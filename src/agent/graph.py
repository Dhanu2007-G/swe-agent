"""
src/agent/graph.py — LangGraph state machine assembly.
The graph is compiled once at startup and reused across runs (thread-safe).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import structlog
from langgraph.graph import END, StateGraph

import src.agent.nodes as nodes
from src.agent.state import AgentState, RunStatus
from src.config import get_settings

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

log = structlog.get_logger(__name__)


# ── Routing Functions ─────────────────────────────────────────────────────────


def route_entry(state: AgentState) -> Literal["read_issue", "read_review"]:
    """Route initial entry point based on job_type."""
    if state.get("job_type") == "review_refinement":
        return "read_review"
    return "read_issue"


def route_after_read(
    state: AgentState,
) -> Literal["plan", "__end__"]:
    """Route after read_issue node. If ambiguous, terminates early."""
    if state.get("status") == RunStatus.FAILED.value:
        return cast("Literal['plan', '__end__']", END)
    return "plan"


def route_after_test(
    state: AgentState,
) -> Literal["open_pr", "correct", "fail"]:
    """
    Route after test node:
    - All tests pass → open_pr
    - Budget cap exceeded → fail
    - Tests fail & retry limit not reached → correct
    - Tests fail & retry limit reached → fail
    """
    test_result = state.get("test_result")
    retry_count = state.get("retry_count", 0)
    settings = get_settings()

    if test_result and getattr(test_result, "passed", False):
        return "open_pr"

    if state.get("budget_exceeded", False):
        return "fail"

    if retry_count < settings.agent_max_retries:
        return "correct"

    return "fail"


# ── Graph Builder ─────────────────────────────────────────────────────────────


async def _wrap_read_issue(state: AgentState) -> dict[str, Any]:
    return await nodes.read_issue_node(state)


async def _wrap_read_review(state: AgentState) -> dict[str, Any]:
    return await nodes.read_review_node(state)


async def _wrap_refine(state: AgentState) -> dict[str, Any]:
    return await nodes.refine_from_review_node(state)


async def _wrap_plan(state: AgentState) -> dict[str, Any]:
    return await nodes.plan_node(state)


async def _wrap_code(state: AgentState) -> dict[str, Any]:
    return await nodes.code_node(state)


async def _wrap_test(state: AgentState) -> dict[str, Any]:
    return await nodes.test_node(state)


async def _wrap_correct(state: AgentState) -> dict[str, Any]:
    return await nodes.correct_node(state)


async def _wrap_open_pr(state: AgentState) -> dict[str, Any]:
    return await nodes.open_pr_node(state)


async def _wrap_fail(state: AgentState) -> dict[str, Any]:
    return await nodes.fail_node(state)


def build_graph() -> StateGraph[AgentState]:
    """
    Assemble the state machine. Node order:
    read_issue ─→ plan ─→ code ─┐
                                │
    read_review → refine_review ┴→ test ─┬→ open_pr → END
                                   ↑     ├→ correct ──┘
                                   └─────┴→ fail → END
    """
    graph = StateGraph(AgentState)

    # Add nodes with dynamic dispatch
    graph.add_node("read_issue", _wrap_read_issue)
    graph.add_node("read_review", _wrap_read_review)
    graph.add_node("refine_from_review", _wrap_refine)
    graph.add_node("plan", _wrap_plan)
    graph.add_node("code", _wrap_code)
    graph.add_node("test", _wrap_test)
    graph.add_node("correct", _wrap_correct)
    graph.add_node("open_pr", _wrap_open_pr)
    graph.add_node("fail", _wrap_fail)

    # Entry point
    graph.set_conditional_entry_point(
        route_entry,
        {
            "read_issue": "read_issue",
            "read_review": "read_review",
        },
    )

    # Fixed edges
    graph.add_edge("read_review", "refine_from_review")
    graph.add_edge("refine_from_review", "test")
    graph.add_edge("plan", "code")
    graph.add_edge("code", "test")
    graph.add_edge("correct", "test")  # correction always feeds back into test
    graph.add_edge("open_pr", END)
    graph.add_edge("fail", END)

    # Conditional edges
    graph.add_conditional_edges(
        "read_issue",
        route_after_read,
        {"plan": "plan", END: END},
    )
    graph.add_conditional_edges(
        "test",
        route_after_test,
        {"open_pr": "open_pr", "correct": "correct", "fail": "fail"},
    )

    return graph


async def get_compiled_graph(checkpointer: AsyncPostgresSaver | None = None) -> Any:
    """
    Compile the graph with optional Postgres checkpointing for persistence.
    Checkpointing enables:
    - Resuming failed runs
    - Full audit trail of state snapshots
    - Human-in-the-loop breakpoints (future feature)
    """
    graph = build_graph()

    if checkpointer is not None:
        compiled = graph.compile(checkpointer=checkpointer)
        log.info("graph.compiled_with_checkpointing")
    else:
        compiled = graph.compile()
        log.info("graph.compiled_without_checkpointing")

    return compiled


# ── Runner ────────────────────────────────────────────────────────────────────


async def run_agent(
    initial_state: AgentState,
    checkpointer: AsyncPostgresSaver | None = None,
) -> AgentState:
    """
    Execute the graph for a single issue.
    Returns the final state after the run completes or fails.
    """
    from src.observability.tracing import record_run_metrics

    graph = await get_compiled_graph(checkpointer=checkpointer)

    run_id = initial_state.get("run_id", "unknown")
    issue = initial_state.get("issue")

    config: RunnableConfig = {
        "configurable": {"thread_id": run_id},
        "tags": [
            f"issue:{issue.issue_number if issue else 'unknown'}",
            f"repo:{issue.repo_full_name if issue else 'unknown'}",
        ],
        "metadata": {
            "run_id": run_id,
            "issue_number": issue.issue_number if issue else None,
        },
    }

    log.info("agent.run_start", run_id=run_id, issue=issue.issue_number if issue else None)

    try:
        final_state: AgentState = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        log.error("agent.run_exception", run_id=run_id, error=str(e), exc_info=True)
        raise

    status = final_state.get("status", RunStatus.FAILED.value)
    pr = final_state.get("pull_request")
    pr_url = getattr(pr, "pr_url", None) if pr else None
    if pr_url is None and isinstance(pr, dict):
        pr_url = pr.get("pr_url")

    log.info(
        "agent.run_complete",
        run_id=run_id,
        status=status,
        pr_url=pr_url,
        retries=final_state.get("retry_count", 0),
        tokens=final_state.get("total_tokens_used", 0),
    )

    await record_run_metrics(cast("dict[str, Any]", final_state))

    return final_state
