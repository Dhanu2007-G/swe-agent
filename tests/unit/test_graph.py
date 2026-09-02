from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.state import AgentState, GithubIssue, RunStatus


@pytest.fixture
def sample_issue() -> GithubIssue:
    return GithubIssue(
        issue_number=42,
        repo_full_name="owner/repo",
        title="Bug",
        body="Fix it",
        labels=["agent-fix"],
        comments=[],
        linked_prs=[],
        assignees=[],
        created_at=datetime.now(UTC),
        html_url="https://github.com/owner/repo/issues/42",
    )


class TestGetCompiledGraph:
    @pytest.mark.asyncio
    async def test_compiles_with_checkpointing_when_provided(self) -> None:
        from src.agent.graph import get_compiled_graph

        graph = MagicMock()
        graph.compile.return_value = "compiled-graph"
        checkpointer = object()

        with patch("src.agent.graph.build_graph", return_value=graph):
            compiled = await get_compiled_graph(checkpointer=checkpointer)

        assert compiled == "compiled-graph"
        graph.compile.assert_called_once_with(checkpointer=checkpointer)

    @pytest.mark.asyncio
    async def test_compiles_without_checkpointing_by_default(self) -> None:
        from src.agent.graph import get_compiled_graph

        graph = MagicMock()
        graph.compile.return_value = "compiled-graph"

        with patch("src.agent.graph.build_graph", return_value=graph):
            compiled = await get_compiled_graph()

        assert compiled == "compiled-graph"
        graph.compile.assert_called_once_with()


class TestRunAgent:
    @pytest.mark.asyncio
    async def test_invokes_graph_with_expected_config_and_records_metrics(
        self,
        sample_issue: GithubIssue,
    ) -> None:
        from src.agent.graph import run_agent

        initial_state: AgentState = {
            "run_id": "run-123",
            "issue": sample_issue,
            "retry_count": 0,
            "attempt_history": [],
            "started_at": datetime.now(UTC).isoformat(),
            "total_tokens_used": 0,
        }
        final_state: AgentState = {
            **initial_state,
            "status": RunStatus.SUCCEEDED.value,
            "pull_request": {"pr_url": "https://github.com/owner/repo/pull/1"},
        }
        compiled = SimpleNamespace(ainvoke=AsyncMock(return_value=final_state))

        with (
            patch(
                "src.agent.graph.get_compiled_graph",
                new_callable=AsyncMock,
                return_value=compiled,
            ),
            patch(
                "src.observability.tracing.record_run_metrics",
                new_callable=AsyncMock,
            ) as record_metrics,
        ):
            result = await run_agent(initial_state)

        assert result == final_state
        compiled.ainvoke.assert_awaited_once()
        call_args = compiled.ainvoke.await_args
        assert call_args.args[0] == initial_state
        assert call_args.kwargs["config"] == {
            "configurable": {"thread_id": "run-123"},
            "tags": ["issue:42", "repo:owner/repo"],
            "metadata": {"run_id": "run-123", "issue_number": 42},
        }
        record_metrics.assert_awaited_once_with(final_state)

    @pytest.mark.asyncio
    async def test_reraises_graph_exceptions_without_recording_metrics(
        self,
        sample_issue: GithubIssue,
    ) -> None:
        from src.agent.graph import run_agent

        initial_state: AgentState = {
            "run_id": "run-123",
            "issue": sample_issue,
        }
        compiled = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("graph failed")))

        with (
            patch(
                "src.agent.graph.get_compiled_graph",
                new_callable=AsyncMock,
                return_value=compiled,
            ),
            patch(
                "src.observability.tracing.record_run_metrics",
                new_callable=AsyncMock,
            ) as record_metrics,
            pytest.raises(RuntimeError, match="graph failed"),
        ):
            await run_agent(initial_state)

        record_metrics.assert_not_awaited()
