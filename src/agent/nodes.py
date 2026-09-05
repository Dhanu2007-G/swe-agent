"""
src/agent/nodes.py — The 6 nodes of the SWE agent graph.
Each node is a pure async function: AgentState -> dict[partial updates].
Nodes are independently testable and observable via LangSmith.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.agent.prompts import (
    CLARIFIER_SYSTEM,
    CLARIFIER_USER,
    CODER_SYSTEM,
    CODER_USER,
    CORRECTOR_SYSTEM,
    CORRECTOR_USER,
    DRAFT_PR_BODY,
    ERROR_CLASSIFIER_SYSTEM,
    ERROR_CLASSIFIER_USER,
    PLANNER_SYSTEM,
    PLANNER_USER,
    PR_BODY_SYSTEM,
    PR_BODY_USER,
)
from src.agent.state import (
    AgentState,
    AttemptRecord,
    CodePatch,
    ErrorCategory,
    GithubIssue,
    RunStatus,
    TaskPlan,
    TestResult,
)
from src.config import get_settings
from src.observability.tracing import get_tracer

log = structlog.get_logger(__name__)


def _build_llm(temperature: float = 0.0, max_tokens: int | None = None) -> Any:
    """Build an LLM client based on configured provider (Anthropic, Gemini, OpenAI)."""
    settings = get_settings()
    provider = getattr(settings, "llm_provider", "anthropic").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key_value or "dummy-gemini-key",
            temperature=temperature,
            max_output_tokens=max_tokens or settings.anthropic_max_tokens,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(  # type: ignore[call-arg]
            model=settings.openai_model,
            api_key=settings.openai_api_key_value or "dummy-openai-key",  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens or settings.anthropic_max_tokens,
        )

    return ChatAnthropic(  # type: ignore[call-arg]
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key_value or "dummy-anthropic-key",  # type: ignore[arg-type]
        max_tokens=max_tokens or settings.anthropic_max_tokens,
        temperature=temperature,
        timeout=settings.anthropic_timeout_seconds,
        max_retries=settings.anthropic_max_retries,
    )


async def _invoke_with_timeout(
    llm: Any,
    messages: list[Any],
    timeout: int | None = None,
    run_name: str = "llm_call",
) -> Any:
    """Invoke LLM with timeout and structured retry on rate limits."""
    settings = get_settings()
    effective_timeout = timeout or settings.anthropic_timeout_seconds

    from anthropic import APIConnectionError, APIStatusError, RateLimitError

    async for attempt in AsyncRetrying(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    ):
        with attempt:
            return await asyncio.wait_for(
                llm.ainvoke(messages, config={"run_name": run_name}),
                timeout=float(effective_timeout),
            )


# ── Node 1: Read Issue ────────────────────────────────────────────────────────


async def read_issue_node(state: AgentState) -> dict[str, Any]:
    """
    Validates the issue is implementable. Sets run_id and timestamps.
    Short-circuits if issue is ambiguous and marks status accordingly.
    """
    issue: GithubIssue = state["issue"]
    # Preserve existing run_id from worker; only generate if missing
    run_id = state.get("run_id") or str(uuid4())

    log.info("read_issue.start", run_id=run_id, issue=issue.issue_number, repo=issue.repo_full_name)

    settings = get_settings()
    llm = _build_llm(temperature=0.0, max_tokens=512)

    clarifier_messages = [
        SystemMessage(content=CLARIFIER_SYSTEM),
        HumanMessage(content=CLARIFIER_USER.format(issue_context=issue.to_prompt_context())),
    ]

    try:
        response = await _invoke_with_timeout(llm, clarifier_messages, timeout=30, run_name="clarifier")
        clarity_data = json.loads(response.content)
        can_proceed = clarity_data.get("can_proceed", True)
    except (json.JSONDecodeError, Exception) as e:
        log.warning("read_issue.clarifier_failed", error=str(e), run_id=run_id)
        can_proceed = True  # fail open — attempt the issue

    updates: dict[str, Any] = {
        "run_id": run_id,
        "retry_count": 0,
        "attempt_history": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "total_tokens_used": 0,
    }

    if not can_proceed:
        log.warning("read_issue.ambiguous", run_id=run_id, issue=issue.issue_number)
        updates["status"] = RunStatus.FAILED
        updates["failure_reason"] = "Issue is too ambiguous to implement safely"
        return updates

    updates["status"] = RunStatus.RUNNING
    log.info("read_issue.complete", run_id=run_id)
    return updates


# ── Node 2: Plan ──────────────────────────────────────────────────────────────


async def plan_node(state: AgentState) -> dict[str, Any]:
    """
    Produces a structured TaskPlan using the planner prompt.
    Uses structured output to guarantee schema compliance.
    """
    from src.tools.filesystem import list_repo_tree, list_test_files

    issue: GithubIssue = state["issue"]
    run_id: str = state["run_id"]

    log.info("plan.start", run_id=run_id, issue=issue.issue_number)
    start_time = time.monotonic()

    # Gather repo context
    repo_tree = await list_repo_tree(issue.repo_full_name)
    test_files = await list_test_files(issue.repo_full_name)

    llm = _build_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(TaskPlan)

    messages = [
        SystemMessage(content=PLANNER_SYSTEM),
        HumanMessage(
            content=PLANNER_USER.format(
                issue_context=issue.to_prompt_context(),
                repo_tree=repo_tree,
                test_files="\n".join(test_files[:20]),
            )
        ),
    ]

    try:
        plan: TaskPlan = await _invoke_with_timeout(structured_llm, messages, run_name="planner")
    except Exception as e:
        log.error("plan.failed", run_id=run_id, error=str(e))
        raise

    elapsed = time.monotonic() - start_time
    log.info(
        "plan.complete",
        run_id=run_id,
        tasks=len(plan.tasks),
        breaking_change=plan.breaking_change_risk,
        elapsed_s=f"{elapsed:.1f}",
    )

    return {
        "plan": plan,
        "current_task_index": 0,
    }


# ── Node 3: Code ──────────────────────────────────────────────────────────────


async def code_node(state: AgentState) -> dict[str, Any]:
    """
    Loads relevant file context and generates a CodePatch for ALL tasks in the plan.
    Uses ContextBuilder for AST-aware, token-budgeted context preparation.
    """
    from src.agent.context import build_context_from_file_dicts
    from src.tools.filesystem import load_file_contexts
    from src.tools.search import find_relevant_files

    issue: GithubIssue = state["issue"]
    plan: TaskPlan = state["plan"]
    run_id: str = state["run_id"]
    settings = get_settings()

    # ── Aggregate all tasks into a single coding request ──────────────────────
    all_planned_files: list[str] = []
    task_descriptions: list[str] = []
    all_acceptance_criteria: list[str] = []

    for i, task in enumerate(plan.tasks):
        task_descriptions.append(f"### Task {i + 1}: {task.description}")
        all_acceptance_criteria.extend(task.acceptance_criteria)
        all_planned_files.extend(task.files_to_modify)
        all_planned_files.extend(task.files_to_create)

    # Deduplicate while preserving order
    all_planned_files = list(dict.fromkeys(all_planned_files))
    combined_description = "\n\n".join(task_descriptions)

    log.info("code.start", run_id=run_id, total_tasks=len(plan.tasks), planned_files=len(all_planned_files))

    # ── Discover additional relevant files via BM25 search ────────────────────
    additional_files = await find_relevant_files(
        repo=issue.repo_full_name,
        query=combined_description,
        exclude=all_planned_files,
        max_results=3,
    )
    all_files = list(dict.fromkeys(all_planned_files + additional_files))
    all_files = all_files[: settings.agent_max_files_in_context]

    # ── Load and build AST-aware context ──────────────────────────────────────
    file_contexts = await load_file_contexts(
        repo=issue.repo_full_name,
        paths=all_files,
        max_tokens_per_file=4000,
    )

    # Use ContextBuilder for AST-aware, token-budgeted context
    file_context_str = build_context_from_file_dicts(
        file_dicts=file_contexts,
        task_description=combined_description,
    )

    # ── Generate unified patch covering all tasks ─────────────────────────────
    llm = _build_llm(temperature=0.1)  # slight creativity for code gen
    structured_llm = llm.with_structured_output(CodePatch)

    messages = [
        SystemMessage(content=CODER_SYSTEM),
        HumanMessage(
            content=CODER_USER.format(
                task_description=combined_description,
                acceptance_criteria="\n".join(f"- {c}" for c in all_acceptance_criteria),
                file_contexts=file_context_str,
                issue_summary=issue.title,
            )
        ),
    ]

    try:
        patch: CodePatch = await _invoke_with_timeout(structured_llm, messages, run_name="coder")
    except Exception as e:
        log.error("code.failed", run_id=run_id, error=str(e))
        raise

    log.info("code.complete", run_id=run_id, patches=len(patch.patches), tasks_covered=len(plan.tasks))

    return {
        "file_contexts": file_contexts,
        "code_patch": patch,
    }


# ── Node 4: Test ──────────────────────────────────────────────────────────────


async def test_node(state: AgentState) -> dict[str, Any]:
    """
    Applies the patch to the sandbox and runs the test suite.
    Returns a structured TestResult regardless of outcome.
    """
    from src.tools.sandbox import SandboxRunner

    issue: GithubIssue = state["issue"]
    patch: CodePatch = state["code_patch"]
    run_id: str = state["run_id"]

    log.info("test.start", run_id=run_id, patches=len(patch.patches))

    async with SandboxRunner(
        repo_full_name=issue.repo_full_name,
        run_id=run_id,
    ) as sandbox:
        # Apply patches one by one — atomically roll back on any failure
        apply_result = await sandbox.apply_patches(patch.patches)  # type: ignore[arg-type]
        if not apply_result.success:
            log.warning("test.patch_apply_failed", run_id=run_id, error=apply_result.error)
            return {
                "test_result": TestResult(
                    passed=False,
                    failures=[],
                    stderr=apply_result.error,
                )
            }

        # Install any new dependencies declared by the coder
        if patch.new_dependencies:
            await sandbox.install_packages(patch.new_dependencies)

        # Run the test suite
        test_result = await sandbox.run_tests(
            test_command=patch.test_command,
        )

    log.info(
        "test.complete",
        run_id=run_id,
        passed=test_result.passed,
        total=test_result.total,
        failed=test_result.failed_count,
        coverage=f"{test_result.coverage_pct:.1f}%",
        duration=f"{test_result.duration_seconds:.1f}s",
    )

    return {"test_result": test_result}


# ── Node 5: Self-Correct ──────────────────────────────────────────────────────


async def correct_node(state: AgentState) -> dict[str, Any]:
    """
    Classifies the failure, builds correction context, and generates a new patch.
    Passes the full history of previous attempts to prevent looping.
    """
    issue: GithubIssue = state["issue"]
    plan: TaskPlan = state["plan"]
    task_index: int = state.get("current_task_index", 0)
    current_patch: CodePatch = state["code_patch"]
    test_result: TestResult = state["test_result"]
    attempt_history: list[dict[str, Any]] = state.get("attempt_history", [])
    retry_count: int = state.get("retry_count", 0)
    run_id: str = state["run_id"]

    current_task = plan.tasks[task_index]
    new_retry_count = retry_count + 1

    log.info("correct.start", run_id=run_id, attempt=new_retry_count, failures=test_result.failed_count)

    # ── Classify error ────────────────────────────────────────────────────────
    error_category = await _classify_error(test_result)
    log.info("correct.error_classified", run_id=run_id, category=error_category)

    # ── Record this attempt ───────────────────────────────────────────────────
    attempt = AttemptRecord(
        attempt_number=new_retry_count,
        patch=current_patch,
        test_result=test_result,
        error_category=error_category,
    )
    updated_history = attempt_history + [attempt.model_dump(mode="json")]

    # ── Format previous attempts for prompt ───────────────────────────────────
    previous_attempts_str = _format_previous_attempts(attempt_history)

    # ── Generate corrected patch ──────────────────────────────────────────────
    patch_str = "\n".join([f"File: {p.file_path}\n{p.unified_diff}" for p in current_patch.patches])

    llm = _build_llm(temperature=0.2)  # slightly more creative for corrections
    structured_llm = llm.with_structured_output(CodePatch)

    messages = [
        SystemMessage(content=CORRECTOR_SYSTEM),
        HumanMessage(
            content=CORRECTOR_USER.format(
                task_description=current_task.description,
                applied_patch=patch_str,
                failure_summary=test_result.failure_summary(),
                previous_attempts=previous_attempts_str,
            )
        ),
    ]

    try:
        corrected_patch: CodePatch = await _invoke_with_timeout(structured_llm, messages, run_name="corrector")
    except Exception as e:
        log.error("correct.llm_failed", run_id=run_id, error=str(e))
        raise

    log.info("correct.patch_generated", run_id=run_id, attempt=new_retry_count, patches=len(corrected_patch.patches))

    return {
        "code_patch": corrected_patch,
        "retry_count": new_retry_count,
        "attempt_history": updated_history,
        "last_error_category": error_category.value,
    }


# ── Node 6: Open PR ───────────────────────────────────────────────────────────


async def open_pr_node(state: AgentState) -> dict[str, Any]:
    """
    Generates a PR description, creates the branch, commits patches, and opens the PR.
    Handles both success PRs and draft failure PRs.
    """
    from src.tools.github import GitHubClient

    issue: GithubIssue = state["issue"]
    plan: TaskPlan = state["plan"]
    patch: CodePatch = state["code_patch"]
    test_result: TestResult = state.get("test_result", TestResult(passed=True))
    retry_count: int = state.get("retry_count", 0)
    run_id: str = state["run_id"]
    settings = get_settings()

    is_draft = not test_result.passed
    log.info("open_pr.start", run_id=run_id, is_draft=is_draft, issue=issue.issue_number)

    # ── Generate PR description ───────────────────────────────────────────────
    if is_draft:
        changes_summary = "\n".join([f"- {p.description} ({p.file_path})" for p in patch.patches])
        pr_body = DRAFT_PR_BODY.format(
            max_retries=settings.agent_max_retries,
            issue_number=issue.issue_number,
            changes_summary=changes_summary,
            failure_summary=test_result.failure_summary(),
        )
    else:
        pr_body = await _generate_pr_description(
            issue=issue,
            patch=patch,
            test_result=test_result,
        )

    # ── Create branch and open PR ─────────────────────────────────────────────
    branch_name = f"swe-agent/issue-{issue.issue_number}-{run_id[:8]}"

    async with GitHubClient() as github:
        pr = await github.create_pull_request(
            repo_full_name=issue.repo_full_name,
            branch_name=branch_name,
            base_branch="main",
            title=f"fix: resolve #{issue.issue_number} — {issue.title[:60]}",
            body=pr_body,
            patches=patch.patches,
            issue_number=issue.issue_number,
            labels=[settings.github_pr_label] + (["draft"] if is_draft else []),
            draft=is_draft,
        )

    status = RunStatus.PARTIAL if is_draft else RunStatus.SUCCEEDED
    log.info("open_pr.complete", run_id=run_id, pr=pr.pr_number, url=pr.pr_url, status=status)

    return {
        "pull_request": pr.model_dump(),
        "status": status.value,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Node: Fail ────────────────────────────────────────────────────────────────


async def fail_node(state: AgentState) -> dict[str, Any]:
    """Terminal failure node — exhausted retries without a passing patch."""
    run_id: str = state["run_id"]
    issue: GithubIssue = state["issue"]
    test_result: TestResult = state.get("test_result", TestResult(passed=False))

    log.error(
        "agent.exhausted_retries",
        run_id=run_id,
        issue=issue.issue_number,
        failures=test_result.failed_count,
    )

    # Still open a draft PR so humans can see what was attempted
    if get_settings().agent_pr_draft_on_failure:
        try:
            draft_state = await open_pr_node({**state, "test_result": test_result})
            return {**draft_state, "status": RunStatus.PARTIAL.value}
        except Exception as e:
            log.error("fail_node.draft_pr_failed", run_id=run_id, error=str(e))

    return {
        "status": RunStatus.FAILED.value,
        "failure_reason": f"Exhausted {get_settings().agent_max_retries} retries",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Private Helpers ───────────────────────────────────────────────────────────


async def _classify_error(test_result: TestResult) -> ErrorCategory:
    """Use LLM to classify the error category for routing."""
    if test_result.timed_out:
        return ErrorCategory.TIMEOUT

    if not test_result.failures:
        return ErrorCategory.UNKNOWN

    first_failure = test_result.failures[0]
    llm = _build_llm(temperature=0.0, max_tokens=20)
    traceback_tail = "\n".join(first_failure.traceback.strip().split("\n")[-10:])

    messages = [
        SystemMessage(content=ERROR_CLASSIFIER_SYSTEM),
        HumanMessage(
            content=ERROR_CLASSIFIER_USER.format(
                error_message=first_failure.error_message[:500],
                traceback_tail=traceback_tail,
            )
        ),
    ]

    try:
        response = await _invoke_with_timeout(llm, messages, timeout=15, run_name="error_classifier")
        category_str = response.content.strip().lower()
        return ErrorCategory(category_str)
    except (ValueError, Exception):
        return ErrorCategory.UNKNOWN


def _format_previous_attempts(attempts: list[dict[str, Any]]) -> str:
    """Format attempt history for the correction prompt."""
    if not attempts:
        return "No previous attempts."

    lines = []
    for attempt in attempts:
        lines.append(f"### Attempt {attempt['attempt_number']} [{attempt['error_category']}]")
        patch = attempt.get("patch", {})
        for p in patch.get("patches", []):
            lines.append(f"Modified: {p['file_path']}")
            diff_preview = p.get("unified_diff", "")[:300]
            lines.append(f"Diff preview:\n{diff_preview}")
        lines.append("")

    return "\n".join(lines)


async def _generate_pr_description(
    issue: GithubIssue,
    patch: CodePatch,
    test_result: TestResult,
) -> str:
    """Generate a human-readable PR description."""
    changes_summary = "\n".join([f"- {p.description} in `{p.file_path}`" for p in patch.patches])

    llm = _build_llm(temperature=0.3, max_tokens=600)
    messages = [
        SystemMessage(content=PR_BODY_SYSTEM),
        HumanMessage(
            content=PR_BODY_USER.format(
                issue_number=issue.issue_number,
                issue_title=issue.title,
                issue_url=issue.html_url,
                changes_summary=changes_summary,
                total_tests=test_result.total,
                coverage_pct=test_result.coverage_pct,
            )
        ),
    ]

    try:
        response = await _invoke_with_timeout(llm, messages, timeout=30, run_name="pr_description")
        return response.content
    except Exception:
        return f"Automated fix for #{issue.issue_number}: {issue.title}\n\n{changes_summary}"
