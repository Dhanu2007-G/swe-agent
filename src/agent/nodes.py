"""
src/agent/nodes.py -- The 6 nodes of the SWE agent graph.
Each node is a pure async function: AgentState -> dict[partial updates].
Nodes are independently testable and observable via LangSmith.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
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
    BUDGET_EXCEEDED_BODY,
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
    REPRO_TEST_SYSTEM,
    REPRO_TEST_USER,
    REVIEW_REFINEMENT_SYSTEM,
    REVIEW_REFINEMENT_USER,
)
from src.agent.state import (
    AgentState,
    AttemptRecord,
    CodePatch,
    ErrorCategory,
    FilePatch,
    GithubIssue,
    ReproTest,
    RunStatus,
    TaskPlan,
    TestResult,
)
from src.config import get_settings

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

    from anthropic import APIConnectionError, RateLimitError

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


def _update_tokens_and_budget(state: AgentState, tokens_added: int) -> dict[str, Any]:
    """Calculate cumulative tokens and cost, checking against configured budget caps."""
    settings = get_settings()
    current_tokens = state.get("total_tokens_used", 0) + tokens_added
    # Estimated blended price per token for Claude 3.5 Sonnet ($0.015 / 1k tokens)
    current_cost = current_tokens * 0.000015
    max_tokens = getattr(settings, "agent_max_tokens_per_run", 150_000)
    max_cost = getattr(settings, "agent_max_cost_per_run_usd", 3.00)

    budget_exceeded = current_tokens >= max_tokens or current_cost >= max_cost
    if budget_exceeded:
        log.warning(
            "agent.budget_cap_exceeded",
            tokens=current_tokens,
            cost_usd=round(current_cost, 4),
            max_tokens=max_tokens,
            max_cost=max_cost,
        )
    return {
        "total_tokens_used": current_tokens,
        "cumulative_cost_usd": round(current_cost, 4),
        "budget_exceeded": budget_exceeded,
    }


# ── Node 1: Read Issue ────────────────────────────────────────────────────────


async def read_issue_node(state: AgentState) -> dict[str, Any]:
    """
    Validates the issue is implementable. Sets run_id and timestamps.
    Auto-detects repository language ecosystem and initializes budget tracking.
    Short-circuits if issue is ambiguous and marks status accordingly.
    """
    issue: GithubIssue = state["issue"]
    # Preserve existing run_id from worker; only generate if missing
    run_id = state.get("run_id") or str(uuid4())

    log.info("read_issue.start", run_id=run_id, issue=issue.issue_number, repo=issue.repo_full_name)

    # Detect language ecosystem from local cached repo if available
    from src.tools._repo_cache import get_local_repo_path
    from src.tools.detector import detect_repo_ecosystem

    try:
        repo_dir = await get_local_repo_path(issue.repo_full_name)
        ecosystem = detect_repo_ecosystem(repo_dir) if repo_dir else None
    except Exception:
        ecosystem = None
    detected_lang = ecosystem.language if ecosystem else "python"
    detected_test_cmd = ecosystem.test_command if ecosystem else "pytest"

    llm = _build_llm(temperature=0.0, max_tokens=512)

    clarifier_messages = [
        SystemMessage(content=CLARIFIER_SYSTEM),
        HumanMessage(content=CLARIFIER_USER.format(issue_context=issue.to_prompt_context())),
    ]

    try:
        response = await _invoke_with_timeout(
            llm, clarifier_messages, timeout=30, run_name="clarifier"
        )
        clarity_data = json.loads(response.content)
        can_proceed = clarity_data.get("can_proceed", True)
    except (json.JSONDecodeError, Exception) as e:
        log.warning("read_issue.clarifier_failed", error=str(e), run_id=run_id)
        can_proceed = True  # fail open -- attempt the issue

    updates: dict[str, Any] = {
        "run_id": run_id,
        "retry_count": 0,
        "attempt_history": [],
        "started_at": datetime.now(UTC).isoformat(),
        "total_tokens_used": 0,
        "cumulative_cost_usd": 0.0,
        "budget_exceeded": False,
        "detected_language": detected_lang,
        "detected_test_command": detected_test_cmd,
    }

    if not can_proceed:
        log.warning("read_issue.ambiguous", run_id=run_id, issue=issue.issue_number)
        updates["status"] = RunStatus.FAILED
        updates["failure_reason"] = "Issue is too ambiguous to implement safely"
        return updates

    updates["status"] = RunStatus.RUNNING
    log.info("read_issue.complete", run_id=run_id, language=detected_lang)
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

    budget_updates = _update_tokens_and_budget(state, 1000)
    return {
        "plan": plan,
        "current_task_index": 0,
        **budget_updates,
    }


# ── Node 2b: Generate Reproduction Test (TDD) ──────────────────────────────────


async def repro_test_node(state: AgentState) -> dict[str, Any]:
    """
    Generates a minimal reproduction test verifying the reported issue (TDD).
    Saves the test to state so the coder and sandbox can execute it.
    """
    settings = get_settings()
    if not getattr(settings, "agent_enable_repro_test", True):
        return {}

    issue: GithubIssue = state["issue"]
    run_id: str = state["run_id"]
    detected_lang = state.get("detected_language", "python")

    log.info("repro.start", run_id=run_id, issue=issue.issue_number, lang=detected_lang)

    from src.tools.filesystem import list_test_files

    test_files = await list_test_files(issue.repo_full_name)

    llm = _build_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ReproTest)

    messages = [
        SystemMessage(content=REPRO_TEST_SYSTEM),
        HumanMessage(
            content=REPRO_TEST_USER.format(
                issue_context=issue.to_prompt_context(),
                ecosystem=detected_lang,
                test_files="\n".join(test_files[:10]),
            )
        ),
    ]

    try:
        repro: ReproTest = await _invoke_with_timeout(structured_llm, messages, run_name="repro")
        log.info("repro.generated", file_path=repro.file_path, run_id=run_id)
        budget_updates = _update_tokens_and_budget(state, 500)
        return {
            "repro_test_code": repro.test_code,
            **budget_updates,
        }
    except Exception as e:
        log.warning("repro.failed_open", run_id=run_id, error=str(e))
        return {}


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

    log.info(
        "code.start",
        run_id=run_id,
        total_tasks=len(plan.tasks),
        planned_files=len(all_planned_files),
    )

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

    budget_updates = _update_tokens_and_budget(state, 2000)

    # If a reproduction test exists, bundle it as a FilePatch so it runs in sandbox
    repro_code = state.get("repro_test_code")
    patches_list = list(patch.patches)
    if repro_code and not any("test_repro" in p.file_path for p in patches_list):
        ext = ".py" if state.get("detected_language", "python") == "python" else ".test.ts"
        repro_path = f"tests/test_repro_issue_{issue.issue_number}{ext}"
        patches_list.append(
            FilePatch(
                file_path=repro_path,
                change_type="create",
                full_content=repro_code,
                description="TDD Reproduction test verifying defect",
            )
        )
        patch = patch.model_copy(update={"patches": patches_list})

    log.info(
        "code.complete", run_id=run_id, patches=len(patch.patches), tasks_covered=len(plan.tasks)
    )

    return {
        "file_contexts": file_contexts,
        "code_patch": patch,
        **budget_updates,
    }


# ── Node 4: Test ──────────────────────────────────────────────────────────────


async def test_node(state: AgentState) -> dict[str, Any]:
    """
    Applies the patch to the sandbox and runs the test suite.
    Executes two-phase dependency preparation and uses detected test runners.
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
        # Prepare dependencies (two-phase execution)
        detected_lang = state.get("detected_language", "python")
        await sandbox.prepare_ecosystem_dependencies(detected_lang)

        # Apply patches one by one -- atomically roll back on any failure
        apply_result = await sandbox.apply_patches(patch.patches)
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

        # Run the test suite: use patch.test_command or fallback to detected_test_command
        test_cmd = patch.test_command
        if test_cmd == "pytest" and state.get("detected_test_command"):
            test_cmd = state["detected_test_command"]

        test_result = await sandbox.run_tests(
            test_command=test_cmd,
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
    plan: TaskPlan = state["plan"]
    task_index: int = state.get("current_task_index", 0)
    current_patch: CodePatch = state["code_patch"]
    test_result: TestResult = state["test_result"]
    attempt_history: list[dict[str, Any]] = state.get("attempt_history", [])
    retry_count: int = state.get("retry_count", 0)
    run_id: str = state["run_id"]

    current_task = plan.tasks[task_index]
    new_retry_count = retry_count + 1

    log.info(
        "correct.start", run_id=run_id, attempt=new_retry_count, failures=test_result.failed_count
    )

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
        corrected_patch: CodePatch = await _invoke_with_timeout(
            structured_llm, messages, run_name="corrector"
        )
    except Exception as e:
        log.error("correct.llm_failed", run_id=run_id, error=str(e))
        raise

    log.info(
        "correct.patch_generated",
        run_id=run_id,
        attempt=new_retry_count,
        patches=len(corrected_patch.patches),
    )

    budget_updates = _update_tokens_and_budget(state, 1500)
    return {
        "code_patch": corrected_patch,
        "retry_count": new_retry_count,
        "attempt_history": updated_history,
        "last_error_category": error_category.value,
        **budget_updates,
    }


# ── Node: Refine From Review ──────────────────────────────────────────────────


async def refine_from_review_node(state: AgentState) -> dict[str, Any]:
    """
    Refines an existing patch based on human maintainer review comments.
    """
    issue: GithubIssue = state["issue"]
    run_id: str = state["run_id"]
    current_patch: CodePatch = state.get("code_patch", CodePatch(patches=[], explanation=""))
    review_feedback: str = state.get("review_feedback") or ""

    log.info("refine.start", run_id=run_id, issue=issue.issue_number)

    patch_summary = "\n".join([f"{p.file_path}: {p.description}" for p in current_patch.patches])

    llm = _build_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(CodePatch)

    messages = [
        SystemMessage(content=REVIEW_REFINEMENT_SYSTEM),
        HumanMessage(
            content=REVIEW_REFINEMENT_USER.format(
                issue_number=issue.issue_number,
                issue_title=issue.title,
                current_patch_summary=patch_summary,
                review_feedback=review_feedback,
            )
        ),
    ]

    try:
        refined_patch: CodePatch = await _invoke_with_timeout(
            structured_llm, messages, run_name="refine_review"
        )
        budget_updates = _update_tokens_and_budget(state, 1500)
        return {
            "code_patch": refined_patch,
            **budget_updates,
        }
    except Exception as e:
        log.error("refine.failed", run_id=run_id, error=str(e))
        raise


# ── Node 6: Open PR ───────────────────────────────────────────────────────────


async def open_pr_node(state: AgentState) -> dict[str, Any]:
    """
    Generates a PR description, creates the branch, commits patches, and opens the PR.
    Handles both success PRs and draft failure PRs, including cross-repo forks and budget notices.
    """
    from src.tools.github import GitHubClient

    issue: GithubIssue = state["issue"]
    patch: CodePatch = state["code_patch"]
    test_result: TestResult = state.get("test_result", TestResult(passed=True))
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

    # If budget cap was reached, append notice
    if state.get("budget_exceeded"):
        pr_body += "\n\n" + BUDGET_EXCEEDED_BODY.format(
            issue_number=issue.issue_number,
            tokens_used=state.get("total_tokens_used", 0),
            max_tokens=getattr(settings, "agent_max_tokens_per_run", 150_000),
            cost_usd=state.get("cumulative_cost_usd", 0.0),
            max_cost=getattr(settings, "agent_max_cost_per_run_usd", 3.00),
        )

    # ── Create branch and open PR ─────────────────────────────────────────────
    branch_name = f"swe-agent/issue-{issue.issue_number}-{run_id[:8]}"

    async with GitHubClient() as github:
        pr = await github.create_pull_request(
            repo_full_name=issue.repo_full_name,
            branch_name=branch_name,
            base_branch="main",
            title=f"fix: resolve #{issue.issue_number} -- {issue.title[:60]}",
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
        "completed_at": datetime.now(UTC).isoformat(),
        "is_fork": getattr(pr, "is_fork", False),
        "fork_repo_full_name": getattr(pr, "fork_repo_full_name", None),
    }


# ── Node: Fail ────────────────────────────────────────────────────────────────


async def fail_node(state: AgentState) -> dict[str, Any]:
    """Terminal failure node -- exhausted retries without a passing patch or budget breached."""
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

    failure_reason = f"Exhausted {get_settings().agent_max_retries} retries"
    if state.get("budget_exceeded"):
        tokens = state.get("total_tokens_used", 0)
        cost = state.get("cumulative_cost_usd", 0.0)
        failure_reason = f"Budget cap reached ({tokens} tokens, ${cost:.2f})"

    return {
        "status": RunStatus.FAILED.value,
        "failure_reason": failure_reason,
        "completed_at": datetime.now(UTC).isoformat(),
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
        response = await _invoke_with_timeout(
            llm, messages, timeout=15, run_name="error_classifier"
        )
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
        return str(response.content)
    except Exception:
        return f"Automated fix for #{issue.issue_number}: {issue.title}\n\n{changes_summary}"
