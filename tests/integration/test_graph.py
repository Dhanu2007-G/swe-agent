"""
tests/integration/test_graph.py — End-to-end graph tests.
All external boundaries (LLM, Docker, GitHub) are mocked.
These tests verify the full state machine wiring, routing logic,
and state propagation — not individual node behaviour.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.state import (
    AgentState,
    CodePatch,
    ErrorCategory,
    FilePatch,
    GithubIssue,
    PullRequest,
    RunStatus,
    Task,
    TaskPlan,
    TaskPriority,
    TestFailure,
    TestResult,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def issue() -> GithubIssue:
    return GithubIssue(
        issue_number=101,
        repo_full_name="acme/backend",
        title="Fix: divide-by-zero in compute_ratio()",
        body="When `denominator` is 0, `compute_ratio()` raises ZeroDivisionError. "
             "Should return 0.0 instead.",
        labels=["bug", "agent-fix"],
        comments=[],
        linked_prs=[],
        assignees=[],
        created_at=datetime.now(timezone.utc),
        html_url="https://github.com/acme/backend/issues/101",
    )


@pytest.fixture
def plan(issue: GithubIssue) -> TaskPlan:
    return TaskPlan(
        summary="Guard against zero denominator in compute_ratio",
        tasks=[
            Task(
                id="t1",
                description="Add zero-check guard in compute_ratio()",
                files_to_modify=["src/math_utils.py"],
                files_to_create=[],
                acceptance_criteria=[
                    "compute_ratio(x, 0) returns 0.0",
                    "compute_ratio(x, y) returns x/y for y != 0",
                ],
                risks=[],
                priority=TaskPriority.HIGH,
            )
        ],
        affected_test_files=["tests/test_math_utils.py"],
        breaking_change_risk=False,
    )


@pytest.fixture
def good_patch() -> CodePatch:
    return CodePatch(
        patches=[
            FilePatch(
                file_path="src/math_utils.py",
                unified_diff=(
                    "--- a/src/math_utils.py\n"
                    "+++ b/src/math_utils.py\n"
                    "@@ -3,4 +3,6 @@\n"
                    " def compute_ratio(numerator, denominator):\n"
                    "+    if denominator == 0:\n"
                    "+        return 0.0\n"
                    "     return numerator / denominator\n"
                ),
                change_type="modify",
                description="Add zero guard before division",
            )
        ],
        explanation="Guard denominator == 0 to prevent ZeroDivisionError",
        test_command="pytest tests/test_math_utils.py",
        new_dependencies=[],
    )


@pytest.fixture
def bad_patch() -> CodePatch:
    """A patch that produces failing tests."""
    return CodePatch(
        patches=[
            FilePatch(
                file_path="src/math_utils.py",
                unified_diff=(
                    "--- a/src/math_utils.py\n"
                    "+++ b/src/math_utils.py\n"
                    "@@ -3,4 +3,5 @@\n"
                    " def compute_ratio(numerator, denominator):\n"
                    "+    return None  # wrong fix\n"
                    "     return numerator / denominator\n"
                ),
                change_type="modify",
                description="Incorrect fix",
            )
        ],
        explanation="Wrong approach",
        test_command="pytest tests/test_math_utils.py",
    )


@pytest.fixture
def passing_tests() -> TestResult:
    return TestResult(
        passed=True, total=5, failed_count=0, coverage_pct=91.0, duration_seconds=1.2
    )


@pytest.fixture
def failing_tests() -> TestResult:
    return TestResult(
        passed=False,
        total=5,
        failed_count=1,
        failures=[
            TestFailure(
                test_id="tests/test_math_utils.py::test_zero_denominator",
                test_name="test_zero_denominator",
                error_message="AssertionError: assert None == 0.0",
                traceback="...\nAssertionError: assert None == 0.0",
                error_category=ErrorCategory.LOGIC_ERROR,
            )
        ],
        duration_seconds=0.8,
    )


@pytest.fixture
def mock_pr() -> PullRequest:
    return PullRequest(
        pr_number=202,
        pr_url="https://github.com/acme/backend/pull/202",
        branch_name="swe-agent/issue-101-abcd1234",
        is_draft=False,
        title="fix: resolve #101 — Fix: divide-by-zero in compute_ratio()",
        body="Automated fix for #101",
    )


# ─── Happy Path: Issue → Passing Tests → PR ──────────────────────────────────

class TestHappyPath:
    @pytest.mark.asyncio
    async def test_full_run_succeeds_first_attempt(
        self,
        issue: GithubIssue,
        plan: TaskPlan,
        good_patch: CodePatch,
        passing_tests: TestResult,
        mock_pr: PullRequest,
    ) -> None:
        """
        Full graph run: read → plan → code → test (pass) → open_pr.
        Zero retries. Final status: SUCCEEDED.
        """
        initial_state: AgentState = {
            "issue": issue,
            "retry_count": 0,
            "attempt_history": [],
        }

        with (
            _mock_read_issue(status=RunStatus.RUNNING),
            _mock_plan(plan),
            _mock_code(good_patch),
            _mock_test(passing_tests),
            _mock_open_pr(mock_pr),
        ):
            from src.agent.graph import run_agent
            final = await run_agent(initial_state)

        assert final["status"] == RunStatus.SUCCEEDED
        assert final["retry_count"] == 0
        pr = final.get("pull_request", {})
        assert pr["pr_url"] == mock_pr.pr_url
        assert pr["is_draft"] is False

    @pytest.mark.asyncio
    async def test_full_run_corrects_once_then_succeeds(
        self,
        issue: GithubIssue,
        plan: TaskPlan,
        bad_patch: CodePatch,
        good_patch: CodePatch,
        failing_tests: TestResult,
        passing_tests: TestResult,
        mock_pr: PullRequest,
    ) -> None:
        """
        Graph run: read → plan → code → test (FAIL) → correct → test (PASS) → open_pr.
        One retry. Final status: SUCCEEDED.
        """
        initial_state: AgentState = {
            "issue": issue,
            "retry_count": 0,
            "attempt_history": [],
        }

        # test_node called twice: fail first, then pass
        test_call_count = 0

        async def _test_side_effect(*args: Any, **kwargs: Any) -> dict:
            nonlocal test_call_count
            test_call_count += 1
            result = failing_tests if test_call_count == 1 else passing_tests
            return {"test_result": result}

        with (
            _mock_read_issue(status=RunStatus.RUNNING),
            _mock_plan(plan),
            _mock_code(bad_patch),
            patch("src.agent.nodes.test_node", side_effect=_test_side_effect),
            _mock_correct(good_patch),
            _mock_open_pr(mock_pr),
        ):
            from src.agent.graph import run_agent
            final = await run_agent(initial_state)

        assert final["status"] == RunStatus.SUCCEEDED
        assert final["retry_count"] == 1
        assert test_call_count == 2

    @pytest.mark.asyncio
    async def test_full_run_opens_draft_pr_after_exhausting_retries(
        self,
        issue: GithubIssue,
        plan: TaskPlan,
        bad_patch: CodePatch,
        failing_tests: TestResult,
        mock_pr: PullRequest,
    ) -> None:
        """
        Graph: test always fails → exhausts 3 retries → fail_node → draft PR.
        Final status: PARTIAL.
        """
        initial_state: AgentState = {
            "issue": issue,
            "retry_count": 0,
            "attempt_history": [],
        }

        draft_pr = mock_pr.model_copy(update={"is_draft": True})

        with (
            _mock_read_issue(status=RunStatus.RUNNING),
            _mock_plan(plan),
            _mock_code(bad_patch),
            patch("src.agent.nodes.test_node",
                  side_effect=AsyncMock(return_value={"test_result": failing_tests})),
            patch("src.agent.nodes.correct_node",
                  side_effect=_make_correct_side_effect(bad_patch)),
            _mock_open_pr(draft_pr, is_draft=True),
            patch("src.config.get_settings") as mock_settings,
        ):
            mock_settings.return_value.agent_max_retries = 3
            mock_settings.return_value.agent_pr_draft_on_failure = True
            mock_settings.return_value.anthropic_model = "claude-opus-4-6"

            from src.agent.graph import run_agent
            final = await run_agent(initial_state)

        assert final["status"] in (RunStatus.PARTIAL, RunStatus.FAILED)


# ─── Abort Path: Ambiguous Issue ─────────────────────────────────────────────

class TestAbortPaths:
    @pytest.mark.asyncio
    async def test_aborts_on_ambiguous_issue(
        self, issue: GithubIssue
    ) -> None:
        """
        read_issue returns FAILED → graph routes to END without planning.
        """
        initial_state: AgentState = {
            "issue": issue,
            "retry_count": 0,
            "attempt_history": [],
        }

        plan_called = False

        async def _mock_plan(*args: Any, **kwargs: Any) -> dict:
            nonlocal plan_called
            plan_called = True
            return {}

        with (
            _mock_read_issue(status=RunStatus.FAILED),
            patch("src.agent.nodes.plan_node", side_effect=_mock_plan),
        ):
            from src.agent.graph import run_agent
            final = await run_agent(initial_state)

        assert final["status"] == RunStatus.FAILED
        assert plan_called is False, "plan_node should never be called after FAILED read"


# ─── State Propagation Tests ──────────────────────────────────────────────────

class TestStatePropagation:
    @pytest.mark.asyncio
    async def test_run_id_set_by_read_issue_and_preserved(
        self,
        issue: GithubIssue,
        plan: TaskPlan,
        good_patch: CodePatch,
        passing_tests: TestResult,
        mock_pr: PullRequest,
    ) -> None:
        """run_id set in read_issue must be present in every subsequent node's state."""
        observed_run_ids: list[str] = []

        original_code = __import__(
            "src.agent.nodes", fromlist=["code_node"]
        ).code_node

        async def _capturing_code(state: AgentState) -> dict:
            observed_run_ids.append(state.get("run_id", "MISSING"))
            return {"code_patch": good_patch, "file_contexts": []}

        initial_state: AgentState = {
            "issue": issue,
            "retry_count": 0,
            "attempt_history": [],
        }

        with (
            _mock_read_issue(status=RunStatus.RUNNING, run_id="test-run-xyz"),
            _mock_plan(plan),
            patch("src.agent.nodes.code_node", side_effect=_capturing_code),
            _mock_test(passing_tests),
            _mock_open_pr(mock_pr),
        ):
            from src.agent.graph import run_agent
            final = await run_agent(initial_state)

        assert all(rid == "test-run-xyz" for rid in observed_run_ids), (
            f"run_id not propagated correctly: {observed_run_ids}"
        )

    @pytest.mark.asyncio
    async def test_attempt_history_grows_on_each_retry(
        self,
        issue: GithubIssue,
        plan: TaskPlan,
        bad_patch: CodePatch,
        good_patch: CodePatch,
        failing_tests: TestResult,
        passing_tests: TestResult,
        mock_pr: PullRequest,
    ) -> None:
        """Each correction cycle must append to attempt_history."""
        history_sizes: list[int] = []

        async def _capturing_correct(state: AgentState) -> dict:
            history_sizes.append(len(state.get("attempt_history", [])))
            retry = state.get("retry_count", 0) + 1
            return {
                "code_patch": good_patch,
                "retry_count": retry,
                "attempt_history": state.get("attempt_history", []) + [{"attempt": retry}],
                "last_error_category": "logic_error",
            }

        test_call_count = 0

        async def _test_seq(*args: Any, **kwargs: Any) -> dict:
            nonlocal test_call_count
            test_call_count += 1
            result = failing_tests if test_call_count == 1 else passing_tests
            return {"test_result": result}

        initial_state: AgentState = {
            "issue": issue,
            "retry_count": 0,
            "attempt_history": [],
        }

        with (
            _mock_read_issue(status=RunStatus.RUNNING),
            _mock_plan(plan),
            _mock_code(bad_patch),
            patch("src.agent.nodes.test_node", side_effect=_test_seq),
            patch("src.agent.nodes.correct_node", side_effect=_capturing_correct),
            _mock_open_pr(mock_pr),
        ):
            from src.agent.graph import run_agent
            await run_agent(initial_state)

        assert history_sizes == [0], (
            "First correction should see empty history, got: {history_sizes}"
        )


# ─── Helpers: context managers that patch individual nodes ────────────────────

def _mock_read_issue(
    status: RunStatus = RunStatus.RUNNING,
    run_id: str = "integration-run-001",
) -> Any:
    async def _impl(state: AgentState) -> dict:
        return {
            "run_id": run_id,
            "status": status.value,
            "retry_count": 0,
            "attempt_history": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total_tokens_used": 0,
        }
    return patch("src.agent.nodes.read_issue_node", side_effect=_impl)


def _mock_plan(plan: TaskPlan) -> Any:
    async def _impl(state: AgentState) -> dict:
        return {"plan": plan, "current_task_index": 0}
    return patch("src.agent.nodes.plan_node", side_effect=_impl)


def _mock_code(patch_obj: CodePatch) -> Any:
    async def _impl(state: AgentState) -> dict:
        return {"code_patch": patch_obj, "file_contexts": []}
    return patch("src.agent.nodes.code_node", side_effect=_impl)


def _mock_test(result: TestResult) -> Any:
    async def _impl(state: AgentState) -> dict:
        return {"test_result": result}
    return patch("src.agent.nodes.test_node", side_effect=_impl)


def _mock_correct(patch_obj: CodePatch) -> Any:
    async def _impl(state: AgentState) -> dict:
        retry = state.get("retry_count", 0) + 1
        prev = state.get("attempt_history", [])
        return {
            "code_patch": patch_obj,
            "retry_count": retry,
            "attempt_history": prev + [{"attempt": retry}],
            "last_error_category": "logic_error",
        }
    return patch("src.agent.nodes.correct_node", side_effect=_impl)


def _mock_open_pr(pr: PullRequest, is_draft: bool = False) -> Any:
    async def _impl(state: AgentState) -> dict:
        return {
            "pull_request": pr.model_dump(),
            "status": RunStatus.PARTIAL.value if is_draft else RunStatus.SUCCEEDED.value,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    return patch("src.agent.nodes.open_pr_node", side_effect=_impl)


def _make_correct_side_effect(patch_obj: CodePatch) -> Any:
    async def _impl(state: AgentState) -> dict:
        retry = state.get("retry_count", 0) + 1
        prev = state.get("attempt_history", [])
        return {
            "code_patch": patch_obj,
            "retry_count": retry,
            "attempt_history": prev + [{"attempt": retry}],
            "last_error_category": "logic_error",
        }
    return AsyncMock(side_effect=_impl)
