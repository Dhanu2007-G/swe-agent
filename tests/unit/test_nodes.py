"""
tests/unit/test_nodes.py — Unit tests for every agent node.
All external dependencies (LLM, GitHub, Docker) are mocked.
Each node tested: happy path, timeout, malformed output, edge cases.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.agent.state import (
    AgentState,
    CodePatch,
    ErrorCategory,
    FilePatch,
    GithubIssue,
    RunStatus,
    Task,
    TaskPlan,
    TaskPriority,
    TestFailure,
    TestResult,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_issue() -> GithubIssue:
    return GithubIssue(
        issue_number=42,
        repo_full_name="owner/repo",
        title="Fix: NullPointerException in UserService.getUser()",
        body="When `user_id` is None, `get_user()` raises an unhandled exception. "
        "Should return 404 instead.",
        labels=["bug", "agent-fix"],
        comments=["Confirmed — happens in prod"],
        linked_prs=[],
        assignees=[],
        created_at=datetime.now(UTC),
        html_url="https://github.com/owner/repo/issues/42",
    )


@pytest.fixture
def sample_plan() -> TaskPlan:
    return TaskPlan(
        summary="Add None check in get_user and return 404 response",
        tasks=[
            Task(
                id="task-1",
                description="Add None guard in UserService.get_user()",
                files_to_modify=["src/services/user_service.py"],
                files_to_create=[],
                acceptance_criteria=[
                    "Returns HTTP 404 when user_id is None",
                    "Existing tests still pass",
                ],
                risks=["Could affect callers that expect an exception"],
                priority=TaskPriority.HIGH,
            )
        ],
        affected_test_files=["tests/test_user_service.py"],
        breaking_change_risk=False,
    )


@pytest.fixture
def sample_patch() -> CodePatch:
    return CodePatch(
        patches=[
            FilePatch(
                file_path="src/services/user_service.py",
                unified_diff=(
                    "--- a/src/services/user_service.py\n"
                    "+++ b/src/services/user_service.py\n"
                    "@@ -10,6 +10,9 @@\n"
                    " def get_user(user_id):\n"
                    "+    if user_id is None:\n"
                    "+        return None, 404\n"
                    "     user = db.query(User).filter(User.id == user_id).first()\n"
                ),
                change_type="modify",
                description="Add None guard for user_id parameter",
            )
        ],
        explanation="Guard against None user_id before DB query",
        test_command="pytest tests/test_user_service.py",
        new_dependencies=[],
    )


@pytest.fixture
def passing_test_result() -> TestResult:
    return TestResult(
        passed=True,
        total=12,
        failed_count=0,
        error_count=0,
        coverage_pct=84.2,
        duration_seconds=3.1,
    )


@pytest.fixture
def failing_test_result() -> TestResult:
    return TestResult(
        passed=False,
        total=12,
        failed_count=2,
        failures=[
            TestFailure(
                test_id="tests/test_user_service.py::test_get_user_none",
                test_name="test_get_user_none",
                error_message="AssertionError: Expected (None, 404), got AttributeError",
                traceback="Traceback (most recent call last):\n  ...\nAssertionError",
                error_category=ErrorCategory.LOGIC_ERROR,
            )
        ],
        duration_seconds=1.2,
    )


@pytest.fixture
def base_state(sample_issue: GithubIssue) -> AgentState:
    return AgentState(
        run_id="test-run-001",
        issue=sample_issue,
        retry_count=0,
        attempt_history=[],
        started_at=datetime.now(UTC).isoformat(),
        total_tokens_used=0,
    )


# ── test_read_issue_node ──────────────────────────────────────────────────────


class TestReadIssueNode:
    @pytest.mark.asyncio
    async def test_happy_path_returns_running_status(self, sample_issue: GithubIssue) -> None:
        from src.agent.nodes import read_issue_node

        mock_response = MagicMock()
        mock_response.content = json.dumps({"can_proceed": True, "blocking_questions": []})

        with patch("src.agent.nodes._invoke_with_timeout", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            state: AgentState = {"issue": sample_issue}
            result = await read_issue_node(state)

        assert result["status"] == RunStatus.RUNNING
        assert "run_id" in result
        assert result["retry_count"] == 0
        assert result["attempt_history"] == []

    @pytest.mark.asyncio
    async def test_ambiguous_issue_returns_failed(self, sample_issue: GithubIssue) -> None:
        from src.agent.nodes import read_issue_node

        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {
                "can_proceed": False,
                "blocking_questions": ["Which endpoint?", "What version?"],
            }
        )

        with patch("src.agent.nodes._invoke_with_timeout", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            state: AgentState = {"issue": sample_issue}
            result = await read_issue_node(state)

        assert result["status"] == RunStatus.FAILED
        assert "failure_reason" in result

    @pytest.mark.asyncio
    async def test_llm_failure_fails_open(self, sample_issue: GithubIssue) -> None:
        """If clarifier LLM fails, we still proceed (fail-open)."""
        from src.agent.nodes import read_issue_node

        with patch("src.agent.nodes._invoke_with_timeout", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = TimeoutError("LLM timed out")
            state: AgentState = {"issue": sample_issue}
            result = await read_issue_node(state)

        # Should fail open and set RUNNING, not crash
        assert result["status"] == RunStatus.RUNNING


# ── test_plan_node ────────────────────────────────────────────────────────────


class TestPlanNode:
    @pytest.mark.asyncio
    async def test_returns_structured_plan(
        self,
        base_state: AgentState,
        sample_plan: TaskPlan,
    ) -> None:
        from src.agent.nodes import plan_node

        with (
            patch(
                "src.tools.filesystem.list_repo_tree",
                new_callable=AsyncMock,
                return_value="src/\n  services/\n    user_service.py",
            ),
            patch(
                "src.tools.filesystem.list_test_files",
                new_callable=AsyncMock,
                return_value=["tests/test_user_service.py"],
            ),
            patch(
                "src.agent.nodes._invoke_with_timeout",
                new_callable=AsyncMock,
                return_value=sample_plan,
            ),
        ):
            result = await plan_node(base_state)

        assert "plan" in result
        plan: TaskPlan = result["plan"]
        assert len(plan.tasks) >= 1
        assert result["current_task_index"] == 0

    @pytest.mark.asyncio
    async def test_propagates_llm_exception(self, base_state: AgentState) -> None:
        from anthropic import APIConnectionError

        from src.agent.nodes import plan_node

        with (
            patch("src.tools.filesystem.list_repo_tree", new_callable=AsyncMock, return_value=""),
            patch("src.tools.filesystem.list_test_files", new_callable=AsyncMock, return_value=[]),
            patch(
                "src.agent.nodes._invoke_with_timeout",
                new_callable=AsyncMock,
                side_effect=APIConnectionError(request=MagicMock()),
            ),
            pytest.raises(APIConnectionError),
        ):
            await plan_node(base_state)


# ── test_code_node ────────────────────────────────────────────────────────────


class TestCodeNode:
    @pytest.mark.asyncio
    async def test_returns_valid_patch(
        self,
        base_state: AgentState,
        sample_plan: TaskPlan,
        sample_patch: CodePatch,
    ) -> None:
        from src.agent.nodes import code_node

        state = {**base_state, "plan": sample_plan, "current_task_index": 0}

        with (
            patch("src.tools.search.find_relevant_files", new_callable=AsyncMock, return_value=[]),
            patch(
                "src.tools.filesystem.load_file_contexts",
                new_callable=AsyncMock,
                return_value=[
                    {
                        "path": "src/services/user_service.py",
                        "content": "def get_user(user_id):\n    ...",
                        "language": "python",
                        "size_bytes": 200,
                        "token_count": 50,
                    }
                ],
            ),
            patch(
                "src.agent.nodes._invoke_with_timeout",
                new_callable=AsyncMock,
                return_value=sample_patch,
            ),
        ):
            result = await code_node(state)

        assert "code_patch" in result
        patch_result: CodePatch = result["code_patch"]
        assert len(patch_result.patches) >= 1
        assert patch_result.patches[0].file_path == "src/services/user_service.py"


# ── test_correct_node ─────────────────────────────────────────────────────────


class TestCorrectNode:
    @pytest.mark.asyncio
    async def test_increments_retry_count(
        self,
        base_state: AgentState,
        sample_plan: TaskPlan,
        sample_patch: CodePatch,
        failing_test_result: TestResult,
    ) -> None:
        from src.agent.nodes import correct_node

        state = {
            **base_state,
            "plan": sample_plan,
            "current_task_index": 0,
            "code_patch": sample_patch,
            "test_result": failing_test_result,
            "retry_count": 0,
            "attempt_history": [],
        }

        corrected_patch = sample_patch.model_copy(update={"explanation": "Corrected version"})

        with (
            patch(
                "src.agent.nodes._classify_error",
                new_callable=AsyncMock,
                return_value=ErrorCategory.LOGIC_ERROR,
            ),
            patch(
                "src.agent.nodes._invoke_with_timeout",
                new_callable=AsyncMock,
                return_value=corrected_patch,
            ),
        ):
            result = await correct_node(state)

        assert result["retry_count"] == 1
        assert len(result["attempt_history"]) == 1
        assert result["last_error_category"] == ErrorCategory.LOGIC_ERROR

    @pytest.mark.asyncio
    async def test_includes_previous_attempts_in_history(
        self,
        base_state: AgentState,
        sample_plan: TaskPlan,
        sample_patch: CodePatch,
        failing_test_result: TestResult,
    ) -> None:
        from src.agent.nodes import correct_node

        existing_attempt = {
            "attempt_number": 1,
            "patch": sample_patch.model_dump(),
            "test_result": failing_test_result.model_dump(),
            "error_category": "logic_error",
        }

        state = {
            **base_state,
            "plan": sample_plan,
            "current_task_index": 0,
            "code_patch": sample_patch,
            "test_result": failing_test_result,
            "retry_count": 1,
            "attempt_history": [existing_attempt],
        }

        with (
            patch(
                "src.agent.nodes._classify_error",
                new_callable=AsyncMock,
                return_value=ErrorCategory.LOGIC_ERROR,
            ),
            patch(
                "src.agent.nodes._invoke_with_timeout",
                new_callable=AsyncMock,
                return_value=sample_patch,
            ),
        ):
            result = await correct_node(state)

        assert result["retry_count"] == 2
        assert len(result["attempt_history"]) == 2


# ── test_routing ──────────────────────────────────────────────────────────────


class TestRouting:
    def test_route_after_test_passes(
        self,
        base_state: AgentState,
        passing_test_result: TestResult,
    ) -> None:
        from src.agent.graph import route_after_test

        state = {**base_state, "test_result": passing_test_result, "retry_count": 0}
        assert route_after_test(state) == "open_pr"

    def test_route_after_test_retries_when_under_limit(
        self,
        base_state: AgentState,
        failing_test_result: TestResult,
    ) -> None:
        from src.agent.graph import route_after_test

        with patch("src.agent.graph.get_settings") as mock_settings:
            mock_settings.return_value.agent_max_retries = 3
            state = {**base_state, "test_result": failing_test_result, "retry_count": 1}
            assert route_after_test(state) == "correct"

    def test_route_after_test_fails_when_exhausted(
        self,
        base_state: AgentState,
        failing_test_result: TestResult,
    ) -> None:
        from src.agent.graph import route_after_test

        with patch("src.agent.graph.get_settings") as mock_settings:
            mock_settings.return_value.agent_max_retries = 3
            state = {**base_state, "test_result": failing_test_result, "retry_count": 3}
            assert route_after_test(state) == "fail"

    def test_route_after_read_aborts_on_failure(self) -> None:
        from langgraph.graph import END

        from src.agent.graph import route_after_read

        state: AgentState = {"status": RunStatus.FAILED}
        assert route_after_read(state) == END

    def test_route_after_read_proceeds_on_running(self) -> None:
        from src.agent.graph import route_after_read

        state: AgentState = {"status": RunStatus.RUNNING}
        assert route_after_read(state) == "plan"


class TestNodeExecutionDeepCoverage:
    @pytest.mark.asyncio
    async def test_invoke_with_timeout(self) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import _invoke_with_timeout

        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=SimpleNamespace(content="ok"))
        res = await _invoke_with_timeout(llm, [], timeout=5, run_name="test_call")
        assert res.content == "ok"

    @pytest.mark.asyncio
    async def test_test_node_patch_apply_failure(
        self,
        base_state: AgentState,
        sample_patch: CodePatch,
    ) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import test_node

        state = {**base_state, "code_patch": sample_patch}
        mock_sandbox = AsyncMock()
        mock_sandbox.apply_patches.return_value = SimpleNamespace(
            success=False, error="Conflict on line 12"
        )

        with patch("src.tools.sandbox.SandboxRunner") as mock_sb_cls:
            mock_sb_cls.return_value.__aenter__.return_value = mock_sandbox
            mock_sb_cls.return_value.__aexit__.return_value = None
            res = await test_node(state)

        assert res["test_result"].passed is False
        assert "Conflict" in res["test_result"].stderr

    @pytest.mark.asyncio
    async def test_test_node_installs_dependencies_and_runs_tests(
        self,
        base_state: AgentState,
        sample_patch: CodePatch,
        passing_test_result: TestResult,
    ) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import test_node

        patch_with_deps = sample_patch.model_copy(update={"new_dependencies": ["pytest-cov>=5.0"]})
        state = {**base_state, "code_patch": patch_with_deps}
        mock_sandbox = AsyncMock()
        mock_sandbox.apply_patches.return_value = SimpleNamespace(success=True, error="")
        mock_sandbox.run_tests.return_value = passing_test_result

        with patch("src.tools.sandbox.SandboxRunner") as mock_sb_cls:
            mock_sb_cls.return_value.__aenter__.return_value = mock_sandbox
            mock_sb_cls.return_value.__aexit__.return_value = None
            res = await test_node(state)

        assert res["test_result"].passed is True
        mock_sandbox.install_packages.assert_awaited_once_with(["pytest-cov>=5.0"])

    @pytest.mark.asyncio
    async def test_open_pr_node_and_fail_node(
        self,
        base_state: AgentState,
        sample_plan: TaskPlan,
        sample_patch: CodePatch,
        passing_test_result: TestResult,
    ) -> None:

        from src.agent.nodes import fail_node, open_pr_node
        from src.agent.state import PullRequest

        state = {
            **base_state,
            "plan": sample_plan,
            "code_patch": sample_patch,
            "test_result": passing_test_result,
        }

        pr_mock = PullRequest(
            pr_number=101,
            pr_url="https://github.com/owner/repo/pull/101",
            branch_name="agent/fix-42",
            is_draft=False,
            title="Fix bug",
            body="Fixed",
        )

        mock_gh = AsyncMock()
        mock_gh.create_branch_and_commit.return_value = "agent/fix-42"
        mock_gh.create_pull_request.return_value = pr_mock

        with (
            patch("src.tools.github.GitHubClient") as mock_gh_cls,
            patch(
                "src.agent.nodes._generate_pr_description", return_value="PR description generated"
            ),
            patch(
                "src.tools._repo_cache.get_local_repo_path",
                new_callable=AsyncMock,
                return_value="/tmp/repo",
            ),
            patch("src.tools.github._apply_patch_to_worktree"),
        ):
            mock_gh_cls.return_value.__aenter__.return_value = mock_gh
            mock_gh_cls.return_value.__aexit__.return_value = None
            pr_state = await open_pr_node(state)

        assert pr_state["pull_request"]["pr_number"] == 101
        assert pr_state["status"] == RunStatus.SUCCEEDED

        # Test fail_node
        with patch(
            "src.agent.nodes.open_pr_node",
            new_callable=AsyncMock,
            return_value={"pull_request": pr_mock},
        ):
            failed_res = await fail_node(state)
            assert failed_res["status"] == RunStatus.PARTIAL.value

        # Test fail_node when open_pr_node throws
        with patch(
            "src.agent.nodes.open_pr_node",
            new_callable=AsyncMock,
            side_effect=RuntimeError("PR error"),
        ):
            failed_res2 = await fail_node(state)
            assert failed_res2["status"] == RunStatus.FAILED.value

    @pytest.mark.asyncio
    async def test_classify_error_and_format_attempts(self) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import (
            _classify_error,
            _format_previous_attempts,
            _generate_pr_description,
        )
        from src.agent.state import CodePatch, ErrorCategory, GithubIssue, TestFailure, TestResult

        # Timeout classification
        res_timeout = await _classify_error(TestResult(passed=False, timed_out=True))
        assert res_timeout == ErrorCategory.TIMEOUT

        # Empty failures
        res_empty = await _classify_error(TestResult(passed=False, failures=[]))
        assert res_empty == ErrorCategory.UNKNOWN

        # LLM classification
        tf = TestFailure(
            test_id="test_1",
            test_name="test_1",
            error_message="SyntaxError in line 1",
            traceback="SyntaxError: invalid syntax",
        )
        with patch("src.agent.nodes._invoke_with_timeout", new_callable=AsyncMock) as mock_inv:
            mock_inv.return_value = SimpleNamespace(content="syntax_error")
            res_syntax = await _classify_error(TestResult(passed=False, failures=[tf]))
            assert res_syntax == ErrorCategory.SYNTAX_ERROR

            # Invalid LLM response falls back to UNKNOWN
            mock_inv.return_value = SimpleNamespace(content="not_a_category")
            res_fallback = await _classify_error(TestResult(passed=False, failures=[tf]))
            assert res_fallback == ErrorCategory.UNKNOWN

        # Format attempts
        assert _format_previous_attempts([]) == "No previous attempts."
        att = [
            {
                "attempt_number": 1,
                "error_category": "syntax_error",
                "patch": {"patches": [{"file_path": "app.py", "unified_diff": "diff..."}]},
            }
        ]
        assert "app.py" in _format_previous_attempts(att)

        # Generate PR description fallback
        issue = GithubIssue(
            issue_number=1,
            repo_full_name="o/r",
            title="Bug",
            body="",
            labels=[],
            comments=[],
            linked_prs=[],
            assignees=[],
            created_at=datetime.now(UTC),
            html_url="",
        )
        patch_obj = CodePatch(
            patches=[], explanation="none", test_command="pytest", new_dependencies=[]
        )
        with patch("src.agent.nodes._invoke_with_timeout", side_effect=RuntimeError("fail")):
            desc = await _generate_pr_description(issue, patch_obj, TestResult(passed=True))
            assert "Automated fix for #1" in desc

        with patch("src.agent.nodes._invoke_with_timeout", new_callable=AsyncMock) as mock_desc:
            mock_desc.return_value = SimpleNamespace(content="LLM generated PR body")
            desc_success = await _generate_pr_description(issue, patch_obj, TestResult(passed=True))
            assert desc_success == "LLM generated PR body"

    @pytest.mark.asyncio
    async def test_code_and_correct_node_error_handling(
        self,
        base_state: AgentState,
        sample_plan: TaskPlan,
        sample_patch: CodePatch,
        failing_test_result: TestResult,
    ) -> None:

        from src.agent.nodes import code_node, correct_node, open_pr_node
        from src.agent.state import PullRequest

        state = {
            **base_state,
            "plan": sample_plan,
            "code_patch": sample_patch,
            "test_result": failing_test_result,
            "current_task_index": 0,
        }

        with (
            patch("src.tools.search.find_relevant_files", new_callable=AsyncMock, return_value=[]),
            patch(
                "src.tools.filesystem.load_file_contexts", new_callable=AsyncMock, return_value=[]
            ),
            patch(
                "src.agent.nodes._invoke_with_timeout", side_effect=RuntimeError("coder LLM down")
            ),
            pytest.raises(RuntimeError, match="coder LLM down"),
        ):
            await code_node(state)

        with (
            patch("src.tools.search.find_relevant_files", new_callable=AsyncMock, return_value=[]),
            patch(
                "src.tools.filesystem.load_file_contexts", new_callable=AsyncMock, return_value=[]
            ),
            patch(
                "src.agent.nodes._classify_error",
                new=AsyncMock(return_value=ErrorCategory.LOGIC_ERROR),
            ),
            patch(
                "src.agent.nodes._invoke_with_timeout", side_effect=RuntimeError("correct LLM down")
            ),
            pytest.raises(RuntimeError, match="correct LLM down"),
        ):
            await correct_node(state)

        # Test draft PR in open_pr_node
        pr_mock = PullRequest(
            pr_number=102,
            pr_url="https://github.com/owner/repo/pull/102",
            branch_name="agent/fix-draft",
            is_draft=True,
            title="Fix bug (draft)",
            body="Draft",
        )
        mock_gh = AsyncMock()
        mock_gh.create_pull_request.return_value = pr_mock
        with (
            patch("src.tools.github.GitHubClient"),
            patch(
                "src.tools._repo_cache.get_local_repo_path",
                new_callable=AsyncMock,
                return_value="/tmp/repo",
            ),
            patch("src.tools.github._apply_patch_to_worktree"),
        ):
            draft_res = await open_pr_node(state)
            assert draft_res["status"] == RunStatus.PARTIAL

    def test_build_llm_providers(self) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import _build_llm

        # Anthropic
        s_anthropic = SimpleNamespace(
            llm_provider="anthropic",
            anthropic_model="claude-3-5-sonnet-latest",
            anthropic_api_key_value="test-ant-key",
            anthropic_max_tokens=4096,
            anthropic_timeout_seconds=60,
            anthropic_max_retries=2,
        )
        with (
            patch("src.agent.nodes.get_settings", return_value=s_anthropic),
            patch("src.agent.nodes.ChatAnthropic") as mock_ant,
        ):
            llm_ant = _build_llm()
            assert llm_ant == mock_ant.return_value

        # Gemini
        s_gemini = SimpleNamespace(
            llm_provider="gemini",
            gemini_model="gemini-1.5-pro",
            gemini_api_key_value="test-gemini-key",
            anthropic_max_tokens=4096,
        )
        with (
            patch("src.agent.nodes.get_settings", return_value=s_gemini),
            patch("langchain_google_genai.ChatGoogleGenerativeAI") as mock_gem,
        ):
            llm_gem = _build_llm()
            assert llm_gem == mock_gem.return_value

        # OpenAI
        s_openai = SimpleNamespace(
            llm_provider="openai",
            openai_model="gpt-4o",
            openai_api_key_value="test-openai-key",
            anthropic_max_tokens=4096,
        )
        with (
            patch("src.agent.nodes.get_settings", return_value=s_openai),
            patch("langchain_openai.ChatOpenAI") as mock_oai,
        ):
            llm_oai = _build_llm()
            assert llm_oai == mock_oai.return_value

    def test_update_tokens_and_budget(self) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import _update_tokens_and_budget

        s = SimpleNamespace(agent_max_tokens_per_run=1000, agent_max_cost_per_run_usd=1.0)
        with patch("src.agent.nodes.get_settings", return_value=s):
            # Normal usage
            u1 = _update_tokens_and_budget({"total_tokens_used": 200}, 300)
            assert u1["total_tokens_used"] == 500
            assert u1["budget_exceeded"] is False

            # Exceeded usage
            u2 = _update_tokens_and_budget({"total_tokens_used": 900}, 200)
            assert u2["total_tokens_used"] == 1100
            assert u2["budget_exceeded"] is True

    @pytest.mark.asyncio
    async def test_repro_test_node(self, sample_issue: GithubIssue) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import repro_test_node
        from src.agent.state import ReproTest

        state: AgentState = {
            "issue": sample_issue,
            "run_id": "run-repro",
            "detected_language": "python",
        }

        # 1. Disabled
        s_disabled = SimpleNamespace(agent_enable_repro_test=False)
        with patch("src.agent.nodes.get_settings", return_value=s_disabled):
            res_disabled = await repro_test_node(state)
            assert res_disabled == {}

        # 2. Enabled success
        s_enabled = SimpleNamespace(
            agent_enable_repro_test=True,
            agent_max_tokens_per_run=150000,
            agent_max_cost_per_run_usd=3.0,
            llm_provider="anthropic",
            anthropic_model="test",
            anthropic_api_key_value="test",
            anthropic_timeout_seconds=30,
            anthropic_max_retries=1,
            anthropic_max_tokens=2048,
        )
        mock_repro = ReproTest(
            file_path="tests/test_repro_issue_42.py",
            test_code="def test_repro(): assert False",
            explanation="Verifies the bug",
        )
        with (
            patch("src.agent.nodes.get_settings", return_value=s_enabled),
            patch("src.tools.filesystem.list_test_files", new_callable=AsyncMock, return_value=[]),
            patch(
                "src.agent.nodes._invoke_with_timeout",
                new_callable=AsyncMock,
                return_value=mock_repro,
            ),
        ):
            res = await repro_test_node(state)
            assert res.get("repro_test_code") == "def test_repro(): assert False"

        # 3. LLM error fails gracefully
        with (
            patch("src.agent.nodes.get_settings", return_value=s_enabled),
            patch("src.tools.filesystem.list_test_files", new_callable=AsyncMock, return_value=[]),
            patch(
                "src.agent.nodes._invoke_with_timeout",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM error"),
            ),
        ):
            res_err = await repro_test_node(state)
            assert res_err == {}

    @pytest.mark.asyncio
    async def test_refine_from_review_node(self, sample_issue: GithubIssue) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import refine_from_review_node
        from src.agent.state import CodePatch

        s_enabled = SimpleNamespace(
            agent_max_tokens_per_run=150000,
            agent_max_cost_per_run_usd=3.0,
            llm_provider="anthropic",
            anthropic_model="test",
            anthropic_api_key_value="test",
            anthropic_timeout_seconds=30,
            anthropic_max_retries=1,
            anthropic_max_tokens=2048,
        )
        state: AgentState = {
            "issue": sample_issue,
            "run_id": "run-refine",
            "code_patch": CodePatch(patches=[], explanation="Initial"),
            "review_feedback": "Please add null checks",
        }
        mock_refined = CodePatch(patches=[], explanation="Refined with checks")

        with (
            patch("src.agent.nodes.get_settings", return_value=s_enabled),
            patch(
                "src.agent.nodes._invoke_with_timeout",
                new_callable=AsyncMock,
                return_value=mock_refined,
            ),
        ):
            res = await refine_from_review_node(state)
            assert res["code_patch"].explanation == "Refined with checks"

    @pytest.mark.asyncio
    async def test_code_node_bundles_repro_test(
        self, sample_issue: GithubIssue, sample_plan: TaskPlan
    ) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import code_node
        from src.agent.state import CodePatch

        s = SimpleNamespace(
            agent_max_files_in_context=10,
            agent_max_tokens_per_run=150000,
            agent_max_cost_per_run_usd=3.0,
            llm_provider="anthropic",
            anthropic_model="test",
            anthropic_api_key_value="test",
            anthropic_timeout_seconds=30,
            anthropic_max_retries=1,
            anthropic_max_tokens=2048,
        )
        state: AgentState = {
            "issue": sample_issue,
            "plan": sample_plan,
            "run_id": "run-code-repro",
            "detected_language": "typescript",
            "repro_test_code": "test('repro', () => { expect(1).toBe(2); });",
        }
        mock_patch = CodePatch(patches=[], explanation="Implemented fix")

        with (
            patch("src.agent.nodes.get_settings", return_value=s),
            patch("src.tools.search.find_relevant_files", new_callable=AsyncMock, return_value=[]),
            patch(
                "src.tools.filesystem.load_file_contexts",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "src.agent.nodes._invoke_with_timeout",
                new_callable=AsyncMock,
                return_value=mock_patch,
            ),
        ):
            res = await code_node(state)
            patches = res["code_patch"].patches
            assert len(patches) == 1
            assert "test_repro_issue_42.test.ts" in patches[0].file_path

    @pytest.mark.asyncio
    async def test_open_pr_node_with_budget_exceeded(self, sample_issue: GithubIssue) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import open_pr_node
        from src.agent.state import CodePatch, PullRequest, TestResult

        s = SimpleNamespace(
            github_pr_label="agent-fix",
            agent_max_retries=3,
            agent_max_tokens_per_run=150000,
            agent_max_cost_per_run_usd=3.0,
        )
        mock_pr = PullRequest(
            pr_number=88,
            pr_url="https://github.com/owner/repo/pull/88",
            branch_name="agent-branch",
            is_draft=False,
            title="Fix bug",
            body="Initial body",
        )
        state: AgentState = {
            "issue": sample_issue,
            "run_id": "run-budget-pr",
            "code_patch": CodePatch(patches=[], explanation="Patch"),
            "test_result": TestResult(passed=True),
            "budget_exceeded": True,
            "total_tokens_used": 160000,
            "cumulative_cost_usd": 3.2,
        }

        with (
            patch("src.agent.nodes.get_settings", return_value=s),
            patch(
                "src.agent.nodes._generate_pr_description",
                new_callable=AsyncMock,
                return_value="Base description",
            ),
            patch("src.tools.github.GitHubClient") as mock_gh_cls,
        ):
            mock_gh = AsyncMock()
            mock_gh.create_pull_request.return_value = mock_pr
            mock_gh_cls.return_value.__aenter__.return_value = mock_gh

            res = await open_pr_node(state)
            assert res["status"] == "succeeded"
            call_body = mock_gh.create_pull_request.call_args[1]["body"]
            assert "Budget Cap Reached" in call_body

    @pytest.mark.asyncio
    async def test_fail_node_with_budget_exceeded(self, sample_issue: GithubIssue) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import fail_node

        s = SimpleNamespace(agent_max_retries=3, agent_pr_draft_on_failure=False)
        state: AgentState = {
            "issue": sample_issue,
            "run_id": "run-fail-budget",
            "budget_exceeded": True,
            "total_tokens_used": 155000,
            "cumulative_cost_usd": 3.10,
        }
        with patch("src.agent.nodes.get_settings", return_value=s):
            res = await fail_node(state)
            assert res["status"] == "failed"
            assert "Budget cap reached" in res["failure_reason"]

    @pytest.mark.asyncio
    async def test_read_issue_node_with_detected_ecosystem(
        self, sample_issue: GithubIssue, tmp_path: Path
    ) -> None:
        from src.agent.nodes import read_issue_node

        (tmp_path / "go.mod").write_text("module example.com/pkg\n")
        mock_response = MagicMock()
        mock_response.content = json.dumps({"can_proceed": True, "blocking_questions": []})

        with (
            patch(
                "src.agent.nodes._invoke_with_timeout",
                new_callable=AsyncMock,
                return_value=mock_response,
            ),
            patch(
                "src.tools._repo_cache.get_local_repo_path",
                new_callable=AsyncMock,
                return_value=str(tmp_path),
            ),
        ):
            res = await read_issue_node({"issue": sample_issue})
            assert res["detected_language"] == "go"
            assert res["detected_test_command"] == "go test ./..."

    @pytest.mark.asyncio
    async def test_test_node_uses_detected_test_command(self, sample_issue: GithubIssue) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import test_node
        from src.agent.state import CodePatch, TestResult

        state: AgentState = {
            "issue": sample_issue,
            "run_id": "run-test-cmd",
            "code_patch": CodePatch(patches=[], explanation="Fix", test_command="pytest"),
            "detected_test_command": "npm test",
        }
        mock_runner = AsyncMock()
        mock_runner.apply_patches.return_value = SimpleNamespace(success=True, error="")
        mock_runner.run_tests.return_value = TestResult(passed=True, total=1)

        with patch("src.tools.sandbox.SandboxRunner") as mock_runner_cls:
            mock_runner_cls.return_value.__aenter__.return_value = mock_runner
            res = await test_node(state)
            assert res["test_result"].passed is True
            mock_runner.run_tests.assert_called_once_with(test_command="npm test")

    @pytest.mark.asyncio
    async def test_refine_from_review_node_error_raises(self, sample_issue: GithubIssue) -> None:
        from types import SimpleNamespace

        from src.agent.nodes import refine_from_review_node
        from src.agent.state import CodePatch

        s_enabled = SimpleNamespace(
            agent_max_tokens_per_run=150000,
            agent_max_cost_per_run_usd=3.0,
            llm_provider="anthropic",
            anthropic_model="test",
            anthropic_api_key_value="test",
            anthropic_timeout_seconds=30,
            anthropic_max_retries=1,
            anthropic_max_tokens=2048,
        )
        state: AgentState = {
            "issue": sample_issue,
            "run_id": "run-refine-err",
            "code_patch": CodePatch(patches=[], explanation="Initial"),
            "review_feedback": "Crash feedback",
        }
        with (
            patch("src.agent.nodes.get_settings", return_value=s_enabled),
            patch(
                "src.agent.nodes._invoke_with_timeout",
                new_callable=AsyncMock,
                side_effect=RuntimeError("LLM fail"),
            ),
            pytest.raises(RuntimeError, match="LLM fail"),
        ):
            await refine_from_review_node(state)
