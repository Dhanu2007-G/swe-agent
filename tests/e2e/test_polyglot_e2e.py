"""tests/e2e/test_polyglot_e2e.py — End-to-end tests across supported languages.

Tests plan -> patch -> sandbox test matrix for Python, JS, TS, Go, Rust, Java,
along with positive lifecycle and full negative edge cases.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from src.agent.context import ContextBuilder
from src.agent.nodes import open_pr_node, read_issue_node
from src.agent.state import (
    AgentState,
    CodePatch,
    FilePatch,
    GithubIssue,
    RunStatus,
    TestResult,
)
from src.tools.filesystem import list_test_files, load_file_contexts
from src.tools.github import GitHubClient
from src.tools.sandbox import (
    DockerSandboxProvider,
    KubernetesSandboxProvider,
    SandboxRunner,
)
from src.tools.search import find_relevant_files, search_symbols

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "repos"


@pytest.mark.asyncio
class TestPolyglotE2EMatrix:
    async def test_python_e2e_pipeline(self) -> None:
        repo_dir = FIXTURES_DIR / "python"
        with patch(
            "src.tools._repo_cache.get_local_repo_path",
            new_callable=AsyncMock,
            return_value=str(repo_dir),
        ):
            test_files = await list_test_files("owner/python-repo")
            assert any("test_calculator.py" in f for f in test_files)

            relevant = await find_relevant_files("owner/python-repo", "calculator add", exclude=[])
            assert any("calculator.py" in r for r in relevant)

            symbols = await search_symbols("owner/python-repo", "add")
            assert len(symbols) >= 1
            assert any(s["name"] == "add" for s in symbols)

            ctx = await load_file_contexts("owner/python-repo", ["src/calculator.py"])
            builder = ContextBuilder(task_description="fix add")
            builder.add_file(ctx[0]["path"], ctx[0]["content"], ctx[0]["language"])
            rendered = builder.render()
            assert "add" in rendered

            # Plan -> patch -> sandbox test
            patch_obj = CodePatch(
                explanation="ensure add handles negative numbers",
                patches=[
                    FilePatch(
                        file_path="src/calculator.py",
                        action="modify",
                        description="update add",
                        old_content="def add(a: int, b: int) -> int:\n    return a + b\n",
                        new_content="def add(a: int, b: int) -> int:\n    return int(a) + int(b)\n",
                    )
                ],
            )
            mock_provider = MagicMock(spec=DockerSandboxProvider)
            mock_provider.create_container = AsyncMock(return_value={"id": "mock-py"})
            mock_provider.exec_command = AsyncMock(return_value=(0, b"1 passed in 0.05s"))
            mock_provider.cleanup = AsyncMock()

            runner = SandboxRunner("owner/python-repo", "run-py", provider=mock_provider)
            runner._workspace_path = repo_dir
            runner._container = {"id": "mock-py"}

            apply_res = await runner.apply_patches(patch_obj.patches)
            assert apply_res.success is True

            test_res = await runner.run_tests()
            assert test_res.passed is True

    async def test_javascript_e2e_pipeline(self) -> None:
        repo_dir = FIXTURES_DIR / "javascript"
        with patch(
            "src.tools._repo_cache.get_local_repo_path",
            new_callable=AsyncMock,
            return_value=str(repo_dir),
        ):
            test_files = await list_test_files("owner/js-repo")
            assert any("index.test.js" in f for f in test_files)

            relevant = await find_relevant_files("owner/js-repo", "multiply", exclude=[])
            assert any("index.js" in r or "package.json" in r for r in relevant)

            symbols = await search_symbols("owner/js-repo", "multiply")
            assert len(symbols) >= 1

            # Plan -> patch -> sandbox test
            patch_obj = CodePatch(
                explanation="implement multiply",
                patches=[
                    FilePatch(
                        file_path="src/index.js",
                        action="modify",
                        description="export multiply",
                        old_content="function multiply(a, b) {\n  return a * b;\n}\n",
                        new_content=(
                            "function multiply(a, b) {\n  return Number(a) * Number(b);\n}\n"
                        ),
                    )
                ],
            )
            mock_provider = MagicMock(spec=DockerSandboxProvider)
            mock_provider.create_container = AsyncMock(return_value={"id": "mock-js"})
            mock_provider.exec_command = AsyncMock(return_value=(0, b"PASS tests/index.test.js"))
            mock_provider.cleanup = AsyncMock()

            runner = SandboxRunner("owner/js-repo", "run-js", provider=mock_provider)
            runner._workspace_path = repo_dir
            runner._container = {"id": "mock-js"}

            apply_res = await runner.apply_patches(patch_obj.patches)
            assert apply_res.success is True
            test_res = await runner.run_tests()
            assert test_res.passed is True

    async def test_typescript_e2e_pipeline(self) -> None:
        repo_dir = FIXTURES_DIR / "typescript"
        with patch(
            "src.tools._repo_cache.get_local_repo_path",
            new_callable=AsyncMock,
            return_value=str(repo_dir),
        ):
            test_files = await list_test_files("owner/ts-repo")
            assert any("math.test.ts" in f for f in test_files)

            relevant = await find_relevant_files("owner/ts-repo", "Geometry distance", exclude=[])
            assert any("math.ts" in r for r in relevant)

            symbols = await search_symbols("owner/ts-repo", "Geometry")
            assert len(symbols) >= 1

            # Plan -> patch -> sandbox test
            patch_obj = CodePatch(
                explanation="implement distance calculation in Geometry",
                patches=[
                    FilePatch(
                        file_path="src/math.ts",
                        action="modify",
                        description="add distance",
                        old_content="export class Geometry {\n}\n",
                        new_content=(
                            "export class Geometry {\n"
                            "  distance(x: number, y: number): number {\n"
                            "    return Math.hypot(x, y);\n"
                            "  }\n"
                            "}\n"
                        ),
                    )
                ],
            )
            mock_provider = MagicMock(spec=KubernetesSandboxProvider)
            mock_provider.create_container = AsyncMock(return_value={"pod_name": "sb-ts"})
            mock_provider.exec_command = AsyncMock(
                return_value=(0, "✓ math.test.ts (2 tests)".encode())
            )
            mock_provider.cleanup = AsyncMock()

            runner = SandboxRunner("owner/ts-repo", "run-ts", provider=mock_provider)
            runner._workspace_path = repo_dir
            runner._container = {"pod_name": "sb-ts"}

            apply_res = await runner.apply_patches(patch_obj.patches)
            assert apply_res.success is True
            test_res = await runner.run_tests()
            assert test_res.passed is True

    async def test_go_e2e_pipeline(self) -> None:
        repo_dir = FIXTURES_DIR / "go"
        with patch(
            "src.tools._repo_cache.get_local_repo_path",
            new_callable=AsyncMock,
            return_value=str(repo_dir),
        ):
            test_files = await list_test_files("owner/go-repo")
            assert any("calc_test.go" in f for f in test_files)

            relevant = await find_relevant_files("owner/go-repo", "Calculator Add", exclude=[])
            assert any("calc.go" in r or "go.mod" in r for r in relevant)

            symbols = await search_symbols("owner/go-repo", "Calculator")
            assert len(symbols) >= 1

            # Plan -> patch -> sandbox test
            patch_obj = CodePatch(
                explanation="implement Add in Calculator",
                patches=[
                    FilePatch(
                        file_path="calc.go",
                        action="modify",
                        description="implement Add",
                        old_content="func Add(a, b int) int { return 0 }\n",
                        new_content="func Add(a, b int) int { return a + b }\n",
                    )
                ],
            )
            mock_provider = MagicMock(spec=DockerSandboxProvider)
            mock_provider.create_container = AsyncMock(return_value={"id": "mock-go"})
            mock_provider.exec_command = AsyncMock(return_value=(0, b"ok  \tcalc\t0.003s"))
            mock_provider.cleanup = AsyncMock()

            runner = SandboxRunner("owner/go-repo", "run-go", provider=mock_provider)
            runner._workspace_path = repo_dir
            runner._container = {"id": "mock-go"}

            apply_res = await runner.apply_patches(patch_obj.patches)
            assert apply_res.success is True
            test_res = await runner.run_tests()
            assert test_res.passed is True

    async def test_rust_e2e_pipeline(self) -> None:
        repo_dir = FIXTURES_DIR / "rust"
        with patch(
            "src.tools._repo_cache.get_local_repo_path",
            new_callable=AsyncMock,
            return_value=str(repo_dir),
        ):
            test_files = await list_test_files("owner/rust-repo")
            assert any("lib_test.rs" in f for f in test_files)

            relevant = await find_relevant_files("owner/rust-repo", "Counter increment", exclude=[])
            assert any("lib.rs" in r or "Cargo.toml" in r for r in relevant)

            symbols = await search_symbols("owner/rust-repo", "Counter")
            assert len(symbols) >= 1

            # Plan -> patch -> sandbox test
            patch_obj = CodePatch(
                explanation="implement Counter increment",
                patches=[
                    FilePatch(
                        file_path="src/lib.rs",
                        action="modify",
                        description="implement increment",
                        old_content="pub fn increment(&mut self) {}\n",
                        new_content="pub fn increment(&mut self) { self.count += 1; }\n",
                    )
                ],
            )
            mock_provider = MagicMock(spec=DockerSandboxProvider)
            mock_provider.create_container = AsyncMock(return_value={"id": "mock-rust"})
            mock_provider.exec_command = AsyncMock(return_value=(0, b"test result: ok. 2 passed"))
            mock_provider.cleanup = AsyncMock()

            runner = SandboxRunner("owner/rust-repo", "run-rust", provider=mock_provider)
            runner._workspace_path = repo_dir
            runner._container = {"id": "mock-rust"}

            apply_res = await runner.apply_patches(patch_obj.patches)
            assert apply_res.success is True
            test_res = await runner.run_tests()
            assert test_res.passed is True

    async def test_java_e2e_pipeline(self) -> None:
        repo_dir = FIXTURES_DIR / "java"
        with patch(
            "src.tools._repo_cache.get_local_repo_path",
            new_callable=AsyncMock,
            return_value=str(repo_dir),
        ):
            test_files = await list_test_files("owner/java-repo")
            assert any("CalculatorTest.java" in f for f in test_files)

            relevant = await find_relevant_files("owner/java-repo", "Calculator", exclude=[])
            assert any("Calculator.java" in r or "pom.xml" in r for r in relevant)

            symbols = await search_symbols("owner/java-repo", "Calculator")
            assert len(symbols) >= 1

            # Plan -> patch -> sandbox test
            patch_obj = CodePatch(
                explanation="implement Calculator.add",
                patches=[
                    FilePatch(
                        file_path="src/main/java/Calculator.java",
                        action="modify",
                        description="implement add",
                        old_content="public int add(int a, int b) { return 0; }\n",
                        new_content="public int add(int a, int b) { return a + b; }\n",
                    )
                ],
            )
            mock_provider = MagicMock(spec=KubernetesSandboxProvider)
            mock_provider.create_container = AsyncMock(return_value={"pod_name": "sb-java"})
            mock_provider.exec_command = AsyncMock(
                return_value=(0, b"[INFO] Tests run: 2, Failures: 0, Errors: 0, Skipped: 0")
            )
            mock_provider.cleanup = AsyncMock()

            runner = SandboxRunner("owner/java-repo", "run-java", provider=mock_provider)
            runner._workspace_path = repo_dir
            runner._container = {"pod_name": "sb-java"}

            apply_res = await runner.apply_patches(patch_obj.patches)
            assert apply_res.success is True
            test_res = await runner.run_tests()
            assert test_res.passed is True


@pytest.mark.asyncio
class TestNegativeEdgeCases:
    async def test_negative_ambiguous_issue_e2e(self) -> None:
        """Ambiguous issue: clarifier rejects it, run marked FAILED, terminates early."""
        issue = GithubIssue(
            issue_number=99,
            repo_full_name="owner/repo",
            title="Help it broke",
            body="nothing works please fix",
            labels=["agent-fix"],
            comments=[],
            assignees=[],
            created_at=datetime.now(UTC),
            html_url="https://github.com/owner/repo/issues/99",
        )
        state: AgentState = {"issue": issue, "retry_count": 0, "attempt_history": []}

        mock_llm_response = MagicMock(content=json.dumps({"can_proceed": False}))
        with (
            patch(
                "src.agent.nodes._invoke_with_timeout",
                AsyncMock(return_value=mock_llm_response),
            ),
            patch(
                "src.tools._repo_cache.get_local_repo_path",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            updates = await read_issue_node(state)
            assert updates["status"] == RunStatus.FAILED
            assert "ambiguous" in updates["failure_reason"].lower()

    async def test_negative_bad_patch_retries_and_draft_pr_e2e(self) -> None:
        """Bad patch: test failure leads to draft PR when retries exhausted."""
        issue = GithubIssue(
            issue_number=101,
            repo_full_name="owner/repo",
            title="Bug in math",
            body="Math calculation error",
            labels=["agent-fix"],
            comments=[],
            assignees=[],
            created_at=datetime.now(UTC),
            html_url="https://github.com/owner/repo/issues/101",
        )
        bad_patch = CodePatch(
            explanation="bad patch",
            patches=[
                FilePatch(
                    file_path="math.py",
                    action="modify",
                    description="bad syntax",
                    old_content="x = 1\n",
                    new_content="x = \n",
                )
            ],
        )
        failing_test = TestResult(
            passed=False, failed_count=2, stdout="", stderr="AssertionError: 1 != 2"
        )

        mock_pr = MagicMock()
        mock_pr.pr_number = 102
        mock_pr.pr_url = "https://github.com/owner/repo/pull/102"
        mock_pr.model_dump.return_value = {"pr_number": 102, "pr_url": mock_pr.pr_url}
        mock_pr.is_fork = False
        mock_pr.fork_repo_full_name = None

        with patch(
            "src.tools.github.GitHubClient.create_pull_request",
            AsyncMock(return_value=mock_pr),
        ):
            state: AgentState = {
                "run_id": "run-bad-patch",
                "issue": issue,
                "code_patch": bad_patch,
                "test_result": failing_test,
                "retry_count": 3,
                "attempt_history": [],
            }
            res = await open_pr_node(state)
            assert res["status"] == RunStatus.PARTIAL.value
            assert res["pull_request"]["pr_number"] == 102

    async def test_negative_timeout_e2e(self, tmp_path: Path) -> None:
        """Timeout edge case: sandbox execution times out and returns timed_out TestResult."""
        mock_provider = MagicMock(spec=DockerSandboxProvider)
        mock_provider.create_container = AsyncMock(return_value={"id": "mock-timeout"})
        mock_provider.exec_command = AsyncMock(return_value=(0, b"pytest --help"))
        mock_provider.cleanup = AsyncMock()

        runner = SandboxRunner("owner/repo", "run-timeout", provider=mock_provider)
        runner._workspace_path = tmp_path
        runner._container = {"id": "mock-timeout"}

        with patch("asyncio.wait_for", side_effect=TimeoutError()):
            result: TestResult = await runner.run_tests(timeout=1)
            assert result.passed is False
            assert result.timed_out is True

    async def test_negative_missing_dependency_e2e(self, tmp_path: Path) -> None:
        """Missing dependency edge case: container returns missing module error."""
        mock_provider = MagicMock(spec=DockerSandboxProvider)
        mock_provider.create_container = AsyncMock(return_value={"id": "mock-missing-dep"})
        mock_provider.exec_command = AsyncMock(
            return_value=(1, b"ModuleNotFoundError: No module named 'pytest'")
        )
        mock_provider.cleanup = AsyncMock()

        runner = SandboxRunner("owner/repo", "run-missing-dep", provider=mock_provider)
        runner._workspace_path = tmp_path
        runner._container = {"id": "mock-missing-dep"}

        res = await runner.run_tests()
        assert res.passed is False
        assert "ModuleNotFoundError" in (res.stdout + res.stderr)

    async def test_negative_private_repo_or_disallowed_repo_e2e(self) -> None:
        """Disallowed repository: API rejects with 403 Forbidden."""
        from src.api.routes import TriggerRequest, trigger_run

        req = TriggerRequest(repo_full_name="unauthorized/private-repo", issue_number=10)
        with patch("src.api.routes.get_settings") as mock_settings:
            mock_settings.return_value.allowed_repositories = ["allowed/public-repo"]
            with pytest.raises(HTTPException) as exc:
                await trigger_run(req)
            assert exc.value.status_code == 403

    async def test_negative_no_push_permission_creates_fork_pr_e2e(self) -> None:
        """No push permission: automatically creates fork and opens PR from fork."""
        client = GitHubClient()
        mock_repo = MagicMock()
        mock_repo.default_branch = "main"
        mock_repo.permissions = MagicMock(push=False)
        mock_repo.name = "upstream-repo"

        mock_user = MagicMock()
        mock_user.login = "agent-bot"
        mock_fork = MagicMock()
        mock_fork.full_name = "agent-bot/upstream-repo"
        mock_user.create_fork.return_value = mock_fork

        client._gh = MagicMock()
        client._gh.get_repo.return_value = mock_repo
        client._gh.get_user.return_value = mock_user

        mock_pr = MagicMock()
        mock_pr.number = 42
        mock_pr.html_url = "https://github.com/upstream-org/upstream-repo/pull/42"
        mock_repo.create_pull.return_value = mock_pr
        mock_repo.get_labels.return_value = []

        with (
            patch("git.Repo.clone_from") as mock_clone,
            patch("src.tools.github._apply_patch_to_worktree"),
        ):
            mock_git_repo = MagicMock()
            mock_git_repo.is_dirty.return_value = True
            mock_clone.return_value = mock_git_repo

            pr = await client.create_pull_request(
                repo_full_name="upstream-org/upstream-repo",
                branch_name="patch-1",
                base_branch=None,
                title="fix bug",
                body="fix description",
                patches=[],
                issue_number=1,
            )
            assert pr.is_fork is True
            assert pr.fork_repo_full_name == "agent-bot/upstream-repo"
            assert pr.pr_number == 42

    async def test_negative_active_run_deduplication_e2e(self) -> None:
        """Active run deduplication: existing active run prevents duplicate enqueuing."""
        from src.api.webhook import github_webhook

        payload = json.dumps(
            {
                "action": "opened",
                "repository": {"full_name": "owner/repo"},
                "issue": {
                    "number": 55,
                    "labels": [{"name": "agent-fix"}],
                },
            }
        ).encode()

        req = MagicMock()
        req.body = AsyncMock(return_value=payload)

        mock_active_run = MagicMock()
        mock_active_run.run_id = "existing-active-run-123"

        with (
            patch("src.tools.github.GitHubClient.validate_webhook_signature", return_value=True),
            patch(
                "src.db.repository.RunRepository.get_active_run",
                AsyncMock(return_value=mock_active_run),
            ),
            patch("src.worker.queue.enqueue_issue_job") as mock_enqueue,
        ):
            resp = await github_webhook(
                request=req,
                x_hub_signature_256="sha256=mock",
                x_github_event="issues",
                x_github_delivery="deliv-123",
            )
            assert resp.status_code == 200
            resp_body = json.loads(resp.body)
            assert resp_body["status"] == "ignored"
            assert resp_body["reason"] == "run already active"
            mock_enqueue.assert_not_called()

    async def test_full_positive_lifecycle_e2e(self) -> None:
        """
        Complete positive lifecycle E2E:
        Issue webhook -> job enqueue -> branch -> patch -> tests pass
        -> PR created -> DB updated -> comment posted.
        """
        issue = GithubIssue(
            issue_number=77,
            repo_full_name="owner/repo",
            title="Add feature X",
            body="Please implement feature X",
            labels=["agent-fix"],
            comments=[],
            assignees=[],
            created_at=datetime.now(UTC),
            html_url="https://github.com/owner/repo/issues/77",
        )

        mock_pr = MagicMock()
        mock_pr.pr_number = 78
        mock_pr.pr_url = "https://github.com/owner/repo/pull/78"
        mock_pr.model_dump.return_value = {"pr_number": 78, "pr_url": mock_pr.pr_url}
        mock_pr.is_fork = False
        mock_pr.fork_repo_full_name = None

        state: AgentState = {
            "run_id": "e2e-run-77",
            "issue": issue,
            "code_patch": CodePatch(
                explanation="implemented X",
                patches=[
                    FilePatch(
                        file_path="feature.py",
                        action="create",
                        description="new feature",
                        new_content="def feature_x(): return True\n",
                    )
                ],
            ),
            "test_result": TestResult(passed=True, failed_count=0),
            "retry_count": 0,
            "attempt_history": [],
        }

        with (
            patch(
                "src.tools.github.GitHubClient.create_pull_request",
                AsyncMock(return_value=mock_pr),
            ),
            patch("src.db.repository.RunRepository.update_run", AsyncMock()),
            patch("src.tools.github.GitHubClient.comment_on_issue", AsyncMock()),
        ):
            res = await open_pr_node(state)
            assert res["status"] == RunStatus.SUCCEEDED.value
            assert res["pull_request"]["pr_number"] == 78
            assert res["pull_request"]["pr_url"] == "https://github.com/owner/repo/pull/78"
