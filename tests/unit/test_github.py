from __future__ import annotations

import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from github import GithubException

from src.agent.state import FilePatch

if TYPE_CHECKING:
    from pathlib import Path


def make_settings() -> SimpleNamespace:
    return SimpleNamespace(
        github_token_value="ghp_test_token",
        github_webhook_secret_value="webhook-secret",
        github_bot_username="swe-agent[bot]",
        github_app_id=None,
        github_app_private_key=None,
    )


async def run_now(fn: object) -> object:
    return fn()


@contextmanager
def fixed_tempdir(path: Path) -> object:
    path.mkdir(parents=True, exist_ok=True)
    yield str(path)


@contextmanager
def case_dir(prefix: str) -> object:
    from pathlib import Path

    root = (
        Path(__file__).resolve().parents[2]
        / "test_out"
        / "github_cases"
        / f"{prefix}-{uuid4().hex}"
    )
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class TestGitHubClientContext:
    @pytest.mark.asyncio
    async def test_context_manager_opens_and_closes_client(self) -> None:
        from src.tools.github import GitHubClient

        github_handle = MagicMock()
        loop = MagicMock()

        async def run_in_executor(_: object, fn: object) -> None:
            fn()

        loop.run_in_executor = AsyncMock(side_effect=run_in_executor)

        with (
            patch("src.tools.github.get_settings", return_value=make_settings()),
            patch("src.tools.github.Github", return_value=github_handle) as mock_github,
            patch("src.tools.github.asyncio.get_running_loop", return_value=loop),
        ):
            client = GitHubClient()
            entered = await client.__aenter__()
            await client.__aexit__(None, None, None)

        assert entered is client
        mock_github.assert_called_once_with("ghp_test_token", retry=3, per_page=100)
        loop.run_in_executor.assert_awaited_once()


class TestIssueOperations:
    @pytest.mark.asyncio
    async def test_get_issue_parses_comments_and_linked_prs(self) -> None:
        from src.tools.github import GitHubClient

        comment_bodies = [SimpleNamespace(body=f"comment {i} body") for i in range(12)]
        issue = SimpleNamespace(
            number=42,
            title="Crash",
            body="Fix it",
            labels=[SimpleNamespace(name="bug"), SimpleNamespace(name="agent-fix")],
            assignees=[SimpleNamespace(login="alice")],
            created_at=datetime.now(UTC),
            html_url="https://github.com/owner/repo/issues/42",
            get_comments=lambda: comment_bodies,
            get_timeline=lambda: [
                SimpleNamespace(
                    event="cross-referenced",
                    source=SimpleNamespace(issue=SimpleNamespace(number=7)),
                ),
                SimpleNamespace(event="closed", source=None),
            ],
        )
        repo = MagicMock()
        repo.get_issue.return_value = issue
        gh = MagicMock()
        gh.get_repo.return_value = repo

        with (
            patch("src.tools.github.get_settings", return_value=make_settings()),
            patch.object(GitHubClient, "_with_retry", new=AsyncMock(side_effect=run_now)),
        ):
            client = GitHubClient()
            client._gh = gh
            parsed = await client.get_issue("owner/repo", 42)

        assert parsed.repo_full_name == "owner/repo"
        assert parsed.linked_prs == [7]
        assert len(parsed.comments) == 10
        assert parsed.assignees == ["alice"]

    @pytest.mark.asyncio
    async def test_get_issue_tolerates_timeline_failures(self) -> None:
        from src.tools.github import GitHubClient

        issue = SimpleNamespace(
            number=42,
            title="Crash",
            body=None,
            labels=[],
            assignees=[],
            created_at=datetime.now(UTC),
            html_url="https://github.com/owner/repo/issues/42",
            get_comments=lambda: [],
            get_timeline=MagicMock(side_effect=RuntimeError("timeline disabled")),
        )
        repo = MagicMock()
        repo.get_issue.return_value = issue
        gh = MagicMock()
        gh.get_repo.return_value = repo

        with (
            patch("src.tools.github.get_settings", return_value=make_settings()),
            patch.object(GitHubClient, "_with_retry", new=AsyncMock(side_effect=run_now)),
        ):
            client = GitHubClient()
            client._gh = gh
            parsed = await client.get_issue("owner/repo", 42)

        assert parsed.linked_prs == []
        assert parsed.body == ""

    @pytest.mark.asyncio
    async def test_comment_on_issue_posts_comment(self) -> None:
        from src.tools.github import GitHubClient

        issue = MagicMock()
        repo = MagicMock()
        repo.get_issue.return_value = issue
        gh = MagicMock()
        gh.get_repo.return_value = repo

        with (
            patch("src.tools.github.get_settings", return_value=make_settings()),
            patch.object(GitHubClient, "_with_retry", new=AsyncMock(side_effect=run_now)),
        ):
            client = GitHubClient()
            client._gh = gh
            await client.comment_on_issue("owner/repo", 42, "hello")

        issue.create_comment.assert_called_once_with("hello")


class TestPullRequestCreation:
    @pytest.mark.asyncio
    async def test_create_pull_request_uses_default_branch_and_labels(self) -> None:
        from src.tools.github import GitHubClient

        fake_repo = MagicMock()
        fake_repo.default_branch = "trunk"
        fake_repo.get_branch.return_value.commit.sha = "sha-123"
        fake_repo.get_labels.return_value = [SimpleNamespace(name="existing")]

        fake_pr = MagicMock()
        fake_pr.number = 9
        fake_pr.html_url = "https://github.com/owner/repo/pull/9"
        fake_repo.create_pull.return_value = fake_pr

        gh = MagicMock()
        gh.get_repo.return_value = fake_repo

        clone_repo = MagicMock()
        writer = MagicMock()
        writer.set_value.return_value = writer
        clone_repo.config_writer.return_value = writer
        clone_repo.is_dirty.return_value = True
        origin = MagicMock()
        clone_repo.remote.return_value = origin

        patches = [
            FilePatch(
                file_path="src/app.py",
                unified_diff="@@ -1 +1 @@\n-print('bad')\n+print('good')\n",
                change_type="modify",
                description="fix output",
            )
        ]

        with (
            case_dir("create-pr") as root,
            patch("src.tools.github.get_settings", return_value=make_settings()),
            patch.object(
                GitHubClient,
                "_with_retry",
                new=AsyncMock(side_effect=run_now),
            ),
            patch("src.tools.github.git.Repo.clone_from", return_value=clone_repo),
            patch(
                "src.tools.github._build_authenticated_clone_kwargs",
                return_value={
                    "url": "https://github.com/owner/repo.git",
                    "multi_options": ["-c", "header"],
                },
            ),
            patch(
                "tempfile.TemporaryDirectory",
                return_value=fixed_tempdir(root / "repo"),
            ),
            patch("src.tools.github._apply_patch_to_worktree") as apply_patch,
        ):
            client = GitHubClient()
            client._gh = gh
            pr = await client.create_pull_request(
                repo_full_name="owner/repo",
                branch_name="swe-agent/fix-42",
                base_branch=None,
                title="Fix bug",
                body="Automated fix",
                patches=patches,
                issue_number=42,
                labels=["new-label"],
                draft=True,
            )

        fake_repo.get_branch.assert_called_once_with("trunk")
        clone_repo.git.checkout.assert_called_once_with("-b", "swe-agent/fix-42")
        clone_repo.index.commit.assert_called_once()
        origin.push.assert_called_once_with(
            refspec="swe-agent/fix-42:swe-agent/fix-42",
            force=False,
        )
        apply_patch.assert_called_once()
        fake_repo.create_label.assert_called_once_with("new-label", "0075ca")
        fake_pr.add_to_labels.assert_called_once_with("new-label")
        fake_pr.create_issue_comment.assert_called_once()
        assert pr.pr_number == 9
        assert pr.is_draft is True

    @pytest.mark.asyncio
    async def test_create_pull_request_ignores_existing_branch(self) -> None:
        from src.tools.github import GitHubClient

        fake_repo = MagicMock()
        fake_repo.default_branch = "main"
        fake_repo.get_branch.return_value.commit.sha = "sha-123"
        fake_repo.create_git_ref.side_effect = GithubException(422, data={}, headers={})
        fake_pr = MagicMock(number=1, html_url="https://github.com/owner/repo/pull/1")
        fake_repo.create_pull.return_value = fake_pr

        gh = MagicMock()
        gh.get_repo.return_value = fake_repo

        clone_repo = MagicMock()
        writer = MagicMock()
        writer.set_value.return_value = writer
        clone_repo.config_writer.return_value = writer
        clone_repo.is_dirty.return_value = False
        clone_repo.remote.return_value = MagicMock()

        with (
            case_dir("create-pr-existing") as root,
            patch("src.tools.github.get_settings", return_value=make_settings()),
            patch.object(
                GitHubClient,
                "_with_retry",
                new=AsyncMock(side_effect=run_now),
            ),
            patch("src.tools.github.git.Repo.clone_from", return_value=clone_repo),
            patch(
                "src.tools.github._build_authenticated_clone_kwargs",
                return_value={
                    "url": "https://github.com/owner/repo.git",
                    "multi_options": ["-c", "header"],
                },
            ),
            patch(
                "tempfile.TemporaryDirectory",
                return_value=fixed_tempdir(root / "repo-existing"),
            ),
        ):
            client = GitHubClient()
            client._gh = gh
            pr = await client.create_pull_request(
                repo_full_name="owner/repo",
                branch_name="swe-agent/fix-42",
                base_branch="main",
                title="Fix bug",
                body="Automated fix",
                patches=[],
                issue_number=42,
                labels=None,
                draft=False,
            )

        assert pr.pr_number == 1
        clone_repo.index.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_pull_request_raises_for_non_422_branch_errors(
        self,
    ) -> None:
        from src.tools.github import GitHubClient

        fake_repo = MagicMock()
        fake_repo.default_branch = "main"
        fake_repo.get_branch.return_value.commit.sha = "sha-123"
        fake_repo.create_git_ref.side_effect = GithubException(500, data={}, headers={})

        gh = MagicMock()
        gh.get_repo.return_value = fake_repo

        with (
            patch("src.tools.github.get_settings", return_value=make_settings()),
            patch.object(GitHubClient, "_with_retry", new=AsyncMock(side_effect=run_now)),
            pytest.raises(GithubException),
        ):
            client = GitHubClient()
            client._gh = gh
            await client.create_pull_request(
                repo_full_name="owner/repo",
                branch_name="swe-agent/fix-42",
                base_branch="main",
                title="Fix bug",
                body="Automated fix",
                patches=[],
                issue_number=42,
            )


class TestRetryAndPatchHelpers:
    @pytest.mark.asyncio
    async def test_with_retry_executes_callable_in_executor(self) -> None:
        from src.tools.github import GitHubClient

        class DummyAttempt:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *_: object) -> None:
                return None

        class DummyRetrying:
            def __aiter__(self) -> DummyRetrying:
                self.used = False
                return self

            async def __anext__(self) -> DummyAttempt:
                if self.used:
                    raise StopAsyncIteration
                self.used = True
                return DummyAttempt()

        loop = MagicMock()

        async def run_in_executor(_: object, fn: object) -> object:
            return fn()

        loop.run_in_executor = AsyncMock(side_effect=run_in_executor)

        with (
            patch("src.tools.github.AsyncRetrying", return_value=DummyRetrying()),
            patch("src.tools.github.asyncio.get_running_loop", return_value=loop),
        ):
            result = await GitHubClient._with_retry(lambda: "ok")

        assert result == "ok"

    def test_apply_patch_to_worktree_creates_files(self) -> None:
        from src.tools.github import _apply_patch_to_worktree

        patch_file = FilePatch(
            file_path="src/new_file.py",
            unified_diff="+++ src/new_file.py\n+print('hello')\n+print('world')\n",
            change_type="create",
            description="create file",
        )

        with case_dir("patch-create") as root:
            _apply_patch_to_worktree(root, patch_file)

            assert (root / "src" / "new_file.py").read_text() == "print('hello')\nprint('world')"

    def test_apply_patch_to_worktree_deletes_files(self) -> None:
        from src.tools.github import _apply_patch_to_worktree

        with case_dir("patch-delete") as root:
            target = root / "src" / "old_file.py"
            target.parent.mkdir(parents=True)
            target.write_text("print('hello')")

            patch_file = FilePatch(
                file_path="src/old_file.py",
                unified_diff="",
                change_type="delete",
                description="delete file",
            )

            _apply_patch_to_worktree(root, patch_file)

            assert not target.exists()

    def test_apply_patch_to_worktree_uses_patch_binary_for_modifications(self) -> None:
        from src.tools.github import _apply_patch_to_worktree

        patch_file = FilePatch(
            file_path="src/app.py",
            unified_diff="@@ -1 +1 @@\n-print('bad')\n+print('good')\n",
            change_type="modify",
            description="modify file",
        )
        completed = SimpleNamespace(returncode=0, stderr=b"")

        with case_dir("patch-modify") as root:
            with patch("subprocess.run", return_value=completed) as mock_run:
                _apply_patch_to_worktree(root, patch_file)

            mock_run.assert_called_once()

    def test_apply_patch_to_worktree_raises_on_patch_failures(self) -> None:
        from src.tools.github import _apply_patch_to_worktree

        patch_file = FilePatch(
            file_path="src/app.py",
            unified_diff="@@ -1 +1 @@\n-print('bad')\n+print('good')\n",
            change_type="modify",
            description="modify file",
        )
        completed = SimpleNamespace(returncode=1, stderr=b"patch failed")

        with (
            case_dir("patch-fail") as root,
            patch("subprocess.run", return_value=completed),
            pytest.raises(
                RuntimeError,
                match="Patch failed for src/app.py: patch failed",
            ),
        ):
            _apply_patch_to_worktree(root, patch_file)

    def test_apply_patch_to_worktree_full_content_fallbacks(self) -> None:
        from src.tools.github import _apply_patch_to_worktree

        # 1. Create with full_content
        with case_dir("patch-create-full") as root:
            patch_create = FilePatch(
                file_path="src/new.py",
                change_type="create",
                full_content="print('created')\n",
                description="create file",
            )
            _apply_patch_to_worktree(root, patch_create)
            assert (root / "src/new.py").read_text() == "print('created')\n"

        # 2. Modify with full_content directly (empty diff)
        with case_dir("patch-mod-full") as root:
            patch_mod_direct = FilePatch(
                file_path="src/mod.py",
                change_type="modify",
                unified_diff="",
                full_content="print('replaced')\n",
                description="replace file",
            )
            _apply_patch_to_worktree(root, patch_mod_direct)
            assert (root / "src/mod.py").read_text() == "print('replaced')\n"

        # 3. Modify with diff failure fallback to full_content
        with case_dir("patch-fallback-full") as root:
            patch_fallback = FilePatch(
                file_path="src/app.py",
                unified_diff="invalid diff",
                change_type="modify",
                full_content="print('fallback success')\n",
                description="fallback modify",
            )
            completed_fail = SimpleNamespace(returncode=1, stderr=b"diff fail")
            with patch("subprocess.run", return_value=completed_fail):
                _apply_patch_to_worktree(root, patch_fallback)
                assert (root / "src/app.py").read_text() == "print('fallback success')\n"

        # 4. Delete non-existent file
        with case_dir("patch-del-nonexist") as root:
            patch_del = FilePatch(
                file_path="nonexistent.py",
                change_type="delete",
                description="delete",
            )
            _apply_patch_to_worktree(root, patch_del)  # should not throw

    def test_build_authenticated_clone_kwargs_includes_basic_auth_header(self) -> None:
        from src.tools.github import _build_authenticated_clone_kwargs

        with patch("src.tools.github.get_settings", return_value=make_settings()):
            clone_kwargs = _build_authenticated_clone_kwargs("owner/repo")

        assert clone_kwargs["url"] == "https://github.com/owner/repo.git"
        assert clone_kwargs["multi_options"][0] == "-c"
        assert "AUTHORIZATION: basic " in clone_kwargs["multi_options"][1]

    @pytest.mark.asyncio
    async def test_create_pull_request_handles_create_label_exception(self) -> None:
        from src.tools.github import GitHubClient

        gh_pr = SimpleNamespace(
            number=99,
            html_url="https://github.com/owner/repo/pull/99",
            add_to_labels=MagicMock(),
            create_issue_comment=MagicMock(),
        )
        gh_repo = SimpleNamespace(
            get_branch=MagicMock(
                return_value=SimpleNamespace(name="main", commit=SimpleNamespace(sha="deadbeef"))
            ),
            default_branch="main",
            get_labels=MagicMock(return_value=[]),
            create_label=MagicMock(side_effect=GithubException(422, "label exists", None)),
            create_pull=MagicMock(return_value=gh_pr),
            create_git_ref=MagicMock(),
        )
        gh_instance = SimpleNamespace(get_repo=MagicMock(return_value=gh_repo))

        with (
            patch("src.tools.github.get_settings", return_value=make_settings()),
            patch("src.tools.github.git.Repo.clone_from") as mock_clone,
        ):
            mock_repo_obj = MagicMock()
            mock_clone.return_value = mock_repo_obj
            client = GitHubClient()
            client._gh = gh_instance

            pr = await client.create_pull_request(
                repo_full_name="owner/repo",
                branch_name="agent/fix-42",
                title="Fix bug",
                body="Fixed",
                issue_number=42,
                labels=["agent-fix"],
            )
            assert pr.pr_number == 99
            gh_repo.create_label.assert_called_once_with("agent-fix", "0075ca")

    @pytest.mark.asyncio
    async def test_with_retry_coroutine_and_plain_value(self) -> None:
        from src.tools.github import GitHubClient

        async def some_coro():
            return "from-coro"

        res1 = await GitHubClient._with_retry(some_coro())
        assert res1 == "from-coro"

        async def async_callable():
            return "from-async-fn"

        res_callable = await GitHubClient._with_retry(async_callable)
        assert res_callable == "from-async-fn"

        res2 = await GitHubClient._with_retry("plain-value")
        assert res2 == "plain-value"

    def test_validate_webhook_signature_edge_cases(self) -> None:
        import hashlib
        import hmac

        from src.tools.github import GitHubClient

        secret = "custom-sec-123"
        payload = b'{"action":"opened"}'
        sig = "sha256=" + hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

        # Settings with github_webhook_secret (not _value)
        settings = SimpleNamespace(github_webhook_secret=secret, github_webhook_secret_value=None)
        with patch("src.tools.github.get_settings", return_value=settings):
            assert GitHubClient.validate_webhook_signature(payload, sig) is True

        # Settings raises exception
        with patch("src.tools.github.get_settings", side_effect=RuntimeError("settings fail")):
            assert GitHubClient.validate_webhook_signature(payload, "sha256=123456") is False

        # Key encoding exception in keys_to_try
        with (
            patch.dict("os.environ", {"GITHUB_WEBHOOK_SECRET": "valid-sec"}),
            patch("hmac.new", side_effect=Exception("hmac error")),
        ):
            assert GitHubClient.validate_webhook_signature(payload, "sha256=123456") is False
