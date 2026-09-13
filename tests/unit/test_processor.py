from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from src.agent.state import GithubIssue


class TestRunAgentAsync:
    @pytest.mark.asyncio
    async def test_marks_existing_run_running_instead_of_recreating(self) -> None:
        from src.worker.processor import _run_agent_async

        issue = GithubIssue(
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

        github_client = AsyncMock()
        github_client.__aenter__.return_value.get_issue = AsyncMock(return_value=issue)
        github_client.__aexit__.return_value = None

        repo = AsyncMock()
        redis = AsyncMock()

        with (
            patch("src.worker.processor.RunRepository", return_value=repo),
            patch(
                "src.worker.processor.get_redis_connection",
                new_callable=AsyncMock,
                return_value=redis,
            ),
            patch("src.worker.processor.GitHubClient", return_value=github_client),
            patch(
                "src.worker.processor.run_agent",
                new_callable=AsyncMock,
                return_value={"status": "failed", "retry_count": 0, "failure_reason": "x"},
            ),
            patch("src.worker.processor.increment_active_runs") as mock_increment,
            patch("src.worker.processor.decrement_active_runs") as mock_decrement,
            patch("src.worker.processor.release_active_job_lock", new_callable=AsyncMock),
            patch("src.worker.processor._post_status_comment", new_callable=AsyncMock),
        ):
            await _run_agent_async("owner/repo", 42, "run-123")

        repo.mark_run_running.assert_awaited_once()
        repo.create_run.assert_not_called()
        mock_increment.assert_called_once()
        mock_decrement.assert_called_once()

    @pytest.mark.asyncio
    async def test_persists_pr_url_and_ignores_comment_failures(self) -> None:
        from src.worker.processor import _run_agent_async

        issue = GithubIssue(
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

        github_client = AsyncMock()
        github_client.__aenter__.return_value.get_issue = AsyncMock(return_value=issue)
        github_client.__aexit__.return_value = None

        repo = AsyncMock()
        redis = AsyncMock()
        final_state = {
            "status": "succeeded",
            "retry_count": 2,
            "pull_request": {"pr_url": "https://github.com/owner/repo/pull/7"},
        }

        with (
            patch("src.worker.processor.RunRepository", return_value=repo),
            patch(
                "src.worker.processor.get_redis_connection",
                new_callable=AsyncMock,
                return_value=redis,
            ),
            patch("src.worker.processor.GitHubClient", return_value=github_client),
            patch(
                "src.worker.processor.run_agent",
                new_callable=AsyncMock,
                return_value=final_state,
            ),
            patch(
                "src.worker.processor._post_status_comment",
                new_callable=AsyncMock,
                side_effect=RuntimeError("comment failed"),
            ),
            patch("src.worker.processor.increment_active_runs"),
            patch("src.worker.processor.decrement_active_runs"),
            patch("src.worker.processor.release_active_job_lock", new_callable=AsyncMock),
        ):
            result = await _run_agent_async("owner/repo", 42, "run-123")

        repo.update_run.assert_awaited_once()
        update_kwargs = repo.update_run.await_args.kwargs
        assert update_kwargs["status"] == "succeeded"
        assert update_kwargs["pr_url"] == "https://github.com/owner/repo/pull/7"
        assert update_kwargs["retry_count"] == 2
        assert redis.hset.await_args_list[-1].kwargs["mapping"] == {
            "status": "succeeded",
            "pr_url": "https://github.com/owner/repo/pull/7",
        }
        assert result == {
            "run_id": "run-123",
            "status": "succeeded",
            "pr_url": "https://github.com/owner/repo/pull/7",
            "retry_count": 2,
        }


class TestPersistFailure:
    @pytest.mark.asyncio
    async def test_updates_redis_status_on_failure(self) -> None:
        from src.worker.processor import _persist_failure

        repo = AsyncMock()
        redis = AsyncMock()

        with (
            patch("src.worker.processor.RunRepository", return_value=repo),
            patch(
                "src.worker.processor.get_redis_connection",
                new_callable=AsyncMock,
                return_value=redis,
            ),
            patch("src.worker.processor.release_active_job_lock", new_callable=AsyncMock),
        ):
            await _persist_failure("run-123", "owner/repo", 42, "fatal boom")

        repo.update_run.assert_awaited_once()
        redis.hset.assert_awaited_once()
        assert redis.hset.await_args.kwargs["mapping"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_swallows_persistence_errors_but_releases_lock(self) -> None:
        from src.worker.processor import _persist_failure

        repo = AsyncMock()
        repo.update_run.side_effect = RuntimeError("db down")

        with (
            patch("src.worker.processor.RunRepository", return_value=repo),
            patch(
                "src.worker.processor.get_redis_connection",
                new_callable=AsyncMock,
            ) as get_redis,
            patch(
                "src.worker.processor.release_active_job_lock",
                new_callable=AsyncMock,
            ) as release_lock,
        ):
            await _persist_failure("run-123", "owner/repo", 42, "fatal boom")

        get_redis.assert_not_awaited()
        release_lock.assert_awaited_once_with("owner/repo", 42, "run-123")


class TestProcessIssueJob:
    def test_process_issue_job_returns_async_result(self) -> None:
        from src.worker.processor import process_issue_job

        settings = type("Settings", (), {"log_level": "INFO", "log_format": "json"})()

        with (
            patch("src.worker.processor.get_settings", return_value=settings),
            patch("src.worker.processor.configure_logging"),
            patch(
                "src.worker.processor._run_agent_async",
                new=AsyncMock(return_value={"status": "succeeded", "run_id": "run-123"}),
            ),
        ):
            result = process_issue_job("owner/repo", 42, "run-123")

        assert result["status"] == "succeeded"

    def test_process_issue_job_persists_fatal_failures(self) -> None:
        from src.worker.processor import process_issue_job

        settings = type("Settings", (), {"log_level": "INFO", "log_format": "json"})()

        with (
            patch("src.worker.processor.get_settings", return_value=settings),
            patch("src.worker.processor.configure_logging"),
            patch(
                "src.worker.processor._run_agent_async",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
            patch("src.worker.processor._persist_failure", new=AsyncMock()) as persist_failure,
            pytest.raises(RuntimeError, match="boom"),
        ):
            process_issue_job("owner/repo", 42, "run-123")

        persist_failure.assert_awaited_once()


class TestPostStatusComment:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("status", "snippet"),
        [
            ("succeeded", "submitted a fix"),
            ("partial", "opened a draft PR"),
            ("failed", "unable to generate a passing fix"),
        ],
    )
    async def test_posts_status_specific_comment(
        self,
        status: str,
        snippet: str,
    ) -> None:
        from src.worker.processor import _post_status_comment

        issue = GithubIssue(
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
        github_client = AsyncMock()
        github_client.__aenter__.return_value = github_client
        github_client.__aexit__.return_value = None

        with patch("src.worker.processor.GitHubClient", return_value=github_client):
            await _post_status_comment(
                issue=issue,
                status=status,
                pr_url="https://github.com/owner/repo/pull/1",
                retry_count=2,
            )

        body = github_client.comment_on_issue.await_args.kwargs["body"]
        assert snippet in body

    @pytest.mark.asyncio
    async def test_handles_json_serialization_failure(self) -> None:
        from src.worker.processor import _run_agent_async

        issue = GithubIssue(
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

        github_client = AsyncMock()
        github_client.__aenter__.return_value.get_issue = AsyncMock(return_value=issue)
        github_client.__aexit__.return_value = None

        repo = AsyncMock()
        redis = AsyncMock()

        with (
            patch("src.worker.processor.RunRepository", return_value=repo),
            patch(
                "src.worker.processor.get_redis_connection",
                new_callable=AsyncMock,
                return_value=redis,
            ),
            patch("src.worker.processor.GitHubClient", return_value=github_client),
            patch(
                "src.worker.processor.run_agent",
                new_callable=AsyncMock,
                return_value={"status": "succeeded"},
            ),
            patch("src.worker.processor.increment_active_runs"),
            patch("src.worker.processor.decrement_active_runs"),
            patch("src.worker.processor.release_active_job_lock", new_callable=AsyncMock),
            patch("src.worker.processor._post_status_comment", new_callable=AsyncMock),
            patch("json.dumps", side_effect=TypeError("cannot serialize")),
        ):
            await _run_agent_async("owner/repo", 42, "run-123")

        repo.update_run.assert_awaited_once()
        assert repo.update_run.await_args.kwargs["state_snapshot"] is None
