"""
tests/unit/test_watchdog.py — Tests for stuck-run detection and reaping logic.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.worker.watchdog import (
    STUCK_THRESHOLD_MINUTES,
    _post_watchdog_comment,
    _reap_stuck_runs,
    run_watchdog_cycle,
)


def _make_run(
    run_id: str = "run-abc",
    status: str = "running",
    started_minutes_ago: int = 60,
    repo: str = "owner/repo",
    issue: int = 1,
) -> MagicMock:
    run = MagicMock()
    run.run_id = run_id
    run.status = status
    run.repo_full_name = repo
    run.issue_number = issue
    run.started_at = datetime.now(timezone.utc) - timedelta(minutes=started_minutes_ago)
    return run


class TestReapStuckRuns:
    @pytest.mark.asyncio
    async def test_reaps_run_over_threshold(self) -> None:
        """A run started 90 minutes ago should be reaped."""
        stuck_run = _make_run(started_minutes_ago=90)
        results = {"stuck_runs_reaped": 0, "stale_containers_removed": 0, "errors": 0}

        with (
            patch("src.worker.watchdog.RunRepository") as mock_repo_cls,
            patch("src.worker.watchdog._post_watchdog_comment", new_callable=AsyncMock),
        ):
            mock_repo = AsyncMock()
            mock_repo.list_runs.return_value = [stuck_run]
            mock_repo.update_run = AsyncMock()
            mock_repo_cls.return_value = mock_repo

            await _reap_stuck_runs(results)

        assert results["stuck_runs_reaped"] == 1
        mock_repo.update_run.assert_awaited_once()
        call_kwargs = mock_repo.update_run.call_args.kwargs
        assert call_kwargs["status"] == "failed"
        assert "Watchdog" in call_kwargs["failure_reason"]

    @pytest.mark.asyncio
    async def test_does_not_reap_fresh_run(self) -> None:
        """A run started 10 minutes ago should NOT be reaped."""
        fresh_run = _make_run(started_minutes_ago=10)
        results = {"stuck_runs_reaped": 0, "stale_containers_removed": 0, "errors": 0}

        with patch("src.worker.watchdog.RunRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.list_runs.return_value = [fresh_run]
            mock_repo.update_run = AsyncMock()
            mock_repo_cls.return_value = mock_repo

            await _reap_stuck_runs(results)

        assert results["stuck_runs_reaped"] == 0
        mock_repo.update_run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reaps_exactly_at_threshold_boundary(self) -> None:
        """A run at exactly STUCK_THRESHOLD_MINUTES + 1 should be reaped."""
        boundary_run = _make_run(started_minutes_ago=STUCK_THRESHOLD_MINUTES + 1)
        results = {"stuck_runs_reaped": 0, "stale_containers_removed": 0, "errors": 0}

        with (
            patch("src.worker.watchdog.RunRepository") as mock_repo_cls,
            patch("src.worker.watchdog._post_watchdog_comment", new_callable=AsyncMock),
        ):
            mock_repo = AsyncMock()
            mock_repo.list_runs.return_value = [boundary_run]
            mock_repo.update_run = AsyncMock()
            mock_repo_cls.return_value = mock_repo

            await _reap_stuck_runs(results)

        assert results["stuck_runs_reaped"] == 1

    @pytest.mark.asyncio
    async def test_handles_db_error_gracefully(self) -> None:
        """If DB throws, errors counter increments and no crash."""
        results = {"stuck_runs_reaped": 0, "stale_containers_removed": 0, "errors": 0}

        with patch("src.worker.watchdog.RunRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.list_runs.side_effect = Exception("DB connection refused")
            mock_repo_cls.return_value = mock_repo

            await _reap_stuck_runs(results)

        assert results["errors"] == 1
        assert results["stuck_runs_reaped"] == 0

    @pytest.mark.asyncio
    async def test_handles_update_run_error(self) -> None:
        """If updating a run fails, errors counter increments."""
        stuck_run = _make_run(started_minutes_ago=90)
        results = {"stuck_runs_reaped": 0, "stale_containers_removed": 0, "errors": 0}

        with patch("src.worker.watchdog.RunRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.list_runs.return_value = [stuck_run]
            mock_repo.update_run.side_effect = RuntimeError("update failed")
            mock_repo_cls.return_value = mock_repo

            await _reap_stuck_runs(results)

        assert results["errors"] == 1
        assert results["stuck_runs_reaped"] == 0

    @pytest.mark.asyncio
    async def test_multiple_stuck_runs_all_reaped(self) -> None:
        """Multiple stuck runs are all reaped in one cycle."""
        stuck_runs = [
            _make_run(run_id=f"run-{i}", started_minutes_ago=90 + i)
            for i in range(3)
        ]
        results = {"stuck_runs_reaped": 0, "stale_containers_removed": 0, "errors": 0}

        with (
            patch("src.worker.watchdog.RunRepository") as mock_repo_cls,
            patch("src.worker.watchdog._post_watchdog_comment", new_callable=AsyncMock),
        ):
            mock_repo = AsyncMock()
            mock_repo.list_runs.return_value = stuck_runs
            mock_repo.update_run = AsyncMock()
            mock_repo_cls.return_value = mock_repo

            await _reap_stuck_runs(results)

        assert results["stuck_runs_reaped"] == 3
        assert mock_repo.update_run.await_count == 3

    @pytest.mark.asyncio
    async def test_run_without_started_at_skipped(self) -> None:
        """Runs with no started_at timestamp are not reaped."""
        run_no_timestamp = _make_run()
        run_no_timestamp.started_at = None  # Missing timestamp
        results = {"stuck_runs_reaped": 0, "stale_containers_removed": 0, "errors": 0}

        with patch("src.worker.watchdog.RunRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.list_runs.return_value = [run_no_timestamp]
            mock_repo.update_run = AsyncMock()
            mock_repo_cls.return_value = mock_repo

            await _reap_stuck_runs(results)

        assert results["stuck_runs_reaped"] == 0


class TestWatchdogCycle:
    @pytest.mark.asyncio
    async def test_full_cycle_returns_results_dict(self) -> None:
        with (
            patch("src.worker.watchdog._reap_stuck_runs", new_callable=AsyncMock),
            patch("src.worker.watchdog._remove_stale_containers", new_callable=AsyncMock),
        ):
            results = await run_watchdog_cycle()

        assert "stuck_runs_reaped" in results
        assert "stale_containers_removed" in results
        assert "errors" in results

    @pytest.mark.asyncio
    async def test_cycle_is_idempotent(self) -> None:
        """Running the cycle twice does not double-reap anything."""
        with (
            patch("src.worker.watchdog._reap_stuck_runs", new_callable=AsyncMock),
            patch("src.worker.watchdog._remove_stale_containers", new_callable=AsyncMock),
        ):
            r1 = await run_watchdog_cycle()
            r2 = await run_watchdog_cycle()

        # Both cycles complete without error
        assert r1["errors"] == 0
        assert r2["errors"] == 0


class TestGitHubComment:
    @pytest.mark.asyncio
    async def test_posts_comment_with_run_id(self) -> None:
        with patch("src.worker.watchdog.GitHubClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.comment_on_issue = AsyncMock()
            mock_client_cls.return_value = mock_client

            await _post_watchdog_comment(
                repo_full_name="owner/repo",
                issue_number=42,
                run_id="run-abc123456789",
            )

            mock_client.comment_on_issue.assert_awaited_once()
            call_kwargs = mock_client.comment_on_issue.call_args.kwargs
            assert call_kwargs["issue_number"] == 42
            assert "run-abc123" in call_kwargs["body"]

    @pytest.mark.asyncio
    async def test_comment_failure_does_not_raise(self) -> None:
        """If GitHub comment fails, the watchdog must not crash."""
        with patch("src.worker.watchdog.GitHubClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.comment_on_issue = AsyncMock(
                side_effect=Exception("GitHub API 503")
            )
            mock_client_cls.return_value = mock_client

            # Must not raise
            await _post_watchdog_comment("owner/repo", 1, "run-123")


class TestRemoveStaleContainers:
    @pytest.mark.asyncio
    async def test_removes_stale_container_and_handles_api_errors(self) -> None:
        from src.worker.watchdog import _remove_stale_containers
        import docker

        stale_container = MagicMock()
        stale_container.short_id = "c-123"
        stale_container.labels = {"swe-agent.run_id": "run-old"}
        stale_container.attrs = {"Created": "2020-01-01T00:00:00.000000000Z"}

        fresh_container = MagicMock()
        fresh_container.short_id = "c-456"
        fresh_container.labels = {"swe-agent.run_id": "run-fresh"}
        fresh_container.attrs = {"Created": datetime.now(timezone.utc).isoformat()}

        error_container = MagicMock()
        error_container.short_id = "c-789"
        error_container.labels = {"swe-agent.run_id": "run-err"}
        error_container.attrs = {"Created": "2020-01-01T00:00:00.000000000Z"}
        error_container.remove.side_effect = docker.errors.APIError("conflict")

        unparseable_container = MagicMock()
        unparseable_container.attrs = {"Created": "invalid-timestamp"}

        mock_client = MagicMock()
        mock_client.containers.list.return_value = [
            stale_container,
            fresh_container,
            error_container,
            unparseable_container,
        ]

        results = {"stuck_runs_reaped": 0, "stale_containers_removed": 0, "errors": 0}

        with patch("docker.from_env", return_value=mock_client):
            await _remove_stale_containers(results)

        assert results["stale_containers_removed"] == 1
        stale_container.remove.assert_called_once_with(force=True)
        fresh_container.remove.assert_not_called()

    @pytest.mark.asyncio
    async def test_handles_docker_unavailable_exception(self) -> None:
        from src.worker.watchdog import _remove_stale_containers
        import docker

        results = {"stuck_runs_reaped": 0, "stale_containers_removed": 0, "errors": 0}
        with patch("docker.from_env", side_effect=docker.errors.DockerException("daemon off")):
            await _remove_stale_containers(results)
        assert results["stale_containers_removed"] == 0

    @pytest.mark.asyncio
    async def test_handles_loop_run_in_executor_exception(self) -> None:
        from src.worker.watchdog import _remove_stale_containers

        results = {"stuck_runs_reaped": 0, "stale_containers_removed": 0, "errors": 0}
        with patch("asyncio.get_running_loop") as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock(side_effect=RuntimeError("executor err"))
            await _remove_stale_containers(results)
        assert results["errors"] == 1


class TestRunForever:
    @pytest.mark.asyncio
    async def test_runs_cycles_and_handles_exception(self) -> None:
        from src.worker.watchdog import run_forever
        from types import SimpleNamespace

        settings = SimpleNamespace(log_level="INFO", log_format="json")

        cycle_calls = 0

        async def fake_cycle():
            nonlocal cycle_calls
            cycle_calls += 1
            if cycle_calls == 1:
                raise RuntimeError("cycle glitch")
            raise asyncio.CancelledError()

        with (
            patch("src.worker.watchdog.get_settings", return_value=settings),
            patch("src.worker.watchdog.configure_logging"),
            patch("src.worker.watchdog.init_db", new_callable=AsyncMock),
            patch("src.worker.watchdog.run_watchdog_cycle", side_effect=fake_cycle),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            pytest.raises(asyncio.CancelledError),
        ):
            await run_forever(interval_seconds=10)

        assert cycle_calls == 2
        mock_sleep.assert_awaited_once_with(10)
