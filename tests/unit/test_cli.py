from __future__ import annotations

import runpy
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import typer

from tests.evals.eval_runner import EvalSummary


@contextmanager
def case_dir(prefix: str) -> object:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "test_out" / "cli_cases" / f"{prefix}-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


class TestRunCommand:
    def test_run_delegates_to_async_helper(self) -> None:
        from src import cli

        with (
            patch("src.cli._run_local", new=AsyncMock()) as mock_run_local,
            patch("src.cli.console.print"),
        ):
            cli.run("owner/repo", 42, False, True)

        mock_run_local.assert_awaited_once_with("owner/repo", 42, False, True)

    @pytest.mark.asyncio
    async def test_run_local_handles_dry_run_and_result_output(self) -> None:
        from src import cli

        issue = SimpleNamespace(title="Crash on None")
        github_client = AsyncMock()
        github_client.__aenter__.return_value = github_client
        github_client.__aexit__.return_value = None
        github_client.get_issue.return_value = issue

        final_state = {
            "status": "partial",
            "pull_request": {"pr_url": "https://github.com/owner/repo/pull/1"},
            "failure_reason": "tests still failing",
            "retry_count": 2,
        }

        with (
            patch("src.config.get_settings"),
            patch("src.observability.tracing.configure_logging"),
            patch("src.cli._inject_mock_responses"),
            patch("src.tools.github.GitHubClient", return_value=github_client),
            patch("src.agent.graph.run_agent", new=AsyncMock(return_value=final_state)),
            patch("src.cli.console.print") as mock_print,
        ):
            await cli._run_local("owner/repo", 42, True, True)

        assert mock_print.call_count >= 5

    @pytest.mark.asyncio
    async def test_run_local_non_dry_run_with_enum_status(self) -> None:
        from src import cli
        from src.agent.state import RunStatus

        issue = SimpleNamespace(title="Crash on None")
        github_client = AsyncMock()
        github_client.__aenter__.return_value = github_client
        github_client.__aexit__.return_value = None
        github_client.get_issue.return_value = issue

        final_state = {
            "status": RunStatus.SUCCEEDED,
            "pull_request": {"pr_url": "https://github.com/owner/repo/pull/1"},
            "retry_count": 0,
        }

        with (
            patch("src.config.get_settings"),
            patch("src.observability.tracing.configure_logging"),
            patch("src.tools.github.GitHubClient", return_value=github_client),
            patch("src.agent.graph.run_agent", new=AsyncMock(return_value=final_state)),
            patch("src.cli.console.print") as mock_print,
        ):
            await cli._run_local("owner/repo", 42, False, False)

        assert mock_print.call_count >= 3


class TestStatusCommand:
    def test_status_delegates_to_async_helper(self) -> None:
        from src import cli

        with patch("src.cli._show_status", new=AsyncMock()) as mock_show_status:
            cli.status("run-123")

        mock_show_status.assert_awaited_once_with("run-123")

    @pytest.mark.asyncio
    async def test_show_status_raises_for_missing_run(self) -> None:
        from src import cli

        repo = AsyncMock()
        repo.get_run.return_value = None

        with (
            patch("src.db.database.init_db", new=AsyncMock()),
            patch("src.db.repository.RunRepository", return_value=repo),
            patch("src.cli.console.print"),
            pytest.raises(typer.Exit) as exc_info,
        ):
            await cli._show_status("run-missing")

        assert exc_info.value.exit_code == 1

    @pytest.mark.asyncio
    async def test_show_status_renders_found_run(self) -> None:
        from src import cli

        run = SimpleNamespace(
            run_id="run-123",
            status="failed",
            repo_full_name="owner/repo",
            issue_number=42,
            retry_count=2,
            pr_url=None,
            started_at="2026-03-29T10:00:00",
            completed_at=None,
            failure_reason="boom",
        )
        repo = AsyncMock()
        repo.get_run.return_value = run

        with (
            patch("src.db.database.init_db", new=AsyncMock()),
            patch("src.db.repository.RunRepository", return_value=repo),
            patch("src.cli.console.print") as mock_print,
        ):
            await cli._show_status("run-123")

        mock_print.assert_called_once()


class TestListRunsCommand:
    def test_list_runs_delegates_to_async_helper(self) -> None:
        from src import cli

        with patch("src.cli._list_runs", new=AsyncMock()) as mock_list_runs:
            cli.list_runs("owner/repo", 5)

        mock_list_runs.assert_awaited_once_with("owner/repo", 5)

    @pytest.mark.asyncio
    async def test_list_runs_prints_empty_state(self) -> None:
        from src import cli

        repo = AsyncMock()
        repo.list_runs.return_value = []

        with (
            patch("src.db.database.init_db", new=AsyncMock()),
            patch("src.db.repository.RunRepository", return_value=repo),
            patch("src.cli.console.print") as mock_print,
        ):
            await cli._list_runs("owner/repo", 5)

        mock_print.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_runs_renders_rows_with_duration(self) -> None:
        from src import cli

        run = SimpleNamespace(
            run_id="run-123456789012345678",
            status="succeeded",
            repo_full_name="owner/repo",
            issue_number=42,
            retry_count=1,
            pr_url="https://github.com/owner/repo/pull/1",
            started_at=datetime(2026, 3, 29, 0, 0, tzinfo=UTC),
            completed_at=datetime(2026, 3, 29, 0, 0, 5, tzinfo=UTC),
        )
        repo = AsyncMock()
        repo.list_runs.return_value = [run]

        with (
            patch("src.db.database.init_db", new=AsyncMock()),
            patch("src.db.repository.RunRepository", return_value=repo),
            patch("src.cli.console.print") as mock_print,
        ):
            await cli._list_runs("owner/repo", 5)

        mock_print.assert_called_once()


class TestEvalsCommand:
    def test_real_mode_requires_confirmation(self) -> None:
        from src import cli

        with (
            patch("src.cli.console.print"),
            patch("src.cli.typer.confirm", return_value=False),
            pytest.raises(typer.Exit) as exc_info,
        ):
            cli.evals(False, 2, None)

        assert exc_info.value.exit_code == 0

    def test_evals_writes_summary_output(self) -> None:
        from src import cli

        summary = EvalSummary(
            total=10,
            passed=8,
            failed=2,
            avg_retries=1.2,
            avg_duration_seconds=3.4,
            avg_tokens=5678.0,
            solve_rate_pct=80.0,
        )

        with case_dir("evals") as root:
            output = root / "eval-summary.json"

            with (
                patch("src.cli._run_evals", new=AsyncMock(return_value=summary)),
                patch("tests.evals.eval_runner.print_eval_table"),
                patch("src.cli.console.print"),
            ):
                cli.evals(True, 3, output)

            assert output.exists()
            assert '"solve_rate_pct": 80.0' in output.read_text()

    @pytest.mark.asyncio
    async def test_run_evals_delegates_to_eval_runner(self) -> None:
        from src import cli

        summary = object()

        with patch(
            "tests.evals.eval_runner.run_all_evals",
            new=AsyncMock(return_value=summary),
        ) as mock_run_all:
            result = await cli._run_evals(True, 4)

        assert result is summary
        mock_run_all.assert_awaited_once_with(max_concurrent=4, dry_run=True)


class TestSandboxCommand:
    def test_sandbox_check_delegates_to_async_helper(self) -> None:
        from src import cli

        with (
            patch("src.cli._check_sandbox", new=AsyncMock()) as mock_check_sandbox,
            patch("src.cli.console.print"),
        ):
            cli.sandbox_check()

        mock_check_sandbox.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_sandbox_passes_when_network_is_blocked(self) -> None:
        from src import cli

        client = MagicMock()
        client.containers.run.side_effect = RuntimeError("network blocked")

        with (
            patch("docker.from_env", return_value=client),
            patch("src.cli.console.print") as mock_print,
        ):
            await cli._check_sandbox()

        assert mock_print.call_count >= 3

    @pytest.mark.asyncio
    async def test_check_sandbox_raises_when_network_is_available(self) -> None:
        from src import cli

        client = MagicMock()
        client.containers.run.return_value = None

        with (
            patch("docker.from_env", return_value=client),
            patch("src.cli.console.print"),
            pytest.raises(typer.Exit) as exc_info,
        ):
            await cli._check_sandbox()

        assert exc_info.value.exit_code == 1


class TestConfigAndHelpers:
    def test_config_show_renders_redacted_settings(self) -> None:
        from src import cli

        settings = SimpleNamespace(
            model_dump=lambda: {
                "github_token": "ghp_very_secret_token",
                "public_url": "https://example.com",
                "api_key_hint": "secret-key-value",
            }
        )

        with (
            patch("src.config.get_settings", return_value=settings),
            patch("src.cli.console.print") as mock_print,
        ):
            cli.config_show()

        mock_print.assert_called_once()

    def test_inject_mock_responses_is_a_noop(self) -> None:
        from src import cli

        assert cli._inject_mock_responses() is None

    def test_module_executes_app_when_run_as_script(self) -> None:
        cli_path = Path(__file__).resolve().parents[2] / "src" / "cli.py"

        with patch("typer.main.Typer.__call__", return_value=None) as app_call:
            runpy.run_path(str(cli_path), run_name="__main__")

        app_call.assert_called_once()
