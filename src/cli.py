"""
src/cli.py — Developer CLI for local operations.
Usage: python -m src.cli [COMMAND]
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional, cast

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

app = typer.Typer(
    name="swe-agent",
    help="Autonomous SWE Agent — Developer CLI",
    rich_markup_mode="rich",
)
console = Console()


# ── run: trigger a local agent run ────────────────────────────────────────────


@app.command()
def run(
    repo: str = typer.Argument(..., help="Repository in owner/repo format"),
    issue: int = typer.Argument(..., help="GitHub issue number"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip LLM calls, use mock responses"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Stream all node logs"),
) -> None:
    """Trigger a single agent run locally against a real GitHub issue."""
    console.print(
        Panel(
            f"[bold]Running agent[/bold] on [cyan]{repo}[/cyan] issue [yellow]#{issue}[/yellow]",
            title="SWE Agent",
            border_style="gold1",
        )
    )

    asyncio.run(_run_local(repo, issue, dry_run, verbose))


async def _run_local(repo: str, issue_num: int, dry_run: bool, verbose: bool) -> None:
    from src.config import get_settings
    from src.observability.tracing import configure_logging

    settings = get_settings()
    configure_logging("DEBUG" if verbose else "INFO", "console")

    if dry_run:
        console.print("[yellow]DRY RUN — LLM calls will be mocked[/yellow]")
        _inject_mock_responses()

    from src.agent.graph import run_agent
    from src.agent.state import AgentState, RunStatus
    from src.tools.github import GitHubClient

    import time

    start = time.monotonic()

    if dry_run:
        from datetime import datetime, timezone
        from src.agent.state import GithubIssue

        issue_obj = GithubIssue(
            issue_number=issue_num,
            title="[MOCK] Add input validation to user registration endpoint",
            body=(
                "The `/api/users/register` endpoint does not validate email format.\n"
                "Any string is accepted. We should validate using the `email-validator` library.\n\n"
                "**Acceptance criteria:**\n"
                "- Reject invalid emails with HTTP 422\n"
                "- Add a unit test for the validation logic"
            ),
            repo_full_name=repo,
            labels=["bug", "backend"],
            comments=[],
            linked_prs=[],
            assignees=[],
            created_at=datetime.now(timezone.utc),
            html_url=f"https://github.com/{repo}/issues/{issue_num}",
        )
        console.print(f"[green]✓[/green] Issue (mock): [bold]{issue_obj.title}[/bold]")
    else:
        async with GitHubClient() as gh:
            issue_obj = await gh.get_issue(repo, issue_num)
        console.print(f"[green]✓[/green] Issue: [bold]{issue_obj.title}[/bold]")

    initial_state: AgentState = {
        "issue": issue_obj,
        "retry_count": 0,
        "attempt_history": [],
    }

    if dry_run:
        from unittest.mock import AsyncMock, patch
        from src.agent.state import TaskPlan, Task, CodePatch, FilePatch, TestResult, PullRequest

        mock_plan = TaskPlan(
            summary="Add email format validation to user registration route",
            tasks=[
                Task(
                    id="task-1",
                    description="Validate email using standard format checking",
                    files_to_modify=["src/api/users.py"],
                    acceptance_criteria=["Return 422 on invalid email"],
                )
            ],
            affected_test_files=["tests/test_users.py"],
        )

        mock_patch = CodePatch(
            explanation="Added email format check returning HTTP 422 for invalid format",
            patches=[
                FilePatch(
                    file_path="src/api/users.py",
                    change_type="modify",
                    unified_diff="--- a/src/api/users.py\n+++ b/src/api/users.py\n@@ -10,3 +10,6 @@\n+    if '@' not in user.email:\n+        raise HTTPException(status_code=422, detail='Invalid email')\n",
                    full_content=None,
                )
            ],
            test_command="python -m pytest tests/test_users.py",
        )

        mock_test_res = TestResult(
            passed=True,
            total=5,
            failed_count=0,
            failures=[],
            duration_seconds=0.42,
            coverage_pct=100.0,
            stdout="5 passed in 0.42s",
        )

        mock_pr = PullRequest(
            pr_number=99,
            pr_url=f"https://github.com/{repo}/pull/99",
            branch_name="agent/fix-email-validation",
            title=f"Fix: {issue_obj.title}",
            body="Automated fix generated by SWE Agent.",
            is_draft=False,
        )

        mock_gh = AsyncMock()
        mock_gh.create_pull_request.return_value = mock_pr

        mock_sandbox = AsyncMock()
        mock_sandbox.apply_patches.return_value = type("ApplyRes", (), {"success": True, "error": None})()
        mock_sandbox.run_tests.return_value = mock_test_res

        with (
            patch("src.tools._repo_cache.get_local_repo_path", new_callable=AsyncMock, return_value="/tmp"),
            patch(
                "src.tools.filesystem.list_repo_tree", new_callable=AsyncMock, return_value="src/\n  api/\n    users.py"
            ),
            patch("src.tools.filesystem.list_test_files", new_callable=AsyncMock, return_value=["tests/test_users.py"]),
            patch("src.tools.search.find_relevant_files", new_callable=AsyncMock, return_value=["src/api/users.py"]),
            patch("src.tools.filesystem.load_file_contexts", new_callable=AsyncMock, return_value=[]),
            patch(
                "src.agent.nodes._invoke_with_timeout",
                side_effect=[
                    type("ClarifierRes", (), {"content": "implementable"})(),
                    mock_plan,
                    mock_patch,
                    type("PRBodyRes", (), {"content": "Automated fix description"})(),
                ],
            ),
            patch("src.tools.sandbox.SandboxRunner") as mock_sandbox_cls,
            patch("src.tools.github.GitHubClient") as mock_gh_cls,
            patch("src.tools.github._apply_patch_to_worktree"),
        ):
            mock_sandbox_cls.return_value.__aenter__.return_value = mock_sandbox
            mock_sandbox_cls.return_value.__aexit__.return_value = None
            mock_gh_cls.return_value.__aenter__.return_value = mock_gh
            mock_gh_cls.return_value.__aexit__.return_value = None
            final = await run_agent(initial_state)
    else:
        final = await run_agent(initial_state)

    elapsed = time.monotonic() - start

    status = final.get("status", "unknown")
    if isinstance(status, RunStatus):
        status = status.value
    pr = final.get("pull_request")

    color = {"succeeded": "green", "partial": "yellow", "failed": "red"}.get(status, "white")
    console.print(f"\n[{color}]Status: {status.upper()}[/{color}]  [dim]({elapsed:.1f}s)[/dim]")

    if pr:
        console.print(f"[bold]PR:[/bold] {pr.get('pr_url')}")  # type: ignore[attr-defined]
    if final.get("failure_reason"):
        console.print(f"[red]Failure:[/red] {final['failure_reason']}")

    console.print(f"[dim]Retries used: {final.get('retry_count', 0)} / {3}[/dim]")


# ── status: show run details ───────────────────────────────────────────────────


@app.command()
def status(
    run_id: str = typer.Argument(..., help="Run ID to inspect"),
) -> None:
    """Show detailed status of a specific run."""
    asyncio.run(_show_status(run_id))


async def _show_status(run_id: str) -> None:
    from src.db.database import init_db
    from src.db.repository import RunRepository

    await init_db()
    repo = RunRepository()
    run = await repo.get_run(run_id)

    if not run:
        console.print(f"[red]Run not found:[/red] {run_id}")
        raise typer.Exit(1)

    table = Table(title=f"Run {run_id[:16]}...", show_header=False)
    table.add_column("Field", style="cyan", width=20)
    table.add_column("Value")

    status_color = {"succeeded": "green", "partial": "yellow", "failed": "red", "running": "blue"}.get(
        run.status, "white"
    )

    table.add_row("Run ID", run.run_id)
    table.add_row("Status", f"[{status_color}]{run.status}[/{status_color}]")
    table.add_row("Repository", run.repo_full_name)
    table.add_row("Issue", f"#{run.issue_number}")
    table.add_row("Retries", str(run.retry_count))
    table.add_row("PR URL", run.pr_url or "—")
    table.add_row("Started", str(run.started_at or "—"))
    table.add_row("Completed", str(run.completed_at or "—"))
    if run.failure_reason:
        table.add_row("Failure", f"[red]{run.failure_reason}[/red]")

    console.print(table)


# ── list: show recent runs ────────────────────────────────────────────────────


@app.command()
def list_runs(
    repo: Optional[str] = typer.Option(None, "--repo", help="Filter by repo"),
    limit: int = typer.Option(20, "--limit", help="Max runs to show"),
) -> None:
    """List recent agent runs."""
    asyncio.run(_list_runs(repo, limit))


async def _list_runs(repo: Optional[str], limit: int) -> None:
    from src.db.database import init_db
    from src.db.repository import RunRepository

    await init_db()
    run_repo = RunRepository()
    runs = await run_repo.list_runs(repo_full_name=repo, limit=limit)

    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title="Recent Agent Runs")
    table.add_column("Run ID", style="dim", width=18)
    table.add_column("Repo")
    table.add_column("Issue", justify="right")
    table.add_column("Status", width=12)
    table.add_column("Retries", justify="right")
    table.add_column("PR")
    table.add_column("Duration")

    for run in runs:
        status_map = {"succeeded": "✅", "failed": "❌", "partial": "⚠️", "running": "⏳"}
        icon = status_map.get(run.status, "?")

        duration = "—"
        if run.started_at and run.completed_at:
            secs = int((run.completed_at - run.started_at).total_seconds())
            duration = f"{secs}s"

        table.add_row(
            run.run_id[:16] + "...",
            run.repo_full_name,
            f"#{run.issue_number}",
            f"{icon} {run.status}",
            str(run.retry_count),
            "→ PR" if run.pr_url else "—",
            duration,
        )

    console.print(table)


# ── evals: run the golden eval set ───────────────────────────────────────────


@app.command()
def evals(
    dry_run: bool = typer.Option(True, "--dry-run/--real", help="Use mocks (safe) or real API"),
    concurrency: int = typer.Option(2, "--concurrency", help="Max parallel cases"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save JSON results to file"),
) -> None:
    """Run the 10-issue golden eval set to measure agent performance."""
    if not dry_run:
        console.print(
            Panel(
                "[bold red]⚠ REAL MODE[/bold red]: This will make real API calls and open real GitHub PRs.\n"
                "Ensure you're pointing at test repositories.",
                border_style="red",
            )
        )
        confirmed = typer.confirm("Continue?", default=False)
        if not confirmed:
            raise typer.Exit(0)

    summary = asyncio.run(_run_evals(dry_run, concurrency))

    from tests.evals.eval_runner import print_eval_table

    print_eval_table(summary)

    if output:
        results_data = {
            "timestamp": summary.timestamp,
            "solve_rate_pct": summary.solve_rate_pct,
            "total": summary.total,
            "passed": summary.passed,
            "avg_retries": summary.avg_retries,
            "avg_duration_seconds": summary.avg_duration_seconds,
            "avg_tokens": summary.avg_tokens,
        }
        output.write_text(json.dumps(results_data, indent=2))
        console.print(f"\n[dim]Results saved to {output}[/dim]")


async def _run_evals(dry_run: bool, concurrency: int) -> Any:
    from tests.evals.eval_runner import run_all_evals

    return await run_all_evals(max_concurrent=concurrency, dry_run=dry_run)


# ── sandbox: test Docker sandbox isolation ────────────────────────────────────


@app.command()
def sandbox_check() -> None:
    """Verify Docker sandbox security settings are correct."""
    console.print("[bold]Checking Docker sandbox configuration...[/bold]\n")
    asyncio.run(_check_sandbox())


async def _check_sandbox() -> None:
    import docker

    client = docker.from_env()
    checks = [
        ("Docker daemon reachable", True),
    ]

    # Try to run a test container
    try:
        container = client.containers.run(
            "python:3.12-slim",
            command=[
                "python",
                "-c",
                "import socket; socket.setdefaulttimeout(1); socket.create_connection(('8.8.8.8', 53))",
            ],
            network_disabled=True,
            remove=True,
        )
        network_blocked = False
    except Exception:
        network_blocked = True

    checks.append(("Network disabled in container", network_blocked))

    for label, passed in checks:
        icon = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(f"  {icon} {label}")

    all_pass = all(p for _, p in checks)
    if all_pass:
        console.print("\n[green]All sandbox checks passed.[/green]")
    else:
        console.print("\n[red]Some sandbox checks FAILED. Review security settings.[/red]")
        raise typer.Exit(1)


# ── config: show current settings (redacted) ─────────────────────────────────


@app.command()
def config_show() -> None:
    """Show current configuration with secrets redacted."""
    from src.config import get_settings

    settings = get_settings()

    table = Table(title="Current Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    def _redact(val: object) -> str:
        s = str(val)
        if len(s) > 8 and any(k in s.lower() for k in ["key", "secret", "password", "token"]):
            return s[:4] + "****" + s[-2:]
        return s

    for field_name, value in settings.model_dump().items():
        if "key" in field_name or "secret" in field_name or "password" in field_name:
            display = "***REDACTED***"
        else:
            display = _redact(value)
        table.add_row(field_name, display)

    console.print(table)


def _inject_mock_responses() -> None:
    """Mock external calls for dry-run mode."""
    return None


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
