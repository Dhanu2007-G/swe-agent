"""
tests/evals/eval_runner.py — Evaluation harness against a golden issue set.
Run in CI to track solve rate, latency, and token cost over time.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

GOLDEN_ISSUES_FILE = Path(__file__).parent / "golden_issues.json"


@dataclass
class EvalCase:
    id: str
    repo: str
    issue_number: int
    description: str
    expected_files_modified: list[str]
    expected_keywords_in_diff: list[str]
    max_retries_expected: int = 2


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    status: str
    retries: int
    duration_seconds: float
    tokens_used: int
    pr_url: str | None
    failure_reason: str | None
    diff_contains_expected: bool = False


@dataclass
class EvalSummary:
    total: int
    passed: int
    failed: int
    avg_retries: float
    avg_duration_seconds: float
    avg_tokens: float
    solve_rate_pct: float
    results: list[EvalResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── Golden Issue Set ──────────────────────────────────────────────────────────

GOLDEN_CASES: list[EvalCase] = [
    EvalCase(
        id="eval-001",
        repo="pallets/flask",
        issue_number=5478,
        description="Fix missing return type annotation in route decorators",
        expected_files_modified=["src/flask/app.py"],
        expected_keywords_in_diff=["-> None", "def route"],
        max_retries_expected=1,
    ),
    EvalCase(
        id="eval-002",
        repo="psf/requests",
        issue_number=6462,
        description="Handle ConnectionError in retry logic",
        expected_files_modified=["requests/adapters.py"],
        expected_keywords_in_diff=["ConnectionError", "retry"],
        max_retries_expected=2,
    ),
    EvalCase(
        id="eval-003",
        repo="encode/httpx",
        issue_number=2756,
        description="Add timeout parameter validation",
        expected_files_modified=["httpx/_config.py"],
        expected_keywords_in_diff=["ValueError", "timeout"],
        max_retries_expected=1,
    ),
    EvalCase(
        id="eval-004",
        repo="tiangolo/fastapi",
        issue_number=10984,
        description="Fix missing 422 response in OpenAPI schema for query params",
        expected_files_modified=["fastapi/routing.py"],
        expected_keywords_in_diff=["422", "responses"],
        max_retries_expected=2,
    ),
    EvalCase(
        id="eval-005",
        repo="pydantic/pydantic",
        issue_number=8972,
        description="Fix model_validator not called on None field",
        expected_files_modified=["pydantic/main.py"],
        expected_keywords_in_diff=["model_validator", "None"],
        max_retries_expected=2,
    ),
    EvalCase(
        id="eval-006",
        repo="sqlalchemy/sqlalchemy",
        issue_number=10234,
        description="Fix async session not committing on context exit",
        expected_files_modified=["lib/sqlalchemy/ext/asyncio/session.py"],
        expected_keywords_in_diff=["__aexit__", "commit"],
        max_retries_expected=3,
    ),
    EvalCase(
        id="eval-007",
        repo="celery/celery",
        issue_number=8765,
        description="Fix task retry countdown not respected",
        expected_files_modified=["celery/app/task.py"],
        expected_keywords_in_diff=["countdown", "retry"],
        max_retries_expected=2,
    ),
    EvalCase(
        id="eval-008",
        repo="aio-libs/aiohttp",
        issue_number=7890,
        description="Handle cancelled futures in connector cleanup",
        expected_files_modified=["aiohttp/connector.py"],
        expected_keywords_in_diff=["CancelledError", "cleanup"],
        max_retries_expected=2,
    ),
    EvalCase(
        id="eval-009",
        repo="pytest-dev/pytest",
        issue_number=11345,
        description="Fix fixture scope not propagating to sub-fixtures",
        expected_files_modified=["src/_pytest/fixtures.py"],
        expected_keywords_in_diff=["scope", "fixture"],
        max_retries_expected=2,
    ),
    EvalCase(
        id="eval-010",
        repo="django/django",
        issue_number=35123,
        description="Fix migration autodetector missing index change",
        expected_files_modified=["django/db/migrations/autodetector.py"],
        expected_keywords_in_diff=["Index", "detect"],
        max_retries_expected=3,
    ),
]


# ── Eval Runner ───────────────────────────────────────────────────────────────

async def run_eval(case: EvalCase, dry_run: bool = False) -> EvalResult:
    """Run a single eval case against the real agent (or mock in dry_run)."""
    start = time.monotonic()

    if dry_run:
        # Simulate a run for testing the eval framework itself
        await asyncio.sleep(0.1)
        return EvalResult(
            case_id=case.id,
            passed=True,
            status="succeeded",
            retries=1,
            duration_seconds=0.1,
            tokens_used=5000,
            pr_url=f"https://github.com/{case.repo}/pull/999",
            failure_reason=None,
            diff_contains_expected=True,
        )

    from src.agent.graph import run_agent
    from src.agent.state import AgentState
    from src.tools.github import GitHubClient

    try:
        async with GitHubClient() as github:
            issue = await github.get_issue(case.repo, case.issue_number)

        initial_state: AgentState = {
            "issue": issue,
            "retry_count": 0,
            "attempt_history": [],
        }

        final_state = await run_agent(initial_state)

        duration = time.monotonic() - start
        status = final_state.get("status", "failed")
        pr = final_state.get("pull_request")
        pr_url = pr.get("pr_url") if pr else None
        retries = final_state.get("retry_count", 0)
        tokens = final_state.get("total_tokens_used", 0)

        # Check if expected keywords appear in the patch
        patch = final_state.get("code_patch")
        diff_text = ""
        if patch:
            for p in patch.patches:
                diff_text += p.unified_diff

        diff_ok = all(kw in diff_text for kw in case.expected_keywords_in_diff)
        passed = status in ("succeeded", "partial") and diff_ok

        return EvalResult(
            case_id=case.id,
            passed=passed,
            status=status,
            retries=retries,
            duration_seconds=duration,
            tokens_used=tokens,
            pr_url=pr_url,
            failure_reason=final_state.get("failure_reason"),
            diff_contains_expected=diff_ok,
        )

    except Exception as e:
        duration = time.monotonic() - start
        log.error("eval.case_exception", case_id=case.id, error=str(e))
        return EvalResult(
            case_id=case.id,
            passed=False,
            status="exception",
            retries=0,
            duration_seconds=duration,
            tokens_used=0,
            pr_url=None,
            failure_reason=str(e),
        )


async def run_all_evals(
    cases: list[EvalCase] | None = None,
    max_concurrent: int = 2,
    dry_run: bool = False,
) -> EvalSummary:
    """Run all eval cases with bounded concurrency."""
    target_cases = cases or GOLDEN_CASES
    sem = asyncio.Semaphore(max_concurrent)

    async def run_with_sem(case: EvalCase) -> EvalResult:
        async with sem:
            log.info("eval.running", case_id=case.id, repo=case.repo)
            result = await run_eval(case, dry_run=dry_run)
            status_icon = "✅" if result.passed else "❌"
            log.info(
                "eval.result",
                case_id=case.id,
                passed=result.passed,
                status=result.status,
                retries=result.retries,
                duration=f"{result.duration_seconds:.1f}s",
                icon=status_icon,
            )
            return result

    results = await asyncio.gather(*[run_with_sem(c) for c in target_cases])
    results_list = list(results)

    passed = sum(1 for r in results_list if r.passed)
    total = len(results_list)

    return EvalSummary(
        total=total,
        passed=passed,
        failed=total - passed,
        avg_retries=sum(r.retries for r in results_list) / max(total, 1),
        avg_duration_seconds=sum(r.duration_seconds for r in results_list) / max(total, 1),
        avg_tokens=sum(r.tokens_used for r in results_list) / max(total, 1),
        solve_rate_pct=passed / max(total, 1) * 100,
        results=results_list,
    )


def print_eval_table(summary: EvalSummary) -> None:
    """Print eval results as a formatted table."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Eval Results — {summary.timestamp}")

    table.add_column("Case", style="cyan")
    table.add_column("Status")
    table.add_column("Retries", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Diff OK")
    table.add_column("PR URL")

    for r in summary.results:
        status_color = "green" if r.passed else "red"
        table.add_row(
            r.case_id,
            f"[{status_color}]{r.status}[/{status_color}]",
            str(r.retries),
            f"{r.duration_seconds:.1f}s",
            f"{r.tokens_used:,}" if r.tokens_used else "—",
            "✅" if r.diff_contains_expected else "❌",
            r.pr_url or "—",
        )

    console.print(table)
    console.print(f"\n[bold]Solve rate:[/bold] {summary.solve_rate_pct:.1f}%  "
                  f"({summary.passed}/{summary.total})")
    console.print(f"[bold]Avg retries:[/bold] {summary.avg_retries:.1f}")
    console.print(f"[bold]Avg duration:[/bold] {summary.avg_duration_seconds:.1f}s")
    console.print(f"[bold]Avg tokens:[/bold] {summary.avg_tokens:,.0f}")


if __name__ == "__main__":
    import typer

    def main(
        dry_run: bool = typer.Option(False, help="Run without real API calls"),
        concurrency: int = typer.Option(2, help="Max concurrent eval cases"),
    ) -> None:
        summary = asyncio.run(
            run_all_evals(max_concurrent=concurrency, dry_run=dry_run)
        )
        print_eval_table(summary)

        # Write results to file for CI tracking
        results_file = Path("eval-results.json")
        results_file.write_text(json.dumps({
            "timestamp": summary.timestamp,
            "solve_rate_pct": summary.solve_rate_pct,
            "total": summary.total,
            "passed": summary.passed,
            "avg_retries": summary.avg_retries,
            "avg_tokens": summary.avg_tokens,
        }, indent=2))

        if summary.solve_rate_pct < 50.0:
            raise SystemExit(f"Eval failed: solve rate {summary.solve_rate_pct:.1f}% < 50%")

    typer.run(main)
