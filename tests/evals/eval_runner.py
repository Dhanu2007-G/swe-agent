"""
tests/evals/eval_runner.py — Evaluation harness against a golden issue set.
Run in CI to track solve rate, latency, and token cost over time.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
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
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# ── Golden Issue Set ──────────────────────────────────────────────────────────


def _load_golden_cases() -> list[EvalCase]:
    """Load eval cases from golden_issues.json (next to this file).

    Separating benchmark data from code allows adding/editing cases without
    touching Python source or re-running linters/formatters.
    """
    if not GOLDEN_ISSUES_FILE.exists():
        log.error(
            "eval.golden_issues_missing",
            path=str(GOLDEN_ISSUES_FILE),
            hint="Run: cp tests/evals/golden_issues.json.example tests/evals/golden_issues.json",
        )
        return []

    raw = json.loads(GOLDEN_ISSUES_FILE.read_text(encoding="utf-8"))
    cases: list[EvalCase] = []
    for entry in raw:
        cases.append(
            EvalCase(
                id=entry["id"],
                repo=entry["repo"],
                issue_number=entry["issue_number"],
                description=entry["description"],
                expected_files_modified=entry["expected_files_modified"],
                expected_keywords_in_diff=entry["expected_keywords_in_diff"],
                max_retries_expected=entry.get("max_retries_expected", 2),
            )
        )
    return cases


GOLDEN_CASES: list[EvalCase] = _load_golden_cases()


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
    from src.tools.github import GitHubClient

    try:
        async with GitHubClient() as github:
            issue = await github.get_issue(case.repo, case.issue_number)

        initial_state: dict[str, Any] = {
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
    console.print(
        f"\n[bold]Solve rate:[/bold] {summary.solve_rate_pct:.1f}%  "
        f"({summary.passed}/{summary.total})"
    )
    console.print(f"[bold]Avg retries:[/bold] {summary.avg_retries:.1f}")
    console.print(f"[bold]Avg duration:[/bold] {summary.avg_duration_seconds:.1f}s")
    console.print(f"[bold]Avg tokens:[/bold] {summary.avg_tokens:,.0f}")


if __name__ == "__main__":
    import typer

    def main(
        dry_run: bool = typer.Option(False, help="Run without real API calls"),
        concurrency: int = typer.Option(2, help="Max concurrent eval cases"),
    ) -> None:
        summary = asyncio.run(run_all_evals(max_concurrent=concurrency, dry_run=dry_run))
        print_eval_table(summary)

        # Write results to file for CI tracking
        results_file = Path("eval-results.json")
        results_file.write_text(
            json.dumps(
                {
                    "timestamp": summary.timestamp,
                    "solve_rate_pct": summary.solve_rate_pct,
                    "total": summary.total,
                    "passed": summary.passed,
                    "avg_retries": summary.avg_retries,
                    "avg_tokens": summary.avg_tokens,
                },
                indent=2,
            )
        )

        if summary.solve_rate_pct < 50.0:
            raise SystemExit(f"Eval failed: solve rate {summary.solve_rate_pct:.1f}% < 50%")

    typer.run(main)
