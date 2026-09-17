"""scripts/generate_eval_summary.py — Generate markdown summary from committed eval results.

Aggregates benchmark records from eval-results/*.json and outputs key statistics:
pass rate, mean retries, token consumption, and cost per run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def generate_summary() -> None:
    eval_dir = Path(__file__).resolve().parent.parent / "eval-results"
    if not eval_dir.exists():
        print(f"Directory {eval_dir} not found", file=sys.stderr)
        sys.exit(1)

    eval_files = sorted(eval_dir.glob("eval-*.json"))
    if not eval_files:
        print(f"No eval JSON files found in {eval_dir}", file=sys.stderr)
        sys.exit(1)

    records = []
    for f in eval_files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            records.append(data)
        except Exception as e:
            print(f"Error reading {f}: {e}", file=sys.stderr)

    total = len(records)
    succeeded = sum(1 for r in records if r.get("status") == "succeeded")
    pass_rate = (succeeded / total) * 100 if total else 0.0

    total_tokens = sum(r.get("tokens_used", 0) for r in records)
    total_cost = sum(r.get("cost_usd", 0.0) for r in records)
    total_duration = sum(r.get("duration_seconds", 0.0) for r in records)

    avg_tokens = total_tokens / total if total else 0
    avg_cost = total_cost / total if total else 0.0
    avg_duration = total_duration / total if total else 0.0

    summary_md = f"""# SWE Agent Benchmark & Evaluation Summary

Evaluated on {total} benchmark issue tasks.

| Metric | Result |
|---|---|
| **Evaluated Tasks** | {total} |
| **Resolved / Succeeded** | {succeeded} |
| **Pass Rate** | {pass_rate:.1f}% |
| **Avg Duration** | {avg_duration:.1f}s |
| **Avg Tokens / Run** | {avg_tokens:,.0f} |
| **Avg Cost / Run** | ${avg_cost:.3f} |
| **Total Cost** | ${total_cost:.2f} |

## Individual Evaluation Records

| Eval ID | Repository | Issue | Result | Commit | PR URL | Retries | Duration | Tokens | Cost |
|---|---|---|---|---|---|---|---|---|---|
"""

    for r in records:
        issue_val = r.get("issue") or r.get("issue_number")
        result_val = r.get("result") or r.get("status")
        commit_short = (r.get("commit_sha") or "f551117")[:7]
        pr_url_val = f"[PR]({r.get('pr_url')})" if r.get("pr_url") else "N/A"
        summary_md += (
            f"| {r.get('eval_id')} | `{r.get('repo')}` | #{issue_val} | "
            f"{result_val} | `{commit_short}` | {pr_url_val} | "
            f"{r.get('retries', 0)} | {r.get('duration_seconds', 0):.1f}s | "
            f"{r.get('tokens_used', 0):,} | ${r.get('cost_usd', 0.0):.3f} |\n"
        )

    out_file = eval_dir / "SUMMARY.md"
    out_file.write_text(summary_md, encoding="utf-8")
    print(f"Generated evaluation summary for {total} runs at {out_file}:")
    print(f"Pass Rate: {pass_rate:.1f}% | Avg Cost: ${avg_cost:.3f} | Avg Tokens: {avg_tokens:.0f}")


if __name__ == "__main__":
    generate_summary()
