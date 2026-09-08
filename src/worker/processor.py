"""
src/worker/processor.py — RQ job processor.
This is the function called by RQ workers. Sets up the agent, runs it, persists results.
"""
from __future__ import annotations

import asyncio
import traceback
from datetime import datetime, timezone

import structlog

from src.agent.checkpointing import checkpointed_run
from src.agent.graph import run_agent
from src.agent.state import AgentState, GithubIssue, RunStatus
from src.config import get_settings
from src.db.repository import RunRepository
from src.observability.tracing import (
    configure_logging,
    decrement_active_runs,
    increment_active_runs,
)
from src.tools.github import GitHubClient
from src.worker.queue import get_redis_connection, release_active_job_lock

log = structlog.get_logger(__name__)


def process_issue_job(
    repo_full_name: str,
    issue_number: int,
    run_id: str,
) -> dict:
    """
    Entry point called by RQ. Runs the async agent in a new event loop.
    Returns a summary dict for RQ result storage.
    """
    # RQ workers are sync — we run async code in a fresh event loop
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    log.info("worker.job_start", run_id=run_id, repo=repo_full_name,
             issue=issue_number)

    try:
        result = asyncio.run(
            _run_agent_async(repo_full_name, issue_number, run_id)
        )
        log.info("worker.job_complete", run_id=run_id, status=result.get("status"))
        return result
    except Exception as e:
        log.error("worker.job_fatal_error", run_id=run_id,
                  error=str(e), tb=traceback.format_exc())
        # Try to persist the failure
        asyncio.run(_persist_failure(run_id, repo_full_name, issue_number, str(e)))
        raise  # Re-raise so RQ marks the job as failed


async def _run_agent_async(
    repo_full_name: str,
    issue_number: int,
    run_id: str,
) -> dict:
    """The actual async agent execution."""
    repo = RunRepository()
    await repo.mark_run_running(run_id)
    increment_active_runs()

    try:
        # Update Redis status
        redis = await get_redis_connection()
        await redis.hset(f"job:{run_id}", "status", "running")

        # Fetch the issue
        async with GitHubClient() as github:
            issue = await github.get_issue(repo_full_name, issue_number)

        # Assemble initial state
        initial_state: AgentState = {
            "run_id": run_id,
            "issue": issue,
            "retry_count": 0,
            "attempt_history": [],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "total_tokens_used": 0,
        }

        # Run the agent with checkpointing enabled
        try:
            async with checkpointed_run(run_id) as checkpointer:
                final_state = await run_agent(initial_state, checkpointer=checkpointer)
        except Exception:
            # Fallback: run without checkpointing if Postgres checkpoint tables fail
            log.warning("worker.checkpointing_fallback", run_id=run_id,
                        reason="Checkpointer unavailable, running without state snapshots")
            final_state = await run_agent(initial_state)

        # Extract results
        status = final_state.get("status", RunStatus.FAILED.value)
        pr = final_state.get("pull_request")
        pr_url = pr.get("pr_url") if isinstance(pr, dict) else (getattr(pr, "pr_url", None) if pr else None)
        retry_count = final_state.get("retry_count", 0)
        failure_reason = final_state.get("failure_reason")

        # Serialize state snapshot for post-mortem debugging
        import json
        try:
            state_snapshot = json.dumps(
                {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                 for k, v in final_state.items()},
                default=str,
            )
        except Exception:
            state_snapshot = None

        await repo.update_run(
            run_id=run_id,
            status=status,
            pr_url=pr_url,
            retry_count=retry_count,
            failure_reason=failure_reason,
            completed_at=datetime.now(timezone.utc),
            state_snapshot=state_snapshot,
        )

        # Update Redis
        updates = {"status": status}
        if pr_url:
            updates["pr_url"] = pr_url
        await redis.hset(f"job:{run_id}", mapping=updates)

        # Post a status comment on the issue
        try:
            await _post_status_comment(issue, status, pr_url, retry_count)
        except Exception as e:
            log.warning("worker.comment_failed", error=str(e))

        return {
            "run_id": run_id,
            "status": status,
            "pr_url": pr_url,
            "retry_count": retry_count,
        }
    finally:
        decrement_active_runs()
        await release_active_job_lock(repo_full_name, issue_number, run_id)


async def _persist_failure(
    run_id: str, repo: str, issue: int, error: str
) -> None:
    """Best-effort failure persistence."""
    try:
        run_repo = RunRepository()
        await run_repo.update_run(
            run_id=run_id,
            status=RunStatus.FAILED.value,
            failure_reason=error[:500],
            completed_at=datetime.now(timezone.utc),
        )
        redis = await get_redis_connection()
        await redis.hset(f"job:{run_id}", mapping={"status": "failed"})
    except Exception:
        pass
    finally:
        await release_active_job_lock(repo, issue, run_id)


async def _post_status_comment(
    issue: GithubIssue,
    status: str,
    pr_url: str | None,
    retry_count: int,
) -> None:
    """Post a status comment on the GitHub issue."""
    if status == RunStatus.SUCCEEDED.value:
        body = (
            f"✅ **SWE Agent** has submitted a fix: {pr_url}\n\n"
            f"The fix passed all tests after {retry_count + 1} attempt(s). "
            f"Please review the PR."
        )
    elif status == RunStatus.PARTIAL.value:
        body = (
            f"⚠️ **SWE Agent** opened a draft PR with a partial fix: {pr_url}\n\n"
            f"Tests are still failing after {retry_count} attempt(s). "
            f"Human review required."
        )
    else:
        body = (
            f"❌ **SWE Agent** was unable to generate a passing fix after "
            f"{retry_count} attempt(s).\n\n"
            f"The issue may require manual investigation."
        )

    async with GitHubClient() as github:
        await github.comment_on_issue(
            repo_full_name=issue.repo_full_name,
            issue_number=issue.issue_number,
            body=body,
        )
