"""
src/worker/watchdog.py — Stuck run detector and reaper.

Without this, a crashed worker leaves runs permanently in "running" state,
Docker containers accumulate, and the queue fills with ghost jobs.

Runs as a separate process (or scheduled task) — checks every 5 minutes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import docker
import docker.errors
import structlog

from src.config import get_settings
from src.db.database import init_db
from src.db.repository import RunRepository
from src.observability.tracing import configure_logging
from src.tools.github import GitHubClient

log = structlog.get_logger(__name__)

# A run that has been "running" for longer than this is considered stuck
STUCK_THRESHOLD_MINUTES = 45

# Docker containers from this agent older than this are force-removed
STALE_CONTAINER_THRESHOLD_MINUTES = 60


async def run_watchdog_cycle() -> dict[str, int]:
    """
    Single watchdog cycle. Returns counts of actions taken.
    Safe to call repeatedly — all operations are idempotent.
    """
    results = {
        "stuck_runs_reaped": 0,
        "stale_containers_removed": 0,
        "errors": 0,
    }

    await asyncio.gather(
        _reap_stuck_runs(results),
        _remove_stale_containers(results),
        return_exceptions=True,
    )

    log.info(
        "watchdog.cycle_complete",
        stuck_runs=results["stuck_runs_reaped"],
        containers=results["stale_containers_removed"],
        errors=results["errors"],
    )
    return results


async def _reap_stuck_runs(results: dict[str, int]) -> None:
    """
    Find runs that have been in 'running' state too long and mark them failed.
    Also attempts to post a failure comment on the GitHub issue.
    """
    repo = RunRepository()
    threshold = datetime.now(timezone.utc) - timedelta(minutes=STUCK_THRESHOLD_MINUTES)

    try:
        # Get all runs that started before the threshold and are still running
        all_running = await repo.list_runs(status="running", limit=100)
        stuck = [r for r in all_running if r.started_at and r.started_at.replace(tzinfo=timezone.utc) < threshold]

        for run in stuck:
            log.warning(
                "watchdog.stuck_run_detected",
                run_id=run.run_id,
                issue=run.issue_number,
                repo=run.repo_full_name,
                started_at=str(run.started_at),
                threshold_minutes=STUCK_THRESHOLD_MINUTES,
            )

            try:
                await repo.update_run(
                    run_id=run.run_id,
                    status="failed",
                    failure_reason=(
                        f"Watchdog: run exceeded {STUCK_THRESHOLD_MINUTES}min timeout. "
                        f"The worker process likely crashed."
                    ),
                    completed_at=datetime.now(timezone.utc),
                )

                # Try to notify on GitHub
                await _post_watchdog_comment(
                    repo_full_name=run.repo_full_name,
                    issue_number=run.issue_number,
                    run_id=run.run_id,
                )

                results["stuck_runs_reaped"] += 1
                log.info("watchdog.run_reaped", run_id=run.run_id)

            except Exception as e:
                log.error("watchdog.reap_failed", run_id=run.run_id, error=str(e))
                results["errors"] += 1

    except Exception as e:
        log.error("watchdog.db_error", error=str(e))
        results["errors"] += 1


async def _remove_stale_containers(results: dict[str, int]) -> None:
    """
    Remove Docker containers from previous agent runs that were not cleaned up.
    Only removes containers labelled with 'swe-agent.run_id'.
    """
    loop = asyncio.get_running_loop()

    def _cleanup() -> int:
        removed = 0
        try:
            client = docker.from_env(timeout=10)
            threshold = datetime.now(timezone.utc) - timedelta(minutes=STALE_CONTAINER_THRESHOLD_MINUTES)

            containers = client.containers.list(
                all=True,
                filters={"label": "swe-agent.run_id"},
            )

            for container in containers:
                try:
                    # Parse container creation time
                    created_str = container.attrs.get("Created", "")
                    # Docker returns ISO 8601 with nanoseconds — trim to microseconds
                    created_str = created_str[:26] + "Z" if len(created_str) > 26 else created_str
                    created_dt = datetime.fromisoformat(created_str.rstrip("Z").replace("T", " ")).replace(
                        tzinfo=timezone.utc
                    )

                    if created_dt < threshold:
                        run_id = container.labels.get("swe-agent.run_id", "unknown")
                        log.warning(
                            "watchdog.stale_container",
                            container_id=container.short_id,
                            run_id=run_id,
                            created=str(created_dt),
                        )
                        container.remove(force=True)
                        removed += 1
                        log.info(
                            "watchdog.container_removed",
                            container_id=container.short_id,
                        )
                except docker.errors.APIError as e:
                    log.warning("watchdog.container_remove_failed", error=str(e))
                except (ValueError, KeyError):
                    log.debug("watchdog.container_time_parse_skipped", container_id=container.short_id)

        except docker.errors.DockerException as e:
            log.warning("watchdog.docker_unavailable", error=str(e))
        return removed

    try:
        removed = await loop.run_in_executor(None, _cleanup)
        results["stale_containers_removed"] = removed
    except Exception as e:
        log.error("watchdog.container_cleanup_error", error=str(e))
        results["errors"] += 1


async def _post_watchdog_comment(
    repo_full_name: str,
    issue_number: int,
    run_id: str,
) -> None:
    """Post a failure notice on the GitHub issue when a run is reaped."""
    body = (
        f"⏰ **SWE Agent** run `{run_id[:12]}...` was automatically terminated.\n\n"
        f"The agent exceeded the {STUCK_THRESHOLD_MINUTES}-minute timeout, likely due to a worker crash. "
        f"You can re-trigger by removing and re-adding the `agent-fix` label."
    )

    try:
        async with GitHubClient() as github:
            await github.comment_on_issue(
                repo_full_name=repo_full_name,
                issue_number=issue_number,
                body=body,
            )
    except Exception as e:
        log.warning("watchdog.comment_failed", error=str(e))


# ── Watchdog Runner ───────────────────────────────────────────────────────────


async def run_forever(interval_seconds: int = 300) -> None:
    """Run the watchdog in an infinite loop. Called from the entrypoint."""
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    await init_db()

    log.info(
        "watchdog.started",
        interval_seconds=interval_seconds,
        stuck_threshold_minutes=STUCK_THRESHOLD_MINUTES,
    )

    while True:
        try:
            await run_watchdog_cycle()
        except Exception as e:
            log.error("watchdog.cycle_error", error=str(e), exc_info=True)

        await asyncio.sleep(interval_seconds)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(run_forever())
