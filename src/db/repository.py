"""
src/db/repository.py — Repository pattern for all DB operations.
Keeps SQLAlchemy out of business logic. All methods async.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.database import AgentRun, get_session

log = structlog.get_logger(__name__)


class RunRepository:
    """All persistence operations for AgentRun records."""

    async def create_run(
        self,
        run_id: str,
        repo_full_name: str,
        issue_number: int,
        status: str = "running",
        started_at: datetime | None = None,
    ) -> AgentRun:
        async for session in get_session():
            run = AgentRun(
                run_id=run_id,
                repo_full_name=repo_full_name,
                issue_number=issue_number,
                status=status,
                started_at=started_at or datetime.now(timezone.utc),
            )
            session.add(run)
            await session.flush()
            log.info("db.run_created", run_id=run_id)
            return run
        raise RuntimeError("Session exhausted")

    async def mark_run_running(self, run_id: str) -> None:
        """Mark a pending run as running."""
        await self.update_run(
            run_id=run_id,
            status="running",
            started_at=datetime.now(timezone.utc),
        )

    async def get_run(self, run_id: str) -> AgentRun | None:
        async for session in get_session():
            result = await session.execute(
                select(AgentRun).where(AgentRun.run_id == run_id)
            )
            return result.scalar_one_or_none()
        return None

    async def get_active_run(
        self, repo_full_name: str, issue_number: int
    ) -> AgentRun | None:
        """Check if an active (running/queued) run exists for this issue."""
        async for session in get_session():
            result = await session.execute(
                select(AgentRun).where(
                    AgentRun.repo_full_name == repo_full_name,
                    AgentRun.issue_number == issue_number,
                    AgentRun.status.in_(["pending", "running"]),
                )
            )
            return result.scalar_one_or_none()
        return None

    async def update_run(
        self,
        run_id: str,
        status: str,
        pr_url: str | None = None,
        retry_count: int = 0,
        failure_reason: str | None = None,
        completed_at: datetime | None = None,
        state_snapshot: str | None = None,
        started_at: datetime | None = None,
    ) -> None:
        values: dict[str, Any] = {"status": status, "retry_count": retry_count}
        if pr_url:
            values["pr_url"] = pr_url
        if failure_reason:
            values["failure_reason"] = failure_reason[:500]
        if completed_at:
            values["completed_at"] = completed_at
        if state_snapshot:
            values["state_snapshot"] = state_snapshot
        if started_at:
            values["started_at"] = started_at

        async for session in get_session():
            await session.execute(
                update(AgentRun)
                .where(AgentRun.run_id == run_id)
                .values(**values)
            )
            log.info("db.run_updated", run_id=run_id, status=status)

    async def list_runs(
        self,
        repo_full_name: str | None = None,
        status: str | None = None,
        limit: int = 20,
    ) -> list[AgentRun]:
        async for session in get_session():
            query = select(AgentRun).order_by(AgentRun.created_at.desc()).limit(limit)
            if repo_full_name:
                query = query.where(AgentRun.repo_full_name == repo_full_name)
            if status:
                query = query.where(AgentRun.status == status)
            result = await session.execute(query)
            return list(result.scalars().all())
        return []
