"""
src/api/routes.py — REST API for run management.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from src.api.rate_limit import API_TRIGGER_LIMITER
from src.db.repository import RunRepository
from src.worker.queue import enqueue_issue_job

log = structlog.get_logger(__name__)
router = APIRouter()


class TriggerRequest(BaseModel):
    repo_full_name: str
    issue_number: int


class RunResponse(BaseModel):
    run_id: str
    status: str
    issue_number: int
    repo_full_name: str
    pr_url: str | None = None
    retry_count: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    failure_reason: str | None = None


@router.post("/runs/trigger", status_code=status.HTTP_202_ACCEPTED)
async def trigger_run(body: TriggerRequest) -> dict[str, Any]:
    """Manually trigger an agent run for a GitHub issue."""
    allowed, _ = await API_TRIGGER_LIMITER.check(body.repo_full_name)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    repo = RunRepository()
    get_run_coro = repo.get_active_run(body.repo_full_name, body.issue_number)
    existing = await get_run_coro if hasattr(get_run_coro, "__await__") else get_run_coro
    if existing:
        return {"job_id": getattr(existing, "run_id", ""), "status": getattr(existing, "status", "")}

    enqueue_coro = enqueue_issue_job(
        repo_full_name=body.repo_full_name,
        issue_number=body.issue_number,
        delivery_id="manual",
    )
    job_id = await enqueue_coro if hasattr(enqueue_coro, "__await__") else enqueue_coro
    log.info("api.run_triggered", job_id=job_id, issue=body.issue_number, repo=body.repo_full_name)
    return {"job_id": job_id, "status": "queued"}


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str) -> RunResponse:
    """Get the status and result of a specific run."""
    repo = RunRepository()
    run = await repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return RunResponse(
        run_id=run.run_id,
        status=run.status,
        issue_number=run.issue_number,
        repo_full_name=run.repo_full_name,
        pr_url=run.pr_url,
        retry_count=run.retry_count,
        started_at=run.started_at.isoformat() if run.started_at else None,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
        failure_reason=run.failure_reason,
    )


@router.get("/runs", response_model=list[RunResponse])
async def list_runs(
    repo: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[RunResponse]:
    """List recent agent runs with optional filters."""
    run_repo = RunRepository()
    runs = await run_repo.list_runs(repo_full_name=repo, status=status, limit=min(limit, 100))
    return [
        RunResponse(
            run_id=r.run_id,
            status=r.status,
            issue_number=r.issue_number,
            repo_full_name=r.repo_full_name,
            pr_url=r.pr_url,
            retry_count=r.retry_count,
            started_at=r.started_at.isoformat() if r.started_at else None,
            completed_at=r.completed_at.isoformat() if r.completed_at else None,
        )
        for r in runs
    ]
