"""
src/api/webhook.py — GitHub webhook receiver.
Validates HMAC signature, filters relevant events, enqueues jobs idempotently.
"""

from __future__ import annotations

import json
from typing import Annotated

import structlog
from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

import src.worker.queue as queue_module
from src.api.rate_limit import WEBHOOK_LIMITER
from src.db.repository import RunRepository
from src.tools.github import GitHubClient
from src.worker.queue import enqueue_issue_job

log = structlog.get_logger(__name__)
router = APIRouter()

# Events we care about
HANDLED_ISSUE_ACTIONS = {"opened", "reopened", "labeled"}
AGENT_TRIGGER_LABEL = "agent-fix"


@router.post("/github")
async def github_webhook(
    request: Request,
    x_github_event: Annotated[str | None, Header()] = None,
    x_hub_signature_256: Annotated[str | None, Header()] = None,
    x_github_delivery: Annotated[str | None, Header()] = None,
) -> JSONResponse:
    """
    Receive GitHub webhook events.
    Only triggers on issues labeled with 'agent-fix'.
    """
    # ── Validate signature ────────────────────────────────────────────────────
    payload = await request.body()

    if not x_hub_signature_256:
        log.warning("webhook.missing_signature", delivery=x_github_delivery)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing webhook signature",
        )

    if not GitHubClient.validate_webhook_signature(payload, x_hub_signature_256):
        log.warning("webhook.invalid_signature", delivery=x_github_delivery)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        )

    # ── Parse event ───────────────────────────────────────────────────────────
    try:
        event_data = json.loads(payload)
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON payload",
        ) from err

    event_type = x_github_event or "unknown"
    action = event_data.get("action", "")
    delivery_id = x_github_delivery or "unknown"

    log.info("webhook.received", event_type=event_type, action=action, delivery=delivery_id)

    # ── Filter to actionable events ───────────────────────────────────────────
    if event_type != "issues":
        return JSONResponse({"status": "ignored", "reason": f"event type '{event_type}'"})

    repo_data = event_data.get("repository")
    repo_full_name = repo_data.get("full_name") if isinstance(repo_data, dict) else None

    # ── Rate limiting ─────────────────────────────────────────────────────────
    if repo_full_name:
        allowed, _ = await WEBHOOK_LIMITER.check(repo_full_name)
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
            )

    issue_data = event_data.get("issue")
    if (
        not isinstance(issue_data, dict)
        or not isinstance(repo_data, dict)
        or not issue_data.get("number")
        or not repo_full_name
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing issue or repository data",
        )

    issue_number = issue_data["number"]

    # ── Delivery deduplication ────────────────────────────────────────────────
    if x_github_delivery:
        try:
            redis = await queue_module.get_redis_connection()
            dedup_key = f"webhook:delivery:{x_github_delivery}"
            res = redis.set(dedup_key, "1", nx=True, ex=3600)
            is_new = await res if hasattr(res, "__await__") else res
            if not is_new:
                return JSONResponse(
                    {
                        "status": "ignored",
                        "reason": "duplicate delivery",
                    }
                )
        except Exception:
            pass

    if action not in HANDLED_ISSUE_ACTIONS:
        return JSONResponse({"status": "ignored", "reason": f"action '{action}'"})

    # Check for trigger label
    labels = [
        item.get("name", "") for item in issue_data.get("labels", []) if isinstance(item, dict)
    ]
    if AGENT_TRIGGER_LABEL not in labels:
        return JSONResponse(
            {
                "status": "ignored",
                "reason": f"label '{AGENT_TRIGGER_LABEL}' not present",
            }
        )

    # ── Idempotency check — don't re-run for same issue ───────────────────────
    repo = RunRepository()
    existing = await repo.get_active_run(repo_full_name, issue_number)
    if existing:
        run_id_val = str(getattr(existing, "run_id", "unknown"))
        log.info("webhook.duplicate_ignored", issue=issue_number, run_id=run_id_val)
        return JSONResponse(
            {
                "status": "ignored",
                "reason": "run already active",
                "run_id": run_id_val,
            }
        )

    # ── Enqueue job ───────────────────────────────────────────────────────────
    job_id = await enqueue_issue_job(
        repo_full_name=repo_full_name,
        issue_number=issue_number,
        delivery_id=delivery_id,
    )

    log.info("webhook.job_enqueued", job_id=job_id, issue=issue_number, repo=repo_full_name)

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"status": "accepted", "job_id": job_id},
    )
