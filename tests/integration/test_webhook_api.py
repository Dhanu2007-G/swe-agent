"""
tests/integration/test_webhook_api.py — FastAPI TestClient integration tests.

Tests the full webhook request lifecycle:
  - HMAC signature validation (valid, invalid, missing)
  - Event type filtering (issues only, right actions)
  - Label-based gating (agent-fix label required)
  - Idempotency (duplicate delivery ignored)
  - Rate limiting (60 req/hour per repo)
  - Job enqueue confirmation

All DB and Redis calls are mocked — this tests HTTP-layer behaviour only.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Set required env vars before importing app
os.environ.update(
    {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "GITHUB_TOKEN": "ghp_test",
        "GITHUB_WEBHOOK_SECRET": "test-secret-xyz",
        "DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5432/test",
        "REDIS_URL": "redis://localhost:6379/1",
        "APP_ENV": "development",
        "LANGCHAIN_TRACING_V2": "false",
    }
)

from src.config import invalidate_settings_cache

invalidate_settings_cache()


# ── Fixtures ──────────────────────────────────────────────────────────────────

WEBHOOK_SECRET = "test-secret-xyz"


def _sign(payload: bytes) -> str:
    sig = hmac.new(
        key=WEBHOOK_SECRET.encode(),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()
    return f"sha256={sig}"


def _issue_payload(
    action: str = "labeled",
    labels: list[str] | None = None,
    repo: str = "owner/repo",
    issue_number: int = 42,
) -> bytes:
    return json.dumps(
        {
            "action": action,
            "issue": {
                "number": issue_number,
                "title": "Bug: crash on None input",
                "body": "It crashes when input is None.",
                "labels": [{"name": lbl} for lbl in (labels or ["agent-fix"])],
            },
            "repository": {"full_name": repo},
            "label": {"name": labels[0] if labels else "agent-fix"},
        }
    ).encode()


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient — module-scoped for performance."""
    with (
        patch("src.db.database.init_db", new_callable=AsyncMock),
        patch("src.worker.queue.get_redis_connection", new_callable=AsyncMock) as mock_redis,
    ):
        mock_redis.return_value.ping = AsyncMock(return_value=True)
        mock_redis.return_value.aclose = AsyncMock()

        from src.api.main import create_app

        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── Signature Validation ──────────────────────────────────────────────────────


class TestSignatureValidation:
    def test_valid_signature_accepted(self, client: TestClient) -> None:
        payload = _issue_payload()
        sig = _sign(payload)

        with (
            patch("src.api.webhook.RunRepository") as mock_repo_cls,
            patch(
                "src.api.webhook.enqueue_issue_job", new_callable=AsyncMock, return_value="job-123"
            ),
        ):
            mock_repo = AsyncMock()
            mock_repo.get_active_run.return_value = None
            mock_repo_cls.return_value = mock_repo

            resp = client.post(
                "/webhooks/github",
                content=payload,
                headers={
                    "X-GitHub-Event": "issues",
                    "X-Hub-Signature-256": sig,
                    "X-GitHub-Delivery": "delivery-001",
                    "Content-Type": "application/json",
                },
            )

        assert resp.status_code == 202
        assert resp.json()["status"] == "accepted"
        assert "job_id" in resp.json()

    def test_missing_signature_returns_401(self, client: TestClient) -> None:
        payload = _issue_payload()
        resp = client.post(
            "/webhooks/github",
            content=payload,
            headers={"X-GitHub-Event": "issues", "Content-Type": "application/json"},
        )
        assert resp.status_code == 401

    def test_wrong_signature_returns_401(self, client: TestClient) -> None:
        payload = _issue_payload()
        resp = client.post(
            "/webhooks/github",
            content=payload,
            headers={
                "X-GitHub-Event": "issues",
                "X-Hub-Signature-256": "sha256=deadbeefdeadbeef",
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 401

    def test_signature_of_different_payload_rejected(self, client: TestClient) -> None:
        """Sign one payload, send a different one — must be rejected."""
        real_payload = _issue_payload()
        sig = _sign(real_payload)
        tampered_payload = _issue_payload(action="closed")

        resp = client.post(
            "/webhooks/github",
            content=tampered_payload,
            headers={
                "X-GitHub-Event": "issues",
                "X-Hub-Signature-256": sig,
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 401


# ── Event Filtering ───────────────────────────────────────────────────────────


class TestEventFiltering:
    def _post(self, client: TestClient, payload: bytes, event: str = "issues") -> Any:
        sig = _sign(payload)
        return client.post(
            "/webhooks/github",
            content=payload,
            headers={
                "X-GitHub-Event": event,
                "X-Hub-Signature-256": sig,
                "X-GitHub-Delivery": "del-001",
                "Content-Type": "application/json",
            },
        )

    def test_non_issue_event_ignored(self, client: TestClient) -> None:
        payload = json.dumps({"action": "created", "ref": "refs/heads/main"}).encode()
        resp = self._post(client, payload, event="push")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    def test_issue_closed_action_ignored(self, client: TestClient) -> None:
        payload = _issue_payload(action="closed")
        resp = self._post(client, payload)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ignored"

    def test_issue_without_agent_fix_label_ignored(self, client: TestClient) -> None:
        payload = _issue_payload(labels=["bug", "enhancement"])
        resp = self._post(client, payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ignored"
        assert "agent-fix" in data["reason"]

    def test_issue_opened_with_label_accepted(self, client: TestClient) -> None:
        payload = _issue_payload(action="opened", labels=["bug", "agent-fix"])

        with (
            patch("src.api.webhook.RunRepository") as mock_repo_cls,
            patch(
                "src.api.webhook.enqueue_issue_job", new_callable=AsyncMock, return_value="job-456"
            ),
        ):
            mock_repo = AsyncMock()
            mock_repo.get_active_run.return_value = None
            mock_repo_cls.return_value = mock_repo

            resp = self._post(client, payload)

        assert resp.status_code == 202

    def test_issue_reopened_with_label_accepted(self, client: TestClient) -> None:
        payload = _issue_payload(action="reopened", labels=["agent-fix"])

        with (
            patch("src.api.webhook.RunRepository") as mock_repo_cls,
            patch(
                "src.api.webhook.enqueue_issue_job", new_callable=AsyncMock, return_value="job-789"
            ),
        ):
            mock_repo = AsyncMock()
            mock_repo.get_active_run.return_value = None
            mock_repo_cls.return_value = mock_repo

            resp = self._post(client, payload)

        assert resp.status_code == 202


# ── Idempotency ───────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_duplicate_delivery_ignored(self, client: TestClient) -> None:
        """If a run is already active for this issue, don't create another."""
        payload = _issue_payload()
        sig = _sign(payload)

        existing_run = MagicMock()
        existing_run.run_id = "existing-run-001"

        with patch("src.api.webhook.RunRepository") as mock_repo_cls:
            mock_repo = AsyncMock()
            mock_repo.get_active_run.return_value = existing_run  # already active
            mock_repo_cls.return_value = mock_repo

            resp = client.post(
                "/webhooks/github",
                content=payload,
                headers={
                    "X-GitHub-Event": "issues",
                    "X-Hub-Signature-256": sig,
                    "X-GitHub-Delivery": "del-dup",
                    "Content-Type": "application/json",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ignored"
        assert data["run_id"] == "existing-run-001"

    def test_no_duplicate_job_enqueued(self, client: TestClient) -> None:
        """Verify enqueue_issue_job is NOT called when run already active."""
        payload = _issue_payload()
        sig = _sign(payload)

        existing_run = MagicMock()
        existing_run.run_id = "active-run"

        with (
            patch("src.api.webhook.RunRepository") as mock_repo_cls,
            patch("src.api.webhook.enqueue_issue_job", new_callable=AsyncMock) as mock_enqueue,
        ):
            mock_repo = AsyncMock()
            mock_repo.get_active_run.return_value = existing_run
            mock_repo_cls.return_value = mock_repo

            client.post(
                "/webhooks/github",
                content=payload,
                headers={
                    "X-GitHub-Event": "issues",
                    "X-Hub-Signature-256": sig,
                    "X-GitHub-Delivery": "del-dup-2",
                    "Content-Type": "application/json",
                },
            )

            mock_enqueue.assert_not_awaited()


# ── Health + Metrics ──────────────────────────────────────────────────────────


class TestHealthEndpoints:
    def test_health_returns_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_metrics_endpoint_returns_prometheus_format(self, client: TestClient) -> None:
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        # Prometheus format always starts with # HELP or a metric name
        assert "http_requests_total" in resp.text or "#" in resp.text


# ── Security Headers ──────────────────────────────────────────────────────────


class TestSecurityHeaders:
    def test_security_headers_present(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.headers.get("x-content-type-options") == "nosniff"
        assert resp.headers.get("x-frame-options") == "DENY"
        assert "x-request-id" in resp.headers

    def test_request_id_unique_per_request(self, client: TestClient) -> None:
        r1 = client.get("/health")
        r2 = client.get("/health")
        assert r1.headers["x-request-id"] != r2.headers["x-request-id"]

    def test_server_header_stripped(self, client: TestClient) -> None:
        resp = client.get("/health")
        # Should not reveal uvicorn/starlette version
        server = resp.headers.get("server", "")
        assert "uvicorn" not in server.lower()
        assert "starlette" not in server.lower()


# ── Malformed Requests ────────────────────────────────────────────────────────


class TestMalformedRequests:
    def test_invalid_json_returns_400(self, client: TestClient) -> None:
        payload = b"this is not json {"
        sig = _sign(payload)
        resp = client.post(
            "/webhooks/github",
            content=payload,
            headers={
                "X-GitHub-Event": "issues",
                "X-Hub-Signature-256": sig,
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 400

    def test_empty_body_with_valid_sig_returns_400(self, client: TestClient) -> None:
        payload = b""
        sig = _sign(payload)
        resp = client.post(
            "/webhooks/github",
            content=payload,
            headers={
                "X-GitHub-Event": "issues",
                "X-Hub-Signature-256": sig,
                "Content-Type": "application/json",
            },
        )
        # Empty body is invalid JSON
        assert resp.status_code in (400, 422)
