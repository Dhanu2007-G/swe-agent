from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


class TestWebhookCoverage:
    @pytest.fixture
    def client(self) -> TestClient:
        from src.api.main import create_app

        settings = SimpleNamespace(
            is_production=False,
            allowed_hosts=["localhost"],
            allowed_repos=None,
        )
        with patch("src.api.main.get_settings", return_value=settings):
            app = create_app()
        return TestClient(app)

    @patch("src.worker.queue.get_redis_connection")
    @patch("src.tools.github.GitHubClient.validate_webhook_signature", return_value=True)
    def test_webhook_invalid_payload(self, mock_validate, mock_get_redis, client) -> None:
        response = client.post(
            "/webhooks/github",
            content=b"not a json",
            headers={
                "X-GitHub-Event": "issues",
                "X-GitHub-Delivery": "foo",
                "X-Hub-Signature-256": "sha256=123",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 400

    def test_webhook_missing_signature(self, client) -> None:
        response = client.post(
            "/webhooks/github",
            json={"action": "opened"},
            headers={"X-GitHub-Event": "issues", "X-GitHub-Delivery": "foo"},
        )
        assert response.status_code == 401

    @patch("src.tools.github.GitHubClient.validate_webhook_signature", return_value=False)
    def test_webhook_invalid_signature_fails(self, mock_validate, client) -> None:
        response = client.post(
            "/webhooks/github",
            json={"action": "opened"},
            headers={
                "X-GitHub-Event": "issues",
                "X-GitHub-Delivery": "foo",
                "X-Hub-Signature-256": "sha256=123",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 401

    @patch("src.tools.github.GitHubClient.validate_webhook_signature", return_value=True)
    @patch("src.worker.queue.get_redis_connection")
    def test_webhook_missing_issue_or_repo(self, mock_get_redis, mock_validate, client) -> None:
        response = client.post(
            "/webhooks/github",
            json={"action": "opened", "issue": {}, "repository": {}},
            headers={
                "X-GitHub-Event": "issues",
                "X-GitHub-Delivery": "foo",
                "X-Hub-Signature-256": "sha256=123",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 400

    @patch("src.tools.github.GitHubClient.validate_webhook_signature", return_value=True)
    @patch("src.worker.queue.get_redis_connection")
    @patch("src.api.webhook.RunRepository")
    def test_webhook_duplicate_delivery(
        self, mock_repo, mock_get_redis, mock_validate, client
    ) -> None:
        mock_redis = AsyncMock()
        mock_redis.set.return_value = False
        mock_get_redis.return_value = mock_redis
        response = client.post(
            "/webhooks/github",
            json={
                "action": "opened",
                "issue": {"number": 1, "labels": [{"name": "agent-fix"}]},
                "repository": {"full_name": "owner/repo"},
            },
            headers={
                "X-GitHub-Event": "issues",
                "X-GitHub-Delivery": "foo",
                "X-Hub-Signature-256": "sha256=123",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200
        assert response.json()["reason"] == "duplicate delivery"

    @patch("src.tools.github.GitHubClient.validate_webhook_signature", return_value=True)
    @patch("src.worker.queue.get_redis_connection")
    @patch("src.api.webhook.RunRepository")
    def test_webhook_active_run_exists(
        self, mock_repo_cls, mock_get_redis, mock_validate, client
    ) -> None:
        run = SimpleNamespace(run_id="run-123")
        mock_repo = mock_repo_cls.return_value
        mock_repo.get_active_run = AsyncMock(return_value=run)

        mock_redis = AsyncMock()
        mock_redis.set.return_value = True
        mock_get_redis.return_value = mock_redis

        response = client.post(
            "/webhooks/github",
            json={
                "action": "opened",
                "issue": {"number": 1, "labels": [{"name": "agent-fix"}]},
                "repository": {"full_name": "owner/repo"},
            },
            headers={
                "X-GitHub-Event": "issues",
                "X-GitHub-Delivery": "foo",
                "X-Hub-Signature-256": "sha256=123",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200
        assert response.json()["reason"] == "run already active"

    @patch("src.tools.github.GitHubClient.validate_webhook_signature", return_value=True)
    @patch("src.api.webhook.WEBHOOK_LIMITER.check", new_callable=AsyncMock)
    def test_webhook_rate_limit_exceeded(self, mock_check, mock_validate, client) -> None:
        mock_check.return_value = (False, {})
        response = client.post(
            "/webhooks/github",
            json={"action": "opened", "repository": {"full_name": "owner/repo"}},
            headers={
                "X-GitHub-Event": "issues",
                "X-GitHub-Delivery": "foo",
                "X-Hub-Signature-256": "sha256=123",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 429

    @patch("src.tools.github.GitHubClient.validate_webhook_signature", return_value=True)
    @patch("src.worker.queue.get_redis_connection", new_callable=AsyncMock)
    @patch("src.api.webhook.RunRepository")
    @patch("src.api.webhook.WEBHOOK_LIMITER.check", new_callable=AsyncMock)
    def test_webhook_redis_exception(
        self, mock_check, mock_repo_cls, mock_get_redis, mock_validate, client
    ) -> None:
        mock_check.return_value = (True, {})
        mock_redis = AsyncMock()
        mock_redis.set.side_effect = Exception("Redis dedup error")
        mock_get_redis.return_value = mock_redis

        run = SimpleNamespace(run_id="run-123")
        mock_repo = mock_repo_cls.return_value
        mock_repo.get_active_run = AsyncMock(return_value=run)

        response = client.post(
            "/webhooks/github",
            json={
                "action": "opened",
                "issue": {"number": 1, "labels": [{"name": "agent-fix"}]},
                "repository": {"full_name": "owner/repo"},
            },
            headers={
                "X-GitHub-Event": "issues",
                "X-GitHub-Delivery": "foo",
                "X-Hub-Signature-256": "sha256=123",
                "Content-Type": "application/json",
            },
        )
        assert response.status_code == 200
        assert response.json()["reason"] == "run already active"
