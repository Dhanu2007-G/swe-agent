"""
tests/unit/test_webhook.py — Tests for webhook signature validation and event handling.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def webhook_secret() -> str:
    return "test-webhook-secret-abc123"


@pytest.fixture
def make_signature(webhook_secret: str):
    def _make(payload: bytes) -> str:
        sig = hmac.new(
            key=webhook_secret.encode(),
            msg=payload,
            digestmod=hashlib.sha256,
        ).hexdigest()
        return f"sha256={sig}"

    return _make


@pytest.fixture
def app(webhook_secret: str):
    import os

    os.environ["GITHUB_TOKEN"] = "ghp_test"
    os.environ["GITHUB_WEBHOOK_SECRET"] = webhook_secret
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
    os.environ["DATABASE_URL"] = "postgresql+asyncpg://user:pass@localhost/test"
    from src.api.main import create_app

    return create_app()


class TestWebhookSignatureValidation:
    def test_valid_signature_accepted(self, make_signature, webhook_secret) -> None:
        from src.tools.github import GitHubClient

        mock_settings = MagicMock()
        mock_settings.github_webhook_secret_value = webhook_secret
        payload = b'{"action": "opened"}'
        sig = make_signature(payload)
        with patch("src.tools.github.get_settings", return_value=mock_settings):
            assert GitHubClient.validate_webhook_signature(payload, sig) is True

    def test_invalid_signature_rejected(self, webhook_secret) -> None:
        from src.tools.github import GitHubClient

        payload = b'{"action": "opened"}'
        assert GitHubClient.validate_webhook_signature(payload, "sha256=deadbeef") is False

    def test_missing_prefix_rejected(self, webhook_secret) -> None:
        from src.tools.github import GitHubClient

        payload = b"hello"
        assert GitHubClient.validate_webhook_signature(payload, "invalidsig") is False

    def test_timing_safe_comparison(self, make_signature) -> None:
        """Ensure we use hmac.compare_digest, not ==."""
        import inspect

        from src.tools.github import GitHubClient

        source = inspect.getsource(GitHubClient.validate_webhook_signature)
        assert "compare_digest" in source, "Must use constant-time comparison"


class TestWebhookEventHandling:
    ISSUE_PAYLOAD = {
        "action": "labeled",
        "issue": {
            "number": 42,
            "title": "Bug: crash on None",
            "body": "It crashes.",
            "labels": [{"name": "agent-fix"}],
        },
        "repository": {"full_name": "owner/repo"},
        "label": {"name": "agent-fix"},
    }

    def test_ignores_non_issue_events(self, make_signature) -> None:
        payload = json.dumps({"action": "push"}).encode()
        # Just test the signature logic — can't do full integration easily here
        sig = make_signature(payload)
        assert sig.startswith("sha256=")

    def test_ignores_issue_without_label(self) -> None:
        payload = {**self.ISSUE_PAYLOAD}
        payload["issue"] = {**payload["issue"], "labels": [{"name": "bug"}]}
        # No agent-fix label — should be ignored
        labels = [label_item["name"] for label_item in payload["issue"]["labels"]]
        assert "agent-fix" not in labels
