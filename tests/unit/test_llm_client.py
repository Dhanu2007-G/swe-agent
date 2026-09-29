"""Tests for LiteLLM multi-provider client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.llm_client import get_llm_response
from src.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(
        ANTHROPIC_API_KEY="test-key",
        LLM_MODEL="claude-3-5-sonnet-20241022",
    )


def _make_mock_response(content: str | None) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    return mock_response


def test_get_llm_response_returns_content(settings: Settings) -> None:
    """Normal path: response with string content is returned."""
    with patch(
        "src.agent.llm_client.completion", return_value=_make_mock_response("Hello world")
    ) as mock_completion:
        result = get_llm_response(settings, [{"role": "user", "content": "Hi"}])
    assert result == "Hello world"
    mock_completion.assert_called_once()


def test_get_llm_response_none_content_returns_empty(settings: Settings) -> None:
    """When model returns None content, empty string is returned."""
    with patch("src.agent.llm_client.completion", return_value=_make_mock_response(None)):
        result = get_llm_response(settings, [{"role": "user", "content": "Hi"}])
    assert result == ""


def test_get_llm_response_sets_anthropic_env(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """API keys from settings are set as environment variables for litellm."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with (
        patch("src.agent.llm_client.completion", return_value=_make_mock_response("ok")),
        patch.object(Settings, "anthropic_api_key_value", "sk-ant-test-123"),
    ):
        import os

        get_llm_response(settings, [])
        assert os.environ.get("ANTHROPIC_API_KEY") == "sk-ant-test-123"


def test_get_llm_response_with_gemini_model(settings: Settings) -> None:
    """LLM_MODEL can be set to a Gemini model."""
    settings.LLM_MODEL = "gemini/gemini-1.5-pro"  # type: ignore[assignment]
    settings.GEMINI_API_KEY = "test-gemini-key"  # type: ignore[assignment]
    with patch(
        "src.agent.llm_client.completion", return_value=_make_mock_response("gemini response")
    ) as mock_comp:
        result = get_llm_response(settings, [{"role": "user", "content": "Hi"}])
    assert result == "gemini response"
    call_kwargs = mock_comp.call_args
    assert call_kwargs[1]["model"] == "gemini/gemini-1.5-pro"


def test_get_llm_response_falls_back_to_anthropic_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When LLM_MODEL is not set, falls back to ANTHROPIC_MODEL."""
    monkeypatch.delenv("LLM_MODEL", raising=False)
    settings = Settings(
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_MODEL="claude-3-haiku-20240307",
        LLM_MODEL="",
    )
    with patch(
        "src.agent.llm_client.completion", return_value=_make_mock_response("ok")
    ) as mock_comp:
        get_llm_response(settings, [])
    call_model = mock_comp.call_args[1]["model"]
    assert "claude" in call_model


def test_get_llm_response_skips_none_api_keys(settings: Settings) -> None:
    """None API keys are not set in environment."""
    settings.OPENAI_API_KEY = None  # type: ignore[assignment]
    with patch("src.agent.llm_client.completion", return_value=_make_mock_response("ok")):
        import os

        original = os.environ.get("OPENAI_API_KEY")
        get_llm_response(settings, [])
        # Should not set None values
        current = os.environ.get("OPENAI_API_KEY")
        assert current == original


def test_get_llm_response_custom_params(settings: Settings) -> None:
    """max_tokens and temperature are forwarded to completion."""
    with patch(
        "src.agent.llm_client.completion", return_value=_make_mock_response("ok")
    ) as mock_comp:
        get_llm_response(settings, [], max_tokens=1024, temperature=0.0)
    kwargs = mock_comp.call_args[1]
    assert kwargs["max_tokens"] == 1024
    assert kwargs["temperature"] == 0.0
