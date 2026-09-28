"""
Helios SWE-Agent — LiteLLM multi-provider LLM client.

Supports: Anthropic (Claude), OpenAI (GPT-4o), Google Gemini,
          Groq, OpenRouter, and local Ollama models.
Configuration is via environment variables in .env.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import litellm
from litellm import completion

if TYPE_CHECKING:
    from src.config import Settings

# Suppress litellm verbose output
litellm.suppress_debug_info = True
litellm.set_verbose = False  # type: ignore[attr-defined]


def get_llm_response(
    settings: Settings,
    messages: list[dict[str, str]],
    max_tokens: int = 4096,
    temperature: float = 0.2,
) -> str:
    """
    Send a chat completion request to the configured LLM provider.

    Provider selection is automatic based on the LLM_MODEL setting:
    - claude-3-5-sonnet-20241022  → Anthropic (uses ANTHROPIC_API_KEY)
    - gpt-4o                      → OpenAI (uses OPENAI_API_KEY)
    - gemini/gemini-1.5-pro       → Google (uses GEMINI_API_KEY)
    - groq/llama-3.1-70b-versatile → Groq (uses GROQ_API_KEY)
    - openrouter/...              → OpenRouter (uses OPENROUTER_API_KEY)
    - ollama/deepseek-coder       → Local Ollama (no key needed)

    Returns the text content of the model's first response message.
    """
    api_keys: dict[str, str | None] = {
        "ANTHROPIC_API_KEY": settings.anthropic_api_key_value
        or getattr(settings, "ANTHROPIC_API_KEY", None)
        or os.environ.get("ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": getattr(settings, "OPENAI_API_KEY", None)
        or settings.openai_api_key_value
        or os.environ.get("OPENAI_API_KEY"),
        "GEMINI_API_KEY": getattr(settings, "GEMINI_API_KEY", None)
        or settings.gemini_api_key_value
        or os.environ.get("GEMINI_API_KEY"),
        "GROQ_API_KEY": getattr(settings, "GROQ_API_KEY", None) or os.environ.get("GROQ_API_KEY"),
        "OPENROUTER_API_KEY": getattr(settings, "OPENROUTER_API_KEY", None)
        or os.environ.get("OPENROUTER_API_KEY"),
    }

    # Set env vars so litellm picks them up
    for key, value in api_keys.items():
        if value:
            os.environ[key] = value

    model = getattr(settings, "LLM_MODEL", settings.anthropic_model)

    response = completion(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    content = response.choices[0].message.content
    if content is None:
        return ""
    return str(content)
