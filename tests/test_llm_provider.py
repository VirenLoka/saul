"""Unit tests for the LLM provider abstraction.

These tests never perform a real model forward pass: they verify the factory
toggling logic and the offline MockProvider only.
"""

from __future__ import annotations

import pytest

from config_loader import load_config
from llm_provider import (
    AnthropicProvider,
    LLMProviderError,
    LocalQwenProvider,
    MockProvider,
    OpenAIProvider,
    get_provider,
)


def test_mock_provider_runs_no_model():
    out = MockProvider().generate("sys", "user")
    assert "MOCK LLM OUTPUT" in out
    assert "no model was run" in out


def test_factory_returns_mock_when_selected():
    cfg = load_config()
    cfg = _with_provider(cfg, "mock")
    provider = get_provider(cfg)
    assert isinstance(provider, MockProvider)


def test_factory_returns_local_qwen_by_default():
    cfg = load_config()  # bundled config defaults to local Qwen via vLLM
    provider = get_provider(cfg)
    assert isinstance(provider, LocalQwenProvider)
    assert provider.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert "Qwen2.5-7B-Instruct" in provider.describe()
    assert "vllm" in provider.describe()


def test_local_provider_builds_correct_url():
    cfg = load_config()
    provider = LocalQwenProvider(cfg)
    assert provider.settings.base_url == (
        f"http://{cfg.local_inference.host}:{cfg.local_inference.port}"
    )


def test_api_providers_require_keys():
    cfg = load_config()
    with pytest.raises(LLMProviderError):
        OpenAIProvider(_with_provider(cfg, "openai"))
    with pytest.raises(LLMProviderError):
        AnthropicProvider(_with_provider(cfg, "anthropic"))


# --- small helpers ----------------------------------------------------------
from dataclasses import replace as _replace  # noqa: E402


def _with_provider(cfg, provider):
    return _replace(cfg, model_selection=_replace(cfg.model_selection,
                                                   provider=provider))
