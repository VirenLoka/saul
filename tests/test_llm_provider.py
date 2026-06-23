"""Unit tests for the LLM provider abstraction.

No real model forward pass occurs: these tests cover the factory, payload
shaping, and the deterministic MockStreamingProvider event stream.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace

import pytest
import yaml

from config_loader import load_config
from llm_provider import (
    LLMProviderError,
    MockStreamingProvider,
    OllamaProvider,
    StreamEvent,
    VLLMProvider,
    get_provider,
)


def _with_provider(cfg, provider):
    """Re-load the config with a different provider so the active engine block
    (host/port/model) is re-resolved, not just the provider name."""
    raw = dict(cfg.raw)
    raw["model_selection"] = {**raw.get("model_selection", {}), "provider": provider}
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as fh:
        yaml.safe_dump(raw, fh)
    try:
        return load_config(path)
    finally:
        os.unlink(path)


def test_factory_returns_vllm_by_default():
    cfg = load_config()
    provider = get_provider(cfg)
    assert isinstance(provider, VLLMProvider)
    assert provider.model == "Qwen/Qwen2.5-7B-Instruct"
    assert "vllm" in provider.describe()
    assert provider.supports_server_side_tools is True


def test_factory_returns_ollama_when_selected():
    cfg = _with_provider(load_config(), "ollama")
    provider = get_provider(cfg)
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen2.5:7b-instruct"
    assert "ollama" in provider.describe()
    # Ollama has no --tool-server, so the CLI runs it tool-free.
    assert provider.supports_server_side_tools is False


def test_factory_returns_mock_when_selected():
    cfg = _with_provider(load_config(), "mock")
    provider = get_provider(cfg)
    assert isinstance(provider, MockStreamingProvider)
    assert provider.supports_server_side_tools is True


def test_unknown_provider_raises():
    cfg = load_config()
    # Bypass loader validation to hit the factory's own guard.
    cfg = replace(cfg, model_selection=replace(cfg.model_selection, provider="bogus"))
    with pytest.raises(LLMProviderError):
        get_provider(cfg)


def test_mock_stream_emits_expected_event_sequence():
    provider = MockStreamingProvider()
    messages = [{"role": "user", "content": "What's the quote for Reliance?"}]
    events = list(provider.stream_chat(messages, tools=None))

    types = [e.type for e in events]
    assert types[0] == "reasoning"
    assert "tool_call" in types
    assert "content" in types
    assert types[-1] == "done"

    tool_calls = [e for e in events if e.type == "tool_call"]
    assert tool_calls[0].name == "get_indian_stock_quote"
    # No model was actually run.
    assert any("MOCK" in e.text for e in events if e.type == "reasoning")


def test_mock_routes_sector_questions_to_sector_tool():
    provider = MockStreamingProvider()
    messages = [{"role": "user", "content": "How is the IT sector doing?"}]
    tool_calls = [
        e for e in provider.stream_chat(messages, tools=None) if e.type == "tool_call"
    ]
    assert tool_calls[0].name == "get_indian_sector_performance"


def test_stream_event_defaults():
    ev = StreamEvent(type="content", text="hi")
    assert ev.name == "" and ev.arguments == "" and ev.finish_reason is None
