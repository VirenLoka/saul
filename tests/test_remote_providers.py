"""Tests for the Groq remote provider + search config (no network).

Provider construction stores config only; no forward pass or HTTP call occurs.
"""

from __future__ import annotations

from config_loader import load_config
from llm_provider import GroqProvider, get_provider


class TestRemoteProviderConfig:
    def test_groq_block_resolves(self, monkeypatch):
        # Model is a user-editable config value, so assert the stable invariants
        # (engine, endpoint, env-key precedence) rather than a specific model.
        monkeypatch.setenv("GROQ_API_KEY", "gsk-groq")
        cfg = load_config(provider_override="groq")
        assert cfg.local_inference.engine == "groq"
        assert cfg.local_inference.model  # some model configured
        assert cfg.local_inference.openai_base_url == "https://api.groq.com/openai/v1"
        assert cfg.local_inference.api_key == "gsk-groq"  # env wins over file

    def test_configured_base_url_strips_trailing_v1(self, monkeypatch):
        # openai_base_url re-appends /v1; base_url must not double it.
        monkeypatch.setenv("GROQ_API_KEY", "gsk-groq")
        cfg = load_config(provider_override="groq")
        assert cfg.local_inference.base_url == "https://api.groq.com/openai"
        assert cfg.local_inference.openai_base_url.endswith("/v1")
        assert not cfg.local_inference.openai_base_url.endswith("/v1/v1")


class TestRemovedProviders:
    """The remote DeepSeek/Grok APIs and Ollama self-hosting are gone.

    DeepSeek is served through self-hosted vLLM instead; Groq is the only
    remote API. Selecting a removed provider must fail loudly rather than
    silently falling back to another engine.
    """

    @staticmethod
    def _rejects(name: str) -> bool:
        from config_loader import ConfigError

        try:
            load_config(provider_override=name)
        except ConfigError:
            return True
        return False

    def test_removed_providers_are_rejected(self):
        assert self._rejects("deepseek")
        assert self._rejects("grok")
        assert self._rejects("ollama")

    def test_supported_providers_are_exactly_vllm_groq_mock(self):
        from config_loader import SUPPORTED_PROVIDERS

        assert set(SUPPORTED_PROVIDERS) == {"vllm", "groq", "mock"}


class TestSearchConfig:
    def test_search_defaults(self):
        cfg = load_config()
        assert cfg.search.base_url == "http://localhost:8080"
        assert cfg.search.max_results == 6
        assert cfg.search.use_live is True

    def test_graphs_path_present(self):
        cfg = load_config()
        assert cfg.storage_paths.graphs.endswith("graphs")


class TestRemoteProviderFactory:
    def test_factory_groq(self):
        cfg = load_config(provider_override="groq")
        p = get_provider(cfg)
        assert isinstance(p, GroqProvider)
        assert p.supports_tools is True
        # Remote API: tools are executed client-side, not on a --tool-server.
        assert p.supports_server_side_tools is False
        assert "groq" in p.describe()
