"""Unit tests for config_loader (MCP / vLLM schema)."""

from __future__ import annotations

import textwrap

import pytest

from config_loader import ConfigError, load_config


def _write(tmp_path, text: str):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_loads_bundled_config_defaults():
    """The repo's own config.yaml should default to vLLM + Qwen."""
    cfg = load_config()
    assert cfg.model_selection.provider == "vllm"
    assert cfg.local_inference.engine == "vllm"
    assert cfg.local_inference.model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.local_inference.openai_base_url == "http://127.0.0.1:8000/v1"
    assert cfg.mcp.tool_server_url.endswith("/sse")
    assert cfg.mcp.market_data.default_exchange == "NS"
    assert cfg.storage_paths.default_portfolio.endswith("sample_portfolio.csv")
    assert cfg.analysis.target_allocation["Equity"] == 60
    # newsapi section is parsed with its documented defaults.
    assert cfg.newsapi.base_url == "https://newsapi.org/v2/everything"
    assert cfg.newsapi.page_size == 8
    assert cfg.newsapi.use_live is True


def test_newsapi_section_parsed(tmp_path):
    cfg_path = _write(
        tmp_path,
        """
        model_selection: {provider: vllm}
        local_inference_settings: {}
        mcp: {market_data: {}}
        storage_paths: {}
        newsapi:
          api_key: "from-file"
          page_size: 3
          sort_by: relevancy
          use_live: false
        """,
    )
    cfg = load_config(cfg_path)
    assert cfg.newsapi.api_key == "from-file"
    assert cfg.newsapi.page_size == 3
    assert cfg.newsapi.sort_by == "relevancy"
    assert cfg.newsapi.use_live is False


def test_newsapi_env_key_wins_over_file(tmp_path, monkeypatch):
    cfg_path = _write(
        tmp_path,
        """
        model_selection: {provider: vllm}
        local_inference_settings: {}
        mcp: {market_data: {}}
        storage_paths: {}
        newsapi: {api_key: "from-file"}
        """,
    )
    monkeypatch.setenv("NEWSAPI_KEY", "from-env")
    cfg = load_config(cfg_path)
    assert cfg.newsapi.api_key == "from-env"


def test_newsapi_falls_back_to_legacy_credentials(tmp_path, monkeypatch):
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)
    cfg_path = _write(
        tmp_path,
        """
        model_selection: {provider: vllm}
        local_inference_settings: {}
        mcp: {market_data: {}}
        storage_paths: {}
        api_credentials: {newsapi_key: "legacy-key"}
        """,
    )
    cfg = load_config(cfg_path)
    assert cfg.newsapi.api_key == "legacy-key"


def test_ollama_provider_selects_ollama_block(tmp_path):
    """Switching provider to ollama resolves the ollama connection block."""
    cfg_path = _write(
        tmp_path,
        """
        model_selection: {provider: ollama}
        local_inference_settings:
          vllm: {model: "Qwen/Qwen2.5-7B-Instruct", port: 8000}
          ollama: {model: "qwen2.5:7b-instruct", port: 11434}
        mcp: {market_data: {}}
        storage_paths: {}
        """,
    )
    cfg = load_config(cfg_path)
    assert cfg.local_inference.engine == "ollama"
    assert cfg.local_inference.model == "qwen2.5:7b-instruct"
    assert cfg.local_inference.openai_base_url == "http://127.0.0.1:11434/v1"


def test_env_var_overrides_vllm_api_key(tmp_path, monkeypatch):
    cfg_path = _write(
        tmp_path,
        """
        model_selection: {provider: vllm}
        local_inference_settings:
          vllm: {api_key: "from-file"}
        mcp: {market_data: {}}
        storage_paths: {}
        """,
    )
    monkeypatch.setenv("SPARKS_API_KEY", "from-env")
    cfg = load_config(cfg_path)
    assert cfg.local_inference.api_key == "from-env"


def test_env_var_overrides_ollama_api_key(tmp_path, monkeypatch):
    cfg_path = _write(
        tmp_path,
        """
        model_selection: {provider: ollama}
        local_inference_settings:
          ollama: {api_key: "from-file"}
        mcp: {market_data: {}}
        storage_paths: {}
        """,
    )
    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-env")
    cfg = load_config(cfg_path)
    assert cfg.local_inference.api_key == "ollama-env"


def test_invalid_provider_raises(tmp_path):
    cfg_path = _write(
        tmp_path,
        """
        model_selection: {provider: not-a-provider}
        local_inference_settings: {}
        mcp: {market_data: {}}
        storage_paths: {}
        """,
    )
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_missing_mcp_section_raises(tmp_path):
    cfg_path = _write(
        tmp_path,
        """
        model_selection: {provider: vllm, model: x}
        local_inference_settings: {}
        storage_paths: {}
        """,
    )
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("/nonexistent/path/config.yaml")
