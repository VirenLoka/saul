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
    """The repo's own config.yaml should load and default to vLLM + Qwen."""
    cfg = load_config()
    assert cfg.model_selection.provider == "vllm"
    assert cfg.model_selection.model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.local_inference.openai_base_url == "http://127.0.0.1:8000/v1"
    assert cfg.mcp.tool_server_url.endswith("/sse")
    assert cfg.mcp.market_data.default_exchange == "NS"
    assert cfg.storage_paths.default_portfolio.endswith("sample_portfolio.csv")
    assert cfg.analysis.target_allocation["Equity"] == 60


def test_env_var_overrides_vllm_api_key(tmp_path, monkeypatch):
    cfg_path = _write(
        tmp_path,
        """
        model_selection: {provider: vllm, model: "Qwen/Qwen2.5-7B-Instruct"}
        local_inference_settings: {api_key: "from-file"}
        mcp: {market_data: {}}
        storage_paths: {}
        """,
    )
    monkeypatch.setenv("SPARKS_API_KEY", "from-env")
    cfg = load_config(cfg_path)
    assert cfg.local_inference.api_key == "from-env"


def test_invalid_provider_raises(tmp_path):
    cfg_path = _write(
        tmp_path,
        """
        model_selection: {provider: not-a-provider, model: "x"}
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
