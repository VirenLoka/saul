"""Unit tests for config_loader."""

from __future__ import annotations

import textwrap

import pytest

from config_loader import ConfigError, load_config


def _write(tmp_path, text: str):
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_loads_bundled_config_defaults():
    """The repo's own config.yaml should load and expose Qwen as default."""
    cfg = load_config()
    assert cfg.model_selection.provider == "local"
    assert cfg.model_selection.local_model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.storage_paths.default_portfolio.endswith("sample_portfolio.csv")
    assert cfg.analysis.target_allocation["Equity"] == 60


def test_env_var_overrides_api_key(tmp_path, monkeypatch):
    cfg_path = _write(
        tmp_path,
        """
        model_selection:
          provider: openai
          local_model: "Qwen/Qwen2.5-7B-Instruct"
        local_inference_settings: {}
        api_credentials:
          openai:
            api_key: "from-file"
        storage_paths: {}
        """,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    cfg = load_config(cfg_path)
    assert cfg.api_credentials.openai_api_key == "from-env"


def test_invalid_provider_raises(tmp_path):
    cfg_path = _write(
        tmp_path,
        """
        model_selection:
          provider: not-a-provider
          local_model: "x"
        local_inference_settings: {}
        storage_paths: {}
        """,
    )
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_missing_section_raises(tmp_path):
    cfg_path = _write(tmp_path, "model_selection: {provider: local, local_model: x}\n")
    with pytest.raises(ConfigError):
        load_config(cfg_path)


def test_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("/nonexistent/path/config.yaml")
