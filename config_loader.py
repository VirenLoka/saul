"""Configuration loader for the Financial Advisor AI Agent.

Parses ``config.yaml`` once at startup and exposes a typed, read-only view of
the settings. Source code should depend on this module rather than reading the
YAML (or environment) directly, so configuration concerns stay in one place.

Secrets policy
--------------
API keys are *never* required to live in the YAML file. Environment variables
always take precedence over any value found in the file:
  - ``SPARKS_API_KEY``   — vLLM bearer token (matches the serve.sh variable)
  - ``OPENAI_API_KEY``   — OpenAI fallback
  - ``ANTHROPIC_API_KEY``— Anthropic fallback
This keeps real secrets out of version control.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Default location of the config file, resolved relative to this module so the
# app works regardless of the current working directory.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or structurally invalid."""


# --------------------------------------------------------------------------- #
# Typed config sections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSelection:
    provider: str
    local_model: str
    openai_model: str
    anthropic_model: str


@dataclass(frozen=True)
class LocalInferenceSettings:
    host: str       # IP / hostname (no scheme, no port) — matches SPARKS_HOST
    port: int       # TCP port of the vLLM server       — matches SPARKS_PORT
    api_key: str    # Bearer token                       — from SPARKS_API_KEY env var
    temperature: float
    max_tokens: int
    request_timeout: int

    @property
    def base_url(self) -> str:
        """Full base URL of the vLLM OpenAI-compatible endpoint."""
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class ApiCredentials:
    openai_api_key: str
    openai_base_url: str
    anthropic_api_key: str


@dataclass(frozen=True)
class StoragePaths:
    knowledge_root: str
    portfolios: str
    market_data: str
    vector_db: str
    default_portfolio: str


@dataclass(frozen=True)
class AnalysisSettings:
    target_allocation: Dict[str, float]
    drift_tolerance_pct: float


@dataclass(frozen=True)
class AppConfig:
    """Top-level, immutable configuration object passed around the app."""

    model_selection: ModelSelection
    local_inference: LocalInferenceSettings
    api_credentials: ApiCredentials
    storage_paths: StoragePaths
    analysis: AnalysisSettings
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _require(section: Dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing required key '{key}' in '{where}' section.")
    return section[key]


def _get_section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"Config is missing the required '{name}' section.")
    return section


def load_config(path: Optional[os.PathLike | str] = None) -> AppConfig:
    """Load, validate, and return the application configuration.

    Parameters
    ----------
    path:
        Optional override of the config file location. Defaults to the
        ``config.yaml`` sitting next to this module.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise ConfigError("Top level of config.yaml must be a mapping.")

    # ---- model_selection ---------------------------------------------------
    ms = _get_section(data, "model_selection")
    provider = str(_require(ms, "provider", "model_selection")).lower()
    valid_providers = {"local", "openai", "anthropic", "mock"}
    if provider not in valid_providers:
        raise ConfigError(
            f"model_selection.provider '{provider}' is invalid; "
            f"expected one of {sorted(valid_providers)}."
        )
    model_selection = ModelSelection(
        provider=provider,
        local_model=str(_require(ms, "local_model", "model_selection")),
        openai_model=str(ms.get("openai_model", "gpt-4o-mini")),
        anthropic_model=str(ms.get("anthropic_model", "claude-opus-4-8")),
    )

    # ---- local_inference_settings ------------------------------------------
    li = _get_section(data, "local_inference_settings")
    local_inference = LocalInferenceSettings(
        host=str(li.get("host", "127.0.0.1")),
        port=int(li.get("port", 8000)),
        # SPARKS_API_KEY env var takes precedence over the file value.
        api_key=os.environ.get("SPARKS_API_KEY", "")
        or str(li.get("api_key", "")),
        temperature=float(li.get("temperature", 0.2)),
        max_tokens=int(li.get("max_tokens", 1024)),
        request_timeout=int(li.get("request_timeout", 120)),
    )

    # ---- api_credentials (env vars win over file) --------------------------
    ac = data.get("api_credentials", {}) or {}
    openai_block = ac.get("openai", {}) or {}
    anthropic_block = ac.get("anthropic", {}) or {}
    api_credentials = ApiCredentials(
        openai_api_key=os.environ.get("OPENAI_API_KEY", "")
        or str(openai_block.get("api_key", "")),
        openai_base_url=os.environ.get("OPENAI_BASE_URL", "")
        or str(openai_block.get("base_url", "")),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "")
        or str(anthropic_block.get("api_key", "")),
    )

    # ---- storage_paths -----------------------------------------------------
    sp = _get_section(data, "storage_paths")
    storage_paths = StoragePaths(
        knowledge_root=str(sp.get("knowledge_root", "knowledge")),
        portfolios=str(sp.get("portfolios", "knowledge/portfolios")),
        market_data=str(sp.get("market_data", "knowledge/market_data")),
        vector_db=str(sp.get("vector_db", "knowledge/vector_db")),
        default_portfolio=str(
            sp.get("default_portfolio", "knowledge/portfolios/sample_portfolio.csv")
        ),
    )

    # ---- analysis ----------------------------------------------------------
    an = data.get("analysis", {}) or {}
    target = an.get("target_allocation", {}) or {}
    analysis = AnalysisSettings(
        target_allocation={str(k): float(v) for k, v in target.items()},
        drift_tolerance_pct=float(an.get("drift_tolerance_pct", 10)),
    )

    return AppConfig(
        model_selection=model_selection,
        local_inference=local_inference,
        api_credentials=api_credentials,
        storage_paths=storage_paths,
        analysis=analysis,
        raw=data,
    )


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    cfg = load_config()
    print("Provider    :", cfg.model_selection.provider)
    print("Local model :", cfg.model_selection.local_model)
    print("vLLM URL    :", cfg.local_inference.base_url)
    print("API key set :", bool(cfg.local_inference.api_key))
    print("Portfolios  :", cfg.storage_paths.portfolios)
