"""Configuration loader for the MCP-Powered Financial Advisor AI Agent.

Parses ``config.yaml`` once at startup and exposes a typed, read-only view of
the settings. Source code depends on this module rather than reading the YAML
(or environment) directly, so configuration concerns stay in one place.

Secrets policy
--------------
API keys are never required to live in the YAML file. Environment variables
always take precedence over any value found in the file:
  - ``SPARKS_API_KEY``  — vLLM bearer token (matches serve.sh)
  - ``OPENAI_API_KEY``  — optional external OpenAI-compatible fallback
  - ``OPENAI_BASE_URL`` — optional external base URL
  - ``NEWSAPI_KEY``     — future market_data ingestion
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
    provider: str   # "vllm" | "ollama" | "mock"


@dataclass(frozen=True)
class LocalInferenceSettings:
    """Resolved connection + sampling settings for the ACTIVE engine.

    ``engine`` records which backend these belong to (vllm/ollama); ``model``,
    ``host``, ``port``, and ``api_key`` come from that engine's config block.
    """

    engine: str
    model: str
    host: str
    port: int
    api_key: str
    temperature: float
    max_tokens: int
    request_timeout: int
    stream: bool

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL of the engine (without /v1)."""
        return f"http://{self.host}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        """Base URL including the /v1 suffix expected by the OpenAI SDK."""
        return f"{self.base_url}/v1"


@dataclass(frozen=True)
class MarketDataSettings:
    default_exchange: str   # "NS" | "BO"
    use_live: bool
    cache_ttl_seconds: int


@dataclass(frozen=True)
class McpSettings:
    host: str
    port: int
    transport: str          # "sse" | "streamable-http"
    tool_server_url: str
    market_data: MarketDataSettings


@dataclass(frozen=True)
class ApiCredentials:
    newsapi_key: str
    external_openai_api_key: str
    external_openai_base_url: str


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
    mcp: McpSettings
    api_credentials: ApiCredentials
    storage_paths: StoragePaths
    analysis: AnalysisSettings
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _get_section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ConfigError(f"Config is missing the required '{name}' section.")
    return section


def _require(section: Dict[str, Any], key: str, where: str) -> Any:
    if key not in section:
        raise ConfigError(f"Missing required key '{key}' in '{where}' section.")
    return section[key]


def load_config(
    path: Optional[os.PathLike | str] = None,
    *,
    provider_override: Optional[str] = None,
) -> AppConfig:
    """Load, validate, and return the application configuration.

    ``provider_override`` (e.g. from a ``--provider`` CLI flag) replaces
    ``model_selection.provider`` before the active engine block is resolved, so
    the connection settings track the chosen engine.
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
    provider = str(provider_override or _require(ms, "provider", "model_selection")).lower()
    valid_providers = {"vllm", "ollama", "mock"}
    if provider not in valid_providers:
        raise ConfigError(
            f"model_selection.provider '{provider}' is invalid; "
            f"expected one of {sorted(valid_providers)}."
        )
    model_selection = ModelSelection(provider=provider)

    # ---- local_inference_settings ------------------------------------------
    # Resolve the connection block for the active engine. 'mock' needs no live
    # engine, so it harmlessly borrows the vllm block's defaults.
    li = _get_section(data, "local_inference_settings")
    engine = provider if provider in {"vllm", "ollama"} else "vllm"
    block = li.get(engine, {}) or {}

    _ENGINE_DEFAULTS = {
        "vllm": ("Qwen/Qwen2.5-7B-Instruct", 8000, "SPARKS_API_KEY"),
        "ollama": ("qwen2.5:7b-instruct", 11434, "OLLAMA_API_KEY"),
    }
    default_model, default_port, env_key = _ENGINE_DEFAULTS[engine]
    api_key = os.environ.get(env_key, "") or str(block.get("api_key", ""))

    local_inference = LocalInferenceSettings(
        engine=engine,
        model=str(block.get("model", default_model)),
        host=str(block.get("host", "127.0.0.1")),
        port=int(block.get("port", default_port)),
        api_key=api_key,
        temperature=float(li.get("temperature", 0.2)),
        max_tokens=int(li.get("max_tokens", 1024)),
        request_timeout=int(li.get("request_timeout", 180)),
        stream=bool(li.get("stream", True)),
    )

    # ---- mcp ---------------------------------------------------------------
    mcp_section = _get_section(data, "mcp")
    md = mcp_section.get("market_data", {}) or {}
    mcp = McpSettings(
        host=str(mcp_section.get("host", "127.0.0.1")),
        port=int(mcp_section.get("port", 8001)),
        transport=str(mcp_section.get("transport", "sse")).lower(),
        tool_server_url=str(
            mcp_section.get("tool_server_url", "http://127.0.0.1:8001/sse")
        ),
        market_data=MarketDataSettings(
            default_exchange=str(md.get("default_exchange", "NS")).upper(),
            use_live=bool(md.get("use_live", True)),
            cache_ttl_seconds=int(md.get("cache_ttl_seconds", 60)),
        ),
    )

    # ---- api_credentials (env vars win over file) --------------------------
    ac = data.get("api_credentials", {}) or {}
    ext = ac.get("external_openai", {}) or {}
    api_credentials = ApiCredentials(
        newsapi_key=os.environ.get("NEWSAPI_KEY", "") or str(ac.get("newsapi_key", "")),
        external_openai_api_key=os.environ.get("OPENAI_API_KEY", "")
        or str(ext.get("api_key", "")),
        external_openai_base_url=os.environ.get("OPENAI_BASE_URL", "")
        or str(ext.get("base_url", "")),
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
        mcp=mcp,
        api_credentials=api_credentials,
        storage_paths=storage_paths,
        analysis=analysis,
        raw=data,
    )


if __name__ == "__main__":  # pragma: no cover - manual sanity check
    cfg = load_config()
    print("Provider       :", cfg.model_selection.provider)
    print("Engine         :", cfg.local_inference.engine)
    print("Model          :", cfg.local_inference.model)
    print("Engine URL     :", cfg.local_inference.openai_base_url)
    print("API key set    :", bool(cfg.local_inference.api_key))
    print("MCP tool server:", cfg.mcp.tool_server_url)
    print("Market live     :", cfg.mcp.market_data.use_live)
    print("Portfolios     :", cfg.storage_paths.portfolios)
