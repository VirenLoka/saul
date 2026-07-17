"""Configuration loader for the MCP-Powered Financial Advisor AI Agent.

Parses ``config.yaml`` once at startup and exposes a typed, read-only view of
the settings. Source code depends on this module rather than reading the YAML
(or environment) directly, so configuration concerns stay in one place.

Secrets policy
--------------
API keys are never required to live in the YAML file. ``config.yaml`` is
untracked (see .gitignore) and ``config.example.yaml`` is the committed
placeholder-only template, so real credentials cannot be pushed. Environment
variables always take precedence over any value found in the file:
  - ``SPARKS_API_KEY``   — vLLM bearer token (matches serve.sh)
  - ``GROQ_API_KEY``     — remote Groq API
  - ``NEWSAPI_KEY``      — stock-news context provider
  - ``NEWSDATA_API_KEY`` — newsdata.io archive news (backtesting)
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
#: Every provider the app supports. ``vllm`` is the only self-hosting path
#: (and the way DeepSeek models are served); ``groq`` is the only remote API;
#: ``mock`` is the offline stub used by the tests.
SUPPORTED_PROVIDERS = ("vllm", "groq", "mock")

#: Providers backed by a real engine block under ``local_inference_settings``.
#: ``mock`` is excluded: it needs no connection settings.
LIVE_ENGINES = ("vllm", "groq")


@dataclass(frozen=True)
class ModelSelection:
    provider: str   # "vllm" | "groq" | "mock"


@dataclass(frozen=True)
class LocalInferenceSettings:
    """Resolved connection + sampling settings for the ACTIVE engine.

    ``engine`` records which backend these belong to (vllm/groq); ``model``,
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
    # Request-size / rate-limit guards (see config.yaml). 0 disables the first two.
    max_request_tokens: int = 0
    max_tool_result_chars: int = 0
    request_max_retries: int = 2
    # Optional full base-URL override (env SPARKS_BASE_URL). Useful in Docker
    # when the engine lives in another container and host:port from the file
    # isn't reachable, e.g. "http://172.23.0.4:8000". Wins over host/port.
    base_url_override: str = ""
    # Full base URL configured for a remote engine (groq), e.g.
    # "https://api.groq.com/openai/v1". Used when there is no host/port to dial.
    configured_base_url: str = ""

    @staticmethod
    def _strip_v1(url: str) -> str:
        url = url.rstrip("/")
        return url[: -len("/v1")] if url.endswith("/v1") else url

    @property
    def base_url(self) -> str:
        """OpenAI-compatible base URL of the engine (without /v1)."""
        # Precedence: explicit env override > configured remote URL > host:port.
        if self.base_url_override:
            return self._strip_v1(self.base_url_override)
        if self.configured_base_url:
            return self._strip_v1(self.configured_base_url)
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
class NewsApiSettings:
    """Settings for the stock-news context provider (NewsAPI)."""

    api_key: str
    base_url: str
    page_size: int
    language: str
    sort_by: str
    lookback_days: int
    use_live: bool


@dataclass(frozen=True)
class SearchSettings:
    """Settings for the autonomous web-search tool (SearXNG)."""

    base_url: str
    max_results: int
    language: str
    use_live: bool
    request_timeout: int


@dataclass(frozen=True)
class NewsDataSettings:
    """Settings for the newsdata.io archive news provider (backtesting)."""

    api_key: str
    language: str
    earliest_date: str      # hard floor, e.g. "2025-08-05" (no earlier archive reads)
    max_articles: int
    use_live: bool
    request_timeout: int


@dataclass(frozen=True)
class BacktestSettings:
    """Settings for the periodic-rebalancing backtest engine."""

    start_date: str
    end_date: str           # "" -> today
    rebalance: str          # weekly | monthly | quarterly
    initial_capital: float
    benchmark: str
    sectors: list
    risk_profile: str
    lookback_days: int
    news_lookback_days: int
    use_graph: bool
    use_web_search: bool
    results_dir: str


@dataclass(frozen=True)
class PortfolioGraphSettings:
    """Settings for the point-in-time user-portfolio graph builder."""

    portfolio: str
    start_date: str
    end_date: str           # "" -> start_date + window_days
    window_days: int
    lookback_days: int
    correlation_threshold: float
    min_validations: int
    allow_web_search: bool


@dataclass(frozen=True)
class ApiCredentials:
    #: Deprecated fallback for the `newsapi` section; env NEWSAPI_KEY wins.
    newsapi_key: str


@dataclass(frozen=True)
class StoragePaths:
    knowledge_root: str
    portfolios: str
    market_data: str
    vector_db: str
    graphs: str
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
    newsapi: NewsApiSettings
    search: SearchSettings
    newsdata: NewsDataSettings
    backtesting: BacktestSettings
    portfolio_graph: PortfolioGraphSettings
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
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"model_selection.provider '{provider}' is invalid; "
            f"expected one of {sorted(SUPPORTED_PROVIDERS)}."
        )
    model_selection = ModelSelection(provider=provider)

    # ---- local_inference_settings ------------------------------------------
    # Resolve the connection block for the active engine. 'mock' needs no live
    # engine, so it harmlessly borrows the vllm block's defaults.
    li = _get_section(data, "local_inference_settings")
    engine = provider if provider in LIVE_ENGINES else "vllm"
    block = li.get(engine, {}) or {}

    # (default_model, default_port, primary_env_key, default_base_url).
    # The remote engine carries a default HTTPS base_url; vLLM is dialled by
    # host:port instead, so its base_url default is empty.
    _ENGINE_DEFAULTS = {
        "vllm": ("Qwen/Qwen2.5-7B-Instruct", 8000, "SPARKS_API_KEY", ""),
        "groq": ("llama3-8b-8192", 443, "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    }
    default_model, default_port, env_key, default_base_url = _ENGINE_DEFAULTS[engine]
    # Env keys win over any file value (repo-wide secrets policy).
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
        base_url_override=os.environ.get("SPARKS_BASE_URL", "").strip(),
        configured_base_url=str(block.get("base_url", default_base_url)),
        max_request_tokens=int(li.get("max_request_tokens", 0)),
        max_tool_result_chars=int(li.get("max_tool_result_chars", 0)),
        request_max_retries=int(li.get("request_max_retries", 2)),
    )

    # ---- mcp ---------------------------------------------------------------
    mcp_section = _get_section(data, "mcp")
    md = mcp_section.get("market_data", {}) or {}
    mcp = McpSettings(
        host=str(mcp_section.get("host", "127.0.0.1")),
        port=int(mcp_section.get("port", 8001)),
        transport=str(mcp_section.get("transport", "sse")).lower(),
        # The CLI's MCP client connects here. Env SPARKS_MCP_URL wins, so a
        # different-container MCP server can be reached without editing the file
        # (e.g. SPARKS_MCP_URL=http://172.23.0.5:8001/sse). Note this is a full
        # client URL with scheme + /sse, distinct from vLLM's --tool-server arg.
        tool_server_url=os.environ.get("SPARKS_MCP_URL", "").strip()
        or str(mcp_section.get("tool_server_url", "http://127.0.0.1:8001/sse")),
        market_data=MarketDataSettings(
            default_exchange=str(md.get("default_exchange", "NS")).upper(),
            use_live=bool(md.get("use_live", True)),
            cache_ttl_seconds=int(md.get("cache_ttl_seconds", 60)),
        ),
    )

    # ---- api_credentials (env vars win over file) --------------------------
    ac = data.get("api_credentials", {}) or {}
    api_credentials = ApiCredentials(
        newsapi_key=os.environ.get("NEWSAPI_KEY", "") or str(ac.get("newsapi_key", "")),
    )

    # ---- newsapi (optional section; env key wins, then this section, then the
    # legacy api_credentials.newsapi_key fallback) ---------------------------
    na = data.get("newsapi", {}) or {}
    newsapi = NewsApiSettings(
        api_key=(
            os.environ.get("NEWSAPI_KEY", "")
            or str(na.get("api_key", ""))
            or api_credentials.newsapi_key
        ),
        base_url=str(na.get("base_url", "https://newsapi.org/v2/everything")),
        page_size=int(na.get("page_size", 8)),
        language=str(na.get("language", "en")),
        sort_by=str(na.get("sort_by", "publishedAt")),
        lookback_days=int(na.get("lookback_days", 7)),
        use_live=bool(na.get("use_live", True)),
    )

    # ---- search (optional section; SearXNG web-search tool) ----------------
    se = data.get("search", {}) or {}
    search = SearchSettings(
        base_url=str(se.get("base_url", "http://localhost:8080")),
        max_results=int(se.get("max_results", 6)),
        language=str(se.get("language", "en")),
        use_live=bool(se.get("use_live", True)),
        request_timeout=int(se.get("request_timeout", 15)),
    )

    # ---- newsdata (archive news for backtesting; env key wins) -------------
    nd = data.get("newsdata", {}) or {}
    newsdata = NewsDataSettings(
        api_key=os.environ.get("NEWSDATA_API_KEY", "") or str(nd.get("api_key", "")),
        language=str(nd.get("language", "en")),
        earliest_date=str(nd.get("earliest_date", "2025-08-05")),
        max_articles=int(nd.get("max_articles", 8)),
        use_live=bool(nd.get("use_live", True)),
        request_timeout=int(nd.get("request_timeout", 20)),
    )

    # ---- backtesting -------------------------------------------------------
    bt = data.get("backtesting", {}) or {}
    default_sectors = ["it", "banking", "energy", "auto", "pharma", "fmcg"]
    sectors_raw = bt.get("sectors") or default_sectors
    backtesting = BacktestSettings(
        start_date=str(bt.get("start_date", newsdata.earliest_date)),
        end_date=str(bt.get("end_date", "")),
        rebalance=str(bt.get("rebalance", "monthly")).lower(),
        initial_capital=float(bt.get("initial_capital", 1_000_000)),
        benchmark=str(bt.get("benchmark", "^NSEI")),
        sectors=[str(s).strip().lower() for s in sectors_raw if str(s).strip()],
        risk_profile=str(bt.get("risk_profile", "balanced")).lower(),
        lookback_days=int(bt.get("lookback_days", 126)),
        news_lookback_days=int(bt.get("news_lookback_days", 14)),
        use_graph=bool(bt.get("use_graph", True)),
        use_web_search=bool(bt.get("use_web_search", False)),
        results_dir=str(bt.get("results_dir", "backtesting/results")),
    )

    # ---- portfolio_graph (point-in-time user graph) ------------------------
    pg = data.get("portfolio_graph", {}) or {}
    portfolio_graph = PortfolioGraphSettings(
        portfolio=str(pg.get("portfolio", "knowledge/portfolios/sample_portfolio.csv")),
        start_date=str(pg.get("start_date", "2025-08-08")),
        end_date=str(pg.get("end_date", "")),
        window_days=int(pg.get("window_days", 30)),
        lookback_days=int(pg.get("lookback_days", 126)),
        correlation_threshold=float(pg.get("correlation_threshold", 0.4)),
        min_validations=int(pg.get("min_validations", 2)),
        allow_web_search=bool(pg.get("allow_web_search", False)),
    )

    # ---- storage_paths -----------------------------------------------------
    sp = _get_section(data, "storage_paths")
    market_data_path = str(sp.get("market_data", "knowledge/market_data"))
    storage_paths = StoragePaths(
        knowledge_root=str(sp.get("knowledge_root", "knowledge")),
        portfolios=str(sp.get("portfolios", "knowledge/portfolios")),
        market_data=market_data_path,
        vector_db=str(sp.get("vector_db", "knowledge/vector_db")),
        graphs=str(sp.get("graphs", f"{market_data_path}/graphs")),
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
        newsapi=newsapi,
        search=search,
        newsdata=newsdata,
        backtesting=backtesting,
        portfolio_graph=portfolio_graph,
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
    print("NewsAPI key set :", bool(cfg.newsapi.api_key))
    print("News live       :", cfg.newsapi.use_live)
    print("Portfolios     :", cfg.storage_paths.portfolios)
