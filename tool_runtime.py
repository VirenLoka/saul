"""Tool execution runtime for the agentic loop.

When the model emits a tool call, the CLI must actually run the tool and feed
the result back so the model can continue. Two executors share one interface
``__call__(name: str, arguments: str | dict) -> str`` (returns a JSON string):

  * :class:`MCPToolExecutor`     — invokes tools on the running FastMCP server
    over the MCP protocol (the production path for vLLM). ``fastmcp`` is
    imported lazily so this module imports fine without it.
  * :class:`InProcessToolExecutor` — calls the ``market_data`` functions
    directly (offline; used by the mock provider and the test suite).

Both never raise to the caller: failures are returned as a JSON ``{"error": …}``
string so the model can read and react to them.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from config_loader import (
        MarketDataSettings,
        NewsApiSettings,
        NewsDataSettings,
        SearchSettings,
    )

logger = logging.getLogger("saul.tools")

ArgsType = Union[str, dict, None]


def parse_args(arguments: ArgsType) -> dict:
    """Coerce a tool-call argument payload (JSON string or dict) to a dict."""
    if isinstance(arguments, dict):
        return arguments
    s = (arguments or "").strip()
    if not s:
        return {}
    return json.loads(s)


def _err(msg: str) -> str:
    return json.dumps({"error": msg})


# --------------------------------------------------------------------------- #
# Production: execute against the running FastMCP server over MCP
# --------------------------------------------------------------------------- #
class MCPToolExecutor:
    """Calls tools on the FastMCP server at ``url`` via an MCP client.

    Uses ``asyncio.run`` per call (the CLI is synchronous and not inside an
    event loop), opening a short-lived client connection each time. Robust and
    simple for an interactive CLI where tool calls are infrequent.
    """

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.url = url
        self.timeout = timeout

    def __call__(self, name: str, arguments: ArgsType) -> str:
        try:
            args = parse_args(arguments)
        except Exception as exc:  # noqa: BLE001
            return _err(f"could not parse arguments for {name}: {exc}")
        logger.info("MCP tool call -> %s %s @ %s", name, args, self.url)
        try:
            result = asyncio.run(self._call(name, args))
            logger.debug("MCP tool result [%s]: %s", name, result[:300])
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP tool '%s' failed: %s", name, exc)
            logger.debug("MCP tool traceback", exc_info=True)
            return _err(
                f"tool '{name}' failed against MCP server {self.url}: {exc}"
            )

    async def _call(self, name: str, args: dict) -> str:
        try:
            from fastmcp import Client  # lazy import
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "fastmcp not installed. Install with: pip install fastmcp"
            ) from exc

        # Client infers the transport from the URL (e.g. an /sse suffix).
        async with Client(self.url) as client:
            res = await client.call_tool(name, args)
            return _extract_result_text(res)


def _extract_result_text(res: object) -> str:
    """Normalize a FastMCP CallToolResult into a JSON/text string.

    Handles the common shapes across fastmcp versions: ``.data`` (deserialized
    structured output), ``.structured_content``, or a list of content blocks.
    """
    data = getattr(res, "data", None)
    if data is not None:
        try:
            return json.dumps(data)
        except TypeError:
            return str(data)

    structured = getattr(res, "structured_content", None)
    if structured:
        try:
            return json.dumps(structured)
        except TypeError:
            return str(structured)

    content = getattr(res, "content", None)
    if content:
        texts = [getattr(block, "text", "") for block in content]
        joined = "\n".join(t for t in texts if t)
        if joined:
            return joined

    return str(res)


# --------------------------------------------------------------------------- #
# Offline: execute the market_data functions directly (mock / tests)
# --------------------------------------------------------------------------- #
class InProcessToolExecutor:
    """Runs the market-data tools in-process, with no network or MCP server.

    Used by the mock provider and tests. ``use_live`` defaults to False so the
    deterministic mock data layer is used.
    """

    def __init__(
        self,
        default_exchange: str = "NS",
        use_live: bool = False,
        cache_ttl_seconds: int = 60,
        newsapi: "NewsApiSettings | None" = None,
        search: "SearchSettings | None" = None,
        graphs_dir: str | None = None,
        portfolios_dir: str | None = None,
        newsdata: "NewsDataSettings | None" = None,
    ) -> None:
        self.default_exchange = default_exchange
        self.use_live = use_live
        self.cache_ttl_seconds = cache_ttl_seconds
        self.newsapi = newsapi
        self.search = search
        self.newsdata = newsdata
        self.default_portfolio: str | None = None
        self.graphs_dir = graphs_dir
        self.portfolios_dir = portfolios_dir
        if graphs_dir:
            from sector_graph import set_graphs_dir

            set_graphs_dir(graphs_dir)

    @classmethod
    def from_settings(
        cls,
        md: "MarketDataSettings",
        *,
        use_live: bool | None = None,
        newsapi: "NewsApiSettings | None" = None,
        search: "SearchSettings | None" = None,
        graphs_dir: str | None = None,
        portfolios_dir: str | None = None,
        default_portfolio: str | None = None,
        newsdata: "NewsDataSettings | None" = None,
    ) -> "InProcessToolExecutor":
        ex = cls(
            default_exchange=md.default_exchange,
            use_live=md.use_live if use_live is None else use_live,
            cache_ttl_seconds=md.cache_ttl_seconds,
            newsapi=newsapi,
            search=search,
            graphs_dir=graphs_dir,
            portfolios_dir=portfolios_dir,
            newsdata=newsdata,
        )
        ex.default_portfolio = default_portfolio
        return ex

    def __call__(self, name: str, arguments: ArgsType) -> str:
        from market_data import (  # local import keeps module deps light
            MarketDataError,
            get_sector_performance,
            get_stock_quote,
        )
        from backtesting.news_archive import NewsArchiveError
        from graph_viz import GraphVizError
        from news_data import NewsDataError, get_stock_news
        from portfolio_builder import PortfolioBuildError
        from portfolio_parser import PortfolioParseError
        from sector_graph import GraphError
        from stock_stats import StatsError
        from web_search import WebSearchError

        try:
            args = parse_args(arguments)
        except Exception as exc:  # noqa: BLE001
            return _err(f"could not parse arguments for {name}: {exc}")

        logger.info("In-process tool call -> %s %s", name, args)
        try:
            if name == "get_indian_stock_quote":
                result = get_stock_quote(
                    args.get("query", ""),
                    args.get("exchange", self.default_exchange),
                    use_live=self.use_live,
                    cache_ttl_seconds=self.cache_ttl_seconds,
                )
            elif name == "get_indian_sector_performance":
                result = get_sector_performance(
                    args.get("sector", ""),
                    args.get("exchange", self.default_exchange),
                    use_live=self.use_live,
                    cache_ttl_seconds=self.cache_ttl_seconds,
                )
            elif name == "get_stock_news":
                ns = self.newsapi
                result = get_stock_news(
                    args.get("query", ""),
                    api_key=ns.api_key if ns else "",
                    base_url=ns.base_url if ns else "https://newsapi.org/v2/everything",
                    page_size=int(args.get("max_articles") or (ns.page_size if ns else 8)),
                    language=ns.language if ns else "en",
                    sort_by=ns.sort_by if ns else "publishedAt",
                    lookback_days=ns.lookback_days if ns else 7,
                    # honour both this executor's offline flag and the news config:
                    # in-process mode is used by the offline mock, so default to mock
                    # unless live is explicitly enabled on both.
                    use_live=self.use_live and (ns.use_live if ns else True),
                )
            elif name in _STATS_TOOLS:
                result = self._call_stats(name, args)
            elif name in _GRAPH_TOOLS:
                result = self._call_graph(name, args)
            elif name == "web_search":
                result = self._call_search(args)
            elif name == "fetch_news_archive":
                result = self._call_news_archive(args)
            elif name in _PORTFOLIO_TOOLS:
                result = self._call_portfolio(name, args)
            else:
                return _err(f"unknown tool '{name}'")
            return json.dumps(result)
        except (
            MarketDataError, NewsDataError, StatsError, GraphError,
            GraphVizError, WebSearchError, PortfolioBuildError,
            PortfolioParseError, FileNotFoundError, NewsArchiveError,
        ) as exc:
            return _err(str(exc))

    def _call_news_archive(self, args: dict) -> dict:
        from backtesting.news_archive import fetch_news_archive

        nd = self.newsdata
        return fetch_news_archive(
            args.get("query", ""),
            args.get("from_date", ""),
            args.get("to_date", ""),
            api_key=nd.api_key if nd else "",
            language=nd.language if nd else "en",
            earliest_date=nd.earliest_date if nd else "2025-08-05",
            max_articles=int(args.get("max_articles") or (nd.max_articles if nd else 8)),
            use_live=self.use_live and (nd.use_live if nd else True),
        )

    def _call_portfolio(self, name: str, args: dict) -> dict:
        exchange = args.get("exchange") or self.default_exchange
        if name == "fetch_sector_analytics":
            from portfolio_builder import fetch_sector_analytics

            return fetch_sector_analytics(
                args.get("sectors") or [], exchange, use_live=self.use_live
            )
        # generate_final_portfolio
        from portfolio_builder import generate_final_portfolio

        return generate_final_portfolio(
            args.get("ticker_weights") or {},
            float(args.get("total_amount") or 0.0),
            exchange=exchange,
            reasoning=args.get("reasoning") or "",
            use_live=self.use_live,
            output_dir=self.portfolios_dir,
        )

    def _call_search(self, args: dict) -> dict:
        from web_search import web_search

        se = self.search
        return web_search(
            args.get("query", ""),
            base_url=se.base_url if se else "http://localhost:8080",
            max_results=int(args.get("max_results") or (se.max_results if se else 6)),
            language=se.language if se else "en",
            # in-process executor backs the offline mock provider by default;
            # honour the search config's live flag when present.
            use_live=self.use_live and (se.use_live if se else True),
            request_timeout=se.request_timeout if se else 15.0,
        )

    def _call_stats(self, name: str, args: dict) -> dict:
        from stock_stats import (
            get_correlation_matrix,
            get_fundamentals,
            get_return_statistics,
            get_risk_metrics,
            get_technical_indicators,
        )

        exchange = args.get("exchange") or self.default_exchange
        period = int(args.get("period_days") or 252)
        if name == "get_return_statistics":
            return get_return_statistics(
                args.get("query", ""), exchange,
                period_days=period, use_live=self.use_live,
            )
        if name == "get_technical_indicators":
            return get_technical_indicators(
                args.get("query", ""), exchange,
                period_days=period, use_live=self.use_live,
            )
        if name == "get_risk_metrics":
            return get_risk_metrics(
                args.get("query", ""), exchange,
                benchmark=args.get("benchmark") or "^NSEI",
                period_days=period, use_live=self.use_live,
            )
        if name == "get_correlation_matrix":
            return get_correlation_matrix(
                args.get("queries") or [], exchange,
                period_days=period, use_live=self.use_live,
            )
        # get_stock_fundamentals
        return get_fundamentals(args.get("query", ""), exchange, use_live=self.use_live)

    def _call_graph(self, name: str, args: dict) -> dict:
        from sector_graph import (
            NewsSentimentProvider,
            build_sector_graph,
            get_all_graphs,
            get_sector_graph,
            list_graphs,
            propose_graph_edge,
            validate_graph_edge,
        )

        if name == "build_sector_graph":
            return build_sector_graph(
                args.get("sectors") or [],
                args.get("exchange") or self.default_exchange,
                period_days=int(args.get("period_days") or 252),
                correlation_threshold=float(args.get("correlation_threshold") or 0.4),
                use_live=self.use_live,
                sentiment_provider=NewsSentimentProvider(self.newsapi),
            )
        if name == "propose_graph_edge":
            return propose_graph_edge(
                args.get("graph_id", ""), args.get("source", ""),
                args.get("target", ""), args.get("relation", ""),
                args.get("rationale", ""), float(args.get("weight") or 0.0),
            )
        if name == "validate_graph_edge":
            return validate_graph_edge(
                args.get("graph_id", ""), args.get("source", ""),
                args.get("target", ""), args.get("relation", ""),
                args.get("verdict", ""), args.get("reasoning", ""),
            )
        if name == "list_saved_graphs":
            return {"graphs": list_graphs()}
        if name == "get_all_graphs":
            return get_all_graphs(
                include_features=bool(args.get("include_features", False)),
                status=args.get("status", ""),
                ticker=args.get("ticker", ""),
                sector=args.get("sector", ""),
            )
        if name == "build_portfolio_graph":
            from graph_agent import build_portfolio_graph

            # In-process tool exec uses the deterministic heuristic (no nested
            # LLM); the model can still refine edges afterwards.
            return build_portfolio_graph(
                args.get("portfolio_path") or self.default_portfolio
                or "knowledge/portfolios/sample_portfolio.csv",
                args.get("exchange") or self.default_exchange,
                min_validations=int(args.get("min_validations") or 2),
                use_live=self.use_live,
                sentiment_provider=NewsSentimentProvider(self.newsapi),
            )
        if name == "visualize_sector_graph":
            from graph_viz import visualize_sector_graph

            kwargs = {"image_format": args.get("image_format") or "svg"}
            if self.graphs_dir:
                kwargs["out_dir"] = self.graphs_dir
            return visualize_sector_graph(args.get("graph_id", ""), **kwargs)
        # get_sector_graph
        return get_sector_graph(
            args.get("graph_id", ""),
            include_features=bool(args.get("include_features", False)),
            status=args.get("status", ""),
        )


_STATS_TOOLS = {
    "get_return_statistics",
    "get_technical_indicators",
    "get_risk_metrics",
    "get_correlation_matrix",
    "get_stock_fundamentals",
}

_GRAPH_TOOLS = {
    "build_sector_graph",
    "propose_graph_edge",
    "validate_graph_edge",
    "get_sector_graph",
    "list_saved_graphs",
    "get_all_graphs",
    "build_portfolio_graph",
    "visualize_sector_graph",
}

_PORTFOLIO_TOOLS = {
    "fetch_sector_analytics",
    "generate_final_portfolio",
}
