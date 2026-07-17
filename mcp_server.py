"""FastMCP tool server — real-time Indian market data.

A standalone MCP server exposing read-only Indian market data tools. It is
launched separately from the model and ATTACHED to the vLLM engine via
``--tool-server`` (see serve.sh / README), so vLLM can invoke these tools
server-side during generation.

Run directly:
    python mcp_server.py
    # honours mcp.host / mcp.port / mcp.transport from config.yaml

The actual data logic lives in ``market_data.py`` (dependency-free and
unit-tested); this file is only the thin MCP wiring. ``fastmcp`` is imported
lazily inside ``build_server`` so the module can be imported (and the tools'
logic tested) without fastmcp installed.

Scope guardrail: every tool here is observational. None place orders or take
financial actions.
"""

from __future__ import annotations

from typing import Any, List

from config_loader import AppConfig, load_config
from market_data import get_sector_performance, get_stock_quote


def build_server(config: "AppConfig | None" = None) -> Any:
    """Construct and return a configured FastMCP server instance.

    Tools read live/mock behaviour from ``config.mcp.market_data`` so nothing
    about the data source is hardcoded.
    """
    try:
        from fastmcp import FastMCP  # lazy import — only needed to actually serve
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "fastmcp is not installed. Install with: pip install fastmcp"
        ) from exc

    cfg = config or load_config()
    md = cfg.mcp.market_data
    news = cfg.newsapi
    search = cfg.search
    newsdata = cfg.newsdata

    # Persist knowledge graphs to the configured location so they survive across
    # sessions and can be queried/visualized later.
    from sector_graph import set_graphs_dir

    set_graphs_dir(cfg.storage_paths.graphs)

    mcp = FastMCP(name="indian-market-data")

    @mcp.tool
    def get_indian_stock_quote(query: str, exchange: str = "") -> dict:
        """Fetch a real-time quote for a single Indian stock.

        Args:
            query: Company name (e.g. "Reliance", "TCS") or ticker (e.g. "INFY").
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
        """
        return get_stock_quote(
            query,
            exchange or md.default_exchange,
            use_live=md.use_live,
            cache_ttl_seconds=md.cache_ttl_seconds,
        )

    @mcp.tool
    def get_indian_sector_performance(sector: str, exchange: str = "") -> dict:
        """Fetch aggregate performance for an Indian market sector.

        Args:
            sector: Sector name, e.g. "IT", "banking", "auto", "pharma", "fmcg".
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
        """
        return get_sector_performance(
            sector,
            exchange or md.default_exchange,
            use_live=md.use_live,
            cache_ttl_seconds=md.cache_ttl_seconds,
        )

    @mcp.tool
    def get_stock_news(query: str, max_articles: int = 0) -> dict:
        """Fetch recent news articles relevant to a stock's price.

        Returns headlines, sources, dates and summaries so the model can factor
        recent events and sentiment into its analysis. The NewsAPI key comes
        from the ``newsapi`` config section (or env NEWSAPI_KEY); without a key,
        deterministic mock headlines are returned.

        Args:
            query: Company name (e.g. "Reliance", "TCS") or ticker (e.g. "INFY").
            max_articles: Optional cap on articles returned. 0 = configured default.
        """
        from news_data import get_stock_news as _fetch  # avoid shadowing the tool

        return _fetch(
            query,
            api_key=news.api_key,
            base_url=news.base_url,
            page_size=max_articles or news.page_size,
            language=news.language,
            sort_by=news.sort_by,
            lookback_days=news.lookback_days,
            use_live=news.use_live,
        )

    # ---- statistical metrics (stock_stats.py) ------------------------------
    @mcp.tool
    def get_return_statistics(query: str, exchange: str = "", period_days: int = 252) -> dict:
        """Compute return/risk statistics for one Indian stock.

        Covers cumulative & annualized return, annualized volatility, Sharpe &
        Sortino ratios, max drawdown, and historical VaR/CVaR (95%) over the
        lookback window.

        Args:
            query: Company name (e.g. "Reliance") or ticker (e.g. "INFY").
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
            period_days: Trading-day lookback window (default 252 ≈ 1 year).
        """
        from stock_stats import get_return_statistics as _stats

        return _stats(
            query,
            exchange or md.default_exchange,
            period_days=period_days,
            use_live=md.use_live,
        )

    @mcp.tool
    def get_technical_indicators(query: str, exchange: str = "", period_days: int = 252) -> dict:
        """Compute technical indicators for one Indian stock.

        Returns SMA (20/50/200), EMA (12/26), RSI-14, MACD (+signal/histogram),
        Bollinger bands with %B, 1m/3m momentum, and distance from the window
        high/low.

        Args:
            query: Company name (e.g. "Reliance") or ticker (e.g. "INFY").
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
            period_days: Trading-day lookback window (default 252 ≈ 1 year).
        """
        from stock_stats import get_technical_indicators as _ind

        return _ind(
            query,
            exchange or md.default_exchange,
            period_days=period_days,
            use_live=md.use_live,
        )

    @mcp.tool
    def get_risk_metrics(
        query: str, exchange: str = "", benchmark: str = "^NSEI", period_days: int = 252
    ) -> dict:
        """Compute benchmark-relative risk metrics for one Indian stock.

        Returns beta, annualized Jensen's alpha, correlation, R², tracking
        error and information ratio versus the benchmark index.

        Args:
            query: Company name (e.g. "Reliance") or ticker (e.g. "INFY").
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
            benchmark: Benchmark Yahoo symbol (default "^NSEI" = NIFTY 50).
            period_days: Trading-day lookback window (default 252 ≈ 1 year).
        """
        from stock_stats import get_risk_metrics as _risk

        return _risk(
            query,
            exchange or md.default_exchange,
            benchmark=benchmark,
            period_days=period_days,
            use_live=md.use_live,
        )

    @mcp.tool
    def get_correlation_matrix(queries: List[str], exchange: str = "", period_days: int = 252) -> dict:
        """Compute pairwise daily-return correlations for two or more stocks.

        Also returns the pairs ranked by |correlation| — useful for
        diversification and pair analysis.

        Args:
            queries: Two or more company names or tickers.
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
            period_days: Trading-day lookback window (default 252 ≈ 1 year).
        """
        from stock_stats import get_correlation_matrix as _corr

        return _corr(
            queries,
            exchange or md.default_exchange,
            period_days=period_days,
            use_live=md.use_live,
        )

    @mcp.tool
    def get_stock_fundamentals(query: str, exchange: str = "") -> dict:
        """Fetch a fundamentals snapshot for one Indian stock.

        Returns market cap, trailing/forward P/E, P/B, ROE, debt-to-equity,
        margins, revenue/earnings growth, dividend yield, EPS and book value.

        Args:
            query: Company name (e.g. "Reliance") or ticker (e.g. "INFY").
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
        """
        from stock_stats import get_fundamentals as _fund

        return _fund(query, exchange or md.default_exchange, use_live=md.use_live)

    # ---- sector knowledge graph (sector_graph.py, early prototype) ---------
    @mcp.tool
    def build_sector_graph(
        sectors: List[str],
        exchange: str = "",
        period_days: int = 252,
        correlation_threshold: float = 0.4,
    ) -> dict:
        """Build a knowledge graph over one or more Indian market sectors.

        Nodes are the sectors' stocks, each carrying a feature bundle: live
        quote, return statistics, technical indicators, fundamentals, alpha
        factors, news sentiment, and financial-filings features (placeholder
        until a filings backend exists). Edges are candidate associations
        seeded with quantitative evidence (return correlation, sector
        membership, sentiment alignment) in 'proposed' state — reason over the
        evidence and validate each edge via validate_graph_edge (multiple
        passes required), and add your own associations via propose_graph_edge.

        Args:
            sectors: Sector names, e.g. ["IT", "banking"].
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
            period_days: Trading-day lookback for node features (default 252).
            correlation_threshold: Min |correlation| to auto-seed a candidate edge.
        """
        from sector_graph import NewsSentimentProvider, build_sector_graph as _build

        return _build(
            sectors,
            exchange or md.default_exchange,
            period_days=period_days,
            correlation_threshold=correlation_threshold,
            use_live=md.use_live,
            sentiment_provider=NewsSentimentProvider(news),
        )

    @mcp.tool
    def propose_graph_edge(
        graph_id: str,
        source: str,
        target: str,
        relation: str,
        rationale: str,
        weight: float = 0.0,
    ) -> dict:
        """Add a new association between two stocks in an existing graph.

        Use this for associations you infer from the data that were not
        auto-seeded (e.g. supply-chain link, shared macro driver, competitor).
        The edge starts 'proposed' and still needs validate_graph_edge passes.

        Args:
            graph_id: Id returned by build_sector_graph.
            source: Ticker of one endpoint (must be a node in the graph).
            target: Ticker of the other endpoint.
            relation: Association type, e.g. "competitor".
            rationale: Why this association holds, grounded in the data.
            weight: Optional association strength in [-1, 1].
        """
        from sector_graph import propose_graph_edge as _propose

        return _propose(graph_id, source, target, relation, rationale, weight)

    @mcp.tool
    def validate_graph_edge(
        graph_id: str,
        source: str,
        target: str,
        relation: str,
        verdict: str,
        reasoning: str,
    ) -> dict:
        """Record one reason/reflect validation pass over a graph edge.

        Edges require multiple separate confirming passes to become
        'validated'; a single 'reject' marks them 'rejected'. Every pass and
        its reasoning is kept on the edge as an audit trail.

        Args:
            graph_id: Id returned by build_sector_graph.
            source: Ticker of one endpoint.
            target: Ticker of the other endpoint.
            relation: The edge's relation label.
            verdict: "confirm" or "reject".
            reasoning: Which evidence supports or contradicts the edge.
        """
        from sector_graph import validate_graph_edge as _validate

        return _validate(graph_id, source, target, relation, verdict, reasoning)

    @mcp.tool
    def get_sector_graph(graph_id: str, include_features: bool = False, status: str = "") -> dict:
        """Fetch the current state of a previously built sector graph.

        Args:
            graph_id: Id returned by build_sector_graph.
            include_features: Include full node feature bundles (verbose).
            status: Optional edge filter: "proposed" | "validated" | "rejected".
        """
        from sector_graph import get_sector_graph as _get

        return _get(graph_id, include_features=include_features, status=status)

    @mcp.tool
    def list_saved_graphs() -> dict:
        """List all knowledge graphs persisted on disk (to query later).

        Returns each graph's id, sectors, node/edge counts and validation
        status, so a graph built in an earlier session can be rediscovered.
        """
        from sector_graph import list_graphs

        return {"graphs": list_graphs()}

    # ---- autonomous web search (web_search.py, SearXNG) --------------------
    @mcp.tool
    def web_search(query: str, max_results: int = 0) -> dict:
        """Search the open web via a self-hosted SearXNG instance.

        Use autonomously when information beyond the market/news/graph tools and
        the portfolio context would improve the analysis (macro events,
        regulatory changes, companies outside the reference set). Returns
        titles, URLs and snippets. Without a reachable instance it degrades to
        deterministic mock results.

        Args:
            query: The search query (natural language or keywords).
            max_results: Optional cap on results. 0 = configured default.
        """
        from web_search import web_search as _search

        return _search(
            query,
            base_url=search.base_url,
            max_results=max_results or search.max_results,
            language=search.language,
            use_live=search.use_live,
            request_timeout=search.request_timeout,
        )

    # ---- historical archive news for backtesting (newsdata.io) -------------
    @mcp.tool
    def fetch_news_archive(query: str, from_date: str, to_date: str, max_articles: int = 0) -> dict:
        """Fetch historical news for a query between two dates (newsdata.io).

        For point-in-time sentiment during backtesting. Dates are clamped to the
        configured earliest date (no look-ahead before the model's training
        cutoff). Without a key it returns deterministic mock articles.

        Args:
            query: Company name or ticker/topic.
            from_date: Start date YYYY-MM-DD (clamped up to the floor).
            to_date: End date YYYY-MM-DD (must not precede the floor).
            max_articles: Optional cap. 0 = configured default.
        """
        from backtesting.news_archive import fetch_news_archive as _archive

        return _archive(
            query, from_date, to_date,
            api_key=newsdata.api_key,
            language=newsdata.language,
            earliest_date=newsdata.earliest_date,
            max_articles=max_articles or newsdata.max_articles,
            use_live=newsdata.use_live,
        )

    # ---- autonomous portfolio -> graph reasoning loop (graph_agent.py) -----
    @mcp.tool
    def build_portfolio_graph(
        portfolio_path: str = "", exchange: str = "", min_validations: int = 2
    ) -> dict:
        """Take a portfolio and autonomously build + reason over a graph.

        Builds a knowledge graph over the portfolio's equity holdings (nodes =
        tickers with alpha/indicator/fundamental/sentiment/filings features),
        proposes candidate associations, and validates each through repeated
        reason/reflect passes (deterministic heuristic here). The graph is
        persisted so it can be queried/visualized later.

        Args:
            portfolio_path: Portfolio CSV path. Empty = configured default.
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
            min_validations: Confirming passes required to accept an edge.
        """
        from graph_agent import build_portfolio_graph as _build
        from sector_graph import NewsSentimentProvider

        return _build(
            portfolio_path or cfg.storage_paths.default_portfolio,
            exchange or md.default_exchange,
            min_validations=min_validations,
            use_live=md.use_live,
            sentiment_provider=NewsSentimentProvider(news),
        )

    # ---- portfolio construction: two-step, LLM-driven (portfolio_builder.py)
    @mcp.tool
    def fetch_sector_analytics(sectors: List[str], exchange: str = "") -> dict:
        """Return raw per-stock metrics for the sectors' reference stocks.

        Volatility, P/E, Sharpe ratio, annualized return and price — this buys
        and sizes nothing. Use it to gather the numbers, then decide target
        weights yourself (dropping/shrinking negative-Sharpe names) and call
        generate_final_portfolio.

        Args:
            sectors: Sectors to analyze, e.g. ["it", "banking", "pharma"].
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
        """
        from portfolio_builder import fetch_sector_analytics as _fetch

        return _fetch(sectors, exchange or md.default_exchange, use_live=md.use_live)

    @mcp.tool
    def generate_final_portfolio(
        ticker_weights: dict,
        total_amount: float,
        reasoning: str = "",
        exchange: str = "",
    ) -> dict:
        """Build the final portfolio from the target weights you chose.

        Weights are literal fractions of capital (e.g. {"TCS": 0.25,
        "HDFCBANK": 0.2}); if they sum to <1 the remainder is held as cash.
        Python sizes each position, rounds DOWN to whole shares, and writes the
        CSV plus a reasoning file (your rationale + the share math) under the
        configured portfolios directory.

        Args:
            ticker_weights: Map of ticker -> weight (fraction of capital, 0-1).
            total_amount: Total capital to allocate, in INR.
            reasoning: Your explanation for the mix (written to the reasoning file).
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
        """
        from portfolio_builder import generate_final_portfolio as _generate

        return _generate(
            ticker_weights,
            total_amount,
            exchange=exchange or md.default_exchange,
            reasoning=reasoning,
            use_live=md.use_live,
            output_dir=cfg.storage_paths.portfolios,
        )

    # ---- graph visualization (graph_viz.py, Graphviz DOT) ------------------
    @mcp.tool
    def visualize_sector_graph(graph_id: str, image_format: str = "svg") -> dict:
        """Render a built graph to Graphviz DOT (and an image if available).

        Writes ``<graphs_dir>/<graph_id>.dot`` (plus an SVG/PNG if the Graphviz
        ``dot`` binary is on PATH) so the graph's nodes, edges and validation
        status can be inspected visually to evaluate the reasoning.

        Args:
            graph_id: Id returned by build_sector_graph / build_portfolio_graph.
            image_format: "svg" | "png" | "pdf" if Graphviz is installed.
        """
        from graph_viz import visualize_sector_graph as _viz

        return _viz(
            graph_id,
            out_dir=cfg.storage_paths.graphs,
            image_format=image_format,
        )

    return mcp


def main() -> None:  # pragma: no cover - exercised only when serving live
    cfg = load_config()
    server = build_server(cfg)
    transport = cfg.mcp.transport
    print(
        f"[mcp_server] starting FastMCP 'indian-market-data' on "
        f"{cfg.mcp.host}:{cfg.mcp.port} (transport={transport}, "
        f"live={cfg.mcp.market_data.use_live})"
    )
    # FastMCP networked transports take host/port; stdio takes neither.
    if transport in {"sse", "streamable-http", "http"}:
        server.run(transport=transport, host=cfg.mcp.host, port=cfg.mcp.port)
    else:
        server.run(transport=transport)


if __name__ == "__main__":
    main()
