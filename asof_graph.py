"""Point-in-time user-portfolio knowledge graph.

Builds a graph over a *user's* portfolio tickers **as of a past date** (after the
model's training cutoff), so the LLM reasons over a historical snapshot rather
than present-day data:

* **Nodes** are the portfolio's tickers. Each carries **as-of statistical
  metrics** — return / volatility / Sharpe / momentum / max-drawdown computed
  from a historical price matrix over a trailing window ending at the chosen date
  (point-in-time, no look-ahead) — plus **news**: newsdata.io *archive* articles
  for the window with a computed lexicon **sentiment score**.
* **Edges** are candidate associations seeded from as-of return correlation and
  sector membership, which the LLM then validates through the reason/reflect loop
  in ``graph_agent`` (the node digests it reasons over include the as-of stats
  and recent headlines).

Backtest-style historical prices come from ``backtesting.engine.load_price_matrix``
(yfinance live, deterministic mock offline); archive news from
``backtesting.news_archive``. To keep the historical graph honest, ``web_search``
(which returns present-day results) is left out of the loop by default.

Scope guardrail: read-only/analytical. It never trades.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import logging
import math
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from backtesting.engine import load_price_matrix
from backtesting.news_archive import fetch_news_archive
from market_data import SECTORS
from portfolio_parser import load_portfolio
from sector_graph import (
    GraphEdge,
    GraphError,
    GraphNode,
    SectorGraph,
    edge_key,
    register_graph,
    reverse_sector_lookup,
)

logger = logging.getLogger("saul.asof_graph")

TRADING_DAYS = 252

# Compact sentiment lexicon (mirrors sector_graph's news scorer).
_POSITIVE = {
    "beat", "beats", "strong", "stronger", "growth", "gains", "rally", "surge",
    "upgrade", "upgraded", "record", "profit", "expansion", "buy", "outperform",
    "positive", "wins", "boost", "jump", "soars",
}
_NEGATIVE = {
    "miss", "misses", "weak", "weaker", "fall", "falls", "drop", "plunge",
    "downgrade", "downgraded", "loss", "losses", "probe", "fraud", "sell",
    "underperform", "negative", "cuts", "warning", "slump", "decline",
}


# --------------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------------- #
def _d(value: str) -> _dt.date:
    return _dt.date.fromisoformat(str(value).strip())


def resolve_window(start_date: str, end_date: str, window_days: int) -> Tuple[_dt.date, _dt.date]:
    start = _d(start_date)
    end = _d(end_date) if end_date else start + _dt.timedelta(days=max(1, window_days))
    if end <= start:
        raise GraphError(f"end_date {end} must be after start_date {start}.")
    return start, end


# --------------------------------------------------------------------------- #
# As-of statistics from the price matrix
# --------------------------------------------------------------------------- #
def _returns(closes: List[float]) -> List[float]:
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1]]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _correlation(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    a, b = a[-n:], b[-n:]
    sa, sb = _std(a), _std(b)
    if sa == 0.0 or sb == 0.0:
        return 0.0
    ma, mb = _mean(a), _mean(b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n - 1)
    return cov / (sa * sb)


def _asof_stats(closes: List[float], rf_annual: float = 0.05) -> Optional[Dict[str, object]]:
    if len(closes) < 3:
        return None
    rets = _returns(closes)
    mean_d, std_d = _mean(rets), _std(rets)
    total_ret = closes[-1] / closes[0] - 1.0
    ann_ret = total_ret * (TRADING_DAYS / max(1, len(rets)))
    ann_vol = std_d * math.sqrt(TRADING_DAYS)
    rf_daily = rf_annual / TRADING_DAYS
    sharpe = ((mean_d - rf_daily) / std_d * math.sqrt(TRADING_DAYS)) if std_d else 0.0
    peak, max_dd = closes[0], 0.0
    for c in closes:
        peak = max(peak, c)
        max_dd = min(max_dd, c / peak - 1.0)
    mom = (closes[-1] / closes[-21] - 1.0) if len(closes) > 21 else total_ret
    return {
        "price": round(closes[-1], 2),
        "return_pct": round(total_ret * 100.0, 2),
        "annualized_return_pct": round(ann_ret * 100.0, 2),
        "volatility_pct": round(ann_vol * 100.0, 2),
        "sharpe": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "momentum_pct": round(mom * 100.0, 2),
    }


# --------------------------------------------------------------------------- #
# Archive news + sentiment per ticker
# --------------------------------------------------------------------------- #
def _score_articles(articles: List[Dict[str, object]]) -> Tuple[float, int, int]:
    pos = neg = 0
    for a in articles:
        text = f"{a.get('title', '')} {a.get('description', '')}".lower()
        words = set(text.replace(",", " ").replace(".", " ").split())
        pos += len(words & _POSITIVE)
        neg += len(words & _NEGATIVE)
    score = (pos - neg) / (pos + neg) if (pos + neg) else 0.0
    return round(score, 3), pos, neg


def _news_feature(
    ticker: str, start: _dt.date, end: _dt.date, *, newsdata, use_live: bool, max_titles: int = 5
) -> Dict[str, object]:
    from news_data import resolve_company_name

    company = resolve_company_name(ticker)
    payload = fetch_news_archive(
        company, start.isoformat(), end.isoformat(),
        api_key=getattr(newsdata, "api_key", "") if newsdata else "",
        language=getattr(newsdata, "language", "en") if newsdata else "en",
        earliest_date=getattr(newsdata, "earliest_date", "2025-08-05") if newsdata else "2025-08-05",
        max_articles=getattr(newsdata, "max_articles", 8) if newsdata else 8,
        use_live=use_live and (getattr(newsdata, "use_live", True) if newsdata else True),
    )
    articles = payload.get("articles", [])
    score, pos, neg = _score_articles(articles)
    return {
        "company": company,
        "window": {"from": payload.get("from_date"), "to": payload.get("to_date")},
        "article_count": payload.get("article_count", 0),
        "source": payload.get("source"),
        "sentiment": {"score": score, "positive_hits": pos, "negative_hits": neg},
        "top_headlines": [a.get("title") for a in articles[:max_titles] if a.get("title")],
        "articles": [
            {"title": a.get("title"), "published_at": a.get("published_at"),
             "source": a.get("source")}
            for a in articles[:max_titles]
        ],
    }


# --------------------------------------------------------------------------- #
# Portfolio -> (ticker, sector) specs
# --------------------------------------------------------------------------- #
def _portfolio_specs(portfolio_path: str, only_known: bool = True) -> Tuple[Dict[str, str], List[str]]:
    pf = load_portfolio(portfolio_path)
    sector_map: Dict[str, str] = {}
    skipped: List[str] = []
    for h in pf.holdings:
        sector = reverse_sector_lookup(h.ticker)
        if h.asset_class.strip().lower() != "equity" or (only_known and sector == "other"):
            skipped.append(h.ticker)
            continue
        sector_map[h.ticker] = sector
    return sector_map, skipped


# --------------------------------------------------------------------------- #
# Build the point-in-time graph
# --------------------------------------------------------------------------- #
def build_asof_portfolio_graph(
    portfolio_path: str,
    *,
    start_date: str,
    end_date: str = "",
    window_days: int = 30,
    exchange: str = "NS",
    lookback_days: int = 126,
    correlation_threshold: float = 0.4,
    min_validations: int = 2,
    benchmark: str = "^NSEI",
    use_live: bool = True,
    newsdata=None,
    only_known: bool = True,
) -> Dict[str, object]:
    """Assemble a persisted as-of graph over a portfolio's tickers (no reasoning).

    Nodes carry as-of stats + archive news/sentiment; edges are seeded from as-of
    correlation + sector membership. Call the ``graph_agent`` reasoning loop next
    to validate the edges. Returns the graph_id, summary and candidate edges.
    """
    start, end = resolve_window(start_date, end_date, window_days)
    sector_map, skipped = _portfolio_specs(portfolio_path, only_known=only_known)
    if not sector_map:
        raise GraphError(
            f"No graphable equity holdings in {portfolio_path} "
            "(need tickers in the known Indian universe)."
        )
    tickers = list(sector_map)

    # Historical prices spanning both the news window and the metric lookback.
    matrix_start = min(start, end - _dt.timedelta(days=max(lookback_days, window_days) * 2))
    dates, matrix = load_price_matrix(tickers, benchmark, matrix_start, end, use_live=use_live)
    window = [d for d in dates if d <= end][-lookback_days:]

    returns_by_ticker: Dict[str, List[float]] = {}
    nodes: Dict[str, GraphNode] = {}
    for tick in tickers:
        closes = [matrix[tick][d] for d in window if d in matrix[tick]]
        stats = _asof_stats(closes)
        if stats is None:
            skipped.append(tick)
            continue
        returns_by_ticker[tick] = _returns(closes)
        news = _news_feature(tick, start, end, newsdata=newsdata, use_live=use_live)
        nodes[tick] = GraphNode(
            ticker=tick,
            sector=sector_map[tick],
            features={
                "as_of": end.isoformat(),
                "window": {"start": start.isoformat(), "end": end.isoformat()},
                "asof_stats": stats,
                "sentiment": news["sentiment"],
                "news": news,
            },
        )
    if len(nodes) < 1:
        raise GraphError("Could not compute as-of features for any ticker.")

    label = [f"portfolio:{Path(portfolio_path).stem}", f"asof:{end.isoformat()}"] + sorted(
        {sector_map[t] for t in nodes}
    )
    graph = SectorGraph(
        graph_id=uuid.uuid4().hex[:12],
        sectors=label,
        exchange=exchange,
        min_validations=max(1, int(min_validations)),
        created_at=time.time(),
    )
    graph.nodes = nodes

    # Seed candidate edges from as-of correlation + sector membership.
    node_list = list(nodes)
    for a, b in itertools.combinations(node_list, 2):
        ra, rb = returns_by_ticker[a], returns_by_ticker[b]
        n = min(len(ra), len(rb))
        corr = _correlation(ra[-n:], rb[-n:]) if n >= 2 else 0.0
        same_sector = nodes[a].sector == nodes[b].sector
        if abs(corr) < correlation_threshold and not same_sector:
            continue
        sent_a = nodes[a].features["sentiment"]["score"]
        sent_b = nodes[b].features["sentiment"]["score"]
        relation = "return_correlation" if abs(corr) >= correlation_threshold else "sector_peer"
        edge = GraphEdge(
            source=a, target=b, relation=relation, weight=corr, status="proposed",
            evidence={
                "return_correlation": round(corr, 3),
                "overlap_days": n,
                "same_sector": same_sector,
                "sentiment_scores": {a: sent_a, b: sent_b},
                "sentiment_aligned": (sent_a * sent_b) > 0,
                "as_of": end.isoformat(),
            },
        )
        graph.edges[edge.key()] = edge

    register_graph(graph)
    return {
        **graph.summary(),
        "portfolio": portfolio_path,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "as_of": end.isoformat(),
        "tickers": node_list,
        "skipped_holdings": skipped,
        "nodes": [n.as_dict() for n in graph.nodes.values()],
        "candidate_edges": [e.as_dict() for e in graph.edges.values()],
    }
