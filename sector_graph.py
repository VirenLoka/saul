"""Sector knowledge-graph construction — early prototype, decoupled core logic.

Builds an in-memory graph over the stocks of one or more sectors:

* **Nodes** are tickers, carrying a feature bundle assembled from the existing
  data layers — live quote (``market_data``), alpha factors & technical
  indicators & return stats (``stock_stats``), fundamentals (``stock_stats``),
  and news sentiment (``news_data`` scored by a naive lexicon).
* **Edges** are associations *the agent chooses to make* between stocks. The
  builder seeds candidate edges with quantitative evidence (return correlation,
  shared sector membership, sentiment alignment) in a ``proposed`` state; the
  agent then reasons over the node features + evidence and validates each edge
  through one or more reflection rounds (``validate_graph_edge``). An edge is
  only ``validated`` after ``min_validations`` confirming passes, or
  ``rejected`` on any rejecting pass — forcing the multiple reason/reflect
  cycles the design calls for.

Data-provider seams
-------------------
Retrieval that already exists in the repo is used directly. Retrieval that does
NOT exist yet is declared as an abstract base class so future implementations
slot in without touching the graph logic:

* :class:`SentimentProvider`   — implemented by :class:`NewsSentimentProvider`
  (NewsAPI headlines via ``news_data`` + a naive keyword lexicon). Swap in an
  LLM- or model-based scorer later.
* :class:`FilingsProvider`     — **abstract only**; no financial-filings
  retrieval exists in this repo yet. :class:`UnimplementedFilingsProvider`
  returns an explicit "unavailable" payload so node features degrade
  gracefully until an EDGAR/NSE-filings backend is written.

Like the other core modules this file has no MCP/FastMCP dependency and works
fully offline via the deterministic mock layers underneath it.

Scope guardrail: read-only analytics. Nothing here places orders or takes any
financial action.

PROTOTYPE NOTES: graphs live in a process-local registry (lost on restart);
persistence to ``knowledge/market_data/`` and richer edge typing are future
work.
"""

from __future__ import annotations

import itertools
import json
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from market_data import SECTORS, get_stock_quote
from stock_stats import (
    daily_returns,
    get_fundamentals,
    get_price_series,
    get_return_statistics,
    get_technical_indicators,
    _correlation,
)


class GraphError(ValueError):
    """Raised for unknown graphs/nodes/edges or invalid graph operations."""


# --------------------------------------------------------------------------- #
# Abstract data-provider seams (future retrieval backends plug in here)
# --------------------------------------------------------------------------- #
class SentimentProvider(ABC):
    """Turns recent coverage of a ticker into a scalar sentiment score."""

    @abstractmethod
    def get_sentiment(self, ticker: str) -> Dict[str, object]:
        """Return at least ``{"score": float in [-1, 1], "source": str}``."""


class FilingsProvider(ABC):
    """Retrieves financial-filings features (annual reports, quarterlies…).

    No filings backend exists in this repo yet — implement this against
    EDGAR / NSE corporate filings / an internal store and pass it to
    ``build_sector_graph``.
    """

    @abstractmethod
    def get_filing_features(self, ticker: str) -> Dict[str, object]:
        """Return filings-derived features for ``ticker``."""


class UnimplementedFilingsProvider(FilingsProvider):
    """Explicit placeholder until a real filings retriever is written."""

    def get_filing_features(self, ticker: str) -> Dict[str, object]:
        return {
            "available": False,
            "note": "No filings retrieval backend implemented yet.",
        }


# Naive keyword lexicon for headline sentiment. Deliberately simple for the
# prototype; replace via a custom SentimentProvider when a real model lands.
_POSITIVE_WORDS = {
    "beat", "beats", "strong", "stronger", "growth", "gains", "rally", "surge",
    "upgrade", "upgraded", "record", "profit", "expansion", "buy", "outperform",
    "lifting", "positive", "wins", "boost",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "weak", "weaker", "fall", "falls", "drop", "plunge",
    "downgrade", "downgraded", "loss", "losses", "probe", "fraud", "sell",
    "underperform", "negative", "cuts", "warning", "slump",
}


class NewsSentimentProvider(SentimentProvider):
    """Scores recent NewsAPI headlines (via ``news_data``) with a keyword lexicon."""

    def __init__(self, newsapi_settings: Optional[object] = None) -> None:
        # Duck-typed NewsApiSettings (config_loader); None -> mock headlines.
        self.settings = newsapi_settings

    def get_sentiment(self, ticker: str) -> Dict[str, object]:
        from news_data import get_stock_news

        ns = self.settings
        payload = get_stock_news(
            ticker,
            api_key=getattr(ns, "api_key", "") if ns else "",
            base_url=getattr(ns, "base_url", "https://newsapi.org/v2/everything")
            if ns else "https://newsapi.org/v2/everything",
            page_size=getattr(ns, "page_size", 8) if ns else 8,
            language=getattr(ns, "language", "en") if ns else "en",
            sort_by=getattr(ns, "sort_by", "publishedAt") if ns else "publishedAt",
            lookback_days=getattr(ns, "lookback_days", 7) if ns else 7,
            use_live=getattr(ns, "use_live", False) if ns else False,
        )

        pos = neg = 0
        for a in payload.get("articles", []):
            text = f"{a.get('title', '')} {a.get('description', '')}".lower()
            words = set(text.replace(",", " ").replace(".", " ").split())
            pos += len(words & _POSITIVE_WORDS)
            neg += len(words & _NEGATIVE_WORDS)
        score = (pos - neg) / (pos + neg) if (pos + neg) else 0.0
        return {
            "score": round(score, 3),
            "positive_hits": pos,
            "negative_hits": neg,
            "article_count": payload.get("article_count", 0),
            "source": f"news-lexicon/{payload.get('source', 'mock')}",
        }


# --------------------------------------------------------------------------- #
# Graph model
# --------------------------------------------------------------------------- #
@dataclass
class GraphNode:
    ticker: str
    sector: str
    features: Dict[str, object] = field(default_factory=dict)

    def as_dict(self, include_features: bool = True) -> Dict[str, object]:
        d: Dict[str, object] = {"ticker": self.ticker, "sector": self.sector}
        if include_features:
            d["features"] = self.features
        return d


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str                 # e.g. "return_correlation", "sector_peer", custom
    weight: float = 0.0
    status: str = "proposed"      # proposed | validated | rejected
    evidence: Dict[str, object] = field(default_factory=dict)
    # One entry per agent reason/reflect pass over this edge.
    validations: List[Dict[str, object]] = field(default_factory=list)

    def key(self) -> str:
        return edge_key(self.source, self.target, self.relation)

    def as_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": round(self.weight, 3),
            "status": self.status,
            "evidence": self.evidence,
            "validations": self.validations,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "GraphEdge":
        return cls(
            source=str(d["source"]),
            target=str(d["target"]),
            relation=str(d["relation"]),
            weight=float(d.get("weight", 0.0)),
            status=str(d.get("status", "proposed")),
            evidence=dict(d.get("evidence", {}) or {}),
            validations=list(d.get("validations", []) or []),
        )


def edge_key(source: str, target: str, relation: str) -> str:
    a, b = sorted([source.upper(), target.upper()])   # undirected for now
    return f"{a}|{b}|{relation}"


@dataclass
class SectorGraph:
    graph_id: str
    sectors: List[str]
    exchange: str
    min_validations: int
    created_at: float
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[str, GraphEdge] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        by_status: Dict[str, int] = {}
        for e in self.edges.values():
            by_status[e.status] = by_status.get(e.status, 0) + 1
        return {
            "graph_id": self.graph_id,
            "sectors": self.sectors,
            "exchange": self.exchange,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "edges_by_status": by_status,
            "min_validations": self.min_validations,
        }

    def to_dict(self) -> Dict[str, object]:
        """Full JSON-serializable snapshot (round-trips via ``from_dict``)."""
        return {
            "graph_id": self.graph_id,
            "sectors": self.sectors,
            "exchange": self.exchange,
            "min_validations": self.min_validations,
            "created_at": self.created_at,
            "nodes": {t: n.as_dict() for t, n in self.nodes.items()},
            "edges": {k: e.as_dict() for k, e in self.edges.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "SectorGraph":
        g = cls(
            graph_id=str(d["graph_id"]),
            sectors=list(d.get("sectors", []) or []),
            exchange=str(d.get("exchange", "NS")),
            min_validations=int(d.get("min_validations", 2)),
            created_at=float(d.get("created_at", 0.0)),
        )
        for t, nd in (d.get("nodes", {}) or {}).items():
            g.nodes[str(t)] = GraphNode(
                ticker=str(nd["ticker"]),
                sector=str(nd.get("sector", "")),
                features=dict(nd.get("features", {}) or {}),
            )
        for k, ed in (d.get("edges", {}) or {}).items():
            g.edges[str(k)] = GraphEdge.from_dict(ed)
        return g


# --------------------------------------------------------------------------- #
# Persistence — one JSON file per graph, so a graph built in one session can be
# queried/visualized in a later one. The directory is configurable (the CLI /
# MCP server point it at storage_paths.graphs); saves are best-effort so an
# unwritable disk never breaks the in-memory prototype.
# --------------------------------------------------------------------------- #
_DEFAULT_GRAPHS_DIR = Path("knowledge/market_data/graphs")
_graphs_dir: Path = _DEFAULT_GRAPHS_DIR

# Process-local registry (fast path; disk is the durable store).
_GRAPHS: Dict[str, SectorGraph] = {}


def set_graphs_dir(path: "str | Path") -> None:
    """Point graph persistence at ``path`` (called from config at startup)."""
    global _graphs_dir
    _graphs_dir = Path(path)


def _graph_path(graph_id: str) -> Path:
    return _graphs_dir / f"{graph_id}.json"


def save_graph(graph: SectorGraph) -> None:
    """Persist ``graph`` to ``<graphs_dir>/<graph_id>.json`` (best-effort)."""
    try:
        _graphs_dir.mkdir(parents=True, exist_ok=True)
        _graph_path(graph.graph_id).write_text(
            json.dumps(graph.to_dict(), indent=2), encoding="utf-8"
        )
    except OSError:  # persistence is best-effort; memory registry still works
        pass


def load_graph(graph_id: str) -> Optional[SectorGraph]:
    """Load a persisted graph from disk, or None if absent/corrupt."""
    p = _graph_path(graph_id)
    if not p.exists():
        return None
    try:
        return SectorGraph.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError):
        return None


def clear_graphs() -> None:
    """Drop all in-memory graphs (used by tests). Does not touch disk."""
    _GRAPHS.clear()


def list_graphs() -> List[Dict[str, object]]:
    """Summaries of every persisted graph on disk (for 'query later')."""
    out: List[Dict[str, object]] = []
    if not _graphs_dir.exists():
        return out
    for p in sorted(_graphs_dir.glob("*.json")):
        g = load_graph(p.stem)
        if g is not None:
            out.append(g.summary())
    return out


def get_graph_object(graph_id: str) -> SectorGraph:
    """Return a graph by id from memory, falling back to disk."""
    g = _GRAPHS.get(graph_id)
    if g is None:
        g = load_graph(graph_id)
        if g is not None:
            _GRAPHS[graph_id] = g
    if g is None:
        raise GraphError(
            f"Unknown graph_id '{graph_id}'. Build one with build_sector_graph "
            "or build_portfolio_graph first."
        )
    return g


# Internal alias kept for readability at call sites.
_get_graph = get_graph_object


def register_graph(graph: SectorGraph) -> None:
    """Register a directly-constructed graph in the registry and persist it.

    For callers (e.g. the as-of portfolio-graph builder) that assemble a
    :class:`SectorGraph` with custom node features rather than via
    ``build_sector_graph``.
    """
    _GRAPHS[graph.graph_id] = graph
    save_graph(graph)


# --------------------------------------------------------------------------- #
# Node feature assembly
# --------------------------------------------------------------------------- #
def _alpha_factors(ticker: str, exchange: str, *, period_days: int,
                   use_live: bool) -> Dict[str, object]:
    """Cross-sectional alpha-factor style signals from the price series."""
    _, closes, source = get_price_series(
        ticker, exchange, days=period_days, use_live=use_live,
    )
    rets = daily_returns(closes)
    last = closes[-1]

    def _mom(days: int) -> Optional[float]:
        return (last / closes[-days] - 1.0) if len(closes) > days else None

    # Classic 12-1 momentum: 12-month return skipping the most recent month.
    mom_12_1 = (
        closes[-21] / closes[0] - 1.0 if len(closes) >= 200 else None
    )
    vol_series = rets[-63:] if len(rets) >= 63 else rets
    vol = (sum((r - sum(vol_series) / len(vol_series)) ** 2
               for r in vol_series) / max(1, len(vol_series) - 1)) ** 0.5

    return {
        "momentum_1m": round(_mom(21) * 100.0, 2) if _mom(21) is not None else None,
        "momentum_3m": round(_mom(63) * 100.0, 2) if _mom(63) is not None else None,
        "momentum_12_1": round(mom_12_1 * 100.0, 2) if mom_12_1 is not None else None,
        "short_term_reversal_5d": round(_mom(5) * -100.0, 2) if _mom(5) is not None else None,
        "volatility_3m_daily_pct": round(vol * 100.0, 3),
        "source": source,
    }


def _build_node(
    ticker: str,
    sector: str,
    exchange: str,
    *,
    period_days: int,
    use_live: bool,
    sentiment_provider: SentimentProvider,
    filings_provider: FilingsProvider,
    include_sentiment: bool = True,
) -> GraphNode:
    features: Dict[str, object] = {}
    features["quote"] = get_stock_quote(ticker, exchange, use_live=use_live)
    features["return_stats"] = get_return_statistics(
        ticker, exchange, period_days=period_days, use_live=use_live,
    )
    features["indicators"] = get_technical_indicators(
        ticker, exchange, period_days=period_days, use_live=use_live,
    )
    features["fundamentals"] = get_fundamentals(ticker, exchange, use_live=use_live)
    features["alpha_factors"] = _alpha_factors(
        ticker, exchange, period_days=period_days, use_live=use_live,
    )
    # Sentiment is optional — excluded when the caller defers it (e.g. the graph
    # built by run_graph_reasoning computes sentiment separately, later).
    if include_sentiment:
        features["sentiment"] = sentiment_provider.get_sentiment(ticker)
    features["filings"] = filings_provider.get_filing_features(ticker)
    return GraphNode(ticker=ticker, sector=sector, features=features)


# --------------------------------------------------------------------------- #
# Candidate-edge seeding (evidence for the agent to reason over)
# --------------------------------------------------------------------------- #
def _seed_candidate_edges(
    graph: SectorGraph,
    *,
    period_days: int,
    use_live: bool,
    correlation_threshold: float,
) -> None:
    tickers = list(graph.nodes)
    returns: Dict[str, List[float]] = {}
    for t in tickers:
        _, closes, _ = get_price_series(
            t, graph.exchange, days=period_days, use_live=use_live,
        )
        returns[t] = daily_returns(closes)

    for a, b in itertools.combinations(tickers, 2):
        n = min(len(returns[a]), len(returns[b]))
        corr = _correlation(returns[a][-n:], returns[b][-n:]) if n >= 2 else 0.0
        same_sector = graph.nodes[a].sector == graph.nodes[b].sector
        sent_a = graph.nodes[a].features.get("sentiment", {}).get("score", 0.0)
        sent_b = graph.nodes[b].features.get("sentiment", {}).get("score", 0.0)
        sentiment_aligned = (sent_a * sent_b) > 0

        # Seed only pairs with at least one quantitative reason; the agent can
        # still propose_graph_edge anything it spots in the features.
        if abs(corr) < correlation_threshold and not same_sector:
            continue

        relation = "return_correlation" if abs(corr) >= correlation_threshold else "sector_peer"
        edge = GraphEdge(
            source=a,
            target=b,
            relation=relation,
            weight=corr,
            status="proposed",
            evidence={
                "return_correlation": round(corr, 3),
                "overlap_days": n,
                "same_sector": same_sector,
                "sentiment_scores": {a: sent_a, b: sent_b},
                "sentiment_aligned": sentiment_aligned,
            },
        )
        graph.edges[edge.key()] = edge


# --------------------------------------------------------------------------- #
# Public API (wrapped as MCP tools in mcp_server.py)
# --------------------------------------------------------------------------- #
def reverse_sector_lookup(ticker: str) -> str:
    """Return the sector a base ticker belongs to (or 'other' if unknown)."""
    t = ticker.upper()
    for sector, members in SECTORS.items():
        if t in members:
            return sector
    return "other"


def _assemble_and_store(
    *,
    node_specs: Sequence[Tuple[str, str]],
    sectors_label: List[str],
    exchange: str,
    period_days: int,
    correlation_threshold: float,
    min_validations: int,
    use_live: bool,
    sentiment_provider: Optional[SentimentProvider],
    filings_provider: Optional[FilingsProvider],
    include_sentiment: bool = True,
) -> SectorGraph:
    """Build nodes + seed candidate edges for ``(ticker, sector)`` specs, store
    the graph in the registry, and persist it to disk."""
    sp = sentiment_provider or NewsSentimentProvider()
    fp = filings_provider or UnimplementedFilingsProvider()

    graph = SectorGraph(
        graph_id=uuid.uuid4().hex[:12],
        sectors=sectors_label,
        exchange=exchange,
        min_validations=max(1, int(min_validations)),
        created_at=time.time(),
    )
    for ticker, sector in node_specs:
        if ticker not in graph.nodes:   # de-dupe shared constituents
            graph.nodes[ticker] = _build_node(
                ticker, sector, exchange,
                period_days=period_days, use_live=use_live,
                sentiment_provider=sp, filings_provider=fp,
                include_sentiment=include_sentiment,
            )

    _seed_candidate_edges(
        graph, period_days=period_days, use_live=use_live,
        correlation_threshold=correlation_threshold,
    )
    _GRAPHS[graph.graph_id] = graph
    save_graph(graph)
    return graph


def _build_result(graph: SectorGraph) -> Dict[str, object]:
    return {
        **graph.summary(),
        "nodes": [n.as_dict() for n in graph.nodes.values()],
        "candidate_edges": [e.as_dict() for e in graph.edges.values()],
        "next_steps": (
            "Reason over each candidate edge's evidence together with the node "
            "features, then call validate_graph_edge for every edge at least "
            f"{graph.min_validations} time(s) (separate reflection passes). "
            "Use propose_graph_edge for associations you infer that were not "
            "auto-seeded."
        ),
    }


def build_sector_graph(
    sectors: Sequence[str] | str,
    exchange: str = "NS",
    *,
    period_days: int = 252,
    correlation_threshold: float = 0.4,
    min_validations: int = 2,
    use_live: bool = True,
    sentiment_provider: Optional[SentimentProvider] = None,
    filings_provider: Optional[FilingsProvider] = None,
    include_sentiment: bool = True,
) -> Dict[str, object]:
    """Build a graph over the given sector(s) and return it with evidence.

    Nodes carry quote/stats/indicator/fundamental/alpha/sentiment/filings
    features; edges are seeded as ``proposed`` candidates with quantitative
    evidence for the agent to validate (see ``validate_graph_edge``). Set
    ``include_sentiment=False`` to omit the news-sentiment feature (deferred).
    """
    if isinstance(sectors, str):
        sectors = [s for s in (p.strip() for p in sectors.split(",")) if s]
    keys = [s.lower() for s in sectors if (s or "").strip()]
    if not keys:
        raise GraphError("No sectors given.")
    unknown = [s for s in keys if s not in SECTORS]
    if unknown:
        raise GraphError(f"Unknown sectors {unknown}. Known: {sorted(SECTORS)}.")

    node_specs: List[Tuple[str, str]] = []
    seen = set()
    for sector in keys:
        for ticker in SECTORS[sector]:
            if ticker not in seen:
                seen.add(ticker)
                node_specs.append((ticker, sector))

    graph = _assemble_and_store(
        node_specs=node_specs, sectors_label=keys, exchange=exchange,
        period_days=period_days, correlation_threshold=correlation_threshold,
        min_validations=min_validations, use_live=use_live,
        sentiment_provider=sentiment_provider, filings_provider=filings_provider,
        include_sentiment=include_sentiment,
    )
    return _build_result(graph)


def build_ticker_graph(
    tickers: Sequence[str],
    exchange: str = "NS",
    *,
    label: Optional[Sequence[str]] = None,
    sector_map: Optional[Dict[str, str]] = None,
    period_days: int = 252,
    correlation_threshold: float = 0.4,
    min_validations: int = 2,
    use_live: bool = True,
    sentiment_provider: Optional[SentimentProvider] = None,
    filings_provider: Optional[FilingsProvider] = None,
    include_sentiment: bool = True,
) -> Dict[str, object]:
    """Build a graph over an explicit ticker list (e.g. a portfolio's holdings).

    Each ticker's sector is taken from ``sector_map`` if given, else inferred
    via :func:`reverse_sector_lookup`. Used by the autonomous
    portfolio-graph builder. ``include_sentiment=False`` omits the sentiment
    feature.
    """
    node_specs: List[Tuple[str, str]] = []
    seen = set()
    for raw in tickers:
        t = (raw or "").strip().upper()
        if not t or t in seen:
            continue
        seen.add(t)
        sector = (sector_map or {}).get(t) or reverse_sector_lookup(t)
        node_specs.append((t, sector))
    if not node_specs:
        raise GraphError("No usable tickers to build a graph from.")

    sectors_label = list(label) if label is not None else sorted(
        {s for _, s in node_specs}
    )
    graph = _assemble_and_store(
        node_specs=node_specs, sectors_label=sectors_label, exchange=exchange,
        period_days=period_days, correlation_threshold=correlation_threshold,
        min_validations=min_validations, use_live=use_live,
        sentiment_provider=sentiment_provider, filings_provider=filings_provider,
        include_sentiment=include_sentiment,
    )
    return _build_result(graph)


def propose_graph_edge(
    graph_id: str,
    source: str,
    target: str,
    relation: str,
    rationale: str,
    weight: float = 0.0,
) -> Dict[str, object]:
    """Agent-initiated association between two nodes (starts as ``proposed``)."""
    g = _get_graph(graph_id)
    s, t = source.upper().strip(), target.upper().strip()
    for tick in (s, t):
        if tick not in g.nodes:
            raise GraphError(f"Ticker '{tick}' is not a node in graph {graph_id}.")
    if s == t:
        raise GraphError("Self-edges are not allowed.")
    if not (relation or "").strip() or not (rationale or "").strip():
        raise GraphError("Both 'relation' and 'rationale' are required.")

    key = edge_key(s, t, relation.strip())
    if key in g.edges:
        raise GraphError(f"Edge already exists: {key}. Validate it instead.")

    edge = GraphEdge(
        source=s, target=t, relation=relation.strip(), weight=float(weight),
        status="proposed", evidence={"agent_rationale": rationale.strip()},
    )
    g.edges[key] = edge
    save_graph(g)
    return {"graph_id": graph_id, "edge": edge.as_dict(), "graph": g.summary()}


def validate_graph_edge(
    graph_id: str,
    source: str,
    target: str,
    relation: str,
    verdict: str,
    reasoning: str,
) -> Dict[str, object]:
    """Record one reason/reflect pass over an edge.

    ``verdict`` is ``confirm`` or ``reject``. An edge is promoted to
    ``validated`` only after the graph's ``min_validations`` confirming passes;
    a single ``reject`` marks it ``rejected``. Each pass (with its reasoning)
    is kept on the edge as an audit trail.
    """
    g = _get_graph(graph_id)
    v = (verdict or "").strip().lower()
    if v not in {"confirm", "reject"}:
        raise GraphError("verdict must be 'confirm' or 'reject'.")
    if not (reasoning or "").strip():
        raise GraphError("Non-empty reasoning is required for every validation pass.")

    key = edge_key(source, target, (relation or "").strip())
    edge = g.edges.get(key)
    if edge is None:
        raise GraphError(
            f"No edge {source}-{target} [{relation}] in graph {graph_id}. "
            "Propose it first with propose_graph_edge."
        )
    if edge.status == "rejected":
        raise GraphError("Edge was already rejected; propose a new relation instead.")

    edge.validations.append({
        "pass": len(edge.validations) + 1,
        "verdict": v,
        "reasoning": reasoning.strip(),
        "at": time.time(),
    })
    if v == "reject":
        edge.status = "rejected"
    else:
        confirms = sum(1 for x in edge.validations if x["verdict"] == "confirm")
        edge.status = "validated" if confirms >= g.min_validations else "proposed"

    save_graph(g)
    remaining = 0 if edge.status != "proposed" else (
        g.min_validations
        - sum(1 for x in edge.validations if x["verdict"] == "confirm")
    )
    return {
        "graph_id": graph_id,
        "edge": edge.as_dict(),
        "confirmations_still_needed": max(0, remaining),
        "graph": g.summary(),
    }


def get_sector_graph(
    graph_id: str,
    *,
    include_features: bool = False,
    status: str = "",
) -> Dict[str, object]:
    """Return a graph's current nodes/edges (optionally filtered by status)."""
    g = _get_graph(graph_id)
    wanted = (status or "").strip().lower()
    edges = [
        e.as_dict() for e in g.edges.values()
        if not wanted or e.status == wanted
    ]
    return {
        **g.summary(),
        "nodes": [n.as_dict(include_features=include_features) for n in g.nodes.values()],
        "edges": edges,
    }


def get_all_graphs(
    *,
    include_features: bool = False,
    status: str = "",
    ticker: str = "",
    sector: str = "",
) -> Dict[str, object]:
    """Fetch EVERY persisted graph on disk at once, plus a cross-graph aggregate.

    Loads every graph in the graphs directory and returns their nodes/edges
    together with an index tying the whole collection together — which graphs
    each ticker appears in, and the union of ``validated`` peer associations
    across all graphs — so the agent can reason over all graphs for various
    purposes (find every graph containing a stock, gather all validated links,
    compare sectors, etc.).

    Optional filters narrow the returned graphs: ``ticker`` (only graphs with
    that node), ``sector`` (only graphs covering it), and ``status`` (edge
    filter). ``include_features`` attaches the full node feature bundles
    (verbose — off by default to keep the payload small).
    """
    wanted = (status or "").strip().lower()
    tick = (ticker or "").strip().upper()
    sect = (sector or "").strip().lower()

    graphs_out: List[Dict[str, object]] = []
    ticker_index: Dict[str, List[str]] = {}
    validated_peers: Dict[str, set] = {}

    if _graphs_dir.exists():
        for path in sorted(_graphs_dir.glob("*.json")):
            g = load_graph(path.stem)
            if g is None:
                continue
            if tick and tick not in g.nodes:
                continue
            if sect and sect not in {s.lower() for s in g.sectors}:
                continue

            edges = [
                e.as_dict() for e in g.edges.values()
                if not wanted or e.status == wanted
            ]
            graphs_out.append({
                **g.summary(),
                "nodes": [n.as_dict(include_features=include_features) for n in g.nodes.values()],
                "edges": edges,
            })
            for t in g.nodes:
                ticker_index.setdefault(t, []).append(g.graph_id)
            for e in g.edges.values():
                if e.status == "validated":
                    validated_peers.setdefault(e.source, set()).add(e.target)
                    validated_peers.setdefault(e.target, set()).add(e.source)

    return {
        "graphs_dir": str(_graphs_dir),
        "graph_count": len(graphs_out),
        "unique_ticker_count": len(ticker_index),
        "unique_tickers": sorted(ticker_index),
        "ticker_index": {t: gids for t, gids in sorted(ticker_index.items())},
        "validated_associations": {
            t: sorted(peers) for t, peers in sorted(validated_peers.items())
        },
        "graphs": graphs_out,
    }


# --------------------------------------------------------------------------- #
# OpenAI-format tool schemas (mirror the MCP tools; passed to the LLM payload).
# --------------------------------------------------------------------------- #
GRAPH_TOOL_SPECS: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "build_sector_graph",
            "description": (
                "Build a knowledge graph for one or more Indian market sectors "
                "(e.g. 'IT', 'banking'). Nodes are the sector's stocks with "
                "features (live quote, return stats, technical indicators, "
                "fundamentals, alpha factors, news sentiment, filings). Edges "
                "are candidate associations seeded with quantitative evidence "
                "(return correlation, sector membership, sentiment alignment) "
                "in 'proposed' state — you must then reason over the evidence "
                "and validate each edge via validate_graph_edge, and may add "
                "your own associations via propose_graph_edge. Returns a "
                "graph_id for follow-up calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sector names, e.g. ['IT', 'banking'].",
                    },
                    "exchange": {
                        "type": "string",
                        "enum": ["NS", "BO"],
                        "description": "NS = NSE (.NS), BO = BSE (.BO). Default NS.",
                    },
                    "period_days": {
                        "type": "integer",
                        "description": "Trading-day lookback for features (default 252).",
                    },
                    "correlation_threshold": {
                        "type": "number",
                        "description": (
                            "Min |return correlation| to auto-seed a candidate "
                            "edge between cross-sector pairs (default 0.4)."
                        ),
                    },
                },
                "required": ["sectors"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_graph_edge",
            "description": (
                "Add a new association between two stocks in an existing graph, "
                "with your rationale (e.g. supply-chain link, shared macro "
                "driver, competitor). Starts as 'proposed' and still requires "
                "validation passes via validate_graph_edge."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string", "description": "From build_sector_graph."},
                    "source": {"type": "string", "description": "Ticker of one endpoint."},
                    "target": {"type": "string", "description": "Ticker of the other endpoint."},
                    "relation": {
                        "type": "string",
                        "description": "Association type, e.g. 'competitor', 'macro_rate_sensitivity'.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Why this association holds, grounded in the node features/data.",
                    },
                    "weight": {
                        "type": "number",
                        "description": "Optional association strength in [-1, 1].",
                    },
                },
                "required": ["graph_id", "source", "target", "relation", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_graph_edge",
            "description": (
                "Record one reason/reflect validation pass over an edge. Give "
                "'confirm' or 'reject' plus your reasoning grounded in the "
                "node features and edge evidence. Edges need multiple separate "
                "confirming passes before they become 'validated'; one reject "
                "marks them 'rejected'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string", "description": "From build_sector_graph."},
                    "source": {"type": "string", "description": "Ticker of one endpoint."},
                    "target": {"type": "string", "description": "Ticker of the other endpoint."},
                    "relation": {"type": "string", "description": "The edge's relation label."},
                    "verdict": {
                        "type": "string",
                        "enum": ["confirm", "reject"],
                        "description": "Outcome of this validation pass.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Your reflection: which evidence supports or contradicts the edge.",
                    },
                },
                "required": ["graph_id", "source", "target", "relation", "verdict", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sector_graph",
            "description": (
                "Fetch the current state of a previously built sector graph: "
                "nodes, edges and validation status (optionally filtered to "
                "'proposed' | 'validated' | 'rejected')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "graph_id": {"type": "string", "description": "From build_sector_graph."},
                    "include_features": {
                        "type": "boolean",
                        "description": "Include full node feature bundles (verbose).",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["proposed", "validated", "rejected"],
                        "description": "Optional edge-status filter.",
                    },
                },
                "required": ["graph_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_saved_graphs",
            "description": (
                "List all previously built graphs persisted on disk (graph_id, "
                "sectors, node/edge counts, validation status). Use this to "
                "discover graphs built in an earlier session so you can query "
                "or visualize them."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_graphs",
            "description": (
                "Fetch EVERY persisted graph at once — all their nodes and edges "
                "plus a cross-graph aggregate: which graphs each ticker appears "
                "in (ticker_index) and the union of validated peer associations "
                "across all graphs (validated_associations). Use it to reason "
                "over the whole graph collection at once — e.g. find every graph "
                "containing a stock, gather all validated links for a name, or "
                "compare coverage across sectors. Filter with ticker / sector / "
                "status; set include_features to attach full node features "
                "(verbose). For just a directory of summaries, use "
                "list_saved_graphs instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Only graphs containing this ticker (e.g. 'TCS').",
                    },
                    "sector": {
                        "type": "string",
                        "description": "Only graphs covering this sector (e.g. 'banking').",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["proposed", "validated", "rejected"],
                        "description": "Optional edge-status filter.",
                    },
                    "include_features": {
                        "type": "boolean",
                        "description": "Attach full node feature bundles (verbose).",
                    },
                },
            },
        },
    },
]
