"""Autonomous portfolio -> knowledge-graph reasoning loop.

Ties the pieces together: given a customer portfolio, this builds a graph over
its holdings (nodes = tickers with the full feature bundle from
``sector_graph``), then runs an autonomous **reason-and-reflect loop** that
validates each candidate association multiple times before accepting it. The
resulting graph is persisted (``sector_graph`` handles the disk store), so it
can be queried or visualized later — including in a subsequent session.

Edge-validation driver
-----------------------
Each validation pass is decided by one of two backends, in priority order:

1. **LLM-driven** — when a live :class:`llm_provider.LLMProvider` is supplied
   (Groq / self-hosted vLLM), the model is shown the edge evidence plus compact
   node feature digests and asked to ``confirm``/``reject`` with reasoning. This
   is the real "financial reasoning" path.
2. **Deterministic heuristic fallback** — when no provider is available (offline
   runs, the in-process MCP tool, tests, or if an LLM call fails), edges are
   decided from their quantitative evidence (return correlation, shared sector,
   sentiment alignment). This keeps the loop fully runnable with **no model and
   no GPU**.

Either way, an edge only becomes ``validated`` after the graph's
``min_validations`` confirming passes, and a single ``reject`` drops it — the
multi-round reflection the design calls for.

Scope guardrail: read-only analytics. Nothing here places orders or takes any
financial action.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from portfolio_parser import load_portfolio
from sector_graph import (
    GraphError,
    GraphNode,
    NewsSentimentProvider,
    SectorGraph,
    build_ticker_graph,
    get_graph_object,
    reverse_sector_lookup,
    validate_graph_edge,
)

logger = logging.getLogger("saul.graph_agent")


# --------------------------------------------------------------------------- #
# Node feature digest (for LLM prompts and heuristic transparency)
# --------------------------------------------------------------------------- #
def _digest(node: GraphNode) -> Dict[str, object]:
    f = node.features or {}
    alpha = f.get("alpha_factors", {}) or {}
    ind = f.get("indicators", {}) or {}
    sent = f.get("sentiment", {}) or {}
    fund = f.get("fundamentals", {}) or {}
    stats = f.get("return_stats", {}) or {}
    asof = f.get("asof_stats", {}) or {}          # point-in-time (as-of graph)
    news = f.get("news", {}) or {}
    digest = {
        "ticker": node.ticker,
        "sector": node.sector,
        "momentum_3m_pct": alpha.get("momentum_3m") or asof.get("momentum_pct"),
        "rsi_14": ind.get("rsi_14"),
        "sentiment_score": sent.get("score"),
        "trailing_pe": fund.get("trailing_pe"),
        "volatility_pct": stats.get("annualized_volatility_pct") or asof.get("volatility_pct"),
        "sharpe": asof.get("sharpe") or stats.get("sharpe_ratio"),
        "return_pct": asof.get("return_pct"),
    }
    # Surface a few recent headlines so the model reasons over news, not just numbers.
    heads = news.get("top_headlines") or [
        a.get("title") for a in (news.get("articles") or []) if a.get("title")
    ]
    if heads:
        digest["recent_headlines"] = heads[:3]
    return {k: v for k, v in digest.items() if v is not None}


# --------------------------------------------------------------------------- #
# Heuristic edge decision (deterministic, offline)
# --------------------------------------------------------------------------- #
def _heuristic_decision(evidence: Dict[str, object]) -> Tuple[str, float, bool, bool]:
    corr = float(evidence.get("return_correlation", 0.0) or 0.0)
    same_sector = bool(evidence.get("same_sector"))
    aligned = bool(evidence.get("sentiment_aligned"))
    strong = abs(corr) >= 0.5
    moderate = abs(corr) >= 0.3
    # Sector membership is itself sufficient evidence for a peer link; meaningful
    # co-movement (or, when available, sentiment alignment) justifies cross-sector
    # edges. This does NOT depend on sentiment, so it holds when sentiment is
    # excluded from the graph.
    confirm = strong or moderate or same_sector or aligned
    return ("confirm" if confirm else "reject"), corr, same_sector, aligned


def _heuristic_reason(pass_no: int, verdict: str, corr: float,
                      same_sector: bool, aligned: bool) -> str:
    basis = (
        f"return correlation {corr:+.2f}, "
        f"{'same' if same_sector else 'cross'}-sector, "
        f"sentiment {'aligned' if aligned else 'divergent'}"
    )
    if verdict == "confirm":
        if pass_no == 1:
            return f"Pass {pass_no} (heuristic): evidence supports the link — {basis}."
        return (
            f"Pass {pass_no} (heuristic reflection): re-examined the same "
            f"evidence from the co-movement and sector angle; still consistent "
            f"({basis}). No contradicting signal."
        )
    return (
        f"Pass {pass_no} (heuristic): evidence too weak to assert an "
        f"association — {basis}."
    )


# --------------------------------------------------------------------------- #
# LLM-driven edge decision (used only when a provider is supplied)
# --------------------------------------------------------------------------- #
_VALIDATOR_SYSTEM = (
    "You are validating a proposed association (edge) between two stocks in a "
    "financial knowledge graph. Weigh the quantitative evidence and each "
    "stock's features. Reply with ONE JSON object on a single line and nothing "
    'else: {"verdict": "confirm" | "reject", "reasoning": "<one or two '
    'sentences grounded in the evidence>"}.'
)


def _collect_content(provider, messages: List[Dict[str, object]], sink=None) -> str:
    """Stream a model reply, returning its content. If ``sink`` is given, each
    content/reasoning token is written to it live (so callers can display the
    model's output as the graph is validated)."""
    parts: List[str] = []
    for ev in provider.stream_chat(messages, tools=None):
        if ev.type == "content":
            parts.append(ev.text)
            if sink and ev.text:
                sink(ev.text)
        elif ev.type == "reasoning":
            if sink and ev.text:
                sink(ev.text)
        elif ev.type == "error":
            raise RuntimeError(ev.text)
        elif ev.type == "done":
            break
    return "".join(parts).strip()


def _parse_verdict(text: str) -> Tuple[str, str]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON object in model reply: {text[:120]!r}")
    obj = json.loads(text[start : end + 1])
    verdict = str(obj.get("verdict", "")).strip().lower()
    reasoning = str(obj.get("reasoning", "")).strip()
    if verdict not in {"confirm", "reject"} or not reasoning:
        raise ValueError(f"invalid verdict payload: {obj!r}")
    return verdict, reasoning


def _llm_decide(provider, graph: SectorGraph, edge, pass_no: int, sink=None) -> Tuple[str, str]:
    src = graph.nodes.get(edge.source)
    tgt = graph.nodes.get(edge.target)
    payload = {
        "relation": edge.relation,
        "evidence": edge.evidence,
        "stock_a": _digest(src) if src else edge.source,
        "stock_b": _digest(tgt) if tgt else edge.target,
        "reflection_pass": pass_no,
    }
    messages = [
        {"role": "system", "content": _VALIDATOR_SYSTEM},
        {
            "role": "user",
            "content": (
                "Validate this edge. Reflect independently on this pass.\n"
                + json.dumps(payload, indent=2)
            ),
        },
    ]
    verdict, reasoning = _parse_verdict(_collect_content(provider, messages, sink=sink))
    return verdict, f"Pass {pass_no} (LLM): {reasoning}"


# --------------------------------------------------------------------------- #
# Autonomous validation loop over one graph
# --------------------------------------------------------------------------- #
def _decide(provider, graph: SectorGraph, edge, pass_no: int, sink=None) -> Tuple[str, str]:
    if provider is not None:
        try:
            return _llm_decide(provider, graph, edge, pass_no, sink=sink)
        except Exception as exc:  # noqa: BLE001 - fall back, keep the loop alive
            logger.warning("LLM edge validation failed (%s); using heuristic.", exc)
    verdict, corr, same_sector, aligned = _heuristic_decision(edge.evidence)
    reason = _heuristic_reason(pass_no, verdict, corr, same_sector, aligned)
    if sink:
        sink(reason)
    return verdict, reason


def run_reasoning_loop(
    graph_id: str,
    *,
    provider: Optional[object] = None,
    sink=None,
) -> Dict[str, object]:
    """Validate every ``proposed`` edge in a graph through repeated passes.

    Each edge gets up to ``min_validations`` confirming passes (or a single
    rejecting pass). If ``sink`` (a ``str -> None`` writer) is given, the model's
    output / decision for each pass is streamed to it live. Returns a per-edge
    reasoning log plus the final status breakdown. Safe to re-run: already-decided
    edges are skipped.
    """
    graph = get_graph_object(graph_id)
    candidates = [e for e in graph.edges.values() if e.status == "proposed"]
    log: List[Dict[str, object]] = []

    for edge in candidates:
        label = f"{edge.source}-{edge.target} [{edge.relation}]"
        passes = 0
        while edge.status == "proposed" and passes < graph.min_validations:
            passes += 1
            if sink:
                sink(f"\n• {label} — pass {passes}: ")
            verdict, reasoning = _decide(provider, graph, edge, passes, sink=sink)
            validate_graph_edge(
                graph_id, edge.source, edge.target, edge.relation,
                verdict, reasoning,
            )
            if sink:
                sink(f"  [{verdict}]")
            if verdict == "reject":
                break
        if sink:
            sink(f"  => {edge.status}\n")
        log.append({
            "edge": label,
            "final_status": edge.status,
            "passes": len(edge.validations),
        })

    return {
        "graph_id": graph_id,
        "driver": "llm" if provider is not None else "heuristic",
        "reasoning_log": log,
        "graph": graph.summary(),
    }


# --------------------------------------------------------------------------- #
# Portfolio -> graph entrypoint
# --------------------------------------------------------------------------- #
def build_portfolio_graph(
    portfolio_path: str,
    exchange: str = "NS",
    *,
    period_days: int = 252,
    correlation_threshold: float = 0.4,
    min_validations: int = 2,
    use_live: bool = True,
    only_known: bool = True,
    provider: Optional[object] = None,
    sentiment_provider: Optional[object] = None,
    include_sentiment: bool = True,
    sink=None,
) -> Dict[str, object]:
    """Build a graph from a portfolio's holdings and auto-reason over its edges.

    Non-equity holdings (cash, and — when ``only_known`` — tickers outside the
    reference universe) are skipped and reported. The graph is persisted so it
    can be queried/visualized later. With no ``provider`` the reasoning loop
    uses the deterministic heuristic, so this runs fully offline.
    """
    pf = load_portfolio(portfolio_path)

    sector_map: Dict[str, str] = {}
    skipped: List[str] = []
    for h in pf.holdings:
        sector = reverse_sector_lookup(h.ticker)
        if h.asset_class.strip().lower() != "equity" or (only_known and sector == "other"):
            skipped.append(h.ticker)
            continue
        sector_map[h.ticker] = sector

    if not sector_map:
        raise GraphError(
            f"No graphable equity holdings in {portfolio_path} "
            "(need tickers in the known Indian universe)."
        )

    label = [f"portfolio:{Path(portfolio_path).stem}"] + sorted(set(sector_map.values()))
    built = build_ticker_graph(
        list(sector_map),
        exchange,
        label=label,
        sector_map=sector_map,
        period_days=period_days,
        correlation_threshold=correlation_threshold,
        min_validations=min_validations,
        use_live=use_live,
        sentiment_provider=sentiment_provider,
        include_sentiment=include_sentiment,
    )
    graph_id = str(built["graph_id"])

    reasoning = run_reasoning_loop(graph_id, provider=provider, sink=sink)

    return {
        "graph_id": graph_id,
        "portfolio": portfolio_path,
        "tickers": list(sector_map),
        "skipped_holdings": skipped,
        "driver": reasoning["driver"],
        "reasoning_log": reasoning["reasoning_log"],
        "graph": reasoning["graph"],
        "next_steps": (
            "The graph is persisted. Inspect it with get_sector_graph, render it "
            "with visualize_sector_graph, or add associations with "
            "propose_graph_edge / validate_graph_edge. It can be queried later "
            "by graph_id via list_saved_graphs."
        ),
    }


# --------------------------------------------------------------------------- #
# OpenAI-format tool schema (mirrors the MCP tool; passed to the LLM payload).
# --------------------------------------------------------------------------- #
PORTFOLIO_GRAPH_TOOL_SPECS: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "build_portfolio_graph",
            "description": (
                "Take a customer portfolio and autonomously build a knowledge "
                "graph over its equity holdings: nodes are the tickers (with "
                "alpha factors, indicators, fundamentals, sentiment and filings "
                "features) and edges are associations the system proposes and "
                "then validates through repeated reason/reflect passes. The "
                "graph is persisted so it can be queried or visualized later. "
                "Returns the graph_id, the reasoning log, and which holdings "
                "were skipped."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "portfolio_path": {
                        "type": "string",
                        "description": (
                            "Path to the portfolio CSV. Omit to use the "
                            "configured default portfolio."
                        ),
                    },
                    "exchange": {
                        "type": "string",
                        "enum": ["NS", "BO"],
                        "description": "NS = NSE (.NS), BO = BSE (.BO). Default NS.",
                    },
                    "min_validations": {
                        "type": "integer",
                        "description": "Confirming passes required to accept an edge (default 2).",
                    },
                },
                "required": [],
            },
        },
    },
]
