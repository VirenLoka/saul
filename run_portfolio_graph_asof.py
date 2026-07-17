#!/usr/bin/env python3
"""Standalone: LLM builds a POINT-IN-TIME user-portfolio knowledge graph.

Takes a user's portfolio and builds a graph *as of a past date* (after the
model's training cutoff, default 2025-08-08). Nodes are the portfolio's tickers;
their features are **as-of statistical metrics** (return / volatility / Sharpe /
momentum / drawdown over a trailing window ending at the chosen date — no
look-ahead) plus **newsdata.io archive news** for the window with a sentiment
score. An LLM then runs a reason/reflect loop over that historical snapshot to
validate the associations (edges), streaming its output live. The graph is
persisted and rendered to Graphviz DOT.

To keep the historical graph honest, ``web_search`` (present-day results) is left
out of the loop; the reasoning is grounded in the as-of stats + archive news
embedded in each node. Runs fully offline with ``--mock`` (mock prices +
deterministic heuristic edge validation, no model/GPU/network).

Examples
--------
    python run_portfolio_graph_asof.py --mock
    python run_portfolio_graph_asof.py --provider groq --start 2025-08-08 --end 2025-09-08
    python run_portfolio_graph_asof.py --portfolio knowledge/portfolios/banking_portfolio.csv --provider groq
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, TextIO

import sector_graph
from asof_graph import build_asof_portfolio_graph
from config_loader import ConfigError, load_config
from graph_agent import run_reasoning_loop
from graph_viz import visualize_sector_graph


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--portfolio", default=None, help="User portfolio CSV (default: config).")
    p.add_argument("--provider", default=None,
                   choices=["vllm", "groq", "mock"],
                   help="Provider for LLM edge validation (omit / mock -> heuristic).")
    p.add_argument("--start", default=None, help="Window start YYYY-MM-DD (>= training cutoff).")
    p.add_argument("--end", default=None, help="Window end YYYY-MM-DD (default: start + window_days).")
    p.add_argument("--window-days", type=int, default=None, help="Span when --end is omitted.")
    p.add_argument("--lookback-days", type=int, default=None, help="Trailing window for as-of metrics.")
    p.add_argument("--min-validations", type=int, default=None, help="Confirming passes per edge.")
    p.add_argument("--exchange", default=None, help="NS or BO (default: configured).")
    p.add_argument("--mock", action="store_true", help="Deterministic offline data (no network).")
    p.add_argument("--no-visualize", action="store_true", help="Skip the Graphviz render.")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logs to stderr.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None, out: Optional[TextIO] = None) -> int:
    out = out if out is not None else sys.stdout
    args = parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        cfg = load_config(args.config, provider_override=args.provider)
    except ConfigError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    pg = cfg.portfolio_graph
    portfolio = args.portfolio or pg.portfolio
    exchange = (args.exchange or cfg.mcp.market_data.default_exchange).upper()
    use_live = (not args.mock) and cfg.mcp.market_data.use_live

    # LLM edge validation only for a real provider (mock -> deterministic heuristic).
    provider = None
    if args.provider and args.provider != "mock":
        from llm_provider import get_provider

        provider = get_provider(cfg)
        out.write(f"LLM edge validation: {provider.describe()}\n")
    else:
        out.write("Edge validation: deterministic heuristic\n")

    sector_graph.set_graphs_dir(cfg.storage_paths.graphs)

    start = args.start or pg.start_date
    end = args.end if args.end is not None else pg.end_date
    out.write(
        f"Building point-in-time graph for {portfolio}\n"
        f"  window start {start}  (end {end or f'start + {pg.window_days}d'}), "
        f"as-of metrics lookback {args.lookback_days or pg.lookback_days}d\n"
        f"  news: newsdata.io archive (live={use_live and cfg.newsdata.use_live})  |  "
        f"web_search: OFF (present-day look-ahead)\n"
    )

    try:
        built = build_asof_portfolio_graph(
            portfolio,
            start_date=start,
            end_date=end,
            window_days=args.window_days or pg.window_days,
            exchange=exchange,
            lookback_days=args.lookback_days or pg.lookback_days,
            correlation_threshold=pg.correlation_threshold,
            min_validations=args.min_validations or pg.min_validations,
            benchmark=cfg.backtesting.benchmark,
            use_live=use_live,
            newsdata=cfg.newsdata,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[graph error] {exc}", file=sys.stderr)
        return 1

    graph_id = str(built["graph_id"])
    if built.get("skipped_holdings"):
        out.write(f"Skipped non-graphable holdings: {built['skipped_holdings']}\n")
    out.write(
        f"\nGraph {graph_id}: {built['node_count']} nodes, "
        f"{built['edge_count']} candidate edges (as of {built['as_of']}).\n"
    )

    def sink(text: str) -> None:
        out.write(text)
        out.flush()

    out.write("Reasoning over candidate associations (live):\n")
    loop = run_reasoning_loop(graph_id, provider=provider, sink=sink)

    summary, driver = loop["graph"], loop["driver"]
    out.write(
        f"\n\nGraph {graph_id}  (validation driver: {driver})\n"
        f"  nodes: {summary['node_count']}  edges: {summary['edge_count']}  "
        f"status: {summary['edges_by_status']}\n"
    )
    persisted = Path(cfg.storage_paths.graphs) / f"{graph_id}.json"
    out.write(f"Persisted graph: {persisted}\n")

    if not args.no_visualize:
        viz = visualize_sector_graph(graph_id, out_dir=cfg.storage_paths.graphs)
        out.write(f"DOT written:     {viz['dot_path']}\n")
        if viz.get("image_path"):
            out.write(f"Image written:   {viz['image_path']}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
