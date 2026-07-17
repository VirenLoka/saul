#!/usr/bin/env python3
"""Standalone runner: build a knowledge graph, run the reason/reflect loop, persist + visualize.

This drives the graph-building reasoning loop end-to-end *outside* the chat CLI:

  1. Build a graph over sectors (``--sectors``) or a portfolio's holdings
     (``--portfolio``) — nodes carry alpha/indicator/fundamental/sentiment/
     filings features, edges are candidate associations with evidence.
  2. Run the validation loop that confirms/reflects on each edge over multiple
     passes (deterministic heuristic by default; ``--llm`` uses the configured
     provider to drive it).
  3. **Persist** the graph as JSON at the configured path
     (``storage_paths.graphs``) so it can be queried later, and render a
     **Graphviz DOT** (+ image if ``dot`` is installed) via ``graph_viz``.

Everything runs offline with ``--mock`` (deterministic data, no network/GPU).

Examples
--------
    python run_graph_reasoning.py --sectors it,banking --mock
    python run_graph_reasoning.py --portfolio knowledge/portfolios/banking_portfolio.csv
    python run_graph_reasoning.py --sectors it,banking --llm --provider groq
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, TextIO

import sector_graph
from config_loader import ConfigError, load_config
from graph_agent import build_portfolio_graph, run_reasoning_loop
from graph_viz import visualize_sector_graph
from sector_graph import NewsSentimentProvider, build_sector_graph

logger = logging.getLogger("saul.scripts.graph")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--sectors", default="it,banking",
                     help="Comma-separated sectors to graph (default: it,banking).")
    src.add_argument("--portfolio", default=None,
                     help="Build the graph from a portfolio CSV's holdings instead of sectors.")
    p.add_argument("--exchange", default=None, help="NS or BO (default: configured).")
    p.add_argument("--min-validations", type=int, default=2,
                   help="Confirming passes required to accept an edge (default 2).")
    p.add_argument("--llm", action="store_true",
                   help="Drive edge validation with the configured LLM (needs network/API).")
    p.add_argument("--provider", default='groq',
                   choices=["vllm", "groq", "mock"],
                   help="Override model_selection.provider (used with --llm).")
    p.add_argument("--mock", action="store_true",
                   help="Use deterministic mock market data (offline).")
    p.add_argument("--sentiment", dest="sentiment", action="store_true", default=False,
                   help="Include the news-sentiment node feature (excluded by default; "
                        "sentiment is computed separately later).")
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

    # Persist graphs at the configured path.
    sector_graph.set_graphs_dir(cfg.storage_paths.graphs)
    exchange = (args.exchange or cfg.mcp.market_data.default_exchange).upper()
    use_live = (not args.mock) and cfg.mcp.market_data.use_live
    sentiment = NewsSentimentProvider(cfg.newsapi)

    provider = None
    if args.llm:
        from llm_provider import get_provider

        provider = get_provider(cfg)
        out.write(f"Using LLM-driven validation via {provider.describe()}.\n")

    # Stream every model/decision token to stdout as the graph is validated.
    def sink(text: str) -> None:
        out.write(text)
        out.flush()

    include_sentiment = bool(args.sentiment)
    out.write(f"Sentiment node feature: {'included' if include_sentiment else 'EXCLUDED (deferred)'}\n")

    try:
        if args.portfolio:
            out.write(f"Building portfolio graph from {args.portfolio} …\n")
            out.write("Reasoning over candidate edges (live):\n")
            result = build_portfolio_graph(
                args.portfolio, exchange,
                min_validations=args.min_validations, use_live=use_live,
                provider=provider, sentiment_provider=sentiment,
                include_sentiment=include_sentiment, sink=sink,
            )
            graph_id = str(result["graph_id"])
            driver, summary = result["driver"], result["graph"]
            if result.get("skipped_holdings"):
                out.write(f"\nSkipped non-graphable holdings: {result['skipped_holdings']}\n")
        else:
            sectors = [s.strip() for s in args.sectors.split(",") if s.strip()]
            out.write(f"Building sector graph for {sectors} …\n")
            built = build_sector_graph(
                sectors, exchange,
                min_validations=args.min_validations, use_live=use_live,
                sentiment_provider=sentiment, include_sentiment=include_sentiment,
            )
            graph_id = str(built["graph_id"])
            out.write("Reasoning over candidate edges (live):\n")
            loop = run_reasoning_loop(graph_id, provider=provider, sink=sink)
            driver, summary = loop["driver"], loop["graph"]
    except Exception as exc:  # noqa: BLE001
        print(f"[graph error] {exc}", file=sys.stderr)
        return 1

    out.write(f"\n\nGraph {graph_id}  (validation driver: {driver})\n")
    out.write(f"  nodes: {summary['node_count']}  edges: {summary['edge_count']}  "
              f"status: {summary['edges_by_status']}\n")

    persisted = Path(cfg.storage_paths.graphs) / f"{graph_id}.json"
    out.write(f"\nPersisted graph: {persisted}\n")

    if not args.no_visualize:
        viz = visualize_sector_graph(graph_id, out_dir=cfg.storage_paths.graphs)
        out.write(f"DOT written:     {viz['dot_path']}\n")
        if viz.get("image_path"):
            out.write(f"Image written:   {viz['image_path']}\n")
        else:
            out.write("Image skipped (Graphviz 'dot' binary not found on PATH).\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
