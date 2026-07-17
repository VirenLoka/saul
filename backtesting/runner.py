#!/usr/bin/env python3
"""Standalone backtest runner: periodic-rebalancing simulation of the portfolio flow.

Builds (or reuses) a knowledge graph for context — with the **sentiment feature
excluded** (deferred) — then runs the backtest: at each rebalance the LLM (or the
deterministic baseline offline) chooses weights from point-in-time analytics +
newsdata.io archive news + graph peers, sized at that date's prices and marked to
market vs the benchmark. **web_search is disabled** and the window is clamped to
the newsdata training-cutoff floor.

Everything is configurable via ``config.yaml`` (``backtesting`` / ``newsdata``);
flags override. Runs fully offline with ``--mock`` (no model / GPU / network).

Examples
--------
    python -m backtesting.runner --mock                       # offline baseline
    python -m backtesting.runner --provider groq              # LLM weight engine
    python -m backtesting.runner --graph-id 3ebe3042b963 --provider groq
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional, TextIO

import sector_graph
from backtesting.engine import BacktestError, run_backtest
from config_loader import ConfigError, load_config


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default='groq',
                   choices=["vllm", "groq", "mock"],
                   help="Provider for the LLM weight engine (omit / mock -> baseline).")
    p.add_argument("--graph-id", default='2b16ae9dcbff',
                   help="Reuse an existing graph for context (else one is built).")
    p.add_argument("--start", default=None, help="Window start YYYY-MM-DD (>= newsdata floor).")
    p.add_argument("--end", default=None, help="Window end YYYY-MM-DD (default: today).")
    p.add_argument("--rebalance", default=None, choices=["weekly", "monthly", "quarterly"])
    p.add_argument("--capital", type=float, default=None, help="Initial capital, INR.")
    p.add_argument("--sectors", default=None, help="Comma-separated sectors override.")
    p.add_argument("--no-graph", action="store_true", help="Skip graph context entirely.")
    p.add_argument("--mock", action="store_true", help="Deterministic offline data (no network).")
    p.add_argument("--name", default="backtest", help="Results file prefix.")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logs to stderr.")
    return p.parse_args(argv)


def _apply_overrides(raw: dict, args: argparse.Namespace) -> dict:
    bt = dict(raw.get("backtesting", {}) or {})
    if args.start:
        bt["start_date"] = args.start
    if args.end:
        bt["end_date"] = args.end
    if args.rebalance:
        bt["rebalance"] = args.rebalance
    if args.capital:
        bt["initial_capital"] = args.capital
    if args.sectors:
        bt["sectors"] = [s.strip().lower() for s in args.sectors.split(",") if s.strip()]
    if args.no_graph:
        bt["use_graph"] = False
    raw = dict(raw)
    raw["backtesting"] = bt
    return raw


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

    # Apply CLI overrides by re-loading a patched raw config (keeps it config-driven).
    if any([args.start, args.end, args.rebalance, args.capital, args.sectors, args.no_graph]):
        import tempfile, os, yaml  # local import; only when overriding

        patched = _apply_overrides(cfg.raw, args)
        fd, path = tempfile.mkstemp(suffix=".yaml")
        with os.fdopen(fd, "w") as fh:
            yaml.safe_dump(patched, fh)
        try:
            cfg = load_config(path, provider_override=args.provider)
        finally:
            os.unlink(path)

    use_live = (not args.mock) and cfg.mcp.market_data.use_live

    # LLM weight engine only for a real provider (mock -> deterministic baseline).
    provider = None
    if args.provider and args.provider != "mock":
        from llm_provider import get_provider

        provider = get_provider(cfg)
        out.write(f"LLM weight engine: {provider.describe()}\n")
    else:
        out.write("Weight engine: deterministic baseline\n")

    sector_graph.set_graphs_dir(cfg.storage_paths.graphs)

    # Build a context graph (sentiment EXCLUDED) unless one was supplied or disabled.
    graph_id = args.graph_id
    if cfg.backtesting.use_graph and not graph_id:
        from graph_agent import run_reasoning_loop
        from sector_graph import NewsSentimentProvider, build_sector_graph

        out.write("Building context graph (sentiment excluded)…\n")
        built = build_sector_graph(
            cfg.backtesting.sectors, cfg.mcp.market_data.default_exchange,
            use_live=use_live, sentiment_provider=NewsSentimentProvider(cfg.newsapi),
            include_sentiment=False,
        )
        graph_id = str(built["graph_id"])
        run_reasoning_loop(graph_id)  # heuristic validation -> peer associations
        out.write(f"Context graph: {graph_id}\n")

    def sink(text: str) -> None:
        out.write(text)
        out.flush()

    out.write("Running backtest…\n")
    try:
        result = run_backtest(
            cfg, provider=provider, graph_id=graph_id,
            use_live=use_live, name=args.name, sink=sink,
        )
    except BacktestError as exc:
        print(f"[backtest error] {exc}", file=sys.stderr)
        return 1

    p, b = result["portfolio"], result["benchmark"]
    out.write(
        f"\n\n=== Backtest complete ({result['engine']}) ===\n"
        f"  window {result['window']['start']} → {result['window']['end']} "
        f"({result['rebalance']}, {result['rebalance_count']} rebalances)\n"
        f"  graph context: {result['graph_id']}  |  web_search: "
        f"{'on' if result['web_search_enabled'] else 'OFF'}\n"
        f"  portfolio: total {p.get('total_return_pct')}%  CAGR {p.get('cagr_pct')}%  "
        f"Sharpe {p.get('sharpe_ratio')}  maxDD {p.get('max_drawdown_pct')}%\n"
        f"  benchmark: total {b.get('total_return_pct')}%  CAGR {b.get('cagr_pct')}%  "
        f"Sharpe {b.get('sharpe_ratio')}\n"
        f"  equity curve: {result['equity_curve_csv']}\n"
        f"  report:       {result['report_md']}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
