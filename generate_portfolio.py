#!/usr/bin/env python3
"""Standalone: the LLM builds a diversified portfolio in two steps, outside cli.py.

The reasoning model has the FINAL say on the allocation:

  1. It calls ``fetch_sector_analytics`` to get raw metrics (volatility, P/E,
     Sharpe, return, price) for candidate stocks — and may call the live-quote,
     news, statistics, or ``web_search`` tools on demand for extra context.
  2. It reasons over those metrics (dropping / shrinking negative-Sharpe names,
     tilting toward risk-adjusted return) and chooses target ``ticker_weights``.
  3. It calls ``generate_final_portfolio`` with its weights + rationale; Python
     rounds to whole shares and writes the CSV + reasoning file.

All model output is streamed live. If no reasoning model produces weights (e.g.
the offline ``mock`` provider), the script falls back to a deterministic
baseline weighting so it still yields output — in real use the model's weights
are authoritative.

Examples
--------
    python generate_portfolio.py --provider groq --amount 1000000 --risk balanced
    python generate_portfolio.py --provider mock --mock            # offline fallback
    python generate_portfolio.py --sectors it,banking,pharma --provider groq
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Dict, List, Optional, TextIO

from config_loader import ConfigError, load_config
from llm_provider import LLMProviderError, get_provider
from market_data import TOOL_SPECS
from news_data import NEWS_TOOL_SPECS
from portfolio_builder import (
    PORTFOLIO_TOOL_SPECS,
    compute_baseline_weights,
    generate_final_portfolio,
)
from stock_stats import STATS_TOOL_SPECS
from tool_runtime import InProcessToolExecutor
from web_search import SEARCH_TOOL_SPECS

logger = logging.getLogger("saul.scripts.portfolio")

# The model may fetch live data / news / stats / web-search on demand, then use
# the two portfolio tools. (Graph/viz tools are left out to keep the request
# small for tight rate-limit tiers.)
_TOOLS = (
    TOOL_SPECS + NEWS_TOOL_SPECS + STATS_TOOL_SPECS + SEARCH_TOOL_SPECS
    + PORTFOLIO_TOOL_SPECS
)

_SYSTEM = (
    "You are a portfolio-construction assistant for Indian equities operating in "
    "a strictly READ-ONLY capacity (no trades). Build a diversified portfolio in "
    "two steps and YOU make the allocation decision:\n"
    "1. Call fetch_sector_analytics(sectors) to get volatility, P/E, Sharpe, "
    "return and price for candidate stocks. You MAY also call "
    "get_indian_stock_quote, get_stock_news, the statistics tools, or web_search "
    "for extra context.\n"
    "2. Decide target weights per ticker yourself: DROP or shrink any name with a "
    "negative Sharpe ratio, tilt toward better risk-adjusted return, and keep the "
    "book diversified across sectors. Weights are fractions of capital and may sum "
    "to less than 1 (the remainder is held as cash).\n"
    "Then call generate_final_portfolio(ticker_weights, total_amount, reasoning) "
    "with your chosen weights and a clear rationale. Finally give a short summary "
    "and end with a one-line not-financial-advice disclaimer."
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default='groq',
                   choices=["vllm", "groq", "mock"],
                   help="Override model_selection.provider.")
    p.add_argument("--amount", type=float, default=1_300_000.0,
                   help="Total capital to allocate, INR (default 1,000,000).")
    p.add_argument("--risk", default="balanced",
                   choices=["conservative", "balanced", "aggressive"],
                   help="Risk hint passed to the model / baseline fallback (default balanced).")
    p.add_argument("--sectors", default=None,
                   help="Comma-separated sectors to steer toward (default: a broad set).")
    p.add_argument("--mock", action="store_true",
                   help="Use deterministic mock market data for the tools (offline).")
    p.add_argument("--max-rounds", type=int, default=6,
                   help="Max model tool-calling rounds before falling back (default 6).")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument("-v", "--verbose", action="store_true", help="DEBUG logs to stderr.")
    return p.parse_args(argv)


def _run_model(provider, messages, tools, out) -> tuple:
    """Stream one model turn; return (tool_calls, content_text, fatal)."""
    calls: List[Dict[str, str]] = []
    parts: List[str] = []
    fatal = False
    for ev in provider.stream_chat(messages, tools=tools):
        if ev.type == "tool_call":
            out.write(f"\n🔧 {ev.name}({ev.arguments})\n")
            out.flush()
            calls.append({"name": ev.name, "arguments": ev.arguments})
        elif ev.type in ("content", "reasoning"):
            if ev.type == "content":
                parts.append(ev.text)
            out.write(ev.text)
            out.flush()
        elif ev.type == "error":
            fatal = fatal or getattr(ev, "fatal", False)
            out.write(f"\n[model error] {ev.text}\n")
        elif ev.type == "done":
            break
    return calls, "".join(parts), fatal


def _summarize(out, payload: Dict[str, object]) -> None:
    if "error" in payload:
        out.write(f"\n[tool error] {payload['error']}\n")
        return
    out.write(
        f"\n✅ Portfolio '{payload.get('name')}' — {payload.get('holding_count')} holdings\n"
        f"   invested ₹{payload.get('total_invested'):,.0f} "
        f"(cash ₹{payload.get('cash_remaining'):,.0f}), "
        f"avg correlation {payload.get('avg_correlation')}, "
        f"normalized={payload.get('weights_normalized')}\n"
        f"   CSV:       {payload.get('csv_path')}\n"
        f"   reasoning: {payload.get('reasoning_path')}\n"
    )


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
        provider = get_provider(cfg)
    except (ConfigError, LLMProviderError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2

    use_live = (not args.mock) and cfg.mcp.market_data.use_live
    sectors = [s.strip() for s in args.sectors.split(",")] if args.sectors else None

    executor = InProcessToolExecutor.from_settings(
        cfg.mcp.market_data,
        use_live=use_live,
        newsapi=cfg.newsapi,
        search=cfg.search,
        graphs_dir=cfg.storage_paths.graphs,
        portfolios_dir=cfg.storage_paths.portfolios,
    )

    ask = (
        f"Build a diversified Indian-equity portfolio of about "
        f"₹{args.amount:,.0f} with a {args.risk} risk profile"
        + (f", focused on these sectors: {args.sectors}." if sectors else ".")
        + " Start by calling fetch_sector_analytics, decide the weights yourself, "
        "then call generate_final_portfolio."
    )
    messages: List[Dict[str, object]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": ask},
    ]

    out.write(f"Provider: {provider.describe()}  |  live data: {use_live}\n")
    out.write("Model is constructing the portfolio (two-step, streaming)…\n")

    tool_payload: Optional[Dict[str, object]] = None
    tool_seq = 0
    for _round in range(max(1, args.max_rounds)):
        calls, _content, fatal = _run_model(provider, messages, _TOOLS, out)
        if fatal:
            out.write("\n(Model request failed; falling back to a direct build.)\n")
            break
        if not calls:
            break  # model produced a plain answer with no tool call

        norm = []
        for c in calls:
            tool_seq += 1
            norm.append({"id": f"call_{tool_seq}", **c})
        messages.append({
            "role": "assistant", "content": None,
            "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in norm
            ],
        })
        for c in norm:
            result = executor(c["name"], c["arguments"])
            messages.append({"role": "tool", "tool_call_id": c["id"],
                             "name": c["name"], "content": result})
            if c["name"] == "generate_final_portfolio":
                parsed = json.loads(result)
                if "error" not in parsed:
                    tool_payload = parsed

        if tool_payload is not None:
            _run_model(provider, messages, None, out)  # closing summary, then stop
            break

    # Fallback: the model never produced a portfolio -> deterministic baseline.
    if tool_payload is None:
        logger.warning("Model did not build a portfolio; using deterministic baseline.")
        out.write("\nModel did not finalize a portfolio — using a deterministic "
                  "baseline weighting…\n")
        weights = compute_baseline_weights(
            sectors, exchange=cfg.mcp.market_data.default_exchange,
            risk_profile=args.risk, use_live=use_live,
        )
        tool_payload = generate_final_portfolio(
            weights, args.amount,
            exchange=cfg.mcp.market_data.default_exchange,
            reasoning=(
                f"Deterministic baseline ({args.risk} risk): top risk-adjusted "
                "pick per sector, inverse-volatility weighted. No reasoning model "
                "produced weights for this run."
            ),
            use_live=use_live,
            output_dir=cfg.storage_paths.portfolios,
        )

    _summarize(out, tool_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
