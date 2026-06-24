"""Orchestration entrypoint for the Financial Advisor AI Agent.

Wires the decoupled modules together into the MVP analytical loop:

    config -> ingest portfolio CSV -> deterministic analysis ->
    build prompts -> LLM narrative -> terminal report

The agent is strictly read-only: it observes portfolio data and reports
conclusions. It never executes trades or any financial action.

Usage
-----
    python main.py                          # uses defaults from config.yaml
    python main.py --portfolio path.csv     # analyze a specific CSV
    python main.py --provider mock          # force offline stub (no model run)
    python main.py --config alt_config.yaml # use an alternate config file
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

from analysis import analyze_portfolio
from config_loader import ConfigError, load_config
from data_ingestion import PortfolioParseError, load_portfolio
from llm_provider import LLMProviderError, get_provider
from prompts import SYSTEM_PROMPT, build_user_prompt
from report import render_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Financial Advisor AI Agent (portfolio analysis)."
    )
    parser.add_argument(
        "--config", default=None, help="Path to config.yaml (default: bundled)."
    )
    parser.add_argument(
        "--portfolio",
        default=None,
        help="Path to a portfolio CSV (default: storage_paths.default_portfolio).",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=["local", "openai", "anthropic", "mock"],
        help="Override model_selection.provider for this run.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # 1. Load configuration.
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    # Allow a per-run provider override without mutating the file.
    if args.provider:
        config = replace(
            config,
            model_selection=replace(
                config.model_selection, provider=args.provider
            ),
        )

    # 2. Ingest the portfolio CSV.
    portfolio_path = args.portfolio or config.storage_paths.default_portfolio
    try:
        portfolio = load_portfolio(portfolio_path)
    except (FileNotFoundError, PortfolioParseError) as exc:
        print(f"[ingestion error] {exc}", file=sys.stderr)
        return 3

    # 3. Deterministic analysis (no LLM).
    result = analyze_portfolio(portfolio, config.analysis)

    # 4. Build prompts and call the configured LLM backend.
    system_prompt = SYSTEM_PROMPT
    user_prompt = build_user_prompt(result.as_summary_dict())
    try:
        provider = get_provider(config)
        narrative = provider.generate(system_prompt, user_prompt)
        provider_name = provider.describe()
    except LLMProviderError as exc:
        # Degrade gracefully: still print the quantitative report.
        narrative = (
            f"[LLM unavailable: {exc}]\n"
            "Quantitative analysis above was produced without the model."
        )
        provider_name = f"{config.model_selection.provider} (unavailable)"

    # 5. Render the terminal report.
    report = render_report(
        result=result,
        source=portfolio.source,
        provider_name=provider_name,
        llm_narrative=narrative,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
