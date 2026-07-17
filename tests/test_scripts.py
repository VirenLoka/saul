"""Tests for the two standalone scripts — offline (mock provider/data), no GPU.

Both scripts are driven through their ``main(argv, out)`` entry points against a
temp config whose storage paths point at ``tmp_path``, so nothing touches the
repo and no network/model is used.
"""

from __future__ import annotations

import io

import pytest
import yaml

import generate_portfolio
import run_graph_reasoning


def _temp_config(tmp_path):
    """Copy the repo config, redirect storage to tmp, force all sources offline."""
    with open("config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["storage_paths"]["graphs"] = str(tmp_path / "graphs")
    cfg["storage_paths"]["portfolios"] = str(tmp_path / "portfolios")
    cfg["mcp"]["market_data"]["use_live"] = False
    cfg["newsapi"]["use_live"] = False
    cfg["search"]["use_live"] = False
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# Script 1: run_graph_reasoning.py
# --------------------------------------------------------------------------- #
class TestGraphReasoningScript:
    def test_sector_graph_persists(self, tmp_path):
        cfg = _temp_config(tmp_path)
        out = io.StringIO()
        rc = run_graph_reasoning.main(
            ["--config", cfg, "--sectors", "it", "--mock", "--no-visualize"], out=out
        )
        assert rc == 0
        assert list((tmp_path / "graphs").glob("*.json"))  # persisted at config path
        assert "validated" in out.getvalue()

    def test_visualization_writes_dot(self, tmp_path):
        cfg = _temp_config(tmp_path)
        out = io.StringIO()
        rc = run_graph_reasoning.main(
            ["--config", cfg, "--sectors", "auto", "--mock"], out=out
        )
        assert rc == 0
        assert list((tmp_path / "graphs").glob("*.dot"))  # DOT always written

    def test_from_portfolio(self, tmp_path):
        cfg = _temp_config(tmp_path)
        out = io.StringIO()
        rc = run_graph_reasoning.main(
            ["--config", cfg, "--portfolio",
             "knowledge/portfolios/banking_portfolio.csv", "--mock", "--no-visualize"],
            out=out,
        )
        assert rc == 0
        assert list((tmp_path / "graphs").glob("*.json"))


# --------------------------------------------------------------------------- #
# Script 2: generate_portfolio.py
# --------------------------------------------------------------------------- #
class TestGeneratePortfolioScript:
    def test_produces_csv_and_reasoning_via_fallback(self, tmp_path):
        # The mock provider never calls the tool -> the script's deterministic
        # fallback still builds the portfolio and writes both files.
        cfg = _temp_config(tmp_path)
        out = io.StringIO()
        rc = generate_portfolio.main(
            ["--config", cfg, "--provider", "mock", "--amount", "500000",
             "--risk", "conservative", "--mock", "--max-rounds", "2"],
            out=out,
        )
        assert rc == 0
        assert (tmp_path / "portfolios" / "diversified_portfolio.csv").exists()
        assert (tmp_path / "portfolios" / "diversified_portfolio.reasoning.md").exists()
        assert "Portfolio" in out.getvalue()

    def test_csv_output_is_loadable(self, tmp_path):
        from portfolio_parser import load_portfolio

        cfg = _temp_config(tmp_path)
        generate_portfolio.main(
            ["--config", cfg, "--provider", "mock", "--mock", "--max-rounds", "1"],
            out=io.StringIO(),
        )
        pf = load_portfolio(str(tmp_path / "portfolios" / "diversified_portfolio.csv"))
        assert len(pf) >= 4
        assert pf.total_value > 0
