"""Tests for the two-step, LLM-driven portfolio builder — mock data, no network.

fetch_sector_analytics returns raw metrics (no sizing); generate_final_portfolio
executes the weights the model would choose. compute_baseline_weights is the
deterministic offline fallback.
"""

from __future__ import annotations

import json

import pytest

from portfolio_builder import (
    FETCH_ANALYTICS_TOOL_SPECS,
    GENERATE_PORTFOLIO_TOOL_SPECS,
    PORTFOLIO_TOOL_SPECS,
    PortfolioBuildError,
    compute_baseline_weights,
    fetch_sector_analytics,
    generate_final_portfolio,
)
from portfolio_parser import load_portfolio


# --------------------------------------------------------------------------- #
# Tool 1: fetch_sector_analytics
# --------------------------------------------------------------------------- #
class TestFetchAnalytics:
    def test_returns_metrics_per_sector(self):
        r = fetch_sector_analytics(["it", "banking"], use_live=False)
        assert set(r["analytics"]) == {"it", "banking"}
        row = r["analytics"]["it"][0]
        for key in ("ticker", "price", "sharpe_ratio",
                    "annualized_volatility_pct", "trailing_pe"):
            assert key in row, key
        # It must not size anything — no weights/quantities here.
        assert "quantity" not in row and "weight" not in row

    def test_accepts_comma_string(self):
        r = fetch_sector_analytics("it, pharma", use_live=False)
        assert set(r["sectors"]) == {"it", "pharma"}

    def test_unknown_sector_raises(self):
        with pytest.raises(PortfolioBuildError):
            fetch_sector_analytics(["crypto"], use_live=False)

    def test_empty_raises(self):
        with pytest.raises(PortfolioBuildError):
            fetch_sector_analytics([], use_live=False)


# --------------------------------------------------------------------------- #
# Tool 2: generate_final_portfolio
# --------------------------------------------------------------------------- #
class TestGenerateFinalPortfolio:
    def test_literal_fractions_leave_cash(self):
        # Weights sum to 0.6 -> ~40% should remain as cash.
        r = generate_final_portfolio(
            {"TCS": 0.3, "HDFCBANK": 0.3}, 1_000_000, use_live=False, write_files=False
        )
        assert r["weights_normalized"] is False
        assert r["cash_remaining"] > 0
        assert r["total_invested"] <= 1_000_000
        # Deployed roughly 60% (minus share-rounding residual).
        assert 0.5 * 1_000_000 < r["total_invested"] <= 0.6 * 1_000_000

    def test_oversum_weights_are_normalized(self):
        r = generate_final_portfolio(
            {"TCS": 0.8, "HDFCBANK": 0.8}, 1_000_000, use_live=False, write_files=False
        )
        assert r["weights_normalized"] is True
        # After normalizing to sum 1, near-full deployment (minus rounding).
        assert r["total_invested"] > 0.9 * 1_000_000

    def test_negative_and_zero_weights_dropped(self):
        r = generate_final_portfolio(
            {"TCS": 0.5, "INFY": 0.0, "WIPRO": -0.2}, 1_000_000,
            use_live=False, write_files=False,
        )
        assert [h["ticker"] for h in r["holdings"]] == ["TCS"]

    def test_all_dropped_raises(self):
        with pytest.raises(PortfolioBuildError):
            generate_final_portfolio({"TCS": 0.0, "INFY": -1.0}, 1_000_000, use_live=False)

    def test_writes_csv_and_reasoning_with_rationale(self, tmp_path):
        r = generate_final_portfolio(
            {"TCS": 0.4, "HDFCBANK": 0.3, "SUNPHARMA": 0.3}, 1_000_000,
            reasoning="Dropped negative-Sharpe names; tilted to IT for momentum.",
            use_live=False, output_dir=str(tmp_path), name="p",
        )
        assert (tmp_path / "p.csv").exists()
        assert (tmp_path / "p.reasoning.md").exists()
        # The model's rationale is recorded in the reasoning file.
        assert "Dropped negative-Sharpe names" in r["reasoning"]
        assert "## Model rationale" in r["reasoning"]
        for h in r["holdings"]:
            assert h["ticker"] in r["reasoning"]

    def test_csv_is_loadable_as_portfolio(self, tmp_path):
        r = generate_final_portfolio(
            {"TCS": 0.5, "HDFCBANK": 0.5}, 500_000,
            use_live=False, output_dir=str(tmp_path), name="p",
        )
        pf = load_portfolio(r["csv_path"])
        assert len(pf) == r["holding_count"]
        assert pf.total_value > 0

    def test_shares_are_whole_and_rounded_down(self):
        r = generate_final_portfolio({"TCS": 1.0}, 1_000_000, use_live=False, write_files=False)
        h = r["holdings"][0]
        assert isinstance(h["quantity"], int)
        assert h["quantity"] * h["price"] <= 1_000_000  # never over-deploys

    def test_accepts_json_string_weights(self):
        r = generate_final_portfolio('{"TCS": 0.5}', 500_000, use_live=False, write_files=False)
        assert r["holdings"][0]["ticker"] == "TCS"

    def test_bad_weights_raise(self):
        with pytest.raises(PortfolioBuildError):
            generate_final_portfolio("not json", 1000, use_live=False)
        with pytest.raises(PortfolioBuildError):
            generate_final_portfolio({}, 1000, use_live=False)

    def test_nonpositive_amount_raises(self):
        with pytest.raises(PortfolioBuildError):
            generate_final_portfolio({"TCS": 1.0}, 0, use_live=False)


# --------------------------------------------------------------------------- #
# Deterministic baseline fallback
# --------------------------------------------------------------------------- #
class TestBaselineWeights:
    def test_weights_sum_to_about_one(self):
        w = compute_baseline_weights(["it", "banking", "pharma"], use_live=False)
        assert sum(w.values()) == pytest.approx(1.0, abs=0.02)
        assert all(v > 0 for v in w.values())

    def test_one_per_sector_by_default(self):
        w = compute_baseline_weights(["it", "banking", "pharma"], use_live=False)
        assert len(w) == 3

    def test_bad_risk_profile_raises(self):
        with pytest.raises(PortfolioBuildError):
            compute_baseline_weights(risk_profile="yolo", use_live=False)

    def test_baseline_feeds_generate(self):
        w = compute_baseline_weights(["it", "banking"], use_live=False)
        r = generate_final_portfolio(w, 1_000_000, use_live=False, write_files=False)
        assert r["holding_count"] >= 2


# --------------------------------------------------------------------------- #
# Tool specs + in-process dispatch
# --------------------------------------------------------------------------- #
class TestToolIntegration:
    def test_tool_specs(self):
        names = {s["function"]["name"] for s in PORTFOLIO_TOOL_SPECS}
        assert names == {"fetch_sector_analytics", "generate_final_portfolio"}
        assert len(FETCH_ANALYTICS_TOOL_SPECS) == 1
        assert len(GENERATE_PORTFOLIO_TOOL_SPECS) == 1

    def test_dispatch_fetch(self):
        from tool_runtime import InProcessToolExecutor

        ex = InProcessToolExecutor(use_live=False)
        out = json.loads(ex("fetch_sector_analytics", '{"sectors": ["it"]}'))
        assert "analytics" in out and out["analytics"]["it"]

    def test_dispatch_generate(self, tmp_path):
        from tool_runtime import InProcessToolExecutor

        ex = InProcessToolExecutor(use_live=False, portfolios_dir=str(tmp_path))
        out = json.loads(ex(
            "generate_final_portfolio",
            '{"ticker_weights": {"TCS": 0.5, "HDFCBANK": 0.5}, '
            '"total_amount": 500000, "reasoning": "balanced two-name test"}',
        ))
        assert out["holding_count"] == 2
        assert (tmp_path / "diversified_portfolio.csv").exists()

    def test_dispatch_surfaces_error(self, tmp_path):
        from tool_runtime import InProcessToolExecutor

        ex = InProcessToolExecutor(use_live=False, portfolios_dir=str(tmp_path))
        out = json.loads(ex("fetch_sector_analytics", '{"sectors": ["nope"]}'))
        assert "error" in out
