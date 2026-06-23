"""Unit tests for deterministic portfolio analytics."""

from __future__ import annotations

import pytest

from analysis import analyze_portfolio
from config_loader import AnalysisSettings
from portfolio_parser import Holding, Portfolio

SETTINGS = AnalysisSettings(
    target_allocation={"Equity": 60, "Bond": 25, "Cash": 5, "Commodity": 5,
                       "Real Estate": 5},
    drift_tolerance_pct=10,
)


def _pf(holdings):
    return Portfolio(holdings=holdings, source="test")


def test_allocation_percentages_sum_to_100():
    pf = _pf(
        [
            Holding("AAPL", "Equity", 1, 60.0),
            Holding("BND", "Bond", 1, 25.0),
            Holding("CASH", "Cash", 1, 15.0),
        ]
    )
    res = analyze_portfolio(pf, SETTINGS)
    assert sum(ln.pct for ln in res.lines) == pytest.approx(100.0)


def test_overweight_and_underweight_flags():
    # 90% equity (overweight vs 60+10 tol), 10% bond (underweight vs 25-10).
    pf = _pf(
        [
            Holding("AAPL", "Equity", 1, 90.0),
            Holding("BND", "Bond", 1, 10.0),
        ]
    )
    res = analyze_portfolio(pf, SETTINGS)
    status = {ln.asset_class: ln.status for ln in res.lines}
    assert status["Equity"] == "overweight"
    assert status["Bond"] == "underweight"


def test_untracked_asset_class():
    pf = _pf([Holding("BTC", "Crypto", 1, 100.0)])
    res = analyze_portfolio(pf, SETTINGS)
    assert res.lines[0].status == "untracked"


def test_missing_target_classes_reported():
    pf = _pf([Holding("AAPL", "Equity", 1, 100.0)])
    res = analyze_portfolio(pf, SETTINGS)
    assert set(res.missing_classes) == {"Bond", "Cash", "Commodity", "Real Estate"}


def test_top_holding_concentration():
    pf = _pf(
        [
            Holding("AAPL", "Equity", 1, 80.0),
            Holding("BND", "Bond", 1, 20.0),
        ]
    )
    res = analyze_portfolio(pf, SETTINGS)
    assert res.top_holding_ticker == "AAPL"
    assert res.top_holding_pct == pytest.approx(80.0)
