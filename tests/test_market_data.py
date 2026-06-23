"""Unit tests for the Indian market data core.

All tests run in mock mode (use_live=False) — no network / yfinance forward
pass. One test monkeypatches the live path to confirm graceful fallback.
"""

from __future__ import annotations

import pytest

import market_data
from market_data import (
    MarketDataError,
    TOOL_SPECS,
    clear_cache,
    get_sector_performance,
    get_stock_quote,
    resolve_symbol,
)


def setup_function(_):
    clear_cache()


def test_resolve_friendly_name_and_suffix():
    base, yahoo = resolve_symbol("Reliance", "NS")
    assert base == "RELIANCE"
    assert yahoo == "RELIANCE.NS"


def test_resolve_bse_exchange():
    base, yahoo = resolve_symbol("TCS", "BO")
    assert yahoo == "TCS.BO"


def test_resolve_already_suffixed():
    base, yahoo = resolve_symbol("INFY.NS", "BO")
    assert base == "INFY"
    assert yahoo == "INFY.NS"  # explicit suffix wins


def test_invalid_exchange_raises():
    with pytest.raises(MarketDataError):
        resolve_symbol("TCS", "XX")


def test_mock_quote_is_deterministic():
    q1 = get_stock_quote("TCS", "NS", use_live=False)
    clear_cache()
    q2 = get_stock_quote("TCS", "NS", use_live=False)
    assert q1 == q2
    assert q1["source"] == "mock"
    assert q1["currency"] == "INR"
    assert q1["yahoo_symbol"] == "TCS.NS"


def test_sector_performance_aggregates():
    perf = get_sector_performance("IT", use_live=False)
    assert perf["sector"] == "it"
    assert len(perf["constituents"]) == 4
    assert "avg_change_pct" in perf
    assert perf["advancers"] + perf["decliners"] <= len(perf["constituents"])


def test_unknown_sector_raises():
    with pytest.raises(MarketDataError):
        get_sector_performance("crypto", use_live=False)


def test_live_failure_falls_back_to_mock(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(market_data, "_live_quote", boom)
    q = get_stock_quote("Reliance", "NS", use_live=True)
    assert q["source"] == "mock"  # degraded gracefully


def test_tool_specs_shape():
    names = {t["function"]["name"] for t in TOOL_SPECS}
    assert names == {"get_indian_stock_quote", "get_indian_sector_performance"}
    for t in TOOL_SPECS:
        assert t["type"] == "function"
        assert "parameters" in t["function"]
