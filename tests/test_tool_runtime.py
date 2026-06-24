"""Unit tests for the tool execution runtime (offline, no fastmcp / no network)."""

from __future__ import annotations

import json

from tool_runtime import InProcessToolExecutor, parse_args


def test_parse_args_variants():
    assert parse_args('{"query": "TCS"}') == {"query": "TCS"}
    assert parse_args({"query": "TCS"}) == {"query": "TCS"}
    assert parse_args("") == {}
    assert parse_args(None) == {}


def test_in_process_executes_stock_quote():
    ex = InProcessToolExecutor(use_live=False)
    out = json.loads(ex("get_indian_stock_quote", '{"query": "Reliance"}'))
    assert out["symbol"] == "RELIANCE"
    assert out["yahoo_symbol"] == "RELIANCE.NS"
    assert out["source"] == "mock"


def test_in_process_executes_sector():
    ex = InProcessToolExecutor(use_live=False)
    out = json.loads(ex("get_indian_sector_performance", '{"sector": "IT"}'))
    assert out["sector"] == "it"
    assert "avg_change_pct" in out


def test_in_process_unknown_tool_returns_error():
    ex = InProcessToolExecutor(use_live=False)
    out = json.loads(ex("nope", "{}"))
    assert "error" in out


def test_in_process_bad_arguments_returns_error():
    ex = InProcessToolExecutor(use_live=False)
    out = json.loads(ex("get_indian_stock_quote", "{not json"))
    assert "error" in out


def test_in_process_market_data_error_is_caught():
    ex = InProcessToolExecutor(use_live=False)
    out = json.loads(ex("get_indian_sector_performance", '{"sector": "crypto"}'))
    assert "error" in out  # unknown sector surfaced as a readable error, not a crash
