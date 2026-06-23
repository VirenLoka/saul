"""Unit tests for the decoupled portfolio ingestion module."""

from __future__ import annotations

import pytest

from portfolio_parser import PortfolioParseError, load_portfolio

SAMPLE = "knowledge/portfolios/sample_portfolio.csv"


def test_loads_sample_portfolio():
    pf = load_portfolio(SAMPLE)
    assert len(pf) >= 1
    assert pf.total_value > 0
    by_class = pf.value_by_asset_class()
    assert "Equity" in by_class
    assert pf.tickers()  # non-empty


def test_header_aliases_and_currency_parsing(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text(
        "Symbol,Class,Qty,Market Value\n"
        'RELIANCE,Equity,10,"₹1,250.50"\n',
        encoding="utf-8",
    )
    pf = load_portfolio(p)
    assert pf.holdings[0].ticker == "RELIANCE"
    assert pf.holdings[0].current_value == pytest.approx(1250.50)


def test_missing_column_raises(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("Ticker,Quantity,Current Value\nTCS,1,100\n", encoding="utf-8")
    with pytest.raises(PortfolioParseError):
        load_portfolio(p)


def test_bad_number_raises(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text(
        "Ticker,Asset Class,Quantity,Current Value\nTCS,Equity,ten,100\n",
        encoding="utf-8",
    )
    with pytest.raises(PortfolioParseError):
        load_portfolio(p)


def test_blank_lines_skipped(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text(
        "Ticker,Asset Class,Quantity,Current Value\n"
        "TCS,Equity,1,100\n\nINFY,Equity,2,200\n",
        encoding="utf-8",
    )
    assert len(load_portfolio(p)) == 2


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_portfolio("does/not/exist.csv")
