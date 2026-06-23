"""Unit tests for the decoupled data ingestion module."""

from __future__ import annotations

import pytest

from data_ingestion import PortfolioParseError, load_portfolio

SAMPLE = "knowledge/portfolios/sample_portfolio.csv"


def test_loads_sample_portfolio():
    pf = load_portfolio(SAMPLE)
    assert len(pf) == 9
    # 10500+12600+24000+9800+7200+4500+5500+2550+3000
    assert pf.total_value == pytest.approx(79650.0)
    by_class = pf.value_by_asset_class()
    assert by_class["Equity"] == pytest.approx(56900.0)
    assert by_class["Bond"] == pytest.approx(11700.0)
    assert set(by_class) == {"Equity", "Bond", "Commodity", "Real Estate", "Cash"}


def test_header_aliases_and_currency_parsing(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text(
        "Symbol,Class,Qty,Market Value\n"
        "aapl,Equity,10,\"$1,250.50\"\n",
        encoding="utf-8",
    )
    pf = load_portfolio(p)
    assert pf.holdings[0].ticker == "AAPL"  # uppercased
    assert pf.holdings[0].current_value == pytest.approx(1250.50)


def test_missing_column_raises(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("Ticker,Quantity,Current Value\nAAPL,1,100\n", encoding="utf-8")
    with pytest.raises(PortfolioParseError):
        load_portfolio(p)


def test_bad_number_raises(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text(
        "Ticker,Asset Class,Quantity,Current Value\nAAPL,Equity,ten,100\n",
        encoding="utf-8",
    )
    with pytest.raises(PortfolioParseError):
        load_portfolio(p)


def test_blank_lines_skipped(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text(
        "Ticker,Asset Class,Quantity,Current Value\n"
        "AAPL,Equity,1,100\n"
        "\n"
        "MSFT,Equity,2,200\n",
        encoding="utf-8",
    )
    pf = load_portfolio(p)
    assert len(pf) == 2


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_portfolio("does/not/exist.csv")
