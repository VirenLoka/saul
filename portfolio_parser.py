"""Decoupled portfolio ingestion.

All on-disk portfolio data enters the system through this module, which converts
raw CSV rows into validated, typed domain objects (:class:`Holding` /
:class:`Portfolio`). Swapping the source later (a database, JSON, an API) only
requires adding a new loader here — call sites stay unchanged.

Expected CSV columns (case/whitespace-insensitive; common aliases accepted):
    Ticker, Asset Class, Quantity, Current Value
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


class PortfolioParseError(ValueError):
    """Raised when a portfolio file is missing columns or malformed."""


# Canonical column names -> set of accepted header aliases (lowercased).
_COLUMN_ALIASES: Dict[str, set[str]] = {
    "ticker": {"ticker", "symbol", "asset"},
    "asset_class": {"asset class", "asset_class", "assetclass", "class"},
    "quantity": {"quantity", "qty", "shares", "units"},
    "current_value": {"current value", "current_value", "value", "market value"},
}


@dataclass(frozen=True)
class Holding:
    """A single position in a portfolio."""

    ticker: str
    asset_class: str
    quantity: float
    current_value: float


@dataclass(frozen=True)
class Portfolio:
    """A parsed collection of holdings plus convenience aggregates."""

    holdings: List[Holding]
    source: str = "<unknown>"

    @property
    def total_value(self) -> float:
        return sum(h.current_value for h in self.holdings)

    def value_by_asset_class(self) -> Dict[str, float]:
        """Sum of current value grouped by asset class."""
        totals: Dict[str, float] = {}
        for h in self.holdings:
            totals[h.asset_class] = totals.get(h.asset_class, 0.0) + h.current_value
        return totals

    def tickers(self) -> List[str]:
        """All tickers in declaration order."""
        return [h.ticker for h in self.holdings]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.holdings)


def _normalize_header(header: List[str]) -> Dict[str, int]:
    """Map canonical column names to their column index in the CSV header."""
    lookup: Dict[str, int] = {}
    for idx, raw in enumerate(header):
        key = (raw or "").strip().lower()
        for canonical, aliases in _COLUMN_ALIASES.items():
            if key in aliases:
                lookup[canonical] = idx
    missing = [c for c in _COLUMN_ALIASES if c not in lookup]
    if missing:
        raise PortfolioParseError(
            "Portfolio CSV is missing required column(s): "
            + ", ".join(missing)
            + f". Found headers: {header}"
        )
    return lookup


def _to_float(value: str, *, field: str, row_num: int) -> float:
    """Parse a numeric cell, tolerating currency symbols, commas, and spaces."""
    cleaned = (value or "").strip().replace(",", "").replace("$", "").replace("₹", "")
    if cleaned == "":
        raise PortfolioParseError(
            f"Row {row_num}: empty value for required numeric field '{field}'."
        )
    try:
        return float(cleaned)
    except ValueError as exc:
        raise PortfolioParseError(
            f"Row {row_num}: could not parse '{value}' as a number for "
            f"field '{field}'."
        ) from exc


def load_portfolio(path: str | Path) -> Portfolio:
    """Read a portfolio CSV from ``path`` and return a validated Portfolio.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    PortfolioParseError
        If headers are missing or any row is malformed / empty.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Portfolio file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise PortfolioParseError("Portfolio CSV is empty.") from exc

        cols = _normalize_header(header)
        holdings: List[Holding] = []

        # row_num starts at 2 to reflect the human-visible line (header == 1).
        for row_num, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue  # skip blank lines gracefully
            if len(row) < len(header):
                raise PortfolioParseError(
                    f"Row {row_num}: expected {len(header)} columns, got {len(row)}."
                )
            ticker = row[cols["ticker"]].strip().upper()
            asset_class = row[cols["asset_class"]].strip()
            if not ticker:
                raise PortfolioParseError(f"Row {row_num}: empty ticker.")
            if not asset_class:
                raise PortfolioParseError(
                    f"Row {row_num}: empty asset class for {ticker}."
                )
            holdings.append(
                Holding(
                    ticker=ticker,
                    asset_class=asset_class,
                    quantity=_to_float(
                        row[cols["quantity"]], field="quantity", row_num=row_num
                    ),
                    current_value=_to_float(
                        row[cols["current_value"]],
                        field="current_value",
                        row_num=row_num,
                    ),
                )
            )

    if not holdings:
        raise PortfolioParseError("Portfolio CSV contained no data rows.")

    return Portfolio(holdings=holdings, source=str(csv_path))
