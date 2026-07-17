"""Indian market data — decoupled core logic.

This module holds the *pure* data-fetching logic for Indian equities and is
deliberately free of any MCP/FastMCP dependency, so it can be unit-tested
without a server and without network access. ``mcp_server.py`` imports these
functions and exposes them as MCP tools.

Data source
-----------
* ``use_live=True``  -> real quotes via ``yfinance`` (lazy-imported), appending
  the exchange suffix ``.NS`` (NSE) or ``.BO`` (BSE) to the base ticker.
* ``use_live=False`` -> a deterministic mock generator (stable per symbol), used
  by the test suite and as an automatic fallback when yfinance is unavailable
  or a live fetch fails.

Scope guardrail: this module only *reads/observes* market data. It never places
orders or takes any financial action.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Reference data: friendly names -> NSE base tickers, and sector constituents.
# (Base tickers carry no exchange suffix; that is appended at fetch time.)
# --------------------------------------------------------------------------- #
NAME_TO_TICKER: Dict[str, str] = {
    "reliance": "RELIANCE",
    "ril": "RELIANCE",
    "tcs": "TCS",
    "tata consultancy": "TCS",
    "infosys": "INFY",
    "infy": "INFY",
    "hdfc bank": "HDFCBANK",
    "hdfcbank": "HDFCBANK",
    "icici bank": "ICICIBANK",
    "icici": "ICICIBANK",
    "sbi": "SBIN",
    "state bank": "SBIN",
    "kotak": "KOTAKBANK",
    "kotak mahindra": "KOTAKBANK",
    "kotak bank": "KOTAKBANK",
    "axis": "AXISBANK",
    "axis bank": "AXISBANK",
    "indusind": "INDUSINDBK",
    "indusind bank": "INDUSINDBK",
    "bank of baroda": "BANKBARODA",
    "bob": "BANKBARODA",
    "pnb": "PNB",
    "punjab national bank": "PNB",
    "idfc first": "IDFCFIRSTB",
    "idfc first bank": "IDFCFIRSTB",
    "federal bank": "FEDERALBNK",
    "au small finance": "AUBANK",
    "au bank": "AUBANK",
    "canara bank": "CANBK",
    "canara": "CANBK",
    "wipro": "WIPRO",
    "hcl": "HCLTECH",
    "hcl tech": "HCLTECH",
    "itc": "ITC",
    "larsen": "LT",
    "l&t": "LT",
    "bharti airtel": "BHARTIARTL",
    "airtel": "BHARTIARTL",
    "maruti": "MARUTI",
    "tata motors": "TATAMOTORS",
    "sun pharma": "SUNPHARMA",
    "asian paints": "ASIANPAINT",
}

SECTORS: Dict[str, List[str]] = {
    "it": ["TCS", "INFY", "WIPRO", "HCLTECH"],
    "technology": ["TCS", "INFY", "WIPRO", "HCLTECH"],
    "banking": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
        "INDUSINDBK", "BANKBARODA", "PNB", "IDFCFIRSTB", "FEDERALBNK",
        "AUBANK", "CANBK",
    ],
    "financials": [
        "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK",
        "INDUSINDBK", "BANKBARODA", "PNB",
    ],
    "energy": ["RELIANCE"],
    "auto": ["MARUTI", "TATAMOTORS"],
    "pharma": ["SUNPHARMA"],
    "fmcg": ["ITC", "ASIANPAINT"],
}

VALID_EXCHANGES = {"NS", "BO"}


class MarketDataError(ValueError):
    """Raised when a symbol or sector cannot be resolved."""


@dataclass(frozen=True)
class Quote:
    """A single resolved quote (JSON-serializable via ``as_dict``)."""

    symbol: str          # base ticker, e.g. RELIANCE
    yahoo_symbol: str    # e.g. RELIANCE.NS
    exchange: str        # NS | BO
    price: float
    currency: str
    change: float
    change_pct: float
    source: str          # "live" | "mock"

    def as_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "yahoo_symbol": self.yahoo_symbol,
            "exchange": self.exchange,
            "price": round(self.price, 2),
            "currency": self.currency,
            "change": round(self.change, 2),
            "change_pct": round(self.change_pct, 2),
            "source": self.source,
        }


# --------------------------------------------------------------------------- #
# Symbol resolution
# --------------------------------------------------------------------------- #
def resolve_symbol(query: str, exchange: str) -> Tuple[str, str]:
    """Resolve a free-text query to ``(base_ticker, yahoo_symbol)``.

    Accepts a friendly name ("Reliance", "TCS") or a raw ticker
    ("RELIANCE", "RELIANCE.NS").
    """
    exchange = (exchange or "NS").upper()
    if exchange not in VALID_EXCHANGES:
        raise MarketDataError(
            f"Unsupported exchange '{exchange}'. Use one of {sorted(VALID_EXCHANGES)}."
        )

    q = (query or "").strip()
    if not q:
        raise MarketDataError("Empty symbol/query.")

    # Already an exchange-suffixed Yahoo symbol?
    upper = q.upper()
    if upper.endswith(".NS") or upper.endswith(".BO"):
        base, suffix = upper.rsplit(".", 1)
        return base, f"{base}.{suffix}"

    # Friendly name lookup, else treat the query itself as the base ticker.
    base = NAME_TO_TICKER.get(q.lower(), upper)
    return base, f"{base}.{exchange}"


# --------------------------------------------------------------------------- #
# Mock generator (deterministic, stable per symbol)
# --------------------------------------------------------------------------- #
def _mock_quote(base: str, yahoo_symbol: str, exchange: str) -> Quote:
    seed = sum(ord(c) for c in base)
    price = 100.0 + (seed % 4000)          # ₹100 – ₹4099, stable per symbol
    change_pct = ((seed % 1100) / 100.0) - 5.0   # roughly -5%..+6%
    change = price * change_pct / 100.0
    return Quote(
        symbol=base,
        yahoo_symbol=yahoo_symbol,
        exchange=exchange,
        price=price,
        currency="INR",
        change=change,
        change_pct=change_pct,
        source="mock",
    )


# --------------------------------------------------------------------------- #
# Live fetch (yfinance, lazy-imported)
# --------------------------------------------------------------------------- #
def _live_quote(base: str, yahoo_symbol: str, exchange: str) -> Quote:
    import yfinance as yf  # lazy import; only needed for live mode

    ticker = yf.Ticker(yahoo_symbol)
    info = getattr(ticker, "fast_info", None) or {}
    last = info.get("last_price") if isinstance(info, dict) else getattr(info, "last_price", None)
    prev = info.get("previous_close") if isinstance(info, dict) else getattr(info, "previous_close", None)

    if last is None:
        # Fall back to recent history if fast_info is unavailable.
        hist = ticker.history(period="2d")
        if hist is None or hist.empty:
            raise MarketDataError(f"No live data returned for {yahoo_symbol}.")
        last = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[0])

    last = float(last)
    prev = float(prev) if prev else last
    change = last - prev
    change_pct = (change / prev * 100.0) if prev else 0.0
    return Quote(
        symbol=base,
        yahoo_symbol=yahoo_symbol,
        exchange=exchange,
        price=last,
        currency="INR",
        change=change,
        change_pct=change_pct,
        source="live",
    )


# --------------------------------------------------------------------------- #
# Small in-process TTL cache
# --------------------------------------------------------------------------- #
_CACHE: Dict[str, Tuple[float, Quote]] = {}


def _cached(key: str, ttl: int) -> Optional[Quote]:
    hit = _CACHE.get(key)
    if hit and (time.monotonic() - hit[0]) < ttl:
        return hit[1]
    return None


def clear_cache() -> None:
    """Reset the quote cache (used by tests)."""
    _CACHE.clear()


# --------------------------------------------------------------------------- #
# Public API (wrapped as MCP tools in mcp_server.py)
# --------------------------------------------------------------------------- #
def get_stock_quote(
    query: str,
    exchange: str = "NS",
    *,
    use_live: bool = True,
    cache_ttl_seconds: int = 60,
) -> Dict[str, object]:
    """Return a quote for a single Indian stock by name or ticker.

    Falls back to the deterministic mock if ``use_live`` is False, if yfinance
    is not installed, or if a live fetch fails — the ``source`` field records
    which path produced the data.
    """
    base, yahoo_symbol = resolve_symbol(query, exchange)
    exchange = yahoo_symbol.rsplit(".", 1)[1]

    cache_key = f"q:{yahoo_symbol}:{'live' if use_live else 'mock'}"
    cached = _cached(cache_key, cache_ttl_seconds)
    if cached is not None:
        return cached.as_dict()

    quote: Quote
    if use_live:
        try:
            quote = _live_quote(base, yahoo_symbol, exchange)
        except Exception:  # noqa: BLE001 - degrade gracefully to mock
            quote = _mock_quote(base, yahoo_symbol, exchange)
    else:
        quote = _mock_quote(base, yahoo_symbol, exchange)

    _CACHE[cache_key] = (time.monotonic(), quote)
    return quote.as_dict()


def get_sector_performance(
    sector: str,
    exchange: str = "NS",
    *,
    use_live: bool = True,
    cache_ttl_seconds: int = 60,
) -> Dict[str, object]:
    """Return aggregate performance for a sector's constituent Indian stocks."""
    key = (sector or "").strip().lower()
    constituents = SECTORS.get(key)
    if not constituents:
        raise MarketDataError(
            f"Unknown sector '{sector}'. Known sectors: {sorted(SECTORS)}."
        )

    quotes = [
        get_stock_quote(
            base,
            exchange,
            use_live=use_live,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        for base in constituents
    ]
    avg_change_pct = sum(q["change_pct"] for q in quotes) / len(quotes)
    sources = {q["source"] for q in quotes}
    return {
        "sector": key,
        "exchange": exchange,
        "constituents": quotes,
        "avg_change_pct": round(avg_change_pct, 2),
        "advancers": sum(1 for q in quotes if q["change_pct"] > 0),
        "decliners": sum(1 for q in quotes if q["change_pct"] < 0),
        "source": "mixed" if len(sources) > 1 else next(iter(sources)),
    }


# --------------------------------------------------------------------------- #
# OpenAI-format tool schemas (mirror the MCP tools; passed to the LLM payload).
# Kept here, next to the implementations, to avoid schema drift.
# --------------------------------------------------------------------------- #
TOOL_SPECS: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "get_indian_stock_quote",
            "description": (
                "Fetch a real-time quote for a single Indian stock by company "
                "name (e.g. 'Reliance', 'TCS') or ticker (e.g. 'INFY')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Company name or ticker symbol.",
                    },
                    "exchange": {
                        "type": "string",
                        "enum": ["NS", "BO"],
                        "description": "NS = NSE (.NS), BO = BSE (.BO). Default NS.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_indian_sector_performance",
            "description": (
                "Fetch aggregate performance for an Indian market sector "
                "(e.g. 'IT', 'banking', 'auto', 'pharma', 'fmcg', 'energy')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "Sector name, e.g. 'IT' or 'banking'.",
                    },
                    "exchange": {
                        "type": "string",
                        "enum": ["NS", "BO"],
                        "description": "NS = NSE (.NS), BO = BSE (.BO). Default NS.",
                    },
                },
                "required": ["sector"],
            },
        },
    },
]
