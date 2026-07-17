"""Statistical metrics for stock analysis — decoupled core logic.

This module computes common quantitative metrics used to analyze a single
stock (or a basket of stocks) from historical price data. It deliberately
mirrors ``market_data.py``: pure Python with no MCP/FastMCP dependency, so
everything is unit-testable without a server and without network access.

Data source
-----------
* ``use_live=True``  -> historical closes via ``yfinance`` (lazy-imported),
  using the same ``.NS``/``.BO`` Yahoo symbols resolved by
  ``market_data.resolve_symbol``.
* ``use_live=False`` -> a deterministic mock price series (stable per symbol),
  used by the test suite and as an automatic fallback when yfinance is
  unavailable or a live fetch fails. The ``source`` field records which path
  produced the data.

All math is pure Python (no numpy/pandas required), matching the repo's
offline-testable ethos.

Scope guardrail: read-only analytics. Nothing here places orders or takes any
financial action.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

from market_data import resolve_symbol

# Trading days per year, used for annualization.
TRADING_DAYS = 252

# Default benchmark for beta/alpha: NIFTY 50 index on Yahoo.
DEFAULT_BENCHMARK = "^NSEI"


class StatsError(ValueError):
    """Raised when a metric cannot be computed (bad input / no data)."""


# --------------------------------------------------------------------------- #
# Basic statistics helpers (pure Python, sample statistics)
# --------------------------------------------------------------------------- #
def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _covariance(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0
    mx, my = _mean(xs[:n]), _mean(ys[:n])
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / (n - 1)


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    sx, sy = _std(xs), _std(ys)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return _covariance(xs, ys) / (sx * sy)


def _percentile(xs: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of ``xs`` (pct in 0-100)."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * pct / 100.0
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def daily_returns(closes: Sequence[float]) -> List[float]:
    """Simple daily returns from a close-price series."""
    return [
        closes[i] / closes[i - 1] - 1.0
        for i in range(1, len(closes))
        if closes[i - 1]
    ]


# --------------------------------------------------------------------------- #
# Price series: deterministic mock + live yfinance fetch
# --------------------------------------------------------------------------- #
def _mock_price_series(base: str, days: int) -> List[float]:
    """Deterministic pseudo-random walk, stable per symbol (mirrors _mock_quote)."""
    seed = sum(ord(c) * 17 for c in base) or 7
    price = 100.0 + (seed % 4000)
    state = seed
    closes: List[float] = []
    for _ in range(max(2, days)):
        state = (state * 1103515245 + 12345) % (2**31)
        step = ((state % 2001) - 1000) / 1000.0  # uniform-ish in [-1, 1]
        price = max(1.0, price * (1.0 + step * 0.02))
        closes.append(round(price, 2))
    return closes


def _period_for(days: int) -> str:
    """Map a requested trading-day window to a generous yfinance period."""
    if days <= 20:
        return "3mo"
    if days <= 65:
        return "6mo"
    if days <= 130:
        return "1y"
    if days <= 260:
        return "2y"
    return "5y"


def _live_price_series(yahoo_symbol: str, days: int) -> List[float]:
    import yfinance as yf  # lazy import; only needed for live mode

    hist = yf.Ticker(yahoo_symbol).history(period=_period_for(days))
    if hist is None or hist.empty:
        raise StatsError(f"No price history returned for {yahoo_symbol}.")
    closes = [float(c) for c in hist["Close"].tolist() if c and not math.isnan(c)]
    if len(closes) < 2:
        raise StatsError(f"Insufficient price history for {yahoo_symbol}.")
    return closes[-days:]


# Small in-process TTL cache for price series (mirrors market_data._CACHE).
_SERIES_CACHE: Dict[str, Tuple[float, List[float], str]] = {}


def clear_cache() -> None:
    """Reset the price-series cache (used by tests)."""
    _SERIES_CACHE.clear()


def get_price_series(
    query: str,
    exchange: str = "NS",
    *,
    days: int = TRADING_DAYS,
    use_live: bool = True,
    cache_ttl_seconds: int = 300,
) -> Tuple[str, List[float], str]:
    """Resolve ``query`` and return ``(yahoo_symbol, closes, source)``.

    ``query`` may also be a raw Yahoo symbol like ``^NSEI`` (benchmark index),
    which bypasses the Indian-exchange suffix logic.
    """
    if (query or "").strip().startswith("^"):
        base = yahoo_symbol = query.strip().upper()
    else:
        base, yahoo_symbol = resolve_symbol(query, exchange)

    cache_key = f"s:{yahoo_symbol}:{days}:{'live' if use_live else 'mock'}"
    hit = _SERIES_CACHE.get(cache_key)
    if hit and (time.monotonic() - hit[0]) < cache_ttl_seconds:
        return yahoo_symbol, hit[1], hit[2]

    source = "mock"
    if use_live:
        try:
            closes = _live_price_series(yahoo_symbol, days)
            source = "live"
        except Exception:  # noqa: BLE001 - degrade gracefully to mock
            closes = _mock_price_series(base, days)
    else:
        closes = _mock_price_series(base, days)

    _SERIES_CACHE[cache_key] = (time.monotonic(), closes, source)
    return yahoo_symbol, closes, source


# --------------------------------------------------------------------------- #
# Public API: return / risk statistics
# --------------------------------------------------------------------------- #
def get_return_statistics(
    query: str,
    exchange: str = "NS",
    *,
    period_days: int = TRADING_DAYS,
    risk_free_rate_annual: float = 0.05,
    use_live: bool = True,
    cache_ttl_seconds: int = 300,
) -> Dict[str, object]:
    """Return distribution / risk-adjusted return metrics for one stock.

    Covers cumulative & annualized return, annualized volatility, Sharpe and
    Sortino ratios, max drawdown, and historical VaR/CVaR (95%).
    """
    yahoo_symbol, closes, source = get_price_series(
        query, exchange, days=period_days, use_live=use_live,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    rets = daily_returns(closes)
    if not rets:
        raise StatsError(f"Not enough history to compute returns for {query}.")

    n = len(rets)
    mean_d, std_d = _mean(rets), _std(rets)
    cumulative = closes[-1] / closes[0] - 1.0
    ann_return = (1.0 + cumulative) ** (TRADING_DAYS / n) - 1.0
    ann_vol = std_d * math.sqrt(TRADING_DAYS)

    rf_daily = risk_free_rate_annual / TRADING_DAYS
    excess = [r - rf_daily for r in rets]
    sharpe = (_mean(excess) / std_d * math.sqrt(TRADING_DAYS)) if std_d else 0.0
    downside = [min(r - rf_daily, 0.0) for r in rets]
    ddev = math.sqrt(_mean([d * d for d in downside])) if downside else 0.0
    sortino = (_mean(excess) / ddev * math.sqrt(TRADING_DAYS)) if ddev else 0.0

    # Max drawdown over the window.
    peak, max_dd = closes[0], 0.0
    for c in closes:
        peak = max(peak, c)
        max_dd = min(max_dd, c / peak - 1.0)

    var_95 = _percentile(rets, 5.0)                     # 5th pct daily return
    tail = [r for r in rets if r <= var_95]
    cvar_95 = _mean(tail) if tail else var_95

    return {
        "symbol": yahoo_symbol,
        "period_days": n,
        "source": source,
        "last_price": round(closes[-1], 2),
        "cumulative_return_pct": round(cumulative * 100.0, 2),
        "annualized_return_pct": round(ann_return * 100.0, 2),
        "annualized_volatility_pct": round(ann_vol * 100.0, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "var_95_daily_pct": round(var_95 * 100.0, 2),
        "cvar_95_daily_pct": round(cvar_95 * 100.0, 2),
        "mean_daily_return_pct": round(mean_d * 100.0, 4),
        "risk_free_rate_annual_pct": round(risk_free_rate_annual * 100.0, 2),
    }


# --------------------------------------------------------------------------- #
# Public API: technical indicators
# --------------------------------------------------------------------------- #
def _sma(closes: Sequence[float], window: int) -> Optional[float]:
    if len(closes) < window:
        return None
    return _mean(closes[-window:])


def _ema_series(closes: Sequence[float], span: int) -> List[float]:
    if not closes:
        return []
    alpha = 2.0 / (span + 1.0)
    out = [closes[0]]
    for c in closes[1:]:
        out.append(alpha * c + (1.0 - alpha) * out[-1])
    return out


def _rsi(closes: Sequence[float], window: int = 14) -> Optional[float]:
    rets = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    if len(rets) < window:
        return None
    recent = rets[-window:]
    gains = [r for r in recent if r > 0]
    losses = [-r for r in recent if r < 0]
    avg_gain = sum(gains) / window
    avg_loss = sum(losses) / window
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def get_technical_indicators(
    query: str,
    exchange: str = "NS",
    *,
    period_days: int = TRADING_DAYS,
    use_live: bool = True,
    cache_ttl_seconds: int = 300,
) -> Dict[str, object]:
    """Common technical indicators: SMA/EMA, RSI, MACD, Bollinger, momentum."""
    yahoo_symbol, closes, source = get_price_series(
        query, exchange, days=period_days, use_live=use_live,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    last = closes[-1]

    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal = _ema_series(macd_line, 9)
    macd, macd_signal = macd_line[-1], signal[-1]

    boll_mid = _sma(closes, 20)
    boll_sd = _std(closes[-20:]) if len(closes) >= 20 else None
    boll_upper = boll_mid + 2 * boll_sd if boll_mid is not None and boll_sd else None
    boll_lower = boll_mid - 2 * boll_sd if boll_mid is not None and boll_sd else None
    # %B: where price sits inside the bands (0=lower, 1=upper).
    pct_b = (
        (last - boll_lower) / (boll_upper - boll_lower)
        if boll_upper is not None and boll_upper != boll_lower
        else None
    )

    hi, lo = max(closes), min(closes)
    momentum_1m = (last / closes[-21] - 1.0) if len(closes) > 21 else None
    momentum_3m = (last / closes[-63] - 1.0) if len(closes) > 63 else None

    def _r(x: Optional[float], nd: int = 2) -> Optional[float]:
        return round(x, nd) if x is not None else None

    return {
        "symbol": yahoo_symbol,
        "period_days": len(closes),
        "source": source,
        "last_price": _r(last),
        "sma_20": _r(_sma(closes, 20)),
        "sma_50": _r(_sma(closes, 50)),
        "sma_200": _r(_sma(closes, 200)),
        "ema_12": _r(ema12[-1]),
        "ema_26": _r(ema26[-1]),
        "rsi_14": _r(_rsi(closes)),
        "macd": _r(macd, 4),
        "macd_signal": _r(macd_signal, 4),
        "macd_histogram": _r(macd - macd_signal, 4),
        "bollinger_upper": _r(boll_upper),
        "bollinger_middle": _r(boll_mid),
        "bollinger_lower": _r(boll_lower),
        "bollinger_pct_b": _r(pct_b, 3),
        "momentum_1m_pct": _r(momentum_1m * 100.0 if momentum_1m is not None else None),
        "momentum_3m_pct": _r(momentum_3m * 100.0 if momentum_3m is not None else None),
        "window_high": _r(hi),
        "window_low": _r(lo),
        "pct_off_window_high": _r((last / hi - 1.0) * 100.0),
    }


# --------------------------------------------------------------------------- #
# Public API: benchmark-relative risk (beta / alpha / correlation)
# --------------------------------------------------------------------------- #
def get_risk_metrics(
    query: str,
    exchange: str = "NS",
    *,
    benchmark: str = DEFAULT_BENCHMARK,
    period_days: int = TRADING_DAYS,
    risk_free_rate_annual: float = 0.05,
    use_live: bool = True,
    cache_ttl_seconds: int = 300,
) -> Dict[str, object]:
    """Benchmark-relative metrics: beta, Jensen's alpha, correlation, R²."""
    yahoo_symbol, closes, source = get_price_series(
        query, exchange, days=period_days, use_live=use_live,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    bench_symbol, bench_closes, bench_source = get_price_series(
        benchmark, exchange, days=period_days, use_live=use_live,
        cache_ttl_seconds=cache_ttl_seconds,
    )

    r_s, r_b = daily_returns(closes), daily_returns(bench_closes)
    n = min(len(r_s), len(r_b))
    if n < 2:
        raise StatsError(f"Not enough overlapping history for {query} vs {benchmark}.")
    r_s, r_b = r_s[-n:], r_b[-n:]

    var_b = _std(r_b) ** 2
    beta = _covariance(r_s, r_b) / var_b if var_b else 0.0
    corr = _correlation(r_s, r_b)

    rf_daily = risk_free_rate_annual / TRADING_DAYS
    # Jensen's alpha, annualized: mean excess return not explained by beta.
    alpha_daily = _mean(r_s) - rf_daily - beta * (_mean(r_b) - rf_daily)
    alpha_annual = alpha_daily * TRADING_DAYS

    # Tracking error and information ratio vs the benchmark.
    active = [r_s[i] - r_b[i] for i in range(n)]
    te = _std(active) * math.sqrt(TRADING_DAYS)
    ir = (_mean(active) * TRADING_DAYS / te) if te else 0.0

    return {
        "symbol": yahoo_symbol,
        "benchmark": bench_symbol,
        "period_days": n,
        "source": source if source == bench_source else "mixed",
        "beta": round(beta, 3),
        "alpha_annualized_pct": round(alpha_annual * 100.0, 2),
        "correlation_to_benchmark": round(corr, 3),
        "r_squared": round(corr * corr, 3),
        "tracking_error_pct": round(te * 100.0, 2),
        "information_ratio": round(ir, 3),
    }


# --------------------------------------------------------------------------- #
# Public API: pairwise correlations across a basket
# --------------------------------------------------------------------------- #
def get_correlation_matrix(
    queries: Sequence[str],
    exchange: str = "NS",
    *,
    period_days: int = TRADING_DAYS,
    use_live: bool = True,
    cache_ttl_seconds: int = 300,
) -> Dict[str, object]:
    """Pairwise daily-return correlations for a list of stocks."""
    names = [q for q in (queries or []) if (q or "").strip()]
    if len(names) < 2:
        raise StatsError("Need at least two stocks for a correlation matrix.")

    series: Dict[str, List[float]] = {}
    sources = set()
    for q in names:
        sym, closes, src = get_price_series(
            q, exchange, days=period_days, use_live=use_live,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        series[sym] = daily_returns(closes)
        sources.add(src)

    symbols = list(series)
    n = min(len(r) for r in series.values())
    matrix: Dict[str, Dict[str, float]] = {}
    pairs: List[Dict[str, object]] = []
    for i, a in enumerate(symbols):
        matrix[a] = {}
        for j, b in enumerate(symbols):
            c = 1.0 if a == b else _correlation(series[a][-n:], series[b][-n:])
            matrix[a][b] = round(c, 3)
            if j > i:
                pairs.append({"pair": [a, b], "correlation": round(c, 3)})

    pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)
    return {
        "symbols": symbols,
        "period_days": n,
        "source": "mixed" if len(sources) > 1 else next(iter(sources)),
        "matrix": matrix,
        "ranked_pairs": pairs,
    }


# --------------------------------------------------------------------------- #
# Public API: fundamentals snapshot (valuation / profitability)
# --------------------------------------------------------------------------- #
_FUNDAMENTAL_FIELDS = {
    "market_cap": "marketCap",
    "trailing_pe": "trailingPE",
    "forward_pe": "forwardPE",
    "price_to_book": "priceToBook",
    "return_on_equity": "returnOnEquity",
    "debt_to_equity": "debtToEquity",
    "profit_margin": "profitMargins",
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "dividend_yield": "dividendYield",
    "trailing_eps": "trailingEps",
    "book_value": "bookValue",
}


def _mock_fundamentals(base: str) -> Dict[str, object]:
    seed = sum(ord(c) for c in base)
    return {
        "market_cap": (seed % 90 + 10) * 10_000_000_000,
        "trailing_pe": round(8.0 + (seed % 400) / 10.0, 2),
        "forward_pe": round(7.0 + (seed % 350) / 10.0, 2),
        "price_to_book": round(0.8 + (seed % 90) / 10.0, 2),
        "return_on_equity": round(0.05 + (seed % 25) / 100.0, 4),
        "debt_to_equity": round((seed % 180) / 1.0, 2),
        "profit_margin": round(0.04 + (seed % 22) / 100.0, 4),
        "revenue_growth": round(-0.05 + (seed % 30) / 100.0, 4),
        "earnings_growth": round(-0.08 + (seed % 35) / 100.0, 4),
        "dividend_yield": round((seed % 40) / 1000.0, 4),
        "trailing_eps": round(5.0 + (seed % 900) / 10.0, 2),
        "book_value": round(50.0 + (seed % 1500), 2),
    }


def get_fundamentals(
    query: str,
    exchange: str = "NS",
    *,
    use_live: bool = True,
) -> Dict[str, object]:
    """Valuation & profitability snapshot from yfinance ``Ticker.info``."""
    base, yahoo_symbol = resolve_symbol(query, exchange)

    source = "mock"
    values: Dict[str, object] = {}
    if use_live:
        try:
            import yfinance as yf  # lazy import

            info = yf.Ticker(yahoo_symbol).info or {}
            values = {ours: info.get(theirs) for ours, theirs in _FUNDAMENTAL_FIELDS.items()}
            if any(v is not None for v in values.values()):
                source = "live"
            else:
                values = _mock_fundamentals(base)
        except Exception:  # noqa: BLE001 - degrade gracefully to mock
            values = _mock_fundamentals(base)
    else:
        values = _mock_fundamentals(base)

    return {"symbol": yahoo_symbol, "source": source, **values}


# --------------------------------------------------------------------------- #
# OpenAI-format tool schemas (mirror the MCP tools; passed to the LLM payload).
# Kept here, next to the implementations, to avoid schema drift.
# --------------------------------------------------------------------------- #
_COMMON_PROPS = {
    "query": {
        "type": "string",
        "description": "Company name (e.g. 'Reliance') or ticker (e.g. 'INFY').",
    },
    "exchange": {
        "type": "string",
        "enum": ["NS", "BO"],
        "description": "NS = NSE (.NS), BO = BSE (.BO). Default NS.",
    },
    "period_days": {
        "type": "integer",
        "description": "Trading-day lookback window (default 252 = ~1 year).",
    },
}

STATS_TOOL_SPECS: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "get_return_statistics",
            "description": (
                "Compute return/risk statistics for one Indian stock over a "
                "lookback window: cumulative & annualized return, annualized "
                "volatility, Sharpe & Sortino ratios, max drawdown, and "
                "historical VaR/CVaR (95%)."
            ),
            "parameters": {
                "type": "object",
                "properties": dict(_COMMON_PROPS),
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_technical_indicators",
            "description": (
                "Compute technical indicators for one Indian stock: SMA "
                "(20/50/200), EMA (12/26), RSI-14, MACD (+signal/histogram), "
                "Bollinger bands with %B, 1m/3m momentum, and distance from "
                "the window high/low."
            ),
            "parameters": {
                "type": "object",
                "properties": dict(_COMMON_PROPS),
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_metrics",
            "description": (
                "Compute benchmark-relative risk metrics for one Indian stock "
                "versus an index (default NIFTY 50 ^NSEI): beta, annualized "
                "Jensen's alpha, correlation, R-squared, tracking error, and "
                "information ratio."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_COMMON_PROPS,
                    "benchmark": {
                        "type": "string",
                        "description": "Benchmark Yahoo symbol (default '^NSEI').",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_correlation_matrix",
            "description": (
                "Compute pairwise daily-return correlations for two or more "
                "Indian stocks, plus a ranking of the most-correlated pairs. "
                "Useful for diversification and pair analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Two or more company names or tickers.",
                    },
                    "exchange": _COMMON_PROPS["exchange"],
                    "period_days": _COMMON_PROPS["period_days"],
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_fundamentals",
            "description": (
                "Fetch a fundamentals snapshot for one Indian stock: market "
                "cap, trailing/forward P/E, P/B, ROE, debt-to-equity, margins, "
                "revenue/earnings growth, dividend yield, EPS, book value."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": _COMMON_PROPS["query"],
                    "exchange": _COMMON_PROPS["exchange"],
                },
                "required": ["query"],
            },
        },
    },
]
