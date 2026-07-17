"""Backtesting engine — periodic-rebalancing market simulation.

At each rebalance date the engine:

1. Computes **point-in-time analytics** (return / volatility / Sharpe over a
   trailing window ending at that date) from a historical price matrix — no
   look-ahead.
2. Fetches **archive news** for the preceding window via newsdata.io (clamped to
   the training-cutoff floor).
3. Optionally attaches **knowledge-graph context** (validated peer associations
   from a graph built by ``run_graph_reasoning``).
4. Lets the **LLM choose target weights** from that context (the two-step
   portfolio flow), falling back to a deterministic baseline offline / on
   failure. **web_search is disabled** throughout.
5. Sizes the book at that date's prices (whole shares) and marks it to market
   daily until the next rebalance.

It then reports the equity curve and performance vs a benchmark, writing a CSV +
markdown report to the results directory.

The whole thing runs offline (mock prices + deterministic baseline weights) so it
is testable with no model, GPU or network.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from backtesting.news_archive import fetch_news_archive
from market_data import SECTORS
from portfolio_builder import compute_baseline_weights, generate_final_portfolio

logger = logging.getLogger("saul.backtest")

TRADING_DAYS = 252
_REBALANCE_STEP = {"weekly": 7, "monthly": 30, "quarterly": 91}


class BacktestError(ValueError):
    """Raised when a backtest cannot be set up or run."""


# --------------------------------------------------------------------------- #
# Date helpers
# --------------------------------------------------------------------------- #
def _d(value: str) -> _dt.date:
    return _dt.date.fromisoformat(str(value).strip())


def _business_days(start: _dt.date, end: _dt.date) -> List[_dt.date]:
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:  # Mon-Fri (holidays ignored for the prototype)
            days.append(cur)
        cur += _dt.timedelta(days=1)
    return days


def _rebalance_dates(dates: List[_dt.date], rebalance: str) -> List[_dt.date]:
    """Pick rebalance dates from the trading-date axis by cadence."""
    if not dates:
        return []
    step = _REBALANCE_STEP.get(rebalance, 30)
    picks, last = [dates[0]], dates[0]
    for d in dates[1:]:
        if (d - last).days >= step:
            picks.append(d)
            last = d
    return picks


# --------------------------------------------------------------------------- #
# Price matrix (historical closes; yfinance live, deterministic mock offline)
# --------------------------------------------------------------------------- #
def _mock_series(symbol: str, dates: List[_dt.date]) -> Dict[_dt.date, float]:
    """Deterministic pseudo-random walk keyed by date (stable per symbol)."""
    seed = sum(ord(c) * 17 for c in symbol) or 7
    price, state = 100.0 + (seed % 4000), seed
    out: Dict[_dt.date, float] = {}
    for d in dates:
        state = (state * 1103515245 + 12345) % (2**31)
        step = ((state % 2001) - 1000) / 1000.0
        price = max(1.0, price * (1.0 + step * 0.02))
        out[d] = round(price, 2)
    return out


def _live_series(symbol: str, start: _dt.date, end: _dt.date) -> Dict[_dt.date, float]:
    import yfinance as yf  # lazy import

    hist = yf.Ticker(symbol).history(
        start=start.isoformat(), end=(end + _dt.timedelta(days=1)).isoformat()
    )
    if hist is None or hist.empty:
        raise BacktestError(f"No price history for {symbol} in the window.")
    out: Dict[_dt.date, float] = {}
    for ts, close in hist["Close"].items():
        if close and not math.isnan(close):
            out[ts.date()] = float(close)
    return out


def _resolve_symbol(query: str) -> str:
    from market_data import resolve_symbol

    if query.startswith("^"):
        return query
    return resolve_symbol(query, "NS")[1]


def load_price_matrix(
    tickers: List[str],
    benchmark: str,
    start: _dt.date,
    end: _dt.date,
    *,
    use_live: bool,
) -> Tuple[List[_dt.date], Dict[str, Dict[_dt.date, float]]]:
    """Return ``(trading_dates, {ticker: {date: close}})`` incl. the benchmark."""
    dates = _business_days(start, end)
    if not dates:
        raise BacktestError("Empty date range.")
    matrix: Dict[str, Dict[_dt.date, float]] = {}
    for tick in list(dict.fromkeys(tickers + [benchmark])):
        series: Dict[_dt.date, float] = {}
        if use_live:
            try:
                series = _live_series(_resolve_symbol(tick), start, end)
            except Exception:  # noqa: BLE001 - fall back to deterministic mock
                series = {}
        if not series:
            series = _mock_series(tick, dates)
        matrix[tick] = series
    return dates, matrix


def _price_asof(series: Dict[_dt.date, float], day: _dt.date) -> Optional[float]:
    """Most recent close on or before ``day`` (forward-fill for gaps)."""
    if day in series:
        return series[day]
    prior = [d for d in series if d <= day]
    return series[max(prior)] if prior else None


# --------------------------------------------------------------------------- #
# Point-in-time analytics (trailing window ending at the rebalance date)
# --------------------------------------------------------------------------- #
def _returns(closes: List[float]) -> List[float]:
    return [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes)) if closes[i - 1]]


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def asof_analytics(
    tickers_by_sector: Dict[str, List[str]],
    matrix: Dict[str, Dict[_dt.date, float]],
    dates: List[_dt.date],
    as_of: _dt.date,
    *,
    lookback_days: int,
    rf_annual: float = 0.05,
) -> Dict[str, object]:
    """Return/vol/Sharpe per stock from the trailing window ending at ``as_of``."""
    window = [d for d in dates if d <= as_of][-lookback_days:]
    analytics: Dict[str, List[Dict[str, object]]] = {}
    for sector, tickers in tickers_by_sector.items():
        rows: List[Dict[str, object]] = []
        for tick in tickers:
            series = matrix.get(tick, {})
            closes = [series[d] for d in window if d in series]
            if len(closes) < 3:
                continue
            rets = _returns(closes)
            mean_d, std_d = _mean(rets), _std(rets)
            ann_ret = (closes[-1] / closes[0] - 1.0) * (TRADING_DAYS / max(1, len(rets)))
            ann_vol = std_d * math.sqrt(TRADING_DAYS)
            rf_daily = rf_annual / TRADING_DAYS
            sharpe = ((mean_d - rf_daily) / std_d * math.sqrt(TRADING_DAYS)) if std_d else 0.0
            rows.append({
                "ticker": tick,
                "price": round(closes[-1], 2),
                "sharpe_ratio": round(sharpe, 3),
                "annualized_volatility_pct": round(ann_vol * 100.0, 2),
                "annualized_return_pct": round(ann_ret * 100.0, 2),
            })
        analytics[sector] = rows
    return {"as_of": as_of.isoformat(), "analytics": analytics}


# --------------------------------------------------------------------------- #
# Knowledge-graph context (validated peer associations)
# --------------------------------------------------------------------------- #
def graph_context(graph_id: str, tickers: List[str]) -> Dict[str, List[str]]:
    """Map each ticker -> its validated peers in the persisted graph (or {})."""
    try:
        from sector_graph import get_graph_object
    except Exception:  # noqa: BLE001
        return {}
    try:
        graph = get_graph_object(graph_id)
    except Exception:  # noqa: BLE001 - graph missing -> no context
        return {}
    peers: Dict[str, List[str]] = {t: [] for t in tickers}
    for edge in graph.edges.values():
        if edge.status != "validated":
            continue
        for a, b in ((edge.source, edge.target), (edge.target, edge.source)):
            if a in peers and b not in peers[a]:
                peers[a].append(b)
    return {t: p for t, p in peers.items() if p}


# --------------------------------------------------------------------------- #
# Weight engine: LLM two-step (curated tools, web_search OFF) + baseline fallback
# --------------------------------------------------------------------------- #
_WEIGHT_SYSTEM = (
    "You are a portfolio-construction assistant for Indian equities (READ-ONLY, "
    "no trades) running inside a BACKTEST as of a historical date. You are given "
    "point-in-time analytics (return/volatility/Sharpe), archive news for the "
    "window, and knowledge-graph peer associations. Choose target weights per "
    "ticker (fractions of capital, may sum to <1 for cash). DROP or shrink "
    "negative-Sharpe names, avoid over-weighting strongly-linked peers, and keep "
    "the book diversified. Call generate_final_portfolio with your ticker_weights "
    "and a short reasoning note. web_search is unavailable."
)


def _llm_weights(
    provider,
    analytics: Dict[str, object],
    news: Dict[str, object],
    peers: Dict[str, List[str]],
    total_amount: float,
    as_of: str,
    sink: Optional[Callable[[str], None]] = None,
) -> Optional[Tuple[Dict[str, float], str]]:
    """Ask the model for weights; return ``(weights, reasoning)`` or None on failure."""
    from llm_provider import StreamEvent  # noqa: F401  (type hint clarity)

    context = {
        "as_of": as_of,
        "analytics": analytics.get("analytics", analytics),
        "recent_news": [a.get("title") for a in news.get("articles", [])][:6],
        "graph_peers": peers,
        "total_amount": total_amount,
    }
    messages = [
        {"role": "system", "content": _WEIGHT_SYSTEM},
        {"role": "user", "content": (
            "Choose the portfolio weights for this rebalance and call "
            "generate_final_portfolio.\n" + json.dumps(context, indent=2)
        )},
    ]
    # Single tool round: force the model to emit its weights via the tool call.
    calls: List[Dict[str, str]] = []
    try:
        for ev in provider.stream_chat(messages, tools=_weight_tool_specs()):
            if ev.type == "tool_call":
                calls.append({"name": ev.name, "arguments": ev.arguments})
            elif ev.type in ("content", "reasoning"):
                if sink and ev.text:
                    sink(ev.text)
            elif ev.type == "error":
                raise RuntimeError(ev.text)
            elif ev.type == "done":
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM weight selection failed (%s); using baseline.", exc)
        return None

    for c in calls:
        if c["name"] == "generate_final_portfolio":
            try:
                args = json.loads(c["arguments"] or "{}")
                weights = args.get("ticker_weights") or {}
                if isinstance(weights, str):
                    weights = json.loads(weights)
                weights = {str(k).upper(): float(v) for k, v in weights.items() if float(v) > 0}
                if weights:
                    return weights, str(args.get("reasoning") or "LLM-selected weights.")
            except (ValueError, TypeError):
                continue
    return None


def _weight_tool_specs() -> List[Dict[str, object]]:
    from portfolio_builder import GENERATE_PORTFOLIO_TOOL_SPECS

    return GENERATE_PORTFOLIO_TOOL_SPECS


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _performance(curve: List[Tuple[_dt.date, float]], rf_annual: float = 0.05) -> Dict[str, object]:
    if len(curve) < 2:
        return {}
    values = [v for _, v in curve]
    start_v, end_v = values[0], values[-1]
    total_ret = end_v / start_v - 1.0
    days = max(1, (curve[-1][0] - curve[0][0]).days)
    cagr = (end_v / start_v) ** (365.0 / days) - 1.0 if start_v > 0 else 0.0
    rets = _returns(values)
    ann_vol = _std(rets) * math.sqrt(TRADING_DAYS)
    rf_daily = rf_annual / TRADING_DAYS
    sharpe = ((_mean(rets) - rf_daily) / _std(rets) * math.sqrt(TRADING_DAYS)) if _std(rets) else 0.0
    peak, max_dd = values[0], 0.0
    for v in values:
        peak = max(peak, v)
        max_dd = min(max_dd, v / peak - 1.0)
    return {
        "total_return_pct": round(total_ret * 100.0, 2),
        "cagr_pct": round(cagr * 100.0, 2),
        "annualized_volatility_pct": round(ann_vol * 100.0, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 2),
        "final_value": round(end_v, 2),
    }


# --------------------------------------------------------------------------- #
# Results writing
# --------------------------------------------------------------------------- #
def _write_results(
    results_dir: str, name: str, curve: List[Tuple[_dt.date, float]],
    bench_curve: List[Tuple[_dt.date, float]], report_md: str,
) -> Dict[str, str]:
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{name}_equity_curve.csv"
    lines = ["date,portfolio_value,benchmark_value"]
    bench = dict(bench_curve)
    for d, v in curve:
        lines.append(f"{d.isoformat()},{v:.2f},{bench.get(d, ''):.2f}"
                     if d in bench else f"{d.isoformat()},{v:.2f},")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_path = out / f"{name}_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    return {"equity_curve_csv": str(csv_path), "report_md": str(report_path)}


def _render_report(
    *, name: str, cfg_summary: Dict[str, object], rebalances: List[Dict[str, object]],
    port_perf: Dict[str, object], bench_perf: Dict[str, object],
) -> str:
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    lines = [
        f"# Backtest report — {name}",
        "",
        f"- Generated: {ts}",
        f"- Window: {cfg_summary['start']} → {cfg_summary['end']}  "
        f"(rebalance: {cfg_summary['rebalance']})",
        f"- Capital: ₹{cfg_summary['capital']:,.0f}  |  Benchmark: {cfg_summary['benchmark']}",
        f"- Weight engine: {cfg_summary['engine']}  |  graph context: "
        f"{cfg_summary['use_graph']}  |  web_search: DISABLED",
        "",
        "## Performance vs benchmark",
        "",
        "| Metric | Portfolio | Benchmark |",
        "|---|---|---|",
    ]
    for label, key in [
        ("Total return %", "total_return_pct"), ("CAGR %", "cagr_pct"),
        ("Volatility %", "annualized_volatility_pct"), ("Sharpe", "sharpe_ratio"),
        ("Max drawdown %", "max_drawdown_pct"), ("Final value ₹", "final_value"),
    ]:
        lines.append(f"| {label} | {port_perf.get(key)} | {bench_perf.get(key)} |")
    lines += ["", "## Rebalances", ""]
    for r in rebalances:
        lines.append(
            f"### {r['date']}  ({r['engine']})\n\n"
            f"- Holdings: {r['holdings']}\n- Weights: {r['weights']}\n"
            f"- Value at rebalance: ₹{r['value']:,.0f}\n- Rationale: {r['reasoning']}\n"
        )
    lines += ["", "_Illustrative backtest — not financial advice; no trades executed._", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #
def run_backtest(
    config,
    *,
    provider=None,
    graph_id: Optional[str] = None,
    use_live: bool = True,
    name: str = "backtest",
    sink: Optional[Callable[[str], None]] = None,
) -> Dict[str, object]:
    """Run the periodic-rebalancing backtest and return results + written paths.

    ``config`` is the app config; ``provider`` (optional) drives LLM weight
    selection, otherwise the deterministic baseline is used. ``graph_id`` (if
    ``config.backtesting.use_graph``) supplies peer-association context.
    """
    bt = config.backtesting
    nd = config.newsdata
    exchange = config.mcp.market_data.default_exchange

    start = _d(bt.start_date)
    floor = _d(nd.earliest_date)
    if start < floor:  # enforce the training-cutoff floor
        logger.warning("start_date %s < floor %s; clamping to floor.", start, floor)
        start = floor
    end = _d(bt.end_date) if bt.end_date else _dt.date.today()
    if end <= start:
        raise BacktestError(f"end_date {end} must be after start_date {start}.")

    sectors = [s for s in bt.sectors if s in SECTORS]
    if not sectors:
        raise BacktestError(f"No known sectors in {bt.sectors}.")
    tickers_by_sector = {s: list(SECTORS[s]) for s in sectors}
    all_tickers = sorted({t for ts in tickers_by_sector.values() for t in ts})

    dates, matrix = load_price_matrix(all_tickers, bt.benchmark, start, end, use_live=use_live)
    reb_dates = _rebalance_dates(dates, bt.rebalance)
    if not reb_dates:
        raise BacktestError("No rebalance dates in the window.")

    peers_full = graph_context(graph_id, all_tickers) if (bt.use_graph and graph_id) else {}

    def emit(msg: str) -> None:
        if sink:
            sink(msg)

    capital = float(bt.initial_capital)
    holdings: Dict[str, int] = {}
    cash = capital
    curve: List[Tuple[_dt.date, float]] = []
    rebalance_log: List[Dict[str, object]] = []
    engine_label = "llm+baseline" if provider is not None else "baseline"

    def book_value(day: _dt.date) -> float:
        val = cash
        for tick, qty in holdings.items():
            price = _price_asof(matrix[tick], day)
            if price:
                val += qty * price
        return val

    reb_set = set(reb_dates)
    for day in dates:
        if day in reb_set:
            value = book_value(day)  # mark to market before rebalancing
            prices = {t: _price_asof(matrix[t], day) for t in all_tickers}
            prices = {t: p for t, p in prices.items() if p and p > 0}

            analytics = asof_analytics(
                tickers_by_sector, matrix, dates, day, lookback_days=bt.lookback_days
            )
            news = fetch_news_archive(
                " OR ".join(sectors) + " India stocks",
                (day - _dt.timedelta(days=bt.news_lookback_days)).isoformat(),
                day.isoformat(),
                api_key=nd.api_key, language=nd.language,
                earliest_date=nd.earliest_date, max_articles=nd.max_articles,
                use_live=(use_live and nd.use_live),
            )
            peers = {t: peers_full[t] for t in prices if t in peers_full}

            emit(f"\n=== Rebalance {day.isoformat()} — value ₹{value:,.0f} ===\n")
            weights_reason = None
            if provider is not None:
                weights_reason = _llm_weights(
                    provider, analytics, news, peers, value, day.isoformat(), sink=sink
                )
            if weights_reason is None:  # deterministic fallback
                weights = compute_baseline_weights(
                    sectors, exchange=exchange, risk_profile=bt.risk_profile, use_live=False
                )
                # Re-price baseline picks at the as-of date.
                weights = {t: w for t, w in weights.items() if t in prices}
                reasoning = f"Deterministic baseline ({bt.risk_profile}) as of {day}."
            else:
                weights, reasoning = weights_reason
                weights = {t: w for t, w in weights.items() if t in prices}
            if not weights:
                emit("  (no priceable weights; holding cash)\n")
                holdings, cash = {}, value
            else:
                built = generate_final_portfolio(
                    weights, value, exchange=exchange, reasoning=reasoning,
                    use_live=False, write_files=False,
                    price_overrides={t: prices[t] for t in weights},
                )
                holdings = {h["ticker"]: h["quantity"] for h in built["holdings"]}
                cash = built["cash_remaining"]
                rebalance_log.append({
                    "date": day.isoformat(),
                    "engine": "llm" if weights_reason is not None else "baseline",
                    "holdings": list(holdings),
                    "weights": {t: round(w, 3) for t, w in weights.items()},
                    "value": value,
                    "reasoning": reasoning,
                })
                emit(f"  holdings: {list(holdings)}  cash ₹{cash:,.0f}\n")
        curve.append((day, book_value(day)))

    bench_series = matrix[bt.benchmark]
    bench0 = _price_asof(bench_series, dates[0]) or 1.0
    bench_curve = [
        (d, round(capital * (_price_asof(bench_series, d) or bench0) / bench0, 2))
        for d in dates
    ]

    port_perf = _performance(curve)
    bench_perf = _performance(bench_curve)
    cfg_summary = {
        "start": start.isoformat(), "end": end.isoformat(), "rebalance": bt.rebalance,
        "capital": capital, "benchmark": bt.benchmark, "engine": engine_label,
        "use_graph": bool(bt.use_graph and graph_id),
    }
    report = _render_report(
        name=name, cfg_summary=cfg_summary, rebalances=rebalance_log,
        port_perf=port_perf, bench_perf=bench_perf,
    )
    paths = _write_results(bt.results_dir, name, curve, bench_curve, report)

    return {
        "name": name,
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "rebalance": bt.rebalance,
        "rebalance_count": len(rebalance_log),
        "engine": engine_label,
        "graph_id": graph_id if cfg_summary["use_graph"] else None,
        "web_search_enabled": False,
        "portfolio": port_perf,
        "benchmark": bench_perf,
        "rebalances": rebalance_log,
        **paths,
    }
