"""Two-step, LLM-driven diversified-portfolio construction.

The allocation decision belongs to the **reasoning model**, not to Python. So
this module is split into two tools the LLM orchestrates:

1. :func:`fetch_sector_analytics` — *reads only*. For the given sectors it hits
   the market-data layer (yfinance, mock fallback) and returns raw per-stock
   metrics: volatility, P/E, Sharpe ratio, annualized return, and price. It buys
   and sizes nothing; it just hands the numbers to the model.

2. :func:`generate_final_portfolio` — *executes only*. It takes the weights the
   model chose (``ticker_weights``, fractions of capital) plus the reasoning it
   wrote, does the baseline mechanical work (size each position, round DOWN to
   whole shares, tally cash left), and writes the **CSV** portfolio plus a
   **reasoning** markdown file that records the model's rationale and the share
   math.

Between the two, the model examines the metrics and decides the mix (e.g.
"drop negative-Sharpe names, tilt toward higher risk-adjusted return").

:func:`compute_baseline_weights` provides a deterministic weighting (top
risk-adjusted pick per sector, inverse-volatility) used only as an *offline
fallback* (mock provider / tests) so the pipeline still runs with no model — in
real use the model's weights are authoritative.

Weights are **literal fractions of capital**: a weight of 0.4 deploys 40% of the
amount to that name; if the weights sum to less than 1 the remainder stays as
cash (so dropping an asset genuinely reduces deployed capital); if they sum to
more than 1 they are normalized down (with a flag).

Scope guardrail: read-only/analytical. It proposes an *illustrative* allocation;
it never executes trades.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from market_data import SECTORS, get_stock_quote

# A spread across defensive + cyclical sectors present in the reference universe.
DEFAULT_SECTORS: List[str] = ["it", "banking", "energy", "auto", "pharma", "fmcg"]
DEFAULT_OUTPUT_DIR = "knowledge/portfolios"

# risk_profile -> (return_weight, vol_weight, pe_weight) for the baseline scorer.
_RISK_WEIGHTS = {
    "conservative": (0.2, 1.0, 0.4),
    "balanced": (0.5, 0.5, 0.2),
    "aggressive": (1.0, 0.2, 0.1),
}

_WEIGHT_SUM_TOLERANCE = 1e-6


class PortfolioBuildError(ValueError):
    """Raised when analytics or a final portfolio cannot be produced."""


# --------------------------------------------------------------------------- #
# Tool 1: fetch_sector_analytics — raw metrics for the model to reason over
# --------------------------------------------------------------------------- #
def _stock_metrics(
    ticker: str, exchange: str, *, period_days: int, use_live: bool
) -> Optional[Dict[str, object]]:
    """Volatility / P/E / Sharpe / return / price for one stock (None if unresolvable)."""
    from stock_stats import StatsError, get_fundamentals, get_return_statistics

    try:
        stats = get_return_statistics(
            ticker, exchange, period_days=period_days, use_live=use_live
        )
        fundamentals = get_fundamentals(ticker, exchange, use_live=use_live)
        quote = get_stock_quote(ticker, exchange, use_live=use_live)
    except (StatsError, Exception):  # noqa: BLE001 - skip names we cannot resolve
        return None
    return {
        "ticker": ticker,
        "price": round(float(quote.get("price") or 0.0), 2),
        "sharpe_ratio": stats.get("sharpe_ratio"),
        "annualized_volatility_pct": stats.get("annualized_volatility_pct"),
        "trailing_pe": fundamentals.get("trailing_pe"),
        "annualized_return_pct": stats.get("annualized_return_pct"),
        "max_drawdown_pct": stats.get("max_drawdown_pct"),
        "source": quote.get("source"),
    }


def fetch_sector_analytics(
    sectors: Sequence[str] | str,
    exchange: str = "NS",
    *,
    period_days: int = 252,
    use_live: bool = True,
) -> Dict[str, object]:
    """Return raw per-stock metrics (volatility, P/E, Sharpe, return, price).

    This buys and sizes nothing — it exists so the reasoning model can decide the
    allocation from real numbers. Hand the result to the model, let it pick
    ``ticker_weights``, then call :func:`generate_final_portfolio`.
    """
    if isinstance(sectors, str):
        sectors = [s for s in (p.strip() for p in sectors.split(",")) if s]
    keys = [s.strip().lower() for s in sectors if (s or "").strip()]
    if not keys:
        raise PortfolioBuildError("No sectors given.")
    unknown = [s for s in keys if s not in SECTORS]
    if unknown:
        raise PortfolioBuildError(f"Unknown sectors {unknown}. Known: {sorted(SECTORS)}.")

    analytics: Dict[str, List[Dict[str, object]]] = {}
    sources: set = set()
    for sector in keys:
        rows: List[Dict[str, object]] = []
        for ticker in SECTORS[sector]:
            m = _stock_metrics(ticker, exchange, period_days=period_days, use_live=use_live)
            if m is not None:
                rows.append(m)
                sources.add(m["source"])
        analytics[sector] = rows

    if not any(analytics.values()):
        raise PortfolioBuildError("Could not resolve any stocks for the given sectors.")

    return {
        "sectors": keys,
        "exchange": exchange,
        "source": "mixed" if len(sources) > 1 else next(iter(sources), "mock"),
        "analytics": analytics,
        "instructions": (
            "Choose target weights per ticker as fractions of capital (each 0-1). "
            "Reduce or DROP names with a negative Sharpe ratio, and tilt toward "
            "higher risk-adjusted return while keeping the book diversified across "
            "sectors. Weights may sum to less than 1 (the remainder is held as "
            "cash). Then call generate_final_portfolio with your ticker_weights, "
            "the total_amount, and a short reasoning note explaining the mix."
        ),
    }


# --------------------------------------------------------------------------- #
# Deterministic baseline weights (offline fallback only)
# --------------------------------------------------------------------------- #
def _score(metrics: Dict[str, object], risk_profile: str) -> float:
    w_ret, w_vol, w_pe = _RISK_WEIGHTS.get(risk_profile, _RISK_WEIGHTS["balanced"])
    sharpe = float(metrics.get("sharpe_ratio") or 0.0)
    ann_ret = float(metrics.get("annualized_return_pct") or 0.0) / 100.0
    vol = float(metrics.get("annualized_volatility_pct") or 0.0) / 100.0
    pe = float(metrics.get("trailing_pe") or 0.0)
    pe_penalty = max(0.0, (pe - 25.0) / 25.0)
    return sharpe + w_ret * ann_ret - w_vol * vol - w_pe * pe_penalty


def compute_baseline_weights(
    sectors: Optional[Sequence[str]] = None,
    *,
    exchange: str = "NS",
    per_sector: int = 1,
    risk_profile: str = "balanced",
    period_days: int = 252,
    use_live: bool = True,
) -> Dict[str, float]:
    """Deterministic ticker->weight mix (top risk-adjusted pick per sector,
    inverse-volatility weighted, normalized to sum ~1). Offline fallback only."""
    if risk_profile not in _RISK_WEIGHTS:
        raise PortfolioBuildError(
            f"Unknown risk_profile '{risk_profile}'. Use one of {sorted(_RISK_WEIGHTS)}."
        )
    analytics = fetch_sector_analytics(
        sectors or DEFAULT_SECTORS, exchange, period_days=period_days, use_live=use_live
    )["analytics"]

    picks: List[Dict[str, object]] = []
    for rows in analytics.values():  # type: ignore[union-attr]
        ranked = sorted(rows, key=lambda m: _score(m, risk_profile), reverse=True)
        picks.extend(ranked[: max(1, per_sector)])
    if not picks:
        raise PortfolioBuildError("No candidate stocks for baseline weights.")

    inv = [1.0 / max(float(p["annualized_volatility_pct"] or 0.0), 0.1) for p in picks]
    total = sum(inv) or 1.0
    return {str(p["ticker"]): round(iv / total, 4) for p, iv in zip(picks, inv)}


# --------------------------------------------------------------------------- #
# Tool 2: generate_final_portfolio — execute the model's chosen weights
# --------------------------------------------------------------------------- #
def _coerce_weights(ticker_weights: object) -> Dict[str, float]:
    if isinstance(ticker_weights, str):
        try:
            ticker_weights = json.loads(ticker_weights)
        except ValueError as exc:
            raise PortfolioBuildError(f"ticker_weights is not valid JSON: {exc}") from exc
    if not isinstance(ticker_weights, dict) or not ticker_weights:
        raise PortfolioBuildError("ticker_weights must be a non-empty {ticker: weight} map.")
    out: Dict[str, float] = {}
    for tick, w in ticker_weights.items():
        try:
            wf = float(w)
        except (TypeError, ValueError) as exc:
            raise PortfolioBuildError(f"Weight for {tick} is not a number: {w!r}") from exc
        if wf > 0:  # ignore zero / negative (a dropped asset)
            out[str(tick).strip().upper()] = wf
    if not out:
        raise PortfolioBuildError("No positive weights supplied (every asset dropped).")
    return out


def _render_reasoning(
    *,
    name: str,
    rationale: str,
    rows: List[Dict[str, object]],
    total_amount: float,
    total_invested: float,
    cash_remaining: float,
    normalized: bool,
    avg_correlation: Optional[float],
    source: str,
) -> str:
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    out: List[str] = [
        f"# Portfolio reasoning — {name}",
        "",
        f"- Generated: {ts}",
        f"- Capital: ₹{total_amount:,.0f}  |  Deployed: ₹{total_invested:,.0f}  |  "
        f"Cash: ₹{cash_remaining:,.0f}",
        f"- Weights normalized (summed > 1): {normalized}",
        f"- Data source: {source}",
        "",
        "## Model rationale",
        "",
        rationale.strip() or "_No model rationale provided (deterministic baseline)._",
        "",
        "## Allocation (weights chosen by the reasoning model → share math)",
        "",
        "| Ticker | Weight % | Price ₹ | Target ₹ | Shares | Value ₹ |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['ticker']} | {r['weight_pct']} | {r['price']:.2f} | "
            f"{r['target_alloc']:.2f} | {r['quantity']} | {r['current_value']:.2f} |"
        )
    out += [
        "",
        "## Diversification check",
        "",
        (
            f"Average pairwise return correlation of the basket: "
            f"**{avg_correlation}** (lower is better for diversification)."
            if avg_correlation is not None
            else "Correlation check unavailable (need at least two holdings)."
        ),
        "",
        "## Method (baseline mechanics only)",
        "",
        "Each position is sized as `weight × capital`, then rounded DOWN to whole "
        "shares at the current price; the un-deployed remainder is held as cash. "
        "The weighting decision itself was made by the reasoning model from the "
        "sector analytics (volatility / P/E / Sharpe).",
        "",
        "## Caveats",
        "",
        "- Illustrative, educational allocation — NOT personalized advice, and no "
        "trades are executed.",
        "",
    ]
    return "\n".join(out)


def generate_final_portfolio(
    ticker_weights: object,
    total_amount: float,
    *,
    exchange: str = "NS",
    reasoning: str = "",
    name: str = "diversified_portfolio",
    period_days: int = 252,
    use_live: bool = True,
    output_dir: Optional[str] = None,
    write_files: bool = True,
    price_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Size the model's chosen weights into whole shares and write the artifacts.

    ``ticker_weights`` are literal fractions of ``total_amount``; if they sum to
    <1 the remainder is cash, if >1 they are normalized down. Rounds each
    position DOWN to whole shares, writes the CSV and a reasoning markdown file
    (the model's ``reasoning`` plus the share math), and returns the holdings.

    ``price_overrides`` (ticker -> price) forces point-in-time prices instead of
    the live quote — used by the backtester to size at a historical date.
    """
    if total_amount <= 0:
        raise PortfolioBuildError("total_amount must be positive.")
    weights = _coerce_weights(ticker_weights)

    weight_sum = sum(weights.values())
    normalized = weight_sum > 1.0 + _WEIGHT_SUM_TOLERANCE
    if normalized:  # scale down proportionally; never lever above the capital
        weights = {t: w / weight_sum for t, w in weights.items()}

    overrides = {str(k).strip().upper(): float(v) for k, v in (price_overrides or {}).items()}
    rows: List[Dict[str, object]] = []
    sources: set = set()
    for tick, weight in weights.items():
        if tick in overrides:
            price, src = overrides[tick], "as_of"
        else:
            try:
                quote = get_stock_quote(tick, exchange, use_live=use_live)
            except Exception as exc:  # noqa: BLE001
                raise PortfolioBuildError(f"Could not price ticker '{tick}': {exc}") from exc
            price, src = float(quote.get("price") or 0.0), quote.get("source")
        if price <= 0:
            continue
        target = total_amount * weight
        qty = int(target // price)  # round DOWN to whole shares
        rows.append({
            "ticker": tick,
            "asset_class": "Equity",
            "weight": round(weight, 4),
            "weight_pct": round(weight * 100.0, 2),
            "price": round(price, 2),
            "target_alloc": round(target, 2),
            "quantity": qty,
            "current_value": round(qty * price, 2),
            "source": src,
        })
        sources.add(src)

    holdings = [r for r in rows if r["quantity"] > 0]
    if not holdings:
        raise PortfolioBuildError(
            "Every position rounded to zero shares; increase total_amount or weights."
        )

    total_invested = round(sum(h["current_value"] for h in holdings), 2)
    cash_remaining = round(total_amount - total_invested, 2)
    source = "mixed" if len(sources) > 1 else next(iter(sources), "mock")

    from stock_stats import StatsError, get_correlation_matrix

    avg_corr = None
    tickers = [h["ticker"] for h in holdings]
    if len(tickers) >= 2:
        try:
            cm = get_correlation_matrix(
                tickers, exchange, period_days=period_days, use_live=use_live
            )
            pairs = cm.get("ranked_pairs") or []
            if pairs:
                avg_corr = round(sum(p["correlation"] for p in pairs) / len(pairs), 3)
        except StatsError:
            avg_corr = None

    csv_text = "Ticker,Asset Class,Quantity,Current Value\n" + "".join(
        f"{h['ticker']},{h['asset_class']},{h['quantity']},{h['current_value']:.2f}\n"
        for h in holdings
    )
    reasoning_text = _render_reasoning(
        name=name, rationale=reasoning, rows=holdings, total_amount=total_amount,
        total_invested=total_invested, cash_remaining=cash_remaining,
        normalized=normalized, avg_correlation=avg_corr, source=str(source),
    )

    csv_path = reasoning_path = None
    if write_files:
        out_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_p = out_dir / f"{name}.csv"
        reason_p = out_dir / f"{name}.reasoning.md"
        csv_p.write_text(csv_text, encoding="utf-8")
        reason_p.write_text(reasoning_text, encoding="utf-8")
        csv_path, reasoning_path = str(csv_p), str(reason_p)

    return {
        "name": name,
        "holding_count": len(holdings),
        "holdings": holdings,
        "weights_normalized": normalized,
        "total_amount": total_amount,
        "total_invested": total_invested,
        "cash_remaining": cash_remaining,
        "avg_correlation": avg_corr,
        "source": source,
        "csv": csv_text,
        "reasoning": reasoning_text,
        "csv_path": csv_path,
        "reasoning_path": reasoning_path,
    }


# --------------------------------------------------------------------------- #
# OpenAI-format tool schemas (mirror the MCP tools; passed to the LLM payload).
# --------------------------------------------------------------------------- #
FETCH_ANALYTICS_TOOL_SPECS: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_sector_analytics",
            "description": (
                "Return raw per-stock metrics (volatility, P/E, Sharpe ratio, "
                "annualized return, price) for the reference stocks in the given "
                "Indian market sectors. This buys/sizes nothing — use it to gather "
                "the numbers, then YOU decide target weights (dropping or shrinking "
                "negative-Sharpe names) and call generate_final_portfolio."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sectors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Sectors to analyze, e.g. ['it','banking','pharma'].",
                    },
                },
                "required": ["sectors"],
            },
        },
    },
]

GENERATE_PORTFOLIO_TOOL_SPECS: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "generate_final_portfolio",
            "description": (
                "Build the final portfolio from the target weights YOU chose. "
                "Weights are literal fractions of capital (e.g. {'TCS':0.25,"
                "'HDFCBANK':0.2}); if they sum to <1 the remainder is held as cash. "
                "Python sizes each position, rounds DOWN to whole shares, writes the "
                "CSV and a reasoning file (your rationale + the share math), and "
                "returns the holdings and file paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker_weights": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": (
                            "Map of ticker -> target weight (fraction of capital, "
                            "0-1). Omit or zero-weight any asset you drop."
                        ),
                    },
                    "total_amount": {
                        "type": "number",
                        "description": "Total capital to allocate, in INR.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Your explanation for this mix (why these weights, which "
                            "names you dropped and why). Written into the reasoning file."
                        ),
                    },
                },
                "required": ["ticker_weights", "total_amount"],
            },
        },
    },
]

# Convenience: the full two-step portfolio tool set.
PORTFOLIO_TOOL_SPECS: List[Dict[str, object]] = (
    FETCH_ANALYTICS_TOOL_SPECS + GENERATE_PORTFOLIO_TOOL_SPECS
)
