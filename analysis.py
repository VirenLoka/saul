"""Deterministic portfolio analytics.

These computations are pure Python and run with **no** LLM and no network.
They produce the quantitative backbone of the report (allocation percentages,
drift versus target bands, concentration) which is then handed to the LLM for
qualitative narrative. Keeping the math here means it is fully unit-testable
and reproducible regardless of which model backend is configured.

Nothing in this module places trades or recommends actions to execute — it is
strictly observational/analytical, per the agent's read-only scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from config_loader import AnalysisSettings
from data_ingestion import Portfolio


@dataclass(frozen=True)
class AllocationLine:
    """Allocation detail for one asset class."""

    asset_class: str
    value: float
    pct: float            # actual share of portfolio, 0-100
    target_pct: float     # configured target, 0-100 (0 if unspecified)
    drift_pct: float      # actual - target (percentage points)
    status: str           # "on-target" | "overweight" | "underweight" | "untracked"


@dataclass(frozen=True)
class AnalysisResult:
    total_value: float
    num_holdings: int
    lines: List[AllocationLine]
    top_holding_ticker: str
    top_holding_pct: float
    missing_classes: List[str]   # target classes absent from the portfolio

    def as_summary_dict(self) -> Dict[str, object]:
        """Compact dict suitable for embedding in an LLM prompt."""
        return {
            "total_value": round(self.total_value, 2),
            "num_holdings": self.num_holdings,
            "allocation": {
                ln.asset_class: {
                    "value": round(ln.value, 2),
                    "pct": round(ln.pct, 2),
                    "target_pct": ln.target_pct,
                    "drift_pct": round(ln.drift_pct, 2),
                    "status": ln.status,
                }
                for ln in self.lines
            },
            "top_holding": {
                "ticker": self.top_holding_ticker,
                "pct": round(self.top_holding_pct, 2),
            },
            "target_classes_missing": self.missing_classes,
        }


def analyze_portfolio(
    portfolio: Portfolio, settings: AnalysisSettings
) -> AnalysisResult:
    """Compute allocation, drift versus target bands, and concentration."""
    total = portfolio.total_value
    by_class = portfolio.value_by_asset_class()
    targets = settings.target_allocation
    tol = settings.drift_tolerance_pct

    lines: List[AllocationLine] = []
    for asset_class, value in sorted(
        by_class.items(), key=lambda kv: kv[1], reverse=True
    ):
        pct = (value / total * 100.0) if total else 0.0
        target_pct = float(targets.get(asset_class, 0.0))
        drift = pct - target_pct
        if target_pct == 0.0:
            status = "untracked"
        elif drift > tol:
            status = "overweight"
        elif drift < -tol:
            status = "underweight"
        else:
            status = "on-target"
        lines.append(
            AllocationLine(
                asset_class=asset_class,
                value=value,
                pct=pct,
                target_pct=target_pct,
                drift_pct=drift,
                status=status,
            )
        )

    # Target classes the portfolio holds none of (potential gaps).
    present = set(by_class)
    missing = [c for c in targets if c not in present]

    # Single-position concentration risk.
    top_ticker, top_pct = "", 0.0
    for h in portfolio.holdings:
        p = (h.current_value / total * 100.0) if total else 0.0
        if p > top_pct:
            top_ticker, top_pct = h.ticker, p

    return AnalysisResult(
        total_value=total,
        num_holdings=len(portfolio.holdings),
        lines=lines,
        top_holding_ticker=top_ticker,
        top_holding_pct=top_pct,
        missing_classes=missing,
    )
