"""Terminal report rendering.

Renders the deterministic quantitative analysis as a clean, dependency-free
ASCII report, then appends the LLM's qualitative narrative. Kept separate from
analysis (the math) and orchestration (main.py) so output formatting can evolve
independently.
"""

from __future__ import annotations

from analysis import AnalysisResult

_STATUS_MARK = {
    "on-target": "OK ",
    "overweight": "OVER",
    "underweight": "UNDR",
    "untracked": "??? ",
}


def _bar(pct: float, width: int = 20) -> str:
    filled = int(round((pct / 100.0) * width))
    filled = max(0, min(width, filled))
    return "#" * filled + "." * (width - filled)


def render_report(
    result: AnalysisResult, source: str, provider_name: str, llm_narrative: str
) -> str:
    """Assemble the full terminal report as a single string."""
    lines: list[str] = []
    w = 72
    lines.append("=" * w)
    lines.append("  FINANCIAL ADVISOR AI AGENT  —  PORTFOLIO ANALYSIS (READ-ONLY)")
    lines.append("=" * w)
    lines.append(f"  Source       : {source}")
    lines.append(f"  LLM backend  : {provider_name}")
    lines.append(f"  Holdings     : {result.num_holdings}")
    lines.append(f"  Total value  : ${result.total_value:,.2f}")
    lines.append("-" * w)
    lines.append("  ASSET ALLOCATION")
    lines.append("-" * w)
    header = f"  {'Asset Class':<16}{'Value':>14}{'Actual':>9}{'Target':>9}  Status"
    lines.append(header)
    for ln in result.lines:
        lines.append(
            f"  {ln.asset_class:<16}"
            f"{('$' + format(ln.value, ',.0f')):>14}"
            f"{ln.pct:>8.1f}%"
            f"{ln.target_pct:>8.0f}%"
            f"  [{_STATUS_MARK.get(ln.status, '?')}] {_bar(ln.pct)}"
        )
    lines.append("-" * w)
    lines.append(
        f"  Largest single position: {result.top_holding_ticker} "
        f"({result.top_holding_pct:.1f}% of portfolio)"
    )
    if result.missing_classes:
        lines.append(
            "  Target classes not held : " + ", ".join(result.missing_classes)
        )
    lines.append("=" * w)
    lines.append("  ANALYST NARRATIVE (LLM)")
    lines.append("=" * w)
    lines.append(llm_narrative.strip() or "(no narrative returned)")
    lines.append("=" * w)
    lines.append(
        "  NOTE: Observational analysis only. Not financial advice. "
        "No trades executed."
    )
    lines.append("=" * w)
    return "\n".join(lines)
