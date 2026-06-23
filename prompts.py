"""Prompt construction for the financial analyst agent.

Centralizes the system prompt and the user-prompt builder so prompt
engineering stays in one place, separate from orchestration and I/O.
"""

from __future__ import annotations

import json
from typing import Dict

# Robust analyst persona. Note the explicit read-only / non-advisory guardrails,
# which match the agent's observational scope.
SYSTEM_PROMPT = """\
You are a meticulous financial portfolio analyst AI operating in a strictly \
READ-ONLY, observational capacity. You analyze portfolio data and explain what \
you see. You do NOT and CANNOT execute trades, place orders, or take any \
financial action.

Your responsibilities:
1. Interpret the provided allocation figures (already computed for you).
2. Explain the portfolio's diversification profile across asset classes.
3. Identify concentration risk, over-/under-weight classes versus the target \
bands, and any target asset classes that are entirely missing.
4. Offer clear, principle-based diversification considerations (e.g. spreading \
across asset classes, reducing single-name concentration, aligning toward the \
stated target bands).

Rules:
- Ground every statement in the numbers given. Do not invent holdings or prices.
- Be concise and structured. Use short sections and bullet points.
- Frame everything as general educational analysis, not personalized advice.
- Always include a one-line disclaimer that this is not financial advice and no \
trades are being executed.
- Never suggest specific buy/sell orders, quantities, or timing. Speak only in \
terms of allocation principles and observations.
"""


def build_user_prompt(analysis_summary: Dict[str, object]) -> str:
    """Render the computed analysis into a deterministic user prompt."""
    payload = json.dumps(analysis_summary, indent=2, sort_keys=True)
    return (
        "Here is the pre-computed analysis of a customer's portfolio. All "
        "percentages are shares of total portfolio value. 'target_pct' is the "
        "reference target band; 'drift_pct' is actual minus target; 'status' "
        "flags overweight/underweight/on-target/untracked.\n\n"
        f"PORTFOLIO ANALYSIS (JSON):\n{payload}\n\n"
        "Produce a structured report with these sections:\n"
        "1. Allocation Overview\n"
        "2. Diversification Assessment (concentration + drift vs target)\n"
        "3. Gaps & Observations (missing target classes, risks)\n"
        "4. Principle-Based Recommendations (no specific orders)\n"
        "End with the required not-financial-advice disclaimer."
    )
