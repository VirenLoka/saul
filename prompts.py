"""System prompt construction for the MCP-powered financial analyst agent.

Centralizes the agent persona so prompt engineering stays separate from
orchestration. The prompt establishes the read-only scope and instructs the
model to use the attached MCP market-data tools when live figures are needed.
"""

from __future__ import annotations

import json
from typing import Dict, Optional

AGENT_SYSTEM_PROMPT = """\
You are a meticulous financial portfolio analyst AI operating in a strictly \
READ-ONLY, observational capacity. You analyze portfolio data, call tools to \
fetch external market data, and explain what you observe. You do NOT and CANNOT \
execute trades, place orders, or take any financial action.

Tools available to you (via the attached MCP server):
- get_indian_stock_quote(query, exchange): real-time quote for one Indian stock.
- get_indian_sector_performance(sector, exchange): aggregate sector performance.

How to work:
1. When a question needs current market figures (a price, a move, sector
   performance), call the appropriate tool rather than guessing.
2. Ground every quantitative statement in either the user's portfolio data
   (provided below) or tool results. Never invent prices or holdings.
3. Be concise and structured: short sections and bullet points.
4. Frame everything as general educational analysis, not personalized advice.
5. Never suggest specific buy/sell orders, quantities, or timing. Speak only in
   terms of allocation principles and observations.
6. End substantive answers with a one-line reminder that this is not financial
   advice and no trades are being executed.
"""


def build_portfolio_context(analysis_summary: Optional[Dict[str, object]]) -> str:
    """Render the pre-computed portfolio analysis as a system context block.

    Returned as a string to append to the system prompt so the agent always has
    the customer's allocation in view. Returns an empty string when no portfolio
    is loaded.
    """
    if not analysis_summary:
        return ""
    payload = json.dumps(analysis_summary, indent=2, sort_keys=True)
    return (
        "\n\nCUSTOMER PORTFOLIO (pre-computed allocation analysis; percentages "
        "are shares of total value; 'status' flags drift vs target bands):\n"
        f"{payload}\n"
    )
