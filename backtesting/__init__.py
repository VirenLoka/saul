"""Backtesting package — periodic-rebalancing market simulation.

Drives the portfolio-generation flow over a historical window (starting no
earlier than the model's training cutoff), using point-in-time analytics,
newsdata.io archive news, and optional knowledge-graph context, and marks the
book to market against a benchmark.

Read-only/analytical: it simulates, it never trades.
"""

from backtesting.engine import BacktestError, run_backtest  # noqa: F401
from backtesting.news_archive import (  # noqa: F401
    NEWS_ARCHIVE_TOOL_SPECS,
    NewsArchiveError,
    fetch_news_archive,
)

__all__ = [
    "run_backtest",
    "BacktestError",
    "fetch_news_archive",
    "NewsArchiveError",
    "NEWS_ARCHIVE_TOOL_SPECS",
]
