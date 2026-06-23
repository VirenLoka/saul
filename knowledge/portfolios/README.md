# `knowledge/portfolios/` — Customer Portfolio Data

Holds customer portfolio files the agent ingests. Tracked in git for the
**sample** only; real customer files are git-ignored (see root `.gitignore`).

## Expected CSV format

Columns (header names are case- and whitespace-insensitive; common aliases such
as `Symbol`, `Qty`, `Value` are also accepted):

| Ticker   | Asset Class | Quantity | Current Value |
|----------|-------------|----------|---------------|
| RELIANCE | Equity      | 50       | 148000.00     |

* `Current Value` is the market value of the position (currency symbols and
  thousands separators are tolerated, e.g. `₹1,48,000.00` / `$10,500.00`).
* Recognized asset classes for target-band analysis: `Equity`, `Bond`, `Cash`,
  `Commodity`, `Real Estate`. Others are reported as `untracked`.
* Tickers are base symbols (no exchange suffix). The MCP market-data tools
  append `.NS` (NSE) or `.BO` (BSE) at fetch time.

`sample_portfolio.csv` holds Indian instruments (equities, a sovereign gold
bond, liquid/gold ETFs, a REIT, cash) for testing and demos. JSON ingestion can
be added later as a sibling loader in `portfolio_parser.py` without changing
call sites.
