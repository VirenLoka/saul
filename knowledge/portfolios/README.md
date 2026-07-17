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

Two portfolios are tracked for testing and demos:

* `sample_portfolio.csv` — a diversified Indian book (equities across sectors, a
  sovereign gold bond, liquid/gold ETFs, a REIT, cash).
* `banking_portfolio.csv` — an n-ticker book focused purely on the **Indian
  banking sector** (HDFC/ICICI/SBI/Kotak/Axis/IndusInd/BoB/PNB/IDFC First/
  Federal/AU/Canara). Feed it to `build_portfolio_graph` to autonomously build
  and reason over a banking knowledge graph.

JSON ingestion can be added later as a sibling loader in `portfolio_parser.py`
without changing call sites.
