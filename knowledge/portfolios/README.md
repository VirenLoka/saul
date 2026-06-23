# `knowledge/portfolios/` — Customer Portfolio Data

Holds customer portfolio files the agent ingests. Tracked in git for the
**sample** only; real customer files are git-ignored (see root `.gitignore`).

## Expected CSV format

Columns (header names are case- and whitespace-insensitive; common aliases such
as `Symbol`, `Qty`, `Value` are also accepted):

| Ticker | Asset Class | Quantity | Current Value |
|--------|-------------|----------|---------------|
| AAPL   | Equity      | 50       | 10500.00      |

* `Current Value` is the market value of the position (currency symbols and
  thousands separators are tolerated, e.g. `$10,500.00`).
* Recognized asset classes for target-band analysis: `Equity`, `Bond`, `Cash`,
  `Commodity`, `Real Estate`. Others are reported as `untracked`.

`sample_portfolio.csv` is provided for testing and demos. JSON ingestion can be
added later as a sibling loader in `data_ingestion.py` without changing call
sites.
