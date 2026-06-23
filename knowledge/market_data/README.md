# `knowledge/market_data/` — Market Data Drop Zone (future)

Destination for external market intelligence the agent will consume for
context. **Empty by design at the MVP stage** — only this README and `.gitkeep`
are tracked; fetched payloads and scraped files are git-ignored.

## What will live here

* `news/` — raw NewsAPI JSON payloads (`*.json`), one file per fetch.
* `research/` — web-scraped articles converted to Markdown (`*.md`).
* `prices/` — cached quote snapshots (`*.json` / `*.csv`).
* `*.log` — temporary ingestion logs (always ignored).

## How it plugs in later (no changes to call sites)

Paths come from `storage_paths.market_data` in `config.yaml`, so the seam is
already in place:

1. Add a `market_data_ingest.py` module with fetchers, e.g.
   `fetch_news(tickers) -> list[Path]` (NewsAPI) and
   `scrape_to_markdown(url) -> Path`. Write outputs under
   `config.storage_paths.market_data`.
2. Add a loader that reads the drop zone into typed objects (mirroring how
   `data_ingestion.load_portfolio` works for CSVs).
3. Optionally embed these documents into `knowledge/vector_db/` for retrieval,
   then surface the most relevant snippets in the prompt.

Real-time API keys (e.g. NewsAPI) follow the same secret hygiene as the LLM
providers: supply them via environment variables, never commit them.

## Suggested config block (add when implementing)

```yaml
market_data:
  newsapi_key: ""            # or env NEWSAPI_KEY
  refresh_interval_minutes: 60
  watchlist: ["AAPL", "MSFT", "NVDA"]
```
