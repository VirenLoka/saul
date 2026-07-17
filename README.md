# MCP-Powered Financial Advisor AI Agent

A **read-only, analytical** financial advisor agent. It loads a customer
portfolio (CSV), computes asset allocation deterministically, and runs an
interactive chat loop against a local **vLLM** engine serving
`Qwen/Qwen2.5-7B-Instruct`. Live **Indian market data** (NSE/BSE) is provided by
a standalone **FastMCP** tool server attached to vLLM via `--tool-server`, so
the model can fetch quotes and sector performance server-side during a turn.

> **Scope guardrail:** strictly observational. It never executes trades, places
> orders, or takes any financial action. Output is general educational
> analysis, not personalized advice.

## Architecture

```
                       ┌──────────────────────────────────────────┐
   config.yaml ──▶ config_loader.py ──▶ AppConfig (typed, frozen)  │
                       └──────────────────────────────────────────┘
                                          │
 knowledge/portfolios/*.csv ─▶ portfolio_parser.py ─▶ Portfolio
                                          │
                                   analysis.py (pure math) ─▶ allocation summary
                                          │
                                   prompts.py (system prompt + portfolio context)
                                          │
   ┌──────────────────────────── cli.py (interactive loop) ───────────────────────┐
   │  • conversational memory (system/user/assistant/tool array)                   │
   │  • streams reasoning, announces each MCP tool call, then the answer           │
   └───────────────────────────────────┬───────────────────────────────────────────┘
                                        │ llm_provider.py (OpenAI-compatible, streaming)
                                        ▼
                              vLLM server  (serve.sh)
                              Qwen/Qwen2.5-7B-Instruct
                              --enable-auto-tool-choice
                              --tool-server ───────────────┐  (server-side tool exec)
                                                            ▼
                                          mcp_server.py  (FastMCP: indian-market-data)
                                                            │
                                  ┌─────────────────────────┴─────────────────────────┐
                                  ▼                                                     ▼
                  market_data.py (yfinance .NS/.BO + mock)        news_data.py (NewsAPI + mock fallback)
```

Why this shape:
* **Config-driven** — no model names, ports, or paths hardcoded; read once via
  `config_loader.py`.
* **MCP-native tools** — `mcp_server.py` is a standalone server; vLLM executes
  its tools server-side. It exposes a live stock quote, sector performance, and
  **`get_stock_news`** (recent NewsAPI articles for a stock, fed back to the
  model as grounding context). Adding tools (RAG, etc.) never touches the CLI.
* **Decoupled, testable core** — allocation math (`analysis.py`), market-data
  logic (`market_data.py`), and news logic (`news_data.py`) are pure Python with
  no MCP/LLM dependency, so the whole suite runs offline with **no forward
  passes**. The news fetch uses only the standard library (`urllib`).
* **Uniform stream contract** — `llm_provider.py` normalizes both the real vLLM
  stream and an offline mock into the same `StreamEvent` sequence, so `cli.py`
  and its tests are backend-agnostic.

## Directory layout

```
saul/
├── config.example.yaml         # committed TEMPLATE (no secrets) — cp to config.yaml
├── config.yaml                 # YOUR live config + keys; gitignored, never committed
│                               #   model_selection, local_inference_settings, mcp,
│                               #   newsapi, api_credentials, storage_paths, analysis
├── config_loader.py            # parse YAML -> typed, immutable AppConfig
├── portfolio_parser.py         # decoupled CSV -> validated Portfolio
├── market_data.py              # Indian market core (yfinance + mock) + tool schemas
├── news_data.py                # stock-news core (NewsAPI + mock) + tool schema
├── mcp_server.py               # FastMCP server wrapping the market-data + news tools
├── llm_provider.py             # OpenAI-compatible streaming client to vLLM + mock
├── analysis.py                 # deterministic allocation / drift / concentration
├── prompts.py                  # agent system prompt + portfolio context builder
├── cli.py                      # interactive chat loop (memory + reasoning/tool display)
├── serve.sh                    # launch vLLM with --tool-server (MCP attached)
├── requirements.txt · .gitignore · README.md
├── knowledge/                  # central data layer
│   ├── portfolios/             # customer portfolio CSVs (sample tracked)
│   │   ├── sample_portfolio.csv
│   │   ├── README.md · .gitkeep
│   ├── vector_db/              # FUTURE: Chroma/FAISS (contents ignored)
│   │   ├── README.md · .gitkeep
│   └── market_data/            # FUTURE: NewsAPI payloads / scrapes (contents ignored)
│       ├── README.md · .gitkeep
└── tests/
    ├── conftest.py
    ├── test_config_loader.py
    ├── test_portfolio_parser.py
    ├── test_analysis.py
    ├── test_market_data.py     # mocked yfinance — no network
    ├── test_news_data.py       # mocked NewsAPI — no network
    ├── test_llm_provider.py    # factory + mock stream contract — no model
    └── test_cli.py             # full loop via mock provider — no forward pass
```

## Choosing an engine: vLLM or Groq

Switch the engine with one config value — `model_selection.provider`
(`vllm` | `groq` | `mock`). All three speak the OpenAI-compatible chat API, so
the client code is identical. They differ in where they run and how tools are
executed:

| | vLLM | Groq |
|---|------|------|
| Runs | self-hosted (**needs a CUDA GPU**) | remote API |
| Endpoint | `http://127.0.0.1:8000/v1` | `https://api.groq.com/openai/v1` |
| Model id | any HF repo id, incl. DeepSeek | Groq-hosted model id |
| API key | `SPARKS_API_KEY` (required) | `GROQ_API_KEY` |
| **MCP tools** | **server-side via `--tool-server`** | client-side (in-process) |

> **vLLM is the only self-hosting path**, and the way DeepSeek models are served
> — point `local_inference_settings.vllm.model` at a DeepSeek repo id (e.g.
> `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`) and launch it with `serve.sh` on a
> GPU host. There is no remote DeepSeek API client.
>
> A remote API can't reach a local `--tool-server`, so under `groq` the CLI
> executes tool calls itself in-process (the provider's
> `supports_server_side_tools` flag is what selects the executor). Both engines
> get the full tool suite; only the execution site differs.

Use `mock` to exercise the whole pipeline offline, with no GPU and no API key.

## Execution — vLLM (full MCP tools)

Three terminals. The MCP server must start **before** vLLM so `--tool-server`
can connect.

```bash
# 0. (once) install — serving host needs vllm + fastmcp + yfinance; client needs openai + pyyaml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "vllm>=0.6.0"          # serving host only

# Shared bearer token (CLI + vLLM must agree)
export SPARKS_API_KEY="$(openssl rand -hex 32)"

# NewsAPI key for the get_stock_news tool (optional; mock headlines without it)
export NEWSAPI_KEY="your-newsapi-key"   # or set newsapi.api_key in config.yaml

# --- Terminal 1: MCP tool server (SSE on 127.0.0.1:8001 per config.yaml) -----
python mcp_server.py

# --- Terminal 2: vLLM engine with the MCP server attached --------------------
# Serves Qwen by default; export SPARKS_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# (plus SPARKS_TOOL_PARSER=deepseek_v3) to serve DeepSeek instead.
bash serve.sh

# --- Terminal 3: the interactive agent ---------------------------------------
python cli.py                       # config: model_selection.provider: vllm
```

## Execution — Groq (no GPU, remote API)

```bash
export GROQ_API_KEY="your-groq-key"   # or set local_inference_settings.groq.api_key
python cli.py --provider groq         # tools run client-side; mcp_server.py not needed
```

## Execution — offline (no GPU, no API key)

```bash
python cli.py --provider mock         # deterministic stub; runs no model
```

Example session:
```
you> How is the IT sector doing, and how does it relate to my equity weight?
🧠 Reasoning: ...
🔧 Invoking MCP tool: get_indian_sector_performance({"sector": "IT"})
📊 Tool result [get_indian_sector_performance]: {...}
💬 Answer: ...
```

In-chat commands: `/help`, `/reset`, `/memory`, `/portfolio`, `/exit`.

## Offline / no-GPU usage

Everything runs without a model, network, or GPU using the deterministic mock
provider and mock market data:

```bash
python cli.py --provider mock                       # interactive, offline
python cli.py --provider mock --once "Quote for Reliance?"   # single turn
```

Set `mcp.market_data.use_live: false` in `config.yaml` to force the MCP server
to serve mock quotes too (no yfinance calls).

## Testing (no forward passes)

The suite uses the mock provider and mocked market data — no model inference and
no live network:

```bash
python3 -m pytest -q          # 191 tests
python3 -m pytest tests/test_market_data.py -q     # market tool logic only
python3 -m pytest tests/test_cli.py -q             # interactive loop + memory
```

## Configuration & secrets

* `model_selection.provider`: `vllm`, `groq`, or `mock`. `vllm` is the only
  self-hosting path (and how DeepSeek is served); `groq` is a remote
  OpenAI-compatible API whose tools run **client-side** (in-process, live data)
  since a remote API can't reach a local `--tool-server`.
* Engine connection: `local_inference_settings.vllm.{model,host,port,api_key}`
  (Groq uses `groq.{model,base_url,api_key}`) plus shared
  `{temperature,max_tokens,request_timeout,stream}`. The loader resolves the
  block matching the active provider.
* Search: `search.{base_url,max_results,language,use_live}` drives the
  autonomous `web_search` tool (SearXNG at `http://localhost:8080` by default).
* MCP: `mcp.{host,port,transport,tllool_server_url}` and
  `mcp.market_data.{default_exchange,use_live,cache_ttl_seconds}`.
* News: `newsapi.{api_key,base_url,page_size,language,sort_by,lookback_days,use_live}`
  drives the `get_stock_news` tool. With no key (or a failed fetch) it returns
  deterministic mock headlines, so the pipeline still runs offline.
* Output length: `local_inference_settings.max_tokens` (default `4096`) caps the
  answer budget; the agent prompt asks for an in-depth, multi-section report
  rather than a few lines.
* Rate-limit / size guards (`local_inference_settings`): remote tiers cap tokens
  per request/minute (Groq's free tier is 8000 TPM). To avoid HTTP 413s and the
  429 retry loop they trigger, the client (a) truncates each tool result fed
  back to the model to `max_tool_result_chars`, (b) trims old history and clamps
  `max_tokens` so prompt + completion stays under `max_request_tokens` (set
  *below* the hard limit, e.g. `7000`; `0` disables — use `0` for self-hosted
  vLLM, which has no such cap), and (c) bounds the client's auto-retries with
  `request_max_retries`. On an unrecoverable rate-limit/size/auth error the CLI
  **aborts the turn** instead of cascading the failure through the next phase.
### Where credentials live

**`config.yaml` is gitignored and is the only file that holds real keys.**
`config.example.yaml` is the committed, placeholder-only template — the app
never reads it. First-time setup:

```bash
cp config.example.yaml config.yaml   # then fill in your keys in config.yaml
```

This makes a key leak structurally impossible via `git add`: the file carrying
the secrets simply isn't tracked. When you add or rename a setting, mirror it
into `config.example.yaml` (minus the secret) so the template stays complete.

Every key is also readable from the environment, which **wins over any file
value**:

| Env var | Used for |
|---|---|
| `SPARKS_API_KEY` | vLLM bearer token (must match `serve.sh`) |
| `GROQ_API_KEY` | remote Groq API |
| `NEWSAPI_KEY` | NewsAPI (wins over `newsapi.api_key`) |
| `NEWSDATA_API_KEY` | newsdata.io archive news (backtesting) |

> Scratch notebooks (`*.ipynb`) are gitignored too — they tend to accumulate
> hardcoded keys in cell source and outputs. Read credentials from
> `config_loader.load_config()` inside notebooks rather than pasting them in.

## Connecting in Docker (the CLI ↔ engine link)

The engine `host`/`port` in `config.yaml` is the **connect** address the CLI
dials — *not* a bind address. `0.0.0.0` is a valid bind for the server but is
**not** a connect target.

* **CLI in the same container/host as vLLM** → `host: 127.0.0.1` (the default).
* **CLI in a different container** → point it at the vLLM service. The cleanest
  way is the env override (no file edit), which wins over `host`/`port`:

  ```bash
  SPARKS_BASE_URL=http://<vllm-host-or-ip>:8000 python cli.py
  ```

The CLI also connects to the **MCP server** to execute tools (see below).
If that server is in another container, override its URL too:

```bash
SPARKS_MCP_URL=http://<mcp-host-or-ip>:8001/sse python cli.py
```

If a turn prints `Could not reach the <engine> server at …`, the URL in that
message is exactly what the CLI dialed — fix the host or set `SPARKS_BASE_URL`.

## Logging

Diagnostics go to **stderr** (never stdout), so they never corrupt the chat UI.

```bash
python cli.py -v                       # DEBUG to stderr (shows the engine URL per request)
python cli.py --log-level INFO         # or pick a level explicitly
python cli.py --log-file agent.log     # also tee logs to a file
SPARKS_LOG_LEVEL=DEBUG python cli.py   # via env
```

The default level is `WARNING` (quiet). The startup line logs the resolved
provider, engine, endpoint, and model — handy for confirming where it connects.

## How a turn works (plan → act → reflect)

Each question runs through a forced reasoning loop so the model thinks before it
answers, and so tool calls actually complete:

1. **🧭 Plan** — the model first lays out a short numbered plan (no tools). This
   is shown live and forces chain-of-thought before any answer-from-memory.
2. **🔧 Act** — the model calls MCP tools to gather live data. *The CLI executes
   each call against the FastMCP server over the MCP protocol*, prints the
   result, feeds it back into the conversation, and lets the model call more
   tools if needed (up to a round cap). This is what makes a tool call resolve
   into a real answer instead of dead-ending.
3. **🔍 Reflection → 💬 Answer** — the model reflects on the gathered data, then
   gives the grounded final answer (split on an `ANSWER:` marker).

Tool execution by engine:
* **vLLM** → calls the live FastMCP server (`mcp.tool_server_url` / `SPARKS_MCP_URL`).
* **Groq** → runs the tools in-process against live data (a remote API can't
  reach a local `--tool-server`).
* **mock** → runs the `market_data` functions in-process (offline, deterministic).

Plan and reflection text are shown but kept out of long-term memory; only the
user message, tool calls/results, and the final answer are persisted (so the
transcript stays valid and re-sendable). The cap is `MAX_TOOL_ROUNDS` in
`cli.py`.

## Analytics, web search & knowledge graphs

Beyond quotes/sector/news, the MCP server exposes (all with a deterministic
mock fallback, so the whole surface runs offline):

* **Statistical metrics** (`stock_stats.py`) — `get_return_statistics`
  (returns/vol/Sharpe/Sortino/drawdown/VaR), `get_technical_indicators`
  (SMA/EMA/RSI/MACD/Bollinger/momentum), `get_risk_metrics` (beta, Jensen's
  alpha, correlation, tracking error vs `^NSEI`), `get_correlation_matrix`, and
  `get_stock_fundamentals`. All derive from the same yfinance history path used
  by the quote tool.
* **Autonomous web search** (`web_search.py`) — `web_search` hits a self-hosted
  **SearXNG** instance (`curl "$SEARXNG/search?q=…&format=json"`), configured
  under `search:`. The model calls it on its own when external facts help.
* **Knowledge graphs** (`sector_graph.py`, `graph_agent.py`, `graph_viz.py`) —
  `build_sector_graph(sectors)` and `build_portfolio_graph()` create a graph
  whose **nodes** are tickers carrying alpha factors, indicators, fundamentals,
  sentiment (news lexicon) and filings features (a `FilingsProvider` ABC — no
  filings backend exists yet, so it returns an explicit placeholder). **Edges**
  are associations seeded with quantitative evidence, then confirmed through
  repeated reason/reflect passes (`propose_graph_edge` / `validate_graph_edge`;
  an edge needs `min_validations` confirms and one reject drops it). The
  portfolio builder runs this loop autonomously — **LLM-driven** when a live
  provider is attached, with a **deterministic evidence heuristic** fallback so
  it works with no model/GPU. Graphs **persist** to `storage_paths.graphs`
  (`knowledge/market_data/graphs/*.json`) so `list_saved_graphs` and
  `get_sector_graph` can query them in a later session — or `get_all_graphs`
  pulls **every** persisted graph at once with a cross-graph index (which graphs
  each ticker is in + the union of validated associations) for reasoning across
  the whole collection. `visualize_sector_graph` renders **Graphviz DOT** (plus
  an SVG if the `dot` binary is present) to eyeball the model's reasoning.
* **Portfolio construction — two steps, the LLM decides** (`portfolio_builder.py`).
  The allocation decision belongs to the reasoning model, not to Python:
  1. `fetch_sector_analytics(sectors)` reads only — it returns raw per-stock
     metrics (volatility, P/E, Sharpe, return, price) and sizes nothing.
  2. The model reasons over those metrics (dropping/shrinking negative-Sharpe
     names) and passes its chosen `ticker_weights` to
     `generate_final_portfolio(ticker_weights, total_amount, reasoning)`, which
     does only the mechanical work: sizes each position, **rounds down to whole
     shares**, and writes the **CSV** plus a **reasoning** markdown file holding
     the model's own rationale + the share math. Weights are literal fractions
     of capital — summing to <1 leaves the remainder as cash. A deterministic
     `compute_baseline_weights` exists purely as an offline/mock fallback.
* **Portfolios** — `sample_portfolio.csv` (diversified) and
  `banking_portfolio.csv` (an n-ticker Indian-banking book) live under
  `knowledge/portfolios/`; the banking universe in `market_data.py` was expanded
  to ~12 names to back it.

## Standalone scripts (outside the chat CLI)

Two entry points drive the pipeline without the interactive loop:

```bash
# 1. Build a graph, run the reason/reflect validation loop, persist + visualize.
python run_graph_reasoning.py --sectors it,banking --mock          # offline
python run_graph_reasoning.py --portfolio knowledge/portfolios/banking_portfolio.csv
python run_graph_reasoning.py --sectors it,banking --llm --provider groq  # LLM-driven

# 2. Have the configured LLM construct a diversified portfolio: it calls
#    fetch_sector_analytics, decides the weights itself, then calls
#    generate_final_portfolio (writes CSV + reasoning under storage_paths.portfolios).
python generate_portfolio.py --provider groq --amount 1000000 --risk balanced
python generate_portfolio.py --provider mock --mock                # offline fallback
```

```bash
# 3. Build a POINT-IN-TIME user-portfolio graph as of a past date (after the
#    training cutoff): nodes = tickers with as-of stats + newsdata.io archive
#    news + sentiment; the LLM validates the associations over that snapshot.
python run_portfolio_graph_asof.py --mock                                    # offline
python run_portfolio_graph_asof.py --provider groq --start 2025-08-08 --end 2025-09-08
```

`run_portfolio_graph_asof.py` (config: `portfolio_graph:`) builds a graph over a
user's portfolio *as of a historical window* (default start `2025-08-08`). Node
features are **point-in-time** statistical metrics (return/vol/Sharpe/momentum/
drawdown from a trailing window ending at the chosen date — no look-ahead, via
the backtest price matrix) plus **newsdata.io archive** articles for the window
with a sentiment score. The LLM then runs the reason/reflect loop over that
snapshot (its node digests carry the as-of stats + recent headlines). `web_search`
is deliberately **off** — its present-day results would be look-ahead relative to
the historical graph. Runs fully offline with `--mock`.

`run_graph_reasoning.py` persists graphs to `storage_paths.graphs`, renders a DOT
(+ image if Graphviz is installed), and **streams the model's edge-validation
output live** (token-by-token under `--llm`). It **excludes the sentiment node
feature by default** (deferred; `--sentiment` re-enables it). `generate_portfolio.py`
runs the two-step flow with the full data/analytics/search tool suite available on
demand (live quotes, news, statistics, `web_search`), streaming all model output;
if no model finalizes a portfolio (e.g. the offline `mock` provider) it falls back
to a deterministic baseline so it always produces output.

## Backtesting (`backtesting/`)

A periodic-rebalancing market simulation that drives the portfolio flow over a
historical window and marks it to market vs a benchmark (`^NSEI`). Config lives
in the `backtesting:` + `newsdata:` sections; flags override.

```bash
python -m backtesting.runner --mock                  # offline, deterministic baseline
python -m backtesting.runner --provider groq         # LLM chooses weights each rebalance
python -m backtesting.runner --provider groq --graph-id <id> --rebalance monthly
```

At each rebalance the engine (`backtesting/engine.py`) computes **point-in-time**
analytics (trailing return/vol/Sharpe — no look-ahead), fetches **archive news**
for the preceding window via **newsdata.io** (`backtesting/news_archive.py`,
`fetch_news_archive` tool), attaches **knowledge-graph peer context** (from a
graph built with sentiment excluded), and lets the **LLM choose the weights**
(deterministic baseline fallback offline / on rate-limit). It sizes at that
date's prices, holds to the next rebalance, and writes an equity-curve CSV + a
markdown report to `backtesting.results_dir`.

Guardrails baked in: **`web_search` is disabled during backtesting**, and the
window start + every news read are **clamped to `newsdata.earliest_date`**
(default `2025-08-05`, the model's training cutoff) so the backtest can't read
anything from before the cutoff. Runs fully offline with `--mock` (mock prices +
deterministic weights, no model/GPU/network).

## Future integration points

* **Vector DB / RAG** → `knowledge/vector_db/` (see its README): add
  `vector_store.py`, persist to `storage_paths.vector_db`, expose retrieval as
  another MCP tool and/or inject into the system context.
* **Financial filings** → implement the `FilingsProvider` ABC in
  `sector_graph.py` against EDGAR / NSE corporate filings and pass it to
  `build_sector_graph`; nodes will then carry filings features automatically.
* **Real-time news** → ✅ implemented: `news_data.py` fetches NewsAPI articles
  for a stock and `mcp_server.py` exposes them as the `get_stock_news` MCP tool
  (configure under the `newsapi` section). Cache payloads to
  `knowledge/market_data/` if you want to persist them.
