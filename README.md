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
                                          market_data.py  (yfinance .NS/.BO + mock fallback)
```

Why this shape:
* **Config-driven** — no model names, ports, or paths hardcoded; read once via
  `config_loader.py`.
* **MCP-native tools** — `mcp_server.py` is a standalone server; vLLM executes
  its tools server-side. Adding tools (RAG, news) never touches the CLI.
* **Decoupled, testable core** — allocation math (`analysis.py`) and market-data
  logic (`market_data.py`) are pure Python with no MCP/LLM dependency, so the
  whole suite runs offline with **no forward passes**.
* **Uniform stream contract** — `llm_provider.py` normalizes both the real vLLM
  stream and an offline mock into the same `StreamEvent` sequence, so `cli.py`
  and its tests are backend-agnostic.

## Directory layout

```
saul/
├── config.yaml                 # model_selection, local_inference_settings,
│                               #   mcp, api_credentials, storage_paths, analysis
├── config_loader.py            # parse YAML -> typed, immutable AppConfig
├── portfolio_parser.py         # decoupled CSV -> validated Portfolio
├── market_data.py              # Indian market core (yfinance + mock) + tool schemas
├── mcp_server.py               # FastMCP server wrapping the market-data tools
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
    ├── test_llm_provider.py    # factory + mock stream contract — no model
    └── test_cli.py             # full loop via mock provider — no forward pass
```

## Choosing an engine: vLLM or Ollama

Switch the local engine with one config value — `model_selection.provider`
(`vllm` | `ollama` | `mock`) — and launch the matching engine via
`SPARKS_ENGINE` in `serve.sh`. Both speak the OpenAI-compatible chat API, so the
client code is identical. The one difference:

| | vLLM | Ollama |
|---|------|--------|
| Endpoint | `http://127.0.0.1:8000/v1` | `http://127.0.0.1:11434/v1` |
| Model id | `Qwen/Qwen2.5-7B-Instruct` (HF) | `qwen2.5:7b-instruct` (tag) |
| API key | `SPARKS_API_KEY` (required) | `OLLAMA_API_KEY` (usually unused) |
| **MCP tools** | **server-side via `--tool-server`** | **none** — runs tool-free |

> Ollama has no `--tool-server`, so the live MCP market-data tools are not
> executed server-side in Ollama mode. The CLI detects this (via the provider's
> `supports_server_side_tools` flag), runs the engine tool-free, and still feeds
> it the customer portfolio analysis in the system context. Use vLLM for the
> full live-tool experience.

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

# --- Terminal 1: MCP tool server (SSE on 127.0.0.1:8001 per config.yaml) -----
python mcp_server.py

# --- Terminal 2: vLLM engine with the MCP server attached --------------------
SPARKS_ENGINE=vllm bash serve.sh

# --- Terminal 3: the interactive agent ---------------------------------------
python cli.py                       # config: model_selection.provider: vllm
```

## Execution — Ollama (no GPU / no MCP tools)

```bash
# config.yaml: set model_selection.provider: ollama   (or pass --provider ollama)
SPARKS_ENGINE=ollama bash serve.sh  # pulls qwen2.5:7b-instruct, starts the daemon
python cli.py --provider ollama
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
python3 -m pytest -q          # 40 tests
python3 -m pytest tests/test_market_data.py -q     # market tool logic only
python3 -m pytest tests/test_cli.py -q             # interactive loop + memory
```

## Configuration & secrets

* `model_selection.provider`: `vllm` (default), `ollama`, or `mock`.
* Engine connection: `local_inference_settings.{vllm,ollama}.{model,host,port,api_key}`
  plus shared `{temperature,max_tokens,request_timeout,stream}`. The loader
  resolves the block matching the active provider.
* MCP: `mcp.{host,port,transport,tool_server_url}` and
  `mcp.market_data.{default_exchange,use_live,cache_ttl_seconds}`.
* Secrets are read from env first — `SPARKS_API_KEY` (vLLM), `OLLAMA_API_KEY`
  (Ollama), `OPENAI_API_KEY` / `OPENAI_BASE_URL` (optional external),
  `NEWSAPI_KEY` (future). Never commit real keys; the YAML holds placeholders.

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

## Future integration points

* **Vector DB / RAG** → `knowledge/vector_db/` (see its README): add
  `vector_store.py`, persist to `storage_paths.vector_db`, expose retrieval as
  another MCP tool and/or inject into the system context.
* **Real-time news** → `knowledge/market_data/` (see its README): add NewsAPI /
  scraper fetchers and surface them as new MCP tools in `mcp_server.py`.
