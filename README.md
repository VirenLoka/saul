# Financial Advisor AI Agent — MVP

A **read-only, analytical** financial advisor agent. It ingests a customer
portfolio (CSV), computes asset allocation deterministically, asks a configured
LLM for qualitative diversification commentary, and prints a structured terminal
report.

> **Scope guardrail:** This agent is strictly observational. It does **not**
> execute trades, place orders, or take any financial action. Output is general
> educational analysis, not personalized financial advice.

## Architecture at a glance

Clean, decoupled layers — each swappable without touching the others:

```
config.yaml ──▶ config_loader.py ──▶ AppConfig (typed, immutable)
                                        │
   knowledge/portfolios/*.csv ──▶ data_ingestion.py ──▶ Portfolio
                                        │
                                  analysis.py (pure math, no LLM) ──▶ AnalysisResult
                                        │
                                  prompts.py (system + user prompt)
                                        │
                                  llm_provider.py  ◀── toggled by config
                                  ┌── local: Qwen/Qwen2.5-7B-Instruct
                                  │     (ollama | vllm | transformers)
                                  ├── openai  (fallback)
                                  ├── anthropic (fallback)
                                  └── mock (offline, tests/CI)
                                        │
                                  report.py ──▶ terminal report
                                        ▲
                                  main.py orchestrates the loop
```

Why this shape:
* **Config-driven** — no model names or paths hardcoded; everything in
  `config.yaml`, read once via `config_loader.py`.
* **Provider abstraction** — `llm_provider.LLMProvider` is the only seam the app
  depends on; switching Qwen ↔ OpenAI ↔ Anthropic is a one-line config change.
* **Deterministic core** — all allocation math lives in `analysis.py`, fully
  unit-tested without any model.
* **Future-proof knowledge layer** — `knowledge/vector_db/` (RAG) and
  `knowledge/market_data/` (NewsAPI/scrapes) are scaffolded with plug-in docs.

## Directory layout

```
saul/
├── config.yaml                 # central config: model_selection, local_inference_settings,
│                               #   api_credentials (placeholders), storage_paths, analysis
├── config_loader.py            # parse YAML -> typed, immutable AppConfig (env vars win for secrets)
├── llm_provider.py             # abstract LLMProvider + Local Qwen / OpenAI / Anthropic / Mock + factory
├── data_ingestion.py           # decoupled CSV -> Portfolio (validated, alias-tolerant)
├── analysis.py                 # deterministic allocation / drift / concentration analytics
├── prompts.py                  # system prompt + user-prompt builder
├── report.py                   # terminal report renderer (dependency-free)
├── main.py                     # orchestration entrypoint (CLI)
├── requirements.txt
├── .gitignore
├── README.md
├── knowledge/                  # central data layer
│   ├── portfolios/             # customer portfolio CSVs/JSONs
│   │   ├── sample_portfolio.csv    # sample for testing/demo (tracked)
│   │   ├── README.md
│   │   └── .gitkeep
│   ├── vector_db/              # FUTURE: Chroma/FAISS embeddings (contents ignored)
│   │   ├── README.md
│   │   └── .gitkeep
│   └── market_data/            # FUTURE: NewsAPI payloads + scraped markdown (contents ignored)
│       ├── README.md
│       └── .gitkeep
└── tests/
    ├── conftest.py
    ├── test_config_loader.py
    ├── test_data_ingestion.py
    ├── test_analysis.py
    ├── test_llm_provider.py    # factory toggle + offline mock (no model run)
    └── test_smoke.py           # full pipeline via mock provider
```

## Quick start

```bash
# 1. (optional) virtual env
python3 -m venv .venv && source .venv/bin/activate

# 2. install core deps
pip install -r requirements.txt

# 3a. run OFFLINE (no model, deterministic stub) — works anywhere
python main.py --provider mock

# 3b. run against local Qwen via Ollama (requires the model pulled & daemon up)
#     ollama pull qwen2.5:7b-instruct
python main.py                       # provider defaults to 'local' in config.yaml

# analyze a specific portfolio
python main.py --portfolio knowledge/portfolios/sample_portfolio.csv
```

## Switching the LLM backend

Edit `model_selection.provider` in `config.yaml`:

| provider    | uses                                              |
|-------------|---------------------------------------------------|
| `local`     | `Qwen/Qwen2.5-7B-Instruct` via `runner` (ollama/vllm/transformers) |
| `openai`    | `OPENAI_API_KEY` env var (preferred) + `openai_model` |
| `anthropic` | `ANTHROPIC_API_KEY` env var (preferred) + `anthropic_model` |
| `mock`      | offline deterministic stub (tests/CI, no network) |

Secrets are read from environment variables first; never commit real keys.

## Testing

Tests use the offline `mock` provider and pure-Python analytics — **no model
forward passes** are performed.

```bash
python3 -m pytest -q          # 23 tests
```

## `.gitignore` policy (summary)

Ignores `.venv`/`venv`, Python caches, secret files (`.env`, `*.secret`,
`config.local.yaml`), local data caches, and the *contents* of
`knowledge/vector_db/` and `knowledge/market_data/` plus `*.log` — while keeping
the folder skeleton (`.gitkeep` + `README.md`) and the sample portfolio tracked.

## Future integration points

* **Vector DB / RAG** → `knowledge/vector_db/` (see its README). Add
  `vector_store.py` behind a small interface, persist to
  `storage_paths.vector_db`, retrieve context into `prompts.build_user_prompt`.
* **Real-time market data** → `knowledge/market_data/` (see its README). Add
  `market_data_ingest.py` (NewsAPI + scraper), optionally embed into the vector
  store. No changes needed to `llm_provider.py` or `analysis.py`.
