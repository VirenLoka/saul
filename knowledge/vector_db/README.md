# `knowledge/vector_db/` — Embedding Store (future)

Reserved home for the agent's semantic memory / RAG index. **Empty by design
at the MVP stage** — only this README and `.gitkeep` are tracked; all generated
index artifacts are git-ignored (see the repo root `.gitignore`).

## What will live here

* Chroma persistent collections (`chroma.sqlite3`, `*/` segment dirs), **or**
* FAISS index files (`*.faiss`, `*.index`) plus their id/metadata sidecars.

## How it plugs in later (no changes to call sites)

The clean seam already exists: everything reads paths from
`storage_paths.vector_db` in `config.yaml`. To add retrieval:

1. Add a `vector_store.py` module exposing a small interface, e.g.
   `class VectorStore: upsert(docs); query(text, k) -> list[Chunk]`.
2. Implement `ChromaVectorStore` / `FaissVectorStore` behind that interface,
   persisting to `config.storage_paths.vector_db`.
3. In `main.py`, after ingestion, retrieve relevant context (e.g. prior
   analyses, research notes, market summaries) and append it to the user prompt
   built in `prompts.build_user_prompt`.

Because the LLM and prompt layers are already decoupled, wiring RAG in does not
touch `llm_provider.py` or `analysis.py`.

## Suggested config block (add when implementing)

```yaml
vector_db:
  backend: "chroma"          # or "faiss"
  collection: "advisor_kb"
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  top_k: 5
```
