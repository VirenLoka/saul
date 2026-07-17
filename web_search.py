"""Autonomous web search — decoupled core logic (SearXNG backend).

Lets the agent look things up on the open web when its market/news/graph tools
and portfolio context are not enough (e.g. a macro event, a regulatory change,
a company the reference data doesn't cover). It targets a self-hosted **SearXNG**
instance, which exposes a JSON API equivalent to::

    curl "http://localhost:8080/search?q=<query>&format=json"

Like the other core modules this file is *pure* logic with no MCP/FastMCP
dependency and uses only the standard library (``urllib``), so importing it
pulls in no third-party packages and it is unit-testable without a server.

Data source
-----------
* ``use_live=True`` and a reachable instance -> real SearXNG results.
* ``use_live=False`` (or the instance is unreachable / errors) -> a
  deterministic mock generator (stable per query), used by the test suite and
  as an automatic offline fallback. The ``source`` field records which path
  produced the data.

Scope guardrail: read-only. This module only retrieves public search results;
it never places orders or takes any financial action.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DEFAULT_BASE_URL = "http://localhost:8080"


class WebSearchError(ValueError):
    """Raised when a query is empty or the search backend rejects it."""


@dataclass(frozen=True)
class SearchResult:
    """A single normalized search result (JSON-serializable via ``as_dict``)."""

    title: str
    url: str
    content: str
    engine: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "engine": self.engine,
        }


# --------------------------------------------------------------------------- #
# Mock generator (deterministic, stable per query)
# --------------------------------------------------------------------------- #
def _mock_results(query: str, n: int) -> List[SearchResult]:
    seed = sum(ord(c) for c in query) or 1
    slug = query.strip().lower().replace(" ", "-") or "query"
    out: List[SearchResult] = []
    for i in range(max(1, n)):
        out.append(
            SearchResult(
                title=f"[mock] {query} — result {i + 1}",
                url=f"https://mock.search.local/{slug}/{(seed + i) % 1000}",
                content=(
                    f"Deterministic offline snippet #{i + 1} for '{query}'. "
                    "No live SearXNG instance was reached; this is mock context "
                    "so the pipeline still runs without network."
                ),
                engine="mock",
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Live fetch (SearXNG JSON API via urllib, lazy-imported)
# --------------------------------------------------------------------------- #
def _normalize_result(raw: Dict[str, object]) -> SearchResult:
    engines = raw.get("engines")
    engine = (
        ", ".join(str(e) for e in engines) if isinstance(engines, list)
        else str(raw.get("engine") or "searxng")
    )
    return SearchResult(
        title=str(raw.get("title") or "").strip(),
        url=str(raw.get("url") or "").strip(),
        content=str(raw.get("content") or "").strip(),
        engine=engine,
    )


def _live_results(
    query: str,
    *,
    base_url: str,
    max_results: int,
    language: str,
    timeout: float,
) -> List[SearchResult]:
    import json
    import urllib.parse
    import urllib.request

    params = {"q": query, "format": "json", "language": language}
    url = f"{base_url.rstrip('/')}/search?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "saul-financial-advisor/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - configured host
        payload = json.loads(resp.read().decode("utf-8"))

    raw_results = payload.get("results") or []
    return [_normalize_result(r) for r in raw_results[: max(1, int(max_results))]]


# --------------------------------------------------------------------------- #
# Small in-process TTL cache (mirrors news_data / market_data)
# --------------------------------------------------------------------------- #
_CACHE: Dict[str, Tuple[float, Dict[str, object]]] = {}


def _cached(key: str, ttl: int) -> Optional[Dict[str, object]]:
    hit = _CACHE.get(key)
    if hit and (time.monotonic() - hit[0]) < ttl:
        return hit[1]
    return None


def clear_cache() -> None:
    """Reset the search cache (used by tests)."""
    _CACHE.clear()


# --------------------------------------------------------------------------- #
# Public API (wrapped as an MCP tool in mcp_server.py)
# --------------------------------------------------------------------------- #
def web_search(
    query: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    max_results: int = 6,
    language: str = "en",
    use_live: bool = True,
    cache_ttl_seconds: int = 300,
    request_timeout: float = 15.0,
) -> Dict[str, object]:
    """Search the web via SearXNG and return compact results as model context.

    Falls back to deterministic mock results when ``use_live`` is False or a
    live fetch fails — the ``source`` field records which path produced the
    data, so the model can weigh how much to trust it.
    """
    q = (query or "").strip()
    if not q:
        raise WebSearchError("Empty search query.")
    n = max(1, int(max_results))

    cache_key = f"w:{q}:{n}:{language}:{'live' if use_live else 'mock'}"
    cached = _cached(cache_key, cache_ttl_seconds)
    if cached is not None:
        return cached

    source = "mock"
    results: List[SearchResult]
    if use_live:
        try:
            results = _live_results(
                q,
                base_url=base_url or DEFAULT_BASE_URL,
                max_results=n,
                language=language,
                timeout=request_timeout,
            )
            source = "live"
            if not results:  # instance reachable but nothing found
                results = _mock_results(q, n)
                source = "mock"
        except Exception:  # noqa: BLE001 - degrade gracefully to mock results
            results = _mock_results(q, n)
            source = "mock"
    else:
        results = _mock_results(q, n)

    result: Dict[str, object] = {
        "query": q,
        "result_count": len(results),
        "source": source,
        "results": [r.as_dict() for r in results],
    }
    _CACHE[cache_key] = (time.monotonic(), result)
    return result


# --------------------------------------------------------------------------- #
# OpenAI-format tool schema (mirrors the MCP tool; passed to the LLM payload).
# --------------------------------------------------------------------------- #
SEARCH_TOOL_SPECS: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the open web (via a self-hosted SearXNG instance) for "
                "information not covered by the market/news/graph tools or the "
                "portfolio context — e.g. macro events, regulatory changes, or "
                "companies outside the reference set. Returns titles, URLs and "
                "snippets. Use autonomously whenever current external facts "
                "would improve the analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (natural language or keywords).",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Optional cap on results (defaults to the configured max).",
                    },
                },
                "required": ["query"],
            },
        },
    },
]
