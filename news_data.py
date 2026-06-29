"""Stock news — decoupled core logic.

Fetches recent news articles relevant to a given stock and returns a compact,
model-friendly summary that can be fed back to the LLM as grounding context.
News comes from NewsAPI (https://newsapi.org/v2/everything); the API key is
supplied via configuration (``newsapi`` section of config.yaml, or env
``NEWSAPI_KEY``).

This module deliberately mirrors ``market_data.py``: it is *pure* logic with no
MCP/FastMCP dependency, so it can be unit-tested without a server. The live HTTP
call uses only the standard library (``urllib``), so importing this module pulls
in no third-party packages.

Data source
-----------
* ``use_live=True`` and an API key present -> real articles via NewsAPI.
* ``use_live=False`` (or no key, or a live fetch fails) -> a deterministic mock
  generator (stable per company), used by the test suite and as an automatic
  offline fallback. The ``source`` field records which path produced the data.

Scope guardrail: read-only. This module only retrieves and summarizes public
news; it never places orders or takes any financial action.
"""

from __future__ import annotations

import datetime as _dt
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Friendly company names for the known Indian tickers. Used to turn a ticker
# (e.g. "TCS") into a natural-language news query ("Tata Consultancy Services"),
# which returns far more relevant articles than the bare symbol.
TICKER_TO_NAME: Dict[str, str] = {
    "RELIANCE": "Reliance Industries",
    "TCS": "Tata Consultancy Services",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "SBIN": "State Bank of India",
    "WIPRO": "Wipro",
    "HCLTECH": "HCL Technologies",
    "ITC": "ITC Limited",
    "LT": "Larsen & Toubro",
    "BHARTIARTL": "Bharti Airtel",
    "MARUTI": "Maruti Suzuki",
    "TATAMOTORS": "Tata Motors",
    "SUNPHARMA": "Sun Pharmaceutical",
    "ASIANPAINT": "Asian Paints",
}

# Default NewsAPI "everything" endpoint; overridable via config.
DEFAULT_BASE_URL = "https://newsapi.org/v2/everything"


class NewsDataError(ValueError):
    """Raised when a stock query cannot be resolved or NewsAPI rejects it."""


@dataclass(frozen=True)
class Article:
    """A single normalized news article (JSON-serializable via ``as_dict``)."""

    title: str
    source: str
    published_at: str
    description: str
    url: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "title": self.title,
            "source": self.source,
            "published_at": self.published_at,
            "description": self.description,
            "url": self.url,
        }


# --------------------------------------------------------------------------- #
# Query resolution
# --------------------------------------------------------------------------- #
def resolve_company_name(query: str) -> str:
    """Resolve a free-text stock query to a human-readable company name.

    Accepts a friendly name ("Reliance"), a ticker ("TCS", "INFY"), or a
    Yahoo-style suffixed symbol ("RELIANCE.NS"). Unknown queries are returned
    as-is so news can still be searched for stocks outside the reference set.
    """
    from market_data import NAME_TO_TICKER  # reuse the existing friendly-name map

    q = (query or "").strip()
    if not q:
        raise NewsDataError("Empty stock query.")

    # Strip a Yahoo exchange suffix if present (RELIANCE.NS -> RELIANCE).
    core = q.rsplit(".", 1)[0] if q.upper().endswith((".NS", ".BO")) else q
    base = NAME_TO_TICKER.get(core.lower(), core.upper())
    return TICKER_TO_NAME.get(base, core)


def build_search_query(company: str) -> str:
    """Build a NewsAPI boolean query biased toward stock-price relevance."""
    return f'"{company}" AND (stock OR shares OR "share price" OR earnings OR results)'


# --------------------------------------------------------------------------- #
# Mock generator (deterministic, stable per company)
# --------------------------------------------------------------------------- #
_MOCK_TEMPLATES: List[Tuple[str, str, str]] = [
    (
        "{c} shares in focus as quarterly results beat street estimates",
        "Mock Markets Daily",
        "{c} reported stronger-than-expected revenue and margins, lifting "
        "sentiment around the stock in early trade.",
    ),
    (
        "Analysts revise price targets on {c} after management commentary",
        "Mock Business Wire",
        "Several brokerages updated their outlook on {c}, citing demand trends "
        "and guidance for the coming quarters.",
    ),
    (
        "{c} stock moves with broader index amid sector rotation",
        "Mock Financial Times",
        "Shares of {c} tracked the wider market as investors rotated across "
        "sectors on macro cues.",
    ),
    (
        "{c} announces capacity expansion; investors weigh capex impact",
        "Mock Economic Review",
        "The company outlined fresh investment plans, prompting debate over "
        "near-term margins versus long-term growth for {c}.",
    ),
    (
        "Institutional flows shift in {c} ahead of earnings",
        "Mock Equity Insights",
        "Positioning data suggests changing institutional interest in {c} as "
        "the next results date approaches.",
    ),
    (
        "{c} in the news: regulatory and demand updates this week",
        "Mock Sector Watch",
        "A round-up of developments affecting {c}, spanning regulation, demand "
        "signals and competitive dynamics.",
    ),
]


def _mock_articles(company: str, n: int) -> List[Article]:
    seed = sum(ord(c) for c in company)
    today = _dt.date.today()
    out: List[Article] = []
    for i in range(max(0, n)):
        title_t, src, desc_t = _MOCK_TEMPLATES[(seed + i) % len(_MOCK_TEMPLATES)]
        day = today - _dt.timedelta(days=(seed + i) % 7)
        out.append(
            Article(
                title=title_t.format(c=company),
                source=src,
                published_at=day.isoformat(),
                description=desc_t.format(c=company),
                url=f"https://news.example.com/{company.lower().replace(' ', '-')}/{i}",
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Live fetch (NewsAPI via urllib, lazy-imported)
# --------------------------------------------------------------------------- #
def _normalize_article(raw: Dict[str, object]) -> Article:
    src = raw.get("source") or {}
    source_name = src.get("name") if isinstance(src, dict) else None
    return Article(
        title=str(raw.get("title") or "").strip(),
        source=str(source_name or "Unknown").strip(),
        published_at=str(raw.get("publishedAt") or "").strip(),
        description=str(raw.get("description") or "").strip(),
        url=str(raw.get("url") or "").strip(),
    )


def _live_articles(
    query_str: str,
    *,
    api_key: str,
    base_url: str,
    page_size: int,
    language: str,
    sort_by: str,
    lookback_days: int,
    timeout: float,
) -> List[Article]:
    import json
    import urllib.parse
    import urllib.request

    from_date = (_dt.date.today() - _dt.timedelta(days=max(0, lookback_days))).isoformat()
    params = {
        "q": query_str,
        "language": language,
        "sortBy": sort_by,
        "pageSize": max(1, min(int(page_size), 100)),
        "from": from_date,
        "apiKey": api_key,
    }
    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "saul-financial-advisor/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed host
        payload = json.loads(resp.read().decode("utf-8"))

    if payload.get("status") != "ok":
        raise NewsDataError(
            f"NewsAPI error: {payload.get('message', 'unknown error')}"
        )
    raw_articles = payload.get("articles") or []
    return [_normalize_article(a) for a in raw_articles[: int(page_size)]]


# --------------------------------------------------------------------------- #
# Small in-process TTL cache (mirrors market_data)
# --------------------------------------------------------------------------- #
_CACHE: Dict[str, Tuple[float, Dict[str, object]]] = {}


def _cached(key: str, ttl: int) -> Optional[Dict[str, object]]:
    hit = _CACHE.get(key)
    if hit and (time.monotonic() - hit[0]) < ttl:
        return hit[1]
    return None


def clear_cache() -> None:
    """Reset the news cache (used by tests)."""
    _CACHE.clear()


# --------------------------------------------------------------------------- #
# Public API (wrapped as an MCP tool in mcp_server.py)
# --------------------------------------------------------------------------- #
def get_stock_news(
    query: str,
    *,
    api_key: str = "",
    base_url: str = DEFAULT_BASE_URL,
    page_size: int = 8,
    language: str = "en",
    sort_by: str = "publishedAt",
    lookback_days: int = 7,
    use_live: bool = True,
    cache_ttl_seconds: int = 300,
    request_timeout: float = 20.0,
) -> Dict[str, object]:
    """Return recent news articles relevant to a stock, as model context.

    Resolves ``query`` (company name or ticker) to a readable company name,
    searches NewsAPI for stock-price-relevant coverage, and returns a compact,
    JSON-serializable summary. Falls back to deterministic mock headlines when
    ``use_live`` is False, no API key is configured, or a live fetch fails — the
    ``source`` field records which path produced the data.
    """
    company = resolve_company_name(query)
    search_query = build_search_query(company)
    n = max(1, int(page_size))

    go_live = bool(use_live and api_key)
    cache_key = f"n:{company}:{n}:{language}:{sort_by}:{lookback_days}:{'live' if go_live else 'mock'}"
    cached = _cached(cache_key, cache_ttl_seconds)
    if cached is not None:
        return cached

    source = "mock"
    articles: List[Article]
    if go_live:
        try:
            articles = _live_articles(
                search_query,
                api_key=api_key,
                base_url=base_url or DEFAULT_BASE_URL,
                page_size=n,
                language=language,
                sort_by=sort_by,
                lookback_days=lookback_days,
                timeout=request_timeout,
            )
            source = "live"
        except Exception:  # noqa: BLE001 - degrade gracefully to mock headlines
            articles = _mock_articles(company, n)
            source = "mock"
    else:
        articles = _mock_articles(company, n)

    result: Dict[str, object] = {
        "query": (query or "").strip(),
        "company": company,
        "search_query": search_query,
        "article_count": len(articles),
        "source": source,
        "articles": [a.as_dict() for a in articles],
    }
    _CACHE[cache_key] = (time.monotonic(), result)
    return result


# --------------------------------------------------------------------------- #
# OpenAI-format tool schema (mirrors the MCP tool; passed to the LLM payload).
# Kept here, next to the implementation, to avoid schema drift.
# --------------------------------------------------------------------------- #
NEWS_TOOL_SPECS: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_news",
            "description": (
                "Fetch recent news articles relevant to a single stock's price "
                "(by company name e.g. 'Reliance' or ticker e.g. 'TCS') and "
                "return them as grounding context: headlines, sources, dates and "
                "summaries. Use this to factor recent events and sentiment into "
                "the analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Company name or ticker symbol.",
                    },
                    "max_articles": {
                        "type": "integer",
                        "description": (
                            "Optional cap on the number of articles to return "
                            "(defaults to the configured page size)."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
]
