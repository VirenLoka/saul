"""Historical news for backtesting — newsdata.io archive endpoint.

Fetches date-bounded articles for a query via ``newsdataapi.NewsDataApiClient``'s
``archive_api``. To prevent look-ahead past the model's training cutoff, an
``earliest_date`` floor is enforced: a ``from_date`` before it is clamped up, and
a ``to_date`` before it is rejected outright.

Like the other data modules this is *pure* logic with no MCP/LLM dependency; the
``newsdataapi`` client is lazy-imported, and without a key (or with
``use_live=False``, or on any failure) it degrades to deterministic mock
articles so the backtest still runs offline.

Scope guardrail: read-only. It only retrieves public news.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Dict, List

# Default hard floor — the LLM's training cutoff (configurable via newsdata.earliest_date).
EARLIEST_DEFAULT = "2025-08-05"


class NewsArchiveError(ValueError):
    """Raised for an empty query or a date range that violates the floor."""


@dataclass(frozen=True)
class ArchiveArticle:
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


def _parse_date(value: str) -> _dt.date:
    try:
        return _dt.date.fromisoformat(str(value).strip())
    except (ValueError, TypeError) as exc:
        raise NewsArchiveError(f"Invalid date '{value}' (expected YYYY-MM-DD).") from exc


def clamp_date_range(from_date: str, to_date: str, earliest_date: str) -> tuple:
    """Clamp ``[from_date, to_date]`` to the ``earliest_date`` floor.

    Raises if ``to_date`` is entirely before the floor (no valid window), else
    returns ``(from_date, to_date)`` with ``from_date`` clamped up to the floor.
    """
    floor = _parse_date(earliest_date)
    f, t = _parse_date(from_date), _parse_date(to_date)
    if t < floor:
        raise NewsArchiveError(
            f"to_date {to_date} precedes the earliest allowed date {earliest_date} "
            "— refusing to read news before the model's training cutoff (look-ahead)."
        )
    if f < floor:
        f = floor
    if f > t:
        f = t
    return f.isoformat(), t.isoformat()


# --------------------------------------------------------------------------- #
# Mock generator (deterministic, stable per query+window)
# --------------------------------------------------------------------------- #
_MOCK_TEMPLATES = [
    ("{q}: quarterly numbers and guidance in focus", "Mock Archive Wire",
     "Coverage of {q} discussing results, guidance and demand trends in the window."),
    ("Analysts weigh in on {q} after sector moves", "Mock Archive Times",
     "Brokerage commentary on {q} amid sector rotation and macro cues."),
    ("{q} in the news: regulatory and demand updates", "Mock Archive Daily",
     "A round-up of developments affecting {q} across the requested dates."),
]


def _mock_articles(query: str, from_date: str, to_date: str, n: int) -> List[ArchiveArticle]:
    seed = sum(ord(c) for c in query) or 1
    start = _parse_date(from_date)
    span = max(1, (_parse_date(to_date) - start).days)
    out: List[ArchiveArticle] = []
    for i in range(max(1, n)):
        title_t, src, desc_t = _MOCK_TEMPLATES[(seed + i) % len(_MOCK_TEMPLATES)]
        day = start + _dt.timedelta(days=(seed + i) % span)
        out.append(ArchiveArticle(
            title=title_t.format(q=query),
            source=src,
            published_at=day.isoformat(),
            description=desc_t.format(q=query),
            url=f"https://mock.newsdata.local/{query.lower().replace(' ', '-')}/{i}",
        ))
    return out


# --------------------------------------------------------------------------- #
# Live fetch (newsdata.io archive, lazy-imported)
# --------------------------------------------------------------------------- #
def _normalize(raw: Dict[str, object]) -> ArchiveArticle:
    return ArchiveArticle(
        title=str(raw.get("title") or "").strip(),
        source=str(raw.get("source_id") or "newsdata").strip(),
        published_at=str(raw.get("pubDate") or "").strip(),
        description=str(raw.get("description") or "").strip(),
        url=str(raw.get("link") or "").strip(),
    )


def _live_articles(
    query: str, from_date: str, to_date: str, *,
    api_key: str, language: str, max_articles: int,
) -> List[ArchiveArticle]:
    from newsdataapi import NewsDataApiClient  # lazy import

    api = NewsDataApiClient(apikey=api_key)
    response = api.archive_api(
        q=query, language=language, from_date=from_date, to_date=to_date
    )
    if not isinstance(response, dict) or response.get("status") != "success":
        raise NewsArchiveError(f"newsdata.io error: {response}")
    results = response.get("results") or []
    return [_normalize(a) for a in results[: max(1, int(max_articles))]]


# --------------------------------------------------------------------------- #
# Public API (wrapped as an MCP tool)
# --------------------------------------------------------------------------- #
def fetch_news_archive(
    query: str,
    from_date: str,
    to_date: str,
    *,
    api_key: str = "",
    language: str = "en",
    earliest_date: str = EARLIEST_DEFAULT,
    max_articles: int = 8,
    use_live: bool = True,
    request_timeout: float = 20.0,  # accepted for parity; client manages its own timeout
) -> Dict[str, object]:
    """Return archive news for ``query`` between ``from_date`` and ``to_date``.

    The range is clamped to ``earliest_date`` (no look-ahead before the model's
    training cutoff). Falls back to deterministic mock articles when there is no
    key, ``use_live`` is False, or a live fetch fails.
    """
    q = (query or "").strip()
    if not q:
        raise NewsArchiveError("Empty news query.")
    f_date, t_date = clamp_date_range(from_date, to_date, earliest_date)
    n = max(1, int(max_articles))

    source = "mock"
    articles: List[ArchiveArticle]
    if use_live and api_key:
        try:
            articles = _live_articles(
                q, f_date, t_date,
                api_key=api_key, language=language, max_articles=n,
            )
            source = "live"
            if not articles:
                articles = _mock_articles(q, f_date, t_date, n)
                source = "mock"
        except Exception:  # noqa: BLE001 - degrade to mock so the backtest continues
            articles = _mock_articles(q, f_date, t_date, n)
            source = "mock"
    else:
        articles = _mock_articles(q, f_date, t_date, n)

    return {
        "query": q,
        "from_date": f_date,
        "to_date": t_date,
        "article_count": len(articles),
        "source": source,
        "articles": [a.as_dict() for a in articles],
    }


# --------------------------------------------------------------------------- #
# OpenAI-format tool schema (mirrors the MCP tool; passed to the LLM payload).
# --------------------------------------------------------------------------- #
NEWS_ARCHIVE_TOOL_SPECS: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "fetch_news_archive",
            "description": (
                "Fetch historical news articles for a company/topic between two "
                "dates (YYYY-MM-DD) from the newsdata.io archive — used during "
                "backtesting for point-in-time sentiment/events. Dates are clamped "
                "to the configured earliest date (no look-ahead before the model's "
                "training cutoff). Returns headlines, sources, dates and summaries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Company name or ticker/topic to search for.",
                    },
                    "from_date": {
                        "type": "string",
                        "description": "Start date, YYYY-MM-DD (clamped up to the floor).",
                    },
                    "to_date": {
                        "type": "string",
                        "description": "End date, YYYY-MM-DD (must not precede the floor).",
                    },
                },
                "required": ["query", "from_date", "to_date"],
            },
        },
    },
]
