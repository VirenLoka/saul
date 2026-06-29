"""Unit tests for the stock-news core.

All tests run offline (no network / NewsAPI key) — the module degrades to a
deterministic mock generator. Two tests monkeypatch the live path to confirm
both the live route and graceful fallback.
"""

from __future__ import annotations

import pytest

import news_data
from news_data import (
    NEWS_TOOL_SPECS,
    Article,
    NewsDataError,
    build_search_query,
    clear_cache,
    get_stock_news,
    resolve_company_name,
)


def setup_function(_):
    clear_cache()


def test_resolve_company_name_friendly_and_ticker():
    assert resolve_company_name("Reliance") == "Reliance Industries"
    assert resolve_company_name("TCS") == "Tata Consultancy Services"
    assert resolve_company_name("INFY") == "Infosys"


def test_resolve_company_name_strips_yahoo_suffix():
    assert resolve_company_name("RELIANCE.NS") == "Reliance Industries"


def test_resolve_company_name_unknown_passthrough():
    # Unknown stocks are searched by the raw query (not in the reference map).
    assert resolve_company_name("Zomato") == "Zomato"


def test_resolve_empty_raises():
    with pytest.raises(NewsDataError):
        resolve_company_name("   ")


def test_build_search_query_biases_toward_price():
    q = build_search_query("Reliance Industries")
    assert "Reliance Industries" in q
    assert "stock" in q


def test_mock_news_is_deterministic():
    n1 = get_stock_news("TCS", use_live=False, page_size=5)
    clear_cache()
    n2 = get_stock_news("TCS", use_live=False, page_size=5)
    assert n1 == n2
    assert n1["source"] == "mock"
    assert n1["company"] == "Tata Consultancy Services"
    assert n1["article_count"] == 5
    assert len(n1["articles"]) == 5
    # Each article carries the model-facing fields.
    art = n1["articles"][0]
    assert set(art) == {"title", "source", "published_at", "description", "url"}
    assert "Tata Consultancy Services" in art["title"]


def test_no_api_key_forces_mock_even_when_use_live_true():
    out = get_stock_news("Reliance", api_key="", use_live=True)
    assert out["source"] == "mock"
    assert out["article_count"] == 8  # default page_size


def test_live_path_used_when_key_present(monkeypatch):
    def fake_live(query_str, **kwargs):
        return [
            Article(
                title="Live headline for the stock",
                source="Reuters",
                published_at="2026-06-24T00:00:00Z",
                description="A live description.",
                url="https://example.com/a",
            )
        ]

    monkeypatch.setattr(news_data, "_live_articles", fake_live)
    out = get_stock_news("Reliance", api_key="secret", use_live=True, page_size=1)
    assert out["source"] == "live"
    assert out["article_count"] == 1
    assert out["articles"][0]["source"] == "Reuters"


def test_live_failure_falls_back_to_mock(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(news_data, "_live_articles", boom)
    out = get_stock_news("Reliance", api_key="secret", use_live=True)
    assert out["source"] == "mock"  # degraded gracefully


def test_news_tool_specs_shape():
    names = {t["function"]["name"] for t in NEWS_TOOL_SPECS}
    assert names == {"get_stock_news"}
    for t in NEWS_TOOL_SPECS:
        assert t["type"] == "function"
        assert "query" in t["function"]["parameters"]["properties"]
        assert t["function"]["parameters"]["required"] == ["query"]
