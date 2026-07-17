"""Tests for the SearXNG web-search tool — mock mode only, no network."""

from __future__ import annotations

import pytest

import web_search
from web_search import SEARCH_TOOL_SPECS, WebSearchError, web_search as search


@pytest.fixture(autouse=True)
def _fresh():
    web_search.clear_cache()
    yield
    web_search.clear_cache()


def test_mock_results_are_deterministic():
    a = search("RBI repo rate", use_live=False, max_results=4)
    web_search.clear_cache()
    b = search("RBI repo rate", use_live=False, max_results=4)
    assert a == b
    assert a["source"] == "mock"
    assert a["result_count"] == 4
    assert a["results"][0]["url"].startswith("https://mock.search.local/")


def test_result_shape():
    r = search("nifty outlook", use_live=False, max_results=2)
    for item in r["results"]:
        assert set(item) == {"title", "url", "content", "engine"}


def test_empty_query_raises():
    with pytest.raises(WebSearchError):
        search("   ", use_live=False)


def test_live_falls_back_to_mock_on_unreachable_host():
    # Nothing is listening here; the tool must degrade to mock, not raise.
    r = search(
        "anything",
        base_url="http://127.0.0.1:9",  # unreachable port
        use_live=True,
        max_results=3,
        request_timeout=0.2,
    )
    assert r["source"] == "mock"
    assert r["result_count"] == 3


def test_tool_spec_present():
    names = {s["function"]["name"] for s in SEARCH_TOOL_SPECS}
    assert "web_search" in names
