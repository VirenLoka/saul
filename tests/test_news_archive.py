"""Tests for the newsdata.io archive news tool — mock only, no network.

The training-cutoff floor (no look-ahead) is the key invariant here.
"""

from __future__ import annotations

import json

import pytest

from backtesting.news_archive import (
    NEWS_ARCHIVE_TOOL_SPECS,
    NewsArchiveError,
    clamp_date_range,
    fetch_news_archive,
)

FLOOR = "2025-08-05"


class TestDateFloor:
    def test_from_date_before_floor_is_clamped_up(self):
        f, t = clamp_date_range("2025-01-01", "2025-09-01", FLOOR)
        assert f == FLOOR
        assert t == "2025-09-01"

    def test_to_date_before_floor_raises(self):
        with pytest.raises(NewsArchiveError):
            clamp_date_range("2025-01-01", "2025-07-01", FLOOR)

    def test_fetch_clamps_from_date(self):
        r = fetch_news_archive(
            "TCS", "2025-01-01", "2025-09-01", earliest_date=FLOOR, use_live=False
        )
        assert r["from_date"] == FLOOR
        # Every mock article falls within the (clamped) window.
        for a in r["articles"]:
            assert FLOOR <= a["published_at"] <= "2025-09-01"

    def test_fetch_rejects_window_entirely_before_floor(self):
        with pytest.raises(NewsArchiveError):
            fetch_news_archive("TCS", "2025-01-01", "2025-06-01", earliest_date=FLOOR, use_live=False)


class TestMock:
    def test_deterministic(self):
        a = fetch_news_archive("Reliance", "2025-08-10", "2025-08-20", earliest_date=FLOOR, use_live=False)
        b = fetch_news_archive("Reliance", "2025-08-10", "2025-08-20", earliest_date=FLOOR, use_live=False)
        assert a == b
        assert a["source"] == "mock"
        assert a["article_count"] >= 1

    def test_respects_max_articles(self):
        r = fetch_news_archive(
            "INFY", "2025-08-10", "2025-08-30", earliest_date=FLOOR,
            max_articles=2, use_live=False,
        )
        assert r["article_count"] == 2

    def test_empty_query_raises(self):
        with pytest.raises(NewsArchiveError):
            fetch_news_archive("  ", "2025-08-10", "2025-08-20", earliest_date=FLOOR, use_live=False)


class TestToolIntegration:
    def test_tool_spec_present(self):
        assert {s["function"]["name"] for s in NEWS_ARCHIVE_TOOL_SPECS} == {"fetch_news_archive"}

    def test_in_process_dispatch(self):
        from tool_runtime import InProcessToolExecutor

        ex = InProcessToolExecutor(use_live=False)
        out = json.loads(ex(
            "fetch_news_archive",
            '{"query": "HDFC Bank", "from_date": "2025-08-10", "to_date": "2025-08-25"}',
        ))
        assert out["source"] == "mock"
        assert out["articles"]

    def test_in_process_dispatch_surfaces_floor_error(self):
        from tool_runtime import InProcessToolExecutor

        ex = InProcessToolExecutor(use_live=False)
        out = json.loads(ex(
            "fetch_news_archive",
            '{"query": "TCS", "from_date": "2025-01-01", "to_date": "2025-06-01"}',
        ))
        assert "error" in out  # window before the default floor -> readable error
