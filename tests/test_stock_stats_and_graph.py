"""Tests for stock_stats.py and sector_graph.py — mock mode only, no network."""

from __future__ import annotations

import pytest

import sector_graph
import stock_stats
from sector_graph import (
    GraphError,
    NewsSentimentProvider,
    UnimplementedFilingsProvider,
    build_sector_graph,
    get_sector_graph,
    propose_graph_edge,
    validate_graph_edge,
)
from stock_stats import (
    StatsError,
    daily_returns,
    get_correlation_matrix,
    get_fundamentals,
    get_return_statistics,
    get_risk_metrics,
    get_technical_indicators,
)


@pytest.fixture(autouse=True)
def _fresh_state(tmp_path):
    # Redirect graph persistence to a temp dir so tests never write into the repo.
    sector_graph.set_graphs_dir(tmp_path / "graphs")
    stock_stats.clear_cache()
    sector_graph.clear_graphs()
    yield
    stock_stats.clear_cache()
    sector_graph.clear_graphs()


# --------------------------------------------------------------------------- #
# stock_stats
# --------------------------------------------------------------------------- #
class TestReturnStatistics:
    def test_mock_is_deterministic(self):
        a = get_return_statistics("TCS", use_live=False)
        stock_stats.clear_cache()
        b = get_return_statistics("TCS", use_live=False)
        assert a == b
        assert a["symbol"] == "TCS.NS"
        assert a["source"] == "mock"

    def test_has_all_metrics(self):
        r = get_return_statistics("Reliance", use_live=False)
        for key in (
            "cumulative_return_pct", "annualized_return_pct",
            "annualized_volatility_pct", "sharpe_ratio", "sortino_ratio",
            "max_drawdown_pct", "var_95_daily_pct", "cvar_95_daily_pct",
        ):
            assert key in r, key
        assert r["max_drawdown_pct"] <= 0.0
        assert r["var_95_daily_pct"] <= 0.0
        assert r["cvar_95_daily_pct"] <= r["var_95_daily_pct"]


class TestTechnicalIndicators:
    def test_core_indicators_present(self):
        r = get_technical_indicators("INFY", use_live=False)
        assert r["sma_20"] is not None
        assert r["sma_200"] is not None
        assert 0.0 <= r["rsi_14"] <= 100.0
        assert r["bollinger_lower"] < r["bollinger_middle"] < r["bollinger_upper"]
        assert r["window_low"] <= r["last_price"] <= r["window_high"]

    def test_short_window_omits_long_smas(self):
        r = get_technical_indicators("INFY", period_days=30, use_live=False)
        assert r["sma_200"] is None


class TestRiskMetrics:
    def test_benchmark_relative(self):
        r = get_risk_metrics("HDFC Bank", use_live=False)
        assert r["benchmark"] == "^NSEI"
        assert -1.0 <= r["correlation_to_benchmark"] <= 1.0
        assert r["r_squared"] == pytest.approx(
            r["correlation_to_benchmark"] ** 2, abs=0.01
        )


class TestCorrelationMatrix:
    def test_matrix_shape_and_diagonal(self):
        r = get_correlation_matrix(["TCS", "INFY", "WIPRO"], use_live=False)
        assert len(r["symbols"]) == 3
        for s in r["symbols"]:
            assert r["matrix"][s][s] == 1.0
        assert len(r["ranked_pairs"]) == 3  # C(3,2)

    def test_requires_two_stocks(self):
        with pytest.raises(StatsError):
            get_correlation_matrix(["TCS"], use_live=False)


class TestFundamentals:
    def test_mock_fundamentals(self):
        r = get_fundamentals("ITC", use_live=False)
        assert r["symbol"] == "ITC.NS"
        assert r["source"] == "mock"
        assert r["market_cap"] > 0
        assert r["trailing_pe"] > 0


def test_daily_returns():
    assert daily_returns([100.0, 110.0, 99.0]) == pytest.approx([0.1, -0.1])


# --------------------------------------------------------------------------- #
# sector_graph
# --------------------------------------------------------------------------- #
def _build(sectors=("it", "banking")):
    return build_sector_graph(list(sectors), use_live=False)


class TestBuildSectorGraph:
    def test_nodes_carry_full_feature_bundle(self):
        from market_data import SECTORS

        g = _build()
        expected = len(set(SECTORS["it"]) | set(SECTORS["banking"]))
        assert g["node_count"] == expected
        node = g["nodes"][0]
        for feat in (
            "quote", "return_stats", "indicators", "fundamentals",
            "alpha_factors", "sentiment", "filings",
        ):
            assert feat in node["features"], feat
        # Filings backend does not exist yet — placeholder must say so.
        assert node["features"]["filings"]["available"] is False

    def test_candidate_edges_are_proposed_with_evidence(self):
        g = _build()
        assert g["edge_count"] > 0
        for e in g["candidate_edges"]:
            assert e["status"] == "proposed"
            assert "return_correlation" in e["evidence"]

    def test_include_sentiment_false_omits_sentiment_feature(self):
        g = build_sector_graph(["it"], use_live=False, include_sentiment=False)
        node = g["nodes"][0]
        assert "sentiment" not in node["features"]
        # Other features are still present.
        assert "return_stats" in node["features"] and "fundamentals" in node["features"]
        # Default keeps sentiment.
        g2 = build_sector_graph(["it"], use_live=False)
        assert "sentiment" in g2["nodes"][0]["features"]

    def test_accepts_comma_separated_string(self):
        g = build_sector_graph("it, pharma", use_live=False)
        assert set(g["sectors"]) == {"it", "pharma"}

    def test_unknown_sector_rejected(self):
        with pytest.raises(GraphError):
            build_sector_graph(["crypto"], use_live=False)


class TestEdgeLifecycle:
    def test_needs_multiple_confirms_then_validated(self):
        g = _build()
        gid = g["graph_id"]
        e = g["candidate_edges"][0]
        r1 = validate_graph_edge(
            gid, e["source"], e["target"], e["relation"], "confirm", "pass one"
        )
        assert r1["edge"]["status"] == "proposed"
        assert r1["confirmations_still_needed"] == 1
        r2 = validate_graph_edge(
            gid, e["source"], e["target"], e["relation"], "confirm", "pass two"
        )
        assert r2["edge"]["status"] == "validated"
        assert len(r2["edge"]["validations"]) == 2

    def test_single_reject_kills_edge(self):
        g = _build()
        e = g["candidate_edges"][0]
        r = validate_graph_edge(
            g["graph_id"], e["source"], e["target"], e["relation"],
            "reject", "evidence contradicts",
        )
        assert r["edge"]["status"] == "rejected"
        with pytest.raises(GraphError):
            validate_graph_edge(
                g["graph_id"], e["source"], e["target"], e["relation"],
                "confirm", "too late",
            )

    def test_reasoning_is_mandatory(self):
        g = _build()
        e = g["candidate_edges"][0]
        with pytest.raises(GraphError):
            validate_graph_edge(
                g["graph_id"], e["source"], e["target"], e["relation"],
                "confirm", "   ",
            )

    def test_agent_can_propose_new_edge(self):
        g = _build()
        r = propose_graph_edge(
            g["graph_id"], "TCS", "HDFCBANK", "macro_rate_sensitivity",
            "both react to RBI policy", 0.3,
        )
        assert r["edge"]["status"] == "proposed"
        with pytest.raises(GraphError):  # duplicates rejected
            propose_graph_edge(
                g["graph_id"], "HDFCBANK", "TCS", "macro_rate_sensitivity", "dup"
            )

    def test_propose_requires_known_nodes(self):
        g = _build(sectors=("it",))
        with pytest.raises(GraphError):
            propose_graph_edge(
                g["graph_id"], "TCS", "SBIN", "x", "SBIN not in an IT-only graph"
            )


class TestGetSectorGraph:
    def test_status_filter(self):
        g = _build()
        gid = g["graph_id"]
        e = g["candidate_edges"][0]
        validate_graph_edge(gid, e["source"], e["target"], e["relation"], "reject", "no")
        out = get_sector_graph(gid, status="rejected")
        assert len(out["edges"]) == 1
        assert out["edges"][0]["status"] == "rejected"
        # Features excluded by default to keep payloads small.
        assert "features" not in out["nodes"][0]

    def test_unknown_graph(self):
        with pytest.raises(GraphError):
            get_sector_graph("nope")


class TestProviders:
    def test_news_sentiment_provider_scores_mock_headlines(self):
        s = NewsSentimentProvider().get_sentiment("TCS")
        assert -1.0 <= s["score"] <= 1.0
        assert s["source"].startswith("news-lexicon/")

    def test_filings_placeholder(self):
        assert UnimplementedFilingsProvider().get_filing_features("TCS")["available"] is False
