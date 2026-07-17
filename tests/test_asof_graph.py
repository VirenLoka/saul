"""Tests for the point-in-time (as-of) user-portfolio graph — offline, no model.

Mock prices + mock archive news only; graph persistence redirected to a temp
dir so nothing touches the repo.
"""

from __future__ import annotations

import io

import pytest
import yaml

import sector_graph
from asof_graph import build_asof_portfolio_graph, resolve_window
from config_loader import load_config
from graph_agent import run_reasoning_loop
from sector_graph import GraphError, get_graph_object

BANKING = "knowledge/portfolios/banking_portfolio.csv"
SAMPLE = "knowledge/portfolios/sample_portfolio.csv"


@pytest.fixture(autouse=True)
def _tmp_graphs(tmp_path):
    sector_graph.set_graphs_dir(tmp_path / "graphs")
    sector_graph.clear_graphs()
    yield
    sector_graph.clear_graphs()


def _newsdata():
    return load_config().newsdata


# --------------------------------------------------------------------------- #
# Window resolution
# --------------------------------------------------------------------------- #
class TestWindow:
    def test_end_defaults_to_start_plus_window(self):
        start, end = resolve_window("2025-08-08", "", 30)
        assert start.isoformat() == "2025-08-08"
        assert (end - start).days == 30

    def test_explicit_end(self):
        start, end = resolve_window("2025-08-08", "2025-09-08", 30)
        assert end.isoformat() == "2025-09-08"

    def test_end_before_start_raises(self):
        with pytest.raises(GraphError):
            resolve_window("2025-09-08", "2025-08-10", 30)


# --------------------------------------------------------------------------- #
# Building the as-of graph
# --------------------------------------------------------------------------- #
class TestBuild:
    def _build(self, portfolio=BANKING):
        return build_asof_portfolio_graph(
            portfolio, start_date="2025-08-08", end_date="2025-09-08",
            use_live=False, newsdata=_newsdata(),
        )

    def test_nodes_carry_asof_stats_and_news(self):
        b = self._build()
        assert b["as_of"] == "2025-09-08"
        assert b["node_count"] >= 10
        feats = b["nodes"][0]["features"]
        # Point-in-time statistical metrics.
        for key in ("price", "return_pct", "volatility_pct", "sharpe", "momentum_pct"):
            assert key in feats["asof_stats"], key
        # News from the window + a sentiment score.
        assert feats["news"]["window"] == {"from": "2025-08-08", "to": "2025-09-08"}
        assert "score" in feats["sentiment"]
        assert "top_headlines" in feats["news"]

    def test_edges_seeded_proposed_with_asof_evidence(self):
        b = self._build()
        assert b["edge_count"] > 0
        for e in b["candidate_edges"]:
            assert e["status"] == "proposed"
            assert e["evidence"]["as_of"] == "2025-09-08"
            assert "return_correlation" in e["evidence"]

    def test_persisted_to_disk(self):
        b = self._build()
        sector_graph.clear_graphs()  # simulate a fresh session
        g = get_graph_object(b["graph_id"])
        assert len(g.nodes) == b["node_count"]

    def test_skips_non_equity_holdings(self):
        b = self._build(SAMPLE)
        assert "CASH" in b["skipped_holdings"]
        assert all(t not in b["tickers"] for t in ("CASH", "SGBSEP28"))

    def test_empty_portfolio_would_raise(self, tmp_path):
        p = tmp_path / "cashonly.csv"
        p.write_text("Ticker,Asset Class,Quantity,Current Value\nCASH,Cash,1,1000\n",
                     encoding="utf-8")
        with pytest.raises(GraphError):
            build_asof_portfolio_graph(str(p), start_date="2025-08-08",
                                       end_date="2025-09-08", use_live=False)


# --------------------------------------------------------------------------- #
# Reasoning loop integration + digest
# --------------------------------------------------------------------------- #
class TestReasoning:
    def test_heuristic_loop_resolves_edges(self):
        b = build_asof_portfolio_graph(
            BANKING, start_date="2025-08-08", end_date="2025-09-08",
            use_live=False, newsdata=_newsdata(),
        )
        loop = run_reasoning_loop(b["graph_id"])  # no provider -> heuristic
        assert loop["driver"] == "heuristic"
        assert loop["graph"]["edges_by_status"].get("proposed", 0) == 0

    def test_digest_surfaces_asof_stats_and_headlines(self):
        from graph_agent import _digest

        b = build_asof_portfolio_graph(
            BANKING, start_date="2025-08-08", end_date="2025-09-08",
            use_live=False, newsdata=_newsdata(),
        )
        g = get_graph_object(b["graph_id"])
        node = next(iter(g.nodes.values()))
        d = _digest(node)
        assert "sharpe" in d and "return_pct" in d and "volatility_pct" in d
        assert "recent_headlines" in d


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_config_defaults():
    pg = load_config().portfolio_graph
    assert pg.start_date == "2025-08-08"
    assert pg.allow_web_search is False


# --------------------------------------------------------------------------- #
# Script entrypoint
# --------------------------------------------------------------------------- #
def test_script_main_offline(tmp_path):
    import run_portfolio_graph_asof as script

    raw = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    raw["storage_paths"]["graphs"] = str(tmp_path / "graphs")
    raw["mcp"]["market_data"]["use_live"] = False
    raw["newsdata"]["use_live"] = False
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    out = io.StringIO()
    rc = script.main(
        ["--config", str(cfg_path), "--portfolio", BANKING,
         "--start", "2025-08-08", "--end", "2025-09-08", "--mock", "--no-visualize"],
        out=out,
    )
    assert rc == 0
    assert "Persisted graph" in out.getvalue()
    assert list((tmp_path / "graphs").glob("*.json"))
