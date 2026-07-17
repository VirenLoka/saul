"""Tests for graph persistence, the autonomous portfolio loop, and DOT viz.

All offline: mock data layers only, no model forward pass, no GPU, no network.
Graph persistence is redirected to a temp dir so nothing touches the repo.
"""

from __future__ import annotations

import pytest

import sector_graph
from graph_agent import build_portfolio_graph, run_reasoning_loop
from graph_viz import GRAPH_VIZ_TOOL_SPECS, render_graph_dot, visualize_sector_graph
from sector_graph import (
    build_sector_graph,
    edge_key,
    get_all_graphs,
    get_graph_object,
    list_graphs,
    load_graph,
    reverse_sector_lookup,
    validate_graph_edge,
)

BANKING_CSV = "knowledge/portfolios/banking_portfolio.csv"


@pytest.fixture(autouse=True)
def _tmp_graphs(tmp_path):
    sector_graph.set_graphs_dir(tmp_path / "graphs")
    sector_graph.clear_graphs()
    yield
    sector_graph.clear_graphs()


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
class TestPersistence:
    def test_build_persists_to_disk(self):
        g = build_sector_graph(["it"], use_live=False)
        gid = g["graph_id"]
        # A fresh load (bypassing the in-memory registry) round-trips the graph.
        loaded = load_graph(gid)
        assert loaded is not None
        assert loaded.graph_id == gid
        assert len(loaded.nodes) == g["node_count"]

    def test_get_graph_object_reloads_after_memory_cleared(self):
        gid = build_sector_graph(["pharma", "auto"], use_live=False)["graph_id"]
        sector_graph.clear_graphs()  # simulate a new session (memory gone)
        g = get_graph_object(gid)
        assert g.graph_id == gid
        assert g.nodes  # rehydrated from disk

    def test_validation_survives_reload(self):
        g = build_sector_graph(["banking"], use_live=False)
        gid = g["graph_id"]
        e = g["candidate_edges"][0]
        for _ in range(g["min_validations"]):
            sector_graph.validate_graph_edge(
                gid, e["source"], e["target"], e["relation"], "confirm", "ok"
            )
        sector_graph.clear_graphs()
        reloaded = get_graph_object(gid)
        assert reloaded.edges[
            sector_graph.edge_key(e["source"], e["target"], e["relation"])
        ].status == "validated"

    def test_list_graphs_reports_saved(self):
        a = build_sector_graph(["it"], use_live=False)["graph_id"]
        b = build_sector_graph(["auto"], use_live=False)["graph_id"]
        ids = {row["graph_id"] for row in list_graphs()}
        assert {a, b} <= ids

    def test_unknown_graph_id_raises(self):
        with pytest.raises(sector_graph.GraphError):
            get_graph_object("does-not-exist")


# --------------------------------------------------------------------------- #
# Fetch ALL graphs at once (cross-graph view)
# --------------------------------------------------------------------------- #
class TestGetAllGraphs:
    def test_empty_when_no_graphs(self):
        out = get_all_graphs()
        assert out["graph_count"] == 0
        assert out["graphs"] == [] and out["unique_tickers"] == []

    def test_fetches_every_graph_with_index(self):
        build_sector_graph(["it"], use_live=False)
        build_sector_graph(["banking"], use_live=False)
        out = get_all_graphs()
        assert out["graph_count"] == 2
        # Nodes from both graphs are indexed to their graph_ids.
        assert "TCS" in out["ticker_index"] and "HDFCBANK" in out["ticker_index"]
        assert set(out["unique_tickers"]) >= {"TCS", "HDFCBANK", "SBIN"}
        # Features excluded by default (compact payload).
        assert "features" not in out["graphs"][0]["nodes"][0]

    def test_validated_associations_union(self):
        g = build_sector_graph(["it"], use_live=False)
        e = g["candidate_edges"][0]
        for _ in range(g["min_validations"]):
            validate_graph_edge(g["graph_id"], e["source"], e["target"], e["relation"],
                                "confirm", "peer link")
        out = get_all_graphs()
        assert e["target"] in out["validated_associations"].get(e["source"], [])
        assert e["source"] in out["validated_associations"].get(e["target"], [])

    def test_ticker_filter(self):
        build_sector_graph(["it"], use_live=False)
        build_sector_graph(["banking"], use_live=False)
        out = get_all_graphs(ticker="TCS")
        assert out["graph_count"] == 1
        assert all("TCS" in [n["ticker"] for n in g["nodes"]] for g in out["graphs"])

    def test_sector_filter(self):
        build_sector_graph(["it"], use_live=False)
        build_sector_graph(["banking"], use_live=False)
        out = get_all_graphs(sector="banking")
        assert out["graph_count"] == 1
        assert "banking" in out["graphs"][0]["sectors"]

    def test_include_features(self):
        build_sector_graph(["it"], use_live=False)
        out = get_all_graphs(include_features=True)
        assert "features" in out["graphs"][0]["nodes"][0]

    def test_in_process_dispatch(self):
        import json

        from tool_runtime import InProcessToolExecutor

        build_sector_graph(["it"], use_live=False)
        # The executor points at the same (temp) graphs dir set by the fixture.
        ex = InProcessToolExecutor(use_live=False)
        out = json.loads(ex("get_all_graphs", "{}"))
        assert out["graph_count"] >= 1 and "ticker_index" in out


# --------------------------------------------------------------------------- #
# Banking universe + reverse lookup
# --------------------------------------------------------------------------- #
class TestBankingUniverse:
    def test_banking_expanded(self):
        g = build_sector_graph(["banking"], use_live=False)
        assert g["node_count"] >= 10  # was 3 before the expansion

    def test_reverse_sector_lookup(self):
        assert reverse_sector_lookup("KOTAKBANK") == "banking"
        assert reverse_sector_lookup("TCS") == "it"
        assert reverse_sector_lookup("UNKNOWNX") == "other"


# --------------------------------------------------------------------------- #
# Autonomous portfolio -> graph reasoning loop (heuristic)
# --------------------------------------------------------------------------- #
class TestPortfolioGraph:
    def test_builds_and_validates_offline(self):
        r = build_portfolio_graph(BANKING_CSV, use_live=False, min_validations=2)
        assert r["driver"] == "heuristic"
        assert len(r["tickers"]) >= 10
        assert r["reasoning_log"]
        # Every candidate edge is resolved to validated or rejected (none left proposed).
        statuses = r["graph"]["edges_by_status"]
        assert statuses.get("proposed", 0) == 0

    def test_skips_non_equity_and_unknown(self):
        r = build_portfolio_graph(
            "knowledge/portfolios/sample_portfolio.csv", use_live=False
        )
        # Cash / bonds / unknown ETFs are not graphable nodes.
        assert "CASH" in r["skipped_holdings"]
        assert all(t not in r["tickers"] for t in ("CASH", "SGBSEP28"))

    def test_reasoning_log_has_multiple_passes(self):
        r = build_portfolio_graph(BANKING_CSV, use_live=False, min_validations=2)
        validated = [row for row in r["reasoning_log"] if row["final_status"] == "validated"]
        assert validated, "expected at least one validated edge"
        assert all(row["passes"] >= 2 for row in validated)

    def test_rerun_loop_is_idempotent(self):
        r = build_portfolio_graph(BANKING_CSV, use_live=False)
        again = run_reasoning_loop(r["graph_id"])
        # Nothing left proposed -> second pass decides nothing new.
        assert again["reasoning_log"] == []


# --------------------------------------------------------------------------- #
# Graphviz DOT rendering
# --------------------------------------------------------------------------- #
class TestGraphViz:
    def test_render_dot_contains_nodes_and_edges(self):
        g = build_sector_graph(["it"], use_live=False)
        dot = render_graph_dot(get_graph_object(g["graph_id"]))
        assert dot.startswith("digraph sector_graph {")
        assert '"TCS"' in dot
        assert "->" in dot  # at least one edge

    def test_visualize_writes_dot_file(self, tmp_path):
        g = build_sector_graph(["auto"], use_live=False)
        out = visualize_sector_graph(g["graph_id"], out_dir=str(tmp_path))
        dot_file = tmp_path / f"{g['graph_id']}.dot"
        assert dot_file.exists()
        assert out["dot_path"] == str(dot_file)
        assert out["node_count"] == g["node_count"]

    def test_tool_spec_present(self):
        names = {s["function"]["name"] for s in GRAPH_VIZ_TOOL_SPECS}
        assert "visualize_sector_graph" in names
