"""Tests for the backtesting engine + runner — fully offline (mock, no model).

Uses a temp config with short windows and temp output dirs, so nothing touches
the repo and no network/model is used.
"""

from __future__ import annotations

import datetime as _dt
import io

import pytest
import yaml

import sector_graph
from backtesting import engine
from backtesting.engine import (
    BacktestError,
    _business_days,
    _rebalance_dates,
    asof_analytics,
    load_price_matrix,
    run_backtest,
)
from config_loader import load_config


def _cfg(tmp_path, **bt_overrides):
    with open("config.yaml", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    raw["storage_paths"]["graphs"] = str(tmp_path / "graphs")
    raw["mcp"]["market_data"]["use_live"] = False
    raw["newsdata"]["use_live"] = False
    bt = raw.setdefault("backtesting", {})
    bt["results_dir"] = str(tmp_path / "results")
    bt.setdefault("start_date", "2025-08-05")
    bt["end_date"] = "2025-10-20"          # short window for a fast test
    bt.setdefault("rebalance", "monthly")
    bt.setdefault("sectors", ["it", "banking"])
    bt.update(bt_overrides)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(str(p))


@pytest.fixture(autouse=True)
def _tmp_graphs(tmp_path):
    sector_graph.set_graphs_dir(tmp_path / "graphs")
    sector_graph.clear_graphs()
    yield
    sector_graph.clear_graphs()


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class TestPrimitives:
    def test_business_days_excludes_weekends(self):
        days = _business_days(_dt.date(2025, 8, 4), _dt.date(2025, 8, 10))  # Mon..Sun
        assert all(d.weekday() < 5 for d in days)
        assert len(days) == 5

    def test_rebalance_cadence(self):
        days = _business_days(_dt.date(2025, 8, 5), _dt.date(2025, 12, 5))
        monthly = _rebalance_dates(days, "monthly")
        weekly = _rebalance_dates(days, "weekly")
        assert len(weekly) > len(monthly) >= 3
        assert monthly[0] == days[0]

    def test_price_matrix_mock_has_benchmark(self):
        dates, matrix = load_price_matrix(
            ["TCS", "INFY"], "^NSEI", _dt.date(2025, 8, 5), _dt.date(2025, 9, 5),
            use_live=False,
        )
        assert dates and "^NSEI" in matrix
        assert all(len(matrix[t]) == len(dates) for t in ("TCS", "INFY", "^NSEI"))

    def test_asof_analytics_shape(self):
        dates, matrix = load_price_matrix(
            ["TCS", "INFY", "WIPRO", "HCLTECH"], "^NSEI",
            _dt.date(2025, 8, 5), _dt.date(2025, 11, 5), use_live=False,
        )
        a = asof_analytics(
            {"it": ["TCS", "INFY", "WIPRO", "HCLTECH"]}, matrix, dates,
            dates[-1], lookback_days=40,
        )
        rows = a["analytics"]["it"]
        assert rows and {"ticker", "sharpe_ratio", "annualized_volatility_pct"} <= set(rows[0])


# --------------------------------------------------------------------------- #
# Full backtest (baseline weight engine, offline)
# --------------------------------------------------------------------------- #
class TestRunBacktest:
    def test_produces_curve_metrics_and_files(self, tmp_path):
        cfg = _cfg(tmp_path)
        r = run_backtest(cfg, provider=None, graph_id=None, use_live=False)
        assert r["engine"] == "baseline"
        assert r["rebalance_count"] >= 1
        assert r["web_search_enabled"] is False  # never searches during backtest
        for key in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "final_value"):
            assert key in r["portfolio"] and key in r["benchmark"]
        assert (tmp_path / "results" / "backtest_equity_curve.csv").exists()
        assert (tmp_path / "results" / "backtest_report.md").exists()

    def test_start_date_clamped_to_floor(self, tmp_path):
        # Ask to start before the training-cutoff floor -> engine clamps up.
        cfg = _cfg(tmp_path, start_date="2025-01-01")
        r = run_backtest(cfg, provider=None, use_live=False)
        assert r["window"]["start"] == cfg.newsdata.earliest_date

    def test_uses_graph_context_when_provided(self, tmp_path):
        cfg = _cfg(tmp_path)
        built = sector_graph.build_sector_graph(
            ["it", "banking"], use_live=False, include_sentiment=False
        )
        r = run_backtest(cfg, provider=None, graph_id=built["graph_id"], use_live=False)
        assert r["graph_id"] == built["graph_id"]

    def test_end_before_start_raises(self, tmp_path):
        cfg = _cfg(tmp_path, start_date="2025-09-01", end_date="2025-08-10")
        with pytest.raises(BacktestError):
            run_backtest(cfg, provider=None, use_live=False)

    def test_report_marks_web_search_disabled(self, tmp_path):
        cfg = _cfg(tmp_path)
        run_backtest(cfg, provider=None, use_live=False, name="bt")
        report = (tmp_path / "results" / "bt_report.md").read_text(encoding="utf-8")
        assert "web_search: DISABLED" in report


# --------------------------------------------------------------------------- #
# Runner entrypoint
# --------------------------------------------------------------------------- #
class TestRunner:
    def test_main_offline(self, tmp_path):
        from backtesting import runner

        _cfg(tmp_path)  # writes tmp_path/config.yaml with offline paths
        out = io.StringIO()
        rc = runner.main(["--config", str(tmp_path / "config.yaml"), "--mock"], out=out)
        assert rc == 0
        assert "Backtest complete" in out.getvalue()
        assert list((tmp_path / "results").glob("*_report.md"))
