"""End-to-end smoke test of the full pipeline.

Runs config -> ingest -> analyze -> prompt -> (MOCK) LLM -> report using the
offline MockProvider. NO real model forward pass occurs, so this is safe to run
on any machine / in CI.
"""

from __future__ import annotations

from main import run


def test_pipeline_runs_with_mock_provider(capsys):
    exit_code = run(["--provider", "mock"])
    assert exit_code == 0
    out = capsys.readouterr().out
    # Report scaffolding present.
    assert "PORTFOLIO ANALYSIS (READ-ONLY)" in out
    assert "ASSET ALLOCATION" in out
    assert "ANALYST NARRATIVE" in out
    # Mock narrative made it into the report.
    assert "MOCK LLM OUTPUT" in out
    # Read-only guardrail disclaimer is present.
    assert "No trades executed." in out
    # Total of the sample portfolio shows up.
    assert "79,650" in out


def test_missing_portfolio_returns_error_code(capsys):
    exit_code = run(["--provider", "mock", "--portfolio", "nope.csv"])
    assert exit_code == 3
