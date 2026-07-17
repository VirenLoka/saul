"""Smoke tests for the interactive CLI.

Driven by the offline MockStreamingProvider + the in-process tool executor, with
output captured to an in-memory buffer — no model forward pass, no network, no
MCP server.
"""

from __future__ import annotations

import io

from cli import ConversationMemory, main, run_turn
from llm_provider import MockStreamingProvider
from market_data import TOOL_SPECS
from tool_runtime import InProcessToolExecutor

# Offline executor: runs market_data with deterministic mock data.
EXEC = InProcessToolExecutor(use_live=False)


def test_run_turn_shows_plan_tool_and_answer():
    out = io.StringIO()
    mem = ConversationMemory("system prompt")
    answer = run_turn(
        MockStreamingProvider(),
        mem,
        "What's the quote for Reliance?",
        tools=TOOL_SPECS,
        executor=EXEC,
        out=out,
        use_color=False,
    )
    text = out.getvalue()
    # Plan -> act -> reflect -> answer are all visible.
    assert "Plan" in text
    assert "Invoking MCP tool: get_indian_stock_quote" in text
    assert "Tool result" in text
    assert "Answer" in text
    assert "MOCK LLM OUTPUT" in answer
    assert "not financial advice" in answer.lower()


def test_run_turn_executes_tool_and_records_real_result():
    out = io.StringIO()
    mem = ConversationMemory("sys")
    run_turn(MockStreamingProvider(), mem, "Quote for Reliance?",
             tools=TOOL_SPECS, executor=EXEC, out=out)
    # The tool was actually executed in-process: the result shows the symbol.
    text = out.getvalue()
    assert "RELIANCE.NS" in text
    # And the deterministic mock source is recorded in the tool message.
    tool_msgs = [m for m in mem.messages if m.get("role") == "tool"]
    assert tool_msgs and "mock" in tool_msgs[0]["content"]


def test_run_turn_updates_memory_with_tool_messages():
    mem = ConversationMemory("system prompt")
    assert [m["role"] for m in mem.messages] == ["system"]

    run_turn(MockStreamingProvider(), mem, "Quote for TCS?",
             tools=TOOL_SPECS, executor=EXEC, out=io.StringIO())

    roles = [m["role"] for m in mem.messages]
    # system, user, assistant(tool_calls), tool(result), assistant(answer).
    # Plan/reflection are ephemeral and NOT persisted.
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    tool_call_msg = mem.messages[2]
    assert tool_call_msg["tool_calls"][0]["function"]["name"] == "get_indian_stock_quote"
    assert mem.messages[3]["tool_call_id"] == tool_call_msg["tool_calls"][0]["id"]


def test_run_turn_without_executor_skips_tools():
    """No executor -> no ACT phase; still plans and answers."""
    out = io.StringIO()
    mem = ConversationMemory("sys")
    run_turn(MockStreamingProvider(), mem, "Quote for Reliance?",
             tools=None, executor=None, out=out)
    text = out.getvalue()
    assert "Plan" in text
    assert "Invoking MCP tool" not in text  # tools skipped
    # Memory has no tool messages.
    assert not any(m.get("role") == "tool" for m in mem.messages)


def test_memory_reset_keeps_system_only():
    mem = ConversationMemory("sys")
    run_turn(MockStreamingProvider(), mem, "hi",
             tools=TOOL_SPECS, executor=EXEC, out=io.StringIO())
    assert mem.turn_count() == 1
    mem.reset()
    assert [m["role"] for m in mem.messages] == ["system"]
    assert mem.turn_count() == 0


def test_multi_turn_memory_accumulates():
    mem = ConversationMemory("sys")
    provider = MockStreamingProvider()
    run_turn(provider, mem, "Quote for Reliance?",
             tools=TOOL_SPECS, executor=EXEC, out=io.StringIO())
    run_turn(provider, mem, "How is the IT sector?",
             tools=TOOL_SPECS, executor=EXEC, out=io.StringIO())
    assert mem.turn_count() == 2
    sector_calls = [
        m for m in mem.messages
        if m.get("role") == "assistant" and m.get("tool_calls")
        and m["tool_calls"][0]["function"]["name"] == "get_indian_sector_performance"
    ]
    assert sector_calls


def test_main_once_mode_offline(capsys):
    rc = main(["--provider", "mock", "--once", "Quote for Reliance?"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Financial Advisor AI Agent" in captured
    assert "Invoking MCP tool" in captured
    assert "get_indian_stock_quote" in captured


def test_main_once_mode_no_portfolio(capsys):
    rc = main(["--provider", "mock", "--no-portfolio", "--once", "hi"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Portfolio" in out
    assert "disabled" in out


def test_main_verbose_logs_go_to_stderr_not_stdout(capsys):
    rc = main(["--provider", "mock", "-v", "--no-portfolio", "--once", "hi"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Startup |" in captured.err
    assert "Startup |" not in captured.out
