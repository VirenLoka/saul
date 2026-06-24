"""Smoke tests for the interactive CLI.

Driven entirely by the offline MockStreamingProvider with output captured to an
in-memory buffer — no model forward pass, no network.
"""

from __future__ import annotations

import io

from cli import ConversationMemory, main, run_turn
from llm_provider import MockStreamingProvider
from market_data import TOOL_SPECS


def test_run_turn_displays_reasoning_tool_and_answer():
    out = io.StringIO()
    mem = ConversationMemory("system prompt")
    answer = run_turn(
        MockStreamingProvider(),
        mem,
        "What's the quote for Reliance?",
        tools=TOOL_SPECS,
        out=out,
        use_color=False,
    )
    text = out.getvalue()
    # The reasoning, the tool invocation banner, and the final answer all show.
    assert "Reasoning" in text
    assert "Invoking MCP tool: get_indian_stock_quote" in text
    assert "Answer" in text
    assert "MOCK LLM OUTPUT" in answer
    assert "not financial advice" in answer.lower()


def test_run_turn_updates_memory_with_tool_messages():
    mem = ConversationMemory("system prompt")
    assert [m["role"] for m in mem.messages] == ["system"]

    run_turn(MockStreamingProvider(), mem, "Quote for TCS?",
             tools=TOOL_SPECS, out=io.StringIO())

    roles = [m["role"] for m in mem.messages]
    # system, user, assistant(tool_calls), tool(result), assistant(answer)
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    # The tool-call assistant message carries a structured tool_calls payload.
    tool_call_msg = mem.messages[2]
    assert tool_call_msg["tool_calls"][0]["function"]["name"] == "get_indian_stock_quote"
    # The tool result message references the same call id.
    assert mem.messages[3]["tool_call_id"] == tool_call_msg["tool_calls"][0]["id"]


def test_memory_reset_keeps_system_only():
    mem = ConversationMemory("sys")
    run_turn(MockStreamingProvider(), mem, "hi", tools=TOOL_SPECS, out=io.StringIO())
    assert mem.turn_count() == 1
    mem.reset()
    assert [m["role"] for m in mem.messages] == ["system"]
    assert mem.turn_count() == 0


def test_multi_turn_memory_accumulates():
    mem = ConversationMemory("sys")
    provider = MockStreamingProvider()
    run_turn(provider, mem, "Quote for Reliance?", tools=TOOL_SPECS, out=io.StringIO())
    run_turn(provider, mem, "How is the IT sector?", tools=TOOL_SPECS, out=io.StringIO())
    assert mem.turn_count() == 2
    # Second turn routed to the sector tool.
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
    # Banner shows the portfolio row as disabled.
    assert "Portfolio" in out
    assert "disabled" in out


def test_main_verbose_logs_go_to_stderr_not_stdout(capsys):
    rc = main(["--provider", "mock", "-v", "--no-portfolio", "--once", "hi"])
    assert rc == 0
    captured = capsys.readouterr()
    # Diagnostic logs must not pollute the chat UI on stdout.
    assert "Startup |" in captured.err
    assert "Startup |" not in captured.out
