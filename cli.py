"""Interactive CLI for the MCP-powered Financial Advisor AI Agent.

Runs a conversational loop against the configured LLM backend (vLLM, with the
Indian-market MCP tools attached server-side via ``--tool-server``). It:

  * maintains conversational memory (system / user / assistant / tool messages
    in a single array, resent each turn),
  * loads the customer portfolio at startup and injects the pre-computed
    allocation analysis into the system context,
  * visually streams the agent's reasoning, announces each MCP tool invocation
    before it runs, surfaces tool results, then streams the final answer.

Read-only scope: the agent observes and analyzes; it never executes trades.

Usage
-----
    python cli.py                       # interactive chat
    python cli.py --once "Quote for TCS?"   # single turn, then exit
    python cli.py --provider mock       # offline (no model / network)
    python cli.py --no-portfolio        # skip portfolio context

In-chat commands: /help  /reset  /memory  /portfolio  /exit
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, TextIO

from analysis import analyze_portfolio
from config_loader import AppConfig, ConfigError, load_config
from llm_provider import LLMProvider, StreamEvent, get_provider
from market_data import TOOL_SPECS
from portfolio_parser import PortfolioParseError, load_portfolio
from prompts import AGENT_SYSTEM_PROMPT, build_portfolio_context

# ANSI styling (used only when writing to a real terminal).
_COLORS = {
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "yellow": "\033[33m",
    "green": "\033[32m",
    "red": "\033[31m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


# --------------------------------------------------------------------------- #
# Conversational memory
# --------------------------------------------------------------------------- #
class ConversationMemory:
    """Ordered array of chat messages (system/user/assistant/tool).

    The full array is resent to the model each turn. Tool interactions are
    recorded in canonical OpenAI format (assistant ``tool_calls`` followed by
    ``tool`` result messages) so the transcript is faithful and re-sendable.
    """

    def __init__(self, system_prompt: str) -> None:
        self._system = {"role": "system", "content": system_prompt}
        self.messages: List[Dict[str, object]] = [self._system]
        self._tool_seq = 0

    def append_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def append_assistant_text(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def append_tool_calls(self, calls: List[Dict[str, str]]) -> None:
        self.messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": c["id"],
                        "type": "function",
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in calls
                ],
            }
        )

    def append_tool_result(self, call_id: str, name: str, content: str) -> None:
        self.messages.append(
            {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}
        )

    def next_tool_id(self) -> str:
        self._tool_seq += 1
        return f"call_{self._tool_seq}"

    def reset(self) -> None:
        self.messages = [self._system]
        self._tool_seq = 0

    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.get("role") == "user")


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
class Renderer:
    """Streams events to an output, printing section headers as types change."""

    def __init__(self, out: TextIO, use_color: bool) -> None:
        self.out = out
        self.color = use_color
        self._section: Optional[str] = None

    def _c(self, text: str, code: str) -> str:
        if not self.color:
            return text
        return f"{_COLORS[code]}{text}{_COLORS['reset']}"

    def _w(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()

    def _header(self, section: str, label: str, code: str) -> None:
        if self._section != section:
            self._section = section
            self._w("\n" + self._c(label, code) + "\n")

    def reasoning(self, text: str) -> None:
        self._header("reasoning", "🧠 Reasoning:", "dim")
        self._w(self._c(text, "dim"))

    def tool_call(self, name: str, arguments: str) -> None:
        self._section = "tool"  # always break onto its own line
        self._w(
            "\n" + self._c(f"🔧 Invoking MCP tool: {name}({arguments})", "yellow") + "\n"
        )

    def tool_result(self, name: str, text: str) -> None:
        self._section = "tool"
        self._w(self._c(f"📊 Tool result [{name}]: {text}", "cyan") + "\n")

    def content(self, text: str) -> None:
        self._header("content", "💬 Answer:", "green")
        self._w(text)

    def error(self, text: str) -> None:
        self._w("\n" + self._c(f"⚠️  {text}", "red") + "\n")

    def end_turn(self) -> None:
        self._w("\n")
        self._section = None


# --------------------------------------------------------------------------- #
# One conversational turn (factored out so it is unit-testable)
# --------------------------------------------------------------------------- #
def run_turn(
    provider: LLMProvider,
    memory: ConversationMemory,
    user_input: str,
    tools: Optional[List[Dict[str, object]]] = None,
    out: Optional[TextIO] = None,
    use_color: bool = False,
) -> str:
    """Execute one turn: append user input, stream the response, update memory.

    Returns the final assistant answer text.
    """
    out = out if out is not None else sys.stdout
    memory.append_user(user_input)
    renderer = Renderer(out, use_color)

    answer_parts: List[str] = []
    calls: List[Dict[str, str]] = []              # {id, name, arguments}
    results: List[Dict[str, str]] = []            # {id, name, content}

    for event in provider.stream_chat(memory.messages, tools=tools):
        if event.type == "reasoning":
            renderer.reasoning(event.text)
        elif event.type == "tool_call":
            # One complete event per tool (uniform across providers): announce
            # and record it.
            renderer.tool_call(event.name, event.arguments)
            calls.append(
                {
                    "id": memory.next_tool_id(),
                    "name": event.name,
                    "arguments": event.arguments,
                }
            )
        elif event.type == "tool_result":
            renderer.tool_result(event.name, event.text)
            # Pair with the most recent un-resulted call of the same name.
            for c in calls:
                if c["name"] == event.name and not any(r["id"] == c["id"] for r in results):
                    results.append({"id": c["id"], "name": c["name"], "content": event.text})
                    break
        elif event.type == "content":
            renderer.content(event.text)
            answer_parts.append(event.text)
        elif event.type == "error":
            renderer.error(event.text)
        elif event.type == "done":
            break

    renderer.end_turn()

    # Update memory in canonical order: tool_calls -> tool results -> answer.
    if calls:
        memory.append_tool_calls(calls)
        for r in results:
            memory.append_tool_result(r["id"], r["name"], r["content"])

    answer = "".join(answer_parts)
    memory.append_assistant_text(answer)
    return answer


# --------------------------------------------------------------------------- #
# Startup helpers
# --------------------------------------------------------------------------- #
def build_system_prompt(config: AppConfig, portfolio_path: Optional[str]) -> str:
    """Assemble the system prompt, injecting portfolio analysis if available."""
    if portfolio_path is None:
        return AGENT_SYSTEM_PROMPT
    portfolio = load_portfolio(portfolio_path)
    result = analyze_portfolio(portfolio, config.analysis)
    return AGENT_SYSTEM_PROMPT + build_portfolio_context(result.as_summary_dict())


def _print_help(out: TextIO) -> None:
    out.write(
        "\nCommands:\n"
        "  /help       show this help\n"
        "  /reset      clear conversation memory (keep system context)\n"
        "  /memory     print the current message array size + roles\n"
        "  /portfolio  reprint the loaded portfolio context status\n"
        "  /exit       quit\n\n"
    )


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCP-powered Financial Advisor AI Agent CLI.")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument("--portfolio", default=None, help="Portfolio CSV (default from config).")
    p.add_argument("--no-portfolio", action="store_true", help="Skip portfolio context.")
    p.add_argument("--provider", choices=["vllm", "ollama", "mock"], default=None,
                   help="Override model_selection.provider.")
    p.add_argument("--once", default=None, metavar="TEXT",
                   help="Run a single turn with TEXT, print the answer, and exit.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None, out: Optional[TextIO] = None) -> int:
    out = out if out is not None else sys.stdout
    args = parse_args(argv)

    try:
        # provider_override re-resolves the active engine block (host/port/model),
        # not just the provider name.
        config = load_config(args.config, provider_override=args.provider)
    except ConfigError as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    portfolio_path = (
        None if args.no_portfolio else (args.portfolio or config.storage_paths.default_portfolio)
    )
    try:
        system_prompt = build_system_prompt(config, portfolio_path)
    except (FileNotFoundError, PortfolioParseError) as exc:
        print(f"[portfolio error] {exc}", file=sys.stderr)
        return 3

    provider = get_provider(config)
    memory = ConversationMemory(system_prompt)
    # Only attach MCP tools for engines that execute them server-side
    # (vLLM --tool-server). Ollama has no equivalent, so it runs tool-free.
    tools = TOOL_SPECS if provider.supports_server_side_tools else None
    if tools:
        tools_line = ", ".join(t["function"]["name"] for t in tools)
    else:
        tools_line = "disabled for this engine (no server-side --tool-server)"
    use_color = (not args.no_color) and hasattr(out, "isatty") and out.isatty()

    out.write(
        f"=== Financial Advisor AI Agent (READ-ONLY) ===\n"
        f"Backend : {provider.describe()}\n"
        f"Tools   : {tools_line}\n"
        f"Portfolio context: {'loaded' if portfolio_path else 'disabled'}\n"
        f"Type /help for commands, /exit to quit.\n"
    )

    # Single-shot mode (handy for scripting / smoke tests).
    if args.once is not None:
        run_turn(provider, memory, args.once, tools=tools, out=out, use_color=use_color)
        return 0

    # Interactive loop.
    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            out.write("\nGoodbye.\n")
            return 0

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            out.write("Goodbye.\n")
            return 0
        if user_input == "/help":
            _print_help(out)
            continue
        if user_input == "/reset":
            memory.reset()
            out.write("(memory cleared)\n")
            continue
        if user_input == "/memory":
            roles = [m["role"] for m in memory.messages]
            out.write(f"({len(memory.messages)} messages: {roles})\n")
            continue
        if user_input == "/portfolio":
            out.write(
                f"(portfolio context: {'loaded' if portfolio_path else 'disabled'})\n"
            )
            continue

        run_turn(provider, memory, user_input, tools=tools, out=out, use_color=use_color)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
