"""Interactive CLI for the MCP-powered Financial Advisor AI Agent.

Runs a conversational loop against the configured LLM engine (self-hosted vLLM,
with the Indian-market MCP tools attached server-side via ``--tool-server``; the
remote Groq API, which runs those tools client-side; or the offline mock). It:

  * maintains conversational memory (system / user / assistant / tool messages
    in a single array, resent each turn),
  * loads the customer portfolio at startup and injects the pre-computed
    allocation analysis into the system context,
  * streams the agent's reasoning, announces each MCP tool invocation before it
    runs, surfaces tool results, then streams the final answer.

Read-only scope: the agent observes and analyzes; it never executes trades.

Logging
-------
Diagnostic logs go to STDERR (never stdout), so they don't corrupt the streamed
chat UI. Control with ``--log-level``, ``-v/--verbose`` (DEBUG), ``--log-file``,
or the ``SPARKS_LOG_LEVEL`` env var. At DEBUG you'll see the exact engine URL
each request targets — useful for diagnosing Docker connectivity.

Usage
-----
    python cli.py                          # interactive chat
    python cli.py --once "Quote for TCS?"  # single turn, then exit
    python cli.py --provider mock          # offline (no model / network)
    python cli.py --no-portfolio           # skip portfolio context
    python cli.py -v                       # verbose logs to stderr

In-chat commands: /help  /reset  /memory  /portfolio  /exit
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from typing import Dict, List, Optional, TextIO

from analysis import analyze_portfolio
from config_loader import AppConfig, ConfigError, load_config
from llm_provider import LLMProvider, get_provider
from graph_agent import PORTFOLIO_GRAPH_TOOL_SPECS
from graph_viz import GRAPH_VIZ_TOOL_SPECS
from market_data import TOOL_SPECS
from news_data import NEWS_TOOL_SPECS
from portfolio_builder import PORTFOLIO_TOOL_SPECS
from sector_graph import GRAPH_TOOL_SPECS
from stock_stats import STATS_TOOL_SPECS
from web_search import SEARCH_TOOL_SPECS
from portfolio_parser import PortfolioParseError, load_portfolio
from prompts import (
    ACT_DIRECTIVE,
    AGENT_SYSTEM_PROMPT,
    ANSWER_DIRECTIVE,
    ANSWER_MARKER,
    PLAN_DIRECTIVE,
    build_portfolio_context,
)
from tool_runtime import InProcessToolExecutor, MCPToolExecutor

logger = logging.getLogger("saul.cli")

# Max tool-execution rounds in the ACT phase before forcing the answer.
MAX_TOOL_ROUNDS = 4

# Cap the tool result echoed to the terminal (the full result still goes to the
# model, subject to the provider's own request-size guard). Keeps a huge graph
# payload from flooding the screen.
MAX_TOOL_RESULT_DISPLAY_CHARS = 1200

# Committed to memory when a turn is aborted after an unrecoverable request
# failure, so the transcript stays valid and re-sendable.
_ABORT_NOTE = (
    "(Turn aborted: the model request failed — see the error above. No answer "
    "was produced and no trades were executed.)"
)

# Callable that runs a tool: executor(name, arguments_json) -> result_json_string.
ToolExecutor = object  # documented alias; both executors are plain callables


def _clip(text: str, max_chars: int) -> str:
    """Shorten ``text`` for terminal display, marking how much was hidden."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f" …[+{len(text) - max_chars} chars]"


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"


def _fg(n: int) -> str:
    return f"\033[38;5;{n}m"


# Semantic palette (256-color).
_STYLE = {
    "title": _BOLD + _fg(45),     # bright cyan
    "rule": _fg(240),             # grey box lines
    "label": _BOLD + _fg(250),    # row labels
    "value": _fg(252),            # row values
    "muted": _DIM + _fg(245),     # reasoning / hints
    "plan_hdr": _BOLD + _fg(111),   # plan header (periwinkle)
    "plan": _fg(111),             # plan body
    "reflect_hdr": _BOLD + _fg(147),  # reflection header (lavender)
    "reflect": _DIM + _fg(247),   # reflection body
    "tool": _BOLD + _fg(214),     # tool invocation (amber)
    "result": _fg(80),            # tool result (teal)
    "answer": _fg(255),           # final answer (bright)
    "answer_hdr": _BOLD + _fg(42),  # answer header (green)
    "prompt": _BOLD + _fg(45),    # input prompt
    "warn": _fg(214),
    "error": _BOLD + _fg(203),    # red/coral
}


def _style(text: str, key: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_STYLE.get(key, '')}{text}{_RESET}"


# --------------------------------------------------------------------------- #
# Logging setup
# --------------------------------------------------------------------------- #
def setup_logging(level: str = "WARNING", log_file: Optional[str] = None) -> None:
    """Configure the ``saul`` package logger.

    Logs go to STDERR (and optionally a file) so they never mix with the chat
    UI on STDOUT. Idempotent: re-clears handlers so repeated calls (e.g. in
    tests) don't duplicate output.
    """
    pkg = logging.getLogger("saul")
    pkg.setLevel(getattr(logging, level.upper(), logging.WARNING))
    pkg.propagate = False  # don't double-log via the root logger
    for h in list(pkg.handlers):
        pkg.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s", datefmt="%H:%M:%S"
    )
    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setFormatter(fmt)
    pkg.addHandler(stderr_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        pkg.addHandler(file_handler)
        logger.debug("Logging to file: %s", log_file)


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
    """Streams events with a clear separation of three registers:

    * **Reasoning** (the model's live chain-of-thought) is *ephemeral*: on a TTY
      it renders to a single, self-updating status line (spinner + the latest
      thought) that is wiped as soon as durable output arrives — so it never
      floods or scrolls the transcript. On a non-TTY (pipe/tests) it is
      suppressed entirely.
    * **Durable sections** — 🧭 Plan, 🔧 tool calls / 📊 results, 🔍 Reflection —
      scroll normally, each with its own styled header so they are visually
      distinct.
    * The final **💬 Answer** is set off by a rule so it is unmistakable.
    """

    _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, out: TextIO, use_color: bool) -> None:
        self.out = out
        self.color = use_color
        # The live status line needs ANSI + an interactive terminal.
        self.live = use_color and hasattr(out, "isatty") and out.isatty()
        self._section: Optional[str] = None
        self._fresh = True          # cursor is at the start of a blank line
        self._status_on = False     # a reasoning status line is currently drawn
        self._reason = ""           # accumulated reasoning for the status tail
        self._spin = 0

    # ---- low-level ---------------------------------------------------------
    def _raw(self, text: str) -> None:
        if not text:
            return
        self.out.write(text)
        self.out.flush()
        self._fresh = text.endswith("\n")

    def _width(self) -> int:
        return max(40, shutil.get_terminal_size((80, 24)).columns)

    def _newline_if_needed(self) -> None:
        if not self._fresh:
            self._raw("\n")

    # ---- ephemeral reasoning (live status line) ----------------------------
    def reasoning(self, text: str) -> None:
        if not self.live or not text:
            return  # ephemeral by design: shown live on a TTY, suppressed otherwise
        if not self._status_on:
            self._newline_if_needed()   # drop below any partial durable line
            self._status_on = True
            self._reason = ""
        self._reason += text
        self._spin = (self._spin + 1) % len(self._SPINNER)
        tail = " ".join(self._reason.split())[-(self._width() - 8):]
        frame = self._SPINNER[self._spin]
        # \r + clear-line keeps everything on ONE self-updating row.
        self.out.write(f"\r\033[2K{_STYLE['muted']}{frame} 🧠 {tail}{_RESET}")
        self.out.flush()
        self._fresh = False

    def _clear_status(self) -> None:
        if self._status_on:
            self.out.write("\r\033[2K")   # wipe the status line, cursor at col 0
            self.out.flush()
            self._status_on = False
            self._reason = ""
            self._fresh = True

    # ---- durable sections --------------------------------------------------
    def _sep_into(self, section: str) -> bool:
        """Wipe any status line and, when entering a NEW section, emit one blank
        separator line. Returns True if the section changed."""
        self._clear_status()
        changed = self._section != section
        self._section = section
        if changed:
            self._newline_if_needed()
            self._raw("\n")
        return changed

    def _header(self, section: str, label: str, key: str, rule: bool = False) -> None:
        if self._sep_into(section):
            if rule:
                self._raw(_style("─" * min(self._width(), 60), "rule", self.color) + "\n")
            self._raw(_style(label, key, self.color) + "\n")

    def plan(self, text: str) -> None:
        self._header("plan", "🧭 Plan", "plan_hdr")
        self._raw(_style(text, "plan", self.color))

    def working(self, text: str) -> None:
        if not text.strip():
            return
        self._header("working", "⚙️  Working", "muted")
        self._raw(_style(text, "muted", self.color))

    def tool_call(self, name: str, arguments: str) -> None:
        self._sep_into("tool")
        line = f"🔧 Invoking MCP tool: {name}({arguments})"
        self._raw(_style(line, "tool", self.color) + "\n")

    def tool_result(self, name: str, text: str) -> None:
        self._sep_into("tool")  # same section as the call -> no extra blank line
        self._raw(_style(f"📊 Tool result [{name}]: {text}", "result", self.color) + "\n")

    def reflection(self, text: str) -> None:
        self._header("reflection", "🔍 Reflection", "reflect_hdr")
        self._raw(_style(text, "reflect", self.color))

    def answer(self, text: str) -> None:
        self._header("answer", "💬 Answer", "answer_hdr", rule=True)
        self._raw(_style(text, "answer", self.color))

    def error(self, text: str) -> None:
        self._clear_status()
        self._newline_if_needed()
        self._raw(_style(f"⚠️  {text}", "error", self.color) + "\n")

    def end_turn(self) -> None:
        self._clear_status()
        self._newline_if_needed()
        self._raw("\n")
        self._section = None


# --------------------------------------------------------------------------- #
# Streaming-marker splitter (for splitting REFLECTION from the final ANSWER)
# --------------------------------------------------------------------------- #
class _MarkerSplitter:
    """Splits a streamed text on the first occurrence of ``marker``.

    ``feed`` yields ``(section, text)`` where section is "before" until the
    marker is seen and "after" thereafter. Handles a marker split across chunk
    boundaries by retaining a small tail.
    """

    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.buf = ""
        self.found = False

    def feed(self, text: str) -> List[tuple[str, str]]:
        if self.found:
            return [("after", text)] if text else []
        self.buf += text
        idx = self.buf.find(self.marker)
        if idx != -1:
            out: List[tuple[str, str]] = []
            before, after = self.buf[:idx], self.buf[idx + len(self.marker):]
            if before:
                out.append(("before", before))
            self.found = True
            self.buf = ""
            if after:
                out.append(("after", after))
            return out
        # Marker not yet seen: emit all but a possible partial-marker tail.
        keep = 0
        for k in range(1, len(self.marker)):
            if self.buf.endswith(self.marker[:k]):
                keep = k
        emit, self.buf = self.buf[: len(self.buf) - keep], self.buf[len(self.buf) - keep:]
        return [("before", emit)] if emit else []

    def flush(self) -> List[tuple[str, str]]:
        if not self.buf:
            return []
        section = "after" if self.found else "before"
        out = [(section, self.buf)]
        self.buf = ""
        return out


# --------------------------------------------------------------------------- #
# Phase streaming helpers
# --------------------------------------------------------------------------- #
def _stream_phase(
    provider: LLMProvider,
    messages: List[Dict[str, object]],
    tools: Optional[List[Dict[str, object]]],
    renderer: Renderer,
    kind: str,
) -> tuple[List[Dict[str, str]], str, bool, bool]:
    """Stream one model response. ``kind`` selects how content is rendered:
    'plan' -> Plan section, 'act' -> Working section. Returns
    (tool_calls, content_text, had_error, fatal)."""
    calls: List[Dict[str, str]] = []
    parts: List[str] = []
    had_error = False
    fatal = False
    for event in provider.stream_chat(messages, tools=tools):
        if event.type == "reasoning":
            renderer.reasoning(event.text)
        elif event.type == "tool_call":
            logger.info("Tool call: %s(%s)", event.name, event.arguments)
            renderer.tool_call(event.name, event.arguments)
            calls.append({"name": event.name, "arguments": event.arguments})
        elif event.type == "content":
            parts.append(event.text)
            if kind == "plan":
                renderer.plan(event.text)
            elif kind == "act":
                renderer.working(event.text)
        elif event.type == "error":
            had_error = True
            fatal = fatal or event.fatal
            logger.error("Stream error: %s", event.text)
            renderer.error(event.text)
        elif event.type == "done":
            break
    return calls, "".join(parts), had_error, fatal


def _stream_answer(
    provider: LLMProvider,
    messages: List[Dict[str, object]],
    renderer: Renderer,
) -> tuple[str, bool]:
    """Stream the final phase, splitting REFLECTION from the ANSWER on the
    ``ANSWER:`` marker. Returns (answer_text, had_error)."""
    splitter = _MarkerSplitter(ANSWER_MARKER)
    reflection_parts: List[str] = []
    answer_parts: List[str] = []
    had_error = False

    def _route(segments: List[tuple[str, str]]) -> None:
        for section, text in segments:
            if section == "before":
                renderer.reflection(text)
                reflection_parts.append(text)
            else:
                renderer.answer(text)
                answer_parts.append(text)

    for event in provider.stream_chat(messages, tools=None):
        if event.type == "reasoning":
            renderer.reasoning(event.text)
        elif event.type == "content":
            _route(splitter.feed(event.text))
        elif event.type == "error":
            had_error = True
            logger.error("Stream error: %s", event.text)
            renderer.error(event.text)
        elif event.type == "done":
            break
    _route(splitter.flush())

    answer = "".join(answer_parts).strip()
    if not answer:  # model omitted the ANSWER: marker -> treat all of it as the answer
        answer = "".join(reflection_parts).strip()
    return answer, had_error


# --------------------------------------------------------------------------- #
# One conversational turn: PLAN -> ACT (tool loop) -> REFLECT + ANSWER
# --------------------------------------------------------------------------- #
def run_turn(
    provider: LLMProvider,
    memory: ConversationMemory,
    user_input: str,
    tools: Optional[List[Dict[str, object]]] = None,
    executor: Optional[object] = None,
    out: Optional[TextIO] = None,
    use_color: bool = False,
    max_tool_rounds: int = MAX_TOOL_ROUNDS,
) -> str:
    """Run one turn through the plan -> act -> reflect loop.

    Reasoning phases (plan, reflection) are displayed but kept ephemeral; only
    the user message, tool calls/results, and the final answer are committed to
    long-term memory (keeping the transcript valid and re-sendable). Tools are
    used only when both ``tools`` and ``executor`` are provided.
    """
    out = out if out is not None else sys.stdout
    memory.append_user(user_input)
    logger.info("Turn %d | user: %s", memory.turn_count(), user_input[:120])
    renderer = Renderer(out, use_color)

    can_use_tools = tools is not None and executor is not None
    working: List[Dict[str, object]] = list(memory.messages)
    committed_calls: List[Dict[str, str]] = []
    committed_results: List[Dict[str, str]] = []

    def _commit_and_return(answer_text: str) -> str:
        """Flush gathered tool calls/results + the answer to long-term memory."""
        if committed_calls:
            memory.append_tool_calls(committed_calls)
            for r in committed_results:
                memory.append_tool_result(r["id"], r["name"], r["content"])
        memory.append_assistant_text(answer_text)
        logger.debug(
            "Turn complete | tool_calls=%d answer chars=%d",
            len(committed_calls), len(answer_text),
        )
        return answer_text

    # ---- PHASE 1: PLAN (no tools) -----------------------------------------
    _, plan_text, _plan_err, plan_fatal = _stream_phase(
        provider, working + [{"role": "user", "content": PLAN_DIRECTIVE}],
        None, renderer, "plan",
    )
    if plan_fatal:
        # An oversized/rate-limited request will only recur on the next phase —
        # abort now rather than looping through ACT and ANSWER.
        renderer.end_turn()
        logger.warning("Aborting turn after fatal error in PLAN phase.")
        return _commit_and_return(_ABORT_NOTE)
    if plan_text.strip():
        working.append({"role": "user", "content": PLAN_DIRECTIVE})
        working.append({"role": "assistant", "content": plan_text})

    # ---- PHASE 2: ACT (tool loop) -----------------------------------------
    if can_use_tools:
        working.append({"role": "user", "content": ACT_DIRECTIVE})
        for round_i in range(max_tool_rounds):
            calls, _act_text, had_error, act_fatal = _stream_phase(
                provider, working, tools, renderer, "act"
            )
            if act_fatal:
                renderer.end_turn()
                logger.warning("Aborting turn after fatal error in ACT phase.")
                return _commit_and_return(_ABORT_NOTE)
            if had_error or not calls:
                break
            norm: List[Dict[str, str]] = [
                {"id": memory.next_tool_id(), "name": c["name"], "arguments": c["arguments"]}
                for c in calls
            ]
            working.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]},
                        }
                        for c in norm
                    ],
                }
            )
            committed_calls.extend(norm)
            for c in norm:
                result = executor(c["name"], c["arguments"])  # type: ignore[operator]
                renderer.tool_result(c["name"], _clip(result, MAX_TOOL_RESULT_DISPLAY_CHARS))
                working.append(
                    {"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": result}
                )
                committed_results.append({"id": c["id"], "name": c["name"], "content": result})
        else:
            logger.warning("Reached max tool rounds (%d); forcing answer.", max_tool_rounds)

    # ---- PHASE 3: REFLECT + ANSWER (no tools) -----------------------------
    answer, _ans_err = _stream_answer(
        provider, working + [{"role": "user", "content": ANSWER_DIRECTIVE}], renderer
    )
    renderer.end_turn()
    if not answer:  # streaming failed (e.g. rate limit) -> commit a clear note
        answer = _ABORT_NOTE
    return _commit_and_return(answer)


# --------------------------------------------------------------------------- #
# Startup helpers
# --------------------------------------------------------------------------- #
def build_system_prompt(config: AppConfig, portfolio_path: Optional[str]) -> str:
    """Assemble the system prompt, injecting portfolio analysis if available."""
    if portfolio_path is None:
        return AGENT_SYSTEM_PROMPT
    portfolio = load_portfolio(portfolio_path)
    result = analyze_portfolio(portfolio, config.analysis)
    logger.info(
        "Portfolio loaded: %s (%d holdings, total %.2f)",
        portfolio_path,
        len(portfolio),
        portfolio.total_value,
    )
    return AGENT_SYSTEM_PROMPT + build_portfolio_context(result.as_summary_dict())


def render_banner(
    title: str,
    rows: List[tuple[str, str]],
    use_color: bool,
) -> str:
    """Build a boxed banner with a centered title and aligned info rows."""
    subtitle = "READ-ONLY · observational · no trades executed"
    box_lines = [title, subtitle]
    inner = max(len(s) for s in box_lines) + 4

    def _center(s: str) -> str:
        pad = inner - len(s)
        left = pad // 2
        return " " * left + s + " " * (pad - left)

    top = _style("╭" + "─" * inner + "╮", "rule", use_color)
    bot = _style("╰" + "─" * inner + "╯", "rule", use_color)
    bar = _style("│", "rule", use_color)
    title_row = f"{bar}{_style(_center(title), 'title', use_color)}{bar}"
    sub_row = f"{bar}{_style(_center(subtitle), 'muted', use_color)}{bar}"

    label_w = max(len(lbl) for lbl, _ in rows) if rows else 0
    info: List[str] = []
    for lbl, val in rows:
        info.append(
            "  "
            + _style(f"{lbl:<{label_w}}", "label", use_color)
            + "  "
            + _style(val, "value", use_color)
        )

    return "\n".join([top, title_row, sub_row, bot, *info]) + "\n"


def _print_help(out: TextIO, use_color: bool) -> None:
    items = [
        ("/help", "show this help"),
        ("/reset", "clear conversation memory (keep system context)"),
        ("/memory", "print the current message array size + roles"),
        ("/portfolio", "show the loaded portfolio context status"),
        ("/exit", "quit"),
    ]
    out.write("\n" + _style("Commands", "label", use_color) + "\n")
    for cmd, desc in items:
        out.write("  " + _style(f"{cmd:<11}", "tool", use_color) + desc + "\n")
    out.write("\n")


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MCP-powered Financial Advisor AI Agent CLI.")
    p.add_argument("--config", default=None, help="Path to config.yaml.")
    p.add_argument("--portfolio", default=None, help="Portfolio CSV (default from config).")
    p.add_argument("--no-portfolio", action="store_true", help="Skip portfolio context.")
    p.add_argument("--provider",
                   choices=["vllm", "groq", "mock"],
                   default=None, help="Override model_selection.provider.")
    p.add_argument("--once", default=None, metavar="TEXT",
                   help="Run a single turn with TEXT, print the answer, and exit.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    p.add_argument("--log-level", default=None,
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Log level for stderr diagnostics (default: WARNING).")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Shortcut for --log-level DEBUG.")
    p.add_argument("--log-file", default=None, help="Also write logs to this file.")
    return p.parse_args(argv)


def _resolve_log_level(args: argparse.Namespace) -> str:
    if args.verbose:
        return "DEBUG"
    if args.log_level:
        return args.log_level
    return os.environ.get("SPARKS_LOG_LEVEL", "WARNING")


def main(argv: Optional[List[str]] = None, out: Optional[TextIO] = None) -> int:
    out = out if out is not None else sys.stdout
    args = parse_args(argv)
    setup_logging(_resolve_log_level(args), args.log_file)

    try:
        # provider_override re-resolves the active engine block (host/port/model),
        # not just the provider name.
        config = load_config(args.config, provider_override=args.provider)
    except ConfigError as exc:
        logger.error("Config error: %s", exc)
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    portfolio_path = (
        None if args.no_portfolio else (args.portfolio or config.storage_paths.default_portfolio)
    )
    try:
        system_prompt = build_system_prompt(config, portfolio_path)
    except (FileNotFoundError, PortfolioParseError) as exc:
        logger.error("Portfolio error: %s", exc)
        print(f"[portfolio error] {exc}", file=sys.stderr)
        return 3

    provider = get_provider(config)
    memory = ConversationMemory(system_prompt)

    # Attach the full tool suite for tool-capable engines. The CLI executes tool
    # calls itself (agentic loop); the executor differs by engine:
    #   * vllm  -> call the live FastMCP server over MCP
    #   * groq  -> run tools in-process against live data (a remote API cannot
    #              reach a local --tool-server)
    #   * mock  -> run tools in-process, offline/deterministic
    tools = (
        (TOOL_SPECS + NEWS_TOOL_SPECS + STATS_TOOL_SPECS + GRAPH_TOOL_SPECS
         + SEARCH_TOOL_SPECS + GRAPH_VIZ_TOOL_SPECS + PORTFOLIO_GRAPH_TOOL_SPECS
         + PORTFOLIO_TOOL_SPECS)
        if provider.supports_tools
        else None
    )
    executor: Optional[object] = None
    if tools:
        tool_names = ", ".join(t["function"]["name"] for t in tools)
        provider_name = config.model_selection.provider
        if provider.supports_server_side_tools and provider_name != "mock":
            # vLLM: tools executed against the live FastMCP server over MCP.
            executor = MCPToolExecutor(
                config.mcp.tool_server_url,
                timeout=config.local_inference.request_timeout,
            )
            tools_line = f"{tool_names} (via MCP @ {config.mcp.tool_server_url})"
        else:
            # mock -> deterministic offline; remote APIs -> in-process live data.
            use_live = provider_name != "mock"
            executor = InProcessToolExecutor.from_settings(
                config.mcp.market_data,
                use_live=use_live,
                newsapi=config.newsapi,
                search=config.search,
                graphs_dir=config.storage_paths.graphs,
                portfolios_dir=config.storage_paths.portfolios,
                newsdata=config.newsdata,
                default_portfolio=(portfolio_path or config.storage_paths.default_portfolio),
            )
            where = "offline" if provider_name == "mock" else f"in-process, live · {provider_name}"
            tools_line = f"{tool_names} ({where})"
    else:
        tools_line = "disabled for this engine (runs tool-free)"

    use_color = (not args.no_color) and hasattr(out, "isatty") and out.isatty()
    endpoint = (
        "offline (mock)"
        if config.model_selection.provider == "mock"
        else config.local_inference.openai_base_url
    )
    portfolio_status = (
        f"loaded — {portfolio_path}" if portfolio_path else "disabled"
    )

    logger.info(
        "Startup | provider=%s engine=%s endpoint=%s model=%s portfolio=%s",
        config.model_selection.provider,
        config.local_inference.engine,
        endpoint,
        config.local_inference.model,
        portfolio_status,
    )

    out.write(
        render_banner(
            "Financial Advisor AI Agent",
            [
                ("Backend", provider.describe()),
                ("Endpoint", endpoint),
                ("Tools", tools_line),
                ("Portfolio", portfolio_status),
                ("Commands", "/help  /reset  /memory  /portfolio  /exit"),
            ],
            use_color,
        )
    )

    # Single-shot mode (handy for scripting / smoke tests).
    if args.once is not None:
        run_turn(provider, memory, args.once, tools=tools, executor=executor,
                 out=out, use_color=use_color)
        return 0

    # Interactive loop.
    prompt = _style("\nyou ›", "prompt", use_color) + " "
    while True:
        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            out.write("\n" + _style("Goodbye.", "muted", use_color) + "\n")
            return 0

        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            out.write(_style("Goodbye.", "muted", use_color) + "\n")
            return 0
        if user_input == "/help":
            _print_help(out, use_color)
            continue
        if user_input == "/reset":
            memory.reset()
            logger.info("Memory reset")
            out.write(_style("(memory cleared)", "muted", use_color) + "\n")
            continue
        if user_input == "/memory":
            roles = [m["role"] for m in memory.messages]
            out.write(
                _style(f"({len(memory.messages)} messages: {roles})", "muted", use_color)
                + "\n"
            )
            continue
        if user_input == "/portfolio":
            out.write(
                _style(f"(portfolio context: {portfolio_status})", "muted", use_color)
                + "\n"
            )
            continue

        run_turn(provider, memory, user_input, tools=tools, executor=executor,
                 out=out, use_color=use_color)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
