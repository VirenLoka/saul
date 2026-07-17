"""Request-size guards — keep outgoing chat requests under token limits.

Remote engines enforce per-request / per-minute token caps (e.g. Groq's free
tier is 8000 TPM, counting prompt + reserved completion). A single turn here can
blow that: the system prompt, the portfolio context, ~16 tool schemas, and —
above all — large tool RESULTS (a graph build returns every node's features)
accumulate in the conversation. Sending that unguarded returns HTTP 413, and the
OpenAI client then auto-retries into a 429 back-off loop.

This module trims a request so it fits a budget, without corrupting the caller's
stored memory (it operates on a copy) and without breaking tool-call pairing:

* Tool-result contents are truncated to ``max_tool_result_chars``.
* Oldest non-system messages are dropped in whole prefixes until the prompt
  fits (never leaving an orphan ``tool`` message whose ``assistant`` tool_calls
  was dropped).
* The completion budget (``max_tokens``) is clamped so prompt + completion stays
  under ``max_request_tokens``.

Token counts are estimated with a cheap ~4-chars/token heuristic (no tokenizer
dependency); the goal is a safety margin, not exactness — pick a budget below
the hard limit.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

CHARS_PER_TOKEN = 4
# Always leave room for at least this many completion tokens after trimming.
MIN_COMPLETION_TOKENS = 256
# Rough per-message structural overhead (role, delimiters).
_MSG_OVERHEAD = 4

Message = Dict[str, object]


def estimate_tokens(text: str) -> int:
    """Estimate tokens for a text blob (~4 chars/token, rounded up)."""
    if not text:
        return 0
    return len(text) // CHARS_PER_TOKEN + 1


def _message_text(msg: Message) -> str:
    parts: List[str] = []
    content = msg.get("content")
    if isinstance(content, str):
        parts.append(content)
    for tc in msg.get("tool_calls") or []:  # type: ignore[union-attr]
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        parts.append(str(fn.get("name", "")))
        parts.append(str(fn.get("arguments", "")))
    if msg.get("name"):
        parts.append(str(msg["name"]))
    return " ".join(p for p in parts if p)


def estimate_message_tokens(msg: Message) -> int:
    return _MSG_OVERHEAD + estimate_tokens(_message_text(msg))


def estimate_tools_tokens(tools: Optional[List[Dict[str, object]]]) -> int:
    if not tools:
        return 0
    try:
        return estimate_tokens(json.dumps(tools))
    except (TypeError, ValueError):
        return 0


def estimate_request_tokens(
    messages: List[Message], tools: Optional[List[Dict[str, object]]] = None
) -> int:
    return sum(estimate_message_tokens(m) for m in messages) + estimate_tools_tokens(tools)


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + f"\n…[truncated {len(text) - max_chars} chars]"
    return text


def truncate_tool_results(messages: List[Message], max_chars: int) -> List[Message]:
    """Return a copy with each ``tool`` message's content capped to ``max_chars``."""
    if not max_chars or max_chars <= 0:
        return list(messages)
    out: List[Message] = []
    for m in messages:
        if m.get("role") == "tool" and isinstance(m.get("content"), str):
            m = {**m, "content": _truncate_text(m["content"], max_chars)}  # type: ignore[arg-type]
        out.append(m)
    return out


def fit_request(
    messages: List[Message],
    tools: Optional[List[Dict[str, object]]],
    *,
    max_request_tokens: int,
    configured_max_tokens: int,
    max_tool_result_chars: int = 0,
) -> Tuple[List[Message], int]:
    """Trim ``messages`` and clamp completion so the request fits the budget.

    Returns ``(trimmed_messages, effective_max_tokens)``. Operates on a copy, so
    the caller's stored conversation is untouched. When ``max_request_tokens`` is
    0/negative only tool-result truncation is applied (no history trimming and
    the configured completion budget is preserved).
    """
    msgs = truncate_tool_results(messages, max_tool_result_chars)

    if not max_request_tokens or max_request_tokens <= 0:
        return msgs, configured_max_tokens

    tools_toks = estimate_tools_tokens(tools)

    # Preserve all leading system messages; trim only the body.
    i = 0
    while i < len(msgs) and msgs[i].get("role") == "system":
        i += 1
    system, body = msgs[:i], msgs[i:]
    sys_toks = sum(estimate_message_tokens(m) for m in system)

    def prompt_tokens(b: List[Message]) -> int:
        return sys_toks + tools_toks + sum(estimate_message_tokens(m) for m in b)

    # Drop oldest body messages (whole prefixes) until the prompt leaves room for
    # a minimum completion. Never keep the last message from being dropped.
    while len(body) > 1 and prompt_tokens(body) + MIN_COMPLETION_TOKENS > max_request_tokens:
        body = body[1:]
        # Don't leave a leading orphan tool result (its tool_calls was dropped).
        while len(body) > 1 and body[0].get("role") == "tool":
            body = body[1:]

    trimmed = system + body
    remaining = max_request_tokens - prompt_tokens(body)
    effective = min(configured_max_tokens, remaining)
    effective = max(1, effective)  # never send a non-positive max_tokens
    return trimmed, effective
