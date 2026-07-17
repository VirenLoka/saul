"""LLM provider abstraction layer.

Decouples the CLI from the inference backend. Everything downstream depends only
on :class:`LLMProvider` and the normalized :class:`StreamEvent` stream; the
concrete backend is selected at runtime from config:

  * ``vllm`` — self-hosted vLLM over an OpenAI-compatible API. The only
    self-hosting path, and how DeepSeek models are served (point
    ``local_inference_settings.vllm.model`` at a DeepSeek repo id).
  * ``groq`` — the remote Groq API (OpenAI-compatible).
  * ``mock`` — an offline stub that runs no model.

Streaming contract
------------------
``stream_chat(messages, tools)`` yields :class:`StreamEvent` objects in arrival
order. Event types:

  * ``reasoning``   — a chunk of the model's thinking (vLLM ``reasoning_content``
                      extension; may be absent on non-reasoning models).
  * ``tool_call``   — the model/engine is invoking an MCP tool. ``name`` /
                      ``arguments`` are populated. With vLLM ``--tool-server``
                      the tool runs server-side; this event lets the CLI
                      announce it.
  * ``tool_result`` — a tool's result surfaced back in the stream (when the
                      engine reports it).
  * ``content``     — a chunk of the final user-facing answer.
  * ``error``       — a recoverable error message.
  * ``done``        — terminal event; ``finish_reason`` set.

The ``openai`` SDK is imported lazily inside the vLLM provider, so importing
this module (and running the unit/smoke tests via the mock) needs no network and
no extra packages.
"""

from __future__ import annotations

import abc
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

if TYPE_CHECKING:  # avoid a hard runtime dependency / import cycle
    from config_loader import AppConfig

logger = logging.getLogger("saul.llm")


class LLMProviderError(RuntimeError):
    """Raised when a provider is misconfigured or a backend call fails."""


@dataclass
class StreamEvent:
    """One normalized event in a streamed model turn."""

    type: str  # reasoning | tool_call | tool_result | content | error | done
    text: str = ""
    name: str = ""                       # tool name (tool_call / tool_result)
    arguments: str = ""                  # raw JSON args (tool_call)
    finish_reason: Optional[str] = None  # set on `done`
    # On `error`: True for unrecoverable failures (rate limit, request too
    # large, auth, other 4xx) where retrying the turn would just loop. The CLI
    # aborts the turn instead of proceeding to the next phase.
    fatal: bool = False


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class LLMProvider(abc.ABC):
    """Abstract base every concrete LLM backend implements."""

    name: str = "abstract"
    #: True if the engine executes attached tools server-side (e.g. vLLM
    #: --tool-server). Distinct from ``supports_tools``: this only affects
    #: which executor the CLI wires up (live MCP server vs in-process).
    supports_server_side_tools: bool = False
    #: True if tool specs should be attached for this provider at all. The
    #: remote Groq API supports tools but returns the calls to the client, which
    #: executes them — so this is True while ``supports_server_side_tools`` is
    #: False.
    supports_tools: bool = False

    @abc.abstractmethod
    def stream_chat(
        self,
        messages: List[Dict[str, object]],
        tools: Optional[List[Dict[str, object]]] = None,
    ) -> Iterator[StreamEvent]:
        """Yield normalized stream events for one assistant turn."""
        raise NotImplementedError

    def describe(self) -> str:
        return self.name


# --------------------------------------------------------------------------- #
# OpenAI-compatible streaming backend (shared by vLLM and Groq)
# --------------------------------------------------------------------------- #
class OpenAICompatibleProvider(LLMProvider):
    """Streams chat completions from any OpenAI-compatible engine.

    Self-hosted vLLM and the remote Groq API both expose this API at
    ``<base_url>/v1``, so the request/stream-parsing logic is identical;
    subclasses only differ in their label and whether the engine executes tools
    server-side.

    Tool definitions may be passed via ``tools`` (payload formatting per the
    deliverable). With a server-side tool engine (vLLM ``--tool-server``), tool
    invocations/results appear inline in the stream.
    """

    engine_label: str = "openai"

    def __init__(self, config: "AppConfig") -> None:
        self.config = config
        self.settings = config.local_inference
        self.model = self.settings.model
        self.name = f"{self.engine_label}:{self.model}"

    def _client(self):
        try:
            from openai import OpenAI  # lazy import
        except Exception as exc:  # noqa: BLE001
            raise LLMProviderError(
                "openai package not installed. Install with: pip install openai"
            ) from exc
        logger.debug(
            "Creating OpenAI client engine=%s base_url=%s timeout=%ss api_key=%s",
            self.engine_label,
            self.settings.openai_base_url,
            self.settings.request_timeout,
            "set" if self.settings.api_key else "EMPTY",
        )
        return OpenAI(
            base_url=self.settings.openai_base_url,
            # vLLM requires a non-empty key even when it enforces none.
            api_key=self.settings.api_key or "EMPTY",
            timeout=self.settings.request_timeout,
            # Bound automatic retries so a rate-limit 429 can't spin in a long
            # exponential back-off loop (the caller aborts on fatal errors).
            max_retries=max(0, self.settings.request_max_retries),
        )

    def _connection_hint(self, exc: Exception) -> str:
        """Build an actionable error message for connection-type failures."""
        url = self.settings.openai_base_url
        return (
            f"Could not reach the {self.engine_label} server at {url} ({exc}). "
            f"Check the server is running and reachable from here. In Docker, the "
            f"engine may be in another container — set SPARKS_BASE_URL="
            f"http://<host>:<port> (note: 0.0.0.0 is a bind address, not a "
            f"connect address)."
        )

    def stream_chat(
        self,
        messages: List[Dict[str, object]],
        tools: Optional[List[Dict[str, object]]] = None,
    ) -> Iterator[StreamEvent]:
        client = self._client()

        # Guard against oversized requests: truncate large tool results, trim old
        # history, and clamp the completion budget so prompt + completion stays
        # under the configured cap (prevents provider 413s / rate-limit loops).
        max_tokens = self.settings.max_tokens
        if self.settings.max_request_tokens > 0 or self.settings.max_tool_result_chars > 0:
            from context_budget import estimate_request_tokens, fit_request

            messages, max_tokens = fit_request(
                messages,
                tools,
                max_request_tokens=self.settings.max_request_tokens,
                configured_max_tokens=self.settings.max_tokens,
                max_tool_result_chars=self.settings.max_tool_result_chars,
            )
            logger.debug(
                "Request budget: ~%d prompt tokens, max_tokens=%d (cap=%d)",
                estimate_request_tokens(messages, tools),
                max_tokens,
                self.settings.max_request_tokens,
            )

        kwargs: Dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.info(
            "Chat request -> %s model=%s msgs=%d tools=%s",
            self.settings.openai_base_url,
            self.model,
            len(messages),
            bool(tools),
        )
        try:
            stream = client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # Classify: connection issues get an actionable hint; rate-limit /
            # request-too-large / auth (4xx) are FATAL — retrying the turn would
            # just loop, so the CLI aborts on these. Traceback logged at debug.
            name = type(exc).__name__
            status = getattr(exc, "status_code", None)
            low = str(exc).lower()
            is_conn = "Connection" in name or "Timeout" in name or "APIConnection" in name
            is_rate_or_size = (
                status in (413, 429)
                or "rate_limit" in low
                or "too large" in low
                or "tokens per minute" in low
                or "context length" in low
                or "maximum context" in low
            )
            is_client_4xx = isinstance(status, int) and 400 <= status < 500

            if is_conn:
                msg, fatal = self._connection_hint(exc), False
            elif is_rate_or_size:
                msg = (
                    f"{self.engine_label} rejected the request as too large / "
                    f"rate-limited ({exc}). The prompt + reserved completion "
                    "exceeded the model's token-per-minute or context limit. "
                    "Lower local_inference_settings.max_request_tokens or "
                    "max_tokens, reduce tool output, or upgrade your API tier. "
                    "Aborting this turn (not retrying) to avoid a request loop."
                )
                fatal = True
            elif is_client_4xx:
                msg = f"{self.engine_label} request rejected (HTTP {status}): {exc}"
                fatal = True
            else:
                msg, fatal = f"{self.engine_label} request failed: {exc}", False

            logger.error("Chat request failed (%s): %s", name, exc)
            logger.debug("Chat request traceback", exc_info=True)
            yield StreamEvent(type="error", text=msg, fatal=fatal)
            yield StreamEvent(type="done", finish_reason="error")
            return

        # Accumulate tool-call fragments by index, then flush each as ONE
        # complete tool_call event before the first content token (tool deltas
        # always precede content in vLLM's stream). This keeps the event
        # contract identical to the mock provider: one tool_call per tool, with
        # full arguments, emitted before the answer.
        acc: Dict[int, Dict[str, str]] = {}
        tools_flushed = False
        finish_reason: Optional[str] = None

        def _flush_tools() -> Iterator[StreamEvent]:
            for idx in sorted(acc):
                slot = acc[idx]
                if slot["name"]:
                    yield StreamEvent(
                        type="tool_call",
                        name=slot["name"],
                        arguments=slot["arguments"],
                    )

        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason or finish_reason

            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield StreamEvent(type="reasoning", text=reasoning)

            for tc in getattr(delta, "tool_calls", None) or []:
                idx = getattr(tc, "index", 0) or 0
                slot = acc.setdefault(idx, {"name": "", "arguments": ""})
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments

            if getattr(delta, "content", None):
                if acc and not tools_flushed:
                    yield from _flush_tools()
                    tools_flushed = True
                yield StreamEvent(type="content", text=delta.content)

        if acc and not tools_flushed:
            yield from _flush_tools()

        yield StreamEvent(type="done", finish_reason=finish_reason or "stop")


class VLLMProvider(OpenAICompatibleProvider):
    """Local vLLM engine. Executes MCP tools server-side via ``--tool-server``."""

    engine_label = "vllm"
    supports_server_side_tools = True
    supports_tools = True


class GroqProvider(OpenAICompatibleProvider):
    """Remote Groq API (OpenAI-compatible, fast LPU inference).

    Speaks the OpenAI API at ``https://api.groq.com/openai/v1`` and returns
    tool_calls the CLI executes client-side (in-process), feeding results back.
    It cannot reach a local ``--tool-server``, so ``supports_server_side_tools``
    is False.
    """

    engine_label = "groq"
    supports_server_side_tools = False
    supports_tools = True


# --------------------------------------------------------------------------- #
# Offline mock — deterministic, dependency-free (tests / CI / no-network demos)
# --------------------------------------------------------------------------- #
class MockStreamingProvider(LLMProvider):
    """Deterministic provider that performs NO model forward pass.

    It emits a fixed, clearly-labelled event sequence — reasoning, a tool call,
    a tool result, then a final answer — so the entire CLI pipeline (memory,
    reasoning display, tool-invocation logging, final render) can be exercised
    offline. If the latest user message mentions a sector it "calls" the sector
    tool; otherwise the single-quote tool.
    """

    name = "mock:offline"
    supports_server_side_tools = True  # simulates the full tool flow for demos/tests
    supports_tools = True

    _PHASE_MARKERS = ("PLANNING STEP", "ACTING STEP", "ANSWER STEP")

    @staticmethod
    def _emit_text(text: str) -> Iterator[StreamEvent]:
        # Stream in a few chunks so the renderer exercises incremental writes.
        for line in text.splitlines(keepends=True):
            yield StreamEvent(type="content", text=line)

    def stream_chat(
        self,
        messages: List[Dict[str, object]],
        tools: Optional[List[Dict[str, object]]] = None,
    ) -> Iterator[StreamEvent]:
        # Last user message = the phase directive (in the plan/act/reflect loop).
        directive = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                directive = str(m.get("content", ""))
                break

        # Current question = the LATEST user message that is NOT a phase
        # directive (so multi-turn histories pick the current turn's question).
        question = ""
        question_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if m.get("role") == "user":
                c = str(m.get("content", ""))
                if not any(mk in c for mk in self._PHASE_MARKERS):
                    question = c
                    question_idx = i
                    break

        wants_sector = "sector" in question.lower()
        tool_name = (
            "get_indian_sector_performance" if wants_sector else "get_indian_stock_quote"
        )
        args = (
            json.dumps({"sector": "IT"})
            if wants_sector
            else json.dumps({"query": "Reliance", "exchange": "NS"})
        )
        # Only count tool results from the CURRENT turn (after the current
        # question), so prior turns' tool messages don't end this turn early.
        has_tool_result = any(
            m.get("role") == "tool" for m in messages[question_idx + 1:]
        )

        # ---- PLANNING phase ------------------------------------------------
        if "PLANNING STEP" in directive:
            plan = (
                "[MOCK PLAN]\n"
                "1. Identify the relevant holding/sector from the portfolio context.\n"
                f"2. Call {tool_name} to fetch the live figure.\n"
                "3. Interpret the result and answer."
            )
            yield from self._emit_text(plan)
            yield StreamEvent(type="done", finish_reason="stop")
            return

        # ---- ACTING phase --------------------------------------------------
        if "ACTING STEP" in directive:
            if has_tool_result:
                # Data already gathered; nothing more to call.
                yield StreamEvent(type="done", finish_reason="stop")
                return
            yield StreamEvent(
                type="reasoning", text="[MOCK] Calling the tool my plan requires."
            )
            yield StreamEvent(type="tool_call", name=tool_name, arguments=args)
            yield StreamEvent(type="done", finish_reason="tool_calls")
            return

        # ---- ANSWER (reflect + answer) phase -------------------------------
        if "ANSWER STEP" in directive:
            text = (
                "[MOCK REFLECTION] The (mock) tool data is in hand and is "
                "consistent with the portfolio context.\n"
                "ANSWER: [MOCK LLM OUTPUT] Based on the mock tool data, here is the "
                "deterministic offline answer. This is not financial advice; no "
                "trades executed."
            )
            yield from self._emit_text(text)
            yield StreamEvent(type="done", finish_reason="stop")
            return

        # ---- Fallback: single-shot (non-phased) direct call ----------------
        yield StreamEvent(
            type="reasoning",
            text="[MOCK] No model was run. Deciding which MCP tool answers this.",
        )
        yield StreamEvent(type="tool_call", name=tool_name, arguments=args)
        for token in (
            "[MOCK LLM OUTPUT] ",
            "Based on the (mock) tool data, ",
            "this is a deterministic offline answer. ",
            "Not financial advice; no trades executed.",
        ):
            yield StreamEvent(type="content", text=token)
        yield StreamEvent(type="done", finish_reason="stop")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
_PROVIDERS = {
    "vllm": VLLMProvider,
    "groq": GroqProvider,
    "mock": MockStreamingProvider,
}


def get_provider(config: "AppConfig") -> LLMProvider:
    """Construct the provider selected by ``model_selection.provider``."""
    key = config.model_selection.provider
    cls = _PROVIDERS.get(key)
    if cls is None:
        raise LLMProviderError(
            f"Unsupported provider '{key}'. Choose one of: {sorted(_PROVIDERS)}."
        )
    if cls is MockStreamingProvider:
        return MockStreamingProvider()
    return cls(config)
