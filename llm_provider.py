"""LLM provider abstraction layer.

Decouples the CLI from the inference backend. Everything downstream depends only
on :class:`LLMProvider` and the normalized :class:`StreamEvent` stream; the
concrete backend (vLLM over an OpenAI-compatible API, or an offline mock) is
selected at runtime from config.

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


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class LLMProvider(abc.ABC):
    """Abstract base every concrete LLM backend implements."""

    name: str = "abstract"
    #: True if the engine executes attached tools server-side (e.g. vLLM
    #: --tool-server). When False, the CLI runs tool-free for this engine.
    supports_server_side_tools: bool = False

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
# OpenAI-compatible streaming backend (shared by vLLM and Ollama)
# --------------------------------------------------------------------------- #
class OpenAICompatibleProvider(LLMProvider):
    """Streams chat completions from any OpenAI-compatible local engine.

    Both vLLM and Ollama expose this API at ``http://<host>:<port>/v1``, so the
    request/stream-parsing logic is identical; subclasses only differ in their
    label and whether the engine executes tools server-side.

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
            # vLLM requires a non-empty key; Ollama ignores it. "EMPTY" satisfies both.
            api_key=self.settings.api_key or "EMPTY",
            timeout=self.settings.request_timeout,
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
        kwargs: Dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
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
            # Connection-class failures get an actionable hint; everything else
            # is surfaced verbatim. Full traceback is logged at debug level.
            name = type(exc).__name__
            is_conn = "Connection" in name or "Timeout" in name or "APIConnection" in name
            msg = self._connection_hint(exc) if is_conn else (
                f"{self.engine_label} request failed: {exc}"
            )
            logger.error("Chat request failed (%s): %s", name, exc)
            logger.debug("Chat request traceback", exc_info=True)
            yield StreamEvent(type="error", text=msg)
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


class OllamaProvider(OpenAICompatibleProvider):
    """Local Ollama daemon (OpenAI-compatible at /v1).

    Ollama has no ``--tool-server`` equivalent, so the MCP tools are not executed
    server-side; the CLI runs this engine tool-free (it still has the customer
    portfolio analysis in its system context).
    """

    engine_label = "ollama"
    supports_server_side_tools = False


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

    def stream_chat(
        self,
        messages: List[Dict[str, object]],
        tools: Optional[List[Dict[str, object]]] = None,
    ) -> Iterator[StreamEvent]:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = str(m.get("content", ""))
                break
        text = last_user.lower()

        if "sector" in text or any(s in text for s in ("it sector", "banking", "auto")):
            tool_name = "get_indian_sector_performance"
            args = json.dumps({"sector": "IT"})
        else:
            tool_name = "get_indian_stock_quote"
            args = json.dumps({"query": "Reliance", "exchange": "NS"})

        yield StreamEvent(
            type="reasoning",
            text="[MOCK] No model was run. Deciding which MCP tool answers this.",
        )
        yield StreamEvent(type="tool_call", name=tool_name, arguments=args)
        yield StreamEvent(
            type="tool_result",
            name=tool_name,
            text='{"source": "mock", "note": "deterministic offline result"}',
        )
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
    "ollama": OllamaProvider,
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
