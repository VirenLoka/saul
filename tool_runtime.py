"""Tool execution runtime for the agentic loop.

When the model emits a tool call, the CLI must actually run the tool and feed
the result back so the model can continue. Two executors share one interface
``__call__(name: str, arguments: str | dict) -> str`` (returns a JSON string):

  * :class:`MCPToolExecutor`     — invokes tools on the running FastMCP server
    over the MCP protocol (the production path for vLLM). ``fastmcp`` is
    imported lazily so this module imports fine without it.
  * :class:`InProcessToolExecutor` — calls the ``market_data`` functions
    directly (offline; used by the mock provider and the test suite).

Both never raise to the caller: failures are returned as a JSON ``{"error": …}``
string so the model can read and react to them.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from config_loader import MarketDataSettings, NewsApiSettings

logger = logging.getLogger("saul.tools")

ArgsType = Union[str, dict, None]


def parse_args(arguments: ArgsType) -> dict:
    """Coerce a tool-call argument payload (JSON string or dict) to a dict."""
    if isinstance(arguments, dict):
        return arguments
    s = (arguments or "").strip()
    if not s:
        return {}
    return json.loads(s)


def _err(msg: str) -> str:
    return json.dumps({"error": msg})


# --------------------------------------------------------------------------- #
# Production: execute against the running FastMCP server over MCP
# --------------------------------------------------------------------------- #
class MCPToolExecutor:
    """Calls tools on the FastMCP server at ``url`` via an MCP client.

    Uses ``asyncio.run`` per call (the CLI is synchronous and not inside an
    event loop), opening a short-lived client connection each time. Robust and
    simple for an interactive CLI where tool calls are infrequent.
    """

    def __init__(self, url: str, timeout: float = 30.0) -> None:
        self.url = url
        self.timeout = timeout

    def __call__(self, name: str, arguments: ArgsType) -> str:
        try:
            args = parse_args(arguments)
        except Exception as exc:  # noqa: BLE001
            return _err(f"could not parse arguments for {name}: {exc}")
        logger.info("MCP tool call -> %s %s @ %s", name, args, self.url)
        try:
            result = asyncio.run(self._call(name, args))
            logger.debug("MCP tool result [%s]: %s", name, result[:300])
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error("MCP tool '%s' failed: %s", name, exc)
            logger.debug("MCP tool traceback", exc_info=True)
            return _err(
                f"tool '{name}' failed against MCP server {self.url}: {exc}"
            )

    async def _call(self, name: str, args: dict) -> str:
        try:
            from fastmcp import Client  # lazy import
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "fastmcp not installed. Install with: pip install fastmcp"
            ) from exc

        # Client infers the transport from the URL (e.g. an /sse suffix).
        async with Client(self.url) as client:
            res = await client.call_tool(name, args)
            return _extract_result_text(res)


def _extract_result_text(res: object) -> str:
    """Normalize a FastMCP CallToolResult into a JSON/text string.

    Handles the common shapes across fastmcp versions: ``.data`` (deserialized
    structured output), ``.structured_content``, or a list of content blocks.
    """
    data = getattr(res, "data", None)
    if data is not None:
        try:
            return json.dumps(data)
        except TypeError:
            return str(data)

    structured = getattr(res, "structured_content", None)
    if structured:
        try:
            return json.dumps(structured)
        except TypeError:
            return str(structured)

    content = getattr(res, "content", None)
    if content:
        texts = [getattr(block, "text", "") for block in content]
        joined = "\n".join(t for t in texts if t)
        if joined:
            return joined

    return str(res)


# --------------------------------------------------------------------------- #
# Offline: execute the market_data functions directly (mock / tests)
# --------------------------------------------------------------------------- #
class InProcessToolExecutor:
    """Runs the market-data tools in-process, with no network or MCP server.

    Used by the mock provider and tests. ``use_live`` defaults to False so the
    deterministic mock data layer is used.
    """

    def __init__(
        self,
        default_exchange: str = "NS",
        use_live: bool = False,
        cache_ttl_seconds: int = 60,
        newsapi: "NewsApiSettings | None" = None,
    ) -> None:
        self.default_exchange = default_exchange
        self.use_live = use_live
        self.cache_ttl_seconds = cache_ttl_seconds
        self.newsapi = newsapi

    @classmethod
    def from_settings(
        cls,
        md: "MarketDataSettings",
        *,
        use_live: bool | None = None,
        newsapi: "NewsApiSettings | None" = None,
    ) -> "InProcessToolExecutor":
        return cls(
            default_exchange=md.default_exchange,
            use_live=md.use_live if use_live is None else use_live,
            cache_ttl_seconds=md.cache_ttl_seconds,
            newsapi=newsapi,
        )

    def __call__(self, name: str, arguments: ArgsType) -> str:
        from market_data import (  # local import keeps module deps light
            MarketDataError,
            get_sector_performance,
            get_stock_quote,
        )
        from news_data import NewsDataError, get_stock_news

        try:
            args = parse_args(arguments)
        except Exception as exc:  # noqa: BLE001
            return _err(f"could not parse arguments for {name}: {exc}")

        logger.info("In-process tool call -> %s %s", name, args)
        try:
            if name == "get_indian_stock_quote":
                result = get_stock_quote(
                    args.get("query", ""),
                    args.get("exchange", self.default_exchange),
                    use_live=self.use_live,
                    cache_ttl_seconds=self.cache_ttl_seconds,
                )
            elif name == "get_indian_sector_performance":
                result = get_sector_performance(
                    args.get("sector", ""),
                    args.get("exchange", self.default_exchange),
                    use_live=self.use_live,
                    cache_ttl_seconds=self.cache_ttl_seconds,
                )
            elif name == "get_stock_news":
                ns = self.newsapi
                result = get_stock_news(
                    args.get("query", ""),
                    api_key=ns.api_key if ns else "",
                    base_url=ns.base_url if ns else "https://newsapi.org/v2/everything",
                    page_size=int(args.get("max_articles") or (ns.page_size if ns else 8)),
                    language=ns.language if ns else "en",
                    sort_by=ns.sort_by if ns else "publishedAt",
                    lookback_days=ns.lookback_days if ns else 7,
                    # honour both this executor's offline flag and the news config:
                    # in-process mode is used by the offline mock, so default to mock
                    # unless live is explicitly enabled on both.
                    use_live=self.use_live and (ns.use_live if ns else True),
                )
            else:
                return _err(f"unknown tool '{name}'")
            return json.dumps(result)
        except (MarketDataError, NewsDataError) as exc:
            return _err(str(exc))
