"""FastMCP tool server — real-time Indian market data.

A standalone MCP server exposing read-only Indian market data tools. It is
launched separately from the model and ATTACHED to the vLLM engine via
``--tool-server`` (see serve.sh / README), so vLLM can invoke these tools
server-side during generation.

Run directly:
    python mcp_server.py
    # honours mcp.host / mcp.port / mcp.transport from config.yaml

The actual data logic lives in ``market_data.py`` (dependency-free and
unit-tested); this file is only the thin MCP wiring. ``fastmcp`` is imported
lazily inside ``build_server`` so the module can be imported (and the tools'
logic tested) without fastmcp installed.

Scope guardrail: every tool here is observational. None place orders or take
financial actions.
"""

from __future__ import annotations

from typing import Any

from config_loader import AppConfig, load_config
from market_data import get_sector_performance, get_stock_quote


def build_server(config: "AppConfig | None" = None) -> Any:
    """Construct and return a configured FastMCP server instance.

    Tools read live/mock behaviour from ``config.mcp.market_data`` so nothing
    about the data source is hardcoded.
    """
    try:
        from fastmcp import FastMCP  # lazy import — only needed to actually serve
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "fastmcp is not installed. Install with: pip install fastmcp"
        ) from exc

    cfg = config or load_config()
    md = cfg.mcp.market_data

    mcp = FastMCP(name="indian-market-data")

    @mcp.tool
    def get_indian_stock_quote(query: str, exchange: str = "") -> dict:
        """Fetch a real-time quote for a single Indian stock.

        Args:
            query: Company name (e.g. "Reliance", "TCS") or ticker (e.g. "INFY").
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
        """
        return get_stock_quote(
            query,
            exchange or md.default_exchange,
            use_live=md.use_live,
            cache_ttl_seconds=md.cache_ttl_seconds,
        )

    @mcp.tool
    def get_indian_sector_performance(sector: str, exchange: str = "") -> dict:
        """Fetch aggregate performance for an Indian market sector.

        Args:
            sector: Sector name, e.g. "IT", "banking", "auto", "pharma", "fmcg".
            exchange: "NS" (NSE) or "BO" (BSE). Defaults to the configured exchange.
        """
        return get_sector_performance(
            sector,
            exchange or md.default_exchange,
            use_live=md.use_live,
            cache_ttl_seconds=md.cache_ttl_seconds,
        )

    return mcp


def main() -> None:  # pragma: no cover - exercised only when serving live
    cfg = load_config()
    server = build_server(cfg)
    transport = cfg.mcp.transport
    print(
        f"[mcp_server] starting FastMCP 'indian-market-data' on "
        f"{cfg.mcp.host}:{cfg.mcp.port} (transport={transport}, "
        f"live={cfg.mcp.market_data.use_live})"
    )
    # FastMCP networked transports take host/port; stdio takes neither.
    if transport in {"sse", "streamable-http", "http"}:
        server.run(transport=transport, host=cfg.mcp.host, port=cfg.mcp.port)
    else:
        server.run(transport=transport)


if __name__ == "__main__":
    main()
