"""Graphviz visualization for sector/portfolio knowledge graphs.

Renders a :class:`sector_graph.SectorGraph` to **Graphviz DOT** so the model's
financial reasoning can be inspected visually:

* **Nodes** are tickers, filled by sector and labelled with a compact feature
  digest (momentum, RSI, sentiment, P/E) so you can eyeball each stock.
* **Edges** are the agent's associations, styled by validation status:
  ``validated`` = solid green, ``proposed`` = dashed grey, ``rejected`` =
  dotted red. The edge label carries the relation, weight, and (tooltip) the
  reasoning trail from each validation pass — i.e. *why* the model drew it.

DOT is emitted with the standard library only (pure string building). If the
Graphviz ``dot`` binary happens to be installed, ``visualize_sector_graph`` also
renders an image (SVG by default); otherwise it still writes the ``.dot`` source
so nothing depends on a system package being present.

Scope guardrail: read-only. This module only visualizes existing analysis.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Sector -> fill color (Graphviz X11 names). Unknown sectors fall back to grey.
_SECTOR_COLORS: Dict[str, str] = {
    "it": "#cfe8ff",
    "technology": "#cfe8ff",
    "banking": "#d8f5d0",
    "financials": "#d8f5d0",
    "energy": "#ffe6cc",
    "auto": "#ffd9d9",
    "pharma": "#e8d9ff",
    "fmcg": "#fff2cc",
}

# Edge status -> (color, style).
_STATUS_STYLE: Dict[str, tuple] = {
    "validated": ("#2e7d32", "solid"),
    "proposed": ("#9e9e9e", "dashed"),
    "rejected": ("#c62828", "dotted"),
}


class GraphVizError(ValueError):
    """Raised when a graph cannot be rendered."""


def _esc(text: object) -> str:
    """Escape a string for a DOT double-quoted label."""
    return str(text).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _fmt(value: object) -> str:
    """Compact numeric formatting for node labels ('—' when missing)."""
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _node_label(node) -> str:
    """Build a compact multi-line node label from the node's feature bundle."""
    f = getattr(node, "features", {}) or {}
    sent = f.get("sentiment", {}) or {}
    asof = f.get("asof_stats", {}) or {}
    if asof:  # point-in-time (as-of) node
        lines = [
            _esc(node.ticker),
            f"ret {_fmt(asof.get('return_pct'))}%",
            f"vol {_fmt(asof.get('volatility_pct'))}%",
            f"Sharpe {_fmt(asof.get('sharpe'))}",
            f"sent {_fmt(sent.get('score'))}",
        ]
    else:
        alpha = f.get("alpha_factors", {}) or {}
        ind = f.get("indicators", {}) or {}
        fund = f.get("fundamentals", {}) or {}
        lines = [
            _esc(node.ticker),
            f"mom3m {_fmt(alpha.get('momentum_3m'))}%",
            f"RSI {_fmt(ind.get('rsi_14'))}",
            f"sent {_fmt(sent.get('score'))}",
            f"P/E {_fmt(fund.get('trailing_pe'))}",
        ]
    return "\\n".join(lines)


def _edge_tooltip(edge) -> str:
    """Assemble the validation reasoning trail into an edge tooltip."""
    passes = getattr(edge, "validations", []) or []
    if not passes:
        rationale = (getattr(edge, "evidence", {}) or {}).get("agent_rationale")
        return _esc(rationale or "candidate edge (not yet validated)")
    trail = [f"{p['pass']}. {p['verdict']}: {p['reasoning']}" for p in passes]
    return _esc(" | ".join(trail))


def render_graph_dot(graph) -> str:
    """Render a :class:`sector_graph.SectorGraph` to a DOT source string."""
    nodes = getattr(graph, "nodes", {})
    edges = getattr(graph, "edges", {})
    if not nodes:
        raise GraphVizError("Graph has no nodes to visualize.")

    sectors = ", ".join(getattr(graph, "sectors", []) or [])
    title = f"Sector graph {getattr(graph, 'graph_id', '')} — {sectors}"

    out: List[str] = [
        "digraph sector_graph {",
        "  rankdir=LR;",
        '  graph [fontname="Helvetica", labelloc="t", '
        f'label="{_esc(title)}"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", '
        'fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=9];',
    ]

    for ticker, node in nodes.items():
        color = _SECTOR_COLORS.get(getattr(node, "sector", ""), "#eeeeee")
        out.append(
            f'  "{_esc(ticker)}" [label="{_node_label(node)}", '
            f'fillcolor="{color}", tooltip="sector: {_esc(node.sector)}"];'
        )

    for edge in edges.values():
        color, style = _STATUS_STYLE.get(edge.status, ("#9e9e9e", "dashed"))
        weight = getattr(edge, "weight", 0.0) or 0.0
        label = f"{_esc(edge.relation)} ({weight:+.2f})"
        out.append(
            f'  "{_esc(edge.source)}" -> "{_esc(edge.target)}" '
            f'[label="{label}", color="{color}", fontcolor="{color}", '
            f'style="{style}", dir=none, tooltip="{_edge_tooltip(edge)}"];'
        )

    out.append("}")
    return "\n".join(out)


def _render_image(dot_path: Path, fmt: str) -> Optional[str]:
    """Render an image from a .dot file if the Graphviz binary is available."""
    dot_bin = shutil.which("dot")
    if not dot_bin:
        return None
    img_path = dot_path.with_suffix(f".{fmt}")
    try:
        subprocess.run(
            [dot_bin, f"-T{fmt}", str(dot_path), "-o", str(img_path)],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except Exception:  # noqa: BLE001 - image is best-effort; DOT is the contract
        return None
    return str(img_path)


def visualize_sector_graph(
    graph_id: str,
    *,
    out_dir: str = "knowledge/market_data/graphs",
    image_format: str = "svg",
) -> Dict[str, object]:
    """Write a DOT (and, if Graphviz is installed, an image) for a graph.

    Loads the graph by id (from memory or disk) and returns the paths written
    plus a small status breakdown, so the caller can open the file to evaluate
    the model's reasoning.
    """
    from sector_graph import get_graph_object  # lazy import avoids a cycle

    graph = get_graph_object(graph_id)
    dot = render_graph_dot(graph)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dot_file = out_path / f"{graph_id}.dot"
    dot_file.write_text(dot, encoding="utf-8")

    image_file = _render_image(dot_file, image_format)

    by_status: Dict[str, int] = {}
    for e in graph.edges.values():
        by_status[e.status] = by_status.get(e.status, 0) + 1

    return {
        "graph_id": graph_id,
        "dot_path": str(dot_file),
        "image_path": image_file,
        "image_rendered": image_file is not None,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "edges_by_status": by_status,
        "note": (
            "Open the .dot in any Graphviz viewer if no image was rendered "
            "(the 'dot' binary was not found on PATH)."
        ),
    }


# --------------------------------------------------------------------------- #
# OpenAI-format tool schema (mirrors the MCP tool; passed to the LLM payload).
# --------------------------------------------------------------------------- #
GRAPH_VIZ_TOOL_SPECS: List[Dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "visualize_sector_graph",
            "description": (
                "Render a previously built sector/portfolio graph to Graphviz "
                "DOT (and an SVG image if Graphviz is installed) so its nodes, "
                "edges and validation status can be inspected visually to "
                "evaluate the financial reasoning. Returns the written file "
                "paths."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "graph_id": {
                        "type": "string",
                        "description": "Id returned by build_sector_graph / build_portfolio_graph.",
                    },
                    "image_format": {
                        "type": "string",
                        "enum": ["svg", "png", "pdf"],
                        "description": "Image format if Graphviz is available (default svg).",
                    },
                },
                "required": ["graph_id"],
            },
        },
    },
]
