"""Render PyTorch FX graphs and TVM Relax call graphs as SVG via Graphviz.

Functions
---------
fx_graph_to_svg   – torch.fx.Graph → SVG string
fx_node_table_html – node table → HTML for Gradio
relax_callgraph_to_svg – tvm.ir.IRModule → SVG string (best-effort)
"""

from __future__ import annotations

import html
import logging
from typing import Any, List, Optional

log = logging.getLogger(__name__)

# Graphviz is optional at import time
try:
    import graphviz  # type: ignore
    _GV_AVAILABLE = True
except ImportError:
    graphviz = None  # type: ignore
    _GV_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
# Colour palette for FX node op types
# ──────────────────────────────────────────────────────────────────────

_OP_COLORS = {
    "placeholder":    "#9E9E9E",   # gray
    "call_module":    "#42A5F5",   # blue
    "call_function":  "#66BB6A",   # green
    "call_method":    "#AB47BC",   # purple
    "get_attr":       "#FFA726",   # orange
    "output":         "#EF5350",   # red
}
_DEFAULT_COLOR = "#BDBDBD"


# ──────────────────────────────────────────────────────────────────────
# FX Graph → SVG
# ──────────────────────────────────────────────────────────────────────

def fx_graph_to_svg(fx_graph: Any, max_nodes: int = 300) -> str:
    """Render a ``torch.fx.Graph`` as an SVG string.

    Falls back to a text description if graphviz is not installed.
    """
    if not _GV_AVAILABLE:
        return _fx_graph_text_fallback(fx_graph)

    dot = graphviz.Digraph(
        format="svg",
        graph_attr={
            "rankdir": "TB",
            "fontname": "Helvetica",
            "fontsize": "10",
            "bgcolor": "transparent",
        },
        node_attr={
            "shape": "box",
            "style": "filled,rounded",
            "fontname": "Helvetica",
            "fontsize": "9",
            "margin": "0.15,0.08",
        },
        edge_attr={"fontsize": "8", "arrowsize": "0.6"},
    )

    nodes = list(fx_graph.nodes)
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]

    node_ids: dict = {}
    for i, node in enumerate(nodes):
        nid = f"n{i}"
        node_ids[node.name] = nid

        label = _fx_node_label(node)
        color = _OP_COLORS.get(node.op, _DEFAULT_COLOR)
        dot.node(nid, label=label, fillcolor=color, fontcolor="white")

    for node in nodes:
        src_id = node_ids.get(node.name)
        if src_id is None:
            continue
        for arg in node.args:
            if hasattr(arg, "name"):
                dst_id = node_ids.get(arg.name)
                if dst_id is not None:
                    dot.edge(dst_id, src_id)

    try:
        return dot.pipe(format="svg").decode("utf-8")
    except Exception as exc:
        log.warning("graphviz dot binary failed (%s), using text fallback", exc)
        return _fx_graph_text_fallback(fx_graph)


def _fx_node_label(node: Any) -> str:
    target = str(node.target) if node.target is not None else ""
    if "." in target and len(target) > 40:
        target = "…" + target.rsplit(".", 1)[-1]
    elif len(target) > 40:
        target = target[:37] + "…"

    if node.op == "placeholder":
        return f"input\\n{node.name}"
    if node.op == "output":
        return "output"
    return f"{node.name}\\n{node.op}: {target}"


def _fx_graph_text_fallback(fx_graph: Any) -> str:
    """Plain-text fallback when graphviz is not installed."""
    lines = ["<pre style='font-size:12px'>"]
    for node in fx_graph.nodes:
        lines.append(
            f"  {node.name:25s}  op={node.op:16s}  target={node.target}"
        )
    lines.append("</pre>")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# FX Node Table → HTML
# ──────────────────────────────────────────────────────────────────────

def fx_node_table_html(node_table: List[dict]) -> str:
    """Render the node table as an HTML <table> for Gradio display."""
    header = (
        "<tr>"
        "<th>#</th><th>Name</th><th>Op</th>"
        "<th>Target</th><th>Inputs</th><th>Users</th>"
        "</tr>"
    )
    rows = []
    for i, row in enumerate(node_table):
        color = _OP_COLORS.get(row["op"], _DEFAULT_COLOR)
        badge = (
            f'<span style="background:{color};color:#fff;'
            f'padding:1px 6px;border-radius:3px;font-size:11px">'
            f'{html.escape(row["op"])}</span>'
        )
        rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td><code>{html.escape(row['name'])}</code></td>"
            f"<td>{badge}</td>"
            f"<td><code>{html.escape(row['target'])}</code></td>"
            f"<td>{html.escape(row['inputs'])}</td>"
            f"<td>{row['num_users']}</td>"
            f"</tr>"
        )
    return (
        '<table style="border-collapse:collapse;width:100%;font-size:12px">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


# ──────────────────────────────────────────────────────────────────────
# TVM Relax Call Graph → SVG
# ──────────────────────────────────────────────────────────────────────

def relax_callgraph_to_svg(mod: Any) -> str:
    """Render the Relax main function's call targets as a graph.

    Each ``R.call_tir`` / ``R.call_dps_packed`` becomes a node; edges
    follow dataflow order.  Falls back to a function-list text if
    parsing fails.
    """
    if not _GV_AVAILABLE:
        return _relax_text_fallback(mod)

    try:
        return _build_relax_callgraph(mod)
    except Exception as exc:
        log.warning("Relax call-graph rendering failed: %s", exc)
        return _relax_text_fallback(mod)


def _build_relax_callgraph(mod: Any) -> str:
    """Parse the printed IR to extract call_tir targets and build a DOT graph."""
    import re

    ir_text = mod.script() if hasattr(mod, "script") else str(mod)

    call_pattern = re.compile(
        r'(?:call_tir|call_dps_packed)\s*\(\s*(?:cls\.)?(\w+)'
    )
    targets = call_pattern.findall(ir_text)
    if not targets:
        return _relax_text_fallback(mod)

    seen_order: list = []
    seen_set: set = set()
    for t in targets:
        if t not in seen_set:
            seen_order.append(t)
            seen_set.add(t)

    dot = graphviz.Digraph(
        format="svg",
        graph_attr={
            "rankdir": "TB",
            "fontname": "Helvetica",
            "fontsize": "10",
            "bgcolor": "transparent",
        },
        node_attr={
            "shape": "box",
            "style": "filled,rounded",
            "fontname": "Courier",
            "fontsize": "9",
            "fillcolor": "#E3F2FD",
            "margin": "0.12,0.06",
        },
    )

    dot.node("input", label="input", shape="ellipse", fillcolor="#9E9E9E", fontcolor="white")
    prev = "input"
    for t in seen_order:
        dot.node(t, label=t)
        dot.edge(prev, t)
        prev = t
    dot.node("output", label="output", shape="ellipse", fillcolor="#EF5350", fontcolor="white")
    dot.edge(prev, "output")

    return dot.pipe(format="svg").decode("utf-8")


_tvm_prevent_free: list = []


def _relax_text_fallback(mod: Any) -> str:
    """Plain-text listing of IRModule global functions."""
    try:
        names = []
        items = list(mod.functions.items())
        _tvm_prevent_free.append(items)
        for gv, func in items:
            kind = type(func).__name__
            name = gv.name_hint if hasattr(gv, "name_hint") else str(gv)
            names.append(f"  {name:30s}  ({kind})")
        return "<pre style='font-size:12px'>" + "\n".join(names) + "</pre>"
    except Exception:
        return "<pre>Unable to list functions</pre>"
