"""Utilities for displaying and comparing TVM IR and TIR artifacts.

Functions
---------
highlight_ir          – truncate + add line numbers for Gradio display
ir_diff               – unified diff between two IR snapshots
format_pass_delta     – one-line summary of what a pass changed
operator_table_html   – list[dict] → HTML table of extracted operators
tir_ast_tree_html     – AST summary dict → collapsible HTML tree
tir_loop_table_html   – AST summary dict → HTML table of loop nest
tir_buffer_table_html – AST summary dict → HTML table of buffers
"""

from __future__ import annotations

import difflib
import html
import logging
from typing import Any, Dict, List

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# IR text display
# ──────────────────────────────────────────────────────────────────────

def highlight_ir(ir_text: str, max_lines: int = 200) -> str:
    """Return IR text with line numbers, truncated to *max_lines*.

    Suitable for use with ``gr.Code(language="python")`` or plain text.
    """
    lines = ir_text.split("\n")
    total = len(lines)
    if total > max_lines:
        lines = lines[:max_lines]
        lines.append(f"# ... ({total - max_lines} more lines omitted)")

    width = len(str(len(lines)))
    numbered = [f"{i + 1:>{width}} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered)


# ──────────────────────────────────────────────────────────────────────
# IR diff
# ──────────────────────────────────────────────────────────────────────

def ir_diff(
    before: str,
    after: str,
    before_label: str = "before",
    after_label: str = "after",
    context_lines: int = 3,
) -> str:
    """Produce a unified diff between two IR snapshots.

    Returns a string suitable for ``gr.Code(language="diff")``.
    """
    before_lines = before.split("\n")
    after_lines = after.split("\n")
    diff = difflib.unified_diff(
        before_lines,
        after_lines,
        fromfile=before_label,
        tofile=after_label,
        n=context_lines,
        lineterm="",
    )
    return "\n".join(diff)


def ir_diff_stats(before: str, after: str) -> dict:
    """Quick numeric summary of changes between two IR texts."""
    b_lines = before.split("\n")
    a_lines = after.split("\n")
    added = removed = 0
    for tag, _, _, _, _ in difflib.SequenceMatcher(
        None, b_lines, a_lines
    ).get_opcodes():
        if tag == "insert":
            added += 1
        elif tag == "delete":
            removed += 1
        elif tag == "replace":
            added += 1
            removed += 1
    return {
        "lines_before": len(b_lines),
        "lines_after": len(a_lines),
        "sections_added": added,
        "sections_removed": removed,
    }


# ──────────────────────────────────────────────────────────────────────
# Pass delta summary
# ──────────────────────────────────────────────────────────────────────

def format_pass_delta(name: str, delta: dict) -> str:
    """One-line human-readable summary of what a pass changed."""
    parts = [f"**{name}**"]

    fb = delta.get("functions_before", "?")
    fa = delta.get("functions_after", "?")
    if fb != fa:
        parts.append(f"functions {fb} → {fa}")

    tb = delta.get("tir_before", "?")
    ta = delta.get("tir_after", "?")
    if tb != ta:
        parts.append(f"TIR funcs {tb} → {ta}")

    elapsed = delta.get("elapsed_s", 0)
    parts.append(f"({elapsed:.3f}s)")

    return "  —  ".join(parts)


def format_all_pass_deltas(
    pass_order: List[str],
    deltas: Dict[str, dict],
) -> str:
    """Multi-line markdown summary of every pass applied."""
    if not pass_order:
        return "_No passes applied._"
    lines = ["| # | Pass | Functions | TIR Funcs | Time |",
             "|---|------|-----------|-----------|------|"]
    for i, name in enumerate(pass_order, 1):
        d = deltas.get(name, {})
        lines.append(
            f"| {i} | {name} "
            f"| {d.get('functions_before','?')} → {d.get('functions_after','?')} "
            f"| {d.get('tir_before','?')} → {d.get('tir_after','?')} "
            f"| {d.get('elapsed_s',0):.3f}s |"
        )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Operator table (Stage 5)
# ──────────────────────────────────────────────────────────────────────

def operator_table_html(operators: List[dict]) -> str:
    """Render the extracted-operator list as an HTML table with kind summary."""
    if not operators:
        return "<p><em>No TIR operators extracted.</em></p>"

    from collections import Counter
    kind_counts = Counter(op.get("op_kind", "other") for op in operators)
    summary_parts = [
        f'{_kind_badge(k)} <b>{c}</b>' for k, c in kind_counts.most_common()
    ]
    summary_html = (
        '<div style="margin-bottom:10px;padding:8px 12px;background:#F5F5F5;'
        'border-radius:6px;font-size:12px">'
        f'<b>{len(operators)} operators</b> &mdash; '
        + " &nbsp; ".join(summary_parts)
        + '</div>'
    )

    header = (
        "<tr>"
        "<th>#</th><th>Name</th><th>Kind</th>"
        "<th>Params</th><th>Blocks</th><th>IR Lines</th>"
        "</tr>"
    )
    rows = []
    for i, op in enumerate(operators):
        param_summary = _format_param_shapes(op.get("params", []))
        kind = op.get("op_kind", "other")
        kind_badge = _kind_badge(kind)
        rows.append(
            f"<tr>"
            f"<td>{i + 1}</td>"
            f"<td><code>{html.escape(op['name'])}</code></td>"
            f"<td>{kind_badge}</td>"
            f"<td style='font-size:11px'>{html.escape(param_summary)}</td>"
            f"<td>{op.get('num_blocks', '?')}</td>"
            f"<td>{op.get('ir_lines', '?')}</td>"
            f"</tr>"
        )
    return (
        summary_html
        + '<table style="border-collapse:collapse;width:100%;font-size:12px">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


_KIND_COLORS = {
    "conv": "#1976D2",
    "matmul": "#7B1FA2",
    "dense": "#7B1FA2",
    "elemwise": "#388E3C",
    "relu": "#F57C00",
    "batchnorm": "#00796B",
    "pool": "#5D4037",
    "softmax": "#C62828",
    "reshape": "#546E7A",
    "transpose": "#546E7A",
    "layernorm": "#00796B",
}


def _kind_badge(kind: str) -> str:
    color = _KIND_COLORS.get(kind, "#757575")
    return (
        f'<span style="background:{color};color:#fff;'
        f'padding:1px 6px;border-radius:3px;font-size:11px">'
        f'{html.escape(kind)}</span>'
    )


def _format_param_shapes(params: List[dict]) -> str:
    """Compact summary like 'f32[1,64,56,56], f32[64,64,3,3]'."""
    parts = []
    for p in params:
        dtype = p.get("dtype", "?")
        shape = p.get("shape", "?")
        if isinstance(shape, list):
            shape_str = "×".join(str(s) for s in shape)
        else:
            shape_str = str(shape)
        short_dtype = dtype.replace("float", "f").replace("int", "i")
        parts.append(f"{short_dtype}[{shape_str}]")
    return ", ".join(parts) if parts else "—"


# ──────────────────────────────────────────────────────────────────────
# TIR AST display (Stage 6)
# ──────────────────────────────────────────────────────────────────────

def tir_ast_tree_html(ast_summary: dict) -> str:
    """Render the AST summary as an indented, collapsible HTML tree."""
    lines = ['<div style="font-family:monospace;font-size:12px;line-height:1.6">']

    blocks = ast_summary.get("blocks", [])
    loops = ast_summary.get("loops", [])
    buffers = ast_summary.get("buffers", [])

    if buffers:
        lines.append("<b>Buffers</b>")
        lines.append("<ul style='margin:2px 0'>")
        for buf in buffers:
            shape_str = "×".join(str(s) for s in buf.get("shape", []))
            scope = buf.get("scope", "")
            scope_tag = f' <span style="color:#888">scope={scope}</span>' if scope else ""
            lines.append(
                f"<li><code>{html.escape(buf['name'])}</code> "
                f"{buf.get('dtype','?')}[{shape_str}]{scope_tag}</li>"
            )
        lines.append("</ul>")

    if loops:
        lines.append("<b>Iteration Structure</b>")
        lines.append("<ul style='margin:2px 0'>")
        for lp in loops:
            binding = lp.get("thread_binding", "")
            binding_tag = (
                f' <span style="color:#1976D2;font-weight:bold">'
                f'→ {html.escape(binding)}</span>'
                if binding else ""
            )
            source = lp.get("source", "for_loop")
            kind = lp.get("kind", "")
            kind_tag = ""
            if kind == "S":
                kind_tag = ' <span style="color:#1976D2;font-size:10px">[Spatial]</span>'
            elif kind == "R":
                kind_tag = ' <span style="color:#C62828;font-size:10px">[Reduce]</span>'
            prefix = "iter" if source == "block_iter" else "for"
            lines.append(
                f"<li>{prefix} <code>{html.escape(lp['var'])}</code> "
                f"in [0, {lp.get('extent','?')})"
                f"{kind_tag}{binding_tag}</li>"
            )
        lines.append("</ul>")

    if blocks:
        lines.append("<b>Blocks</b>")
        lines.append("<ul style='margin:2px 0'>")
        for blk in blocks:
            iters = ", ".join(blk.get("iter_vars", []))
            lines.append(
                f"<li><code>{html.escape(blk.get('name',''))}</code> "
                f"iters=[{html.escape(iters)}] "
                f"reads={blk.get('num_reads','?')} "
                f"writes={blk.get('num_writes','?')}</li>"
            )
        lines.append("</ul>")

    if not (blocks or loops or buffers):
        lines.append("<em>No AST structure extracted.</em>")

    lines.append("</div>")
    return "\n".join(lines)


def tir_loop_table_html(ast_summary: dict) -> str:
    """Render the loop/iterator table as HTML.

    Handles both explicit ``For`` loops and block iter_vars (extracted as
    pseudo-loops with ``source='block_iter'``).
    """
    loops = ast_summary.get("loops", [])
    if not loops:
        return "<p><em>No iteration structure found.</em></p>"

    header = (
        "<tr><th>#</th><th>Variable</th><th>Extent</th>"
        "<th>Kind</th><th>Thread Binding</th><th>Source</th></tr>"
    )
    rows = []
    for i, lp in enumerate(loops):
        binding = lp.get("thread_binding", "")
        source = lp.get("source", "for_loop")
        kind_raw = lp.get("kind", "")
        kind_display = kind_raw
        if kind_raw == "S":
            kind_display = '<span style="color:#1976D2">Spatial</span>'
        elif kind_raw == "R":
            kind_display = '<span style="color:#C62828">Reduce</span>'
        source_badge = (
            '<span style="background:#E3F2FD;color:#1565C0;padding:1px 5px;'
            'border-radius:3px;font-size:10px">block iter</span>'
            if source == "block_iter" else
            '<span style="background:#E8F5E9;color:#2E7D32;padding:1px 5px;'
            'border-radius:3px;font-size:10px">for loop</span>'
        )
        rows.append(
            f"<tr>"
            f"<td>{i}</td>"
            f"<td><code>{html.escape(lp['var'])}</code></td>"
            f"<td>{lp.get('extent', '?')}</td>"
            f"<td>{kind_display}</td>"
            f"<td><code>{html.escape(binding)}</code></td>"
            f"<td>{source_badge}</td>"
            f"</tr>"
        )
    return (
        '<table style="border-collapse:collapse;width:100%;font-size:12px">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def tir_buffer_table_html(ast_summary: dict) -> str:
    """Render the buffer list as an HTML table."""
    buffers = ast_summary.get("buffers", [])
    if not buffers:
        return "<p><em>No buffer info available.</em></p>"

    header = "<tr><th>Name</th><th>Shape</th><th>Dtype</th><th>Scope</th></tr>"
    rows = []
    for buf in buffers:
        shape_str = "×".join(str(s) for s in buf.get("shape", []))
        rows.append(
            f"<tr>"
            f"<td><code>{html.escape(buf['name'])}</code></td>"
            f"<td>{shape_str}</td>"
            f"<td>{buf.get('dtype', '?')}</td>"
            f"<td>{html.escape(buf.get('scope', ''))}</td>"
            f"</tr>"
        )
    return (
        '<table style="border-collapse:collapse;width:100%;font-size:12px">'
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
