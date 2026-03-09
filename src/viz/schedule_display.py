"""Render MetaSchedule traces and candidate cards as HTML.

Functions
---------
trace_to_readable    -- parse trace text into human-readable instruction list
trace_to_card_html   -- render one candidate as a styled HTML card
candidate_cards_html -- render all candidates as a scrollable card grid
best_candidate_banner_html -- highlight banner for the selected winner
"""

from __future__ import annotations

import html
import re
from typing import List, Optional


# ──────────────────────────────────────────────────────────────────────
# Trace parsing
# ──────────────────────────────────────────────────────────────────────

_INSTRUCTION_PATTERN = re.compile(
    r'(split|reorder|fuse|bind|vectorize|unroll|parallel|'
    r'cache_read|cache_write|storage_align|set_scope|'
    r'compute_at|compute_inline|reverse_compute_at|'
    r'tile|annotate|decompose_reduction|pad_einsum|'
    r'rfactor|sample_perfect_tile|sample_categorical)\b'
    r'[^)]*\)?',
    re.IGNORECASE,
)

_CATEGORY_COLORS = {
    "split": "#1976D2",
    "tile": "#1976D2",
    "reorder": "#7B1FA2",
    "fuse": "#7B1FA2",
    "bind": "#E65100",
    "vectorize": "#2E7D32",
    "unroll": "#2E7D32",
    "parallel": "#2E7D32",
    "cache_read": "#00838F",
    "cache_write": "#00838F",
    "set_scope": "#00838F",
    "compute_at": "#5D4037",
    "compute_inline": "#5D4037",
    "reverse_compute_at": "#5D4037",
    "reverse_compute_inline": "#5D4037",
    "sample_perfect_tile": "#455A64",
    "sample_categorical": "#455A64",
    "annotate": "#455A64",
    # Real MetaSchedule instructions (sch.xxx format)
    "sch.split": "#1976D2",
    "sch.fuse": "#7B1FA2",
    "sch.reorder": "#7B1FA2",
    "sch.bind": "#E65100",
    "sch.vectorize": "#2E7D32",
    "sch.unroll": "#2E7D32",
    "sch.parallel": "#2E7D32",
    "sch.cache_read": "#00838F",
    "sch.cache_write": "#00838F",
    "sch.compute_inline": "#5D4037",
    "sch.reverse_compute_inline": "#5D4037",
    "sch.get_sblock": "#37474F",
    "sch.get_loops": "#37474F",
    "sch.sample_perfect_tile": "#455A64",
    "sch.sample_categorical": "#455A64",
    "sch.annotate": "#455A64",
}


_REAL_SCHEDULE_CALL = re.compile(
    r'sch\.(get_sblock|get_loops|split|fuse|reorder|bind|vectorize|'
    r'unroll|parallel|compute_inline|reverse_compute_inline|'
    r'cache_read|cache_write|set_scope|storage_align|'
    r'sample_perfect_tile|sample_categorical|'
    r'annotate|decompose_reduction|pad_einsum|rfactor)\b[^)]*\)',
    re.IGNORECASE,
)


def trace_to_readable(trace_text: str) -> List[str]:
    """Parse a MetaSchedule trace string into a list of instruction strings.

    Handles:
    - Real MetaSchedule trace (Python code with sch.split(...) etc.)
    - JSON array of instruction strings
    - Arrow-separated compact format (synthetic records)
    """
    if not trace_text:
        return ["(empty trace)"]

    # JSON array format: '["sch.split(...)", "sch.fuse(...)"]'
    if trace_text.startswith("["):
        try:
            import json
            items = json.loads(trace_text)
            if isinstance(items, list) and items:
                return [str(i).strip().strip('"') for i in items if str(i).strip()]
        except Exception:
            pass

    if " -> " in trace_text:
        return [s.strip() for s in trace_text.split(" -> ") if s.strip()]

    # Real MetaSchedule traces: extract sch.xxx(...) calls
    real_matches = _REAL_SCHEDULE_CALL.findall(trace_text)
    if real_matches:
        full_matches = _REAL_SCHEDULE_CALL.finditer(trace_text)
        return [m.group().strip() for m in full_matches]

    matches = _INSTRUCTION_PATTERN.findall(trace_text)
    if matches:
        return [m.strip().rstrip(",") for m in matches]

    lines = [l.strip() for l in trace_text.split("\n") if l.strip()]
    return lines if lines else [trace_text[:200]]


def _instruction_badge(instr: str) -> str:
    """Wrap an instruction in a coloured badge."""
    keyword = instr.split("(")[0].strip().lower()
    # Try full key first (e.g., "sch.split"), then just the method name
    color = _CATEGORY_COLORS.get(keyword)
    if color is None:
        short = keyword.split(".")[-1] if "." in keyword else keyword
        color = _CATEGORY_COLORS.get(short, "#757575")
    return (
        f'<span style="display:inline-block;background:{color};color:#fff;'
        f'padding:2px 8px;border-radius:4px;font-size:11px;'
        f'font-family:monospace;margin:2px 2px">'
        f'{html.escape(instr)}</span>'
    )


# ──────────────────────────────────────────────────────────────────────
# Candidate card
# ──────────────────────────────────────────────────────────────────────

def trace_to_card_html(
    record: dict,
    rank: int = 0,
) -> str:
    """Render a single tuning candidate as a styled HTML card.

    Parameters
    ----------
    record : dict with keys candidate_id, task_name, trace_text, run_ms, is_best
    rank : display rank (1-based)
    """
    is_best = record.get("is_best", False)
    border_color = "#4CAF50" if is_best else "#E0E0E0"
    border_width = "3px" if is_best else "1px"
    bg = "#E8F5E9" if is_best else "#FAFAFA"

    badge = ""
    if is_best:
        badge = (
            '<span style="background:#4CAF50;color:#fff;padding:2px 10px;'
            'border-radius:4px;font-size:11px;font-weight:bold;float:right">'
            'BEST</span>'
        )

    synthetic_tag = ""
    if record.get("_synthetic"):
        synthetic_tag = (
            ' <span style="background:#FF9800;color:#fff;padding:1px 6px;'
            'border-radius:3px;font-size:10px">synthetic</span>'
        )

    instructions = trace_to_readable(record.get("trace_text", ""))
    instr_html = " ".join(_instruction_badge(i) for i in instructions)

    run_ms = record.get("run_ms", float("inf"))
    latency_str = f"{run_ms:.4f} ms" if run_ms < 1e6 else "timeout"

    return (
        f'<div style="border:{border_width} solid {border_color};'
        f'border-radius:8px;padding:12px;margin:8px 0;background:{bg}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<span style="font-weight:bold;font-size:13px">'
        f'#{html.escape(str(record.get("candidate_id", rank)))} '
        f'<code>{html.escape(record.get("task_name", ""))}</code>'
        f'{synthetic_tag}</span>'
        f'{badge}'
        f'</div>'
        f'<div style="margin:6px 0;font-size:12px;color:#333">'
        f'Latency: <b>{latency_str}</b>'
        f'</div>'
        f'<div style="margin:4px 0">{instr_html}</div>'
        f'</div>'
    )


def candidate_cards_html(
    records: List[dict],
    max_display: int = 20,
) -> str:
    """Render all candidate cards as a scrollable container."""
    if not records:
        return "<p><em>No tuning records available.</em></p>"

    sorted_recs = sorted(records, key=lambda r: r.get("run_ms", float("inf")))
    display = sorted_recs[:max_display]

    cards = [trace_to_card_html(rec, rank=i + 1) for i, rec in enumerate(display)]

    header = (
        f'<div style="font-size:13px;color:#666;margin-bottom:8px">'
        f'Showing {len(display)} of {len(records)} candidates '
        f'(sorted by latency)</div>'
    )

    overflow = ""
    if len(records) > max_display:
        overflow = (
            f'<div style="color:#999;font-size:12px;text-align:center;'
            f'margin-top:8px">... and {len(records) - max_display} more</div>'
        )

    return (
        f'{header}'
        f'<div style="max-height:600px;overflow-y:auto">'
        f'{"".join(cards)}'
        f'{overflow}'
        f'</div>'
    )


# ──────────────────────────────────────────────────────────────────────
# Best candidate banner
# ──────────────────────────────────────────────────────────────────────

def best_candidate_banner_html(best: Optional[dict]) -> str:
    """Large highlight banner for the winning candidate."""
    if best is None:
        return "<p><em>No best candidate selected.</em></p>"

    run_ms = best.get("run_ms", 0)
    instructions = trace_to_readable(best.get("trace_text", ""))

    return (
        '<div style="background:linear-gradient(135deg,#1B5E20,#4CAF50);'
        'color:#fff;border-radius:12px;padding:20px;margin:12px 0">'
        '<div style="font-size:18px;font-weight:bold;margin-bottom:8px">'
        f'Selected Candidate #{html.escape(str(best.get("candidate_id", "?")))}</div>'
        f'<div style="font-size:14px;margin-bottom:6px">'
        f'Task: <code style="color:#E8F5E9">{html.escape(best.get("task_name", ""))}</code>'
        f'</div>'
        f'<div style="font-size:22px;font-weight:bold;margin:8px 0">'
        f'{run_ms:.4f} ms</div>'
        f'<div style="font-size:12px;opacity:0.9;margin-top:8px">'
        f'Schedule: {html.escape(" -> ".join(instructions))}</div>'
        '</div>'
    )
