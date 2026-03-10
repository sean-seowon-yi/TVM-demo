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


_SETUP_OPS = frozenset({
    "get_sblock", "get_block", "get_loops", "get_child_blocks",
    "get_consumers", "get_producers", "sample_perfect_tile",
    "sample_categorical", "annotate", "unannotate",
    "enter_postproc",
})


def trace_to_readable(trace_text: str, concise: bool = False) -> List[str]:
    """Parse a MetaSchedule trace string into a list of instruction strings.

    Parameters
    ----------
    concise : if True, filter out setup instructions (get_sblock, get_loops,
              sample_*, annotate) and keep only actual scheduling decisions.
    """
    if not trace_text:
        return ["(empty trace)"]

    if trace_text.startswith("["):
        try:
            import json
            items = json.loads(trace_text)
            if isinstance(items, list) and items:
                raw = [str(i).strip().strip('"') for i in items if str(i).strip()]
                return _filter_setup(raw) if concise else raw
        except Exception:
            pass

    if " -> " in trace_text:
        raw = [s.strip() for s in trace_text.split(" -> ") if s.strip()]
        return _filter_setup(raw) if concise else raw

    real_matches = _REAL_SCHEDULE_CALL.findall(trace_text)
    if real_matches:
        full_matches = _REAL_SCHEDULE_CALL.finditer(trace_text)
        raw = [m.group().strip() for m in full_matches]
        return _filter_setup(raw) if concise else raw

    matches = _INSTRUCTION_PATTERN.findall(trace_text)
    if matches:
        raw = [m.strip().rstrip(",") for m in matches]
        return _filter_setup(raw) if concise else raw

    lines = [l.strip() for l in trace_text.split("\n") if l.strip()]
    raw = lines if lines else [trace_text[:200]]
    return _filter_setup(raw) if concise else raw


def _filter_setup(instructions: List[str]) -> List[str]:
    """Remove setup/bookkeeping instructions, keeping only scheduling decisions."""
    result = []
    for inst in instructions:
        op = inst.split("(")[0].strip().lower()
        op = op.split(".")[-1] if "." in op else op
        if op not in _SETUP_OPS:
            result.append(inst)
    return result if result else ["(default schedule)"]


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
    tuned_task_names: Optional[List[str]] = None,
) -> str:
    """Render a single tuning candidate as a styled HTML card.

    Parameters
    ----------
    record : dict with keys candidate_id, task_name, trace_text, run_ms, is_best
    rank : display rank (1-based)
    tuned_task_names : (unused, kept for backwards compat)
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

    task_name = record.get("task_name", "")
    task_label = ""
    if task_name and task_name != "main" and not task_name.startswith("task_"):
        task_label = (
            f' <span style="background:#1976D2;color:#fff;padding:1px 6px;'
            f'border-radius:3px;font-size:10px">'
            f'{html.escape(task_name)}</span>'
        )

    instructions = trace_to_readable(record.get("trace_text", ""), concise=True)
    instr_html = " ".join(_instruction_badge(i) for i in instructions[:12])
    if len(instructions) > 12:
        instr_html += (
            f' <span style="color:#888;font-size:10px;font-style:italic">'
            f'... +{len(instructions) - 12} more</span>'
        )

    run_ms = record.get("run_ms", float("inf"))
    latency_str = f"{run_ms:.4f} ms" if run_ms < 1e6 else "timeout"

    return (
        f'<div style="border:{border_width} solid {border_color};'
        f'border-radius:8px;padding:12px;margin:8px 0;background:{bg}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center">'
        f'<span style="font-weight:bold;font-size:13px">'
        f'#{html.escape(str(record.get("candidate_id", rank)))}'
        f'{synthetic_tag}{task_label}</span>'
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
    tuned_task_names: Optional[List[str]] = None,
) -> str:
    """Render candidates grouped by task, best + runner-ups per group."""
    if not records:
        return "<p><em>No tuning records available.</em></p>"

    groups: dict = {}
    for r in records:
        tn = r.get("task_name", "unknown")
        groups.setdefault(tn, []).append(r)

    for recs in groups.values():
        recs.sort(key=lambda r: r.get("run_ms", float("inf")))
        if recs:
            for r in recs:
                r["_is_task_best"] = False
            recs[0]["_is_task_best"] = True

    MAX_PER_TASK = 3
    sections: List[str] = []
    for tn in sorted(groups, key=lambda t: groups[t][0].get("run_ms", float("inf"))):
        recs = groups[tn]
        best_ms = recs[0].get("run_ms", float("inf"))
        finite_recs = [r for r in recs if r.get("run_ms", float("inf")) < 1e6]
        worst_finite = finite_recs[-1].get("run_ms", best_ms) if len(finite_recs) > 1 else best_ms
        n_failed = len(recs) - len(finite_recs)

        sec_header = (
            f'<div style="margin:16px 0 6px;padding:8px 12px;'
            f'background:#E3F2FD;border-radius:6px;border-left:4px solid #1976D2">'
            f'<span style="font-weight:bold;font-size:14px">{html.escape(tn)}</span>'
            f' &mdash; {len(recs)} candidate{"s" if len(recs) != 1 else ""}'
            f' &nbsp; best <b>{best_ms:.4f} ms</b>' if best_ms < 1e6 else
            f'<div style="margin:16px 0 6px;padding:8px 12px;'
            f'background:#E3F2FD;border-radius:6px;border-left:4px solid #1976D2">'
            f'<span style="font-weight:bold;font-size:14px">{html.escape(tn)}</span>'
            f' &mdash; {len(recs)} candidate{"s" if len(recs) != 1 else ""}'
            f' &nbsp; best <b>timeout</b>'
        )
        if len(finite_recs) > 1 and best_ms > 0 and best_ms < 1e6:
            spread = worst_finite / best_ms
            sec_header += f' &nbsp; spread {spread:.1f}x'
        if n_failed > 0:
            sec_header += f' &nbsp; <span style="color:#E65100">{n_failed} timed out</span>'
        sec_header += '</div>'
        sections.append(sec_header)

        shown = recs[:MAX_PER_TASK]
        for i, rec in enumerate(shown):
            rec_copy = dict(rec)
            if i == 0:
                rec_copy["is_best"] = True
            sections.append(trace_to_card_html(rec_copy, rank=i + 1))

        if len(recs) > MAX_PER_TASK:
            sections.append(
                f'<div style="color:#888;font-size:11px;padding:4px 12px;font-style:italic">'
                f'... and {len(recs) - MAX_PER_TASK} more candidates</div>'
            )

    n_tasks = len(groups)
    header = (
        f'<div style="font-size:13px;color:#666;margin-bottom:8px">'
        f'{len(records)} candidates across {n_tasks} task{"s" if n_tasks != 1 else ""} '
        f'(grouped by task, best first)</div>'
    )

    return (
        f'{header}'
        f'<div style="max-height:700px;overflow-y:auto">'
        f'{"".join(sections)}'
        f'</div>'
    )


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

_SCHEDULE_OP_RE = re.compile(
    r'(?:sch\.)?(split|fuse|reorder|bind|vectorize|unroll|parallel|'
    r'cache_read|cache_write|set_scope|compute_at|compute_inline|'
    r'reverse_compute_at|annotate|sample_categorical|sample_perfect_tile)',
    re.IGNORECASE,
)


def _concise_ops_summary(trace_text: str) -> str:
    """Summarise a schedule trace as a compact list of op types with counts."""
    if not trace_text:
        return "(default)"
    ops = _SCHEDULE_OP_RE.findall(trace_text)
    if not ops:
        return "(default)"
    from collections import Counter
    counts = Counter(op.lower() for op in ops if op.lower() not in _SETUP_OPS)
    if not counts:
        return "(default)"
    parts = []
    for op, cnt in counts.most_common():
        parts.append(f"{op}\u00d7{cnt}" if cnt > 1 else op)
    return ", ".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Per-task summary (replaces global-best banner)
# ──────────────────────────────────────────────────────────────────────

def per_task_summary_html(records: List[dict], total_tasks: int = 0) -> str:
    """Summary showing best schedule per task — the actual chosen schedules."""
    if not records:
        return "<p><em>No tuning records available.</em></p>"

    groups: dict = {}
    for r in records:
        tn = r.get("task_name", "unknown")
        groups.setdefault(tn, []).append(r)

    rows: List[str] = []
    for tn in sorted(groups, key=lambda t: min(r.get("run_ms", 1e9) for r in groups[t])):
        recs = sorted(groups[tn], key=lambda r: r.get("run_ms", float("inf")))
        best = recs[0]
        best_ms = best.get("run_ms", float("inf"))
        sched_summary = _concise_ops_summary(best.get("trace_text", ""))

        rows.append(
            f'<tr>'
            f'<td style="font-weight:bold;white-space:nowrap"><code>{html.escape(tn)}</code></td>'
            f'<td style="text-align:right"><b>{best_ms:.4f}</b> ms</td>'
            f'<td style="text-align:center">{len(recs)}</td>'
            f'<td style="font-size:11px;color:#555">{html.escape(sched_summary)}</td>'
            f'</tr>'
        )

    n_tuned = len(groups)
    n_untuned = max(0, total_tasks - n_tuned) if total_tasks else 0

    untuned_note = ""
    if n_untuned > 0:
        untuned_note = (
            f'<div style="margin-top:8px;font-size:12px;color:#888;font-style:italic">'
            f'{n_untuned} remaining tasks use DLight default schedules (not tuned).</div>'
        )

    return (
        '<div style="background:#F5F5F5;border-radius:8px;padding:14px;margin:8px 0">'
        '<div style="font-weight:bold;font-size:14px;margin-bottom:8px">'
        f'Best Schedule Per Task ({n_tuned} tuned)</div>'
        '<table style="border-collapse:collapse;width:100%;font-size:12px">'
        '<thead><tr>'
        '<th style="text-align:left">Task</th>'
        '<th style="text-align:right">Best Latency</th>'
        '<th style="text-align:center">Candidates</th>'
        '<th style="text-align:left">Key Schedule Ops</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
        f'{untuned_note}'
        '</div>'
    )
