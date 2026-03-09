"""Per-candidate structural feature tables for cost-model education.

Functions
---------
build_feature_dataframe   -- list[dict] -> pd.DataFrame
feature_table_html        -- DataFrame -> styled HTML table
tir_features_table_html   -- single-operator TIR features -> HTML card
cost_model_explanation_html -- educational text card about the cost model
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

try:
    import pandas as pd  # type: ignore
    _PD_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore
    _PD_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
# DataFrame builder
# ──────────────────────────────────────────────────────────────────────

_FEATURE_COLUMNS = [
    "candidate_id",
    "task_name",
    "run_ms",
    "num_splits",
    "num_reorders",
    "num_thread_bindings",
    "has_cache_read",
    "has_cache_write",
    "has_shared_memory",
    "has_vectorize",
    "has_unroll",
    "trace_length",
    "is_best",
]


def build_feature_dataframe(features: List[dict]) -> Any:
    """Convert the per-candidate feature dicts into a pandas DataFrame.

    Returns a plain list-of-dicts if pandas is unavailable.
    """
    if not features:
        return pd.DataFrame() if _PD_AVAILABLE else []

    if not _PD_AVAILABLE:
        return features

    rows = []
    for f in features:
        row = {col: f.get(col, None) for col in _FEATURE_COLUMNS}
        rows.append(row)

    df = pd.DataFrame(rows)
    bool_cols = ["has_cache_read", "has_cache_write", "has_shared_memory",
                 "has_vectorize", "has_unroll", "is_best"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    return df.sort_values("run_ms", ascending=True).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# Feature table -> HTML
# ──────────────────────────────────────────────────────────────────────

_BOOL_YES = '<span style="color:#4CAF50;font-weight:bold">Yes</span>'
_BOOL_NO = '<span style="color:#BDBDBD">--</span>'

_COLUMN_LABELS = {
    "candidate_id": "#",
    "task_name": "Task",
    "run_ms": "Latency (ms)",
    "num_splits": "Splits",
    "num_reorders": "Reorders",
    "num_thread_bindings": "Thread Binds",
    "has_cache_read": "Cache Read",
    "has_cache_write": "Cache Write",
    "has_shared_memory": "Shared Mem",
    "has_vectorize": "Vectorize",
    "has_unroll": "Unroll",
    "trace_length": "Trace Len",
    "is_best": "Best?",
}


def feature_table_html(features: Any, max_rows: int = 30) -> str:
    """Render the feature DataFrame (or list-of-dicts) as an HTML table."""
    rows_data: List[dict] = []
    if _PD_AVAILABLE and hasattr(features, "to_dict"):
        rows_data = features.head(max_rows).to_dict("records")
    elif isinstance(features, list):
        rows_data = features[:max_rows]
    else:
        return "<p><em>No feature data available.</em></p>"

    if not rows_data:
        return "<p><em>No candidates to display.</em></p>"

    columns = [c for c in _FEATURE_COLUMNS if c in rows_data[0]]
    header = "".join(f"<th>{_COLUMN_LABELS.get(c, c)}</th>" for c in columns)

    body_rows = []
    for row in rows_data:
        is_best = row.get("is_best", False)
        bg = "background:#E8F5E9;" if is_best else ""
        cells = []
        for c in columns:
            val = row.get(c)
            if isinstance(val, bool):
                cells.append(f"<td>{_BOOL_YES if val else _BOOL_NO}</td>")
            elif isinstance(val, float):
                cells.append(f"<td>{val:.4f}</td>")
            elif c == "task_name":
                cells.append(f"<td><code>{html.escape(str(val or ''))}</code></td>")
            else:
                cells.append(f"<td>{html.escape(str(val if val is not None else ''))}</td>")
        body_rows.append(f'<tr style="{bg}">{"".join(cells)}</tr>')

    total_note = ""
    total = len(features) if hasattr(features, "__len__") else "?"
    if isinstance(total, int) and total > max_rows:
        total_note = (
            f'<div style="color:#999;font-size:11px;margin-top:4px">'
            f'Showing {max_rows} of {total} candidates</div>'
        )

    return (
        '<div style="overflow-x:auto">'
        '<table style="border-collapse:collapse;width:100%;font-size:12px">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        f'</table>{total_note}</div>'
    )


# ──────────────────────────────────────────────────────────────────────
# Single-operator TIR features card
# ──────────────────────────────────────────────────────────────────────

def tir_features_table_html(features: Dict[str, Any]) -> str:
    """Render TIR structural features (from compute_tir_structural_features)."""
    if not features:
        return "<p><em>No TIR features available.</em></p>"

    display_labels = {
        "op_name": "Operator",
        "num_loops": "Loop Count",
        "num_thread_bindings": "Thread Bindings",
        "num_shared_buffers": "Shared-Memory Buffers",
        "num_vectorized_loops": "Vectorized Loops",
        "num_unrolled_loops": "Unrolled Loops",
        "num_blocks": "TIR Blocks",
        "num_buffer_stores": "Buffer Stores",
        "total_loop_extent_product": "Loop Extent Product",
        "arithmetic_intensity_proxy": "Arith. Intensity (proxy)",
    }

    rows = []
    for key, label in display_labels.items():
        val = features.get(key)
        if val is None:
            continue
        if isinstance(val, float):
            val_str = f"{val:,.2f}"
        elif isinstance(val, int) and val > 9999:
            val_str = f"{val:,}"
        else:
            val_str = str(val)
        rows.append(
            f"<tr><td style='font-weight:bold;padding-right:16px'>{label}</td>"
            f"<td><code>{html.escape(val_str)}</code></td></tr>"
        )

    return (
        '<table style="border-collapse:collapse;font-size:12px;'
        'margin:8px 0">'
        f'{"".join(rows)}</table>'
    )


# ──────────────────────────────────────────────────────────────────────
# Cost model educational card
# ──────────────────────────────────────────────────────────────────────

def cost_model_explanation_html() -> str:
    """Render the educational explanation card for the cost model."""
    return (
        '<div style="background:#FFF8E1;border-left:4px solid #FFC107;'
        'border-radius:4px;padding:16px;margin:12px 0;font-size:13px;'
        'line-height:1.6">'
        '<div style="font-weight:bold;font-size:14px;margin-bottom:8px">'
        'About the Cost Model (Paper Section 5.2, Figure 13)</div>'
        '<p>TVM uses a <b>gradient tree boosting</b> (XGBoost) cost model '
        'trained on features extracted from the loop AST of each candidate '
        'schedule.  The model predicts execution time <em>without</em> '
        'running the candidate on hardware, allowing the search to prune '
        'the schedule space efficiently.</p>'
        '<p>The features shown above are <b>educational approximations</b> '
        'of the kind of information the cost model considers:</p>'
        '<ul style="margin:4px 0 4px 20px">'
        '<li><b>Splits / Reorders</b> -- how the loop nest is tiled and '
        'reordered for locality</li>'
        '<li><b>Thread Bindings</b> -- how loops map to GPU thread/block '
        'dimensions</li>'
        '<li><b>Cache Read/Write</b> -- whether shared memory staging is used '
        '(cooperative fetching, paper Section 4.2)</li>'
        '<li><b>Vectorize / Unroll</b> -- instruction-level parallelism '
        'optimizations</li>'
        '</ul>'
        '<p style="color:#795548;margin-top:8px"><em>Note: The exact internal '
        'cost-model features and predictor math are not fully surfaced in '
        'the public high-level APIs.  The structural features shown are an '
        'educational approximation.</em></p>'
        '</div>'
    )
