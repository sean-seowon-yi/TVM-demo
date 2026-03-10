"""Latency comparison and convergence charts using matplotlib.

Functions
---------
latency_comparison_chart -- PyTorch vs TVM grouped bar chart
convergence_chart        -- trial index vs best latency (paper Figure 12)
task_weight_pie_chart    -- task weight distribution
"""

from __future__ import annotations

import io
import base64
import logging
from typing import Any, List, Optional, Tuple

log = logging.getLogger(__name__)

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.ticker as ticker  # type: ignore
    _MPL_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore
    ticker = None  # type: ignore
    _MPL_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
# Helper: figure -> embeddable content
# ──────────────────────────────────────────────────────────────────────

def _fig_to_base64_img(fig: Any, dpi: int = 120) -> str:
    """Convert a matplotlib Figure to an HTML <img> tag with base64 data."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'


def _fig_to_pil(fig: Any, dpi: int = 120) -> Any:
    """Convert a matplotlib Figure to a PIL Image (for Gradio gr.Image)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image
    return Image.open(buf)


def _check_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise RuntimeError("matplotlib is not installed")


# ──────────────────────────────────────────────────────────────────────
# Latency comparison (PyTorch vs TVM bar chart)
# ──────────────────────────────────────────────────────────────────────

def latency_comparison_chart(
    pytorch_ms: float,
    tvm_ms: float,
    title: str = "Inference Latency Comparison",
    return_format: str = "html",
) -> Any:
    """Grouped bar chart comparing PyTorch vs TVM latency.

    Paper mapping: Section 6, Figure 14.

    Parameters
    ----------
    pytorch_ms : PyTorch median latency in milliseconds
    tvm_ms : TVM median latency in milliseconds
    title : chart title
    return_format : "html" for base64 <img>, "pil" for PIL Image, "fig" for raw Figure

    Returns
    -------
    HTML string, PIL Image, or matplotlib Figure depending on return_format.
    """
    _check_mpl()

    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#FAFAFA")
    bars = ax.bar(
        ["PyTorch\n(eager)", "TVM\n(optimized)"],
        [pytorch_ms, tvm_ms],
        color=["#1976D2", "#4CAF50"],
        width=0.5,
        edgecolor="white",
        linewidth=1.5,
    )

    for bar, val in zip(bars, [pytorch_ms, tvm_ms]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(pytorch_ms, tvm_ms) * 0.02,
            f"{val:.2f} ms",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    speedup = pytorch_ms / tvm_ms if tvm_ms > 0 else float("inf")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_ylim(0, max(pytorch_ms, tvm_ms) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate(
        f"{speedup:.2f}x speedup",
        xy=(1, tvm_ms), xytext=(1.35, (pytorch_ms + tvm_ms) / 2),
        fontsize=10, fontweight="bold", color="#E65100",
        arrowprops=dict(arrowstyle="->", color="#E65100", lw=1.5),
        ha="left",
    )

    fig.tight_layout()

    if return_format == "pil":
        return _fig_to_pil(fig)
    elif return_format == "fig":
        return fig
    return _fig_to_base64_img(fig)


# ──────────────────────────────────────────────────────────────────────
# Convergence chart (trial index vs best latency)
# ──────────────────────────────────────────────────────────────────────

def convergence_chart(
    convergence_data: List[dict],
    title: str = "Tuning Convergence",
    return_format: str = "html",
) -> Any:
    """Line chart of best latency vs trial index.

    Paper mapping: Section 5.3, mirrors Figure 12.

    Parameters
    ----------
    convergence_data : list of {"trial_index": int, "best_latency_ms": float}
    """
    _check_mpl()

    if not convergence_data:
        fig, ax = plt.subplots(figsize=(6, 3), facecolor="#FAFAFA")
        ax.text(0.5, 0.5, "No convergence data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#999")
        ax.set_axis_off()
        if return_format == "pil":
            return _fig_to_pil(fig)
        elif return_format == "fig":
            return fig
        return _fig_to_base64_img(fig)

    trials = [d["trial_index"] for d in convergence_data]
    best_ms = [d["best_latency_ms"] for d in convergence_data]

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#FAFAFA")

    ax.fill_between(trials, best_ms, alpha=0.15, color="#1976D2")
    ax.plot(trials, best_ms, color="#1976D2", linewidth=2, marker="o",
            markersize=3, markerfacecolor="#1976D2")

    ax.set_xlabel("Trial Index", fontsize=10)
    ax.set_ylabel("Best Latency (ms)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if best_ms:
        final_best = best_ms[-1]
        ax.axhline(y=final_best, color="#4CAF50", linestyle="--",
                    linewidth=1, alpha=0.7)
        ax.annotate(
            f"Best: {final_best:.4f} ms",
            xy=(trials[-1], final_best),
            xytext=(trials[-1] * 0.7, final_best * 1.15),
            fontsize=9, color="#4CAF50", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1),
        )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()

    if return_format == "pil":
        return _fig_to_pil(fig)
    elif return_format == "fig":
        return fig
    return _fig_to_base64_img(fig)


# ──────────────────────────────────────────────────────────────────────
# Task weight pie chart
# ──────────────────────────────────────────────────────────────────────

def task_weight_pie_chart(
    task_dicts: List[dict],
    max_slices: int = 8,
    title: str = "Task Weight Distribution",
    return_format: str = "html",
) -> Any:
    """Pie chart showing relative weight of each tuning task.

    Parameters
    ----------
    task_dicts : list of dicts with "name" and "weight" keys
    """
    _check_mpl()

    if not task_dicts:
        fig, ax = plt.subplots(figsize=(4, 4), facecolor="#FAFAFA")
        ax.text(0.5, 0.5, "No tasks", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#999")
        ax.set_axis_off()
        if return_format == "pil":
            return _fig_to_pil(fig)
        elif return_format == "fig":
            return fig
        return _fig_to_base64_img(fig)

    sorted_tasks = sorted(task_dicts, key=lambda t: t.get("weight", 0), reverse=True)

    if len(sorted_tasks) > max_slices:
        top = sorted_tasks[:max_slices - 1]
        other_weight = sum(t.get("weight", 0) for t in sorted_tasks[max_slices - 1:])
        names = [_short_name(t["name"]) for t in top] + ["other"]
        weights = [t.get("weight", 1) for t in top] + [other_weight]
    else:
        names = [_short_name(t["name"]) for t in sorted_tasks]
        weights = [t.get("weight", 1) for t in sorted_tasks]

    colors = plt.cm.Set3([i / max(len(names), 1) for i in range(len(names))])

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#FAFAFA")
    wedges, texts, autotexts = ax.pie(
        weights, labels=names, autopct="%1.0f%%",
        colors=colors, startangle=90, textprops={"fontsize": 9},
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()

    if return_format == "pil":
        return _fig_to_pil(fig)
    elif return_format == "fig":
        return fig
    return _fig_to_base64_img(fig)


def three_bar_latency_chart(
    pytorch_ms: float,
    tvm_live_ms: float,
    tvm_precomputed_ms: float,
    live_trials: int = 8,
    precomputed_trials: int = 512,
    title: str = "Inference Latency: PyTorch vs TVM",
    return_format: str = "html",
) -> Any:
    """Three-bar chart showing the tuning progression story.

    Bar 1 (blue):  PyTorch baseline -- cuDNN, no compilation needed
    Bar 2 (amber): TVM live demo   -- few trials, mostly default schedules
    Bar 3 (green): TVM precomputed -- many trials, all operators optimized
    """
    _check_mpl()

    labels = [
        f"PyTorch\nBaseline",
        f"TVM Live\n({live_trials} trials)",
        f"TVM Tuned\n({precomputed_trials} trials)",
    ]
    values = [pytorch_ms, tvm_live_ms, tvm_precomputed_ms]
    colors = ["#546E7A", "#FF8F00", "#2E7D32"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), facecolor="#FAFAFA")
    bars = ax.bar(labels, values, color=colors, width=0.52,
                  edgecolor="white", linewidth=1.5, zorder=3)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    ymax = max(values)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.02,
            f"{val:.2f} ms",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    live_speedup = pytorch_ms / tvm_live_ms if tvm_live_ms > 0 else 0
    pre_speedup = pytorch_ms / tvm_precomputed_ms if tvm_precomputed_ms > 0 else 0

    if live_speedup < 1:
        live_label = f"{live_speedup:.2f}x\n(slower)"
        live_color = "#D84315"
    else:
        live_label = f"{live_speedup:.2f}x"
        live_color = "#E65100"

    pre_label = f"{pre_speedup:.2f}x faster"
    pre_color = "#1B5E20"

    ax.annotate(
        live_label,
        xy=(1, tvm_live_ms + ymax * 0.01),
        xytext=(1.45, tvm_live_ms + ymax * 0.18),
        fontsize=9, fontweight="bold", color=live_color, ha="center",
        arrowprops=dict(arrowstyle="->", color=live_color, lw=1.2),
    )
    ax.annotate(
        pre_label,
        xy=(2, tvm_precomputed_ms + ymax * 0.01),
        xytext=(1.55, tvm_precomputed_ms + ymax * 0.22),
        fontsize=10, fontweight="bold", color=pre_color, ha="center",
        arrowprops=dict(arrowstyle="->", color=pre_color, lw=1.5),
    )

    # Dashed line at PyTorch level for easy visual comparison
    ax.axhline(y=pytorch_ms, color="#546E7A", linestyle=":", linewidth=1, alpha=0.5, zorder=2)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
    ax.set_ylabel("Latency (ms)", fontsize=10)
    ax.set_ylim(0, ymax * 1.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if return_format == "pil":
        return _fig_to_pil(fig)
    elif return_format == "fig":
        return fig
    return _fig_to_base64_img(fig)


def _short_name(name: str, max_len: int = 20) -> str:
    """Shorten a task name for chart labels."""
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."


# ──────────────────────────────────────────────────────────────────────
# Per-task tuning summary (horizontal bar chart)
# ──────────────────────────────────────────────────────────────────────

def per_task_summary_chart(
    total_tasks: int,
    tuned_task_names: List[str],
    records: Optional[List[dict]] = None,
    title: str = "Per-Task Tuning Summary",
    return_format: str = "html",
) -> Any:
    """Horizontal bar chart showing tuned vs untuned tasks.

    Parameters
    ----------
    total_tasks : total number of extracted tuning tasks (e.g. 28)
    tuned_task_names : real task names from the MetaSchedule DB
    records : tuning records (used to count candidates per task, optional)
    """
    _check_mpl()
    import numpy as np

    n_records = len(records) if records else 0

    if not tuned_task_names and total_tasks == 0:
        fig, ax = plt.subplots(figsize=(6, 2), facecolor="#FAFAFA")
        ax.text(0.5, 0.5, "No tuning data", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#999")
        ax.set_axis_off()
        return _fig_to_base64_img(fig)

    n_tuned = len(tuned_task_names)

    per_task_count: dict = {}
    if records:
        for r in records:
            tn = r.get("task_name", "")
            if tn and tn != "main" and not tn.startswith("task_"):
                per_task_count[tn] = per_task_count.get(tn, 0) + 1

    names = []
    latencies = []
    colors = []
    annotations = []

    for task_name in tuned_task_names:
        short = _short_name(task_name, 25)
        names.append(short)
        latencies.append(1.0)
        colors.append("#43A047")
        count = per_task_count.get(task_name, 0)
        if count > 0:
            annotations.append(f"tuned ({count} candidates)")
        else:
            n_avg = max(1, n_records // n_tuned) if n_tuned > 0 else 0
            annotations.append(f"tuned (~{n_avg} candidates)")

    n_untuned = max(0, total_tasks - n_tuned)

    if n_untuned > 0:
        names.append(f"{n_untuned} untuned tasks")
        latencies.append(1.0)
        colors.append("#E0E0E0")
        annotations.append("using DLight default schedules")

    n = len(names)
    fig_height = max(2.5, n * 0.38 + 1.2)
    fig, ax = plt.subplots(figsize=(7, fig_height), facecolor="#FAFAFA")

    y_pos = np.arange(n)
    ax.barh(y_pos, latencies, color=colors, edgecolor="none", height=0.55)

    for i, ann in enumerate(annotations):
        txt_color = "#333" if colors[i] == "#43A047" else "#999"
        style = "italic" if colors[i] == "#E0E0E0" else "normal"
        ax.text(1.05, i, ann, va="center", fontsize=8, color=txt_color, style=style)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlim(0, 2.5)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    fig.tight_layout()

    if return_format == "pil":
        return _fig_to_pil(fig)
    elif return_format == "fig":
        return fig
    return _fig_to_base64_img(fig)


# ──────────────────────────────────────────────────────────────────────
# Candidate latency scatter (search space exploration)
# ──────────────────────────────────────────────────────────────────────

def candidate_scatter_chart(
    records: List[dict],
    title: str = "Schedule Search Space Exploration",
    return_format: str = "html",
) -> Any:
    """Scatter plot of all candidate latencies grouped by task.

    Each dot is a measured schedule candidate. The x-axis is the trial
    index, y-axis is the measured latency. Color encodes task identity.
    This shows *how* the tuner explored different schedule configurations.
    """
    _check_mpl()
    import numpy as np

    if not records:
        fig, ax = plt.subplots(figsize=(7, 3), facecolor="#FAFAFA")
        ax.text(0.5, 0.5, "No candidates", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#999")
        ax.set_axis_off()
        return _fig_to_base64_img(fig)

    valid = [r for r in records if r.get("run_ms", float("inf")) < 1e6]
    if not valid:
        fig, ax = plt.subplots(figsize=(7, 3), facecolor="#FAFAFA")
        ax.text(0.5, 0.5, "No valid measurements", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="#999")
        ax.set_axis_off()
        return _fig_to_base64_img(fig)

    task_names = sorted(set(r["task_name"] for r in valid))
    cmap = plt.cm.Set2(np.linspace(0, 1, max(len(task_names), 1)))
    task_color = {name: cmap[i] for i, name in enumerate(task_names)}

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#FAFAFA")

    for name in task_names:
        task_recs = [r for r in valid if r["task_name"] == name]
        xs = [r["candidate_id"] for r in task_recs]
        ys = [r["run_ms"] for r in task_recs]
        ax.scatter(xs, ys, label=_short_name(name, 18), color=task_color[name],
                   s=30, alpha=0.7, edgecolors="white", linewidths=0.5)

    ax.set_xlabel("Trial Index", fontsize=9)
    ax.set_ylabel("Measured Latency (ms)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if len(task_names) <= 8:
        ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

    fig.tight_layout()

    if return_format == "pil":
        return _fig_to_pil(fig)
    elif return_format == "fig":
        return fig
    return _fig_to_base64_img(fig)
