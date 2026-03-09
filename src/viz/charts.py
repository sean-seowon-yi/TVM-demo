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


def _short_name(name: str, max_len: int = 20) -> str:
    """Shorten a task name for chart labels."""
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."
