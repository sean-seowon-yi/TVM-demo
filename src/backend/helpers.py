"""Low-level helpers used across the pipeline and visualization layers."""

from __future__ import annotations

import io
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────────────
# Environment introspection
# ──────────────────────────────────────────────────────────────────────

def get_device_info() -> Dict[str, Any]:
    """Return a dict describing the runtime environment.

    Keys: python, torch, cuda_available, gpu_name, cuda_version,
          tvm_available, tvm_version, tvm_cuda_target.
    """
    import sys
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None,
        "cuda_version": None,
        "tvm_available": False,
        "tvm_version": None,
        "tvm_cuda_target": False,
    }

    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda

    try:
        import tvm  # type: ignore
        info["tvm_available"] = True
        info["tvm_version"] = tvm.__version__
        try:
            info["tvm_cuda_target"] = tvm.cuda(0).exist
        except Exception:
            pass
    except ImportError:
        pass

    return info


def format_device_banner(info: Dict[str, Any]) -> str:
    """Pretty-print the device info dict as a multi-line banner."""
    lines = [
        "═══════════════════════════════════════",
        "  TVM Demo — Environment",
        "═══════════════════════════════════════",
        f"  Python:       {info['python']}",
        f"  PyTorch:      {info['torch']}",
        f"  CUDA:         {'Yes' if info['cuda_available'] else 'No'}",
    ]
    if info["cuda_available"]:
        lines.append(f"  GPU:          {info['gpu_name']}")
        lines.append(f"  CUDA version: {info['cuda_version']}")
    lines.append(f"  TVM:          {info['tvm_version'] or 'not installed'}")
    if info["tvm_available"]:
        lines.append(f"  TVM → CUDA:   {'Yes' if info['tvm_cuda_target'] else 'No'}")
    lines.append("═══════════════════════════════════════")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Latency measurement
# ──────────────────────────────────────────────────────────────────────

def measure_latency(
    fn: Callable[[], Any],
    warmup: int = 10,
    repeat: int = 100,
    sync_fn: Optional[Callable[[], None]] = None,
) -> Tuple[float, List[float]]:
    """Run *fn* repeatedly and return (median_ms, all_times_ms).

    Parameters
    ----------
    fn : callable
        Zero-arg function to benchmark.
    warmup : int
        Warm-up iterations (not measured).
    repeat : int
        Measured iterations.
    sync_fn : callable or None
        Called before each timing boundary (e.g. ``torch.cuda.synchronize``).
    """
    for _ in range(warmup):
        fn()
    if sync_fn:
        sync_fn()

    times: List[float] = []
    for _ in range(repeat):
        if sync_fn:
            sync_fn()
        t0 = time.perf_counter()
        fn()
        if sync_fn:
            sync_fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return statistics.median(times), times


# ──────────────────────────────────────────────────────────────────────
# Numeric helpers
# ──────────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flat vectors."""
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    dot = np.dot(a_flat, b_flat)
    norm = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if norm < 1e-12:
        return 0.0
    return float(dot / norm)


def top_k_predictions(
    logits: np.ndarray,
    categories: List[str],
    k: int = 5,
) -> List[dict]:
    """Return top-*k* predictions as ``[{"class": str, "prob": float}, ...]``."""
    probs = _softmax(logits.squeeze())
    idx = np.argsort(probs)[::-1][:k]
    return [
        {"class": categories[int(i)], "prob": float(probs[i]), "index": int(i)}
        for i in idx
    ]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


# ──────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────

def load_image(source: str | Image.Image) -> Image.Image:
    """Open an image from a file path or return an already-opened PIL Image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    return Image.open(source).convert("RGB")


def prepare_input_tensor(
    image: Image.Image,
    transform: Callable,
) -> torch.Tensor:
    """Apply *transform* to a PIL image and return a batch-1 tensor."""
    tensor = transform(image)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    return tensor


def download_sample_image(
    url: str = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg",
) -> Image.Image:
    """Download a sample image for testing (defaults to a Wikimedia cat)."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "TVM-Demo/1.0"})
    with urllib.request.urlopen(req) as resp:
        data = resp.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


# ──────────────────────────────────────────────────────────────────────
# IR / text helpers
# ──────────────────────────────────────────────────────────────────────

def count_ir_lines(ir_text: str) -> int:
    return ir_text.count("\n") + 1


def truncate_text(text: str, max_lines: int = 200) -> str:
    """Truncate text to *max_lines*, appending a note if trimmed."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    kept = lines[:max_lines]
    kept.append(f"\n... ({len(lines) - max_lines} more lines truncated)")
    return "\n".join(kept)


def model_summary(model: torch.nn.Module) -> Tuple[str, int]:
    """Return (summary_str, total_param_count) for a PyTorch model."""
    total = sum(p.numel() for p in model.parameters())
    lines = [f"Model: {model.__class__.__name__}", f"Parameters: {total:,}"]
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        lines.append(f"  {name}: {module.__class__.__name__}  ({n:,} params)")
    return "\n".join(lines), total
