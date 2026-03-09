"""Core pipeline functions — one function per demo stage.

Every function is stateless: it takes explicit inputs, returns explicit
outputs, and never mutates global state.  The caller (``app.py`` or a test)
is responsible for storing results in a :class:`DemoState`.

TVM imports are guarded so the module is importable even when TVM is
missing (useful for linting and unit-testing the PyTorch-only stages).
"""

from __future__ import annotations

import copy
import logging
import os
import textwrap
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Ensure nvcc and gcc from the conda/tvm_env are on PATH
_TVM_ENV_BIN = os.path.expanduser("~/tvm_env/bin")
if os.path.isdir(_TVM_ENV_BIN) and _TVM_ENV_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _TVM_ENV_BIN + ":" + os.environ.get("PATH", "")

_tvm_prevent_free: list = []


def _safe_mod_functions(mod: Any) -> list:
    """Return list(mod.functions.items()) and pin TVM wrappers in memory.

    Prevents TVMFFIObjectDecRef segfault by keeping refcount > 0.
    """
    items = list(mod.functions.items())
    _tvm_prevent_free.append(items)
    return items

import numpy as np
import torch
import torchvision.models as models
from PIL import Image

from .helpers import (
    cosine_similarity,
    measure_latency,
    model_summary,
    prepare_input_tensor,
    top_k_predictions,
)

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# TVM availability guard
# ──────────────────────────────────────────────────────────────────────

_TVM_AVAILABLE = False
try:
    import tvm  # type: ignore
    from tvm import relax as tvm_relax  # type: ignore

    _TVM_AVAILABLE = True

    # ---- Neutralise TVM FFI destructors --------------------------------
    # mlc-ai-nightly has a use-after-free bug: C++ objects backing Python
    # wrappers can be freed internally by pass transforms, but the Python
    # wrapper still holds a stale pointer.  When Python's normal refcount
    # drops to 0 (on ANY thread — including Gradio's asyncio thread),
    # tp_dealloc → TVMFFIObjectDecRef dereferences the freed pointer → SIGSEGV.
    #
    # TVM does NOT use Python __del__; the destructor is the C-level
    # tp_dealloc function pointer on tvm_ffi.core.Object and
    # tvm_ffi.core.Function.  We use ctypes to replace those function
    # pointers with object's tp_dealloc (which just calls tp_free to
    # reclaim the Python wrapper memory, without touching C++ pointers).
    # TVM C++ objects are leaked for the process lifetime, which is
    # acceptable for a short-lived demo app.
    import ctypes as _ctypes
    _TP_DEALLOC_OFFSET = 48  # PyTypeObject layout on CPython 3.10+ 64-bit

    def _get_tp_dealloc(cls):
        return _ctypes.cast(
            id(cls) + _TP_DEALLOC_OFFSET,
            _ctypes.POINTER(_ctypes.c_void_p),
        )[0]

    def _set_tp_dealloc(cls, addr):
        _ctypes.cast(
            id(cls) + _TP_DEALLOC_OFFSET,
            _ctypes.POINTER(_ctypes.c_void_p),
        )[0] = addr

    _TestCls = type("_T", (object,), {})
    _subtype_dealloc = _get_tp_dealloc(_TestCls)
    _object_dealloc = _get_tp_dealloc(object)
    del _TestCls

    if _subtype_dealloc != _object_dealloc:
        _probe_types = []
        for _attr in ("tvm.ir.IRModule", "tvm.tir.PrimFunc",
                       "tvm.runtime.Module", "tvm.runtime.PackedFunc"):
            try:
                _obj = tvm
                for _p in _attr.split(".")[1:]:
                    _obj = getattr(_obj, _p)
                _probe_types.append(_obj)
            except AttributeError:
                pass

        _patched_ids: set = set()
        for _probe in _probe_types:
            for _cls in _probe.__mro__:
                if _cls is object or id(_cls) in _patched_ids:
                    continue
                _d = _get_tp_dealloc(_cls)
                if _d != _subtype_dealloc and _d != _object_dealloc:
                    _set_tp_dealloc(_cls, _object_dealloc)
                    _patched_ids.add(id(_cls))
                    print(
                        f"[TVM-fix] Replaced tp_dealloc on "
                        f"{_cls.__module__}.{_cls.__qualname__}"
                    )
        del _patched_ids, _probe_types
    else:
        print("[TVM-fix] WARNING: tp_dealloc offset verification failed, "
              "skipping patch")

    del _get_tp_dealloc, _set_tp_dealloc, _TP_DEALLOC_OFFSET, _ctypes
    # --------------------------------------------------------------------

except ImportError:
    tvm = None  # type: ignore
    tvm_relax = None  # type: ignore


# ---- NDArray helper (tvm.nd.array is absent in mlc-ai builds) ----------
def _nd_array(np_arr: "np.ndarray", dev: Any) -> Any:
    """Create a TVM tensor on *dev* from a numpy array.

    Works across both standard TVM (tvm.nd.array) and mlc-ai-nightly
    where tvm.nd doesn't exist.
    """
    if hasattr(tvm, "nd") and hasattr(tvm.nd, "array"):
        return tvm.nd.array(np_arr, dev)
    import torch as _torch
    t = _torch.from_numpy(np_arr)
    is_cuda = "cuda" in str(dev) or getattr(dev, "device_name", "") == "cuda"
    if is_cuda:
        dev_id = getattr(dev, "device_id", 0)
        t = t.cuda(dev_id)
    return tvm.runtime.from_dlpack(t)


# ---- DLight (GPU default scheduling) -----------------------------------
_dlight_available = False
_dlight = None
try:
    from tvm.s_tir import dlight as _dlight  # type: ignore
    _dlight_available = True
except ImportError:
    try:
        import tvm.dlight as _dlight  # type: ignore
        _dlight_available = True
    except ImportError:
        pass

# ---- MetaSchedule (task extraction & tuning) ----------------------------
_ms_mod = None
try:
    from tvm.s_tir import meta_schedule as _ms_mod  # type: ignore
except ImportError:
    try:
        from tvm import meta_schedule as _ms_mod  # type: ignore
    except ImportError:
        pass


def _require_tvm() -> None:
    if not _TVM_AVAILABLE:
        raise RuntimeError(
            "TVM is not installed or not importable.  "
            "Install apache-tvm or build from source — see PLAN.md §2.2."
        )


# ──────────────────────────────────────────────────────────────────────
# Stage 0 — Load Model
# ──────────────────────────────────────────────────────────────────────

def check_environment() -> Dict[str, Any]:
    """Return device / version info.  Raises nothing; always succeeds."""
    from .helpers import get_device_info
    return get_device_info()


_SUPPORTED_MODELS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
}


def load_model(
    model_name: str = "resnet18",
) -> Tuple[torch.nn.Module, Callable, List[str], str, int]:
    """Load a pretrained classifier and its ImageNet transform.

    Returns
    -------
    model : nn.Module  (eval mode, CPU)
    transform : callable
    categories : list[str]   – 1000 ImageNet class names
    summary_text : str
    param_count : int
    """
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported: {list(_SUPPORTED_MODELS)}"
        )

    factory, weights = _SUPPORTED_MODELS[model_name]
    model = factory(weights=weights).eval()
    transform = weights.transforms()
    categories = list(weights.meta["categories"])

    summary_text, param_count = model_summary(model)
    log.info("Loaded %s  (%s params)", model_name, f"{param_count:,}")

    return model, transform, categories, summary_text, param_count


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — PyTorch Baseline Inference
# ──────────────────────────────────────────────────────────────────────

def prepare_input(
    image: str | Image.Image,
    transform: Callable,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Apply *transform* to an image, return (batch tensor, numpy copy)."""
    from .helpers import load_image
    img = load_image(image)
    tensor = prepare_input_tensor(img, transform)
    return tensor, tensor.numpy().copy()


def run_pytorch_inference(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    categories: List[str],
    n_runs: int = 100,
    use_cuda: bool = True,
) -> Tuple[np.ndarray, List[dict], float]:
    """Run PyTorch eager inference with latency measurement.

    Returns
    -------
    logits : np.ndarray  (1, 1000)
    top5 : list of dicts
    median_ms : float
    """
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    m = model.to(device)
    x = input_tensor.to(device)

    with torch.no_grad():
        logits_t = m(x)
    logits_np = logits_t.cpu().numpy()
    top5 = top_k_predictions(logits_np, categories, k=5)

    sync = torch.cuda.synchronize if device.type == "cuda" else None
    median_ms, _ = measure_latency(
        fn=lambda: m(x),
        warmup=10,
        repeat=n_runs,
        sync_fn=sync,
    )
    log.info("PyTorch inference: %s  (%.2f ms)", top5[0]["class"], median_ms)
    return logits_np, top5, median_ms


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — PyTorch Computation Graph
# ──────────────────────────────────────────────────────────────────────

def trace_pytorch_graph(
    model: torch.nn.Module,
    example_input: torch.Tensor,
) -> Tuple[Any, str, List[dict], Optional[Any]]:
    """Capture the PyTorch computation graph via FX and (optionally) export.

    Returns
    -------
    fx_graph : torch.fx.Graph
    fx_code : str                  – human-readable generated Python
    node_table : list[dict]        – one dict per node (name, op, target, inputs)
    exported_program : ExportedProgram or None
    """
    import torch.fx

    model_cpu = copy.deepcopy(model).cpu().eval()
    example_cpu = example_input.cpu()

    traced = torch.fx.symbolic_trace(model_cpu)
    fx_graph = traced.graph
    fx_code = traced.code

    node_table = _build_node_table(fx_graph)
    log.info("FX graph: %d nodes", len(node_table))

    exported = _try_torch_export(model_cpu, example_cpu)

    return fx_graph, fx_code, node_table, exported


def _build_node_table(graph: Any) -> List[dict]:
    """Walk FX graph nodes and build a display-friendly table."""
    rows: List[dict] = []
    for node in graph.nodes:
        target_str = str(node.target) if node.target is not None else ""
        if len(target_str) > 80:
            target_str = target_str[:77] + "..."

        input_names = [str(a) for a in node.args if hasattr(a, "name")]
        rows.append({
            "name": node.name,
            "op": node.op,
            "target": target_str,
            "inputs": ", ".join(input_names),
            "num_users": len(node.users),
        })
    return rows


def _try_torch_export(
    model_cpu: torch.nn.Module,
    example_cpu: torch.Tensor,
) -> Optional[Any]:
    """Best-effort ``torch.export.export``.  Returns None on failure."""
    try:
        from torch.export import export as torch_export
        with torch.no_grad():
            return torch_export(model_cpu, (example_cpu,))
    except Exception as exc:
        log.warning("torch.export.export failed: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — Import into TVM (Relax IR)
# ──────────────────────────────────────────────────────────────────────

def import_to_tvm(
    model: torch.nn.Module,
    example_input: torch.Tensor,
) -> Tuple[Any, List[np.ndarray], str]:
    """Convert a PyTorch model into a TVM Relax IRModule.

    Uses ``torch.export.export`` → ``from_exported_program`` (modern path).
    Falls back to ``torch.fx.symbolic_trace`` → ``from_fx`` if needed.

    Returns
    -------
    mod : tvm.ir.IRModule
    params_np : list[np.ndarray]   – model parameters in function-arg order
    ir_text : str                  – the ``mod.script()`` source
    """
    _require_tvm()
    model_cpu = copy.deepcopy(model).cpu().eval()
    example_cpu = example_input.cpu()

    mod, params_np = _import_via_export(model_cpu, example_cpu)

    ir_text = mod.script()
    log.info(
        "TVM import: IRModule with %d global functions, IR is %d lines",
        len(mod.functions),
        ir_text.count("\n"),
    )
    return mod, params_np, ir_text


def _import_via_export(
    model_cpu: torch.nn.Module,
    example_cpu: torch.Tensor,
) -> Tuple[Any, List[np.ndarray]]:
    """Primary import path: torch.export → from_exported_program."""
    from torch.export import export as torch_export

    try:
        from tvm.relax.frontend.torch import from_exported_program  # type: ignore
    except ImportError:
        log.warning("from_exported_program unavailable, trying from_fx fallback")
        return _import_via_fx(model_cpu, example_cpu)

    with torch.no_grad():
        exported = torch_export(model_cpu, (example_cpu,))

    try:
        mod = from_exported_program(
            exported,
            keep_params_as_input=True,
            unwrap_unit_return_tuple=True,
        )
    except TypeError:
        mod = from_exported_program(exported, keep_params_as_input=True)

    params_np = _extract_params_matching_tvm(model_cpu, mod)
    if params_np is None:
        params_np = _extract_params_from_state_dict(model_cpu, exported)
    _tvm_prevent_free.append(mod)
    return mod, params_np


def _import_via_fx(
    model_cpu: torch.nn.Module,
    example_cpu: torch.Tensor,
) -> Tuple[Any, List[np.ndarray]]:
    """Fallback import path: torch.fx.symbolic_trace → from_fx."""
    import torch.fx

    try:
        from tvm.relax.frontend.torch import from_fx  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Neither from_exported_program nor from_fx is available in your "
            "TVM build.  Please upgrade TVM."
        ) from exc

    traced = torch.fx.symbolic_trace(model_cpu)
    shape = list(example_cpu.shape)
    mod = from_fx(traced, input_info=[("input0", shape, "float32")])
    _tvm_prevent_free.append(mod)

    params_np = [
        v.detach().cpu().numpy()
        for v in model_cpu.state_dict().values()
    ]
    return mod, params_np


def _extract_params_matching_tvm(
    model_cpu: torch.nn.Module,
    tvm_mod: Any,
) -> Optional[List[np.ndarray]]:
    """Match TVM function params to state_dict by name convention.

    TVM names params like ``p_conv1_weight`` which corresponds to
    ``conv1.weight`` in the PyTorch state_dict (dots→underscores, with a
    ``p_`` prefix).
    """
    try:
        main_fn = tvm_mod["main"]
        # Skip the first param (input tensor)
        tvm_param_names = [p.name_hint for p in main_fn.params[1:]]
    except Exception:
        return None

    state = model_cpu.state_dict()
    ordered: List[np.ndarray] = []

    for tvm_name in tvm_param_names:
        sd_name = tvm_name.removeprefix("p_").replace("_", ".")
        if sd_name in state:
            ordered.append(state[sd_name].detach().cpu().numpy())
            continue
        # Try replacing last segment separator: p_fc_weight → fc.weight
        # Need smarter matching: try all possible dot-placements
        found = False
        clean = tvm_name.removeprefix("p_")
        for sd_key in state:
            if sd_key.replace(".", "_") == clean:
                ordered.append(state[sd_key].detach().cpu().numpy())
                found = True
                break
        if not found:
            log.debug("Cannot match TVM param '%s' to state_dict", tvm_name)
            return None

    log.info("Matched %d/%d TVM params to state_dict by name", len(ordered), len(tvm_param_names))
    return ordered


def _extract_params_from_state_dict(
    model_cpu: torch.nn.Module,
    exported: Any,
) -> List[np.ndarray]:
    """Extract model parameters in the order the Relax function expects.

    ``from_exported_program`` with ``keep_params_as_input=True`` lifts
    parameters and buffers as additional function inputs.  Their order
    follows ``graph_signature.parameters`` then ``graph_signature.buffers``
    in the ExportedProgram.  We map those names back to the state dict.
    """
    state = model_cpu.state_dict()

    try:
        sig = exported.graph_signature
        lifted_names: List[str] = []

        param_input_map = getattr(sig, "inputs_to_parameters", None) or {}
        buffer_input_map = getattr(sig, "inputs_to_buffers", None) or {}

        if param_input_map or buffer_input_map:
            for sd_name in param_input_map.values():
                lifted_names.append(sd_name)
            for sd_name in buffer_input_map.values():
                lifted_names.append(sd_name)
        else:
            lifted_names.extend(getattr(sig, "parameters", []))
            lifted_names.extend(getattr(sig, "buffers", []))

        if lifted_names:
            _verify_param_names(lifted_names, state)
            return [state[n].detach().cpu().numpy() for n in lifted_names]
    except Exception as exc:
        log.warning("Could not use graph_signature for param ordering: %s", exc)

    log.info("Falling back to state_dict iteration order for params")
    all_params = [v.detach().cpu().numpy() for v in state.values()]

    # If the exported program didn't give us a signature, try to match the
    # count expected by the Relax function (first param is the input tensor,
    # remaining are model params).
    try:
        mod_obj = getattr(exported, "_tvm_mod", None)
        if mod_obj is None:
            return all_params
        main_fn = mod_obj["main"]
        expected = len(main_fn.params) - 1
        if expected > 0 and expected != len(all_params):
            log.info(
                "State dict has %d entries but TVM expects %d params; "
                "truncating (BN running stats likely folded as constants).",
                len(all_params), expected,
            )
            return all_params[:expected]
    except Exception:
        pass
    return all_params


def _verify_param_names(names: List[str], state: Dict[str, Any]) -> None:
    missing = [n for n in names if n not in state]
    if missing:
        raise KeyError(
            f"Parameter name mismatch — expected keys in state_dict but "
            f"got: {missing[:5]}{'…' if len(missing) > 5 else ''}"
        )


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — Apply TVM Graph-Level Passes (one by one)
# ──────────────────────────────────────────────────────────────────────

def apply_passes_stepwise(
    mod: Any,
) -> Tuple[Any, Dict[str, str], List[str], Dict[str, dict]]:
    """Apply Relax passes individually and snapshot the IR after each.

    Returns
    -------
    current_mod : IRModule          – module after all passes
    snapshots : dict[str, str]      – pass_name → IR text
    pass_order : list[str]          – ordered pass names
    deltas : dict[str, dict]        – pass_name → {"functions_before", "functions_after", …}
    """
    _require_tvm()
    import tvm.relax.transform as T  # type: ignore

    pass_defs = _get_pass_sequence()

    snapshots: Dict[str, str] = {"imported": mod.script()}
    pass_order: List[str] = []
    deltas: Dict[str, dict] = {}
    current = mod
    _tvm_prevent_free.append(mod)

    for name, make_pass in pass_defs:
        funcs_before = len(current.functions)
        tir_before = _count_tir_funcs(current)
        t0 = time.time()

        try:
            p = make_pass()
            current = p(current)
            _tvm_prevent_free.append(current)
        except Exception as exc:
            log.warning("Pass '%s' failed: %s — skipping", name, exc)
            continue

        elapsed = time.time() - t0
        funcs_after = len(current.functions)
        tir_after = _count_tir_funcs(current)
        ir_text = current.script()

        snapshots[name] = ir_text
        pass_order.append(name)
        deltas[name] = {
            "functions_before": funcs_before,
            "functions_after": funcs_after,
            "tir_before": tir_before,
            "tir_after": tir_after,
            "ir_lines": ir_text.count("\n"),
            "elapsed_s": round(elapsed, 3),
        }
        log.info(
            "Pass %-28s funcs %d→%d  tir %d→%d  (%.3fs)",
            name, funcs_before, funcs_after, tir_before, tir_after, elapsed,
        )

    return current, snapshots, pass_order, deltas


def _get_pass_sequence() -> List[Tuple[str, Callable]]:
    """Return the ordered list of (name, pass_factory) tuples.

    Each entry is a zero-arg callable that creates the pass object.
    We use factories so that import-time errors are deferred to execution.
    """
    import tvm.relax.transform as T  # type: ignore

    sequence: List[Tuple[str, Callable]] = []

    _try_append(sequence, "LegalizeOps",          lambda: T.LegalizeOps())
    _try_append(sequence, "AnnotateTIROpPattern", lambda: T.AnnotateTIROpPattern())
    _try_append(sequence, "FuseOps",              lambda: T.FuseOps())
    _try_append(sequence, "FuseTIR",              lambda: T.FuseTIR())
    _try_append(sequence, "DeadCodeElimination",  lambda: T.DeadCodeElimination())

    return sequence


def _try_append(
    seq: List[Tuple[str, Callable]],
    name: str,
    factory: Callable,
) -> None:
    """Append to *seq* only if the factory callable looks valid."""
    try:
        factory()
        seq.append((name, factory))
    except AttributeError:
        log.warning("Pass '%s' not found in this TVM build — skipping", name)
    except Exception:
        seq.append((name, factory))


def _count_tir_funcs(mod: Any) -> int:
    count = 0
    for _, func in _safe_mod_functions(mod):
        if hasattr(tvm, "tir") and isinstance(func, tvm.tir.PrimFunc):
            count += 1
    return count


# ──────────────────────────────────────────────────────────────────────
# Stage 5 — Operator Extraction (Lowered TIR Functions)
# ──────────────────────────────────────────────────────────────────────

def extract_operators(mod: Any) -> List[dict]:
    """Extract all TIR PrimFuncs from the IRModule after passes.

    Returns a list of dicts, one per operator:
        name, params (list of {name, dtype, shape}), num_blocks,
        tir_source, op_kind (inferred from name).
    """
    _require_tvm()
    operators: List[dict] = []

    for gv, func in _safe_mod_functions(mod):
        if not isinstance(func, tvm.tir.PrimFunc):
            continue

        name = gv.name_hint if hasattr(gv, "name_hint") else str(gv)
        params_info = _extract_prim_params(func)
        tir_src = _safe_script(func)

        operators.append({
            "name": name,
            "params": params_info,
            "num_blocks": _count_blocks(func.body),
            "tir_source": tir_src,
            "op_kind": _infer_op_kind(name),
            "ir_lines": tir_src.count("\n") + 1,
        })

    operators.sort(key=lambda o: o["name"])
    log.info("Extracted %d TIR operators", len(operators))
    return operators


def _extract_prim_params(func: Any) -> List[dict]:
    """Build a list of {name, dtype, shape} for a PrimFunc's buffer parameters."""
    result: List[dict] = []
    for param in func.params:
        entry: dict = {"name": str(param)}
        buf = func.buffer_map.get(param)
        if buf is not None:
            entry["dtype"] = str(buf.dtype)
            try:
                entry["shape"] = [int(s) if hasattr(s, "value") else str(s) for s in buf.shape]
            except Exception:
                entry["shape"] = str(buf.shape)
        result.append(entry)
    return result


def _count_blocks(node: Any, _depth: int = 0) -> int:
    """Recursively count tvm.tir.Block nodes in a TIR AST."""
    if _depth > 200:
        return 0
    count = 0
    if type(node).__name__ == "Block":
        count += 1
    for attr_name in ("body", "then_case", "else_case", "block"):
        child = getattr(node, attr_name, None)
        if child is not None:
            count += _count_blocks(child, _depth + 1)
    seq = getattr(node, "seq", None)
    if seq is not None:
        for s in seq:
            count += _count_blocks(s, _depth + 1)
    return count


_OP_KIND_KEYWORDS = [
    ("conv", "conv"),
    ("matmul", "matmul"),
    ("dense", "dense"),
    ("batch_norm", "batchnorm"),
    ("relu", "relu"),
    ("add", "elemwise"),
    ("multiply", "elemwise"),
    ("pool", "pool"),
    ("softmax", "softmax"),
    ("reshape", "reshape"),
    ("transpose", "transpose"),
    ("layer_norm", "layernorm"),
]


def _infer_op_kind(name: str) -> str:
    lower = name.lower()
    for kw, kind in _OP_KIND_KEYWORDS:
        if kw in lower:
            return kind
    return "other"


def _safe_script(obj: Any) -> str:
    """Call .script() with a fallback to str()."""
    try:
        return obj.script()
    except Exception:
        return str(obj)


# ──────────────────────────────────────────────────────────────────────
# Stage 6 — TensorIR / AST Visualization
# ──────────────────────────────────────────────────────────────────────

def get_tir_ast(
    mod: Any,
    op_name: str,
) -> Tuple[str, dict]:
    """Return the TIR source and a structured AST summary for one PrimFunc.

    Parameters
    ----------
    mod : IRModule (post-passes)
    op_name : name of the PrimFunc to inspect

    Returns
    -------
    tir_source : str
    ast_summary : dict  with keys "blocks", "loops", "buffers"
    """
    _require_tvm()

    func = _find_prim_func(mod, op_name)
    tir_source = _safe_script(func)
    ast_summary = _walk_tir_ast(func.body)

    buf_info = []
    for param in func.params:
        buf = func.buffer_map.get(param)
        if buf is not None:
            buf_info.append({
                "name": str(buf.name),
                "shape": [int(s) if hasattr(s, "value") else str(s) for s in buf.shape],
                "dtype": str(buf.dtype),
                "scope": str(getattr(buf, "scope", "")),
            })
    ast_summary["buffers"] = buf_info

    log.info(
        "TIR AST for '%s': %d blocks, %d loops, %d buffers",
        op_name,
        len(ast_summary.get("blocks", [])),
        len(ast_summary.get("loops", [])),
        len(buf_info),
    )
    return tir_source, ast_summary


def _find_prim_func(mod: Any, name: str) -> Any:
    """Lookup a PrimFunc by name in the IRModule."""
    for gv, func in _safe_mod_functions(mod):
        gv_name = gv.name_hint if hasattr(gv, "name_hint") else str(gv)
        if gv_name == name and isinstance(func, tvm.tir.PrimFunc):
            return func
    available = [
        gv_name for gv_name, f in [
            (gv.name_hint if hasattr(gv, "name_hint") else str(gv), f)
            for gv, f in _safe_mod_functions(mod)
        ]
        if isinstance(f, tvm.tir.PrimFunc)
    ]
    raise KeyError(
        f"PrimFunc '{name}' not found. Available: {available[:10]}"
    )


def _walk_tir_ast(node: Any, _depth: int = 0) -> dict:
    """Recursively walk a TIR body and collect blocks and loops.

    Returns {"blocks": [...], "loops": [...]}.
    """
    result: dict = {"blocks": [], "loops": []}
    if _depth > 200:
        return result
    _walk_tir_ast_impl(node, result, _depth)
    return result


def _walk_tir_ast_impl(node: Any, out: dict, depth: int) -> None:
    if depth > 200:
        return
    node_type = type(node).__name__

    if node_type == "For":
        loop_info: dict = {
            "var": str(getattr(node, "loop_var", "?")),
            "extent": _tir_value(getattr(node, "extent", None)),
            "kind": str(getattr(node, "kind", "")),
        }
        thread = getattr(node, "thread_binding", None)
        if thread is not None:
            tag = getattr(thread, "thread_tag", None)
            loop_info["thread_binding"] = str(tag) if tag else ""
        else:
            loop_info["thread_binding"] = ""
        out["loops"].append(loop_info)
        _walk_tir_ast_impl(node.body, out, depth + 1)
        return

    if node_type == "Block":
        block_info: dict = {"name": str(getattr(node, "name_hint", ""))}
        iters = getattr(node, "iter_vars", [])
        block_info["iter_vars"] = [str(iv) for iv in iters]
        reads = getattr(node, "reads", [])
        writes = getattr(node, "writes", [])
        block_info["num_reads"] = len(reads)
        block_info["num_writes"] = len(writes)
        out["blocks"].append(block_info)

        for iv in iters:
            dom = getattr(iv, "dom", None)
            extent_val = "?"
            if dom is not None:
                extent_val = _tir_value(getattr(dom, "extent", None))
            iter_type_raw = getattr(iv, "iter_type", None)
            if iter_type_raw is not None:
                it_str = str(iter_type_raw)
                if "Spatial" in it_str or it_str == "0":
                    kind = "S"
                elif "Reduction" in it_str or it_str == "2":
                    kind = "R"
                else:
                    kind = it_str
            else:
                kind = ""
            out["loops"].append({
                "var": str(getattr(iv, "var", iv)),
                "extent": extent_val,
                "kind": kind,
                "thread_binding": "",
                "source": "block_iter",
            })

        body = getattr(node, "body", None)
        if body is not None:
            _walk_tir_ast_impl(body, out, depth + 1)
        return

    if node_type == "BlockRealize":
        block = getattr(node, "block", None)
        if block is not None:
            _walk_tir_ast_impl(block, out, depth + 1)
        return

    if node_type == "SeqStmt":
        for s in node:
            _walk_tir_ast_impl(s, out, depth + 1)
        return

    for attr in ("body", "then_case", "else_case"):
        child = getattr(node, attr, None)
        if child is not None:
            _walk_tir_ast_impl(child, out, depth + 1)


def _tir_value(v: Any) -> str:
    """Convert a TIR expression to a display string."""
    if v is None:
        return "?"
    if hasattr(v, "value"):
        return str(v.value)
    return str(v)


# ──────────────────────────────────────────────────────────────────────
# Stage 7 — Tensor Expression Microscope (standalone TE/TOPI conv2d)
# ──────────────────────────────────────────────────────────────────────

def build_te_microscope(
    n: int = 1,
    ci: int = 64,
    h: int = 56,
    w: int = 56,
    co: int = 64,
    kh: int = 3,
    kw: int = 3,
    stride: int = 1,
    padding: int = 1,
) -> Tuple[str, str, str]:
    """Build a standalone conv2d with TE/TOPI and lower it to TIR.

    This is the "microscope operator" that shows the compute/schedule
    separation (paper Section 4.1, Figure 5).

    Returns
    -------
    compute_source : str   – readable description of the TE declaration
    naive_tir : str        – TIR of the un-scheduled (naive) loop nest
    explanation : str      – educational text about compute/schedule separation
    """
    _require_tvm()

    compute_source, naive_tir = _te_conv2d(n, ci, h, w, co, kh, kw, stride, padding)

    explanation = textwrap.dedent(f"""\
    Microscope Operator: Conv2D ({co}×{ci}×{kh}×{kw}, stride={stride}, pad={padding})

    TVM separates the *what* (tensor expression) from the *how* (schedule).

    The tensor expression above declares a 2-D convolution using a reduction
    over the input channels and kernel spatial dimensions.  No loop ordering,
    tiling, or threading is specified — that is the job of the schedule.

    The "Naive TIR" panel shows the default lowered loop nest: a simple set of
    nested for-loops that directly implement the compute declaration.  In the
    next stages (Schedule Search, Tuning), TVM's MetaSchedule will explore
    thousands of schedule transformations — tile sizes, loop reordering, thread
    bindings, cache reads/writes — to find an optimized version of this program
    for the target GPU.

    This maps directly to paper Section 4.1 and Figure 5 (left column →
    schedule transformation → optimized low-level code).
    """)

    log.info("TE microscope: conv2d %dx%dx%dx%d", n, co, h, w)
    return compute_source, naive_tir, explanation


def _te_conv2d(
    n: int, ci: int, h: int, w: int,
    co: int, kh: int, kw: int,
    stride: int, padding: int,
) -> Tuple[str, str]:
    """Create a TE conv2d and lower it.  Returns (compute_text, tir_text)."""
    try:
        from tvm import te, topi  # type: ignore

        data = te.placeholder((n, ci, h, w), name="data", dtype="float32")
        weight = te.placeholder((co, ci, kh, kw), name="weight", dtype="float32")

        out = topi.nn.conv2d(
            data, weight,
            strides=stride,
            padding=padding,
            dilation=1,
            data_layout="NCHW",
        )

        compute_text = (
            f'data   = te.placeholder(({n}, {ci}, {h}, {w}), name="data", dtype="float32")\n'
            f'weight = te.placeholder(({co}, {ci}, {kh}, {kw}), name="weight", dtype="float32")\n'
            f"\n"
            f"out = topi.nn.conv2d(\n"
            f"    data, weight,\n"
            f"    strides={stride}, padding={padding}, dilation=1,\n"
            f'    data_layout="NCHW",\n'
            f")"
        )

        naive_tir = _lower_te_to_tir(te, data, weight, out)
        return compute_text, naive_tir

    except ImportError:
        return (
            "# tvm.te / tvm.topi not available in this build",
            "# TIR lowering unavailable",
        )
    except Exception as exc:
        log.warning("TE microscope conv2d failed: %s", exc)
        return (
            f"# TE conv2d construction failed: {exc}",
            f"# TIR lowering failed: {exc}",
        )


def _lower_te_to_tir(te: Any, data: Any, weight: Any, out: Any) -> str:
    """Lower a TE compute to TIR, trying modern API first then legacy."""
    # Modern TVM (0.15+): te.create_prim_func
    if hasattr(te, "create_prim_func"):
        try:
            func = te.create_prim_func([data, weight, out])
            return _safe_script(func)
        except Exception as exc:
            log.debug("te.create_prim_func failed: %s", exc)

    # Legacy TVM: te.create_schedule + tvm.lower
    if hasattr(te, "create_schedule"):
        try:
            s = te.create_schedule(out.op)
            lowered = tvm.lower(s, [data, weight, out], simple_mode=True)
            return _safe_script(lowered)
        except Exception as exc:
            log.debug("te.create_schedule failed: %s", exc)

    return "# Neither te.create_prim_func nor te.create_schedule available"


# ──────────────────────────────────────────────────────────────────────
# Stage 8 — MetaSchedule Task Extraction
# ──────────────────────────────────────────────────────────────────────

def extract_tuning_tasks(
    mod: Any,
    target_str: str = "cuda",
) -> Tuple[List[dict], Any, Any]:
    """Extract tunable tasks from the IRModule via MetaSchedule.

    Paper mapping: Section 5.1 -- "Schedule Space Specification."

    Returns
    -------
    task_dicts : list[dict]  -- one dict per task (name, shape_key, flop_estimate,
                                tir_source, target)
    tasks_raw  : list        -- raw ExtractedTask objects (for run_tuning)
    target     : tvm.target.Target
    """
    _require_tvm()

    target = _resolve_target(target_str)
    tasks_raw = _ms_extract_tasks(mod, target)

    task_dicts: List[dict] = []
    for i, task in enumerate(tasks_raw):
        name = getattr(task, "task_name", None) or getattr(task, "name", f"task_{i}")
        tir_src = ""
        try:
            dispatch_mod = getattr(task, "dispatched", None)
            if dispatch_mod is None:
                dispatch_mod = getattr(task, "mod", None)
            if dispatch_mod is not None:
                # dispatched can be a list/Array of IRModules
                if hasattr(dispatch_mod, "__len__") and not hasattr(dispatch_mod, "script"):
                    dispatch_mod = dispatch_mod[0] if len(dispatch_mod) > 0 else None
            if dispatch_mod is not None:
                tir_src = dispatch_mod.script() if hasattr(dispatch_mod, "script") else str(dispatch_mod)
        except Exception:
            pass

        flop_est = _estimate_flops(task)
        weight = float(getattr(task, "weight", 1.0))

        task_dicts.append({
            "index": i,
            "name": str(name),
            "flop_estimate": flop_est,
            "weight": weight,
            "tir_source": tir_src,
            "tir_lines": tir_src.count("\n") + 1 if tir_src else 0,
            "target": str(target),
        })

    task_dicts.sort(key=lambda t: t["weight"], reverse=True)
    log.info("Extracted %d tuning tasks", len(task_dicts))
    return task_dicts, tasks_raw, target


def _ms_extract_tasks(mod: Any, target: Any) -> list:
    """Call MetaSchedule's task extraction with fallbacks for API drift."""
    ms = _ms_mod

    if ms is not None:
        # Path 1: relax_integration.extract_tasks (works in both tvm.s_tir.meta_schedule and tvm.meta_schedule)
        extract_fn = getattr(
            getattr(ms, "relax_integration", None), "extract_tasks", None
        )
        if extract_fn is not None:
            try:
                tasks = extract_fn(mod, target)
                if tasks:
                    log.info("extract_tasks (relax_integration) returned %d tasks", len(tasks))
                    return list(tasks)
            except Exception as exc:
                log.warning("relax_integration.extract_tasks failed: %s", exc)

        # Path 2: extract_task_from_relay (legacy, if mod was somehow relay)
        extract_fn2 = getattr(ms, "extract_task_from_relay", None)
        if extract_fn2 is not None:
            try:
                tasks, weights = extract_fn2(mod, target=target, params={})
                if tasks:
                    log.info("extract_task_from_relay returned %d tasks", len(tasks))
                    return list(tasks)
            except Exception as exc:
                log.warning("extract_task_from_relay failed: %s", exc)

    # Path 3: manually build ExtractedTask-like objects from PrimFuncs
    log.warning("MetaSchedule task extraction APIs unavailable, building manual task list")
    return _manual_task_extraction(mod, target)


def _manual_task_extraction(mod: Any, target: Any) -> list:
    """Fallback: wrap each PrimFunc as a pseudo-task for display purposes."""

    class _PseudoTask:
        def __init__(self, name: str, prim_mod: Any, weight: float = 1.0):
            self.task_name = name
            self.dispatched = prim_mod
            self.weight = weight

    tasks = []
    for gv, func in _safe_mod_functions(mod):
        if not isinstance(func, tvm.tir.PrimFunc):
            continue
        name = gv.name_hint if hasattr(gv, "name_hint") else str(gv)
        try:
            single_mod = tvm.IRModule({gv: func})
            _tvm_prevent_free.append(single_mod)
        except Exception:
            single_mod = None
        tasks.append(_PseudoTask(name, single_mod))
    return tasks


def _estimate_flops(task: Any) -> int:
    """Best-effort FLOP estimate for a tuning task."""
    try:
        dispatch_mod = getattr(task, "dispatched", None) or getattr(task, "mod", None)
        if dispatch_mod is None:
            return 0
        if hasattr(dispatch_mod, "__len__") and not hasattr(dispatch_mod, "functions"):
            dispatch_mod = dispatch_mod[0] if len(dispatch_mod) > 0 else None
        if dispatch_mod is None:
            return 0
        from tvm import tir as _tir  # type: ignore
        total = 0
        for _, func in _safe_mod_functions(dispatch_mod):
            if isinstance(func, _tir.PrimFunc):
                total += _count_tir_ops(func.body)
        return total
    except Exception:
        return 0


def _count_tir_ops(node: Any, _depth: int = 0) -> int:
    """Rough count of arithmetic ops in a TIR body (addition proxy for FLOPs)."""
    if _depth > 200:
        return 0
    count = 0
    node_type = type(node).__name__
    if node_type in ("Add", "Mul", "Sub", "Div", "FloorDiv", "Mod"):
        count += 1
    if node_type == "BufferStore":
        count += 1
    for attr in ("body", "then_case", "else_case", "block", "value"):
        child = getattr(node, attr, None)
        if child is not None:
            count += _count_tir_ops(child, _depth + 1)
    seq = getattr(node, "seq", None)
    if seq is not None:
        for s in seq:
            count += _count_tir_ops(s, _depth + 1)
    return count


# ──────────────────────────────────────────────────────────────────────
# Stage 9 — Candidate Schedule Generation (Tuning)
# ──────────────────────────────────────────────────────────────────────

def run_tuning(
    mod: Any,
    target: Any,
    work_dir: str = "./tuning_logs",
    max_trials_global: int = 64,
    num_trials_per_iter: int = 16,
    max_tasks: int = 3,
) -> Tuple[List[dict], List[dict], str]:
    """Run MetaSchedule tuning on the hottest tasks.

    Paper mapping: Section 5.3 -- "Schedule Exploration."

    Parameters
    ----------
    mod : IRModule (post-passes)
    target : tvm.target.Target
    work_dir : directory for tuning database / logs
    max_trials_global : total trial budget across all tasks
    num_trials_per_iter : candidates per iteration batch
    max_tasks : maximum number of tasks to tune (pick the heaviest)

    Returns
    -------
    tuning_records : list[dict]  -- per-candidate: task_name, trace_text,
                                    run_secs, is_best
    convergence_data : list[dict] -- trial_index, best_latency_ms
    work_dir : str  -- where tuning logs were written
    """
    _require_tvm()
    import os
    os.makedirs(work_dir, exist_ok=True)

    if target is None:
        target = _resolve_target("cuda")
        log.info("No target provided to run_tuning; resolved to %s", target)

    ms = _ms_mod
    if ms is None:
        log.warning("MetaSchedule not available, will use synthetic tuning records")

    records: List[dict] = []
    convergence: List[dict] = []

    if ms is not None:
        log.info(
            "Starting MetaSchedule tuning: max_trials=%d, trials_per_iter=%d, work_dir=%s",
            max_trials_global, num_trials_per_iter, work_dir,
        )
        records, convergence = _try_tune(
            ms, mod, target, work_dir,
            max_trials_global, num_trials_per_iter, max_tasks,
        )

    if not records:
        log.warning("Tuning produced no records; generating synthetic records for demo")
        records, convergence = _synthetic_tuning_records(mod, target)

    log.info("Tuning complete: %d records, %d convergence points", len(records), len(convergence))
    return records, convergence, work_dir


def _try_tune(
    ms: Any,
    mod: Any,
    target: Any,
    work_dir: str,
    max_trials_global: int,
    num_trials_per_iter: int,
    max_tasks: int,
) -> Tuple[List[dict], List[dict]]:
    """Attempt actual MetaSchedule tuning with multiple API paths."""

    has_tune_config = hasattr(ms, "TuneConfig")

    # Path 1: tune_relax (whole-module tuning — preferred for Relax IRModules)
    tune_relax_fn = getattr(
        getattr(ms, "relax_integration", None), "tune_relax", None
    )
    if tune_relax_fn is not None:
        try:
            if has_tune_config:
                db = tune_relax_fn(
                    mod=mod, target=target, params={},
                    config=ms.TuneConfig(
                        max_trials_global=max_trials_global,
                        num_trials_per_iter=num_trials_per_iter,
                    ),
                    work_dir=work_dir,
                )
            else:
                db = tune_relax_fn(
                    mod=mod, params={}, target=target,
                    work_dir=work_dir,
                    max_trials_global=max_trials_global,
                    num_trials_per_iter=num_trials_per_iter,
                )
            return _collect_records_from_db(db, work_dir)
        except Exception as exc:
            log.warning("tune_relax failed: %s", exc)

    # Path 2: ms.tune_tir (direct TIR tuning)
    tune_tir = getattr(ms, "tune_tir", None)
    if tune_tir is not None:
        try:
            if has_tune_config:
                db = tune_tir(
                    mod=mod, target=target,
                    config=ms.TuneConfig(
                        max_trials_global=max_trials_global,
                        num_trials_per_iter=num_trials_per_iter,
                    ),
                    work_dir=work_dir,
                )
            else:
                db = tune_tir(
                    mod=mod, target=target,
                    work_dir=work_dir,
                    max_trials_global=max_trials_global,
                    num_trials_per_iter=num_trials_per_iter,
                )
            return _collect_records_from_db(db, work_dir)
        except Exception as exc:
            log.warning("ms.tune_tir failed: %s", exc)

    # Path 3: manual per-task tuning loop
    try:
        return _tune_per_task(ms, mod, target, work_dir, max_trials_global, num_trials_per_iter, max_tasks)
    except Exception as exc:
        log.warning("Per-task tuning failed: %s", exc)

    return [], []


def _tune_per_task(
    ms: Any,
    mod: Any,
    target: Any,
    work_dir: str,
    max_trials_global: int,
    num_trials_per_iter: int,
    max_tasks: int,
) -> Tuple[List[dict], List[dict]]:
    """Tune individual tasks using ms.tune_tir on single-function modules."""
    all_records: List[dict] = []
    all_convergence: List[dict] = []
    trial_counter = 0

    extract_fn = getattr(getattr(ms, "relax_integration", None), "extract_tasks", None)
    if extract_fn is None:
        return [], []

    tasks = extract_fn(mod, target)
    tasks = list(tasks)[:max_tasks]
    trials_per_task = max(4, max_trials_global // max(len(tasks), 1))
    has_tune_config = hasattr(ms, "TuneConfig")

    for task in tasks:
        task_name = getattr(task, "task_name", "unknown")
        task_mod = getattr(task, "dispatched", None) or getattr(task, "mod", None)
        if task_mod is None:
            continue
        # dispatched can be a list/Array of IRModules — use the first one
        if hasattr(task_mod, "__len__") and not hasattr(task_mod, "functions"):
            task_mod = task_mod[0] if len(task_mod) > 0 else None
        if task_mod is None:
            continue
        try:
            if has_tune_config:
                db = ms.tune_tir(
                    mod=task_mod, target=target,
                    config=ms.TuneConfig(
                        max_trials_global=trials_per_task,
                        num_trials_per_iter=min(num_trials_per_iter, trials_per_task),
                    ),
                    work_dir=work_dir,
                )
            else:
                db = ms.tune_tir(
                    mod=task_mod, target=target,
                    work_dir=work_dir,
                    max_trials_global=trials_per_task,
                    num_trials_per_iter=min(num_trials_per_iter, trials_per_task),
                )
            recs, conv = _collect_records_from_db(db, work_dir, task_name, trial_counter)
            all_records.extend(recs)
            all_convergence.extend(conv)
            trial_counter += len(recs)
        except Exception as exc:
            log.warning("Task '%s' tuning failed: %s", task_name, exc)

    return all_records, all_convergence


def _collect_records_from_db(
    db: Any,
    work_dir: str,
    task_name_override: str = "",
    trial_offset: int = 0,
) -> Tuple[List[dict], List[dict]]:
    """Read tuning records from a MetaSchedule database object or work_dir."""
    records: List[dict] = []
    convergence: List[dict] = []

    raw_records = _read_raw_records(db, work_dir)

    best_so_far = float("inf")
    for i, rec in enumerate(raw_records):
        run_secs = _extract_run_secs(rec)
        trace_text = _extract_trace(rec)
        task_name = task_name_override or _extract_task_name(rec, i)

        run_ms = run_secs * 1000.0 if run_secs < 1e6 else float("inf")
        best_so_far = min(best_so_far, run_ms)

        records.append({
            "candidate_id": trial_offset + i,
            "task_name": task_name,
            "trace_text": trace_text,
            "run_secs": run_secs,
            "run_ms": run_ms,
            "is_best": False,
        })
        convergence.append({
            "trial_index": trial_offset + i,
            "best_latency_ms": best_so_far,
        })

    if records:
        best_idx = min(range(len(records)), key=lambda j: records[j]["run_ms"])
        records[best_idx]["is_best"] = True

    return records, convergence


def _read_raw_records(db: Any, work_dir: str) -> list:
    """Extract raw record objects from the database."""
    # Try get_all_tuning_records (common API)
    if hasattr(db, "get_all_tuning_records"):
        try:
            return list(db.get_all_tuning_records())
        except Exception:
            pass

    # Try iterating the db directly
    if hasattr(db, "__iter__"):
        try:
            return list(db)
        except Exception:
            pass

    # Try reading from JSON files in work_dir
    import json
    import glob as _glob
    result = []
    for f in sorted(_glob.glob(f"{work_dir}/**/*.json", recursive=True)):
        try:
            with open(f, "r") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        result.append(json.loads(line))
        except Exception:
            continue
    return result


def _extract_run_secs(rec: Any) -> float:
    """Get the measured runtime from a tuning record."""
    if isinstance(rec, dict):
        for key in ("run_secs", "time_cost", "latency"):
            val = rec.get(key)
            if val is not None:
                if isinstance(val, (list, tuple)):
                    return float(sum(float(v) for v in val) / len(val)) if val else float("inf")
                return float(val)
        return float("inf")

    for attr in ("run_secs", "time_cost"):
        val = getattr(rec, attr, None)
        if val is None:
            continue
        try:
            vals = list(val)
            if vals:
                return float(sum(float(v) for v in vals) / len(vals))
        except (TypeError, ValueError):
            pass
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    return float("inf")


def _extract_trace(rec: Any) -> str:
    """Get the schedule trace text from a tuning record."""
    if isinstance(rec, dict):
        return str(rec.get("trace", rec.get("schedule", "")))

    trace = getattr(rec, "trace", None)
    if trace is not None:
        # TVM Trace objects have as_python() for readable output
        for method in ["as_python", "show", "__str__"]:
            fn = getattr(trace, method, None)
            if fn is not None:
                try:
                    result = fn() if method != "__str__" else str(trace)
                    if result:
                        return str(result)
                except Exception:
                    continue
        return repr(trace)

    return str(rec)[:500]


def _extract_task_name(rec: Any, fallback_idx: int) -> str:
    """Get the task name from a tuning record."""
    if isinstance(rec, dict):
        return str(rec.get("task_name", rec.get("workload_key", f"task_{fallback_idx}")))
    for attr in ("task_name", "workload_key"):
        name = getattr(rec, attr, None)
        if name:
            return str(name)
    # Try to get name from workload's IRModule
    wl = getattr(rec, "workload", None)
    if wl is not None:
        wl_mod = getattr(wl, "mod", None)
        if wl_mod is not None:
            try:
                for gv in wl_mod.functions:
                    return gv.name_hint
            except Exception:
                pass
    return f"task_{fallback_idx}"


def _synthetic_tuning_records(
    mod: Any,
    target: Any,
) -> Tuple[List[dict], List[dict]]:
    """Generate educational synthetic tuning records when real tuning fails.

    This allows the demo to show the *structure* of tuning results even
    when MetaSchedule cannot run (no CUDA, broken API, etc.).
    """
    import random
    random.seed(42)

    op_names = []
    for gv, func in _safe_mod_functions(mod):
        if isinstance(func, tvm.tir.PrimFunc):
            name = gv.name_hint if hasattr(gv, "name_hint") else str(gv)
            op_names.append(name)

    if not op_names:
        op_names = ["conv2d_0", "conv2d_1", "dense_0"]

    hot_ops = op_names[:3]
    records: List[dict] = []
    convergence: List[dict] = []
    best_so_far = float("inf")

    schedule_templates = [
        "split(i, [4, 8]) -> reorder(i_0, j_0, i_1, j_1) -> bind(blockIdx.x, i_0) -> vectorize(j_1, 4)",
        "split(i, [2, 16]) -> reorder(i_0, j_0, i_1, j_1) -> bind(blockIdx.x, i_0) -> bind(threadIdx.x, i_1)",
        "split(i, [8, 4]) -> split(j, [8, 4]) -> reorder(i_0, j_0, i_1, j_1) -> cache_read(shared) -> bind(blockIdx.x, i_0)",
        "tile(i, j, 16, 16) -> bind(blockIdx.x, i_0) -> bind(threadIdx.x, j_0) -> unroll(i_1, 4)",
        "split(i, [1, 32]) -> bind(blockIdx.x, i_0) -> bind(threadIdx.x, i_1) -> vectorize(j, 4) -> cache_write(local)",
    ]

    for trial_idx in range(32):
        task_name = hot_ops[trial_idx % len(hot_ops)]
        base_ms = 0.08 + random.random() * 0.4
        improvement = max(0, 1.0 - (trial_idx / 32) * 0.6)
        noise = random.gauss(0, 0.02)
        run_ms = max(0.01, base_ms * improvement + noise)

        trace = random.choice(schedule_templates)
        best_so_far = min(best_so_far, run_ms)

        records.append({
            "candidate_id": trial_idx,
            "task_name": task_name,
            "trace_text": trace,
            "run_secs": run_ms / 1000.0,
            "run_ms": run_ms,
            "is_best": False,
            "_synthetic": True,
        })
        convergence.append({
            "trial_index": trial_idx,
            "best_latency_ms": best_so_far,
        })

    best_idx = min(range(len(records)), key=lambda j: records[j]["run_ms"])
    records[best_idx]["is_best"] = True

    return records, convergence


# ──────────────────────────────────────────────────────────────────────
# Stage 10 — Cost Model & Schedule Selection
# ──────────────────────────────────────────────────────────────────────

def select_best_candidate(
    tuning_records: List[dict],
    mod: Any,
) -> Tuple[Optional[dict], List[dict]]:
    """Analyse tuning records: pick the best, compute structural features.

    Paper mapping: Section 5.2 -- "ML-Based Cost Model."

    This function implements "Layer A" (direct from tuning data) and
    "Layer B" (educational structural features) as described in the plan.

    Returns
    -------
    best : dict or None   -- the winning candidate record
    features : list[dict] -- per-candidate structural features
    """
    if not tuning_records:
        return None, []

    valid = [r for r in tuning_records if r["run_ms"] < 1e6]
    if not valid:
        return None, []

    best = min(valid, key=lambda r: r["run_ms"])

    features = []
    for rec in valid:
        feat = _compute_trace_features(rec)
        feat["candidate_id"] = rec["candidate_id"]
        feat["task_name"] = rec["task_name"]
        feat["run_ms"] = rec["run_ms"]
        feat["is_best"] = rec.get("is_best", False)
        features.append(feat)

    features.sort(key=lambda f: f["run_ms"])

    log.info(
        "Best candidate: #%d (%s) at %.4f ms",
        best["candidate_id"], best["task_name"], best["run_ms"],
    )
    return best, features


def _compute_trace_features(rec: dict) -> dict:
    """Extract human-readable structural features from a schedule trace.

    These are "Layer B" educational features (plan Section Stage 10).
    """
    trace = rec.get("trace_text", "")
    lower_trace = trace.lower()

    return {
        "num_splits": lower_trace.count("split"),
        "num_reorders": lower_trace.count("reorder"),
        "has_cache_read": "cache_read" in lower_trace,
        "has_cache_write": "cache_write" in lower_trace,
        "num_thread_bindings": (
            lower_trace.count("bind(blockidx") +
            lower_trace.count("bind(threadidx") +
            lower_trace.count("blockidx") +
            lower_trace.count("threadidx")
        ),
        "has_vectorize": "vectorize" in lower_trace,
        "has_unroll": "unroll" in lower_trace,
        "has_shared_memory": "shared" in lower_trace,
        "trace_length": len(trace),
    }


def compute_tir_structural_features(mod: Any, op_name: str) -> dict:
    """Compute structural features from a TIR PrimFunc for cost-model education.

    Walks the TIR AST to count loops, thread bindings, shared-memory buffers,
    vectorized loops, unrolled loops, and estimate arithmetic intensity.

    Paper mapping: Section 5.2, Figure 13.
    """
    _require_tvm()
    try:
        func = _find_prim_func(mod, op_name)
    except KeyError:
        return {}

    features: dict = {
        "op_name": op_name,
        "num_loops": 0,
        "num_thread_bindings": 0,
        "num_shared_buffers": 0,
        "num_vectorized_loops": 0,
        "num_unrolled_loops": 0,
        "num_blocks": 0,
        "num_buffer_stores": 0,
        "total_loop_extent_product": 1,
    }

    _walk_tir_for_features(func.body, features)

    for param in func.params:
        buf = func.buffer_map.get(param)
        if buf is not None:
            scope = str(getattr(buf, "scope", ""))
            if "shared" in scope.lower():
                features["num_shared_buffers"] += 1

    if features["num_buffer_stores"] > 0 and features["total_loop_extent_product"] > 1:
        features["arithmetic_intensity_proxy"] = round(
            features["total_loop_extent_product"] / max(features["num_buffer_stores"], 1),
            2,
        )
    else:
        features["arithmetic_intensity_proxy"] = 0.0

    return features


def _walk_tir_for_features(node: Any, features: dict, _depth: int = 0) -> None:
    """Recursively walk TIR to compute structural features."""
    if _depth > 200:
        return
    node_type = type(node).__name__

    if node_type == "For":
        features["num_loops"] += 1
        extent = getattr(node, "extent", None)
        if extent is not None and hasattr(extent, "value"):
            features["total_loop_extent_product"] *= int(extent.value)
        thread = getattr(node, "thread_binding", None)
        if thread is not None:
            features["num_thread_bindings"] += 1
            tag = str(getattr(thread, "thread_tag", ""))
            if "threadIdx" in tag:
                pass  # already counted
        kind = str(getattr(node, "kind", ""))
        if "Vectorized" in kind:
            features["num_vectorized_loops"] += 1
        if "Unrolled" in kind:
            features["num_unrolled_loops"] += 1
        _walk_tir_for_features(node.body, features, _depth + 1)
        return

    if node_type == "Block":
        features["num_blocks"] += 1
        body = getattr(node, "body", None)
        if body is not None:
            _walk_tir_for_features(body, features, _depth + 1)
        return

    if node_type == "BufferStore":
        features["num_buffer_stores"] += 1

    if node_type == "BlockRealize":
        block = getattr(node, "block", None)
        if block is not None:
            _walk_tir_for_features(block, features, _depth + 1)
        return

    if node_type == "SeqStmt":
        for s in node:
            _walk_tir_for_features(s, features, _depth + 1)
        return

    for attr in ("body", "then_case", "else_case", "value"):
        child = getattr(node, attr, None)
        if child is not None:
            _walk_tir_for_features(child, features, _depth + 1)


# ──────────────────────────────────────────────────────────────────────
# Stage 11 — Build Final CUDA Module
# ──────────────────────────────────────────────────────────────────────

def build_tvm_module(
    mod: Any,
    params_np: Optional[List[np.ndarray]] = None,
    target_str: str = "cuda",
) -> Tuple[Any, str, str, bool]:
    """Compile the (optionally tuned) IRModule into a runnable artifact.

    If *params_np* is provided the parameters are bound as constants before
    building so the VM only needs the user input at call time.

    Returns
    -------
    lib : tvm runtime module
    target_used : str
    cuda_source : str   – generated CUDA source (best-effort, may be empty)
    params_bound : bool – True if params were successfully bound as constants
    """
    _require_tvm()

    target = _resolve_target(target_str)
    log.info("Building TVM module for target: %s", target)

    build_mod = mod
    params_bound = False
    if params_np is not None:
        build_mod, params_bound = _bind_params(mod, params_np)

    # Apply DLight default GPU scheduling (tiles loops + binds to GPU threads).
    # Without this, naive TIR will fail CUDA memory verification.
    if _dlight_available and "cuda" in str(target).lower():
        build_mod = _apply_dlight(build_mod, target)

    lib = tvm_relax.build(build_mod, target=target)

    cuda_src = _try_get_cuda_source(lib)
    return lib, str(target), cuda_src, params_bound


def _apply_dlight(mod: Any, target: Any) -> Any:
    """Apply DLight default GPU schedule rules to all TIR PrimFuncs."""
    try:
        gpu = _dlight.gpu
        rules = []
        for rule_cls in [gpu.Matmul, gpu.GEMV, gpu.Reduction,
                         gpu.GeneralReduction, gpu.Fallback]:
            try:
                rules.append(rule_cls())
            except Exception:
                pass
        if not rules:
            rules = [gpu.Fallback()]
        with target:
            mod = _dlight.ApplyDefaultSchedule(*rules)(mod)
        log.info("DLight applied %d GPU schedule rules", len(rules))
        return mod
    except Exception as exc:
        log.warning("DLight scheduling failed: %s — building without it", exc)
        return mod


def _resolve_target(target_str: str) -> Any:
    """Create a TVM Target, trying specific GPU names first.

    For CUDA targets, ensure ``max_threads_per_block`` is present
    (required by MetaSchedule).
    """
    try:
        t = tvm.target.Target(target_str)
    except Exception:
        t = tvm.target.Target("cuda" if "cuda" in target_str.lower() else "llvm")

    if "cuda" in t.kind.name:
        try:
            attrs = dict(t.attrs)
            if "max_threads_per_block" not in attrs:
                attrs["max_threads_per_block"] = attrs.get("max_num_threads", 1024)
            if "max_shared_memory_per_block" not in attrs:
                attrs["max_shared_memory_per_block"] = 49152
            t = tvm.target.Target(
                {
                    "kind": "cuda",
                    "arch": attrs.get("arch", "sm_86"),
                    "max_threads_per_block": attrs["max_threads_per_block"],
                    "max_shared_memory_per_block": attrs["max_shared_memory_per_block"],
                    "max_num_threads": attrs.get("max_num_threads", 1024),
                    "thread_warp_size": attrs.get("thread_warp_size", 32),
                }
            )
        except Exception as exc:
            log.debug("Could not enrich CUDA target: %s", exc)
    return t


def _bind_params(mod: Any, params_np: List[np.ndarray]) -> Tuple[Any, bool]:
    """Bind numpy parameters as constants in the IRModule.

    Tries ``tvm.relax.transform.BindParams`` first.  If unavailable,
    returns the module unchanged (params will be passed at runtime).

    Returns (module, True) if binding succeeded, (module, False) otherwise.
    """
    try:
        main_fn = mod["main"]
        _tvm_prevent_free.append(main_fn)
        param_vars = list(main_fn.params[1:])
        _tvm_prevent_free.append(param_vars)

        if len(param_vars) != len(params_np):
            log.warning(
                "Param count mismatch: function has %d params, got %d arrays. "
                "Skipping bind, will pass at runtime.",
                len(param_vars), len(params_np),
            )
            return mod, False

        param_dict = {}
        for var, arr in zip(param_vars, params_np):
            param_dict[var.name_hint] = _nd_array(arr, tvm.cpu(0))
        _tvm_prevent_free.append(param_dict)

        bound = tvm.relax.transform.BindParams("main", param_dict)(mod)
        _tvm_prevent_free.append(bound)
        log.info("Bound %d params as constants", len(param_dict))
        return bound, True
    except Exception as exc:
        log.warning("BindParams failed (%s), params will be passed at runtime", exc)
        return mod, False


def _try_get_cuda_source(lib: Any) -> str:
    """Best-effort extraction of generated CUDA source."""
    def _is_cuda(src: str) -> bool:
        return bool(src and ("__global__" in src or "__device__" in src or "threadIdx" in src))

    def _search_module(mod: Any) -> str:
        """Recursively search a module tree for CUDA source."""
        for getter in ["inspect_source", "get_source"]:
            fn = getattr(mod, getter, None)
            if fn is None:
                continue
            try:
                src = fn() if getter == "inspect_source" else fn("cuda")
                if _is_cuda(src):
                    return src
            except Exception:
                pass
            if getter == "get_source":
                for fmt in ["cu", ""]:
                    try:
                        src = fn(fmt)
                        if _is_cuda(src):
                            return src
                    except Exception:
                        pass
        for child in getattr(mod, "imports", getattr(mod, "imported_modules", [])):
            result = _search_module(child)
            if result:
                return result
        return ""

    # For VMExecutable, start from lib.mod
    root = getattr(lib, "mod", lib)
    return _search_module(root)


# ──────────────────────────────────────────────────────────────────────
# Stage 12 — Run TVM Inference & Comparison
# ──────────────────────────────────────────────────────────────────────

def run_tvm_inference(
    lib: Any,
    input_np: np.ndarray,
    categories: List[str],
    params_np: Optional[List[np.ndarray]] = None,
    n_runs: int = 100,
) -> Tuple[np.ndarray, List[dict], float]:
    """Run inference through the compiled TVM module.

    Parameters
    ----------
    lib : compiled TVM runtime module
    input_np : (1,3,224,224) float32
    categories : ImageNet labels
    params_np : model params (only needed if they were NOT bound at build time)
    n_runs : latency measurement iterations

    Returns
    -------
    logits : np.ndarray
    top5 : list of dicts
    median_ms : float
    """
    _require_tvm()

    try:
        dev = tvm.cuda(0) if tvm.cuda(0).exist else tvm.cpu(0)
    except Exception:
        dev = tvm.cpu(0)
    vm = tvm_relax.VirtualMachine(lib, dev)
    _tvm_prevent_free.append(vm)

    tvm_input = _nd_array(input_np.astype("float32"), dev)
    _tvm_prevent_free.append(tvm_input)

    if params_np is not None:
        tvm_params = [_nd_array(p.astype("float32"), dev) for p in params_np]
        _tvm_prevent_free.append(tvm_params)
        run_args = [tvm_input] + tvm_params
    else:
        run_args = [tvm_input]

    out = vm["main"](*run_args)
    _tvm_prevent_free.append(out)
    logits_np = out.numpy() if hasattr(out, "numpy") else np.array(out)
    top5 = top_k_predictions(logits_np, categories, k=5)

    sync = dev.sync if hasattr(dev, "sync") else None

    def _bench():
        vm["main"](*run_args)

    median_ms, _ = measure_latency(
        fn=_bench,
        warmup=10,
        repeat=n_runs,
        sync_fn=sync,
    )
    log.info("TVM inference: %s  (%.2f ms)", top5[0]["class"], median_ms)
    return logits_np, top5, median_ms


def compare_results(
    pytorch_logits: np.ndarray,
    tvm_logits: np.ndarray,
    pytorch_ms: float,
    tvm_ms: float,
) -> dict:
    """Produce a comparison summary between PyTorch and TVM outputs."""
    max_diff = float(np.max(np.abs(
        pytorch_logits.astype(np.float64) - tvm_logits.astype(np.float64)
    )))
    cos_sim = cosine_similarity(pytorch_logits, tvm_logits)
    speedup = pytorch_ms / tvm_ms if tvm_ms > 0 else float("inf")

    return {
        "max_abs_diff": max_diff,
        "cosine_similarity": cos_sim,
        "pytorch_ms": pytorch_ms,
        "tvm_ms": tvm_ms,
        "speedup": round(speedup, 2),
        "match": max_diff < 1e-2,
    }
