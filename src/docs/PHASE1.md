# Phase 1 -- Core Pipeline (Headless)

> **Status note (2026-03-11):** Historical phase log. Some specifics may differ from current behavior after later fixes/refactors; use README.md + code as source of truth.

> **Goal**: Build the foundational backend that runs the full TVM compilation
> pipeline end-to-end without any UI -- from loading a PyTorch model through
> TVM import, graph-level passes, CUDA module build, and inference comparison.

---

## Files Created

| File | Size | Purpose |
|------|------|---------|
| `requirements.txt` | root | Core Python dependencies (torch, torchvision, numpy, Pillow, gradio, graphviz, networkx, matplotlib, pandas). TVM installed separately. |
| `src/__init__.py` | empty | Makes `src/` a Python package |
| `src/backend/__init__.py` | 478 B | Package docstring with lazy imports (avoids pulling torch/tvm at package-import time) |
| `src/backend/state.py` | 6.4 KB | `DemoState` dataclass + `StageStatus` enum |
| `src/backend/helpers.py` | 8.5 KB | Environment checks, latency measurement, numeric helpers, image I/O |
| `src/backend/pipeline.py` | 20 KB (Phase 1 portion) | Stage functions: 0, 1, 3, 4, 11, 12. Imports TVM, DLight, MetaSchedule; uses `_nd_array` (from_dlpack fallback); adds `~/tvm_env/bin` to PATH for nvcc/gcc. |
| `src/viz/__init__.py` | placeholder | Docstring only, modules added in Phase 2 |
| `src/tests/__init__.py` | empty | Makes `tests/` a package |
| `src/tests/test_pipeline.py` | 10 KB (Phase 1 portion) | Headless smoke test covering all Phase 1 stages |

---

## Architecture Decisions

### Stateless pipeline functions + central DemoState

Every pipeline function is **stateless**: it takes explicit inputs and returns
explicit outputs.  The caller (`app.py` or a test) stores results in a single
`DemoState` dataclass.  This separation means:

- Functions are independently testable.
- The Gradio UI (Phase 4) can call any function and cache results in state.
- No hidden global mutation.

### TVM guard pattern

TVM may not be installed on every dev machine.  `pipeline.py` handles this
with a top-level try/except:

```python
_TVM_AVAILABLE = False
try:
    import tvm
    from tvm import relax as tvm_relax
    _TVM_AVAILABLE = True
except ImportError:
    tvm = None
    tvm_relax = None

def _require_tvm() -> None:
    if not _TVM_AVAILABLE:
        raise RuntimeError("TVM is not installed ...")
```

- PyTorch-only stages (0, 1) work without TVM.
- TVM stages (3, 4, 11, 12) call `_require_tvm()` at entry.
- The module is importable for linting/structure checks even without TVM.

### Dual-path TVM import with fallback

The import function tries two paths in order:

1. **Primary**: `torch.export.export()` then `from_exported_program()` (modern
   Relax frontend).
2. **Fallback**: `torch.fx.symbolic_trace()` then `from_fx()` (older but
   stable).

This handles TVM API drift across versions.

### Parameter management

When importing with `keep_params_as_input=True`, model parameters become
additional inputs to the Relax function.  Phase 1 handles this with:

1. **Extraction**: `_extract_params_matching_tvm()` maps TVM param names
   (e.g. `p_conv1_weight`) back to PyTorch `state_dict` keys (e.g.
   `conv1.weight`) via name-based matching (prefix `p_`, underscores↔dots).
   Falls back to `_extract_params_from_state_dict()` using
   `ExportedProgram.graph_signature` when available.

2. **Binding at build time**: `_bind_params()` attempts to use
   `tvm.relax.transform.BindParams("main", param_dict)` to bake parameters
   as constants in the IRModule before `relax.build()`.  This keeps the VM
   call simple (only the user input is passed).

3. **Fallback**: If binding fails, `build_tvm_module()` returns
   `params_bound=False` and the test/app passes params at runtime.

### NDArray creation and PATH setup

- **NDArray helper**: `tvm.nd.array` is absent in mlc-ai builds.  The pipeline
  uses `_nd_array()` which calls `tvm.nd.array` when available, otherwise
  creates tensors via `tvm.runtime.from_dlpack(torch.from_numpy(...))`.

- **nvcc/gcc**: Compilers are installed via conda into `~/tvm_env/bin`.  The
  pipeline auto-adds this directory to `PATH` at import time so CUDA build
  succeeds without manual PATH setup.

- **Segfault fix**: A `tp_dealloc` replacement (via ctypes) neutralises TVM
  FFI destructors to prevent use-after-free segfaults in mlc-ai-nightly.

---

## Detailed Module Descriptions

### `src/backend/state.py`

#### `StageStatus` (Enum)

Tracks the lifecycle of each demo stage:

| Value | Meaning |
|-------|---------|
| `PENDING` | Not yet started |
| `RUNNING` | Currently executing |
| `DONE` | Completed successfully |
| `FAILED` | Threw an error |
| `SKIPPED` | Intentionally skipped (e.g. missing dependency) |

#### `DemoState` (dataclass)

A single mutable container with one field-group per stage.  All 14 stages
(0-13) have a status entry in `stage_status`.

**Stage tracking fields:**
- `stage_status: Dict[str, StageStatus]` -- initialized with `stage_0` through
  `stage_13`, all `PENDING`.
- `stage_logs: Dict[str, str]` -- free-form log text per stage.

**Stage 0 fields** (Load Model):
- `model` -- the `torch.nn.Module` in eval mode.
- `model_name` -- `"resnet18"` or `"mobilenet_v2"`.
- `model_summary` -- human-readable layer tree with param counts.
- `param_count` -- total parameter count (int).
- `transform` -- torchvision image transform.
- `categories` -- 1000 ImageNet class labels.

**Stage 1 fields** (PyTorch Baseline):
- `input_image` -- PIL.Image of the user-uploaded photo.
- `input_tensor` -- `torch.Tensor` of shape `(1, 3, 224, 224)`.
- `input_np` -- numpy copy of the same tensor (for TVM).
- `pytorch_logits` -- `np.ndarray` of shape `(1, 1000)`.
- `pytorch_top5` -- list of `{"class": str, "prob": float, "index": int}`.
- `pytorch_latency_ms` -- median inference latency in milliseconds.

**Stage 3 fields** (TVM Import):
- `imported_mod` -- the `tvm.ir.IRModule` immediately after import.
- `ir_snapshots` -- `Dict[str, str]` keyed by stage name, holding
  `mod.script()` text.
- `model_params_np` -- `List[np.ndarray]`, parameters in function-arg order.

**Stage 4 fields** (TVM Passes):
- `current_mod` -- IRModule after all passes have been applied.
- `pass_order` -- `List[str]` of pass names in execution order.
- `pass_deltas` -- `Dict[str, dict]` mapping pass names to delta summaries
  (function count before/after, TIR function count before/after, elapsed
  seconds).

**Stages 11-12 fields** (Build + Inference):
- `compiled_lib` -- the `tvm.runtime.Module`.
- `target_str` -- e.g. `"cuda"`.
- `cuda_source` -- generated `.cu` code (best-effort extraction).
- `tvm_logits`, `tvm_top5`, `tvm_latency_ms` -- TVM inference outputs.
- `max_abs_diff`, `cosine_sim` -- correctness metrics vs PyTorch.

**Helper methods:**
- `mark(stage_id, status, log="")` -- update a stage's status and append log
  text.
- `is_done(stage_id)` -- check if a stage completed.
- `reset()` -- re-initialize all fields to defaults (preserves `model_name`).

---

### `src/backend/helpers.py`

#### Environment introspection

**`get_device_info() -> Dict[str, Any]`**

Probes the runtime for: Python version, PyTorch version, CUDA availability,
GPU name, CUDA version, TVM availability, TVM version, and whether TVM
sees a CUDA device.  Never raises.

**`format_device_banner(info) -> str`**

Pretty-prints the info dict as a boxed multi-line banner:

```
======================================
  TVM Demo -- Environment
======================================
  Python:       3.10.11
  PyTorch:      2.3.0
  CUDA:         Yes
  GPU:          NVIDIA GeForce RTX 3080
  CUDA version: 12.1
  TVM:          0.18.0
  TVM -> CUDA:  Yes
======================================
```

#### Latency measurement

**`measure_latency(fn, warmup=10, repeat=100, sync_fn=None) -> (median_ms, all_times_ms)`**

Generic benchmark harness.  Parameters:
- `fn` -- zero-arg callable to measure.
- `warmup` -- warm-up iterations (not timed).
- `repeat` -- measured iterations.
- `sync_fn` -- called before each timing boundary (e.g.
  `torch.cuda.synchronize`).

Returns the median and the full list of per-iteration millisecond timings.
Uses `time.perf_counter` for sub-millisecond resolution.

#### Numeric helpers

**`cosine_similarity(a, b) -> float`**

Flattens both arrays to float64, computes `dot(a,b) / (||a|| * ||b||)`.
Returns 0.0 if norm product is near-zero.

**`top_k_predictions(logits, categories, k=5) -> List[dict]`**

Applies softmax to raw logits, sorts descending, returns the top-k entries
as `[{"class": "tabby cat", "prob": 0.89, "index": 281}, ...]`.

**`_softmax(x) -> np.ndarray`** -- numerically stable softmax
(`exp(x - max(x))`).

#### Image helpers

**`load_image(source) -> PIL.Image`** -- accepts a file path string or an
existing PIL Image.  Always returns RGB.

**`prepare_input_tensor(image, transform) -> torch.Tensor`** -- applies the
torchvision transform and unsqueezes to batch dimension if needed.

**`download_sample_image(url=...) -> PIL.Image`** -- downloads a Wikimedia cat
photo for testing.  Falls back cleanly if network is unavailable.

#### IR / text helpers

**`count_ir_lines(ir_text) -> int`** -- line count.

**`truncate_text(text, max_lines=200) -> str`** -- truncates long IR texts
with a note about omitted lines.

**`model_summary(model) -> (summary_str, param_count)`** -- lists each
top-level child module with its class name and parameter count.

---

### `src/backend/pipeline.py` (Phase 1 functions)

#### `check_environment() -> Dict[str, Any]`

Delegates to `helpers.get_device_info()`.  Always succeeds.

#### `load_model(model_name="resnet18") -> (model, transform, categories, summary, param_count)`

**Paper mapping**: Figure 2 top ("Frameworks" box).

- Loads a pretrained classifier from `torchvision.models`.
- Supported: `resnet18`, `mobilenet_v2`.
- Returns the model in eval mode on CPU, the ImageNet transform, 1000 class
  labels, a summary string, and total param count.

#### `prepare_input(image, transform) -> (tensor, numpy_copy)`

Converts a file path or PIL Image into a `(1, 3, 224, 224)` float32 tensor
and its numpy equivalent (for TVM).

#### `run_pytorch_inference(model, input_tensor, categories, n_runs=100, use_cuda=True) -> (logits, top5, median_ms)`

**Paper mapping**: Section 1 motivation ("before TVM" baseline).

- Moves model + input to the appropriate device.
- Runs inference once to get logits, then `top_k_predictions` for top-5.
- Benchmarks with `measure_latency` (warmup=10, repeat=n_runs) using
  `torch.cuda.synchronize` as the sync barrier on CUDA.

#### `import_to_tvm(model, example_input) -> (mod, params_np, ir_text)`

**Paper mapping**: Figure 2, "Computational Graph" -> "High Level Graph
Rewriting".

- Deep-copies the model to CPU.
- Primary path: `torch.export.export()` then `from_exported_program()` with
  `keep_params_as_input=True`.
- Fallback path: `torch.fx.symbolic_trace()` then `from_fx()`.
- Extracts parameters in correct order via `graph_signature` introspection.
- Returns the `tvm.ir.IRModule`, parameter numpy arrays, and the printed IR
  text.

**Internal helpers:**
- `_import_via_export()` -- primary path implementation.
- `_import_via_fx()` -- fallback path.
- `_extract_params_from_state_dict()` -- parameter ordering from
  `ExportedProgram.graph_signature`.
- `_verify_param_names()` -- checks all lifted names exist in `state_dict`.

#### `apply_passes_stepwise(mod) -> (current_mod, snapshots, pass_order, deltas)`

**Paper mapping**: Section 3 -- "Optimizing Computational Graphs."

Applies TVM Relax passes one at a time, snapshotting the IR after each:

| # | Pass | Paper Concept |
|---|------|---------------|
| 1 | `LegalizeOps` | Maps high-level Relax ops to concrete TIR |
| 2 | `AnnotateTIROpPattern` | Tags ops by fusion category (injective, reduction, complex-out-fusable) |
| 3 | `FuseOps` | Operator fusion (paper Section 3 "Operator Fusion") |
| 4 | `FuseTIR` | Merges fused groups into single TIR PrimFuncs |
| 5 | `DeadCodeElimination` | Removes unreachable IR nodes |

Each pass is individually try/excepted.  If a pass fails or is missing from
the TVM build, it is skipped and logged.

The `deltas` dict records for each pass: function count before/after, TIR
function count before/after, IR line count, and elapsed seconds.

**Internal helpers:**
- `_get_pass_sequence()` -- builds the ordered list of (name, factory) tuples.
  Uses lazy factories so import errors are deferred.
- `_try_append()` -- validates that a pass factory works before adding it.
- `_count_tir_funcs()` -- counts `tvm.tir.PrimFunc` entries in a module.

#### `build_tvm_module(mod, params_np=None, target_str="cuda") -> (lib, target_used, cuda_source, params_bound)`

**Paper mapping**: Figure 2 bottom -- "Deployable Module."

- Resolves the target string (tries specific GPU names, falls back to generic
  `"cuda"`, then `"llvm"`).
- If `params_np` is provided, attempts to bind them as constants via
  `BindParams`.  Returns `params_bound=True/False` to indicate success.
- Calls `tvm.relax.build(mod, target=target)`.
- Extracts generated CUDA source by navigating `lib.mod.imports` (or
  `imported_modules`) recursively, calling `inspect_source()` or `get_source()`
  on each module until CUDA code is found.

#### `run_tvm_inference(lib, input_np, categories, params_np=None, n_runs=100) -> (logits, top5, median_ms)`

**Paper mapping**: Section 6 -- "Evaluation."

- Creates a `VirtualMachine` from the compiled library.
- Prepares TVM NDArrays via `_nd_array()` (from_dlpack when `tvm.nd.array` is
  absent).
- Runs inference, collects logits, computes top-5 predictions.
- Benchmarks with `measure_latency` using `dev.sync` as barrier.

#### `compare_results(pytorch_logits, tvm_logits, pytorch_ms, tvm_ms) -> dict`

Computes:
- `max_abs_diff` -- maximum absolute difference between logit vectors.
- `cosine_similarity` -- cosine similarity of flattened logits.
- `speedup` -- `pytorch_ms / tvm_ms`.
- `match` -- True if max_abs_diff < 0.01.

---

### `src/tests/test_pipeline.py` (Phase 1 portion)

A runnable script that exercises the full Phase 1 pipeline:

```
python -m src.tests.test_pipeline          # full run (TVM + CUDA)
python -m src.tests.test_pipeline --cpu    # PyTorch-only
python -m src.tests.test_pipeline --model mobilenet_v2
```

**Test flow:**
1. `test_environment` -- prints version banner.
2. `test_load_model` -- loads the model, prints summary.
3. `test_pytorch_inference` -- downloads sample image, runs inference, prints
   top-5 and latency.
4. `test_import_tvm` -- imports into TVM, prints function/param counts.
5. `test_passes` -- applies passes, prints before/after deltas.
6. `test_build` -- compiles CUDA module, prints target and binding status.
7. `test_tvm_inference` -- runs TVM inference, prints comparison metrics.
8. `assert_correctness` -- verifies max_abs_diff < 0.05, cosine > 0.99,
   top-1 class agreement.

Each test function writes its results into a shared `DemoState` instance.
The `--cpu` flag skips all TVM stages and only runs PyTorch stages.

---

## Data Flow Diagram

```
load_model()
  |
  v
prepare_input(image)
  |
  v
run_pytorch_inference()  -->  logits, top5, latency   (baseline)
  |
  v
import_to_tvm()  -->  IRModule, params_np, ir_text
  |
  v
apply_passes_stepwise()  -->  transformed IRModule, snapshots, deltas
  |
  v
build_tvm_module()  -->  compiled lib, cuda_source
  |
  v
run_tvm_inference()  -->  logits, top5, latency
  |
  v
compare_results()  -->  max_abs_diff, cosine_sim, speedup
```

---

## Exit Criterion

The test script prints PyTorch and TVM predictions + latencies, they match
(max absolute difference < 0.05, cosine similarity > 0.99, top-1 class
agrees), and all assertions pass.
