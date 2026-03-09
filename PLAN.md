# TVM Demo — Full Implementation Plan

> **Paper**: *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*
> (Chen et al., OSDI '18)
>
> **Goal**: An interactive Gradio web app that walks the viewer through every
> stage of TVM's compilation pipeline — from a PyTorch model to an optimized
> CUDA module — using live code, real compiler artifacts, and actual tuning
> data.  The demo runs on a local NVIDIA desktop and is accessed from any
> browser (including a MacBook) via a tunnel.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────┐
│  NVIDIA Desktop  (server)                       │
│                                                 │
│  Python  ·  PyTorch  ·  TVM  ·  CUDA            │
│  Gradio app  (or FastAPI + Gradio Blocks)        │
│                                                 │
│  ngrok / Cloudflare Tunnel  ──► public URL       │
└─────────────────────────────────────────────────┘
            │
            ▼  HTTPS
┌──────────────────────┐
│  MacBook (or any PC) │
│  browser only        │
└──────────────────────┘
```

### Why This Architecture

| Concern | Solution |
|---|---|
| Need NVIDIA GPU for CUDA target | Run everything on the NVIDIA desktop |
| Avoid Colab install/build headaches | Local install, full control |
| MacBook only has browser | Access via tunnel URL |
| Reproducibility | Pin TVM commit, lock deps |

### What Must Be True

1. The NVIDIA machine stays powered on and keeps the server process running.
2. The machine has internet access.
3. A tunnel (ngrok or Cloudflare Tunnel) is running, OR the router forwards a
   port.  With a tunnel, no manual router config is needed.

---

## 2. Platform & Deployment

### 2.1 Server Setup (NVIDIA Desktop)

| Item | Detail |
|---|---|
| OS | Windows or Linux (WSL2 is acceptable) |
| GPU | Any NVIDIA GPU with CUDA ≥ 11.x |
| Python | 3.10 or 3.11 (match TVM build) |
| CUDA toolkit | 11.8 or 12.x, matching the TVM build |
| Driver | ≥ 525 |

### 2.2 TVM Installation

Preferred order:

1. **Pre-built wheel** — `pip install apache-tvm-cu1xx` if a compatible wheel
   exists for the local Python + CUDA combination.
2. **Build from source** — clone `https://github.com/apache/tvm`, checkout a
   known-good tag (e.g. `v0.18.0` or latest release), build with
   `USE_CUDA=ON`, `USE_LLVM=ON`, `USE_RELAY=ON`.  Follow
   `docs.tvm.ai/install/from_source`.

After install, verify:

```python
import tvm
print(tvm.__version__)
print(tvm.cuda(0).exist)          # must be True
print(tvm.target.Target("cuda"))  # must parse
```

### 2.3 Tunnel / Exposure

Choose **one** of:

| Method | Command | Notes |
|---|---|---|
| **ngrok** (recommended for demo) | `ngrok http 7860` | Gives a public HTTPS URL. Free tier allows one tunnel. |
| **Cloudflare Tunnel** | `cloudflared tunnel --url http://localhost:7860` | No account needed for quick tunnels. |
| **Tailscale** | Install on both machines | Private network, no public URL, but zero config. |

The Gradio app will listen on `0.0.0.0:7860` by default.

### 2.4 Launch Sequence

```bash
# 1.  Activate the project venv
conda activate tvm_demo   # or: source .venv/bin/activate

# 2.  Start the Gradio app
python app.py

# 3.  In a second terminal, start the tunnel
ngrok http 7860
# Copy the Forwarding URL and open it on the MacBook.
```

---

## 3. Software Stack & Dependencies

### 3.1 Core Runtime

| Package | Purpose |
|---|---|
| `torch` (≥ 2.2) | Model source, baseline inference, FX tracing |
| `torchvision` | Pre-trained ResNet-18, image transforms |
| `numpy` | Tensor manipulation |
| `Pillow` | Image I/O |

### 3.2 TVM / Compiler

| Package / Module | Purpose |
|---|---|
| `tvm` | Root runtime |
| `tvm.relax` | Modern high-level IR, frontend import, Relax VM |
| `tvm.relax.frontend.torch` | `from_exported_program()` — PyTorch → Relax |
| `tvm.ir` | `IRModule`, pass infrastructure |
| `tvm.tir` | Tensor-level IR (`PrimFunc`, loops, buffers) |
| `tvm.topi` | Operator inventory (TE/TOPI compute declarations) |
| `tvm.te` | Tensor expression language (for the microscope panel) |
| `tvm.meta_schedule` | Tuning: task extraction, search, database |
| `tvm.transform` / `tvm.relax.transform` | Pass objects |
| `tvm.contrib.graph_executor` | Legacy graph runtime (fallback) |

### 3.3 Visualization

| Package | Purpose |
|---|---|
| `gradio` (≥ 4.x) | Web UI |
| `graphviz` | Render FX graph and Relax call graph as DOT/SVG |
| `networkx` | Graph construction helper for call-graph extraction |
| `matplotlib` | Latency bar charts, roofline-style diagrams |
| `pandas` | Tables for operator lists, tuning records |

### 3.4 Optional Helpers

| Package / Module | Purpose |
|---|---|
| `torch.fx` | `symbolic_trace` for human-readable node graph |
| `torch.export` | `export()` for normalized ATen graph |
| `json` | Serialize/deserialize tuning records |
| `sqlite3` | Read MetaSchedule tuning DB |
| `difflib` | Side-by-side IR diffs |
| `textwrap` | Pretty-print long IR strings |
| `inspect` | Source introspection for TIR functions |
| `time` / `statistics` | Latency measurement |

### 3.5 `requirements.txt`

```
torch>=2.2
torchvision>=0.17
numpy
Pillow
gradio>=4.0
graphviz
networkx
matplotlib
pandas
pyngrok          # optional, for programmatic ngrok launch
```

TVM is installed separately (wheel or source build).

---

## 4. Project Structure

```
TVM_demo/
├── PLAN.md                   ← this file
├── requirements.txt
├── app.py                    ← Gradio entry-point (Section E+F)
├── backend/
│   ├── __init__.py
│   ├── pipeline.py           ← stage functions (Section B)
│   ├── state.py              ← DemoState dataclass (Section D)
│   └── helpers.py            ← small utilities (format, diff, timing)
├── viz/
│   ├── __init__.py
│   ├── graph_render.py       ← FX/Relax graph → SVG (Section C)
│   ├── ir_display.py         ← IR text, TIR AST, syntax highlight
│   ├── schedule_display.py   ← schedule trace → readable cards
│   ├── feature_table.py      ← per-candidate structural features
│   └── charts.py             ← latency bar chart, comparison plots
├── assets/
│   └── imagenet_classes.json ← human-readable class labels
└── tests/
    └── test_pipeline.py      ← smoke test: headless run of all stages
```

---

## 5. Demo Stages — Complete Specification

Each stage below specifies:
- **Paper mapping** — which section of the OSDI '18 paper it illustrates.
- **What to compute** — the exact function calls.
- **What to display** — the Gradio components.
- **Why it matters** — the educational purpose.

All stage outputs are cached in a single `DemoState` object so the user can
navigate back and forth without recomputation.

---

### Stage 0 — Load Model

**Paper mapping**: Figure 2, top ("Frameworks" box).

**Compute**:
```python
import torchvision.models as models
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.eval().cuda()
```
Build the standard ImageNet transform (resize 256 → center-crop 224 →
normalize with ImageNet mean/std).

**Display**:
| Component | Content |
|---|---|
| Model summary text | `str(model)` layer tree |
| Parameter count | total params, trainable params |
| Input spec | `(1, 3, 224, 224)` float32 CUDA |

**Why**: This is the starting point before any compiler work.  The viewer sees
the model as a PyTorch object.

---

### Stage 1 — PyTorch Baseline Inference

**Paper mapping**: Motivation (Section 1) — "current frameworks rely on
vendor-specific operator libraries."

**Compute**:
```python
from PIL import Image
img = transform(Image.open(uploaded_path)).unsqueeze(0).cuda()
with torch.no_grad():
    logits = model(img)
probs = torch.softmax(logits, dim=1)
top5 = torch.topk(probs, 5)

# Latency
times = []
for _ in range(100):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model(img)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append((t1 - t0) * 1000)
baseline_ms = statistics.median(times)
```

**Display**:
| Component | Content |
|---|---|
| Uploaded image thumbnail | the input image |
| Top-5 class probabilities | bar chart or table |
| Median latency | e.g. "2.3 ms (100 runs, PyTorch eager, CUDA)" |

**Why**: Establishes the correctness and performance baseline before TVM
touches the model.

---

### Stage 2 — PyTorch Computation Graph

**Paper mapping**: Section 3, opening — "Computational graphs are a common way
to represent programs in DL frameworks."

This stage uses PyTorch's own graph capture to show the concept of a
computation graph *before* importing into TVM.

**Compute** (two complementary views):

1. **`torch.fx.symbolic_trace`** — produces a human-readable `Graph` with
   named nodes.
   ```python
   import torch.fx
   traced = torch.fx.symbolic_trace(model.cpu())
   fx_graph = traced.graph
   fx_code  = traced.code       # readable Python code
   ```
2. **`torch.export.export`** — produces an `ExportedProgram` with ATen ops,
   closer to compiler-ready form.
   ```python
   from torch.export import export
   example_input = (torch.randn(1, 3, 224, 224),)
   exported = export(model.cpu(), example_input)
   ```

**Display**:
| Component | Content |
|---|---|
| Node table | columns: name, op type (`call_function` / `call_module`), target, input edges |
| Graphviz SVG | FX nodes laid out as a DAG (use `graph_render.py`) |
| Generated code | the `traced.code` string |

**Why**: The viewer sees that a neural network *is* a dataflow graph of tensor
operations.  This is the conceptual input to TVM (Section 3, Figure 3).

---

### Stage 3 — Import into TVM (Relax IR)

**Paper mapping**: Figure 2, "Computational Graph" → "High Level Graph
Rewriting".  Modern TVM uses Relax as the high-level IR that replaces the old
Relay graph.

**Compute**:
```python
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

exported = export(model.cpu().eval(), (torch.randn(1, 3, 224, 224),))
mod = from_exported_program(exported, keep_params_as_input=True)
# mod is a tvm.ir.IRModule
```

Save `mod.script()` as `ir_snapshot["imported"]`.

**Display**:
| Component | Content |
|---|---|
| Raw IR text | `mod.script()`, truncated with expandable detail |
| Function list | names of all `relax.Function` and `tir.PrimFunc` entries in the module |
| Op/call-site count | number of `R.call_tir`, `R.call_dps_packed`, etc. |
| Relax call graph (optional) | Graphviz rendering of the main function's call structure |

**Why**: This is the *first real TVM artifact*.  The paper's IRModule is "the
central data structure" — it bundles the graph, tensor programs, and external
calls (Section 3 / modern docs).

---

### Stage 4 — TVM Graph-Level Passes (One by One)

**Paper mapping**: Section 3 — "Optimizing Computational Graphs."  Covers
operator fusion (Section 3, "Operator Fusion"), constant folding, data layout
transformation.

This is where the demo becomes excellent: instead of an opaque "compile"
button, each pass is applied individually and the IR snapshot is kept.

**Passes to surface (in order)**:

| # | Pass | Paper concept | What it does |
|---|---|---|---|
| 1 | `LegalizeOps()` | Maps Relax high-level ops to concrete TIR implementations | Bridges graph ops → tensor programs |
| 2 | `AnnotateTIROpPattern()` | Tags ops by fusion category (injective, reduction, complex-out-fusable, opaque) | Prepares for fusion (paper Section 3 categories) |
| 3 | `FuseOps()` | Operator fusion | Paper Section 3 "Operator Fusion" — fuses element-wise, broadcast, reduction ops into one kernel |
| 4 | `FuseTIR()` | Merges fused Relax groups into single TIR PrimFuncs | Creates the actual fused kernels |
| 5 | `DeadCodeElimination()` | Removes unreachable IR nodes | Standard compiler pass |
| 6 | (Optional) Layout transforms, `ConvertToDataflow()` | Data layout optimization | Paper Section 3 "Data Layout Transformation" |

**Implementation**:
```python
import tvm.relax.transform as T

passes = [
    ("LegalizeOps",           T.LegalizeOps()),
    ("AnnotateTIROpPattern",  T.AnnotateTIROpPattern()),
    ("FuseOps",               T.FuseOps()),
    ("FuseTIR",               T.FuseTIR()),
    ("DeadCodeElimination",   T.DeadCodeElimination()),
]
snapshots = {"imported": mod.script()}
current = mod
for name, p in passes:
    current = p(current)
    snapshots[name] = current.script()
```

**Display**:
| Component | Content |
|---|---|
| Pass selector dropdown | pick any pass to see its before/after |
| Before / After IR panels | side-by-side or diff view (use `difflib`) |
| Delta summary | "FuseOps created N fused functions; call_tir count went from X → Y" |
| Pass execution order list | vertical flow showing pass names |

**Why**: This transforms TVM from a black box into a visible sequence of
compiler rewrites.  It directly demonstrates Figure 2 "High Level Graph
Rewriting" and Section 3's fusion rules.

---

### Stage 5 — Operator Extraction (Lowered TIR Functions)

**Paper mapping**: Transition from Section 3 → Section 4.  After graph-level
fusion, the resulting operators are the units that Section 4 will optimize.

**Compute**:
```python
operators = []
for gv, func in current.functions.items():
    if isinstance(func, tvm.tir.PrimFunc):
        info = {
            "name": gv.name_hint,
            "params": [(p.name, p.dtype, list(p.shape)) for p in func.params
                       if hasattr(p, 'shape')],
            "num_blocks": count_tir_blocks(func),
            "tir_source": func.script(),
        }
        operators.append(info)
```

**Display**:
| Component | Content |
|---|---|
| Operator table | columns: name, input shapes, output shapes, op kind (conv, matmul, elemwise, etc. inferred from name), estimated FLOPs |
| Click-to-expand TIR source | for each function |

**Why**: Shows how graph-level ops become concrete tensor programs — the
bridge to Section 4 "Generating Tensor Operations."

---

### Stage 6 — TensorIR / AST Visualization

**Paper mapping**: Section 4 — the "low-level loop program" and its AST.
Figure 13 shows the "Loop AST" fed into the cost model.

For each extracted `PrimFunc`, the TIR is a loop-based program with explicit
buffers, thread bindings, and memory scopes.  This is the representation TVM
schedules and lowers to CUDA.

**Compute** (for one selected PrimFunc):
```python
prim = current[selected_op_name]
tir_source = prim.script()
ast_summary = extract_tir_ast_summary(prim)
# ast_summary includes:
#   - list of blocks (name, iter_vars, reads, writes)
#   - loop nest (var, extent, annotation, thread_binding)
#   - buffer declarations (name, shape, dtype, scope)
```

`extract_tir_ast_summary` is custom code that walks `prim.body` recursively
and collects `tvm.tir.For`, `tvm.tir.Block`, `tvm.tir.BufferStore`, etc.

**Display**:
| Component | Content |
|---|---|
| Operator selector | dropdown of all extracted PrimFuncs |
| Raw TIR code | syntax-highlighted source |
| AST tree view | collapsible tree: blocks → loops → buffer ops |
| Loop nest summary table | columns: loop var, extent, annotation, thread binding, memory scope |

**Why**: The viewer sees the low-level program TVM will schedule and lower to
CUDA.  This is the concrete representation referenced in Figure 13 and
Section 5.2 where the cost model extracts features from the loop AST.

---

### Stage 7 — Tensor Expression Definition (Microscope Panel)

**Paper mapping**: Section 4.1 — "Tensor Expression and Schedule Space."
The paper's core abstraction: declare *what* to compute without specifying
*how* (the compute/schedule separation from Halide).

For an imported full model, TVM does not expose a single pretty TE lambda.
The modern Relax import path goes directly to TIR.  So we do two things:

1. **Real model**: point back to the TIR blocks from Stage 6 and explain
   "these *are* the lowered tensor programs for the imported model."
2. **Microscope operator**: build a standalone `conv2d` from TE/TOPI, show
   the compute declaration, lower it to TIR, and show how it corresponds to
   a real conv-like TIR extracted from the model.

**Compute (microscope conv2d)**:
```python
import tvm
from tvm import te, topi

N, CI, H, W = 1, 64, 56, 56
CO, KH, KW = 64, 3, 3
data = te.placeholder((N, CI, H, W), name="data", dtype="float32")
weight = te.placeholder((CO, CI, KH, KW), name="weight", dtype="float32")

out = topi.nn.conv2d(data, weight, strides=1, padding=1, dilation=1)

s = te.create_schedule(out.op)
lowered = tvm.lower(s, [data, weight, out], simple_mode=True)
```

**Display**:
| Component | Content |
|---|---|
| TE compute definition text | the `topi.nn.conv2d` call and its lambda body |
| Default (naive) TIR | `lowered.script()` — the un-scheduled loop nest |
| Side-by-side with real model TIR | align the microscope conv2d TIR with the corresponding extracted PrimFunc from Stage 6 |
| Explanation text | "TVM separates the *what* (tensor expression) from the *how* (schedule). The next stages show how schedules transform this loop nest." |

**Why**: This satisfies the "tensor expression definition" requirement with
live code.  It maps directly to Section 4.1 and Figure 5 (left column: compute
definition → schedule → low-level code).

---

### Stage 8 — Schedule Space & Task Extraction

**Paper mapping**: Section 5.1 — "Schedule Space Specification" and
Section 5 intro.  MetaSchedule is the modern realization of the paper's
"automated optimizer" (Figure 11).

**Compute**:
```python
from tvm import meta_schedule as ms

target = tvm.target.Target("nvidia/nvidia-<gpu-model>")

# Extract tunable tasks from the module
database = ms.database.JSONDatabase(work_dir="./tuning_logs")
tasks, task_weights = ms.relay_integration.extracted_tasks_to_tune_contexts(
    # For Relax, the API may differ; use:
    #   ms.relax_integration.extract_tasks(mod, target)
    # Adjust to available API.
)
```

Alternatively (Relax path):
```python
tasks = ms.relax_integration.extract_tasks(current, target)
```

For each task, record:
- task name / workload key
- target
- associated TIR body
- number of schedule knobs (if exposed).

**Display**:
| Component | Content |
|---|---|
| Task table | columns: #, task name, workload key, target, associated TIR snippet |
| Total tasks count | e.g. "12 tunable tasks extracted" |
| Selected task detail | full TIR of one selected task (pre-tuning) |
| Explanation text | "Each task is a TIR PrimFunc that MetaSchedule will explore. The schedule space is the set of all valid transformations (tile sizes, loop orders, thread bindings, cache strategies) for this program." |

**Why**: This directly shows Section 5.1's schedule space and Figure 11's
"Schedule Space" box.

---

### Stage 9 — Candidate Schedule Generation (Tuning)

**Paper mapping**: Section 5.3 — "Schedule Exploration."  The explorer
proposes configurations, measures them on hardware, and feeds results back.

**Compute**:

Run a short tuning session on 1–3 selected hot tasks with a small trial
budget.  This keeps the demo fast (minutes, not hours).

```python
tuning_tasks = tasks[:3]   # pick the hottest conv-like tasks

with ms.database.JSONDatabase(work_dir="./tuning_logs") as db:
    ms.tune_tir(
        mod=current,
        target=target,
        config=ms.TuneConfig(
            max_trials_global=64,      # small budget for demo
            num_trials_per_iter=16,
            strategy="evolutionary",   # maps to simulated annealing in paper
        ),
        work_dir="./tuning_logs",
        database=db,
        task_name=...,  # or tune all extracted tasks
    )
```

For each candidate tried, capture:
- schedule trace (sequence of schedule instructions: split, reorder, bind,
  cache_read, cache_write, vectorize, unroll)
- transformed loop nest summary
- measured runtime on GPU.

**Display**:
| Component | Content |
|---|---|
| Candidate cards | for each measured candidate: trace text, loop nest summary, measured latency |
| Schedule instruction breakdown | e.g. "split(i, 32) → reorder(i_outer, j_outer, i_inner) → bind(blockIdx.x) → vectorize(j_inner, 4)" |
| Convergence chart | trials vs. best latency found so far (mirrors Figure 12) |
| Best candidate highlight | border / badge on the winner |

**Why**: This is one of the most important educational stages.  It makes
visible the core automated search loop from Section 5.3 and Figure 12.

---

### Stage 10 — Cost Model & Schedule Selection

**Paper mapping**: Section 5.2 — "ML-Based Cost Model."  The paper uses
gradient tree boosting (XGBoost) on features extracted from the loop AST
(Figure 13).

This stage is split into two honest layers:

#### Layer A — What TVM Actually Exposes

Show:
- Tuning record table: candidate ID, schedule trace, measured runtime.
- Which candidate was selected as best.
- Total tuning wall-clock time.

These come directly from the tuning database:
```python
records = db.get_all_tuning_records()
for r in records:
    trace  = r.trace          # schedule instructions
    run_ms = r.run_secs       # measured latency
    # ...
```

#### Layer B — Educational Structural Features (Custom Code)

Compute human-readable per-candidate features from TIR:

| Feature | How to compute |
|---|---|
| Number of loops | count `tvm.tir.For` nodes |
| Thread bindings | count `threadIdx.*` / `blockIdx.*` annotations |
| Shared-memory buffers | count buffers with `"shared"` scope |
| Vectorized loops | count loops annotated `vectorize` |
| Unrolled loops | count loops annotated `unroll` |
| Arithmetic intensity proxy | FLOPs ÷ bytes-moved estimate |

Then display with a clear disclaimer:

> "These are human-readable structural features of each candidate schedule.
> TVM's internal tuner uses its own cost model (gradient tree boosting on loop
> program features — paper Section 5.2, Figure 13) to rank candidates between
> measurements.  This panel approximates the *kind* of information the cost
> model considers."

**Display**:
| Component | Content |
|---|---|
| Tuning records table | candidate ID, trace summary, measured ms, selected? |
| Feature comparison table | rows = candidates, columns = structural features |
| Selection outcome | "Candidate #7 selected — 0.031 ms, schedule: split+reorder+bind+vectorize" |
| Cost model explanation card | text block referencing paper Section 5.2 and Figure 13 |

**Why**: Section 5.2 is the paper's most novel contribution (ML-based cost
model).  This stage is honest about what's surfaced while still being
educational.

---

### Stage 11 — Build Final CUDA Module

**Paper mapping**: Figure 2, bottom — "Deployable Module."  The optimized
IRModule is compiled to target-specific code (LLVM IR, CUDA, Metal, etc.).

**Compute**:
```python
# Apply best tuning records to the module
with database:
    tuned_mod = ms.relax_integration.compile_relax(
        current, target=target, database=database
    )
# OR, using the lower-level path:
#   tuned_mod = tvm.relax.transform.MetaScheduleApplyDatabase()(current)
#   lib = tvm.relax.build(tuned_mod, target=target)

lib = tvm.relax.build(tuned_mod, target=target)
```

Inspect the built artifact:
```python
# List compiled function names
func_names = [f.name for f in lib.imported_modules[0].get_function(...)]

# Optionally extract generated CUDA source
cuda_src = lib.imported_modules[0].get_source("cuda")
```

**Display**:
| Component | Content |
|---|---|
| Target string | e.g. `"nvidia/geforce-rtx-3080"` |
| Compiled function names | list |
| Final IR snapshot | the post-tuning, fully-lowered IRModule |
| Generated CUDA source (optional) | raw `.cu` code for one kernel, if the backend exposes it |

**Why**: This closes the compiler story — from model to deployable artifact.

---

### Stage 12 — TVM Inference & Comparison

**Paper mapping**: Section 6 — "Evaluation."  The paper compares TVM against
framework baselines (Figure 14).

**Compute**:
```python
dev = tvm.cuda(0)
vm = tvm.relax.VirtualMachine(lib, dev)

# Prepare TVM input
tvm_input = tvm.nd.array(img_np, dev)
tvm_params = ...  # load params into TVM nd arrays

# Run inference
vm.set_input("main", tvm_input, *tvm_params)
vm.invoke_stateful("main")
tvm_output = vm.get_outputs("main")[0].numpy()

# Correctness
max_abs_diff = np.max(np.abs(pytorch_logits - tvm_output))
cosine_sim   = cosine_similarity(pytorch_logits.flatten(), tvm_output.flatten())

# Latency (mirror the PyTorch measurement)
tvm_times = []
for _ in range(100):
    dev.sync()
    t0 = time.perf_counter()
    vm.invoke_stateful("main")
    dev.sync()
    t1 = time.perf_counter()
    tvm_times.append((t1 - t0) * 1000)
tvm_ms = statistics.median(tvm_times)
```

**Display**:
| Component | Content |
|---|---|
| Side-by-side predictions | PyTorch top-5 vs. TVM top-5 |
| Correctness metrics | max abs diff, cosine similarity |
| Latency bar chart | PyTorch eager vs. TVM optimized (matplotlib) |
| Speedup callout | e.g. "TVM is 1.8× faster" |

**Why**: Proves correctness (same predictions) and demonstrates the
performance benefit of TVM's optimizations, mirroring the paper's evaluation
methodology.

---

### Stage 13 — Pipeline Timeline (Full Story)

**Paper mapping**: Figure 2 (the full system overview diagram).

A single summary tab that lays out the entire journey as a vertical flow
diagram:

```
PyTorch Model (Stage 0)
    │
    ▼
PyTorch Eager Inference — baseline (Stage 1)
    │
    ▼
Computation Graph Capture — FX / export (Stage 2)
    │
    ▼
Relax IR Import (Stage 3)
    │
    ▼
Graph-Level Passes — LegalizeOps → FuseOps → … (Stage 4)
    │  Paper Section 3
    ▼
Extracted TIR Operators (Stage 5)
    │
    ▼
TensorIR / AST (Stage 6)
    │  Paper Section 4
    ▼
Tensor Expression — microscope conv2d (Stage 7)
    │  Paper Section 4.1
    ▼
MetaSchedule Task Extraction (Stage 8)
    │  Paper Section 5.1
    ▼
Schedule Exploration & Tuning (Stage 9)
    │  Paper Section 5.3
    ▼
Cost Model & Selection (Stage 10)
    │  Paper Section 5.2
    ▼
Build CUDA Module (Stage 11)
    │  Paper Figure 2 bottom
    ▼
TVM Inference — optimized (Stage 12)
    │  Paper Section 6
    ▼
 ✓  Comparison: correctness + speedup
```

Each node in the timeline links back to the corresponding tab and shows a
one-line summary of the key artifact produced (e.g. "IRModule with 14
functions" or "Best schedule: 0.031 ms").

---

## 6. DemoState Object (Section D)

A single `@dataclass` holding all artifacts so tabs can cross-reference
without recomputation.

```python
@dataclass
class DemoState:
    # Stage 0
    model: torch.nn.Module = None
    model_summary: str = ""

    # Stage 1
    input_image: Image.Image = None
    input_tensor: torch.Tensor = None
    pytorch_logits: np.ndarray = None
    pytorch_top5: list = None
    pytorch_latency_ms: float = 0.0

    # Stage 2
    fx_graph: torch.fx.Graph = None
    fx_code: str = ""
    exported_program: Any = None

    # Stage 3
    imported_mod: tvm.ir.IRModule = None
    ir_snapshots: dict = field(default_factory=dict)  # pass_name → IR text

    # Stage 4
    pass_deltas: dict = field(default_factory=dict)    # pass_name → delta summary
    current_mod: tvm.ir.IRModule = None

    # Stage 5
    operators: list = field(default_factory=list)      # list of operator dicts

    # Stage 6
    selected_tir_source: str = ""
    tir_ast_summary: dict = None

    # Stage 7
    te_microscope_source: str = ""
    te_lowered_tir: str = ""

    # Stage 8
    tasks: list = field(default_factory=list)

    # Stage 9
    tuning_records: list = field(default_factory=list)
    convergence_data: list = field(default_factory=list)

    # Stage 10
    candidate_features: pd.DataFrame = None
    best_candidate: dict = None

    # Stage 11
    compiled_lib: Any = None
    cuda_source: str = ""
    final_ir: str = ""

    # Stage 12
    tvm_logits: np.ndarray = None
    tvm_top5: list = None
    tvm_latency_ms: float = 0.0
    max_abs_diff: float = 0.0
    cosine_sim: float = 0.0
```

---

## 7. Visualization Utilities (Section C)

### 7.1 `viz/graph_render.py`

- `fx_graph_to_svg(fx_graph) → str`
  Walk `fx_graph.nodes`, build a `graphviz.Digraph`, return SVG string.
  Node color by op type: `call_module` = blue, `call_function` = green,
  `placeholder` = gray, `output` = orange.

- `relax_callgraph_to_svg(mod) → str`
  For the Relax main function, extract call targets (`R.call_tir`,
  `R.call_dps_packed`), build a DAG, render.

### 7.2 `viz/ir_display.py`

- `highlight_ir(ir_text, max_lines=200) → str`
  Truncate, add line numbers, apply Gradio `gr.Code` language hints.

- `ir_diff(before, after) → str`
  Use `difflib.unified_diff`, return as Gradio-compatible diff text.

- `extract_tir_ast_summary(prim_func) → dict`
  Recursively walk `prim_func.body`, collect:
  - blocks: name, iter_vars, reads, writes
  - loops: var, extent, kind, thread_binding
  - buffer_stores: count, scopes
  Return nested dict.

### 7.3 `viz/schedule_display.py`

- `trace_to_readable(trace) → list[str]`
  Parse MetaSchedule trace object into human-readable instruction list:
  `["split(i, factors=[4, 8])", "reorder(i_0, j_0, i_1, j_1)", ...]`

- `trace_to_card_html(trace, runtime_ms) → str`
  Render a styled HTML card for one candidate.

### 7.4 `viz/feature_table.py`

- `compute_structural_features(prim_func) → dict`
  Walk TIR, count loops, thread bindings, shared buffers, vectorized loops,
  unrolled loops, estimate arithmetic intensity.

- `build_feature_dataframe(records) → pd.DataFrame`
  One row per candidate, columns = features + measured latency.

### 7.5 `viz/charts.py`

- `latency_comparison_chart(pytorch_ms, tvm_ms) → matplotlib.Figure`
  Grouped bar chart: PyTorch vs TVM.

- `convergence_chart(trial_latencies) → matplotlib.Figure`
  X = trial number, Y = best latency so far.  Mirrors Figure 12.

---

## 8. Gradio UI Layout (Section E)

Use `gr.Blocks` with `gr.Tabs`.  Each tab corresponds to one stage (or a
logical grouping).

```
Tab 1:  "Task & Input"           → Stages 0–1
Tab 2:  "PyTorch Graph"          → Stage 2
Tab 3:  "TVM IR Import"          → Stage 3
Tab 4:  "TVM Passes"             → Stage 4
Tab 5:  "Extracted Operators"    → Stage 5
Tab 6:  "TensorIR / AST"        → Stage 6
Tab 7:  "Tensor Expression"      → Stage 7  (microscope panel)
Tab 8:  "Schedule Search"        → Stages 8–9
Tab 9:  "Cost Model & Selection" → Stage 10
Tab 10: "Build & Results"        → Stages 11–12
Tab 11: "Pipeline Timeline"      → Stage 13
```

**Controls at the top of the page**:
- Model selector: `ResNet-18` (default), optionally `MobileNetV2`.
- Image upload widget.
- "Run All Stages" button — runs the full pipeline sequentially.
- "Run Next Stage" button — runs only the next uncomputed stage.
- Stage progress indicator — shows which stages are complete.

**Design principles**:
- Every tab is populated by *actual computation*, not mock data.
- All artifacts are cached in `DemoState`.
- The user can click "Run All" once, or advance tab by tab.
- Long operations (tuning) show a progress bar.

---

## 9. Implementation Order

Build in four incremental passes.  Each pass produces a working artifact.

### Pass 1 — Core Pipeline (headless)

Goal: A plain Python script that runs the full pipeline end-to-end without any
UI.

1. Install TVM and verify CUDA target.
2. Implement `backend/pipeline.py`:
   - `load_model()` → Stage 0
   - `run_pytorch_inference()` → Stage 1
   - `import_to_tvm()` → Stage 3
   - `apply_passes_stepwise()` → Stage 4
   - `build_and_run_tvm()` → Stages 11–12
3. Implement `backend/state.py` (`DemoState`).
4. Write `tests/test_pipeline.py` — runs all stages, asserts correctness
   (max abs diff < 1e-3), prints latencies.

**Exit criterion**: The test script prints PyTorch and TVM predictions +
latencies, and they match.

### Pass 2 — Graph & Operator Visualization

5. Implement `backend/pipeline.py`:
   - `trace_pytorch_graph()` → Stage 2
   - `extract_operators()` → Stage 5
   - `get_tir_ast()` → Stage 6
   - `build_te_microscope()` → Stage 7
6. Implement `viz/graph_render.py`, `viz/ir_display.py`.
7. Extend `tests/test_pipeline.py` — asserts FX graph has > 0 nodes, operator
   table is non-empty, TIR source is non-empty.

**Exit criterion**: All visualization artifacts are generated and serializable.

### Pass 3 — MetaSchedule Tuning

8. Implement `backend/pipeline.py`:
   - `extract_tuning_tasks()` → Stage 8
   - `run_tuning()` → Stage 9
   - `read_tuning_records()` → Stage 10
9. Implement `viz/schedule_display.py`, `viz/feature_table.py`,
   `viz/charts.py`.
10. Extend test — runs tuning with `max_trials_global=16`, asserts ≥ 1 record.

**Exit criterion**: Tuning records are collected, feature table is populated,
convergence chart renders.

### Pass 4 — Gradio App

11. Build `app.py` — wire all backend functions and viz utilities into Gradio
    Blocks/Tabs.
12. Add progress indicators, error handling, loading spinners.
13. Test end-to-end in browser.
14. Set up tunnel (ngrok) and verify access from another machine.

**Exit criterion**: The full demo is accessible from a browser on the MacBook,
all tabs are interactive, and the pipeline completes within a reasonable time
(< 10 min including tuning).

---

## 10. Performance & Practicality Constraints

| Constraint | Mitigation |
|---|---|
| Tuning is slow | Tune only 1–3 hot tasks; cap `max_trials_global` at 64–128 |
| Full model TIR is huge | Truncate IR display, use expandable sections |
| MetaSchedule API may differ between TVM versions | Gate optional features behind `hasattr` / `try-except`; print version info at startup |
| Some internal cost model details are not surfaced | Use Layer A / Layer B approach (Stage 10); label approximations clearly |
| Generated CUDA source may not be exposed | Wrap in `try-except`; show "CUDA source not available for this backend" fallback |

---

## 11. Risk Mitigation

### 11.1 TVM Installation Failure

- **Fallback 1**: Use a Docker container with TVM + CUDA pre-built.
- **Fallback 2**: Use Colab with GPU runtime as a last resort (the original
  plan).
- **Fallback 3**: If only the CPU target works, the demo still functions — just
  set `target = tvm.target.Target("llvm")` and note the difference.

### 11.2 API Drift

TVM's Relax APIs are actively evolving.  Protect against drift:

```python
# Example: version-gated import
try:
    from tvm.relax.frontend.torch import from_exported_program
except ImportError:
    from tvm.relax.frontend.torch import from_fx
    # adapt arguments
```

Print a version banner at app startup:

```
TVM version:    0.18.0
PyTorch version: 2.3.0
CUDA available:  True (RTX 3080)
Target:          nvidia/geforce-rtx-3080
```

### 11.3 MetaSchedule Not Available / Broken

If MetaSchedule is non-functional:
- Skip Stages 8–10.
- Build the module without tuning (still applies graph passes).
- Add a banner: "MetaSchedule tuning is unavailable in this TVM build. Stages
  8–10 are skipped."

### 11.4 Tunnel Drops

Use `pyngrok` to programmatically restart ngrok, or run Gradio with
`share=True` (Gradio's built-in tunnel via HuggingFace infrastructure):

```python
demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
```

---

## 12. Paper Section ↔ Demo Stage Mapping

| Paper Section | Paper Concept | Demo Stage |
|---|---|---|
| §1 Introduction | Motivation, hardware diversity | Stage 1 (baseline shows "before TVM") |
| §3 Optimizing Computational Graphs | Operator fusion, data layout | **Stage 4** (pass-by-pass) |
| §3 Figure 3 | Computational graph | **Stage 2** (FX graph) + **Stage 3** (Relax IR) |
| §3 Figure 4 | Fusion speedup | Stage 4 delta summary |
| §4 Generating Tensor Operations | Tensor expressions, schedule primitives | **Stage 7** (TE microscope) |
| §4.1 Tensor Expression & Schedule Space | Compute/schedule separation | **Stage 7** + **Stage 8** |
| §4 Figure 5 | Schedule transformations on matrix multiply | Stage 9 (candidate schedule traces) |
| §4 Figure 6 | Schedule lowering & code gen process | Stage 6 (TIR) → Stage 11 (CUDA build) |
| §4.2 Nested Parallelism | Cooperative memory fetching, thread groups | Visible in candidate trace thread bindings (Stage 9) |
| §4.3 Tensorization | Hardware intrinsic mapping | (Mentioned in explanation text, Stage 7) |
| §5 Automating Optimization | ML-based cost model + search | **Stages 8–10** |
| §5.1 Schedule Space Specification | Template / knobs | **Stage 8** |
| §5.2 ML-Based Cost Model | XGBoost on loop AST features | **Stage 10** (Layer A + B) |
| §5.2 Figure 13 | Feature extraction → cost prediction | Stage 10 feature table |
| §5.3 Schedule Exploration | Simulated annealing, batch measurement | **Stage 9** convergence chart |
| §6 Evaluation | End-to-end perf comparison | **Stage 12** latency chart |
| §6 Figure 14 | GPU end-to-end: TVM vs frameworks | Stage 12 bar chart |
| Figure 2 | Full system overview | **Stage 13** pipeline timeline |

---

## 13. What Is Fully Automatic vs. Custom Glue

### Fully automatic from live TVM/PyTorch code

- Model loading and inference
- FX / export graph capture
- Relax IR import (`from_exported_program`)
- IRModule pass snapshots (each pass returns a new module)
- TIR function extraction
- MetaSchedule task extraction
- MetaSchedule tuning (records, traces)
- Module build and TVM inference

### Custom glue code to write

- FX graph → Graphviz SVG renderer
- IR diff viewer (before/after text diff)
- TIR AST walker / summarizer
- TE microscope panel (standalone TOPI conv2d example)
- Schedule trace parser (trace object → readable instruction list)
- Structural feature extractor (count loops, bindings, scopes)
- Feature comparison table
- Latency / convergence charts (matplotlib)
- Gradio UI wiring and state management

The compiler artifacts are real.  The custom code is the "microscope" that
makes them visible.

---

## 14. Honest Boundaries

The demo will *genuinely demonstrate*:

1. How a PyTorch image classifier starts as a framework model.
2. How the computation graph is captured.
3. How TVM imports it into an IRModule.
4. How graph-level passes (fusion, legalization) transform it step by step.
5. How lowered tensor programs (TIR PrimFuncs) appear after fusion.
6. How TIR represents the low-level loop/buffer structure.
7. How tensor expressions define computation declaratively (via TOPI microscope).
8. How MetaSchedule extracts tunable tasks.
9. How candidate schedules are generated and measured.
10. How the best schedule is selected.
11. How the final CUDA module is built.
12. How TVM's optimized inference compares to PyTorch in correctness and speed.

The demo will *clearly label* (not overclaim):

- "The exact internal cost-model features and predictor math are not fully
  surfaced in the public high-level APIs.  The structural features shown are
  an educational approximation."
- "MetaSchedule replaces the paper's original template-based schedule space
  with a more automated approach.  The underlying search principles (Section
  5.3) remain the same."
- "The paper's XGBoost / TreeRNN cost model (Section 5.2, Figure 13) operates
  internally during tuning.  This demo shows the *inputs* (loop structure) and
  *outputs* (measured rankings) of that process."
- **MetaSchedule in mlc-ai-nightly builds**: The `tvm.meta_schedule` top-level
  module is absent, but the full MetaSchedule infrastructure lives under
  `tvm.s_tir.meta_schedule`.  The pipeline imports from both paths, so real
  MetaSchedule task extraction (Stage 8) and real tuning (Stage 9) work
  out-of-the-box.  DLight default GPU scheduling (`tvm.s_tir.dlight`) is
  applied before CUDA compilation to provide thread-bound kernels.  All stages
  (8–12) use genuine TVM APIs — no synthetic fallbacks.

---

*End of plan.*
