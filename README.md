# TVM Demo — End-to-End Optimizing Compiler for Deep Learning

An **interactive Gradio web app** that walks through every stage of [Apache TVM](https://tvm.apache.org/)'s compilation pipeline: from a **PyTorch image classifier** to an **optimized CUDA module**, with live compiler artifacts, real tuning data, and direct mappings to the TVM OSDI '18 paper.

**Paper**: *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning* (Chen et al., OSDI '18)

---

## What This Demo Does

The demo runs on a machine with an **NVIDIA GPU** (or CPU fallback) and is accessed from any browser. It:

1. **Loads a pretrained model** (ResNet-18 or MobileNetV2) and runs **PyTorch baseline inference** to establish correctness and latency.
2. **Captures the computation graph** via PyTorch FX and `torch.export`.
3. **Imports the model into TVM** as a **Relax IRModule** (TVM’s high-level graph IR).
4. **Applies graph-level passes** (LegalizeOps, AnnotateTIROpPattern, FuseOps, FuseTIR, DeadCodeElimination) and shows **before/after IR** for each pass.
5. **Extracts lowered TIR operators** (PrimFuncs) and lets you inspect **TensorIR source** and **loop AST** for any operator.
6. **Demonstrates Tensor Expressions** with a standalone conv2d “microscope” (compute vs. schedule separation).
7. **Extracts tuning tasks** with MetaSchedule and runs **automated schedule search** (candidate schedules, convergence chart).
8. **Selects the best candidate** and explains the **cost model** (structural features, paper Section 5.2).
9. **Builds the final CUDA module** (with DLight default GPU scheduling when MetaSchedule is unavailable).
10. **Runs TVM inference** and **compares** predictions and latency to PyTorch.

All artifacts are **real**: IR snapshots, TIR, tuning records, and generated code come from TVM and PyTorch; the app is the “microscope” that makes them visible.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Server (NVIDIA desktop or WSL2)                                 │
│  Python · PyTorch · TVM · CUDA                                    │
│  Gradio app (app.py) on 0.0.0.0:7860                             │
│  Optional: ngrok / Cloudflare Tunnel → public URL                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼  HTTP
┌─────────────────────────────────────────────────────────────────┐
│  Client (any device with a browser)                               │
│  Tabs 1–11: run stages, view IR, graphs, tuning, comparison      │
└─────────────────────────────────────────────────────────────────┘
```

- **Backend** (`src/backend/pipeline.py`): Stateless stage functions (load model, trace, import, passes, extract ops, TIR, TE microscope, tuning, build, inference).
- **State** (`src/backend/state.py`): Single `DemoState` object holding all artifacts so tabs can reuse results without recomputation.
- **Viz** (`src/viz/`): Graph rendering (FX/Relax), IR display/diff, schedule cards, feature tables, latency/convergence charts.
- **App** (`app.py`): Gradio Blocks with 11 tabs, progress badges, “Run All” and per-tab “Run Stage” buttons.

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python** | 3.10 or 3.11 (match your TVM build) |
| **PyTorch** | ≥ 2.2 (with CUDA 12.x recommended for GPU) |
| **TVM** | Apache TVM or mlc-ai-nightly; built with CUDA if you want GPU tuning and CUDA build |
| **CUDA** | 11.x or 12.x and matching driver (e.g. ≥ 525) if using GPU |
| **OS** | Windows, Linux, or WSL2 on Windows |

TVM is **not** in `requirements.txt`; it must be installed separately (see below).

---

## Installation

### 1. Clone and enter the project

```bash
cd /path/to/TVM-demo
```

### 2. Create a virtual environment and install Python dependencies

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# or:  venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

For **CUDA 12.8** PyTorch (as noted in `requirements.txt`):

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

For **CPU-only** PyTorch, install without the extra index (PyPI default).

### 3. Install TVM

TVM is not in `requirements.txt`. You can install it in the project venv or in a separate environment (e.g. `~/tvm_env` in WSL); the app works with either. If you use a separate env, run the app with that Python (see *Running from Windows (WSL one-liner)* below).

Choose one:

- **Pre-built wheel** (if available for your Python + CUDA):
  ```bash
  pip install apache-tvm-cu118   # or apache-tvm-cu121, etc.
  ```
- **From source**: Clone [apache/tvm](https://github.com/apache/tvm), check out a release tag (e.g. `v0.18.0`), build with `USE_CUDA=ON`, `USE_LLVM=ON`. See [TVM install from source](https://tvm.apache.org/docs/install/from_source.html).
- **mlc-ai-nightly** (alternative with Relax + MetaSchedule under `tvm.s_tir`):
  ```bash
  pip install mlc-ai-nightly -f https://mlc.ai/wheels
  ```

Verify TVM and CUDA:

```python
import tvm
print(tvm.__version__)
print(tvm.cuda(0).exist)   # True if CUDA target works
```

---

## Running the App

### Local

```bash
python app.py
```

Then open **http://localhost:7860** in your browser.

### WSL2 (from Windows)

**Option A — One-liner from Windows (PowerShell or CMD)**  
If you use a separate Python environment where TVM is installed (e.g. `~/tvm_env` in WSL), you can launch the app and a Gradio share link in one command:

```bash
wsl -d Ubuntu -- bash -c "cd /path/to/TVM-demo && ~/tvm_env/bin/python app.py --share"
```

Replace `/path/to/TVM-demo` with your project path inside WSL (e.g. `/mnt/c/Users/<YourUser>/Documents/TVM-demo`). Gradio will print a temporary public URL you can open in any browser. The pipeline also adds `~/tvm_env/bin` to `PATH` when present (for `nvcc`/`gcc` used by TVM).

**Option B — From a WSL2 terminal**  
From inside Ubuntu (or another WSL distro):

```bash
cd /path/to/TVM-demo
source venv/bin/activate   # or: source ~/tvm_env/bin/activate
pip install -r requirements.txt   # if not already done
python app.py
```

On Windows, open **http://localhost:7860**. If that fails, in WSL run `hostname -I` and use **http://<WSL_IP>:7860**.

### Optional: share link or tunnel

- **Gradio share link** (temporary public URL): run with `--share` (e.g. `python app.py --share` or the WSL one-liner above).
- **Custom host/port** (e.g. for Cloudflare or ngrok):
  ```bash
  python app.py --host 0.0.0.0 --port 7860
  ```
  Then run your tunnel (e.g. `ngrok http 7860`) and open the tunnel URL in the browser.

---

## UI Overview: Tabs and Stages

The app has **11 tabs**. Each tab corresponds to one or more **pipeline stages** (0–13). Run **“Run All Stages”** once to fill every tab, or use the **per-tab buttons** to run stages step by step. Progress badges at the top show which stages are done, running, or failed.

| Tab | Stages | What it shows | Paper reference |
|-----|--------|----------------|------------------|
| **1. Task & Input** | 0–1 | Load model (ResNet-18 / MobileNetV2), device banner, optional image upload, PyTorch inference (top-5, median latency) | Fig. 2 top, §1 |
| **2. PyTorch Graph** | 2 | FX graph (node table + Graphviz SVG), generated code; `torch.export` path when available | §3, Fig. 3 |
| **3. TVM IR Import** | 3 | Relax IRModule after import (`mod.script()`), function list, Relax call graph (SVG) | Fig. 2, §3 |
| **4. TVM Passes** | 4 | Pass-by-pass application (LegalizeOps → … → DeadCodeElimination), delta table, before/after IR diff viewer | §3 (fusion, legalization) |
| **5. Extracted Operators** | 5 | Table of TIR operators (name, shapes, op kind, block count); expandable TIR source | §3 → §4 |
| **6. TensorIR / AST** | 6 | Per-PrimFunc TIR source, block/loop/buffer AST tree, loop table (extent, bindings) | §4, Fig. 13 |
| **7. Tensor Expression** | 7 | Standalone conv2d TE compute, naive lowered TIR, and short explanation of compute/schedule separation | §4.1, Fig. 5 |
| **8. Schedule Search** | 8–9 | Extracted tuning tasks, MetaSchedule tuning (or synthetic fallback), candidate cards, convergence chart, task-weight pie | §5.1, §5.3, Fig. 12 |
| **9. Cost Model** | 10 | Best candidate banner, structural features table, cost-model explanation (paper §5.2, Fig. 13) | §5.2, Fig. 13 |
| **10. Build & Results** | 11–12 | Target string, built module, TVM top-5, correctness (max diff, cosine sim), latency bar chart (PyTorch vs TVM), optional CUDA source | Fig. 2 bottom, §6, Fig. 14 |
| **11. Pipeline Timeline** | 13 | Full pipeline as a vertical timeline with status and paper refs per stage | Fig. 2 (overview) |

---

## Project Structure

```
TVM-demo/
├── README.md                 # This file
├── PLAN.md                   # Full implementation plan and paper mapping
├── requirements.txt          # Python deps (PyTorch, Gradio, graphviz, etc.); TVM separate
├── app.py                    # Gradio entry point (tabs, buttons, state wiring)
├── _find_nvcc.py             # Helper to locate nvcc for TVM build
├── _tune_probe.py            # Tuning / MetaSchedule probe script
├── src/
│   ├── backend/
│   │   ├── pipeline.py       # All stage functions (load, trace, import, passes, ops, TIR, TE, tuning, build, inference)
│   │   ├── state.py          # DemoState dataclass and stage status
│   │   └── helpers.py        # Device info, latency measurement, image/transform, cosine similarity
│   ├── viz/
│   │   ├── graph_render.py   # FX graph and Relax call graph → SVG
│   │   ├── ir_display.py     # IR formatting, diff, TIR AST tree/loop tables
│   │   ├── schedule_display.py  # Schedule trace → readable cards, best-candidate banner
│   │   ├── feature_table.py  # Structural features, cost-model explanation HTML
│   │   └── charts.py        # Latency comparison, convergence chart, task-weight pie
│   ├── docs/
│   │   ├── PHASE1.md … PHASE4.md  # Implementation phases (core pipeline → viz → tuning → Gradio)
│   └── tests/
│       └── test_pipeline.py  # Headless smoke test: run full pipeline, check correctness/latency
├── tuning_logs/              # MetaSchedule tuning DB (optional; can be .gitignored)
└── .gitignore
```

---

## Pipeline Stages (Summary)

- **Stage 0**: Load pretrained ResNet-18 or MobileNetV2, ImageNet transform, model summary and param count.
- **Stage 1**: Prepare input (upload or sample image), run PyTorch inference, top-5 and median latency (CUDA or CPU).
- **Stage 2**: `torch.fx.symbolic_trace` and optionally `torch.export.export`; FX graph + node table + generated code.
- **Stage 3**: `tvm.relax.frontend.torch.from_exported_program` (or FX fallback); IRModule, params, Relax call graph.
- **Stage 4**: Apply LegalizeOps, AnnotateTIROpPattern, FuseOps, FuseTIR, DeadCodeElimination; store IR snapshots and deltas.
- **Stage 5**: Collect all TIR PrimFuncs from the module; name, params, shapes, op kind, block count.
- **Stage 6**: For a chosen PrimFunc: TIR source and AST summary (blocks, loops, buffers).
- **Stage 7**: Build a small TE/TOPI conv2d, show compute declaration and naive lowered TIR (microscope).
- **Stage 8**: MetaSchedule task extraction (or manual extraction); list of tunable tasks and target.
- **Stage 9**: Run tuning (or synthetic records if tuning unavailable); candidate cards and convergence data.
- **Stage 10**: Select best candidate by measured latency; structural features table; cost-model explanation.
- **Stage 11**: Bind params, apply DLight if needed, build Relax module for CUDA (or LLVM); optionally capture generated CUDA source.
- **Stage 12**: Run TVM inference, compare logits to PyTorch (max abs diff, cosine similarity), compare latency.
- **Stage 13**: Timeline view of all stages and paper references.

---

## Testing

Headless smoke test (no browser). Run from the **project root**:

```bash
python -m src.tests.test_pipeline
```

Use `--cpu` to run only PyTorch-only stages on CPU (no TVM/CUDA). The full run executes the entire pipeline (model load → inference → trace → import → passes → operators → TIR → TE → tuning → build → TVM inference → comparison) and checks correctness (e.g. max abs diff) and that latencies are produced. Tuning trial count is kept small so the test stays relatively fast.

---

## Tuning and Performance

- **Tuning trials**: The “Tuning Trials” slider (default 32, range 4–128) sets the MetaSchedule trial budget for stages 8–9. Higher values can improve performance but take longer.
- **Tuning logs**: MetaSchedule may write to `tuning_logs/` (database and workload JSON). This directory can be recreated; you can add `tuning_logs/` to `.gitignore` if you don’t want to commit it.
- **Fallbacks**: If MetaSchedule is unavailable (e.g. no `tvm.meta_schedule` or `tvm.s_tir.meta_schedule`), the app can use synthetic tuning records and still build and run with DLight default GPU scheduling so stages 8–12 remain functional.

---

## Paper Section ↔ Demo Mapping

| Paper | Concept | Demo |
|-------|---------|------|
| §1 | Motivation, baseline frameworks | Stage 1 (PyTorch inference) |
| §3 | Computational graph, fusion, legalization | Stages 2–4 (FX, Relax import, passes) |
| §3 → §4 | From graph to tensor programs | Stages 5–6 (operators, TIR/AST) |
| §4.1 | Tensor expressions, schedule space | Stage 7 (TE microscope) |
| §5.1 | Schedule space, tasks | Stage 8 (task extraction) |
| §5.2 | ML-based cost model, features | Stage 10 (selection, feature table) |
| §5.3 | Schedule exploration | Stage 9 (tuning, convergence) |
| §6, Fig. 14 | Evaluation, TVM vs frameworks | Stage 12 (comparison, latency chart) |
| Fig. 2 | End-to-end system | Stages 0–13, Tab 11 (timeline) |

---

## License

See [LICENSE](LICENSE) in the repository.
