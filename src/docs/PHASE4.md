# Phase 4 -- Gradio App

> **Goal**: Wire every backend pipeline function and visualization utility into
> an interactive Gradio web UI with 11 tabs, top-level controls, progress
> tracking, and per-stage error handling.  This is the final integration pass
> that makes the entire TVM demo accessible from a browser.

---

## New / Modified Files

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `app.py` | **new** | ~560 lines | Main Gradio Blocks application -- 11 tabs, model selector, image upload, tuning slider, "Run All" button, pipeline timeline |
| `src/backend/state.py` | **modified** | +2 fields | Added `_tasks_raw` and `_tuning_target` fields to `DemoState` for MetaSchedule inter-stage data flow |

---

## Paper Mapping (UI ↔ Paper Sections)

| Tab | Stages | Paper Reference | What It Shows |
|-----|--------|-----------------|---------------|
| 1. Task & Input | 0-1 | Figure 2 (top), Section 1 | Load a pretrained model and run PyTorch eager inference as the baseline |
| 2. Computation Graph | 2 | Section 3, Figure 3 | Capture the computation graph via `torch.fx` / `torch.export` |
| 3. Relax IR Import | 3 | Figure 2, Section 3 | Import the model into TVM's Relax IR, show call graph |
| 4. Graph Passes | 4 | Section 3 -- Operator Fusion, Legalization | Apply graph-level compiler passes with before/after diffs |
| 5. Operators | 5 | Section 3 → 4 transition | List all TIR operators with kind summary and per-type counts |
| 6. TensorIR | 6 | Section 4, Figure 13 | Inspect a PrimFunc's TIR source, block iterators, loop nest, and buffers |
| 7. Tensor Expressions | 7 | Section 4.1, Figure 5 | Demonstrate compute/schedule separation with a standalone conv2d |
| 8. Auto-Tuning | 8-9 | Sections 5.1, 5.3, Figure 12 | Extract tuning tasks, run MetaSchedule, show candidate cards and convergence |
| 9. Cost Model | 10 | Section 5.2, Figure 13 | Select the best candidate, display structural features and cost model explanation |
| 10. Build & Compare | 11-12 | Figure 2 (bottom), Section 6, Figure 14 | Compile to CUDA with DLight scheduling, run TVM inference, compare against PyTorch |
| 11. Summary | 13 | Figure 2 (system overview) | Vertical timeline showing every stage's status, detail, and paper reference |

---

## Architecture

### Top-Level Controls

```
┌──────────────────────────────────────────────────────────────────┐
│  Progress: [0. Load] [1. Infer] [2. FX] ... (color-coded)      │
│  Model: [resnet18 ▼]   [Image]   Trials: [32]   [Run All]      │
└──────────────────────────────────────────────────────────────────┘
```

- **Progress bar** (top): Color-coded badges for all 13 stages. Green = done,
  orange = running, red = failed, grey = pending. Positioned above the
  controls so it's always visible.
- **Model dropdown**: `resnet18` (default) or `mobilenet_v2`.
- **Image upload**: Optional PIL image; falls back to a sample ImageNet image
  via `helpers.download_sample_image()`.
- **Tuning slider**: 4–128 trials (default 32).  Controls the MetaSchedule
  budget for stages 8-9.
- **Run All button**: Placed inline with controls. Executes every stage
  sequentially using `gr.Progress` to report percent completion.

### State Management

A single module-level `DemoState` instance (`STATE`) holds all artifacts.
Each stage function:

1. Calls `STATE.mark(stage_id, StageStatus.RUNNING)`.
2. Calls the corresponding `pipeline.*` function.
3. Stores results in `STATE` fields.
4. Calls `STATE.mark(stage_id, StageStatus.DONE)` (or `FAILED`).
5. Returns a tuple of `(status_html, *viz_outputs, progress_html)`.

The last element of every return tuple is the refreshed progress bar HTML,
so individual tab buttons can update the global badge strip.

### Per-Tab Architecture

Each tab follows a consistent pattern:

```
┌───────────────────────────────────────────────────────┐
│ Gradient header: stage title + paper ref + explanation │
│ [ Run Stage X ]  button (primary variant)             │
│ Status badge (gr.HTML)                                │
│ Primary output (Code / HTML)                          │
│ Secondary output (table / chart)                      │
└───────────────────────────────────────────────────────┘
```

Every tab has a gradient-colored header div that names the stage, cites the
paper section, and gives a one-sentence educational explanation.  Each tab
has its own "Run Stage X" button (primary variant), so stages can be
executed individually.  The "Run All Stages" button calls every stage
function in sequence and populates all tabs at once.

### Error Handling

Every stage function wraps its body in `try/except`:

- On failure, the stage is marked `FAILED` in `DemoState`, and the status
  output shows a red error banner with the exception message.
- Subsequent stages that depend on a missing artifact return an orange
  prerequisite warning (e.g., "Run Stage 0-1 first").
- The "Run All" path does not short-circuit on failure -- it continues to
  attempt all stages, so partial results are still visible.

---

## Gradio Component Mapping

### Tab 1: Task & Input (Stages 0-1)

| Component | Type | Source |
|-----------|------|--------|
| Status | `gr.HTML` | `_ok()` / `_err()` with model info |
| Environment | `gr.Markdown` | `format_device_banner(check_environment())` (inside accordion) |
| Top-5 predictions | `gr.Markdown` | Markdown table from `run_pytorch_inference().top5` |
| Latency | `gr.Markdown` | Median latency string with device label |

**Pipeline functions**: `check_environment`, `load_model`, `prepare_input`,
`run_pytorch_inference`.

### Tab 2: PyTorch Graph (Stage 2)

| Component | Type | Source |
|-----------|------|--------|
| Status | `gr.HTML` | Node count, torch.export availability |
| FX Graph SVG | `gr.HTML` | `viz.graph_render.fx_graph_to_svg()` |
| Node Table | `gr.HTML` | `viz.graph_render.fx_node_table_html()` |

**Pipeline function**: `trace_pytorch_graph`.

### Tab 3: TVM IR Import (Stage 3)

| Component | Type | Source |
|-----------|------|--------|
| Status | `gr.HTML` | Function count, param count, IR line count |
| Relax IR | `gr.Code(language="python")` | Raw IR text (truncated to 300 lines) |
| Call Graph SVG | `gr.HTML` | `viz.graph_render.relax_callgraph_to_svg()` |

**Pipeline function**: `import_to_tvm`.

### Tab 4: TVM Passes (Stage 4)

| Component | Type | Source |
|-----------|------|--------|
| Status | `gr.HTML` | Pass count |
| Pass Delta Table | `gr.Markdown` | `viz.ir_display.format_all_pass_deltas()` |
| Post-Pass IR | `gr.Code(language="python")` | Final IR snapshot (accordion) |
| Pass Diff | `gr.Code(language="diff")` | `viz.ir_display.ir_diff()` (accordion with dropdown) |

**Pipeline function**: `apply_passes_stepwise`.  
**Interactive feature**: Pass dropdown populates after stage runs; selecting
a pass shows the unified diff between the IR before and after that pass.

### Tab 5: Operators (Stage 5)

| Component | Type | Source |
|-----------|------|--------|
| Status | `gr.HTML` | Operator count |
| Operator Table | `gr.HTML` | `viz.ir_display.operator_table_html()` -- includes kind summary bar (counts per type) |

**Pipeline function**: `extract_operators`.

### Tab 6: TensorIR (Stage 6)

| Component | Type | Source |
|-----------|------|--------|
| Operator dropdown | `gr.Dropdown` | Populated from `get_op_names()`, labels include `[kind]` suffix |
| Status | `gr.HTML` | Block/iterator/buffer counts with for-loop vs block-iter breakdown |
| TIR Source | `gr.Code(language="python")` | Raw TIR text (truncated to 200 lines) |
| AST Tree | `gr.HTML` | `viz.ir_display.tir_ast_tree_html()` -- shows both `For` loops and block iter_vars |
| Iterator Table | `gr.HTML` | `viz.ir_display.tir_loop_table_html()` -- includes Source column (for loop / block iter), Spatial/Reduce kind |

**Pipeline function**: `get_tir_ast`.  
**Interactive feature**: Selecting an operator from the dropdown re-runs the
TIR analysis for that specific PrimFunc.  The dropdown shows `[kind]` labels
(e.g. `[conv]`, `[matmul]`, `[elemwise]`) so users can quickly navigate to
operators of interest.

**Block iterators**: After `FuseTIR`, many PrimFuncs use `Block` nodes with
`iter_vars` (spatial/reduction iterators) instead of explicit `For` loops.
The walker extracts these as pseudo-loop entries with `source=block_iter`,
ensuring the iteration table is populated for all operators.

### Tab 7: Tensor Expression (Stage 7)

| Component | Type | Source |
|-----------|------|--------|
| Status | `gr.HTML` | Success message |
| TE Compute | `gr.Code(language="python")` | Compute declaration source |
| Naive TIR | `gr.Code(language="python")` | Lowered (un-scheduled) TIR |
| Explanation | `gr.Markdown` | Educational text about compute/schedule separation |

**Pipeline function**: `build_te_microscope`.

### Tab 8: Schedule Search (Stages 8-9)

| Component | Type | Source |
|-----------|------|--------|
| Task Status | `gr.HTML` | Task count |
| Tune Status | `gr.HTML` | Candidate count, synthetic flag |
| Candidate Cards | `gr.HTML` | `viz.schedule_display.candidate_cards_html()` |
| Convergence Chart | `gr.HTML` | `viz.charts.convergence_chart()` (base64 img) |
| Task Pie Chart | `gr.HTML` | `viz.charts.task_weight_pie_chart()` (base64 img) |

**Pipeline functions**: `extract_tuning_tasks`, `run_tuning`.

### Tab 9: Cost Model (Stage 10)

| Component | Type | Source |
|-----------|------|--------|
| Status | `gr.HTML` | Best candidate info |
| Banner + TIR Features | `gr.HTML` | `viz.schedule_display.best_candidate_banner_html()` + `viz.feature_table.tir_features_table_html()` |
| Feature Table | `gr.HTML` | `viz.feature_table.feature_table_html()` |
| Explanation | `gr.HTML` | `viz.feature_table.cost_model_explanation_html()` |

**Pipeline function**: `select_best_candidate`, `compute_tir_structural_features`.

### Tab 10: Build & Results (Stages 11-12)

Stage 11 applies **DLight GPU scheduling** (`tvm.s_tir.dlight`) before the
CUDA build, adding GPU thread bindings to TIR PrimFuncs.  The CUDA build
works end-to-end: real kernels (~1.2M chars of source), compiled via nvcc/gcc
from `~/tvm_env/bin`.

| Component | Type | Source |
|-----------|------|--------|
| Build Status | `gr.HTML` | Target info, params-bound flag |
| TVM Top-5 | `gr.Markdown` | Markdown table |
| Comparison | `gr.Markdown` | Max diff, cosine similarity, speedup |
| Latency Chart | `gr.HTML` | `viz.charts.latency_comparison_chart()` (base64 img) |
| CUDA Source | `gr.Code(language="c")` | Generated CUDA kernel (accordion, 3000 char limit) |
| Done | `gr.HTML` | "Pipeline complete!" banner |

**Pipeline functions**: `build_tvm_module`, `run_tvm_inference`, `compare_results`.

### Tab 11: Pipeline Timeline (Stage 13)

| Component | Type | Source |
|-----------|------|--------|
| Timeline | `gr.HTML` | Vertical step-by-step timeline with status icons, paper references, and summary details per stage |

**Function**: `build_timeline()` -- reads `STATE.stage_status` and all
summary fields to build a visual timeline.

---

## "Run All Stages" Integration

The `run_all_stages` function:

1. Accepts `(model_choice, image, max_trials)` plus a `gr.Progress` tracker.
2. Calls each `run_stage_*` function sequentially, reporting progress at each
   step (0% → 15% → 25% → ... → 100%).
3. Each sub-function returns a tuple ending with a progress HTML string.
4. The final return strips the trailing progress element from each
   sub-result and concatenates them into a single flat tuple of 40 values
   matching the 40 output components.

Output count verification:

| Source | Raw returns | Strip last | Cumulative |
|--------|-------------|------------|------------|
| s01 | 6 | 5 | 5 |
| s2 | 4 | 3 | 8 |
| s3 | 4 | 3 | 11 |
| s4 | 4 | 3 | 14 |
| s5 | 3 | 2 | 16 |
| s6 | 5 | 4 | 20 |
| s7 | 5 | 4 | 24 |
| s89 | 6 | 5 | 29 |
| s10 | 5 | 4 | 33 |
| s1112 | 7 | 6 | 39 |
| timeline | -- | -- | 40 |
| progress | -- | -- | **41** |

---

## Launching the App

### Local

```bash
python app.py
# Opens at http://localhost:7860
```

### With Gradio share link

```bash
python app.py --share
# Prints a public https://*.gradio.live URL
```

### For Cloudflare Tunnel

```bash
# Terminal 1: Start the app
python app.py --host 0.0.0.0 --port 7860

# Terminal 2: Start the tunnel
cloudflared tunnel --url http://localhost:7860
```

This exposes the NVIDIA desktop's GPU-powered demo to any browser
(e.g., a MacBook on a different network) via a public HTTPS URL.

### CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Server bind address |
| `--port` | `7860` | Server port |
| `--share` | `false` | Create a Gradio share link |

---

## Design Decisions

### 1. Single Global State vs. Per-Session State

We use a single module-level `DemoState` instance.  This is simpler and
sufficient for a single-user demo.  For multi-user deployment, Gradio's
`gr.State` could wrap `DemoState`, but the trade-off is increased
complexity for a demo that is designed for one presenter at a time.

### 2. Non-Generator "Run All"

The `run_all_stages` function is a **regular function** (not a generator).
While a generator would allow intermediate UI updates per stage,
maintaining exactly 40 output values in every intermediate yield is
error-prone.  Instead, `gr.Progress` provides visual feedback during
execution, and all outputs populate simultaneously when the full pipeline
completes.

### 3. Raw Text for gr.Code Components

The `gr.Code` component renders its own syntax highlighting and line
numbers.  We pass raw IR / TIR / CUDA text (truncated at sensible limits)
rather than pre-highlighted HTML, to leverage Gradio's native code viewer.

### 4. Accordion Sections for Large Outputs

The post-pass IR (Tab 4) and generated CUDA source (Tab 10) are wrapped
in `gr.Accordion` with `open=False`, keeping the default view clean while
making full details available on click.

### 5. Interactive Pass Diff Viewer

Tab 4 includes a dropdown that, when a pass is selected, calls
`view_pass_diff()` to compute a unified diff between that pass's
before/after IR snapshots.  This leverages the `ir_snapshots` dictionary
stored in `DemoState` from `apply_passes_stepwise()`.

### 6. Interactive Operator Selector

Tab 6 has a dropdown that populates from `get_op_names()` (reading
`STATE.operators`).  Each entry includes a `[kind]` label (e.g. `[conv]`,
`[matmul]`) to help users navigate to operators of interest.  Changing the
selection triggers `run_stage_6()` for that specific PrimFunc, allowing
users to browse different operators without re-running the full pipeline.

### 7. Educational Tab Headers

Every tab opens with a gradient-colored banner containing the stage name,
paper citation, and a one-sentence explanation of what happens in that stage.
This reduces the need for separate documentation and makes the demo
self-documenting for presenters and audiences.

### 8. Layer-Focused Navigation

Rather than viewing the model as an opaque whole, the UI provides
layer-focused navigation at three levels:

- **Tab 5** (Operators): Kind summary bar at the top shows counts by type
  (conv, matmul, elemwise, etc.).
- **Tab 6** (TensorIR): Dropdown with `[kind]` labels lets users navigate
  directly to conv, matmul, or other layer types.
- **Tab 8** (Auto-Tuning): Task extraction maps to individual operators,
  showing which layers consume tuning budget.

---

## Visualization Modules Used

| Module | Functions Used | Tab(s) |
|--------|---------------|--------|
| `viz.graph_render` | `fx_graph_to_svg`, `fx_node_table_html`, `relax_callgraph_to_svg` | 2, 3 |
| `viz.ir_display` | `format_all_pass_deltas`, `ir_diff`, `operator_table_html`, `tir_ast_tree_html`, `tir_loop_table_html` | 4, 5, 6 |
| `viz.schedule_display` | `candidate_cards_html`, `best_candidate_banner_html` | 8, 9 |
| `viz.feature_table` | `build_feature_dataframe`, `feature_table_html`, `tir_features_table_html`, `cost_model_explanation_html` | 9 |
| `viz.charts` | `latency_comparison_chart`, `convergence_chart`, `task_weight_pie_chart` | 8, 10 |

---

## Error States and Edge Cases

| Scenario | Behavior |
|----------|----------|
| No image uploaded | Falls back to `download_sample_image()` |
| CUDA not available | PyTorch runs on CPU; TVM build attempts CPU fallback |
| TVM not installed | Stages 3+ fail gracefully with red error banners |
| MetaSchedule unavailable | Stages 8-9 fall back to synthetic tuning data (MetaSchedule *is* available via `tvm.s_tir.meta_schedule` in the standard setup) |
| Stage skipped | Dependent stages show orange prerequisite warnings |
| Very large IR | Truncated to 250-300 lines in `gr.Code` components |
| CUDA source unavailable | Displays "(CUDA source not available for this backend)" (CUDA source *is* available now via `inspect_source()` on `VMExecutable.mod.imports`) |

---

## Dependencies

All dependencies from `requirements.txt` are used:

| Package | Where Used |
|---------|------------|
| `torch`, `torchvision` | Model loading, inference, FX tracing |
| `numpy` | Array manipulation throughout |
| `Pillow` | Image loading and processing |
| `gradio` | The entire UI layer |
| `graphviz` | FX and Relax graph rendering (optional fallback) |
| `networkx` | Graph analysis (used internally by graph_render) |
| `matplotlib` | Charts (convergence, latency, task pie) |
| `pandas` | Feature DataFrame in cost model tab |
| `xgboost` | MetaSchedule cost model (gradient tree boosting) |

**Environment**: nvcc and gcc are installed via conda into `~/tvm_env/bin` and
auto-added to PATH by the pipeline.  A `tp_dealloc` replacement (ctypes) is
applied to prevent TVM FFI use-after-free segfaults in mlc-ai-nightly.

---

## Exit Criterion (from PLAN.md)

> The full demo is accessible from a browser on the MacBook, all tabs are
> interactive, and the pipeline completes within a reasonable time
> (< 10 min including tuning).

This is satisfied when:
1. `python app.py` launches without errors.
2. All 11 tabs are visible and functional.
3. "Run All Stages" populates every tab with computed (not mock) data.
4. Individual tab buttons work independently for re-running stages.
5. The pipeline timeline (Tab 11) shows all stages with correct statuses.
6. Cloudflare tunnel provides external browser access.
