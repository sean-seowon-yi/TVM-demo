# Phase 3 -- MetaSchedule Tuning

> **Status note (2026-03-11):** Historical phase log. Some specifics may differ from current behavior after later fixes/refactors; use README.md + code as source of truth.

> **Goal**: Implement the automated optimization loop -- task extraction,
> schedule search, and cost-model-based selection -- along with three new
> visualization modules for schedule traces, structural features, and
> performance charts.  This phase makes visible the paper's most novel
> contribution (Section 5: "Automating Optimization").

---

## New / Modified Files

| File | Status | Lines | Purpose |
|------|--------|-------|---------|
| `src/backend/pipeline.py` | **modified** | +665 lines | Added `extract_tuning_tasks` (Stage 8), `run_tuning` (Stage 9), `select_best_candidate` (Stage 10), `compute_tir_structural_features`, and 15+ internal helpers |
| `src/viz/schedule_display.py` | **new** | ~190 lines | Trace parsing, candidate cards, best-candidate banner |
| `src/viz/feature_table.py` | **new** | ~210 lines | Structural feature DataFrame, HTML feature table, TIR feature cards, cost model explanation |
| `src/viz/charts.py` | **new** | ~250 lines | Latency comparison bar chart, tuning convergence chart, task weight pie chart |
| `src/viz/__init__.py` | **modified** | docstring | Updated to list the three new Pass 3 submodules |
| `src/tests/test_pipeline.py` | **modified** | +150 lines | Added `test_extract_tuning_tasks`, `test_run_tuning`, `test_select_best`; new Stage 8/9/10 assertions; latency chart in Stage 12 test |

---

## Paper Mapping

| Stage | Paper Reference | What It Shows |
|-------|-----------------|---------------|
| Stage 8 | Section 5.1 -- "Schedule Space Specification" | MetaSchedule extracts tunable tasks (TIR PrimFuncs) from the IRModule. Each task defines a schedule search space. |
| Stage 9 | Section 5.3 -- "Schedule Exploration" | The search loop proposes candidate schedules, measures them on hardware, and feeds results back.  Convergence chart mirrors Figure 12. |
| Stage 10 | Section 5.2 -- "ML-Based Cost Model" (Figure 13) | The cost model ranks candidates by predicted performance.  Educational structural features approximate what the internal XGBoost model considers. |

---

## Architecture: Real MetaSchedule via tvm.s_tir.meta_schedule

MetaSchedule is now **fully functional** for real tuning via
`tvm.s_tir.meta_schedule` (not `tvm.meta_schedule`).  Stage 8 extracts
**28 real tasks** from the IRModule; Stage 9 produces **real tuning**
**records** with measured GPU latencies (not synthetic).

The pipeline still uses fallbacks for older or incompatible builds:

```
extract_tuning_tasks()
  ├── Path 1: ms.relax_integration.extract_tasks()  (tvm.s_tir.meta_schedule)
  ├── Path 2: ms.extract_task_from_relay()            (legacy)
  └── Path 3: _manual_task_extraction()               (PrimFunc wrapping)

run_tuning()
  ├── Path 1: ms.tune_tir()                           (direct)
  ├── Path 2: ms.relax_integration.tune_relax()       (high-level)
  ├── Path 3: _tune_per_task()                        (iterate tasks)
  └── Fallback: _synthetic_tuning_records()           (educational demo data)
```

When real tuning succeeds (default with `tvm.s_tir.meta_schedule`), the
records contain actual measured latencies on the GPU.  When it fails
(missing MetaSchedule, no CUDA device, API incompatibility), the demo
generates **synthetic records** -- structurally identical data with
realistic schedule traces.  Synthetic records are tagged with
`_synthetic: True` and displayed with an orange "synthetic" badge in the UI.

---

## Detailed Module Descriptions

### `src/backend/pipeline.py` -- New Functions

#### Stage 8: `extract_tuning_tasks(mod, target_str="cuda")`

```
Signature:
    extract_tuning_tasks(
        mod: tvm.ir.IRModule,
        target_str: str = "cuda",
    ) -> Tuple[task_dicts, tasks_raw, target]
```

**What it does:**

1. Resolves the TVM target (reuses `_resolve_target()` from Phase 1).
2. Calls `_ms_extract_tasks()` which tries three paths in order:
   - `ms.relax_integration.extract_tasks(mod, target)` -- the real API (via
     `tvm.s_tir.meta_schedule`), extracts ~28 tunable tasks from ResNet-style
     models.
   - `ms.extract_task_from_relay(mod, target=target, params={})` -- legacy Relay.
   - `_manual_task_extraction(mod, target)` -- wraps each `PrimFunc` as a
     pseudo-task object.
3. For each extracted task, collects:

   | Key | Type | Description |
   |-----|------|-------------|
   | `index` | int | Original extraction order |
   | `name` | str | Task name / workload key |
   | `flop_estimate` | int | Rough FLOP count from `_estimate_flops()` |
   | `weight` | float | Relative importance weight (from MetaSchedule) |
   | `tir_source` | str | TIR source of the dispatched module |
   | `tir_lines` | int | Line count of the TIR source |
   | `target` | str | Target string used |

4. Sorts tasks by weight (descending) so the heaviest tasks appear first.

**Returns:**
- `task_dicts` -- display-ready list of dicts.
- `tasks_raw` -- raw ExtractedTask objects (passed to `run_tuning()`).
- `target` -- resolved `tvm.target.Target`.

**Internal helpers:**
- `_ms_extract_tasks(mod, target)` -- three-path task extraction with fallbacks.
- `_manual_task_extraction(mod, target)` -- creates `_PseudoTask` wrappers
  around each `PrimFunc` for display purposes when MetaSchedule APIs are
  unavailable.
- `_estimate_flops(task)` -- walks the dispatched TIR body counting
  arithmetic ops and buffer stores as a rough FLOP proxy.
- `_count_tir_ops(node, depth)` -- recursive counter for `Add`, `Mul`, `Sub`,
  `Div`, `FloorDiv`, `Mod`, `BufferStore` nodes (depth-limited to 200).

---

#### Stage 9: `run_tuning(mod, target, work_dir, max_trials_global, num_trials_per_iter, max_tasks)`

```
Signature:
    run_tuning(
        mod: tvm.ir.IRModule,
        target: tvm.target.Target,
        work_dir: str = "./tuning_logs",
        max_trials_global: int = 64,
        num_trials_per_iter: int = 16,
        max_tasks: int = 3,
    ) -> Tuple[tuning_records, convergence_data, work_dir]
```

**What it does:**

1. Creates the work directory.
2. Calls `_try_tune()` which attempts three tuning paths:
   - `ms.tune_tir(mod, target, config, work_dir)` -- direct TIR tuning.
   - `ms.relax_integration.tune_relax(mod, target, ...)` -- high-level Relax
     tuning.
   - `_tune_per_task()` -- extracts tasks and tunes each individually with a
     per-task trial budget.
3. If all real tuning fails, falls back to `_synthetic_tuning_records()`.
4. Returns structured records and convergence data.

**`tuning_records` schema** (one dict per candidate):

| Key | Type | Description |
|-----|------|-------------|
| `candidate_id` | int | Sequential ID across all tasks |
| `task_name` | str | Which task this candidate belongs to |
| `trace_text` | str | Schedule trace (split, reorder, bind, ...) |
| `run_secs` | float | Measured latency in seconds |
| `run_ms` | float | Measured latency in milliseconds |
| `is_best` | bool | True for the winning candidate |
| `_synthetic` | bool | (optional) True if data is synthetic |

**`convergence_data` schema** (one dict per trial):

| Key | Type | Description |
|-----|------|-------------|
| `trial_index` | int | Sequential trial number |
| `best_latency_ms` | float | Best latency found so far at this trial |

**Internal helpers:**
- `_try_tune(ms, mod, target, ...)` -- orchestrates the three tuning paths.
- `_tune_per_task(ms, mod, target, ...)` -- per-task loop, distributes trial
  budget evenly across tasks.
- `_collect_records_from_db(db, work_dir, ...)` -- reads raw records from a
  MetaSchedule database object or JSON files in the work directory.
- `_read_raw_records(db, work_dir)` -- tries `db.get_all_tuning_records()`,
  iteration, and JSON file parsing.
- `_extract_run_secs(rec)` -- extracts measured runtime from a record (handles
  both dict and object forms, and list-valued times).
- `_extract_trace(rec)` -- extracts the schedule trace text.
- `_extract_task_name(rec, fallback_idx)` -- gets the task name from a record.

**Synthetic tuning records:**

`_synthetic_tuning_records(mod, target)` generates 32 candidates with:
- Task names drawn from actual PrimFunc names in the module.
- Realistic schedule instruction sequences (split, reorder, bind, vectorize,
  cache_read, cache_write, unroll).
- Latencies that converge downward over trials (simulating the search
  improving).
- Deterministic (seeded with 42) for reproducibility.

---

#### Stage 10: `select_best_candidate(tuning_records, mod)`

```
Signature:
    select_best_candidate(
        tuning_records: List[dict],
        mod: tvm.ir.IRModule,
    ) -> Tuple[best, features]
```

**What it does:**

This implements the two-layer approach from the plan:

**Layer A -- What TVM Actually Exposes:**
- Filters valid records (run_ms < 1e6).
- Selects the candidate with the lowest measured latency as `best`.

**Layer B -- Educational Structural Features:**
- For each valid candidate, calls `_compute_trace_features()` which parses
  the trace text to extract:

  | Feature | How Computed |
  |---------|--------------|
  | `num_splits` | Count of "split" in trace |
  | `num_reorders` | Count of "reorder" in trace |
  | `has_cache_read` | "cache_read" present in trace |
  | `has_cache_write` | "cache_write" present in trace |
  | `num_thread_bindings` | Count of "blockidx" + "threadidx" in trace |
  | `has_vectorize` | "vectorize" present in trace |
  | `has_unroll` | "unroll" present in trace |
  | `has_shared_memory` | "shared" present in trace |
  | `trace_length` | Character length of the trace string |

- Each feature dict also includes `candidate_id`, `task_name`, `run_ms`,
  `is_best` for display.
- Features are sorted by latency (ascending).

**Returns:**
- `best` -- the winning record dict (or None).
- `features` -- list of feature dicts for all valid candidates.

---

#### `compute_tir_structural_features(mod, op_name)`

```
Signature:
    compute_tir_structural_features(
        mod: tvm.ir.IRModule,
        op_name: str,
    ) -> dict
```

An auxiliary function (not a stage function) that walks a specific PrimFunc's
TIR AST to extract structural features for cost-model education.

**Feature dict:**

| Feature | Description |
|---------|-------------|
| `op_name` | Operator name |
| `num_loops` | Total `tvm.tir.For` nodes |
| `num_thread_bindings` | Loops with `thread_binding` annotations |
| `num_shared_buffers` | Buffers with "shared" in their scope |
| `num_vectorized_loops` | Loops with `Vectorized` kind |
| `num_unrolled_loops` | Loops with `Unrolled` kind |
| `num_blocks` | Total `tvm.tir.Block` nodes |
| `num_buffer_stores` | Total `tvm.tir.BufferStore` nodes |
| `total_loop_extent_product` | Product of all loop extents (proxy for work volume) |
| `arithmetic_intensity_proxy` | `total_loop_extent_product / num_buffer_stores` |

Paper mapping: Section 5.2, Figure 13 -- "Feature extraction from loop AST."

**Internal helper:**
- `_walk_tir_for_features(node, features, depth)` -- recursive walk handling
  `For`, `Block`, `BlockRealize`, `SeqStmt`, `BufferStore`, and generic nodes.

---

### `src/viz/schedule_display.py`

Parses MetaSchedule schedule traces and renders candidate cards as HTML.

#### Trace Parsing

**`trace_to_readable(trace_text) -> List[str]`**

Handles two trace formats:
1. **Arrow-separated** (from synthetic records):
   `"split(i, [4, 8]) -> reorder(i_0, j_0) -> bind(blockIdx.x, i_0)"`
   is split on ` -> ` into a list.
2. **Structured** (from real MetaSchedule traces): uses a regex to match
   schedule instruction keywords like `split`, `reorder`, `bind`,
   `vectorize`, `unroll`, `cache_read`, `cache_write`, `compute_at`, etc.

Returns a list of instruction strings.

**Instruction Color Palette:**

| Instruction Type | Color | Category |
|------------------|-------|----------|
| `split`, `tile` | `#1976D2` (blue) | Tiling |
| `reorder`, `fuse` | `#7B1FA2` (purple) | Loop ordering |
| `bind` | `#E65100` (deep orange) | Thread binding |
| `vectorize`, `unroll`, `parallel` | `#2E7D32` (green) | ILP |
| `cache_read`, `cache_write`, `set_scope` | `#00838F` (cyan) | Memory |
| `compute_at`, `compute_inline`, `reverse_compute_at` | `#5D4037` (brown) | Compute location |
| `sample_*`, `annotate` | `#455A64` (blue-gray) | MetaSchedule internals |

#### Candidate Card Rendering

**`trace_to_card_html(record, rank=0) -> str`**

Renders one candidate as a styled card with:
- Candidate ID and task name header.
- Latency value.
- Schedule instructions as coloured badges.
- Green border + "BEST" badge if `is_best=True`.
- Orange "synthetic" tag if `_synthetic=True`.

**`candidate_cards_html(records, max_display=20) -> str`**

Wraps all candidates in a scrollable container, sorted by latency.
Shows a header with count and an overflow note if truncated.

**`best_candidate_banner_html(best) -> str`**

Large gradient-background banner for the winner:
- Dark green gradient background with white text.
- Candidate ID, task name, latency, schedule summary.

---

### `src/viz/feature_table.py`

Structural feature DataFrame and HTML rendering for cost-model education.

#### DataFrame Builder

**`build_feature_dataframe(features) -> pd.DataFrame`**

Converts the list of feature dicts (from `select_best_candidate()`) into a
pandas DataFrame with typed columns.  Boolean columns (`has_cache_read`,
`has_cache_write`, etc.) are cast to bool.  Sorted by `run_ms` ascending.

Falls back to returning the raw list if pandas is unavailable.

**Column order:**

```
candidate_id, task_name, run_ms, num_splits, num_reorders,
num_thread_bindings, has_cache_read, has_cache_write,
has_shared_memory, has_vectorize, has_unroll, trace_length, is_best
```

#### HTML Rendering

**`feature_table_html(features, max_rows=30) -> str`**

Renders the DataFrame as a styled HTML table with:
- Column headers from `_COLUMN_LABELS` mapping (e.g. `num_splits` -> "Splits").
- Boolean values as green "Yes" or gray "--".
- Float values formatted to 4 decimal places.
- Best-candidate row highlighted with green background.
- Overflow note if truncated.

**`tir_features_table_html(features) -> str`**

Renders a single-operator TIR structural feature dict (from
`compute_tir_structural_features()`) as a two-column key-value table.
Large integers formatted with commas.

**`cost_model_explanation_html() -> str`**

Static educational card (yellow background, warning-style border) explaining:
- The paper's gradient tree boosting (XGBoost) cost model.
- How it predicts execution time from loop AST features.
- What each displayed feature approximates.
- A disclaimer that the exact internal features are not fully surfaced.

References: Paper Section 5.2, Figure 13.  **xgboost** is installed for the
MetaSchedule cost model (gradient tree boosting).

---

### `src/viz/charts.py`

Matplotlib-based charts with multiple output formats (`html` base64 `<img>`,
`pil` PIL Image, `fig` raw matplotlib Figure).

#### Output Format Helper

All chart functions accept `return_format` parameter:
- `"html"` (default) -- returns `<img src="data:image/png;base64,...">` for
  embedding in Gradio `gr.HTML`.
- `"pil"` -- returns a `PIL.Image` for `gr.Image`.
- `"fig"` -- returns the raw matplotlib Figure (for further customization).

#### `latency_comparison_chart(pytorch_ms, tvm_ms, title, return_format)`

**Paper mapping**: Section 6, Figure 14.

Grouped bar chart:
- Two bars: "PyTorch (eager)" in blue, "TVM (optimized)" in green.
- Value labels above each bar.
- Annotated arrow showing speedup factor (e.g. "1.8x speedup") in orange.
- Clean styling: hidden top/right spines, adequate headroom.

#### `convergence_chart(convergence_data, title, return_format)`

**Paper mapping**: Section 5.3, mirrors Figure 12.

Line chart of best-latency-so-far vs. trial index:
- Blue filled area under the curve.
- Marker dots at each trial.
- Green dashed horizontal line at the final best latency.
- Annotated label showing the final best value.
- Integer x-axis ticks (trial indices).

Handles empty data gracefully with a centered "No convergence data" message.

#### `task_weight_pie_chart(task_dicts, max_slices, title, return_format)`

Pie chart showing the relative weight distribution of tuning tasks:
- Up to `max_slices` segments (default 8), with an "other" wedge for
  remaining tasks.
- Task names truncated to 20 characters.
- Percentage labels on each wedge.
- `Set3` colormap for distinct, readable colors.

---

## Test Extensions

### New test functions in `test_pipeline.py`

#### `test_extract_tuning_tasks(state)` -- Stage 8

- Calls `extract_tuning_tasks()` on the post-pass IRModule.
- Stores `tuning_tasks`, `_tasks_raw`, `_tuning_target` in state.
- Prints: task count, top-5 tasks with name/weight/flops/TIR lines.
- Viz test: `task_weight_pie_chart()`, prints HTML size.
- Marks stage 8 done.

#### `test_run_tuning(state)` -- Stage 9

- Calls `run_tuning()` with `max_trials_global=16`, `num_trials_per_iter=8`,
  `max_tasks=2`.
- Stores `tuning_records` and `convergence_data` in state.
- Prints: record count (with synthetic label if applicable), convergence
  points, best candidate info.
- Viz tests:
  - `candidate_cards_html()` with `max_display=5`.
  - `trace_to_readable()` on the first record.
  - `convergence_chart()`.
- Marks stage 9 done.

#### `test_select_best(state)` -- Stage 10

- Calls `select_best_candidate()` on the tuning records.
- Stores `best_candidate` and `candidate_features` in state.
- Prints: best candidate info, feature record count and keys.
- Viz tests:
  - `build_feature_dataframe()` and `feature_table_html()`.
  - `cost_model_explanation_html()`.
  - `best_candidate_banner_html()`.
- If operators are available, calls `compute_tir_structural_features()` on
  the first operator and renders `tir_features_table_html()`.
- Marks stage 10 done.

#### `test_tvm_inference(state)` -- Stage 12 (extended)

- Added: `latency_comparison_chart()` rendered at the end of the inference
  test, verifying the chart generation pipeline works end-to-end.

### New assertions

```python
# Stage 8
assert len(state.tuning_tasks) > 0
assert all("name" in t for t in state.tuning_tasks)

# Stage 9
assert len(state.tuning_records) > 0
assert all("run_ms" in r for r in state.tuning_records)
assert state.convergence_data  # non-empty
assert any(r.get("is_best") for r in state.tuning_records)

# Stage 10
assert state.best_candidate["run_ms"] < 1e6
assert state.candidate_features  # non-empty
```

### Updated main flow

```
test_environment()               # Stage 0a
test_load_model()                # Stage 0b
test_pytorch_inference()         # Stage 1
test_pytorch_graph()             # Stage 2

if not --cpu:
    test_import_tvm()            # Stage 3
    test_passes()                # Stage 4
    test_extract_operators()     # Stage 5
    test_tir_ast()               # Stage 6
    test_te_microscope()         # Stage 7
    test_extract_tuning_tasks()  # Stage 8   ** NEW
    test_run_tuning()            # Stage 9   ** NEW
    test_select_best()           # Stage 10  ** NEW
    test_build()                 # Stage 11
    test_tvm_inference()         # Stage 12  (extended with latency chart)

assert_correctness()
```

---

## DemoState Fields Used by Pass 3

These fields were pre-allocated in `state.py` during Phase 1:

```python
# Stage 8
tuning_tasks: List[dict]           # populated by extract_tuning_tasks()

# Stage 9
tuning_records: List[dict]         # populated by run_tuning()
convergence_data: List[dict]       # populated by run_tuning()

# Stage 10
candidate_features: Any            # populated by select_best_candidate() (list or DataFrame)
best_candidate: Optional[dict]     # populated by select_best_candidate()
```

Additionally, two private attributes are stashed for inter-stage communication:

```python
state._tasks_raw      # raw ExtractedTask objects (Stage 8 -> Stage 9)
state._tuning_target  # resolved tvm.target.Target (Stage 8 -> Stage 9)
```

---

## Data Flow Diagram (Full Pipeline with Pass 3)

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
trace_pytorch_graph()  -->  fx_graph, fx_code, node_table
  |
  v
import_to_tvm()  -->  IRModule, params_np
  |
  v
apply_passes_stepwise()  -->  transformed IRModule, snapshots, deltas
  |
  v
extract_operators()  -->  operator list with TIR sources
  |
  v
get_tir_ast()  -->  TIR source + AST summary for one operator
  |
  v
build_te_microscope()  -->  TE compute source + naive TIR
  |
  v
extract_tuning_tasks()  -->  task_dicts, tasks_raw, target    [NEW]
  |
  v
run_tuning()  -->  tuning_records, convergence_data           [NEW]
  |
  v
select_best_candidate()  -->  best, features                  [NEW]
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

## Robustness Strategy Summary

| Failure Mode | Handling |
|---|---|
| `tvm.s_tir.meta_schedule` not importable | Falls back to `tvm.meta_schedule`; if both fail, `RuntimeError` with clear message |
| `extract_tasks` API missing or fails | Falls back to `extract_task_from_relay`, then to manual PrimFunc wrapping |
| `tune_tir` fails | Falls back to `tune_relax`, then to per-task tuning loop |
| All real tuning fails | Generates 32 synthetic records with realistic traces and convergence |
| `pandas` not installed | `build_feature_dataframe` returns raw list; `feature_table_html` accepts both |
| `matplotlib` not installed | Chart functions raise `RuntimeError` with clear message |
| `graphviz` not installed | Already handled in Phase 2 -- text fallbacks |
| MetaSchedule database has no records | `_read_raw_records` tries three extraction methods, returns empty list on total failure |
| Tuning record format varies across TVM versions | `_extract_run_secs` and `_extract_trace` handle both dict and object forms, with list-valued times |

---

## Relationship to Phase 1 and Phase 2

- Phase 1 functions are completely unchanged.
- Phase 2 functions are completely unchanged.
- Phase 3 stages slot between Stage 7 (TE microscope) and Stage 11 (build):

```
Phase 1:  0 -> 1 -> 3 -> 4 ->                      11 -> 12
Phase 2:       2         5 -> 6 -> 7
Phase 3:                             8 -> 9 -> 10
                                     ^^^^^^^^^^^^^^
```

- Phase 3 visualization modules (`schedule_display`, `feature_table`, `charts`)
  are independent of Phase 2 viz modules (`graph_render`, `ir_display`).
  They share no imports.

---

## What Comes Next

Phase 4 (**Gradio App**) will:
- Wire all backend functions and viz utilities into `gr.Blocks` with tabs.
- One tab per stage or logical grouping.
- Progress indicators for long operations (tuning).
- Error handling and loading spinners.
- Cloudflare Tunnel setup for remote browser access.
