# Phase 2 -- Graph & Operator Visualization

> **Status note (2026-03-11):** Historical phase log. Some specifics may differ from current behavior after later fixes/refactors; use README.md + code as source of truth.

> **Goal**: Add pipeline stages 2, 5, 6, and 7, plus two visualization modules
> (`graph_render` and `ir_display`), so every intermediate compiler artifact
> produced by Phase 1 can be inspected visually.  This phase is the "lens"
> through which the Gradio UI (Phase 4) will present TVM internals.

---

## New / Modified Files

| File | Status | Purpose |
|------|--------|---------|
| `src/backend/pipeline.py` | **modified** | Added `trace_pytorch_graph` (Stage 2), `extract_operators` (Stage 5), `get_tir_ast` (Stage 6), `build_te_microscope` (Stage 7) and their internal helpers |
| `src/viz/__init__.py` | **modified** | Updated docstring listing new submodules |
| `src/viz/graph_render.py` | **new** | FX graph SVG, node table HTML, Relax call-graph SVG |
| `src/viz/ir_display.py` | **new** | IR highlighting, diffing, pass-delta tables, operator tables, TIR AST tree/loop/buffer views |
| `src/tests/test_pipeline.py` | **modified** | Extended with Stage 2/5/6/7 tests + new assertions |
| `src/backend/state.py` | unchanged (fields for Stages 2/5/6/7 were pre-allocated in Phase 1) | -- |

---

## Paper Mapping

Each new stage connects to a specific section in the OSDI '18 paper:

| Stage | Paper Reference | What It Shows |
|-------|-----------------|---------------|
| Stage 2 | Figure 2, "Computational Graph" | The PyTorch FX graph -- the framework-level IR that TVM's frontend ingests |
| Stage 5 | Section 4 "Generating Tensor Operators" | The concrete tensor operators (PrimFuncs) that result from graph lowering and fusion |
| Stage 6 | Section 4.1, Figure 5 (left column) | TensorIR (TIR) -- the low-level loop-level representation of one operator, with its AST structure |
| Stage 7 | Section 4.1, Figure 5 "Compute/Schedule" | The compute/schedule separation: a standalone `conv2d` declared with Tensor Expressions, lowered to a naive loop nest |

---

## Detailed Module Descriptions

### `src/backend/pipeline.py` -- New Functions

#### Stage 2: `trace_pytorch_graph(model, example_input)`

```
Signature:
    trace_pytorch_graph(
        model: torch.nn.Module,
        example_input: torch.Tensor,
    ) -> Tuple[fx_graph, fx_code, node_table, exported_program]
```

**What it does:**

1. Deep-copies the model to CPU and puts it in eval mode.
2. Calls `torch.fx.symbolic_trace()` to produce a traced `GraphModule`.
3. Extracts the `Graph` object and the generated Python source code.
4. Iterates over every FX node to build a `node_table` -- a list of dicts:

   | Key | Type | Description |
   |-----|------|-------------|
   | `name` | str | Node name (e.g. `"conv1"`, `"bn1"`, `"add"`) |
   | `op` | str | One of: `placeholder`, `call_module`, `call_function`, `call_method`, `get_attr`, `output` |
   | `target` | str | The module path or function (truncated at 80 chars) |
   | `inputs` | str | Comma-separated names of input nodes |
   | `num_users` | int | How many downstream nodes consume this output |

5. Attempts `torch.export.export()` for the modern ExportedProgram.  Returns
   `None` on failure with a logged warning (non-blocking).

**Internal helpers:**
- `_build_node_table(graph)` -- walks `graph.nodes`, builds the dicts above.
- `_try_torch_export(model_cpu, example_cpu)` -- wraps `torch.export.export()`
  in a try/except.

---

#### Stage 5: `extract_operators(mod)`

```
Signature:
    extract_operators(mod: tvm.ir.IRModule) -> List[dict]
```

**What it does:**

Iterates over all global functions in the post-pass IRModule.  For every
`tvm.tir.PrimFunc` it encounters, it extracts:

| Key | Type | Description |
|-----|------|-------------|
| `name` | str | GlobalVar name hint (e.g. `"fused_nn_conv2d_nn_relu"`) |
| `params` | List[dict] | Each param has `name`, `dtype`, `shape` |
| `num_blocks` | int | Count of `tir.Block` nodes in the AST |
| `tir_source` | str | Full TIR source via `.script()` |
| `op_kind` | str | Inferred category based on name keywords |
| `ir_lines` | int | Line count of the TIR source |

**Operator kind inference** uses a keyword-to-category table:

| Keyword in name | Assigned kind |
|-----------------|---------------|
| `conv` | `conv` |
| `matmul` | `matmul` |
| `dense` | `dense` |
| `batch_norm` | `batchnorm` |
| `relu` | `relu` |
| `add`, `multiply` | `elemwise` |
| `pool` | `pool` |
| `softmax` | `softmax` |
| `reshape` | `reshape` |
| `transpose` | `transpose` |
| `layer_norm` | `layernorm` |
| (no match) | `other` |

The result list is sorted alphabetically by name.

**Internal helpers:**
- `_extract_prim_params(func)` -- reads `func.params` and `func.buffer_map`
  to extract shapes and dtypes.
- `_count_blocks(node, depth=0)` -- recursive walk over TIR AST counting
  `Block` nodes (with depth limit of 200 to avoid infinite recursion).
- `_infer_op_kind(name)` -- keyword matching against `_OP_KIND_KEYWORDS`.
- `_safe_script(obj)` -- calls `.script()` with fallback to `str()`.

---

#### Stage 6: `get_tir_ast(mod, op_name)`

```
Signature:
    get_tir_ast(
        mod: tvm.ir.IRModule,
        op_name: str,
    ) -> Tuple[tir_source, ast_summary]
```

**What it does:**

1. Looks up the named `PrimFunc` in the IRModule via `_find_prim_func()`.
2. Produces the full TIR source text via `_safe_script()`.
3. Walks the TIR AST body with `_walk_tir_ast()` to collect:

   **`ast_summary` structure:**
   ```python
   {
       "blocks": [
           {
               "name": "conv2d",
               "iter_vars": ["n", "oc", "oh", "ow", "ic", "kh", "kw"],
               "num_reads": 2,
               "num_writes": 1,
           },
           ...
       ],
       "loops": [
           # From explicit For nodes:
           {
               "var": "ax0",
               "extent": "64",
               "kind": "DataPar",
               "thread_binding": "blockIdx.x",
               "source": "for_loop",
           },
           # From Block iter_vars (sblock-style TIR):
           {
               "var": "v_n",
               "extent": "1",
               "kind": "S",      # Spatial
               "thread_binding": "",
               "source": "block_iter",
           },
           {
               "var": "v_ic",
               "extent": "64",
               "kind": "R",      # Reduction
               "thread_binding": "",
               "source": "block_iter",
           },
           ...
       ],
       "buffers": [
           {
               "name": "data",
               "shape": [1, 64, 56, 56],
               "dtype": "float32",
               "scope": "",
           },
           ...
       ],
   }
   ```

   **Note on block iterators**: After `FuseTIR`, many PrimFuncs use `Block`
   nodes with `iter_vars` (spatial/reduction iterators) instead of explicit
   `For` loops.  The walker extracts these as entries in `loops` with
   `"source": "block_iter"` and kind `"S"` (spatial) or `"R"` (reduction).
   This ensures the iteration table is always populated.

4. Buffer info is extracted separately from `func.params` / `func.buffer_map`.

**Internal helpers:**
- `_find_prim_func(mod, name)` -- lookup by `name_hint` with a useful
  `KeyError` listing available PrimFuncs.
- `_walk_tir_ast(node, depth=0)` -- returns `{"blocks": [...], "loops": [...]}`.
  Handles both classic TIR (`For`, `Block`) and sblock-style TIR
  (`BlockRealize` wrapping `Block` nodes).
- `_walk_tir_ast_impl(node, out, depth)` -- recursive implementation handling
  `For`, `Block`, `BlockRealize`, `SeqStmt`, and generic body/branch nodes.
- `_tir_value(v)` -- converts TIR expressions (e.g. `IntImm`) to display
  strings.

---

#### Stage 7: `build_te_microscope(n, ci, h, w, co, kh, kw, stride, padding)`

```
Signature:
    build_te_microscope(
        n=1, ci=64, h=56, w=56, co=64, kh=3, kw=3,
        stride=1, padding=1,
    ) -> Tuple[compute_source, naive_tir, explanation]
```

**What it does:**

This is the "microscope operator" -- a standalone demonstration of TVM's
compute/schedule separation (paper Section 4.1, Figure 5).

1. Creates TE placeholders for `data` and `weight`.
2. Calls `topi.nn.conv2d()` with NCHW layout.
3. Records the compute declaration as a human-readable string.
4. Creates a default schedule (`te.create_schedule`) and lowers it with
   `tvm.lower(s, ..., simple_mode=True)` to produce the **naive TIR** --
   a simple nested-loop implementation with no tiling or threading.
5. Generates an educational explanation paragraph that maps the concept back
   to the paper.

**Returns:**
- `compute_source` -- the TE declaration code (readable, not executable).
- `naive_tir` -- the fully lowered TIR of the un-optimized loop nest.
- `explanation` -- multi-paragraph text explaining compute/schedule separation.

**Internal helper:**
- `_te_conv2d(n, ci, h, w, co, kh, kw, stride, padding)` -- wrapped in
  try/except for robustness against missing `tvm.te` / `tvm.topi`.

---

### `src/viz/graph_render.py`

Renders computational graphs to SVG using the `graphviz` Python package.
All functions fall back to plain-text HTML if graphviz is not installed.

#### Color Palette

FX node types are color-coded for visual clarity:

| Op Type | Color | Hex |
|---------|-------|-----|
| `placeholder` | Gray | `#9E9E9E` |
| `call_module` | Blue | `#42A5F5` |
| `call_function` | Green | `#66BB6A` |
| `call_method` | Purple | `#AB47BC` |
| `get_attr` | Orange | `#FFA726` |
| `output` | Red | `#EF5350` |

#### `fx_graph_to_svg(fx_graph, max_nodes=300) -> str`

- Creates a Graphviz `Digraph` with top-to-bottom rank direction.
- Iterates over FX nodes (capped at `max_nodes`).
- Each node is rendered as a colored rounded box with a label showing
  `name\nop: target`.
- Edges follow `node.args` references.
- Returns the SVG as a UTF-8 string via `dot.pipe(format="svg")`.

**Fallback:** `_fx_graph_text_fallback()` returns a `<pre>` block listing each
node's name, op, and target.

#### `fx_node_table_html(node_table) -> str`

Takes the `List[dict]` from `_build_node_table()` and renders it as an HTML
`<table>` with columns: #, Name, Op (color-coded badge), Target, Inputs,
Users.  Intended for display in Gradio's `gr.HTML` component.

#### `relax_callgraph_to_svg(mod) -> str`

Extracts TVM Relax `call_tir` / `call_dps_packed` targets from the printed IR
using a regex pattern.  Builds a sequential dataflow graph:

```
[input] --> [op1] --> [op2] --> ... --> [output]
```

Where each operation node gets a light-blue rounded box.  Input and output are
ellipses with distinct colors.

**Fallback:** `_relax_text_fallback()` lists IRModule global functions with
their types.

---

### `src/viz/ir_display.py`

Provides utilities for textual display, comparison, and structured rendering
of TVM IR and TIR artifacts.

#### IR Text Display

**`highlight_ir(ir_text, max_lines=200) -> str`**

Adds line numbers and truncates long IR texts:

```
  1 | @I.ir_module
  2 | class Module:
  3 |     @T.prim_func
...
# ... (347 more lines omitted)
```

Suitable for `gr.Code(language="python")` or plain text areas.

#### IR Diffing

**`ir_diff(before, after, before_label, after_label, context_lines=3) -> str`**

Produces a unified diff between two IR snapshots using `difflib.unified_diff`.
The output follows standard diff format and can be rendered with
`gr.Code(language="diff")`.

**`ir_diff_stats(before, after) -> dict`**

Returns a numeric summary:

```python
{
    "lines_before": 450,
    "lines_after": 380,
    "sections_added": 12,
    "sections_removed": 8,
}
```

Uses `difflib.SequenceMatcher.get_opcodes()` to classify change regions.

#### Pass Delta Summary

**`format_pass_delta(name, delta) -> str`**

One-line markdown summary of a single pass:

```
**FuseOps**  --  functions 15 -> 35  --  TIR funcs 14 -> 34  --  (0.012s)
```

**`format_all_pass_deltas(pass_order, deltas) -> str`**

Multi-line markdown table:

```
| # | Pass | Functions | TIR Funcs | Time |
|---|------|-----------|-----------|------|
| 1 | LegalizeOps | 2 -> 15 | 0 -> 14 | 0.045s |
| 2 | AnnotateTIROpPattern | 15 -> 15 | 14 -> 14 | 0.003s |
| 3 | FuseOps | 15 -> 35 | 14 -> 34 | 0.012s |
| 4 | FuseTIR | 35 -> 22 | 34 -> 21 | 0.008s |
| 5 | DeadCodeElimination | 22 -> 22 | 21 -> 21 | 0.001s |
```

#### Operator Table (Stage 5)

**`operator_table_html(operators) -> str`**

Renders the extracted-operator list as an HTML table with columns: #, Name,
Kind (color-coded badge), Params (compact shape notation), Blocks, IR Lines.

**Kind badges** use per-category colors:

| Kind | Color |
|------|-------|
| `conv` | `#1976D2` (blue) |
| `matmul`, `dense` | `#7B1FA2` (purple) |
| `elemwise` | `#388E3C` (green) |
| `relu` | `#F57C00` (orange) |
| `batchnorm`, `layernorm` | `#00796B` (teal) |
| `pool` | `#5D4037` (brown) |
| `softmax` | `#C62828` (deep red) |
| `reshape`, `transpose` | `#546E7A` (blue-gray) |

**Parameter shapes** are rendered in compact notation, e.g.
`f32[1x64x56x56], f32[64x64x3x3]`.

#### TIR AST Visualization (Stage 6)

Three complementary views of the AST summary:

**`tir_ast_tree_html(ast_summary) -> str`**

A monospace HTML `<div>` with three sections:

- **Buffers** -- bulleted list with name, dtype, shape, and scope.
- **Loop Nest** -- bulleted list with loop variable, extent, and thread binding
  (highlighted in blue if bound).
- **Blocks** -- bulleted list with block name, iteration variables, read count,
  write count.

**`tir_loop_table_html(ast_summary) -> str`**

HTML `<table>` with columns: #, Variable, Extent, Kind, Thread Binding.

**`tir_buffer_table_html(ast_summary) -> str`**

HTML `<table>` with columns: Name, Shape, Dtype, Scope.

---

## Test Extensions

### New test functions in `test_pipeline.py`

#### `test_pytorch_graph(state)`

- Calls `trace_pytorch_graph()` with the loaded model and input tensor.
- Stores `fx_graph`, `fx_code`, and `exported_program` in state.
- Prints: FX node count, op type distribution, generated code length,
  `torch.export` availability.
- Tests visualization: calls `fx_graph_to_svg()` and `fx_node_table_html()`,
  prints their output sizes.

#### `test_extract_operators(state)`

- Calls `extract_operators()` on the post-pass IRModule.
- Stores the operator list in state.
- Prints: total count, first 10 operators with name/kind/block count.
- Tests visualization: calls `operator_table_html()`, prints HTML size.

#### `test_tir_ast(state)`

- Picks the first operator from the extracted list.
- Calls `get_tir_ast()` to get TIR source and AST summary.
- Stores `selected_tir_name`, `selected_tir_source`, `tir_ast_summary`.
- Prints: operator name, source length, block/loop/buffer counts.
- Tests all four viz functions: `highlight_ir()`, `tir_ast_tree_html()`,
  `tir_loop_table_html()`, `tir_buffer_table_html()`, printing sizes.

#### `test_te_microscope(state)`

- Calls `build_te_microscope()` with default dimensions.
- Stores `te_compute_source` and `te_lowered_tir`.
- Prints: compute source code, naive TIR line count, explanation length.
- If Stage 6 produced TIR, runs `ir_diff()` and `ir_diff_stats()` between the
  microscope TIR and the real model operator TIR.

### New assertions

```python
# Stage 2
assert len(list(state.fx_graph.nodes)) > 0
assert len(state.fx_code) > 0

# Stage 5
assert len(state.operators) > 0
assert all("tir_source" in op for op in state.operators)

# Stage 6
assert len(state.selected_tir_source) > 0
assert state.tir_ast_summary is not None

# Stage 7
assert "placeholder" in state.te_compute_source or "topi" in state.te_compute_source
assert len(state.te_lowered_tir) > 0
```

### Updated main flow

```
test_environment()            # Stage 0a
test_load_model()             # Stage 0b
test_pytorch_inference()      # Stage 1     (PyTorch only)
test_pytorch_graph()          # Stage 2  ** NEW -- PyTorch only, no TVM needed

if not --cpu:
    test_import_tvm()         # Stage 3
    test_passes()             # Stage 4
    test_extract_operators()  # Stage 5  ** NEW
    test_tir_ast()            # Stage 6  ** NEW
    test_te_microscope()      # Stage 7  ** NEW
    test_build()              # Stage 11
    test_tvm_inference()      # Stage 12

assert_correctness()
```

Stage 2 runs even in `--cpu` mode because it only uses `torch.fx`.

---

## Fixes Applied During Phase 2

### Unicode encoding on Windows

`ir_display.py` initially used box-drawing characters (`│`) for tree
connectors and ellipsis (`…`) for truncation.  These caused
`UnicodeEncodeError` on Windows consoles with `cp1252` encoding.

**Fix:** Replaced `│` with `|` and `…` with `...` in all console-facing
strings.  HTML output (which runs in a browser) can still use Unicode freely
via `html.escape()`.

### Viz package init update

`src/viz/__init__.py` was updated from an empty file to include a docstring
documenting the new `graph_render` and `ir_display` submodules, plus
placeholders for Phase 3 modules (`schedule_display`, `feature_table`,
`charts`).

---

## Relationship to Phase 1

Phase 2 does **not** change any Phase 1 behavior.  All Phase 1 functions
(`load_model`, `import_to_tvm`, `apply_passes_stepwise`, `build_tvm_module`,
`run_tvm_inference`, `compare_results`) continue to work identically.

Phase 2 adds *new* pipeline functions and *new* visualization modules that
consume the same `DemoState` artifacts produced by Phase 1.  The test script
calls the new stages in logical order between Phase 1's existing stages:

```
Phase 1 stages:  0 → 1 → 3 → 4 → 11 → 12
Phase 2 inserts:       ↑   ↑    ↑↑
                       2   5    6 7
```

---

## What Comes Next

Phase 3 (**MetaSchedule Tuning**) will add:
- Stage 8: Task extraction from the IRModule
- Stage 9: Schedule search (candidate generation, mini-tuning)
- Stage 10: Cost model & best-candidate selection
- Visualization modules: `schedule_display.py`, `feature_table.py`, `charts.py`
  (convergence plots, latency charts)
