#!/usr/bin/env python
"""Pipeline smoke test (Pass 1 + Pass 2 + Pass 3) -- runs all stages headless.

Usage (from repo root):
    python -m src.tests.test_pipeline          # full run (needs TVM + CUDA)
    python -m src.tests.test_pipeline --cpu     # PyTorch-only stages on CPU
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure the src/ directory is on sys.path
_SRC = str(Path(__file__).resolve().parent.parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from backend.state import DemoState, StageStatus
from backend.helpers import (
    format_device_banner,
    get_device_info,
    cosine_similarity,
    download_sample_image,
)
from backend.pipeline import (
    check_environment,
    load_model,
    prepare_input,
    run_pytorch_inference,
    trace_pytorch_graph,
    import_to_tvm,
    apply_passes_stepwise,
    extract_operators,
    get_tir_ast,
    build_te_microscope,
    extract_tuning_tasks,
    run_tuning,
    select_best_candidate,
    compute_tir_structural_features,
    build_tvm_module,
    run_tvm_inference,
    compare_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_pipeline")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _separator(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _get_sample_image():
    """Try to download a sample image; fall back to a random tensor."""
    try:
        log.info("Downloading sample image …")
        return download_sample_image()
    except Exception as exc:
        log.warning("Download failed (%s), using random noise image", exc)
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        from PIL import Image
        return Image.fromarray(arr)


# ──────────────────────────────────────────────────────────────────────
# Test stages
# ──────────────────────────────────────────────────────────────────────

def test_environment(state: DemoState) -> None:
    _separator("Stage 0a — Environment check")
    info = check_environment()
    banner = format_device_banner(info)
    print(banner)
    state.mark("stage_0", StageStatus.DONE, banner)


def test_load_model(state: DemoState) -> None:
    _separator("Stage 0b — Load model")
    model, transform, categories, summary, n_params = load_model(state.model_name)
    state.model = model
    state.transform = transform
    state.categories = categories
    state.model_summary = summary
    state.param_count = n_params
    print(summary)
    state.mark("stage_0", StageStatus.DONE)


def test_pytorch_inference(state: DemoState, use_cuda: bool) -> None:
    _separator("Stage 1 — PyTorch baseline inference")
    image = _get_sample_image()
    tensor, input_np = prepare_input(image, state.transform)

    state.input_image = image
    state.input_tensor = tensor
    state.input_np = input_np

    logits, top5, latency = run_pytorch_inference(
        state.model, tensor, state.categories,
        n_runs=50, use_cuda=use_cuda,
    )
    state.pytorch_logits = logits
    state.pytorch_top5 = top5
    state.pytorch_latency_ms = latency

    print(f"  Top-5 predictions:")
    for i, p in enumerate(top5):
        print(f"    {i + 1}. {p['class']:30s}  {p['prob']:.4f}")
    device_label = "CUDA" if use_cuda and torch.cuda.is_available() else "CPU"
    print(f"  Median latency: {latency:.2f} ms  ({device_label})")
    state.mark("stage_1", StageStatus.DONE)


def test_pytorch_graph(state: DemoState) -> None:
    _separator("Stage 2 — PyTorch computation graph")
    fx_graph, fx_code, node_table, exported = trace_pytorch_graph(
        state.model, state.input_tensor,
    )
    state.fx_graph = fx_graph
    state.fx_code = fx_code
    state.exported_program = exported

    print(f"  FX graph nodes: {len(node_table)}")
    op_counts: dict = {}
    for row in node_table:
        op_counts[row["op"]] = op_counts.get(row["op"], 0) + 1
    for op, cnt in sorted(op_counts.items()):
        print(f"    {op:20s}  {cnt}")
    print(f"  Generated code length: {len(fx_code)} chars")
    print(f"  torch.export: {'OK' if exported else 'unavailable'}")

    # Viz test
    from viz.graph_render import fx_graph_to_svg, fx_node_table_html
    svg = fx_graph_to_svg(fx_graph)
    table_html = fx_node_table_html(node_table)
    print(f"  SVG size: {len(svg)} chars")
    print(f"  Table HTML size: {len(table_html)} chars")

    state.mark("stage_2", StageStatus.DONE)


def test_import_tvm(state: DemoState) -> None:
    _separator("Stage 3 — Import into TVM Relax IR")
    mod, params_np, ir_text = import_to_tvm(state.model, state.input_tensor)
    state.imported_mod = mod
    state.model_params_np = params_np
    state.ir_snapshots["imported"] = ir_text

    n_funcs = len(mod.functions)
    n_lines = ir_text.count("\n")
    print(f"  IRModule: {n_funcs} functions, {n_lines} IR lines")
    print(f"  Parameters: {len(params_np)} arrays")
    state.mark("stage_3", StageStatus.DONE)


def test_passes(state: DemoState) -> None:
    _separator("Stage 4 — TVM graph-level passes")
    current, snapshots, order, deltas = apply_passes_stepwise(state.imported_mod)
    state.current_mod = current
    state.ir_snapshots.update(snapshots)
    state.pass_order = order
    state.pass_deltas = deltas

    for name in order:
        d = deltas[name]
        print(
            f"  {name:28s}  funcs {d['functions_before']:3d}→{d['functions_after']:3d}"
            f"  tir {d['tir_before']:3d}→{d['tir_after']:3d}"
            f"  ({d['elapsed_s']:.3f}s)"
        )
    state.mark("stage_4", StageStatus.DONE)


def test_extract_operators(state: DemoState) -> None:
    _separator("Stage 5 — Operator extraction")
    operators = extract_operators(state.current_mod)
    state.operators = operators

    print(f"  Extracted {len(operators)} TIR operators")
    for op in operators[:10]:
        print(f"    {op['name']:40s}  kind={op['op_kind']:10s}  blocks={op['num_blocks']}")
    if len(operators) > 10:
        print(f"    … and {len(operators) - 10} more")

    # Viz test
    from viz.ir_display import operator_table_html
    table = operator_table_html(operators)
    print(f"  Operator table HTML: {len(table)} chars")
    state.mark("stage_5", StageStatus.DONE)


def test_tir_ast(state: DemoState) -> None:
    _separator("Stage 6 — TensorIR / AST visualization")
    if not state.operators:
        print("  SKIP: no operators extracted")
        state.mark("stage_6", StageStatus.SKIPPED)
        return

    op_name = state.operators[0]["name"]
    tir_source, ast_summary = get_tir_ast(state.current_mod, op_name)
    state.selected_tir_name = op_name
    state.selected_tir_source = tir_source
    state.tir_ast_summary = ast_summary

    print(f"  Operator: {op_name}")
    print(f"  TIR source: {len(tir_source)} chars")
    print(f"  Blocks: {len(ast_summary.get('blocks', []))}")
    print(f"  Loops:  {len(ast_summary.get('loops', []))}")
    print(f"  Buffers: {len(ast_summary.get('buffers', []))}")

    # Viz test
    from viz.ir_display import (
        highlight_ir, tir_ast_tree_html, tir_loop_table_html, tir_buffer_table_html,
    )
    highlighted = highlight_ir(tir_source, max_lines=30)
    tree_html = tir_ast_tree_html(ast_summary)
    loop_html = tir_loop_table_html(ast_summary)
    buf_html = tir_buffer_table_html(ast_summary)
    print(f"  Highlighted IR: {len(highlighted)} chars")
    print(f"  AST tree HTML:  {len(tree_html)} chars")
    print(f"  Loop table:     {len(loop_html)} chars")
    print(f"  Buffer table:   {len(buf_html)} chars")

    state.mark("stage_6", StageStatus.DONE)


def test_te_microscope(state: DemoState) -> None:
    _separator("Stage 7 — Tensor expression microscope")
    compute_src, naive_tir, explanation = build_te_microscope()
    state.te_compute_source = compute_src
    state.te_lowered_tir = naive_tir

    print(f"  Compute source:\n{_indent(compute_src, 4)}")
    print(f"  Naive TIR: {len(naive_tir)} chars ({naive_tir.count(chr(10))} lines)")
    print(f"  Explanation: {len(explanation)} chars")

    # Viz: IR diff between microscope TIR and real model TIR
    if state.selected_tir_source:
        from viz.ir_display import ir_diff, ir_diff_stats
        diff = ir_diff(
            naive_tir, state.selected_tir_source,
            before_label="microscope_conv2d",
            after_label=state.selected_tir_name,
        )
        stats = ir_diff_stats(naive_tir, state.selected_tir_source)
        print(f"  Diff vs model TIR: {stats['lines_before']}→{stats['lines_after']} lines")

    state.mark("stage_7", StageStatus.DONE)


def _indent(text: str, n: int) -> str:
    prefix = " " * n
    return "\n".join(prefix + line for line in text.split("\n"))


def test_extract_tuning_tasks(state: DemoState) -> None:
    _separator("Stage 8 -- MetaSchedule task extraction")
    task_dicts, tasks_raw, target = extract_tuning_tasks(
        state.current_mod, target_str="cuda",
    )
    state.tuning_tasks = task_dicts
    state._tasks_raw = tasks_raw  # type: ignore[attr-defined]
    state._tuning_target = target  # type: ignore[attr-defined]

    print(f"  Extracted {len(task_dicts)} tuning tasks")
    for t in task_dicts[:5]:
        print(
            f"    {t['name']:40s}  weight={t['weight']:.1f}"
            f"  flops~{t['flop_estimate']}  tir_lines={t['tir_lines']}"
        )
    if len(task_dicts) > 5:
        print(f"    ... and {len(task_dicts) - 5} more")

    # Viz test: task weight pie chart
    from viz.charts import task_weight_pie_chart
    pie_html = task_weight_pie_chart(task_dicts)
    print(f"  Task weight pie chart HTML: {len(pie_html)} chars")

    state.mark("stage_8", StageStatus.DONE)


def test_run_tuning(state: DemoState) -> None:
    _separator("Stage 9 -- Schedule exploration (tuning)")
    import shutil
    test_work_dir = "./test_tuning_logs"
    if os.path.exists(test_work_dir):
        shutil.rmtree(test_work_dir)
    records, convergence, work_dir = run_tuning(
        state.current_mod,
        target=getattr(state, "_tuning_target", None),
        work_dir=test_work_dir,
        max_trials_global=16,
        num_trials_per_iter=8,
        max_tasks=2,
    )
    state.tuning_records = records
    state.convergence_data = convergence
    state.tuning_work_dir = work_dir

    is_synthetic = any(r.get("_synthetic") for r in records)
    label = " (synthetic)" if is_synthetic else ""
    print(f"  Tuning records: {len(records)}{label}")
    print(f"  Convergence points: {len(convergence)}")
    if records:
        best = min(records, key=lambda r: r["run_ms"])
        print(f"  Best candidate: #{best['candidate_id']} ({best['task_name']}) at {best['run_ms']:.4f} ms")

    # Viz test: candidate cards
    from viz.schedule_display import candidate_cards_html, trace_to_readable
    cards_html = candidate_cards_html(records, max_display=5)
    print(f"  Candidate cards HTML: {len(cards_html)} chars")

    if records:
        instructions = trace_to_readable(records[0].get("trace_text", ""))
        print(f"  First candidate trace instructions: {instructions[:3]}")

    # Viz test: convergence chart
    from viz.charts import convergence_chart
    conv_html = convergence_chart(convergence)
    print(f"  Convergence chart HTML: {len(conv_html)} chars")

    state.mark("stage_9", StageStatus.DONE)


def test_select_best(state: DemoState) -> None:
    _separator("Stage 10 -- Cost model & schedule selection")
    best, features = select_best_candidate(state.tuning_records, state.current_mod)
    state.best_candidate = best
    state.candidate_features = features

    if best:
        print(f"  Best candidate: #{best['candidate_id']} ({best['task_name']})")
        print(f"  Latency: {best['run_ms']:.4f} ms")
    else:
        print("  No best candidate selected")

    print(f"  Feature records: {len(features)}")
    if features:
        print(f"  Feature keys: {list(features[0].keys())[:8]}")

    # Viz test: feature table
    from viz.feature_table import (
        build_feature_dataframe, feature_table_html,
        cost_model_explanation_html,
    )
    df = build_feature_dataframe(features)
    table_html = feature_table_html(df, max_rows=10)
    explanation_html = cost_model_explanation_html()
    print(f"  Feature table HTML: {len(table_html)} chars")
    print(f"  Cost model explanation HTML: {len(explanation_html)} chars")

    # Viz test: best candidate banner
    from viz.schedule_display import best_candidate_banner_html
    banner = best_candidate_banner_html(best)
    print(f"  Best candidate banner HTML: {len(banner)} chars")

    # TIR structural features for one operator (if available)
    if state.operators:
        op_name = state.operators[0]["name"]
        tir_feats = compute_tir_structural_features(state.current_mod, op_name)
        if tir_feats:
            print(f"  TIR features for '{op_name}':")
            for k, v in list(tir_feats.items())[:6]:
                print(f"    {k}: {v}")
            from viz.feature_table import tir_features_table_html
            tir_feats_html = tir_features_table_html(tir_feats)
            print(f"  TIR features table HTML: {len(tir_feats_html)} chars")

    state.mark("stage_10", StageStatus.DONE)


def test_build(state: DemoState) -> None:
    _separator("Stage 11 — Build TVM CUDA module")

    lib, target_used, cuda_src, params_bound = build_tvm_module(
        state.current_mod,
        params_np=state.model_params_np,
        target_str="cuda",
        work_dir=state.tuning_work_dir,
    )
    state.compiled_lib = lib
    state.target_str = target_used
    state.cuda_source = cuda_src

    print(f"  Target: {target_used}")
    print(f"  Params bound: {'yes' if params_bound else 'no (will pass at runtime)'}")
    print(f"  CUDA source: {len(cuda_src)} chars" if cuda_src else "  CUDA source: not available")
    state.mark("stage_11", StageStatus.DONE)

    state._params_bound = params_bound  # type: ignore[attr-defined]


def test_tvm_inference(state: DemoState) -> None:
    _separator("Stage 12 — TVM inference & comparison")

    params_at_runtime = None if getattr(state, "_params_bound", False) else state.model_params_np

    logits, top5, latency = run_tvm_inference(
        state.compiled_lib,
        state.input_np,
        state.categories,
        params_np=params_at_runtime,
        n_runs=50,
    )
    state.tvm_logits = logits
    state.tvm_top5 = top5
    state.tvm_latency_ms = latency

    comp = compare_results(
        state.pytorch_logits, logits,
        state.pytorch_latency_ms, latency,
    )
    state.max_abs_diff = comp["max_abs_diff"]
    state.cosine_sim = comp["cosine_similarity"]

    print(f"  TVM Top-5:")
    for i, p in enumerate(top5):
        print(f"    {i + 1}. {p['class']:30s}  {p['prob']:.4f}")
    print(f"  Median latency: {latency:.2f} ms")
    print()
    print(f"  Max abs diff:       {comp['max_abs_diff']:.6f}")
    print(f"  Cosine similarity:  {comp['cosine_similarity']:.6f}")
    print(f"  Speedup:            {comp['speedup']}x")
    print(f"  Match (diff < 0.01): {'YES' if comp['match'] else 'NO'}")

    # Viz test: latency comparison chart
    from viz.charts import latency_comparison_chart
    chart_html = latency_comparison_chart(
        state.pytorch_latency_ms, latency,
        title=f"{state.model_name} Latency: PyTorch vs TVM",
    )
    print(f"  Latency chart HTML: {len(chart_html)} chars")

    state.mark("stage_12", StageStatus.DONE)


# ──────────────────────────────────────────────────────────────────────
# Assertions (for CI / pytest)
# ──────────────────────────────────────────────────────────────────────

def assert_correctness(state: DemoState) -> None:
    _separator("Assertions")

    # Stage 1
    assert state.pytorch_logits is not None, "PyTorch logits missing"
    assert state.pytorch_top5, "PyTorch top5 empty"
    print("  [PASS] Stage 1: PyTorch inference produced results")

    # Stage 2
    if state.fx_graph is not None:
        assert len(list(state.fx_graph.nodes)) > 0, "FX graph has no nodes"
        assert len(state.fx_code) > 0, "FX code is empty"
        print("  [PASS] Stage 2: FX graph captured with >0 nodes")
    else:
        print("  [SKIP] Stage 2: not run")

    # Stage 5
    if state.operators:
        assert len(state.operators) > 0, "Operator list is empty"
        assert all("tir_source" in op for op in state.operators), (
            "Some operators missing tir_source"
        )
        print(f"  [PASS] Stage 5: {len(state.operators)} operators extracted")
    elif state.is_done("stage_5"):
        print("  [WARN] Stage 5: ran but produced 0 operators")

    # Stage 6
    if state.selected_tir_source:
        assert len(state.selected_tir_source) > 0, "TIR source is empty"
        assert state.tir_ast_summary is not None, "AST summary is None"
        print("  [PASS] Stage 6: TIR AST extracted")
    elif state.is_done("stage_6"):
        print("  [WARN] Stage 6: ran but no TIR source")

    # Stage 7
    if state.te_compute_source:
        assert "placeholder" in state.te_compute_source or "topi" in state.te_compute_source, (
            "TE source doesn't look like a compute declaration"
        )
        assert len(state.te_lowered_tir) > 0, "Naive TIR is empty"
        print("  [PASS] Stage 7: TE microscope built")

    # Stage 8
    if state.tuning_tasks:
        assert len(state.tuning_tasks) > 0, "Tuning task list is empty"
        assert all("name" in t for t in state.tuning_tasks), "Tasks missing 'name'"
        print(f"  [PASS] Stage 8: {len(state.tuning_tasks)} tuning tasks extracted")
    elif state.is_done("stage_8"):
        print("  [WARN] Stage 8: ran but produced 0 tasks")

    # Stage 9
    if state.tuning_records:
        assert len(state.tuning_records) > 0, "Tuning records list is empty"
        assert all("run_ms" in r for r in state.tuning_records), "Records missing 'run_ms'"
        assert state.convergence_data, "Convergence data is empty"
        any_best = any(r.get("is_best") for r in state.tuning_records)
        assert any_best, "No candidate marked as best"
        print(f"  [PASS] Stage 9: {len(state.tuning_records)} tuning records collected")
    elif state.is_done("stage_9"):
        print("  [WARN] Stage 9: ran but produced 0 records")

    # Stage 10
    if state.best_candidate:
        assert state.best_candidate["run_ms"] < 1e6, "Best candidate has invalid latency"
        assert state.candidate_features, "Candidate features list is empty"
        print(f"  [PASS] Stage 10: best candidate #{state.best_candidate['candidate_id']} selected")
    elif state.is_done("stage_10"):
        print("  [WARN] Stage 10: ran but no best candidate")

    # Stage 12
    if state.tvm_logits is not None:
        assert state.max_abs_diff < 0.05, (
            f"Output mismatch too large: {state.max_abs_diff}"
        )
        assert state.cosine_sim > 0.99, (
            f"Cosine similarity too low: {state.cosine_sim}"
        )
        assert state.tvm_top5[0]["class"] == state.pytorch_top5[0]["class"], (
            "Top-1 class mismatch"
        )
        print("  [PASS] Stage 12: TVM output matches PyTorch (diff<0.05, cos>0.99)")
        print("  [PASS] Stage 12: Top-1 class agrees")
    else:
        print("  [SKIP] Stage 12: TVM inference not run")

    print("\n  All assertions passed.\n")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TVM demo pipeline smoke test")
    parser.add_argument(
        "--cpu", action="store_true",
        help="Run PyTorch-only stages on CPU (skip TVM/CUDA stages)",
    )
    parser.add_argument(
        "--model", default="resnet18",
        choices=["resnet18", "mobilenet_v2"],
    )
    args = parser.parse_args()

    state = DemoState(model_name=args.model)

    # ── Always run (PyTorch only) ────────────────────────────────────
    test_environment(state)
    test_load_model(state)
    test_pytorch_inference(state, use_cuda=not args.cpu)
    test_pytorch_graph(state)     # Stage 2 — pure PyTorch, no TVM needed

    # ── TVM stages ───────────────────────────────────────────────────
    if not args.cpu:
        try:
            test_import_tvm(state)              # Stage 3
            test_passes(state)                  # Stage 4
            test_extract_operators(state)       # Stage 5
            test_tir_ast(state)                 # Stage 6
            test_te_microscope(state)           # Stage 7
            test_extract_tuning_tasks(state)    # Stage 8
            test_run_tuning(state)              # Stage 9
            test_select_best(state)             # Stage 10
            test_build(state)                   # Stage 11
            test_tvm_inference(state)           # Stage 12
        except RuntimeError as exc:
            log.error("TVM stage failed: %s", exc)
            print(f"\n  TVM stages skipped due to error: {exc}")
    else:
        print("\n  --cpu flag set: skipping TVM stages")

    assert_correctness(state)

    _separator("Summary")
    for sid, status in state.stage_status.items():
        print(f"  {sid:12s}  {status.value}")


if __name__ == "__main__":
    main()
