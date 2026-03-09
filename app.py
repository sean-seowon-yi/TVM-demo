#!/usr/bin/env python
"""TVM Demo -- Interactive Gradio app (Pass 4).

Wires all backend pipeline functions and visualization utilities into
a tabbed Gradio Blocks interface.  Each tab corresponds to one or more
demo stages from PLAN.md Section 8.

Launch:
    python app.py                         # local
    python app.py --share                 # Gradio built-in tunnel
    python app.py --port 7860 --host 0.0.0.0  # for Cloudflare tunnel
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import gradio as gr
import numpy as np

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from backend.state import DemoState, StageStatus
from backend.helpers import format_device_banner, get_device_info
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

import concurrent.futures
import gc

# Disable Python's cyclic garbage collector.  TVM's FFI wrapper objects
# can contain C++ pointers that become dangling after pass transforms.
# When Python's cyclic GC tries to finalize these objects it calls
# TVMFFIObjectDecRef on a freed pointer, causing a SIGSEGV.  Disabling
# the cyclic GC avoids this; normal reference-counting still works, so
# non-cyclic objects are freed immediately.  The only cost is potential
# leaks from reference cycles, which is acceptable for a demo app.
gc.disable()

_tvm_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="tvm-worker",
)

def _on_tvm_thread(fn, *args, **kwargs):
    """Run *fn* on a single dedicated thread so all TVM FFI objects are
    created and destroyed on the same OS thread.  Gradio's async event-loop
    threads never touch TVM C++ pointers, preventing the TVMFFIObjectDecRef
    segfault."""
    future = _tvm_executor.submit(fn, *args, **kwargs)
    return future.result()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("app")

STATE = DemoState()

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _err(msg: str) -> str:
    return f'<div style="color:#9B1C1C;padding:8px 12px;background:#FDE8E8;border-left:3px solid #E53E3E;border-radius:4px;font-size:13px">{msg}</div>'


def _ok(msg: str) -> str:
    return f'<div style="color:#276749;padding:8px 12px;background:#F0FFF4;border-left:3px solid #48BB78;border-radius:4px;font-size:13px">{msg}</div>'


def _info(msg: str) -> str:
    return f'<div style="color:#2B6CB0;padding:8px 12px;background:#EBF8FF;border-left:3px solid #4299E1;border-radius:4px;font-size:13px">{msg}</div>'



def _progress_html() -> str:
    names = [
        "0:Load", "1:Infer", "2:FX", "3:Import", "4:Passes",
        "5:Ops", "6:TIR", "7:TE", "8:Tasks", "9:Tune",
        "10:Cost", "11:Build", "12:Compare",
    ]
    parts = []
    for i, name in enumerate(names):
        st = STATE.stage_status.get(f"stage_{i}", StageStatus.PENDING)
        bg, fg = {
            StageStatus.DONE: ("#48BB78", "#fff"),
            StageStatus.FAILED: ("#E53E3E", "#fff"),
            StageStatus.RUNNING: ("#ED8936", "#fff"),
            StageStatus.SKIPPED: ("#A0AEC0", "#fff"),
        }.get(st, ("#EDF2F7", "#718096"))
        parts.append(
            f'<span style="background:{bg};color:{fg};padding:2px 8px;'
            f'border-radius:12px;font-size:11px;font-weight:500">'
            f'{name}</span>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:5px;padding:4px 0;'
        'align-items:center">'
        + "".join(parts)
        + '</div>'
    )


# ──────────────────────────────────────────────────────────────────────
# Tab 1 — Task & Input (Stages 0-1)
# ──────────────────────────────────────────────────────────────────────

def run_stage_0_1(model_choice: str, image) -> tuple:
    global STATE
    STATE = DemoState(model_name=model_choice)

    try:
        STATE.mark("stage_0", StageStatus.RUNNING)
        env = check_environment()
        banner = format_device_banner(env)

        model, transform, categories, summary, n_params = load_model(model_choice)
        STATE.model = model
        STATE.transform = transform
        STATE.categories = categories
        STATE.model_summary = summary
        STATE.param_count = n_params
        STATE.mark("stage_0", StageStatus.DONE, banner)
    except Exception as exc:
        STATE.mark("stage_0", StageStatus.FAILED, str(exc))
        return _err(f"Stage 0 failed: {exc}"), "", None, "", "", _progress_html()

    try:
        STATE.mark("stage_1", StageStatus.RUNNING)
        if image is None:
            from backend.helpers import download_sample_image
            image = download_sample_image()

        tensor, input_np = prepare_input(image, STATE.transform)
        STATE.input_image = image
        STATE.input_tensor = tensor
        STATE.input_np = input_np

        import torch
        use_cuda = torch.cuda.is_available()
        logits, top5, latency = run_pytorch_inference(
            STATE.model, tensor, STATE.categories, n_runs=50, use_cuda=use_cuda,
        )
        STATE.pytorch_logits = logits
        STATE.pytorch_top5 = top5
        STATE.pytorch_latency_ms = latency
        STATE.mark("stage_1", StageStatus.DONE)

        top5_md = "| # | Class | Probability |\n|---|-------|-------------|\n"
        for i, p in enumerate(top5):
            top5_md += f"| {i+1} | {p['class']} | {p['prob']:.4f} |\n"
        device_label = "CUDA" if use_cuda else "CPU"
        latency_text = f"**Median latency**: {latency:.2f} ms ({device_label}, 50 runs)"

        return (
            _ok(f"Model loaded: {model_choice} ({n_params:,} params)"),
            f"```\n{banner}\n```",
            image,
            top5_md,
            latency_text,
            _progress_html(),
        )
    except Exception as exc:
        STATE.mark("stage_1", StageStatus.FAILED, str(exc))
        return _err(f"Stage 1 failed: {exc}"), "", None, "", "", _progress_html()


# ──────────────────────────────────────────────────────────────────────
# Tab 2 — PyTorch Graph (Stage 2)
# ──────────────────────────────────────────────────────────────────────

def run_stage_2() -> tuple:
    if STATE.model is None:
        return _err("Run Stage 0-1 first"), "", "", _progress_html()

    try:
        STATE.mark("stage_2", StageStatus.RUNNING)
        fx_graph, fx_code, node_table, exported = trace_pytorch_graph(
            STATE.model, STATE.input_tensor,
        )
        STATE.fx_graph = fx_graph
        STATE._fx_node_count = len(node_table)
        STATE.fx_code = fx_code
        STATE.exported_program = exported
        STATE.mark("stage_2", StageStatus.DONE)

        from viz.graph_render import fx_graph_to_svg, fx_node_table_html
        svg = fx_graph_to_svg(fx_graph)
        table_html = fx_node_table_html(node_table)

        status = _ok(f"FX graph: {len(node_table)} nodes captured. torch.export: {'OK' if exported else 'unavailable'}")
        return status, svg, table_html, _progress_html()
    except Exception as exc:
        STATE.mark("stage_2", StageStatus.FAILED, str(exc))
        return _err(f"Stage 2 failed: {exc}"), "", "", _progress_html()


# ──────────────────────────────────────────────────────────────────────
# Tab 3 — TVM IR Import (Stage 3)
# ──────────────────────────────────────────────────────────────────────

def run_stage_3() -> tuple:
    if STATE.model is None:
        return _err("Run Stage 0-1 first"), "", "", _progress_html()

    def _impl():
        STATE.mark("stage_3", StageStatus.RUNNING)
        mod, params_np, ir_text = import_to_tvm(STATE.model, STATE.input_tensor)
        STATE.imported_mod = mod
        STATE.model_params_np = params_np
        STATE.ir_snapshots["imported"] = ir_text
        num_funcs = len(mod.functions)
        STATE.imported_mod_num_funcs = num_funcs
        STATE.mark("stage_3", StageStatus.DONE)

        from viz.graph_render import relax_callgraph_to_svg
        callgraph_svg = relax_callgraph_to_svg(mod)

        lines = ir_text.split("\n")
        truncated = "\n".join(lines[:300]) + (f"\n# ... ({len(lines)-300} more lines)" if len(lines) > 300 else "")

        status = _ok(f"IRModule: {num_funcs} functions, {len(params_np)} parameter arrays, {len(lines)} IR lines")
        return status, truncated, callgraph_svg

    try:
        status, truncated, callgraph_svg = _on_tvm_thread(_impl)
        return status, truncated, callgraph_svg, _progress_html()
    except Exception as exc:
        STATE.mark("stage_3", StageStatus.FAILED, str(exc))
        return _err(f"Stage 3 failed: {exc}"), "", "", _progress_html()


# ──────────────────────────────────────────────────────────────────────
# Tab 4 — TVM Passes (Stage 4)
# ──────────────────────────────────────────────────────────────────────

def run_stage_4() -> tuple:
    if STATE.imported_mod is None:
        return _err("Run Stage 3 first"), "", "", _progress_html()

    def _impl():
        STATE.mark("stage_4", StageStatus.RUNNING)
        current, snapshots, order, deltas = apply_passes_stepwise(STATE.imported_mod)
        STATE.current_mod = current
        STATE.imported_mod = None
        STATE.ir_snapshots.update(snapshots)
        STATE.pass_order = order
        STATE.pass_deltas = deltas
        STATE.mark("stage_4", StageStatus.DONE)

        from viz.ir_display import format_all_pass_deltas
        delta_table = format_all_pass_deltas(order, deltas)
        final_ir = snapshots.get(order[-1], "") if order else ""

        lines = final_ir.split("\n")
        truncated = "\n".join(lines[:250]) + (f"\n# ... ({len(lines)-250} more lines)" if len(lines) > 250 else "")

        status = _ok(f"{len(order)} passes applied successfully")
        return status, delta_table, truncated

    try:
        status, delta_table, truncated = _on_tvm_thread(_impl)
        return status, delta_table, truncated, _progress_html()
    except Exception as exc:
        STATE.mark("stage_4", StageStatus.FAILED, str(exc))
        return _err(f"Stage 4 failed: {exc}"), "", "", _progress_html()


def view_pass_diff(pass_name: str) -> str:
    if not pass_name or not STATE.ir_snapshots:
        return ""
    idx = STATE.pass_order.index(pass_name) if pass_name in STATE.pass_order else -1
    if idx < 0:
        return ""
    before_key = STATE.pass_order[idx - 1] if idx > 0 else "imported"
    before = STATE.ir_snapshots.get(before_key, "")
    after = STATE.ir_snapshots.get(pass_name, "")
    from viz.ir_display import ir_diff
    return ir_diff(before, after, before_label=before_key, after_label=pass_name)


# ──────────────────────────────────────────────────────────────────────
# Tab 5 — Extracted Operators (Stage 5)
# ──────────────────────────────────────────────────────────────────────

def run_stage_5() -> tuple:
    if STATE.current_mod is None:
        return _err("Run Stage 4 first"), "", _progress_html()

    def _impl():
        STATE.mark("stage_5", StageStatus.RUNNING)
        operators = extract_operators(STATE.current_mod)
        STATE.operators = operators
        STATE.mark("stage_5", StageStatus.DONE)

        from viz.ir_display import operator_table_html
        table = operator_table_html(operators)
        status = _ok(f"Extracted {len(operators)} TIR operators")
        return status, table

    try:
        status, table = _on_tvm_thread(_impl)
        return status, table, _progress_html()
    except Exception as exc:
        STATE.mark("stage_5", StageStatus.FAILED, str(exc))
        return _err(f"Stage 5 failed: {exc}"), "", _progress_html()


# ──────────────────────────────────────────────────────────────────────
# Tab 6 — TensorIR / AST (Stage 6)
# ──────────────────────────────────────────────────────────────────────

def get_op_names() -> list:
    if STATE.operators:
        return [
            f"{op['name']}  [{op.get('op_kind', 'other')}]"
            for op in STATE.operators
        ]
    return []


def _strip_op_kind_label(display_name: str) -> str:
    """Remove the ' [kind]' suffix added by get_op_names for display."""
    if "  [" in display_name:
        return display_name.rsplit("  [", 1)[0]
    return display_name


def run_stage_6(op_display_name: str) -> tuple:
    if STATE.current_mod is None:
        return _err("Run Stage 5 first"), "", "", "", _progress_html()
    if not op_display_name:
        return _info("Select an operator from the dropdown"), "", "", "", _progress_html()

    op_name = _strip_op_kind_label(op_display_name)

    def _impl():
        STATE.mark("stage_6", StageStatus.RUNNING)
        tir_source, ast_summary = get_tir_ast(STATE.current_mod, op_name)
        STATE.selected_tir_name = op_name
        STATE.selected_tir_source = tir_source
        STATE.tir_ast_summary = ast_summary
        STATE.mark("stage_6", StageStatus.DONE)

        from viz.ir_display import tir_ast_tree_html, tir_loop_table_html
        tree_html = tir_ast_tree_html(ast_summary)
        loop_html = tir_loop_table_html(ast_summary)

        lines = tir_source.split("\n")
        truncated = "\n".join(lines[:200]) + (f"\n# ... ({len(lines)-200} more lines)" if len(lines) > 200 else "")

        n_blocks = len(ast_summary.get("blocks", []))
        n_loops = len(ast_summary.get("loops", []))
        n_bufs = len(ast_summary.get("buffers", []))
        n_block_iters = sum(1 for lp in ast_summary.get("loops", []) if lp.get("source") == "block_iter")
        n_for_loops = n_loops - n_block_iters
        iter_desc = []
        if n_for_loops:
            iter_desc.append(f"{n_for_loops} for-loops")
        if n_block_iters:
            iter_desc.append(f"{n_block_iters} block iterators")
        iter_str = ", ".join(iter_desc) if iter_desc else "no iteration"
        status = _ok(f"TIR for '{op_name}': {n_blocks} blocks, {iter_str}, {n_bufs} buffers")
        return status, truncated, tree_html, loop_html

    try:
        status, truncated, tree_html, loop_html = _on_tvm_thread(_impl)
        return status, truncated, tree_html, loop_html, _progress_html()
    except Exception as exc:
        STATE.mark("stage_6", StageStatus.FAILED, str(exc))
        return _err(f"Stage 6 failed: {exc}"), "", "", "", _progress_html()


# ──────────────────────────────────────────────────────────────────────
# Tab 7 — Tensor Expression Microscope (Stage 7)
# ──────────────────────────────────────────────────────────────────────

def run_stage_7() -> tuple:
    def _impl():
        STATE.mark("stage_7", StageStatus.RUNNING)
        compute_src, naive_tir, explanation = build_te_microscope()
        STATE.te_compute_source = compute_src
        STATE.te_lowered_tir = naive_tir
        STATE.mark("stage_7", StageStatus.DONE)

        lines = naive_tir.split("\n")
        truncated_tir = "\n".join(lines[:160]) + (f"\n# ... ({len(lines)-160} more lines)" if len(lines) > 160 else "")

        status = _ok("Microscope operator built: conv2d 64x64x3x3")
        return status, compute_src, truncated_tir, explanation

    try:
        status, compute_src, truncated_tir, explanation = _on_tvm_thread(_impl)
        return status, compute_src, truncated_tir, explanation, _progress_html()
    except Exception as exc:
        STATE.mark("stage_7", StageStatus.FAILED, str(exc))
        return _err(f"Stage 7 failed: {exc}"), "", "", "", _progress_html()


# ──────────────────────────────────────────────────────────────────────
# Tab 8 — Schedule Search (Stages 8-9)
# ──────────────────────────────────────────────────────────────────────

def run_stage_8_9(max_trials: int) -> tuple:
    if STATE.current_mod is None:
        return _err("Run Stage 4 first"), "", "", "", "", _progress_html()

    def _impl():
        STATE.mark("stage_8", StageStatus.RUNNING)
        task_dicts, tasks_raw, target = extract_tuning_tasks(STATE.current_mod, target_str="cuda")
        STATE.tuning_tasks = task_dicts
        STATE._tasks_raw = tasks_raw
        STATE._tuning_target = target
        STATE.mark("stage_8", StageStatus.DONE)

        task_info = _ok(f"Extracted {len(task_dicts)} tuning tasks")

        STATE.mark("stage_9", StageStatus.RUNNING)
        records, convergence, work_dir = run_tuning(
            STATE.current_mod,
            target=STATE._tuning_target,
            max_trials_global=max(4, int(max_trials)),
            num_trials_per_iter=min(16, max(4, int(max_trials) // 4)),
            max_tasks=3,
        )
        STATE.tuning_records = records
        STATE.convergence_data = convergence
        STATE.mark("stage_9", StageStatus.DONE)

        from viz.schedule_display import candidate_cards_html
        from viz.charts import convergence_chart, task_weight_pie_chart
        cards = candidate_cards_html(records, max_display=15)
        conv_html = convergence_chart(convergence) if convergence else _info("No convergence data")
        pie_html = task_weight_pie_chart(task_dicts)

        is_synthetic = any(r.get("_synthetic") for r in records)
        label = " (synthetic -- real tuning unavailable)" if is_synthetic else ""
        tune_info = _ok(f"{len(records)} candidates measured{label}")

        return task_info, tune_info, cards, conv_html, pie_html

    try:
        task_info, tune_info, cards, conv_html, pie_html = _on_tvm_thread(_impl)
        return task_info, tune_info, cards, conv_html, pie_html, _progress_html()
    except Exception as exc:
        failed_stage = "stage_8" if not STATE.tuning_tasks else "stage_9"
        STATE.mark(failed_stage, StageStatus.FAILED, str(exc))
        if STATE.tuning_tasks:
            task_info = _ok(f"Extracted {len(STATE.tuning_tasks)} tuning tasks")
            return task_info, _err(f"Stage 9 failed: {exc}"), "", "", "", _progress_html()
        return _err(f"Stage 8 failed: {exc}"), "", "", "", "", _progress_html()


# ──────────────────────────────────────────────────────────────────────
# Tab 9 — Cost Model & Selection (Stage 10)
# ──────────────────────────────────────────────────────────────────────

def run_stage_10() -> tuple:
    if not STATE.tuning_records:
        return _err("Run Stage 9 first"), "", "", "", _progress_html()

    def _impl():
        STATE.mark("stage_10", StageStatus.RUNNING)
        best, features = select_best_candidate(STATE.tuning_records, STATE.current_mod)
        STATE.best_candidate = best
        STATE.candidate_features = features
        STATE.mark("stage_10", StageStatus.DONE)

        from viz.schedule_display import best_candidate_banner_html
        from viz.feature_table import (
            build_feature_dataframe, feature_table_html,
            cost_model_explanation_html,
        )
        banner = best_candidate_banner_html(best)
        df = build_feature_dataframe(features)
        table = feature_table_html(df, max_rows=20)
        explanation = cost_model_explanation_html()

        tir_feats_html = ""
        if STATE.operators and STATE.current_mod is not None:
            try:
                op_name = STATE.operators[0]["name"]
                tir_feats = compute_tir_structural_features(STATE.current_mod, op_name)
                if tir_feats:
                    from viz.feature_table import tir_features_table_html
                    tir_feats_html = (
                        f'<div style="margin-top:12px"><b>TIR Structural Features for '
                        f'<code>{op_name}</code></b></div>'
                        + tir_features_table_html(tir_feats)
                    )
            except Exception:
                pass

        status = _ok(f"Best: #{best['candidate_id']} ({best['task_name']}) at {best['run_ms']:.4f} ms") if best else _info("No best candidate")
        return status, banner + tir_feats_html, table, explanation

    try:
        status, banner_html, table, explanation = _on_tvm_thread(_impl)
        return status, banner_html, table, explanation, _progress_html()
    except Exception as exc:
        STATE.mark("stage_10", StageStatus.FAILED, str(exc))
        return _err(f"Stage 10 failed: {exc}"), "", "", "", _progress_html()


# ──────────────────────────────────────────────────────────────────────
# Tab 10 — Build & Results (Stages 11-12)
# ──────────────────────────────────────────────────────────────────────

def run_stage_11_12() -> tuple:
    if STATE.current_mod is None:
        return _err("Run Stage 4 first"), "", "", "", "", "", _progress_html()

    def _impl():
        STATE.mark("stage_11", StageStatus.RUNNING)
        lib, target_used, cuda_src, params_bound = build_tvm_module(
            STATE.current_mod, params_np=STATE.model_params_np, target_str="cuda",
        )
        STATE.compiled_lib = lib
        STATE.target_str = target_used
        STATE.cuda_source = cuda_src
        STATE.mark("stage_11", StageStatus.DONE)

        build_info = _ok(f"Built for target: {target_used}. Params bound: {'yes' if params_bound else 'no'}")

        STATE.mark("stage_12", StageStatus.RUNNING)
        params_at_runtime = None if params_bound else STATE.model_params_np
        logits, top5, latency = run_tvm_inference(
            STATE.compiled_lib, STATE.input_np, STATE.categories,
            params_np=params_at_runtime, n_runs=50,
        )
        STATE.tvm_logits = logits
        STATE.tvm_top5 = top5
        STATE.tvm_latency_ms = latency

        comp = compare_results(
            STATE.pytorch_logits, logits,
            STATE.pytorch_latency_ms, latency,
        )
        STATE.max_abs_diff = comp["max_abs_diff"]
        STATE.cosine_sim = comp["cosine_similarity"]
        STATE.mark("stage_12", StageStatus.DONE)

        tvm_md = "| # | Class | Probability |\n|---|-------|-------------|\n"
        for i, p in enumerate(top5):
            tvm_md += f"| {i+1} | {p['class']} | {p['prob']:.4f} |\n"

        comparison_md = (
            f"| Metric | Value |\n|---|---|\n"
            f"| Max absolute diff | {comp['max_abs_diff']:.6f} |\n"
            f"| Cosine similarity | {comp['cosine_similarity']:.6f} |\n"
            f"| PyTorch latency | {comp['pytorch_ms']:.2f} ms |\n"
            f"| TVM latency | {comp['tvm_ms']:.2f} ms |\n"
            f"| Speedup | {comp['speedup']}x |\n"
            f"| Match (diff < 0.01) | {'YES' if comp['match'] else 'NO'} |\n"
        )

        from viz.charts import latency_comparison_chart
        chart_html = latency_comparison_chart(
            STATE.pytorch_latency_ms, latency,
            title=f"{STATE.model_name}: PyTorch vs TVM",
        )

        cuda_display = cuda_src[:3000] if cuda_src else "(CUDA source not available for this backend)"

        return build_info, tvm_md, comparison_md, chart_html, cuda_display

    try:
        build_info, tvm_md, comparison_md, chart_html, cuda_display = _on_tvm_thread(_impl)
        return build_info, tvm_md, comparison_md, chart_html, cuda_display, _ok("Pipeline complete!"), _progress_html()
    except Exception as exc:
        stage = "stage_11" if STATE.compiled_lib is None else "stage_12"
        STATE.mark(stage, StageStatus.FAILED, str(exc))
        if STATE.compiled_lib is not None:
            build_info = _ok(f"Built for target: {STATE.target_str}")
            return build_info, "", _err(f"Stage 12 failed: {exc}"), "", "", "", _progress_html()
        return _err(f"Stage 11 failed: {exc}"), "", "", "", "", "", _progress_html()


# ──────────────────────────────────────────────────────────────────────
# Tab 11 — Pipeline Timeline (Stage 13)
# ──────────────────────────────────────────────────────────────────────

def build_timeline() -> str:
    stages = [
        ("Stage 0", "Load Model", "stage_0", f"{STATE.model_name}, {STATE.param_count:,} params" if STATE.model else ""),
        ("Stage 1", "PyTorch Inference", "stage_1", f"Top-1: {STATE.pytorch_top5[0]['class']}, {STATE.pytorch_latency_ms:.2f} ms" if STATE.pytorch_top5 else ""),
        ("Stage 2", "FX Graph Capture", "stage_2", f"{STATE._fx_node_count} nodes" if STATE._fx_node_count else ""),
        ("Stage 3", "TVM Relax Import", "stage_3", f"{STATE.imported_mod_num_funcs} functions" if STATE.imported_mod_num_funcs else ""),
        ("Stage 4", "Graph-Level Passes", "stage_4", f"{len(STATE.pass_order)} passes" if STATE.pass_order else ""),
        ("Stage 5", "Operator Extraction", "stage_5", f"{len(STATE.operators)} TIR operators" if STATE.operators else ""),
        ("Stage 6", "TensorIR / AST", "stage_6", STATE.selected_tir_name or ""),
        ("Stage 7", "Tensor Expression", "stage_7", "conv2d microscope" if STATE.te_compute_source else ""),
        ("Stage 8", "Task Extraction", "stage_8", f"{len(STATE.tuning_tasks)} tasks" if STATE.tuning_tasks else ""),
        ("Stage 9", "Schedule Search", "stage_9", f"{len(STATE.tuning_records)} candidates" if STATE.tuning_records else ""),
        ("Stage 10", "Cost Model", "stage_10", f"Best: {STATE.best_candidate['run_ms']:.4f} ms" if STATE.best_candidate else ""),
        ("Stage 11", "Build CUDA Module", "stage_11", STATE.target_str or ""),
        ("Stage 12", "TVM Inference", "stage_12", f"{STATE.tvm_latency_ms:.2f} ms, {STATE.cosine_sim:.4f} cosine" if STATE.tvm_logits is not None else ""),
    ]

    paper_refs = [
        "Figure 2 top",
        "Section 1",
        "Section 3, Figure 3",
        "Figure 2",
        "Section 3",
        "Section 3 -> 4",
        "Section 4, Figure 13",
        "Section 4.1, Figure 5",
        "Section 5.1",
        "Section 5.3, Figure 12",
        "Section 5.2, Figure 13",
        "Figure 2 bottom",
        "Section 6, Figure 14",
    ]

    html_parts = ['<div style="max-width:700px;margin:0 auto;font-family:sans-serif">']
    for i, (label, title, sid, detail) in enumerate(stages):
        st = STATE.stage_status.get(sid, StageStatus.PENDING)
        color = {
            StageStatus.DONE: "#4CAF50",
            StageStatus.FAILED: "#F44336",
            StageStatus.RUNNING: "#FF9800",
            StageStatus.SKIPPED: "#607D8B",
        }.get(st, "#BDBDBD")

        icon = {StageStatus.DONE: "&#10003;", StageStatus.FAILED: "&#10007;", StageStatus.RUNNING: "&#9679;"}.get(st, "&#9675;")

        html_parts.append(
            f'<div style="display:flex;align-items:flex-start;margin:4px 0">'
            f'<div style="min-width:36px;text-align:center">'
            f'<span style="color:{color};font-size:18px">{icon}</span>'
            f'</div>'
            f'<div style="flex:1;border-left:2px solid {color};padding:4px 0 12px 16px">'
            f'<div style="font-weight:bold;font-size:13px">{label}: {title}</div>'
            f'<div style="font-size:11px;color:#666">Paper: {paper_refs[i]}</div>'
            + (f'<div style="font-size:12px;color:#333;margin-top:2px">{detail}</div>' if detail else '')
            + f'</div></div>'
        )
    html_parts.append('</div>')
    return "\n".join(html_parts)


# ──────────────────────────────────────────────────────────────────────
# Run All Stages
# ──────────────────────────────────────────────────────────────────────

def run_all_stages(model_choice: str, image, max_trials: int, progress=gr.Progress(track_tqdm=True)):
    """Execute every stage sequentially, populating all tab outputs at once."""

    progress(0, desc="Stage 0-1: Loading model & inference...")
    s01 = run_stage_0_1(model_choice, image)

    progress(0.15, desc="Stage 2: Tracing PyTorch graph...")
    s2 = run_stage_2()

    progress(0.25, desc="Stage 3: Importing into TVM...")
    s3 = run_stage_3()

    progress(0.35, desc="Stage 4: Applying passes...")
    s4 = run_stage_4()

    progress(0.45, desc="Stage 5: Extracting operators...")
    s5 = run_stage_5()

    progress(0.50, desc="Stage 6: TIR analysis...")
    op_names = get_op_names()
    first_op = op_names[0] if op_names else ""
    s6 = run_stage_6(first_op)

    progress(0.55, desc="Stage 7: TE microscope...")
    s7 = run_stage_7()

    progress(0.60, desc="Stages 8-9: Tuning...")
    s89 = run_stage_8_9(max_trials)

    progress(0.80, desc="Stage 10: Cost model...")
    s10 = run_stage_10()

    progress(0.90, desc="Stages 11-12: Build & inference...")
    s1112 = run_stage_11_12()

    progress(1.0, desc="Complete!")
    timeline = build_timeline()

    # strip trailing progress_bar from each sub-result (last element)
    return (
        *s01[:-1],
        *s2[:-1],
        *s3[:-1],
        *s4[:-1],
        *s5[:-1],
        *s6[:-1],
        *s7[:-1],
        *s89[:-1],
        *s10[:-1],
        *s1112[:-1],
        timeline,
        _progress_html(),
    )


# ──────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="TVM Demo -- OSDI '18") as app:

        gr.Markdown(
            "# TVM: End-to-End Optimizing Compiler for Deep Learning\n"
            "*Interactive demo based on Chen et al., OSDI '18*"
        )

        # ── Top controls ─────────────────────────────────────────────
        with gr.Row():
            model_dd = gr.Dropdown(
                choices=["resnet18", "mobilenet_v2"],
                value="resnet18",
                label="Model",
                scale=1,
            )
            image_in = gr.Image(type="pil", label="Upload Image (optional)", scale=2)
            tuning_slider = gr.Slider(
                minimum=4, maximum=128, value=32, step=4,
                label="Tuning Trials",
                scale=1,
            )

        with gr.Row():
            run_all_btn = gr.Button("Run All Stages", variant="primary", scale=2)

        progress_bar = gr.HTML(value=_progress_html(), label="Progress")

        # ── Tabs: create components inside each tab (Gradio 6: no .render()) ─────

        with gr.Tabs():

            # --- Tab 1: Task & Input ---
            with gr.TabItem("1. Task & Input"):
                gr.Markdown("### Stages 0-1: Load Model & PyTorch Baseline\n*Paper: Figure 2 top, Section 1*")
                run_01_btn = gr.Button("Run Stages 0-1")
                s01_status = gr.HTML()
                s01_env = gr.Markdown()
                with gr.Row():
                    s01_image = gr.Image(label="Input Image", type="pil", interactive=False, scale=1)
                    with gr.Column(scale=2):
                        s01_top5 = gr.Markdown()
                        s01_latency = gr.Markdown()
                run_01_btn.click(
                    run_stage_0_1,
                    inputs=[model_dd, image_in],
                    outputs=[s01_status, s01_env, s01_image, s01_top5, s01_latency, progress_bar],
                )

            # --- Tab 2: PyTorch Graph ---
            with gr.TabItem("2. PyTorch Graph"):
                gr.Markdown("### Stage 2: Computation Graph Capture\n*Paper: Section 3, Figure 3*")
                run_2_btn = gr.Button("Run Stage 2")
                s2_status = gr.HTML()
                s2_svg = gr.HTML()
                s2_table = gr.HTML()
                run_2_btn.click(
                    run_stage_2,
                    outputs=[s2_status, s2_svg, s2_table, progress_bar],
                )

            # --- Tab 3: TVM IR Import ---
            with gr.TabItem("3. TVM IR Import"):
                gr.Markdown("### Stage 3: Import into TVM Relax IR\n*Paper: Figure 2, Section 3*")
                run_3_btn = gr.Button("Run Stage 3")
                s3_status = gr.HTML()
                with gr.Row():
                    s3_ir = gr.Code(language=None, label="Relax IR")
                    s3_callgraph = gr.HTML()
                run_3_btn.click(
                    run_stage_3,
                    outputs=[s3_status, s3_ir, s3_callgraph, progress_bar],
                )

            # --- Tab 4: TVM Passes ---
            with gr.TabItem("4. TVM Passes"):
                gr.Markdown("### Stage 4: Graph-Level Passes\n*Paper: Section 3 -- Operator Fusion, Legalization*")
                run_4_btn = gr.Button("Run Stage 4")
                s4_status = gr.HTML()
                s4_deltas = gr.Markdown()
                with gr.Accordion("Post-Pass IR (click to expand)", open=False):
                    s4_ir = gr.Code(language=None, label="Post-Pass IR")
                with gr.Accordion("Pass Diff Viewer", open=False):
                    pass_dd = gr.Dropdown(label="Select pass", choices=[], interactive=True)
                    diff_code = gr.Code(language=None, label="Diff")
                    pass_dd.change(view_pass_diff, inputs=[pass_dd], outputs=[diff_code])
                run_4_btn.click(
                    run_stage_4,
                    outputs=[s4_status, s4_deltas, s4_ir, progress_bar],
                ).then(
                    lambda: gr.update(choices=STATE.pass_order if STATE.pass_order else []),
                    outputs=[pass_dd],
                )

            # --- Tab 5: Extracted Operators ---
            with gr.TabItem("5. Extracted Operators"):
                gr.Markdown("### Stage 5: Operator Extraction\n*Paper: Section 3 -> 4 transition*")
                run_5_btn = gr.Button("Run Stage 5")
                s5_status = gr.HTML()
                s5_table = gr.HTML()

            # --- Tab 6: TensorIR / AST ---
            with gr.TabItem("6. TensorIR / AST"):
                gr.Markdown("### Stage 6: TIR Low-Level Representation\n*Paper: Section 4, Figure 13*")
                op_dd = gr.Dropdown(label="Select PrimFunc", choices=[], interactive=True)
                refresh_ops_btn = gr.Button("Refresh operator list", size="sm")
                s6_status = gr.HTML()
                with gr.Row():
                    with gr.Column():
                        s6_tir = gr.Code(language=None, label="TIR Source")
                    with gr.Column():
                        s6_tree = gr.HTML()
                        s6_loops = gr.HTML()

            run_5_btn.click(
                run_stage_5,
                outputs=[s5_status, s5_table, progress_bar],
            ).then(
                lambda: gr.update(choices=get_op_names()),
                outputs=[op_dd],
            )
            refresh_ops_btn.click(
                lambda: gr.update(choices=get_op_names()),
                outputs=[op_dd],
            )
            op_dd.change(
                run_stage_6,
                inputs=[op_dd],
                outputs=[s6_status, s6_tir, s6_tree, s6_loops, progress_bar],
            )

            # --- Tab 7: Tensor Expression ---
            with gr.TabItem("7. Tensor Expression"):
                gr.Markdown("### Stage 7: Compute / Schedule Separation\n*Paper: Section 4.1, Figure 5*")
                run_7_btn = gr.Button("Run Stage 7")
                s7_status = gr.HTML()
                with gr.Row():
                    s7_compute = gr.Code(language="python", label="TE Compute Declaration")
                    s7_tir = gr.Code(language=None, label="Naive TIR (un-scheduled)")
                s7_explain = gr.Markdown()
                run_7_btn.click(
                    run_stage_7,
                    outputs=[s7_status, s7_compute, s7_tir, s7_explain, progress_bar],
                )

            # --- Tab 8: Schedule Search ---
            with gr.TabItem("8. Schedule Search"):
                gr.Markdown("### Stages 8-9: Task Extraction & Tuning\n*Paper: Sections 5.1, 5.3 -- Figure 12*")
                run_89_btn = gr.Button("Run Stages 8-9")
                s89_task_status = gr.HTML()
                s89_tune_status = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=2):
                        s89_cards = gr.HTML()
                    with gr.Column(scale=1):
                        s89_convergence = gr.HTML()
                        s89_pie = gr.HTML()
                run_89_btn.click(
                    run_stage_8_9,
                    inputs=[tuning_slider],
                    outputs=[s89_task_status, s89_tune_status, s89_cards, s89_convergence, s89_pie, progress_bar],
                )

            # --- Tab 9: Cost Model ---
            with gr.TabItem("9. Cost Model"):
                gr.Markdown("### Stage 10: Cost Model & Schedule Selection\n*Paper: Section 5.2, Figure 13*")
                run_10_btn = gr.Button("Run Stage 10")
                s10_status = gr.HTML()
                s10_banner = gr.HTML()
                s10_features = gr.HTML()
                s10_explanation = gr.HTML()
                run_10_btn.click(
                    run_stage_10,
                    outputs=[s10_status, s10_banner, s10_features, s10_explanation, progress_bar],
                )

            # --- Tab 10: Build & Results ---
            with gr.TabItem("10. Build & Results"):
                gr.Markdown("### Stages 11-12: CUDA Build & Inference Comparison\n*Paper: Figure 2 bottom, Section 6, Figure 14*")
                run_1112_btn = gr.Button("Run Stages 11-12")
                s1112_build = gr.HTML()
                with gr.Row():
                    s1112_tvm_top5 = gr.Markdown()
                    s1112_comparison = gr.Markdown()
                s1112_chart = gr.HTML()
                with gr.Accordion("Generated CUDA Source", open=False):
                    s1112_cuda = gr.Code(language="c", label="Generated CUDA Source")
                s1112_done = gr.HTML()
                run_1112_btn.click(
                    run_stage_11_12,
                    outputs=[s1112_build, s1112_tvm_top5, s1112_comparison, s1112_chart, s1112_cuda, s1112_done, progress_bar],
                )

            # --- Tab 11: Timeline ---
            with gr.TabItem("11. Pipeline Timeline"):
                gr.Markdown("### Full Pipeline Journey\n*Paper: Figure 2 (system overview)*")
                refresh_tl_btn = gr.Button("Refresh Timeline")
                timeline_html = gr.HTML()
                refresh_tl_btn.click(build_timeline, outputs=[timeline_html])

        # ── "Run All" wiring ─────────────────────────────────────────
        all_outputs = [
            s01_status, s01_env, s01_image, s01_top5, s01_latency,
            s2_status, s2_svg, s2_table,
            s3_status, s3_ir, s3_callgraph,
            s4_status, s4_deltas, s4_ir,
            s5_status, s5_table,
            s6_status, s6_tir, s6_tree, s6_loops,
            s7_status, s7_compute, s7_tir, s7_explain,
            s89_task_status, s89_tune_status, s89_cards, s89_convergence, s89_pie,
            s10_status, s10_banner, s10_features, s10_explanation,
            s1112_build, s1112_tvm_top5, s1112_comparison, s1112_chart, s1112_cuda, s1112_done,
            timeline_html,
            progress_bar,
        ]
        run_all_btn.click(
            run_all_stages,
            inputs=[model_dd, image_in, tuning_slider],
            outputs=all_outputs,
        )

    return app


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TVM Demo Gradio App")
    parser.add_argument("--host", default="0.0.0.0", help="Server host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default 7860)")
    parser.add_argument("--share", action="store_true", help="Create Gradio share link")
    args = parser.parse_args()

    app = build_app()
    app.queue()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="slate",
            neutral_hue="slate",
            font=("Inter", "system-ui", "sans-serif"),
        ),
        css="""
        footer { display: none !important; }
        """,
    )


if __name__ == "__main__":
    main()
