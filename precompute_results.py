#!/usr/bin/env python
"""Run a high-trial tuning session and save results to precomputed_results.json.

Usage (from WSL with TVM):
    ~/tvm_env/bin/python precompute_results.py --trials 512
    ~/tvm_env/bin/python precompute_results.py --trials 512 --model mobilenet_v2

ResNet-18 has 28 tunable tasks.  At least ~500 trials are needed so the
MetaSchedule task scheduler covers all tasks (especially the heavy conv2d
ones).  128 trials only covers ~6 tasks and won't beat PyTorch.

This takes 15-30+ minutes depending on GPU and trial count.
The saved results are shown in Tab 10 during the live demo
so the audience can see the benefit of more tuning without waiting.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_SRC = str(Path(__file__).resolve().parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("precompute")

from backend.pipeline import (
    check_environment,
    load_model,
    prepare_input,
    run_pytorch_inference,
    import_to_tvm,
    apply_passes_stepwise,
    extract_tuning_tasks,
    run_tuning,
    count_tuned_tasks_from_db,
    select_best_candidate,
    build_tvm_module,
    run_tvm_inference,
    compare_results,
)
from backend.helpers import download_sample_image


def main():
    parser = argparse.ArgumentParser(description="Precompute high-trial TVM results")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "mobilenet_v2"])
    parser.add_argument("--trials", type=int, default=512, help="MetaSchedule trial budget (512+ recommended)")
    parser.add_argument("--output", default="precomputed_results.json")
    args = parser.parse_args()

    import torch
    use_cuda = torch.cuda.is_available()
    assert use_cuda, "CUDA required for precomputing results"

    log.info("Loading model: %s", args.model)
    model, transform, categories, summary, n_params = load_model(args.model)

    image = download_sample_image()
    tensor, input_np = prepare_input(image, transform)

    log.info("Running PyTorch baseline (CUDA)...")
    pytorch_logits, pytorch_top5, pytorch_ms = run_pytorch_inference(
        model, tensor, categories, n_runs=100, use_cuda=True,
    )
    log.info("PyTorch: %s (%.2f ms)", pytorch_top5[0]["class"], pytorch_ms)

    log.info("Importing into TVM...")
    mod, params_np, _ = import_to_tvm(model, tensor)

    log.info("Applying passes...")
    current_mod, _, _, _ = apply_passes_stepwise(mod)

    log.info("Extracting tuning tasks...")
    task_dicts, tasks_raw, target = extract_tuning_tasks(current_mod, target_str="cuda")
    log.info("Found %d tuning tasks", len(task_dicts))

    precompute_work_dir = "./precompute_tuning_logs"
    import shutil, os
    if os.path.exists(precompute_work_dir):
        shutil.rmtree(precompute_work_dir)

    log.info("Running tuning with %d trials (this may take a while)...", args.trials)
    records, convergence, work_dir = run_tuning(
        current_mod, target=target,
        work_dir=precompute_work_dir,
        max_trials_global=args.trials,
        num_trials_per_iter=min(16, args.trials // 4),
        max_tasks=min(5, len(task_dicts)),
    )
    n_tasks_tuned, tuned_names = count_tuned_tasks_from_db(work_dir)
    log.info("Tuning complete: %d records, %d unique tasks tuned", len(records), n_tasks_tuned)

    log.info("Building TVM module (with tuning database)...")
    lib, target_used, cuda_src, params_bound = build_tvm_module(
        current_mod, params_np=params_np, target_str="cuda",
        work_dir=work_dir,
    )

    log.info("Running TVM inference...")
    params_at_runtime = None if params_bound else params_np
    tvm_logits, tvm_top5, tvm_ms = run_tvm_inference(
        lib, input_np, categories, params_np=params_at_runtime, n_runs=100,
    )

    comp = compare_results(pytorch_logits, tvm_logits, pytorch_ms, tvm_ms)

    result = {
        "tuning_trials": args.trials,
        "tasks_tuned": n_tasks_tuned,
        "tvm_latency_ms": round(tvm_ms, 4),
        "pytorch_latency_ms": round(pytorch_ms, 4),
        "speedup": comp["speedup"],
        "max_abs_diff": comp["max_abs_diff"],
        "cosine_similarity": comp["cosine_similarity"],
        "top1_class": tvm_top5[0]["class"],
        "target": target_used,
    }

    log.info("Results: %s", result)

    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
    else:
        data = {}

    data[args.model] = result
    data.setdefault("_comment", "Precomputed high-trial results. Update by running precompute_results.py.")

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    log.info("Saved to %s", output_path)
    print(f"\n  {args.model}: PyTorch {pytorch_ms:.2f} ms -> TVM {tvm_ms:.2f} ms ({comp['speedup']}x speedup)")
    print(f"  Correctness: max_diff={comp['max_abs_diff']:.6f}, cosine={comp['cosine_similarity']:.6f}")


if __name__ == "__main__":
    main()
