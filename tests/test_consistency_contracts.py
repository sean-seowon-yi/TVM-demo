#!/usr/bin/env python
"""End-to-end consistency contracts for pipeline + UI reporting.

Purpose:
- Catch inconsistencies that are easy to miss in manual inspection:
  * requested trial budget vs actual measured candidates
  * record/convergence accounting invariants
  * best-candidate marker correctness
  * DB task coverage vs record task names
  * Stage 8/10 display text contract consistency

Run from repo root:
    python tests/test_consistency_contracts.py
"""

from __future__ import annotations

import os
import sys
import shutil
import random
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SRC = str(_ROOT / "src")
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from backend.helpers import download_sample_image
from backend.pipeline import (
    load_model,
    prepare_input,
    import_to_tvm,
    apply_passes_stepwise,
    extract_tuning_tasks,
    run_tuning,
    select_best_candidate,
    count_tuned_tasks_from_db,
)

PASS = 0
FAIL = 0
SKIP = 0


def check(cond: bool, label: str) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


def skip(label: str) -> None:
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {label}")


def _sample_image():
    try:
        return download_sample_image()
    except Exception:
        # fallback: random RGB image
        from PIL import Image
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        return Image.fromarray(arr)


def test_backend_invariants():
    print("\n=== Backend invariants (real run) ===")
    try:
        model, transform, _cats, _summary, _params = load_model("resnet18")
        img = _sample_image()
        inp, _inp_np = prepare_input(img, transform)
        mod, _params_np, _ir = import_to_tvm(model, inp)
        current, _snapshots, _order, _deltas = apply_passes_stepwise(mod)
        tasks, _raw, target = extract_tuning_tasks(current, target_str="cuda")
        check(len(tasks) > 0, "Stage 8 extracts >=1 tuning task")

        work_dir = "./consistency_contract_logs"
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)

        records, conv, out_dir = run_tuning(
            current,
            target=target,
            work_dir=work_dir,
            max_trials_global=16,
            num_trials_per_iter=8,
            max_tasks=2,
        )
        check(len(records) > 0, "Stage 9 produces tuning records")
        check(len(records) == len(conv), "records count equals convergence count")

        # Candidate IDs should be unique
        ids = [r["candidate_id"] for r in records]
        check(len(ids) == len(set(ids)), "candidate_id values are unique")

        # Exactly one best marker and it must match true min run_ms
        best_marked = [r for r in records if r.get("is_best")]
        check(len(best_marked) == 1, "exactly one record has is_best=True")
        if best_marked:
            valid = [r for r in records if r.get("run_ms", float("inf")) < 1e6]
            true_best = min(valid, key=lambda r: r["run_ms"])
            check(
                best_marked[0]["candidate_id"] == true_best["candidate_id"],
                "is_best marker matches minimum-latency candidate",
            )

        # DB task coverage should match record task names
        db_n, db_names = count_tuned_tasks_from_db(out_dir)
        rec_names = list(dict.fromkeys(
            r["task_name"] for r in records
            if r.get("task_name")
            and r["task_name"] != "main"
            and not r["task_name"].startswith("task_")
        ))
        if db_names and rec_names:
            check(set(db_names) == set(rec_names), "DB task names == record task names")
            check(db_n == len(db_names), "DB task count matches DB name list length")
        else:
            skip("DB/record task-name comparison (names not available)")

        best, feats = select_best_candidate(records, current)
        check(best is not None, "Stage 10 selects a best candidate")
        check(len(feats) > 0, "Stage 10 produces feature rows")
        if best and best_marked:
            check(
                best["candidate_id"] == best_marked[0]["candidate_id"],
                "selected best matches is_best-marked record",
            )
    except Exception as exc:
        check(False, f"backend invariant run failed: {exc}")


def test_app_display_contracts():
    print("\n=== App display/report contracts ===")
    try:
        import app as demo_app
    except Exception as exc:
        skip(f"app import failed: {exc}")
        return

    try:
        # Run up to stage 9 with small budget through app wiring
        demo_app.run_stage_0_1("resnet18", None)
        demo_app.run_stage_2()
        demo_app.run_stage_3()
        demo_app.run_stage_4()
        demo_app.run_stage_5()
        op_names = demo_app.get_op_names()
        if op_names:
            demo_app.run_stage_6(op_names[0])
        demo_app.run_stage_7()

        last = None
        for out in demo_app.run_stage_8_9(16):
            last = out
        check(last is not None, "Stage 8-9 generator produced output")
        if last is None:
            return

        # Output tuple: task_info, tune_info, cards, task_chart, summary_html, progress
        _task_info, tune_info, cards_html, _task_chart, summary_html, _progress = last
        requested = demo_app.STATE.tuning_trials_used
        actual = len(demo_app.STATE.tuning_records)
        task_names = list(dict.fromkeys(
            r["task_name"] for r in demo_app.STATE.tuning_records
            if r.get("task_name")
            and r["task_name"] != "main"
            and not r["task_name"].startswith("task_")
        ))

        # Contract 1: Stage-8 tune status always includes requested/actual wording
        check("requested" in tune_info.lower(), "Stage 8 tune status includes requested budget wording")
        check("actual" in tune_info.lower(), "Stage 8 tune status includes actual measured wording")
        check(str(requested) in tune_info, "Stage 8 tune status includes requested count")
        check(str(actual) in tune_info, "Stage 8 tune status includes actual count")

        # Contract 2: Candidate card header count matches actual records/tasks
        expected_header = f"{actual} candidates across {len(task_names)} task"
        check(expected_header in cards_html, "Stage 8 cards header matches actual counts")

        # Contract 3: Per-task summary tuned count matches record groups
        check(
            f"({len(task_names)} tuned)" in summary_html,
            "Stage 8 per-task summary tuned-count matches records",
        )

        # Stage 10 status wording contract
        s10_status, *_ = demo_app.run_stage_10()
        check("requested budget" in s10_status.lower(), "Stage 10 status includes requested budget wording")
        check("valid measured candidates" in s10_status.lower(), "Stage 10 status uses valid-candidate wording")
    except Exception as exc:
        check(False, f"app display contract run failed: {exc}")


if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    test_backend_invariants()
    test_app_display_contracts()

    print(f"\n{'='*60}")
    print(f"  Results:  {PASS} passed,  {FAIL} failed,  {SKIP} skipped")
    print(f"{'='*60}")
    if FAIL > 0:
        sys.exit(1)
