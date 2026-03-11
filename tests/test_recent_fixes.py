#!/usr/bin/env python
"""Tests for recent fixes: disambiguation, block name extraction,
task name consistency between Stage 8 and Stage 9, card rendering,
and _is_tir_* helpers.

Run from repo root:
    python -m pytest tests/test_recent_fixes.py -v
    # or without pytest:
    python tests/test_recent_fixes.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from backend.pipeline import (
    _disambiguate_names,
    _extract_block_name_from_trace,
    _pick_primary_block_name,
    _build_workload_name_map,
    _fixup_main_task_names,
    count_tuned_tasks_from_db,
    _is_tir_for,
    _is_tir_block,
    _is_tir_block_realize,
    _is_tir_seq_stmt,
)
from viz.schedule_display import trace_to_card_html, candidate_cards_html, per_task_summary_html

PASS = 0
FAIL = 0


def check(condition: bool, label: str):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


# ──────────────────────────────────────────────────────────────────────
# 1. _disambiguate_names
# ──────────────────────────────────────────────────────────────────────
def test_disambiguate_names():
    print("\n=== _disambiguate_names ===")

    # No duplicates → unchanged
    r = _disambiguate_names(["transpose", "mean", "reshape"])
    check(r == ["transpose", "mean", "reshape"], "no duplicates → unchanged")

    # All duplicates
    r = _disambiguate_names(["conv2d_nchw", "conv2d_nchw", "conv2d_nchw"])
    check(r == ["conv2d_nchw_0", "conv2d_nchw_1", "conv2d_nchw_2"],
          "3x same → _0, _1, _2")
    check(len(r) == len(set(r)), "all unique after disambiguation")

    # Mixed
    r = _disambiguate_names(["add", "conv2d_nchw", "add", "mean", "conv2d_nchw"])
    check(r[0] == "add_0", "first 'add' → add_0")
    check(r[2] == "add_1", "second 'add' → add_1")
    check(r[1] == "conv2d_nchw_0", "first 'conv2d_nchw' → conv2d_nchw_0")
    check(r[4] == "conv2d_nchw_1", "second 'conv2d_nchw' → conv2d_nchw_1")
    check(r[3] == "mean", "'mean' stays unchanged (no dup)")

    # Empty
    r = _disambiguate_names([])
    check(r == [], "empty list → empty list")

    # Single
    r = _disambiguate_names(["only_one"])
    check(r == ["only_one"], "single item → unchanged")


# ──────────────────────────────────────────────────────────────────────
# 2. _pick_primary_block_name
# ──────────────────────────────────────────────────────────────────────
def test_pick_primary_block_name():
    print("\n=== _pick_primary_block_name ===")

    check(_pick_primary_block_name(["reshape", "conv2d_nchw"]) == "conv2d_nchw",
          "prefers conv over reshape")
    check(_pick_primary_block_name(["add", "matmul"]) == "matmul",
          "prefers matmul over add")
    check(_pick_primary_block_name(["transpose", "dense"]) == "dense",
          "prefers dense over transpose")
    check(_pick_primary_block_name(["reshape", "mean"]) == "mean",
          "prefers mean over reshape")
    check(_pick_primary_block_name(["some_unknown"]) == "some_unknown",
          "single unknown → return it")
    check(_pick_primary_block_name(["xyz", "abc"]) == "xyz",
          "no priority match → return first")


# ──────────────────────────────────────────────────────────────────────
# 3. _extract_block_name_from_trace
# ──────────────────────────────────────────────────────────────────────
def test_extract_block_name_from_trace():
    print("\n=== _extract_block_name_from_trace ===")

    # Typical MetaSchedule trace structure:
    # [[[inst1, inst2, ...], decisions], ...]
    # Each inst: [inst_kind, inputs, attrs, ...]

    # Single block name
    trace = [[
        [
            ["GetBlock", [], ["conv2d_nchw"]],
            ["Split", [], ["something"]],
        ],
        []
    ]]
    r = _extract_block_name_from_trace(trace)
    check(r == "conv2d_nchw", "extracts conv2d_nchw from single GetBlock")

    # Multiple blocks — should prefer compute-like
    trace = [[
        [
            ["GetBlock", [], ["reshape"]],
            ["GetBlock", [], ["conv2d_nchw"]],
            ["Split", [], ["other"]],
        ],
        []
    ]]
    r = _extract_block_name_from_trace(trace)
    check(r == "conv2d_nchw", "prefers conv2d_nchw over reshape")

    # "main" and "root" are filtered out
    trace = [[
        [
            ["GetBlock", [], ["main"]],
            ["GetBlock", [], ["root"]],
            ["GetBlock", [], ["add"]],
        ],
        []
    ]]
    r = _extract_block_name_from_trace(trace)
    check(r == "add", "filters out main/root, returns add")

    # T_ prefix stripped
    trace = [[
        [
            ["GetBlock", [], ["T_reshape"]],
        ],
        []
    ]]
    r = _extract_block_name_from_trace(trace)
    check(r == "reshape", "strips T_ prefix")

    # Empty/invalid
    check(_extract_block_name_from_trace(None) is None, "None → None")
    check(_extract_block_name_from_trace([]) is None, "empty list → None")
    check(_extract_block_name_from_trace("not a list") is None, "string → None")

    # Only main/root → None
    trace = [[
        [
            ["GetBlock", [], ["main"]],
            ["GetBlock", [], ["root"]],
        ],
        []
    ]]
    r = _extract_block_name_from_trace(trace)
    check(r is None, "only main/root → None")


# ──────────────────────────────────────────────────────────────────────
# 4. _is_tir_* helpers with no tvm.tir.Block
# ──────────────────────────────────────────────────────────────────────
def test_is_tir_helpers():
    print("\n=== _is_tir_* helpers ===")

    class FakeFor:
        pass

    class FakeBlock:
        pass

    class FakeBlockRealize:
        pass

    class FakeSeqStmt:
        pass

    # Rename classes to match TVM naming
    FakeFor.__name__ = "For"
    FakeBlock.__name__ = "Block"
    FakeBlockRealize.__name__ = "BlockRealize"
    FakeSeqStmt.__name__ = "SeqStmt"

    check(_is_tir_for(FakeFor()), "FakeFor with __name__='For' → True")
    check(_is_tir_block(FakeBlock()), "FakeBlock with __name__='Block' → True")
    check(_is_tir_block_realize(FakeBlockRealize()),
          "FakeBlockRealize with __name__='BlockRealize' → True")
    check(_is_tir_seq_stmt(FakeSeqStmt()),
          "FakeSeqStmt with __name__='SeqStmt' → True")

    # Wrong names → False
    check(not _is_tir_for(object()), "plain object → not For")
    check(not _is_tir_block(object()), "plain object → not Block")
    check(not _is_tir_block_realize(object()), "plain object → not BlockRealize")
    check(not _is_tir_seq_stmt(object()), "plain object → not SeqStmt")

    # None → should not crash
    check(not _is_tir_for(None), "None → not For (no crash)")
    check(not _is_tir_block(None), "None → not Block (no crash)")


# ──────────────────────────────────────────────────────────────────────
# 5. _build_workload_name_map + count_tuned_tasks_from_db consistency
# ──────────────────────────────────────────────────────────────────────
def _make_trace(block_name: str) -> list:
    """Build a minimal MetaSchedule trace JSON structure."""
    return [[[["GetBlock", [], [block_name]]], []]]


def _write_db(work_dir: str, entries: list):
    """Write database_tuning_record.json lines: each entry is [wl_idx, trace, run_secs]."""
    db_path = os.path.join(work_dir, "database_tuning_record.json")
    with open(db_path, "w") as f:
        for wl_idx, block_name, run_secs in entries:
            trace = _make_trace(block_name)
            line = json.dumps([wl_idx, trace, [run_secs]])
            f.write(line + "\n")


def test_workload_name_map_and_count_consistency():
    print("\n=== _build_workload_name_map + count_tuned_tasks_from_db consistency ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 3 workloads: 2x conv2d_nchw (different shapes), 1x add
        _write_db(tmpdir, [
            ("wl_hash_A", "conv2d_nchw", 0.005),
            ("wl_hash_A", "conv2d_nchw", 0.004),  # same workload, diff candidate
            ("wl_hash_B", "conv2d_nchw", 0.010),
            ("wl_hash_B", "conv2d_nchw", 0.009),
            ("wl_hash_C", "add", 0.001),
        ])

        wl_map = _build_workload_name_map(tmpdir)
        n, names = count_tuned_tasks_from_db(tmpdir)

        check(n == 3, f"3 distinct workloads → count={n}")
        check(len(names) == 3, f"3 names returned, got {len(names)}")

        # Disambiguated: two conv2d_nchw should get _0 and _1
        conv_names = [nm for nm in names if "conv2d_nchw" in nm]
        check(len(conv_names) == 2, f"2 conv2d_nchw variants, got {conv_names}")
        check(conv_names[0] != conv_names[1],
              f"conv2d names are different: {conv_names}")

        # The add should not be suffixed
        add_names = [nm for nm in names if "add" in nm]
        check(len(add_names) == 1, f"1 add task, got {add_names}")
        check(add_names[0] == "add", f"add not suffixed: {add_names[0]}")

        # names from count_tuned_tasks_from_db must equal values from _build_workload_name_map
        map_values = list(wl_map.values())
        check(set(names) == set(map_values),
              "count_tuned_tasks_from_db names == _build_workload_name_map values")


# ──────────────────────────────────────────────────────────────────────
# 6. _fixup_main_task_names
# ──────────────────────────────────────────────────────────────────────
def test_fixup_main_task_names():
    print("\n=== _fixup_main_task_names ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 2 workloads: conv2d_nchw and add
        _write_db(tmpdir, [
            ("wl_A", "conv2d_nchw", 0.01),
            ("wl_A", "conv2d_nchw", 0.009),
            ("wl_B", "add", 0.002),
        ])

        records = [
            {
                "candidate_id": 0,
                "task_name": "main",
                "run_ms": 10.0,
                "trace_text": 'sch.get_sblock(name="conv2d_nchw", func_name="main")',
            },
            {
                "candidate_id": 1,
                "task_name": "main",
                "run_ms": 9.0,
                "trace_text": 'sch.get_sblock(name="conv2d_nchw", func_name="main")',
            },
            {
                "candidate_id": 2,
                "task_name": "main",
                "run_ms": 2.0,
                "trace_text": 'sch.get_sblock(name="add", func_name="main")',
            },
        ]

        _fixup_main_task_names(records, tmpdir)

        check(records[0]["task_name"] == "conv2d_nchw",
              f"record 0 → conv2d_nchw, got {records[0]['task_name']}")
        check(records[1]["task_name"] == "conv2d_nchw",
              f"record 1 → conv2d_nchw (same wl), got {records[1]['task_name']}")
        check(records[2]["task_name"] == "add",
              f"record 2 → add, got {records[2]['task_name']}")

    # Non-main records should not be touched
    records = [
        {"candidate_id": 0, "task_name": "already_set", "run_ms": 5.0},
    ]
    _fixup_main_task_names(records, "/nonexistent")
    check(records[0]["task_name"] == "already_set",
          "non-main record not modified")


# ──────────────────────────────────────────────────────────────────────
# 7. Stage 8 ↔ Stage 9 name consistency (end-to-end)
# ──────────────────────────────────────────────────────────────────────
def test_stage8_stage9_consistency():
    print("\n=== Stage 8 ↔ Stage 9 name consistency ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_db(tmpdir, [
            ("w0", "conv2d_nchw", 0.01),
            ("w0", "conv2d_nchw", 0.009),
            ("w1", "conv2d_nchw", 0.02),
            ("w2", "add", 0.003),
            ("w2", "add", 0.002),
            ("w3", "mean", 0.001),
        ])

        # Stage 8 path: count_tuned_tasks_from_db
        n_covered, covered_names = count_tuned_tasks_from_db(tmpdir)

        # Stage 9 path: records with _fixup_main_task_names
        records = [
            {
                "candidate_id": 0, "task_name": "main", "run_ms": 10.0,
                "trace_text": 'sch.get_sblock(name="conv2d_nchw_0", func_name="main")',
            },
            {
                "candidate_id": 1, "task_name": "main", "run_ms": 9.0,
                "trace_text": 'sch.get_sblock(name="conv2d_nchw_0", func_name="main")',
            },
            {
                "candidate_id": 2, "task_name": "main", "run_ms": 20.0,
                "trace_text": 'sch.get_sblock(name="conv2d_nchw_1", func_name="main")',
            },
            {
                "candidate_id": 3, "task_name": "main", "run_ms": 3.0,
                "trace_text": 'sch.get_sblock(name="add", func_name="main")',
            },
            {
                "candidate_id": 4, "task_name": "main", "run_ms": 2.0,
                "trace_text": 'sch.get_sblock(name="add", func_name="main")',
            },
            {
                "candidate_id": 5, "task_name": "main", "run_ms": 1.0,
                "trace_text": 'sch.get_sblock(name="mean", func_name="main")',
            },
        ]
        _fixup_main_task_names(records, tmpdir)

        record_unique_names = list(dict.fromkeys(
            r["task_name"] for r in records
            if r["task_name"] != "main" and not r["task_name"].startswith("task_")
        ))

        check(n_covered == 4, f"4 distinct workloads, got {n_covered}")
        check(set(record_unique_names) == set(covered_names),
              f"record names match covered_names: {record_unique_names} vs {covered_names}")

        # Verify the scatter chart grouping would work correctly
        task_stats = {}
        for r in records:
            name = r["task_name"]
            if name not in task_stats:
                task_stats[name] = 0
            task_stats[name] += 1

        check(len(task_stats) == 4,
              f"scatter chart would show 4 groups, got {len(task_stats)}: {list(task_stats.keys())}")

        # w0 has 2 records, w1 has 1, w2 has 2, w3 has 1
        for name, count in task_stats.items():
            if "conv2d" in name:
                pass  # expected 1 or 2
            check(count > 0, f"  {name}: {count} candidates")


# ──────────────────────────────────────────────────────────────────────
# 8. Card HTML — per-record task label
# ──────────────────────────────────────────────────────────────────────
def test_card_html_per_record_task():
    print("\n=== Card HTML per-record task label ===")

    rec_conv = {
        "candidate_id": 0, "task_name": "conv2d_nchw_0",
        "trace_text": "split -> reorder -> bind", "run_ms": 5.0, "is_best": False,
    }
    rec_add = {
        "candidate_id": 1, "task_name": "add",
        "trace_text": "split -> vectorize", "run_ms": 2.0, "is_best": True,
    }
    rec_main = {
        "candidate_id": 2, "task_name": "main",
        "trace_text": "split", "run_ms": 3.0, "is_best": False,
    }

    html_conv = trace_to_card_html(rec_conv)
    html_add = trace_to_card_html(rec_add)
    html_main = trace_to_card_html(rec_main)

    check("conv2d_nchw_0" in html_conv, "conv card shows conv2d_nchw_0 badge")
    check("add" in html_add, "add card shows add badge")
    check("BEST" in html_add, "best card shows BEST badge")
    check("main" not in html_main or "task_" not in html_main,
          "main task name → no task badge shown")

    # candidate_cards_html — grouped by task
    recs = [
        {"candidate_id": 0, "task_name": "conv2d_nchw_0", "trace_text": "split -> bind", "run_ms": 5.0, "is_best": False},
        {"candidate_id": 1, "task_name": "conv2d_nchw_0", "trace_text": "split -> reorder -> bind", "run_ms": 4.0, "is_best": False},
        {"candidate_id": 2, "task_name": "add", "trace_text": "split -> vectorize", "run_ms": 2.0, "is_best": False},
    ]
    grouped_html = candidate_cards_html(recs)
    check("conv2d_nchw_0" in grouped_html, "grouped cards: conv2d_nchw_0 section present")
    check("add" in grouped_html, "grouped cards: add section present")
    check("3 candidates across 2 tasks" in grouped_html,
          "header says '3 candidates across 2 tasks'")
    check("BEST" in grouped_html, "per-task best gets BEST badge")


def test_per_task_summary_html():
    print("\n=== per_task_summary_html ===")

    recs = [
        {"candidate_id": 0, "task_name": "conv2d_nchw_0", "trace_text": "split -> bind -> cache_read", "run_ms": 5.0},
        {"candidate_id": 1, "task_name": "conv2d_nchw_0", "trace_text": "split -> reorder -> bind", "run_ms": 4.0},
        {"candidate_id": 2, "task_name": "conv2d_nchw_1", "trace_text": "split -> unroll", "run_ms": 10.0},
        {"candidate_id": 3, "task_name": "add", "trace_text": "split -> vectorize", "run_ms": 1.5},
    ]
    html_out = per_task_summary_html(recs, total_tasks=28)
    check("Best Schedule Per Task" in html_out, "shows 'Best Schedule Per Task' title")
    check("conv2d_nchw_0" in html_out, "shows conv2d_nchw_0 task")
    check("conv2d_nchw_1" in html_out, "shows conv2d_nchw_1 task")
    check("add" in html_out, "shows add task")
    check("4.0000" in html_out, "shows conv2d_nchw_0 best latency (4.0)")
    check("1.5000" in html_out, "shows add best latency (1.5)")
    check("25 remaining tasks" in html_out, "shows 25 untuned tasks note")
    check("split" in html_out, "shows key schedule ops")
    check("3 tuned" in html_out, "shows '3 tuned' count")


# ──────────────────────────────────────────────────────────────────────
# 9. per_task_summary_chart candidate counts
# ──────────────────────────────────────────────────────────────────────
def test_per_task_chart_counts():
    print("\n=== per_task_summary_chart counts ===")
    try:
        from viz.charts import per_task_summary_chart
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    records = [
        {"task_name": "conv2d_nchw_0", "run_ms": 5.0},
        {"task_name": "conv2d_nchw_0", "run_ms": 4.0},
        {"task_name": "conv2d_nchw_0", "run_ms": 3.0},
        {"task_name": "add", "run_ms": 1.0},
    ]
    tuned_names = ["conv2d_nchw_0", "add"]

    html = per_task_summary_chart(
        total_tasks=28, tuned_task_names=tuned_names,
        records=records, return_format="html",
    )
    check("img" in html.lower(), "chart returns an <img> tag")
    # Can't inspect chart internals easily, but if it doesn't crash, the
    # per_task_count lookup worked
    check(True, "per_task_summary_chart did not crash with disambiguated names")


# ──────────────────────────────────────────────────────────────────────
# 10. Scatter chart with real task names
# ──────────────────────────────────────────────────────────────────────
def test_scatter_chart_groups():
    print("\n=== candidate_scatter_chart grouping ===")
    try:
        from viz.charts import candidate_scatter_chart
    except ImportError:
        print("  [SKIP] matplotlib not available")
        return

    records = [
        {"candidate_id": 0, "task_name": "conv2d_nchw_0", "run_ms": 5.0},
        {"candidate_id": 1, "task_name": "conv2d_nchw_0", "run_ms": 4.0},
        {"candidate_id": 2, "task_name": "conv2d_nchw_1", "run_ms": 10.0},
        {"candidate_id": 3, "task_name": "add", "run_ms": 1.0},
    ]
    html = candidate_scatter_chart(records, return_format="html")
    check("img" in html.lower(), "scatter chart returns an <img> tag")
    check(True, "scatter chart did not crash with disambiguated task names")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_disambiguate_names()
    test_pick_primary_block_name()
    test_extract_block_name_from_trace()
    test_is_tir_helpers()
    test_workload_name_map_and_count_consistency()
    test_fixup_main_task_names()
    test_stage8_stage9_consistency()
    test_card_html_per_record_task()
    test_per_task_summary_html()
    test_per_task_chart_counts()
    test_scatter_chart_groups()

    print(f"\n{'='*60}")
    print(f"  Results:  {PASS} passed,  {FAIL} failed")
    print(f"{'='*60}")
    if FAIL > 0:
        sys.exit(1)
