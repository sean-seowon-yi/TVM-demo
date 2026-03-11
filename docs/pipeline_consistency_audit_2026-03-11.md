# Pipeline Consistency Audit (2026-03-11)

This audit checks whether stage outputs, UI displays, tests, and docs are aligned with the current implementation in `app.py`, `src/backend/pipeline.py`, and `src/viz/*`.

## Scope

- End-to-end headless run in WSL
- Stage/report consistency review for Tabs 1–10
- Test harness sanity checks
- Documentation alignment updates

## Verification Runs

## 1) End-to-end smoke test (WSL)

Command:

```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/woom2/Documents/TVM-demo && ~/tvm_env/bin/python -m src.tests.test_pipeline"
```

Result: **PASS** (exit code `0`)

Highlights from output:

- Stage 4 passes executed with expected deltas (`1→33→46→29` functions, `0→32→32→28` TIR funcs)
- Stage 5 extracted **28 TIR operators**
- Stage 6 reported **0 blocks, 11 loops, 3 buffers** for `conv2d`
- Stage 8 extracted **28 tuning tasks**
- Stage 9 produced **22 records** in smoke-test budget
- Stage 10 selected best candidate (`reshape`)
- Stage 12 correctness passed (`max abs diff < 0.05`, cosine > 0.99, top-1 match)
- Stage statuses: `stage_0` … `stage_12` all `done`

## 2) Recent-fixes regression script

Command:

```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/woom2/Documents/TVM-demo && ~/tvm_env/bin/python tests/test_recent_fixes.py"
```

Result: **PASS** (`73 passed, 0 failed`)

## Findings and Resolutions (No pipeline functionality changes)

1. **Stale smoke test API usage**
   - `src/tests/test_pipeline.py` called `candidate_cards_html(..., max_display=5)` but the function no longer accepts `max_display`.
   - **Resolution**: updated test harness call to `candidate_cards_html(records)`.

2. **Removed visualization helper still imported by tests**
   - `src/tests/test_pipeline.py` imported `best_candidate_banner_html`, which is no longer exported/used.
   - **Resolution**: removed stale import + dead usage from test harness.

3. **Recent-fixes fixtures out of date with current task-name fixup path**
   - `tests/test_recent_fixes.py` expected `_fixup_main_task_names` behavior without trace text.
   - Current logic resolves names from each record’s `trace_text`.
   - **Resolution**: updated fixtures to include `sch.get_sblock(...)` traces and aligned expected assertions.

## Documentation Updates Completed

- `README.md`
  - Updated Tab 8/9/10 descriptions to match current UI outputs.
  - Updated stage summaries for stages 9–10.
  - Added regression test command (`python tests/test_recent_fixes.py`).

- `docs/run_128_trials.md`
  - Added a new revalidation section with current WSL smoke-test outputs.
  - Explicitly distinguishes 128-trial manual run log vs smoke-test consistency run.

- `PLAN.md`
  - Added a status note that this is a historical plan; current code + README are source of truth.

- `src/docs/PHASE1.md`, `src/docs/PHASE2.md`, `src/docs/PHASE3.md`, `src/docs/PHASE4.md`
  - Added status notes indicating historical phase logs and pointing to current source-of-truth docs/code.

## Final Status

- Pipeline behavior: **unchanged**
- Tests: **green for end-to-end smoke + recent-fixes script**
- Documentation: **updated to reflect current UI and stage behavior**
