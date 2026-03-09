"""Probe real MetaSchedule tuning via tvm.s_tir.meta_schedule."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import pipeline first to get tp_dealloc fix
from backend.pipeline import _ms_mod, _dlight_available

ms = _ms_mod
print(f"MetaSchedule module: {ms}")
print(f"tune_tir: {hasattr(ms, 'tune_tir')}")
print(f"TuneConfig: {hasattr(ms, 'TuneConfig')}")

# Check tune_tir signature
if hasattr(ms, 'tune_tir'):
    fn = ms.tune_tir
    print(f"tune_tir: {fn}")
    import inspect
    try:
        sig = inspect.signature(fn)
        print(f"  signature: {sig}")
    except Exception:
        print(f"  (can't inspect signature)")

# Check relax_integration.tune_relax
ri = getattr(ms, "relax_integration", None)
if ri:
    print(f"\nrelax_integration.tune_relax: {hasattr(ri, 'tune_relax')}")
    print(f"relax_integration.tune_tasks: {hasattr(ri, 'tune_tasks')}")
    if hasattr(ri, "tune_relax"):
        fn = ri.tune_relax
        print(f"  tune_relax: {fn}")
        try:
            sig = inspect.signature(fn)
            print(f"  signature: {sig}")
        except Exception:
            pass

# Check TuneConfig alternatives
for attr in dir(ms):
    if "tune" in attr.lower() or "config" in attr.lower():
        print(f"  ms.{attr}: {getattr(ms, attr)}")

# Try a quick tune on a tiny function
print("\n=== Quick tuning test ===")
import tvm
from tvm import tir

# Create a tiny test module
from tvm.script import tir as T

@T.prim_func
def matmul(A: T.Buffer((128, 128), "float32"),
           B: T.Buffer((128, 128), "float32"),
           C: T.Buffer((128, 128), "float32")):
    T.func_attr({"tir.noalias": True})
    for i, j, k in T.grid(128, 128, 128):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

test_mod = tvm.IRModule({"main": matmul})
target = tvm.target.Target("cuda")

# Try tune_tir
try:
    db = ms.tune_tir(
        mod=test_mod,
        target=target,
        config=ms.TuneConfig(
            max_trials_global=8,
            num_trials_per_iter=4,
        ),
        work_dir="/tmp/tvm_tune_test",
    )
    print(f"[OK] tune_tir returned: {type(db)}")
except Exception as e:
    print(f"[FAIL] tune_tir: {e}")

# Try tune_relax
if ri and hasattr(ri, "tune_relax"):
    try:
        db = ri.tune_relax(
            mod=test_mod,
            target=target,
            params={},
            config=ms.TuneConfig(
                max_trials_global=8,
                num_trials_per_iter=4,
            ),
            work_dir="/tmp/tvm_tune_test2",
        )
        print(f"[OK] tune_relax returned: {type(db)}")
    except Exception as e:
        print(f"[FAIL] tune_relax: {e}")
