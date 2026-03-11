## TVM Demo – 128-Trial Tuning Run

**Date**: 2026-03-10  
**Model**: ResNet-18  
**Tuning work dir**: `tuning_logs/`  
**Global trial budget (slider)**: 128  
**Total MetaSchedule trials actually run**: 134  
**Total measured candidate records**: 134  

---

### 1. Earlier Stages (0–7): End-to-End Context

These are the key outputs produced *before* tuning for this run. Values are based on the live demo configuration and match what you see in the corresponding tabs.

#### Stage 0–1: PyTorch Baseline (Tab 1)

- **Model loaded**: `resnet18` (**11,689,512 parameters**)
- **Environment** (from the demo banner):
  - Python: **3.10.19**
  - PyTorch: **2.10.0+cu128**
  - CUDA: **Yes**
  - GPU: **NVIDIA GeForce RTX 3060 Ti**
  - CUDA version: **12.8**
  - TVM: **0.20.dev851+gbfd7787ab**
  - TVM → CUDA: **Yes**
- **Input**: 1×3×224×224 ImageNet image (sample cat photo)
- **PyTorch inference results** (this run):
  - Top‑1: **Egyptian cat** (`0.4931`)
  - Top‑5:
    1. Egyptian cat — 0.4931  
    2. tabby — 0.2862  
    3. tiger cat — 0.2059  
    4. lynx — 0.0038  
    5. plastic bag — 0.0030  
  - **Median latency**: **3.58 ms** (CUDA, 50 runs)

#### Stage 2: PyTorch Graph Capture (Tab 2)

- **FX graph nodes captured**: **71**
- **torch.export**: **OK** (used as the source for Relax import)
- The node table shows the full ResNet‑18 structure, including:
  - Stem: `conv1 → bn1 → relu → maxpool`
  - Residual blocks in layers 1–4 with skip connections (`add`, `add_1`, …, `add_7`)
  - Global average pool, flatten, fully‑connected layer, and `output`

Snippet (first few nodes):

```text
x                 op=placeholder  target=x
conv1             op=call_module  target=conv1
bn1               op=call_module  target=bn1
relu              op=call_module  target=relu
maxpool           op=call_module  target=maxpool
layer1_0_conv1    op=call_module  target=layer1.0.conv1
...
fc                op=call_module  target=fc
output            op=output       target=output
```

#### Stage 3: Relax IR Import (Tab 3)

- **IRModule**: **1 functions, 62 parameter arrays, 108 IR lines**

The full Relax IR begins with the 62‑parameter function signature and a `R.dataflow()` region:

```python
@I.ir_module
class Module:
    @R.function
    def main(
        x: R.Tensor((1, 3, 224, 224), dtype="float32"),
        p_conv1_weight: R.Tensor((64, 3, 7, 7), dtype="float32"),
        p_bn1_weight: R.Tensor((64,), dtype="float32"),
        p_bn1_bias: R.Tensor((64,), dtype="float32"),
        ...,    # 62 parameter tensors total
        p_fc_weight: R.Tensor((1000, 512), dtype="float32"),
        p_fc_bias: R.Tensor((1000,), dtype="float32"),
    ) -> R.Tensor((1, 1000), dtype="float32"):
        R.func_attr({"num_input": 1, "params": [...], ...})
        with R.dataflow():
            lv  = R.nn.conv2d(x, p_conv1_weight, strides=[2, 2], padding=[3, 3, 3, 3], ...)
            lv1 = R.nn.batch_norm(lv, p_bn1_weight, p_bn1_bias, ..., training=False)
            lv2 = lv1[0]
            lv3 = R.nn.relu(lv2)
            lv4 = R.nn.max_pool2d(lv3, pool_size=[3, 3], strides=[2, 2], ...)
            ...   # residual blocks for layers 1–4
            lv88 = R.nn.relu(lv87)
            lv89 = R.mean(lv88, axis=[-1, -2], keepdims=True)
            lv90 = R.reshape(lv89, R.shape([1, 512]))
            lv91 = R.permute_dims(p_fc_weight, axes=[1, 0])
            lv92 = R.matmul(lv90, lv91, out_dtype="float32")
            lv93 = R.add(p_fc_bias, lv92)
            gv   = lv93
            R.output(gv)
        return gv
```

The function list below the IR viewer shows a single entry: `main (Function)`.

#### Stage 4: Graph Passes (Tab 4)

For this 128‑trial run, **five graph‑level passes** were applied successfully to the Relax IRModule:

| # | Pass                   | Functions (before → after) | TIR funcs (before → after) | Time |
|---|------------------------|----------------------------|-----------------------------|------|
| 1 | `LegalizeOps`          | 1 → 33                     | 0 → 32                      | 0.106 s |
| 2 | `AnnotateTIROpPattern` | 33 → 33                    | 32 → 32                     | 0.000 s |
| 3 | `FuseOps`              | 33 → 46                    | 32 → 32                     | 0.006 s |
| 4 | `FuseTIR`              | 46 → 29                    | 32 → 28                     | 0.019 s |
| 5 | `DeadCodeElimination`  | 29 → 29                    | 28 → 28                     | 0.001 s |

**FuseOps diff** (selected pass: `FuseOps`). New primitive fused functions are introduced and `main` is rewritten to call them instead of individual ops.

New fused functions added (`+` lines):

```diff
+    @R.function(private=True)
+    def fused_batch_norm4_batch_norm4_add3_relu4(
+        lv72: R.Tensor((1, 512, 7, 7), ...),
+        p_layer4_0_bn2_weight, p_layer4_0_bn2_bias, param_0, param_1,
+        lv75: R.Tensor((1, 512, 7, 7), ...),
+        p_layer4_0_downsample_1_weight, p_layer4_0_downsample_1_bias, param_2, param_3
+    ) -> R.Tensor((1, 512, 7, 7), dtype="float32"):
+        R.func_attr({"Primitive": True})
+        cls = Module
+        with R.dataflow():
+            lv73 = R.call_tir(cls.batch_norm4, (lv72, ...), ...)
+            lv74 = lv73[0]
+            lv76 = R.call_tir(cls.batch_norm4, (lv75, ...), ...)
+            lv77 = lv76[0]
+            lv78 = R.call_tir(cls.add3, (lv74, lv77), ...)
+            gv = R.call_tir(cls.relu4, (lv78,), ...)
+            R.output(gv)
+        return gv
+
+    @R.function(private=True)
+    def fused_batch_norm4_relu4(...) -> R.Tensor((1, 512, 7, 7), ...):
+        R.func_attr({"Primitive": True})
+        ...  # batch_norm4 → relu4
+
+    @R.function(private=True)
+    def fused_batch_norm_relu(...) -> R.Tensor((1, 64, 112, 112), ...):
+        R.func_attr({"Primitive": True})
+        ...  # batch_norm → relu (stem)
+
+    @R.function(private=True)
+    def fused_matmul_add4(lv90, lv91, p_fc_bias) -> R.Tensor((1, 1000), ...):
+        R.func_attr({"Primitive": True})
+        ...  # matmul → add4 (classifier head)
```

Corresponding changes in `main` — individual ops replaced by fused calls:

```diff
     lv = R.call_tir(cls.conv2d, (x, p_conv1_weight), ...)
-    lv1 = R.call_tir(cls.batch_norm, (lv, p_bn1_weight, p_bn1_bias, ...), ...)
-    lv2: R.Tensor(...) = lv1[0]
-    lv3 = R.call_tir(cls.relu, (lv2,), ...)
-    lv4 = R.call_tir(cls.max_pool2d, (lv3,), ...)
+    lv_1: R.Tensor(...) = cls.fused_batch_norm_relu(lv, p_bn1_weight, p_bn1_bias, ...)
+    lv4 = R.call_tir(cls.max_pool2d, (lv_1,), ...)
     ...
     lv8 = R.call_tir(cls.conv2d1, (lv7, p_layer1_0_conv1_weight), ...)
-    lv9 = R.call_tir(cls.batch_norm1, (lv8, ...), ...)
-    lv10 = lv9[0]
-    lv11 = R.call_tir(cls.relu1, (lv10,), ...)
-    lv12 = R.call_tir(cls.conv2d1, (lv11, ...), ...)
-    lv13 = R.call_tir(cls.batch_norm1, (lv12, ...), ...)
-    lv14 = lv13[0]
-    lv15 = R.call_tir(cls.add, (lv14, lv7), ...)
-    lv16 = R.call_tir(cls.relu1, (lv15,), ...)
+    lv_2: R.Tensor(...) = cls.fused_batch_norm1_relu1(lv8, ...)
+    lv12 = R.call_tir(cls.conv2d1, (lv_2, ...), ...)
+    lv_3: R.Tensor(...) = cls.fused_batch_norm1_add_relu1(lv12, ..., lv7, ...)
     ...
```

The same pattern repeats through all residual blocks in layers 1–4: sequences of `batch_norm` → `add` → `relu` are collapsed into single fused function calls like `fused_batch_norm2_batch_norm2_add1_relu2`, `fused_batch_norm3_add2_relu3`, etc.

The net result after `FuseTIR` is that the IRModule contains **28 TIR PrimFuncs**, which are exactly the operators exposed to MetaSchedule as tuning tasks in Stage 8.

#### Stage 5: Extracted Operators (Tab 5)

- **Total TIR PrimFuncs**: **28**
- **Kind breakdown**: batchnorm **12**, conv **11**, matmul **1**, pool **1**, other **1**, reshape **1**, transpose **1**
- Blocks = count of explicit TIR Block nodes; many lowered GPU kernels rely mostly on loops, so it is normal to see 0 here.

Full operator table:

| #  | Name | Kind | Params | Blocks* | IR Lines |
|----|------|------|--------|---------|----------|
| 1  | `conv2d` | conv | f32[1×3×224×224], f32[64×3×7×7], f32[1×64×112×112] | 0 | 21 |
| 2  | `conv2d1` | conv | f32[1×64×56×56], f32[64×64×3×3], f32[1×64×56×56] | 0 | 21 |
| 3  | `conv2d10` | conv | f32[1×256×14×14], f32[512×256×1×1], f32[1×512×7×7] | 0 | 21 |
| 4  | `conv2d2` | conv | f32[1×64×56×56], f32[128×64×3×3], f32[1×128×28×28] | 0 | 21 |
| 5  | `conv2d3` | conv | f32[1×128×28×28], f32[128×128×3×3], f32[1×128×28×28] | 0 | 21 |
| 6  | `conv2d4` | conv | f32[1×64×56×56], f32[128×64×1×1], f32[1×128×28×28] | 0 | 21 |
| 7  | `conv2d5` | conv | f32[1×128×28×28], f32[256×128×3×3], f32[1×256×14×14] | 0 | 21 |
| 8  | `conv2d6` | conv | f32[1×256×14×14], f32[256×256×3×3], f32[1×256×14×14] | 0 | 21 |
| 9  | `conv2d7` | conv | f32[1×128×28×28], f32[256×128×1×1], f32[1×256×14×14] | 0 | 21 |
| 10 | `conv2d8` | conv | f32[1×256×14×14], f32[512×256×3×3], f32[1×512×7×7] | 0 | 21 |
| 11 | `conv2d9` | conv | f32[1×512×7×7], f32[512×512×3×3], f32[1×512×7×7] | 0 | 21 |
| 12 | `fused_batch_norm1_add_relu1` | batchnorm | f32[1×64×56×56], f32[64], f32[64], f32[64], f32[64], f32[1×64×56×56], f32[1×64×56×56] | 0 | 103 |
| 13 | `fused_batch_norm1_relu1` | batchnorm | f32[1×64×56×56], f32[64], f32[64], f32[64], f32[64], f32[1×64×56×56] | 0 | 96 |
| 14 | `fused_batch_norm2_add1_relu2` | batchnorm | f32[1×128×28×28], f32[128], f32[128], f32[128], f32[128], f32[1×128×28×28], f32[1×128×28×28] | 0 | 103 |
| 15 | `fused_batch_norm2_batch_norm2_add1_relu2` | batchnorm | f32[1×128×28×28], f32[128]×4, f32[1×128×28×28], f32[128]×4, f32[1×128×28×28] | 0 | 187 |
| 16 | `fused_batch_norm2_relu2` | batchnorm | f32[1×128×28×28], f32[128], f32[128], f32[128], f32[128], f32[1×128×28×28] | 0 | 96 |
| 17 | `fused_batch_norm3_add2_relu3` | batchnorm | f32[1×256×14×14], f32[256], f32[256], f32[256], f32[256], f32[1×256×14×14], f32[1×256×14×14] | 0 | 103 |
| 18 | `fused_batch_norm3_batch_norm3_add2_relu3` | batchnorm | f32[1×256×14×14], f32[256]×4, f32[1×256×14×14], f32[256]×4, f32[1×256×14×14] | 0 | 187 |
| 19 | `fused_batch_norm3_relu3` | batchnorm | f32[1×256×14×14], f32[256], f32[256], f32[256], f32[256], f32[1×256×14×14] | 0 | 96 |
| 20 | `fused_batch_norm4_add3_relu4` | batchnorm | f32[1×512×7×7], f32[512], f32[512], f32[512], f32[512], f32[1×512×7×7], f32[1×512×7×7] | 0 | 103 |
| 21 | `fused_batch_norm4_batch_norm4_add3_relu4` | batchnorm | f32[1×512×7×7], f32[512]×4, f32[1×512×7×7], f32[512]×4, f32[1×512×7×7] | 0 | 187 |
| 22 | `fused_batch_norm4_relu4` | batchnorm | f32[1×512×7×7], f32[512], f32[512], f32[512], f32[512], f32[1×512×7×7] | 0 | 96 |
| 23 | `fused_batch_norm_relu` | batchnorm | f32[1×64×112×112], f32[64], f32[64], f32[64], f32[64], f32[1×64×112×112] | 0 | 96 |
| 24 | `fused_matmul_add4` | matmul | f32[1×512], f32[512×1000], f32[1000], f32[1×1000] | 0 | 21 |
| 25 | `max_pool2d` | pool | f32[1×64×112×112], f32[1×64×56×56] | 0 | 22 |
| 26 | `mean` | other | f32[1×512×7×7], f32[1×512×1×1] | 0 | 21 |
| 27 | `reshape` | reshape | f32[1×512×1×1], f32[1×512] | 0 | 12 |
| 28 | `transpose` | transpose | f32[1000×512], f32[512×1000] | 0 | 12 |

#### Stage 6: TensorIR / AST (Tab 6)

Selected PrimFunc: **`conv2d` [conv]**

- **Summary**: 0 blocks, 11 for-loops, 3 buffers

**TIR Source** (21 lines):

```python
# from tvm.script import tir as T

@T.prim_func(private=True)
def main(
    x: T.Buffer((T.int64(1), T.int64(3), T.int64(224), T.int64(224)), "float32"),
    p_conv1_weight: T.Buffer((T.int64(64), T.int64(3), T.int64(7), T.int64(7)), "float32"),
    conv2d_nchw: T.Buffer((T.int64(1), T.int64(64), T.int64(112), T.int64(112)), "float32"),
):
    T.func_attr({"op_pattern": 4, "tir.noalias": True})
    # with T.sblock("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(230), T.int64(230)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(3), T.int64(230), T.int64(230)):
        with T.sblock("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(x[v_i0, v_i1, v_i2 - T.int64(3), v_i3 - T.int64(3)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(
                T.int64(3) <= v_i2 and v_i2 < T.int64(227)
                and T.int64(3) <= v_i3 and v_i3 < T.int64(227),
                x[v_i0, v_i1, v_i2 - T.int64(3), v_i3 - T.int64(3)],
                T.float32(0.0),
            )
    for nn, ff, yy, xx, rc, ry, rx in T.grid(
        T.int64(1), T.int64(64), T.int64(112), T.int64(112),
        T.int64(3), T.int64(7), T.int64(7),
    ):
        with T.sblock("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap(
                "SSSSRRR", [nn, ff, yy, xx, rc, ry, rx]
            )
            T.reads(
                pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx],
                p_conv1_weight[v_ff, v_rc, v_ry, v_rx],
            )
            T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = (
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx]
                + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx]
                * p_conv1_weight[v_ff, v_rc, v_ry, v_rx]
            )
```

**Buffers**:

| Buffer | Dtype | Shape | Scope |
|--------|-------|-------|-------|
| `x` | float32 | 1×3×224×224 | |
| `p_conv1_weight` | float32 | 64×3×7×7 | |
| `conv2d_nchw` | float32 | 1×64×112×112 | |

**Iteration Structure** (11 for-loops, no thread bindings — this is the *naive* loop nest before scheduling):

```text
for i3 in [0, 230)        # pad width
  for i2 in [0, 230)      # pad height
    for i1 in [0, 3)      # input channels
      for i0 in [0, 1)    # batch
for rx in [0, 7)           # kernel width (reduction)
  for ry in [0, 7)         # kernel height (reduction)
    for rc in [0, 3)       # input channels (reduction)
      for xx in [0, 112)   # output width
        for yy in [0, 112) # output height
          for ff in [0, 64) # output filters
            for nn in [0, 1) # batch
```

Loop detail table:

| # | Variable | Extent | Kind | Thread Binding | Source |
|---|----------|--------|------|----------------|--------|
| 0 | `i3` | 230 | 0 | | for loop |
| 1 | `i2` | 230 | 0 | | for loop |
| 2 | `i1` | 3 | 0 | | for loop |
| 3 | `i0` | 1 | 0 | | for loop |
| 4 | `rx` | 7 | 0 | | for loop |
| 5 | `ry` | 7 | 0 | | for loop |
| 6 | `rc` | 3 | 0 | | for loop |
| 7 | `xx` | 112 | 0 | | for loop |
| 8 | `yy` | 112 | 0 | | for loop |
| 9 | `ff` | 64 | 0 | | for loop |
| 10 | `nn` | 1 | 0 | | for loop |

This is the naive but correct loop nest that schedules will later tile, fuse, and bind to GPU threads.

#### Stage 7: Compute / Schedule Separation — Microscope (Tab 7)

- **Microscope operator built**: `conv2d 64x64x3x3`

**TE Compute Declaration**:

```python
data   = te.placeholder((1, 64, 56, 56), name="data", dtype="float32")
weight = te.placeholder((64, 64, 3, 3), name="weight", dtype="float32")

out = topi.nn.conv2d(
    data, weight,
    strides=1, padding=1, dilation=1,
    data_layout="NCHW",
)
```

**Naive TIR (un-scheduled)**:

```python
# from tvm.script import tir as T

@T.prim_func
def main(data: T.Buffer((1, 64, 56, 56), "float32"),
         weight: T.Buffer((64, 64, 3, 3), "float32"),
         conv2d_nchw: T.Buffer((1, 64, 56, 56), "float32")):
    T.func_attr({"tir.noalias": True})
    # with T.sblock("root"):
    pad_temp = T.alloc_buffer((1, 64, 58, 58))
    for i0, i1, i2, i3 in T.grid(1, 64, 58, 58):
        with T.sblock("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(data[v_i0, v_i1, v_i2 - 1, v_i3 - 1])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(
                1 <= v_i2 and v_i2 < 57 and 1 <= v_i3 and v_i3 < 57,
                data[v_i0, v_i1, v_i2 - 1, v_i3 - 1], T.float32(0.0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(1, 64, 56, 56, 64, 3, 3):
        with T.sblock("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap(
                "SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx],
                    weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(conv2d_nchw[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = T.float32(0.0)
            conv2d_nchw[v_nn, v_ff, v_yy, v_xx] = (
                conv2d_nchw[v_nn, v_ff, v_yy, v_xx]
                + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx]
                * weight[v_ff, v_rc, v_ry, v_rx])
```

**Explanation** (from the UI):

> TVM separates the *what* (tensor expression) from the *how* (schedule).
>
> The tensor expression above declares a 2-D convolution using a reduction over the input channels and kernel spatial dimensions. No loop ordering, tiling, or threading is specified — that is the job of the schedule.
>
> The "Naive TIR" panel shows the default lowered loop nest: a simple set of nested for-loops that directly implement the compute declaration. In the next stages (Schedule Search, Tuning), TVM's MetaSchedule will explore thousands of schedule transformations — tile sizes, loop reordering, thread bindings, cache reads/writes — to find an optimized version of this program for the target GPU.

---

### 2. Stage 8–9: Tuning (Schedule Search & Cost Model)

#### Tuning Progression

Tuning ran from **11:40:13** to **11:40:17** (~4 seconds wall time for 134 trials on RTX 3060 Ti). The task scheduler processed tasks in order, printing an updated table each time a task completed. Timestamps and task‑finish messages from the terminal:

```text
11:40:13 [INFO] [Updated] Task #6: "fused_batch_norm4_relu4"        → Total trials: 86
11:40:13 [INFO] Task #6 has finished. Remaining task(s): 21         → Total trials: 86
11:40:14 [INFO] [Updated] Task #7: "fused_batch_norm3_add2_relu3"   → Total trials: 102
11:40:14 [INFO] Task #7 has finished. Remaining task(s): 20         → Total trials: 102
...
11:40:16 [INFO] [Updated] Task #8: "fused_batch_norm3_relu3"        → Total trials: 118
11:40:16 [INFO] Task #8 has finished. Remaining task(s): 19
11:40:16 [INFO] [Updated] Task #9: "fused_batch_norm1_add_relu1"    → Total trials: 134
11:40:16 [INFO] Task #9 has finished. Remaining task(s): 18
11:40:17 [INFO] Task #10 has finished. Remaining task(s): 17        (DLight — 0 trials)
11:40:17 [INFO] Task #11 has finished. Remaining task(s): 16        (DLight — 0 trials)
...
11:40:17 [INFO] Task #27 has finished. Remaining task(s): 0         (DLight — 0 trials)
```

After the last tuned task (#9) finished at 134 trials, tasks #10–#27 were immediately marked done with 0 trials (they fall back to DLight default GPU schedules).

Between tasks, the XGBoost cost model was retrained. Example from the `fused_batch_norm1_add_relu1` round:

```text
2026-03-10 11:40:16 [DEBUG] XGB iter   0: tr-p-rmse: 0.513037   tr-a-peak@32: 0.980763  tr-rmse: 0.560556
2026-03-10 11:40:16 [DEBUG] XGB iter  25: tr-p-rmse: 0.151580   tr-a-peak@32: 0.999729  tr-rmse: 0.612036
2026-03-10 11:40:16 [DEBUG] XGB iter  50: tr-p-rmse: 0.151584   tr-a-peak@32: 0.999729  tr-rmse: 0.612033
2026-03-10 11:40:16 [DEBUG] XGB stopped. Best iteration: [22] tr-p-rmse:0.15154 tr-a-peak@32:0.99973    tr-rmse:0.61211
```

And an earlier XGB round (after `fused_batch_norm4_relu4`):

```text
2026-03-10 11:40:13 [DEBUG] XGB iter   0: tr-p-rmse: 0.540609   tr-a-peak@32: 0.989087  tr-rmse: 0.577580
2026-03-10 11:40:13 [DEBUG] XGB iter  25: tr-p-rmse: 0.163756   tr-a-peak@32: 0.999729  tr-rmse: 0.615249
2026-03-10 11:40:13 [DEBUG] XGB stopped. Best iteration: [25] tr-p-rmse:0.16376 tr-a-peak@32:0.99973    tr-rmse:0.61525
```

Pipeline summary:

```text
11:40:17  INFO  backend.pipeline  Tuning complete: 134 records, 134 convergence points
11:40:17  INFO  backend.pipeline  Best candidate: #11 (lv88_red) at 0.0088 ms
```

#### Tuning Tasks and Coverage

TVM's MetaSchedule task scheduler initialized **28 tunable operators** from the Relax IRModule (conv layers, fused batch-norm + ReLU blocks, mean, pooling, etc.).  

After using a 128‑trial budget, the scheduler assigned **trials to 10 distinct workloads** (workload indices 0–9), which our naming logic maps to:

- `transpose`
- `reshape`
- `lv88_red` (layer‑norm style reduction)
- `mean`
- `conv2d10`
- `conv2d9`
- `fused_batch_norm4_batch_norm4_add3_relu4`
- `fused_batch_norm4_relu4`
- `fused_batch_norm3_add2_relu3`
- `fused_batch_norm3_relu3`
- `fused_batch_norm1_add_relu1`

From the MetaSchedule task table near the end of the run:

- Light ops such as `transpose`, `reshape`, and `mean` reached **1, 5, and 16 trials** respectively and were marked **Done = Y** early.
- Heavy conv layers (`conv2d10`, `conv2d9`) each accumulated **16 trials**, with latencies on the order of **17.5 µs** (for `conv2d10`) and **430.9 µs** (for `conv2d9`) in the TVM logs.
- Several fused batch‑norm + ReLU workloads (`fused_batch_norm4_*`, `fused_batch_norm3_*`, `fused_batch_norm1_add_relu1`) each reached **16 trials** and were also marked **Done = Y**.
- Remaining tasks (later convolutions, some fused batch‑norms, pooling, matmul) were initialized but kept at **0 trials** and thus fell back to **DLight default schedules**. They remain listed with `Latency (us) = N/A` in the task table.

Net effect: **roughly 10–11 of 28 operators** received real tuning; the rest used DLight defaults.

#### Final MetaSchedule Task Scheduler Table

The final scheduler table after all 134 trials (and all 28 tasks marked done):

```text
ID |                                     Name |      FLOP | Weight | Speed (GFLOPS) | Latency (us) | Weighted Latency (us) | Trials | Done
--------------------------------------------------------------------------------------------------------------------------------------------
 0 |                                transpose |         1 |      1 |         0.0000 |      23.2332 |               23.2332 |      1 |    Y
 1 |                                  reshape |         1 |      1 |         0.0001 |       9.1473 |                9.1473 |      5 |    Y
 2 |                                     mean |     25600 |      1 |         2.8974 |       8.8356 |                8.8356 |     16 |    Y
 3 | fused_batch_norm4_batch_norm4_add3_relu4 |    251904 |      1 |         5.6486 |      44.5958 |               44.5958 |     16 |    Y
 4 |                                 conv2d10 |  12845056 |      1 |       732.3147 |      17.5404 |               17.5404 |     16 |    Y
 5 |                                  conv2d9 | 231211008 |      3 |       536.6018 |     430.8801 |             1292.6403 |     16 |    Y
 6 |                  fused_batch_norm4_relu4 |    125952 |      2 |         4.7836 |      26.3298 |               52.6595 |     16 |    Y
 7 |             fused_batch_norm3_add2_relu3 |    301312 |      1 |        11.2456 |      26.7937 |               26.7937 |     16 |    Y
 8 |                  fused_batch_norm3_relu3 |    251136 |      2 |         9.6547 |      26.0117 |               52.0233 |     16 |    Y
 9 |              fused_batch_norm1_add_relu1 |   1204288 |      2 |        48.9195 |      24.6178 |               49.2355 |     16 |    Y
10 |                    fused_batch_norm_relu |   4014144 |      1 |            N/A |          N/A |                   N/A |      0 |    Y
11 |                               max_pool2d |   1806336 |      1 |            N/A |          N/A |                   N/A |      0 |
12 |                  fused_batch_norm2_relu2 |    501888 |      2 |            N/A |          N/A |                   N/A |      0 |
13 |             fused_batch_norm2_add1_relu2 |    602240 |      1 |            N/A |          N/A |                   N/A |      0 |
14 |                  fused_batch_norm1_relu1 |   1003584 |      2 |            N/A |          N/A |                   N/A |      0 |
15 |                                  conv2d7 |  12845056 |      1 |            N/A |          N/A |                   N/A |      0 |
16 |                                  conv2d2 | 115605504 |      1 |            N/A |          N/A |                   N/A |      0 |
17 |                        fused_matmul_add4 |   1025000 |      1 |            N/A |          N/A |                   N/A |      0 |
18 |             fused_batch_norm4_add3_relu4 |    151040 |      1 |            N/A |          N/A |                   N/A |      0 |
19 |                                  conv2d3 | 231211008 |      3 |            N/A |          N/A |                   N/A |      0 |
20 | fused_batch_norm3_batch_norm3_add2_relu3 |    502272 |      1 |            N/A |          N/A |                   N/A |      0 |
21 |                                  conv2d4 |  12845056 |      1 |            N/A |          N/A |                   N/A |      0 |
22 |                                   conv2d | 236027904 |      1 |            N/A |          N/A |                   N/A |      0 |
23 |                                  conv2d1 | 231211008 |      4 |            N/A |          N/A |                   N/A |      0 |
24 | fused_batch_norm2_batch_norm2_add1_relu2 |   1003776 |      1 |            N/A |          N/A |                   N/A |      0 |
25 |                                  conv2d8 | 115605504 |      1 |            N/A |          N/A |                   N/A |      0 |
26 |                                  conv2d6 | 231211008 |      3 |            N/A |          N/A |                   N/A |      0 |
27 |                                  conv2d5 | 115605504 |      1 |            N/A |          N/A |                   N/A |      0 |
--------------------------------------------------------------------------------------------------------------------------------------------
Total trials: 134
Total latency (us): 1576.7
```

### 3. Per-Task Best Latencies (from Stage 9 Summary)

Using the actual `database_tuning_record.json` for this 128‑trial run, the per‑task best latencies and candidate counts are:

| Task name             | Best latency (ms) | Candidates | Notes                                |
|-----------------------|-------------------|-----------:|--------------------------------------|
| `reshape`             | **0.0089**        | 5          | Light reshape kernel                 |
| `transpose`           | **0.0241**        | 1          | Single candidate explored            |
| `lv88_red`            | **0.0113**        | 16         | Layer‑norm style reduction           |
| `conv2d_nchw_0`       | **0.0227**        | 16 (2 TO)  | First heavy conv (some timeouts)    |
| `conv2d_nchw_1`       | **0.2628**        | 16 (1 TO)  | Later conv, heavier, few good tiles |
| `batch_norm_fused_3`  | **0.0278**        | 16         | Fused batch‑norm + add + ReLU       |
| `batch_norm_fused_1`  | **0.0282**        | 16         | Fused batch‑norm + add + ReLU       |
| `batch_norm_fused_2`  | **0.0286**        | 16         | Fused batch‑norm + add + ReLU       |
| `batch_norm_fused_4`  | **0.0287**        | 16         | Fused batch‑norm + add + ReLU       |
| `batch_norm_fused_0`  | **0.0521**        | 16         | Largest fused BN block               |

TO = timeouts (candidates with `inf` latency); these do **not** affect the "best" latency but are counted in the candidate totals.

### 4. Representative Concise Traces

From the resolved schedule traces (with decisions substituted for symbolic variables):

- **`reshape` best candidate (`#3`, 0.0092 ms)**  
  ```text
  sch.fuse(l1, l2)
  sch.split(loop=l3, factors=[None, 32])
  sch.bind(loop=l5, thread_axis="blockIdx.x")
  sch.bind(loop=l6, thread_axis="threadIdx.x")
  ```
  Other `reshape` candidates differ only in the second factor: `32`, `64`, `128`, `256`, `512` etc., visible directly in the `factors=[None, X]` list.

- **`lv88_red` best candidate (`#17`, 0.0108 ms)**  
  ```text
  sch.split(loop=l7, factors=[None, 32])
  sch.bind(loop=l10, thread_axis="threadIdx.x")
  sch.set_scope(b0)
  sch.fuse(l19, l20)
  sch.split(loop=l21, factors=[None, 32])
  sch.bind(loop=l23, thread_axis="threadIdx.x")
  sch.fuse(l25, l26, l27, l28)
  sch.bind(loop=l30, thread_axis="blockIdx.x")
  ```
  Slower `lv88_red` candidates use smaller or larger thread counts (e.g. 8, 256) in their split factors, which you can see directly in the trace.

- **`conv2d_nchw_0` candidates (tile shapes)** – examples of the first few concise traces:
  ```text
  #49 (0.0382 ms):
    sch.split(loop=l3, factors=[1, 1, 1, 1, 1])
    sch.split(loop=l4, factors=[1, 16, 32, 1, 1])
    sch.split(loop=l5, factors=[7, 1, 1, 1, 1])
    sch.split(loop=l6, factors=[1, 1, 7, 1, 1])
    ...

  #53 (0.0418 ms):
    sch.split(loop=l3, factors=[1, 1, 1, 1, 1])
    sch.split(loop=l4, factors=[8, 2, 32, 1, 1])
    sch.split(loop=l5, factors=[7, 1, 1, 1, 1])
    sch.split(loop=l6, factors=[7, 1, 1, 1, 1])
    ...

  #42 (0.0422 ms):
    sch.split(loop=l3, factors=[1, 1, 1, 1, 1])
    sch.split(loop=l4, factors=[4, 1, 32, 1, 4])
    sch.split(loop=l5, factors=[1, 7, 1, 1, 1])
    sch.split(loop=l6, factors=[7, 1, 1, 1, 1])
    ...
  ```

You can see how different tile factorizations (e.g. `[8, 2, 32, 1, 1]` vs `[4, 1, 32, 1, 4]`) correspond to different performance points for the same conv2d workload.

### 5. Stage 10: Compile & Run (TVM vs PyTorch)

After tuning completed, the compile stage applied tuning records and built a CUDA module:

```text
11:54:06  INFO  backend.pipeline  Building TVM module for target:
  {"kind":"cuda","tag":"","keys":["cuda","gpu"],
   "max_num_threads":1024,"thread_warp_size":32,
   "max_shared_memory_per_block":49152,
   "max_threads_per_block":1024,"arch":"sm_86"}
```

TVM's `MetaScheduleApplyDatabase` loaded the tuning records. For the **18 untuned operators**, it logged warnings:

```text
[11:54:06] Warning: Tuning record is not found for primfunc: conv2d
[11:54:06] Warning: Tuning record is not found for primfunc: max_pool2d
[11:54:06] Warning: Tuning record is not found for primfunc: conv2d1
[11:54:06] Warning: Tuning record is not found for primfunc: conv2d2
[11:54:06] Warning: Tuning record is not found for primfunc: conv2d3
[11:54:06] Warning: Tuning record is not found for primfunc: conv2d4
[11:54:06] Warning: Tuning record is not found for primfunc: conv2d5
[11:54:06] Warning: Tuning record is not found for primfunc: conv2d6
[11:54:06] Warning: Tuning record is not found for primfunc: conv2d7
[11:54:06] Warning: Tuning record is not found for primfunc: conv2d8
[11:54:06] Warning: Tuning record is not found for primfunc: fused_batch_norm1_relu1
[11:54:06] Warning: Tuning record is not found for primfunc: fused_batch_norm2_add1_relu2
[11:54:06] Warning: Tuning record is not found for primfunc: fused_batch_norm2_batch_norm2_add1_relu2
[11:54:06] Warning: Tuning record is not found for primfunc: fused_batch_norm2_relu2
[11:54:06] Warning: Tuning record is not found for primfunc: fused_batch_norm3_batch_norm3_add2_relu3
[11:54:06] Warning: Tuning record is not found for primfunc: fused_batch_norm4_add3_relu4
[11:54:06] Warning: Tuning record is not found for primfunc: fused_batch_norm_relu
[11:54:06] Warning: Tuning record is not found for primfunc: fused_matmul_add4
```

These 18 operators were handled by DLight default GPU schedule rules:

```text
11:54:06  INFO  backend.pipeline  MetaScheduleApplyDatabase applied from ./tuning_logs
11:54:06  INFO  backend.pipeline  Bound 62 params as constants
11:54:06  INFO  backend.pipeline  DLight applied 5 GPU schedule rules
11:54:14  INFO  backend.pipeline  Build with tuning DB succeeded — tuning schedules applied
```

Final inference result:

```text
11:54:15  INFO  backend.pipeline  TVM inference: Egyptian cat  (3.87 ms)
```

| Metric | PyTorch (Stage 1) | TVM (Stage 10) |
|--------|-------------------|----------------|
| Prediction | Egyptian cat | Egyptian cat |
| Median latency | **3.58 ms** | **3.87 ms** |

With only 128 trials (covering 10 of 28 operators), TVM's latency is slightly higher than PyTorch's highly optimized cuDNN kernels. With a larger trial budget (e.g. 2000+), TVM typically matches or beats cuDNN as more operators receive dedicated tuning.

### 6. How to Reproduce

To re-run a similar 128‑trial experiment:

```bash
python app.py        # or use your WSL one-liner that points to ~/tvm_env/bin/python
```

Then in the UI:

1. Run Tabs 1–7 to populate the model and IR.
2. In **Tab 8**, set the tuning slider to **128 trials**.
3. Click **"Run Stages 8–9"** and wait for tuning to complete.
4. Optionally open `tuning_logs/database_tuning_record.json` and re-run the internal probes (or reuse this `docs/run_128_trials.md`) to inspect the resulting tasks, candidates, and latencies.

---

### 7. Consistency Revalidation (2026-03-11)

A full headless end-to-end validation was re-run in WSL using the current code:

```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/woom2/Documents/TVM-demo && ~/tvm_env/bin/python -m src.tests.test_pipeline"
```

Observed outputs (current code path):

- Stage 3: `IRModule: 1 functions, 107 IR lines`
- Stage 4 pass deltas:
  - `LegalizeOps`: `1→33`, TIR `0→32`, `0.088s`
  - `FuseOps`: `33→46`, TIR `32→32`, `0.004s`
  - `FuseTIR`: `46→29`, TIR `32→28`, `0.015s`
- Stage 5: `Extracted 28 TIR operators`
- Stage 6: `TIR AST for 'conv2d': 0 blocks, 11 loops, 3 buffers`
- Stage 8: `Extracted 28 tuning tasks`
- Stage 9: `Tuning complete: 22 records, 22 convergence points` (smoke-test budget)
- Stage 10: `Best candidate: #1 (reshape) at 0.0126 ms`
- Stage 12: `TVM inference: Egyptian cat (6.72 ms)`, max abs diff `0.003178`, cosine `1.000000`

All smoke-test assertions passed (`stage_0` through `stage_12` = `done`).

> Note: this section is a **pipeline consistency check** from the automated smoke test (small trial budget), while the rest of this document is the **manual 128-trial demo run record**.
