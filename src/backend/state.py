"""Central state object that holds every artifact produced by the demo pipeline.

Each stage writes its outputs here so that subsequent stages and UI tabs can
read them without recomputation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DemoState:
    """Mutable container for all demo artifacts, one field-group per stage."""

    # ------------------------------------------------------------------
    # Stage tracking
    # ------------------------------------------------------------------
    stage_status: Dict[str, StageStatus] = field(default_factory=lambda: {
        f"stage_{i}": StageStatus.PENDING for i in range(14)
    })
    stage_logs: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Stage 0 — Load Model
    # ------------------------------------------------------------------
    model: Any = None                         # torch.nn.Module
    model_name: str = "resnet18"
    model_summary: str = ""
    param_count: int = 0
    transform: Any = None                     # torchvision transform
    categories: Optional[List[str]] = None    # ImageNet class labels

    # ------------------------------------------------------------------
    # Stage 1 — PyTorch Baseline Inference
    # ------------------------------------------------------------------
    input_image: Any = None                   # PIL.Image
    input_tensor: Any = None                  # torch.Tensor (1,3,224,224)
    input_np: Optional[np.ndarray] = None     # numpy copy for TVM
    pytorch_logits: Optional[np.ndarray] = None
    pytorch_top5: Optional[List[dict]] = None # [{"class": str, "prob": float}]
    pytorch_latency_ms: float = 0.0

    # ------------------------------------------------------------------
    # Stage 2 — PyTorch Computation Graph  (Pass 2)
    # ------------------------------------------------------------------
    fx_graph: Any = None                      # torch.fx.Graph
    _fx_node_count: int = 0                   # cached for timeline (avoid iterating)
    fx_code: str = ""
    exported_program: Any = None              # torch.export.ExportedProgram

    # ------------------------------------------------------------------
    # Stage 3 — TVM Relax Import
    # ------------------------------------------------------------------
    imported_mod: Any = None                  # tvm.ir.IRModule
    imported_mod_num_funcs: int = 0           # cached for timeline (avoid touching TVM obj)
    ir_snapshots: Dict[str, str] = field(default_factory=dict)
    model_params_np: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # Stage 4 — TVM Passes
    # ------------------------------------------------------------------
    current_mod: Any = None                   # IRModule after all passes
    pass_order: List[str] = field(default_factory=list)
    pass_deltas: Dict[str, dict] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Stage 5 — Extracted Operators  (Pass 2)
    # ------------------------------------------------------------------
    operators: List[dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Stage 6 — TensorIR / AST  (Pass 2)
    # ------------------------------------------------------------------
    selected_tir_name: str = ""
    selected_tir_source: str = ""
    tir_ast_summary: Optional[dict] = None

    # ------------------------------------------------------------------
    # Stage 7 — Tensor Expression Microscope  (Pass 2)
    # ------------------------------------------------------------------
    te_compute_source: str = ""
    te_lowered_tir: str = ""

    # ------------------------------------------------------------------
    # Stage 8 — Schedule Space / Task Extraction  (Pass 3)
    # ------------------------------------------------------------------
    tuning_tasks: List[dict] = field(default_factory=list)
    _tasks_raw: Any = None                    # raw ExtractedTask objects
    _tuning_target: Any = None                # tvm.target.Target

    # ------------------------------------------------------------------
    # Stage 9 — Candidate Schedules / Tuning  (Pass 3)
    # ------------------------------------------------------------------
    tuning_records: List[dict] = field(default_factory=list)
    convergence_data: List[dict] = field(default_factory=list)
    tuning_work_dir: Optional[str] = None
    tuning_trials_used: int = 0
    tuning_tasks_total: int = 0
    tuning_tasks_covered: int = 0
    tuning_task_names_covered: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Stage 10 — Cost Model & Selection  (Pass 3)
    # ------------------------------------------------------------------
    candidate_features: Any = None            # pd.DataFrame
    best_candidate: Optional[dict] = None

    # ------------------------------------------------------------------
    # Stage 11 — Build CUDA Module
    # ------------------------------------------------------------------
    compiled_lib: Any = None                  # tvm.runtime.Module
    target_str: str = ""
    cuda_source: str = ""
    final_ir: str = ""

    # ------------------------------------------------------------------
    # Stage 12 — TVM Inference & Comparison
    # ------------------------------------------------------------------
    tvm_logits: Optional[np.ndarray] = None
    tvm_top5: Optional[List[dict]] = None
    tvm_latency_ms: float = 0.0
    max_abs_diff: float = 0.0
    cosine_sim: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def mark(self, stage_id: str, status: StageStatus, log: str = "") -> None:
        self.stage_status[stage_id] = status
        if log:
            prev = self.stage_logs.get(stage_id, "")
            self.stage_logs[stage_id] = (prev + "\n" + log).strip()

    def is_done(self, stage_id: str) -> bool:
        return self.stage_status.get(stage_id) == StageStatus.DONE

    def reset(self) -> None:
        """Reset all stages back to PENDING (keeps model_name)."""
        name = self.model_name
        self.__init__()  # type: ignore[misc]
        self.model_name = name
