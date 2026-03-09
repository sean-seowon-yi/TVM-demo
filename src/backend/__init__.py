"""TVM demo backend — pipeline functions, state management, and helpers.

Submodules
----------
state : DemoState dataclass
helpers : environment checks, latency measurement, numeric utilities
pipeline : one function per demo stage (Stages 0–12)

Import from submodules directly to avoid pulling in heavy dependencies
(torch, tvm) at package-import time::

    from backend.state import DemoState
    from backend.pipeline import load_model, import_to_tvm
"""
