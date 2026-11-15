"""Slice 2a: the decode JIT prewarm runs at the startup graph capture and the
late-compile guard arms at readiness.

Covers, through the real sources (ast-lifted, never re-implemented):

- ``_capture_startup_decode_graphs`` runs ``_prewarm_decode_jit`` BEFORE
  ``_warmup_cuda_graphs`` (the KV pool is sized and the decode instance live,
  so geometry-keyed kernels compile before readiness);
- ``_prewarm_decode_jit``: manager without the hook is a no-op; the hook gets
  the worker's GPU KV manager, plus the graph bucket ladder exactly when the
  deployment requests whole-model decode graphs; a prewarm failure propagates;
- ``arm_late_compile_guard_at_readiness`` delegates to the manager hook and is
  a no-op without one;
- the main loop arms the guard before signalling readiness;
- GLM-5 PSM: ``prewarm_decode_graphs`` adopts the handed-down bucket ladder
  (nothing else sets it) and ``arm_late_compile_guard`` arms via
  ``triton_prewarm.arm_at_readiness`` only when CUDA is up.
"""

import ast
import pathlib
from types import SimpleNamespace

import pytest

from tests.worker_method_source import load_worker_methods

REPO = pathlib.Path(__file__).resolve().parents[2]
GLM5_PSM = REPO / "batchgen" / "models" / "glm" / "glm5" / "Parallel_Strategy_Manager.py"
MAIN_LOOP = REPO / "batchgen" / "server_worker_main_loop.py"


def load_class_methods(path, class_name, names, namespace):
    """Lift methods out of a class body in ``path`` (worker_method_source's
    technique for modules that cannot import on a CPU-only machine)."""
    wanted = set(names)
    tree = ast.parse(path.read_text(), filename=str(path))
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in wanted:
                    module = ast.Module(body=[item], type_ignores=[])
                    code = compile(module, filename=str(path), mode="exec")
                    scope = dict(namespace)
                    exec(code, scope)
                    out[item.name] = scope[item.name]
    missing = wanted - out.keys()
    assert not missing, f"methods not found in {class_name}: {missing}"
    return out


# ---------------------------------------------------------------------------
# Worker seam
# ---------------------------------------------------------------------------


def _worker_methods():
    return load_worker_methods(
        ["_capture_startup_decode_graphs", "_prewarm_decode_jit",
         "arm_late_compile_guard_at_readiness"],
        {},
    )


def _capture_worker(methods, events, manager):
    worker = SimpleNamespace(
        rank=0,
        parallel_manager=manager,
        _decode_graph_captures=0,
        _sync_decode_moe_rank_counts=lambda *a, **k: events.append("sync"),
        _bind_decode_attention_metadata_for_graph_config=(
            lambda _: events.append("bind")),
        _get_cuda_graph_gpu_manager=lambda: "gpu-manager",
        _whole_model_decode_graphs_requested=lambda: True,
        _decode_graph_bucket_sizes=lambda: [1, 2, 4],
        _decode_graph_state_present=lambda: True,
    )

    def warmed():
        events.append("capture")
        worker._decode_graph_captures += 1

    worker._warmup_cuda_graphs = warmed
    worker._prewarm_decode_jit = (
        lambda: methods["_prewarm_decode_jit"](worker))
    return worker


def test_startup_capture_prewarms_before_graph_capture():
    events = []
    methods = _worker_methods()
    manager = SimpleNamespace(
        prewarm_decode_graphs=lambda gpu, decode_buckets: events.append(
            ("prewarm", gpu, decode_buckets)),
    )
    worker = _capture_worker(methods, events, manager)
    methods["_capture_startup_decode_graphs"](worker)
    assert events == [
        "sync", "bind", ("prewarm", "gpu-manager", [1, 2, 4]), "capture",
    ]


def test_startup_capture_without_manager_hook_still_captures():
    events = []
    methods = _worker_methods()
    worker = _capture_worker(methods, events, SimpleNamespace())
    methods["_capture_startup_decode_graphs"](worker)
    assert events == ["sync", "bind", "capture"]


def test_startup_prewarm_failure_fails_startup():
    events = []
    methods = _worker_methods()

    def boom(gpu, decode_buckets):
        raise RuntimeError("geometry prewarm failed")

    worker = _capture_worker(
        methods, events, SimpleNamespace(prewarm_decode_graphs=boom))
    with pytest.raises(RuntimeError, match="geometry prewarm failed"):
        methods["_capture_startup_decode_graphs"](worker)
    assert "capture" not in events


def test_prewarm_jit_passes_buckets_only_for_whole_model_graphs():
    methods = _worker_methods()
    calls = []
    worker = SimpleNamespace(
        parallel_manager=SimpleNamespace(
            prewarm_decode_graphs=lambda gpu, decode_buckets: calls.append(
                (gpu, decode_buckets))),
        _get_cuda_graph_gpu_manager=lambda: "gpu-manager",
        _whole_model_decode_graphs_requested=lambda: False,
        _decode_graph_bucket_sizes=lambda: [1, 2],
    )
    methods["_prewarm_decode_jit"](worker)
    assert calls == [("gpu-manager", None)]


def test_arm_at_readiness_delegates_to_manager_hook():
    methods = _worker_methods()
    events = []
    worker = SimpleNamespace(parallel_manager=SimpleNamespace(
        arm_late_compile_guard=lambda: events.append("armed")))
    methods["arm_late_compile_guard_at_readiness"](worker)
    assert events == ["armed"]

    bare = SimpleNamespace(parallel_manager=SimpleNamespace())
    methods["arm_late_compile_guard_at_readiness"](bare)  # no hook: no-op


# ---------------------------------------------------------------------------
# Main loop ordering
# ---------------------------------------------------------------------------


def test_main_loop_arms_guard_before_ready_signal():
    """The guard must arm after the global barrier and before the ready
    signal; later and a compile in the gap is invisible, earlier and the
    barrier wait is reported as late."""
    source = MAIN_LOOP.read_text()
    arm = source.index("worker.arm_late_compile_guard_at_readiness()")
    ready = source.index("_signal_local_worker_manager_ready(", source.index(
        "def "))  # first call site inside the function body
    barrier = source.index("dist.barrier()")
    assert barrier < arm < ready


# ---------------------------------------------------------------------------
# GLM-5 PSM seam (module cannot import on CPU: triton)
# ---------------------------------------------------------------------------

PHASE_GEOMETRY = "geometry"


def _glm5_methods(cuda_available, arm_calls):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: cuda_available))

    class FakeTritonPrewarm(SimpleNamespace):
        pass

    return load_class_methods(
        GLM5_PSM, "GLM5ParallelStrategyManager",
        ["prewarm_decode_graphs", "arm_late_compile_guard"],
        {
            "torch": fake_torch,
            "PHASE_GEOMETRY": PHASE_GEOMETRY,
            "triton_prewarm": FakeTritonPrewarm(
                arm_at_readiness=lambda rank: arm_calls.append(rank)),
        },
    )


def test_glm5_prewarm_adopts_the_handed_down_bucket_ladder():
    runs = []
    methods = _glm5_methods(cuda_available=True, arm_calls=[])
    psm = SimpleNamespace(
        decode_graph_buckets=None,
        _prewarm_triton_kernels=lambda phase, gpu_manager=None: runs.append(
            (phase, gpu_manager)) or 0.0,
    )
    methods["prewarm_decode_graphs"](psm, "gpu-manager", decode_buckets=(1, 2, 8))
    assert psm.decode_graph_buckets == [1, 2, 8]
    assert runs == [(PHASE_GEOMETRY, "gpu-manager")]

    # No manager: nothing runs, the recorded ladder is untouched.
    methods["prewarm_decode_graphs"](psm, None, decode_buckets=(4,))
    assert psm.decode_graph_buckets == [1, 2, 8]
    assert runs == [(PHASE_GEOMETRY, "gpu-manager")]

    # No ladder handed down (eager deployment): ladder stays, phase runs.
    methods["prewarm_decode_graphs"](psm, "gpu-manager")
    assert psm.decode_graph_buckets == [1, 2, 8]
    assert len(runs) == 2


def test_glm5_arm_late_compile_guard_requires_cuda():
    arm_calls = []
    methods = _glm5_methods(cuda_available=False, arm_calls=arm_calls)
    methods["arm_late_compile_guard"](SimpleNamespace(global_rank=3))
    assert arm_calls == []


def test_glm5_arm_late_compile_guard_arms_with_rank(monkeypatch):
    arm_calls = []
    methods = _glm5_methods(cuda_available=True, arm_calls=arm_calls)
    import sys
    import types

    pkg = "batchgen.models.glm.glm5"
    fake = types.ModuleType(pkg + ".triton_prewarm")
    fake.arm_at_readiness = lambda rank: arm_calls.append(rank)
    monkeypatch.setitem(sys.modules, pkg + ".triton_prewarm", fake)
    # The method does ``from .triton_prewarm import arm_at_readiness`` with
    # the PSM module's package; the lifted function resolves the relative
    # import against this namespace.
    methods = load_class_methods(
        GLM5_PSM, "GLM5ParallelStrategyManager", ["arm_late_compile_guard"],
        {
            "torch": SimpleNamespace(cuda=SimpleNamespace(
                is_available=lambda: True)),
            "__package__": pkg,
            "__name__": pkg + ".Parallel_Strategy_Manager",
        },
    )
    methods["arm_late_compile_guard"](SimpleNamespace(global_rank=5))
    assert arm_calls == [5]
