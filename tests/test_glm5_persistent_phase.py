import ast
import copy
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from tests.worker.worker_fakes import attach_trajectory_state

import pytest
import torch

from batchgen.phase_lifecycle import PhaseKey


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"


def _isolated_worker_method(name):
    tree = ast.parse(WORKER.read_text())
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    method = copy.deepcopy(
        next(
            node
            for node in worker.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Worker",
                bases=[],
                keywords=[],
                body=[method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    from batchgen.phase_lifecycle import PhaseKey

    # Names the extracted method resolves at call time.
    namespace = {"PhaseKey": PhaseKey}
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return getattr(namespace["Worker"], name)


def _worker_class_ast():
    tree = ast.parse(WORKER.read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )


def _load_server_args_module():
    package_name = "batchgen.server"
    previous = sys.modules.get(package_name)
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    sys.modules[package_name] = package
    try:
        return importlib.import_module("batchgen.server.server_args")
    finally:
        if previous is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = previous


def _fake_cuda_driver(calls):
    class Location:
        pass

    class AllocationProp:
        def __init__(self):
            self.location = Location()

    class AccessDesc:
        def __init__(self):
            self.location = Location()

    driver = types.ModuleType("cuda.bindings.driver")
    driver.CUresult = SimpleNamespace(CUDA_SUCCESS=0)
    driver.CUmemAllocationProp = AllocationProp
    driver.CUmemAccessDesc = AccessDesc
    driver.CUmemAllocationType = SimpleNamespace(
        CU_MEM_ALLOCATION_TYPE_PINNED=1
    )
    driver.CUmemLocationType = SimpleNamespace(
        CU_MEM_LOCATION_TYPE_DEVICE=1
    )
    driver.CUmemAllocationGranularity_flags = SimpleNamespace(
        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED=1
    )
    driver.CUmemAccess_flags = SimpleNamespace(
        CU_MEM_ACCESS_FLAGS_PROT_READWRITE=1
    )
    driver.cuMemGetAllocationGranularity = lambda prop, flag: (0, 64)
    driver.cuMemAddressReserve = lambda size, align, addr, flags: (
        calls.append(("reserve", size, align)) or (0, 0x100000)
    )

    next_handle = iter((101, 102, 103))

    def create(size, prop, flags):
        handle = next(next_handle)
        calls.append(("create", size, handle))
        return 0, handle

    driver.cuMemCreate = create
    driver.cuMemMap = lambda ptr, size, offset, handle, flags: (
        calls.append(("map", ptr, size, handle)) or (0,)
    )
    driver.cuMemSetAccess = lambda ptr, size, access, count: (
        calls.append(("access", ptr, size, count)) or (0,)
    )
    driver.cuMemUnmap = lambda ptr, size: (
        calls.append(("unmap", ptr, size)) or (0,)
    )
    driver.cuMemRelease = lambda handle: (
        calls.append(("release", handle)) or (0,)
    )
    return driver


def _load_vmm_arena(monkeypatch, driver):
    cuda = types.ModuleType("cuda")
    bindings = types.ModuleType("cuda.bindings")
    bindings.driver = driver
    cuda.bindings = bindings
    monkeypatch.setitem(sys.modules, "cuda", cuda)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.driver", driver)

    path = ROOT / "batchgen" / "cuda_graph" / "vmm_arena.py"
    spec = importlib.util.spec_from_file_location("_vmm_arena_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.VmmArena


def test_persistent_phase_cli_is_opt_in_without_context_cap():
    server_args = _load_server_args_module()
    base = [
        "--model",
        "zai-org/GLM-5.2-FP8",
        "--host-kv-cache-size",
        "64",
    ]

    default = server_args.prepare_server_args(base)
    enabled = server_args.prepare_server_args([*base, "--persistent-phase-instances"])

    assert default.persistent_phase_instances is False
    assert enabled.persistent_phase_instances is True

    with pytest.raises(SystemExit):
        server_args.prepare_server_args(
            [*base, "--cuda-graph-max-seqlen", "8192"]
        )


def test_persistent_phase_worker_gate_preserves_flag_off_behavior():
    enabled = _isolated_worker_method("_persistent_phase_enabled")
    manager = SimpleNamespace()
    worker = SimpleNamespace(
        args=SimpleNamespace(persistent_phase_instances=False),
        _max_pool_size=0,
        parallel_manager=manager,
    )

    assert enabled(worker) is False
    assert not hasattr(manager, "persistent_phase_instances")


def test_persistent_phase_worker_gate_rejects_invalid_flag_on_modes():
    enabled = _isolated_worker_method("_persistent_phase_enabled")
    worker = SimpleNamespace(
        args=SimpleNamespace(persistent_phase_instances=True),
        parallel_manager=SimpleNamespace(),
    )
    attach_trajectory_state(worker)

    with pytest.raises(RuntimeError, match="requires the unified trajectory pool"):
        enabled(worker)

    worker._traj.trajectory_book = object()
    with pytest.raises(RuntimeError, match="is not supported"):
        enabled(worker)

    worker.parallel_manager.activate_decoding = lambda: None
    assert enabled(worker) is True
    assert worker.parallel_manager.persistent_phase_instances is True



@pytest.mark.parametrize("graph_requested, compiles_kernels", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_persistent_startup_capture_follows_cuda_graph_enablement(
    graph_requested, compiles_kernels,
):
    build = _isolated_worker_method("_build_persistent_phase_instances")
    build.__globals__["time"] = SimpleNamespace(perf_counter=lambda: 0.0)
    build.__globals__["logging"] = SimpleNamespace(info=lambda *args: None)
    build.__globals__["_startup_transient"] = SimpleNamespace(
        SEAM_PERSISTENT_BUILD="persistent_build"
    )

    calls = []
    manager = SimpleNamespace(
        decode_instance=None,
        set_comm=lambda comm: calls.append(("set_comm", comm)),
        share_skeleton_on_device=lambda: calls.append("share_skeleton"),
    )
    capture_fn = object()
    prewarm_fn = object()
    worker = SimpleNamespace(
        rank=0,
        comm="comm",
        parallel_manager=manager,
        _transition_phase=lambda key, **kw: (
            calls.append(("transition", key.phase, kw)) or (True, False)
        ),
        set_phase=lambda phase: calls.append(("set_phase", phase)),
        _install_prefill_weight_copy_pipeline=lambda profile: calls.append(
            ("install_prefill", profile)
        ),
        _desired_phase_key=lambda phase: PhaseKey(phase, None, ()),
        # GLM-5 declares no prefill MoE modes: the server mode is None.
        _server_prefill_moe_mode=lambda: None,
        _max_decode_rank_bsz=lambda: 128,
        _check_decode_width_within_graph_buckets=lambda width: calls.append(
            ("check_width", width)
        ),
        _assert_phase_switch_clean=lambda label, reused: None,
        _install_decode_weight_copy_queue=lambda: calls.append("install_decode"),
        _init_gpu_kv_with_actual_size=lambda: calls.append("init_gpu_kv"),
        _glm5_whole_model_graph_requested_for_current_batch=(
            lambda: graph_requested
        ),
        _probe_startup_transient=lambda *, seam, capture: calls.append(
            ("probe", seam, capture)
        ),
        _capture_startup_decode_graphs=capture_fn,
        _manager_decode_prewarm_compiles_kernels=lambda: compiles_kernels,
        _prewarm_startup_decode_jit=prewarm_fn,
    )

    build(worker)

    # The decode width is checked and plumbed through the transition.
    assert ("check_width", 128) in calls
    assert any(
        entry[0] == "transition"
        and entry[1] == "decode"
        and entry[2].get("max_num_seq") == 128
        for entry in calls
    )
    # The capture callable reaches the probe only under graph enablement;
    # without graphs, a manager whose decode prewarm compiles kernels still
    # gets that prewarm before readiness.
    probe = next(entry for entry in calls if entry[0] == "probe")
    assert probe[1] == "persistent_build"
    expected = capture_fn if graph_requested else (prewarm_fn if compiles_kernels else None)
    assert probe[2] is expected
    # The build ends back in prefill.
    assert calls[-1][0] == "install_prefill"


def test_worker_only_calls_worker_methods_that_exist():
    worker = _worker_class_ast()
    methods = {
        node.name: node
        for node in worker.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    def _own_calls(method):
        """Calls whose ``self`` is the worker: skip nested classes, whose
        ``self`` is the nested instance (e.g. the comm thread's planner)."""
        stack = list(ast.iter_child_nodes(method))
        while stack:
            node = stack.pop()
            if isinstance(node, ast.ClassDef):
                continue
            stack.extend(ast.iter_child_nodes(node))
            if isinstance(node, ast.Call):
                yield node

    missing = set()
    for name, method in methods.items():
        for node in _own_calls(method):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
                and func.attr.startswith("_")
                and func.attr not in methods
            ):
                missing.add((name, func.attr))

    assert not missing


def test_prefill_nsys_profile_is_inert_under_an_empty_debug_state():
    begin = _isolated_worker_method("_nsys_prefill_profile_begin")
    end = _isolated_worker_method("_nsys_prefill_profile_end")
    worker = SimpleNamespace(
        _debug_flag_enabled=lambda value: False,
        _debug_values=lambda: None,
        # D2: the delegator freezes the context before the engine's gate; an
        # inert window must not touch the device either way.
        _decode_phase_context=lambda: None,
    )

    active = begin(worker, local_bsz=42, global_bsz=336)
    end(worker, active)

    assert active is False


def test_prefill_nsys_profile_reads_the_server_debug_state():
    # The capture window opens from the server's debug state as the worker
    # adopted it, the same for a pool-served wave and any other. D2: the
    # delegator resolves the flag and passes the frozen context — the engine
    # never receives the worker.
    begin = _isolated_worker_method("_nsys_prefill_profile_begin")
    seen = {}

    class _Engine:
        def _nsys_prefill_profile_begin(self, ctx, **kwargs):
            seen["ctx"] = ctx
            seen.update(kwargs)
            return True

    worker = SimpleNamespace(
        _step_observability_engine=_Engine(),
        _debug_values=lambda: {"glm5_prefill_nsys_profile": True},
        _debug_flag_enabled=lambda value: bool(value),
        _decode_phase_context=lambda: "frozen-ctx",
    )

    assert begin(worker, local_bsz=3, global_bsz=24) is True
    assert seen == {
        "ctx": "frozen-ctx",
        "local_bsz": 3,
        "global_bsz": 24,
        "profile_enabled": True,
    }


def test_prefill_nsys_window_wraps_the_only_prefill_dispatch():
    # Both prefill forwards (per-batch and the in-wave pool, which
    # prefill_prepacked dispatches to) are entered from one place; the window
    # must bracket it, and no second caller may bypass the bracket.
    source = (ROOT / "batchgen" / "batchgen_worker.py").read_text()
    begin = source.index("self._nsys_prefill_profile_begin(")
    dispatch = source.index("self.prefill_prepacked(local_prefill_indices)")
    end = source.index("self._nsys_prefill_profile_end(_prefill_profile_active)")
    assert begin < dispatch < end
    assert source.count("self.prefill_prepacked(") == 1
    assert source.count("self.prefill(local_prefill_indices)") == 1
    assert begin < source.index("self.prefill(local_prefill_indices)") < end


def test_cuda_python_runtime_dependency_is_declared():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    assert "cuda-python==13.3.1" in requirements


def test_vmm_arena_maps_and_unmaps_the_reserved_address(monkeypatch):
    calls = []
    VmmArena = _load_vmm_arena(monkeypatch, _fake_cuda_driver(calls))
    arena = VmmArena(torch.device("cuda:0"), nbytes=100, chunk_bytes=64)

    assert arena.nbytes == 128
    assert arena.is_mapped is False

    arena.map()
    assert arena.is_mapped is True
    assert [call[0] for call in calls].count("map") == 2
    assert ("access", 0x100000, 128, 1) in calls
    with pytest.raises(RuntimeError, match="already mapped"):
        arena.map()

    arena.unmap()
    assert arena.is_mapped is False
    assert ("unmap", 0x100000, 128) in calls
    assert ("release", 101) in calls
    assert ("release", 102) in calls
    with pytest.raises(RuntimeError, match="not mapped"):
        arena.unmap()
