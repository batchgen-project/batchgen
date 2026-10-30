import ast
import copy
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


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
    namespace = {}
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
        _trajectory_book=None,
        parallel_manager=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="requires the unified trajectory pool"):
        enabled(worker)

    worker._trajectory_book = object()
    with pytest.raises(RuntimeError, match="is not supported"):
        enabled(worker)

    worker.parallel_manager.activate_decoding = lambda: None
    assert enabled(worker) is True
    assert worker.parallel_manager.persistent_phase_instances is True


def test_persistent_decode_instance_uses_runtime_rank_cap():
    activate = _isolated_worker_method("_activate_persistent_decode_instance")
    activate.__globals__["time"] = SimpleNamespace(perf_counter=lambda: 0.0)
    activate.__globals__["logging"] = SimpleNamespace(info=lambda *args: None)

    configured = []
    manager = SimpleNamespace(
        decode_instance=None,
        decode_experts_resident=False,
    )

    def configure_decoding(*, padding_bsz, comm):
        configured.append((padding_bsz, comm))
        return object(), {}

    manager.configure_decoding = configure_decoding
    trace = []
    worker = SimpleNamespace(
        rank=1,
        parallel_manager=manager,
        _weight_passes=SimpleNamespace(close=lambda: trace.append("close")),
        core_engine=SimpleNamespace(
            stop_h2d_worker=lambda: trace.append("stop"),
            clear_kv_copy_queue=lambda: trace.append("clear_kv"),
            reset_decoding_buffer=lambda: trace.append("reset_decoding"),
        ),
        init_nvshmem=lambda: None,
        _max_decode_rank_bsz=lambda: 128,
        _install_decode_weight_copy_queue=lambda: trace.append("install"),
        set_phase=lambda phase: trace.append(("set_phase", phase)),
    )

    activate(worker, "comm")

    assert configured == [(128, "comm")]
    assert worker._decode_padding_bsz == 128
    # No weight pass is open across the switch, and the decode schedule
    # (empty here) replaces the prefill lists before decode runs.
    assert trace == [
        "close",
        "stop",
        "clear_kv",
        "reset_decoding",
        "install",
        ("set_phase", "decode"),
    ]


def test_persistent_startup_capture_follows_cuda_graph_enablement():
    build = _isolated_worker_method("_build_persistent_phase_instances")
    build.__globals__["time"] = SimpleNamespace(perf_counter=lambda: 0.0)
    build.__globals__["logging"] = SimpleNamespace(info=lambda *args: None)

    calls = []
    manager = SimpleNamespace(
        decode_instance=None,
        set_comm=lambda comm: calls.append(("set_comm", comm)),
        share_skeleton_on_device=lambda: calls.append("share_skeleton"),
        configure_prefill=lambda: calls.append("configure_prefill"),
    )
    worker = SimpleNamespace(
        rank=0,
        comm="comm",
        parallel_manager=manager,
        _activate_persistent_decode_instance=lambda comm: calls.append(
            ("activate_decode", comm)
        ),
        _init_gpu_kv_with_actual_size=lambda: calls.append("init_gpu_kv"),
        _glm5_whole_model_graph_requested_for_current_batch=lambda: True,
        _sync_decode_moe_rank_counts=lambda rows, reason: calls.append(
            ("sync_counts", rows, reason)
        ),
        _bind_decode_attention_metadata_for_graph_config=lambda rows: calls.append(
            ("bind_metadata", rows)
        ),
        _warmup_cuda_graphs=lambda: calls.append("capture_graphs"),
        _release_persistent_decode_instance=lambda: calls.append("release_decode"),
    )

    build(worker)

    assert "capture_graphs" in calls
    assert "release_decode" in calls


def test_worker_only_calls_worker_methods_that_exist():
    worker = _worker_class_ast()
    methods = {
        node.name: node
        for node in worker.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    missing = set()
    for name, method in methods.items():
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
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


def _isolated_prefill_nsys_methods(calls, seam_debug=None, world_size=8):
    """``begin``/``end`` with torch, dist, logging and the wrapper seam faked.

    ``seam_debug`` stands in for ``AttnWrapperBase.batchgen_debug``, which is
    where ``_config_prefill_for_batch`` leaves the per-batch debug dict.
    """
    begin = _isolated_worker_method("_nsys_prefill_profile_begin")
    end = _isolated_worker_method("_nsys_prefill_profile_end")

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda device: calls.append(("synchronize",)),
            nvtx=SimpleNamespace(
                range_push=lambda name: calls.append(("nvtx_push", name)),
                range_pop=lambda: calls.append(("nvtx_pop",)),
            ),
            cudart=lambda: SimpleNamespace(
                cudaProfilerStart=lambda: calls.append(("profiler_start",)),
                cudaProfilerStop=lambda: calls.append(("profiler_stop",)),
            ),
        )
    )
    fake_dist = SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: True,
        get_world_size=lambda: world_size,
        barrier=lambda: calls.append(("barrier",)),
    )
    for method in (begin, end):
        method.__globals__.update(
            torch=fake_torch,
            dist=fake_dist,
            logging=SimpleNamespace(info=lambda *args, **kwargs: None),
            AttnWrapperBase=SimpleNamespace(batchgen_debug=seam_debug),
        )
    return begin, end


def _prefill_nsys_worker(rank, debug=None):
    parse = _isolated_worker_method("_nsys_prefill_profile_controller_ranks")
    worker = SimpleNamespace(
        rank=rank,
        torch_device=f"cuda:{rank}",
        _batchgen_debug=debug,
        _debug_flag_enabled=lambda value: value is True,
        _nsys_prefill_profile_controller=False,
    )
    worker._nsys_prefill_profile_controller_ranks = lambda value: parse(worker, value)
    return worker


def test_prefill_nsys_profile_is_inert_without_batch_debug():
    calls = []
    begin, end = _isolated_prefill_nsys_methods(calls, seam_debug=None)
    worker = _prefill_nsys_worker(0, debug={})

    active = begin(worker, local_bsz=42, global_bsz=336)
    end(worker, active)

    assert active is False
    assert calls == []


def test_prefill_nsys_profile_reads_the_batch_debug_from_the_wrapper_seam():
    """``self._batchgen_debug`` is never populated per batch.

    ``_config_prefill_for_batch`` leaves the batch's dict on
    ``AttnWrapperBase.batchgen_debug`` immediately before the hook runs, so
    reading only ``self._batchgen_debug`` makes the capture unreachable.
    """
    seam = {
        "glm5_prefill_nsys_profile": True,
        "glm5_prefill_nsys_profile_ranks": [6],
    }
    calls = []
    begin, end = _isolated_prefill_nsys_methods(calls, seam_debug=seam)
    worker = _prefill_nsys_worker(6, debug=None)

    active = begin(worker, local_bsz=42, global_bsz=336)
    end(worker, active)

    assert active is True
    assert worker._nsys_prefill_profile_controller is True
    assert ("profiler_start",) in calls


def test_prefill_nsys_profile_rejects_ranks_outside_the_world():
    """A rank no process owns would capture nothing; fail loudly instead."""
    seam = {
        "glm5_prefill_nsys_profile": True,
        "glm5_prefill_nsys_profile_ranks": [99],
    }
    calls = []
    begin, _ = _isolated_prefill_nsys_methods(calls, seam_debug=seam, world_size=8)
    worker = _prefill_nsys_worker(0)

    with pytest.raises(ValueError, match="outside world size 8"):
        begin(worker, local_bsz=42, global_bsz=336)
    # It must raise before any collective, so the ranks cannot desync.
    assert ("barrier",) not in calls


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, {0}),
        (6, {6}),
        ([6], {6}),
        ((0, 6), {0, 6}),
        ([6, 6], {6}),
    ],
)
def test_prefill_nsys_profile_ranks_parses_valid_entries(value, expected):
    parse = _isolated_worker_method("_nsys_prefill_profile_controller_ranks")
    assert parse(SimpleNamespace(), value) == expected


@pytest.mark.parametrize(
    "value",
    [[], (), "6", "0,6", [-1], [True], [1.5], {"rank": 6}, 0.0, True],
)
def test_prefill_nsys_profile_ranks_rejects_malformed_entries(value):
    parse = _isolated_worker_method("_nsys_prefill_profile_controller_ranks")
    with pytest.raises(ValueError, match="glm5_prefill_nsys_profile_ranks"):
        parse(SimpleNamespace(), value)


@pytest.mark.parametrize(
    "ranks_entry, expected_controllers",
    [
        (None, {0}),
        ([6], {6}),
        ([0, 6], {0, 6}),
    ],
)
def test_prefill_nsys_profile_gates_only_the_profiler_on_controller_ranks(
    ranks_entry, expected_controllers
):
    debug = {"glm5_prefill_nsys_profile": True}
    if ranks_entry is not None:
        debug["glm5_prefill_nsys_profile_ranks"] = ranks_entry

    for rank in (0, 6):
        calls = []
        begin, end = _isolated_prefill_nsys_methods(calls, seam_debug=debug)
        worker = _prefill_nsys_worker(rank)

        active = begin(worker, local_bsz=42, global_bsz=336)
        end(worker, active)

        assert active is True
        # The capture window stays collective on EVERY rank: one NVTX range
        # and four barriers (two in begin, two in end). Gating these would
        # desync the ranks at the barrier. The exact sequence is asserted so a
        # regression cannot move the profiler call outside the barrier pair.
        gated = (
            [("profiler_start",)] if rank in expected_controllers else []
        )
        ungated = (
            [("profiler_stop",)] if rank in expected_controllers else []
        )
        assert calls == (
            [("synchronize",), ("barrier",)]
            + gated
            + [
                ("barrier",),
                (
                    "nvtx_push",
                    f"BatchGen_prefill_rank_{rank}_local_bsz_42_global_bsz_336",
                ),
                ("nvtx_pop",),
                ("synchronize",),
                ("barrier",),
            ]
            + ungated
            + [("barrier",)]
        )


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
