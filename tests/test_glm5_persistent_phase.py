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
        _max_pool_size=0,
        parallel_manager=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="requires pool mode"):
        enabled(worker)

    worker._max_pool_size = 1
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
    worker = SimpleNamespace(
        rank=1,
        parallel_manager=manager,
        core_engine=SimpleNamespace(
            stop_h2d_worker=lambda: None,
            clear_kv_copy_queue=lambda: None,
            clear_weight_copy_queue=lambda: None,
            reset_decoding_buffer=lambda: None,
        ),
        init_nvshmem=lambda: None,
        _max_decode_rank_bsz=lambda: 128,
        set_phase=lambda phase: None,
    )

    activate(worker, "comm")

    assert configured == [(128, "comm")]
    assert worker._decode_padding_bsz == 128


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


def test_prefill_nsys_profile_is_inert_without_batch_debug():
    begin = _isolated_worker_method("_nsys_prefill_profile_begin")
    end = _isolated_worker_method("_nsys_prefill_profile_end")
    worker = SimpleNamespace(
        _batchgen_debug={},
        _debug_flag_enabled=lambda value: False,
    )

    active = begin(worker, local_bsz=42, global_bsz=336)
    end(worker, active)

    assert active is False


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
