"""Runtime topology and communication ranges must remain lane-scoped."""

from __future__ import annotations

import ast
import copy
import json
import os
import pickle
import runpy
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Optional
import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
WORKER_MANAGER = ROOT / "batchgen" / "server" / "worker_manager.py"
HTTP_SERVER = ROOT / "batchgen" / "server" / "http_server.py"
PROCESS_UTILS = ROOT / "batchgen" / "server" / "process_utils.py"
GPT_OSS_PARAMETER_SERVER = (
    ROOT
    / "batchgen"
    / "models"
    / "openai"
    / "gpt_oss_120b"
    / "gpt_oss_parameter_server.py"
)
GPT_OSS_PS_MODULE = "batchgen.models.openai.gpt_oss_120b.gpt_oss_parameter_server"
MIXTRAL_PS_MODULE = "batchgen.models.mixtral.mixtral_parameter_server"


def _top_level_function(path: Path, name: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


def _worker_method(name: str, globals_=None):
    tree = ast.parse(WORKER.read_text(), filename=str(WORKER))
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
                name="IsolatedWorker",
                bases=[],
                keywords=[],
                body=[method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = dict(globals_ or {})
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return namespace["IsolatedWorker"]


def _worker_manager_method(name: str, globals_=None):
    tree = ast.parse(WORKER_MANAGER.read_text(), filename=str(WORKER_MANAGER))
    manager = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkerManager"
    )
    method = copy.deepcopy(
        next(
            node
            for node in manager.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="IsolatedManager",
                bases=[],
                keywords=[],
                body=[method],
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = dict(globals_ or {})
    exec(
        compile(ast.fix_missing_locations(module), str(WORKER_MANAGER), "exec"),
        namespace,
    )
    return namespace["IsolatedManager"]


def _isolated_class(path: Path, class_name: str, method_names, globals_=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        copy.deepcopy(
            next(
                node
                for node in source_class.body
                if isinstance(node, ast.FunctionDef) and node.name == method_name
            )
        )
        for method_name in method_names
    ]
    module = ast.Module(
        body=[
            ast.ClassDef(
                name="Isolated",
                bases=[],
                keywords=[],
                body=methods,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )
    namespace = dict(globals_ or {})
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Isolated"]


def _process_utils_namespace():
    """Load process_utils standalone so the test needs no batchgen package import."""
    return runpy.run_path(str(PROCESS_UTILS))


def _gpt_oss_parameter_server(convert_hook, cpp_init_calls):
    fake_cpp = SimpleNamespace(
        Init=lambda *args: cpp_init_calls.append(args),
        get_skeleton_state_dict=lambda: {"model.norm.weight": 1},
    )
    server_type = _isolated_class(
        GPT_OSS_PARAMETER_SERVER,
        "GptOss_Parameter_Server",
        ("reserve_shm_names", "Init"),
        {
            "logging": SimpleNamespace(
                info=lambda *args, **kwargs: None,
                debug=lambda *args, **kwargs: None,
                warning=lambda *args, **kwargs: None,
                error=lambda *args, **kwargs: None,
            ),
            "os": os,
            "shutil": SimpleNamespace(
                disk_usage=lambda path: (0, 0, 200 * 1024**3)
            ),
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(mem_get_info=lambda: (1 << 40, 1 << 40))
            ),
            "uuid": uuid,
            "Parameter_Server": lambda *args: fake_cpp,
        },
    )
    server = server_type()
    server.shm_name = None
    server.tensor_meta_shm_name = None
    server.enable_hugetlbfs = False
    server.enable_memfd = False
    server.converted_ckpt_dir = "/tmp/converted"
    server.state_dict_name_map = {}
    server._parse_state_dict = lambda: None
    server._convert_checkpoint = convert_hook
    return server


def test_gpt_oss_reserve_shm_names_is_stable_and_used_by_init():
    converted_with = []
    cpp_init_calls = []
    server = _gpt_oss_parameter_server(
        lambda: converted_with.append(
            (server.shm_name, server.tensor_meta_shm_name)
        ),
        cpp_init_calls,
    )

    reserved = server.reserve_shm_names()
    assert server.reserve_shm_names() == reserved
    assert reserved[0].startswith("/shm_") and reserved[0] != reserved[1]

    assert server.Init() == reserved
    # Names must already be fixed before the long checkpoint conversion.
    assert converted_with == [reserved]
    assert cpp_init_calls[0][:2] == reserved


def test_gpt_oss_init_still_self_generates_without_reservation():
    cpp_init_calls = []
    server = _gpt_oss_parameter_server(lambda: None, cpp_init_calls)

    shm_name, tensor_meta_shm_name = server.Init()

    assert shm_name.startswith("/shm_")
    assert tensor_meta_shm_name.startswith("/shm_")
    assert shm_name != tensor_meta_shm_name
    assert cpp_init_calls[0][:2] == (shm_name, tensor_meta_shm_name)


def _local_load_manager(process_utils, runtime_dir, model, tmp_path):
    manager_type = _worker_manager_method(
        "_load_model_locally",
        {
            "record_model_shm_provenance": process_utils[
                "record_model_shm_provenance"
            ],
            "logger": SimpleNamespace(
                info=lambda *args, **kwargs: None,
                warning=lambda *args, **kwargs: None,
                error=lambda *args, **kwargs: None,
            ),
            "os": os,
            "Path": Path,
            "tempfile": tempfile,
            "torch": SimpleNamespace(save=lambda obj, path: None),
        },
    )
    manager = manager_type()
    manager.args = SimpleNamespace(
        model=model,
        cache_dir=tmp_path / "cache",
        enable_hugetlbfs=False,
        fast_init=False,
        runtime_identity=SimpleNamespace(runtime_dir=runtime_dir),
    )
    return manager


def test_local_gpt_oss_records_shm_names_before_init_failure(tmp_path, monkeypatch):
    process_utils = _process_utils_namespace()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    neighbor = shm_dir / "shm_neighbor"
    neighbor.touch()

    class FakeGptOssParameterServer:
        def __init__(self, *args, **kwargs):
            self.shm_name = None
            self.tensor_meta_shm_name = None

        def reserve_shm_names(self):
            self.shm_name = "/shm_reserved_weights"
            self.tensor_meta_shm_name = "/shm_reserved_meta"
            return self.shm_name, self.tensor_meta_shm_name

        def Init(self):
            (shm_dir / self.shm_name.lstrip("/")).touch()
            (shm_dir / self.tensor_meta_shm_name.lstrip("/")).touch()
            raise RuntimeError("weight load crashed")

    module = ModuleType(GPT_OSS_PS_MODULE)
    module.GptOss_Parameter_Server = FakeGptOssParameterServer
    monkeypatch.setitem(sys.modules, GPT_OSS_PS_MODULE, module)

    manager = _local_load_manager(
        process_utils, runtime_dir, "openai/gpt-oss-120b", tmp_path
    )

    with pytest.raises(RuntimeError, match="weight load crashed"):
        manager._load_model_locally(tmp_path / "hf", tmp_path / "converted")

    record = json.loads(
        (runtime_dir / process_utils["MODEL_SHM_PROVENANCE_FILE"]).read_text()
    )
    assert record["shm_names"] == ["shm_reserved_weights", "shm_reserved_meta"]
    assert manager.parameter_server_instance is not None
    assert manager.model_info == {
        "shm_name": "/shm_reserved_weights",
        "tensor_meta_shm_name": "/shm_reserved_meta",
    }

    stop_type = _worker_manager_method(
        "stop",
        {
            "cleanup_model_shm_files": lambda info: process_utils[
                "cleanup_model_shm_files"
            ](info, shm_dir=shm_dir),
            "logger": SimpleNamespace(
                info=lambda *args, **kwargs: None,
                error=lambda *args, **kwargs: None,
            ),
        },
    )
    manager._stopping = False
    manager.started = False
    manager._runtime_dir_created = True
    manager._runtime_namespace_owned = False
    manager._runtime_locks = None
    manager._lane_lease = None
    manager.worker_process = None
    manager.distributed_weight_daemon = None
    manager.skeleton_state_dict_file = None
    manager._monitor_stop_event = SimpleNamespace(set=lambda: None)
    manager._monitor_thread = None
    manager._cleanup_skeleton_state_dict_file = lambda: None
    stop_type.stop(manager)

    assert not (shm_dir / "shm_reserved_weights").exists()
    assert not (shm_dir / "shm_reserved_meta").exists()
    assert neighbor.exists()
    assert not runtime_dir.exists()


def test_local_gpt_oss_fails_closed_when_init_drifts_from_reservation(
    tmp_path, monkeypatch
):
    process_utils = _process_utils_namespace()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()

    class DriftingGptOssParameterServer:
        def __init__(self, *args, **kwargs):
            pass

        def reserve_shm_names(self):
            return "/shm_reserved_weights", "/shm_reserved_meta"

        def Init(self):
            return "/shm_other_weights", "/shm_reserved_meta"

    module = ModuleType(GPT_OSS_PS_MODULE)
    module.GptOss_Parameter_Server = DriftingGptOssParameterServer
    monkeypatch.setitem(sys.modules, GPT_OSS_PS_MODULE, module)

    manager = _local_load_manager(
        process_utils, runtime_dir, "openai/gpt-oss-120b", tmp_path
    )

    with pytest.raises(RuntimeError, match="differ from the recorded reservation"):
        manager._load_model_locally(tmp_path / "hf", tmp_path / "converted")

    record = json.loads(
        (runtime_dir / process_utils["MODEL_SHM_PROVENANCE_FILE"]).read_text()
    )
    assert record["shm_names"] == ["shm_reserved_weights", "shm_reserved_meta"]


def test_local_other_model_still_records_shm_names_after_init(tmp_path, monkeypatch):
    process_utils = _process_utils_namespace()
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    record_path = runtime_dir / process_utils["MODEL_SHM_PROVENANCE_FILE"]
    recorded_during_init = []

    class FakeMixtralParameterServer:
        def __init__(self, *args, **kwargs):
            self.parameter_server = SimpleNamespace(
                byte_size=lambda: 1234,
                get_skeleton_state_dict=lambda: {"model.norm.weight": 1},
            )

        def Init(self):
            recorded_during_init.append(record_path.exists())
            return "/shm_mixtral_weights", "/shm_mixtral_meta"

    module = ModuleType(MIXTRAL_PS_MODULE)
    module.Mixtral_Parameter_Server = FakeMixtralParameterServer
    monkeypatch.setitem(sys.modules, MIXTRAL_PS_MODULE, module)

    manager = _local_load_manager(
        process_utils, runtime_dir, "mistralai/Mixtral-8x7B-Instruct-v0.1", tmp_path
    )

    manager._load_model_locally(tmp_path / "hf", tmp_path / "converted")

    assert recorded_during_init == [False]
    assert manager.model_info["shm_name"] == "/shm_mixtral_weights"
    record = json.loads(record_path.read_text())
    assert record["shm_names"] == ["shm_mixtral_weights", "shm_mixtral_meta"]


def test_local_world_size_requires_exact_division_and_visibility():
    resolve = _top_level_function(WORKER_MANAGER, "_resolve_local_world_size")

    assert resolve(8, 1, 8) == 8
    assert resolve(8, 2, 8) == 4
    assert resolve(8, 4, 2, require_exact_visibility=True) == 2
    assert resolve(8, 8, 1, require_exact_visibility=True) == 1

    with pytest.raises(ValueError, match="divisible"):
        resolve(7, 2, 8)
    with pytest.raises(ValueError, match="only 1 GPUs are visible"):
        resolve(4, 1, 1)
    with pytest.raises(ValueError, match="exactly 2 visible GPUs"):
        resolve(4, 2, 8, require_exact_visibility=True)


def test_pynccl_search_never_leaves_assigned_range():
    calls = []

    def _find_available_port(host, start_port, max_attempts):
        calls.append((host, start_port, max_attempts))
        return start_port

    worker_type = _worker_method(
        "_find_available_pynccl_port",
        {"_find_available_port": _find_available_port},
    )
    worker = worker_type()
    worker.pynccl_port_base = 21000
    worker.pynccl_port_span = 4

    assert worker._find_available_pynccl_port("127.0.0.1", 20900) == 21000
    assert calls[-1] == ("127.0.0.1", 21000, 4)
    assert worker._find_available_pynccl_port("127.0.0.1", 21003) == 21003
    assert calls[-1] == ("127.0.0.1", 21003, 1)

    with pytest.raises(RuntimeError, match="range exhausted"):
        worker._find_available_pynccl_port("127.0.0.1", 21004)


def test_worker_rank_math_uses_configured_local_world_size():
    main_loop = (
        ROOT / "batchgen" / "server_worker_main_loop.py"
    ).read_text()
    worker_source = WORKER.read_text()
    migration_source = (ROOT / "batchgen" / "migration.py").read_text()

    assert "args.local_world_size * args.nnode_rank" in main_loop
    assert "NUM_GPUS_PER_NODE" not in worker_source
    assert "NUM_GPUS_PER_NODE" not in migration_source


def test_http_shutdown_has_no_host_global_cleanup_fallback():
    http_source = HTTP_SERVER.read_text()
    manager_source = WORKER_MANAGER.read_text()

    assert "shm_prefix=None" not in http_source
    assert "clean_hugepages=True" not in http_source
    assert "if self._runtime_namespace_owned:" in manager_source
    assert "shm_prefix=(" in manager_source
    assert "self.args.runtime_identity.shm_prefix" in manager_source
    assert "kill_workers=False" in manager_source
    assert http_source.index("worker._acquire_runtime_admission()") < (
        http_source.index("StorageManager(server_args.storage_path)")
    )


def test_runtime_namespace_preflight_does_not_claim_longer_instance_id(tmp_path):
    run_id = "a" * 32
    prefix = f"batchgen_lane_{run_id}"
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()
    neighbor = shm_dir / f"{prefix}_b_{'b' * 32}_host_kv"
    neighbor.touch()
    runtime_dir = tmp_path / "runtime"
    manager_type = _worker_manager_method(
        "_prepare_runtime_dir",
        {"Path": lambda value: shm_dir if value == "/dev/shm" else Path(value)},
    )
    manager = manager_type()
    manager.args = SimpleNamespace(
        runtime_identity=SimpleNamespace(
            shm_prefix=f"{prefix}.", runtime_dir=runtime_dir
        )
    )
    manager._runtime_dir_created = False
    manager._runtime_namespace_owned = False

    manager._prepare_runtime_dir()

    assert manager._runtime_namespace_owned
    assert neighbor.exists()


def test_worker_stop_does_not_clean_longer_instance_id(tmp_path):
    run_id = "a" * 32
    prefix = f"batchgen_lane_{run_id}"
    own = tmp_path / f"{prefix}.host_kv"
    neighbor = tmp_path / f"{prefix}_b_{'b' * 32}.host_kv"
    own.touch()
    neighbor.touch()

    def cleanup_resources(*, shm_prefix, **kwargs):
        for entry in tmp_path.iterdir():
            if entry.name.startswith(shm_prefix):
                entry.unlink()

    fake_logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    manager_type = _worker_manager_method(
        "stop",
        {
            "cleanup_resources": cleanup_resources,
            "cleanup_model_shm_files": lambda model_info: None,
            "logger": fake_logger,
        },
    )
    manager = manager_type()
    manager.args = SimpleNamespace(
        runtime_identity=SimpleNamespace(
            shm_prefix=f"{prefix}.", runtime_dir=tmp_path / "unused"
        )
    )
    manager._stopping = False
    manager.started = True
    manager._runtime_dir_created = False
    manager._runtime_namespace_owned = True
    manager._runtime_locks = None
    manager._lane_lease = None
    manager.worker_process = None
    manager.distributed_weight_daemon = None
    manager.parameter_server_instance = None
    manager.model_info = {}
    manager.skeleton_state_dict_file = None
    manager._monitor_stop_event = SimpleNamespace(set=lambda: None)
    manager._monitor_thread = None
    manager._hugepages_enabled = False
    manager._cleanup_skeleton_state_dict_file = lambda: None

    manager.stop()

    assert not own.exists()
    assert neighbor.exists()


@pytest.mark.parametrize("local_owner", [False, True])
def test_worker_stop_only_unlinks_locally_owned_model_shm(tmp_path, local_owner):
    weight = tmp_path / "shm_weight"
    metadata = tmp_path / "shm_metadata"
    weight.touch()
    metadata.touch()
    cleaned = []

    def cleanup_model_shm_files(model_info):
        cleaned.append(True)
        for key in ("shm_name", "tensor_meta_shm_name"):
            (tmp_path / model_info.pop(key).lstrip("/")).unlink()

    fake_logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    manager_type = _worker_manager_method(
        "stop",
        {"cleanup_model_shm_files": cleanup_model_shm_files, "logger": fake_logger},
    )
    manager = manager_type()
    manager.args = SimpleNamespace(
        runtime_identity=SimpleNamespace(runtime_dir=tmp_path / "unused")
    )
    manager._stopping = False
    manager.started = True
    manager._runtime_dir_created = False
    manager._runtime_namespace_owned = False
    manager._runtime_locks = None
    manager._lane_lease = None
    manager.worker_process = None
    manager.distributed_weight_daemon = None
    manager.parameter_server_instance = object() if local_owner else None
    manager.model_info = {
        "shm_name": f"/{weight.name}",
        "tensor_meta_shm_name": f"/{metadata.name}",
    }
    manager.skeleton_state_dict_file = None
    manager._monitor_stop_event = SimpleNamespace(set=lambda: None)
    manager._monitor_thread = None
    manager._hugepages_enabled = False
    manager._cleanup_skeleton_state_dict_file = lambda: None

    manager.stop()

    assert cleaned == ([True] if local_owner else [])
    assert weight.exists() is not local_owner
    assert metadata.exists() is not local_owner
    assert "shm_name" not in manager.model_info
    assert "tensor_meta_shm_name" not in manager.model_info


def test_worker_start_rolls_back_partial_startup_before_reraising():
    events = []
    manager_type = _worker_manager_method(
        "start",
        {"logger": type("Logger", (), {"exception": lambda *args: None})()},
    )
    manager = manager_type()
    manager.started = False

    def fail_start():
        events.append("start")
        raise RuntimeError("startup failed")

    manager._start_impl = fail_start
    manager.stop = lambda: events.append("stop")

    with pytest.raises(RuntimeError, match="startup failed"):
        manager.start()
    assert events == ["start", "stop"]


def test_worker_stop_preserves_artifacts_and_locks_for_live_owned_pid(
    tmp_path,
):
    events = []
    fake_logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    manager_type = _worker_manager_method(
        "stop",
        {
            "cleanup_resources": lambda **kwargs: events.append("cleanup"),
            "logger": fake_logger,
        },
    )
    manager = manager_type()
    manager._stopping = False
    manager.started = True
    manager._runtime_dir_created = True
    manager._runtime_namespace_owned = True
    manager._runtime_locks = SimpleNamespace(
        close=lambda: events.append("runtime-lock-close")
    )
    manager._lane_lease = SimpleNamespace(
        close=lambda: events.append("lane-lease-close")
    )
    manager.worker_process = SimpleNamespace(processes=[SimpleNamespace(pid=123)])
    manager.distributed_weight_daemon = None
    manager.model_info = {}
    manager.skeleton_state_dict_file = None
    manager._monitor_stop_event = SimpleNamespace(set=lambda: None)
    manager._monitor_thread = None
    manager._stop_workers = lambda: (_ for _ in ()).throw(
        RuntimeError("worker teardown left live owned PIDs")
    )
    manager.request_queue = SimpleNamespace(put=lambda value: None)
    manager._join_lock = nullcontext()
    manager._cleanup_skeleton_state_dict_file = lambda: None
    manager.args = SimpleNamespace(
        runtime_identity=SimpleNamespace(
            shm_prefix="batchgen_lane-0_run.",
            runtime_dir=tmp_path / "runtime",
        )
    )
    manager._hugepages_enabled = False

    with pytest.raises(RuntimeError, match="live owned PIDs"):
        manager.stop()

    assert "cleanup" not in events
    assert "runtime-lock-close" not in events
    assert "lane-lease-close" not in events
    assert manager._runtime_dir_created
    assert manager._runtime_namespace_owned
    assert manager._runtime_locks is not None
    assert manager._lane_lease is not None


@pytest.mark.parametrize("exits_after_term", [True, False])
def test_worker_stop_signals_only_original_child_handles(exits_after_term):
    events = []

    class Child:
        pid = 123
        exitcode = None

        def join(self, timeout):
            events.append(("join", timeout))
            if exits_after_term and ("signal", 9, 15) in events:
                self.exitcode = 0

    child = Child()
    fake_os = SimpleNamespace(
        pidfd_open=lambda pid: events.append(("open", pid)) or 9,
        close=lambda fd: events.append(("close", fd)),
    )
    fake_signal = SimpleNamespace(
        SIGTERM=15,
        SIGKILL=9,
        pidfd_send_signal=lambda fd, sig: events.append(("signal", fd, sig)),
    )
    fake_logger = SimpleNamespace(warning=lambda *args, **kwargs: None)
    manager_type = _worker_manager_method(
        "_stop_workers",
        {
            "logger": fake_logger,
            "os": fake_os,
            "signal": fake_signal,
            "time": __import__("time"),
        },
    )
    manager = manager_type()
    manager.worker_process = SimpleNamespace(processes=[child])
    manager._join_lock = nullcontext()
    manager.request_queue = SimpleNamespace(put=lambda value: events.append("poison"))

    if exits_after_term:
        manager._stop_workers()
        assert ("signal", 9, 9) not in events
    else:
        with pytest.raises(RuntimeError, match="live owned PIDs"):
            manager._stop_workers()
        assert ("signal", 9, 9) in events
    assert events[0:2] == [("open", 123), ("signal", 9, 15)]
    assert events[-1] == ("close", 9)


@pytest.mark.skipif(
    not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"),
    reason="Linux PIDFD support required",
)
def test_worker_stop_real_child_pidfd():
    child_process = subprocess.Popen(["sleep", "30"])

    class Child:
        pid = child_process.pid

        @property
        def exitcode(self):
            return child_process.poll()

        def join(self, timeout):
            try:
                child_process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass

    manager_type = _worker_manager_method(
        "_stop_workers",
        {
            "logger": SimpleNamespace(warning=lambda *args, **kwargs: None),
            "os": os,
            "signal": signal,
            "time": time,
        },
    )
    manager = manager_type()
    manager.worker_process = SimpleNamespace(processes=[Child()])
    manager._join_lock = nullcontext()
    manager.request_queue = SimpleNamespace(put=lambda value: None)
    try:
        manager._stop_workers()
        assert child_process.poll() is not None
    finally:
        if child_process.poll() is None:
            child_process.kill()
        child_process.wait()


def test_worker_monitor_does_not_invoke_context_auto_kill():
    events = []

    class StopEvent:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, timeout):
            self.stopped = True

    manager_type = _worker_manager_method("_monitor_worker_processes")
    manager = manager_type()
    manager._monitor_stop_event = StopEvent()
    manager.worker_process = SimpleNamespace(
        processes=[SimpleNamespace(join=lambda timeout: events.append("child-join"))],
        join=lambda timeout: events.append("context-join"),
    )
    manager._join_lock = nullcontext()
    manager._monitor_interval_s = 1
    manager._collect_worker_exit_reason = lambda: None

    manager._monitor_worker_processes()

    assert events == ["child-join"]


def test_worker_exit_reason_preserves_python_traceback(tmp_path):
    error_file = tmp_path / "worker-error.pickle"
    error_file.write_bytes(pickle.dumps("Traceback: worker ValueError"))
    manager_type = _worker_manager_method(
        "_collect_worker_exit_reason",
        {"Optional": Optional, "os": os, "pickle": pickle},
    )
    manager = manager_type()
    manager.worker_process = SimpleNamespace(
        processes=[SimpleNamespace(pid=123, exitcode=1)],
        error_files=[str(error_file)],
    )

    reason = manager._collect_worker_exit_reason()

    assert "idx=0 pid=123 exitcode=1" in reason
    assert "Traceback: worker ValueError" in reason
