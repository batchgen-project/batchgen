"""Runtime namespaces must prevent cross-instance mutable-resource takeover."""

from __future__ import annotations

import ast
import copy
import gc
import importlib
import importlib.util
import os
import sys
import types
import uuid
from multiprocessing import shared_memory
from pathlib import Path
from typing import Tuple

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
POSIX_SHM = ROOT / "core" / "Parameter_Server" / "posix_shm.cpp"
PARAMETER_SERVER = ROOT / "core" / "Parameter_Server" / "Parameter_Server.cpp"


def _load_runtime_identity_module():
    package_name = "batchgen.server"
    previous = sys.modules.get(package_name)
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    sys.modules[package_name] = package
    try:
        return importlib.import_module("batchgen.server.runtime_identity")
    finally:
        if previous is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = previous


def _create_memfd():
    """Load create_memfd from batchgen/memfd.py without importing batchgen."""
    path = Path(__file__).resolve().parents[1] / "batchgen" / "memfd.py"
    spec = importlib.util.spec_from_file_location("_batchgen_memfd", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_memfd


def _isolated_worker_namespace(*names: str) -> dict:
    """Exec the named top-level worker definitions without importing the worker."""
    tree = ast.parse(WORKER.read_text())
    wanted = [
        copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert len(wanted) == len(names), [node.name for node in wanted]
    module = ast.Module(body=wanted, type_ignores=[])
    namespace = {
        "QueryBookPoolCapacityError": RuntimeError,
        "Tuple": Tuple,
        "os": os,
        "create_memfd": _create_memfd(),
        "torch": torch,
    }
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return namespace


def test_runtime_identity_derives_disjoint_run_names():
    runtime = _load_runtime_identity_module()
    first = runtime.RuntimeIdentity.create(
        "lane-0",
        run_id="0" * 32,
    )
    second = runtime.RuntimeIdentity.create(
        "lane-0",
        run_id="1" * 32,
    )

    assert first.resource_prefix != second.resource_prefix
    assert first.shm_prefix == f"{first.resource_prefix}."
    assert first.host_kv_shm_name.startswith(first.shm_prefix)
    assert first.host_kv_aux_shm_name.startswith(first.shm_prefix)
    assert first.query_book_shm_prefix.startswith(first.shm_prefix)
    assert first.host_kv_shm_name != second.host_kv_shm_name
    assert first.host_kv_aux_shm_name != second.host_kv_aux_shm_name
    assert first.query_book_shm_prefix != second.query_book_shm_prefix
    assert first.reload_status_dir != second.reload_status_dir


@pytest.mark.parametrize(
    "instance_id",
    ["", "Lane-0", "-lane", "lane.name", "a" * 49],
)
def test_runtime_identity_rejects_malformed_instance_ids(instance_id):
    runtime = _load_runtime_identity_module()
    with pytest.raises(ValueError, match="instance_id must match"):
        runtime.RuntimeIdentity.create(instance_id)


def test_runtime_identity_accepts_only_declared_modes():
    runtime = _load_runtime_identity_module()
    identity = runtime.RuntimeIdentity.create(
        "lane-0", mode="shared", run_id="0" * 32
    )
    assert identity.mode == "shared"

    with pytest.raises(ValueError, match="runtime mode must be"):
        runtime.RuntimeIdentity.create("lane-0", mode="invalid")


@pytest.mark.parametrize("run_id", ["", "0" * 31, "G" * 32, 123])
def test_runtime_identity_rejects_malformed_run_ids(run_id):
    runtime = _load_runtime_identity_module()
    with pytest.raises(ValueError, match="run_id must be exactly"):
        runtime.RuntimeIdentity.create("lane-0", run_id=run_id)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="memfd_create requires Linux"
)
def test_query_book_creator_cannot_touch_a_same_named_segment():
    """The QueryBook label is a memfd tag, so it can never collide on a name."""
    namespace = _isolated_worker_namespace(
        "NodeSharedMemfd", "allocate_node_shared_int64"
    )
    allocate = namespace["allocate_node_shared_int64"]
    label = f"batchgen_query_book_collision_{uuid.uuid4().hex}"
    owner = shared_memory.SharedMemory(name=label, create=True, size=64)
    try:
        sentinel = b"alive123"
        owner.buf[: len(sentinel)] = sentinel

        buf, memfd = allocate(label, 1, 1, True, lambda pid, fd: (pid, fd), lambda: None)
        try:
            # A fresh memfd is zero-filled; it did not adopt the foreign bytes.
            assert buf[0, 0].item() == 0
            assert "memfd:" in os.readlink(f"/proc/self/fd/{memfd.fd}")
        finally:
            del buf
            gc.collect()
            memfd.release_fd()

        attached = shared_memory.SharedMemory(name=label)
        try:
            assert attached.size == 64
            assert bytes(attached.buf[: len(sentinel)]) == sentinel
        finally:
            attached.close()
    finally:
        owner.close()
        owner.unlink()


def test_model_shm_creators_and_destructor_preserve_foreign_names():
    shm_source = POSIX_SHM.read_text()
    server_source = PARAMETER_SERVER.read_text()
    # Only the hugetlbfs file is still created by name, and only with O_EXCL.
    assert shm_source.count("create ? O_CREAT | O_EXCL : 0") == 1
    destructor = server_source.split("Parameter_Server::~Parameter_Server()", 1)[1]
    destructor = destructor.split("Parameter_Server::get_skeleton_state_dict", 1)[0]
    assert "if (weight_hugetlbfs_owned_ && !this->weight_hugetlbfs_path_.empty())" in destructor
    assert "unlink(this->weight_hugetlbfs_path_.c_str())" in destructor
    assert "close(this->weights_memfd_fd_)" in destructor
    assert "close(this->tensor_meta_memfd_fd_)" in destructor
    assert "free_shared_pinned_memory(this->weight_ptr_, this->mapped_size_)" in destructor
    assert "if (create && errno == EEXIST)" in shm_source
    assert "memfd creator requires an output fd" in shm_source


def _core_engine():
    """Import the compiled extension or skip; source checks still have to run."""
    try:
        from batchgen.models.engine_loader import core_engine
    except Exception as exc:
        pytest.skip(f"compiled extension unavailable: {exc}")
    return core_engine


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(),
    reason="native memfd lifetime check requires Linux CUDA",
)
@pytest.mark.parametrize("enable_thp", [False, True])
def test_native_parameter_server_releases_full_memfd_mapping(tmp_path, enable_thp):
    """Both regions are anonymous memfds in every mode, THP or not."""
    core_engine = _core_engine()

    weight_name = f"/shm_{uuid.uuid4()}"
    metadata_name = f"/shm_{uuid.uuid4()}"
    parameter_server = core_engine.Parameter_Server(False, enable_thp)
    parameter_server.Init(weight_name, metadata_name, 4096, str(tmp_path), {})
    fd = parameter_server.weights_memfd_fd()
    metadata_fd = parameter_server.tensor_meta_memfd_fd()
    assert fd >= 0
    assert metadata_fd >= 0
    assert "memfd:batchgen_weights" in os.readlink(f"/proc/self/fd/{fd}")
    assert "memfd:batchgen_tensor_meta" in os.readlink(
        f"/proc/self/fd/{metadata_fd}"
    )
    assert not (Path("/dev/shm") / weight_name[1:]).exists()
    assert not (Path("/dev/shm") / metadata_name[1:]).exists()
    del parameter_server
    gc.collect()

    with pytest.raises(OSError):
        os.fstat(fd)
    with pytest.raises(OSError):
        os.fstat(metadata_fd)
    assert "memfd:batchgen_weights" not in Path("/proc/self/maps").read_text()


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="native host KV memfd check requires Linux",
)
def test_native_host_kv_manager_creates_no_named_shm_object():
    """Host KV names are labels; the region itself is an anonymous memfd."""
    core_engine = _core_engine()

    shm_name = f"batchgen_host_kv_{uuid.uuid4().hex}"
    config = core_engine.HostPagedKVConfig()
    config.shm_name = shm_name
    config.num_layers = 1
    config.num_pages = 4
    config.page_size_tokens = 4
    config.num_k_heads = 1
    config.k_head_dim = 8
    config.num_v_heads = 0
    config.k_element_size_bytes = 2
    config.sequence_table_capacity = 8
    config.alignment_bytes = 64

    manager = core_engine.MLAHostPagedKVManager(config)
    manager.initialize(True)
    fd = manager.memfd_fd()
    assert fd >= 0
    assert "memfd:batchgen_kv" in os.readlink(f"/proc/self/fd/{fd}")
    assert not (Path("/dev/shm") / shm_name).exists()

    del manager
    gc.collect()

    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(),
    reason="native parameter-server name check requires Linux CUDA",
)
@pytest.mark.parametrize("shared_label", ["weight", "metadata"])
def test_native_model_names_are_labels_only(tmp_path, shared_label):
    """A foreign /dev/shm object with the same name is neither used nor touched."""
    core_engine = _core_engine()

    weight_name = f"/shm_{uuid.uuid4()}"
    metadata_name = f"/shm_{uuid.uuid4()}"
    label = weight_name if shared_label == "weight" else metadata_name
    owner = shared_memory.SharedMemory(name=label[1:], create=True, size=64)
    sentinel = b"lane-owner-alive"
    owner.buf[: len(sentinel)] = sentinel
    parameter_server = core_engine.Parameter_Server(False, False)
    try:
        parameter_server.Init(
            weight_name, metadata_name, 4096, str(tmp_path), {}
        )
        assert parameter_server.weights_memfd_fd() >= 0
        assert parameter_server.tensor_meta_memfd_fd() >= 0
        del parameter_server

        attached = shared_memory.SharedMemory(name=label[1:])
        try:
            assert attached.size == 64
            assert bytes(attached.buf[: len(sentinel)]) == sentinel
        finally:
            attached.close()
    finally:
        owner.close()
        owner.unlink()
