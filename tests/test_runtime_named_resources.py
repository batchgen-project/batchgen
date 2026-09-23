"""Runtime namespaces must prevent cross-instance mutable-resource takeover."""

from __future__ import annotations

import ast
import copy
import gc
import importlib
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


def _isolated_worker_function(name: str):
    tree = ast.parse(WORKER.read_text())
    function = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "QueryBookPoolCapacityError": RuntimeError,
        "Tuple": Tuple,
        "torch": torch,
    }
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return namespace[name]


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


def test_query_book_duplicate_creator_cannot_unlink_existing_segment():
    allocate = _isolated_worker_function("allocate_node_shared_int64")
    name = f"batchgen_query_book_collision_{uuid.uuid4().hex}"
    owner = shared_memory.SharedMemory(name=name, create=True, size=64)
    try:
        sentinel = b"alive123"
        owner.buf[: len(sentinel)] = sentinel

        with pytest.raises(FileExistsError):
            allocate(name, 1, 1, True, lambda: None)

        attached = shared_memory.SharedMemory(name=name)
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
    assert shm_source.count("create ? O_CREAT | O_EXCL : 0") >= 2
    assert "shm_open(shm_name.c_str(), O_RDWR | O_CREAT | O_EXCL" in shm_source
    destructor = server_source.split("Parameter_Server::~Parameter_Server()", 1)[1]
    destructor = destructor.split("Parameter_Server::get_skeleton_state_dict", 1)[0]
    assert "if (weight_posix_shm_owned_)" in destructor
    assert "if (weight_hugetlbfs_owned_ && !this->weight_hugetlbfs_path_.empty())" in destructor
    assert "unlink(this->weight_hugetlbfs_path_.c_str())" in destructor
    assert "close(this->weights_memfd_fd_)" in destructor
    assert "free_shared_pinned_memory(this->weight_ptr_, this->mapped_size_)" in destructor
    assert "if (tensor_meta_shm_owned_)" in destructor
    assert "if (create && errno == EEXIST)" in shm_source


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(),
    reason="native memfd lifetime check requires Linux CUDA",
)
def test_native_parameter_server_releases_full_memfd_mapping(tmp_path):
    from batchgen.models.engine_loader import core_engine

    weight_name = f"/shm_{uuid.uuid4()}"
    metadata_name = f"/shm_{uuid.uuid4()}"
    parameter_server = core_engine.Parameter_Server(False, True)
    parameter_server.Init(weight_name, metadata_name, 4096, str(tmp_path), {})
    fd = parameter_server.weights_memfd_fd()
    assert fd >= 0
    assert "memfd:batchgen_weights" in os.readlink(f"/proc/self/fd/{fd}")
    del parameter_server
    gc.collect()

    with pytest.raises(OSError):
        os.fstat(fd)
    assert "memfd:batchgen_weights" not in Path("/proc/self/maps").read_text()
    assert not (Path("/dev/shm") / metadata_name[1:]).exists()


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(),
    reason="native parameter-server collision check requires Linux CUDA",
)
@pytest.mark.parametrize("collision", ["weight", "metadata"])
def test_native_model_shm_collision_preserves_existing_region(tmp_path, collision):
    from batchgen.models.engine_loader import core_engine

    weight_name = f"/shm_{uuid.uuid4()}"
    metadata_name = f"/shm_{uuid.uuid4()}"
    collided_name = weight_name if collision == "weight" else metadata_name
    owner = shared_memory.SharedMemory(
        name=collided_name[1:], create=True, size=64
    )
    sentinel = b"lane-owner-alive"
    owner.buf[: len(sentinel)] = sentinel
    parameter_server = core_engine.Parameter_Server(False, False)
    try:
        try:
            with pytest.raises(RuntimeError, match="File exists"):
                parameter_server.Init(
                    weight_name, metadata_name, 4096, str(tmp_path), {}
                )
        finally:
            del parameter_server
        attached = shared_memory.SharedMemory(name=collided_name[1:])
        try:
            assert attached.size == 64
            assert bytes(attached.buf[: len(sentinel)]) == sentinel
        finally:
            attached.close()
        if collision == "metadata":
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=weight_name[1:])
    finally:
        owner.close()
        owner.unlink()
