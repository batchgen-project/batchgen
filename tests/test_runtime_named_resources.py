"""Runtime namespaces must prevent cross-instance mutable-resource takeover."""

from __future__ import annotations

import ast
import copy
import importlib
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


def test_shared_runtime_mode_remains_fail_loud():
    runtime = _load_runtime_identity_module()
    with pytest.raises(ValueError, match="shared runtime mode is not available"):
        runtime.RuntimeIdentity.create("lane-0", mode="shared")


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
