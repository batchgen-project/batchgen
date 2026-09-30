"""The QueryBook input_ids table and the skeleton state dict must be unnamed.

Both used to be named objects — a POSIX segment under /dev/shm and a file in
the runtime dir — so a forced exit of the serving processes left the whole
table, or the whole skeleton, behind for something else to reclaim. Both are
now anonymous memfds: the kernel frees them with their last reference however
the holder died, and the other ranks reach them through /proc/<pid>/fd/<N>.
"""

from __future__ import annotations

import ast
import copy
import gc
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Dict, Tuple
from uuid import uuid4

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
WORKER_MANAGER = ROOT / "batchgen" / "server" / "worker_manager.py"

HELPERS = ("NodeSharedMemfd", "allocate_node_shared_int64")

requires_memfd = pytest.mark.skipif(
    not hasattr(os, "memfd_create"), reason="memfd_create requires Linux"
)


def _worker_tree() -> ast.Module:
    return ast.parse(WORKER.read_text(), filename=str(WORKER))


def _top_level(tree: ast.Module, name: str) -> ast.AST:
    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name
    )


def _method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(), filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _calls_to(node: ast.AST, module_name: str) -> list[str]:
    """Every ``<module_name>.<attr>(...)`` called anywhere under ``node``."""
    names = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == module_name
        ):
            names.append(func.attr)
    return names


def _bare_calls(node: ast.AST, name: str) -> int:
    return sum(
        1
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == name
    )


def _allocator():
    """Exec the memfd helpers out of the worker without importing the worker.

    Re-derived inside spawned children too, so the attaching side of the test
    runs the production code rather than a copy of it.
    """
    tree = _worker_tree()
    module = ast.Module(
        body=[_top_level(tree, name) for name in HELPERS], type_ignores=[]
    )
    namespace = {
        "QueryBookPoolCapacityError": RuntimeError,
        "Tuple": Tuple,
        "os": os,
        "torch": torch,
    }
    exec(compile(ast.fix_missing_locations(module), str(WORKER), "exec"), namespace)
    return namespace["allocate_node_shared_int64"]


# --------------------------------------------------------------------------
# Source checks: nothing on these paths may create a named object.
# --------------------------------------------------------------------------


def test_worker_never_names_the_input_ids_region():
    source = WORKER.read_text()

    assert "shared_memory" not in source
    assert "shm_open" not in source
    assert "os.memfd_create(label, os.MFD_CLOEXEC)" in source
    # Attachers reach the creator's descriptor, and their handle must not
    # survive an exec either.
    assert 'f"/proc/{creator_pid}/fd/{creator_fd}", os.O_RDWR | os.O_CLOEXEC' in source


def test_allocator_creates_no_name_and_unlinks_nothing():
    allocate = _top_level(_worker_tree(), "allocate_node_shared_int64")
    dump = ast.dump(allocate)

    assert "unlink" not in dump
    assert "SharedMemory" not in dump
    assert set(_calls_to(allocate, "os")) == {
        "memfd_create",
        "ftruncate",
        "getpid",
        "open",
        "fstat",
    }


def test_retirement_closes_the_descriptor_instead_of_unlinking():
    method = _method(WORKER, "BatchGenWorker", "_retire_buffer_pool")
    body = method.body[1:] if ast.get_docstring(method) else method.body
    dump = "\n".join(ast.dump(node) for node in body)

    assert "unlink" not in dump
    assert "release_fd" in dump
    # The mapping outlives the descriptor: views handed out before a grow may
    # still be live, so nothing here may unmap.
    assert "munmap" not in dump and "'mapping'" not in dump


# --------------------------------------------------------------------------
# The pid/fd exchange must stay ONE collective, issued by every rank.
# --------------------------------------------------------------------------


def test_identity_exchange_is_exactly_one_collective():
    exchange = _method(WORKER, "BatchGenWorker", "_exchange_node_memfd_identity")

    assert _calls_to(exchange, "dist") == ["all_gather"]


def test_allocator_keeps_one_exchange_and_one_barrier():
    allocate = _top_level(_worker_tree(), "allocate_node_shared_int64")

    # The exchange replaced the post-create barrier rather than adding to it:
    # one call each.
    assert _bare_calls(allocate, "exchange") == 1
    assert _bare_calls(allocate, "barrier") == 1
    # Both sit at the function's top level, so every rank runs them on every
    # path — a rank that skipped one would hang the others.
    for name in ("exchange", "barrier"):
        assert any(
            _bare_calls(stmt, name) == 1 for stmt in allocate.body
        ), f"{name}() must not be conditional"


def test_pool_allocation_passes_the_exchange_and_the_barrier():
    ensure = _method(WORKER, "BatchGenWorker", "_ensure_buffer_pool")
    dump = ast.dump(ensure)

    assert "_exchange_node_memfd_identity" in dump
    assert "barrier" in dump
    assert "unlink" not in dump


# --------------------------------------------------------------------------
# The skeleton state dict must not reach a named path either.
# --------------------------------------------------------------------------


def test_skeleton_is_written_into_a_memfd_not_the_runtime_dir():
    source = WORKER_MANAGER.read_text()

    assert 'os.memfd_create("batchgen_skeleton", os.MFD_CLOEXEC)' in source
    assert 'f"/proc/{os.getpid()}/fd/{fd}"' in source
    # The named temp file, its deletion and the atexit hook that existed only
    # to delete it are all gone.
    assert "mkstemp" not in source
    assert "tempfile" not in source
    assert "atexit" not in source
    assert "os.remove" not in source


def test_skeleton_store_never_touches_the_runtime_dir():
    store = _method(WORKER_MANAGER, "WorkerManager", "_store_skeleton_state_dict")
    dump = ast.dump(store)

    assert "runtime_dir" not in dump
    assert "runtime_identity" not in dump
    assert _calls_to(store, "torch") == ["save"]


def test_worker_loads_the_skeleton_from_whatever_path_the_server_published():
    source = WORKER.read_text()

    assert "torch.load(args.skeleton_state_dict_file)" in source


def _skeleton_holder():
    """A stand-in carrying only the two real skeleton methods.

    Instantiating a WorkerManager would need the compiled extension and a full
    ServerArgs; these two methods touch neither.
    """
    tree = ast.parse(WORKER_MANAGER.read_text(), filename=str(WORKER_MANAGER))
    class_node = copy.deepcopy(
        next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "WorkerManager"
        )
    )
    class_node.body = [
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_store_skeleton_state_dict", "_close_skeleton_memfd")
    ]
    assert len(class_node.body) == 2
    class_node.name = "SkeletonHolder"
    class_node.bases = []
    class_node.keywords = []
    class_node.decorator_list = []
    module = ast.Module(body=[class_node], type_ignores=[])
    namespace = {
        "Any": Any,
        "Dict": Dict,
        "logger": logging.getLogger(__name__),
        "logging": logging,
        "os": os,
        "torch": torch,
    }
    exec(
        compile(ast.fix_missing_locations(module), str(WORKER_MANAGER), "exec"),
        namespace,
    )
    return namespace["SkeletonHolder"]


@requires_memfd
def test_skeleton_round_trips_through_proc_without_a_named_file():
    holder = _skeleton_holder()()
    holder.skeleton_memfd_fd = -1
    holder.skeleton_state_dict_file = None

    state = {"a": torch.arange(4), "b": torch.zeros(2, 2)}
    holder._store_skeleton_state_dict(state)
    try:
        path = holder.skeleton_state_dict_file
        assert path == f"/proc/{os.getpid()}/fd/{holder.skeleton_memfd_fd}"
        assert "memfd:batchgen_skeleton" in os.readlink(path)

        loaded = torch.load(path)
        assert torch.equal(loaded["a"], state["a"])
        assert torch.equal(loaded["b"], state["b"])
    finally:
        holder._close_skeleton_memfd()

    assert holder.skeleton_memfd_fd == -1
    assert holder.skeleton_state_dict_file is None


# --------------------------------------------------------------------------
# Behaviour: two processes really do share one int64 table through /proc.
# --------------------------------------------------------------------------


def _attach_in_child(creator_pid: int, creator_fd: int, rows: int, width: int, out):
    """Attach as a non-creator rank would, echo the table, write one cell back."""
    allocate = _allocator()
    buf, memfd = allocate(
        "batchgen_input_ids_child",
        rows,
        width,
        False,
        lambda pid, fd: (creator_pid, creator_fd),
        lambda: None,
    )
    try:
        out.put(buf.tolist())
        buf[rows - 1, width - 1] = 4242
    finally:
        del buf
        gc.collect()
        memfd.release_fd()


@requires_memfd
def test_two_processes_share_one_unnamed_int64_table():
    rows, width = 3, 4
    label = f"batchgen_input_ids_{uuid4().hex}"
    allocate = _allocator()
    buf, memfd = allocate(
        label, rows, width, True, lambda pid, fd: (pid, fd), lambda: None
    )
    try:
        assert torch.all(buf == 0), "a fresh memfd must be zero-filled"
        buf[0, 0] = 7
        buf[1, 2] = 11
        expected = buf.tolist()

        ctx = mp.get_context("spawn")
        out = ctx.Queue()
        child = ctx.Process(
            target=_attach_in_child,
            args=(os.getpid(), memfd.fd, rows, width, out),
        )
        child.start()
        try:
            echoed = out.get(timeout=120)
        finally:
            child.join(timeout=120)

        assert child.exitcode == 0
        assert echoed == expected, "the child mapped a different region"
        assert buf[rows - 1, width - 1].item() == 4242, (
            "the child's write did not reach the creator's mapping"
        )

        shm_dir = Path("/dev/shm")
        if shm_dir.is_dir():
            assert not any(label in entry.name for entry in shm_dir.iterdir())
        assert "memfd:" in os.readlink(f"/proc/self/fd/{memfd.fd}")
    finally:
        del buf
        gc.collect()
        memfd.release_fd()


@requires_memfd
def test_growth_keeps_the_superseded_mapping_valid():
    """Retiring a generation closes its fd; the old mapping must survive."""
    allocate = _allocator()
    old_buf, old_memfd = allocate(
        "batchgen_input_ids_g1", 2, 4, True, lambda pid, fd: (pid, fd), lambda: None
    )
    new_buf, new_memfd = allocate(
        "batchgen_input_ids_g2", 4, 8, True, lambda pid, fd: (pid, fd), lambda: None
    )
    try:
        old_buf[0, 0] = 5
        new_buf[:2, :4] = old_buf  # what QueryBookBufferPool.adopt does

        # _retire_buffer_pool drops only the descriptor.
        old_memfd.release_fd()
        assert old_memfd.fd == -1

        old_buf[1, 3] = 9
        assert old_buf[0, 0].item() == 5
        assert old_buf[1, 3].item() == 9
        assert new_buf[0, 0].item() == 5
        assert new_buf[1, 3].item() == 0, "the generations must not alias"
    finally:
        del old_buf, new_buf
        gc.collect()
        old_memfd.release_fd()
        new_memfd.release_fd()


@requires_memfd
def test_undersized_region_is_rejected_rather_than_truncated():
    allocate = _allocator()
    fd = os.memfd_create("batchgen_input_ids_small", os.MFD_CLOEXEC)
    os.ftruncate(fd, 8)
    try:
        with pytest.raises(RuntimeError, match="need 64"):
            allocate(
                "batchgen_input_ids_small",
                2,
                4,
                False,
                lambda pid, _fd: (os.getpid(), fd),
                lambda: None,
            )
    finally:
        os.close(fd)


@requires_memfd
def test_attacher_refuses_an_unpublished_creator():
    allocate = _allocator()

    with pytest.raises(RuntimeError, match="no rank published a memfd"):
        allocate(
            "batchgen_input_ids_missing",
            1,
            1,
            False,
            lambda pid, fd: (-1, -1),
            lambda: None,
        )
