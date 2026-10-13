"""A run's liveness lock must prove when its leftovers are safe to reclaim."""

from __future__ import annotations

import ast
import copy
import fcntl
import importlib
import logging
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WORKER_MANAGER = ROOT / "batchgen" / "server" / "worker_manager.py"
WORKER_MAIN_LOOP = ROOT / "batchgen" / "server_worker_main_loop.py"

# Holds the run lock shared, exactly as a live worker of that run does.
_LOCK_HOLDER = """
import fcntl, sys, time
handle = open(sys.argv[1], "r+b", buffering=0)
fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
sys.stdout.write("held\\n")
sys.stdout.flush()
time.sleep(600)
"""


def _load_runtime_modules():
    """Import the runtime modules without importing the batchgen package."""
    package_name = "batchgen.server"
    module_names = (
        "batchgen.server.runtime_identity",
        "batchgen.server.process_utils",
        "batchgen.server.runtime_locks",
    )
    previous_package = sys.modules.get(package_name)
    previous_modules = {
        name: sys.modules.pop(name, None) for name in module_names
    }
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    sys.modules[package_name] = package
    try:
        identity = importlib.import_module(module_names[0])
        locks = importlib.import_module(module_names[-1])
        return identity, locks
    finally:
        for name in module_names:
            sys.modules.pop(name, None)
            if previous_modules[name] is not None:
                sys.modules[name] = previous_modules[name]
        if previous_package is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = previous_package


def _worker_manager_method(name: str, globals_: dict):
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
    namespace = dict(globals_)
    exec(
        compile(ast.fix_missing_locations(module), str(WORKER_MANAGER), "exec"),
        namespace,
    )
    return namespace["IsolatedManager"]


def _exclusive_lock_is_free(path: Path) -> bool:
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


def _dead_run_dir(locks, temp_dir: Path, name: str) -> Path:
    """A leftover run directory whose lock nobody holds."""
    run_dir = temp_dir / name
    run_dir.mkdir()
    (run_dir / locks.RUN_LOCK_NAME).touch()
    return run_dir


def _identity(identity_module, instance_id: str = "lane-0", run_id: str = "f" * 32):
    return identity_module.RuntimeIdentity.create(instance_id, run_id=run_id)


# ---------------------------------------------------------------------------
# run.lock itself
# ---------------------------------------------------------------------------
def test_run_lock_is_shared_so_every_process_of_a_run_can_join(tmp_path):
    _identity_module, locks = _load_runtime_modules()

    server_lock = locks.hold_run_lock(tmp_path, create=True)
    lock_path = tmp_path / locks.RUN_LOCK_NAME
    assert lock_path.is_file()
    assert lock_path.stat().st_mode & 0o777 == 0o600

    worker_lock = locks.hold_run_lock(tmp_path)
    try:
        assert not _exclusive_lock_is_free(lock_path)
    finally:
        locks.release_run_lock(worker_lock)
    # One live process of the run is still enough to refuse a reclaim.
    assert not _exclusive_lock_is_free(lock_path)

    locks.release_run_lock(server_lock)
    assert _exclusive_lock_is_free(lock_path)


def test_worker_run_lock_requires_the_server_created_file(tmp_path):
    _identity_module, locks = _load_runtime_modules()

    with pytest.raises(locks.RuntimeLockError, match="run lock is missing"):
        locks.hold_run_lock(tmp_path)

    assert not (tmp_path / locks.RUN_LOCK_NAME).exists()


# ---------------------------------------------------------------------------
# Dead-run reclaim
# ---------------------------------------------------------------------------
def test_reclaim_removes_a_released_run_with_its_shm_entries(tmp_path, caplog):
    identity_module, locks = _load_runtime_modules()
    identity = _identity(identity_module)
    temp_dir = tmp_path / "tmp"
    shm_dir = tmp_path / "shm"
    temp_dir.mkdir()
    shm_dir.mkdir()

    dead_id = "a" * 32
    dead = _dead_run_dir(locks, temp_dir, f"batchgen_lane-0_{dead_id}")
    other_instance = _dead_run_dir(locks, temp_dir, f"batchgen_lane-1_{'b' * 32}")
    current = temp_dir / identity.resource_prefix
    current.mkdir()

    dead_shm = shm_dir / f"{dead.name}.host_kv"
    other_shm = shm_dir / f"{other_instance.name}.host_kv"
    unrelated_shm = shm_dir / "sem.mp-abcdef"
    for entry in (dead_shm, other_shm, unrelated_shm):
        entry.touch()

    with caplog.at_level(logging.WARNING):
        reclaimed = locks.reclaim_dead_runs(
            identity, temp_dir=temp_dir, shm_dir=shm_dir
        )

    assert reclaimed == [dead_id]
    assert not dead.exists()
    assert not dead_shm.exists()
    # Another instance's run and anything outside this run's prefix is untouched.
    assert other_instance.is_dir()
    assert other_shm.exists()
    assert unrelated_shm.exists()
    assert current.is_dir()

    messages = [record.getMessage() for record in caplog.records]
    reclaim_messages = [m for m in messages if "Reclaimed dead run" in m]
    assert len(reclaim_messages) == 1
    assert dead_id in reclaim_messages[0]
    assert str(dead_shm) in reclaim_messages[0]
    assert str(dead) in reclaim_messages[0]


def test_reclaim_refuses_startup_while_a_run_process_is_alive(tmp_path):
    identity_module, locks = _load_runtime_modules()
    identity = _identity(identity_module)
    temp_dir = tmp_path / "tmp"
    shm_dir = tmp_path / "shm"
    temp_dir.mkdir()
    shm_dir.mkdir()
    live = _dead_run_dir(locks, temp_dir, f"batchgen_lane-0_{'c' * 32}")
    lock_path = live / locks.RUN_LOCK_NAME

    holder = subprocess.Popen(
        [sys.executable, "-c", _LOCK_HOLDER, str(lock_path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(RuntimeError, match="refusing startup") as excinfo:
            locks.reclaim_dead_runs(identity, temp_dir=temp_dir, shm_dir=shm_dir)
        assert str(lock_path) in str(excinfo.value)
        assert live.is_dir()
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()


def test_reclaim_leaves_a_leftover_without_a_run_lock_untouched(tmp_path, caplog):
    identity_module, locks = _load_runtime_modules()
    identity = _identity(identity_module)
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    pre_upgrade = temp_dir / f"batchgen_lane-0_{'d' * 32}"
    pre_upgrade.mkdir()
    (pre_upgrade / "reload_status").mkdir()

    with caplog.at_level(logging.WARNING):
        assert locks.reclaim_dead_runs(identity, temp_dir=temp_dir) == []

    assert (pre_upgrade / "reload_status").is_dir()
    messages = [record.getMessage() for record in caplog.records]
    assert [m for m in messages if str(pre_upgrade) in m and "untouched" in m]


def test_reclaim_skips_a_symlinked_leftover(tmp_path, caplog):
    identity_module, locks = _load_runtime_modules()
    identity = _identity(identity_module)
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    target = _dead_run_dir(locks, tmp_path, "real-run")
    link = temp_dir / f"batchgen_lane-0_{'e' * 32}"
    link.symlink_to(target, target_is_directory=True)

    with caplog.at_level(logging.WARNING):
        assert locks.reclaim_dead_runs(identity, temp_dir=temp_dir) == []

    assert link.is_symlink()
    assert (target / locks.RUN_LOCK_NAME).exists()
    messages = [record.getMessage() for record in caplog.records]
    assert [m for m in messages if str(link) in m and "not a real directory" in m]


def test_reclaim_skips_a_leftover_owned_by_another_user(
    tmp_path, caplog, monkeypatch
):
    identity_module, locks = _load_runtime_modules()
    identity = _identity(identity_module)
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    foreign = _dead_run_dir(locks, temp_dir, f"batchgen_lane-0_{'1' * 32}")
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(foreign).st_uid + 1)

    with caplog.at_level(logging.WARNING):
        assert locks.reclaim_dead_runs(identity, temp_dir=temp_dir) == []

    assert foreign.is_dir()
    messages = [record.getMessage() for record in caplog.records]
    assert [m for m in messages if str(foreign) in m and "another user" in m]


# ---------------------------------------------------------------------------
# Server and worker both join the lock
# ---------------------------------------------------------------------------
def test_server_holds_the_run_lock_shared_from_prepare_runtime_dir(tmp_path):
    _identity_module, locks = _load_runtime_modules()
    resource_prefix = "batchgen_lane-0_" + "0" * 32
    runtime_dir = tmp_path / resource_prefix
    shm_dir = tmp_path / "shm"
    shm_dir.mkdir()

    manager_type = _worker_manager_method(
        "_prepare_runtime_dir",
        {
            "Path": lambda value: shm_dir if value == "/dev/shm" else Path(value),
            "hold_run_lock": locks.hold_run_lock,
        },
    )
    manager = manager_type()
    manager.args = SimpleNamespace(
        runtime_identity=SimpleNamespace(
            shm_prefix=f"{resource_prefix}.",
            runtime_dir=runtime_dir,
        )
    )
    manager._runtime_dir_created = False
    manager._runtime_namespace_owned = False
    manager._run_lock = None

    manager._prepare_runtime_dir()

    lock_path = runtime_dir / locks.RUN_LOCK_NAME
    assert manager._run_lock is not None
    assert lock_path.is_file()
    assert not _exclusive_lock_is_free(lock_path)

    locks.release_run_lock(manager._run_lock)
    assert _exclusive_lock_is_free(lock_path)


def test_worker_joins_the_run_lock_right_after_die_with_parent():
    source = WORKER_MAIN_LOOP.read_text()

    assert "from batchgen.server.runtime_locks import hold_run_lock" in source
    # The worker learns its runtime dir from the reload-status dir it is given.
    assert (
        "_run_lock_file = hold_run_lock(os.path.dirname(args.reload_status_dir))"
        in source
    )
    assert "\tglobal _run_lock_file" in source
    assert source.index("die_with_parent()") < source.index("hold_run_lock(os.path")
    # Only the server creates the file, and nothing ever closes it: the kernel
    # drops this worker's share when the process dies, however it dies.
    assert "create=True" not in source
    assert "_run_lock_file.close" not in source
    assert "release_run_lock" not in source

    module = ast.parse(source, filename=str(WORKER_MAIN_LOOP))
    module_level = {
        target.id
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "_run_lock_file" in module_level
