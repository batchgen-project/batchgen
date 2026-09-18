"""Runtime admission locks must reject conflicting server lifetimes."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_runtime_modules():
    package_name = "batchgen.server"
    module_names = (
        "batchgen.server.runtime_identity",
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
        locks = importlib.import_module(module_names[1])
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


def _identity(identity_type, instance_id: str, mode: str):
    return identity_type(
        mode=mode,
        instance_id=instance_id,
        run_id=("0" if instance_id == "lane-0" else "1") * 32,
    )


def test_shared_host_locks_coexist_for_distinct_instances(tmp_path: Path):
    identity, locks = _load_runtime_modules()
    first = locks.RuntimeLocks.acquire(
        _identity(identity.RuntimeIdentity, "lane-0", "shared"),
        lock_root=tmp_path,
    )
    second = locks.RuntimeLocks.acquire(
        _identity(identity.RuntimeIdentity, "lane-1", "shared"),
        lock_root=tmp_path,
    )
    second.close()
    first.close()


def test_exclusive_runtime_conflicts_with_shared_runtime(tmp_path: Path):
    identity, locks = _load_runtime_modules()
    shared = locks.RuntimeLocks.acquire(
        _identity(identity.RuntimeIdentity, "lane-0", "shared"),
        lock_root=tmp_path,
    )
    try:
        with pytest.raises(locks.RuntimeLockError, match="host runtime mode"):
            locks.RuntimeLocks.acquire(
                _identity(identity.RuntimeIdentity, "lane-1", "exclusive"),
                lock_root=tmp_path,
            )
    finally:
        shared.close()


def test_duplicate_instance_conflicts_without_stale_takeover(tmp_path: Path):
    identity, locks = _load_runtime_modules()
    first = locks.RuntimeLocks.acquire(
        _identity(identity.RuntimeIdentity, "lane-0", "shared"),
        lock_root=tmp_path,
    )
    try:
        with pytest.raises(locks.RuntimeLockError, match="instance 'lane-0'"):
            locks.RuntimeLocks.acquire(
                _identity(identity.RuntimeIdentity, "lane-0", "shared"),
                lock_root=tmp_path,
            )
    finally:
        first.close()

    replacement = locks.RuntimeLocks.acquire(
        _identity(identity.RuntimeIdentity, "lane-0", "shared"),
        lock_root=tmp_path,
    )
    replacement.close()
