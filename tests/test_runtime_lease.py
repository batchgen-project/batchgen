"""Shared runtimes must hold an exact launcher-transferred lease bundle."""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
_CACHE_ENV = {
    "TMPDIR": "tmp",
    "TORCH_EXTENSIONS_DIR": "torch-extensions",
    "TRITON_CACHE_DIR": "triton",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor",
    "CUDA_CACHE_PATH": "cuda",
}


def _load_runtime_lease_module():
    package_name = "batchgen.server"
    module_name = "batchgen.server.runtime_lease"
    previous_package = sys.modules.get(package_name)
    previous_module = sys.modules.pop(module_name, None)
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "batchgen" / "server")]
    sys.modules[package_name] = package
    try:
        return importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if previous_module is not None:
            sys.modules[module_name] = previous_module
        if previous_package is None:
            sys.modules.pop(package_name, None)
        else:
            sys.modules[package_name] = previous_package


def _args(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    for env_name, dirname in _CACHE_ENV.items():
        monkeypatch.setenv(env_name, str(tmp_path / dirname))
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        "GPU-00000000-0000-0000-0000-000000000000",
    )
    return SimpleNamespace(
        lane_lease_manifest_fd=None,
        instance_id="lane-0",
        world_size=1,
        listen_port=11000,
        dist_init_addr="localhost:12000",
        pynccl_port_base=21000,
        pynccl_port_span=4,
        storage_path=tmp_path / "storage",
        converted_ckpt_dir=tmp_path / "converted",
    )


def _open_bundle(runtime_lease, args, lock_root: Path):
    lock_root.mkdir(mode=0o700, exist_ok=True)
    paths = runtime_lease._expected_paths(args)
    gpu_uuids = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    resources = runtime_lease._expected_resources(
        gpu_uuids=gpu_uuids,
        listen_port=args.listen_port,
        dist_port=12000,
        pynccl_port_base=args.pynccl_port_base,
        pynccl_port_span=args.pynccl_port_span,
        world_size=args.world_size,
        paths=paths,
    )
    resource_fds = {}
    for resource in resources:
        path = lock_root / runtime_lease._resource_filename(resource)
        resource_fds[resource] = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    manifest = {
        "version": 1,
        "instance_id": args.instance_id,
        "gpu_uuids": gpu_uuids,
        "listen_port": args.listen_port,
        "dist_init_port": 12000,
        "pynccl_port_base": args.pynccl_port_base,
        "pynccl_port_span": args.pynccl_port_span,
        "paths": paths,
        "resources": resource_fds,
    }
    manifest_count = len(list(lock_root.parent.glob("manifest-*.json")))
    manifest_path = lock_root.parent / f"manifest-{manifest_count}.json"
    manifest_path.write_text(json.dumps(manifest))
    args.lane_lease_manifest_fd = os.open(manifest_path, os.O_RDONLY)
    return resource_fds


def test_lane_lease_holds_exact_resources_until_close(tmp_path, monkeypatch):
    runtime_lease = _load_runtime_lease_module()
    args = _args(tmp_path, monkeypatch)
    lock_root = tmp_path / "locks"
    resource_fds = _open_bundle(runtime_lease, args, lock_root)

    lease = runtime_lease.LaneLease.acquire(args, lock_root=lock_root)
    assert not os.get_inheritable(args.lane_lease_manifest_fd)
    assert all(not os.get_inheritable(fd) for fd in resource_fds.values())
    lease.close()

    with pytest.raises(OSError):
        os.fstat(args.lane_lease_manifest_fd)
    for fd in resource_fds.values():
        with pytest.raises(OSError):
            os.fstat(fd)


def test_lane_lease_rejects_second_owner(tmp_path, monkeypatch):
    runtime_lease = _load_runtime_lease_module()
    first_args = _args(tmp_path, monkeypatch)
    lock_root = tmp_path / "locks"
    _open_bundle(runtime_lease, first_args, lock_root)
    first = runtime_lease.LaneLease.acquire(first_args, lock_root=lock_root)

    second_args = _args(tmp_path, monkeypatch)
    _open_bundle(runtime_lease, second_args, lock_root)
    try:
        with pytest.raises(BlockingIOError):
            runtime_lease.LaneLease.acquire(
                second_args,
                lock_root=lock_root,
            )
    finally:
        first.close()


def test_lane_lease_rejects_visible_gpu_mismatch(tmp_path, monkeypatch):
    runtime_lease = _load_runtime_lease_module()
    args = _args(tmp_path, monkeypatch)
    lock_root = tmp_path / "locks"
    resource_fds = _open_bundle(runtime_lease, args, lock_root)
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES",
        "GPU-11111111-1111-1111-1111-111111111111",
    )

    with pytest.raises(runtime_lease.LaneLeaseError, match="GPU UUID order"):
        runtime_lease.LaneLease.acquire(args, lock_root=lock_root)
    for fd in resource_fds.values():
        os.close(fd)
