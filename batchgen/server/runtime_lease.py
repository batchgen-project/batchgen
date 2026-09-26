"""Validation and lifetime ownership for launcher-transferred lane leases."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


LANE_LOCK_ROOT = Path("/tmp/batchgen-lane-leases")
_MAX_MANIFEST_BYTES = 64 * 1024
_CACHE_ENV_PATHS = {
    "temp": "TMPDIR",
    "torch_extensions": "TORCH_EXTENSIONS_DIR",
    "triton": "TRITON_CACHE_DIR",
    "torchinductor": "TORCHINDUCTOR_CACHE_DIR",
    "cuda": "CUDA_CACHE_PATH",
}


class LaneLeaseError(RuntimeError):
    """Raised when a shared runtime's inherited lease bundle is invalid."""


def _canonical_path(value: os.PathLike[str] | str) -> str:
    return str(Path(value).expanduser().resolve())


def _resource_filename(resource: str) -> str:
    digest = hashlib.sha256(resource.encode("utf-8")).hexdigest()
    return f"{digest}.lock"


def _expected_paths(args: Any) -> dict[str, str]:
    paths = {
        "storage": _canonical_path(args.storage_path),
        "converted_checkpoint": _canonical_path(args.converted_ckpt_dir),
    }
    for name, env_name in _CACHE_ENV_PATHS.items():
        value = os.environ.get(env_name)
        if not value:
            raise LaneLeaseError(
                f"shared runtime requires explicit {env_name}"
            )
        paths[name] = _canonical_path(value)
    return paths


def _expected_resources(
    *,
    gpu_uuids: list[str],
    listen_port: int,
    dist_port: int,
    pynccl_port_base: int,
    pynccl_port_span: int,
    world_size: int,
    paths: dict[str, str],
) -> set[str]:
    resources = {f"gpu:{uuid}" for uuid in gpu_uuids}
    resources.update({f"port:{listen_port}", f"port:{dist_port}"})
    if world_size > 1:
        resources.add(f"pynccl:{pynccl_port_base}:{pynccl_port_span}")
    resources.update(
        f"path:{hashlib.sha256(path.encode('utf-8')).hexdigest()}"
        for path in paths.values()
    )
    return resources


class LaneLease:
    """Own inherited GPU, port, and writable-path lock descriptors."""

    def __init__(self, manifest_fd: int, resource_fds: list[int]) -> None:
        self._manifest_fd = manifest_fd
        self._resource_fds = resource_fds

    @classmethod
    def acquire(
        cls,
        args: Any,
        *,
        lock_root: Path = LANE_LOCK_ROOT,
    ) -> "LaneLease":
        manifest_fd = args.lane_lease_manifest_fd
        if manifest_fd is None:
            raise LaneLeaseError(
                "shared runtime requires --lane-lease-manifest-fd"
            )
        resource_fds: list[int] = []
        try:
            manifest = cls._read_manifest(manifest_fd)
            resource_fds = cls._validate_manifest(
                args,
                manifest,
                manifest_fd=manifest_fd,
                lock_root=lock_root,
            )
            try:
                for fd in resource_fds:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    os.set_inheritable(fd, False)
                os.set_inheritable(manifest_fd, False)
            except BaseException:
                for fd in reversed(resource_fds):
                    cls._close_fd(fd)
                raise
            return cls(manifest_fd, resource_fds)
        except BaseException:
            cls._close_fd(manifest_fd)
            raise

    @staticmethod
    def _read_manifest(manifest_fd: int) -> dict[str, Any]:
        try:
            metadata = os.fstat(manifest_fd)
        except OSError as exc:
            raise LaneLeaseError("lane lease manifest FD is not open") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise LaneLeaseError("lane lease manifest FD must be a regular file")
        if metadata.st_uid != os.geteuid():
            raise LaneLeaseError("lane lease manifest is not owned by this user")
        if metadata.st_size <= 0 or metadata.st_size > _MAX_MANIFEST_BYTES:
            raise LaneLeaseError("lane lease manifest has an invalid size")
        try:
            payload = os.pread(manifest_fd, metadata.st_size, 0)
            manifest = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LaneLeaseError("lane lease manifest is not valid JSON") from exc
        if not isinstance(manifest, dict):
            raise LaneLeaseError("lane lease manifest must be a JSON object")
        return manifest

    @classmethod
    def _validate_manifest(
        cls,
        args: Any,
        manifest: dict[str, Any],
        *,
        manifest_fd: int,
        lock_root: Path,
    ) -> list[int]:
        if manifest.get("version") != 1:
            raise LaneLeaseError("unsupported lane lease manifest version")
        if manifest.get("instance_id") != args.instance_id:
            raise LaneLeaseError("lane lease instance_id does not match server")

        gpu_uuids = manifest.get("gpu_uuids")
        if not isinstance(gpu_uuids, list) or not gpu_uuids:
            raise LaneLeaseError("lane lease gpu_uuids must be a non-empty list")
        if any(
            not isinstance(uuid, str) or not uuid.startswith("GPU-")
            for uuid in gpu_uuids
        ):
            raise LaneLeaseError("lane lease contains an invalid GPU UUID")
        visible = tuple(
            item.strip()
            for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if item.strip()
        )
        if tuple(gpu_uuids) != visible:
            raise LaneLeaseError(
                "lane lease GPU UUID order must equal CUDA_VISIBLE_DEVICES"
            )
        if len(gpu_uuids) != args.world_size:
            raise LaneLeaseError("lane lease GPU count must equal world_size")

        _, dist_port = args.dist_init_addr.rsplit(":", 1)
        expected_scalars = {
            "listen_port": args.listen_port,
            "dist_init_port": int(dist_port),
            "pynccl_port_base": args.pynccl_port_base,
            "pynccl_port_span": args.pynccl_port_span,
        }
        for name, expected in expected_scalars.items():
            if manifest.get(name) != expected:
                raise LaneLeaseError(f"lane lease {name} does not match server")

        paths = _expected_paths(args)
        if manifest.get("paths") != paths:
            raise LaneLeaseError("lane lease writable paths do not match server")
        expected_resources = _expected_resources(
            gpu_uuids=gpu_uuids,
            listen_port=args.listen_port,
            dist_port=int(dist_port),
            pynccl_port_base=args.pynccl_port_base,
            pynccl_port_span=args.pynccl_port_span,
            world_size=args.world_size,
            paths=paths,
        )
        resources = manifest.get("resources")
        if not isinstance(resources, dict) or set(resources) != expected_resources:
            raise LaneLeaseError("lane lease resource set does not match server")

        if lock_root.is_symlink() or not lock_root.is_dir():
            raise LaneLeaseError("lane lease lock root is not a directory")
        root_metadata = lock_root.stat()
        if root_metadata.st_uid != os.geteuid() or root_metadata.st_mode & 0o077:
            raise LaneLeaseError("lane lease lock root ownership or mode is unsafe")
        resolved_root = lock_root.resolve()
        resource_fds: list[int] = []
        for resource in sorted(expected_resources):
            fd = resources[resource]
            if not isinstance(fd, int) or fd < 0 or fd == manifest_fd:
                raise LaneLeaseError(f"invalid descriptor for {resource}")
            if fd in resource_fds:
                raise LaneLeaseError("lane lease resource descriptors must be unique")
            try:
                metadata = os.fstat(fd)
                target = Path(os.readlink(f"/proc/self/fd/{fd}")).resolve()
            except OSError as exc:
                raise LaneLeaseError(
                    f"lane lease descriptor for {resource} is not open"
                ) from exc
            expected_target = resolved_root / _resource_filename(resource)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.geteuid()
                or target != expected_target
            ):
                raise LaneLeaseError(
                    f"lane lease descriptor target does not match {resource}"
                )
            resource_fds.append(fd)
        return resource_fds

    @staticmethod
    def _close_fd(fd: int) -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

    def close(self) -> None:
        for fd in reversed(self._resource_fds):
            self._close_fd(fd)
        self._resource_fds.clear()
        self._close_fd(self._manifest_fd)
        self._manifest_fd = -1
