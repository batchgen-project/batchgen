#!/usr/bin/env python3
"""Fail-closed launcher and lifecycle tool for co-resident BatchGen lanes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


HOST_LOCK_ROOT = Path("/tmp/batchgen-runtime-locks")
LANE_LOCK_ROOT = Path("/tmp/batchgen-lane-leases")
STATE_ROOT = Path("/tmp/batchgen-lanes")
_CACHE_DIRS = {
    "temp": "tmp",
    "torch_extensions": "torch-extensions",
    "triton": "triton-cache",
    "torchinductor": "torchinductor-cache",
    "cuda": "cuda-cache",
}
_GIB = 1024**3
_GPTOSS_WEIGHT_SHM_GIB = 70
_SHM_TRANSIENT_RESERVE_GIB = 16
_HOST_PRIVATE_RESERVE_GIB = 64
_MIN_SAFETY_RESERVE_GIB = 64
_INSTANCE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,47}\Z")
_O200K_BASE_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
_O200K_BASE_CACHE_KEY = "fb374d419588a4632f3f557e76b4b70aebbca790"


class LaneError(RuntimeError):
    """Raised when lane ownership or admission cannot be proven."""


def _ensure_private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.stat()
    if path.is_symlink() or not path.is_dir():
        raise LaneError(f"not a real directory: {path}")
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise LaneError(f"directory ownership or mode is unsafe: {path}")
    return path.resolve()


def _open_lock(path: Path, operation: int) -> int:
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        fcntl.flock(fd, operation | fcntl.LOCK_NB)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _close_fd(fd: int | None) -> None:
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        os.close(fd)
    except OSError:
        pass


def _close_inherited_parent_fd(fd: int | None) -> None:
    """Drop the parent's duplicate without unlocking the child's OFD lock."""
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


def _resource_filename(resource: str) -> str:
    return hashlib.sha256(resource.encode("utf-8")).hexdigest() + ".lock"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_private_dir(path.parent)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temporary, flags, 0o600)
    try:
        data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LaneError(f"cannot read lane manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LaneError(f"lane manifest is not an object: {path}")
    return payload


def _boot_id(proc_root: Path = Path("/proc")) -> str:
    return (proc_root / "sys/kernel/random/boot_id").read_text().strip()


def _proc_start_time(pid: int, proc_root: Path = Path("/proc")) -> int:
    value = (proc_root / str(pid) / "stat").read_text()
    end = value.rfind(")")
    if end < 0:
        raise LaneError(f"malformed /proc/{pid}/stat")
    fields = value[end + 2 :].split()
    return int(fields[19])


def _pid_process_identity_matches(
    manifest: dict[str, Any], proc_root: Path = Path("/proc")
) -> bool:
    pid = manifest.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        return (
            manifest.get("boot_id") == _boot_id(proc_root)
            and manifest.get("pid_start_time")
            == _proc_start_time(pid, proc_root)
            and manifest.get("process_group") == os.getpgid(pid)
        )
    except (
        FileNotFoundError,
        ProcessLookupError,
        PermissionError,
        LaneError,
        ValueError,
        UnicodeError,
    ):
        return False


def _pid_identity_matches(
    manifest: dict[str, Any], proc_root: Path = Path("/proc")
) -> bool:
    command = manifest.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(arg, str) or "\0" in arg for arg in command)
        or not _pid_process_identity_matches(manifest, proc_root)
    ):
        return False
    try:
        return (proc_root / str(manifest["pid"]) / "cmdline").read_bytes() == (
            b"\0".join(os.fsencode(arg) for arg in command) + b"\0"
        )
    except (FileNotFoundError, PermissionError, UnicodeError):
        return False


def _require_pidfd_signaling() -> None:
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise LaneError("lane signaling requires Linux pidfd support")


def _open_verified_owner_pidfd(manifest: dict[str, Any]) -> int:
    _require_pidfd_signaling()
    try:
        pidfd = os.pidfd_open(manifest["pid"])
    except OSError as exc:
        raise LaneError("cannot open lane owner pidfd") from exc
    if not _pid_identity_matches(manifest):
        os.close(pidfd)
        raise LaneError("owner identity changed before lane signal")
    return pidfd


def _signal_pidfd(pidfd: int, sig: signal.Signals) -> None:
    try:
        signal.pidfd_send_signal(pidfd, sig)
    except ProcessLookupError:
        pass


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _parse_meminfo(path: Path = Path("/proc/meminfo")) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        name, raw = line.split(":", 1)
        values[name] = int(raw.strip().split()[0]) * 1024
    return values


def _nonreclaimable_bytes(meminfo: dict[str, int]) -> int:
    reclaimable = (
        meminfo["Buffers"]
        + meminfo["Cached"]
        - meminfo["Shmem"]
        + meminfo["SReclaimable"]
    )
    return meminfo["MemTotal"] - meminfo["MemFree"] - reclaimable


def _state_manifests(state_root: Path) -> Iterable[tuple[Path, dict[str, Any]]]:
    if not state_root.is_dir():
        return ()
    manifests = []
    for path in sorted(state_root.glob("*.json")):
        if path.name.endswith(".lease.json"):
            continue
        manifests.append((path, _read_json(path)))
    return manifests


def _active_manifests(state_root: Path) -> list[dict[str, Any]]:
    active = []
    for path, manifest in _state_manifests(state_root):
        instance_id = manifest.get("instance_id")
        if (
            not isinstance(instance_id, str)
            or not _INSTANCE_ID_RE.fullmatch(instance_id)
            or path.name != f"{instance_id}.json"
        ):
            raise LaneError(f"invalid lane manifest identity: {path}")
        if manifest.get("state") == "stopped":
            if _pid_process_identity_matches(manifest):
                raise LaneError(f"stopped lane manifest has a live owner: {path}")
            continue
        if not _pid_identity_matches(manifest):
            raise LaneError(
                f"ambiguous stale lane manifest blocks admission: {path}"
            )
        _validate_active_manifest(manifest, path)
        active.append(manifest)
    return active


def _validate_active_manifest(manifest: dict[str, Any], path: Path) -> None:
    gpus = manifest.get("gpu_uuids")
    ports = (
        manifest.get("listen_port"),
        manifest.get("dist_init_port"),
        manifest.get("pynccl_port_base"),
        manifest.get("pynccl_port_span"),
    )
    paths = manifest.get("paths")
    lane_root = manifest.get("lane_root")
    budgets = (
        manifest.get("host_memory_reservation_bytes"),
        manifest.get("shm_reservation_bytes"),
        manifest.get("safety_reserve_bytes"),
    )
    if (
        manifest.get("version") != 1
        or manifest.get("state") not in {"starting", "admitted", "stopping", "failed"}
        or not isinstance(gpus, list)
        or len(gpus) not in {1, 2, 4, 8}
        or any(
            not isinstance(gpu, str) or not gpu.startswith("GPU-")
            for gpu in gpus
        )
        or len(set(gpus)) != len(gpus)
        or any(type(port) is not int or port <= 0 for port in ports)
        or any(type(budget) is not int or budget <= 0 for budget in budgets)
        or budgets[2] < _MIN_SAFETY_RESERVE_GIB * _GIB
        or not isinstance(lane_root, str)
        or not Path(lane_root).is_absolute()
        or not isinstance(paths, dict)
        or set(paths) != {"storage", "converted_checkpoint", *_CACHE_DIRS}
        or any(
            not isinstance(value, str) or not Path(value).is_absolute()
            for value in paths.values()
        )
    ):
        raise LaneError(f"invalid active lane manifest resources: {path}")
    if (
        ports[0] > 65535
        or ports[1] > 65535
        or ports[2] + ports[3] > 65536
        or not Path(paths["converted_checkpoint"]).is_relative_to(lane_root)
        or paths != _canonical_paths(
            Path(lane_root), Path(paths["converted_checkpoint"])
        )
    ):
        raise LaneError(f"invalid active lane manifest resources: {path}")


def _ranges_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return max(first[0], second[0]) < min(first[1], second[1])


def _paths_overlap(first: str, second: str) -> bool:
    first_path = Path(first)
    second_path = Path(second)
    return (
        first_path == second_path
        or first_path in second_path.parents
        or second_path in first_path.parents
    )


def _check_lane_root_available(lane_root: Path, active: Iterable[dict[str, Any]]) -> None:
    root = str(lane_root.resolve())
    for manifest in active:
        other = manifest.get("lane_root")
        if not isinstance(other, str) or not Path(other).is_absolute():
            raise LaneError("active lane manifest has no valid lane root")
        if _paths_overlap(root, other):
            raise LaneError("lane root overlaps an active lane")


def _check_no_overlap(candidate: dict[str, Any], active: Iterable[dict[str, Any]]) -> None:
    candidate_gpus = set(candidate["gpu_uuids"])
    candidate_ports = {candidate["listen_port"], candidate["dist_init_port"]}
    candidate_range = (
        candidate["pynccl_port_base"],
        candidate["pynccl_port_base"] + candidate["pynccl_port_span"],
    )
    candidate_paths = list(candidate["paths"].values())
    for index, path in enumerate(candidate_paths):
        remaining_paths = candidate_paths[index + 1 :]
        if any(_paths_overlap(path, other) for other in remaining_paths):
            raise LaneError("lane writable paths overlap each other")
    for manifest in active:
        if manifest.get("instance_id") == candidate["instance_id"]:
            raise LaneError("logical instance is already active")
        if candidate_gpus & set(manifest.get("gpu_uuids", ())):
            raise LaneError("GPU UUID allocation overlaps an active lane")
        other_ports = {
            manifest.get("listen_port"),
            manifest.get("dist_init_port"),
        }
        if candidate_ports & other_ports:
            raise LaneError("TCP port allocation overlaps an active lane")
        other_range = (
            manifest.get("pynccl_port_base", 0),
            manifest.get("pynccl_port_base", 0)
            + manifest.get("pynccl_port_span", 0),
        )
        if (
            _ranges_overlap(candidate_range, other_range)
            or any(port in range(*other_range) for port in candidate_ports)
            or any(port in range(*candidate_range) for port in other_ports)
        ):
            raise LaneError("PyNccl range overlaps an active lane port or range")
        other_paths = manifest.get("paths", {}).values()
        if any(
            _paths_overlap(path, other)
            for path in candidate_paths
            for other in other_paths
        ):
            raise LaneError("writable path allocation overlaps an active lane")


def _check_memory(
    candidate: dict[str, Any],
    active: Iterable[dict[str, Any]],
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    shm_path: Path = Path("/dev/shm"),
) -> None:
    active = list(active)
    meminfo = _parse_meminfo(meminfo_path)
    safety_reserve = max(
        candidate["safety_reserve_bytes"],
        *(item["safety_reserve_bytes"] for item in active),
    )
    memory_limit = meminfo["MemTotal"] - safety_reserve
    reserved = sum(item.get("host_memory_reservation_bytes", 0) for item in active)
    requested = candidate["host_memory_reservation_bytes"]
    if reserved + requested > memory_limit:
        raise LaneError("host-memory reservations exceed the admission limit")
    if _nonreclaimable_bytes(meminfo) + requested > memory_limit:
        raise LaneError("current host use plus the lane reservation is unsafe")

    shm = os.statvfs(shm_path)
    shm_free = shm.f_bavail * shm.f_frsize
    shm_total = shm.f_blocks * shm.f_frsize
    shm_reserved = sum(item.get("shm_reservation_bytes", 0) for item in active)
    if shm_reserved + candidate["shm_reservation_bytes"] > shm_total:
        raise LaneError("lane SHM reservations exceed /dev/shm capacity")
    if candidate["shm_reservation_bytes"] > shm_free:
        raise LaneError("lane SHM reservation exceeds current /dev/shm free space")


def _check_ports_free(ports: Iterable[int]) -> None:
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("", port))
            except OSError as exc:
                raise LaneError(f"TCP port {port} is not available") from exc


def _gpu_processes() -> list[tuple[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid = (part.strip() for part in line.split(",", 1))
        processes.append((uuid, int(pid)))
    return processes


def _gpu_inventory() -> set[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _check_gpus_free(gpu_uuids: Iterable[str]) -> None:
    requested = set(gpu_uuids)
    missing = requested - _gpu_inventory()
    if missing:
        raise LaneError(f"assigned GPU UUIDs are not present: {sorted(missing)}")
    conflicts = [(uuid, pid) for uuid, pid in _gpu_processes() if uuid in requested]
    if conflicts:
        raise LaneError(f"assigned GPU UUIDs have live compute processes: {conflicts}")


def _canonical_paths(lane_root: Path, converted: Path) -> dict[str, str]:
    paths = {
        "storage": str((lane_root / "storage").resolve()),
        "converted_checkpoint": str(converted.resolve()),
    }
    paths.update(
        {name: str((lane_root / dirname).resolve()) for name, dirname in _CACHE_DIRS.items()}
    )
    return paths


def _prepare_paths(
    lane_root: Path,
    worktree: Path,
    converted: Path,
) -> tuple[Path, dict[str, str], Path]:
    worktree = worktree.resolve(strict=True)
    if _paths_overlap(str(lane_root.resolve()), str(worktree)):
        raise LaneError("lane root overlaps worktree")
    lane_root = _ensure_private_dir(lane_root)
    converted = converted.resolve()
    if not converted.is_relative_to(lane_root):
        raise LaneError("converted checkpoint path must be inside lane root")
    for package in ("batchgen", "batchgen_kernels"):
        if not (worktree / package).is_dir():
            raise LaneError(f"worktree is missing {package}: {worktree}")
    paths = _canonical_paths(lane_root, converted)
    for value in paths.values():
        _ensure_private_dir(Path(value))
    log_dir = _ensure_private_dir(lane_root / "logs")
    pyroot = _ensure_private_dir(lane_root / "pyroot")
    for package in ("batchgen", "batchgen_kernels"):
        link = pyroot / package
        target = worktree / package
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise LaneError(f"pyroot package points outside this worktree: {link}")
        elif link.exists():
            raise LaneError(f"pyroot package is not a symlink: {link}")
        else:
            link.symlink_to(target, target_is_directory=True)
    return pyroot, paths, log_dir / "server.log"


def _seed_o200k_base(temp_dir: Path, source: Path) -> Path:
    data = source.read_bytes()
    if hashlib.sha256(data).hexdigest() != _O200K_BASE_SHA256:
        raise LaneError(f"o200k_base encoding checksum mismatch: {source}")
    cache_dir = _ensure_private_dir(temp_dir / "data-gym-cache")
    target = cache_dir / _O200K_BASE_CACHE_KEY
    if target.is_symlink():
        raise LaneError(f"o200k_base cache is a symlink: {target}")
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != _O200K_BASE_SHA256:
            raise LaneError(f"o200k_base cache checksum mismatch: {target}")
    else:
        with target.open("xb") as output:
            output.write(data)
    return cache_dir


def _candidate(args: argparse.Namespace, paths: dict[str, str]) -> dict[str, Any]:
    return {
        "version": 1,
        "instance_id": args.instance_id,
        "model": args.model,
        "gpu_uuids": args.gpu_uuid,
        "listen_port": args.listen_port,
        "dist_init_port": args.dist_port,
        "pynccl_port_base": args.pynccl_port_base,
        "pynccl_port_span": args.pynccl_port_span,
        "paths": paths,
        "host_memory_reservation_bytes": int(args.host_memory_gb * _GIB),
        "shm_reservation_bytes": int(args.shm_gb * _GIB),
        "safety_reserve_bytes": int(args.safety_gb * _GIB),
    }


def _validate_instance_id(instance_id: str) -> None:
    if not _INSTANCE_ID_RE.fullmatch(instance_id):
        raise LaneError("instance_id must match [a-z0-9][a-z0-9_-]{0,47}")


def _validate_start_args(args: argparse.Namespace) -> None:
    _validate_instance_id(args.instance_id)
    if args.model != "openai/gpt-oss-120b":
        raise LaneError("shared lanes are qualified only for openai/gpt-oss-120b")
    if len(args.gpu_uuid) not in {1, 2, 4, 8}:
        raise LaneError("lane GPU count must be one of 1, 2, 4, or 8")
    if len(set(args.gpu_uuid)) != len(args.gpu_uuid) or any(
        not uuid.startswith("GPU-") for uuid in args.gpu_uuid
    ):
        raise LaneError("lane GPU UUIDs must be unique physical UUIDs")
    if any(
        value <= 0
        for value in (
            args.host_kv_cache_gb,
            args.host_memory_gb,
            args.shm_gb,
            args.safety_gb,
            args.pynccl_port_span,
        )
    ):
        raise LaneError("memory budgets and PyNccl span must be positive")
    if args.safety_gb < _MIN_SAFETY_RESERVE_GIB:
        raise LaneError("host safety reserve must be at least 64 GiB")
    min_shm_gb = (
        args.host_kv_cache_gb
        + _GPTOSS_WEIGHT_SHM_GIB
        + _SHM_TRANSIENT_RESERVE_GIB
    )
    if args.shm_gb < min_shm_gb:
        raise LaneError("SHM reservation understates GPT-OSS and Host-KV needs")
    if args.host_memory_gb < args.shm_gb + _HOST_PRIVATE_RESERVE_GIB:
        raise LaneError("host-memory reservation omits private runtime headroom")
    ports = (args.listen_port, args.dist_port, args.pynccl_port_base)
    if any(port <= 0 or port > 65535 for port in ports):
        raise LaneError("lane ports must be in [1, 65535]")
    if args.pynccl_port_base + args.pynccl_port_span > 65536:
        raise LaneError("PyNccl range exceeds the TCP port space")
    pynccl = range(
        args.pynccl_port_base,
        args.pynccl_port_base + args.pynccl_port_span,
    )
    if args.listen_port == args.dist_port or (
        len(args.gpu_uuid) > 1
        and (args.listen_port in pynccl or args.dist_port in pynccl)
    ):
        raise LaneError("lane communication port allocations overlap")
    if not args.checkpoint.is_dir():
        raise LaneError(f"checkpoint directory does not exist: {args.checkpoint}")


def _resource_names(candidate: dict[str, Any]) -> set[str]:
    resources = {f"gpu:{uuid}" for uuid in candidate["gpu_uuids"]}
    resources.update(
        {
            f"port:{candidate['listen_port']}",
            f"port:{candidate['dist_init_port']}",
        }
    )
    if len(candidate["gpu_uuids"]) > 1:
        resources.add(
            "pynccl:"
            f"{candidate['pynccl_port_base']}:{candidate['pynccl_port_span']}"
        )
    resources.update(
        f"path:{hashlib.sha256(path.encode('utf-8')).hexdigest()}"
        for path in candidate["paths"].values()
    )
    return resources


def _server_command(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    manifest_fd: int,
) -> list[str]:
    command = [
        args.python,
        "-m",
        "batchgen.launch_http_server",
        "--model",
        args.model,
        "--instance-id",
        args.instance_id,
        "--runtime-mode",
        "shared",
        "--lane-lease-manifest-fd",
        str(manifest_fd),
        "--cache-dir",
        str(args.checkpoint.resolve()),
        "--converted-ckpt-dir",
        candidate["paths"]["converted_checkpoint"],
        "--storage-path",
        candidate["paths"]["storage"],
        "--listen-port",
        str(args.listen_port),
        "--dist-init-addr",
        f"localhost:{args.dist_port}",
        "--pynccl-port-base",
        str(args.pynccl_port_base),
        "--pynccl-port-span",
        str(args.pynccl_port_span),
        "--host-kv-cache-size",
        str(args.host_kv_cache_gb),
        "--nnodes",
        "1",
        "--node-rank",
        "0",
        "--world-size",
        str(len(args.gpu_uuid)),
    ]
    return command


def _wait_for_instance_lock(instance_id: str, process: subprocess.Popen, timeout: float) -> None:
    path = HOST_LOCK_ROOT / f"instance-{instance_id}.lock"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LaneError(f"server exited before admission, code={process.returncode}")
        if path.exists():
            try:
                fd = _open_lock(path, fcntl.LOCK_EX)
            except BlockingIOError:
                return
            else:
                _close_fd(fd)
        time.sleep(0.1)
    raise LaneError("server did not acquire its instance lock before timeout")


def start_lane(args: argparse.Namespace) -> dict[str, Any]:
    _validate_start_args(args)
    state_root = _ensure_private_dir(STATE_ROOT)
    host_root = _ensure_private_dir(HOST_LOCK_ROOT)
    lock_root = _ensure_private_dir(LANE_LOCK_ROOT)
    admission_fd = _open_lock(host_root / "admission.lock", fcntl.LOCK_EX)
    host_fd = None
    manifest_fd = None
    resource_fds: list[int] = []
    log_file = None
    process: subprocess.Popen | None = None
    process_pidfd: int | None = None
    admitted = False
    try:
        active = _active_manifests(state_root)
        _check_lane_root_available(args.lane_root, active)
        pyroot, paths, log_path = _prepare_paths(
            args.lane_root,
            args.worktree,
            args.converted_ckpt_dir,
        )
        candidate = _candidate(args, paths)
        _check_no_overlap(candidate, active)
        _check_memory(candidate, active)
        ports = [args.listen_port, args.dist_port]
        if len(args.gpu_uuid) > 1:
            ports.extend(
                range(
                    args.pynccl_port_base,
                    args.pynccl_port_base + args.pynccl_port_span,
                )
            )
        _check_ports_free(ports)
        _check_gpus_free(args.gpu_uuid)

        host_fd = _open_lock(host_root / "host.lock", fcntl.LOCK_SH)
        resources: dict[str, int] = {}
        for resource in sorted(_resource_names(candidate)):
            fd = _open_lock(
                lock_root / _resource_filename(resource),
                fcntl.LOCK_EX,
            )
            resources[resource] = fd
            resource_fds.append(fd)

        tiktoken_cache_dir = _seed_o200k_base(
            Path(paths["temp"]), args.o200k_base_file
        )

        lease_path = state_root / f"{args.instance_id}.lease.json"
        lease = {
            "version": 1,
            "instance_id": args.instance_id,
            "gpu_uuids": args.gpu_uuid,
            "listen_port": args.listen_port,
            "dist_init_port": args.dist_port,
            "pynccl_port_base": args.pynccl_port_base,
            "pynccl_port_span": args.pynccl_port_span,
            "paths": paths,
            "resources": resources,
        }
        _atomic_json(lease_path, lease)
        manifest_fd = os.open(lease_path, os.O_RDONLY)

        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": ",".join(args.gpu_uuid),
                "TMPDIR": paths["temp"],
                "TIKTOKEN_CACHE_DIR": str(tiktoken_cache_dir),
                "TORCH_EXTENSIONS_DIR": paths["torch_extensions"],
                "TRITON_CACHE_DIR": paths["triton"],
                "TORCHINDUCTOR_CACHE_DIR": paths["torchinductor"],
                "CUDA_CACHE_PATH": paths["cuda"],
                "PYTHONPATH": str(pyroot)
                + (os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""),
            }
        )
        command = _server_command(args, candidate, manifest_fd)
        log_file = open(log_path, "ab", buffering=0)
        pass_fds = tuple([host_fd, manifest_fd, *resource_fds])
        _require_pidfd_signaling()
        state_path = state_root / f"{args.instance_id}.json"
        # If launch fails after this point, an ambiguous lane blocks admission
        # instead of leaving a detached process without a manifest.
        _atomic_json(state_path, {**candidate, "state": "starting"})
        process = subprocess.Popen(
            command,
            cwd=args.worktree,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            pass_fds=pass_fds,
        )
        manifest = {
            **candidate,
            "state": "starting",
            "pid": process.pid,
            "process_group": os.getpgid(process.pid),
            "pid_start_time": _proc_start_time(process.pid),
            "boot_id": _boot_id(),
            "command": command,
            "worktree": str(args.worktree.resolve()),
            "lane_root": str(args.lane_root.resolve()),
            "pyroot": str(pyroot),
            "log_path": str(log_path),
            "commit": subprocess.run(
                ["git", "-C", str(args.worktree), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "started_at": time.time(),
        }
        _atomic_json(state_path, manifest)
        process_pidfd = os.pidfd_open(process.pid)
        _wait_for_instance_lock(args.instance_id, process, args.admission_timeout)
        manifest["state"] = "admitted"
        _atomic_json(state_path, manifest)
        admitted = True
        return manifest
    finally:
        if process is not None and not admitted and process_pidfd is not None:
            _signal_pidfd(process_pidfd, signal.SIGKILL)
        if process_pidfd is not None:
            os.close(process_pidfd)
        if log_file is not None:
            log_file.close()
        close_passed_fd = (
            _close_inherited_parent_fd if process is not None else _close_fd
        )
        for fd in reversed(resource_fds):
            close_passed_fd(fd)
        close_passed_fd(manifest_fd)
        close_passed_fd(host_fd)
        _close_fd(admission_fd)


def _lane_state_path(args: argparse.Namespace) -> Path:
    _validate_instance_id(args.instance_id)
    return args.state_root / f"{args.instance_id}.json"


def lane_status(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _read_json(_lane_state_path(args))
    alive = _pid_identity_matches(manifest)
    result = dict(manifest)
    result["owner_identity_valid"] = alive
    if alive:
        temp_root = Path(manifest["paths"]["temp"])
        runtime_dirs = sorted(
            temp_root.glob(f"batchgen_{args.instance_id}_*")
        )
        result["runtime_dirs"] = [str(path) for path in runtime_dirs]
    return result


def stop_lane(args: argparse.Namespace) -> dict[str, Any]:
    _validate_instance_id(args.instance_id)
    host_root = _ensure_private_dir(HOST_LOCK_ROOT)
    admission_fd = _open_lock(host_root / "admission.lock", fcntl.LOCK_EX)
    try:
        return _stop_lane_under_admission_lock(args)
    finally:
        _close_fd(admission_fd)


def _stop_lane_under_admission_lock(args: argparse.Namespace) -> dict[str, Any]:
    state_path = _lane_state_path(args)
    manifest = _read_json(state_path)
    process_group = manifest.get("process_group")
    if not isinstance(process_group, int) or process_group <= 0:
        raise LaneError("lane owner identity is unavailable; preserving manifest")
    if _pid_identity_matches(manifest):
        pidfd = _open_verified_owner_pidfd(manifest)
        try:
            manifest["state"] = "stopping"
            _atomic_json(state_path, manifest)
            _signal_pidfd(pidfd, signal.SIGTERM)
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline and _process_group_exists(process_group):
                time.sleep(0.2)
            if _process_group_exists(process_group):
                if not _pid_identity_matches(manifest):
                    manifest["state"] = "failed"
                    _atomic_json(state_path, manifest)
                    raise LaneError("owner identity changed during lane stop")
                _signal_pidfd(pidfd, signal.SIGKILL)
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and _process_group_exists(process_group):
                    time.sleep(0.2)
            if _process_group_exists(process_group):
                manifest["state"] = "failed"
                _atomic_json(state_path, manifest)
                raise LaneError("lane process group remains live after owner SIGKILL")
        finally:
            os.close(pidfd)
    elif _process_group_exists(process_group):
        raise LaneError("refusing to signal an unverified lane owner")

    temp_root = Path(manifest["paths"]["temp"])
    runtime_dirs = list(temp_root.glob(f"batchgen_{args.instance_id}_*"))
    shm_name = re.compile(
        rf"batchgen_{re.escape(args.instance_id)}_[0-9a-f]{{32}}\."
    )
    shm_objects = [
        path
        for path in Path("/dev/shm").glob(f"batchgen_{args.instance_id}_*")
        if shm_name.match(path.name)
    ]
    gpu_processes = [
        (uuid, process_pid)
        for uuid, process_pid in _gpu_processes()
        if uuid in set(manifest["gpu_uuids"])
    ]
    if runtime_dirs or shm_objects or gpu_processes:
        manifest["state"] = "failed"
        manifest["residual_runtime_dirs"] = [str(path) for path in runtime_dirs]
        manifest["residual_shm"] = [str(path) for path in shm_objects]
        manifest["residual_gpu_processes"] = gpu_processes
        _atomic_json(state_path, manifest)
        raise LaneError("lane stopped with residual lane resources; preserving them")
    manifest["state"] = "stopped"
    manifest["stopped_at"] = time.time()
    _atomic_json(state_path, manifest)
    return manifest


def verify_lane(args: argparse.Namespace) -> dict[str, Any]:
    manifest = lane_status(args)
    if not manifest["owner_identity_valid"]:
        raise LaneError("lane owner identity is not live and exact")
    assigned = set(manifest["gpu_uuids"])
    foreign = []
    for uuid, pid in _gpu_processes():
        if uuid not in assigned:
            continue
        try:
            if os.getpgid(pid) != manifest["process_group"]:
                foreign.append((uuid, pid))
        except ProcessLookupError:
            continue
    if foreign:
        raise LaneError(f"foreign GPU processes occupy this lane: {foreign}")

    unheld = []
    for resource in _resource_names(manifest):
        path = LANE_LOCK_ROOT / _resource_filename(resource)
        try:
            fd = _open_lock(path, fcntl.LOCK_EX)
        except BlockingIOError:
            continue
        else:
            unheld.append(resource)
            _close_fd(fd)
    if unheld:
        raise LaneError(f"lane resource locks are not held: {unheld}")
    manifest["verified"] = True
    return manifest


def exclusive_run(args: argparse.Namespace) -> int:
    host_root = _ensure_private_dir(HOST_LOCK_ROOT)
    fd = _open_lock(host_root / "host.lock", fcntl.LOCK_EX)
    try:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        return subprocess.run(command, check=False).returncode
    finally:
        _close_fd(fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    start = subparsers.add_parser("start")
    start.add_argument("--instance-id", required=True)
    start.add_argument("--model", default="openai/gpt-oss-120b")
    start.add_argument("--gpu-uuid", action="append", required=True)
    start.add_argument("--listen-port", type=int, required=True)
    start.add_argument("--dist-port", type=int, required=True)
    start.add_argument("--pynccl-port-base", type=int, required=True)
    start.add_argument("--pynccl-port-span", type=int, default=8)
    start.add_argument("--host-kv-cache-gb", type=int, required=True)
    start.add_argument("--host-memory-gb", type=float, required=True)
    start.add_argument("--shm-gb", type=float, required=True)
    start.add_argument("--safety-gb", type=float, default=64)
    start.add_argument("--checkpoint", type=Path, required=True)
    start.add_argument("--o200k-base-file", type=Path, required=True)
    start.add_argument("--converted-ckpt-dir", type=Path, required=True)
    start.add_argument("--worktree", type=Path, required=True)
    start.add_argument("--lane-root", type=Path, required=True)
    start.add_argument("--python", required=True)
    start.add_argument("--admission-timeout", type=float, default=30)

    for name in ("status", "stop", "verify"):
        command = subparsers.add_parser(name)
        command.add_argument("--instance-id", required=True)
        command.add_argument("--state-root", type=Path, default=STATE_ROOT)
        if name == "stop":
            command.add_argument("--timeout", type=float, default=30)

    exclusive = subparsers.add_parser("exclusive-run")
    exclusive.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command_name == "start":
            result = start_lane(args)
        elif args.command_name == "status":
            result = lane_status(args)
        elif args.command_name == "stop":
            result = stop_lane(args)
        elif args.command_name == "verify":
            result = verify_lane(args)
        else:
            if not args.command:
                raise LaneError("exclusive-run requires a command")
            return exclusive_run(args)
    except (LaneError, BlockingIOError, OSError, subprocess.SubprocessError) as exc:
        print(f"lane-runtime: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
