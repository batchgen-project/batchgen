"""Stage a Kimi-K3 distributed weight store into a managed tmpfs directory.

The store builder's output (``store_path`` and its completion manifest) is the
**source**: a deployment input BatchGen never deletes. When the node config
sets ``store_residency``, the server maps a copy it owns instead:

* the copy lives in ``<store_staging_dir>/node<R>-<sha256(manifest)[:16]>/``
  as ``store.bin``, ``metadata.tsv``, ``store_manifest.json``, ``residency``
  and ``COMPLETE``, the last written only after every manifest segment of the
  copy hashed to the manifest. The id changes with every build, so a stale
  copy is never taken for the current one;
* ``use.lock`` in that directory is held ``LOCK_SH`` by every server mapping
  the copy, for the server's whole life, and a directory is removed only by a
  holder of ``LOCK_EX`` -- so a copy in use is never removed;
* a start removes every idle directory of another id, reuses a ``COMPLETE``
  copy of the current id without reading the source, and otherwise copies the
  source under ``LOCK_EX`` with the copying thread interleaving its pages
  across all NUMA nodes;
* a ``run`` copy is removed when its server stops; a ``resident`` copy stays
  for the next start. A copy left by a killed server is reused by the next
  start of the same build or swept by a start of another.

The daemon and the workers stay path-based: they receive an effective config,
written into the run's runtime directory, that names the staged files.
"""

from __future__ import annotations

import contextlib
import ctypes
import fcntl
import json
import logging
import os
import platform
import shlex
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ContextManager, Iterator, List, Optional, Tuple

from batchgen.models.moonshotai.kimi_linear.distributed_weight_store import (
    DistributedMetadataError,
    load_distributed_weight_config,
    read_distributed_weight_config,
)
from batchgen.models.moonshotai.kimi_linear.store_builder import (
    DEFAULT_HASH_WORKERS,
    DEFAULT_READ_BUFFER_BYTES,
    METADATA_FILE_NAME,
    MANIFEST_FILE_NAME,
    STAGED_COMPLETE_FILE_NAME as COMPLETE_FILE_NAME,
    STORE_FILE_NAME,
    K3StoreBuilderError,
    K3StoreManifestError,
    hash_store_segment,
    manifest_segment_spans,
    complete_staged_store,
    rebuild_command,
    staged_store_id,
    verify_store_manifest,
)

logger = logging.getLogger(__name__)

USE_LOCK_NAME = "use.lock"
RESIDENCY_FILE_NAME = "residency"
PARTIAL_STORE_NAME = STORE_FILE_NAME + ".partial"
#: A directory being removed is first renamed to this prefix, so a start that
#: races the removal can never lock a half-deleted copy under its real name.
REMOVING_PREFIX = ".removing-"
#: ``huge=`` values under which a tmpfs backs files with transparent huge pages.
HUGE_MOUNT_OPTIONS = ("within_size", "always")

# How long a start waits for a peer that holds the current id's lock only for
# the instant between checking for COMPLETE and taking LOCK_EX. A peer that is
# copying holds LOCK_EX, which a start waits out in a blocking LOCK_SH instead.
_EXCLUSIVE_RETRY_S = 0.1
_EXCLUSIVE_RETRIES = 100

_MPOL_DEFAULT = 0
_MPOL_INTERLEAVE = 3
_SYS_SET_MEMPOLICY = {"x86_64": 238, "amd64": 238, "aarch64": 237, "arm64": 237}
_NODE_ONLINE_PATH = "/sys/devices/system/node/online"


class WeightStoreStagingError(RuntimeError):
    """A staged weight store cannot be prepared or is refused."""


# --------------------------------------------------------------------------- #
#  Mount gate                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MountEntry:
    mountpoint: str
    fstype: str
    options: Tuple[str, ...]

    def option(self, name: str) -> Optional[str]:
        prefix = name + "="
        value = None
        for option in self.options:
            if option.startswith(prefix):
                value = option[len(prefix):]
        return value


def read_proc_mounts() -> str:
    return Path("/proc/mounts").read_text()


def _unescape_mount_field(value: str) -> str:
    # /proc/mounts octal-escapes space, tab, newline and backslash.
    for escaped, raw in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, raw)
    return value


def find_mount(path: Path | str, mounts_text: str) -> Optional[MountEntry]:
    """The mount that holds ``path`` (the longest matching mount point; the
    last listed of equal ones, which is the one on top)."""
    target = os.path.realpath(path)
    best: Optional[MountEntry] = None
    for line in mounts_text.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        mountpoint = _unescape_mount_field(fields[1])
        prefix = mountpoint.rstrip("/") + "/"
        if target != mountpoint and not target.startswith(prefix):
            continue
        if best is None or len(mountpoint) >= len(best.mountpoint):
            best = MountEntry(mountpoint, fields[2], tuple(fields[3].split(",")))
    return best


def check_staging_mount(
    staging_dir: Path,
    *,
    fast_init: bool,
    read_mounts: Callable[[], str] = read_proc_mounts,
) -> MountEntry:
    """Require ``staging_dir`` on a tmpfs, mounted ``huge=within_size|always``
    under ``--fast-init``. Raises :class:`WeightStoreStagingError`."""
    try:
        mounts_text = read_mounts()
    except OSError as error:
        raise WeightStoreStagingError(
            f"cannot read the mount table to check that the weight store "
            f"staging directory {staging_dir} is on a tmpfs: {error}"
        ) from error
    entry = find_mount(staging_dir, mounts_text)
    if entry is None or entry.fstype != "tmpfs":
        where = (
            f"{entry.fstype} mount {entry.mountpoint}" if entry else "no known mount"
        )
        raise WeightStoreStagingError(
            f"the weight store staging directory {staging_dir} is on {where}, "
            "not a tmpfs; store_staging_dir must name a directory on a tmpfs "
            "such as /dev/shm"
        )
    if fast_init:
        huge = entry.option("huge") or "never"
        if huge not in HUGE_MOUNT_OPTIONS:
            raise WeightStoreStagingError(
                f"--fast-init requires the weight store staging tmpfs "
                f"{entry.mountpoint} to be mounted huge=within_size or "
                f"huge=always (it is huge={huge}). Remount it with: "
                f"mount -o remount,huge=within_size {shlex.quote(entry.mountpoint)}"
            )
    return entry


def free_bytes(path: Path) -> int:
    stats = os.statvfs(path)
    return stats.f_bavail * stats.f_frsize


# --------------------------------------------------------------------------- #
#  NUMA placement of the copy                                                 #
# --------------------------------------------------------------------------- #


def _online_nodes() -> List[int]:
    try:
        text = Path(_NODE_ONLINE_PATH).read_text().strip()
    except OSError:
        return []
    nodes: List[int] = []
    for part in filter(None, text.split(",")):
        low, _, high = part.partition("-")
        nodes.extend(range(int(low), int(high or low) + 1))
    return nodes


@contextlib.contextmanager
def interleave_all_nodes() -> Iterator[None]:
    """Run the body under an all-nodes ``MPOL_INTERLEAVE`` task mempolicy.

    The policy is per thread, so the staged copy's tmpfs pages, allocated
    when this thread writes them, are spread across every NUMA node instead
    of landing on the copying thread's node. A thread started inside the
    body inherits the policy (the kernel copies it on clone), which is how
    the copy threads get it. ``MPOL_DEFAULT`` is restored on exit. Without NUMA syscalls (not Linux, an unknown architecture) or with
    a single node there is nothing to interleave and the body runs as is.
    """
    number = _SYS_SET_MEMPOLICY.get(platform.machine())
    nodes = _online_nodes() if sys.platform.startswith("linux") else []
    if number is None or len(nodes) < 2:
        yield
        return
    libc = ctypes.CDLL(None, use_errno=True)
    words = max(nodes) // 64 + 1
    mask = (ctypes.c_ulong * words)()
    for node in nodes:
        mask[node // 64] |= 1 << (node % 64)
    if libc.syscall(
        ctypes.c_long(number),
        ctypes.c_int(_MPOL_INTERLEAVE),
        mask,
        ctypes.c_ulong(words * 64 + 1),
    ) != 0:
        logger.warning(
            "set_mempolicy(MPOL_INTERLEAVE, nodes %s) failed: %s; the staged "
            "weight store is placed by the default policy",
            nodes,
            os.strerror(ctypes.get_errno()),
        )
        yield
        return
    try:
        yield
    finally:
        libc.syscall(
            ctypes.c_long(number),
            ctypes.c_int(_MPOL_DEFAULT),
            None,
            ctypes.c_ulong(0),
        )


# --------------------------------------------------------------------------- #
#  use.lock protocol                                                          #
# --------------------------------------------------------------------------- #


def _still_linked(fd: int, path: Path) -> bool:
    """``fd`` is still the file at ``path`` (a remover renames the directory
    away while holding ``LOCK_EX``, so a lock taken on an unlinked inode
    protects nothing)."""
    try:
        linked = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return False
    held = os.fstat(fd)
    return (linked.st_dev, linked.st_ino) == (held.st_dev, held.st_ino)


def _lock_use(directory: Path, operation: int, *, create: bool) -> Optional[int]:
    """``flock`` ``directory/use.lock`` and return the descriptor.

    Returns None when ``operation`` has ``LOCK_NB`` and the lock is held, or,
    without ``create``, when the directory is gone. With ``create`` the
    directory is (re)created as needed, so the caller always ends up holding
    the lock of the directory that is linked at ``directory``.
    """
    lock_path = directory / USE_LOCK_NAME
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    while True:
        if create:
            directory.mkdir(mode=0o755, exist_ok=True)
        try:
            fd = os.open(lock_path, flags, 0o644)
        except FileNotFoundError:
            if create:
                continue
            return None
        try:
            fcntl.flock(fd, operation)
        except BlockingIOError:
            os.close(fd)
            return None
        except BaseException:
            os.close(fd)
            raise
        if _still_linked(fd, lock_path):
            return fd
        os.close(fd)
        if not create:
            return None


def _remove_locked(directory: Path, fd: int) -> None:
    """Remove ``directory`` while holding its ``LOCK_EX`` (``fd``); closes
    ``fd``. The directory leaves its name atomically before any file goes."""
    try:
        trash = directory.with_name(
            f"{REMOVING_PREFIX}{directory.name}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        os.rename(directory, trash)
    finally:
        os.close(fd)
    shutil.rmtree(trash, ignore_errors=True)


def sweep_idle_stores(staging_dir: Path, keep: str) -> List[str]:
    """Remove every directory under ``staging_dir`` but ``keep`` whose
    ``use.lock`` can be taken ``LOCK_EX`` without blocking, whatever its
    residency. Returns the removed names."""
    removed: List[str] = []
    for entry in sorted(staging_dir.iterdir()):
        if entry.name == keep or entry.is_symlink() or not entry.is_dir():
            continue
        if entry.name.startswith(REMOVING_PREFIX):
            # A removal that died part-way; nothing can lock it any more.
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name)
            continue
        fd = _lock_use(entry, fcntl.LOCK_EX | fcntl.LOCK_NB, create=False)
        if fd is None:
            logger.info("keeping staged weight store %s: in use", entry)
            continue
        _remove_locked(entry, fd)
        logger.info("removed idle staged weight store %s", entry)
        removed.append(entry.name)
    return removed


# --------------------------------------------------------------------------- #
#  Staging                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class StagedWeightStore:
    """A staged copy this server maps; holds its ``use.lock`` ``LOCK_SH``."""

    directory: Path
    effective_config_path: Path
    residency: str
    reused: bool
    _lock_fd: Optional[int] = field(default=None, repr=False)

    def release(self) -> None:
        """Drop this server's hold. A ``run`` copy is removed unless another
        server still maps it; a ``resident`` copy is kept."""
        fd, self._lock_fd = self._lock_fd, None
        if fd is None:
            return
        if self.residency != "run":
            os.close(fd)
            logger.info("kept resident staged weight store %s", self.directory)
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            logger.warning(
                "staged weight store %s is still mapped by another server; "
                "leaving it for that server to remove",
                self.directory,
            )
            return
        if not _still_linked(fd, self.directory / USE_LOCK_NAME):
            os.close(fd)
            return
        _remove_locked(self.directory, fd)
        logger.info("removed run-scoped staged weight store %s", self.directory)


def _write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "wb") as handle:
        handle.write(data)
    os.replace(temporary, path)


class _Stager:
    def __init__(
        self,
        config_path: Path,
        config: dict,
        runtime_dir: Path,
        *,
        read_buffer_bytes: int,
        copy_workers: int,
        free_bytes_of: Callable[[Path], int],
        interleave: Callable[[], ContextManager[None]],
    ) -> None:
        self.config_path = config_path
        self.raw_config = json.loads(config_path.read_text())
        self.config = config
        self.residency = self.config["store_residency"]
        self.staging_dir = Path(self.config["store_staging_dir"])
        self.runtime_dir = runtime_dir
        self.read_buffer_bytes = read_buffer_bytes
        self.copy_workers = copy_workers
        self.free_bytes_of = free_bytes_of
        self.interleave = interleave
        self.manifest_bytes = self._read_source_manifest()
        self.staged_id = staged_store_id(self.config["node_rank"], self.manifest_bytes)
        self.directory = self.staging_dir / self.staged_id
        self.effective_config_path = (
            runtime_dir / f"distributed-weight-node{int(self.config['node_rank'])}.json"
        )

    def _read_source_manifest(self) -> bytes:
        manifest_path = self.config.get("manifest_path")
        try:
            return Path(str(manifest_path)).read_bytes()
        except OSError:
            # The cheap gate names the reason and the rebuild command.
            verify_store_manifest(self.config, self.config_path)
            raise

    def write_effective_config(self) -> None:
        effective = dict(self.raw_config)
        effective.update(
            store_path=str(self.directory / STORE_FILE_NAME),
            metadata_path=str(self.directory / METADATA_FILE_NAME),
            manifest_path=str(self.directory / MANIFEST_FILE_NAME),
            staged_from=str(self.config_path),
        )
        _write_atomic(
            self.effective_config_path,
            (json.dumps(effective, indent=2, sort_keys=True) + "\n").encode(),
        )

    def staged_state(self) -> Tuple[str, Optional[str]]:
        """``("absent", None)``, ``("valid", None)`` or ``("invalid", why)``.

        A copy is valid when ``COMPLETE`` is present and the effective config
        that names it passes the manifest gate every consumer runs.
        """
        if not (self.directory / COMPLETE_FILE_NAME).is_file():
            return "absent", None
        self.write_effective_config()
        try:
            load_distributed_weight_config(self.effective_config_path)
        except (K3StoreManifestError, ValueError, OSError) as error:
            return "invalid", str(error)
        return "valid", None

    def record_residency(self) -> None:
        # The gates read this to tell a declared resident copy from residue,
        # so it follows the current deployment's choice for a reused copy too.
        path = self.directory / RESIDENCY_FILE_NAME
        try:
            current = path.read_text().strip()
        except OSError:
            current = None
        if current != self.residency:
            _write_atomic(path, (self.residency + "\n").encode())

    def _check_copied_segment(
        self, index, segment, future, source_path, partial, manifest
    ) -> None:
        try:
            digest = future.result()
        except (OSError, EOFError) as error:
            raise WeightStoreStagingError(
                f"cannot copy segment {index} of {source_path} "
                f"into {partial}: {error}"
            ) from error
        if digest != segment["digest"]:
            begin = int(segment["offset"])
            raise WeightStoreStagingError(
                f"segment {index} of the source weight store "
                f"{source_path} ([{begin}, "
                f"{begin + int(segment['bytes'])})) does not hash "
                "to its completion manifest: the source changed "
                "after the build published it. Rebuild it with: "
                + rebuild_command(self.config_path, self.config, manifest=manifest)
            )

    def copy(self) -> None:
        """Copy and verify the source into this id's directory (``LOCK_EX``
        held). Leaves ``COMPLETE`` absent on any failure."""
        (self.directory / COMPLETE_FILE_NAME).unlink(missing_ok=True)
        for entry in list(self.directory.iterdir()):
            if entry.name == USE_LOCK_NAME:
                continue
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()

        manifest = verify_store_manifest(self.config, self.config_path)
        if json.loads(self.manifest_bytes) != manifest:
            raise WeightStoreStagingError(
                f"the completion manifest of {self.config_path} changed while "
                "the server was staging it; a store build republished the "
                "source -- start the server again"
            )
        try:
            segment_spans = manifest_segment_spans(self.config, manifest)
            metadata = Path(str(self.config["metadata_path"])).read_bytes()
        except (K3StoreBuilderError, DistributedMetadataError, OSError) as error:
            raise WeightStoreStagingError(
                f"cannot read the store layout of {self.config_path}: {error}"
            ) from error
        store_bytes = int(self.config["store_bytes"])
        needed = store_bytes + len(metadata) + len(self.manifest_bytes)
        available = self.free_bytes_of(self.directory)
        if available < needed:
            raise WeightStoreStagingError(
                f"the weight store staging tmpfs holding {self.staging_dir} has "
                f"{available} B free but the staged store needs {needed} B; "
                "free it or remount it larger (mount -o remount,size=<bytes> "
                "<mountpoint>) before starting"
            )

        source_path = str(self.config["store_path"])
        partial = self.directory / PARTIAL_STORE_NAME
        segments = manifest["store"]["segments"]
        logger.info(
            "staging weight store %s (%d B, %d segments, %d copy threads) into %s",
            source_path,
            store_bytes,
            len(segments),
            self.copy_workers,
            self.directory,
        )
        started = time.monotonic()
        with self.interleave():
            try:
                source = os.open(source_path, os.O_RDONLY)
            except OSError as error:
                raise WeightStoreStagingError(
                    f"cannot open the source weight store {source_path}: {error}"
                ) from error
            try:
                sink = os.open(
                    partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
                )
                try:
                    os.ftruncate(sink, store_bytes)
                    # One thread hashes at a fraction of what the source disk
                    # and tmpfs sustain, so segments are copied concurrently;
                    # pread/pwrite at explicit offsets keep the threads
                    # independent. Results are checked in segment order, so a
                    # refusal names the first bad segment whatever finished
                    # first. The threads start inside interleave() and so
                    # inherit its mempolicy.
                    with ThreadPoolExecutor(
                        max_workers=self.copy_workers,
                        thread_name_prefix="weight-store-copy",
                    ) as pool:
                        pending = [
                            pool.submit(
                                hash_store_segment,
                                source,
                                spans,
                                self.read_buffer_bytes,
                                sink,
                            )
                            for spans in segment_spans
                        ]
                        try:
                            for index, (segment, future) in enumerate(
                                zip(segments, pending)
                            ):
                                self._check_copied_segment(
                                    index, segment, future, source_path, partial, manifest
                                )
                        except BaseException:
                            for future in pending:
                                future.cancel()
                            raise
                finally:
                    os.close(sink)
            finally:
                os.close(source)
        os.rename(partial, self.directory / STORE_FILE_NAME)
        _write_atomic(self.directory / METADATA_FILE_NAME, metadata)
        _write_atomic(self.directory / MANIFEST_FILE_NAME, self.manifest_bytes)
        self.record_residency()
        self.write_effective_config()
        try:
            load_distributed_weight_config(self.effective_config_path)
        except (K3StoreManifestError, ValueError, OSError) as error:
            raise WeightStoreStagingError(
                f"the staged weight store {self.directory} fails the manifest "
                f"gate after copying: {error}"
            ) from error
        # The commit point: written only after every segment verified.
        _write_atomic(
            self.directory / COMPLETE_FILE_NAME, (self.staged_id + "\n").encode()
        )
        logger.info(
            "staged weight store %s in %.1fs", self.directory, time.monotonic() - started
        )


def stage_distributed_weight_store(
    config_path: Path | str,
    runtime_dir: Path | str,
    *,
    fast_init: bool,
    read_mounts: Callable[[], str] = read_proc_mounts,
    free_bytes_of: Callable[[Path], int] = free_bytes,
    interleave: Callable[[], ContextManager[None]] = interleave_all_nodes,
    read_buffer_bytes: int = DEFAULT_READ_BUFFER_BYTES,
    copy_workers: int = DEFAULT_HASH_WORKERS,
) -> Optional[StagedWeightStore]:
    """Stage the store ``config_path`` names; None for a legacy config.

    Returns with the copy's ``use.lock`` held ``LOCK_SH`` and an effective
    config written into ``runtime_dir`` that passes the manifest gate. The
    caller hands that config to the daemon and the workers and calls
    :meth:`StagedWeightStore.release` once both have stopped.
    """
    config_path = Path(os.path.abspath(config_path))
    config = read_distributed_weight_config(config_path)
    if config["store_residency"] is None:
        return None
    stager = _Stager(
        config_path,
        config,
        Path(runtime_dir),
        read_buffer_bytes=read_buffer_bytes,
        copy_workers=copy_workers,
        free_bytes_of=free_bytes_of,
        interleave=interleave,
    )
    check_staging_mount(stager.staging_dir, fast_init=fast_init, read_mounts=read_mounts)
    stager.staging_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    sweep_idle_stores(stager.staging_dir, keep=stager.staged_id)

    def staged(fd: int, reused: bool) -> StagedWeightStore:
        stager.record_residency()
        logger.info(
            "%s staged weight store %s (residency=%s); effective config %s",
            "reusing" if reused else "mapping",
            stager.directory,
            stager.residency,
            stager.effective_config_path,
        )
        return StagedWeightStore(
            directory=stager.directory,
            effective_config_path=stager.effective_config_path,
            residency=stager.residency,
            reused=reused,
            _lock_fd=fd,
        )

    busy_retries = 0
    while True:
        # A peer that is copying holds LOCK_EX; this waits it out.
        fd = _lock_use(stager.directory, fcntl.LOCK_SH, create=True)
        try:
            state, why = stager.staged_state()
        except BaseException:
            os.close(fd)
            raise
        if state == "valid":
            return staged(fd, reused=True)
        os.close(fd)
        if state == "invalid":
            logger.warning(
                "staged weight store %s is marked COMPLETE but refused: %s; "
                "staging it again",
                stager.directory,
                why,
            )
        fd = _lock_use(
            stager.directory, fcntl.LOCK_EX | fcntl.LOCK_NB, create=True
        )
        if fd is None:
            if state == "invalid":
                raise WeightStoreStagingError(
                    f"the staged weight store {stager.directory} is refused "
                    f"({why}) but another server maps it, so it cannot be "
                    "staged again; stop that server first"
                )
            busy_retries += 1
            if busy_retries > _EXCLUSIVE_RETRIES:
                raise WeightStoreStagingError(
                    f"could not take {stager.directory / USE_LOCK_NAME} to "
                    "stage the weight store: other holders kept it"
                )
            time.sleep(_EXCLUSIVE_RETRY_S)
            continue
        try:
            # A peer may have finished staging between the two locks.
            state, why = stager.staged_state()
            reused = state == "valid"
            if not reused:
                stager.copy()
        except BaseException:
            _remove_locked(stager.directory, fd)
            raise
        # flock conversion is not atomic: a sweeper may take LOCK_EX in the
        # gap and remove the copy, in which case it is staged again.
        fcntl.flock(fd, fcntl.LOCK_SH)
        lock_path = stager.directory / USE_LOCK_NAME
        if _still_linked(fd, lock_path) and (
            stager.directory / COMPLETE_FILE_NAME
        ).is_file():
            return staged(fd, reused=reused)
        os.close(fd)
        logger.warning(
            "staged weight store %s was removed while its lock was being "
            "downgraded; staging it again",
            stager.directory,
        )
