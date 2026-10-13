"""Host-local lifetime locks for BatchGen runtimes."""

from __future__ import annotations

import fcntl
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import BinaryIO, List, Optional

from batchgen.server.process_utils import cleanup_shm_files
from batchgen.server.runtime_identity import RuntimeIdentity

logger = logging.getLogger(__name__)

# Name of the per-run liveness lock inside a run's runtime directory. Every
# process of the run holds it LOCK_SH for its whole life, so a later run of the
# same instance proves the previous run is dead by taking it LOCK_EX.
RUN_LOCK_NAME = "run.lock"


class RuntimeLockError(RuntimeError):
    """Raised when a conflicting runtime already owns an admission lock."""


class RuntimeLocks:
    """Own the host and logical-instance locks for one server lifetime."""

    def __init__(self, host_file: BinaryIO, instance_file: BinaryIO) -> None:
        self._host_file = host_file
        self._instance_file = instance_file

    @classmethod
    def acquire(
        cls,
        identity: RuntimeIdentity,
        *,
        lock_root: Path | None = None,
    ) -> "RuntimeLocks":
        root = lock_root or Path("/tmp/batchgen-runtime-locks")
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise RuntimeLockError(f"runtime lock root is not a directory: {root}")
        root_metadata = root.stat()
        if root_metadata.st_uid != os.geteuid():
            raise RuntimeLockError(f"runtime lock root is not owned by this user: {root}")
        if root_metadata.st_mode & 0o077:
            raise RuntimeLockError(
                f"runtime lock root must not be group/world accessible: {root}"
            )

        host_file = cls._lock_file(
            root / "host.lock",
            shared=identity.mode == "shared",
            description=f"host runtime mode {identity.mode}",
        )
        try:
            instance_file = cls._lock_file(
                root / f"instance-{identity.instance_id}.lock",
                shared=False,
                description=f"instance {identity.instance_id!r}",
            )
        except BaseException:
            cls._close_file(host_file)
            raise
        return cls(host_file, instance_file)

    @staticmethod
    def _lock_file(
        path: Path,
        *,
        shared: bool,
        description: str,
    ) -> BinaryIO:
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(path, flags, 0o600)
        file_obj = os.fdopen(fd, "r+b", buffering=0)
        operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        try:
            fcntl.flock(file_obj.fileno(), operation | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            file_obj.close()
            raise RuntimeLockError(
                f"cannot acquire {description} lock; another runtime owns it"
            ) from exc
        return file_obj

    @staticmethod
    def _close_file(file_obj: BinaryIO | None) -> None:
        if file_obj is None or file_obj.closed:
            return
        try:
            fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            file_obj.close()
        except OSError:
            pass

    def close(self) -> None:
        """Release the instance lock, then the host admission lock."""
        self._close_file(self._instance_file)
        self._close_file(self._host_file)


def hold_run_lock(runtime_dir: Path | str, *, create: bool = False) -> BinaryIO:
    """Take this run's liveness lock LOCK_SH and return the open file.

    The caller must keep the returned file alive for its whole process life: the
    kernel drops the lock when the last descriptor closes, however the process
    died, so a successful LOCK_EX by a later run proves every process of this
    run is gone.  Only the server process creates the file; a worker that cannot
    find it is looking at a runtime directory its server never prepared.
    """
    path = Path(runtime_dir) / RUN_LOCK_NAME
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    if create:
        flags |= os.O_CREAT
    try:
        fd = os.open(path, flags, 0o600)
    except FileNotFoundError as exc:
        raise RuntimeLockError(f"run lock is missing: {path}") from exc
    file_obj = os.fdopen(fd, "r+b", buffering=0)
    try:
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_SH)
    except BaseException:
        file_obj.close()
        raise
    return file_obj


def release_run_lock(file_obj: Optional[BinaryIO]) -> None:
    """Drop this process's share of the run lock before the dir is removed."""
    RuntimeLocks._close_file(file_obj)


def list_run_named_objects(
    shm_prefix: str,
    runtime_dir: Path | str,
    *,
    shm_dir: Optional[Path] = None,
) -> List[str]:
    """Name everything one run can leave on this node after an abnormal exit.

    The model weights, tensor metadata, host KV and QueryBook regions are
    anonymous memfds the kernel reclaims, so only the run's /dev/shm namespace
    and its runtime directory can survive.
    """
    root = Path("/dev/shm") if shm_dir is None else Path(shm_dir)
    names: List[str] = []
    if root.is_dir():
        names.extend(
            str(root / entry.name)
            for entry in sorted(root.iterdir())
            if entry.name.startswith(shm_prefix)
        )
    runtime_dir = Path(runtime_dir)
    if runtime_dir.is_dir():
        names.append(str(runtime_dir))
    return names


def reclaim_dead_runs(
    identity: RuntimeIdentity,
    *,
    temp_dir: Optional[Path] = None,
    shm_dir: Optional[Path] = None,
) -> List[str]:
    """Remove the leftovers of this instance's previous, fully dead runs.

    Called with the instance admission lock already held and before the new
    runtime directory exists, so only runs of *this* instance are in scope and
    no concurrent run of it can start.  A run whose lock cannot be taken
    exclusively still owns a live process, which refuses startup rather than
    letting two runs share one instance's resources.

    Returns the run ids reclaimed.
    """
    root = Path(tempfile.gettempdir()) if temp_dir is None else Path(temp_dir)
    pattern = re.compile(
        rf"batchgen_{re.escape(identity.instance_id)}_(?P<run_id>[0-9a-f]{{32}})\Z"
    )
    reclaimed: List[str] = []
    if not root.is_dir():
        return reclaimed

    for entry in sorted(root.iterdir()):
        match = pattern.fullmatch(entry.name)
        if match is None or entry.name == identity.resource_prefix:
            continue
        if entry.is_symlink() or not entry.is_dir():
            logger.warning(
                "Skipping runtime leftover that is not a real directory: %s", entry
            )
            continue
        if os.lstat(entry).st_uid != os.geteuid():
            logger.warning(
                "Skipping runtime leftover owned by another user: %s", entry
            )
            continue

        lock_path = entry / RUN_LOCK_NAME
        if not lock_path.exists():
            logger.warning(
                "Leaving runtime leftover without a %s untouched; its liveness "
                "cannot be proven: %s",
                RUN_LOCK_NAME,
                entry,
            )
            continue

        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            fd = os.open(lock_path, flags)
        except OSError as exc:
            logger.warning("Skipping unreadable runtime leftover %s: %s", entry, exc)
            continue
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "a live process of a previous run of instance "
                    f"{identity.instance_id!r} still holds {lock_path}; refusing "
                    "startup"
                ) from exc
            run_shm_prefix = f"{entry.name}."
            removed = list_run_named_objects(
                run_shm_prefix, entry, shm_dir=shm_dir
            )
            cleanup_shm_files(run_shm_prefix, shm_dir=shm_dir)
            shutil.rmtree(entry)
            reclaimed.append(match.group("run_id"))
            logger.warning(
                "Reclaimed dead run %s of instance %s: every process of that run "
                "had exited. Removed: %s",
                match.group("run_id"),
                identity.instance_id,
                ", ".join(removed),
            )
        finally:
            os.close(fd)

    return reclaimed
