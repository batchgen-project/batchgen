"""Host-local lifetime locks for BatchGen runtimes."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import BinaryIO

from batchgen.server.runtime_identity import RuntimeIdentity


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
