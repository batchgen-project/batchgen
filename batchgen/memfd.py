"""Anonymous shared memory for BatchGen processes.

Regions shared between the server and its workers are memfds: they carry no
/dev/shm name, so the kernel frees them when the last process mapping them
exits, however it exits. Peers attach through /proc/<creator-pid>/fd/<N>.
"""

from __future__ import annotations

import ctypes
import os

# memfd_create(2) flag; identical on every Linux architecture.
_MFD_CLOEXEC = 1


def create_memfd(name: str) -> int:
    """Create an anonymous, close-on-exec memfd and return its descriptor.

    Some Python builds (e.g. conda's) lack ``os.memfd_create`` although the
    kernel and glibc provide it, so fall back to glibc through ctypes.
    """
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, os.MFD_CLOEXEC)
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "memfd_create"):
        raise OSError("memfd_create is unavailable: glibc 2.27 or newer is required")
    libc.memfd_create.argtypes = (ctypes.c_char_p, ctypes.c_uint)
    libc.memfd_create.restype = ctypes.c_int
    fd = libc.memfd_create(name.encode(), _MFD_CLOEXEC)
    if fd < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"memfd_create({name}) failed: {os.strerror(errno)}")
    return fd
