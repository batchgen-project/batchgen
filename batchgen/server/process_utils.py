"""Process management utilities."""

from __future__ import annotations

import ctypes
import logging
import multiprocessing
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import psutil

logger = logging.getLogger(__name__)

# Known BatchGen shared memory prefixes
# These are used for safe cleanup to avoid deleting files from other applications
BATCHGEN_SHM_PREFIXES = (
    "shm_",           # Parameter server main memory: /shm_<uuid>
    "skel_",          # Skeleton state dict: skel_<timestamp>_<random>
    "batchgen_skel_", # Temp skeleton files: batchgen_skel_*.pt
    "batchgen_",      # General BatchGen prefix
)

# The model weight and tensor-metadata regions are anonymous memfds in every
# mode, hugetlbfs included, so this run creates no named model region for a
# supervisor to attribute and records no model SHM provenance.

# Default hugepage size (2MB) used as fallback if detection fails
DEFAULT_HUGEPAGE_SIZE = 2 * 1024 * 1024

# Model byte_size lookup table
# These values are from each model's parameter_server.py
# Used to calculate required hugepages before model loading
MODEL_BYTE_SIZES = {
    # GPT-OSS-120B: ~65GB total (61GB MXFP4 experts + 4GB BF16 attn/embed)
    "openai/gpt-oss-120b": 70 * 1024**3,
    # DeepSeek models
    "deepseek-ai/DeepSeek-V2-Lite": 32 * 1024**3,
    "deepseek-ai/DeepSeek-V2": 472 * 1024**3,
    "deepseek-ai/DeepSeek-V3": 675 * 1024**3,
    "deepseek-ai/DeepSeek-R1": 675 * 1024**3,  # Same as V3
    "deepseek-ai/DeepSeek-V4-Flash": 180 * 1024**3,
    "deepseek-ai/DeepSeek-V4-Pro": 700 * 1024**3,
    # Mixtral models
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 96 * 1024**3,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 286 * 1024**3,
    # Kimi K2.5: ~580GB INT4 experts + ~20GB BF16 (attn/shared/embed) ≈ 600GB + buffer
    "moonshotai/Kimi-K2.5": 650 * 1024**3,
    "moonshotai/Kimi-K2.6": 650 * 1024**3,
    # Kimi-Linear-48B-A3B: ~96 GiB BF16 total; 110 GiB with buffer, the
    # constant its parameter server reserves (_KIMI_LINEAR_48B_BYTE_SIZE).
    "moonshotai/Kimi-Linear-48B-A3B-Instruct": 110 * 1024**3,
    # Kimi-K3: the released checkpoint's metadata.total_size
    # (1,560,860,324,864 B) plus one 4 KiB alignment unit per shard (96), the
    # reservation k3_shm_byte_size computes from that index. Kimi-K3.1-Pro is
    # the same rule on its own index (1,561,217,377,280 B, 96 shards).
    "moonshotai/Kimi-K3": 1_560_860_324_864 + 96 * 4096,
    "moonshotai/Kimi-K3.1-Pro": 1_561_217_377_280 + 96 * 4096,
    # MiniMax-M2.5: ~225GB FP8 experts + ~8GB BF16 (attn/embed/router) ≈ 233GB
    "MiniMaxAI/MiniMax-M2.5": 250 * 1024**3,
    # GLM-5-FP8: 675GB FP8 experts + 17.3GB attn+DSA + 4.5GB embed/dense ≈ 700GB + buffer
    "zai-org/GLM-5-FP8": 760 * 1024**3,
    # GLM-5: 1350GB BF16 experts + 17.3GB attn+DSA + 5.4GB shared + 4.5GB embed/dense ≈ 1380GB + buffer
    "zai-org/GLM-5": 1400 * 1024**3,
    # GLM-5.1 / GLM-5.1-FP8: same architecture + param count as GLM-5 (754B), same sizes.
    "zai-org/GLM-5.1-FP8": 760 * 1024**3,
    "zai-org/GLM-5.1": 1400 * 1024**3,
    # GLM-5.2 / GLM-5.2-FP8: same architecture family + sizes as GLM-5.1.
    "zai-org/GLM-5.2-FP8": 760 * 1024**3,
    "zai-org/GLM-5.2": 1400 * 1024**3,
    # GLM-5.3 FP8 has the same released tensor total and host-weight envelope
    # as GLM-5.2 FP8.
    "zai-org/GLM-5.3-FP8": 760 * 1024**3,
    "zai-org/GLM-5.3": 1400 * 1024**3,
}


def _model_byte_size_index():
    from batchgen.config import model_identity

    return model_identity.build_index(MODEL_BYTE_SIZES)


_MODEL_BYTE_SIZE_INDEX = _model_byte_size_index()


class UnknownModelByteSizeError(ValueError):
    """``--model`` names no model with a host weight envelope."""


def get_model_byte_size(model_name: str) -> int:
    """Host weight envelope of ``model_name``, in bytes.

    Args:
        model_name: HuggingFace model name (e.g., "openai/gpt-oss-120b")

    Raises:
        UnknownModelByteSizeError: ``model_name`` names no entry of
            ``MODEL_BYTE_SIZES``. A neighbouring entry or a default would
            size the hugepage pool and the weight region for another model.
    """
    # Exact match first, then the registered id the value names (a local
    # directory named after it, or a Base/Instruct/dated variant of it).
    if model_name in MODEL_BYTE_SIZES:
        return MODEL_BYTE_SIZES[model_name]
    from batchgen.config import model_identity

    named = model_identity.lookup(_MODEL_BYTE_SIZE_INDEX, model_name)
    if named is not None:
        return named

    raise UnknownModelByteSizeError(
        f"Model {model_name!r} has no host weight envelope. Add an exact entry "
        "keyed by its released Hugging Face id to "
        "batchgen.server.process_utils.MODEL_BYTE_SIZES; a local directory "
        "named after that id and its Base/Instruct/dated variants then "
        "resolve to it."
    )


def get_hugepage_size() -> int:
    """Get system hugepage size in bytes from /proc/meminfo.

    Hugepage sizes vary by architecture:
    - x86_64: 2 MB (default) or 1 GB
    - ARM64: 2 MB (default) or 1 GB
    - ARM64 (64K pages): 512 MB

    Returns:
        Hugepage size in bytes. Defaults to 2MB if detection fails.
    """
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("Hugepagesize:"):
                    # Format: "Hugepagesize:     2048 kB"
                    parts = line.split()
                    size_kb = int(parts[1])
                    return size_kb * 1024  # Convert KB to bytes
    except (IOError, ValueError, IndexError) as e:
        logger.debug(f"Failed to read hugepage size from /proc/meminfo: {e}")

    # Default to 2MB if detection fails
    return DEFAULT_HUGEPAGE_SIZE


def calculate_hugepages(byte_size: int) -> int:
    """Calculate required hugepages for given model size.

    NOTE: byte_size values from model configs (70GB, 675GB, etc.)
    already include buffer, so no additional buffer is added.

    Args:
        byte_size: Model size in bytes (already includes buffer)

    Returns:
        Number of hugepages required
    """
    hugepage_size = get_hugepage_size()
    num_pages = (byte_size + hugepage_size - 1) // hugepage_size  # ceil division

    logger.info(
        f"Hugepages: {byte_size / (1024**3):.1f} GB model, "
        f"{hugepage_size / (1024**2):.0f} MB pages, "
        f"{num_pages} pages required"
    )
    return num_pages


def kill_process_tree(
    parent_pid, include_parent: bool = True, skip_pid: int = None
):
    """Kill the process and all its child processes."""
    # Remove sigchld handler to avoid spammy logs.
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGCHLD, signal.SIG_DFL)

    if parent_pid is None:
        parent_pid = os.getpid()
        include_parent = False

    try:
        itself = psutil.Process(parent_pid)
    except psutil.NoSuchProcess:
        return

    children = itself.children(recursive=True)
    for child in children:
        if child.pid == skip_pid:
            continue
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

    if include_parent:
        try:
            if parent_pid == os.getpid():
                itself.kill()
                sys.exit(0)

            itself.kill()

            # Sometime processes cannot be killed with SIGKILL (e.g, PID=1 launched by kubernetes),
            # so we send an additional signal to kill them.
            itself.send_signal(signal.SIGQUIT)
        except psutil.NoSuchProcess:
            pass


def cleanup_shm_files(
    shm_prefix: Optional[str] = "batchgen",
    shm_dir: Optional[Path] = None,
) -> int:
    """Clean up shared memory files in /dev/shm safely using Python.

    This function only deletes files matching known BatchGen prefixes,
    avoiding the unsafe 'rm -rf /dev/shm/*' pattern that could affect
    other applications.

    Args:
        shm_prefix: Prefix to match shared memory files. If None, matches all
                   known BATCHGEN_SHM_PREFIXES. Default is 'batchgen'.
        shm_dir: Directory holding the POSIX shared-memory objects. Defaults to
                 /dev/shm; the dead-run reclaim tests inject a temporary dir.

    Returns:
        Number of files removed.
    """
    shm_dir = Path("/dev/shm") if shm_dir is None else Path(shm_dir)
    if not shm_dir.exists():
        logger.debug("%s does not exist", shm_dir)
        return 0

    # Determine which prefixes to match
    if shm_prefix is not None:
        prefixes = (shm_prefix,)
    else:
        # Clean all known BatchGen prefixes (NOT all files in /dev/shm)
        prefixes = BATCHGEN_SHM_PREFIXES

    removed = 0

    try:
        for entry in shm_dir.iterdir():
            # Skip non-files (directories, sockets, etc.)
            if not entry.is_file():
                continue

            # Only delete files matching our prefixes
            if not any(entry.name.startswith(p) for p in prefixes):
                continue

            try:
                entry.unlink()
                logger.debug(f"Removed {entry}")
                removed += 1
            except PermissionError:
                logger.warning(f"Permission denied: {entry}")
            except OSError as e:
                logger.warning(f"Failed to remove {entry}: {e}")

    except (PermissionError, OSError) as e:
        logger.warning(f"Error accessing {shm_dir}: {e}")

    if removed > 0:
        logger.info(f"Cleaned up {removed} shared memory files from {shm_dir}")

    return removed


def _exit_process(code: int, reason: str) -> None:
    """Write the event log's ``process.exit`` record, then ``os._exit``.

    Callers run in signal handlers and before the worker configured logging,
    so nothing is imported here: a BatchGen worker has loaded
    ``batchgen.eventlog.process`` by then (its entry module imports it). This
    module loaded on its own, by path, logs the plain way instead. The exit
    itself is always this module's ``os._exit``.
    """
    events = sys.modules.get("batchgen.eventlog.process")
    if events is not None:
        events.record_exit(code, reason, logger=logger)
    else:
        logger.critical("process.exit code=%s reason=%s", code, reason)
    os._exit(code)


def install_worker_signal_handlers(
    shutdown_callback: Optional[Callable[[], None]] = None,
) -> None:
    """Install signal handlers for worker processes.

    This enables workers to respond to Ctrl+C and other termination signals
    even when blocked in NCCL operations.

    Args:
        shutdown_callback: Optional callback to execute before exiting.
    """
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Worker received {sig_name}, initiating shutdown...")
        if shutdown_callback:
            try:
                shutdown_callback()
            except Exception as e:
                logger.warning(f"Shutdown callback failed: {e}")
        # Use os._exit to force exit even if blocked in NCCL
        _exit_process(128 + signum, f"worker received {sig_name}")

    # Install handlers for common termination signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.debug("Worker signal handlers installed")


# prctl(2) option number; identical on every Linux architecture.
PR_SET_PDEATHSIG = 1


def _set_pdeathsig() -> None:
    """Arm the kernel parent-death signal (SIGKILL) for this process."""
    libc = ctypes.CDLL(None, use_errno=True)
    ctypes.set_errno(0)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"prctl(PR_SET_PDEATHSIG) failed: {os.strerror(errno)}")


def die_with_parent() -> None:
    """Make the kernel SIGKILL this process when its parent process dies.

    GPU workers must never outlive the server process that spawned them: an
    orphaned worker keeps its CUDA context and shared memory alive.  The signal
    is SIGKILL rather than a catchable one because a Python-level handler cannot
    run while the worker's main thread is inside a native collective (NCCL), so
    a handler-based death signal is not reliable.

    The parent-death signal is delivered when the *thread* that created this
    process exits, so the caller must be spawned from the parent's main thread.

    No-op on platforms without PR_SET_PDEATHSIG.
    """
    if not sys.platform.startswith("linux"):
        return

    _set_pdeathsig()

    # Close the race where the parent died before prctl ran: the kernel would
    # then never deliver the death signal, leaving this process an orphan.
    parent = multiprocessing.parent_process()
    if parent is not None and os.getppid() != parent.pid:
        _exit_process(
            1,
            f"parent process {parent.pid} exited before the parent-death "
            "signal was armed",
        )
