"""Process management utilities."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

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

# Run-owned record of this run's exact model SHM names. Model weight and
# tensor-metadata regions carry random names, so a supervisor can only tell
# them apart from foreign objects through this private provenance file.
MODEL_SHM_PROVENANCE_FILE = "model_shm.json"
_RECORDED_SHM_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}\Z")

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
    # MiniMax-M2.5: ~225GB FP8 experts + ~8GB BF16 (attn/embed/router) ≈ 233GB
    "MiniMaxAI/MiniMax-M2.5": 250 * 1024**3,
    # GLM-5-FP8: 675GB FP8 experts + 17.3GB attn+DSA + 4.5GB embed/dense ≈ 700GB + buffer
    "zai-org/GLM-5-FP8": 760 * 1024**3,
    # GLM-5: 1350GB BF16 experts + 17.3GB attn+DSA + 5.4GB shared + 4.5GB embed/dense ≈ 1380GB + buffer
    "zai-org/GLM-5": 1400 * 1024**3,
    # GLM-5.1 / GLM-5.1-FP8: same architecture + param count as GLM-5 (754B), same sizes.
    "zai-org/GLM-5.1-FP8": 760 * 1024**3,
    "zai-org/GLM-5.1": 1400 * 1024**3,
}

# Default byte_size when model not in lookup (700GB for backwards compatibility)
DEFAULT_MODEL_BYTE_SIZE = 700 * 1024**3


def get_model_byte_size(model_name: str) -> int:
    """Get model byte_size from lookup table.

    Args:
        model_name: HuggingFace model name (e.g., "openai/gpt-oss-120b")

    Returns:
        Byte size for the model. Returns DEFAULT_MODEL_BYTE_SIZE if not found.
    """
    # Try exact match first
    if model_name in MODEL_BYTE_SIZES:
        return MODEL_BYTE_SIZES[model_name]

    # Try case-insensitive partial match
    model_lower = model_name.lower()
    for key, value in MODEL_BYTE_SIZES.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return value

    logger.warning(
        f"Model '{model_name}' not in byte_size lookup, using default {DEFAULT_MODEL_BYTE_SIZE / (1024**3):.0f} GB"
    )
    return DEFAULT_MODEL_BYTE_SIZE


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


def cleanup_shm_files(shm_prefix: Optional[str] = "batchgen") -> int:
    """Clean up shared memory files in /dev/shm safely using Python.

    This function only deletes files matching known BatchGen prefixes,
    avoiding the unsafe 'rm -rf /dev/shm/*' pattern that could affect
    other applications.

    Args:
        shm_prefix: Prefix to match shared memory files. If None, matches all
                   known BATCHGEN_SHM_PREFIXES. Default is 'batchgen'.

    Returns:
        Number of files removed.
    """
    shm_dir = Path("/dev/shm")
    if not shm_dir.exists():
        logger.debug("/dev/shm does not exist")
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
                logger.debug(f"Removed /dev/shm/{entry.name}")
                removed += 1
            except PermissionError:
                logger.warning(f"Permission denied: /dev/shm/{entry.name}")
            except OSError as e:
                logger.warning(f"Failed to remove /dev/shm/{entry.name}: {e}")

    except (PermissionError, OSError) as e:
        logger.warning(f"Error accessing /dev/shm: {e}")

    if removed > 0:
        logger.info(f"Cleaned up {removed} shared memory files from /dev/shm")

    return removed


MODEL_SHM_KEYS = ("shm_name", "tensor_meta_shm_name")


def _validated_shm_entry_name(name: str) -> str:
    """Return the /dev/shm entry name for a model SHM name, or raise."""
    entry_name = name[1:] if name.startswith("/") else name
    if not entry_name or "/" in entry_name or entry_name in (".", ".."):
        raise ValueError(f"Invalid model SHM name: {name!r}")
    return entry_name


def record_model_shm_provenance(
    model_info: Dict[str, Any], runtime_dir: Path
) -> Optional[Path]:
    """Record this run's exact model SHM names inside its private runtime dir.

    Written once, never overwritten: an existing record means another owner
    claimed this runtime directory, which must fail closed.
    """
    names = []
    for key in MODEL_SHM_KEYS:
        if not model_info.get(key):
            continue
        entry_name = _validated_shm_entry_name(model_info[key])
        if not _RECORDED_SHM_NAME_RE.fullmatch(entry_name):
            raise ValueError(f"Invalid model SHM name: {model_info[key]!r}")
        names.append(entry_name)
    if not names:
        return None
    path = Path(runtime_dir) / MODEL_SHM_PROVENANCE_FILE
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        payload = json.dumps({"version": 1, "shm_names": names}, sort_keys=True)
        os.write(fd, payload.encode() + b"\n")
        os.fsync(fd)
    finally:
        os.close(fd)
    return path


def verify_model_shm_absent(
    model_info: Dict[str, Any], *, shm_dir: Path = Path("/dev/shm"),
    hugepages_dir: Optional[Path] = None,
) -> None:
    """Verify the C++ owner released its model SHM; never unlink by name here."""
    paths = []
    for key in MODEL_SHM_KEYS:
        name = model_info.get(key)
        if name:
            entry = _validated_shm_entry_name(name)
            paths.append(shm_dir / entry)
            if key == "shm_name" and hugepages_dir is not None:
                paths.append(hugepages_dir / entry)

    if paths and not shm_dir.is_dir():
        raise RuntimeError(f"Model SHM directory is unavailable: {shm_dir}")
    for path in paths:
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"Model SHM remains after owner release: {path}")

    for key in MODEL_SHM_KEYS:
        model_info.pop(key, None)


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
        os._exit(128 + signum)

    # Install handlers for common termination signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.debug("Worker signal handlers installed")
