"""Process management utilities."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

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
    # Mixtral models
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 96 * 1024**3,
    "mistralai/Mixtral-8x22B-Instruct-v0.1": 286 * 1024**3,
    # Kimi K2.5: ~580GB INT4 experts + ~20GB BF16 (attn/shared/embed) ≈ 600GB + buffer
    "moonshotai/Kimi-K2.5": 650 * 1024**3,
    # GLM-5: 1350GB BF16 experts + 17.3GB attn+DSA + 5.4GB shared + 4.5GB embed/dense ≈ 1380GB + buffer
    "zai-org/GLM-5-FP8": 1400 * 1024**3,
    "zai-org/GLM-5": 1400 * 1024**3,
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


def cleanup_hugepages_files(prefix: Optional[str] = None) -> int:
    """Clean up files in /dev/hugepages safely using Python.

    This function only deletes files matching known BatchGen prefixes,
    avoiding the unsafe 'rm -rf /dev/hugepages/*' pattern.

    Args:
        prefix: Specific prefix to match. If None, matches all
               known BATCHGEN_SHM_PREFIXES.

    Returns:
        Number of files removed.
    """
    import shutil

    hugepages_dir = Path("/dev/hugepages")
    if not hugepages_dir.exists():
        logger.debug("/dev/hugepages does not exist")
        return 0

    # Determine which prefixes to match
    prefixes = (prefix,) if prefix else BATCHGEN_SHM_PREFIXES
    removed = 0

    try:
        for entry in hugepages_dir.iterdir():
            # Only delete entries matching our prefixes
            if not any(entry.name.startswith(p) for p in prefixes):
                continue

            try:
                if entry.is_file() or entry.is_symlink():
                    entry.unlink()
                elif entry.is_dir():
                    shutil.rmtree(entry)
                logger.debug(f"Removed /dev/hugepages/{entry.name}")
                removed += 1
            except PermissionError:
                logger.warning(f"Permission denied: /dev/hugepages/{entry.name}")
            except OSError as e:
                logger.warning(f"Failed to remove /dev/hugepages/{entry.name}: {e}")

    except (PermissionError, OSError) as e:
        logger.warning(f"Error accessing /dev/hugepages: {e}")

    if removed > 0:
        logger.info(f"Cleaned up {removed} files from /dev/hugepages")

    return removed


def unmount_hugetlbfs() -> bool:
    """Unmount all hugetlbfs filesystems.

    Returns:
        True if unmount was successful or no hugetlbfs mounted, False otherwise.
    """
    try:
        # Check if hugetlbfs is mounted
        result = subprocess.run(
            ["mount"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "hugetlbfs" not in result.stdout:
            logger.debug("No hugetlbfs filesystems mounted")
            return True

        # Unmount all hugetlbfs mounts
        max_attempts = 10
        for attempt in range(max_attempts):
            result = subprocess.run(
                ["mount"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if "hugetlbfs" not in result.stdout:
                logger.info("All hugetlbfs filesystems unmounted")
                return True

            # Try to unmount /dev/hugepages
            umount_result = subprocess.run(
                ["umount", "/dev/hugepages"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if umount_result.returncode != 0:
                # umount failed, might need to wait for processes to release
                if attempt < max_attempts - 1:
                    logger.debug(
                        f"Unmount attempt {attempt + 1} failed, retrying..."
                    )
                    time.sleep(0.5)
                else:
                    logger.warning(
                        f"Failed to unmount hugetlbfs: {umount_result.stderr}"
                    )
                    return False

        return True
    except subprocess.TimeoutExpired:
        logger.warning("Timeout while unmounting hugetlbfs")
        return False
    except Exception as e:
        logger.warning(f"Error unmounting hugetlbfs: {e}")
        return False


def reset_hugepages_allocation() -> bool:
    """Reset huge pages allocation to 0.

    Runs both sysctl and direct /proc write for robustness.

    Returns:
        True if at least one method succeeded, False otherwise.
    """
    success = False

    # Method 1: sysctl -w vm.nr_hugepages=0
    try:
        result = subprocess.run(
            ["sysctl", "-w", "vm.nr_hugepages=0"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("Reset vm.nr_hugepages to 0 via sysctl")
            success = True
        else:
            logger.warning(f"sysctl failed: {result.stderr}")
    except subprocess.TimeoutExpired:
        logger.warning("Timeout with sysctl")
    except Exception as e:
        logger.warning(f"sysctl error: {e}")

    # Method 2: echo 0 > /proc/sys/vm/nr_hugepages
    try:
        with open("/proc/sys/vm/nr_hugepages", "w") as f:
            f.write("0\n")
        logger.info("Reset vm.nr_hugepages to 0 via /proc")
        success = True
    except PermissionError:
        logger.warning("Permission denied writing to /proc/sys/vm/nr_hugepages (need root)")
    except Exception as e:
        logger.warning(f"Failed to reset hugepages via /proc: {e}")

    return success


def cleanup_resources(
    shm_prefix: Optional[str] = "batchgen",
    clean_hugepages: bool = True,
    kill_workers: bool = True,
    worker_pids: Optional[List[int]] = None,
) -> None:
    """Comprehensive resource cleanup for BatchGen.

    This function should be called during server shutdown to clean up:
    1. Worker processes (if kill_workers=True)
    2. Shared memory files in /dev/shm
    3. Files in /dev/hugepages
    4. Unmount hugetlbfs and reset hugepages allocation

    Args:
        shm_prefix: Prefix for shared memory files to clean. None cleans all.
        clean_hugepages: Whether to clean and unmount hugepages.
        kill_workers: Whether to kill worker processes.
        worker_pids: List of worker PIDs to kill. If None and kill_workers=True,
                    kills all child processes.
    """
    logger.info("Starting resource cleanup...")

    # Step 1: Kill worker processes
    if kill_workers:
        if worker_pids:
            for pid in worker_pids:
                try:
                    proc = psutil.Process(pid)
                    proc.kill()
                    logger.debug(f"Killed worker process {pid}")
                except psutil.NoSuchProcess:
                    pass
                except Exception as e:
                    logger.warning(f"Failed to kill process {pid}: {e}")
        else:
            # Kill all child processes of current process
            kill_process_tree(os.getpid(), include_parent=False)

    # Step 2: Clean shared memory
    cleanup_shm_files(shm_prefix)

    # Step 3: Clean hugepages
    if clean_hugepages:
        cleanup_hugepages_files()
        unmount_hugetlbfs()
        reset_hugepages_allocation()

    logger.info("Resource cleanup completed")


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
