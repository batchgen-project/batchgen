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
    """Clean up shared memory files in /dev/shm.

    Args:
        shm_prefix: Prefix to match shared memory files. If None, clean all
                   files. Default is 'batchgen'.

    Returns:
        Number of files removed.
    """
    shm_dir = Path("/dev/shm")
    if not shm_dir.exists():
        return 0

    removed = 0
    try:
        for shm_file in shm_dir.iterdir():
            if shm_file.is_file():
                if shm_prefix is None or shm_file.name.startswith(shm_prefix):
                    try:
                        shm_file.unlink()
                        logger.debug(f"Removed /dev/shm/{shm_file.name}")
                        removed += 1
                    except (PermissionError, OSError) as e:
                        logger.warning(f"Failed to remove {shm_file}: {e}")
    except Exception as e:
        logger.warning(f"Error cleaning /dev/shm: {e}")

    if removed > 0:
        logger.info(f"Cleaned up {removed} shared memory files from /dev/shm")
    return removed


def cleanup_hugepages_files() -> int:
    """Clean up files in /dev/hugepages.

    Returns:
        Number of files removed.
    """
    hugepages_dir = Path("/dev/hugepages")
    if not hugepages_dir.exists():
        return 0

    removed = 0
    try:
        for hp_file in hugepages_dir.iterdir():
            if hp_file.is_file():
                try:
                    hp_file.unlink()
                    logger.debug(f"Removed /dev/hugepages/{hp_file.name}")
                    removed += 1
                except (PermissionError, OSError) as e:
                    logger.warning(f"Failed to remove {hp_file}: {e}")
    except Exception as e:
        logger.warning(f"Error cleaning /dev/hugepages: {e}")

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

    Returns:
        True if successful, False otherwise.
    """
    try:
        result = subprocess.run(
            ["sysctl", "-w", "vm.nr_hugepages=0"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("Reset vm.nr_hugepages to 0")
            return True
        else:
            logger.warning(f"Failed to reset hugepages: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("Timeout while resetting hugepages")
        return False
    except Exception as e:
        logger.warning(f"Error resetting hugepages: {e}")
        return False


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
