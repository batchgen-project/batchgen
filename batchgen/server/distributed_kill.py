"""Distributed kill switch for coordinated worker shutdown across all nodes.

This module provides a mechanism to propagate failure signals across all workers
in a distributed setup. When any worker fails (exception, signal, stuck, etc.),
the kill signal is broadcast to all other workers so they can exit gracefully.

This is similar to sglang's recursive kill feature.

IMPORTANT: NCCL operations are NOT thread-safe. The check_and_propagate() method
must only be called from the main thread, never from a background thread.
"""

import atexit
import logging
import os
import signal
import threading
from typing import Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# Kill signal values for broadcast
KILL_SIGNAL_ALIVE = 0
KILL_SIGNAL_SHUTDOWN = 1
KILL_SIGNAL_ERROR = 2


class DistributedKillSwitch:
    """Coordinated shutdown mechanism for distributed workers.

    This class provides:
    1. Local kill signal tracking via threading.Event (thread-safe)
    2. Distributed propagation via NCCL all_reduce (must be called from main thread)
    3. Signal handler integration for Ctrl+C handling

    Usage:
        kill_switch = create_kill_switch(rank, world_size, local_rank)

        # In main loop - call check_and_propagate at safe points:
        if not kill_switch.check_and_propagate():
            break

        # Or just check local state (no NCCL):
        if kill_switch.should_exit():
            break

        # On failure:
        kill_switch.trigger_kill("Worker crashed")
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        device: torch.device,
    ):
        """Initialize the distributed kill switch.

        Args:
            rank: This worker's global rank
            world_size: Total number of workers
            device: CUDA device for tensor operations
        """
        self.rank = rank
        self.world_size = world_size
        self.device = device

        # State - thread-safe Event for local kill tracking
        self._kill_triggered = threading.Event()
        self._kill_reason: Optional[str] = None

        # Tensor for kill signal broadcast (allocated once for efficiency)
        self._kill_tensor = torch.zeros(1, dtype=torch.int32, device=device)

        # Register cleanup
        atexit.register(self._cleanup)

    def trigger_kill(self, reason: str = "Unknown") -> None:
        """Trigger a kill signal locally.

        This sets the local kill flag. The signal will be propagated to other
        workers when check_and_propagate() is called.

        Args:
            reason: Human-readable reason for the kill
        """
        if self._kill_triggered.is_set():
            return  # Already triggered

        self._kill_reason = reason
        self._kill_triggered.set()
        logger.error(f"Rank {self.rank}: Kill switch triggered - {reason}")

    def should_exit(self) -> bool:
        """Check if this worker should exit (local check only, no NCCL).

        This is thread-safe and can be called from any thread.

        Returns:
            True if kill signal has been triggered locally
        """
        return self._kill_triggered.is_set()

    @property
    def kill_reason(self) -> Optional[str]:
        """Get the reason for the kill signal."""
        return self._kill_reason

    def check_and_propagate(self) -> bool:
        """Check for kill signals across all workers and propagate if needed.

        IMPORTANT: This method performs NCCL all_reduce and MUST only be called
        from the main thread. Never call from a background thread.

        This should be called periodically in the main loop at safe points
        (e.g., between inference batches, during idle polling).

        Returns:
            True if should continue, False if should exit
        """
        if not dist.is_initialized():
            return not self._kill_triggered.is_set()

        try:
            # Set local status
            if self._kill_triggered.is_set():
                self._kill_tensor.fill_(KILL_SIGNAL_ERROR)
            else:
                self._kill_tensor.fill_(KILL_SIGNAL_ALIVE)

            # All-reduce with MAX to propagate any non-zero (kill) signal
            dist.all_reduce(self._kill_tensor, op=dist.ReduceOp.MAX)

            # Check result
            signal_value = self._kill_tensor.item()
            if signal_value != KILL_SIGNAL_ALIVE:
                if not self._kill_triggered.is_set():
                    self._kill_reason = "Kill signal received from another worker"
                    self._kill_triggered.set()
                    logger.warning(
                        f"Rank {self.rank}: Received kill signal from cluster "
                        f"(value={signal_value})"
                    )
                return False

            return True

        except Exception as e:
            # NCCL error likely means other workers died
            logger.error(f"Rank {self.rank}: Kill check failed - {e}")
            if not self._kill_triggered.is_set():
                self._kill_reason = f"NCCL communication failed: {e}"
                self._kill_triggered.set()
            return False

    def stop(self) -> None:
        """Stop the kill switch (cleanup)."""
        pass  # No background thread to stop

    def _cleanup(self) -> None:
        """Cleanup on exit."""
        pass


def install_kill_switch_signal_handlers(
    kill_switch: DistributedKillSwitch,
) -> None:
    """Install signal handlers that trigger the kill switch.

    Args:
        kill_switch: The DistributedKillSwitch instance to trigger on signals
    """
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, triggering kill switch...")
        kill_switch.trigger_kill(f"Received signal {sig_name}")
        # Don't force exit here - let the main loop handle graceful shutdown
        # The main loop checks should_exit() and will break out

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.debug("Kill switch signal handlers installed")


def create_kill_switch(
    rank: int,
    world_size: int,
    local_rank: int,
) -> DistributedKillSwitch:
    """Factory function to create a kill switch.

    Args:
        rank: Global rank of this worker
        world_size: Total number of workers
        local_rank: Local rank (for CUDA device)

    Returns:
        DistributedKillSwitch instance
    """
    device = torch.device(f"cuda:{local_rank}")
    kill_switch = DistributedKillSwitch(
        rank=rank,
        world_size=world_size,
        device=device,
    )
    install_kill_switch_signal_handlers(kill_switch)
    logger.info(f"Rank {rank}: Distributed kill switch initialized")
    return kill_switch
