"""Distributed kill switch for coordinated worker shutdown across all nodes.

This module provides a mechanism to propagate failure signals across all workers
in a distributed setup. When any worker fails (exception, signal, stuck, etc.),
the kill signal is broadcast to all other workers so they can exit gracefully.

This is similar to sglang's recursive kill feature.
"""

import atexit
import logging
import os
import signal
import sys
import threading
import time
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
    1. Heartbeat-based health monitoring across all ranks
    2. Fast propagation of kill signals when any worker fails
    3. Graceful shutdown coordination

    Usage:
        kill_switch = DistributedKillSwitch(rank, world_size)
        kill_switch.start()

        # In main loop:
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
        heartbeat_interval: float = 5.0,
        check_interval: float = 0.1,
    ):
        """Initialize the distributed kill switch.

        Args:
            rank: This worker's global rank
            world_size: Total number of workers
            device: CUDA device for tensor operations
            heartbeat_interval: Seconds between heartbeat broadcasts
            check_interval: Seconds between kill signal checks
        """
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.heartbeat_interval = heartbeat_interval
        self.check_interval = check_interval

        # State
        self._kill_triggered = threading.Event()
        self._kill_reason: Optional[str] = None
        self._started = False
        self._stop_event = threading.Event()

        # Thread for background monitoring
        self._monitor_thread: Optional[threading.Thread] = None

        # Tensor for kill signal broadcast (allocated once for efficiency)
        self._kill_tensor = torch.zeros(1, dtype=torch.int32, device=device)

        # Register cleanup
        atexit.register(self._cleanup)

    def start(self) -> None:
        """Start the kill switch monitoring."""
        if self._started:
            return

        self._started = True
        self._stop_event.clear()

        # Start background monitor thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name=f"kill-switch-{self.rank}",
            daemon=True,
        )
        self._monitor_thread.start()

        logger.info(
            f"Rank {self.rank}: Distributed kill switch started "
            f"(heartbeat={self.heartbeat_interval}s)"
        )

    def stop(self) -> None:
        """Stop the kill switch monitoring."""
        if not self._started:
            return

        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self._started = False

    def trigger_kill(self, reason: str = "Unknown") -> None:
        """Trigger a distributed kill signal.

        This should be called when a worker encounters a fatal error.
        The kill signal will be propagated to all other workers.

        Args:
            reason: Human-readable reason for the kill
        """
        if self._kill_triggered.is_set():
            return  # Already triggered

        self._kill_reason = reason
        self._kill_triggered.set()
        logger.error(f"Rank {self.rank}: Kill switch triggered - {reason}")

    def should_exit(self) -> bool:
        """Check if workers should exit.

        Returns:
            True if kill signal has been received/triggered
        """
        return self._kill_triggered.is_set()

    @property
    def kill_reason(self) -> Optional[str]:
        """Get the reason for the kill signal."""
        return self._kill_reason

    def check_and_propagate(self) -> bool:
        """Check for kill signals and propagate if needed.

        This should be called periodically in the main loop.
        It performs a fast all_reduce to check if any worker has triggered kill.

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

    def broadcast_shutdown(self) -> None:
        """Broadcast graceful shutdown signal to all workers.

        Unlike trigger_kill, this signals a clean shutdown rather than error.
        """
        if not dist.is_initialized():
            return

        try:
            self._kill_tensor.fill_(KILL_SIGNAL_SHUTDOWN)
            dist.all_reduce(self._kill_tensor, op=dist.ReduceOp.MAX)
            self._kill_triggered.set()
            self._kill_reason = "Graceful shutdown"
            logger.info(f"Rank {self.rank}: Broadcast shutdown signal")
        except Exception as e:
            logger.warning(f"Rank {self.rank}: Failed to broadcast shutdown: {e}")

    def _monitor_loop(self) -> None:
        """Background thread that periodically checks for kill signals."""
        last_heartbeat = time.time()

        while not self._stop_event.is_set():
            try:
                current_time = time.time()

                # Periodic heartbeat check
                if current_time - last_heartbeat >= self.heartbeat_interval:
                    if not self.check_and_propagate():
                        # Kill signal detected, initiate shutdown
                        self._initiate_shutdown()
                        return
                    last_heartbeat = current_time

                # Check if we should exit
                if self._kill_triggered.is_set():
                    # Give a moment for the signal to propagate
                    time.sleep(0.5)
                    self._initiate_shutdown()
                    return

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Rank {self.rank}: Monitor thread error - {e}")
                self.trigger_kill(f"Monitor thread crashed: {e}")
                self._initiate_shutdown()
                return

    def _initiate_shutdown(self) -> None:
        """Initiate worker shutdown after kill signal."""
        logger.info(
            f"Rank {self.rank}: Initiating shutdown - {self._kill_reason}"
        )

        # Try to do one final broadcast to ensure all workers know
        try:
            if dist.is_initialized():
                self._kill_tensor.fill_(KILL_SIGNAL_ERROR)
                dist.all_reduce(self._kill_tensor, op=dist.ReduceOp.MAX)
        except Exception:
            pass  # Best effort

        # Signal main thread to exit
        # The main loop should check should_exit() and handle cleanup

    def _cleanup(self) -> None:
        """Cleanup on exit."""
        self._stop_event.set()


def install_kill_switch_signal_handlers(
    kill_switch: DistributedKillSwitch,
) -> None:
    """Install signal handlers that trigger the kill switch.

    Args:
        kill_switch: The DistributedKillSwitch instance to trigger on signals
    """
    def signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, triggering distributed kill...")
        kill_switch.trigger_kill(f"Received signal {sig_name}")

        # Give time for kill signal to propagate
        time.sleep(1.0)

        # Force exit if still running
        os._exit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.debug("Kill switch signal handlers installed")


def create_kill_switch(
    rank: int,
    world_size: int,
    local_rank: int,
    heartbeat_interval: float = 5.0,
) -> DistributedKillSwitch:
    """Factory function to create and start a kill switch.

    Args:
        rank: Global rank of this worker
        world_size: Total number of workers
        local_rank: Local rank (for CUDA device)
        heartbeat_interval: Seconds between heartbeat checks

    Returns:
        Started DistributedKillSwitch instance
    """
    device = torch.device(f"cuda:{local_rank}")
    kill_switch = DistributedKillSwitch(
        rank=rank,
        world_size=world_size,
        device=device,
        heartbeat_interval=heartbeat_interval,
    )
    kill_switch.start()
    install_kill_switch_signal_handlers(kill_switch)
    return kill_switch
