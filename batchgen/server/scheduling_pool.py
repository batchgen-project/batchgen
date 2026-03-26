"""SchedulingPool: fixed-size pool of tokenized sequences with lifecycle management.

Manages QueryBook slots. Every sequence here is tokenized and has status
metadata. The SchedulingPool is where prefill/decode scheduling decisions
are made. When sequences complete, slots are freed and refilled from the
IntakePool at decision boundaries.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from batchgen.server.intake_pool import IntakeEntry, IntakePool, Priority

logger = logging.getLogger(__name__)


@dataclass
class BatchTracker:
    """Tracks completion state for a batch across the scheduling pool."""
    batch_id: str
    total_requests: int
    completed_requests: int = 0
    failed_requests: int = 0
    output_path: Optional[str] = None
    priority: Priority = Priority.NORMAL
    created_at: float = field(default_factory=time.time)
    error: Optional[str] = None  # Set on fatal failure (worker crash, timeout)

    @property
    def is_complete(self) -> bool:
        return (self.completed_requests + self.failed_requests) >= self.total_requests

    @property
    def is_failed(self) -> bool:
        return self.error is not None

    @property
    def pending_requests(self) -> int:
        return self.total_requests - self.completed_requests - self.failed_requests


class SchedulingPool:
    """Fixed-size pool of tokenized sequences with lifecycle management.

    Pre-allocated to *capacity* QueryBook slots at init. Sequences flow
    in from the IntakePool, go through prefill/decode, and on completion
    their slot is freed for reuse.

    Thread-safety: external callers (server thread) and the worker process
    communicate via the mp.Queue; this class is used single-threaded within
    the worker process. The _lock is provided for the server-side
    BatchTracker updates which may be called from the async scheduler.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        # Free slot indices (QueryBook slot IDs available for new sequences)
        self._free_slots: List[int] = list(range(capacity))
        # Active sequences: request_id → slot_index
        self._active_slots: Dict[str, int] = {}
        # Batch tracking
        self._batch_trackers: Dict[str, BatchTracker] = {}
        # Idempotency: track completed request IDs to prevent double-counting
        self._completed_requests: Set[str] = set()
        self._lock = threading.Lock()

    # -------------------- Capacity --------------------

    @property
    def capacity(self) -> int:
        return self._capacity

    def num_free_slots(self) -> int:
        with self._lock:
            return len(self._free_slots)

    def num_active_slots(self) -> int:
        with self._lock:
            return len(self._active_slots)

    def has_free_slots(self) -> bool:
        with self._lock:
            return len(self._free_slots) > 0

    # -------------------- Slot Management --------------------

    def allocate_slot(self, request_id: str) -> int:
        """Allocate a free slot for a new sequence.

        Returns:
            The slot index (for QueryBook buffer mapping).

        Raises:
            ValueError: If request_id is already allocated.
            RuntimeError: If no free slots are available.
        """
        with self._lock:
            if request_id in self._active_slots:
                raise ValueError(
                    f"SchedulingPool: request_id '{request_id}' already allocated "
                    f"(slot {self._active_slots[request_id]})"
                )
            if not self._free_slots:
                raise RuntimeError(
                    f"SchedulingPool: no free slots (capacity={self._capacity}, "
                    f"active={len(self._active_slots)})"
                )
            slot = self._free_slots.pop()
            self._active_slots[request_id] = slot
            logger.debug(
                f"[SCHED] Allocated slot {slot} for {request_id} "
                f"({len(self._free_slots)} free)"
            )
            return slot

    def free_slot(self, request_id: str) -> int:
        """Free a slot previously allocated to *request_id*.

        Returns:
            The freed slot index.

        Raises:
            KeyError: If request_id has no allocated slot.
        """
        with self._lock:
            slot = self._active_slots.pop(request_id)
            self._free_slots.append(slot)
            logger.debug(
                f"[SCHED] Freed slot {slot} for {request_id} "
                f"({len(self._free_slots)} free)"
            )
            return slot

    def get_slot(self, request_id: str) -> Optional[int]:
        """Get the slot index for a request, or None."""
        with self._lock:
            return self._active_slots.get(request_id)

    # -------------------- Batch Tracking --------------------

    def register_batch(
        self,
        batch_id: str,
        total_requests: int,
        priority: Priority = Priority.NORMAL,
        output_path: Optional[str] = None,
    ) -> None:
        """Register a new batch for completion tracking."""
        with self._lock:
            self._batch_trackers[batch_id] = BatchTracker(
                batch_id=batch_id,
                total_requests=total_requests,
                priority=priority,
                output_path=output_path,
            )

    def mark_request_completed(self, request_id: str, batch_id: str) -> bool:
        """Mark a request as completed and update its batch tracker.

        Idempotent: duplicate completions for the same request_id are ignored.

        Returns:
            True if this was the last request in the batch (batch is now complete).
        """
        with self._lock:
            if request_id in self._completed_requests:
                logger.warning(
                    f"[SCHED] Duplicate completion for {request_id} in batch {batch_id}, ignoring"
                )
                return False
            self._completed_requests.add(request_id)

            tracker = self._batch_trackers.get(batch_id)
            if tracker is None:
                logger.warning(
                    f"[SCHED] mark_request_completed for unknown batch {batch_id}"
                )
                return False
            tracker.completed_requests += 1
            is_done = tracker.is_complete
            if is_done:
                logger.info(
                    f"[SCHED] Batch {batch_id}: ALL {tracker.total_requests} requests complete"
                )
            else:
                logger.info(
                    f"[SCHED] Batch {batch_id}: {tracker.completed_requests}/{tracker.total_requests} complete"
                )
            return is_done

    def mark_request_failed(self, request_id: str, batch_id: str) -> bool:
        """Mark a request as failed and update its batch tracker.

        Idempotent: duplicate failures for the same request_id are ignored.

        Returns:
            True if this was the last request in the batch.
        """
        with self._lock:
            if request_id in self._completed_requests:
                logger.warning(
                    f"[SCHED] Duplicate fail for {request_id} in batch {batch_id}, ignoring"
                )
                return False
            self._completed_requests.add(request_id)

            tracker = self._batch_trackers.get(batch_id)
            if tracker is None:
                return False
            tracker.failed_requests += 1
            return tracker.is_complete

    def get_batch_tracker(self, batch_id: str) -> Optional[BatchTracker]:
        with self._lock:
            return self._batch_trackers.get(batch_id)

    def remove_batch_tracker(self, batch_id: str) -> Optional[BatchTracker]:
        """Remove and return a batch tracker (e.g., after batch completes).

        Also cleans up completed request IDs for this batch to prevent
        unbounded growth.
        """
        with self._lock:
            tracker = self._batch_trackers.pop(batch_id, None)
            # Clean up completed request IDs for this batch
            # We don't track per-batch request sets, so we can't selectively clean.
            # Instead, prune if total tracked > 2x active batches' total requests.
            total_expected = sum(t.total_requests for t in self._batch_trackers.values())
            if len(self._completed_requests) > max(total_expected * 2, 10000):
                self._completed_requests.clear()
                logger.info("[SCHED] Pruned completed_requests set")
            return tracker

    # -------------------- Intake Selection --------------------

    def select_from_intake(
        self,
        intake: IntakePool,
        max_n: Optional[int] = None,
    ) -> List[IntakeEntry]:
        """Select entries from the IntakePool to fill free slots.

        Currently uses priority-based drain (HIGH first, then NORMAL).
        Future: prefix-aware selection for KV cache reuse.

        Args:
            intake: The IntakePool to drain from.
            max_n: Maximum entries to select (defaults to number of free slots).

        Returns:
            List of IntakeEntry to admit into the scheduling pool.
        """
        if max_n is None:
            max_n = self.num_free_slots()
        else:
            max_n = min(max_n, self.num_free_slots())

        if max_n <= 0:
            return []

        return intake.drain(max_n)

    # -------------------- Diagnostics --------------------

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"SchedulingPool(capacity={self._capacity}, "
                f"active={len(self._active_slots)}, "
                f"free={len(self._free_slots)}, "
                f"batches={len(self._batch_trackers)})"
            )
