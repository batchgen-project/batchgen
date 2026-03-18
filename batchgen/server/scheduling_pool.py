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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

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

    @property
    def is_complete(self) -> bool:
        return (self.completed_requests + self.failed_requests) >= self.total_requests

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
            RuntimeError: If no free slots are available.
        """
        with self._lock:
            if not self._free_slots:
                raise RuntimeError(
                    f"SchedulingPool: no free slots (capacity={self._capacity}, "
                    f"active={len(self._active_slots)})"
                )
            slot = self._free_slots.pop()
            self._active_slots[request_id] = slot
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

        Returns:
            True if this was the last request in the batch (batch is now complete).
        """
        with self._lock:
            tracker = self._batch_trackers.get(batch_id)
            if tracker is None:
                logger.warning(
                    "SchedulingPool: mark_request_completed for unknown batch %s",
                    batch_id,
                )
                return False
            tracker.completed_requests += 1
            return tracker.is_complete

    def mark_request_failed(self, request_id: str, batch_id: str) -> bool:
        """Mark a request as failed and update its batch tracker.

        Returns:
            True if this was the last request in the batch.
        """
        with self._lock:
            tracker = self._batch_trackers.get(batch_id)
            if tracker is None:
                return False
            tracker.failed_requests += 1
            return tracker.is_complete

    def get_batch_tracker(self, batch_id: str) -> Optional[BatchTracker]:
        with self._lock:
            return self._batch_trackers.get(batch_id)

    def remove_batch_tracker(self, batch_id: str) -> Optional[BatchTracker]:
        """Remove and return a batch tracker (e.g., after batch completes)."""
        with self._lock:
            return self._batch_trackers.pop(batch_id, None)

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
