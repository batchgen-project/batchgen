"""IntakePool: capacity-limited pool of raw requests, partitioned by priority.

Accepts raw (unparsed, untokenized) requests from submit_batch() calls.
Requests are drained into the SchedulingPool at decision boundaries,
highest priority first.

Capacity limit prevents OOM under high load. Rejection is all-or-nothing
per batch: if accepting the next batch would overflow, the entire batch
is rejected.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, Deque, List, Optional

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Request priority levels. Higher value = higher priority."""
    NORMAL = 0
    HIGH = 1


@dataclass
class IntakeEntry:
    """A single raw request waiting to enter the SchedulingPool."""
    request_id: str
    batch_id: str
    raw_request: Dict[str, Any]   # Unparsed JSONL body (messages, max_tokens, etc.)
    priority: Priority
    created_at: float = field(default_factory=time.time)


@dataclass
class IntakeBatchInfo:
    """Tracks per-batch state in the IntakePool."""
    batch_id: str
    total_requests: int
    remaining_in_intake: int      # How many are still in the intake pool
    priority: Priority


class IntakePool:
    """Capacity-limited pool of raw requests, partitioned by priority.

    Thread-safe. Multiple batches can coexist. Higher-priority batch
    entries are drained first.

    Args:
        max_capacity: Maximum total entries across all priorities.
            Default 1,000,000. Set to 0 for unlimited (not recommended).
    """

    def __init__(self, max_capacity: int = 1_000_000) -> None:
        self._high: Deque[IntakeEntry] = deque()
        self._normal: Deque[IntakeEntry] = deque()
        self._lock = threading.Lock()
        self._batch_info: Dict[str, IntakeBatchInfo] = {}
        self._max_capacity = max_capacity

    @property
    def max_capacity(self) -> int:
        return self._max_capacity

    # -------------------- Public API --------------------

    def submit_batch(
        self,
        batch_id: str,
        entries: List[IntakeEntry],
        priority: Priority = Priority.NORMAL,
    ) -> bool:
        """Add a batch of entries to the pool.

        All-or-nothing: if accepting would overflow capacity, the entire
        batch is rejected and False is returned.

        Args:
            batch_id: Unique batch identifier.
            entries: Pre-constructed IntakeEntry objects.
            priority: Batch-level priority (for tracking).

        Returns:
            True if accepted, False if rejected (capacity exceeded).
        """
        with self._lock:
            current = len(self._high) + len(self._normal)
            if self._max_capacity > 0 and current + len(entries) > self._max_capacity:
                logger.warning(
                    f"[INTAKE] Batch {batch_id} REJECTED: {len(entries)} entries "
                    f"would exceed capacity ({current}/{self._max_capacity})"
                )
                return False

            self._batch_info[batch_id] = IntakeBatchInfo(
                batch_id=batch_id,
                total_requests=len(entries),
                remaining_in_intake=len(entries),
                priority=priority,
            )
            for entry in entries:
                if entry.priority == Priority.HIGH:
                    self._high.append(entry)
                else:
                    self._normal.append(entry)

            logger.info(
                f"[INTAKE] Batch {batch_id}: {len(entries)} entries accepted "
                f"(pool: {current + len(entries)}/{self._max_capacity})"
            )
            return True

    def drain(self, max_n: int) -> List[IntakeEntry]:
        """Drain up to *max_n* entries, highest priority first.

        Returns a list of IntakeEntry (may be shorter than max_n if pool
        doesn't have enough entries).
        """
        result: List[IntakeEntry] = []
        with self._lock:
            # Drain HIGH first
            while len(result) < max_n and self._high:
                entry = self._high.popleft()
                result.append(entry)
                self._decrement_batch_remaining(entry.batch_id)

            # Then NORMAL
            while len(result) < max_n and self._normal:
                entry = self._normal.popleft()
                result.append(entry)
                self._decrement_batch_remaining(entry.batch_id)

        if result:
            remaining = self.size()
            logger.info(f"[INTAKE] Drained {len(result)} entries (remaining: {remaining}/{self._max_capacity})")
        return result

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._high) == 0 and len(self._normal) == 0

    def size(self) -> int:
        with self._lock:
            return len(self._high) + len(self._normal)

    def batch_remaining(self, batch_id: str) -> int:
        """Return how many entries of *batch_id* are still in the intake pool."""
        with self._lock:
            info = self._batch_info.get(batch_id)
            return info.remaining_in_intake if info else 0

    def get_batch_info(self, batch_id: str) -> Optional[IntakeBatchInfo]:
        with self._lock:
            return self._batch_info.get(batch_id)

    def remove_batch_info(self, batch_id: str) -> None:
        """Clean up batch tracking info after batch is finalized."""
        with self._lock:
            self._batch_info.pop(batch_id, None)

    # -------------------- Internal --------------------

    def _decrement_batch_remaining(self, batch_id: str) -> None:
        """Decrement remaining count for a batch. Caller must hold _lock."""
        info = self._batch_info.get(batch_id)
        if info is not None:
            info.remaining_in_intake = max(0, info.remaining_in_intake - 1)
