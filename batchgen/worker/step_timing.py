"""Continuous-decode step timing reported in the rank-0 ``[DECODE]`` heartbeat.

Two complementary measurements of one decode step:

* A CPU wall split. Each segment is the host time between two fixed points
  of the step. The host blocks at whichever call first waits on the device,
  so a segment containing such a wait also absorbs device time that is not
  its own: the split says where the HOST waits, not where the GPU works.
* A device-side forward time from a fixed ring of reusable timing-event
  pairs that bracket the forward+sample segment on the current stream.

Host-sync contract for the steady-state decode path: nothing in this module
makes the host wait for the device. ``ForwardEventRing`` only records and
queries events. ``elapsed_time`` is read from a pair only after BOTH of its
events have reported ``query() == True``, and a step that finds no free slot
is not timed rather than waiting for one to free up.

The module is torch-free: the caller injects the event factory, so the ring
logic is testable on CPU with fake events.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple


# CPU wall split layout: slot 0 counts steps, slot 1 sums whole-step wall ms,
# then one slot per segment below, in step order.
DECODE_STEP_SPLIT_SEGMENTS: Tuple[str, ...] = (
    # step start -> forward launch begins: batch metadata / host-side prep
    "setup",
    # forward + sampling launch (graph replay or eager); host wall only
    "fwd_launch",
    # sampled-token D2H copy launch + host-KV D2H append-task launch
    "kv_launch",
    # host wait for the sampled-token copy
    "readback",
    # host wait for this step's host-KV append tasks
    "kv_drain",
    # per-sequence token / EOS / repetition updates
    "bookkeeping",
)
DECODE_STEP_SPLIT_SLOTS = 2 + len(DECODE_STEP_SPLIT_SEGMENTS)

# Reusable timing-event pairs per decode round. The decode loop harvests the
# current step's pair after the sampled-token readback, so in steady state
# at most one pair is outstanding; the slack only matters if a pair is still
# incomplete when harvested.
FORWARD_EVENT_RING_SLOTS = 64


def new_decode_step_split() -> List[float]:
    """Return a zeroed CPU wall split accumulator."""
    return [0.0] * DECODE_STEP_SPLIT_SLOTS


def accumulate_decode_step_split(split: List[float], marks: Sequence[float]) -> None:
    """Add one step to ``split``.

    ``marks`` are ``time.perf_counter()`` readings in seconds: the step start
    followed by the end of each segment in ``DECODE_STEP_SPLIT_SEGMENTS``.
    """
    if len(split) != DECODE_STEP_SPLIT_SLOTS:
        raise ValueError(
            f"decode step split has {len(split)} slots, expected {DECODE_STEP_SPLIT_SLOTS}"
        )
    if len(marks) != len(DECODE_STEP_SPLIT_SEGMENTS) + 1:
        raise ValueError(
            f"decode step split needs {len(DECODE_STEP_SPLIT_SEGMENTS) + 1} time marks, "
            f"got {len(marks)}"
        )
    split[0] += 1
    split[1] += (marks[-1] - marks[0]) * 1000.0
    for i in range(len(DECODE_STEP_SPLIT_SEGMENTS)):
        split[2 + i] += (marks[i + 1] - marks[i]) * 1000.0


def format_decode_step_split(split: Sequence[float]) -> str:
    """Render ``step_ms=S (setup a, fwd_launch b, ...)`` as per-step means."""
    n = split[0]
    if n <= 0:
        raise ValueError("cannot format an empty decode step split")
    segments = ", ".join(
        f"{name} {split[2 + i] / n:.1f}"
        for i, name in enumerate(DECODE_STEP_SPLIT_SEGMENTS)
    )
    return f"step_ms={split[1] / n:.1f} ({segments})"


class TimingEvent(Protocol):
    """The subset of ``torch.cuda.Event(enable_timing=True)`` the ring uses."""

    def record(self, stream: Any = None) -> None: ...

    def query(self) -> bool: ...

    def elapsed_time(self, end_event: Any) -> float: ...


class ForwardEventRing:
    """Fixed ring of reusable start/end timing-event pairs.

    ``begin`` takes a free slot and records its start event, or returns
    ``None`` when every slot is still outstanding: that step is not timed,
    and the call never waits. ``end`` records the end event and queues the
    slot for harvest. ``harvest`` is non-blocking: it reads ``elapsed_time``
    only from pairs whose start and end events both report ``query()`` True,
    returns those slots to the free list, and leaves the rest queued.

    Every slot is always in exactly one of: free, open (begun, not ended),
    pending (ended, not yet harvested).
    """

    def __init__(self, num_slots: int, event_factory: Callable[[], TimingEvent]):
        if num_slots <= 0:
            raise ValueError(f"ForwardEventRing needs at least one slot, got {num_slots}")
        self._pairs = [(event_factory(), event_factory()) for _ in range(num_slots)]
        self._free: List[int] = list(range(num_slots))
        self._pending: List[int] = []
        self._sum_ms = 0.0
        self._max_ms = 0.0
        self._count = 0

    @property
    def num_slots(self) -> int:
        return len(self._pairs)

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_pending(self) -> int:
        return len(self._pending)

    def begin(self, stream: Any) -> Optional[int]:
        """Record a start event in a free slot; ``None`` if the ring is full."""
        if not self._free:
            return None
        slot = self._free.pop()
        self._pairs[slot][0].record(stream)
        return slot

    def end(self, slot: Optional[int], stream: Any) -> None:
        """Record the end event for a slot from ``begin``; no-op for ``None``."""
        if slot is None:
            return
        self._pairs[slot][1].record(stream)
        self._pending.append(slot)

    def harvest(self) -> None:
        """Collect every pending pair whose events have both completed."""
        if not self._pending:
            return
        still_pending = []
        for slot in self._pending:
            start, end = self._pairs[slot]
            if start.query() and end.query():
                elapsed_ms = start.elapsed_time(end)
                self._sum_ms += elapsed_ms
                self._count += 1
                if elapsed_ms > self._max_ms:
                    self._max_ms = elapsed_ms
                self._free.append(slot)
            else:
                still_pending.append(slot)
        self._pending = still_pending

    def take_summary(self) -> str:
        """Return `` gpu_fwd_ms=M gpu_fwd_max=X (n=N)`` and reset the totals.

        Empty when no pair has been harvested since the previous call.
        """
        if self._count == 0:
            return ""
        text = (
            f" gpu_fwd_ms={self._sum_ms / self._count:.1f} "
            f"gpu_fwd_max={self._max_ms:.1f} (n={self._count})"
        )
        self._sum_ms = 0.0
        self._max_ms = 0.0
        self._count = 0
        return text
