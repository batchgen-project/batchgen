"""Depth-1 deferred sampled-token readback for the continuous decode loop.

The decode step's only mandatory device->host dependency is the sampled token
ids: the host needs them for the trajectory book, EOS / length / repetition
stop decisions and output. The NEXT step does not: its input ids are the
previous step's sampled DEVICE tensor (``new_tokens`` is handed to the graph
replay as ``input_ids`` and copied device-to-device into the static input
buffer), and every other per-row input it reads -- context length, position,
host-KV write position -- advances by exactly one per live row, independent of
the token's value.

So the loop can enqueue step N+1 before it waits on step N's tokens, as long
as:

* each in-flight step's tokens land in their OWN pinned buffer (two slots,
  alternated: the host has always read slot ``s`` before step N+2 reuses it);
* the token-independent advance of step N (``decoded_length += 1`` for the
  rows live at step N) happens before step N+1 is set up, and the
  token-dependent half (trajectory append, stop flags) happens after the wait;
* anything that needs host tokens -- the page-boundary planner, input-id
  rebuilds, loop exit -- drains the pending step first.

This module is torch-free (buffer and event factories are injected) so the
ring's slot discipline is testable on CPU, like ``step_timing.ForwardEventRing``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Sequence


class DeferredTokenReadbackError(RuntimeError):
    """A host consumer needed sampled tokens that are still deferred, or the
    depth-1 slot discipline was violated."""


@dataclass
class PendingDecodeStep:
    """One launched decode step whose sampled tokens the host has not read.

    ``row_active`` is the completed-row mask decided when the step was
    launched -- exactly the mask the undeferred step computes after its token
    wait, because nothing between the two points changes completion state.
    The token-dependent bookkeeping must reuse it rather than re-evaluate
    completion: the advance already moved ``decoded_length``.
    """

    slot: int
    bs: int
    batch: List[int]
    batch_sequences: List[Any]
    row_active: List[bool]
    local_iteration: int


class TokenReadbackRing:
    """Two pinned host buffers + completion events, alternated per step.

    ``launch`` issues the non-blocking device->host copy of a step's tokens
    into the next slot and records that slot's event on ``stream``. ``read``
    returns a slot's tokens and frees it; it refuses a slot whose event has
    not completed, so a missing wait is a named error instead of a read of
    whatever the previous step left in the buffer. Launching into a slot that
    is still unread means more than one step is in flight, which this design
    never allows.
    """

    SLOTS = 2

    def __init__(
        self,
        capacity: int,
        buffer_factory: Callable[[int], Any],
        event_factory: Callable[[], Any],
    ) -> None:
        capacity = max(int(capacity), 1)
        self._buffer_factory = buffer_factory
        self._buffers = [buffer_factory(capacity) for _ in range(self.SLOTS)]
        self._events = [event_factory() for _ in range(self.SLOTS)]
        self._in_flight = [False] * self.SLOTS
        self._next = 0

    def launch(self, tokens: Any, bs: int, stream: Any) -> int:
        slot = self._next
        if self._in_flight[slot]:
            raise DeferredTokenReadbackError(
                f"token readback slot {slot} is still unread: launching another "
                "step would put more than one deferred step in flight"
            )
        if bs > int(self._buffers[slot].shape[0]):
            # Only the free slot is regrown; its last copy was already read,
            # so no in-flight copy targets the buffer being replaced.
            self._buffers[slot] = self._buffer_factory(bs)
        self._buffers[slot][:bs].copy_(tokens[:bs], non_blocking=True)
        self._events[slot].record(stream)
        self._in_flight[slot] = True
        self._next = (slot + 1) % self.SLOTS
        return slot

    def event(self, slot: int) -> Any:
        return self._events[slot]

    def read(self, slot: int, bs: int) -> List[int]:
        if not self._in_flight[slot]:
            raise DeferredTokenReadbackError(
                f"token readback slot {slot} has no launched step to read"
            )
        if not self._events[slot].query():
            raise DeferredTokenReadbackError(
                f"token readback slot {slot} read before its copy completed"
            )
        token_ids = self._buffers[slot][:bs].reshape(-1).tolist()
        self._in_flight[slot] = False
        return token_ids

    def abandon(self, slot: int) -> None:
        """Teardown: the slot's tokens will never be read."""
        self._in_flight[slot] = False

    def in_flight(self) -> int:
        return sum(self._in_flight)


def deferred_decode_step_marks(
    forward_start: float,
    t_setup: float,
    t_fwd: float,
    t_copy: float,
    t_wait: float,
    t_drain: float,
    t_consumed: float,
    t_kv_launched: float,
    t_end: float,
) -> Sequence[float]:
    """Map a deferred step's timeline onto the canonical split segments.

    The deferred step runs: setup, forward launch, token-copy launch, wait for
    the PREVIOUS step's tokens, drain its host-KV appends, consume its tokens,
    launch this step's host-KV appends, advance this step's rows. The split
    keeps its segment meaning (``kv_launch`` = token copy + host-KV launch,
    ``bookkeeping`` = consume + advance), so the marks are re-accumulated in
    canonical order; their total is still the step's wall time.
    """
    kv_launch = (t_copy - t_fwd) + (t_kv_launched - t_consumed)
    readback = t_wait - t_copy
    kv_drain = t_drain - t_wait
    m_kv = t_fwd + kv_launch
    m_rb = m_kv + readback
    m_drain = m_rb + kv_drain
    return (forward_start, t_setup, t_fwd, m_kv, m_rb, m_drain, t_end)
