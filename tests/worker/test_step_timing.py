"""Unit tests for `batchgen.worker.step_timing`.

Pure CPU: the forward timing ring takes an injected event factory, so these
tests drive it with fake events that model CUDA timing-event semantics
(``elapsed_time`` raises unless both events completed) and fail loudly on any
call that could make the host wait for the device.
"""

from __future__ import annotations

import pytest

from batchgen.worker.step_timing import (
    DECODE_STEP_SPLIT_SEGMENTS,
    DECODE_STEP_SPLIT_SLOTS,
    FORWARD_EVENT_RING_SLOTS,
    ForwardEventRing,
    accumulate_decode_step_split,
    format_decode_step_split,
    new_decode_step_split,
)


class _FakeEvent:
    """A timing event whose completion the test controls."""

    def __init__(self, log):
        self._log = log
        self.recorded = False
        self.recorded_on = None
        self.done = False
        self.timestamp_ms = None

    def record(self, stream=None):
        self.recorded = True
        self.recorded_on = stream
        self.done = False
        self.timestamp_ms = None
        self._log.append(("record", self))

    def complete(self, timestamp_ms):
        assert self.recorded, "completing an event never recorded"
        self.done = True
        self.timestamp_ms = timestamp_ms

    def query(self):
        self._log.append(("query", self))
        return self.done

    def elapsed_time(self, end_event):
        # Mirrors cudaEventElapsedTime: cudaErrorNotReady if either is pending.
        if not (self.done and end_event.done):
            raise RuntimeError("elapsed_time on an incomplete event pair")
        self._log.append(("elapsed", self))
        return end_event.timestamp_ms - self.timestamp_ms

    def synchronize(self):
        raise AssertionError("ring made the host wait: Event.synchronize")

    def wait(self, stream=None):
        raise AssertionError("ring touched Event.wait")


def _ring(num_slots):
    log = []
    events = []

    def factory():
        event = _FakeEvent(log)
        events.append(event)
        return event

    ring = ForwardEventRing(num_slots, factory)
    return ring, events, log


def _pair(events, slot):
    return events[2 * slot], events[2 * slot + 1]


def _assert_conserved(ring, open_slots=0):
    assert ring.num_free + ring.num_pending + open_slots == ring.num_slots


def test_production_ring_size():
    assert FORWARD_EVENT_RING_SLOTS == 64


def test_rejects_empty_ring():
    with pytest.raises(ValueError):
        ForwardEventRing(0, lambda: _FakeEvent([]))


def test_begin_end_harvest_round_trip():
    ring, events, _ = _ring(4)
    stream = object()
    slot = ring.begin(stream)
    assert slot is not None
    _assert_conserved(ring, open_slots=1)
    start, end = _pair(events, slot)
    assert start.recorded_on is stream and end.recorded_on is None

    ring.end(slot, stream)
    assert end.recorded_on is stream
    assert ring.num_pending == 1
    _assert_conserved(ring)

    start.complete(10.0)
    end.complete(13.5)
    ring.harvest()
    assert ring.num_pending == 0 and ring.num_free == 4
    assert ring.take_summary() == " gpu_fwd_ms=3.5 gpu_fwd_max=3.5 (n=1)"
    # take_summary resets the totals.
    assert ring.take_summary() == ""


def test_full_ring_skips_instead_of_waiting():
    ring, events, log = _ring(3)
    stream = object()
    slots = []
    for _ in range(3):
        slot = ring.begin(stream)
        ring.end(slot, stream)
        slots.append(slot)
    assert sorted(slots) == [0, 1, 2]
    assert ring.num_free == 0

    # No pair has completed. Further steps are not timed, and neither
    # begin/end nor harvest may wait for a slot: the fake events raise on
    # synchronize()/wait(), and elapsed_time() raises on incomplete pairs.
    for _ in range(10):
        slot = ring.begin(stream)
        assert slot is None
        ring.end(slot, stream)
        ring.harvest()
        _assert_conserved(ring)
    assert ring.num_pending == 3
    assert not [entry for entry in log if entry[0] == "elapsed"]
    assert ring.take_summary() == ""

    # Once the device catches up, the slots come back.
    for slot in slots:
        start, end = _pair(events, slot)
        start.complete(0.0)
        end.complete(2.0)
    ring.harvest()
    assert ring.num_free == 3 and ring.num_pending == 0
    assert ring.begin(stream) is not None


def test_elapsed_read_only_after_both_events_complete():
    ring, events, log = _ring(2)
    stream = object()
    slot = ring.begin(stream)
    ring.end(slot, stream)
    start, end = _pair(events, slot)

    # Only the start event completed.
    start.complete(5.0)
    ring.harvest()
    assert ring.num_pending == 1
    assert not [entry for entry in log if entry[0] == "elapsed"]

    # Only the end event completed (start pending): still not read.
    start.done = False
    end.complete(9.0)
    ring.harvest()
    assert ring.num_pending == 1
    assert not [entry for entry in log if entry[0] == "elapsed"]

    start.complete(5.0)
    ring.harvest()
    elapsed_at = [i for i, entry in enumerate(log) if entry[0] == "elapsed"]
    assert len(elapsed_at) == 1
    # The read is immediately preceded by a successful query of each event.
    i = elapsed_at[0]
    assert log[i - 2] == ("query", start)
    assert log[i - 1] == ("query", end)
    assert ring.take_summary() == " gpu_fwd_ms=4.0 gpu_fwd_max=4.0 (n=1)"


def test_out_of_order_completion_conserves_slots():
    ring, events, _ = _ring(4)
    stream = object()
    slots = []
    for _ in range(4):
        slot = ring.begin(stream)
        ring.end(slot, stream)
        slots.append(slot)

    # Complete the 2nd and 4th pair only.
    for slot, (t0, t1) in ((slots[1], (0.0, 1.0)), (slots[3], (0.0, 3.0))):
        start, end = _pair(events, slot)
        start.complete(t0)
        end.complete(t1)
    ring.harvest()
    assert ring.num_free == 2 and ring.num_pending == 2
    _assert_conserved(ring)

    # Reuse a freed slot while two older pairs are still pending.
    reused = ring.begin(stream)
    assert reused in (slots[1], slots[3])
    _assert_conserved(ring, open_slots=1)
    ring.end(reused, stream)
    _assert_conserved(ring)

    for slot in (slots[0], slots[2], reused):
        start, end = _pair(events, slot)
        start.complete(0.0)
        end.complete(2.0)
    ring.harvest()
    assert ring.num_free == 4 and ring.num_pending == 0
    # Five harvested pairs: 1, 3, 2, 2, 2 ms.
    assert ring.take_summary() == " gpu_fwd_ms=2.0 gpu_fwd_max=3.0 (n=5)"


def test_many_steps_conserve_slots():
    # Every third timed pair never completes (a stuck device), so those
    # slots stay pending; the ring degrades to skipping steps, never to
    # waiting, and every slot stays accounted for.
    ring, events, _ = _ring(8)
    stream = object()
    clock = 0.0
    timed = skipped = 0
    for step in range(1000):
        slot = ring.begin(stream)
        ring.end(slot, stream)
        if slot is None:
            skipped += 1
        else:
            timed += 1
            if timed % 3 != 0:
                start, end = _pair(events, slot)
                start.complete(clock)
                end.complete(clock + 1.0)
        clock += 1.0
        ring.harvest()
        _assert_conserved(ring)
    assert ring.num_pending == 8 and ring.num_free == 0
    assert timed + skipped == 1000 and skipped > 0


def test_step_split_layout_and_accumulation():
    assert DECODE_STEP_SPLIT_SEGMENTS == (
        "setup", "fwd_launch", "kv_launch", "readback", "kv_drain", "bookkeeping",
    )
    assert DECODE_STEP_SPLIT_SLOTS == 8
    split = new_decode_step_split()
    assert split == [0.0] * 8
    # Marks in seconds: step start, then the end of each segment.
    accumulate_decode_step_split(split, (0.0, 0.001, 0.003, 0.004, 0.010, 0.012, 0.013))
    accumulate_decode_step_split(split, (1.0, 1.001, 1.003, 1.004, 1.010, 1.012, 1.013))
    assert split[0] == 2
    assert split[1:] == pytest.approx([26.0, 2.0, 4.0, 2.0, 12.0, 4.0, 2.0])
    assert format_decode_step_split(split) == (
        "step_ms=13.0 (setup 1.0, fwd_launch 2.0, kv_launch 1.0, readback 6.0, "
        "kv_drain 2.0, bookkeeping 1.0)"
    )


def test_step_split_rejects_layout_mismatch():
    with pytest.raises(ValueError):
        accumulate_decode_step_split([0.0] * 7, (0.0,) * 7)
    with pytest.raises(ValueError):
        accumulate_decode_step_split(new_decode_step_split(), (0.0,) * 6)
    with pytest.raises(ValueError):
        format_decode_step_split(new_decode_step_split())


def test_heartbeat_text_drops_the_old_labels():
    split = new_decode_step_split()
    accumulate_decode_step_split(split, (0.0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006))
    ring, events, _ = _ring(1)
    slot = ring.begin(None)
    ring.end(slot, None)
    start, end = _pair(events, slot)
    start.complete(0.0)
    end.complete(1.0)
    ring.harvest()
    text = format_decode_step_split(split) + ring.take_summary()
    # The old labels changed meaning; a parser must not keep matching them.
    for old_label in ("forward+sample", "kv_flush", "token_readback"):
        assert old_label not in text
    positions = [text.index(f"{name} ") for name in DECODE_STEP_SPLIT_SEGMENTS]
    assert positions == sorted(positions)
    assert "gpu_fwd_ms=1.0 gpu_fwd_max=1.0 (n=1)" in text
