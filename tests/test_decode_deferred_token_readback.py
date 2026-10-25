"""Deferred sampled-token readback (depth-1 decode pipeline).

``batchgen_worker`` imports the whole engine, so the shipping worker methods
are lifted out of the source and executed against fakes (the pattern of the
neighbouring decode-loop tests). The device is a lazy fake stream: queued work
runs only when an event is waited on, so a host read that runs ahead of its
copy, a reused pinned buffer, or a stale metadata value shows up as a wrong
token instead of passing by luck of timing.

The two step shapes are driven by ``_drive``, a model of the
``decoding_continuous`` skeleton (boundary every K steps, setup, forward,
step tail). The deferred tail calls the real worker methods in the order the
real loop calls them -- the static tests at the bottom pin that order in the
shipping loop -- and the undeferred tail reproduces the inline step the
existing decode-loop tests pin.
"""

from __future__ import annotations

import ast
import logging
import textwrap
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import pytest
import torch

from batchgen.worker.completion import CompletionContext, CompletionHandler
from batchgen.worker.deferred_readback import (
    DeferredTokenReadbackError,
    PendingDecodeStep,
    TokenReadbackRing,
    deferred_decode_step_marks,
)
from batchgen.worker.step_timing import (
    DECODE_STEP_SPLIT_SEGMENTS,
    accumulate_decode_step_split,
    new_decode_step_split,
)

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
SOURCE = WORKER.read_text()
TREE = ast.parse(SOURCE)
WORKER_CLASS = next(
    n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "BatchGenWorker"
)


def _method_node(name):
    return next(
        n for n in WORKER_CLASS.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


def _source_of(node):
    lines = SOURCE.splitlines(keepends=True)
    return textwrap.dedent("".join(lines[node.lineno - 1 : node.end_lineno]))


def _module_function(name):
    return next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _module_constant(name):
    node = next(
        n for n in TREE.body
        if isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == [name]
    )
    return ast.literal_eval(node.value)


_METHODS = (
    "_apply_decode_row_token",
    "_require_decode_tokens_read",
    "_begin_deferred_decode_step",
    "_consume_deferred_decode_tokens",
    "_finish_deferred_decode_step",
    "_warn_sampling_bad_rows",
    "_wait_decode_tokens",
    "_cohort_shutdown_requested",
    "_make_completion_context",
    "_should_stop_at_eos",
    "_is_sequence_completed",
    "_trajectory_slot",
    "_trajectory_decoded_tokens",
    "_append_trajectory_tokens",
    "_rebuild_input_tokens",
)


def _namespace(**overrides):
    ns = {
        "time": time,
        "logging": logging,
        "torch": torch,
        "List": List,
        "Optional": Optional,
        "Sequence": Sequence,
        "SequenceEntry": object,
        "BATCHGEN_MULTI_BATCH_DIAG": False,
        "REP_DETECTION": True,
        "SeqEvent": SimpleNamespace(REPETITION="REPETITION"),
        "lifespan": SimpleNamespace(dump_lifespan=lambda *a, **k: None),
        "CompletionHandler": CompletionHandler,
        "CompletionContext": CompletionContext,
        "PendingDecodeStep": PendingDecodeStep,
        "TokenReadbackRing": TokenReadbackRing,
        "DeferredTokenReadbackError": DeferredTokenReadbackError,
    }
    for name in (
        "_DECODE_WAIT_COMPLETED", "_DECODE_WAIT_ABANDONED", "_DECODE_STEP_SHUTDOWN",
        "_DECODE_EVENT_SPIN_BUDGET_S", "_DECODE_EVENT_POLL_INTERVAL_S",
    ):
        ns[name] = _module_constant(name)
    for fn in ("_check_repeating_pattern", "_repetition_check_enabled"):
        exec(compile(_source_of(_module_function(fn)), str(WORKER), "exec"), ns)
    ns.update(overrides)
    for name in _METHODS:
        exec(compile(_source_of(_method_node(name)), str(WORKER), "exec"), ns)
    return ns


# --------------------------------------------------------------- lazy device

class FakeStream:
    """In-order device queue that runs work only when something waits on it."""

    def __init__(self):
        self.queue = []
        self.executed = []

    def enqueue(self, name, fn):
        self.queue.append((name, fn))

    def run_until(self, done):
        while not done():
            if not self.queue:
                raise AssertionError("waited on device work that was never queued")
            name, fn = self.queue.pop(0)
            fn()
            self.executed.append(name)

    def has_queued(self, prefix):
        return any(name.startswith(prefix) for name, _ in self.queue)


class FakeEvent:
    def __init__(self, stream):
        self._stream = stream
        self._recorded = 0
        self._completed = 0

    def record(self, stream=None):
        stream = stream or self._stream
        self._recorded += 1
        gen = self._recorded
        stream.enqueue(f"event{gen}", lambda: setattr(self, "_completed", max(self._completed, gen)))

    def query(self):
        return self._completed >= self._recorded

    def synchronize(self):
        self._stream.run_until(self.query)


class DevTensor:
    """A device tensor whose values exist only after its producer ran."""

    def __init__(self, n, vals=None):
        self.n = n
        self.vals = vals

    @property
    def shape(self):
        return (self.n, 1)

    def __getitem__(self, s):
        assert isinstance(s, slice) and s.start is None
        return _DevView(self, s.stop)


class _DevView:
    def __init__(self, base, n):
        self.base, self.n = base, n


class HostBuf:
    """Pinned host buffer: a non-blocking copy into it lands when it runs."""

    def __init__(self, stream, n):
        self.stream = stream
        self.vals = [None] * n
        self.shape = (n, 1)

    def __getitem__(self, s):
        return _HostView(self, s.stop)


class _HostView:
    def __init__(self, buf, n):
        self.buf, self.n = buf, n

    def copy_(self, src, non_blocking=False):
        assert non_blocking
        buf, n = self.buf, self.n

        def run():
            assert src.base.vals is not None, "copy ran before its producer"
            buf.vals[:n] = list(src.base.vals[:n])

        buf.stream.enqueue("token_copy", run)

    def reshape(self, *_):
        return self

    def tolist(self):
        return list(self.buf.vals[: self.n])


# ----------------------------------------------------------- host-side fakes

class Seq:
    def __init__(self, gid, prompt, max_decode, ignore_eos=False):
        self.uuid = f"u{gid}"
        self.global_idx = gid
        self.original_prompt_length = prompt
        self.prompt_length = prompt
        self.decoded_length = 1  # the prefill token
        self.current_context_length = prompt + 1
        self.max_decode_length = max_decode
        self.ignore_eos = ignore_eos
        self.eos_reached = False
        self._rep_detected = False
        self._rep_last_token = None
        self._rep_count = 0
        self._lifespan_log = []

    def log_event(self, *a, **k):
        pass

    def state(self):
        return (self.decoded_length, self.current_context_length, self.eos_reached,
                self._rep_detected)


class Book:
    def __init__(self, seqs, first_token):
        self.tokens = {s.uuid: [first_token(s)] for s in seqs}
        self._slots = {s.uuid: s.uuid for s in seqs}

    def slot_for(self, uuid):
        return self._slots[uuid]

    def append_tokens(self, slots, ids):
        for slot, tok in zip(slots, ids):
            self.tokens[slot].append(int(tok))

    def generated_tokens(self, slot):
        return torch.tensor(self.tokens[slot], dtype=torch.long)

    def last_generated_token(self, slot):
        return torch.tensor(self.tokens[slot][-1], dtype=torch.long)


EOS = 2
VOCAB = 50


def _make_worker(ns, seqs, stream, log):
    book = Book(seqs, first_token=lambda s: 3 + s.global_idx)
    by_uuid = {s.uuid: s for s in seqs}
    kv_waits = []
    worker = SimpleNamespace(
        rank=0,
        eos_token_ids=[EOS],
        model_context_length=10_000,
        torch_device="cpu",
        _trajectory_book=book,
        _local_to_uuid_map={i: s.uuid for i, s in enumerate(seqs)},
        global_batch=SimpleNamespace(get_sequence=by_uuid.get),
        _cohort_shutdown_probe=None,
        _decode_tokens_unread=False,
        _pending_kv_append_tasks=[],
        kv_waits=kv_waits,
    )

    def wait_kv(**kwargs):
        kv_waits.append(kwargs)
        log.append(("kv_drain", len(worker._pending_kv_append_tasks)))
        worker._pending_kv_append_tasks.clear()
        return 0

    worker._wait_pending_kv_append_tasks = wait_kv
    for name in _METHODS:
        setattr(worker, name, types.MethodType(ns[name], worker))
    real_consume = worker._consume_deferred_decode_tokens

    def consume(step, token_ids):
        log.append(("consume", step.local_iteration, stream.has_queued("forward")))
        real_consume(step, token_ids)

    worker._consume_deferred_decode_tokens = consume
    return worker


def _model_token(seq, ctx, inp, stop_at):
    """Deterministic 'model': depends on row, context length and input id."""
    if seq.global_idx == 5:
        return 9  # degenerate row: trips the 32x repetition breaker
    if stop_at.get(seq.global_idx) == ctx - seq.original_prompt_length:
        return EOS
    tok = (seq.global_idx * 31 + ctx * 7 + inp * 3) % VOCAB
    return 11 if tok == EOS else tok


def _drive(mode, *, interval=4, ns=None):
    """Run one decode round to completion; return everything observable."""
    ns = ns or _namespace()
    stream = FakeStream()
    log = []
    seqs = [
        Seq(0, 10, 40),                    # length-limited
        Seq(1, 7, 40),                     # EOS at decoded step 2
        Seq(2, 12, 40),                    # EOS on a boundary step
        Seq(3, 5, 40),                     # EOS right after a boundary
        Seq(4, 9, 40, ignore_eos=True),    # EOS token ignored, runs to length
        Seq(5, 6, 60),                     # repetition breaker
        Seq(6, 8, 13),                     # short length limit
        Seq(7, 4, 40),                     # EOS at decoded step 7
    ]
    # the model emits EOS as the token of the step whose input context is
    # prompt + stop_at, i.e. the row ends at decoded_length == stop_at + 1
    stop_at = {1: 2, 2: 7, 3: 9, 4: 6, 7: 6}
    worker = _make_worker(ns, seqs, stream, log)
    forward_inputs = []
    completed_at = {}

    batch = list(range(len(seqs)))
    rebuilt = worker._rebuild_input_tokens(batch)
    new_tokens = DevTensor(len(batch), rebuilt.reshape(-1).tolist())
    ring = None
    if mode == "deferred":
        ring = TokenReadbackRing(
            len(batch), lambda n: HostBuf(stream, n), lambda: FakeEvent(stream)
        )
    pinned = HostBuf(stream, len(batch))
    ready = FakeEvent(stream)
    pending = None
    local_iteration = 0
    last_boundary = 0
    while batch:
        local_iteration += 1
        assert local_iteration <= 200, "decode round did not terminate"
        if local_iteration - last_boundary >= interval:
            if pending is not None:
                status = worker._finish_deferred_decode_step(pending, ring, overlapped=False)
                pending = None
                assert status == ns["_DECODE_WAIT_COMPLETED"]
            last_boundary = local_iteration
            # what _page_boundary_fast does first, then the completion pass
            worker._require_decode_tokens_read("_page_boundary_fast")
            log.append(("boundary", local_iteration))
            keep = []
            for row in batch:
                seq = seqs[row]
                if worker._is_sequence_completed(seq):
                    completed_at[seq.uuid] = local_iteration
                else:
                    keep.append(row)
            batch = keep
            if not batch:
                break
            rebuilt = worker._rebuild_input_tokens(batch)
            new_tokens = DevTensor(len(batch), rebuilt.reshape(-1).tolist())

        batch_sequences = [seqs[row] for row in batch]
        # setup: host metadata is read NOW and handed to the device
        for seq in batch_sequences:
            assert seq.current_context_length == seq.original_prompt_length + seq.decoded_length
        ctx = [seq.current_context_length for seq in batch_sequences]
        inp = new_tokens
        out = DevTensor(len(batch))

        def forward(step=local_iteration, rows=list(batch_sequences), ctx=ctx, inp=inp, out=out):
            assert inp.vals is not None
            vals = []
            for seq, c, i in zip(rows, ctx, inp.vals):
                forward_inputs.append((step, seq.global_idx, c, i))
                vals.append(_model_token(seq, c, i, stop_at))
            out.vals = vals

        stream.enqueue(f"forward{local_iteration}", forward)
        log.append(("launch", local_iteration))
        new_tokens = out
        bs = out.n

        if mode == "deferred":
            slot = ring.launch(new_tokens, bs, stream)
            if pending is not None:
                status = worker._finish_deferred_decode_step(pending, ring, overlapped=True)
                pending = None
                assert status == ns["_DECODE_WAIT_COMPLETED"]
            worker._pending_kv_append_tasks.append(("kv", local_iteration))
            pending = worker._begin_deferred_decode_step(
                batch, batch_sequences, slot, bs, local_iteration
            )
            continue

        # undeferred step, as the inline loop tail does it
        pinned[:bs].copy_(new_tokens[:bs], non_blocking=True)
        ready.record(stream)
        worker._pending_kv_append_tasks.append(("kv", local_iteration))
        assert worker._wait_decode_tokens(ready) == ns["_DECODE_WAIT_COMPLETED"]
        if worker._pending_kv_append_tasks:
            worker._wait_pending_kv_append_tasks(defer_errors=True)
        new_token_ids = pinned[:bs].reshape(-1).tolist()
        log.append(("consume", local_iteration, stream.has_queued("forward")))
        _row_active = [not worker._is_sequence_completed(seq) for seq in batch_sequences]
        worker._append_trajectory_tokens(
            [seq for seq, live in zip(batch_sequences, _row_active) if live],
            [tok for tok, live in zip(new_token_ids, _row_active) if live],
        )
        for i, seq in enumerate(batch_sequences):
            if not _row_active[i]:
                continue
            seq.decoded_length += 1
            seq.current_context_length = seq.original_prompt_length + seq.decoded_length
            worker._apply_decode_row_token(seq, new_token_ids[i], i, local_iteration)

    assert pending is None, "loop left with a deferred step unread"
    return SimpleNamespace(
        seqs=seqs,
        tokens=worker._trajectory_book.tokens,
        states={s.uuid: s.state() for s in seqs},
        completed_at=completed_at,
        forward_inputs=forward_inputs,
        log=log,
        worker=worker,
        ring=ring,
        steps=local_iteration,
    )


# ------------------------------------------------------- (a) token identity

@pytest.mark.parametrize("interval", [1, 2, 3, 4, 7, 128])
def test_deferred_readback_is_token_identical_to_the_undeferred_step(interval):
    off = _drive("undeferred", interval=interval)
    on = _drive("deferred", interval=interval)
    assert on.tokens == off.tokens
    assert on.states == off.states
    assert on.completed_at == off.completed_at
    # the device saw the same inputs at every step: same context lengths and
    # same input ids, fed device-side from the previous step's sample
    assert on.forward_inputs == off.forward_inputs
    assert on.steps == off.steps


def test_emitted_token_counts_and_stop_reasons_are_exact():
    on = _drive("deferred", interval=4)
    s = {seq.uuid: seq for seq in on.seqs}
    # emitted = prefill token + one per live decode step; never one extra
    for uuid, seq in s.items():
        assert len(on.tokens[uuid]) == seq.decoded_length
    # EOS stops exactly at the configured step and the EOS token is the last
    assert s["u1"].decoded_length == 3 and on.tokens["u1"][-1] == EOS
    assert s["u2"].decoded_length == 8 and on.tokens["u2"][-1] == EOS
    assert s["u3"].decoded_length == 10 and on.tokens["u3"][-1] == EOS
    assert s["u7"].decoded_length == 7 and on.tokens["u7"][-1] == EOS
    # ignore_eos runs through its EOS token to the length limit
    assert EOS in on.tokens["u4"] and s["u4"].decoded_length == 40
    assert not s["u4"].eos_reached or s["u4"].decoded_length == s["u4"].max_decode_length
    assert s["u0"].decoded_length == 40 and s["u6"].decoded_length == 13
    # repetition breaker: the 32nd identical token retires the row
    # (32 identical decode tokens after the prefill token)
    assert s["u5"]._rep_detected and s["u5"].decoded_length == 33


def test_finished_rows_keep_computing_until_the_boundary_without_emitting():
    """The one-step stop lag is absorbed by the existing design: a finished
    row stays in the batch until the next boundary in BOTH step shapes and is
    skipped by the completed-row mask, so the lag never emits a token."""
    on = _drive("deferred", interval=4)
    off = _drive("undeferred", interval=4)
    row1_steps_on = [step for step, gid, _, _ in on.forward_inputs if gid == 1]
    row1_steps_off = [step for step, gid, _, _ in off.forward_inputs if gid == 1]
    assert row1_steps_on == row1_steps_off == [1, 2, 3]
    assert len(on.tokens["u1"]) == 3  # prefill + 2 decoded, EOS last
    assert on.tokens["u1"][-1] == EOS


# ------------------------------------------------ (b) enqueue-before-wait

def test_next_step_is_enqueued_before_the_host_consumes_the_previous_tokens():
    on = _drive("deferred", interval=4)
    boundary_steps = {step for kind, step, *_ in on.log if kind == "boundary"}
    consumes = [entry for entry in on.log if entry[0] == "consume"]
    assert consumes
    for kind, step, forward_queued in consumes:
        launched_next = ("launch", step + 1) in on.log
        if step + 1 in boundary_steps or not launched_next:
            # drained at the boundary (or the round's last step): nothing
            # may be queued behind it, the planner needs its tokens
            assert on.log.index(("consume", step, forward_queued)) < (
                on.log.index(("boundary", step + 1))
            )
            assert not forward_queued
        else:
            assert on.log.index(("launch", step + 1)) < on.log.index(
                ("consume", step, forward_queued)
            )
            # ...and the GPU still has that forward queued while the host
            # does the previous step's bookkeeping: the overlap itself.
            assert forward_queued


def test_undeferred_step_never_overlaps():
    off = _drive("undeferred", interval=4)
    assert all(not queued for kind, _, queued in
               (e for e in off.log if e[0] == "consume"))


def test_overlapped_kv_drain_skips_only_the_device_wide_sync():
    on = _drive("deferred", interval=4)
    flags = [kw for kw in on.worker.kv_waits]
    assert flags and all(kw["defer_errors"] is True for kw in flags)
    overlapped = [kw for kw in flags if kw["device_sync"] is False]
    drained = [kw for kw in flags if kw["device_sync"] is True]
    assert overlapped and drained
    # at most one step's host-KV appends are ever in flight
    assert all(n <= 1 for kind, n in (e for e in on.log if e[0] == "kv_drain"))


# --------------------------------------------------------- (c) last drain

def test_round_end_leaves_nothing_in_flight():
    on = _drive("deferred", interval=4)
    assert on.ring.in_flight() == 0
    assert on.worker._decode_tokens_unread is False
    assert on.completed_at  # every row retired at a boundary
    assert set(on.completed_at) == {s.uuid for s in on.seqs}


def test_host_token_consumers_fail_loud_while_a_step_is_deferred():
    ns = _namespace()
    stream = FakeStream()
    worker = _make_worker(ns, [Seq(0, 4, 8)], stream, [])
    worker._decode_tokens_unread = True
    with pytest.raises(DeferredTokenReadbackError, match="_page_boundary_fast"):
        worker._require_decode_tokens_read("_page_boundary_fast")
    with pytest.raises(DeferredTokenReadbackError, match="_rebuild_input_tokens"):
        worker._rebuild_input_tokens([0])


def _one_pending(ns, stream, probe=None):
    seq = Seq(0, 4, 8)
    worker = _make_worker(ns, [seq], stream, [])
    worker._cohort_shutdown_probe = probe
    ring = TokenReadbackRing(1, lambda n: HostBuf(stream, n), lambda: FakeEvent(stream))
    out = DevTensor(1)
    stream.enqueue("forward1", lambda: setattr(out, "vals", [7]))
    slot = ring.launch(out, 1, stream)
    step = worker._begin_deferred_decode_step([0], [seq], slot, 1, 1)
    return worker, ring, step, seq


def test_shutdown_after_the_drain_reads_no_token():
    ns = _namespace()
    stream = FakeStream()
    worker, ring, step, seq = _one_pending(ns, stream)
    worker._pending_kv_append_tasks.append("kv")
    worker._cohort_shutdown_requested = lambda: True
    status = worker._finish_deferred_decode_step(step, ring, overlapped=True)
    assert status == ns["_DECODE_STEP_SHUTDOWN"]
    assert worker.kv_waits == [{"defer_errors": True, "device_sync": False}]
    assert worker._trajectory_book.tokens["u0"] == [3]  # nothing appended
    assert ring.in_flight() == 0
    assert worker._decode_tokens_unread is True


def test_abandoned_wait_neither_drains_nor_reads():
    ns = _namespace()
    stream = FakeStream()
    worker, ring, step, seq = _one_pending(ns, stream, probe=lambda: True)
    worker._pending_kv_append_tasks.append("kv")
    status = worker._finish_deferred_decode_step(step, ring, overlapped=True)
    assert status == ns["_DECODE_WAIT_ABANDONED"]
    assert worker.kv_waits == []
    assert worker._trajectory_book.tokens["u0"] == [3]
    assert ring.in_flight() == 0


def test_finish_reports_marks_and_completes():
    ns = _namespace()
    stream = FakeStream()
    worker, ring, step, seq = _one_pending(ns, stream)
    marks = []
    status = worker._finish_deferred_decode_step(step, ring, overlapped=False, marks=marks)
    assert status == ns["_DECODE_WAIT_COMPLETED"]
    assert len(marks) == 3 and marks == sorted(marks)
    assert worker._trajectory_book.tokens["u0"] == [3, 7]
    assert worker._decode_tokens_unread is False


# ---------------------------------------------------------------- the ring

def test_ring_alternates_slots_and_refuses_a_second_step_in_flight():
    stream = FakeStream()
    ring = TokenReadbackRing(2, lambda n: HostBuf(stream, n), lambda: FakeEvent(stream))
    a, b, c = DevTensor(2, [1, 2]), DevTensor(2, [3, 4]), DevTensor(2, [5, 6])
    s0 = ring.launch(a, 2, stream)
    s1 = ring.launch(b, 2, stream)
    assert (s0, s1) == (0, 1)
    with pytest.raises(DeferredTokenReadbackError, match="more than one deferred step"):
        ring.launch(c, 2, stream)
    with pytest.raises(DeferredTokenReadbackError, match="before its copy completed"):
        ring.read(s0, 2)
    ring.event(s0).synchronize()
    assert ring.read(s0, 2) == [1, 2]
    with pytest.raises(DeferredTokenReadbackError, match="no launched step"):
        ring.read(s0, 2)
    assert ring.launch(c, 2, stream) == 0
    ring.event(0).synchronize()
    assert ring.read(1, 2) == [3, 4] and ring.read(0, 2) == [5, 6]


def test_ring_regrows_only_the_free_slot():
    stream = FakeStream()
    made = []

    def factory(n):
        made.append(n)
        return HostBuf(stream, n)

    ring = TokenReadbackRing(1, factory, lambda: FakeEvent(stream))
    ring.launch(DevTensor(1, [9]), 1, stream)
    ring.launch(DevTensor(3, [1, 2, 3]), 3, stream)
    assert made == [1, 1, 3]
    ring.event(1).synchronize()
    assert ring.read(0, 1) == [9] and ring.read(1, 3) == [1, 2, 3]


def test_split_marks_keep_segment_meaning_and_total():
    fs, t0, t1, t2, t3, t4, tc, tk, te = 0.0, 1.0, 3.0, 3.5, 10.0, 11.0, 12.0, 13.5, 14.0
    marks = deferred_decode_step_marks(fs, t0, t1, t2, t3, t4, tc, tk, te)
    split = new_decode_step_split()
    accumulate_decode_step_split(split, marks)
    seg = dict(zip(DECODE_STEP_SPLIT_SEGMENTS, split[2:]))
    ms = 1000.0
    assert split[1] == pytest.approx((te - fs) * ms)
    assert seg["setup"] == pytest.approx(1.0 * ms)
    assert seg["fwd_launch"] == pytest.approx(2.0 * ms)
    assert seg["kv_launch"] == pytest.approx((0.5 + 1.5) * ms)
    assert seg["readback"] == pytest.approx(6.5 * ms)
    assert seg["kv_drain"] == pytest.approx(1.0 * ms)
    assert seg["bookkeeping"] == pytest.approx((1.0 + 0.5) * ms)


# ------------------------------------------- metadata staging rewrite guard

def _bind_metadata(deferred):
    events = []

    class Ev:
        def __init__(self):
            events.append(self)
            self.calls = []

        def synchronize(self):
            self.calls.append("sync")

        def record(self, stream):
            self.calls.append("record")

    fake_cuda = SimpleNamespace(Event=Ev, current_stream=lambda d: "stream")
    fake_torch = SimpleNamespace(cuda=fake_cuda, int64=torch.int64)
    ns = {"torch": fake_torch, "DECODE_DEFERRED_TOKEN_READBACK": deferred,
          "List": List, "Tuple": tuple, "SequenceEntry": object}
    exec(compile(_source_of(_method_node("_bind_decode_attention_metadata")),
                 str(WORKER), "exec"), ns)
    worker = SimpleNamespace(
        torch_device="cpu",
        _decode_cache_seqlens_i32=torch.zeros(4, dtype=torch.int32),
        _decode_position_ids_i64=torch.zeros(4, 1, dtype=torch.int64),
        _decode_cache_seqlens_cpu_staging=torch.zeros(4, dtype=torch.int32),
        _decode_metadata_batch_key=None,
        _decode_metadata_cpu_seqlens=None,
        _decode_metadata_staging_event=None,
        _ensure_decode_metadata_capacity=lambda n: None,
    )
    bind = types.MethodType(ns["_bind_decode_attention_metadata"], worker)
    rows = [SimpleNamespace(global_idx=g) for g in (1, 2)]
    bind(rows, [5, 6])        # seed: staging copy
    bind(rows, [6, 7])        # +1 everywhere: on-device advance, no staging
    bind(rows, [6, 8])        # a frozen (finished) row: staging rewrite
    return worker, events


def test_deferred_mode_waits_the_previous_staging_copy_before_rewriting():
    worker, events = _bind_metadata(True)
    assert len(events) == 1
    assert events[0].calls == ["record", "sync", "record"]
    assert worker._decode_cache_seqlens_i32[:2].tolist() == [6, 8]


def test_undeferred_mode_never_touches_a_staging_event():
    worker, events = _bind_metadata(False)
    assert events == []
    assert worker._decode_metadata_staging_event is None
    assert worker._decode_cache_seqlens_i32[:2].tolist() == [6, 8]


# ------------------------------------------- (d) shipping-loop structure

def _loop():
    fn = _method_node("decoding_continuous")
    return fn, next(
        n for n in fn.body if isinstance(n, ast.While) and ast.unparse(n.test) == "decode_uuids"
    )


def _deferred_block(loop):
    (block,) = [
        n for n in loop.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "_deferred_readback"
    ]
    return block


def _call_index(body, func_src):
    hits = [i for i, stmt in enumerate(body)
            if any(isinstance(c, ast.Call) and ast.unparse(c.func) == func_src
                   for c in ast.walk(stmt))]
    assert len(hits) == 1, (func_src, hits)
    return hits[0]


def test_switch_is_a_module_constant_read_once_per_round():
    assert _module_constant("DECODE_DEFERRED_TOKEN_READBACK") is True
    fn, _ = _loop()
    reads = [n for n in ast.walk(fn)
             if isinstance(n, ast.Name) and n.id == "DECODE_DEFERRED_TOKEN_READBACK"]
    assert len(reads) == 1
    assign = [n for n in fn.body if isinstance(n, ast.Assign)
              and ast.unparse(n.targets[0]) == "_deferred_readback"]
    assert len(assign) == 1
    for name in ("_begin_deferred_decode_step", "_finish_deferred_decode_step",
                 "_consume_deferred_decode_tokens", "_require_decode_tokens_read",
                 "_apply_decode_row_token"):
        body = ast.unparse(_method_node(name))
        assert "os.environ" not in body and "getenv" not in body, name


def test_deferred_tail_enqueues_the_forward_before_waiting_the_previous_step():
    _, loop = _loop()
    block = _deferred_block(loop)
    forward = next(i for i, s in enumerate(loop.body)
                   if isinstance(s, ast.With)
                   and ast.unparse(s.items[0].context_expr) == "torch.inference_mode()")
    end = next(i for i, s in enumerate(loop.body)
               if "_fwd_ring.end" in ast.unparse(s) and isinstance(s, ast.Expr))
    assert forward < end < loop.body.index(block)
    body = block.body
    launch = _call_index(body, "_readback_ring.launch")
    finish = _call_index(body, "self._finish_deferred_decode_step")
    flush = _call_index(body, "self._flush_deferred_kv_to_host")
    begin = _call_index(body, "self._begin_deferred_decode_step")
    assert launch < finish < flush < begin
    assert isinstance(body[-1], ast.Continue)
    call = next(c for c in ast.walk(body[finish]) if isinstance(c, ast.Call)
                and ast.unparse(c.func) == "self._finish_deferred_decode_step")
    assert [(k.arg, ast.unparse(k.value)) for k in call.keywords][0] == ("overlapped", "True")


def test_boundary_drains_the_deferred_step_before_the_planner():
    _, loop = _loop()
    boundary = next(n for n in loop.body if isinstance(n, ast.If)
                    and "self.DECISION_INTERVAL" in ast.unparse(n.test))
    first = boundary.body[0]
    assert ast.unparse(first.test) == "_pending_step is not None"
    call = next(c for c in ast.walk(first) if isinstance(c, ast.Call)
                and ast.unparse(c.func) == "self._finish_deferred_decode_step")
    assert [(k.arg, ast.unparse(k.value)) for k in call.keywords] == [("overlapped", "False")]
    planner = _call_index(boundary.body, "self._page_boundary_fast")
    assert planner > 0


def test_every_deferred_teardown_exit_is_the_ordered_three_statement_exit():
    _, loop = _loop()
    exits = [n for n in ast.walk(loop) if isinstance(n, ast.If)
             and ast.unparse(n.test) == "_finish != _DECODE_WAIT_COMPLETED"]
    assert len(exits) == 2
    for check in exits:
        assert [ast.unparse(s) for s in check.body] == [
            "self._cohort_shutdown_observed = True",
            "self._quiesce_decode_for_cohort_shutdown(pending_async_task)",
            "return (decode_uuids, batch)",
        ]


def test_loop_exit_with_a_step_still_deferred_raises():
    fn, loop = _loop()
    tail = [n for n in fn.body if n.lineno > loop.end_lineno]
    guard = tail[0]
    assert isinstance(guard, ast.If) and ast.unparse(guard.test) == "_pending_step is not None"
    assert isinstance(guard.body[-1], ast.Raise)
    assert "DeferredTokenReadbackError" in ast.unparse(guard.body[-1])


def test_planner_and_input_rebuild_guard_on_unread_tokens():
    for name in ("_page_boundary_fast", "_rebuild_input_tokens"):
        fn = _method_node(name)
        stmts = [s for s in fn.body if not (isinstance(s, ast.Expr)
                                            and isinstance(s.value, ast.Constant))]
        assert ast.unparse(stmts[0]) == f"self._require_decode_tokens_read({name!r})"


def test_both_step_shapes_share_one_stop_implementation():
    fn, loop = _loop()
    off_loop = next(n for n in loop.body if isinstance(n, ast.For)
                    and ast.unparse(n.iter) == "enumerate(zip(batch, batch_sequences))")
    assert "self._apply_decode_row_token(seq, token_id, i, local_iteration)" in ast.unparse(off_loop)
    consume = ast.unparse(_method_node("_consume_deferred_decode_tokens"))
    assert "self._apply_decode_row_token(" in consume
    # the stop checks live only in the shared helper now
    assert not [c for c in ast.walk(fn) if isinstance(c, ast.Call)
                and ast.unparse(c.func) == "self._should_stop_at_eos"]


def test_overlapped_drain_is_the_only_caller_without_device_sync():
    calls = [c for c in ast.walk(TREE) if isinstance(c, ast.Call)
             and ast.unparse(c.func) == "self._wait_pending_kv_append_tasks"
             and any(k.arg == "device_sync" for k in c.keywords)]
    assert len(calls) == 1
    (kw,) = [k for k in calls[0].keywords if k.arg == "device_sync"]
    assert ast.unparse(kw.value) == "not overlapped"
    sig = _method_node("_wait_pending_kv_append_tasks").args
    defaults = dict(zip([a.arg for a in sig.kwonlyargs],
                        [ast.unparse(d) for d in sig.kw_defaults]))
    assert defaults["device_sync"] == "True"
