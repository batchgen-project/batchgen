"""One-step decode token-finalization overlap in ``decoding_continuous``.

Step N's sampled-token CPU finalization (decoded-token book, EOS, length and
repetition completion) runs after step N+1's forward, token copy and host-KV
appends are queued, so it overlaps step N+1's device work. Step N's length
counters advance eagerly (they do not depend on the token value), so step
N+1's metadata is exact before step N's token reaches the CPU.

``batchgen_worker`` imports the whole GPU engine, so this file

* extracts the shipping ``_PendingDecodeTokenResult`` and its three worker
  helpers and runs them on CPU tensors with fake events (wait-before-apply,
  exact pipelined-vs-sequential equivalence, resumable apply), and
* pins the loop-local transitions (two-slot readback ring, finalize before
  every consumer of token-dependent state, loop-exit and exception-path
  drains, no new device sync on the steady path) as AST contracts.
"""

import ast
import logging
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import torch


WORKER_PATH = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"

_HELPERS = (
    "_advance_decode_sequences_for_pending_token",
    "_finalize_pending_decode_token",
    "_apply_pending_decode_token",
)


# ---------------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------------


def _source_and_tree():
    source = WORKER_PATH.read_text()
    return source, ast.parse(source)


def _segment(source, node):
    lines = source.splitlines(keepends=True)
    # Include decorators (``@dataclass``): a def/class node starts at its keyword.
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return textwrap.dedent("".join(lines[start - 1 : node.end_lineno]))


def _worker_class(tree):
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )


def _decoding_continuous():
    _, tree = _source_and_tree()
    return next(
        node
        for node in _worker_class(tree).body
        if isinstance(node, ast.FunctionDef) and node.name == "decoding_continuous"
    )


def _decode_try(fn):
    tries = [
        node
        for node in fn.body
        if isinstance(node, ast.Try)
        and any(
            isinstance(stmt, ast.While) and ast.unparse(stmt.test) == "decode_uuids"
            for stmt in node.body
        )
    ]
    assert len(tries) == 1, "the decode loop must sit in exactly one top-level try"
    return tries[0]


def _decode_loop(fn):
    (loop,) = [
        stmt
        for stmt in _decode_try(fn).body
        if isinstance(stmt, ast.While) and ast.unparse(stmt.test) == "decode_uuids"
    ]
    return loop


def _calls(root, func_src):
    nodes = root if isinstance(root, list) else [root]
    return sorted(
        (
            node
            for top in nodes
            for node in ast.walk(top)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == func_src
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )


def _index(body, predicate, what):
    hits = [i for i, stmt in enumerate(body) if predicate(stmt)]
    assert len(hits) == 1, f"expected exactly one {what}, got {hits}"
    return hits[0]


def _is_call_stmt(func_src):
    def predicate(stmt):
        value = stmt.value if isinstance(stmt, (ast.Expr, ast.Assign)) else None
        return isinstance(value, ast.Call) and ast.unparse(value.func) == func_src
    return predicate


def _assigns(name):
    def predicate(stmt):
        return (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == name
        )
    return predicate


def _is_pending_guard(stmt):
    return (
        isinstance(stmt, ast.If)
        and ast.unparse(stmt.test) == "_pending_decode_token is not None"
    )


def _is_finalize_guard(stmt):
    """``if _pending_decode_token is not None:`` finalize + clear, nothing else."""
    if not _is_pending_guard(stmt) or stmt.orelse or len(stmt.body) != 2:
        return False
    finalize, clear = stmt.body
    return (
        ast.unparse(finalize)
        == "self._finalize_pending_decode_token(_pending_decode_token)"
        and ast.unparse(clear) == "_pending_decode_token = None"
    )


def _parents(root):
    parents = {}
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


# ---------------------------------------------------------------------------
# Extracted pending-token state machine
# ---------------------------------------------------------------------------


def _build_token_state_machine(*, rep_detection=False):
    source, tree = _source_and_tree()
    segments = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_PendingDecodeTokenResult":
            segments[node.name] = _segment(source, node)
        elif isinstance(node, ast.FunctionDef) and node.name in (
            "_check_repeating_pattern",
            "_repetition_check_enabled",
        ):
            segments[node.name] = _segment(source, node)
    for method in _worker_class(tree).body:
        if isinstance(method, ast.FunctionDef) and method.name in _HELPERS:
            segments[method.name] = _segment(source, method)
    expected = set(_HELPERS) | {
        "_PendingDecodeTokenResult",
        "_check_repeating_pattern",
        "_repetition_check_enabled",
    }
    assert set(segments) == expected, f"pending-token source drift: {expected - set(segments)}"

    namespace = {
        "dataclass": dataclass,
        "field": field,
        "List": List,
        "SequenceEntry": object,
        "torch": torch,
        "logging": logging,
        "BATCHGEN_CB_DEBUG": False,
        "BATCHGEN_MULTI_BATCH_DIAG": False,
        "REP_DETECTION": rep_detection,
        "SeqEvent": SimpleNamespace(REPETITION="REPETITION"),
        "lifespan": SimpleNamespace(dump_lifespan=lambda *a, **k: None),
    }
    for source_segment in segments.values():
        exec(compile(source_segment, str(WORKER_PATH), "exec"), namespace)
    worker_type = type("FakeWorker", (), {name: namespace[name] for name in _HELPERS})
    return worker_type, namespace["_PendingDecodeTokenResult"]


class _Event:
    """Fake CUDA event: ``synchronize`` marks the readback complete."""

    def __init__(self, trace=None):
        self.trace = trace if trace is not None else []
        self.ready = False

    def synchronize(self):
        self.ready = True
        self.trace.append("synchronize")


class _Seq:
    def __init__(self, name, *, prompt=8, max_decode_length=16, ignore_eos=False):
        self.uuid = name
        self.global_idx = 0
        self.decoded_length = 0
        self.original_prompt_length = prompt
        self.current_context_length = prompt
        self.max_decode_length = max_decode_length
        self.eos_reached = False
        self.ignore_eos = ignore_eos
        self._rep_detected = False
        self._rep_last_token = None
        self._rep_count = 0
        self._lifespan_log = []

    def log_event(self, *args, **kwargs):
        pass

    def state(self):
        return (
            self.decoded_length,
            self.current_context_length,
            self.eos_reached,
            self._rep_detected,
            self._rep_count,
        )


EOS = 99


def _worker(local_indices, *, width=64, rep_detection=False):
    worker_type, pending_type = _build_token_state_machine(rep_detection=rep_detection)
    worker = worker_type()
    worker.rank = 0
    worker.query_book = {
        idx: SimpleNamespace(decoded_tokens=torch.full((1, width), -1, dtype=torch.long))
        for idx in local_indices
    }
    worker._should_stop_at_eos = lambda token_id, seq: token_id == EOS and not seq.ignore_eos
    worker._is_sequence_completed = lambda seq: (
        seq.eos_reached
        or seq._rep_detected
        or seq.decoded_length >= seq.max_decode_length
    )
    return worker, pending_type


# ---------------------------------------------------------------------------
# Unit tests: pending-token finalization
# ---------------------------------------------------------------------------


def test_finalize_waits_for_its_own_readback_before_applying():
    worker, Pending = _worker([7])
    trace = []
    event = _Event(trace)

    def apply(pending):
        assert pending.ready_event.ready, "applied before the token readback completed"
        trace.append("apply")

    worker._apply_pending_decode_token = apply
    pending = Pending(
        ready_event=event,
        tokens_cpu=torch.tensor([[17]]),
        slot=1,
        local_iteration=1,
    )
    worker._finalize_pending_decode_token(pending)
    assert trace == ["synchronize", "apply"]


def test_advance_is_token_independent_and_apply_decides_eos():
    worker, Pending = _worker([7])
    seq = _Seq("a", max_decode_length=4)
    pending = Pending(_Event(), torch.tensor([[EOS]]), 0, 1)

    worker._advance_decode_sequences_for_pending_token([7], [seq], pending.rows)
    # Counters advance before the token value is known; completion does not.
    assert (seq.decoded_length, seq.current_context_length, seq.eos_reached) == (1, 9, False)
    assert pending.rows == [(0, 7, seq, 0)]

    worker._finalize_pending_decode_token(pending)
    assert worker.query_book[7].decoded_tokens[0, 0].item() == EOS
    assert seq.eos_reached is True
    # The next step's advance sees the completion and skips the row.
    rows = []
    worker._advance_decode_sequences_for_pending_token([7], [seq], rows)
    assert rows == [] and (seq.decoded_length, seq.current_context_length) == (1, 9)


def test_two_slot_ring_keeps_the_pending_slot_intact():
    _, Pending = _build_token_state_machine()
    pinned = torch.full((2, 4, 1), -1, dtype=torch.long)
    slots = []
    for local_iteration in (1, 2, 3, 4):
        slot = local_iteration & 1
        assert not slots or slots[-1] != slot, "consecutive steps must alternate slots"
        slots.append(slot)
    assert slots == [1, 0, 1, 0]

    # Step 1 copies into slot 1 and stays pending; step 2's copy into slot 0
    # must leave the rows step 1's finalization has not read yet untouched.
    step1 = Pending(_Event(), pinned[1, :3], 1, 1)
    step1.tokens_cpu.copy_(torch.tensor([[11], [12], [13]]))
    step2_view = pinned[0, :3]
    step2_view.copy_(torch.tensor([[21], [22], [23]]))
    assert step1.tokens_cpu.data_ptr() != step2_view.data_ptr()
    assert step1.tokens_cpu.flatten().tolist() == [11, 12, 13]


def _run_sequential(worker, Pending, batch, seqs, steps):
    """Pre-overlap order: every step's token applied before the next step."""
    for step, tokens in enumerate(steps, start=1):
        pending = Pending(_Event(), torch.tensor(tokens).view(-1, 1), step & 1, step)
        worker._advance_decode_sequences_for_pending_token(batch, seqs, pending.rows)
        worker._finalize_pending_decode_token(pending)


def _run_pipelined(worker, Pending, batch, seqs, steps, *, boundary_every=None):
    """Loop order: apply(prev) -> new pending(cur) -> advance(cur); final drain."""
    pinned = torch.full((2, len(batch), 1), -1, dtype=torch.long)
    pending = None
    for step, tokens in enumerate(steps, start=1):
        if boundary_every and step > 1 and (step - 1) % boundary_every == 0:
            # Page boundary: finalize before any consumer of token state.
            if pending is not None:
                worker._finalize_pending_decode_token(pending)
                pending = None
        slot = step & 1
        assert pending is None or pending.slot != slot
        view = pinned[slot, : len(batch)]
        view.copy_(torch.tensor(tokens).view(-1, 1))
        event = _Event()
        if pending is not None:
            pending.ready_event.synchronize()
            worker._apply_pending_decode_token(pending)
        pending = Pending(event, view, slot, step)
        worker._advance_decode_sequences_for_pending_token(batch, seqs, pending.rows)
    if pending is not None:
        worker._finalize_pending_decode_token(pending)


@pytest.mark.parametrize("boundary_every", [None, 1, 3])
@pytest.mark.parametrize("rep_detection", [False, True])
def test_pipelined_finalization_matches_sequential_exactly(boundary_every, rep_detection):
    batch = [3, 5, 9]
    # Row 0 hits EOS at step 2, row 1 hits max length 4, row 2 repeats one token
    # and runs to its budget (6). Steps continue past every completion, as a
    # completed sequence stays in the forward until the next boundary.
    steps = [
        [1, 4, 7],
        [EOS, 4, 7],
        [2, 4, 7],
        [2, 4, 7],
        [2, 4, 7],
        [2, 4, 7],
        [2, 4, 7],
        [2, 4, 7],
    ]

    def make():
        worker, Pending = _worker(batch, rep_detection=rep_detection)
        seqs = [
            _Seq("a", max_decode_length=8),
            _Seq("b", max_decode_length=4),
            _Seq("c", max_decode_length=6),
        ]
        return worker, Pending, seqs

    ref_worker, ref_pending, ref_seqs = make()
    _run_sequential(ref_worker, ref_pending, batch, ref_seqs, steps)
    worker, Pending, seqs = make()
    _run_pipelined(worker, Pending, batch, seqs, steps, boundary_every=boundary_every)

    assert [s.state() for s in seqs] == [s.state() for s in ref_seqs]
    for idx in batch:
        assert torch.equal(
            worker.query_book[idx].decoded_tokens, ref_worker.query_book[idx].decoded_tokens
        )
    # Exact accounting: tokens written == decoded_length, nothing past it.
    for idx, seq in zip(batch, seqs):
        written = int((worker.query_book[idx].decoded_tokens[0] >= 0).sum())
        assert written == seq.decoded_length
    assert [s.decoded_length for s in seqs] == [2, 4, 6]


def test_apply_resumes_after_an_exception_without_double_applying():
    worker, Pending = _worker([1, 2, 3], rep_detection=True)
    seqs = [_Seq(name) for name in "abc"]
    pending = Pending(_Event(), torch.tensor([[5], [6], [7]]), 0, 2)
    worker._advance_decode_sequences_for_pending_token([1, 2, 3], seqs, pending.rows)

    calls = []
    real = worker._should_stop_at_eos

    def flaky(token_id, seq):
        calls.append(seq.uuid)
        if seq.uuid == "b" and calls.count("b") == 1:
            raise RuntimeError("injected")
        return real(token_id, seq)

    worker._should_stop_at_eos = flaky
    with pytest.raises(RuntimeError, match="injected"):
        worker._finalize_pending_decode_token(pending)
    assert pending.applied == 1

    # The exception-path drain retries the same pending: it resumes at row 1.
    worker._finalize_pending_decode_token(pending)
    assert pending.applied == 3
    assert calls == ["a", "b", "b", "c"]
    assert [s._rep_count for s in seqs] == [1, 1, 1]
    assert [worker.query_book[i].decoded_tokens[0, 0].item() for i in (1, 2, 3)] == [5, 6, 7]

    # A further finalize of a fully applied result is a no-op.
    worker._finalize_pending_decode_token(pending)
    assert calls == ["a", "b", "b", "c"]


def test_advance_records_rows_as_it_goes():
    worker, Pending = _worker([1, 2])
    seqs = [_Seq("a"), _Seq("b")]
    pending = Pending(_Event(), torch.tensor([[1], [2]]), 1, 1)

    original = worker._is_sequence_completed

    def explode_on_second(seq):
        if seq.uuid == "b":
            raise RuntimeError("injected")
        return original(seq)

    worker._is_sequence_completed = explode_on_second
    with pytest.raises(RuntimeError):
        worker._advance_decode_sequences_for_pending_token([1, 2], seqs, pending.rows)
    # The advanced row is already owned by the pending result, so an
    # exception-path finalize still writes its token.
    assert [row[2].uuid for row in pending.rows] == ["a"]
    worker._finalize_pending_decode_token(pending)
    assert worker.query_book[1].decoded_tokens[0, 0].item() == 1
    assert worker.query_book[2].decoded_tokens[0, 0].item() == -1


# ---------------------------------------------------------------------------
# Loop contracts
# ---------------------------------------------------------------------------


def test_readback_ring_has_two_slots_indexed_by_iteration_parity():
    fn = _decoding_continuous()
    allocations = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_new_tokens_pinned" for t in node.targets)
    ]
    assert len(allocations) == 2  # entry allocation + grow-on-demand
    for allocation in allocations:
        assert ast.unparse(allocation.value.func) == "torch.empty"
        assert ast.unparse(allocation.value.args[0]) == "2"
        assert "pin_memory=True" in ast.unparse(allocation.value)

    (events,) = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_token_ready_events" for t in node.targets)
    ]
    assert ast.unparse(events.value) == "[torch.cuda.Event(), torch.cuda.Event()]"

    body = _decode_loop(fn).body
    slot = _index(body, _assigns("_token_slot"), "_token_slot")
    assert ast.unparse(body[slot].value) == "local_iteration & 1"
    view = _index(body, _assigns("_tokens_cpu"), "_tokens_cpu")
    assert ast.unparse(body[view].value) == "_new_tokens_pinned[_token_slot, :bs]"

    # Overrun guard between the slot choice and the copy: a hard error, never
    # a silent overwrite of the slot the pending finalization still reads.
    guard = body[slot + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "_pending_decode_token is not None and _pending_decode_token.slot == _token_slot"
    )
    assert isinstance(guard.body[0], ast.Raise)
    assert slot < view

    (pending,) = [
        stmt for stmt in body if _assigns("_pending_decode_token")(stmt)
        and isinstance(stmt.value, ast.Call)
    ]
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in pending.value.keywords}
    assert kwargs == {
        "ready_event": "_token_ready_events[_token_slot]",
        "tokens_cpu": "_tokens_cpu",
        "slot": "_token_slot",
        "local_iteration": "local_iteration",
    }

    # Growing the buffer finalizes the in-flight token first.
    (grow,) = [
        stmt for stmt in body
        if isinstance(stmt, ast.If) and ast.unparse(stmt.test) == "bs > _new_tokens_pinned.shape[1]"
    ]
    assert _is_finalize_guard(grow.body[0])
    assert _assigns("_new_tokens_pinned")(grow.body[1])


def test_steady_step_order_and_split_slots():
    body = _decode_loop(_decoding_continuous()).body

    fwd_end = _index(body, _is_call_stmt("_fwd_ring.end"), "_fwd_ring.end")
    copy = _index(body, _is_call_stmt("_tokens_cpu.copy_"), "token copy")
    record = _index(body, _is_call_stmt("_token_ready_events[_token_slot].record"), "token event")
    flush = _index(body, _is_call_stmt("self._flush_deferred_kv_to_host"), "host-KV launch")
    t2 = _index(body, _assigns("_split_t2"), "_split_t2")
    wait = _index(
        body,
        lambda s: _is_pending_guard(s)
        and [ast.unparse(b) for b in s.body]
        == ["_pending_decode_token.ready_event.synchronize()"],
        "previous-step token wait",
    )
    t3 = _index(body, _assigns("_split_t3"), "_split_t3")
    apply = _index(
        body,
        lambda s: _is_pending_guard(s)
        and [ast.unparse(b) for b in s.body]
        == ["self._apply_pending_decode_token(_pending_decode_token)"],
        "previous-step apply",
    )
    new_pending = _index(
        body,
        lambda s: _assigns("_pending_decode_token")(s) and isinstance(s.value, ast.Call),
        "new pending result",
    )
    advance = _index(
        body, _is_call_stmt("self._advance_decode_sequences_for_pending_token"), "advance"
    )
    t4 = _index(body, _assigns("_split_t4"), "_split_t4")
    drain = _index(
        body,
        lambda s: isinstance(s, ast.If)
        and ast.unparse(s.test) == "self._pending_kv_append_tasks",
        "per-step host-KV drain",
    )
    t5 = _index(body, _assigns("_split_t5"), "_split_t5")
    accumulate = _index(body, _is_call_stmt("accumulate_decode_step_split"), "split accumulate")

    # kv_launch | readback (PREVIOUS step) | bookkeeping | kv_drain
    assert (
        fwd_end < copy < record < flush < t2
        < wait < t3
        < apply < new_pending < advance < t4
        < drain < t5 < accumulate
    )
    assert ast.unparse(body[advance].value.args[2]) == "_pending_decode_token.rows"
    marks = ast.unparse(body[accumulate].value.args[1])
    assert marks == (
        "(forward_start, _split_t0, _split_t1, _split_t2, _split_t3, _split_t4, _split_t5)"
    )


def test_steady_path_adds_no_device_sync():
    body = _decode_loop(_decoding_continuous()).body
    fwd_end = _index(body, _is_call_stmt("_fwd_ring.end"), "_fwd_ring.end")
    accumulate = _index(body, _is_call_stmt("accumulate_decode_step_split"), "split accumulate")
    steady = body[fwd_end : accumulate + 1]
    syncs = sorted(
        ast.unparse(node.func)
        for stmt in steady
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "synchronize"
    )
    # The only host waits: the previous step's token event, and the existing
    # per-step host-KV drain (whose device sync lives in the drain helper).
    assert syncs == ["_pending_decode_token.ready_event.synchronize"]
    assert [
        ast.unparse(node)
        for stmt in steady
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "self._wait_pending_kv_append_tasks"
    ] == ["self._wait_pending_kv_append_tasks(defer_errors=True)"]
    # Every finalize in the steady segment is the rare buffer-grow path.
    finalizes = _calls(steady, "self._finalize_pending_decode_token")
    assert len(finalizes) == 1


def test_pending_token_is_finalized_before_boundary_and_admission():
    body = _decode_loop(_decoding_continuous()).body
    (boundary,) = [
        stmt
        for stmt in body
        if isinstance(stmt, ast.If)
        and ast.unparse(stmt.test) == "local_iteration - last_boundary >= self.DECISION_INTERVAL"
    ]
    assert _is_finalize_guard(boundary.body[0]), "boundary must finalize the in-flight token first"
    finalize_line = boundary.body[0].lineno
    for consumer in (
        "self._page_boundary_fast",
        "self._poll_admissions",
        "self._put_sequences_on_hold",
        "self._rebuild_input_tokens",
        "self._wait_pending_kv_append_tasks",
        "gpu_manager.rebuild_page_table",
        "self._rebuild_page_table_for_batch",
    ):
        calls = _calls(boundary, consumer)
        assert calls, consumer
        assert all(call.lineno > finalize_line for call in calls), consumer
    # The boundary sits before any of this step's setup or forward.
    assert body.index(boundary) < _index(body, _assigns("forward_start"), "forward_start")


def test_every_input_rebuild_in_the_loop_follows_a_finalize():
    fn = _decoding_continuous()
    loop = _decode_loop(fn)
    parents = _parents(loop)
    rebuilds = _calls(loop, "self._rebuild_input_tokens")
    assert rebuilds
    for call in rebuilds:
        # Walk up to the nearest statement list and require an earlier
        # finalize guard in an enclosing block of the same iteration.
        node = call
        found = False
        while node is not loop and not found:
            parent = parents[node]
            for field_name in ("body", "orelse"):
                block = getattr(parent, field_name, None)
                if isinstance(block, list) and node in block:
                    found = any(_is_finalize_guard(s) for s in block[: block.index(node)])
                    break
            node = parent
        assert found, f"_rebuild_input_tokens at line {call.lineno} may read an unapplied token"


def test_every_finalize_site_clears_the_pending_result_after_success():
    fn = _decoding_continuous()
    parents = _parents(fn)
    handler = _decode_try(fn).handlers[0]
    in_handler = set(ast.walk(handler))
    sites = _calls(fn, "self._finalize_pending_decode_token")
    # boundary, input rebuild, buffer grow, exception handler, loop exit
    assert len(sites) == 5
    for call in sites:
        if call in in_handler:
            continue  # exception path, see the handler tests below
        guard = parents[parents[call]]
        assert _is_finalize_guard(guard), ast.unparse(guard)


def test_loop_exit_drains_the_pending_token_before_cleanup():
    fn = _decoding_continuous()
    try_stmt = _decode_try(fn)
    after = fn.body[fn.body.index(try_stmt) + 1]
    assert _is_finalize_guard(after), "normal loop exit must finalize the in-flight token"
    cleanup = fn.body[fn.body.index(try_stmt) + 2]
    assert ast.unparse(cleanup) == (
        "self._wait_pending_kv_append_tasks(sync_distributed_errors=True)"
    )
    # Nothing outside the decode try/drain touches the pending result.
    for stmt in fn.body[fn.body.index(try_stmt) + 2 :]:
        assert "_pending_decode_token" not in ast.unparse(stmt)


def _exception_handler_runner():
    """Compile the shipping except-handler body into a callable."""
    fn = _decoding_continuous()
    try_stmt = _decode_try(fn)
    assert len(try_stmt.handlers) == 1 and not try_stmt.finalbody and not try_stmt.orelse
    handler = try_stmt.handlers[0]
    assert ast.unparse(handler.type) == "BaseException"
    assert isinstance(handler.body[-1], ast.Raise) and handler.body[-1].exc is None
    handler_src = textwrap.indent("\n".join(ast.unparse(s) for s in handler.body), " " * 8)
    source = (
        "def run(self, _pending_decode_token, original):\n"
        "    try:\n"
        "        raise original\n"
        "    except BaseException:\n"
        f"{handler_src}\n"
    )
    namespace = {"logging": logging}
    exec(compile(source, str(WORKER_PATH), "exec"), namespace)
    return namespace["run"]


def test_exception_exit_finalizes_the_pending_token_and_reraises():
    run = _exception_handler_runner()
    worker, Pending = _worker([4])
    seq = _Seq("a")
    pending = Pending(_Event(), torch.tensor([[42]]), 0, 6)
    worker._advance_decode_sequences_for_pending_token([4], [seq], pending.rows)

    original = ValueError("boundary failed")
    with pytest.raises(ValueError) as info:
        run(worker, pending, original)
    assert info.value is original
    assert pending.ready_event.ready and pending.applied == 1
    assert worker.query_book[4].decoded_tokens[0, 0].item() == 42


def test_exception_exit_never_masks_the_original_error(caplog):
    run = _exception_handler_runner()

    class _Failing:
        rank = 3

        def _finalize_pending_decode_token(self, pending):
            raise RuntimeError("sticky CUDA error")

    pending = SimpleNamespace(local_iteration=9)
    original = KeyError("forward failed")
    with caplog.at_level(logging.ERROR):
        with pytest.raises(KeyError) as info:
            run(_Failing(), pending, original)
    assert info.value is original
    assert "could not finalize the in-flight decode token of step 9" in caplog.text

    # No pending result: the handler only re-raises.
    with pytest.raises(KeyError):
        run(_Failing(), None, KeyError("x"))
