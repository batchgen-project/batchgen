"""Focused regression coverage for the one-step GLM-5 decode-token overlap.

The token bookkeeping helpers are extracted from the shipping worker source and
executed with CPU tensors.  The enqueue/drain state itself is still local to the
large ``decoding_continuous`` loop, so those inaccessible transitions are
checked as narrow AST contracts rather than by importing the GPU worker.
"""

import ast
import gc
import logging
import textwrap
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch


WORKER_PATH = Path(__file__).resolve().parents[1] / "batchgen" / "batchgen_worker.py"


def _worker_source_and_tree():
    source = WORKER_PATH.read_text()
    return source, ast.parse(source)


def _worker_method_node(name):
    _, tree = _worker_source_and_tree()
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker"
    )
    return next(
        node
        for node in worker.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _source_for_node(source, node):
    lines = source.splitlines(keepends=True)
    first_line = min(
        [node.lineno]
        + [decorator.lineno for decorator in getattr(node, "decorator_list", [])]
    )
    return textwrap.dedent("".join(lines[first_line - 1 : node.end_lineno]))


def _build_token_state_machine():
    """Extract the real pending-token type and helper methods onto a fake worker."""
    source, tree = _worker_source_and_tree()
    wanted_methods = {
        "_advance_decode_sequences_for_pending_token",
        "_finalize_pending_decode_token",
        "_apply_pending_decode_token",
    }
    segments = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "_PendingDecodeTokenResult":
            segments[node.name] = _source_for_node(source, node)
        elif isinstance(node, ast.ClassDef) and node.name == "BatchGenWorker":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in wanted_methods:
                    segments[method.name] = _source_for_node(source, method)

    expected = wanted_methods | {"_PendingDecodeTokenResult"}
    assert set(segments) == expected, f"pending-token source drift: {expected - set(segments)}"

    namespace = {
        "dataclass": dataclass,
        "List": List,
        "SequenceEntry": object,
        "torch": torch,
        "logging": logging,
        "BATCHGEN_CB_DEBUG": False,
        "BATCHGEN_MULTI_BATCH_DIAG": False,
        "REP_DETECTION": False,
    }
    exec(compile(segments["_PendingDecodeTokenResult"], str(WORKER_PATH), "exec"), namespace)
    for name in wanted_methods:
        exec(compile(segments[name], str(WORKER_PATH), "exec"), namespace)

    FakeWorker = type("FakeWorker", (), {name: namespace[name] for name in wanted_methods})
    return FakeWorker, namespace["_PendingDecodeTokenResult"]


def _calls(nodes):
    if not isinstance(nodes, list):
        nodes = [nodes]
    calls = [
        node
        for root in nodes
        for node in ast.walk(root)
        if isinstance(node, ast.Call)
    ]
    return sorted(calls, key=lambda node: (node.lineno, node.col_offset))


def _call_name(call):
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    if isinstance(call.func, ast.Name):
        return call.func.id
    return None


def _calls_named(nodes, name):
    return [call for call in _calls(nodes) if _call_name(call) == name]


def _assignments_named(nodes, name):
    if not isinstance(nodes, list):
        nodes = [nodes]
    assignments = []
    for root in nodes:
        for node in ast.walk(root):
            if not isinstance(node, ast.Assign):
                continue
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                assignments.append(node)
    return sorted(assignments, key=lambda node: (node.lineno, node.col_offset))


def _overlap_if(method):
    matches = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "_overlap_this_step"
    ]
    assert len(matches) == 1
    return matches[0]


class _ReadyEvent:
    def __init__(self, trace=None):
        self.trace = trace if trace is not None else []
        self.ready = False

    def synchronize(self):
        self.ready = True
        self.trace.append("synchronize")


def _token_worker():
    FakeWorker, Pending = _build_token_state_machine()
    worker = FakeWorker()
    worker.rank = 0
    worker.query_book = {
        7: SimpleNamespace(decoded_tokens=torch.full((1, 4), -1, dtype=torch.long))
    }
    worker._is_sequence_completed = lambda seq: seq.eos_reached
    return worker, Pending


def _sequence(*, max_decode_length):
    return SimpleNamespace(
        decoded_length=0,
        current_context_length=8,
        max_decode_length=max_decode_length,
        eos_reached=False,
    )


def test_current_graph_and_d2h_are_enqueued_before_prior_token_wait():
    method = _worker_method_node("decoding_continuous")
    overlap = _overlap_if(method)

    graph_replay = next(
        call
        for call in _calls_named(method, "replay")
        if call.args
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == "glm5_whole_model"
    )
    token_copy = next(
        call
        for call in _calls_named(method, "copy_")
        if ast.unparse(call.func.value) == "_tokens_cpu"
    )
    event_record = next(
        call
        for call in _calls_named(method, "record")
        if ast.unparse(call.func.value).startswith("_token_ready_events[")
    )
    prior_finalize = _calls_named(overlap.body, "_finalize_pending_decode_token")

    assert len(prior_finalize) == 1
    assert graph_replay.lineno < token_copy.lineno < event_record.lineno
    assert event_record.lineno < prior_finalize[0].lineno


def test_graph_window_timing_samples_replay_without_disabling_overlap():
    method = _worker_method_node("decoding_continuous")
    timing = _assignments_named(method, "_glm5_graph_window_timing")
    sample = _assignments_named(method, "_glm5_graph_window_sample")

    assert len(timing) == len(sample) == 1
    assert "'glm5_graph_window_timing'" in ast.unparse(timing[0].value)
    sample_condition = ast.unparse(sample[0].value)
    for term in (
        "_glm5_graph_window_timing",
        "self, '_glm5_whole_model_graph'",
        "len(_glm5_graph_window_samples) < 8",
        "local_iteration - last_boundary + 1",
        "self.DECISION_INTERVAL // 8",
    ):
        assert term in sample_condition

    starts = _calls_named(method, "_start_glm5_graph_window_sample")
    finishes = _calls_named(method, "_finish_glm5_graph_window_sample")
    adapter_replay = next(
        call
        for call in _calls_named(method, "replay")
        if call.args and ast.unparse(call.args[0]) == "_wm_seg_name"
    )
    direct_replay = next(
        call
        for call in _calls_named(method, "replay")
        if call.args and ast.unparse(call.args[0]) == "'glm5_whole_model'"
    )

    assert len(starts) == len(finishes) == 2
    assert starts[0].lineno < adapter_replay.lineno < finishes[0].lineno
    assert starts[1].lineno < direct_replay.lineno < finishes[1].lineno
    assert "_glm5_graph_window_timing" not in ast.unparse(_overlap_if(method).test)

    # Boundaries run before the replay at iterations 128, 256, ... .  For the
    # fixed 1,023-replay trace, the shipping predicate therefore yields eight
    # samples in each of eight windows, including the cleanup-resolved tail.
    last_boundary = 0
    windows = []
    samples = []
    for local_iteration in range(1, 1024):
        if local_iteration - last_boundary >= 128:
            windows.append(samples)
            samples = []
            last_boundary = local_iteration
        if len(samples) < 8 and (local_iteration - last_boundary + 1) % 16 == 0:
            samples.append(local_iteration)
    windows.append(samples)
    assert len(windows) == 8
    assert all(len(window) == 8 for window in windows)
    assert windows[0] == [15, 31, 47, 63, 79, 95, 111, 127]
    assert windows[-1] == [911, 927, 943, 959, 975, 991, 1007, 1023]


def test_finalize_waits_for_readback_before_applying_cpu_state():
    worker, Pending = _token_worker()
    trace = []
    event = _ReadyEvent(trace)

    def apply(pending):
        assert pending.ready_event.ready
        trace.append("apply")

    worker._apply_pending_decode_token = apply
    pending = Pending(event, torch.tensor([[17]], dtype=torch.long), [], 1)

    worker._finalize_pending_decode_token(pending)

    assert trace == ["synchronize", "apply"]


def test_two_slots_alternate_and_keep_pending_tensor_storage_alive():
    method = _worker_method_node("decoding_continuous")
    allocations = _assignments_named(method, "_new_tokens_pinned")
    assert len(allocations) == 2
    assert all(
        isinstance(assignment.value, ast.Call)
        and assignment.value.args
        and isinstance(assignment.value.args[0], ast.Constant)
        and assignment.value.args[0].value == 2
        for assignment in allocations
    )

    event_assignment = _assignments_named(method, "_token_ready_events")
    assert len(event_assignment) == 1
    assert isinstance(event_assignment[0].value, ast.List)
    assert len(event_assignment[0].value.elts) == 2
    assert all(
        isinstance(event, ast.Call) and _call_name(event) == "Event"
        for event in event_assignment[0].value.elts
    )

    slot_assignment = _assignments_named(method, "_token_slot")
    assert len(slot_assignment) == 1
    assert ast.unparse(slot_assignment[0].value) == "local_iteration & 1"

    overlap = _overlap_if(method)
    prior_finalize = _calls_named(overlap.body, "_finalize_pending_decode_token")
    next_pending = _assignments_named(overlap.body, "_pending_decode_token")[-1]
    assert prior_finalize[0].lineno < next_pending.lineno
    assert "ready_event=_token_ready_events[_token_slot]" in ast.unparse(next_pending.value)
    assert "tokens_cpu=_tokens_cpu" in ast.unparse(next_pending.value)

    resize = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and _assignments_named(node.body, "_new_tokens_pinned")
    )
    resize_finalize = _calls_named(resize.body, "_finalize_pending_decode_token")
    resize_allocation = _assignments_named(resize.body, "_new_tokens_pinned")
    assert len(resize_finalize) == len(resize_allocation) == 1
    assert resize_finalize[0].lineno < resize_allocation[0].lineno

    _, Pending = _build_token_state_machine()
    backing = torch.empty((2, 2, 1), dtype=torch.long)
    slot_zero = backing[0, :1]
    slot_one = backing[1, :1]
    backing_ref = weakref.ref(backing)
    slot_zero_ref = weakref.ref(slot_zero)
    pending_zero = Pending(_ReadyEvent(), slot_zero, [], 1)
    pending_one = Pending(_ReadyEvent(), slot_one, [], 2)

    assert pending_zero.tokens_cpu.data_ptr() != pending_one.tokens_cpu.data_ptr()
    del backing, slot_zero, slot_one
    gc.collect()
    assert slot_zero_ref() is pending_zero.tokens_cpu
    assert backing_ref() is not None


def test_eos_is_applied_after_wait_and_blocks_advancing_inflight_extra_result():
    worker, Pending = _token_worker()
    worker._should_stop_at_eos = lambda token_id: token_id == 99
    seq = _sequence(max_decode_length=4)

    rows = worker._advance_decode_sequences_for_pending_token([7], [seq])
    assert seq.decoded_length == 1
    assert seq.current_context_length == 9
    assert seq.eos_reached is False

    event = _ReadyEvent()
    worker._finalize_pending_decode_token(
        Pending(event, torch.tensor([[99]], dtype=torch.long), rows, 1)
    )

    assert event.ready is True
    assert worker.query_book[7].decoded_tokens[0, 0].item() == 99
    assert seq.eos_reached is True
    assert worker._advance_decode_sequences_for_pending_token([7], [seq]) == []
    assert seq.decoded_length == 1
    assert seq.current_context_length == 9


def test_max_length_is_applied_after_wait_and_blocks_further_advancement():
    worker, Pending = _token_worker()
    worker._should_stop_at_eos = lambda token_id: False
    seq = _sequence(max_decode_length=1)

    rows = worker._advance_decode_sequences_for_pending_token([7], [seq])
    assert seq.decoded_length == 1
    assert seq.eos_reached is False

    worker._finalize_pending_decode_token(
        Pending(_ReadyEvent(), torch.tensor([[23]], dtype=torch.long), rows, 1)
    )

    assert seq.eos_reached is True
    assert worker._advance_decode_sequences_for_pending_token([7], [seq]) == []
    assert seq.decoded_length == 1
    assert seq.current_context_length == 9


def test_pending_result_is_drained_before_boundary_and_cleanup_consumers():
    method = _worker_method_node("decoding_continuous")
    boundary = next(
        node
        for node in ast.walk(method)
        if isinstance(node, ast.If)
        and ast.unparse(node.test)
        == "local_iteration - last_boundary >= self.DECISION_INTERVAL"
    )
    boundary_finalize = _calls_named(boundary.body[0], "_finalize_pending_decode_token")
    boundary_call = _calls_named(boundary.body, "_page_boundary_fast")
    boundary_clear = _assignments_named(boundary.body[0], "_pending_decode_token")
    boundary_resolution = _calls_named(
        boundary.body, "_log_glm5_graph_window_timing"
    )

    assert all(
        len(group) == 1
        for group in (
            boundary_finalize,
            boundary_clear,
            boundary_resolution,
            boundary_call,
        )
    )
    assert isinstance(boundary_clear[0].value, ast.Constant)
    assert boundary_clear[0].value.value is None
    assert (
        boundary_finalize[0].lineno
        < boundary_clear[0].lineno
        < boundary_resolution[0].lineno
        < boundary_call[0].lineno
    )
    assert ast.unparse(boundary_resolution[0].args[0]) == "'boundary'"

    decode_loop = next(node for node in method.body if isinstance(node, ast.While))
    loop_index = method.body.index(decode_loop)
    cleanup_drain = method.body[loop_index + 1]
    cleanup_resolution = method.body[loop_index + 2]
    cleanup_flush = method.body[loop_index + 3]

    assert isinstance(cleanup_drain, ast.If)
    assert ast.unparse(cleanup_drain.test) == "_pending_decode_token is not None"
    assert len(_calls_named(cleanup_drain, "_finalize_pending_decode_token")) == 1
    assert len(_assignments_named(cleanup_drain, "_pending_decode_token")) == 1
    assert ast.unparse(cleanup_resolution) == (
        "_log_glm5_graph_window_timing('cleanup')"
    )
    assert len(_calls_named(cleanup_flush, "_flush_boundary_dirty_kv_ranges")) == 1
    assert cleanup_drain.end_lineno < cleanup_resolution.lineno < cleanup_flush.lineno

    resolver = next(
        node
        for node in method.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_log_glm5_graph_window_timing"
    )
    assert len(_calls_named(resolver, "elapsed_time")) == 1
    assert not _calls_named(resolver, "synchronize")


def test_non_overlap_fallback_preserves_synchronous_update_order():
    method = _worker_method_node("decoding_continuous")
    fallback = _overlap_if(method).orelse

    finalize = _calls_named(fallback, "_finalize_pending_decode_token")
    flush = _calls_named(fallback, "_flush_deferred_kv_to_host")
    synchronize = _calls_named(fallback, "synchronize")
    mark_stale = _calls_named(fallback, "_mark_suppressed_decode_host_kv_stale")
    advance = _calls_named(fallback, "_advance_decode_sequences_for_pending_token")
    apply = _calls_named(fallback, "_apply_pending_decode_token")

    assert all(
        len(group) == 1
        for group in (finalize, flush, synchronize, mark_stale, advance, apply)
    )
    assert (
        finalize[0].lineno
        < flush[0].lineno
        < synchronize[0].lineno
        < mark_stale[0].lineno
        < advance[0].lineno
        < apply[0].lineno
    )
    assert "_PendingDecodeTokenResult" in ast.unparse(apply[0])
