"""The prefix_drain metric row records active_attachments once per drain.

Every completion path reports through _report_completion after the sequence
reached COMPLETED and its Host pages and prefix attachments were released, so
that is where the worker observes the live set draining to empty. The row is
edge-triggered: one per drain, re-armed by admission.

batchgen_worker imports the whole engine, so the completion, drain and
admission methods are compiled in isolation against a fake worker.
"""

import ast
import copy
import logging
from pathlib import Path

from batchgen.query_book import release_local_query_slot
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus

WORKER = Path(__file__).resolve().parents[2] / "batchgen" / "batchgen_worker.py"
TREE = ast.parse(WORKER.read_text())


def _method(name):
    return next(
        node for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


_NAMESPACE = {
    "SequenceEntry": SequenceEntry,
    "logging": logging,
    "release_local_query_slot": release_local_query_slot,
    "_release_tokenizer_arenas": lambda **_: None,
}
exec(compile(ast.fix_missing_locations(ast.Module(
    body=[
        copy.deepcopy(_method(name)) for name in (
            "_report_completion",
            "_maybe_emit_prefix_drain_metric",
            "_trajectory_restore_fields",
            "_admit_sequences_from_message",
        )
    ],
    type_ignores=[],
)), str(WORKER), "exec"), _NAMESPACE)


class _Worker:
    rank = 1
    world_size = 1
    max_input_length = 1 << 20
    max_decoding_length = 64
    _trajectory_book = None
    _response_queue = None

    _report_completion = _NAMESPACE["_report_completion"]
    _maybe_emit_prefix_drain_metric = _NAMESPACE["_maybe_emit_prefix_drain_metric"]
    _trajectory_restore_fields = _NAMESPACE["_trajectory_restore_fields"]
    _admit_sequences_from_message = _NAMESPACE["_admit_sequences_from_message"]

    def __init__(self, *, enable_prefix_cache=True, debug_stats=True):
        self.enable_prefix_cache = enable_prefix_cache
        self.prefix_cache_debug_stats = debug_stats
        self._prefix_drain_metric_emitted = False
        self.global_batch = SequenceBatch()
        self._group_uuids = set()
        self._uuid_to_local_map = {}
        self._local_to_uuid_map = {}
        self.query_book = {}
        self._free_local_indices = set()
        self.active_attachments = 0
        # M3 Slice 2: a run that never turned the trajectory flag on has no
        # restorer, so the drain row carries no trajectory fields.
        self._trajectory_restorer = None
        self.emitted = []

    def _publish_trajectory_pool_capacity(self):
        pass

    def _emit_prefix_cache_metric(self, phase, **fields):
        self.emitted.append((phase, fields))

    def _prefix_host_resource_fields(self):
        return {"active_attachments": self.active_attachments, "resident_nodes": 3}

    # Admission collaborators: tokenization and query-book binding are out of
    # scope; the drain edge only needs the sequences to enter global_batch.
    def _tokenize_admitted_sequences(self, uuids):
        for uuid in uuids:
            self.global_batch.get_sequence(uuid).prompt_length = 4

    def _update_config_after_tokenization(self):
        pass

    def _build_local_query_book_for_admitted(self, uuids):
        pass


def _admit(worker, *request_ids):
    worker._admit_sequences_from_message({"entries": [
        {"request_id": rid, "text": "prompt", "batch_id": "b"}
        for rid in request_ids
    ]})


def _to_status(worker, uuid, *path):
    for status in path:
        worker.global_batch.update_status(uuid, status)


def _complete(worker, uuid):
    """Mirror every completion site: status first, then _report_completion."""
    _to_status(
        worker, uuid,
        SequenceStatus.IN_PREFILL,
        SequenceStatus.PREFILLED,
        SequenceStatus.COMPLETED,
    )
    worker._report_completion(uuid)


def _drain_rows(worker):
    return [fields for phase, fields in worker.emitted if phase == "prefix_drain"]


def test_drain_emits_once_per_edge_and_rearms_on_admission():
    worker = _Worker()
    _admit(worker, "a", "b")

    # (a) a live sequence remains -> no row.
    _complete(worker, "a")
    assert _drain_rows(worker) == []

    # (b) the last live sequence completes -> exactly one row, carrying the
    # Host resource snapshot (active_attachments included) and nothing more.
    _complete(worker, "b")
    assert worker.emitted == [
        ("prefix_drain", {"active_attachments": 0, "resident_nodes": 3})
    ]

    # (c) further completion reports while still drained -> no second row.
    worker._report_completion("a")
    worker._report_completion("b")
    assert len(_drain_rows(worker)) == 1

    # (d) admission re-arms; the next drain emits a second row.
    _admit(worker, "c")
    assert len(_drain_rows(worker)) == 1
    worker.active_attachments = 2
    _complete(worker, "c")
    assert _drain_rows(worker) == [
        {"active_attachments": 0, "resident_nodes": 3},
        {"active_attachments": 2, "resident_nodes": 3},
    ]


def test_every_non_completed_status_counts_as_live():
    for path in (
        (),
        (SequenceStatus.IN_PREFILL,),
        (SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED),
        (SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED, SequenceStatus.IN_DECODE),
        (
            SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
            SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD,
        ),
        (
            SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
            SequenceStatus.IN_DECODE, SequenceStatus.EVICTED,
        ),
    ):
        worker = _Worker()
        _admit(worker, "done", "live")
        _to_status(worker, "live", *path)
        _complete(worker, "done")
        assert _drain_rows(worker) == [], path


def test_empty_admission_does_not_rearm():
    worker = _Worker()
    _admit(worker, "a")
    _complete(worker, "a")
    worker._admit_sequences_from_message({"entries": []})
    worker._report_completion("a")
    assert len(_drain_rows(worker)) == 1


def test_no_row_without_prefix_cache_or_debug_stats():
    for kwargs in (
        {"enable_prefix_cache": False},
        {"debug_stats": False},
    ):
        worker = _Worker(**kwargs)
        _admit(worker, "a")
        _complete(worker, "a")
        assert worker.emitted == [], kwargs
        assert worker._prefix_drain_metric_emitted is False


def _calls(node, attr):
    return sorted(
        call.lineno for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == attr
    )


def test_every_completion_site_releases_before_reporting():
    """_report_completion is the single completion funnel: each caller has
    released the completed sequences' Host pages (which drops their prefix
    attachments) before its report loop, so the drain row sees the released
    state."""
    callers = {
        node.name: node for node in ast.walk(TREE)
        if isinstance(node, ast.FunctionDef)
        and node.name != "_report_completion"
        and _calls(node, "_report_completion")
    }
    assert set(callers) == {
        "_finish_prefill_completed_sequences",
        "generate",
        "_page_boundary_fast",
        "_decoding_legacy_modes",
    }
    for name, node in callers.items():
        releases = _calls(node, "_release_host_kv_pages_for_batch")
        previous_report = 0
        for report in _calls(node, "_report_completion"):
            assert any(previous_report < line < report for line in releases), (
                name, report,
            )
            previous_report = report


def test_the_drain_row_carries_the_trajectory_attachment_counter():
    """M3 Slice 2: post-drain live attachments are evidence, not a guess.

    The row is the place the gate reads "post-drain attachments == 0" from,
    for the trajectory path beside the coordinator's own active_attachments.
    """

    class _Restorer:
        def counters(self):
            return {
                "trajectory_lookups": 4,
                "trajectory_hits": 2,
                "trajectory_restores": 1,
                "trajectory_live_attachments": 0,
                "trajectory_declines": 1,
            }

    worker = _Worker()
    worker._trajectory_restorer = _Restorer()
    _admit(worker, "a")
    _complete(worker, "a")

    rows = _drain_rows(worker)
    assert len(rows) == 1
    assert rows[0]["active_attachments"] == 0
    assert rows[0]["trajectory_live_attachments"] == 0
    assert rows[0]["trajectory_restores"] == 1
    assert rows[0]["trajectory_declines"] == 1
