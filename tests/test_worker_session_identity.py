"""Agent-session identity reaches the worker's SequenceEntry and its completion.

A /v1/turns request carries traj_id / session_id / parent_session_id in its
admit entry. The worker stores them next to batch_id, logs them on the owning
rank at admission and at completion, and its completion item reports the
prompt tokens the last prefill took from cache (gathered from the owning rank,
since rank 0 publishes every completion) and the wall-clock admit/complete
stamps the server uses for per-turn overhead.

batchgen_worker imports the whole engine, so the methods are compiled from the
real source against a fake worker (tests/worker_method_source.py).
"""

import logging
import queue
import time
from types import SimpleNamespace

from batchgen.query_book import release_local_query_slot
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from tests.worker_method_source import load_worker_methods

# What the other rank contributes to the next all_gather_object.
_PEER = {"payload": {}}


def _all_gather_object(output, local):
    output[0] = local
    output[1] = _PEER["payload"]


_METHODS = load_worker_methods(
    ["_admit_sequences_from_message", "_report_completion", "_gather_completed_tokens"],
    {
        "SequenceEntry": SequenceEntry,
        "logging": logging,
        "time": time,
        "release_local_query_slot": release_local_query_slot,
        "_release_tokenizer_arenas": lambda **_: None,
        "dist": SimpleNamespace(all_gather_object=_all_gather_object),
    },
)


class _Worker:
    world_size = 2
    max_input_length = 1 << 20
    max_decoding_length = 64
    _trajectory_book = None

    _admit_sequences_from_message = _METHODS["_admit_sequences_from_message"]
    _report_completion = _METHODS["_report_completion"]
    _gather_completed_tokens = _METHODS["_gather_completed_tokens"]

    def __init__(self, rank, owner_rank=0):
        self.rank = rank
        self.owner_rank = owner_rank
        self.global_batch = SequenceBatch()
        self._group_uuids = set()
        self._uuid_to_local_map = {}
        self._local_to_uuid_map = {}
        self.query_book = {}
        self._free_local_indices = set()
        self._prefix_sequence_states = {}
        self._response_queue = queue.Queue() if rank == 0 else None

    def _tokenize_admitted_sequences(self, uuids):
        for uuid in uuids:
            seq = self.global_batch.get_sequence(uuid)
            seq.prompt_length = len(seq.text)
            seq.original_prompt_length = seq.prompt_length
            seq.assigned_rank = self.owner_rank

    def _update_config_after_tokenization(self):
        pass

    def _build_local_query_book_for_admitted(self, uuids):
        pass

    def _publish_trajectory_pool_capacity(self):
        pass

    def _maybe_emit_prefix_drain_metric(self):
        pass

    def _get_finish_reason(self, seq):
        return "stop"

    def _trajectory_decoded_tokens(self, seq):
        return [1, 2, 3]

    def _decode_tokens_to_string(self, tokens):
        return "answer"


_TURN_ENTRY = {
    "request_id": "turn_abc",
    "text": "x" * 40,
    "batch_id": "turn_abc",
    "max_tokens": 16,
    "traj_id": "traj-1",
    "session_id": "sess-sub",
    "parent_session_id": "sess-main",
}


def _complete(worker, uuid):
    for status in (SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED,
                   SequenceStatus.COMPLETED):
        worker.global_batch.update_status(uuid, status)


def test_admit_stores_the_identity_next_to_batch_id(caplog):
    worker = _Worker(rank=0)
    before = time.time()
    with caplog.at_level(logging.INFO):
        worker._admit_sequences_from_message({"entries": [
            _TURN_ENTRY,
            {"request_id": "r@batch_1", "text": "batch prompt", "batch_id": "batch_1"},
        ]})
    turn = worker.global_batch.get_sequence("turn_abc")
    assert (turn.batch_id, turn.traj_id, turn.session_id, turn.parent_session_id) == (
        "turn_abc", "traj-1", "sess-sub", "sess-main")
    assert before <= turn.admitted_at <= time.time()
    batch = worker.global_batch.get_sequence("r@batch_1")
    assert (batch.traj_id, batch.session_id, batch.parent_session_id) == (None, None, None)
    admits = [r.getMessage() for r in caplog.records if "[SESSION] admit" in r.getMessage()]
    assert admits == [
        "[SESSION] admit request=turn_abc traj=traj-1 session=sess-sub "
        "parent=sess-main prompt_tokens=40 rank=0"
    ]


def test_only_the_owning_rank_logs_the_identity(caplog):
    worker = _Worker(rank=1, owner_rank=0)
    with caplog.at_level(logging.INFO):
        worker._admit_sequences_from_message({"entries": [_TURN_ENTRY]})
    assert worker.global_batch.get_sequence("turn_abc").session_id == "sess-sub"
    assert not [r for r in caplog.records if "[SESSION]" in r.getMessage()]


def test_gather_brings_the_owners_cached_tokens_to_rank_zero():
    """Rank 0 does not own the sequence; the owner (rank 1) reports 24 cached."""
    worker = _Worker(rank=0, owner_rank=1)
    worker._admit_sequences_from_message({"entries": [_TURN_ENTRY]})
    _PEER["payload"] = {"turn_abc": ("from owner", 24)}
    try:
        texts = worker._gather_completed_tokens(["turn_abc"])
    finally:
        _PEER["payload"] = {}
    assert texts == {"turn_abc": "from owner"}
    assert worker.global_batch.get_sequence("turn_abc").prefix_cached_tokens == 24


def test_owner_reads_cached_tokens_from_its_prefix_state():
    worker = _Worker(rank=0, owner_rank=0)
    worker._admit_sequences_from_message({"entries": [_TURN_ENTRY]})
    seq = worker.global_batch.get_sequence("turn_abc")
    worker._uuid_to_local_map["turn_abc"] = 0
    worker.query_book[0] = object()
    worker._prefix_sequence_states[seq.global_idx] = SimpleNamespace(
        compute_cached_tokens=32
    )
    assert worker._gather_completed_tokens(["turn_abc"]) == {"turn_abc": "answer"}
    assert seq.prefix_cached_tokens == 32


def test_completion_reports_cached_computed_and_stamps(caplog):
    worker = _Worker(rank=0)
    worker._admit_sequences_from_message({"entries": [_TURN_ENTRY]})
    seq = worker.global_batch.get_sequence("turn_abc")
    seq.prefix_cached_tokens = 32
    seq.decoded_length = 5
    _complete(worker, "turn_abc")
    with caplog.at_level(logging.INFO):
        worker._report_completion("turn_abc", gathered_text="answer")
    item = worker._response_queue.get_nowait()
    assert item["request_id"] == "turn_abc"
    assert (item["cached_tokens"], item["computed_tokens"]) == (32, 8)
    assert item["admitted_at"] == seq.admitted_at
    assert item["admitted_at"] <= item["completed_at"] <= time.time()
    done = [r.getMessage() for r in caplog.records if "[SESSION] complete" in r.getMessage()]
    assert done == [
        "[SESSION] complete request=turn_abc traj=traj-1 session=sess-sub "
        "parent=sess-main prompt_tokens=40 cached_tokens=32 computed_tokens=8 "
        "decoded_tokens=5 rank=0"
    ]


def test_batch_completion_without_prefix_cache_reports_zero_cached():
    worker = _Worker(rank=0)
    worker._admit_sequences_from_message({"entries": [
        {"request_id": "r@batch_1", "text": "y" * 12, "batch_id": "batch_1"},
    ]})
    _complete(worker, "r@batch_1")
    worker._report_completion("r@batch_1", gathered_text="answer")
    item = worker._response_queue.get_nowait()
    assert (item["cached_tokens"], item["computed_tokens"]) == (0, 12)
