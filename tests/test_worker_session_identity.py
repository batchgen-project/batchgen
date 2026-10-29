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

import pytest

from batchgen.query_book import release_local_query_slot
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from tests.worker_method_source import load_worker_methods

# What the other rank contributes to the next all_gather_object.
_PEER = {"payload": {}}


def _all_gather_object(output, local):
    output[0] = local
    output[1] = _PEER["payload"]


_METHODS = load_worker_methods(
    ["_admit_sequences_from_message", "_report_completion", "_gather_completed_tokens",
     "_owns_host_kv", "_decode_attn_tp_size"],
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
    _owns_host_kv = _METHODS["_owns_host_kv"]
    _decode_attn_tp_size = _METHODS["_decode_attn_tp_size"]

    def __init__(self, rank, owner_rank=0, attn_tp_size=1):
        self.rank = rank
        self.owner_rank = owner_rank
        self.parallel_manager = SimpleNamespace(attn_tp_size=attn_tp_size)
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


def _bind_locally(worker, uuid, cached=None):
    """Hold ``uuid`` in this rank's local maps; ``cached`` is its prefix state."""
    seq = worker.global_batch.get_sequence(uuid)
    worker._uuid_to_local_map[uuid] = 0
    worker.query_book[0] = object()
    if cached is not None:
        worker._prefix_sequence_states[seq.global_idx] = SimpleNamespace(
            compute_cached_tokens=cached)
    return seq


def _g2_group():
    """World 2, G=2: both ranks serve group 0; rank 0 is its host-KV leader.

    The follower holds the sequence too, but its prefix state was never
    populated (the leader drives prefix lookups), so it would report 0.
    """
    workers = []
    for rank in (0, 1):
        worker = _Worker(rank=rank, owner_rank=0, attn_tp_size=2)
        worker._admit_sequences_from_message({"entries": [_TURN_ENTRY]})
        seq = _bind_locally(worker, "turn_abc", cached=24 if rank == 0 else None)
        seq.decode_dp_group = 0
        workers.append(worker)
    return workers


def _gather_on_all_ranks(workers, uuids):
    """Run the collective on every rank: each sees all ranks' payloads in order."""
    payloads = {}

    def capture(rank):
        def gather(output, local):
            payloads[rank] = local
        return gather

    original = _METHODS["_gather_completed_tokens"].__globals__["dist"]
    try:
        for worker in workers:
            _METHODS["_gather_completed_tokens"].__globals__["dist"] = SimpleNamespace(
                all_gather_object=capture(worker.rank))
            worker._gather_completed_tokens(uuids)
        results = []
        for worker in workers:
            def replay(output, local):
                for rank in range(len(output)):
                    output[rank] = payloads[rank]
            _METHODS["_gather_completed_tokens"].__globals__["dist"] = SimpleNamespace(
                all_gather_object=replay)
            results.append(worker._gather_completed_tokens(uuids))
        return results
    finally:
        _METHODS["_gather_completed_tokens"].__globals__["dist"] = original


def test_g2_follower_does_not_overwrite_the_leaders_cached_tokens():
    workers = _g2_group()
    texts = _gather_on_all_ranks(workers, ["turn_abc"])
    assert texts == [{"turn_abc": "answer"}] * 2
    assert [w.global_batch.get_sequence("turn_abc").prefix_cached_tokens
            for w in workers] == [24, 24]


def test_g2_only_the_host_kv_leader_contributes_a_cached_count():
    leader, follower = _g2_group()
    seen = {}
    original = _METHODS["_gather_completed_tokens"].__globals__["dist"]

    def record(output, local):
        seen["local"] = local
        output[0] = output[1] = {}

    _METHODS["_gather_completed_tokens"].__globals__["dist"] = SimpleNamespace(
        all_gather_object=record)
    try:
        follower._gather_completed_tokens(["turn_abc"])
    finally:
        _METHODS["_gather_completed_tokens"].__globals__["dist"] = original
    assert seen["local"] == {"turn_abc": ("answer", None)}


def test_disagreeing_owner_counts_for_one_uuid_raise():
    worker = _Worker(rank=0, owner_rank=0)
    worker._admit_sequences_from_message({"entries": [_TURN_ENTRY]})
    _bind_locally(worker, "turn_abc", cached=24)
    _PEER["payload"] = {"turn_abc": ("answer", 30)}
    try:
        with pytest.raises(RuntimeError, match="disagree between host-KV owners"):
            worker._gather_completed_tokens(["turn_abc"])
    finally:
        _PEER["payload"] = {}
