"""Unit tests for `batchgen.worker.indexing`.

Real fixtures only — no mocks of `SequenceBatch` / `SequenceEntry` per the
Phase A §G no-hack rule. Tests run CPU-only and require no GPU.
"""

from __future__ import annotations

import logging

import pytest

from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from batchgen.worker.indexing import IndexLookupRequest, IndexManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_seq(uuid: str, global_idx: int, rank: int = 0) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=8,
        max_decode_length=8,
    )
    seq.assigned_rank = rank
    return seq


@pytest.fixture
def two_rank_batch() -> SequenceBatch:
    """SequenceBatch with 4 sequences split across two ranks.

    rank 0 owns: alice (global=0, local=0), bob (global=2, local=1)
    rank 1 owns: carol (global=1, local=0), dave (global=3, local=1)
    """
    batch = SequenceBatch()
    for seq in [
        _make_seq("alice", global_idx=0, rank=0),
        _make_seq("bob", global_idx=2, rank=0),
        _make_seq("carol", global_idx=1, rank=1),
        _make_seq("dave", global_idx=3, rank=1),
    ]:
        batch.add_sequence(seq)
        batch.assign_rank(seq.uuid, seq.assigned_rank)
    return batch


@pytest.fixture
def rank0_req(two_rank_batch) -> IndexLookupRequest:
    return IndexLookupRequest(
        rank=0,
        local_to_uuid={0: "alice", 1: "bob"},
        uuid_to_local={"alice": 0, "bob": 1},
        global_batch=two_rank_batch,
    )


# ---------------------------------------------------------------------------
# IndexLookupRequest dataclass behavior
# ---------------------------------------------------------------------------

def test_request_is_frozen(rank0_req):
    with pytest.raises((AttributeError, Exception)):
        rank0_req.rank = 99  # type: ignore[misc]


def test_request_construct_with_empty_maps(two_rank_batch):
    req = IndexLookupRequest(
        rank=0, local_to_uuid={}, uuid_to_local={}, global_batch=two_rank_batch
    )
    assert req.rank == 0
    assert IndexManager.local_to_uuid(req, 0) == ""
    assert IndexManager.uuid_to_local(req, "alice") == -1


# ---------------------------------------------------------------------------
# local_to_uuid
# ---------------------------------------------------------------------------

def test_local_to_uuid_hit(rank0_req):
    assert IndexManager.local_to_uuid(rank0_req, 0) == "alice"
    assert IndexManager.local_to_uuid(rank0_req, 1) == "bob"


def test_local_to_uuid_miss_returns_empty_string(rank0_req):
    assert IndexManager.local_to_uuid(rank0_req, 99) == ""
    assert IndexManager.local_to_uuid(rank0_req, -1) == ""


# ---------------------------------------------------------------------------
# uuid_to_local
# ---------------------------------------------------------------------------

def test_uuid_to_local_hit(rank0_req):
    assert IndexManager.uuid_to_local(rank0_req, "alice") == 0
    assert IndexManager.uuid_to_local(rank0_req, "bob") == 1


def test_uuid_to_local_miss_returns_minus_one(rank0_req):
    assert IndexManager.uuid_to_local(rank0_req, "carol") == -1  # owned by rank 1
    assert IndexManager.uuid_to_local(rank0_req, "nonexistent") == -1


# ---------------------------------------------------------------------------
# local_indices_to_global_seq_ids
# ---------------------------------------------------------------------------

def test_local_indices_to_global_seq_ids_happy_path(rank0_req):
    result = IndexManager.local_indices_to_global_seq_ids(rank0_req, [0, 1])
    assert result == [0, 2]  # alice.global_idx=0, bob.global_idx=2


def test_local_indices_to_global_seq_ids_empty_input(rank0_req):
    assert IndexManager.local_indices_to_global_seq_ids(rank0_req, []) == []


def test_local_indices_to_global_seq_ids_logs_on_missing(rank0_req, caplog):
    with caplog.at_level(logging.ERROR, logger="batchgen.worker.indexing"):
        result = IndexManager.local_indices_to_global_seq_ids(rank0_req, [0, 99, 1])
    # Missing index 99 dropped from output; result is the resolvable subset
    assert result == [0, 2]
    assert any(
        "MISSING LOCAL INDICES" in rec.message and "Rank 0" in rec.message
        for rec in caplog.records
    )


def test_local_indices_to_global_seq_ids_preserves_order(rank0_req):
    # Reverse order should produce reverse global IDs
    assert IndexManager.local_indices_to_global_seq_ids(rank0_req, [1, 0]) == [2, 0]


# ---------------------------------------------------------------------------
# get_my_sequences_by_status
# ---------------------------------------------------------------------------

def test_get_my_sequences_by_status_filters_by_rank(rank0_req, two_rank_batch):
    # All 4 sequences start in QUEUEING
    rank0_queueing = IndexManager.get_my_sequences_by_status(
        rank0_req, SequenceStatus.QUEUEING
    )
    assert set(rank0_queueing) == {"alice", "bob"}


def test_get_my_sequences_by_status_empty_when_wrong_status(rank0_req):
    # No sequences are in IN_DECODE yet
    assert IndexManager.get_my_sequences_by_status(rank0_req, SequenceStatus.IN_DECODE) == []


def test_get_my_sequences_by_status_other_rank_view(two_rank_batch):
    req_rank1 = IndexLookupRequest(
        rank=1,
        local_to_uuid={0: "carol", 1: "dave"},
        uuid_to_local={"carol": 0, "dave": 1},
        global_batch=two_rank_batch,
    )
    rank1_queueing = IndexManager.get_my_sequences_by_status(
        req_rank1, SequenceStatus.QUEUEING
    )
    assert set(rank1_queueing) == {"carol", "dave"}


# ---------------------------------------------------------------------------
# get_local_indices_for_uuids
# ---------------------------------------------------------------------------

def test_get_local_indices_for_uuids_happy_path(rank0_req):
    result = IndexManager.get_local_indices_for_uuids(rank0_req, ["alice", "bob"])
    assert result == [0, 1]


def test_get_local_indices_for_uuids_silently_skips_non_owned(rank0_req):
    # carol/dave are owned by rank 1 — rank 0's map doesn't contain them
    result = IndexManager.get_local_indices_for_uuids(
        rank0_req, ["alice", "carol", "bob", "dave"]
    )
    assert result == [0, 1]


def test_get_local_indices_for_uuids_empty_input(rank0_req):
    assert IndexManager.get_local_indices_for_uuids(rank0_req, []) == []


def test_get_local_indices_for_uuids_preserves_input_order(rank0_req):
    result = IndexManager.get_local_indices_for_uuids(rank0_req, ["bob", "alice"])
    assert result == [1, 0]


# ---------------------------------------------------------------------------
# Stateless: shared rank0_req works across many calls without mutation
# ---------------------------------------------------------------------------

def test_handler_is_stateless(rank0_req):
    # Many calls on the same request must not mutate it
    for _ in range(10):
        IndexManager.local_to_uuid(rank0_req, 0)
        IndexManager.uuid_to_local(rank0_req, "alice")
        IndexManager.local_indices_to_global_seq_ids(rank0_req, [0, 1])
        IndexManager.get_local_indices_for_uuids(rank0_req, ["alice"])
    # Snapshot maps remain unchanged
    assert rank0_req.local_to_uuid == {0: "alice", 1: "bob"}
    assert rank0_req.uuid_to_local == {"alice": 0, "bob": 1}
