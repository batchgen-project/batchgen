"""Unit tests for `batchgen.worker.kv_manager` (Phase 5.1 stats tier).

Single-rank tests with a fake ``KVStatsBackend``. Real ``SequenceBatch``
fixtures — no mocks of the underlying batch / status enum per Phase A §G.
"""

from __future__ import annotations

import pytest

from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from batchgen.worker.kv_manager import (
    HostKVUtilization,
    KVCacheManager,
    KVStats,
    KVStatsBackend,
    KVUtilizationRequest,
)


# ---------------------------------------------------------------------------
# Fakes / fixtures
# ---------------------------------------------------------------------------


class FakeKVBackend:
    """Test backend with explicit pre-set stats. No NCCL, no GPU."""

    def __init__(
        self,
        *,
        host: KVStats = KVStats(num_free_pages=80, num_used_pages=20, num_total_pages=100),
        gpu: "KVStats | None" = KVStats(num_free_pages=40, num_used_pages=10, num_total_pages=50),
    ) -> None:
        self._host = host
        self._gpu = gpu
        self.host_calls = 0
        self.gpu_calls = 0

    def get_host_stats(self) -> KVStats:
        self.host_calls += 1
        return self._host

    def get_gpu_stats(self):
        self.gpu_calls += 1
        return self._gpu


def _make_seq(uuid: str, global_idx: int, rank: int, status: SequenceStatus) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=8,
        max_decode_length=16,
    )
    seq.assigned_rank = rank
    seq.status = status
    return seq


@pytest.fixture
def batch_node0() -> SequenceBatch:
    """Sequences distributed across an 8-GPU node-0 (ranks 0-7).

    rank 0 → 1× IN_DECODE
    rank 1 → 1× PREFILLED, 1× ON_HOLD
    rank 4 → 1× QUEUEING (not "valid" for host KV)
    rank 7 → 1× COMPLETED (not "valid")
    Total valid on node 0: 3 (in_decode=1, prefilled=1, onhold=1)
    """
    batch = SequenceBatch()
    seqs = [
        _make_seq("a", 0, 0, SequenceStatus.IN_DECODE),
        _make_seq("b", 1, 1, SequenceStatus.PREFILLED),
        _make_seq("c", 2, 1, SequenceStatus.ON_HOLD),
        _make_seq("d", 3, 4, SequenceStatus.QUEUEING),
        _make_seq("e", 4, 7, SequenceStatus.COMPLETED),
    ]
    for s in seqs:
        batch.add_sequence(s)
        batch.assign_rank(s.uuid, s.assigned_rank)
    return batch


# ---------------------------------------------------------------------------
# get_host_free_pages / get_gpu_free_pages
# ---------------------------------------------------------------------------


def test_get_host_free_pages_returns_backend_value():
    backend = FakeKVBackend(
        host=KVStats(num_free_pages=42, num_used_pages=58, num_total_pages=100)
    )
    mgr = KVCacheManager(backend=backend)
    assert mgr.get_host_free_pages() == 42
    assert backend.host_calls == 1


def test_get_gpu_free_pages_returns_backend_value():
    backend = FakeKVBackend(
        gpu=KVStats(num_free_pages=7, num_used_pages=3, num_total_pages=10)
    )
    mgr = KVCacheManager(backend=backend)
    assert mgr.get_gpu_free_pages() == 7
    assert backend.gpu_calls == 1


def test_get_gpu_free_pages_returns_zero_when_unbound():
    """Production legacy returned 0 when gpu_paged_kv_cache_manager is None."""
    backend = FakeKVBackend(gpu=None)
    mgr = KVCacheManager(backend=backend)
    assert mgr.get_gpu_free_pages() == 0


# ---------------------------------------------------------------------------
# KVStats dataclass behavior
# ---------------------------------------------------------------------------


def test_kvstats_is_frozen():
    s = KVStats(num_free_pages=1, num_used_pages=2, num_total_pages=3)
    with pytest.raises((AttributeError, Exception)):
        s.num_free_pages = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_host_utilization
# ---------------------------------------------------------------------------


def test_get_host_utilization_node_aggregation(batch_node0):
    backend = FakeKVBackend(
        host=KVStats(num_free_pages=200, num_used_pages=800, num_total_pages=1000)
    )
    mgr = KVCacheManager(backend=backend)
    req = KVUtilizationRequest(
        rank=0,
        world_size=16,
        local_rank=0,
        num_gpus_per_node=8,
        global_batch=batch_node0,
    )
    util = mgr.get_host_utilization(req)

    assert util.rank == 0
    assert util.node_id == 0
    assert util.num_free_pages == 200
    assert util.num_used_pages == 800
    assert util.num_total_pages == 1000
    assert util.free_percent == 20  # 200/1000
    # Node 0 spans ranks 0-7. Valid-status sequences on those ranks:
    # IN_DECODE: a (rank 0)
    # PREFILLED: b (rank 1)
    # ON_HOLD: c (rank 1)
    # → 3 valid; queueing (d, rank 4) and completed (e, rank 7) are not.
    assert util.num_in_decode == 1
    assert util.num_prefilled == 1
    assert util.num_onhold == 1
    assert util.num_valid_sequences == 3


def test_get_host_utilization_node_id_from_rank():
    backend = FakeKVBackend()
    mgr = KVCacheManager(backend=backend)
    batch = SequenceBatch()
    req_node1 = KVUtilizationRequest(
        rank=12,  # node 1 (12 // 8)
        world_size=16,
        local_rank=4,
        num_gpus_per_node=8,
        global_batch=batch,
    )
    util = mgr.get_host_utilization(req_node1)
    assert util.node_id == 1
    assert util.num_valid_sequences == 0  # empty batch


def test_get_host_utilization_free_percent_total_zero():
    backend = FakeKVBackend(
        host=KVStats(num_free_pages=0, num_used_pages=0, num_total_pages=0)
    )
    mgr = KVCacheManager(backend=backend)
    batch = SequenceBatch()
    req = KVUtilizationRequest(
        rank=0, world_size=1, local_rank=0, num_gpus_per_node=8, global_batch=batch,
    )
    util = mgr.get_host_utilization(req)
    # Legacy returned 100 when num_total_pages == 0
    assert util.free_percent == 100


def test_get_host_utilization_node_rank_bound_clamp():
    """world_size=10, num_gpus_per_node=8 → node 1 spans ranks 8-9, not 8-15."""
    backend = FakeKVBackend()
    mgr = KVCacheManager(backend=backend)
    batch = SequenceBatch()
    # rank 9 sequence on node 1
    seq = _make_seq("only", 0, 9, SequenceStatus.PREFILLED)
    batch.add_sequence(seq)
    batch.assign_rank(seq.uuid, 9)

    req = KVUtilizationRequest(
        rank=8, world_size=10, local_rank=0, num_gpus_per_node=8, global_batch=batch,
    )
    util = mgr.get_host_utilization(req)
    # Should find the rank-9 prefilled sequence on node 1
    assert util.node_id == 1
    assert util.num_prefilled == 1


# ---------------------------------------------------------------------------
# Backend injection
# ---------------------------------------------------------------------------


def test_can_swap_backend():
    class CountingBackend:
        def __init__(self):
            self.events = []

        def get_host_stats(self):
            self.events.append("host")
            return KVStats(num_free_pages=1, num_used_pages=2, num_total_pages=3)

        def get_gpu_stats(self):
            self.events.append("gpu")
            return KVStats(num_free_pages=4, num_used_pages=5, num_total_pages=9)

    backend = CountingBackend()
    mgr = KVCacheManager(backend=backend)
    mgr.get_host_free_pages()
    mgr.get_gpu_free_pages()
    assert backend.events == ["host", "gpu"]
