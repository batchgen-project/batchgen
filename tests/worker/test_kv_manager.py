"""Unit tests for `batchgen.worker.kv_manager`.

Covers Phases 5.1 stats, 5.2 page-table capacity, 5.3 token-budget cache,
and 5.4a GPU-KV-manager allocation planning.
Single-rank tests with a fake ``KVStatsBackend``. Real ``SequenceBatch``
fixtures — no mocks of the underlying batch / status enum per Phase A §G.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, replace

import pytest

from batchgen.migration import MigrationOp
from batchgen.query_book import QueryBookEntry
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from batchgen.worker.kv_manager import (
    HostKVUtilization,
    KVCacheManager,
    KVStats,
    KVStatsBackend,
    KVUtilizationRequest,
    MigrationCandidate,
    MigrationPlanRequest,
    PageTableCapacityRequest,
    TokenBudgetRequest,
    WatermarkGlobalStats,
    WatermarkTriggerPlan,
    WatermarkTriggerRequest,
)

# NOTE: we deliberately do NOT import GPUPagedKVConfig (or anything under
# batchgen.kv_cache) at module top — that triggers a JIT build of the
# core_engine op, which fails on hosts without ninja. The plan tests use a
# lightweight fake config + a fake config module injected into sys.modules.


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


# ===========================================================================
# Phase 5.2 — Page-table capacity helpers
# ===========================================================================


def _make_cap_req(
    *,
    sequence_tokens=(),
    max_input_length=0,
    max_decoding_length=0,
    engine_max_prompt=None,
    engine_max_decode=None,
    engine_module_global_batch_size=None,
    engine_module_attn_decoding_micro_batch_size=None,
    engine_basic_num_queries=None,
    model_max_position_embeddings=None,
    args_cuda_graph_max_bucket_size=None,
    engine_basic_decode_graph_max_bucket=None,
) -> PageTableCapacityRequest:
    return PageTableCapacityRequest(
        sequence_tokens=tuple(sequence_tokens),
        max_input_length=max_input_length,
        max_decoding_length=max_decoding_length,
        engine_max_prompt=engine_max_prompt,
        engine_max_decode=engine_max_decode,
        engine_module_global_batch_size=engine_module_global_batch_size,
        engine_module_attn_decoding_micro_batch_size=engine_module_attn_decoding_micro_batch_size,
        engine_basic_num_queries=engine_basic_num_queries,
        model_max_position_embeddings=model_max_position_embeddings,
        args_cuda_graph_max_bucket_size=args_cuda_graph_max_bucket_size,
        engine_basic_decode_graph_max_bucket=engine_basic_decode_graph_max_bucket,
    )


# ---------------------------------------------------------------------------
# page_table_token_capacity
# ---------------------------------------------------------------------------


def test_token_capacity_floor_is_16384():
    """With no other inputs, the floor of 16384 wins."""
    req = _make_cap_req()
    assert KVCacheManager.page_table_token_capacity(req) == 16384


def test_token_capacity_sequence_tokens_dominate():
    req = _make_cap_req(sequence_tokens=(8000, 32000, 4096))
    assert KVCacheManager.page_table_token_capacity(req) == 32000


def test_token_capacity_skips_nonpositive_seq_tokens():
    req = _make_cap_req(sequence_tokens=(0, -1, 100))
    # Floor 16384 still wins
    assert KVCacheManager.page_table_token_capacity(req) == 16384


def test_token_capacity_includes_max_input_plus_decode():
    req = _make_cap_req(max_input_length=20000, max_decoding_length=4096)
    assert KVCacheManager.page_table_token_capacity(req) == 24096


def test_token_capacity_max_input_zero_uses_floor():
    """max_input_length=0 is treated as "unset"; doesn't contribute."""
    req = _make_cap_req(max_input_length=0, max_decoding_length=99999)
    # max_input=0 path is skipped → only floor 16384 vs sequence/model
    assert KVCacheManager.page_table_token_capacity(req) == 16384


def test_token_capacity_engine_max_prompt_and_decode():
    req = _make_cap_req(engine_max_prompt=5000, engine_max_decode=2000)
    # 5000 + 2000 = 7000, but floor 16384 still wins
    assert KVCacheManager.page_table_token_capacity(req) == 16384
    # Larger engine values
    req = _make_cap_req(engine_max_prompt=20000, engine_max_decode=8000)
    assert KVCacheManager.page_table_token_capacity(req) == 28000


def test_token_capacity_engine_max_prompt_only():
    req = _make_cap_req(engine_max_prompt=20000)
    assert KVCacheManager.page_table_token_capacity(req) == 20000


def test_token_capacity_engine_max_decode_only():
    req = _make_cap_req(engine_max_decode=20000)
    assert KVCacheManager.page_table_token_capacity(req) == 20000


def test_token_capacity_model_max_position():
    req = _make_cap_req(model_max_position_embeddings=131072)
    assert KVCacheManager.page_table_token_capacity(req) == 131072


def test_token_capacity_max_of_all():
    """When multiple sources contribute, return the max."""
    req = _make_cap_req(
        sequence_tokens=(50000,),
        max_input_length=10000,
        max_decoding_length=10000,
        engine_max_prompt=30000,
        engine_max_decode=5000,
        model_max_position_embeddings=40000,
    )
    # candidates = [16384, 50000, 20000, 35000, 40000] → max 50000
    assert KVCacheManager.page_table_token_capacity(req) == 50000


# ---------------------------------------------------------------------------
# page_table_slot_capacity
# ---------------------------------------------------------------------------


def test_slot_capacity_defaults_to_one():
    """No candidates → fallback 1 (legacy semantics)."""
    req = _make_cap_req()
    assert KVCacheManager.page_table_slot_capacity(req) == 1


def test_slot_capacity_args_bucket_size():
    req = _make_cap_req(args_cuda_graph_max_bucket_size=256)
    assert KVCacheManager.page_table_slot_capacity(req) == 256


def test_slot_capacity_max_of_all():
    req = _make_cap_req(
        args_cuda_graph_max_bucket_size=128,
        engine_module_global_batch_size=64,
        engine_module_attn_decoding_micro_batch_size=512,
        engine_basic_num_queries=256,
    )
    assert KVCacheManager.page_table_slot_capacity(req) == 512


def test_slot_capacity_covers_the_largest_decode_graph_bucket():
    """A whole-model decode graph addresses one row per sequence of its top
    bucket (K3 H200 plans 160 KDA slots > the 128 arg default); the graph
    page table must not fall back to eager for a full bucket."""
    req = _make_cap_req(
        args_cuda_graph_max_bucket_size=128,
        engine_module_attn_decoding_micro_batch_size=64,
        engine_basic_decode_graph_max_bucket=160,
    )
    assert KVCacheManager.page_table_slot_capacity(req) == 160


def test_slot_capacity_skips_nonpositive():
    """0 / None values are skipped, not coerced."""
    req = _make_cap_req(
        args_cuda_graph_max_bucket_size=0,
        engine_module_global_batch_size=128,
        engine_module_attn_decoding_micro_batch_size=None,
        engine_basic_num_queries=0,
    )
    assert KVCacheManager.page_table_slot_capacity(req) == 128


# ---------------------------------------------------------------------------
# apply_page_table_capacity (returns updated config dataclass)
# ---------------------------------------------------------------------------


@dataclass
class _FakeCudaGraphConfig:
    """Stand-in for whatever real config dataclass the worker uses.

    Production callers pass a richer config; we only depend on these 4
    fields, so a minimal local dataclass is enough.
    """
    num_pages: int
    page_size_tokens: int
    cuda_graph_max_pages_per_sequence: int = 0
    cuda_graph_max_slots: int = 0


def test_apply_capacity_normal_case():
    req = _make_cap_req(
        max_input_length=16000,
        max_decoding_length=8000,  # → token_capacity = max(16384, 24000) = 24000
        args_cuda_graph_max_bucket_size=64,
    )
    config = _FakeCudaGraphConfig(num_pages=1000, page_size_tokens=64)
    out = KVCacheManager.apply_page_table_capacity(req, config)
    # ceil(24000 / 64) = 375 — from token capacity only.
    assert out.cuda_graph_max_pages_per_sequence == 375
    assert out.cuda_graph_max_slots == 64


def test_apply_capacity_decoupled_from_num_pages():
    """Table SHAPE comes from token capacity only (measure-once-cache
    contract): a pool smaller than the token capacity no longer clamps the
    page-table width, so captured graphs never depend on the pool's page
    count and a pool resize cannot invalidate the baked table shape."""
    req = _make_cap_req(model_max_position_embeddings=1_000_000)
    config = _FakeCudaGraphConfig(num_pages=100, page_size_tokens=64)
    out = KVCacheManager.apply_page_table_capacity(req, config)
    assert out.cuda_graph_max_pages_per_sequence == 15625  # ceil(1e6 / 64)
    assert out.cuda_graph_max_slots == 1  # no slot inputs → fallback 1


def test_apply_capacity_zero_floor():
    """Page-capacity and slot-capacity always clamp to at least 1."""
    req = _make_cap_req()
    # With huge page_size_tokens, ceil(16384 / 99999999) = 1
    config = _FakeCudaGraphConfig(num_pages=10, page_size_tokens=99_999_999)
    out = KVCacheManager.apply_page_table_capacity(req, config)
    assert out.cuda_graph_max_pages_per_sequence == 1
    assert out.cuda_graph_max_slots == 1


def test_apply_capacity_does_not_mutate_input():
    req = _make_cap_req(args_cuda_graph_max_bucket_size=32)
    config = _FakeCudaGraphConfig(num_pages=100, page_size_tokens=64)
    out = KVCacheManager.apply_page_table_capacity(req, config)
    # Original input unchanged
    assert config.cuda_graph_max_pages_per_sequence == 0
    assert config.cuda_graph_max_slots == 0
    # Output has new values
    assert out is not config
    assert out.cuda_graph_max_slots == 32


# ---------------------------------------------------------------------------
# Frozen dataclass semantics
# ---------------------------------------------------------------------------


def test_capacity_request_is_frozen():
    req = _make_cap_req()
    with pytest.raises((AttributeError, Exception)):
        req.max_input_length = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Token-budget cache (Phase 5.3)
# ---------------------------------------------------------------------------


def _make_token_budget_req(*, query_book=None) -> TokenBudgetRequest:
    return TokenBudgetRequest(query_book=query_book if query_book is not None else {})


def _make_entry(*, kv_token_budget=None) -> QueryBookEntry:
    return QueryBookEntry(
        text="dummy",
        encoded={"input_ids": [1, 2, 3]},
        decoded_tokens=None,
        kv_token_budget=kv_token_budget,
    )


def test_token_budget_returns_admission_value():
    entry = _make_entry(kv_token_budget=999)
    req = _make_token_budget_req(query_book={42: entry})
    assert KVCacheManager.get_sequence_token_budget(req, 42) == 999


def test_token_budget_missing_value_raises():
    """Admission always sets the budget; there is no recompute fallback."""
    entry = _make_entry(kv_token_budget=None)
    req = _make_token_budget_req(query_book={7: entry})
    with pytest.raises(RuntimeError, match="sequence 7 has no kv_token_budget"):
        KVCacheManager.get_sequence_token_budget(req, 7)
    assert entry.kv_token_budget is None


def test_token_budget_raises_runtime_error_when_query_book_empty():
    """Empty query_book ⇒ worker has not initialized it. Legacy raised RuntimeError."""
    req = _make_token_budget_req(query_book={})
    with pytest.raises(RuntimeError, match="query_book is not initialized"):
        KVCacheManager.get_sequence_token_budget(req, 0)


def test_token_budget_raises_keyerror_for_missing_entry():
    """Sequence id not in query_book ⇒ KeyError (legacy parity)."""
    entry = _make_entry(kv_token_budget=42)
    req = _make_token_budget_req(query_book={0: entry})
    with pytest.raises(KeyError, match="Missing query entry for sequence 7"):
        KVCacheManager.get_sequence_token_budget(req, 7)


def test_token_budget_raises_keyerror_for_entry_without_encoded():
    """encoded=None ⇒ entry not ready ⇒ KeyError (legacy parity)."""
    entry = QueryBookEntry(text="x", encoded=None, kv_token_budget=None)
    req = _make_token_budget_req(query_book={5: entry})
    with pytest.raises(KeyError, match="Missing query entry for sequence 5"):
        KVCacheManager.get_sequence_token_budget(req, 5)


def test_compute_host_kv_sequence_tokens_returns_list_in_order():
    req = _make_token_budget_req(
        query_book={
            1: _make_entry(kv_token_budget=111),
            2: _make_entry(kv_token_budget=222),
            3: _make_entry(kv_token_budget=100),
        },
    )
    assert KVCacheManager.compute_host_kv_sequence_tokens(req, [2, 1, 3]) == [222, 111, 100]


def test_compute_host_kv_sequence_tokens_empty_input():
    req = _make_token_budget_req(query_book={})
    assert KVCacheManager.compute_host_kv_sequence_tokens(req, []) == []


def test_token_budget_request_is_frozen():
    req = _make_token_budget_req(query_book={})
    with pytest.raises((AttributeError, Exception)):
        req.query_book = {}  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Host-KV watermark trigger (Phase 5.5)
# ---------------------------------------------------------------------------


def _node_stat(*, free_percent, num_used_pages=0, num_total_pages=100, node_id=0):
    """A per-node host-KV stat dict (shape of get_host_utilization output)."""
    return {
        "free_percent": free_percent,
        "num_used_pages": num_used_pages,
        "num_total_pages": num_total_pages,
        "num_free_pages": num_total_pages - num_used_pages,
        "node_id": node_id,
    }


def _wm_req(node_stats, *, watermark=70, has_queued=False, has_evicted=False):
    return WatermarkTriggerRequest(
        node_stats=tuple(node_stats),
        host_kv_watermark=watermark,
        has_queued=has_queued,
        has_evicted=has_evicted,
    )


def test_watermark_empty_node_stats_no_trigger():
    plan = KVCacheManager.plan_watermark_trigger(_wm_req([]))
    assert plan.should_trigger is False
    assert plan.max_free_percent is None
    assert plan.global_stats is None


def test_watermark_above_threshold_with_queued_triggers():
    plan = KVCacheManager.plan_watermark_trigger(
        _wm_req([_node_stat(free_percent=85)], watermark=70, has_queued=True)
    )
    assert plan.should_trigger is True
    assert plan.max_free_percent == 85


def test_watermark_above_threshold_with_evicted_triggers():
    plan = KVCacheManager.plan_watermark_trigger(
        _wm_req([_node_stat(free_percent=85)], watermark=70, has_evicted=True)
    )
    assert plan.should_trigger is True


def test_watermark_above_threshold_no_work_no_trigger():
    """Free space high but nothing waiting → no preemption."""
    plan = KVCacheManager.plan_watermark_trigger(
        _wm_req([_node_stat(free_percent=85)], watermark=70,
                has_queued=False, has_evicted=False)
    )
    assert plan.should_trigger is False
    assert plan.max_free_percent == 85


def test_watermark_below_threshold_no_trigger():
    """Free space below watermark → busy, keep decoding even with queued work."""
    plan = KVCacheManager.plan_watermark_trigger(
        _wm_req([_node_stat(free_percent=50)], watermark=70, has_queued=True)
    )
    assert plan.should_trigger is False
    assert plan.max_free_percent == 50


def test_watermark_boundary_strictly_greater():
    """`free > watermark` is strict — equal does NOT trigger (legacy parity)."""
    plan = KVCacheManager.plan_watermark_trigger(
        _wm_req([_node_stat(free_percent=70)], watermark=70, has_queued=True)
    )
    assert plan.should_trigger is False


def test_watermark_uses_max_across_nodes():
    """ANY node above watermark triggers; max_free_percent is the highest."""
    plan = KVCacheManager.plan_watermark_trigger(
        _wm_req(
            [_node_stat(free_percent=40, node_id=0),
             _node_stat(free_percent=88, node_id=1)],
            watermark=70, has_queued=True,
        )
    )
    assert plan.should_trigger is True
    assert plan.max_free_percent == 88


def test_watermark_global_stats_aggregation():
    plan = KVCacheManager.plan_watermark_trigger(
        _wm_req(
            [_node_stat(free_percent=60, num_used_pages=40, num_total_pages=100, node_id=0),
             _node_stat(free_percent=80, num_used_pages=20, num_total_pages=100, node_id=1)],
            watermark=70, has_queued=True,
        )
    )
    gs = plan.global_stats
    assert gs == WatermarkGlobalStats(used=60, total=200, free_percent=70, num_nodes=2)


def test_watermark_global_stats_total_zero_safe():
    """num_total_pages == 0 → no ZeroDivision; used%=0 → free%=100."""
    plan = KVCacheManager.plan_watermark_trigger(
        _wm_req([_node_stat(free_percent=0, num_used_pages=0, num_total_pages=0)],
                watermark=70, has_queued=True)
    )
    assert plan.global_stats.free_percent == 100
    assert plan.global_stats.total == 0


def test_watermark_request_and_plan_are_frozen():
    req = _wm_req([_node_stat(free_percent=80)])
    with pytest.raises((AttributeError, Exception)):
        req.host_kv_watermark = 50  # type: ignore[misc]
    plan = WatermarkTriggerPlan(should_trigger=True, max_free_percent=80, global_stats=None)
    with pytest.raises((AttributeError, Exception)):
        plan.should_trigger = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Host-KV migration planning (Phase 5.6)
# ---------------------------------------------------------------------------

_GPUS_PER_NODE = 8
_WORLD = 16  # 2 nodes


def _node(used, total):
    return {"num_used_pages": used, "num_total_pages": total}


def _cand(uuid, rank, gidx, budget, host_pages):
    return MigrationCandidate(
        uuid=uuid,
        assigned_rank=rank,
        global_idx=gidx,
        kv_token_budget=budget,
        host_pages_allocated=host_pages,
    )


def _mig_req(node_stats, candidates, *, gpus_per_node=_GPUS_PER_NODE, world=_WORLD):
    return MigrationPlanRequest(
        node_stats=node_stats,
        candidates=tuple(candidates),
        num_gpus_per_node=gpus_per_node,
        world_size=world,
    )


def test_migration_single_node_no_migration():
    plan = KVCacheManager.plan_kv_migration(
        _mig_req({0: _node(100, 200)}, [_cand("a", 0, 0, 128, 60)])
    )
    assert plan == []


def test_migration_already_balanced():
    # both nodes at 50 == target → no overloaded/underutilized
    plan = KVCacheManager.plan_kv_migration(
        _mig_req({0: _node(50, 200), 1: _node(50, 200)}, [])
    )
    assert plan == []


def test_migration_basic_single_move():
    plan = KVCacheManager.plan_kv_migration(
        _mig_req(
            {0: _node(100, 200), 1: _node(0, 200)},
            [_cand("a", 0, 0, 128, 60)],
        )
    )
    assert plan == [
        MigrationOp(uuid="a", from_rank=0, to_rank=8, pages=60, host_pages=60)
    ]


def test_migration_selects_smallest_budget():
    """Among candidates, the smallest kv_token_budget is migrated first."""
    plan = KVCacheManager.plan_kv_migration(
        _mig_req(
            {0: _node(100, 400), 1: _node(0, 400)},
            [
                _cand("big", 0, 0, 256, 30),
                _cand("small", 1, 1, 64, 30),
            ],
        )
    )
    # target = 50; node0 used 100. First move picks "small" (budget 64).
    assert plan[0].uuid == "small"


def test_migration_global_idx_tiebreak():
    """Equal budget → lower global_idx wins."""
    plan = KVCacheManager.plan_kv_migration(
        _mig_req(
            {0: _node(100, 400), 1: _node(0, 400)},
            [
                _cand("later", 1, 5, 64, 30),
                _cand("earlier", 0, 2, 64, 30),
            ],
        )
    )
    assert plan[0].uuid == "earlier"


def test_migration_round_robin_dest_ranks():
    """Two moves to the same dest node distribute across ranks 8, 9."""
    plan = KVCacheManager.plan_kv_migration(
        _mig_req(
            {0: _node(120, 400), 1: _node(0, 400)},
            [
                _cand("a", 0, 0, 64, 30),
                _cand("b", 1, 1, 64, 30),
            ],
        )
    )
    # target = 60; node0 120 → migrate a (rank8) then b (rank9) until node0=60.
    assert [m.uuid for m in plan] == ["a", "b"]
    assert [m.to_rank for m in plan] == [8, 9]


def test_migration_skips_dest_with_insufficient_pages():
    """A dest node without room is dropped; migration goes to the next node."""
    plan = KVCacheManager.plan_kv_migration(
        _mig_req(
            {0: _node(90, 300), 1: _node(10, 20), 2: _node(10, 300)},
            [_cand("a", 0, 0, 64, 50)],
            world=24,  # 3 nodes
        )
    )
    # target = 36; node1 has only 10 free (< 50 needed) → skip → dest node2 rank16.
    assert plan == [
        MigrationOp(uuid="a", from_rank=0, to_rank=16, pages=50, host_pages=50)
    ]


def test_migration_skips_zero_host_pages():
    """A candidate with no host pages allocated is skipped, not migrated."""
    plan = KVCacheManager.plan_kv_migration(
        _mig_req(
            {0: _node(100, 200), 1: _node(0, 200)},
            [_cand("empty", 0, 0, 128, 0)],
        )
    )
    assert plan == []


def test_migration_multi_move_until_balanced():
    """Keeps migrating until the source node reaches target."""
    plan = KVCacheManager.plan_kv_migration(
        _mig_req(
            {0: _node(120, 400), 1: _node(0, 400)},
            [
                _cand("a", 0, 0, 64, 20),
                _cand("b", 1, 1, 64, 20),
                _cand("c", 2, 2, 64, 20),
                _cand("d", 3, 3, 64, 20),
            ],
        )
    )
    # target = 60; node0 120 → migrate 3×20 = 60 to reach 60.
    assert len(plan) == 3
    assert {m.uuid for m in plan} == {"a", "b", "c"}


def test_migration_no_candidates_returns_empty():
    """Overloaded node but no movable sequences → no migrations."""
    plan = KVCacheManager.plan_kv_migration(
        _mig_req({0: _node(100, 200), 1: _node(0, 200)}, [])
    )
    assert plan == []


def test_migration_request_and_candidate_are_frozen():
    req = _mig_req({0: _node(1, 2)}, [_cand("a", 0, 0, 1, 1)])
    with pytest.raises((AttributeError, Exception)):
        req.world_size = 8  # type: ignore[misc]
    c = _cand("a", 0, 0, 1, 1)
    with pytest.raises((AttributeError, Exception)):
        c.uuid = "b"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# measure-once-cache pure helpers
# ---------------------------------------------------------------------------


def test_compute_final_pool_gb_subtracts_provisional_pool():
    final = KVCacheManager.compute_final_pool_gb(140.0, 0.93, 132.4, 19.1)
    assert abs(final - (140.0 * 0.93 - (132.4 - 19.1))) < 1e-9


def test_compute_final_pool_gb_is_deterministic_identity():
    # Measuring again with the FINAL pool allocated returns the same value:
    # used2 = steady + final, so total*frac - (used2 - final) = final.
    total, frac, steady = 140.0, 0.93, 113.3
    final = KVCacheManager.compute_final_pool_gb(total, frac, steady + 19.1, 19.1)
    again = KVCacheManager.compute_final_pool_gb(total, frac, steady + final, final)
    assert abs(final - again) < 1e-9


def test_pool_floor_pages():
    assert KVCacheManager.pool_floor_pages(128) == 512
    assert KVCacheManager.pool_floor_pages(1) == 4
    assert KVCacheManager.pool_floor_pages(0) == 4  # clamped bucket
