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
    GpuKvManagerPlan,
    GpuKvManagerRequest,
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
    # ceil(24000 / 64) = 375. min(1000, 375) = 375.
    assert out.cuda_graph_max_pages_per_sequence == 375
    # min(1000, 64) = 64.
    assert out.cuda_graph_max_slots == 64


def test_apply_capacity_token_capacity_exceeds_num_pages():
    """When ceil(tokens/page_size) > num_pages, clamped to num_pages."""
    req = _make_cap_req(model_max_position_embeddings=1_000_000)
    config = _FakeCudaGraphConfig(num_pages=100, page_size_tokens=64)
    out = KVCacheManager.apply_page_table_capacity(req, config)
    assert out.cuda_graph_max_pages_per_sequence == 100
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


def _make_token_budget_req(
    *,
    query_book=None,
    local_to_uuid=None,
    sequences=(),
    max_decoding_length: int = 16,
) -> TokenBudgetRequest:
    """Build a TokenBudgetRequest from explicit pieces.

    ``sequences`` is an iterable of ``(uuid, prompt_length)`` pairs which is
    materialized into a real ``SequenceBatch`` so the handler exercises the
    same ``get_sequence`` path the worker uses.
    """
    batch = SequenceBatch()
    for uuid, prompt_length in sequences:
        seq = SequenceEntry(
            uuid=uuid,
            global_idx=0,
            prompt_length=prompt_length,
            max_decode_length=max_decoding_length,
        )
        batch.add_sequence(seq)
    return TokenBudgetRequest(
        query_book=query_book if query_book is not None else {},
        local_to_uuid=local_to_uuid if local_to_uuid is not None else {},
        global_batch=batch,
        max_decoding_length=max_decoding_length,
    )


def _make_entry(*, kv_token_budget=None) -> QueryBookEntry:
    return QueryBookEntry(
        text="dummy",
        encoded={"input_ids": [1, 2, 3]},
        decoded_tokens=None,
        kv_token_budget=kv_token_budget,
    )


def test_token_budget_uses_cached_value_when_set():
    """If the entry already has kv_token_budget, return it without recomputing."""
    entry = _make_entry(kv_token_budget=999)
    req = _make_token_budget_req(query_book={42: entry})
    assert KVCacheManager.get_sequence_token_budget(req, 42) == 999


def test_token_budget_computes_and_memoizes_when_missing():
    """When the entry has no budget, compute from sequence metadata and cache."""
    entry = _make_entry(kv_token_budget=None)
    req = _make_token_budget_req(
        query_book={7: entry},
        local_to_uuid={7: "uuid-7"},
        sequences=[("uuid-7", 100)],
        max_decoding_length=512,
    )
    budget = KVCacheManager.get_sequence_token_budget(req, 7)
    assert budget == 100 + 512  # prompt_length + max_decoding_length
    # Memoized on the entry the worker passed in
    assert entry.kv_token_budget == 612


def test_token_budget_second_call_returns_cache_hit():
    """Second call must hit the just-memoized value (no re-read of metadata)."""
    entry = _make_entry(kv_token_budget=None)
    req = _make_token_budget_req(
        query_book={3: entry},
        local_to_uuid={3: "uuid-3"},
        sequences=[("uuid-3", 50)],
        max_decoding_length=16,
    )
    first = KVCacheManager.get_sequence_token_budget(req, 3)
    # Drop the sequence from the batch — cache hit must not need it
    req.global_batch.sequences.clear()
    second = KVCacheManager.get_sequence_token_budget(req, 3)
    assert first == second == 66


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


def test_token_budget_raises_keyerror_when_sequence_metadata_missing():
    """Compute path needs sequence metadata; missing ⇒ KeyError (legacy parity)."""
    entry = _make_entry(kv_token_budget=None)
    req = _make_token_budget_req(
        query_book={9: entry},
        local_to_uuid={9: "uuid-9"},
        sequences=[],  # No sequence with uuid-9
    )
    with pytest.raises(KeyError, match="No sequence metadata available for sequence 9"):
        KVCacheManager.get_sequence_token_budget(req, 9)


def test_token_budget_raises_keyerror_when_uuid_unknown():
    """If local_to_uuid has no entry for the sequence_id, uuid="" → no metadata."""
    entry = _make_entry(kv_token_budget=None)
    req = _make_token_budget_req(
        query_book={11: entry},
        local_to_uuid={},  # 11 not present
        sequences=[("anything", 100)],
    )
    with pytest.raises(KeyError, match="No sequence metadata available for sequence 11"):
        KVCacheManager.get_sequence_token_budget(req, 11)


def test_token_budget_no_truncation_against_max_input_length():
    """Budget is prompt + max_decode regardless of any worker max_input_length.

    Legacy had an earlier min(...) here which silently undersized KV when
    max_input_length lagged behind the actual prompt length on multi-batch
    admits. The handler must NOT re-introduce that.
    """
    entry = _make_entry(kv_token_budget=None)
    # Prompt is 4096 but no max_input_length in the request — handler must
    # not clamp against any external limit.
    req = _make_token_budget_req(
        query_book={1: entry},
        local_to_uuid={1: "uuid-1"},
        sequences=[("uuid-1", 4096)],
        max_decoding_length=512,
    )
    assert KVCacheManager.get_sequence_token_budget(req, 1) == 4096 + 512


def test_compute_host_kv_sequence_tokens_returns_list_in_order():
    """Bulk variant preserves input ordering and reuses the cache."""
    e1 = _make_entry(kv_token_budget=111)
    e2 = _make_entry(kv_token_budget=222)
    e3 = _make_entry(kv_token_budget=None)
    req = _make_token_budget_req(
        query_book={1: e1, 2: e2, 3: e3},
        local_to_uuid={3: "uuid-3"},
        sequences=[("uuid-3", 60)],
        max_decoding_length=40,
    )
    out = KVCacheManager.compute_host_kv_sequence_tokens(req, [2, 1, 3])
    assert out == [222, 111, 100]
    # The miss path also memoized:
    assert e3.kv_token_budget == 100


def test_compute_host_kv_sequence_tokens_empty_input():
    req = _make_token_budget_req(query_book={})
    assert KVCacheManager.compute_host_kv_sequence_tokens(req, []) == []


def test_token_budget_request_is_frozen():
    req = _make_token_budget_req(query_book={})
    with pytest.raises((AttributeError, Exception)):
        req.max_decoding_length = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# GPU-KV-manager allocation planning (Phase 5.4a)
# ---------------------------------------------------------------------------

_CFG_MODULE = "batchgen.kv_cache.host_kv_mananger_config"


@dataclass(frozen=True)
class _FakeGpuConfig:
    """Stand-in for GPUPagedKVConfig — only the fields the planner touches.

    ``apply_page_table_capacity`` reads ``num_pages`` / ``page_size_tokens``
    and ``dataclasses.replace``s the two cuda_graph fields, so this is
    sufficient to exercise the plan without importing the real (JIT-backed)
    config class.
    """

    num_pages: int
    page_size_tokens: int = 64
    cuda_graph_max_pages_per_sequence: "int | None" = None
    cuda_graph_max_slots: "int | None" = None


def _gpu_config(num_pages: int) -> _FakeGpuConfig:
    return _FakeGpuConfig(num_pages=num_pages)


def _cap_req_for_plan() -> PageTableCapacityRequest:
    """Capacity request with everything off → apply_page_table_capacity
    only sets cuda_graph fields, never touches num_pages."""
    return PageTableCapacityRequest(
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
    )


def _patch_builders(monkeypatch, *, primary_pages: int, aux_pages=None):
    """Inject a fake config module so the handler's local
    ``from batchgen.kv_cache.host_kv_mananger_config import ...`` resolves to
    controlled builders — without importing the real module (which JIT-builds
    the core_engine op). Python's import machinery checks ``sys.modules``
    first, so the fake wins.
    """
    fake_mod = types.ModuleType(_CFG_MODULE)
    fake_mod.build_gpu_kv_config = (
        lambda model_name, sequence_tokens: _gpu_config(primary_pages)
    )
    fake_mod.build_gpu_kv_config_aux = lambda model_name, sequence_tokens: (
        None if aux_pages is None else _gpu_config(aux_pages)
    )
    monkeypatch.setitem(sys.modules, _CFG_MODULE, fake_mod)


def _plan_req(*, has_manager: bool, current_num_pages: int) -> GpuKvManagerRequest:
    return GpuKvManagerRequest(
        model_name="fake-model",
        sequence_tokens=(2048,),
        has_manager=has_manager,
        current_num_pages=current_num_pages,
        capacity=_cap_req_for_plan(),
    )


def test_plan_gpu_kv_no_existing_manager_creates_new(monkeypatch):
    _patch_builders(monkeypatch, primary_pages=100)
    plan = KVCacheManager.plan_gpu_kv_manager(
        _plan_req(has_manager=False, current_num_pages=0)
    )
    assert plan.reuse is False
    assert plan.destroy_existing is False  # nothing to destroy
    assert plan.primary_config is not None and plan.primary_config.num_pages == 100
    assert plan.aux_config is None  # non-DSA model


def test_plan_gpu_kv_reuse_when_enough_pages(monkeypatch):
    _patch_builders(monkeypatch, primary_pages=100)
    plan = KVCacheManager.plan_gpu_kv_manager(
        _plan_req(has_manager=True, current_num_pages=128)
    )
    assert plan.reuse is True
    assert plan.destroy_existing is False
    assert plan.primary_config is None  # reuse → no configs built
    assert plan.aux_config is None


def test_plan_gpu_kv_reuse_boundary_exact_pages(monkeypatch):
    """current == required ⇒ reuse (the `>=` boundary)."""
    _patch_builders(monkeypatch, primary_pages=100)
    plan = KVCacheManager.plan_gpu_kv_manager(
        _plan_req(has_manager=True, current_num_pages=100)
    )
    assert plan.reuse is True


def test_plan_gpu_kv_recreate_when_too_few_pages(monkeypatch):
    """Existing manager too small ⇒ recreate + destroy old."""
    _patch_builders(monkeypatch, primary_pages=200)
    plan = KVCacheManager.plan_gpu_kv_manager(
        _plan_req(has_manager=True, current_num_pages=100)
    )
    assert plan.reuse is False
    assert plan.destroy_existing is True
    assert plan.primary_config.num_pages == 200


def test_plan_gpu_kv_dsa_model_includes_aux(monkeypatch):
    """DSA model (aux builder returns a config) ⇒ aux_config populated."""
    _patch_builders(monkeypatch, primary_pages=200, aux_pages=50)
    plan = KVCacheManager.plan_gpu_kv_manager(
        _plan_req(has_manager=False, current_num_pages=0)
    )
    assert plan.reuse is False
    assert plan.primary_config.num_pages == 200
    assert plan.aux_config is not None and plan.aux_config.num_pages == 50


def test_plan_gpu_kv_reuse_skips_aux_build(monkeypatch):
    """On reuse, neither config is built (aux stays None even for DSA)."""
    _patch_builders(monkeypatch, primary_pages=100, aux_pages=50)
    plan = KVCacheManager.plan_gpu_kv_manager(
        _plan_req(has_manager=True, current_num_pages=128)
    )
    assert plan.reuse is True
    assert plan.aux_config is None


def test_plan_gpu_kv_applies_page_table_capacity(monkeypatch):
    """The returned config carries CUDA-graph fields from the capacity req."""
    _patch_builders(monkeypatch, primary_pages=100)
    req = GpuKvManagerRequest(
        model_name="fake-model",
        sequence_tokens=(2048,),
        has_manager=False,
        current_num_pages=0,
        capacity=PageTableCapacityRequest(
            sequence_tokens=(),
            max_input_length=0,
            max_decoding_length=0,
            engine_max_prompt=None,
            engine_max_decode=None,
            engine_module_global_batch_size=32,  # → slot capacity 32
            engine_module_attn_decoding_micro_batch_size=None,
            engine_basic_num_queries=None,
            model_max_position_embeddings=None,
            args_cuda_graph_max_bucket_size=None,
        ),
    )
    plan = KVCacheManager.plan_gpu_kv_manager(req)
    # num_pages is untouched by capacity application; cuda_graph fields set.
    assert plan.primary_config.num_pages == 100
    assert plan.primary_config.cuda_graph_max_slots == 32
    assert plan.primary_config.cuda_graph_max_pages_per_sequence is not None


def test_plan_gpu_kv_request_and_plan_are_frozen():
    req = _plan_req(has_manager=False, current_num_pages=0)
    with pytest.raises((AttributeError, Exception)):
        req.has_manager = True  # type: ignore[misc]
    plan = GpuKvManagerPlan(
        reuse=True, destroy_existing=False, primary_config=None, aux_config=None
    )
    with pytest.raises((AttributeError, Exception)):
        plan.reuse = False  # type: ignore[misc]


# --- Integration: real config builders for known models -------------------
#
# These exercise the REAL build_gpu_kv_config / _aux through the handler's
# local import. That pulls batchgen.kv_cache, which JIT-builds the core_engine
# op — unavailable on hosts without ninja (e.g. the dev laptop). Skip cleanly
# there; they run where the op is built (remote / CI-with-GPU).


def _require_real_config_module():
    try:
        import batchgen.kv_cache.host_kv_mananger_config  # noqa: F401
    except Exception as exc:  # RuntimeError (ninja/JIT) or ImportError
        pytest.skip(f"core_engine op unavailable on this host: {exc}")


def test_plan_gpu_kv_real_glm5_is_dsa_with_aux():
    """GLM-5-FP8 is a DSA model → real builder returns a non-None aux config."""
    _require_real_config_module()
    req = GpuKvManagerRequest(
        model_name="glm-5-fp8",
        sequence_tokens=(2048,),
        has_manager=False,
        current_num_pages=0,
        capacity=_cap_req_for_plan(),
    )
    plan = KVCacheManager.plan_gpu_kv_manager(req)
    assert plan.reuse is False
    assert plan.primary_config is not None and plan.primary_config.num_pages > 0
    assert plan.aux_config is not None and plan.aux_config.num_pages > 0


def test_plan_gpu_kv_real_gptoss_non_dsa_no_aux():
    """GPT-OSS-120B is not a DSA model → real aux builder returns None."""
    _require_real_config_module()
    req = GpuKvManagerRequest(
        model_name="gpt-oss-120b",
        sequence_tokens=(2048,),
        has_manager=False,
        current_num_pages=0,
        capacity=_cap_req_for_plan(),
    )
    plan = KVCacheManager.plan_gpu_kv_manager(req)
    assert plan.reuse is False
    assert plan.primary_config is not None and plan.primary_config.num_pages > 0
    assert plan.aux_config is None


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
