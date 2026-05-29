"""Unit tests for `batchgen.worker.kv_manager`.

Covers Phases 5.1 stats, 5.2 page-table capacity, and 5.3 token-budget cache.
Single-rank tests with a fake ``KVStatsBackend``. Real ``SequenceBatch``
fixtures — no mocks of the underlying batch / status enum per Phase A §G.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from batchgen.query_book import QueryBookEntry
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus
from batchgen.worker.kv_manager import (
    HostKVUtilization,
    KVCacheManager,
    KVStats,
    KVStatsBackend,
    KVUtilizationRequest,
    PageTableCapacityRequest,
    TokenBudgetRequest,
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
