from types import SimpleNamespace

import pytest
import torch

from batchgen.kv_cache.dual_host_kv_coordinator import DualHostKVCoordinator
from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator
from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
)


def _allocation_result(
    sequence_id=1,
    shared_prefix_pages=None,
    private_pages=None,
    shared_prefix_tokens=4,
):
    shared_prefix_pages = [] if shared_prefix_pages is None else shared_prefix_pages
    private_pages = [8] if private_pages is None else private_pages
    return {
        "sequence_id": sequence_id,
        "shared_prefix_pages": shared_prefix_pages,
        "private_pages": private_pages,
        "shared_prefix_tokens": shared_prefix_tokens,
        "private_start_token": shared_prefix_tokens,
        "logical_page_count": len(shared_prefix_pages) + len(private_pages),
        "physical_pages_allocated": len(private_pages),
        "full_hit": False,
        "miss_reason": "",
    }


def _prefix_stats(entries=1, lookup_hits=0):
    return SimpleNamespace(
        entries=entries,
        lookup_hits=lookup_hits,
        lookup_misses=0,
        shared_pages_attached=0,
        prefix_pin_increments=0,
        prefix_pin_decrements=0,
        host_pages_saved=0,
        eviction_epoch=0,
        eviction_runs=0,
        evicted_entries=0,
        evicted_prefix_pins=0,
        evicted_pages_immediately_freed=0,
        evicted_active_ref_entries=0,
        eviction_protected_skips=0,
        eviction_target_failures=0,
    )


def _eviction_result(entries_removed=0):
    return SimpleNamespace(
        entries_removed=entries_removed,
        pages_immediately_freed=0,
        prefix_pins_released=0,
        protected_entries_skipped=0,
        active_ref_entries_removed=0,
        reached_target=True,
    )


class _FakeHostPrefixView:
    def __init__(
        self,
        allocation_results=None,
        inserted_pages=1,
        stats=None,
        eviction=None,
        shared_pages=None,
        shared_tokens=4,
    ):
        self.allocation_results = (
            [_allocation_result()]
            if allocation_results is None
            else allocation_results
        )
        self.estimate_results = list(self.allocation_results)
        self.inserted_pages = inserted_pages
        self.stats = _prefix_stats() if stats is None else stats
        self.eviction = _eviction_result() if eviction is None else eviction
        self.shared_pages = [] if shared_pages is None else shared_pages
        self.shared_tokens = shared_tokens
        self.allocate_calls = []
        self.estimate_calls = []
        self.commit_calls = []
        self.release_calls = []
        self.clear_calls = 0
        self.prefix_load_calls = []

    def allocate_pages_for_sequences_with_prefix(self, requests):
        self.allocate_calls.append(list(requests))
        return [dict(item) for item in self.allocation_results]

    def estimate_pages_for_sequences_with_prefix(self, requests):
        self.estimate_calls.append(list(requests))
        return [dict(item) for item in self.estimate_results]

    def commit_sequence_prefix_pages(
        self,
        sequence_id,
        token_ids,
        namespace_hash=0,
    ):
        self.commit_calls.append((sequence_id, list(token_ids), namespace_hash))
        return self.inserted_pages

    def release_sequence_pages(self, sequence_ids):
        self.release_calls.append(list(sequence_ids))

    def shared_prefix_pages(self, sequence_id):
        return list(self.shared_pages)

    def shared_prefix_tokens(self, sequence_id):
        return self.shared_tokens

    def get_prefix_cache_stats(self):
        return self.stats

    def prefix_cache_debug_entries(self, limit=0, cold_first=True):
        return []

    def clear_prefix_cache(self):
        self.clear_calls += 1

    def evict_prefix_cache_until_free(
        self,
        target_free_pages,
        protected_pages=None,
        max_entries_to_scan=0,
    ):
        return self.eviction

    def async_load_prefix_pages_to_device(
        self,
        host_page_ids,
        k_device_ptrs,
        v_device_ptrs,
    ):
        task = SimpleNamespace(wait=lambda: None)
        self.prefix_load_calls.append(
            (host_page_ids, k_device_ptrs, v_device_ptrs, task)
        )
        return task


def test_dual_host_prefix_allocation_delegates_to_both_views():
    primary = _FakeHostPrefixView(shared_pages=[3], shared_tokens=4)
    auxiliary = _FakeHostPrefixView(shared_pages=[9], shared_tokens=4)
    coordinator = DualHostKVCoordinator(primary, auxiliary)

    requests = [(1, [10, 11, 12, 13], 8, 99)]
    result = coordinator.allocate_pages_for_sequences_with_prefix(requests)

    assert result == primary.allocation_results
    assert primary.allocate_calls == [requests]
    assert auxiliary.allocate_calls == [requests]
    assert coordinator.shared_prefix_pages(1) == [3]
    assert coordinator.shared_prefix_tokens(1) == 4


def test_dual_host_prefix_allocation_page_id_drift_is_allowed():
    primary = _FakeHostPrefixView(
        allocation_results=[_allocation_result(private_pages=[8])]
    )
    auxiliary = _FakeHostPrefixView(
        allocation_results=[_allocation_result(private_pages=[9])]
    )
    coordinator = DualHostKVCoordinator(primary, auxiliary)

    result = coordinator.allocate_pages_for_sequences_with_prefix(
        [(1, [10, 11, 12, 13], 8, 99)]
    )

    assert result == primary.allocation_results
    assert primary.release_calls == []
    assert auxiliary.release_calls == []


def test_dual_host_prefix_allocation_length_mismatch_raises_and_releases():
    primary = _FakeHostPrefixView(
        allocation_results=[_allocation_result(private_pages=[8, 7])]
    )
    auxiliary = _FakeHostPrefixView(
        allocation_results=[_allocation_result(private_pages=[9])]
    )
    coordinator = DualHostKVCoordinator(primary, auxiliary)

    with pytest.raises(RuntimeError, match="prefix allocation mismatch"):
        coordinator.allocate_pages_for_sequences_with_prefix(
            [(1, [10, 11, 12, 13], 8, 99)]
        )

    assert primary.release_calls == [[1]]
    assert auxiliary.release_calls == [[1]]


def test_dual_host_prefix_commit_and_stats_fail_fast_on_drift():
    primary = _FakeHostPrefixView(inserted_pages=1, stats=_prefix_stats(entries=1))
    auxiliary = _FakeHostPrefixView(inserted_pages=2, stats=_prefix_stats(entries=2))
    coordinator = DualHostKVCoordinator(primary, auxiliary)

    with pytest.raises(RuntimeError, match="inserted-page mismatch"):
        coordinator.commit_sequence_prefix_pages(1, [10, 11, 12, 13], 99)

    with pytest.raises(RuntimeError, match="prefix stats mismatch"):
        coordinator.get_prefix_cache_stats()


def test_dual_host_prefix_estimate_and_eviction_are_mirrored():
    primary = _FakeHostPrefixView(eviction=_eviction_result(entries_removed=1))
    auxiliary = _FakeHostPrefixView(eviction=_eviction_result(entries_removed=1))
    coordinator = DualHostKVCoordinator(primary, auxiliary)

    requests = [(1, [10, 11, 12, 13], 8, 99)]
    assert coordinator.estimate_pages_for_sequences_with_prefix(requests) == (
        primary.estimate_results
    )
    result = coordinator.evict_prefix_cache_until_free(
        4,
        protected_pages=[1],
    )
    assert result.entries_removed == 1
    coordinator.clear_prefix_cache()
    assert primary.clear_calls == 1
    assert auxiliary.clear_calls == 1


def test_dual_host_prefix_materialization_load_uses_primary_only():
    primary = _FakeHostPrefixView()
    auxiliary = _FakeHostPrefixView()
    coordinator = DualHostKVCoordinator(primary, auxiliary)

    host_pages = torch.tensor([1, 2], dtype=torch.int32)
    k_ptrs = object()
    v_ptrs = object()
    task = coordinator.async_load_prefix_pages_to_device(
        host_pages,
        k_ptrs,
        v_ptrs,
    )

    assert task is primary.prefix_load_calls[0][3]
    assert primary.prefix_load_calls[0][:3] == (host_pages, k_ptrs, v_ptrs)
    assert auxiliary.prefix_load_calls == []


def _make_gpu_config() -> GPUPagedKVConfig:
    return GPUPagedKVConfig(
        num_layers=1,
        num_pages=16,
        page_size_tokens=4,
        num_k_heads=1,
        k_head_dim=2,
        num_v_heads=1,
        v_head_dim=2,
        kv_dtype=torch.bfloat16,
    )


def test_dual_gpu_prefix_allocation_mirrors_shared_pages_on_cpu():
    primary = GPUPagedKVCacheManager(config=_make_gpu_config(), device="cpu")
    auxiliary = GPUPagedKVCacheManager(config=_make_gpu_config(), device="cpu")
    primary.initialize()
    auxiliary.initialize()
    coordinator = DualKVCacheCoordinator(primary, auxiliary)

    result = coordinator.allocate_pages_for_sequences_with_prefix(
        sequence_ids=[101, 102],
        num_tokens=[16, 16],
        shared_prefix_pages=[[10, 11], [10, 11]],
    )

    assert result[101] == primary._sequences[101].pages.tolist()
    assert torch.equal(
        primary._sequences[101].pages,
        auxiliary._sequences[101].pages,
    )
    assert torch.equal(
        primary._sequences[102].pages,
        auxiliary._sequences[102].pages,
    )
    assert primary.get_stats().num_shared_prefix_pages == 2
    assert auxiliary.get_stats().num_shared_prefix_pages == 2

    coordinator.free_pages_for_sequences([101, 102])
    assert primary.get_stats().num_used_pages == 0
    assert auxiliary.get_stats().num_used_pages == 0
