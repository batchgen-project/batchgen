import ctypes
import errno
import random
import string

import pytest
import torch

from batchgen.models.engine_loader import core_engine as bg

_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def _random_shm_name() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
    return f"/batchgen_prefix_kv_{suffix}"


def _shm_unlink(name: str) -> None:
    result = _LIBC.shm_unlink(name.encode("utf-8"))
    if result != 0:
        err = ctypes.get_errno()
        if err != errno.ENOENT:
            raise OSError(err, f"shm_unlink({name}) failed")


def _make_config(
    shm_name: str, num_pages: int = 32
) -> bg.HostPagedKVConfig:  # type: ignore[name-defined]
    cfg = bg.HostPagedKVConfig()
    cfg.shm_name = shm_name
    cfg.num_layers = 1
    cfg.num_pages = num_pages
    cfg.page_size_tokens = 4
    cfg.num_k_heads = 1
    cfg.k_head_dim = 1
    cfg.num_v_heads = 0
    cfg.v_head_dim = 0
    cfg.k_element_size_bytes = 2
    cfg.v_element_size_bytes = 0
    cfg.sequence_table_capacity = 64
    cfg.alignment_bytes = 64
    return cfg


def _make_worker(shm_name: str, num_pages: int = 32):
    worker = bg.MLAHostPagedKVWorkerView(_make_config(shm_name, num_pages=num_pages))
    worker.initialize(device_index=0, create_region=True)
    return worker


def test_prefix_lookup_reuses_only_complete_pages():
    shm_name = _random_shm_name()
    worker = None
    try:
        worker = _make_worker(shm_name)
        tokens = list(range(10))  # two full pages plus one partial page

        worker.register_sequences([1])
        first = worker.allocate_pages_for_sequences_with_prefix([(1, tokens, 12)])
        assert first[0]["shared_prefix_tokens"] == 0
        assert len(first[0]["private_pages"]) == 3

        inserted = worker.commit_sequence_prefix_pages(1, tokens)
        assert inserted == 2

        worker.register_sequences([2])
        second = worker.allocate_pages_for_sequences_with_prefix([(2, tokens, 12)])
        assert second[0]["shared_prefix_tokens"] == 8
        assert len(second[0]["shared_prefix_pages"]) == 2
        assert len(second[0]["private_pages"]) == 1

        table = worker.build_page_table([2])[0]
        assert table[:2] == second[0]["shared_prefix_pages"]
        assert table[2:] == second[0]["private_pages"]
        assert worker.shared_prefix_pages(2) == second[0]["shared_prefix_pages"]
        assert worker.private_pages(2) == second[0]["private_pages"]

        stats = worker.get_stats()
        assert stats.num_used_pages == 4
        assert stats.num_sequence_ref_pages == 6
        assert stats.num_prefix_pinned_pages == 2

        prefix_stats = worker.get_prefix_cache_stats()
        assert prefix_stats.entries == 2
        assert prefix_stats.lookup_hits == 1
        assert prefix_stats.host_pages_saved == 2
    finally:
        if worker is not None:
            worker.release_sequence_pages([1, 2])
            worker.clear_prefix_cache()
            worker.shutdown()
        _shm_unlink(shm_name)


def test_parent_page_hash_prevents_invalid_reuse():
    shm_name = _random_shm_name()
    worker = None
    try:
        worker = _make_worker(shm_name)
        first_tokens = [1, 1, 1, 1, 9, 9, 9, 9]
        same_second_page_different_parent = [2, 2, 2, 2, 9, 9, 9, 9]

        worker.register_sequences([1])
        worker.allocate_pages_for_sequences_with_prefix([(1, first_tokens, 8)])
        worker.commit_sequence_prefix_pages(1, first_tokens)

        worker.register_sequences([2])
        result = worker.allocate_pages_for_sequences_with_prefix(
            [(2, same_second_page_different_parent, 8)]
        )[0]
        assert result["shared_prefix_tokens"] == 0
        assert result["shared_prefix_pages"] == []
        assert len(result["private_pages"]) == 2
    finally:
        if worker is not None:
            worker.release_sequence_pages([1, 2])
            worker.clear_prefix_cache()
            worker.shutdown()
        _shm_unlink(shm_name)


def test_prefix_pins_and_sequence_refs_release_independently():
    shm_name = _random_shm_name()
    worker = None
    try:
        worker = _make_worker(shm_name)
        tokens = [3, 3, 3, 3, 4, 4, 4, 4]

        worker.register_sequences([1])
        worker.allocate_pages_for_sequences_with_prefix([(1, tokens, 8)])
        worker.commit_sequence_prefix_pages(1, tokens)
        worker.release_sequence_pages([1])

        after_release = worker.get_stats()
        assert after_release.num_used_pages == 2
        assert after_release.num_sequence_ref_pages == 0
        assert after_release.num_prefix_pinned_pages == 2

        worker.register_sequences([2])
        result = worker.allocate_pages_for_sequences_with_prefix([(2, tokens, 12)])[0]
        assert result["shared_prefix_tokens"] == 8
        assert len(result["private_pages"]) == 1

        worker.clear_prefix_cache()
        after_clear = worker.get_stats()
        assert after_clear.num_prefix_pinned_pages == 0
        assert after_clear.num_used_pages == 3
        assert after_clear.num_sequence_ref_pages == 3

        worker.release_sequence_pages([2])
        final_stats = worker.get_stats()
        assert final_stats.num_used_pages == 0
        assert final_stats.num_sequence_ref_pages == 0
    finally:
        if worker is not None:
            worker.shutdown()
        _shm_unlink(shm_name)


def test_prefix_cache_leaf_eviction_preserves_shorter_prefix():
    shm_name = _random_shm_name()
    worker = None
    try:
        worker = _make_worker(shm_name)
        tokens = list(range(12))  # three full pages

        worker.register_sequences([1])
        first = worker.allocate_pages_for_sequences_with_prefix([(1, tokens, 12)])
        worker.commit_sequence_prefix_pages(1, tokens)
        worker.release_sequence_pages([1])

        free_before = worker.free_page_count()
        eviction = worker.evict_prefix_cache_until_free(free_before + 1)
        assert eviction.reached_target
        assert eviction.entries_removed == 1
        assert worker.get_prefix_cache_stats().entries == 2

        worker.register_sequences([2])
        second = worker.allocate_pages_for_sequences_with_prefix([(2, tokens, 12)])[0]
        assert second["shared_prefix_tokens"] == 8
        assert len(second["shared_prefix_pages"]) == 2
        assert len(second["private_pages"]) == 1
        assert second["shared_prefix_pages"] == first[0]["private_pages"][:2]
    finally:
        if worker is not None:
            try:
                worker.release_sequence_pages([2])
            except Exception:
                pass
            worker.clear_prefix_cache()
            worker.shutdown()
        _shm_unlink(shm_name)


def test_prefix_cache_eviction_skips_protected_leaf_pages():
    shm_name = _random_shm_name()
    worker = None
    try:
        worker = _make_worker(shm_name)
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]

        worker.register_sequences([1])
        first = worker.allocate_pages_for_sequences_with_prefix([(1, tokens, 8)])
        worker.commit_sequence_prefix_pages(1, tokens)
        worker.release_sequence_pages([1])

        leaf_page = first[0]["private_pages"][1]
        free_before = worker.free_page_count()
        eviction = worker.evict_prefix_cache_until_free(
            free_before + 1, protected_pages=[leaf_page]
        )
        assert not eviction.reached_target
        assert eviction.entries_removed == 0
        assert eviction.protected_entries_skipped >= 1
        assert worker.get_prefix_cache_stats().entries == 2
    finally:
        if worker is not None:
            try:
                worker.release_sequence_pages([1, 2])
            except Exception:
                pass
            worker.clear_prefix_cache()
            worker.shutdown()
        _shm_unlink(shm_name)


def test_prefix_cache_eviction_unblocks_allocation_pressure():
    shm_name = _random_shm_name()
    worker = None
    try:
        worker = _make_worker(shm_name, num_pages=5)
        first_tokens = [1, 1, 1, 1, 2, 2, 2, 2]
        second_tokens = [3, 3, 3, 3, 4, 4, 4, 4]
        miss_tokens = [9, 9, 9, 9, 10, 10, 10, 10]

        worker.register_sequences([1])
        worker.allocate_pages_for_sequences_with_prefix([(1, first_tokens, 8)])
        worker.commit_sequence_prefix_pages(1, first_tokens)
        worker.release_sequence_pages([1])

        worker.register_sequences([2])
        worker.allocate_pages_for_sequences_with_prefix([(2, second_tokens, 8)])
        worker.commit_sequence_prefix_pages(2, second_tokens)
        worker.release_sequence_pages([2])

        assert worker.free_page_count() == 1
        worker.register_sequences([3])
        result = worker.allocate_pages_for_sequences_with_prefix([(3, miss_tokens, 8)])[0]
        assert result["shared_prefix_tokens"] == 0
        assert len(result["private_pages"]) == 2

        stats = worker.get_prefix_cache_stats()
        assert stats.eviction_runs >= 1
        assert stats.evicted_entries >= 1
        assert stats.evicted_prefix_pins >= 1
    finally:
        if worker is not None:
            try:
                worker.release_sequence_pages([3])
            except Exception:
                pass
            worker.clear_prefix_cache()
            worker.shutdown()
        _shm_unlink(shm_name)


def test_prefix_cache_eviction_keeps_active_sequence_pages_alive():
    shm_name = _random_shm_name()
    worker = None
    try:
        worker = _make_worker(shm_name)
        tokens = [7, 7, 7, 7]

        worker.register_sequences([1])
        first = worker.allocate_pages_for_sequences_with_prefix([(1, tokens, 4)])
        full_k = torch.arange(
            1,
            5,
            dtype=torch.bfloat16,
            device="cuda:0",
        ).view(1, 4, 1, 1)
        task = worker.async_offload_layer_kv_to_host(
            layer_idx=0,
            sequence_ids=[1],
            k_tensor=full_k,
            v_tensor=None,
            sequence_lengths=[4],
        )
        task.result()
        worker.commit_sequence_prefix_pages(1, tokens)
        leaf_page = first[0]["private_pages"][0]
        worker.release_sequence_pages([1])

        worker.register_sequences([2])
        second = worker.allocate_pages_for_sequences_with_prefix([(2, tokens, 8)])[0]
        assert second["shared_prefix_tokens"] == 4
        assert second["shared_prefix_pages"] == [leaf_page]
        assert len(second["private_pages"]) == 1

        page_table_before = worker.build_page_table([2])[0]
        prefix_tokens_before = worker.shared_prefix_tokens(2)

        free_before = worker.free_page_count()
        eviction = worker.evict_prefix_cache_until_free(free_before + 1)
        assert not eviction.reached_target
        assert eviction.entries_removed == 1
        assert eviction.pages_immediately_freed == 0
        assert eviction.active_ref_entries_removed == 1
        assert worker.build_page_table([2])[0] == page_table_before
        assert worker.shared_prefix_tokens(2) == prefix_tokens_before

        k_cpu, _ = worker.read_sequence_kv_to_cpu(2)
        logical_tokens = k_cpu[0, :, :, 0, 0].reshape(-1).float().tolist()
        assert logical_tokens[:4] == pytest.approx([1, 2, 3, 4])

        ref_state = worker.page_ref_state(leaf_page)
        assert ref_state.sequence_refs == 1
        assert ref_state.prefix_pins == 0
        assert not ref_state.is_free

        worker.release_sequence_pages([2])
        final_ref_state = worker.page_ref_state(leaf_page)
        assert final_ref_state.sequence_refs == 0
        assert final_ref_state.prefix_pins == 0
        assert final_ref_state.is_free
    finally:
        if worker is not None:
            try:
                worker.release_sequence_pages([1, 2])
            except Exception:
                pass
            worker.clear_prefix_cache()
            worker.shutdown()
        _shm_unlink(shm_name)


def test_suffix_offload_uses_explicit_source_and_destination_offsets():
    shm_name = _random_shm_name()
    worker = None
    try:
        worker = _make_worker(shm_name)
        tokens = [10, 11, 12, 13, 14, 15, 16, 17]

        worker.register_sequences([1])
        worker.allocate_pages_for_sequences_with_prefix([(1, tokens, 8)])
        full_k = torch.arange(
            1,
            9,
            dtype=torch.bfloat16,
            device="cuda:0",
        ).view(1, 8, 1, 1)
        task = worker.async_offload_layer_kv_to_host(
            layer_idx=0,
            sequence_ids=[1],
            k_tensor=full_k,
            v_tensor=None,
            sequence_lengths=[8],
        )
        task.result()
        worker.commit_sequence_prefix_pages(1, tokens)

        worker.register_sequences([2])
        result = worker.allocate_pages_for_sequences_with_prefix([(2, tokens, 12)])[0]
        assert result["shared_prefix_tokens"] == 8
        assert len(result["private_pages"]) == 1

        suffix_k = torch.tensor(
            [90, 91],
            dtype=torch.bfloat16,
            device="cuda:0",
        ).view(1, 2, 1, 1)
        task = worker.async_offload_layer_kv_to_host_with_offsets(
            layer_idx=0,
            sequence_ids=[2],
            k_tensor=suffix_k,
            v_tensor=None,
            sequence_lengths=[2],
            source_token_starts=[0],
            destination_token_starts=[8],
        )
        task.result()

        k_cpu, _ = worker.read_sequence_kv_to_cpu(2)
        logical_tokens = k_cpu[0, :, :, 0, 0].reshape(-1).float().tolist()
        assert logical_tokens[:8] == pytest.approx([1, 2, 3, 4, 5, 6, 7, 8])
        assert logical_tokens[8:10] == pytest.approx([90, 91])

        bad_task = worker.async_offload_layer_kv_to_host_with_offsets(
            layer_idx=0,
            sequence_ids=[2],
            k_tensor=suffix_k[:, :1],
            v_tensor=None,
            sequence_lengths=[1],
            source_token_starts=[0],
            destination_token_starts=[0],
        )
        with pytest.raises(Exception, match="shared prefix"):
            bad_task.result()
    finally:
        if worker is not None:
            try:
                worker.release_sequence_pages([1, 2])
            except Exception:
                pass
            worker.clear_prefix_cache()
            worker.shutdown()
        _shm_unlink(shm_name)
