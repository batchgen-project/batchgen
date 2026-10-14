import pytest
import torch

from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
)


def _manager():
    config = GPUPagedKVConfig(
        num_layers=2,
        num_pages=16,
        page_size_tokens=4,
        num_k_heads=1,
        k_head_dim=2,
        num_v_heads=1,
        v_head_dim=2,
        kv_dtype=torch.float32,
    )
    manager = GPUPagedKVCacheManager(config=config, device="cpu")
    manager.initialize()
    return manager


def _read_layer_sequence(manager, sequence_id, length, *, value=False):
    k_cache, v_cache = manager.get_kv_tensors()
    cache = v_cache if value else k_cache
    pages = manager._sequences[sequence_id].pages.tolist()
    chunks = []
    remaining = length
    for page in pages:
        take = min(remaining, manager.config.page_size_tokens)
        chunks.append(cache[0, page, :take].clone())
        remaining -= take
        if remaining == 0:
            break
    return torch.cat(chunks, dim=0)


def test_suffix_append_writes_across_page_boundary():
    manager = _manager()
    manager.allocate_pages_for_sequences([101], [7])
    plan = manager.prepare_prefill_suffix_append(
        sequence_ids=[101], prefix_lens=[3], suffix_lens=[4]
    )
    suffix_k = torch.arange(8, dtype=torch.float32).view(4, 1, 2)
    suffix_v = suffix_k + 10

    manager.append_layer_prefill_suffix_tokens(
        k_tensor=suffix_k,
        v_tensor=suffix_v,
        append_plan=plan,
        layer_idx=0,
    )

    full_k = _read_layer_sequence(manager, 101, 7)
    full_v = _read_layer_sequence(manager, 101, 7, value=True)
    torch.testing.assert_close(full_k[:3], torch.zeros_like(full_k[:3]))
    torch.testing.assert_close(full_v[:3], torch.zeros_like(full_v[:3]))
    torch.testing.assert_close(full_k[3:], suffix_k)
    torch.testing.assert_close(full_v[3:], suffix_v)


def test_suffix_append_handles_hit_miss_and_full_hit_together():
    manager = _manager()
    manager.allocate_pages_for_sequences([101], [5])
    manager.allocate_pages_for_sequences([103], [4])
    plan = manager.prepare_prefill_suffix_append(
        sequence_ids=[101, 102, 103],
        prefix_lens=[3, 0, 3],
        suffix_lens=[2, 3, 1],
    )
    suffix_k = torch.arange(12, dtype=torch.float32).view(6, 1, 2)
    suffix_v = suffix_k + 100

    manager.append_layer_prefill_suffix_tokens(
        k_tensor=suffix_k,
        v_tensor=suffix_v,
        append_plan=plan,
        layer_idx=0,
    )

    torch.testing.assert_close(
        _read_layer_sequence(manager, 101, 5)[3:], suffix_k[:2]
    )
    torch.testing.assert_close(
        _read_layer_sequence(manager, 102, 3), suffix_k[2:5]
    )
    torch.testing.assert_close(
        _read_layer_sequence(manager, 103, 4)[3:], suffix_k[5:]
    )


def test_reused_prefix_requires_preallocated_gpu_pages():
    manager = _manager()

    with pytest.raises(KeyError, match="prefix-reused sequence"):
        manager.prepare_prefill_suffix_append(
            sequence_ids=[101], prefix_lens=[2], suffix_lens=[3]
        )


def test_host_page_identity_shares_gpu_pages_until_last_release():
    manager = _manager()
    before = manager.get_stats()

    manager.allocate_pages_for_sequences_with_page_keys(
        [101, 102],
        [12, 12],
        [
            [1001, 1002, 2001],
            [1001, 1002, 2002],
        ],
    )

    pages_101 = manager._sequences[101].pages.tolist()
    pages_102 = manager._sequences[102].pages.tolist()
    assert pages_101[:2] == pages_102[:2]
    assert pages_101[2] != pages_102[2]
    assert manager.get_stats().num_used_pages == before.num_used_pages + 4

    manager.free_pages_for_sequences([101])
    assert manager.get_stats().num_used_pages == before.num_used_pages + 3
    assert 1001 in manager._shared_page_key_to_gpu_page
    assert 1002 in manager._shared_page_key_to_gpu_page

    manager.free_pages_for_sequences([102])
    assert manager.get_stats().num_used_pages == before.num_used_pages
    assert manager._shared_page_key_to_gpu_page == {}
    assert manager._gpu_page_refcounts == {}


def test_shared_gpu_allocation_is_transactional_on_capacity_failure():
    manager = _manager()
    before = manager.get_stats()

    with pytest.raises(RuntimeError, match="Insufficient free pages"):
        manager.allocate_pages_for_sequences_with_page_keys(
            [101],
            [68],
            [list(range(17))],
        )

    assert manager.get_stats() == before
    assert manager._sequences == {}
