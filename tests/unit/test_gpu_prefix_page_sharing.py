import torch

from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
)


def _make_config() -> GPUPagedKVConfig:
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


def test_gpu_prefix_pages_are_shared_and_refcounted_on_cpu():
    manager = GPUPagedKVCacheManager(config=_make_config(), device="cpu")
    manager.initialize()

    manager.allocate_pages_for_sequences_with_prefix(
        sequence_ids=[101, 102],
        num_tokens=[16, 16],
        shared_prefix_pages=[[10, 11], [10, 11]],
    )

    pages_101 = manager._sequences[101].pages.tolist()
    pages_102 = manager._sequences[102].pages.tolist()
    assert pages_101[:2] == pages_102[:2]
    assert pages_101[2:] != pages_102[2:]

    stats = manager.get_stats()
    assert stats.num_used_pages == 6
    assert stats.num_shared_prefix_pages == 2
    assert stats.num_shared_prefix_refs == 4
    assert stats.shared_prefix_pages_reused == 2

    table = manager.rebuild_page_table([101, 102])
    assert table[0, 0].item() == table[1, 0].item()
    assert table[0, 1].item() == table[1, 1].item()

    manager.free_pages_for_sequences([101])
    stats = manager.get_stats()
    assert stats.num_used_pages == 4
    assert stats.num_shared_prefix_pages == 2
    assert stats.num_shared_prefix_refs == 2

    manager.free_pages_for_sequences([102])
    stats = manager.get_stats()
    assert stats.num_used_pages == 0
    assert stats.num_free_pages == 16
    assert stats.num_shared_prefix_pages == 0
    assert stats.num_shared_prefix_refs == 0
