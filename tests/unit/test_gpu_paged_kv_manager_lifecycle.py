import torch

from batchgen.kv_cache.gpu_paged_kv_manager import (
    GPUPagedKVCacheManager,
    GPUPagedKVConfig,
)


def test_destroy_releases_cuda_cache_even_after_runtime_state_reset():
    manager = object.__new__(GPUPagedKVCacheManager)
    manager._is_initialized = False
    calls = []

    def release_cached_cuda_memory():
        calls.append("released")

    manager._release_cached_cuda_memory = release_cached_cuda_memory

    manager.destroy(empty_cuda_cache=True)

    assert calls == ["released"]


def test_destroy_skips_cuda_cache_release_for_uninitialized_noop():
    manager = object.__new__(GPUPagedKVCacheManager)
    manager._is_initialized = False
    calls = []

    def release_cached_cuda_memory():
        calls.append("released")

    manager._release_cached_cuda_memory = release_cached_cuda_memory

    manager.destroy(empty_cuda_cache=False)

    assert calls == []


def test_append_prefill_suffix_resolves_logical_layer_mapping():
    config = GPUPagedKVConfig(
        num_layers=2,
        num_pages=4,
        page_size_tokens=4,
        num_k_heads=1,
        k_head_dim=2,
        num_v_heads=1,
        v_head_dim=2,
        kv_dtype=torch.float32,
        logical_to_physical_layer=(0, 1, 0),
    )
    manager = GPUPagedKVCacheManager(config=config, device="cpu")
    manager.initialize()
    manager.allocate_pages_for_sequences([101], [6])
    manager.rebuild_page_table([101])
    plan = manager.prepare_prefill_suffix_append(
        sequence_ids=[101],
        prefix_lens=[4],
        suffix_lens=[2],
        rebuild_page_table=False,
    )

    k_tensor = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    v_tensor = torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]])
    manager.append_layer_prefill_suffix_tokens(
        k_tensor=k_tensor,
        v_tensor=v_tensor,
        append_plan=plan,
        layer_idx=2,
    )

    k_cache, v_cache = manager.get_kv_tensors()
    page_for_token_four = int(manager._sequences[101].pages[1].item())
    assert k_cache[0, page_for_token_four, 0, 0].tolist() == [1.0, 2.0]
    assert k_cache[0, page_for_token_four, 1, 0].tolist() == [3.0, 4.0]
    assert v_cache[0, page_for_token_four, 0, 0].tolist() == [5.0, 6.0]
    assert v_cache[0, page_for_token_four, 1, 0].tolist() == [7.0, 8.0]
    assert torch.count_nonzero(k_cache[1]).item() == 0
