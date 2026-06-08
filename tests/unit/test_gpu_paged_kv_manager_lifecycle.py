from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager


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
