import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_gpu_manager_module():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "batchgen" / "config" / "config.py"
    config_spec = importlib.util.spec_from_file_location(
        "_batchgen_config_config_for_gpu_page_table_test",
        config_path,
    )
    config_module = importlib.util.module_from_spec(config_spec)
    sys.modules[config_spec.name] = config_module
    config_spec.loader.exec_module(config_module)

    previous_config_pkg = sys.modules.get("batchgen.config")
    previous_config_module = sys.modules.get("batchgen.config.config")
    previous_gpu_kv_kernels = sys.modules.get("batchgen.kv_cache.gpu_kv_kernels")
    config_pkg = types.ModuleType("batchgen.config")
    config_pkg.__path__ = [str(repo_root / "batchgen" / "config")]
    config_pkg.config = config_module
    sys.modules["batchgen.config"] = config_pkg
    sys.modules["batchgen.config.config"] = config_module
    gpu_kv_kernels = types.ModuleType("batchgen.kv_cache.gpu_kv_kernels")

    def _unused_gpu_kernel(*args, **kwargs):
        raise RuntimeError("GPU KV kernels are not used by this page-table test")

    gpu_kv_kernels.run_paged_kv_token_update = _unused_gpu_kernel
    gpu_kv_kernels.run_paged_kv_token_update_fused = _unused_gpu_kernel
    sys.modules["batchgen.kv_cache.gpu_kv_kernels"] = gpu_kv_kernels
    try:
        manager_path = repo_root / "batchgen" / "kv_cache" / "gpu_paged_kv_manager.py"
        manager_spec = importlib.util.spec_from_file_location(
            "_batchgen_gpu_paged_kv_manager_for_test",
            manager_path,
        )
        manager_module = importlib.util.module_from_spec(manager_spec)
        sys.modules[manager_spec.name] = manager_module
        manager_spec.loader.exec_module(manager_module)
        return manager_module
    finally:
        if previous_config_pkg is None:
            sys.modules.pop("batchgen.config", None)
        else:
            sys.modules["batchgen.config"] = previous_config_pkg
        if previous_config_module is None:
            sys.modules.pop("batchgen.config.config", None)
        else:
            sys.modules["batchgen.config.config"] = previous_config_module
        if previous_gpu_kv_kernels is None:
            sys.modules.pop("batchgen.kv_cache.gpu_kv_kernels", None)
        else:
            sys.modules["batchgen.kv_cache.gpu_kv_kernels"] = previous_gpu_kv_kernels


_gpu_manager_module = _load_gpu_manager_module()
GPUPagedKVCacheManager = _gpu_manager_module.GPUPagedKVCacheManager
GPUPagedKVConfig = _gpu_manager_module.GPUPagedKVConfig


def _make_manager(num_pages: int = 32) -> GPUPagedKVCacheManager:
    config = GPUPagedKVConfig(
        num_layers=1,
        num_pages=num_pages,
        page_size_tokens=4,
        num_k_heads=1,
        k_head_dim=8,
        num_v_heads=0,
        v_head_dim=0,
        kv_dtype=torch.bfloat16,
    )
    manager = GPUPagedKVCacheManager(config=config, device="cpu")
    manager.initialize()
    return manager


def test_gpu_page_table_storage_is_stable_across_rebuilds():
    manager = _make_manager()
    manager.allocate_pages_for_sequences([10, 20, 30], [4, 12, 8])

    first_view = manager.rebuild_page_table([10, 20])
    backing = manager._gpu_page_table_manager.gpu_table
    assert backing is not None
    data_ptr = backing.data_ptr()
    shape = tuple(backing.shape)

    assert first_view.shape == (2, shape[1])
    assert manager._gpu_page_table_manager.seq_id_to_slot == {10: 0, 20: 1}
    torch.testing.assert_close(first_view[0, :1], manager._sequences[10].pages)
    torch.testing.assert_close(first_view[1, :3], manager._sequences[20].pages)

    second_view = manager.rebuild_page_table([30])
    assert manager._gpu_page_table_manager.gpu_table.data_ptr() == data_ptr
    assert tuple(manager._gpu_page_table_manager.gpu_table.shape) == shape
    assert second_view.shape == (1, shape[1])
    assert manager._gpu_page_table_manager.seq_id_to_slot == {30: 0}
    torch.testing.assert_close(second_view[0, :2], manager._sequences[30].pages)
    assert torch.all(manager._gpu_page_table_manager.gpu_table[1:, :] == -1)

    third_view = manager.rebuild_page_table([20, 10, 30])
    assert manager._gpu_page_table_manager.gpu_table.data_ptr() == data_ptr
    assert tuple(manager._gpu_page_table_manager.gpu_table.shape) == shape
    assert third_view.shape == (3, shape[1])
    assert manager._gpu_page_table_manager.seq_id_to_slot == {20: 0, 10: 1, 30: 2}


def test_clear_page_table_preserves_backing_storage():
    manager = _make_manager()
    manager.allocate_pages_for_sequences([1], [4])
    manager.rebuild_page_table([1])

    backing = manager._gpu_page_table_manager.gpu_table
    assert backing is not None
    data_ptr = backing.data_ptr()
    shape = tuple(backing.shape)

    manager.clear_page_table()

    cleared = manager._gpu_page_table_manager.gpu_table
    assert cleared is not None
    assert cleared.data_ptr() == data_ptr
    assert tuple(cleared.shape) == shape
    assert manager._gpu_page_table_manager.seq_id_to_slot == {}
    assert manager._gpu_page_table_manager.slot_to_seq_id == []
    assert torch.all(cleared == -1)
