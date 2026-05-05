import importlib.util
import sys
import types
from pathlib import Path

import pytest
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


def test_rebuild_page_table_preserves_active_dynamic_api():
    manager = _make_manager()
    manager.allocate_pages_for_sequences([10, 20, 30], [4, 12, 8])

    first_view = manager.rebuild_page_table([10, 20])
    assert first_view.shape == (2, 4)
    assert tuple(manager._gpu_page_table_manager.gpu_table.shape) == tuple(first_view.shape)
    assert manager._gpu_page_table_manager.gpu_table.data_ptr() == first_view.data_ptr()
    assert manager._gpu_page_table_manager.seq_id_to_slot == {10: 0, 20: 1}
    torch.testing.assert_close(first_view[0, :1], manager._sequences[10].pages)
    torch.testing.assert_close(first_view[1, :3], manager._sequences[20].pages)

    second_view = manager.rebuild_page_table([30])
    assert second_view.shape == (1, 3)
    assert tuple(manager._gpu_page_table_manager.gpu_table.shape) == tuple(second_view.shape)
    assert manager._gpu_page_table_manager.gpu_table.data_ptr() == second_view.data_ptr()
    assert manager._gpu_page_table_manager.seq_id_to_slot == {30: 0}
    torch.testing.assert_close(second_view[0, :2], manager._sequences[30].pages)

    third_view = manager.rebuild_page_table([20, 10, 30])
    assert third_view.shape == (3, 4)
    assert tuple(manager._gpu_page_table_manager.gpu_table.shape) == tuple(third_view.shape)
    assert manager._gpu_page_table_manager.gpu_table.data_ptr() == third_view.data_ptr()
    assert manager._gpu_page_table_manager.seq_id_to_slot == {20: 0, 10: 1, 30: 2}


def test_cuda_graph_page_table_storage_is_stable_across_rebuilds():
    manager = _make_manager()
    manager.allocate_pages_for_sequences([10, 20, 30], [4, 12, 8])

    manager.rebuild_page_table([10, 20])
    first_state = manager.get_cuda_graph_page_table_state()
    backing = first_state.table
    data_ptr = backing.data_ptr()
    shape = tuple(backing.shape)
    expected_shape = (
        manager._gpu_page_table_manager.max_slots,
        manager._gpu_page_table_manager.graph_max_pages_per_sequence,
    )

    assert shape == expected_shape
    assert first_state.num_valid_slots == 2
    torch.testing.assert_close(first_state.slot_indices, torch.tensor([0, 1], dtype=torch.int32))
    torch.testing.assert_close(first_state.slot_to_seq_id, torch.tensor([10, 20], dtype=torch.int64))
    torch.testing.assert_close(backing[0, :1], manager._sequences[10].pages)
    torch.testing.assert_close(backing[1, :3], manager._sequences[20].pages)

    manager.rebuild_page_table([30])
    second_state = manager.get_cuda_graph_page_table_state()
    assert second_state.table.data_ptr() == data_ptr
    assert tuple(second_state.table.shape) == shape
    assert second_state.num_valid_slots == 1
    torch.testing.assert_close(second_state.slot_indices, torch.tensor([0], dtype=torch.int32))
    torch.testing.assert_close(second_state.slot_to_seq_id, torch.tensor([30], dtype=torch.int64))
    torch.testing.assert_close(second_state.table[0, :2], manager._sequences[30].pages)
    assert torch.all(second_state.table[0, 2:] == -1)
    assert torch.all(second_state.table[1:, :] == -1)

    manager.rebuild_page_table([20, 10, 30])
    third_table = manager.get_cuda_graph_page_table()
    assert third_table.data_ptr() == data_ptr
    assert tuple(third_table.shape) == shape
    torch.testing.assert_close(third_table[0, :3], manager._sequences[20].pages)
    torch.testing.assert_close(third_table[1, :1], manager._sequences[10].pages)
    torch.testing.assert_close(third_table[2, :2], manager._sequences[30].pages)


def test_clear_page_table_preserves_graph_storage_and_active_empty_api():
    manager = _make_manager()
    manager.allocate_pages_for_sequences([1], [4])
    manager.rebuild_page_table([1])

    graph_table = manager.get_cuda_graph_page_table()
    graph_data_ptr = graph_table.data_ptr()
    graph_shape = tuple(graph_table.shape)

    manager.clear_page_table()

    cleared = manager._gpu_page_table_manager.gpu_table
    assert cleared is not None
    assert tuple(cleared.shape) == (0, manager._gpu_page_table_manager.max_pages_per_sequence)
    assert manager._gpu_page_table_manager.seq_id_to_slot == {}
    assert manager._gpu_page_table_manager.slot_to_seq_id == []
    assert torch.all(cleared == -1)

    cleared_graph_state = manager.get_cuda_graph_page_table_state()
    assert cleared_graph_state.table.data_ptr() == graph_data_ptr
    assert tuple(cleared_graph_state.table.shape) == graph_shape
    assert cleared_graph_state.num_valid_slots == 0
    assert torch.all(cleared_graph_state.table == -1)
    assert cleared_graph_state.slot_indices.numel() == 0
    assert cleared_graph_state.slot_to_seq_id.numel() == 0


def test_cuda_graph_capacity_errors_do_not_break_active_dynamic_table(monkeypatch):
    monkeypatch.setenv("BATCHGEN_GPU_PAGE_TABLE_MAX_SLOTS", "2")
    manager = _make_manager(num_pages=8)
    manager.allocate_pages_for_sequences([1, 2, 3], [4, 4, 4])

    manager.rebuild_page_table([1, 2])
    first_state = manager.get_cuda_graph_page_table_state()
    graph_ptr = first_state.table.data_ptr()

    active_over_capacity = manager.rebuild_page_table([1, 2, 3])
    assert active_over_capacity.shape == (3, 2)
    assert tuple(manager._gpu_page_table_manager.gpu_table.shape) == (3, 2)
    with pytest.raises(RuntimeError, match="CUDA graph page table is not valid"):
        manager.get_cuda_graph_page_table()

    active_valid_again = manager.rebuild_page_table([3])
    graph_state = manager.get_cuda_graph_page_table_state()
    assert active_valid_again.shape == (1, 2)
    assert graph_state.table.data_ptr() == graph_ptr
    assert graph_state.num_valid_slots == 1
    torch.testing.assert_close(graph_state.table[0, :1], manager._sequences[3].pages)
    assert torch.all(graph_state.table[0, 1:] == -1)
    assert torch.all(graph_state.table[1:, :] == -1)


def test_ensure_cuda_graph_page_table_recovers_invalid_graph_storage(monkeypatch):
    monkeypatch.setenv("BATCHGEN_GPU_PAGE_TABLE_MAX_SLOTS", "2")
    manager = _make_manager(num_pages=8)
    manager.allocate_pages_for_sequences([1, 2, 3], [4, 4, 4])

    manager.rebuild_page_table([1, 2])
    first_state = manager.get_cuda_graph_page_table_state()
    graph_ptr = first_state.table.data_ptr()

    manager.rebuild_page_table([1, 2, 3])
    with pytest.raises(RuntimeError, match="CUDA graph page table is not valid"):
        manager.get_cuda_graph_page_table()

    refreshed = manager.ensure_cuda_graph_page_table([3])
    assert refreshed.data_ptr() == graph_ptr
    torch.testing.assert_close(refreshed[0, :1], manager._sequences[3].pages)
    assert torch.all(refreshed[0, 1:] == -1)
    assert torch.all(refreshed[1:] == -1)

    emptied = manager.ensure_cuda_graph_page_table([])
    assert emptied.data_ptr() == graph_ptr
    empty_state = manager.get_cuda_graph_page_table_state()
    assert empty_state.num_valid_slots == 0
    assert empty_state.slot_indices.numel() == 0
    assert tuple(manager._gpu_page_table_manager.gpu_table.shape) == (
        0,
        manager._gpu_page_table_manager.max_pages_per_sequence,
    )


def test_cuda_graph_page_table_storage_accessor_does_not_rebuild_active_table(monkeypatch):
    monkeypatch.setenv("BATCHGEN_GPU_PAGE_TABLE_MAX_SLOTS", "2")
    manager = _make_manager(num_pages=8)
    manager.allocate_pages_for_sequences([1, 2, 3], [4, 4, 4])

    manager.rebuild_page_table([1, 2])
    graph_ptr = manager.get_cuda_graph_page_table().data_ptr()

    manager.rebuild_page_table([1, 2, 3])
    with pytest.raises(RuntimeError, match="CUDA graph page table is not valid"):
        manager.get_cuda_graph_page_table()

    storage = manager.get_cuda_graph_page_table_storage()

    assert storage.data_ptr() == graph_ptr
    assert manager._gpu_page_table_manager.slot_to_seq_id == [1, 2, 3]
    assert tuple(manager._gpu_page_table_manager.gpu_table.shape) == (3, 2)
    with pytest.raises(RuntimeError, match="CUDA graph page table is not valid"):
        manager.get_cuda_graph_page_table()
