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
    previous_kv_cache_pkg = sys.modules.get("batchgen.kv_cache")
    previous_gpu_kv_kernels = sys.modules.get("batchgen.kv_cache.gpu_kv_kernels")
    config_pkg = types.ModuleType("batchgen.config")
    config_pkg.__path__ = [str(repo_root / "batchgen" / "config")]
    config_pkg.config = config_module
    sys.modules["batchgen.config"] = config_pkg
    sys.modules["batchgen.config.config"] = config_module
    kv_cache_pkg = types.ModuleType("batchgen.kv_cache")
    kv_cache_pkg.__path__ = [str(repo_root / "batchgen" / "kv_cache")]
    sys.modules["batchgen.kv_cache"] = kv_cache_pkg
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
        if previous_kv_cache_pkg is None:
            sys.modules.pop("batchgen.kv_cache", None)
        else:
            sys.modules["batchgen.kv_cache"] = previous_kv_cache_pkg
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


def test_reset_allocations_preserves_cache_and_graph_storage():
    manager = _make_manager(num_pages=8)
    manager.allocate_pages_for_sequences([10, 20], [8, 4])
    manager.rebuild_page_table([10, 20])
    k_cache, _ = manager.get_kv_tensors()
    k_cache.fill_(7)
    cache_ptr = k_cache.data_ptr()
    graph_ptr = manager.get_cuda_graph_page_table().data_ptr()

    manager.reset_allocations()

    reset_cache, _ = manager.get_kv_tensors()
    state = manager.get_cuda_graph_page_table_state()
    assert reset_cache.data_ptr() == cache_ptr
    assert state.table.data_ptr() == graph_ptr
    assert manager._sequences == {}
    assert manager.get_stats().num_free_pages == manager.config.num_pages
    assert state.num_valid_slots == 0
    assert torch.count_nonzero(reset_cache).item() == 0
    assert torch.all(state.table == -1)


def test_rebuild_and_clear_accept_inference_created_page_tables():
    manager = _make_manager()
    manager.allocate_pages_for_sequences([1, 2], [4, 8])

    with torch.inference_mode():
        manager.rebuild_page_table([1, 2])

    active = manager._gpu_page_table_manager.gpu_table
    graph = manager.get_cuda_graph_page_table_storage()
    assert torch.is_inference(active)
    assert torch.is_inference(graph)

    # Same-order rebuild reuses and mutates the inference-created active table.
    manager.rebuild_page_table([1, 2])
    # Reordering allocates a new active table but refreshes the existing
    # inference-created graph-stable backing table.
    manager.rebuild_page_table([2, 1])
    manager.clear_page_table()

    assert manager._gpu_page_table_manager.gpu_table.shape[0] == 0
    assert torch.all(manager.get_cuda_graph_page_table() == -1)


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


def test_reused_rebuild_refreshes_cuda_graph_page_table_after_extension():
    manager = _make_manager()
    manager.allocate_pages_for_sequences([10, 20], [12, 4])
    manager.rebuild_page_table([10, 20])
    assert manager.get_cuda_graph_page_table()[1, 1].item() == -1

    # Same slots, same order, enough columns: the active table is reused in
    # place. The graph-stable table must still receive the extended page.
    assert manager.extend_pages_for_sequence(20, 8) == 1
    active = manager.rebuild_page_table([10, 20])
    graph = manager.get_cuda_graph_page_table()

    assert manager._gpu_page_table_manager.gpu_table.data_ptr() == active.data_ptr()
    torch.testing.assert_close(graph[1, :2], manager._sequences[20].pages)
    torch.testing.assert_close(graph[:2, : active.shape[1]], active)


def test_layer_kv_cache_is_available_before_any_page_table():
    # A rank that holds no sequence never builds an active page table, yet the
    # GPT-OSS whole-model decode graph still captures on it and needs only the
    # fixed-address cache tensors.
    config = GPUPagedKVConfig(
        num_layers=2,
        num_pages=8,
        page_size_tokens=4,
        num_k_heads=1,
        k_head_dim=8,
        num_v_heads=1,
        v_head_dim=8,
        kv_dtype=torch.bfloat16,
    )
    manager = GPUPagedKVCacheManager(config=config, device="cpu")
    manager.initialize()
    assert manager._gpu_page_table_manager.gpu_table is None

    with pytest.raises(RuntimeError, match="GPU page table is not initialized"):
        manager.get_layer_kv_with_page_table(1)

    k_cache, v_cache = manager.get_layer_kv_cache(1)
    assert k_cache.data_ptr() == manager._k_cache[1].data_ptr()
    assert v_cache.data_ptr() == manager._v_cache[1].data_ptr()

    # Once a page table exists, both accessors return the same tensors.
    manager.allocate_pages_for_sequences([7], [4])
    manager.rebuild_page_table([7])
    k_with_table, v_with_table, table = manager.get_layer_kv_with_page_table(1)
    assert k_with_table.data_ptr() == k_cache.data_ptr()
    assert v_with_table.data_ptr() == v_cache.data_ptr()
    assert table is manager._gpu_page_table_manager.gpu_table


def test_layer_kv_cache_without_v_cache_returns_none():
    manager = _make_manager()
    k_cache, v_cache = manager.get_layer_kv_cache(0)
    assert k_cache.data_ptr() == manager._k_cache[0].data_ptr()
    assert v_cache is None


# --------------------------------------------------------------------------- #
# The graph table is REFRESHED in O(the rows' pages), not O(its reserved width)
# --------------------------------------------------------------------------- #
#
# The CUDA-graph page table is sized for the DEPLOYMENT context bound, so at a
# 1,048,576-token span and 64-token pages it is 16,384 columns wide while a
# decode row owns ~100 of them. The refresh used to re-establish `-1` across
# the whole width of every active row on every rebuild, which is a per-step
# cost proportional to the width instead of to the context: at 128 rows that is
# 128 * 16384 * 4 B = 8 MiB per pool per step, against ~64 KiB for the rows'
# real pages. These tests pin the accounting, not a timing.

_GRAPH_WIDTH = 16384
_GRAPH_SLOTS = 64
_PAGES_PER_ROW = 128


_WRITE_LEDGER = {"elements": 0}
_COUNTED_WRITES = ("fill_", "zero_", "copy_", "__setitem__")


class _WriteCountingTensor(torch.Tensor):
    """A real tensor that records how many ELEMENTS each in-place write touches.

    A subclass rather than a wrapper object, because the write patterns under
    test differ in exactly the place a wrapper hides: a width-proportional
    clear is ``table[:rows, :].fill_(-1)``, whose ``fill_`` lands on the SLICE.
    Slicing a subclass yields a subclass, so the slice's own write is counted;
    slicing a wrapper yields a plain tensor and the write disappears.
    """

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if getattr(func, "__name__", "") in _COUNTED_WRITES:
            target = args[0]
            with torch._C.DisableTorchFunctionSubclass():
                written = (
                    target[args[1]].numel()
                    if func.__name__ == "__setitem__"
                    else target.numel()
                )
            _WRITE_LEDGER["elements"] += int(written)
        return super().__torch_function__(func, types, args, kwargs)


def _reset_write_ledger() -> None:
    _WRITE_LEDGER["elements"] = 0


def _written_elements() -> int:
    return _WRITE_LEDGER["elements"]


def _make_wide_graph_manager(
    rows: int = _GRAPH_SLOTS, pages_per_row: int = _PAGES_PER_ROW
):
    """A manager whose graph table has the production width, write-counted."""
    config = GPUPagedKVConfig(
        num_layers=1,
        # Room for a second batch of rows alongside the first, so a slot
        # handover can be exercised without recycling pages.
        num_pages=2 * rows * (pages_per_row + 1),
        page_size_tokens=64,
        num_k_heads=1,
        k_head_dim=8,
        num_v_heads=0,
        v_head_dim=0,
        kv_dtype=torch.bfloat16,
        cuda_graph_max_pages_per_sequence=_GRAPH_WIDTH,
        cuda_graph_max_slots=_GRAPH_SLOTS,
    )
    manager = GPUPagedKVCacheManager(config=config, device="cpu")
    manager.initialize()
    page_manager = manager._gpu_page_table_manager
    assert page_manager.graph_max_pages_per_sequence == _GRAPH_WIDTH
    # Create the real storage (which also seeds the written-extent bookkeeping),
    # then count every write that reaches it.
    real = page_manager._ensure_cuda_graph_table()
    counted = torch.Tensor._make_subclass(_WriteCountingTensor, real, False)
    page_manager._cuda_graph_table = counted
    _reset_write_ledger()
    return manager, page_manager, counted


def test_graph_table_refresh_writes_the_rows_pages_not_the_reserved_width():
    rows, pages_per_row = _GRAPH_SLOTS, _PAGES_PER_ROW
    manager, page_manager, table = _make_wide_graph_manager(rows, pages_per_row)
    seq_ids = list(range(100, 100 + rows))
    tokens = [pages_per_row * 64] * rows
    manager.allocate_pages_for_sequences(seq_ids, tokens)

    manager.rebuild_page_table(seq_ids)
    first = _written_elements()
    assert first == rows * pages_per_row
    # The point of the fix: the refresh must not scale with the reserved width.
    assert first < rows * _GRAPH_WIDTH // 100

    # A decode step that extends every row by one page writes that step's
    # pages and nothing else -- no clear is owed, because a row only grows.
    for seq_id in seq_ids:
        manager.extend_pages_for_sequence(seq_id, (pages_per_row + 1) * 64)
    _reset_write_ledger()
    manager.rebuild_page_table(seq_ids)
    assert _written_elements() == rows * (pages_per_row + 1)


def test_graph_table_refresh_still_clears_what_a_shorter_occupant_inherits():
    """O(context) must not mean "stale pages survive a slot handover"."""
    rows, pages_per_row = 4, 16
    manager, page_manager, table = _make_wide_graph_manager(rows, pages_per_row)
    long_ids = list(range(10, 10 + rows))
    manager.allocate_pages_for_sequences(long_ids, [pages_per_row * 64] * rows)
    manager.rebuild_page_table(long_ids)

    short_ids = list(range(50, 50 + rows))
    manager.allocate_pages_for_sequences(short_ids, [3 * 64] * rows)
    _reset_write_ledger()
    manager.rebuild_page_table(short_ids)

    # 3 pages written + 13 columns cleared per row: the shorter occupant's
    # tail, never the 16384-column width.
    assert _written_elements() == rows * pages_per_row
    for slot, seq_id in enumerate(short_ids):
        torch.testing.assert_close(table[slot, :3], manager._sequences[seq_id].pages)
        assert torch.all(table[slot, 3:] == -1), slot


def test_graph_table_refresh_clears_slots_the_batch_gave_up():
    rows, pages_per_row = 4, 16
    manager, page_manager, table = _make_wide_graph_manager(rows, pages_per_row)
    seq_ids = list(range(10, 10 + rows))
    manager.allocate_pages_for_sequences(seq_ids, [pages_per_row * 64] * rows)
    manager.rebuild_page_table(seq_ids)

    _reset_write_ledger()
    manager.rebuild_page_table(seq_ids[:2])
    # Two surviving rows rewritten, two retired rows cleared to -1.
    assert _written_elements() == rows * pages_per_row
    assert torch.all(table[2:, :] == -1)
    # A later batch that reclaims those slots inherits no stale page id.
    manager.rebuild_page_table(seq_ids)
    for slot, seq_id in enumerate(seq_ids):
        torch.testing.assert_close(
            table[slot, :pages_per_row], manager._sequences[seq_id].pages
        )
        assert torch.all(table[slot, pages_per_row:] == -1), slot


def test_clear_page_table_resets_without_touching_unused_columns():
    rows, pages_per_row = 4, 16
    manager, page_manager, table = _make_wide_graph_manager(rows, pages_per_row)
    seq_ids = list(range(10, 10 + rows))
    manager.allocate_pages_for_sequences(seq_ids, [pages_per_row * 64] * rows)
    manager.rebuild_page_table(seq_ids)

    _reset_write_ledger()
    manager.clear_page_table()
    assert _written_elements() == rows * pages_per_row
    assert torch.all(table == -1)
    assert page_manager._cuda_graph_slot_count == 0
