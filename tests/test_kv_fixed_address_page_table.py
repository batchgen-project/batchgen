"""Fixed-address GPU KV contract: what a decode graph bakes must not move.

The default GLM-5.2 phase switch recaptured every decode CUDA-graph bucket on
every batch, because the pools a captured graph dereferences were cudaMalloc'd
afresh by every ``initialize()`` and the page-table manager itself was rebuilt
inside ``destroy()``. These tests pin the contract that stops both:

* the KV pools are carved once from one reserved virtual range, so a
  destroy/initialize cycle unmaps and maps the same addresses and keeps the
  carved tensors, the page-pointer tables and the page-table manager;
* the slot-index and slot-to-sequence tensors become prefix views of fixed
  storage instead of per-rebuild allocations. This one is hardening: both are
  part of ``get_cuda_graph_state()`` and the Kimi-K2.5 graph adapter reads them
  directly, while the GLM-5 path binds its slot indices as static graph inputs.

Plus the #424 invariant that the KV region stays mapped across a decode
admission / capacity snapshot.

CPU-only. The page-table and guard paths use plain torch tensors; the
fixed-address pool lifecycle runs against a fake arena, since CUDA VMM is the
only production implementation.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_manager_module():
    """Import the KV manager without running ``batchgen.kv_cache.__init__``.

    The package ``__init__`` pulls in Triton-backed kernels that a CPU-only
    machine does not have; the manager itself needs none of them.

    Every ``sys.modules`` entry this adds is removed again afterwards. The
    loaded module keeps direct references to what it imported, so it still
    works, and other test modules in the same session see an unchanged
    ``sys.modules`` — a leaked stub would otherwise silently change whether
    their own imports succeed.
    """
    saved = dict(sys.modules)
    saved_path = list(sys.path)
    try:
        for name, path in (
            ("batchgen", "batchgen"),
            ("batchgen.config", "batchgen/config"),
            ("batchgen.kv_cache", "batchgen/kv_cache"),
        ):
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT / path)]
            sys.modules[name] = module
        stub = types.ModuleType("batchgen.kv_cache.gpu_kv_kernels")
        stub.run_paged_kv_token_update = lambda *a, **k: None
        stub.run_paged_kv_token_update_fused = lambda *a, **k: None
        sys.modules["batchgen.kv_cache.gpu_kv_kernels"] = stub
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        return importlib.import_module("batchgen.kv_cache.gpu_paged_kv_manager")
    finally:
        for name in [n for n in sys.modules if n not in saved]:
            del sys.modules[name]
        for name, module in saved.items():
            sys.modules[name] = module
        sys.path[:] = saved_path


MANAGER = _load_manager_module()


def _config(*, num_pages=64, max_slots=8, max_pages_per_seq=6, v_cache=False):
    return MANAGER.GPUPagedKVConfig(
        num_layers=2,
        num_pages=num_pages,
        page_size_tokens=4,
        num_k_heads=1,
        k_head_dim=8,
        num_v_heads=1 if v_cache else 0,
        v_head_dim=8 if v_cache else 0,
        kv_dtype=torch.float16,
        cuda_graph_max_pages_per_sequence=max_pages_per_seq,
        cuda_graph_max_slots=max_slots,
    )


def _make_manager(**kwargs):
    manager = MANAGER.GPUPagedKVCacheManager(config=_config(**kwargs), device="cpu")
    manager.initialize()
    return manager


def _allocate(manager, sequence_tokens):
    ids = list(sequence_tokens)
    manager.allocate_pages_for_sequences(ids, [sequence_tokens[i] for i in ids])
    return ids


# ---------------------------------------------------------------------------
# Page table / slot views
# ---------------------------------------------------------------------------


def test_graph_state_addresses_survive_two_simulated_switches():
    manager = _make_manager()
    table_mgr = manager._gpu_page_table_manager

    ids = _allocate(manager, {0: 4, 1: 4})
    manager.rebuild_page_table(ids)
    graph_ptr = table_mgr.get_cuda_graph_table().data_ptr()
    slot_ptr = table_mgr._slot_index_tensor.data_ptr()
    seq_ptr = table_mgr._slot_to_seq_id_tensor.data_ptr()

    # Decode -> prefill: every sequence is dropped, buffers are kept in place
    # (what the worker does with keep_buffers=True).
    manager.reset_allocations()

    # Prefill -> decode with a different slot count AND a different page count
    # per sequence: the two things that used to force a reallocation.
    ids = _allocate(manager, {5: 12, 6: 8, 7: 4})
    manager.rebuild_page_table(ids)

    assert table_mgr.get_cuda_graph_table().data_ptr() == graph_ptr
    assert table_mgr._slot_index_tensor.data_ptr() == slot_ptr
    assert table_mgr._slot_to_seq_id_tensor.data_ptr() == seq_ptr

    # And again, with the batch shrinking back down.
    manager.reset_allocations()
    ids = _allocate(manager, {9: 4})
    manager.rebuild_page_table(ids)
    assert table_mgr.get_cuda_graph_table().data_ptr() == graph_ptr
    assert table_mgr._slot_index_tensor.data_ptr() == slot_ptr
    assert table_mgr._slot_to_seq_id_tensor.data_ptr() == seq_ptr


def test_slot_views_hold_exactly_the_values_they_used_to_be_rebuilt_with():
    manager = _make_manager()
    table_mgr = manager._gpu_page_table_manager

    ids = _allocate(manager, {11: 4, 12: 8, 13: 4})
    manager.rebuild_page_table(ids)

    assert table_mgr._slot_index_tensor.dtype is torch.int32
    assert table_mgr._slot_to_seq_id_tensor.dtype is torch.int64
    assert table_mgr._slot_index_tensor.tolist() == [0, 1, 2]
    assert table_mgr._slot_to_seq_id_tensor.tolist() == [11, 12, 13]
    assert table_mgr.get_slot_to_seq_id_tensor(2).tolist() == [11, 12]

    # A narrower batch must not leave the wider batch's ids visible.
    manager.rebuild_page_table([11])
    assert table_mgr._slot_index_tensor.tolist() == [0]
    assert table_mgr._slot_to_seq_id_tensor.tolist() == [11]


def test_clear_page_table_empties_the_slot_views_without_moving_them():
    manager = _make_manager()
    table_mgr = manager._gpu_page_table_manager

    ids = _allocate(manager, {0: 4, 1: 4})
    manager.rebuild_page_table(ids)
    graph_ptr = table_mgr.get_cuda_graph_table().data_ptr()
    slot_storage_ptr = table_mgr._slot_index_storage.data_ptr()
    seq_storage_ptr = table_mgr._slot_to_seq_id_storage.data_ptr()

    manager.clear_page_table()

    assert table_mgr._slot_index_tensor.numel() == 0
    assert table_mgr._slot_to_seq_id_tensor.numel() == 0
    assert table_mgr._slot_index_tensor.dtype is torch.int32
    assert table_mgr._slot_to_seq_id_tensor.dtype is torch.int64
    assert table_mgr._slot_index_storage.data_ptr() == slot_storage_ptr
    assert table_mgr._slot_to_seq_id_storage.data_ptr() == seq_storage_ptr
    assert table_mgr.get_cuda_graph_table().data_ptr() == graph_ptr
    assert table_mgr._cuda_graph_slot_count == 0


def test_batch_past_slot_capacity_falls_back_without_touching_fixed_storage():
    manager = _make_manager(max_slots=2)
    table_mgr = manager._gpu_page_table_manager

    ids = _allocate(manager, {0: 4, 1: 4})
    manager.rebuild_page_table(ids)
    storage_ptr = table_mgr._slot_index_storage.data_ptr()

    # Three slots against a reserved capacity of two: no captured graph can
    # serve this batch, so the views fall back to per-rebuild allocation and
    # the graph table is explicitly invalid rather than half-updated.
    ids = _allocate(manager, {0: 4, 1: 4, 2: 4})
    manager.rebuild_page_table(ids)

    assert table_mgr._slot_index_tensor.tolist() == [0, 1, 2]
    assert table_mgr._slot_to_seq_id_tensor.tolist() == [0, 1, 2]
    assert table_mgr._slot_index_storage.data_ptr() == storage_ptr
    assert table_mgr._cuda_graph_table_valid is False
    with pytest.raises(RuntimeError, match="not valid"):
        table_mgr.get_cuda_graph_table()


# ---------------------------------------------------------------------------
# #424: the KV region stays mapped across a decode admission / capacity read
# ---------------------------------------------------------------------------


def test_destroy_inside_a_capacity_snapshot_is_refused():
    manager = _make_manager()
    _allocate(manager, {0: 4})
    manager.rebuild_page_table([0])

    with manager.capacity_snapshot():
        with pytest.raises(RuntimeError, match="capacity snapshot"):
            manager.destroy()
        # Still usable: the refusal must not have half-torn-down the manager.
        assert manager.is_initialized is True
        assert manager.get_stats().num_total_pages == 64

    # Outside the section it is allowed again.
    manager.destroy()
    assert manager.is_initialized is False


def test_capacity_snapshot_nests_and_releases():
    manager = _make_manager()
    with manager.capacity_snapshot():
        with manager.capacity_snapshot():
            assert manager._capacity_snapshot_depth == 2
        assert manager._capacity_snapshot_depth == 1
    assert manager._capacity_snapshot_depth == 0
    manager.destroy()


def test_capacity_snapshot_refuses_an_unmapped_pool():
    manager = _make_manager()
    manager.destroy()
    with pytest.raises(RuntimeError, match="not mapped"):
        with manager.capacity_snapshot():
            pass


# ---------------------------------------------------------------------------
# Fixed-address pool lifecycle (fake arena)
# ---------------------------------------------------------------------------


class _FakeArena:
    """Carves CPU tensors out of one buffer and records map/unmap order."""

    def __init__(self, nbytes, events):
        self.nbytes = nbytes
        self._events = events
        self._buffer = torch.empty(nbytes, dtype=torch.uint8)
        self._offset = 0
        self.is_mapped = False

    def map(self):
        assert not self.is_mapped, "map() while already mapped"
        self.is_mapped = True
        self._events.append("map")
        # Physical memory is fresh and undefined after a real map().
        self._buffer.fill_(0xAB)

    def unmap(self):
        assert self.is_mapped, "unmap() while not mapped"
        self.is_mapped = False
        self._events.append("unmap")

    def carve(self, shape, dtype):
        assert self.is_mapped, "carve() before map()"
        element_size = torch.empty((), dtype=dtype).element_size()
        count = 1
        for dim in shape:
            count *= int(dim)
        start = (self._offset + 255) // 256 * 256
        end = start + count * element_size
        assert end <= self.nbytes, "fake arena overflow"
        self._offset = end
        self._events.append(("carve", tuple(shape)))
        return self._buffer[start:end].view(dtype).view(tuple(shape))


class _FakePoolManager(MANAGER.GPUPagedKVCacheManager):
    """Fixed-address manager whose arena and device calls are local."""

    def __init__(self, *args, **kwargs):
        self.arena_events = []
        self.arenas = []
        super().__init__(*args, **kwargs)

    def _make_pool_arena(self, nbytes):
        arena = _FakeArena(nbytes, self.arena_events)
        self.arenas.append(arena)
        return arena

    def _set_device(self):
        return None


def _fixed_manager(**kwargs):
    manager = _FakePoolManager(
        config=_config(**kwargs), device="cpu", fixed_address_pool=True
    )
    manager.initialize()
    return manager


def test_fixed_pool_keeps_one_address_across_destroy_and_initialize():
    manager = _fixed_manager(v_cache=True)
    table_mgr = manager._gpu_page_table_manager

    k_ptr = manager._k_cache.data_ptr()
    v_ptr = manager._v_cache.data_ptr()
    graph_ptr = table_mgr._ensure_cuda_graph_table().data_ptr()

    _allocate(manager, {0: 8, 1: 4})
    manager.rebuild_page_table([0, 1])
    assert manager.pool_is_mapped is True

    # Decode -> prefill: physical pages go back, addresses stay reserved.
    manager.destroy()
    assert manager.is_initialized is False
    assert manager.pool_is_mapped is False
    assert manager.arenas[0].is_mapped is False
    # The page-table manager is NOT rebuilt, unlike the allocator-backed path.
    assert manager._gpu_page_table_manager is table_mgr

    # Prefill -> decode.
    manager.initialize()

    assert manager.pool_is_mapped is True
    assert manager._k_cache.data_ptr() == k_ptr
    assert manager._v_cache.data_ptr() == v_ptr
    assert manager._gpu_page_table_manager is table_mgr
    assert table_mgr._ensure_cuda_graph_table().data_ptr() == graph_ptr
    # One arena for the manager's whole life: reserved once, mapped twice.
    assert len(manager.arenas) == 1
    assert [e for e in manager.arena_events if e in ("map", "unmap")] == [
        "map",
        "unmap",
        "map",
    ]
    # The page-pointer table is rebuilt, but from the same pool addresses, so
    # the device pointers it publishes are unchanged.
    assert int(manager._k_page_ptr_table[0, 0]) == manager._k_cache[0].data_ptr()


def test_fixed_pool_is_zeroed_after_every_remap():
    manager = _fixed_manager()
    assert int(manager._k_cache.abs().sum()) == 0

    manager._k_cache.fill_(3.0)
    manager.destroy()
    manager.initialize()

    # map() handed back undefined (0xAB-filled) memory; a freshly initialized
    # manager must still look zeroed to its callers.
    assert int(manager._k_cache.abs().sum()) == 0


def test_fixed_pool_destroy_is_refused_inside_a_capacity_snapshot():
    manager = _fixed_manager()
    _allocate(manager, {0: 4})
    manager.rebuild_page_table([0])

    with manager.capacity_snapshot():
        with pytest.raises(RuntimeError, match="capacity snapshot"):
            manager.destroy()
        assert manager.pool_is_mapped is True
        assert manager.arenas[0].is_mapped is True


def test_fixed_pool_assert_mapped_fails_loud_when_unmapped():
    manager = _fixed_manager()
    manager.destroy()

    with pytest.raises(RuntimeError, match="not mapped"):
        manager.assert_pool_mapped("decode replay")


def test_fixed_pool_requires_an_indexed_cuda_device_in_production():
    manager = MANAGER.GPUPagedKVCacheManager(
        config=_config(), device="cpu", fixed_address_pool=True
    )
    with pytest.raises(RuntimeError, match="indexed CUDA device"):
        manager.initialize()


# ---------------------------------------------------------------------------
# The DSA coordinator mirrors the lifecycle state of both pools
# ---------------------------------------------------------------------------
#
# GLM-5.2 serves through DualKVCacheCoordinator (MLA primary + indexer aux).
# The phase switch asks the bound manager — i.e. the coordinator — whether the
# pool is mapped, holds it mapped for a capacity snapshot, and releases it, so
# the coordinator must answer for both pools and refuse to answer at all if
# they ever disagree.


def _load_coordinator_module():
    saved = dict(sys.modules)
    saved_path = list(sys.path)
    try:
        for name, path in (
            ("batchgen", "batchgen"),
            ("batchgen.config", "batchgen/config"),
            ("batchgen.kv_cache", "batchgen/kv_cache"),
        ):
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT / path)]
            sys.modules[name] = module
        stub = types.ModuleType("batchgen.kv_cache.gpu_kv_kernels")
        stub.run_paged_kv_token_update = lambda *a, **k: None
        stub.run_paged_kv_token_update_fused = lambda *a, **k: None
        sys.modules["batchgen.kv_cache.gpu_kv_kernels"] = stub
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        return importlib.import_module("batchgen.kv_cache.dual_kv_cache_coordinator")
    finally:
        for name in [n for n in sys.modules if n not in saved]:
            del sys.modules[name]
        for name, module in saved.items():
            sys.modules[name] = module
        sys.path[:] = saved_path


COORDINATOR = _load_coordinator_module()


class _StubPool:
    """Only the lifecycle surface the coordinator delegates."""

    def __init__(self, *, mapped=True, fixed=True):
        self.config = types.SimpleNamespace(page_size_tokens=64)
        self.pool_is_mapped = mapped
        self.uses_fixed_address_pool = fixed
        self.depth = 0

    def assert_pool_mapped(self, context):
        if not self.pool_is_mapped:
            raise RuntimeError(f"{context}: GPU KV pool is not mapped")

    def capacity_snapshot(self):
        pool = self

        class _Section:
            def __enter__(self_inner):
                pool.assert_pool_mapped("capacity_snapshot")
                pool.depth += 1
                return pool

            def __exit__(self_inner, *exc):
                pool.depth -= 1
                return False

        return _Section()


def _coordinator(**kwargs):
    primary = _StubPool(**kwargs)
    auxiliary = _StubPool(**kwargs)
    return COORDINATOR.DualKVCacheCoordinator(primary, auxiliary), primary, auxiliary


def test_coordinator_reports_the_shared_pool_state():
    mapped, _, _ = _coordinator(mapped=True)
    assert mapped.pool_is_mapped is True
    assert mapped.uses_fixed_address_pool is True

    released, _, _ = _coordinator(mapped=False)
    assert released.pool_is_mapped is False


def test_coordinator_refuses_to_answer_for_diverged_pools():
    coordinator, _, auxiliary = _coordinator(mapped=True)
    auxiliary.pool_is_mapped = False

    with pytest.raises(RuntimeError, match="diverged in mapped state"):
        coordinator.pool_is_mapped

    coordinator, _, auxiliary = _coordinator(fixed=True)
    auxiliary.uses_fixed_address_pool = False

    with pytest.raises(RuntimeError, match="disagree on fixed addressing"):
        coordinator.uses_fixed_address_pool


def test_coordinator_capacity_snapshot_holds_both_pools():
    coordinator, primary, auxiliary = _coordinator()

    with coordinator.capacity_snapshot():
        assert (primary.depth, auxiliary.depth) == (1, 1)

    assert (primary.depth, auxiliary.depth) == (0, 0)


def test_coordinator_capacity_snapshot_refuses_an_unmapped_pool():
    coordinator, _, auxiliary = _coordinator()
    auxiliary.pool_is_mapped = False

    with pytest.raises(RuntimeError, match="not mapped"):
        with coordinator.capacity_snapshot():
            pass


def test_coordinator_assert_pool_mapped_names_the_failing_side():
    coordinator, _, auxiliary = _coordinator()
    auxiliary.pool_is_mapped = False

    with pytest.raises(RuntimeError, match=r"auxiliary"):
        coordinator.assert_pool_mapped("decode replay")
