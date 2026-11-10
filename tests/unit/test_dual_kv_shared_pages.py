"""GPU page-key dedup through the DSA dual KV coordinator.

Pure CPU. The identity under test: a prefix several rows read is ONE physical
GPU page in each pool, refcounted, and reclaimed only when the last reader of
THAT pool releases it -- while every page a row WRITES stays private.

The two pools (primary MLA latent KV, auxiliary indexer KV) are backed by two
separate Host KV regions, so their Host page ids are two key spaces. The tests
therefore give the same rows different key values per pool, which is also what
makes the pools' physical page ids legitimately differ.
"""

from __future__ import annotations

import importlib
import logging
import math
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytest
import torch

from tests.worker_method_source import load_worker_methods

ROOT = Path(__file__).resolve().parents[2]


def _load_coordinator_module():
    """Import the dual coordinator without running ``batchgen.kv_cache.__init__``.

    The package ``__init__`` pulls in Triton-backed kernels a CPU-only machine
    does not have; neither the coordinator nor the page allocator needs them.
    Every ``sys.modules`` entry this adds is removed again afterwards.
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
        return importlib.import_module(
            "batchgen.kv_cache.dual_kv_cache_coordinator"
        )
    finally:
        for name in [n for n in sys.modules if n not in saved]:
            del sys.modules[name]
        for name, module in saved.items():
            sys.modules[name] = module
        sys.path[:] = saved_path


KV = _load_coordinator_module()

PAGE_TOKENS = 4
POOL_PAGES = 16

# One row of three logical pages: two pages of a shared book, one private tail.
PRIMARY_KEYS = ([10, 11, 90], [10, 11, 91])
# Same grouping in the indexer pool, different ids -- and key 10, which is
# SHARED in the primary pool, is a private page id here. The pools must not
# read anything into each other's numbers.
AUX_KEYS = ([210, 211, 10], [210, 211, 291])


def _config(num_pages=POOL_PAGES):
    return KV.GPUPagedKVConfig(
        num_layers=2,
        num_pages=num_pages,
        page_size_tokens=PAGE_TOKENS,
        num_k_heads=1,
        k_head_dim=2,
        num_v_heads=1,
        v_head_dim=2,
        kv_dtype=torch.float32,
    )


def _manager(num_pages=POOL_PAGES):
    manager = KV.GPUPagedKVCacheManager(config=_config(num_pages), device="cpu")
    manager.initialize()
    return manager


def _coordinator(primary_pages=POOL_PAGES, aux_pages=POOL_PAGES):
    return KV.DualKVCacheCoordinator(
        _manager(primary_pages), _manager(aux_pages)
    )


def _keys(primary=PRIMARY_KEYS, auxiliary=AUX_KEYS):
    return KV.DualPoolPageKeys.from_rows(primary, auxiliary)


def _allocate(coordinator, keys=None):
    return coordinator.allocate_pages_for_sequences_with_page_keys(
        [101, 102],
        [3 * PAGE_TOKENS, 3 * PAGE_TOKENS],
        keys if keys is not None else _keys(),
    )


def _pages(manager, sequence_id):
    return manager._sequences[sequence_id].pages.tolist()


# --------------------------------------------------------------- 1. dedup


def test_each_pool_maps_one_physical_page_per_shared_key():
    coordinator = _coordinator()
    before_primary = coordinator.primary.get_stats().num_used_pages
    before_aux = coordinator.auxiliary.get_stats().num_used_pages

    _allocate(coordinator)

    for manager, shared_pages in (
        (coordinator.primary, 2),
        (coordinator.auxiliary, 2),
    ):
        first, second = _pages(manager, 101), _pages(manager, 102)
        assert first[:shared_pages] == second[:shared_pages]
        assert first[shared_pages] != second[shared_pages]

    # Six logical pages, four physical ones, in each pool independently.
    assert coordinator.primary.get_stats().num_used_pages == before_primary + 4
    assert coordinator.auxiliary.get_stats().num_used_pages == before_aux + 4
    assert coordinator.has_shared_pages() is True
    # Page COUNTS still mirror, which is what the dual contract needs.
    coordinator.assert_mirrored_state("test", [101, 102])


def test_the_two_pools_are_independent_key_spaces():
    coordinator = _coordinator()
    _allocate(coordinator)

    assert sorted(coordinator.primary._shared_page_key_to_gpu_page) == [
        10, 11, 90, 91
    ]
    assert sorted(coordinator.auxiliary._shared_page_key_to_gpu_page) == [
        10, 210, 211, 291
    ]
    # Key 10 is a shared book page in the primary pool and a private tail page
    # in the indexer pool; one pool's mapping must never answer for the other.
    primary_10 = coordinator.primary._shared_page_key_to_gpu_page[10]
    assert coordinator.primary._gpu_page_refcounts[primary_10] == 2
    aux_10 = coordinator.auxiliary._shared_page_key_to_gpu_page[10]
    assert coordinator.auxiliary._gpu_page_refcounts[aux_10] == 1


def test_page_ids_may_differ_between_pools_once_keys_are_in_play():
    """The relaxation that keyed dedup requires, stated as a test.

    Without keys the pools hand out identical ids and a mismatch is a bug.
    With keys each pool assigns new pages in the order of ITS OWN Host ids, so
    the ids diverge by design -- and nothing addresses the indexer pool through
    the primary table, so only the shape has to mirror.
    """

    coordinator = _coordinator()
    _allocate(coordinator)

    assert _pages(coordinator.primary, 101) != _pages(coordinator.auxiliary, 101)
    assert coordinator._page_ids_must_mirror() is False
    coordinator.assert_mirrored_state("diverged ids are allowed", [101, 102])


def test_releasing_every_shared_page_does_not_re_arm_id_mirroring():
    """The GLM-5.3 lcb crash, as a unit test.

    Rows sharing one chat-template prefix page diverged the two pools' page
    ids; when the last reader released that page both registries went empty.
    A LIVE "is anything shared right now" predicate then re-armed the page-id
    and page-table CONTENT assertions against tables that had already
    diverged, and the next ``free_pages_for_sequences`` raised
    "primary/auxiliary page-table contents mismatch". Releasing the pages does
    not un-diverge the ids, so the relaxation has to be sticky.
    """

    coordinator = _coordinator()
    _allocate(coordinator)
    diverged_primary = _pages(coordinator.primary, 101)
    diverged_aux = _pages(coordinator.auxiliary, 101)
    assert diverged_primary != diverged_aux

    coordinator.free_pages_for_sequences([101])
    coordinator.free_pages_for_sequences([102])
    # Every shared page is gone, so the live read says "nothing shared" ...
    assert coordinator.has_shared_pages() is False
    # ... but the ids it diverged stay diverged, so mirroring stays relaxed.
    assert coordinator._page_ids_must_mirror() is False

    # The sequence that crashed the run: an unrelated, unkeyed row allocated,
    # page-tabled and freed after the registries emptied.
    coordinator.allocate_pages_for_sequences([7], [2 * PAGE_TOKENS])
    coordinator.rebuild_page_table([7])
    coordinator.get_stats()
    coordinator.free_pages_for_sequences([7])
    coordinator.clear_page_table()


def test_a_rolled_back_keyed_attempt_stays_sticky():
    """Conservative on purpose: an attempt arms the flag before any pop.

    A keyed allocation that gets as far as the primary pool leaves that pool's
    free stack reordered by a pop/push its peer never made, so the ids are
    independent from then on even though the attempt failed. Arming before the
    preflight only ever costs an assertion; it can never cost a false crash.
    """

    coordinator = _coordinator(primary_pages=POOL_PAGES, aux_pages=3)
    with pytest.raises(RuntimeError, match="insufficient auxiliary free pages"):
        _allocate(coordinator)
    assert coordinator._page_ids_must_mirror() is False


def test_reset_allocations_restores_strict_mirroring():
    coordinator = _coordinator()
    _allocate(coordinator)
    assert coordinator._page_ids_must_mirror() is False

    coordinator.reset_allocations()

    # Every page is back and both registries are empty: the pools hand out the
    # same ids again, so the strict assertion is meaningful and re-armed.
    assert coordinator._page_ids_must_mirror() is True
    coordinator.allocate_pages_for_sequences([7], [2 * PAGE_TOKENS])
    assert _pages(coordinator.primary, 7) == _pages(coordinator.auxiliary, 7)
    coordinator.auxiliary._sequences[7].pages = torch.tensor(
        [13, 14], dtype=torch.int32
    )
    with pytest.raises(RuntimeError, match="page-vector mismatch"):
        coordinator.assert_mirrored_state("mirror", [7])


def test_destroy_restores_strict_mirroring():
    coordinator = _coordinator()
    _allocate(coordinator)
    assert coordinator._page_ids_must_mirror() is False

    coordinator.destroy()
    assert coordinator._page_ids_must_mirror() is True

    coordinator.initialize()
    coordinator.allocate_pages_for_sequences([7], [2 * PAGE_TOKENS])
    assert _pages(coordinator.primary, 7) == _pages(coordinator.auxiliary, 7)


def test_wrapping_pools_that_already_hold_keyed_pages_starts_relaxed():
    primary, auxiliary = _manager(), _manager()
    primary.allocate_pages_for_sequences_with_page_keys(
        [1, 2], [2 * PAGE_TOKENS] * 2, [[5, 6], [5, 7]]
    )
    auxiliary.allocate_pages_for_sequences_with_page_keys(
        [1, 2], [2 * PAGE_TOKENS] * 2, [[6, 5], [6, 8]]
    )
    coordinator = KV.DualKVCacheCoordinator(primary, auxiliary)
    assert coordinator._page_ids_must_mirror() is False


def test_without_keys_the_pools_must_still_mirror_page_ids():
    coordinator = _coordinator()
    coordinator.allocate_pages_for_sequences([7], [2 * PAGE_TOKENS])
    assert coordinator._page_ids_must_mirror() is True
    assert _pages(coordinator.primary, 7) == _pages(coordinator.auxiliary, 7)

    # A pool that diverges while nothing is keyed is a mirroring bug.
    coordinator.auxiliary._sequences[7].pages = torch.tensor(
        [13, 14], dtype=torch.int32
    )
    with pytest.raises(RuntimeError, match="page-vector mismatch"):
        coordinator.assert_mirrored_state("mirror", [7])


# ------------------------------------------------- 2. release and refcounts


def test_a_shared_page_comes_back_only_at_the_last_reader_in_each_pool():
    coordinator = _coordinator()
    before = coordinator.primary.get_stats().num_used_pages
    _allocate(coordinator)

    coordinator.free_pages_for_sequences([101])
    # Only the released row's private tail page comes back, in both pools.
    assert coordinator.primary.get_stats().num_used_pages == before + 3
    assert coordinator.auxiliary.get_stats().num_used_pages == before + 3
    assert 10 in coordinator.primary._shared_page_key_to_gpu_page
    assert 210 in coordinator.auxiliary._shared_page_key_to_gpu_page

    coordinator.free_pages_for_sequences([102])
    assert coordinator.primary.get_stats().num_used_pages == before
    assert coordinator.auxiliary.get_stats().num_used_pages == before
    for manager in (coordinator.primary, coordinator.auxiliary):
        assert manager._shared_page_key_to_gpu_page == {}
        assert manager._gpu_page_to_shared_key == {}
        assert manager._gpu_page_refcounts == {}
    assert coordinator.has_shared_pages() is False


def test_releasing_a_page_twice_fails_loud_in_the_pool_that_lost_it():
    coordinator = _coordinator()
    _allocate(coordinator)
    shared_page = coordinator.primary._shared_page_key_to_gpu_page[10]
    coordinator.free_pages_for_sequences([101])
    coordinator.free_pages_for_sequences([102])

    with pytest.raises(RuntimeError, match="no live reference"):
        coordinator.primary._release_gpu_pages(
            torch.tensor([shared_page], dtype=torch.int32)
        )
    with pytest.raises(KeyError, match="not allocated"):
        coordinator.free_pages_for_sequences([101])


def test_a_pool_flush_drops_the_shared_page_registry_with_the_pages():
    """``reset_allocations`` returns every page, so no key may survive it.

    A key left mapped here would hand the next keyed allocation a page that is
    also sitting in the free stack for somebody else.
    """

    manager = _manager()
    manager.allocate_pages_for_sequences_with_page_keys(
        [1, 2], [2 * PAGE_TOKENS] * 2, [[5, 6], [5, 7]]
    )
    manager.reset_allocations()

    assert manager._shared_page_key_to_gpu_page == {}
    assert manager._gpu_page_to_shared_key == {}
    assert manager._gpu_page_refcounts == {}
    assert manager.get_stats().num_used_pages == 0
    # And the keys are reusable: the next batch gets fresh physical pages.
    manager.allocate_pages_for_sequences_with_page_keys(
        [3, 4], [2 * PAGE_TOKENS] * 2, [[5, 6], [5, 7]]
    )
    assert manager.get_stats().num_used_pages == 3


# ------------------------------------------------------- 3. atomicity


def test_an_auxiliary_capacity_failure_allocates_in_neither_pool():
    coordinator = _coordinator(primary_pages=POOL_PAGES, aux_pages=3)
    before_primary = coordinator.primary.get_stats()

    with pytest.raises(RuntimeError, match="insufficient auxiliary free pages"):
        _allocate(coordinator)

    assert coordinator.primary.get_stats() == before_primary
    assert coordinator.primary._sequences == {}
    assert coordinator.auxiliary._sequences == {}
    assert coordinator.primary._shared_page_key_to_gpu_page == {}


def test_an_auxiliary_allocation_error_rolls_the_primary_pool_back():
    coordinator = _coordinator()
    before = coordinator.primary.get_stats()

    def _boom(*args, **kwargs):
        raise RuntimeError("auxiliary pool exploded")

    coordinator.auxiliary.allocate_pages_for_sequences_with_page_keys = _boom
    with pytest.raises(RuntimeError, match="auxiliary pool exploded"):
        _allocate(coordinator)

    assert coordinator.primary.get_stats() == before
    assert coordinator.primary._sequences == {}
    assert coordinator.primary._shared_page_key_to_gpu_page == {}
    assert coordinator.primary._gpu_page_refcounts == {}


def test_a_rollback_keeps_a_page_a_resident_row_still_reads():
    """The rollback is refcounted, so it cannot evict a sibling's prefix."""

    coordinator = _coordinator()
    coordinator.allocate_pages_for_sequences_with_page_keys(
        [55], [2 * PAGE_TOKENS], _keys(([10, 11],), ([210, 211],))
    )
    resident_primary = _pages(coordinator.primary, 55)
    resident_aux = _pages(coordinator.auxiliary, 55)

    def _boom(*args, **kwargs):
        raise RuntimeError("auxiliary pool exploded")

    coordinator.auxiliary.allocate_pages_for_sequences_with_page_keys = _boom
    with pytest.raises(RuntimeError, match="auxiliary pool exploded"):
        _allocate(coordinator)

    assert _pages(coordinator.primary, 55) == resident_primary
    assert _pages(coordinator.auxiliary, 55) == resident_aux
    assert coordinator.primary._shared_page_key_to_gpu_page[10] == resident_primary[0]


def test_a_plain_rollback_also_gives_the_refcount_back():
    """The non-keyed rollback must release, not push pages straight back.

    A page returned to the free stack while its refcount still stands would be
    handed out a second time, and a page a resident row still reads must not be
    reclaimed at all.
    """

    coordinator = _coordinator()
    before = coordinator.primary.get_stats()

    def _boom(*args, **kwargs):
        raise RuntimeError("auxiliary pool exploded")

    coordinator.auxiliary.allocate_pages_for_sequences = _boom
    with pytest.raises(RuntimeError, match="auxiliary pool exploded"):
        coordinator.allocate_pages_for_sequences([7], [2 * PAGE_TOKENS])

    assert coordinator.primary.get_stats() == before
    assert coordinator.primary._sequences == {}
    assert coordinator.primary._gpu_page_refcounts == {}


def _pool_state(manager):
    free = manager._free_pages
    return (
        free._buffer[: free.size].tolist(),
        {seq_id: _pages(manager, seq_id) for seq_id in manager._sequences},
        dict(manager._gpu_page_refcounts),
    )


# Three rows of 2, 3 and 4 pages. A pool of SMALL_POOL_PAGES fits the first
# rows of either call and fails a later one: allocate fails at row 2 (5 pages
# taken, 1 free); grow, behind three resident one-page rows, fails at row 1.
ROWS = [7, 8, 9]
ROW_PAGES = [2, 3, 4]
SMALL_POOL_PAGES = 6


def _allocate_rows(coordinator):
    coordinator.allocate_pages_for_sequences(
        ROWS, [pages * PAGE_TOKENS for pages in ROW_PAGES]
    )


def _grow_rows(coordinator):
    coordinator.grow_pages_for_sequences(ROWS, ROW_PAGES)


@pytest.mark.parametrize("failing_pool", ["primary", "auxiliary"])
@pytest.mark.parametrize(
    "operation, resident_rows",
    [(_allocate_rows, False), (_grow_rows, True)],
    ids=["allocate", "grow"],
)
def test_a_failed_multi_row_call_leaves_both_pools_as_they_were(
    operation, resident_rows, failing_pool
):
    """A mirrored call that fails at row k > 0 of one pool changes neither.

    The coordinator's capacity preflight is bypassed so the pool's own
    per-row check is what fails, after earlier rows of the same call already
    took pages. The pool that raised must have returned those rows itself,
    and the coordinator must have undone the pool that succeeded, both last
    row first, so the next mirrored allocation hands both pools the same ids.
    """

    coordinator = KV.DualKVCacheCoordinator(
        _manager(SMALL_POOL_PAGES if failing_pool == "primary" else POOL_PAGES),
        _manager(SMALL_POOL_PAGES if failing_pool == "auxiliary" else POOL_PAGES),
    )
    if resident_rows:
        coordinator.allocate_pages_for_sequences(ROWS, [PAGE_TOKENS] * len(ROWS))
    before = {
        name: _pool_state(manager)
        for name, manager in (
            ("primary", coordinator.primary),
            ("auxiliary", coordinator.auxiliary),
        )
    }
    coordinator._preflight_allocate = lambda *args, **kwargs: None
    coordinator._preflight_grow = lambda *args, **kwargs: None

    with pytest.raises(RuntimeError, match="(?i)insufficient free pages for seq"):
        operation(coordinator)

    assert _pool_state(coordinator.primary) == before["primary"]
    assert _pool_state(coordinator.auxiliary) == before["auxiliary"]

    coordinator.allocate_pages_for_sequences([20], [PAGE_TOKENS])
    assert _pages(coordinator.primary, 20) == _pages(coordinator.auxiliary, 20)


def test_the_two_pools_must_agree_on_the_shared_page_demand():
    coordinator = _coordinator()
    # The indexer rows claim one more shared page than the primary rows: the
    # pools would stop mirroring page counts, so it is named, not averaged.
    with pytest.raises(RuntimeError, match="shared-page demand mismatch"):
        coordinator.count_new_shared_pages(
            [101, 102],
            [3 * PAGE_TOKENS] * 2,
            _keys(PRIMARY_KEYS, ([210, 211, 290], [210, 211, 290])),
        )


def test_the_demand_preview_matches_what_the_allocation_takes():
    coordinator = _coordinator()
    demand = coordinator.count_new_shared_pages(
        [101, 102], [3 * PAGE_TOKENS] * 2, _keys()
    )
    before = coordinator.primary.get_stats().num_free_pages
    _allocate(coordinator)
    assert demand == 4
    assert before - coordinator.primary.get_stats().num_free_pages == demand
    assert before - coordinator.auxiliary.get_stats().num_free_pages == demand


def test_the_coordinator_refuses_a_single_pool_key_list():
    coordinator = _coordinator()
    with pytest.raises(TypeError, match="DualPoolPageKeys"):
        coordinator.allocate_pages_for_sequences_with_page_keys(
            [101, 102], [3 * PAGE_TOKENS] * 2, list(PRIMARY_KEYS)
        )


def test_the_key_carrier_rejects_rows_the_pools_cannot_mirror():
    with pytest.raises(ValueError, match="row count mismatch"):
        KV.DualPoolPageKeys.from_rows(([1, 2],), ([1, 2], [3, 4]))
    with pytest.raises(ValueError, match="page-key count mismatch"):
        KV.DualPoolPageKeys.from_rows(([1, 2],), ([1, 2, 3],))


# ------------------------------------------- 4. writable pages stay private


def test_a_shared_key_after_a_private_page_is_rejected():
    """Sharing comes from a committed prefix, so it is a LEADING run."""

    manager = _manager()
    with pytest.raises(ValueError, match="leading run"):
        manager.allocate_pages_for_sequences_with_page_keys(
            [1, 2],
            [3 * PAGE_TOKENS] * 2,
            [[1, 2, 3], [9, 2, 8]],
        )
    assert manager._sequences == {}


def test_pages_appended_after_a_keyed_allocation_are_private():
    manager = _manager()
    manager.allocate_pages_for_sequences_with_page_keys(
        [1, 2], [2 * PAGE_TOKENS] * 2, [[5, 6], [5, 7]]
    )
    appended = manager.grow_pages_for_sequences([1], [1])[1]

    assert len(appended) == 1
    page = int(appended[0])
    assert manager._gpu_page_refcounts[page] == 1
    assert page not in manager._gpu_page_to_shared_key


def test_a_prefill_suffix_may_not_be_written_onto_a_shared_page():
    manager = _manager()
    # Both rows map the same single page and nothing else: a write to page 0
    # would land on the sibling's copy.
    manager.allocate_pages_for_sequences_with_page_keys(
        [1, 2], [PAGE_TOKENS] * 2, [[5], [5]]
    )
    with pytest.raises(RuntimeError, match="a shared prefix page is read-only"):
        manager.prepare_prefill_suffix_append(
            sequence_ids=[1], prefix_lens=[0], suffix_lens=[PAGE_TOKENS]
        )


def test_a_prefill_suffix_behind_a_shared_prefix_is_allowed():
    manager = _manager()
    manager.allocate_pages_for_sequences_with_page_keys(
        [1, 2], [2 * PAGE_TOKENS] * 2, [[5, 6], [5, 7]]
    )
    plan = manager.prepare_prefill_suffix_append(
        sequence_ids=[1], prefix_lens=[PAGE_TOKENS], suffix_lens=[PAGE_TOKENS]
    )
    assert plan.total_suffix_tokens == PAGE_TOKENS


# -------------------------------------- 5. the worker's keys and the credit


class _FakeDualHostKVCoordinator:
    """Stands in for the Host-side coordinator (native region, no CPU build)."""

    def __init__(self, primary, auxiliary):
        self.primary = primary
        self.auxiliary = auxiliary

    def views_by_group(self):
        return {0: self.primary, 1: self.auxiliary}


class _HostView:
    def __init__(self, tables):
        self._tables = tables

    def build_page_table(self, sequence_ids):
        return [list(self._tables[int(sid)]) for sid in sequence_ids]


_METHODS = load_worker_methods(
    [
        "_shared_page_identity_sources",
        "_host_identity_page_keys",
        "_pool_host_identity_page_keys",
        "_assert_private_writable_page_keys",
        "_decode_shared_prefix_credit_enabled",
    ],
    {
        "DualKVCacheCoordinator": KV.DualKVCacheCoordinator,
        "DualHostKVCoordinator": _FakeDualHostKVCoordinator,
        "DualPoolPageKeys": KV.DualPoolPageKeys,
        "logging": logging,
        "math": math,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Sequence": Sequence,
    },
)


class _Batch:
    def __init__(self, contexts):
        self._contexts = contexts

    def get_sequence_by_global_idx(self, global_idx):
        context = self._contexts.get(int(global_idx))
        if context is None:
            return None
        return types.SimpleNamespace(current_context_length=context)


class _Worker:
    _shared_page_identity_sources = _METHODS["_shared_page_identity_sources"]
    _host_identity_page_keys = _METHODS["_host_identity_page_keys"]
    _pool_host_identity_page_keys = _METHODS["_pool_host_identity_page_keys"]
    _assert_private_writable_page_keys = _METHODS[
        "_assert_private_writable_page_keys"
    ]
    _decode_shared_prefix_credit_enabled = _METHODS[
        "_decode_shared_prefix_credit_enabled"
    ]

    def __init__(
        self,
        manager,
        *,
        host_view=None,
        contexts=None,
        prefix_cache=True,
        admission=True,
    ):
        self.gpu_paged_kv_cache_manager = manager
        self.enable_prefix_cache = prefix_cache
        self.shared_prefix_admission = admission
        self.host_paged_kv_worker_view = host_view
        self.core_engine = types.SimpleNamespace(
            host_paged_kv_worker_view=(
                host_view.primary
                if isinstance(host_view, _FakeDualHostKVCoordinator)
                else host_view
            )
        )
        self.global_batch = _Batch(contexts or {})


# Two rows of one book: 2 shared pages, 1 private tail page each. The context
# is 2 pages, so the write head is logical page 2 -- the tail.
_CONTEXTS = {101: 2 * PAGE_TOKENS, 102: 2 * PAGE_TOKENS}


def _dual_worker(coordinator, **kwargs):
    host_view = _FakeDualHostKVCoordinator(
        _HostView({101: PRIMARY_KEYS[0], 102: PRIMARY_KEYS[1]}),
        _HostView({101: AUX_KEYS[0], 102: AUX_KEYS[1]}),
    )
    return _Worker(
        coordinator, host_view=host_view, contexts=dict(_CONTEXTS), **kwargs
    )


def test_the_worker_builds_one_key_space_per_pool():
    coordinator = _coordinator()
    worker = _dual_worker(coordinator)

    keys = worker._host_identity_page_keys(
        coordinator, [101, 102], [3 * PAGE_TOKENS] * 2
    )
    assert isinstance(keys, KV.DualPoolPageKeys)
    assert keys.primary == tuple(tuple(row) for row in PRIMARY_KEYS)
    assert keys.auxiliary == tuple(tuple(row) for row in AUX_KEYS)

    # And the coordinator consumes exactly that object.
    coordinator.allocate_pages_for_sequences_with_page_keys(
        [101, 102], [3 * PAGE_TOKENS] * 2, keys
    )
    assert coordinator.has_shared_pages() is True


def test_a_single_manager_still_gets_a_plain_row_list():
    manager = _manager()
    worker = _Worker(
        manager,
        host_view=_HostView({101: PRIMARY_KEYS[0], 102: PRIMARY_KEYS[1]}),
        contexts=dict(_CONTEXTS),
    )
    keys = worker._host_identity_page_keys(
        manager, [101, 102], [3 * PAGE_TOKENS] * 2
    )
    assert keys == [list(PRIMARY_KEYS[0]), list(PRIMARY_KEYS[1])]


def test_a_shared_key_on_the_write_head_is_refused():
    coordinator = _coordinator()
    worker = _dual_worker(coordinator)
    # Both rows' LAST page (the write head at logical page 2) now carries the
    # same Host page id: two rows would write one physical page.
    worker.host_paged_kv_worker_view = _FakeDualHostKVCoordinator(
        _HostView({101: [10, 11, 90], 102: [10, 11, 90]}),
        _HostView({101: AUX_KEYS[0], 102: AUX_KEYS[1]}),
    )
    with pytest.raises(RuntimeError, match="share writable logical page"):
        worker._host_identity_page_keys(
            coordinator, [101, 102], [3 * PAGE_TOKENS] * 2
        )


def test_a_key_already_mapped_on_the_write_head_is_refused():
    coordinator = _coordinator()
    worker = _dual_worker(coordinator)
    _allocate(coordinator)  # 101/102 now hold the book and their tails
    # A third row whose write head reuses a RESIDENT row's page id.
    worker.global_batch = _Batch({103: 2 * PAGE_TOKENS})
    worker.host_paged_kv_worker_view = _FakeDualHostKVCoordinator(
        _HostView({103: [10, 11, 90]}),
        _HostView({103: [210, 211, 292]}),
    )
    with pytest.raises(RuntimeError, match="share writable logical page"):
        worker._host_identity_page_keys(
            coordinator, [103], [3 * PAGE_TOKENS]
        )


def test_a_short_host_table_is_named_with_its_pool():
    coordinator = _coordinator()
    worker = _dual_worker(coordinator)
    worker.host_paged_kv_worker_view = _FakeDualHostKVCoordinator(
        _HostView({101: PRIMARY_KEYS[0], 102: PRIMARY_KEYS[1]}),
        _HostView({101: [210, 211], 102: AUX_KEYS[1]}),
    )
    with pytest.raises(RuntimeError, match="auxiliary pool. is shorter"):
        worker._host_identity_page_keys(
            coordinator, [101, 102], [3 * PAGE_TOKENS] * 2
        )


def test_the_decode_credit_is_live_on_the_dual_path():
    coordinator = _coordinator()
    assert _dual_worker(coordinator)._decode_shared_prefix_credit_enabled() is True


def test_the_decode_credit_follows_the_allocator_and_the_opt_out():
    coordinator = _coordinator()

    # --no-shared-prefix-admission
    assert (
        _dual_worker(coordinator, admission=False)
        ._decode_shared_prefix_credit_enabled()
        is False
    )
    # Prefix cache off: nothing is keyed, so every row loads its own copy.
    assert (
        _dual_worker(coordinator, prefix_cache=False)
        ._decode_shared_prefix_credit_enabled()
        is False
    )
    # A dual GPU coordinator without the dual Host views has no key space.
    half_bound = _Worker(coordinator, host_view=None, contexts=dict(_CONTEXTS))
    assert half_bound._decode_shared_prefix_credit_enabled() is False
    assert half_bound._host_identity_page_keys(
        coordinator, [101], [PAGE_TOKENS]
    ) is None


def test_the_credit_and_the_keys_share_one_condition():
    coordinator = _coordinator()
    for worker in (
        _dual_worker(coordinator),
        _dual_worker(coordinator, prefix_cache=False),
        _Worker(coordinator, host_view=None),
    ):
        has_sources = worker._shared_page_identity_sources(coordinator) is not None
        assert worker._decode_shared_prefix_credit_enabled() is has_sources
