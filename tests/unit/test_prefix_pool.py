import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load(name, *relative):
    path = Path(__file__).parents[2].joinpath(*relative)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_POOL = _load("test_prefix_pool_mod", "batchgen", "prefix_reuse", "pool.py")

PrefillPrefixPool = _POOL.PrefillPrefixPool
pool_pages_from_budget = _POOL.pool_pages_from_budget
scratch_pages_bound = _POOL.scratch_pages_bound

P = 4  # page tokens


def make_pool(num_pages=8, num_layers=2):
    return PrefillPrefixPool(
        num_layers=num_layers,
        num_pages=num_pages,
        page_tokens=P,
        num_kv_heads=1,
        head_dim=2,
        dtype=torch.float32,
        device="cpu",
    )


def test_budget_leaves_room_for_the_workspace():
    assert pool_pages_from_budget(1000, 200, 300) == 2
    assert pool_pages_from_budget(1000, 1000, 300) == 0


def test_budget_rejects_bad_inputs():
    with pytest.raises(ValueError):
        pool_pages_from_budget(1000, 200, 0)
    with pytest.raises(ValueError):
        pool_pages_from_budget(100, 200, 32)


def test_scratch_bound_covers_partial_pages_and_oversized_items():
    # One chunk of 10 tokens over 3 prompts: 3 full pages plus one tail page each.
    assert scratch_pages_bound(10, 8, 3, P) == 6
    # A single item larger than the chunk cap still has to fit.
    assert scratch_pages_bound(10, 21, 3, P) == 9


def test_alloc_takes_the_lowest_free_pages():
    pool = make_pool()
    assert pool.alloc(3) == [0, 1, 2]
    assert pool.alloc(0) == []
    pool.free([1])
    assert pool.alloc(2) == [1, 3]
    assert pool.free_pages == pool.num_pages - 4


def test_page_bytes_counts_k_and_v_over_all_layers():
    assert make_pool(num_layers=2).page_bytes == 2 * 2 * P * 1 * 2 * 4


def test_double_free_raises():
    pool = make_pool()
    pages = pool.alloc(2)
    pool.free(pages)
    with pytest.raises(RuntimeError):
        pool.free(pages)
    with pytest.raises(RuntimeError):
        pool.free([pool.num_pages])


def test_exhaustion_reports_need_and_have():
    pool = make_pool()
    pool.alloc(6)
    with pytest.raises(RuntimeError, match="need 3 pages, have 2"):
        pool.alloc(3)


def test_slots_cross_a_page_boundary_from_an_offset():
    pool = make_pool()
    slots = pool.slots([5, 2], start_offset=2, n_tokens=4)
    assert slots.tolist() == [5 * P + 2, 5 * P + 3, 2 * P, 2 * P + 1]
    assert slots.dtype == torch.long
    with pytest.raises(ValueError):
        pool.slots([5, 2], start_offset=2, n_tokens=7)


def test_release_requires_every_page_back():
    pool = make_pool()
    pages = pool.alloc(2)
    with pytest.raises(RuntimeError):
        pool.release()
    pool.free(pages)
    pool.release()
    assert pool.k is None and pool.v is None
