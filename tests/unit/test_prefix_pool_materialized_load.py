"""M3 Slice 3: the worker loading a materialized segment into the pool.

What the driver must get right, and what these tests hold it to:

* the load runs BEFORE the chunk's forward, so every layer's pool row holds
  the committed prefix by the time any row attends over it;
* each layer expands its OWN slab -- the pool row and the Host physical layer
  are one index, asserted rather than assumed;
* the expansion covers exactly the segment's slots, once, in token order;
* the loaded prefix is read from the frontier's resident pages, and a wave
  that spans more than one Host KV region is refused by name;
* the hit's tail offloads from the frontier boundary, never from token 0;
* a wave that materializes nothing behaves exactly as before.

The worker methods are the real ones, compiled onto a fake holding only what
they read (the pattern of ``test_k3_prefix_pool_group``); the model layers and
the Host KV view are fakes at the H2D and projection boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_k3_prefix_pool_group import (  # noqa: E402
    PAGE_TOKENS,
    _bind,
    _worker_methods,
)

from batchgen.models.wrappers.attention import AttnWrapperBase  # noqa: E402
from batchgen.prefix_reuse.executor import PoolChunkExecutor  # noqa: E402
from batchgen.prefix_reuse.pool import PrefillPrefixPool  # noqa: E402
from batchgen.prefix_reuse.wave_plan import (  # noqa: E402
    MaterializedPrefix,
    plan_wave_prefix_sharing,
)
from batchgen.prefix_reuse.wave_runtime import PoolWave  # noqa: E402

LAYERS = 2  # two MLA layers, with a KDA layer between them
LATENT = 8
FRONTIER = list(range(500, 500 + 3 * PAGE_TOKENS))
HIT_PROMPT = FRONTIER + [901, 902, 903]
SHARED = list(range(700, 700 + 2 * PAGE_TOKENS))
PROMPTS = [HIT_PROMPT, SHARED + [1, 2, 3], SHARED + [4, 5, 6]]
GLOBAL_IDS = [100, 101, 102]
BOUNDARY = 3 * PAGE_TOKENS
FRONTIER_PAGES = [71, 72, 73]
# Engine layer -> pool row, with the KDA layer in between holding none.
LAYER_MAP = (0, -1, 1)

METHODS = (
    "_materialized_pool_layers",
    "_materialized_segment_loader",
    "_load_materialized_segments",
)


# --------------------------------------------------------------------------- #
#  Fakes
# --------------------------------------------------------------------------- #


class _FakeTask:
    def __init__(self, view):
        self._view = view

    def wait(self):
        self._view.waits += 1


class _FakeHostView:
    """Writes a per-(layer, page) marker into the staging the loader names."""

    def __init__(self, *, page_contents=None):
        self.waits = 0
        self.loads: list[tuple[tuple[int, ...], int]] = []
        self.page_contents = page_contents or {}

    def async_load_prefix_pages_to_device(
        self, *, host_page_ids, active_page_counts, k_device_ptrs, v_device_ptrs
    ):
        pages = tuple(int(page) for page in host_page_ids[0].tolist())
        count = int(active_page_counts[0])
        self.loads.append((pages[:count], count))
        assert v_device_ptrs is None
        assert k_device_ptrs.shape[1] == 1
        # The pointers address the staging buffer; the fake writes through the
        # one handle a CPU test can hold, which is the tensor the loader
        # yields. The content check below keys on (layer, host page).
        self._pending = (pages[:count], count, k_device_ptrs)
        return _FakeTask(self)


class _FakeMLALayer:
    """Records what its own layer was asked to expand and where."""

    def __init__(self, layer_idx, pool_row, pool):
        self.layer_idx = layer_idx
        self.pool_row = pool_row
        self._pool = pool
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def materialize_pool_segment(self, *, latent, slots):
        self.calls.append((latent.clone(), slots.clone()))
        # A recognisable non-zero marker per (row, slot), so the pool can be
        # checked for "this layer wrote exactly these slots". The real
        # expansion is covered against the forward's own projection in
        # test_prefix_pool_materialized_segment.
        rows = self._pool.k[self.pool_row].view(-1, *self._pool.k.shape[-2:])
        marker = (
            (slots.to(torch.float32) + 1.0).reshape(-1, 1, 1)
            + float(self.pool_row + 1)
        ).expand(int(slots.numel()), *rows.shape[-2:])
        rows.index_copy_(0, slots, marker.contiguous())
        return int(slots.numel())


class _FakeKDALayer:
    """A layer with no materialization contract; the driver must skip it."""


def _model(pool):
    layers = []
    pool_rows = iter(range(LAYERS))
    for engine_layer, row in enumerate(LAYER_MAP):
        if row < 0:
            layers.append(SimpleNamespace(self_attn=_FakeKDALayer()))
            continue
        layers.append(SimpleNamespace(
            self_attn=_FakeMLALayer(engine_layer, next(pool_rows), pool)
        ))
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def _pool(num_pages=32):
    pool = PrefillPrefixPool(
        num_layers=LAYERS,
        num_pages=num_pages,
        page_tokens=PAGE_TOKENS,
        num_kv_heads=2,
        head_dim=6,
        v_head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )
    pool.k.zero_()
    pool.v.zero_()
    return pool


def _wave(pool, *, hits=(MaterializedPrefix(index=0, tokens=BOUNDARY),)):
    plan = plan_wave_prefix_sharing(
        PROMPTS,
        block_tokens=PAGE_TOKENS,
        pool_pages=24,
        chunk_tokens=64,
        same_chunk_ancestors=True,
        materialized=list(hits),
    )
    wave = PoolWave(
        plan=plan,
        prompts=[list(prompt) for prompt in PROMPTS],
        uuids=["hit", "a", "b"],
        global_ids=list(GLOBAL_IDS),
        page_tokens=PAGE_TOKENS,
        total_pages=pool.num_pages,
        pool=pool,
        executor=PoolChunkExecutor(plan, PROMPTS, pool),
    )
    for segment_id in sorted(wave.materialized):
        wave.bind_materialized_host_pages(segment_id, {0: FRONTIER_PAGES})
    # What allocate_segment_host_pages would have given the COMPUTED segments
    # (the leader's pseudo sequences); a loaded one has none by contract.
    wave.segment_sids = {
        seg.segment_id: -11 - position
        for position, seg in enumerate(wave.computed_segments())
    }
    return wave


def _worker(pool, *, views=None, layer_map=LAYER_MAP, geometry=None):
    methods = _worker_methods(METHODS, {"logging": __import__("logging")})
    host_views = {0: _FakeHostView()} if views is None else views
    worker = SimpleNamespace(
        rank=0,
        torch_device="cpu",
        huggingface_ckpt_name="moonshotai/Kimi-K3",
        model=_model(pool),
        _host_views_by_group=lambda: host_views,
    )
    worker = _bind(worker, methods)
    AttnWrapperBase.prefill_pool_layer_map = layer_map
    return worker, host_views


@pytest.fixture(autouse=True)
def _clear_class_state():
    yield
    AttnWrapperBase.prefill_pool_layer_map = None
    AttnWrapperBase.prefill_pool_materialized_segments = ()


def _stub_geometry(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "batchgen.kv_cache.host_kv_mananger_config",
        SimpleNamespace(build_gpu_kv_config=lambda *a, **k: SimpleNamespace(
            num_layers=LAYERS, num_k_heads=1, k_head_dim=LATENT,
            kv_dtype=torch.float32,
        )),
    )


# --------------------------------------------------------------------------- #
#  1. Which layers load, and in what order
# --------------------------------------------------------------------------- #


def test_the_nth_materializing_layer_must_be_pool_row_n(monkeypatch):
    pool = _pool()
    worker, _ = _worker(pool)
    layers = worker._materialized_pool_layers()
    assert [layer.layer_idx for layer in layers] == [0, 2]
    assert [layer.pool_row for layer in layers] == [0, 1]


def test_a_pool_order_that_contradicts_the_layer_order_is_refused(monkeypatch):
    pool = _pool()
    # The second MLA layer claims pool row 0, which row 0 already holds.
    worker, _ = _worker(pool, layer_map=(0, -1, 0))
    with pytest.raises(RuntimeError, match="orders must agree"):
        worker._materialized_pool_layers()


def test_a_wave_without_the_layer_map_is_refused(monkeypatch):
    pool = _pool()
    worker, _ = _worker(pool, layer_map=None)
    with pytest.raises(RuntimeError, match="engine-layer map"):
        worker._materialized_pool_layers()


# --------------------------------------------------------------------------- #
#  2. The loader the wave gets
# --------------------------------------------------------------------------- #


def test_a_wave_that_materializes_nothing_gets_no_loader(monkeypatch):
    _stub_geometry(monkeypatch)
    pool = _pool()
    worker, _ = _worker(pool)
    wave = _wave(pool, hits=())
    assert wave.materialized == frozenset()
    assert worker._materialized_segment_loader(wave) is None


def test_a_wave_that_materializes_gets_staging_for_every_layer(monkeypatch):
    _stub_geometry(monkeypatch)
    pool = _pool()
    worker, _ = _worker(pool)
    loader = worker._materialized_segment_loader(_wave(pool))
    assert loader is not None
    assert loader.slab_pages >= 1
    assert loader.staging_bytes == (
        LAYERS * loader.slab_pages * PAGE_TOKENS * LATENT * 4
    )


def test_a_second_host_kv_region_is_refused_by_name(monkeypatch):
    _stub_geometry(monkeypatch)
    pool = _pool()
    worker, _ = _worker(
        pool, views={0: _FakeHostView(), 1: _FakeHostView()}
    )
    with pytest.raises(RuntimeError, match="exactly one Host KV region"):
        worker._materialized_segment_loader(_wave(pool))


# --------------------------------------------------------------------------- #
#  3. The load itself
# --------------------------------------------------------------------------- #


def _load_chunk(monkeypatch, *, pool=None, wave=None):
    _stub_geometry(monkeypatch)
    pool = pool or _pool()
    wave = wave or _wave(pool)
    worker, views = _worker(pool)
    loader = worker._materialized_segment_loader(wave)
    chunk = wave.plan.chunks[0]
    rows = wave.executor.build(chunk)
    loaded = worker._load_materialized_segments(wave, rows, loader)
    return worker, wave, rows, views[0], loaded


def test_every_layer_expands_its_own_slab_over_the_segments_slots(monkeypatch):
    worker, wave, rows, view, loaded = _load_chunk(monkeypatch)
    stage = rows.materialized[0]
    assert loaded == BOUNDARY
    layers = worker._materialized_pool_layers()
    for layer in layers:
        covered = [slot for _, slots in layer.calls for slot in slots.tolist()]
        assert covered == stage.slots.tolist(), (
            "each layer fills the segment's slots exactly once, in order"
        )
    # Each slab was waited on before any layer read it.
    assert view.waits == len(view.loads)
    assert sum(count for _, count in view.loads) == len(FRONTIER_PAGES)


def test_the_load_reads_the_frontiers_resident_pages(monkeypatch):
    _, _, _, view, _ = _load_chunk(monkeypatch)
    read = [page for pages, _ in view.loads for page in pages]
    assert read == FRONTIER_PAGES


def test_a_segment_with_no_bound_pages_is_refused_by_name(monkeypatch):
    _stub_geometry(monkeypatch)
    pool = _pool()
    wave = _wave(pool)
    worker, _ = _worker(pool)
    loader = worker._materialized_segment_loader(wave)
    rows = wave.executor.build(wave.plan.chunks[0])
    wave.segment_host_pages.pop(0)
    with pytest.raises(RuntimeError, match="no Host page to load from"):
        worker._load_materialized_segments(wave, rows, loader)


def test_a_segment_spanning_two_regions_is_refused_by_name(monkeypatch):
    _stub_geometry(monkeypatch)
    pool = _pool()
    wave = _wave(pool)
    worker, _ = _worker(pool)
    loader = worker._materialized_segment_loader(wave)
    rows = wave.executor.build(wave.plan.chunks[0])
    wave.segment_host_pages[0] = {0: FRONTIER_PAGES, 1: FRONTIER_PAGES}
    with pytest.raises(RuntimeError, match="Host KV regions"):
        worker._load_materialized_segments(wave, rows, loader)


def test_a_chunk_with_loads_and_no_staging_is_refused_by_name(monkeypatch):
    _stub_geometry(monkeypatch)
    pool = _pool()
    wave = _wave(pool)
    worker, _ = _worker(pool)
    rows = wave.executor.build(wave.plan.chunks[0])
    assert rows.materialized
    with pytest.raises(RuntimeError, match="no\\s+staging"):
        worker._load_materialized_segments(wave, rows, None)


def test_a_chunk_without_loads_does_nothing(monkeypatch):
    _stub_geometry(monkeypatch)
    pool = _pool()
    wave = _wave(pool, hits=())
    worker, views = _worker(pool)
    rows = wave.executor.build(wave.plan.chunks[0])
    assert rows.materialized == ()
    assert worker._load_materialized_segments(wave, rows, None) == 0
    assert views[0].loads == []


# --------------------------------------------------------------------------- #
#  4. What the chunk's rows then read
# --------------------------------------------------------------------------- #


def test_the_hits_tail_reads_the_loaded_pages_through_its_page_table(monkeypatch):
    _, wave, rows, _, _ = _load_chunk(monkeypatch)
    stage = rows.materialized[0]
    tail_row = next(
        row for row, item in enumerate(rows.items)
        if item.kind == "tail" and item.ref == 0
    )
    table = rows.page_table[tail_row].tolist()
    assert table[:len(stage.pages)] == list(stage.pages)
    assert int(rows.cache_seqlens[tail_row]) == len(HIT_PROMPT)
    assert rows.items[tail_row].token_start == BOUNDARY


def test_the_loaded_slots_hold_the_loaded_prefix_before_the_forward(monkeypatch):
    """The pool rows the tail will attend over are written by the load."""

    pool = _pool()
    _, _, rows, _, _ = _load_chunk(monkeypatch, pool=pool)
    stage = rows.materialized[0]
    for row in range(LAYERS):
        written = pool.k[row].view(-1, *pool.k.shape[-2:]).index_select(
            0, stage.slots
        )
        assert torch.count_nonzero(written) == written.numel(), (
            f"pool row {row} still holds unwritten slots"
        )


def test_the_hits_tail_offloads_from_the_frontier_boundary(monkeypatch):
    """Never from token 0: that would overwrite the frontier's shared pages."""

    _, wave, rows, _, _ = _load_chunk(monkeypatch)
    ids, starts = wave.row_host_targets(rows)
    for row, item in enumerate(rows.items):
        if item.kind == "tail" and item.ref == 0:
            assert ids[row] == GLOBAL_IDS[0]
            assert starts[row] == BOUNDARY
            break
    else:  # pragma: no cover - the tail is always built
        raise AssertionError("the hit's tail was not built")
    # And the loaded segment is no row, so nothing offloads it at all.
    assert len(ids) == len(rows.items)
    assert all(item.kind != "materialize" for item in rows.items)


def test_the_hits_host_page_table_starts_with_the_frontier(monkeypatch):
    pool = _pool()
    wave = _wave(pool)
    assert wave.chain_host_pages(0) == {0: FRONTIER_PAGES}
    assert wave.chain_tokens(0) == BOUNDARY
