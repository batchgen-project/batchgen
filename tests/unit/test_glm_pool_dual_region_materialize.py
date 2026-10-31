"""GLM-5.x (DSA) cross-wave pool load: both Host regions, one page list.

A DSA model keeps a token's MLA latent in the primary Host region (group 0)
and its indexer keys in the auxiliary one (group 1). A held / committed prefix
loaded into the in-wave prefill pool must bring BOTH, from the same pages in
the same order, or the pool's indexer scores stale slots (wrong top-k, no
error). These CPU tests hold the bookkeeping to that:

* region page-id alignment -- both regions read the same page ids, in order,
  and a disagreement (other ids, other order, other count, missing or extra
  group) is refused by name before any byte moves;
* accounting -- slab size, staging bytes, tokens loaded, one H2D per region
  per slab and both waited on before the copy;
* the real H2D pointer arithmetic, exercised byte-for-byte on CPU memory: the
  fake Host view copies each named Host page to the device pointer the loader
  computed, and the pool rows must then equal the Host pages exactly;
* GLM attention's ``materialize_pool_segment`` (the real method, compiled
  from ``glm5/wrappers.py``) copies latent and indexer rows into the slots a
  computed segment's forward writes, and refuses a latent-only load;
* the resume helper turns a held session into a page-aligned pool segment;
* GPT-OSS / single-region behaviour is unchanged.

The GPU validation is not a unit test; the note at the end says where each
of its assertions is checked.
"""

from __future__ import annotations

import ast
import copy
import ctypes
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_k3_prefix_pool_group import _bind, _worker_methods  # noqa: E402

from batchgen.models.wrappers.attention import AttnWrapperBase  # noqa: E402
from batchgen.prefix_reuse.executor import PoolChunkExecutor  # noqa: E402
from batchgen.prefix_reuse.pool import PrefillPrefixPool  # noqa: E402
from batchgen.prefix_reuse.pool_materialize import (  # noqa: E402
    DualRegionPageMismatch,
    DualRegionSegmentLoader,
    MaterializedSegmentLoader,
    RegionGeometry,
    aligned_region_pages,
)
from batchgen.models.glm.glm5.pool_segment import (  # noqa: E402
    copy_dsa_pool_segment,
)
from batchgen.prefix_reuse.wave_plan import (  # noqa: E402
    MaterializedPrefix,
    plan_wave_prefix_sharing,
)
from batchgen.prefix_reuse.wave_runtime import PoolWave  # noqa: E402
from batchgen.worker import session_hold as session_hold_lib  # noqa: E402

PAGE_TOKENS = 4
LAYERS = 3  # engine layers; GLM pools one latent row per engine layer
LATENT = 8  # stands in for 576
INDEXER = 4  # stands in for 128
DTYPE = torch.float32
HOST_PAGES = 16
HELD = list(range(500, 500 + 3 * PAGE_TOKENS))
HIT_PROMPT = HELD + [901, 902, 903]
SHARED = list(range(700, 700 + 2 * PAGE_TOKENS))
PROMPTS = [HIT_PROMPT, SHARED + [1, 2, 3], SHARED + [4, 5, 6]]
GLOBAL_IDS = [100, 101, 102]
BOUNDARY = 3 * PAGE_TOKENS
# Deliberately unordered: the load must keep token order, not sort the ids.
HELD_PAGES = [9, 3, 12]

WORKER_METHODS = (
    "_materialized_pool_layers",
    "_materialized_segment_loader",
    "_dual_region_segment_loader",
    "_materialized_dual_region_layers",
    "_load_dual_region_segments",
    "_load_materialized_segments",
)


# --------------------------------------------------------------------------- #
#  The real GLM-5 wrapper method, compiled without importing the CUDA stack
# --------------------------------------------------------------------------- #


def _glm_materialize_method():
    tree = ast.parse((ROOT / "batchgen/models/glm/glm5/wrappers.py").read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GLM5AttnWrapper"
    )
    method = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "materialize_pool_segment"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[copy.deepcopy(method)], type_ignores=[])
    )
    namespace = {"AttnWrapperBase": AttnWrapperBase, "__package__": "batchgen.models.glm.glm5"}
    exec(compile(module, "glm5/wrappers.py", "exec"), namespace)
    return namespace["materialize_pool_segment"]


GLM_MATERIALIZE = _glm_materialize_method()


class _GLMLayer:
    """A GLM-5 attention wrapper reduced to what the seam reads."""

    def __init__(self, layer_idx, *, has_indexer):
        self.layer_idx = layer_idx
        self.module = SimpleNamespace(
            indexer=object() if has_indexer else None
        )
        self.calls = []

    def materialize_pool_segment(self, **kwargs):
        self.calls.append(
            (kwargs["slots"].clone(), kwargs.get("indexer") is not None)
        )
        return GLM_MATERIALIZE(self, **kwargs)


# --------------------------------------------------------------------------- #
#  Host regions and a Host view whose H2D really copies bytes
# --------------------------------------------------------------------------- #


def _host_region(layers, dim, *, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(
        (layers, HOST_PAGES, PAGE_TOKENS, dim), generator=generator, dtype=DTYPE
    )


class _Task:
    def __init__(self, view):
        self._view = view

    def wait(self):
        self._view.waits += 1


class _CopyingHostView:
    """``async_load_prefix_pages_to_device`` over a CPU 'Host' tensor.

    Copies page ``host_page_ids[0][j]`` of physical layer ``l`` to
    ``k_device_ptrs[l][0][j]`` with memmove: the loader's pointer arithmetic
    is exercised for real, against real (CPU) staging memory.
    """

    def __init__(self, host, *, is_page_split=False, fail=False):
        self.host = host
        self.is_page_split = is_page_split
        self.fail = fail
        self.loads = []
        self.waits = 0

    def async_load_prefix_pages_to_device(
        self, *, host_page_ids, active_page_counts, k_device_ptrs, v_device_ptrs
    ):
        if self.fail:
            raise RuntimeError("H2D launch failed")
        assert v_device_ptrs is None
        count = int(active_page_counts[0])
        pages = [int(page) for page in host_page_ids[0].tolist()[:count]]
        assert k_device_ptrs.shape == (self.host.shape[0], 1, count)
        page_bytes = self.host[0, 0].numel() * self.host.element_size()
        for layer in range(self.host.shape[0]):
            for slot, page in enumerate(pages):
                source = self.host[layer, page].contiguous()
                ctypes.memmove(
                    int(k_device_ptrs[layer, 0, slot]), source.data_ptr(),
                    page_bytes,
                )
        self.loads.append(tuple(pages))
        return _Task(self)


# --------------------------------------------------------------------------- #
#  Pool, wave and worker
# --------------------------------------------------------------------------- #


def _pool(indexer_layers=(0, 2), num_pages=32):
    pool = PrefillPrefixPool(
        num_layers=LAYERS,
        num_pages=num_pages,
        page_tokens=PAGE_TOKENS,
        num_kv_heads=1,
        head_dim=LATENT,
        dtype=DTYPE,
        device="cpu",
        with_v=False,
        indexer_layers=indexer_layers,
        indexer_dim=INDEXER,
    )
    pool.k.zero_()
    if pool.indexer is not None:
        pool.indexer.zero_()
    return pool


def _wave(pool, pages_by_group=None):
    plan = plan_wave_prefix_sharing(
        PROMPTS,
        block_tokens=PAGE_TOKENS,
        pool_pages=24,
        chunk_tokens=64,
        same_chunk_ancestors=True,
        materialized=[MaterializedPrefix(index=0, tokens=BOUNDARY)],
    )
    wave = PoolWave(
        plan=plan,
        prompts=[list(prompt) for prompt in PROMPTS],
        uuids=["held", "a", "b"],
        global_ids=list(GLOBAL_IDS),
        page_tokens=PAGE_TOKENS,
        total_pages=pool.num_pages,
        pool=pool,
        executor=PoolChunkExecutor(plan, PROMPTS, pool),
    )
    for segment_id in sorted(wave.materialized):
        wave.bind_materialized_host_pages(
            segment_id,
            pages_by_group or {0: HELD_PAGES, 1: HELD_PAGES},
        )
    wave.segment_sids = {
        seg.segment_id: -11 - position
        for position, seg in enumerate(wave.computed_segments())
    }
    return wave


def _geometry(*, aux_layers=LAYERS, aux_dtype=DTYPE, aux_map=None, aux=True):
    primary = SimpleNamespace(
        num_layers=LAYERS, num_k_heads=1, k_head_dim=LATENT, num_v_heads=0,
        kv_dtype=DTYPE, logical_to_physical_layer=None,
    )
    secondary = SimpleNamespace(
        num_layers=aux_layers, num_k_heads=1, k_head_dim=INDEXER,
        num_v_heads=0, kv_dtype=aux_dtype, logical_to_physical_layer=aux_map,
    ) if aux else None
    return SimpleNamespace(
        build_gpu_kv_config=lambda *a, **k: primary,
        build_gpu_kv_config_aux=lambda *a, **k: secondary,
    )


@pytest.fixture(autouse=True)
def _clear_class_state():
    yield
    AttnWrapperBase.prefill_pool = None
    AttnWrapperBase.prefill_pool_batch = None
    AttnWrapperBase.prefill_pool_layer_map = None


def _worker(monkeypatch, pool, views, *, indexer_layers=(0, 2), **geometry):
    monkeypatch.setitem(
        sys.modules,
        "batchgen.kv_cache.host_kv_mananger_config",
        _geometry(**geometry),
    )
    layers = [
        SimpleNamespace(
            self_attn=_GLMLayer(index, has_indexer=index in indexer_layers)
        )
        for index in range(LAYERS)
    ]
    worker = SimpleNamespace(
        rank=0,
        torch_device="cpu",
        huggingface_ckpt_name="zai-org/GLM-5.3-FP8",
        model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
        _host_views_by_group=lambda: views,
    )
    return _bind(worker, _worker_methods(WORKER_METHODS, {"logging": logging}))


def _regions(aux_layers=LAYERS):
    return (
        _host_region(LAYERS, LATENT, seed=1),
        _host_region(aux_layers, INDEXER, seed=2),
    )


def _load(monkeypatch, *, indexer_layers=(0, 2), views=None, wave_pages=None,
          aux_layers=LAYERS, aux_map=None):
    latent_host, indexer_host = _regions(aux_layers)
    views = views or {
        0: _CopyingHostView(latent_host), 1: _CopyingHostView(indexer_host),
    }
    pool = _pool(indexer_layers)
    wave = _wave(pool, wave_pages)
    worker = _worker(
        monkeypatch, pool, views, indexer_layers=indexer_layers,
        aux_layers=aux_layers, aux_map=aux_map,
    )
    loader = worker._materialized_segment_loader(wave)
    rows = wave.executor.build(wave.plan.chunks[0])
    AttnWrapperBase.prefill_pool = pool
    AttnWrapperBase.prefill_pool_batch = rows
    loaded = worker._load_materialized_segments(wave, rows, loader)
    return SimpleNamespace(
        worker=worker, pool=pool, wave=wave, rows=rows, loader=loader,
        loaded=loaded, views=views, latent_host=latent_host,
        indexer_host=indexer_host,
    )


def _host_rows(host, layer, pages):
    return torch.cat([host[layer, page] for page in pages], dim=0)


# --------------------------------------------------------------------------- #
#  1. Region page-id alignment
# --------------------------------------------------------------------------- #


def test_aligned_regions_return_the_one_page_list_in_token_order():
    pages = aligned_region_pages(
        {0: HELD_PAGES, 1: tuple(HELD_PAGES)}, groups=(0, 1),
        page_tokens=PAGE_TOKENS, tokens=BOUNDARY,
    )
    assert pages == tuple(HELD_PAGES)


@pytest.mark.parametrize(
    "pages_by_group, match",
    [
        ({0: [9, 3, 12], 1: [9, 3, 13]}, "page 2 is 12 in group 0 and 13"),
        ({0: [9, 3, 12], 1: [3, 9, 12]}, "page 0 is 9 in group 0 and 3"),
        ({0: [9, 3, 12], 1: [9, 3]}, "3 page\\(s\\) against 2"),
        ({0: [9, 3, 12]}, "names Host KV groups \\[0\\]"),
        ({0: [9, 3, 12], 1: [9, 3, 12], 2: [9, 3, 12]}, "groups \\[0, 1, 2\\]"),
    ],
)
def test_mismatched_regions_are_refused_by_name(pages_by_group, match):
    with pytest.raises(DualRegionPageMismatch, match=match):
        aligned_region_pages(
            pages_by_group, groups=(0, 1), page_tokens=PAGE_TOKENS,
            tokens=BOUNDARY,
        )


def test_a_page_count_that_does_not_cover_the_tokens_is_refused():
    with pytest.raises(ValueError, match="2 Host page\\(s\\) were named"):
        aligned_region_pages(
            {0: [1, 2], 1: [1, 2]}, groups=(0, 1), page_tokens=PAGE_TOKENS,
            tokens=BOUNDARY,
        )


def test_a_mismatched_wave_is_refused_before_any_host_read(monkeypatch):
    latent_host, indexer_host = _regions()
    views = {0: _CopyingHostView(latent_host), 1: _CopyingHostView(indexer_host)}
    with pytest.raises(DualRegionPageMismatch):
        _load(monkeypatch, views=views,
              wave_pages={0: HELD_PAGES, 1: [9, 3, 13]})
    assert views[0].loads == [] and views[1].loads == []


def test_a_latent_only_segment_is_refused_in_a_dual_region_wave(monkeypatch):
    latent_host, indexer_host = _regions()
    views = {0: _CopyingHostView(latent_host), 1: _CopyingHostView(indexer_host)}
    with pytest.raises(DualRegionPageMismatch, match="groups \\[0\\]"):
        _load(monkeypatch, views=views, wave_pages={0: HELD_PAGES})
    assert views[0].loads == []


# --------------------------------------------------------------------------- #
#  2. Accounting
# --------------------------------------------------------------------------- #


def test_the_slab_is_sized_over_both_regions():
    page_bytes = (
        LAYERS * PAGE_TOKENS * LATENT + LAYERS * PAGE_TOKENS * INDEXER
    ) * 4
    loader = DualRegionSegmentLoader(
        primary_view=object(), aux_view=object(), page_tokens=PAGE_TOKENS,
        primary=RegionGeometry(LAYERS, LATENT, DTYPE),
        aux=RegionGeometry(LAYERS, INDEXER, DTYPE),
        device="cpu", max_slab_bytes=2 * page_bytes + 1,
    )
    assert loader.page_bytes_all_layers == page_bytes
    assert loader.slab_pages == 2
    assert loader.staging_bytes == 2 * page_bytes
    with pytest.raises(ValueError, match="holds no page"):
        DualRegionSegmentLoader(
            primary_view=object(), aux_view=object(), page_tokens=PAGE_TOKENS,
            primary=RegionGeometry(LAYERS, LATENT, DTYPE),
            aux=RegionGeometry(LAYERS, INDEXER, DTYPE),
            device="cpu", max_slab_bytes=page_bytes - 1,
        )


def test_glm53_production_geometry_accounting():
    """GLM-5.3: 78 x 576 latent + 78 x 128 indexer bf16 = 109,824 B/token."""
    loader = DualRegionSegmentLoader(
        primary_view=object(), aux_view=object(), page_tokens=64,
        primary=RegionGeometry(78, 576, torch.bfloat16),
        aux=RegionGeometry(78, 128, torch.bfloat16),
        device="meta", max_slab_bytes=16 * 1024 * 1024,
    )
    assert loader.page_bytes_all_layers == 7_028_736 == 64 * 109_824
    assert loader.slab_pages == 2  # 16 MiB // 6.70 MiB


def test_multi_slab_loads_keep_token_order_in_both_regions():
    latent_host, indexer_host = _regions()
    views = (_CopyingHostView(latent_host), _CopyingHostView(indexer_host))
    one_page = (LAYERS * PAGE_TOKENS * (LATENT + INDEXER)) * 4
    loader = DualRegionSegmentLoader(
        primary_view=views[0], aux_view=views[1], page_tokens=PAGE_TOKENS,
        primary=RegionGeometry(LAYERS, LATENT, DTYPE),
        aux=RegionGeometry(LAYERS, INDEXER, DTYPE),
        device="cpu", max_slab_bytes=one_page,
    )
    assert loader.slab_pages == 1
    starts = []
    for slab, page in zip(
        loader.slabs(
            pages_by_group={0: HELD_PAGES, 1: HELD_PAGES}, tokens=BOUNDARY
        ),
        HELD_PAGES,
    ):
        starts.append(slab.token_start)
        for layer in range(LAYERS):
            assert torch.equal(slab.latent[layer], latent_host[layer, page])
            assert torch.equal(slab.indexer[layer], indexer_host[layer, page])
    assert starts == [0, PAGE_TOKENS, 2 * PAGE_TOKENS]
    assert [view.waits for view in views] == [3, 3]


def test_one_h2d_per_region_per_slab_and_both_waited(monkeypatch):
    run = _load(monkeypatch)
    slabs = -(-len(HELD_PAGES) // run.loader.slab_pages)
    for view in run.views.values():
        assert [page for load in view.loads for page in load] == HELD_PAGES
        assert len(view.loads) == slabs
        assert view.waits == slabs
    assert run.loaded == BOUNDARY


def test_every_layer_fills_the_segments_slots_once_in_order(monkeypatch):
    run = _load(monkeypatch)
    stage = run.rows.materialized[0]
    for decoder_layer in run.worker.model.model.layers:
        layer = decoder_layer.self_attn
        covered = [slot for slots, _ in layer.calls for slot in slots.tolist()]
        assert covered == stage.slots.tolist()
        assert all(
            brought == (layer.layer_idx in (0, 2)) for _, brought in layer.calls
        ), "indexer keys go to exactly the layers with an indexer row"


# --------------------------------------------------------------------------- #
#  3. Bytes: both regions land in the pool exactly
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("indexer_layers", [(0, 1, 2), (0, 2)])
def test_both_regions_land_in_the_pool_byte_equal(monkeypatch, indexer_layers):
    """(0, 1, 2) is the GLM-5.3 shape (78-layer indexer profile); (0, 2) has a
    shared layer that keeps no indexer row (the GLM-5.2 schedule's shape)."""
    run = _load(monkeypatch, indexer_layers=indexer_layers)
    stage = run.rows.materialized[0]
    for layer in range(LAYERS):
        pooled = run.pool.k[layer].view(-1, LATENT).index_select(0, stage.slots)
        assert torch.equal(pooled, _host_rows(run.latent_host, layer, HELD_PAGES))
    for layer, row in run.pool.indexer_row.items():
        pooled = run.pool.indexer[row].view(-1, INDEXER).index_select(
            0, stage.slots
        )
        assert torch.equal(
            pooled, _host_rows(run.indexer_host, layer, HELD_PAGES)
        )


def test_the_load_writes_no_slot_outside_the_segment(monkeypatch):
    run = _load(monkeypatch)
    stage = run.rows.materialized[0]
    outside = torch.ones(run.pool.k[0].view(-1, LATENT).shape[0], dtype=torch.bool)
    outside[stage.slots] = False
    for layer in range(LAYERS):
        assert torch.count_nonzero(run.pool.k[layer].view(-1, LATENT)[outside]) == 0
    for row in range(run.pool.indexer.shape[0]):
        flat = run.pool.indexer[row].view(-1, INDEXER)
        assert torch.count_nonzero(flat[outside]) == 0


def test_a_mapped_indexer_region_reads_each_layers_physical_row(monkeypatch):
    """A dense mapped aux region (engine 0 -> 0, engine 2 -> 1)."""
    run = _load(
        monkeypatch, indexer_layers=(0, 2), aux_layers=2, aux_map=(0, -1, 1)
    )
    stage = run.rows.materialized[0]
    for engine_layer, physical in ((0, 0), (2, 1)):
        row = run.pool.indexer_row[engine_layer]
        pooled = run.pool.indexer[row].view(-1, INDEXER).index_select(
            0, stage.slots
        )
        assert torch.equal(
            pooled, _host_rows(run.indexer_host, physical, HELD_PAGES)
        )


def test_an_indexer_layer_the_aux_region_does_not_keep_is_refused(monkeypatch):
    with pytest.raises(RuntimeError, match="no indexer-key Host layer"):
        _load(monkeypatch, indexer_layers=(0, 1, 2), aux_layers=2,
              aux_map=(0, -1, 1))


# --------------------------------------------------------------------------- #
#  4. Loader-level refusals
# --------------------------------------------------------------------------- #


def test_a_page_split_indexer_region_is_refused_by_name(monkeypatch):
    latent_host, indexer_host = _regions()
    views = {
        0: _CopyingHostView(latent_host),
        1: _CopyingHostView(indexer_host, is_page_split=True),
    }
    pool = _pool()
    worker = _worker(monkeypatch, pool, views)
    with pytest.raises(RuntimeError, match="page-split FP8"):
        worker._materialized_segment_loader(_wave(pool))


def test_an_indexer_region_in_another_dtype_is_refused(monkeypatch):
    latent_host, indexer_host = _regions()
    views = {0: _CopyingHostView(latent_host), 1: _CopyingHostView(indexer_host)}
    pool = _pool()
    worker = _worker(monkeypatch, pool, views, aux_dtype=torch.float16)
    with pytest.raises(RuntimeError, match="lossy"):
        worker._materialized_segment_loader(_wave(pool))


def test_a_failed_aux_launch_drains_the_primary_load(monkeypatch):
    latent_host, indexer_host = _regions()
    views = {
        0: _CopyingHostView(latent_host),
        1: _CopyingHostView(indexer_host, fail=True),
    }
    with pytest.raises(RuntimeError, match="H2D launch failed"):
        _load(monkeypatch, views=views)
    assert len(views[0].loads) == 1 and views[0].waits == 1


def test_the_pool_must_hold_one_latent_row_per_engine_layer(monkeypatch):
    latent_host, indexer_host = _regions()
    views = {0: _CopyingHostView(latent_host), 1: _CopyingHostView(indexer_host)}
    pool = _pool()
    wave = _wave(pool)
    worker = _worker(monkeypatch, pool, views)
    worker.model.model.layers.append(
        SimpleNamespace(self_attn=_GLMLayer(LAYERS, has_indexer=False))
    )
    loader = worker._materialized_segment_loader(wave)
    rows = wave.executor.build(wave.plan.chunks[0])
    with pytest.raises(RuntimeError, match="one latent row per engine layer"):
        worker._load_materialized_segments(wave, rows, loader)
    assert views[0].loads == []


# --------------------------------------------------------------------------- #
#  5. GLM attention's materialize_pool_segment
# --------------------------------------------------------------------------- #


def _seam(pool, rows=object()):
    AttnWrapperBase.prefill_pool = pool
    AttnWrapperBase.prefill_pool_batch = rows


def test_glm_materialize_writes_the_slots_a_forward_writes():
    """Same rows as ``latent_flat[pool.slots] = primary_kv`` and
    ``indexer_flat[pool.slots] = indexer_kv[0, :, 0, :]`` in sparse prefill."""
    pool, reference = _pool(), _pool()
    _seam(pool)
    slots = torch.tensor([5, 6, 7, 8, 13])
    latent = torch.randn(5, LATENT)
    keys = torch.randn(5, INDEXER)
    layer = _GLMLayer(2, has_indexer=True)
    assert layer.materialize_pool_segment(
        latent=latent, indexer=keys, slots=slots
    ) == 5
    reference.k[2].view(-1, LATENT)[slots] = latent
    reference.indexer[reference.indexer_row[2]].view(-1, INDEXER)[slots] = keys
    assert torch.equal(pool.k, reference.k)
    assert torch.equal(pool.indexer, reference.indexer)


def test_glm_materialize_refuses_a_latent_only_load_untouched():
    pool = _pool()
    _seam(pool)
    with pytest.raises(RuntimeError, match="stale"):
        _GLMLayer(0, has_indexer=True).materialize_pool_segment(
            latent=torch.ones(2, LATENT), slots=torch.tensor([0, 1])
        )
    assert torch.count_nonzero(pool.k) == 0


def test_glm_materialize_refuses_keys_for_a_layer_without_indexer():
    pool = _pool()
    _seam(pool)
    with pytest.raises(RuntimeError, match="keeps no indexer keys"):
        _GLMLayer(1, has_indexer=False).materialize_pool_segment(
            latent=torch.ones(2, LATENT), indexer=torch.ones(2, INDEXER),
            slots=torch.tensor([0, 1]),
        )
    assert torch.count_nonzero(pool.k) == 0


def test_glm_materialize_refuses_an_indexer_layer_the_pool_has_no_row_for():
    pool = _pool(indexer_layers=(0,))
    _seam(pool)
    with pytest.raises(RuntimeError, match="keeps no indexer row"):
        _GLMLayer(2, has_indexer=True).materialize_pool_segment(
            latent=torch.ones(1, LATENT), indexer=torch.ones(1, INDEXER),
            slots=torch.tensor([0]),
        )


def test_glm_materialize_outside_a_pool_wave_is_refused():
    AttnWrapperBase.prefill_pool = None
    AttnWrapperBase.prefill_pool_batch = None
    with pytest.raises(RuntimeError, match="outside a prefix-pool wave"):
        _GLMLayer(0, has_indexer=True).materialize_pool_segment(
            latent=torch.ones(1, LATENT), indexer=torch.ones(1, INDEXER),
            slots=torch.tensor([0]),
        )


@pytest.mark.parametrize(
    "latent, keys, slots, match",
    [
        (torch.ones(2, LATENT), torch.ones(2, INDEXER), torch.tensor([0]), "2 token"),
        (torch.ones(2, LATENT + 1), torch.ones(2, INDEXER), torch.tensor([0, 1]), "wide"),
        (torch.ones(2, LATENT), torch.ones(2, INDEXER + 1), torch.tensor([0, 1]), "indexer row"),
        (torch.ones(2, LATENT, dtype=torch.float16), torch.ones(2, INDEXER),
         torch.tensor([0, 1]), "wide"),
    ],
)
def test_copy_refuses_shape_and_dtype_mismatch(latent, keys, slots, match):
    pool = _pool()
    with pytest.raises(ValueError, match=match):
        copy_dsa_pool_segment(
            pool, layer=0, latent=latent, indexer=keys, slots=slots,
            has_indexer=True,
        )
    assert torch.count_nonzero(pool.k) == 0


# --------------------------------------------------------------------------- #
#  6. A held session as a pool segment (R1c's input)
# --------------------------------------------------------------------------- #


def _plan(cached, keep, *, page=64):
    return session_hold_lib.ResumePlan(
        session_id="s", lcp_tokens=cached, cached_tokens=cached,
        keep_pages=keep, release_pages=0, drop_reason=None,
    )


def test_a_resume_loads_the_held_span_rounded_down_to_the_page_grid():
    held = {0: (40, 41, 42, 43), 1: (40, 41, 42, 43)}
    tokens, span = session_hold_lib.pool_segment_for_resume(
        plan=_plan(3 * 64 + 17, 4), pages_by_group=held, page_tokens=64,
    )
    assert tokens == 3 * 64
    assert span == {0: (40, 41, 42), 1: (40, 41, 42)}


def test_a_page_aligned_resume_loads_every_kept_page():
    held = {0: (7, 8), 1: (7, 8)}
    tokens, span = session_hold_lib.pool_segment_for_resume(
        plan=_plan(128, 2), pages_by_group=held, page_tokens=64,
    )
    assert tokens == 128 and span == {0: (7, 8), 1: (7, 8)}


def test_a_dropped_resume_loads_nothing():
    plan = session_hold_lib.plan_resume(
        session_id="s", lcp_tokens=10, kv_tokens=300, prompt_tokens=400,
        page_tokens=64,
    )
    assert session_hold_lib.pool_segment_for_resume(
        plan=plan, pages_by_group={0: (1,), 1: (1,)}, page_tokens=64,
    ) is None


def test_a_resume_whose_regions_disagree_is_refused():
    with pytest.raises(DualRegionPageMismatch):
        session_hold_lib.pool_segment_for_resume(
            plan=_plan(128, 2), pages_by_group={0: (7, 8), 1: (7, 9)},
            page_tokens=64,
        )


def test_a_resume_with_a_short_region_is_refused():
    with pytest.raises(session_hold_lib.SessionHoldInvariantError, match="group 1"):
        session_hold_lib.pool_segment_for_resume(
            plan=_plan(128, 2), pages_by_group={0: (7, 8), 1: (7,)},
            page_tokens=64,
        )


def test_a_resume_plan_feeds_a_wave_that_loads_both_regions(monkeypatch):
    """plan_resume -> pool_segment_for_resume -> bind -> load, end to end."""
    plan = session_hold_lib.plan_resume(
        session_id="s", lcp_tokens=BOUNDARY + 2, kv_tokens=BOUNDARY + 2,
        prompt_tokens=len(HIT_PROMPT), page_tokens=PAGE_TOKENS,
    )
    held = {0: tuple(HELD_PAGES) + (14,), 1: tuple(HELD_PAGES) + (14,)}
    tokens, span = session_hold_lib.pool_segment_for_resume(
        plan=plan, pages_by_group=held, page_tokens=PAGE_TOKENS,
    )
    assert tokens == BOUNDARY
    run = _load(monkeypatch, wave_pages=span)
    assert run.loaded == BOUNDARY
    stage = run.rows.materialized[0]
    assert torch.equal(
        run.pool.k[1].view(-1, LATENT).index_select(0, stage.slots),
        _host_rows(run.latent_host, 1, HELD_PAGES),
    )


# --------------------------------------------------------------------------- #
#  7. GPT-OSS / single region: unchanged
# --------------------------------------------------------------------------- #


def test_a_single_region_wave_keeps_the_latent_expansion_loader(monkeypatch):
    """One Host region (GPT-OSS geometry, K3) never reaches the DSA path."""
    monkeypatch.setitem(
        sys.modules,
        "batchgen.kv_cache.host_kv_mananger_config",
        SimpleNamespace(
            build_gpu_kv_config=lambda *a, **k: SimpleNamespace(
                num_layers=36, num_k_heads=8, k_head_dim=64,
                kv_dtype=torch.bfloat16,
            ),
            build_gpu_kv_config_aux=lambda *a, **k: pytest.fail(
                "a single-region model must not resolve an aux geometry"
            ),
        ),
    )
    pool = _pool()
    wave = _wave(pool, {0: HELD_PAGES})
    worker = SimpleNamespace(
        rank=0, torch_device="meta", huggingface_ckpt_name="openai/gpt-oss-120b",
        _host_views_by_group=lambda: {0: object()},
    )
    worker = _bind(worker, _worker_methods(WORKER_METHODS, {"logging": logging}))
    loader = worker._materialized_segment_loader(wave)
    assert type(loader) is MaterializedSegmentLoader


def test_gpt_oss_session_hold_path_is_untouched_by_this_change():
    """The GPT-OSS resume loads held pages straight into GPU paged KV
    (``materialize_gpt_oss_prefixes``), not through the pool; GLM-5.3 is
    the one model that resumes through the pool load."""
    from batchgen.prefix_reuse.config import require_session_hold_model_support

    assert require_session_hold_model_support("openai/gpt-oss-120b") == "gpt_oss"
    assert require_session_hold_model_support("zai-org/GLM-5.3-FP8") == (
        "glm_moe_dsa_5_3"
    )


# --------------------------------------------------------------------------- #
#  8. GPU validation (8xH200, GLM-5.3) -- not unit tests
# --------------------------------------------------------------------------- #
#
# The five CUDA tests that stood here were docstring stubs: each passed the
# CUDA gate and then called pytest.skip unconditionally, so a GPU run
# reported them as skipped and asserted nothing. They are removed; where each
# assertion is checked now:
#
# * turn-2 cached_tokens == (turn1_kv // 64) * 64 across waves, one loaded
#   segment of that many tokens per region, no pool decline -- a live-server
#   probe (two-turn session, then four sessions x two turns); it needs the
#   HTTP server, not a unit test.
# * held_kv_pages == 0 after close, and the restart reclaim frees the held
#   pages of BOTH Host regions -- the same live-server probe.
# * both Host regions of a held turn byte-equal to a recompute's -- the
#   env-gated Host KV dump (batchgen/worker/hostkv_debug_dump.py, wired at
#   the hold retain): BATCHGEN_DEBUG_HOSTKV_DUMP_DIR + _SESSION_PREFIX dump
#   every matching held turn's pages of every region, and
#   ``python -m batchgen.worker.hostkv_debug_dump compare A B`` compares two
#   dumps page by page. Its writer and comparer are CPU-tested in
#   tests/unit/test_hostkv_debug_dump.py; the live compare runs on the node.
# * greedy turn-2 output with the hold == without it -- dropped as a gate: at
#   ~27K tokens the full-recompute path is not itself greedy-reproducible on
#   GLM-5.3 (identical requests diverge at a near-tied position), so token
#   equality cannot decide the hold's correctness.
# * the real CUDA H2D of DualRegionSegmentLoader (and the pool rows after
#   _load_dual_region_segments) byte-equal to the Host pages -- NOT covered by
#   any test yet. The CPU tests above exercise the pointer arithmetic with a
#   memmove fake; the device copy itself is unverified.
