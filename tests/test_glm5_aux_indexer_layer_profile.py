"""GLM-5.2 aux (DSA indexer) KV pools hold only the 21 indexer layers.

Pure-CPU contract tests: the mapped `glm5_2_indexer` profile must agree with
the model's real skip-topk schedule, route only the GLM-5.2 aliases, and
thread its logical->physical map into the aux GPU KV config.
"""

import types

import pytest

pytest.importorskip("triton")  # host-KV config imports the triton-backed GPU manager

from batchgen.kv_cache.host_kv_mananger_config import (  # noqa: E402
    _GLM5_2_INDEXER_LAYER_MAP,
    _GLM5_2_INDEXER_LOGICAL_LAYERS,
    _resolve_indexer_profile,
    build_gpu_kv_config_aux,
)
from batchgen.models.glm.glm5.dsa_schedule import dsa_layer_skips_topk  # noqa: E402


def test_layer_map_matches_dsa_schedule():
    # GLM-5.2 ships index_topk_freq=4 / index_skip_topk_offset=3; the runtime
    # capture logs confirm 21 full-indexer + 57 reuse layers.
    config = types.SimpleNamespace(index_topk_freq=4, index_skip_topk_offset=3)
    schedule_full_layers = tuple(
        layer for layer in range(78) if not dsa_layer_skips_topk(config, layer)
    )
    assert schedule_full_layers == _GLM5_2_INDEXER_LOGICAL_LAYERS
    assert len(schedule_full_layers) == 21
    # Dense physical ids 0..20 in logical order; -1 for every skip layer.
    assert len(_GLM5_2_INDEXER_LAYER_MAP) == 78
    physical = [p for p in _GLM5_2_INDEXER_LAYER_MAP if p >= 0]
    assert physical == list(range(21))
    for logical, phys in enumerate(_GLM5_2_INDEXER_LAYER_MAP):
        assert (phys >= 0) == (logical in schedule_full_layers)


def test_alias_routing_only_glm52_gets_mapped_profile():
    for alias in ("zai-org/GLM-5.2-FP8", "glm-5.2", "glm-5.2-fp8"):
        profile = _resolve_indexer_profile(alias)
        assert profile is not None, alias
        assert profile.num_layers == 21, alias
        assert profile.logical_to_physical_layer is not None, alias
    for alias in ("zai-org/GLM-5-FP8", "glm-5.1", "zai-org/GLM-5.3-FP8"):
        profile = _resolve_indexer_profile(alias)
        assert profile is not None, alias
        assert profile.num_layers == 78, alias
        assert profile.logical_to_physical_layer is None, alias


def test_aux_gpu_config_threads_layer_map():
    config = build_gpu_kv_config_aux("zai-org/GLM-5.2-FP8", [1024] * 8)
    assert config is not None
    assert config.num_layers == 21
    assert list(config.logical_to_physical_layer) == list(_GLM5_2_INDEXER_LAYER_MAP)
    legacy = build_gpu_kv_config_aux("zai-org/GLM-5-FP8", [1024] * 8)
    assert legacy is not None
    assert legacy.num_layers == 78
    assert legacy.logical_to_physical_layer is None


def test_aux_layers_with_slots_gating_logic():
    # Mirror of BatchGenWorker._aux_layers_with_slots without constructing a
    # worker: the gate drops appends for unmapped layers and passes mapped
    # ones through.
    from batchgen.batchgen_worker import BatchGenWorker

    worker = object.__new__(BatchGenWorker)
    worker.gpu_paged_kv_cache_manager = types.SimpleNamespace(
        auxiliary=types.SimpleNamespace(
            config=types.SimpleNamespace(
                logical_to_physical_layer=list(_GLM5_2_INDEXER_LAYER_MAP)
            )
        )
    )
    valid = BatchGenWorker._aux_layers_with_slots(worker)
    assert valid == frozenset(_GLM5_2_INDEXER_LOGICAL_LAYERS)
    # Unmapped pools (map None) gate nothing.
    worker2 = object.__new__(BatchGenWorker)
    worker2.gpu_paged_kv_cache_manager = types.SimpleNamespace(
        auxiliary=types.SimpleNamespace(
            config=types.SimpleNamespace(logical_to_physical_layer=None)
        )
    )
    assert BatchGenWorker._aux_layers_with_slots(worker2) is None
    # Uninitialized aux: None and NOT cached.
    worker3 = object.__new__(BatchGenWorker)
    worker3.gpu_paged_kv_cache_manager = None
    assert BatchGenWorker._aux_layers_with_slots(worker3) is None
    assert getattr(worker3, "_aux_layers_with_slots_cache", False) is False
