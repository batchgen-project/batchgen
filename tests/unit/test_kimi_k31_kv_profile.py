# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3.1                                                        #
#  Copyright (c) 2025-2026 BatchGen Team                                       #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""Core-side gate for Kimi-K3.1: KV profiles, registry pattern, pool layout.

Three things, all CPU, no CUDA and no compiled engine:

1. the dual (primary + auxiliary indexer) Host/GPU KV profiles.  The
   cross-assertion against the model side's own declaration lives with the
   model set (``tests/test_kimi_k31_kv_agreement.py``);
2. the ``MODEL_NAME_PATTERNS`` fix — ``"Kimi-K3"`` is a substring of
   ``"Kimi-K3.1-Pro"``, so without an explicit earlier pattern a K3.1
   identifier bound to the dense ``kimi_k3`` identity;
3. ``batchgen_worker._prefill_pool_layout``, whose two glm5-shaped assumptions
   had to be generalized before a ``kimi_linear`` model could take the
   dual-coordinator path.  The method is compiled straight out of the worker's
   AST onto a namespace of fakes (the ``tests/unit/test_k3_prefix_pool_group.py``
   pattern), so nothing here imports the worker.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tests.worker_method_source import delegated_method_names, worker_shaped_method_node

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "batchgen" / "batchgen_worker.py"
HOST_KV_CONFIG = ROOT / "batchgen" / "kv_cache" / "host_kv_mananger_config.py"

#: 0-BASED checkpoint/engine layer ids of Kimi-K3's 24 full-attention layers —
#: the layers that hold MLA latent KV and, on K3.1, the indexer keys.
K31_DSA_LAYERS = tuple(range(3, 92, 4)) + (92,)
K3_ENGINE_LAYERS = 93

#: Released geometry, independent of the module under test.
LATENT_DIM = 576            # kv_lora_rank 512 + qk_rope_head_dim 64
INDEX_HEAD_DIM = 128        # dsa_config.index_head_dim
BF16_BYTES = 2
#: Page-split FP8 index key: 128 e4m3 bytes + one fp32 scale per token.
INDEX_K_BYTES = INDEX_HEAD_DIM * 1 + 4                                # 132
EXPECTED_LATENT_BYTES_PER_TOKEN = 24 * LATENT_DIM * BF16_BYTES        # 27,648
EXPECTED_INDEXER_BYTES_PER_TOKEN = 24 * INDEX_K_BYTES                 # 3,168
EXPECTED_PAGED_BYTES_PER_TOKEN = (
    EXPECTED_LATENT_BYTES_PER_TOKEN + EXPECTED_INDEXER_BYTES_PER_TOKEN)  # 30,816

K31_IDS = (
    "moonshotai/Kimi-K3.1-Pro",
    "Kimi-K3.1-Pro",
)


# --------------------------------------------------------------------------- #
#  Loaders                                                                    #
# --------------------------------------------------------------------------- #

def _load_host_kv_config(monkeypatch):
    """``host_kv_mananger_config`` by file path: ``batchgen.kv_cache.__init__``
    imports triton, so the package cannot be imported on a CPU box."""
    gpu_config = types.ModuleType("batchgen.kv_cache.gpu_paged_kv_manager")

    class _GPUPagedKVConfig(SimpleNamespace):
        pass

    gpu_config.GPUPagedKVConfig = _GPUPagedKVConfig
    native_loader = types.ModuleType("batchgen.native_loader")
    native_loader.core_engine = SimpleNamespace()
    monkeypatch.setitem(
        sys.modules, "batchgen.kv_cache.gpu_paged_kv_manager", gpu_config)
    monkeypatch.setitem(sys.modules, "batchgen.native_loader", native_loader)

    module_name = "_k31_host_kv_config_for_test"
    spec = importlib.util.spec_from_file_location(module_name, HOST_KV_CONFIG)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _profile_bytes_per_token(profile) -> int:
    """Per-token bytes this region costs across all of its layers."""
    if profile is None:
        return 0
    element = {"bfloat16": 2, "float16": 2, "float8_e4m3fn": 1}[profile.kv_dtype]
    k = profile.num_k_heads * profile.k_head_dim * element
    v = profile.num_v_heads * profile.v_head_dim * element
    return profile.num_layers * (k + v)


# --------------------------------------------------------------------------- #
#  1. Profiles                                                                #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("model_id", K31_IDS)
def test_k31_reuses_k3s_primary_and_gets_its_own_indexer_profile(monkeypatch, model_id):
    config = _load_host_kv_config(monkeypatch)
    assert config._resolve_profile(model_id) is config._KIMI_K3_MLA_PROFILE
    assert (config._resolve_indexer_profile(model_id)
            is config._KIMI_K31_INDEXER_PROFILE)
    assert config.is_dsa_model(model_id)


def test_kimi_k3_stays_dense(monkeypatch):
    """K3 ships no indexer weights, so it must resolve to no indexer profile —
    that is also what keeps its single-coordinator path intact."""
    config = _load_host_kv_config(monkeypatch)
    for model_id in ("moonshotai/Kimi-K3", "Kimi-K3", "kimi_k3"):
        assert config._resolve_profile(model_id) is config._KIMI_K3_MLA_PROFILE
        assert config._resolve_indexer_profile(model_id) is None
        assert not config.is_dsa_model(model_id)


def test_kimi_linear_48b_is_untouched(monkeypatch):
    config = _load_host_kv_config(monkeypatch)
    model_id = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
    assert config._resolve_profile(model_id) is config._KIMI_LINEAR_MLA_PROFILE
    assert not config.is_dsa_model(model_id)


def test_indexer_profile_geometry(monkeypatch):
    config = _load_host_kv_config(monkeypatch)
    aux = config._KIMI_K31_INDEXER_PROFILE
    assert aux.num_layers == 24
    assert aux.num_k_heads == 1
    assert aux.num_v_heads == 0 and aux.v_head_dim == 0
    # Page-split FP8: the layout DeepGEMM's ReLU-gated paged scorer reads, so
    # k_head_dim counts BYTES (128 e4m3 + one fp32 scale), not elements.
    assert aux.is_page_split
    assert aux.kv_dtype == "float8_e4m3fn"
    assert aux.k_head_dim == INDEX_K_BYTES == 132
    assert aux.k_data_dim() == INDEX_HEAD_DIM


def test_indexer_profile_has_the_page_split_route_geometry(monkeypatch):
    """Per layer, the K3.1 aux record is the record the production page-split
    route already stores and scores; only the layer map differs."""
    config = _load_host_kv_config(monkeypatch)
    aux = config._KIMI_K31_INDEXER_PROFILE
    reference = config._GLM5_2_INDEXER_PROFILE
    for field in ("num_k_heads", "k_head_dim", "num_v_heads", "v_head_dim",
                  "kv_dtype", "page_size", "page_format"):
        assert getattr(aux, field) == getattr(reference, field), field
    assert aux.bytes_per_page() == reference.bytes_per_page()


def test_indexer_layer_map_is_the_full_attention_layers(monkeypatch):
    """One page table serves both regions, so the aux map must put its 24 dense
    rows on the same logical layers the primary latent cache does."""
    config = _load_host_kv_config(monkeypatch)
    aux = config._KIMI_K31_INDEXER_PROFILE
    primary = config._KIMI_K3_MLA_PROFILE
    aux_map = aux.logical_to_physical_layer
    assert len(aux_map) == K3_ENGINE_LAYERS
    mapped = tuple(i for i, p in enumerate(aux_map) if p is not None and p >= 0)
    assert mapped == K31_DSA_LAYERS
    assert aux_map == primary.logical_to_physical_layer
    # Dense rows, in logical order.
    assert [aux_map[i] for i in mapped] == list(range(24))


def test_primary_and_aux_page_sizes_agree(monkeypatch):
    """``_compute_dual_page_count`` refuses a mismatch; pin it here so the two
    profiles cannot drift apart silently."""
    config = _load_host_kv_config(monkeypatch)
    assert (config._KIMI_K31_INDEXER_PROFILE.page_size
            == config._KIMI_K3_MLA_PROFILE.page_size == 64)


def test_per_token_bytes(monkeypatch):
    config = _load_host_kv_config(monkeypatch)
    primary = config._resolve_profile("moonshotai/Kimi-K3.1-Pro")
    aux = config._resolve_indexer_profile("moonshotai/Kimi-K3.1-Pro")
    assert _profile_bytes_per_token(primary) == EXPECTED_LATENT_BYTES_PER_TOKEN
    assert _profile_bytes_per_token(aux) == EXPECTED_INDEXER_BYTES_PER_TOKEN
    assert (_profile_bytes_per_token(primary) + _profile_bytes_per_token(aux)
            == EXPECTED_PAGED_BYTES_PER_TOKEN == 30_816)


def _load_dual_coordinator(monkeypatch, host_kv_config):
    """``dual_host_kv_coordinator`` by file path, sharing the already-loaded
    ``host_kv_mananger_config`` so both see the same profile tables."""
    eventlog = types.ModuleType("batchgen.eventlog.process")
    eventlog.native_log_level = lambda *a, **k: 0
    native_loader = types.ModuleType("batchgen.native_loader")
    native_loader.core_engine = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "batchgen.eventlog.process", eventlog)
    monkeypatch.setitem(sys.modules, "batchgen.native_loader", native_loader)
    monkeypatch.setitem(
        sys.modules, "batchgen.kv_cache.host_kv_mananger_config", host_kv_config)

    module_name = "_k31_dual_coordinator_for_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "batchgen" / "kv_cache" / "dual_host_kv_coordinator.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_dual_host_configs_split_one_page_count(monkeypatch):
    """The dual split is the single source of the shared page count, and it
    refuses a primary/aux page-size mismatch — so K3.1's two regions have to be
    compatible with each other, not merely individually well-formed."""
    config = _load_host_kv_config(monkeypatch)
    dual = _load_dual_coordinator(monkeypatch, config)

    budget = 64 * (1 << 30)
    primary, aux, num_pages = dual._compute_dual_page_count(
        "moonshotai/Kimi-K3.1-Pro", budget)
    assert primary is config._KIMI_K3_MLA_PROFILE
    assert aux is config._KIMI_K31_INDEXER_PROFILE
    assert num_pages > 0
    combined = (primary.bytes_per_page() * primary.num_layers
                + aux.bytes_per_page() * aux.num_layers)
    assert num_pages * combined <= budget
    # One page count, both regions: a token's pages cost latent + index keys.
    assert combined == EXPECTED_PAGED_BYTES_PER_TOKEN * primary.page_size

    # GPU-DEFERRED: DualHostKVCoordinator.build_host_configs turns this split
    # into two bg_lib.HostPagedKVConfig objects, which only the compiled engine
    # can construct. The arithmetic above IS that function's own, and the early
    # return for a dense model is covered below; the config objects themselves
    # are validated on-node.


def test_a_dense_model_gets_no_dual_configs(monkeypatch):
    """CPU-provable leg of ``build_host_configs``: the non-DSA early return,
    which precedes every native call."""
    config = _load_host_kv_config(monkeypatch)
    dual = _load_dual_coordinator(monkeypatch, config)
    for dense in ("moonshotai/Kimi-K3",
                  "moonshotai/Kimi-Linear-48B-A3B-Instruct"):
        assert dual.DualHostKVCoordinator.build_host_configs(
            dense, 64 * (1 << 30)) is None


def test_worker_dual_branch_passes_the_primary_layer_map():
    """The worker's dual (DSA) GPU-KV branch must hand BOTH pools the
    engine-layer map. GLM never exposed the omission (every engine layer is a
    DSA layer, so engine id == physical slot); on Kimi-K3.1 the primary pool
    without its map bounds-checks engine layer 27 against 24 slots and the
    first decode-graph capture dies with resolve_physical_layer out of range
    (measured 2026-11-06, 4x8 H20)."""
    calls = {}
    for node in ast.walk(_worker_method_node("_initialize_gpu_kv_manager_fixed_size")):
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", None) == "GPUPagedKVConfig"
                and isinstance(node.targets[0], ast.Name)):
            calls[node.targets[0].id] = {k.arg for k in node.value.keywords}
    assert "primary_config" in calls and "aux_config" in calls
    assert "logical_to_physical_layer" in calls["primary_config"]
    assert "logical_to_physical_layer" in calls["aux_config"]


def test_aux_gpu_config_is_built_from_the_profile(monkeypatch):
    config = _load_host_kv_config(monkeypatch)
    built = config.build_gpu_kv_config_aux("moonshotai/Kimi-K3.1-Pro", [4096])
    assert built.num_layers == 24
    assert built.num_k_heads == 1
    assert built.k_head_dim == INDEX_K_BYTES
    assert built.kv_dtype is torch.float8_e4m3fn
    assert len(built.logical_to_physical_layer) == K3_ENGINE_LAYERS
    assert config.build_gpu_kv_config_aux("moonshotai/Kimi-K3", [4096]) is None


def test_other_models_profiles_are_unchanged(monkeypatch):
    """Regression: adding K3.1 must not move any existing resolution."""
    config = _load_host_kv_config(monkeypatch)
    expected = {
        "zai-org/GLM-5": ("_GLM5_MLA_PROFILE", "_GLM5_INDEXER_PROFILE"),
        "zai-org/GLM-5.1": ("_GLM5_MLA_PROFILE", "_GLM5_INDEXER_PROFILE"),
        "zai-org/GLM-5.2": ("_GLM5_MLA_PROFILE", "_GLM5_2_INDEXER_PROFILE"),
        "zai-org/GLM-5.3": ("_GLM5_MLA_PROFILE", "_GLM5_2_INDEXER_PROFILE"),
        "deepseek-ai/DeepSeek-V3.2": (
            "_DEEPSEEK_MLA_PROFILE", "_DEEPSEEK_V3_2_INDEXER_PROFILE"),
        "openai/gpt-oss-120b": ("_GPT_OSS_GQA_PROFILE", None),
        "MiniMaxAI/MiniMax-M2.5": ("_MINIMAX_M25_GQA_PROFILE", None),
    }
    for model_id, (primary_name, aux_name) in expected.items():
        assert config._resolve_profile(model_id) is getattr(config, primary_name)
        got_aux = config._resolve_indexer_profile(model_id)
        assert got_aux is (None if aux_name is None
                           else getattr(config, aux_name)), model_id


def test_an_unlisted_kimi_identifier_still_fails_loud(monkeypatch):
    config = _load_host_kv_config(monkeypatch)
    with pytest.raises(ValueError, match="Unsupported model"):
        config._resolve_profile("moonshotai/Kimi-K3.2-Pro")


# --------------------------------------------------------------------------- #
#  2. Config name-pattern registry                                            #
# --------------------------------------------------------------------------- #

def test_k31_identifier_resolves_to_the_k31_config_type():
    """The corrected pin.  Before the explicit pattern, ``"Kimi-K3"`` matched
    ``"Kimi-K3.1-Pro"`` as a substring and bound it to the dense identity."""
    from batchgen.config.model_registry import _detect_model_type_from_identifier

    assert _detect_model_type_from_identifier("moonshotai/Kimi-K3.1-Pro") == "kimi_k31"
    assert _detect_model_type_from_identifier("moonshotai/Kimi-K3.1") == "kimi_k31"
    assert _detect_model_type_from_identifier("moonshotai/Kimi-K3") == "kimi_k3"
    assert _detect_model_type_from_identifier(
        "moonshotai/Kimi-Linear-48B-A3B-Instruct") == "kimi_linear"


def test_the_k31_patterns_precede_the_broad_kimi_k3_pattern():
    """Order is the mechanism: ``_detect_model_type_from_identifier`` returns
    the FIRST matching pattern, so a later K3.1 entry would never be reached."""
    from batchgen.config.model_registry import MODEL_NAME_PATTERNS

    keys = list(MODEL_NAME_PATTERNS)
    assert keys.index("Kimi-K3.1-Pro") < keys.index("Kimi-K3")
    assert keys.index("Kimi-K3.1") < keys.index("Kimi-K3")


# --------------------------------------------------------------------------- #
#  3. Worker: _prefill_pool_layout                                            #
# --------------------------------------------------------------------------- #

WORKER_TREE = ast.parse(WORKER.read_text(), filename=str(WORKER))
ENGINE = WORKER.parents[0] / "worker" / "prefill_engine.py"
ENGINE_TREE = ast.parse(ENGINE.read_text(), filename=str(ENGINE))


def _worker_method_node(name):
    # S10a: the GPU-KV manager init runs in KVCapacityEngine (worker delegates).
    if name in delegated_method_names():
        return worker_shaped_method_node(name)
    tree = ENGINE_TREE if name in {
        "_prefill_pool_layout", "_prefill_pool_psm_layout"
    } else WORKER_TREE
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


class _FakeDualHostKVCoordinator:
    """Stands in for the real coordinator: the method only isinstance-checks it."""


def _layout_worker(*, dual, engine_layers, indexer_layers, indexer_dim,
                   kv_num_layers, kv_layer_map, psm_layout, model_present=True,
                   cached=None):
    """A fake worker holding only what ``_prefill_pool_layout`` reads."""
    names = ["_prefill_pool_layout", "_prefill_pool_psm_layout"]
    # Worker-shaped predicate (S5a carve-out #5): lifted engine bodies call it
    # on the rebound self, so bind it worker-shaped rather than engine-shaped.
    predicate = copy.deepcopy(_worker_method_node("_prefill_host_view_is_dual"))
    wanted = [copy.deepcopy(_worker_method_node(name)) for name in names] + [predicate]
    module = ast.fix_missing_locations(ast.Module(body=wanted, type_ignores=[]))
    namespace = {"DualHostKVCoordinator": _FakeDualHostKVCoordinator}
    exec(compile(module, str(ENGINE), "exec"), namespace)

    layers = []
    for layer_idx in range(engine_layers):
        indexer = (SimpleNamespace(index_head_dim=indexer_dim)
                   if layer_idx in indexer_layers else None)
        layers.append(SimpleNamespace(self_attn=SimpleNamespace(module=SimpleNamespace(
            indexer=indexer))))
    model = SimpleNamespace(model=SimpleNamespace(layers=layers))

    worker = SimpleNamespace(
        host_paged_kv_worker_view=(
            _FakeDualHostKVCoordinator() if dual else object()),
        model=model if model_present else None,
        parallel_manager=SimpleNamespace(
            prefill_pool_layout=(None if psm_layout is None
                                 else (lambda: dict(psm_layout)))),
    )
    if cached is not None:
        worker._prefill_pool_layout_cache = cached
    for name in names:
        engine_method = namespace[name]
        setattr(
            worker,
            name,
            types.MethodType(
                lambda self, *args, _fn=engine_method, **kwargs:
                _fn(None, self, *args, **kwargs),
                worker,
            ),
        )
    worker._prefill_host_view_is_dual = types.MethodType(
        namespace["_prefill_host_view_is_dual"], worker
    )
    geometry = SimpleNamespace(
        num_layers=kv_num_layers, logical_to_physical_layer=kv_layer_map)
    return worker, geometry


#: Kimi-K3.1: 93 engine layers, 24 of them DSA; the primary cache stores those
#: 24 densely behind a 93-entry logical map; the pool row holds the EXPANDED
#: per-head K and V, declared by the PSM.
_K31_KV_MAP = tuple(
    (K31_DSA_LAYERS.index(i) if i in K31_DSA_LAYERS else -1)
    for i in range(K3_ENGINE_LAYERS)
)
_K31_PSM_LAYOUT = {
    "num_layers": 24,
    "layer_map": _K31_KV_MAP,
    "num_kv_heads": 12,
    "k_head_dim": 192,
    "v_head_dim": 128,
}


def test_pool_layout_for_a_kimi_shaped_dsa_model():
    """Both generalizations at once.

    Before them this raised: the depth check compared 93 engine layers against
    the primary profile's 24 dense ROWS, and ``with_v`` was hardcoded False for
    every dual-coordinator model, which would have dropped the V rows of a pool
    whose widths ``_prefill_pool_page_dims`` still took from the PSM.
    """
    worker, geometry = _layout_worker(
        dual=True, engine_layers=K3_ENGINE_LAYERS,
        indexer_layers=set(K31_DSA_LAYERS), indexer_dim=INDEX_HEAD_DIM,
        kv_num_layers=24, kv_layer_map=_K31_KV_MAP,
        psm_layout=_K31_PSM_LAYOUT,
    )
    with_v, indexer_layers, indexer_dim = worker._prefill_pool_layout(geometry)
    assert with_v is True
    assert indexer_layers == K31_DSA_LAYERS
    assert indexer_dim == INDEX_HEAD_DIM


def test_pool_layout_for_a_glm_shaped_dsa_model_is_unchanged():
    """GLM-5 pools one latent per layer and has no PSM layout: no V rows, and
    the depth check still reads ``num_layers`` because the profile declares no
    logical map."""
    worker, geometry = _layout_worker(
        dual=True, engine_layers=78, indexer_layers=set(range(78)),
        indexer_dim=128, kv_num_layers=78, kv_layer_map=None, psm_layout=None,
    )
    with_v, indexer_layers, indexer_dim = worker._prefill_pool_layout(geometry)
    assert with_v is False
    assert indexer_layers == tuple(range(78))
    assert indexer_dim == 128


def test_pool_layout_for_a_glm_52_shaped_subset_of_indexer_layers():
    """Only the top-k computing layers carry an indexer; the rest reuse."""
    indexer_layers = (0, 1, 2) + tuple(range(6, 78, 4))
    worker, geometry = _layout_worker(
        dual=True, engine_layers=78, indexer_layers=set(indexer_layers),
        indexer_dim=128, kv_num_layers=78, kv_layer_map=None, psm_layout=None,
    )
    _, got, _ = worker._prefill_pool_layout(geometry)
    assert got == indexer_layers
    assert len(got) == 21


def test_pool_layout_for_a_non_dsa_model_pools_kv_and_no_index_keys():
    worker, geometry = _layout_worker(
        dual=False, engine_layers=27, indexer_layers=set(), indexer_dim=0,
        kv_num_layers=27, kv_layer_map=None, psm_layout=_K31_PSM_LAYOUT,
    )
    assert worker._prefill_pool_layout(geometry) == (True, (), 0)


def test_pool_layout_rejects_a_depth_mismatch():
    """The check still fires — it just reads the right number now."""
    worker, geometry = _layout_worker(
        dual=True, engine_layers=K3_ENGINE_LAYERS,
        indexer_layers=set(K31_DSA_LAYERS), indexer_dim=INDEX_HEAD_DIM,
        kv_num_layers=24, kv_layer_map=tuple(range(50)),
        psm_layout=_K31_PSM_LAYOUT,
    )
    with pytest.raises(RuntimeError, match="do not match the KV geometry"):
        worker._prefill_pool_layout(geometry)


def test_pool_layout_rejects_a_dsa_model_with_no_indexer_layers():
    worker, geometry = _layout_worker(
        dual=True, engine_layers=K3_ENGINE_LAYERS, indexer_layers=set(),
        indexer_dim=INDEX_HEAD_DIM, kv_num_layers=24,
        kv_layer_map=_K31_KV_MAP, psm_layout=_K31_PSM_LAYOUT,
    )
    with pytest.raises(RuntimeError, match="do not match the KV geometry"):
        worker._prefill_pool_layout(geometry)


def test_pool_layout_rejects_mixed_indexer_head_dims():
    worker, geometry = _layout_worker(
        dual=True, engine_layers=K3_ENGINE_LAYERS,
        indexer_layers=set(K31_DSA_LAYERS), indexer_dim=INDEX_HEAD_DIM,
        kv_num_layers=24, kv_layer_map=_K31_KV_MAP,
        psm_layout=_K31_PSM_LAYOUT,
    )
    worker.model.model.layers[7].self_attn.module.indexer.index_head_dim = 64
    with pytest.raises(RuntimeError, match="mixed indexer head dims"):
        worker._prefill_pool_layout(geometry)


def test_pool_layout_uses_its_cache_when_the_model_is_deep_freed():
    """Wave planning runs in the decode phase, where the model may be gone."""
    cached = (True, K31_DSA_LAYERS, INDEX_HEAD_DIM)
    worker, geometry = _layout_worker(
        dual=True, engine_layers=K3_ENGINE_LAYERS,
        indexer_layers=set(K31_DSA_LAYERS), indexer_dim=INDEX_HEAD_DIM,
        kv_num_layers=24, kv_layer_map=_K31_KV_MAP,
        psm_layout=_K31_PSM_LAYOUT, model_present=False, cached=cached,
    )
    assert worker._prefill_pool_layout(geometry) == cached


def test_pool_layout_refuses_an_unmeasured_cache():
    worker, geometry = _layout_worker(
        dual=True, engine_layers=K3_ENGINE_LAYERS,
        indexer_layers=set(K31_DSA_LAYERS), indexer_dim=INDEX_HEAD_DIM,
        kv_num_layers=24, kv_layer_map=_K31_KV_MAP,
        psm_layout=_K31_PSM_LAYOUT, model_present=False,
    )
    with pytest.raises(RuntimeError, match="before the first prefill measured"):
        worker._prefill_pool_layout(geometry)


def test_pool_layout_caches_the_measured_layout():
    worker, geometry = _layout_worker(
        dual=True, engine_layers=K3_ENGINE_LAYERS,
        indexer_layers=set(K31_DSA_LAYERS), indexer_dim=INDEX_HEAD_DIM,
        kv_num_layers=24, kv_layer_map=_K31_KV_MAP,
        psm_layout=_K31_PSM_LAYOUT,
    )
    layout = worker._prefill_pool_layout(geometry)
    assert worker._prefill_pool_layout_cache == layout
