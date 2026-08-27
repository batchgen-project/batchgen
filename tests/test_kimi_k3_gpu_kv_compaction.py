"""Kimi-K3 paged-KV uses 93 logical layer ids over 24 physical MLA rows."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
K3_MODEL_ID = "moonshotai/Kimi-K3"
GIB = 1024 ** 3


def _config_module():
    try:
        from batchgen.kv_cache import host_kv_mananger_config as config_module
    except Exception as exc:
        pytest.skip(f"core_engine op unavailable on this host: {exc}")
    return config_module


def _checkpoint_mla_layers():
    payload = json.loads(
        (
            ROOT
            / "batchgen"
            / "models"
            / "moonshotai"
            / "kimi_k3"
            / "assets"
            / "config.json"
        ).read_text()
    )["text_config"]
    kda_layers = set(payload["linear_attn_config"]["kda_layers"])
    return [
        layer_idx
        for layer_idx in range(payload["num_hidden_layers"])
        if layer_idx + 1 not in kda_layers
    ]


def test_k3_profile_matches_checkpoint_and_maps_mla_layers_densely():
    module = _config_module()
    profile = module._resolve_profile(K3_MODEL_ID)
    mapping = tuple(profile.logical_to_physical_layer)
    mla_layers = _checkpoint_mla_layers()

    assert len(mapping) == 93
    assert len(mla_layers) == 24
    assert profile.num_layers == 24
    assert [idx for idx, physical in enumerate(mapping) if physical >= 0] == mla_layers
    assert [mapping[idx] for idx in mla_layers] == list(range(24))
    assert all(
        mapping[idx] == -1 for idx in range(93) if idx not in set(mla_layers)
    )


def test_k3_host_and_gpu_configs_share_the_same_24_physical_rows():
    module = _config_module()
    host = module.build_host_kv_config(K3_MODEL_ID, 16 * GIB)
    gpu = module.build_gpu_kv_config_fixed_size(K3_MODEL_ID, 6.98)

    assert host.num_layers == gpu.num_layers == 24
    assert tuple(host.logical_to_physical_layer) == tuple(
        gpu.logical_to_physical_layer
    )
    assert len(host.logical_to_physical_layer) == 93


def test_k3_fixed_gpu_budget_fits_the_64_request_diagnostic_in_one_wave():
    module = _config_module()
    config = module.build_gpu_kv_config_fixed_size(K3_MODEL_ID, 6.98)

    assert config.num_pages == 4235
    usable_pages_across_two_tp8_groups = int(config.num_pages * 0.9) * 2
    assert usable_pages_across_two_tp8_groups >= 5424


def test_k3_gpu_manager_resolves_logical_mla_layers_and_rejects_kda_layers():
    module = _config_module()
    from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager

    config = module.build_gpu_kv_config_fixed_size(K3_MODEL_ID, 6.98)
    manager = GPUPagedKVCacheManager(config=config, device="cpu")

    assert manager.resolve_physical_layer(3) == 0
    assert manager.resolve_physical_layer(91) == 22
    assert manager.resolve_physical_layer(92) == 23
    with pytest.raises(KeyError, match="logical layer 0"):
        manager.resolve_physical_layer(0)


def test_generic_fixed_gpu_profile_remains_identity_mapped():
    module = _config_module()
    config = module.build_gpu_kv_config_fixed_size("gpt-oss-120b", 1.0)

    assert config.num_layers == 36
    assert config.logical_to_physical_layer is None


def test_host_worker_view_factory_selects_mapped_mla_only_for_mapped_profile():
    module = _config_module()
    calls = []

    def view(name):
        return lambda config: calls.append((name, config)) or name

    core = SimpleNamespace(
        MappedMLAHostPagedKVWorkerView=view("mapped-mla"),
        MLAHostPagedKVWorkerView=view("mla"),
        MappedDefaultHostPagedKVWorkerView=view("mapped-default"),
        DefaultHostPagedKVWorkerView=view("default"),
    )
    mapped_mla = SimpleNamespace(
        num_v_heads=0, logical_to_physical_layer=[-1, 0]
    )
    identity_mla = SimpleNamespace(num_v_heads=0, logical_to_physical_layer=[])

    assert module.build_host_kv_worker_view(core, mapped_mla) == "mapped-mla"
    assert module.build_host_kv_worker_view(core, identity_mla) == "mla"
    assert calls == [("mapped-mla", mapped_mla), ("mla", identity_mla)]
