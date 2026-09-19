import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_CONFIG_PATH = (
    Path(__file__).parents[2] / "batchgen" / "prefix_reuse" / "config.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_prefix_cache_runtime_config", _CONFIG_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_CONFIG = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _CONFIG
_SPEC.loader.exec_module(_CONFIG)

PrefixKVGroupSemantic = _CONFIG.PrefixKVGroupSemantic
build_prefix_cache_runtime_config = _CONFIG.build_prefix_cache_runtime_config
require_prefix_cache_model_support = _CONFIG.require_prefix_cache_model_support
unlink_prefix_cache_shared_memory = _CONFIG.unlink_prefix_cache_shared_memory


def _host_config(**overrides):
    values = {
        "page_size_tokens": 64,
        "num_pages": 4096,
        "sequence_table_capacity": 2048,
        "num_layers": 36,
        "num_k_heads": 8,
        "k_head_dim": 64,
        "k_element_size_bytes": 2,
        "num_v_heads": 8,
        "v_head_dim": 64,
        "v_element_size_bytes": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_gpt_oss_runtime_config_is_derived_from_host_geometry():
    config = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
        debug_stats=True,
    )

    assert config.hash_block_tokens == 64
    assert config.publish_boundary_tokens == 64
    assert config.max_nodes == 4097
    assert config.max_group_entries == 4097
    assert config.max_page_handles == 4096
    assert config.max_attachments == 2048
    assert config.host_page_bytes_all_layers == 36 * 64 * 8 * 64 * 2 * 2
    assert config.debug_stats is True
    assert len(config.namespace_digest) == 4
    assert config.group_specs[0].semantic is PrefixKVGroupSemantic.FULL_KV


def test_gpt_oss_aliases_share_namespace_and_shm_name():
    canonical = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
    )
    alias = build_prefix_cache_runtime_config(
        model_name="GPT-OSS-120B",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
    )

    assert alias.namespace_digest == canonical.namespace_digest
    assert alias.shm_name == canonical.shm_name


def test_runtime_config_converts_to_core_types():
    class CoreConfig:
        pass

    class CoreGroupSpec:
        pass

    core = SimpleNamespace(
        HostPrefixCacheConfig=CoreConfig,
        HostKVGroupSpec=CoreGroupSpec,
        HostKVGroupSemantic=SimpleNamespace(FULL_KV="full-kv-enum"),
    )
    runtime = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
    )

    converted = runtime.to_core_config(core)

    assert converted.shm_name == runtime.shm_name
    assert converted.group_specs[0].semantic == "full-kv-enum"
    assert converted.group_specs[0].raw_page_tokens == 64
    assert converted.max_page_handles == 4096


def test_unlink_targets_only_the_derived_prefix_region(monkeypatch):
    runtime = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
    )
    removed = []
    monkeypatch.setattr(_CONFIG.os, "unlink", removed.append)

    unlink_prefix_cache_shared_memory(runtime)

    assert removed == [f"/dev/shm/{runtime.shm_name}"]


def test_namespace_changes_with_kv_dtype_or_page_geometry():
    baseline = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
    )
    other_dtype = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="float16",
        host_kv_config=_host_config(),
    )
    other_page = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(page_size_tokens=128),
    )

    assert baseline.namespace_digest != other_dtype.namespace_digest
    assert baseline.namespace_digest != other_page.namespace_digest


@pytest.mark.parametrize(
    "model_name",
    ["moonshotai/kimi-k3", "zai-org/glm-5", "minimax-m2.5"],
)
def test_unsupported_models_fail_loud(model_name):
    with pytest.raises(ValueError, match="currently supports only"):
        require_prefix_cache_model_support(model_name)


@pytest.mark.parametrize(
    "field",
    ["page_size_tokens", "num_pages", "sequence_table_capacity"],
)
def test_invalid_host_geometry_fails_before_core_allocation(field):
    with pytest.raises(ValueError, match="positive capacities"):
        build_prefix_cache_runtime_config(
            model_name="openai/gpt-oss-120b",
            kv_dtype="bfloat16",
            host_kv_config=_host_config(**{field: 0}),
        )
