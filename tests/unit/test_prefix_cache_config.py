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

_SHM = "batchgen_lane_0123.prefix_cache"


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
        shm_name=_SHM,
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


def test_gpt_oss_aliases_share_namespace_and_use_the_given_shm_name():
    canonical = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
        shm_name=_SHM,
    )
    alias = build_prefix_cache_runtime_config(
        model_name="gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
        shm_name=_SHM,
    )

    assert alias.namespace_digest == canonical.namespace_digest
    # The name is the runtime lane's, never derived from the model: two lanes
    # of one model must not share a prefix index.
    assert alias.shm_name == canonical.shm_name == _SHM


def test_shm_name_defaults_to_a_derived_label():
    # The region is an anonymous memfd; the name is a diagnostic label, so an
    # omitted or empty name derives a stable per-model label instead of failing.
    config = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
        shm_name="",
    )
    assert config.shm_name.startswith("batchgen_prefix_cache_")


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
        shm_name=_SHM,
    )

    converted = runtime.to_core_config(core)

    assert converted.shm_name == runtime.shm_name
    assert converted.group_specs[0].semantic == "full-kv-enum"
    assert converted.group_specs[0].raw_page_tokens == 64
    assert converted.max_page_handles == 4096


def test_memfd_identity_defaults_and_round_trip():
    # Owner builds without identity; workers attach with the published one.
    runtime = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
        shm_name=_SHM,
    )
    assert runtime.memfd_creator_pid == -1
    assert runtime.memfd_fd == -1

    attached = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
        memfd_creator_pid=4321,
        memfd_fd=17,
    )
    class CoreConfig:
        pass

    class CoreGroupSpec:
        pass

    core = SimpleNamespace(
        HostPrefixCacheConfig=CoreConfig,
        HostKVGroupSpec=CoreGroupSpec,
        HostKVGroupSemantic=SimpleNamespace(FULL_KV="full-kv-enum"),
    )
    converted = attached.to_core_config(core)
    assert converted.memfd_creator_pid == 4321
    assert converted.memfd_fd == 17


def test_namespace_changes_with_kv_dtype_or_page_geometry():
    baseline = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
        shm_name=_SHM,
    )
    other_dtype = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="float16",
        host_kv_config=_host_config(),
        shm_name=_SHM,
    )
    other_page = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(page_size_tokens=128),
        shm_name=_SHM,
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
            shm_name=_SHM,
        )


def _mla_host_config(**overrides):
    # GLM DSA primary region: MLA latent, one 576-dim head, no V.
    return _host_config(num_layers=78, num_k_heads=1, k_head_dim=576,
                        num_v_heads=0, v_head_dim=0, **overrides)


def _indexer_host_config(**overrides):
    # GLM DSA aux region: indexer keys, one 128-dim head, no V.
    return _host_config(num_layers=78, num_k_heads=1, k_head_dim=128,
                        num_v_heads=0, v_head_dim=0, **overrides)


def _glm_config(model_name, **kwargs):
    return build_prefix_cache_runtime_config(
        model_name=model_name,
        kv_dtype="bfloat16",
        host_kv_config=kwargs.pop("primary", _mla_host_config()),
        aux_host_kv_config=kwargs.pop("aux", _indexer_host_config()),
        shm_name=_SHM,
        **kwargs,
    )


def test_dsa_model_gets_a_latent_group_and_an_indexer_group():
    config = _glm_config("zai-org/GLM-5.3-FP8")

    assert [spec.semantic for spec in config.group_specs] == [
        PrefixKVGroupSemantic.MLA_LATENT,
        PrefixKVGroupSemantic.DSA_INDEXER_KEYS,
    ]
    assert [spec.group_id for spec in config.group_specs] == [0, 1]
    assert all(spec.required_for_reuse for spec in config.group_specs)
    assert config.max_nodes == 4097
    assert config.max_group_entries == 2 * 4097
    assert config.max_page_handles == 2 * 4096
    assert config.host_page_bytes_all_layers == 78 * 64 * (576 + 128) * 2


def test_support_follows_the_prefill_path_not_a_name_list():
    # GLM-5.2 runs the same sparse DSA prefill as GLM-5.3.
    assert require_prefix_cache_model_support("GLM-5.2-FP8") == "glm_moe_dsa_5_2"
    assert require_prefix_cache_model_support("zai-org/GLM-5.3-FP8") == "glm_moe_dsa_5_3"
    # GLM-5 / 5.1 prefill is dense only.
    with pytest.raises(ValueError, match="currently supports only"):
        require_prefix_cache_model_support("zai-org/GLM-5")


def test_dsa_needs_the_aux_region_and_matching_geometry():
    with pytest.raises(ValueError, match="aux"):
        build_prefix_cache_runtime_config(
            model_name="zai-org/GLM-5.3-FP8",
            kv_dtype="bfloat16",
            host_kv_config=_mla_host_config(),
            shm_name=_SHM,
        )
    with pytest.raises(ValueError, match="share page size and page count"):
        _glm_config("zai-org/GLM-5.3-FP8", aux=_indexer_host_config(num_pages=4095))
    with pytest.raises(ValueError, match="no aux Host-KV region"):
        build_prefix_cache_runtime_config(
            model_name="openai/gpt-oss-120b",
            kv_dtype="bfloat16",
            host_kv_config=_host_config(),
            aux_host_kv_config=_host_config(),
            shm_name=_SHM,
        )


def test_dsa_groups_convert_to_the_core_mla_semantic():
    class CoreConfig:
        pass

    class CoreGroupSpec:
        pass

    core = SimpleNamespace(
        HostPrefixCacheConfig=CoreConfig,
        HostKVGroupSpec=CoreGroupSpec,
        HostKVGroupSemantic=SimpleNamespace(
            FULL_KV="full-kv-enum", MLA_COMPRESSED_KV="mla-enum"
        ),
    )
    converted = _glm_config("zai-org/GLM-5.3-FP8").to_core_config(core)

    assert [spec.semantic for spec in converted.group_specs] == ["mla-enum", "mla-enum"]
    assert [spec.group_id for spec in converted.group_specs] == [0, 1]


def test_dsa_and_full_kv_namespaces_differ():
    glm = _glm_config("zai-org/GLM-5.3-FP8")
    gpt = build_prefix_cache_runtime_config(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        host_kv_config=_host_config(),
        shm_name=_SHM,
    )
    assert glm.namespace_digest != gpt.namespace_digest
