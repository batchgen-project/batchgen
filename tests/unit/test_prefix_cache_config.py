from __future__ import annotations

from types import SimpleNamespace

import pytest

from batchgen.prefix_reuse.config import (
    PrefixKVGroupSemantic,
    PrefixKVGroupSpec,
    build_prefix_cache_namespace_digest,
    build_prefix_cache_runtime_config_from_specs,
    derive_prefix_cache_shm_name,
)
from batchgen.server.server_args import _build_parser


def test_prefix_cache_runtime_config_derives_boundaries_and_capacities():
    config = build_prefix_cache_runtime_config_from_specs(
        model_name="test/model",
        kv_dtype="bfloat16",
        host_kv_pages_per_required_group=128,
        group_specs=[
            PrefixKVGroupSpec(
                group_id=0,
                semantic=PrefixKVGroupSemantic.MLA_COMPRESSED_KV,
                required_for_reuse=True,
                raw_page_tokens=64,
            ),
            PrefixKVGroupSpec(
                group_id=1,
                semantic=PrefixKVGroupSemantic.SWA_KV,
                required_for_reuse=True,
                raw_page_tokens=128,
            ),
        ],
    )

    assert config.hash_block_tokens == 64
    assert config.publish_boundary_tokens == 128
    assert config.max_nodes >= 1024
    assert config.max_group_entries == config.max_nodes * 2
    assert config.max_page_handles >= config.max_group_entries
    assert config.max_attachments >= 1024


def test_prefix_cache_namespace_digest_is_stable_and_group_sensitive():
    group = PrefixKVGroupSpec(
        group_id=0,
        semantic=PrefixKVGroupSemantic.FULL_KV,
        required_for_reuse=True,
        raw_page_tokens=64,
    )
    same = build_prefix_cache_namespace_digest(
        model_name="OpenAI/GPT-OSS-120B",
        kv_dtype="bfloat16",
        group_specs=[group],
    )
    reordered_case = build_prefix_cache_namespace_digest(
        model_name="openai/gpt-oss-120b",
        kv_dtype="BFLOAT16",
        group_specs=[group],
    )
    changed = build_prefix_cache_namespace_digest(
        model_name="openai/gpt-oss-120b",
        kv_dtype="bfloat16",
        group_specs=[
            PrefixKVGroupSpec(
                group_id=0,
                semantic=PrefixKVGroupSemantic.FULL_KV,
                required_for_reuse=True,
                raw_page_tokens=128,
            )
        ],
    )

    assert same == reordered_case
    assert same != changed
    assert len(same) == 4


def test_prefix_cache_core_config_conversion_uses_bound_classes():
    class _CoreGroupSpec(SimpleNamespace):
        pass

    class _CoreConfig(SimpleNamespace):
        pass

    class _Core:
        HostKVGroupSpec = _CoreGroupSpec
        HostPrefixCacheConfig = _CoreConfig
        HostKVGroupSemantic = SimpleNamespace(
            FULL_KV="full",
            MLA_COMPRESSED_KV="mla",
            SWA_KV="swa",
            COMPRESSED_RATIO_KV="compressed",
        )

    config = build_prefix_cache_runtime_config_from_specs(
        model_name="test/model",
        kv_dtype="bfloat16",
        host_kv_pages_per_required_group=8,
        group_specs=[
            PrefixKVGroupSpec(
                group_id=3,
                semantic=PrefixKVGroupSemantic.COMPRESSED_RATIO_KV,
                required_for_reuse=False,
                raw_page_tokens=256,
                compression_ratio=4,
            ),
            PrefixKVGroupSpec(
                group_id=0,
                semantic=PrefixKVGroupSemantic.FULL_KV,
                required_for_reuse=True,
                raw_page_tokens=64,
            ),
        ],
    )

    core_config = config.to_core_config(_Core)

    assert core_config.shm_name == config.shm_name
    assert core_config.hash_block_tokens == 64
    assert len(core_config.group_specs) == 2
    assert core_config.group_specs[0].group_id == 3
    assert core_config.group_specs[0].semantic == "compressed"
    assert core_config.group_specs[0].compression_ratio == 4


def test_prefix_cache_runtime_config_rejects_no_required_group():
    with pytest.raises(ValueError, match="required KV group"):
        build_prefix_cache_runtime_config_from_specs(
            model_name="test/model",
            kv_dtype="bfloat16",
            host_kv_pages_per_required_group=8,
            group_specs=[
                PrefixKVGroupSpec(
                    group_id=0,
                    semantic=PrefixKVGroupSemantic.FULL_KV,
                    required_for_reuse=False,
                    raw_page_tokens=64,
                )
            ],
        )


def test_prefix_cache_shm_name_is_sanitized_and_node_scoped():
    shm_name = derive_prefix_cache_shm_name(
        "Org/Model-Name", node_rank=2
    )

    assert shm_name.startswith("batchgen_prefix_cache_org_model_name_")
    assert shm_name.endswith("_node2")


def test_server_parser_exposes_only_prefix_cache_user_flags():
    parsed = _build_parser().parse_args(
        [
            "--model",
            "openai/gpt-oss-120b",
            "--enable-prefix-cache",
            "--prefix-cache-debug-stats",
        ]
    )

    assert parsed.enable_prefix_cache is True
    assert parsed.prefix_cache_debug_stats is True
    assert not hasattr(parsed, "prefix_cache_size_gb")
    assert not hasattr(parsed, "prefix_cache_hash_block_tokens")
