"""The prefix index of a DSA model is sized from the workers' dual Host-KV split.

Needs the native core engine (HostPagedKVConfig); skipped where it is absent.
"""

import pytest

pytest.importorskip("triton")
dual = pytest.importorskip("batchgen.kv_cache.dual_host_kv_coordinator")
from batchgen.prefix_reuse.config import (  # noqa: E402
    PrefixKVGroupSemantic,
    build_prefix_cache_runtime_config,
)

BUDGET = 280 * 1024**3


def test_non_dsa_model_has_no_dual_configs():
    assert dual.DualHostKVCoordinator.build_host_configs(
        "openai/gpt-oss-120b", BUDGET
    ) is None


def test_glm_dual_configs_share_one_page_count():
    primary, aux = dual.DualHostKVCoordinator.build_host_configs(
        "zai-org/GLM-5.3-FP8", BUDGET, "lane.host_kv", "lane.host_kv_aux"
    )
    _, _, num_pages = dual._compute_dual_page_count("zai-org/GLM-5.3-FP8", BUDGET)

    assert primary.num_pages == aux.num_pages == num_pages
    assert primary.page_size_tokens == aux.page_size_tokens == 64
    assert (primary.k_head_dim, aux.k_head_dim) == (576, 128)
    assert (primary.shm_name, aux.shm_name) == ("lane.host_kv", "lane.host_kv_aux")


def test_glm_prefix_config_from_the_dual_split():
    primary, aux = dual.DualHostKVCoordinator.build_host_configs(
        "zai-org/GLM-5.3-FP8", BUDGET
    )
    config = build_prefix_cache_runtime_config(
        model_name="zai-org/GLM-5.3-FP8",
        kv_dtype="bfloat16",
        host_kv_config=primary,
        aux_host_kv_config=aux,
        shm_name="lane.prefix_cache",
    )

    assert [spec.semantic for spec in config.group_specs] == [
        PrefixKVGroupSemantic.MLA_LATENT,
        PrefixKVGroupSemantic.DSA_INDEXER_KEYS,
    ]
    assert config.max_nodes == primary.num_pages + 1
    assert config.max_page_handles == 2 * primary.num_pages
