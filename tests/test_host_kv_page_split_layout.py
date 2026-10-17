"""The host-KV page-format discriminator and its page-split geometry.

CPU contract tests against the core engine's own layout code (no device, no
shared memory): ``describe_host_kv_page_geometry`` and ``host_kv_config_hash``
are pure functions of a config, so the page arithmetic the C++ copy paths use
is pinned here instead of only on a node.

Checked:
* the GLM-5.2 aux page is [64 x 128 B e4m3][64 x 4 B fp32] = 8,448 B, and the
  two regions tile the page exactly;
* an independent Python address model agrees with the reported geometry for a
  sweep of (layer, page, in-page offset);
* flipping ONLY the page format changes the config hash and the layout
  fingerprint, so a stale shared-memory region can never be reused across the
  migration;
* a contiguous-AoS config keeps its old geometry and has no scale region;
* the fail-loud gate rejects a page-split config that is not single-head e4m3
  with a byte-count k_head_dim and no V cache.
"""

from __future__ import annotations

import pytest


def _core_engine():
    try:
        from batchgen.models.engine_loader import core_engine
    except Exception as exc:  # pragma: no cover - depends on the build
        pytest.skip(f"core_engine extension unavailable: {exc}")
    if not hasattr(core_engine, "describe_host_kv_page_geometry"):
        pytest.skip(
            "core_engine build predates HostKVPageFormat; rebuild core_engine"
        )
    return core_engine


def _aux_config(core_engine, *, page_split: bool = True):
    config = core_engine.HostPagedKVConfig()
    config.shm_name = "test_aux_page_split"
    config.num_layers = 21
    config.num_pages = 3103
    config.page_size_tokens = 64
    config.num_k_heads = 1
    config.num_v_heads = 0
    config.v_head_dim = 0
    config.v_element_size_bytes = 0
    config.sequence_table_capacity = 3103
    config.alignment_bytes = 64
    if page_split:
        config.k_head_dim = 132  # 128 data bytes + one fp32 scale
        config.k_element_size_bytes = 1
        config.page_format = core_engine.HostKVPageFormat.PAGE_SPLIT_FP8_SCALE
    else:
        config.k_head_dim = 128
        config.k_element_size_bytes = 2
        config.page_format = core_engine.HostKVPageFormat.CONTIGUOUS_AOS
    return config


def test_page_split_geometry_is_exact():
    core_engine = _core_engine()
    geometry = core_engine.describe_host_kv_page_geometry(
        _aux_config(core_engine)
    )

    assert geometry["is_page_split"] is True
    assert geometry["has_v_cache"] is False
    assert geometry["k_page_bytes"] == 8448
    assert geometry["k_data_token_bytes"] == 128
    assert geometry["scale_token_bytes"] == 4
    assert geometry["scale_region_offset"] == 64 * 128 == 8192
    # The regions tile the page: no padding, no overlap.
    assert (
        geometry["scale_region_offset"] + 64 * geometry["scale_token_bytes"]
        == geometry["k_page_bytes"]
    )
    assert geometry["layer_stride_bytes"] == 3103 * 8448
    assert geometry["data_section_bytes"] == 21 * 3103 * 8448


def test_page_split_address_model():
    core_engine = _core_engine()
    config = _aux_config(core_engine)
    geometry = core_engine.describe_host_kv_page_geometry(config)

    layer_stride = geometry["layer_stride_bytes"]
    page_bytes = geometry["k_page_bytes"]
    data_bytes = geometry["k_data_token_bytes"]
    scale_bytes = geometry["scale_token_bytes"]
    scale_offset = geometry["scale_region_offset"]

    for layer in (0, 7, 20):
        for page in (0, 1, 3102):
            for offset in (0, 1, 63):
                page_base = layer * layer_stride + page * page_bytes
                # The address a writer/reader of this page must use.
                assert page_base + offset * data_bytes < page_base + scale_offset
                assert (
                    page_base + scale_offset + offset * scale_bytes
                    < page_base + page_bytes
                )
                # The data record never runs into the scale region, and the
                # scale never runs past the page.
                assert (
                    offset * data_bytes + data_bytes <= scale_offset
                )
                assert (
                    scale_offset + offset * scale_bytes + scale_bytes
                    <= page_bytes
                )


def test_contiguous_aos_geometry_is_unchanged():
    core_engine = _core_engine()
    geometry = core_engine.describe_host_kv_page_geometry(
        _aux_config(core_engine, page_split=False)
    )

    assert geometry["is_page_split"] is False
    assert geometry["k_page_bytes"] == 64 * 128 * 2 == 16384
    assert geometry["k_data_token_bytes"] == 128 * 2
    assert geometry["scale_token_bytes"] == 0
    assert geometry["scale_region_offset"] == 0


def test_page_format_flips_the_config_hash_and_fingerprint():
    core_engine = _core_engine()
    split = _aux_config(core_engine)

    # Same dimensions, same element size, ONLY the format differs.
    same_dims = _aux_config(core_engine)
    same_dims.page_format = core_engine.HostKVPageFormat.CONTIGUOUS_AOS

    assert core_engine.host_kv_config_hash(split) != core_engine.host_kv_config_hash(
        same_dims
    )
    split_geometry = core_engine.describe_host_kv_page_geometry(split)
    aos_geometry = core_engine.describe_host_kv_page_geometry(same_dims)
    assert split_geometry["fingerprint"] != aos_geometry["fingerprint"]
    # The page identity itself does not move: only the bytes inside it do.
    assert split_geometry["k_page_bytes"] == aos_geometry["k_page_bytes"]
    assert (
        split_geometry["data_section_bytes"] == aos_geometry["data_section_bytes"]
    )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda c: setattr(c, "k_element_size_bytes", 2), id="bf16_elements"
        ),
        pytest.param(lambda c: setattr(c, "num_k_heads", 2), id="multi_head"),
        pytest.param(lambda c: setattr(c, "k_head_dim", 4), id="no_data_bytes"),
        pytest.param(
            lambda c: setattr(c, "k_head_dim", 140), id="unaligned_data_bytes"
        ),
    ],
)
def test_page_split_rejects_invalid_configs(mutate):
    core_engine = _core_engine()
    config = _aux_config(core_engine)
    mutate(config)
    with pytest.raises(Exception):
        core_engine.describe_host_kv_page_geometry(config)


def test_page_split_rejects_a_v_cache():
    core_engine = _core_engine()
    config = _aux_config(core_engine)
    config.num_v_heads = 1
    config.v_head_dim = 132
    config.v_element_size_bytes = 1
    with pytest.raises(Exception):
        core_engine.describe_host_kv_page_geometry(config)
