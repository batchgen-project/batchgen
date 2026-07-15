from __future__ import annotations

from batchgen.kv_cache.host_kv_mananger_config import (
    build_gpu_kv_config_from_group_profile,
    resolve_host_kv_group_profiles,
)


def test_deepseek_v4_group_profiles_capture_storage_and_raw_rates():
    profiles = resolve_host_kv_group_profiles("deepseek-v4-flash")

    assert [
        (
            profile.group_id,
            profile.group_name,
            profile.storage_page_tokens,
            profile.raw_page_tokens,
            profile.compression_ratio,
        )
        for profile in profiles
    ] == [
        (0, "swa", 64, 64, 1),
        (1, "compressor_c4", 64, 256, 4),
        (2, "compressor_c128", 2, 256, 128),
        (3, "indexer_c4", 64, 256, 4),
    ]


def test_compressed_group_gpu_config_uses_storage_page_capacity():
    profiles = {
        profile.group_name: profile
        for profile in resolve_host_kv_group_profiles("deepseek-v4-flash")
    }

    swa_config = build_gpu_kv_config_from_group_profile(profiles["swa"], [1024])
    c4_config = build_gpu_kv_config_from_group_profile(
        profiles["compressor_c4"], [1024]
    )
    c128_config = build_gpu_kv_config_from_group_profile(
        profiles["compressor_c128"], [1024]
    )

    assert swa_config.page_size_tokens == 64
    assert swa_config.num_pages == 17
    assert c4_config.page_size_tokens == 64
    assert c4_config.num_pages == 5
    assert c128_config.page_size_tokens == 2
    assert c128_config.num_pages == 5
