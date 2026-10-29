import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_host_kv_config(monkeypatch):
    gpu_config = types.ModuleType(
        "batchgen.kv_cache.gpu_paged_kv_manager"
    )
    gpu_config.GPUPagedKVConfig = object
    native_loader = types.ModuleType("batchgen.native_loader")
    native_loader.core_engine = types.SimpleNamespace()
    monkeypatch.setitem(
        sys.modules,
        "batchgen.kv_cache.gpu_paged_kv_manager",
        gpu_config,
    )
    monkeypatch.setitem(
        sys.modules,
        "batchgen.native_loader",
        native_loader,
    )

    path = ROOT / "batchgen" / "kv_cache" / "host_kv_mananger_config.py"
    module_name = "_glm52_host_kv_config_for_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_glm52_uses_glm5_primary_and_its_own_indexer_host_kv_profile(monkeypatch):
    # GLM-5.2 shares the 78-layer MLA primary cache with GLM-5/5.1/5.3, but it
    # owns its aux (indexer) profile: only its 21 top-k computing layers hold
    # indexer keys, and they are stored in the page-split FP8 layout.
    config = _load_host_kv_config(monkeypatch)

    for model_name in (
        "zai-org/GLM-5.2-FP8",
        "zai-org/GLM-5.2",
        "GLM-5.2-FP8",
        "GLM-5.2",
    ):
        assert config._resolve_profile(model_name) is config._GLM5_MLA_PROFILE
        assert (
            config._resolve_indexer_profile(model_name)
            is config._GLM5_2_INDEXER_PROFILE
        )


def test_glm53_uses_glm5_primary_and_indexer_host_kv_profiles(monkeypatch):
    config = _load_host_kv_config(monkeypatch)

    for model_name in (
        "zai-org/GLM-5.3-FP8",
        "zai-org/GLM-5.3",
        "GLM-5.3-FP8",
        "GLM-5.3",
    ):
        assert config._resolve_profile(model_name) is config._GLM5_MLA_PROFILE
        assert (
            config._resolve_indexer_profile(model_name)
            is config._GLM5_INDEXER_PROFILE
        )


def test_only_glm52_aux_is_page_split_fp8(monkeypatch):
    """The page-split FP8 flip is GLM-5.2-aux-only.

    Every other profile — the GLM-5.x MLA primary, the GLM-5/5.1/5.3 indexer,
    DeepSeek-V3.2's indexer, the MLA and GQA caches — must stay bf16 AoS, so a
    reader of any of those regions keeps seeing token-major bf16 records.
    """

    config = _load_host_kv_config(monkeypatch)

    page_split = [
        name
        for name, profile in config._PROFILE_REGISTRY.items()
        if profile.page_format == config.PAGE_FORMAT_FP8_PAGE_SPLIT
    ]
    assert page_split == ["glm5_2_indexer"]

    for name, profile in config._PROFILE_REGISTRY.items():
        if name == "glm5_2_indexer":
            continue
        assert profile.page_format == config.PAGE_FORMAT_CONTIGUOUS_AOS, name
        assert profile.kv_dtype == "bfloat16", name
        assert not profile.is_page_split, name
        assert profile.k_data_dim() == profile.k_head_dim, name


def test_glm52_aux_page_identity(monkeypatch):
    """The exact bytes of the GLM-5.2 aux page and the pair's page identity."""

    config = _load_host_kv_config(monkeypatch)
    aux = config._GLM5_2_INDEXER_PROFILE
    primary = config._GLM5_MLA_PROFILE

    assert aux.kv_dtype == "float8_e4m3fn"
    assert aux.is_page_split
    assert aux.num_k_heads == 1 and aux.num_v_heads == 0
    assert aux.page_size == 64
    assert aux.num_layers == 21
    # k_head_dim is the per-token BYTE count: 128 e4m3 K bytes + one fp32
    # scale. The real indexer head dim stays 128.
    assert aux.k_head_dim == 132
    assert aux.k_data_dim() == 128
    assert aux.bytes_per_page() == 64 * 132 == 8448

    # Page identity of the mirrored pair (both regions share one page count).
    primary_bytes = primary.bytes_per_page() * primary.num_layers
    aux_bytes = aux.bytes_per_page() * aux.num_layers
    assert primary_bytes == 78 * 64 * 576 * 2 == 5_750_784
    assert aux_bytes == 21 * 8448 == 177_408
    assert primary_bytes + aux_bytes == 5_928_192


def test_page_split_profile_validation_rejects_mismatches(monkeypatch):
    import pytest as _pytest

    config = _load_host_kv_config(monkeypatch)
    base = dict(
        num_layers=21,
        num_k_heads=1,
        k_head_dim=132,
        num_v_heads=0,
        v_head_dim=0,
        kv_dtype="float8_e4m3fn",
        page_format=config.PAGE_FORMAT_FP8_PAGE_SPLIT,
    )
    # The declaration itself is the gate: a page-split profile cannot be bf16,
    # cannot have heads or a V cache, and its k_head_dim must be a byte count.
    config._HostKVModelProfile(**base)
    for override in (
        {"kv_dtype": "bfloat16"},
        {"num_k_heads": 2},
        {"num_v_heads": 1, "v_head_dim": 132},
        {"k_head_dim": 4},
        {"k_head_dim": 140},
        {"page_format": "something_else"},
    ):
        with _pytest.raises(ValueError):
            config._HostKVModelProfile(**{**base, **override})
