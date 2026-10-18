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
    engine_loader = types.ModuleType("batchgen.models.engine_loader")
    engine_loader.core_engine = types.SimpleNamespace()
    monkeypatch.setitem(
        sys.modules,
        "batchgen.kv_cache.gpu_paged_kv_manager",
        gpu_config,
    )
    monkeypatch.setitem(
        sys.modules,
        "batchgen.models.engine_loader",
        engine_loader,
    )

    path = ROOT / "batchgen" / "kv_cache" / "host_kv_mananger_config.py"
    module_name = "_glm52_host_kv_config_for_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


GLM52_ALIASES = (
    "zai-org/GLM-5.2-FP8",
    "zai-org/GLM-5.2",
    "GLM-5.2-FP8",
    "GLM-5.2",
)


def test_glm52_owns_both_a_packed_primary_and_its_own_indexer_profile(monkeypatch):
    # GLM-5.2 no longer shares the bf16 78-layer MLA primary cache with
    # GLM-5/5.1/5.3: its primary moves to FlashMLA's packed FP8 record, and it
    # already owned its aux (indexer) profile, where only its 21 top-k
    # computing layers hold keys, in the page-split FP8 layout.
    config = _load_host_kv_config(monkeypatch)

    for model_name in GLM52_ALIASES:
        assert (
            config._resolve_profile(model_name)
            is config._GLM5_2_MLA_PACKED_PROFILE
        )
        assert (
            config._resolve_indexer_profile(model_name)
            is config._GLM5_2_INDEXER_PROFILE
        )


def test_glm52_packed_kv_kill_switch_restores_the_bf16_primary(monkeypatch):
    """With the switch off, resolution is EXACTLY the pre-migration mapping.

    The rollback is a profile selection, not a code path: the bf16 entry stays
    in `_PROFILE_ALIASES` and the packed profile is an override applied only
    while the switch is on. The aux cache is a separate migration and is NOT
    affected by this switch.
    """

    config = _load_host_kv_config(monkeypatch)
    assert config.glm52_packed_kv_enabled() is True
    config.set_glm52_packed_kv(False)
    assert config.glm52_packed_kv_enabled() is False

    for model_name in GLM52_ALIASES:
        assert config._resolve_profile(model_name) is config._GLM5_MLA_PROFILE
        assert (
            config._resolve_indexer_profile(model_name)
            is config._GLM5_2_INDEXER_PROFILE
        )


def test_glm52_packed_kv_switch_latches_after_the_first_resolution(monkeypatch):
    """Two geometries in one process would silently mis-decode pages.

    The GPU pool, the host shm region, the staging buffers and the
    prefix-reuse spans are all sized from this resolution, so a flip after any
    of them exists must raise rather than hand out a second layout.
    """

    import pytest as _pytest

    config = _load_host_kv_config(monkeypatch)
    assert (
        config._resolve_profile("glm-5.2")
        is config._GLM5_2_MLA_PACKED_PROFILE
    )
    # Re-asserting the value already in force stays a no-op: initializers may
    # set it more than once.
    config.set_glm52_packed_kv(True)
    with _pytest.raises(RuntimeError, match="already committed"):
        config.set_glm52_packed_kv(False)
    # And the committed geometry is unchanged by the refused flip.
    assert (
        config._resolve_profile("glm-5.2")
        is config._GLM5_2_MLA_PACKED_PROFILE
    )


def test_glm52_packed_primary_page_identity(monkeypatch):
    """The exact FlashMLA packed-record bytes and the migrated pair identity."""

    config = _load_host_kv_config(monkeypatch)
    primary = config._GLM5_2_MLA_PACKED_PROFILE
    aux = config._GLM5_2_INDEXER_PROFILE

    # The record FlashMLA hard-checks: 512 e4m3 NoPE + 4 fp32 block scales +
    # 64 bf16 RoPE = 656 B, page_block_size 64 (csrc/pybind.cpp:236/249-256).
    assert config.PACKED_MLA_NOPE_DIM == 512
    assert config.PACKED_MLA_ROPE_DIM == 64
    assert config.PACKED_MLA_SCALE_BYTES == 16
    assert config.PACKED_MLA_TOKEN_BYTES == 512 + 16 + 128 == 656
    assert config.PACKED_MLA_SOURCE_DIM == 576

    assert primary.packed_mla_fp8 is True
    assert primary.kv_dtype == "uint8"
    assert primary.k_head_dim == 656
    assert primary.page_size == 64
    assert primary.num_layers == 78
    assert primary.num_k_heads == 1 and primary.num_v_heads == 0
    # Packed is ordinary token-major AoS; only a token's byte count changes.
    assert primary.page_format == config.PAGE_FORMAT_CONTIGUOUS_AOS
    assert not primary.is_page_split
    assert primary.k_data_dim() == 656
    assert primary.bytes_per_page() == 64 * 656 == 41_984

    primary_bytes = primary.bytes_per_page() * primary.num_layers
    aux_bytes = aux.bytes_per_page() * aux.num_layers
    assert primary_bytes == 78 * 41_984 == 3_274_752
    assert aux_bytes == 21 * 8_448 == 177_408
    # The page-count denominator the worker's budget split divides by
    # (batchgen_worker.py `combined_bytes_per_page`).
    assert primary_bytes + aux_bytes == 3_452_160

def test_packed_pair_page_count_identity(monkeypatch):
    """The exact page-count identity the PR must state.

    `batchgen_worker.py` sizes the GPU pool as
    ``num_pages = floor(budget_bytes / combined_bytes_per_page)`` with
    ``combined = primary.bytes_per_page()*78 + aux.bytes_per_page()*21``, so
    the only thing the migration changes is that denominator. Both FP8 steps
    are pinned here against the pre-migration pair so a profile edit that
    moves either one fails with the arithmetic in front of the reader.
    """

    config = _load_host_kv_config(monkeypatch)
    bf16_primary = config._GLM5_MLA_PROFILE
    bf16_aux = config._GLM5_INDEXER_PROFILE
    fp8_aux = config._GLM5_2_INDEXER_PROFILE
    packed_primary = config._GLM5_2_MLA_PACKED_PROFILE

    # GLM-5.2 has 21 indexer layers against the primary's 78 (its 57
    # skip_topk layers reuse the previous full layer's top-k).
    aux_layers = 21
    assert fp8_aux.num_layers == aux_layers

    pre = bf16_primary.bytes_per_page() * 78 + bf16_aux.bytes_per_page() * aux_layers
    after_aux = bf16_primary.bytes_per_page() * 78 + fp8_aux.bytes_per_page() * aux_layers
    after_both = (
        packed_primary.bytes_per_page() * 78 + fp8_aux.bytes_per_page() * aux_layers
    )

    assert pre == 5_750_784 + 344_064 == 6_094_848
    assert after_aux == 5_750_784 + 177_408 == 5_928_192
    assert after_both == 3_274_752 + 177_408 == 3_452_160

    # Page factor against the pre-migration pair, and pages per GiB.
    assert round(pre / after_both, 4) == 1.7655
    gib = 1024**3
    assert gib // pre == 176
    assert gib // after_both == 311

    # And the identity at the reference budget: every byte budget that bought
    # exactly 3,019 pre-migration pages buys 5,330 or 5,331 packed ones.
    lo = 3019 * pre
    hi = 3020 * pre - 1
    assert lo // after_both == 5_330
    assert hi // after_both == 5_331


def test_packed_mla_profile_validation_rejects_mismatches(monkeypatch):
    """Every geometry FlashMLA hard-checks is rejected at declaration time."""

    import pytest as _pytest

    config = _load_host_kv_config(monkeypatch)
    base = dict(
        num_layers=78,
        num_k_heads=1,
        k_head_dim=656,
        num_v_heads=0,
        v_head_dim=0,
        kv_dtype="uint8",
        page_size=64,
        packed_mla_fp8=True,
    )
    config._HostKVModelProfile(**base)
    for override in (
        {"kv_dtype": "bfloat16"},
        {"kv_dtype": "float8_e4m3fn"},
        {"k_head_dim": 576},
        {"k_head_dim": 640},
        {"page_size": 32},
        {"page_size": 128},
        {"num_k_heads": 2},
        {"num_v_heads": 1, "v_head_dim": 656},
        {"page_format": config.PAGE_FORMAT_FP8_PAGE_SPLIT},
    ):
        with _pytest.raises(ValueError):
            config._HostKVModelProfile(**{**base, **override})


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


def test_only_glm52_profiles_leave_bf16_aos(monkeypatch):
    """Both FP8 flips are GLM-5.2-only, and they are the ONLY two.

    `glm5_2_indexer` is page-split FP8 (aux); `glm5_2_mla_packed` is FlashMLA's
    packed FP8 record (primary). Every other profile — the GLM-5/5.1/5.3 MLA
    primary, the GLM-5/5.1/5.3 indexer, DeepSeek-V3.2's indexer, the MLA and
    GQA caches — must stay bf16 AoS, so a reader of any of those regions keeps
    seeing token-major bf16 records.
    """

    config = _load_host_kv_config(monkeypatch)

    page_split = [
        name
        for name, profile in config._PROFILE_REGISTRY.items()
        if profile.page_format == config.PAGE_FORMAT_FP8_PAGE_SPLIT
    ]
    assert page_split == ["glm5_2_indexer"]

    packed = [
        name
        for name, profile in config._PROFILE_REGISTRY.items()
        if profile.packed_mla_fp8
    ]
    assert packed == ["glm5_2_mla_packed"]

    migrated = {"glm5_2_indexer", "glm5_2_mla_packed"}
    for name, profile in config._PROFILE_REGISTRY.items():
        if name in migrated:
            continue
        assert profile.page_format == config.PAGE_FORMAT_CONTIGUOUS_AOS, name
        assert profile.kv_dtype == "bfloat16", name
        assert not profile.is_page_split, name
        assert not profile.packed_mla_fp8, name
        assert profile.k_data_dim() == profile.k_head_dim, name


def test_non_glm52_models_keep_their_exact_primary_geometry(monkeypatch):
    """Collateral gate: the packed flip must not move any other model.

    Asserted as the full (dtype, k_head_dim, page_size, num_layers) tuple per
    alias family, so a future alias edit that re-points one of these at a
    migrated profile fails here rather than at a node run.
    """

    config = _load_host_kv_config(monkeypatch)

    expected = {
        "GLM-5": ("bfloat16", 576, 64, 78),
        "zai-org/GLM-5-FP8": ("bfloat16", 576, 64, 78),
        "GLM-5.1": ("bfloat16", 576, 64, 78),
        "GLM-5.3": ("bfloat16", 576, 64, 78),
        "zai-org/GLM-5.3-FP8": ("bfloat16", 576, 64, 78),
        "deepseek-v3.2": ("bfloat16", 576, 64, 61),
        "deepseek-v3": ("bfloat16", 576, 64, 61),
        "deepseek-v4-flash": ("bfloat16", 512, 64, 43),
        "deepseek-v4-pro": ("bfloat16", 512, 64, 61),
        "gpt-oss-120b": ("bfloat16", 64, 64, 36),
        "minimax-m2.5": ("bfloat16", 128, 64, 62),
        "kimi-linear": ("bfloat16", 576, 64, 27),
        "kimi-k3": ("bfloat16", 576, 64, 24),
    }
    for model_name, want in expected.items():
        profile = config._resolve_profile(model_name)
        got = (
            profile.kv_dtype,
            profile.k_head_dim,
            profile.page_size,
            profile.num_layers,
        )
        assert got == want, (model_name, got, want)

    # DeepSeek-V3.2's indexer is likewise untouched by the GLM-5.2 aux flip.
    dsv32_aux = config._resolve_indexer_profile("deepseek-v3.2")
    assert dsv32_aux is config._DEEPSEEK_V3_2_INDEXER_PROFILE
    assert dsv32_aux.kv_dtype == "bfloat16"
    assert dsv32_aux.k_head_dim == 128


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
