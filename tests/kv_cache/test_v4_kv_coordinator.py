from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
import torch

from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator
from batchgen.kv_cache.deepseek_v4_single_kv_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
)
from batchgen_kernels.attention.v4_fused_qnorm_rope_kv import (
    HEAD_DIM,
    TOKEN_BYTES,
)

FLASHMLA_QUANT_PATH = Path("/root/FlashMLA_v4/tests/quant.py")


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for DeepSeek-V4 KV tests")
    return torch.device("cuda")


def _load_flashmla_quant_module() -> ModuleType:
    if not FLASHMLA_QUANT_PATH.exists():
        pytest.skip(f"FlashMLA quant reference missing: {FLASHMLA_QUANT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "flashmla_quant_reference", FLASHMLA_QUANT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {FLASHMLA_QUANT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _identity_cos_sin_cache(
    device: torch.device, max_pos: int = 8
) -> torch.Tensor:
    cache = torch.zeros((max_pos, 64, 2), dtype=torch.bfloat16, device=device)
    cache[..., 0] = 1
    return cache


def _reference_v4_roundtrip(
    quant_module: ModuleType,
    token: torch.Tensor,
    *,
    page_size: int,
    offset: int,
) -> torch.Tensor:
    blocked = torch.zeros(
        (1, page_size, 1, HEAD_DIM), dtype=torch.bfloat16, device=token.device
    )
    blocked[0, offset, 0] = token
    quantized = quant_module.quantize_k_cache(
        blocked, quant_module.FP8KVCacheLayout.MODEL1_FP8Sparse
    )
    dequantized = quant_module.dequantize_k_cache(
        quantized, quant_module.FP8KVCacheLayout.MODEL1_FP8Sparse
    )
    return dequantized[0, offset, 0]


def _reference_indexer_roundtrip(token: torch.Tensor) -> torch.Tensor:
    scale = torch.abs(token.float()).amax(dim=-1) / 448.0
    scale = torch.clamp_min(scale, 1e-4)
    quantized = (token.float() / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return (quantized.float() * scale.unsqueeze(-1)).to(torch.bfloat16)


@pytest.fixture()
def coordinator() -> DeepSeekV4KVCoordinator:
    device = _require_cuda()
    coord = DeepSeekV4KVCoordinator(
        compress_ratios=[0, 4, 128],
        num_pages=8,
        device=device,
        base_page_size=256,
    )
    coord.initialize()
    yield coord
    coord.destroy()


def test_v4_pool_roundtrip_matches_flashmla_reference(
    coordinator: DeepSeekV4KVCoordinator,
):
    quant_module = _load_flashmla_quant_module()
    device = _require_cuda()
    sequence_id = 17
    coordinator.allocate_pages_for_sequences([sequence_id], [2])
    coordinator.rebuild_page_table([sequence_id])

    q = torch.randn((1, HEAD_DIM), dtype=torch.bfloat16, device=device).clamp_(
        -1, 1
    )
    kv = torch.randn((1, HEAD_DIM), dtype=torch.bfloat16, device=device).clamp_(
        -1, 1
    )
    kv_weight = torch.ones((HEAD_DIM,), dtype=torch.bfloat16, device=device)
    cos_sin_cache = _identity_cos_sin_cache(device)
    positions = torch.tensor([0], dtype=torch.int64, device=device)

    swa_slots = coordinator.swa.sequence_token_slots(sequence_id, [0])
    _, swa_processed = coordinator.swa.store_qnorm_rope_kv(
        layer_idx=0,
        token_slots=swa_slots,
        q=q,
        kv=kv,
        kv_weight=kv_weight,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
    )
    swa_read = coordinator.swa.debug_read_kv(
        layer_idx=0, token_slots=swa_slots
    )[0]
    swa_expected = _reference_v4_roundtrip(
        quant_module,
        swa_processed[0],
        page_size=coordinator.swa.page_size_tokens,
        offset=0,
    )
    assert torch.allclose(swa_read, swa_expected, atol=0.05, rtol=0)

    c4_route = coordinator.get_layer_routing(1)
    assert c4_route.c4_layer_idx is not None
    c4_slots = coordinator.c4.sequence_token_slots(sequence_id, [1])
    _, c4_processed = coordinator.c4.store_qnorm_rope_kv(
        layer_idx=c4_route.c4_layer_idx,
        token_slots=c4_slots,
        q=q,
        kv=kv,
        kv_weight=kv_weight,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
    )
    c4_read = coordinator.c4.debug_read_kv(
        layer_idx=c4_route.c4_layer_idx,
        token_slots=c4_slots,
    )[0]
    c4_expected = _reference_v4_roundtrip(
        quant_module,
        c4_processed[0],
        page_size=coordinator.c4.page_size_tokens,
        offset=1,
    )
    assert torch.allclose(c4_read, c4_expected, atol=0.05, rtol=0)

    c128_route = coordinator.get_layer_routing(2)
    assert c128_route.c128_layer_idx is not None
    c128_slots = coordinator.c128.sequence_token_slots(sequence_id, [1])
    _, c128_processed = coordinator.c128.store_qnorm_rope_kv(
        layer_idx=c128_route.c128_layer_idx,
        token_slots=c128_slots,
        q=q,
        kv=kv,
        kv_weight=kv_weight,
        cos_sin_cache=cos_sin_cache,
        positions=positions,
    )
    c128_read = coordinator.c128.debug_read_kv(
        layer_idx=c128_route.c128_layer_idx,
        token_slots=c128_slots,
    )[0]
    c128_expected = _reference_v4_roundtrip(
        quant_module,
        c128_processed[0],
        page_size=coordinator.c128.page_size_tokens,
        offset=1,
    )
    assert torch.allclose(c128_read, c128_expected, atol=0.05, rtol=0)

    index_token = torch.randn(
        (1, coordinator.indexer_head_dim), dtype=torch.bfloat16, device=device
    ).clamp_(-1, 1)
    index_slots = coordinator.indexer.sequence_token_slots(sequence_id, [1])
    assert c4_route.indexer_layer_idx is not None
    coordinator.indexer.store_indexer(
        layer_idx=c4_route.indexer_layer_idx,
        token_slots=index_slots,
        index_k=index_token,
    )
    index_read = coordinator.indexer.debug_read_indexer(
        layer_idx=c4_route.indexer_layer_idx,
        token_slots=index_slots,
    )
    index_expected = _reference_indexer_roundtrip(index_token)
    assert torch.allclose(index_read, index_expected, atol=0.05, rtol=0)


def test_v4_fp8_layout_is_584_bytes_per_token(
    coordinator: DeepSeekV4KVCoordinator,
):
    coordinator.allocate_pages_for_sequences([1], [1])
    coordinator.rebuild_page_table([1])
    assert TOKEN_BYTES == 584
    assert DeepSeekV4SingleKVPool.bytes_per_token == 584
    assert coordinator.swa.config.bytes_per_token == 584
    expected_padded = (
        (coordinator.swa.page_size_tokens * 584 + 575) // 576
    ) * 576
    assert coordinator.swa.bytes_per_page_padded == expected_padded
    k_cache, _, _ = coordinator.swa.get_layer_kv_with_page_table(0)
    assert k_cache.shape == (
        coordinator.swa.num_pages,
        coordinator.swa.page_size_tokens,
        1,
        584,
    )
    assert k_cache.stride()[0] == coordinator.swa.bytes_per_page_padded
    assert DeepSeekV4IndexerPool.bytes_per_token == 132


def test_v4_layer_routing():
    coord = DeepSeekV4KVCoordinator(
        compress_ratios=[0, 4, 128, 4],
        num_pages=4,
        device=_require_cuda(),
        base_page_size=256,
    )
    try:
        route0 = coord.get_layer_routing(0)
        assert route0.swa_layer_idx == 0
        assert route0.c4_layer_idx is None
        assert route0.c128_layer_idx is None
        assert route0.indexer_layer_idx is None

        route1 = coord.get_layer_routing(1)
        assert route1.swa_layer_idx == 1
        assert route1.c4_layer_idx == 0
        assert route1.c128_layer_idx is None
        assert route1.indexer_layer_idx == 0

        route2 = coord.get_layer_routing(2)
        assert route2.swa_layer_idx == 2
        assert route2.c4_layer_idx is None
        assert route2.c128_layer_idx == 0
        assert route2.indexer_layer_idx is None
    finally:
        coord.destroy()


def test_v4_page_size_must_be_divisible_by_128():
    with pytest.raises(ValueError, match="divisible by 128"):
        DeepSeekV4KVCoordinator(
            compress_ratios=[0, 4, 128],
            num_pages=4,
            device=_require_cuda(),
            base_page_size=64,
        )


def test_v4_decode_resident_guard_raises(coordinator: DeepSeekV4KVCoordinator):
    with pytest.raises(RuntimeError, match="GPU-resident only"):
        coordinator.copy_kv_to_tensor(1)
    with pytest.raises(RuntimeError, match="GPU-resident only"):
        coordinator.async_offload_layer_kv_to_host(layer_idx=0)


def test_v4_compressed_pools_charged_in_compressed_token_space(
    coordinator: DeepSeekV4KVCoordinator,
):
    coordinator.allocate_pages_for_sequences([1], [1024])
    swa_pages = coordinator.swa.get_sequence_pages(1).numel()
    c4_pages = coordinator.c4.get_sequence_pages(1).numel()
    c128_pages = coordinator.c128.get_sequence_pages(1).numel()
    indexer_pages = coordinator.indexer.get_sequence_pages(1).numel()

    # swa keeps raw tokens: ceil(1024/128)=8. Compressed pools store raw/ratio
    # rows: c4/indexer ceil(1024/4 / 64)=4, c128 ceil(1024/128 / 2)=4. The old
    # raw-token bug allocated c128=ceil(1024/2)=512.
    assert swa_pages == 8
    assert c4_pages == 4
    assert indexer_pages == 4
    assert c128_pages == 4


def test_v4_preflight_false_when_one_pool_exhausted(
    coordinator: DeepSeekV4KVCoordinator,
):
    # num_pages=8 per pool. Drain the swa pool with a long raw context while the
    # other pools still have room, so the SUMMED free count stays positive but
    # the per-pool preflight must report False.
    coordinator.allocate_pages_for_sequences([1], [1024])
    assert coordinator.swa.get_stats().num_free_pages == 0
    summed_free = coordinator.get_stats().num_free_pages
    assert summed_free > 0
    assert coordinator.can_allocate_pages_for_sequences([2], [256]) is False


def test_v4_preflight_true_when_all_pools_have_room(
    coordinator: DeepSeekV4KVCoordinator,
):
    assert coordinator.can_allocate_pages_for_sequences([1], [256]) is True
    coordinator.allocate_pages_for_sequences([1], [256])
    assert coordinator.can_allocate_pages_for_sequences([2], [256]) is True


def test_v4_free_worker_pages_reflects_binding_pool(
    coordinator: DeepSeekV4KVCoordinator,
):
    # Empty coordinator: swa binds at 8 pages * 128 tok = 1024 raw tokens =
    # 16 worker pages (64-token). Other pools cover more raw tokens, so the
    # binding (min) is swa.
    assert coordinator.free_worker_pages(64) == 16


def test_v4_tracked_sequence_ids_filters_unknown(
    coordinator: DeepSeekV4KVCoordinator,
):
    coordinator.allocate_pages_for_sequences([7], [256])
    assert coordinator.tracked_sequence_ids([7, 99]) == [7]
    assert coordinator.tracked_sequence_ids([99]) == []
    # Freeing only the tracked id must not raise on the unknown one.
    coordinator.free_pages_for_sequences(
        coordinator.tracked_sequence_ids([7, 99])
    )
    assert coordinator.tracked_sequence_ids([7]) == []
