from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from batchgen.attention.dsa.v4_flashmla_adapter import (
    DeepSeekV4FlashMLADecodeAdapter,
    build_v4_decode_attn_metadata,
)
from batchgen.attention.v4_backend import (
    DeepseekV4AttnBackend,
    build_layer_configs_from_compress_ratios,
)
from batchgen.kv_cache.deepseek_v4_kv_coordinator import DeepSeekV4KVCoordinator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for V4 dense decode"
)

FLASHMLA_QUANT_PATH = Path("/root/FlashMLA_v4/tests/quant.py")


def _load_flashmla_quant_module():
    if not FLASHMLA_QUANT_PATH.exists():
        pytest.skip(f"FlashMLA quant reference missing: {FLASHMLA_QUANT_PATH}")
    spec = importlib.util.spec_from_file_location(
        "flashmla_quant_reference_dense", FLASHMLA_QUANT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec for {FLASHMLA_QUANT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_rope_cache(
    max_pos: int, rope_dim: int = 64, base: float = 10000.0
) -> torch.Tensor:
    device = torch.device("cuda")
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32)
            / rope_dim
        )
    )
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def _apply_rope_ref(
    x: torch.Tensor,
    positions: torch.Tensor,
    rope_cache: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    out = x.clone()
    half = 32
    rope = out[..., -64:].float().view(*out.shape[:-1], half, 2)
    cache = rope_cache.index_select(0, positions.long())
    view_shape = (positions.shape[0],) + (1,) * (rope.ndim - 3) + (half,)
    cos = cache[:, :half].view(view_shape)
    sin = cache[:, half:].view(view_shape)
    even = rope[..., 0]
    odd = rope[..., 1]
    if inverse:
        rot_even = even * cos + odd * sin
        rot_odd = odd * cos - even * sin
    else:
        rot_even = even * cos - odd * sin
        rot_odd = even * sin + odd * cos
    out[..., -64:] = (
        torch.stack((rot_even, rot_odd), dim=-1).flatten(-2).to(x.dtype)
    )
    return out


def _dense_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_sink: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    scores = (
        torch.einsum("bhd,btd->bht", q.float(), kv_cache.float())
        * softmax_scale
    )
    kv_len = kv_cache.size(1)
    key_pos = torch.arange(kv_len, device=kv_cache.device).unsqueeze(0)
    mask = key_pos >= cache_seqlens.to(kv_cache.device).long().unsqueeze(1)
    scores = scores.masked_fill(mask[:, None, :], float("-inf"))

    scores_max = scores.amax(dim=-1, keepdim=True)
    exp_scores = torch.exp(scores - scores_max)
    sink = torch.exp(attn_sink.float().view(1, -1, 1) - scores_max)
    weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + sink)
    return torch.einsum("bht,btd->bhd", weights.to(kv_cache.dtype), kv_cache)


def test_v4_dense_decode_matches_assets_reference():
    device = torch.device("cuda")
    quant_module = _load_flashmla_quant_module()
    batch_size = 2
    num_heads = 64
    head_dim = 512
    rope_head_dim = 64
    compress_ratios = [0, 4, 128]
    cache_seqlens = torch.tensor([3, 5], dtype=torch.int32, device=device)
    sequence_ids = [101, 202]
    max_seq_len = int(cache_seqlens.max().item())
    softmax_scale = head_dim**-0.5

    torch.manual_seed(0)
    coordinator = DeepSeekV4KVCoordinator(
        compress_ratios=compress_ratios,
        num_pages=8,
        device=device,
        base_page_size=256,
    )
    coordinator.initialize()

    try:
        coordinator.allocate_pages_for_sequences(
            sequence_ids, cache_seqlens.tolist()
        )
        page_tables = coordinator.rebuild_page_table(sequence_ids)

        rope_cache = _make_rope_cache(max_pos=max_seq_len + 4)
        positions = cache_seqlens.long() - 1
        history_raw = (
            torch.randn(
                batch_size,
                max_seq_len - 1,
                head_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            .div_(10)
            .clamp_(-1, 1)
        )
        current_kv = (
            torch.randn(
                batch_size, head_dim, dtype=torch.bfloat16, device=device
            )
            .div_(10)
            .clamp_(-1, 1)
        )
        q = torch.randn(
            batch_size,
            num_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + 1e-6)
        q = q.clamp_(-1, 1)
        attn_sink = torch.zeros(num_heads, dtype=torch.float32, device=device)

        full_cache = torch.zeros(
            batch_size,
            max_seq_len,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        for batch_idx, seq_id in enumerate(sequence_ids):
            hist_len = int(cache_seqlens[batch_idx].item()) - 1
            if hist_len > 0:
                hist_pos = torch.arange(
                    hist_len, device=device, dtype=torch.long
                )
                roped_history = _apply_rope_ref(
                    history_raw[batch_idx, :hist_len], hist_pos, rope_cache
                )
                full_cache[batch_idx, :hist_len] = roped_history
                slots = coordinator.swa.sequence_token_slots(seq_id, hist_pos)
                coordinator.swa.store_kv(
                    layer_idx=0,
                    token_slots=slots,
                    kv_processed=roped_history,
                )

            full_cache[batch_idx, hist_len] = _apply_rope_ref(
                current_kv[batch_idx : batch_idx + 1],
                positions[batch_idx : batch_idx + 1],
                rope_cache,
            )[0]

        current_roped = _apply_rope_ref(current_kv, positions, rope_cache)

        metadata = build_v4_decode_attn_metadata(
            coordinator=coordinator,
            sequence_ids=sequence_ids,
            cache_seqlens=cache_seqlens,
            positions=positions,
            page_tables=page_tables,
            rope_cache=rope_cache,
        )
        coordinator.swa.store_kv(
            layer_idx=0,
            token_slots=metadata.extras["swa_token_slots"],
            kv_processed=current_roped,
        )

        # Gate 1: pool pack/dequant correctness for the just-written prefix.
        k_cache, _, _ = coordinator.swa.get_layer_kv_with_page_table(0)
        model1_layout = quant_module.FP8KVCacheLayout.MODEL1_FP8Sparse
        for batch_idx, seq_id in enumerate(sequence_ids):
            page_id = int(coordinator.swa.get_sequence_pages(seq_id)[0].item())
            dequant_page = quant_module.dequantize_k_cache(
                k_cache[page_id : page_id + 1], model1_layout
            )[0, :, 0]
            seq_len = int(cache_seqlens[batch_idx].item())
            assert torch.allclose(
                dequant_page[:seq_len],
                full_cache[batch_idx, :seq_len],
                atol=0.05,
                rtol=0,
            )

            # Gate 2: page bytes must match FlashMLA's official MODEL1 quantizer.
            expected_page = torch.zeros(
                (1, coordinator.swa.page_size_tokens, 1, head_dim),
                dtype=torch.bfloat16,
                device=device,
            )
            expected_page[0, :seq_len, 0] = full_cache[batch_idx, :seq_len]
            expected_quant = quant_module.quantize_k_cache(
                expected_page, model1_layout
            )
            actual_flat = k_cache[page_id].view(torch.uint8).view(-1)
            expected_flat = expected_quant[0].view(torch.uint8).view(-1)
            body_bytes = coordinator.swa.page_size_tokens * 576
            assert torch.equal(
                actual_flat[:body_bytes],
                expected_flat[:body_bytes],
            )
            assert torch.equal(
                actual_flat[body_bytes:].view(
                    coordinator.swa.page_size_tokens, 8
                )[:, :7],
                expected_flat[body_bytes:].view(
                    coordinator.swa.page_size_tokens, 8
                )[:, :7],
            )

        layer_config = build_layer_configs_from_compress_ratios(
            compress_ratios=compress_ratios,
            n_heads=num_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
        )[0]
        backend = DeepseekV4AttnBackend(
            layer_configs=[layer_config],
            page_size=coordinator.swa.page_size_tokens,
            flashmla_backend=DeepSeekV4FlashMLADecodeAdapter(coordinator),
        )
        backend.init_metadata(metadata)

        actual = backend.forward(
            layer_config=layer_config,
            q=q,
            kv=current_kv,
            attn_sink=attn_sink,
            softmax_scale=512**-0.5,
        )

        q_roped = _apply_rope_ref(q, positions, rope_cache)
        expected = _dense_reference(
            q=q_roped,
            kv_cache=full_cache,
            attn_sink=attn_sink,
            cache_seqlens=cache_seqlens,
            softmax_scale=512**-0.5,
        )
        expected = _apply_rope_ref(
            expected, positions, rope_cache, inverse=True
        )

        assert actual.shape == (batch_size, num_heads, head_dim)
        assert torch.allclose(actual, expected, atol=0.05, rtol=0)
    finally:
        coordinator.destroy()
