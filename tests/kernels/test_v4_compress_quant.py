# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import math

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# MXFP4 cvt.e2m1x2 PTX is rejected by ptxas on sm_90a; sm120+ only.
requires_mxfp4 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 12,
    reason="MXFP4 requires sm120+ (cvt.e2m1x2 unsupported on sm_90a)",
)

SPARSE_HEAD_SIZE = 512
INDEXER_HEAD_SIZE = 128
ROPE_DIM = 64
SPARSE_NOPE_DIM = 448
SPARSE_QUANT_BLOCK = 64
SPARSE_TOKEN_STRIDE = 576
SPARSE_SCALE_DIM = 8
MXFP4_BLOCK_SIZE = 32
INDEXER_MXFP4_TOKEN_STRIDE = 64
INDEXER_MXFP4_SCALE_DIM = 4
FP8_MAX = 448.0


def _make_cos_sin_cache(
    max_pos: int, rope_dim: int = ROPE_DIM, device: str = "cuda"
) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32)
            / rope_dim
        )
    )
    positions = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


def _make_sparse_cache(num_blocks: int, block_size: int) -> torch.Tensor:
    return torch.zeros(
        (
            num_blocks,
            block_size * SPARSE_TOKEN_STRIDE + block_size * SPARSE_SCALE_DIM,
        ),
        dtype=torch.uint8,
        device="cuda",
    )


def _make_mxfp4_cache(num_blocks: int, block_size: int) -> torch.Tensor:
    return torch.zeros(
        (
            num_blocks,
            block_size * INDEXER_MXFP4_TOKEN_STRIDE
            + block_size * INDEXER_MXFP4_SCALE_DIM,
        ),
        dtype=torch.uint8,
        device="cuda",
    )


def _load_state_row(
    state_cache: torch.Tensor,
    block_table: torch.Tensor,
    req_idx: int,
    pos: int,
    block_size: int,
) -> torch.Tensor:
    physical_block = int(block_table[req_idx, pos // block_size].item())
    return state_cache[physical_block, pos % block_size].float()


def _compress_norm_ref(
    state_cache: torch.Tensor,
    block_table: torch.Tensor,
    req_idx: int,
    position: int,
    rms_norm_weight: torch.Tensor,
    *,
    head_dim: int,
    block_size: int,
    compress_ratio: int,
    overlap: int,
    eps: float,
) -> torch.Tensor:
    window = []
    scores = []
    start = position - (1 + overlap) * compress_ratio + 1
    state_width = head_dim * 2
    for token_idx in range((1 + overlap) * compress_ratio):
        pos = start + token_idx
        if pos < 0:
            window.append(
                torch.zeros(head_dim, device="cuda", dtype=torch.float32)
            )
            scores.append(torch.full((head_dim,), float("-inf"), device="cuda"))
            continue
        row = _load_state_row(
            state_cache, block_table, req_idx, pos, block_size
        )
        head_offset = head_dim if token_idx >= compress_ratio else 0
        window.append(row[head_offset : head_offset + head_dim])
        scores.append(
            row[
                state_width + head_offset : state_width + head_offset + head_dim
            ]
        )
    score = torch.softmax(torch.stack(scores, dim=0), dim=0)
    compressed = (torch.stack(window, dim=0) * score).sum(dim=0)
    return (
        compressed
        * torch.rsqrt(compressed.square().mean() + eps)
        * rms_norm_weight.float()
    )


def _rope_ref(
    x: torch.Tensor,
    compressed_pos: int,
    cos_sin_cache: torch.Tensor,
    rope_dim: int = ROPE_DIM,
) -> torch.Tensor:
    out = x.float().clone()
    nope_dim = out.numel() - rope_dim
    half = rope_dim // 2
    rope = out[nope_dim:].view(half, 2)
    cos = cos_sin_cache[compressed_pos, :half]
    sin = cos_sin_cache[compressed_pos, half:]
    even = rope[:, 0]
    odd = rope[:, 1]
    out[nope_dim:] = (
        torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
        .flatten()
        .to(torch.bfloat16)
        .float()
    )
    return out


def _sparse_cache_ref(
    state_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    block_size: int,
    compress_ratio: int,
    overlap: int,
    eps: float,
) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    out = []
    for token_idx, position in enumerate(positions.tolist()):
        if (position + 1) % compress_ratio != 0:
            continue
        normed = _compress_norm_ref(
            state_cache,
            block_table,
            int(token_to_req_indices[token_idx].item()),
            int(position),
            rms_norm_weight,
            head_dim=SPARSE_HEAD_SIZE,
            block_size=block_size,
            compress_ratio=compress_ratio,
            overlap=overlap,
            eps=eps,
        )
        quant = (
            normed.to(torch.bfloat16)
            .float()[:SPARSE_NOPE_DIM]
            .view(-1, SPARSE_QUANT_BLOCK)
        )
        absmax = torch.clamp_min(quant.abs().amax(dim=-1), 1e-4)
        exponent = torch.ceil(torch.log2(absmax / FP8_MAX))
        scale = torch.pow(torch.tensor(2.0, device=quant.device), exponent)
        fp8 = torch.clamp(quant / scale.unsqueeze(-1), -FP8_MAX, FP8_MAX).to(
            torch.float8_e4m3fn
        )
        scale_u8 = torch.cat(
            (
                (exponent + 127.0).to(torch.uint8),
                torch.zeros(1, device="cuda", dtype=torch.uint8),
            )
        )
        rotated = _rope_ref(
            normed, (position // compress_ratio) * compress_ratio, cos_sin_cache
        )
        out.append(
            (
                token_idx,
                fp8.reshape(-1).view(torch.uint8),
                scale_u8,
                rotated[SPARSE_NOPE_DIM:].to(torch.bfloat16),
                torch.cat(
                    (
                        fp8.float().reshape(-1)
                        * scale.repeat_interleave(SPARSE_QUANT_BLOCK),
                        rotated[SPARSE_NOPE_DIM:],
                    )
                ),
            )
        )
    return out


def _decode_sparse_slot(
    cache: torch.Tensor, slot: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_idx = slot // block_size
    pos_in_block = slot % block_size
    row = cache[block_idx]
    data_base = pos_in_block * SPARSE_TOKEN_STRIDE
    scale_base = (
        block_size * SPARSE_TOKEN_STRIDE + pos_in_block * SPARSE_SCALE_DIM
    )
    fp8_bytes = (
        row[data_base : data_base + SPARSE_NOPE_DIM].clone().contiguous()
    )
    scale_u8 = (
        row[scale_base : scale_base + SPARSE_SCALE_DIM].clone().contiguous()
    )
    scale = torch.pow(
        torch.tensor(2.0, device=cache.device), scale_u8[:7].float() - 127.0
    )
    fp8 = (
        fp8_bytes.view(torch.float8_e4m3fn).float().view(7, SPARSE_QUANT_BLOCK)
    )
    nope = (fp8 * scale.unsqueeze(-1)).reshape(-1)
    rope = (
        row[data_base + SPARSE_NOPE_DIM : data_base + SPARSE_TOKEN_STRIDE]
        .clone()
        .contiguous()
        .view(torch.bfloat16)
        .float()
    )
    return fp8_bytes, scale_u8, rope, torch.cat((nope, rope))


def _rope_ref_q(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int = ROPE_DIM,
) -> torch.Tensor:
    out = index_q.float().clone()
    half = rope_dim // 2
    rope = out[..., -rope_dim:].view(*out.shape[:-1], half, 2)
    cache = cos_sin_cache.index_select(0, positions)
    cos = cache[:, :half].unsqueeze(1)
    sin = cache[:, half:].unsqueeze(1)
    even = rope[..., 0]
    odd = rope[..., 1]
    out[..., -rope_dim:] = (
        torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
        .flatten(-2)
        .to(torch.bfloat16)
        .float()
    )
    return out


def _fp8_scale_ref(x: torch.Tensor) -> torch.Tensor:
    amax = x.abs().amax(dim=-1)
    scale = torch.clamp_min(amax, 1e-4) / FP8_MAX
    return torch.pow(2.0, torch.ceil(torch.log2(scale)))


def _mxfp4_scale_ref(x: torch.Tensor) -> torch.Tensor:
    pairs = x.float().view(-1, 2)
    even = pairs[:, 0].view(-1, MXFP4_BLOCK_SIZE // 2)
    odd = pairs[:, 1].view(-1, MXFP4_BLOCK_SIZE // 2)
    amax = torch.maximum(even.abs().amax(dim=-1), odd.abs().amax(dim=-1))
    amax = torch.clamp_min(amax, 6.0 * (2**-126))
    exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127.0, 127.0)
    return exponent.to(torch.int32), (exponent + 127.0).to(torch.uint8)


def _decode_mxfp4_slot(
    cache: torch.Tensor, slot: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from batchgen_kernels.common.v4_fp4_dequant import dequant_fp4_e2m1

    block_idx = slot // block_size
    pos_in_block = slot % block_size
    row = cache[block_idx]
    data_base = pos_in_block * INDEXER_MXFP4_TOKEN_STRIDE
    scale_base = (
        block_size * INDEXER_MXFP4_TOKEN_STRIDE
        + pos_in_block * INDEXER_MXFP4_SCALE_DIM
    )
    packed = (
        row[data_base : data_base + INDEXER_MXFP4_TOKEN_STRIDE]
        .clone()
        .contiguous()
    )
    scale_u8 = (
        row[scale_base : scale_base + INDEXER_MXFP4_SCALE_DIM]
        .clone()
        .contiguous()
    )
    scale = torch.pow(
        torch.tensor(2.0, device=cache.device), scale_u8.float() - 127.0
    )
    restored = dequant_fp4_e2m1(
        packed.view(1, -1), scale.view(1, -1), torch.float32
    ).view(-1)
    return packed, scale_u8, restored


def _make_block_table(
    num_tokens: int, block_size: int, permute: bool = False
) -> torch.Tensor:
    num_blocks = max(1, math.ceil(num_tokens / block_size))
    blocks = torch.arange(num_blocks, device="cuda", dtype=torch.int32)
    if permute and num_blocks > 1:
        blocks = torch.flip(blocks, dims=(0,))
    return blocks.view(1, -1)


def test_sparse_attn_quant_roundtrip():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_kv_compress_norm_rope_insert_sparse_attn,
    )

    torch.manual_seed(0)
    T = 8
    block_size = 4
    compress_ratio = 2
    overlap = 1
    block_table = _make_block_table(T, block_size)
    state_cache = torch.randn(
        block_table.shape[1], block_size, SPARSE_HEAD_SIZE * 4, device="cuda"
    )
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    token_to_req_indices = torch.zeros(T, device="cuda", dtype=torch.int32)
    slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    weight = torch.randn(SPARSE_HEAD_SIZE, device="cuda", dtype=torch.float32)
    cache = _make_cos_sin_cache(T + 2)
    k_cache = _make_sparse_cache(2, block_size)

    fused_kv_compress_norm_rope_insert_sparse_attn(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        weight,
        cache,
        k_cache,
        kv_slot_mapping,
        block_size=block_size,
        kv_cache_block_size=block_size,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )
    refs = _sparse_cache_ref(
        state_cache,
        block_table,
        token_to_req_indices,
        positions,
        weight,
        cache,
        block_size=block_size,
        compress_ratio=compress_ratio,
        overlap=overlap,
        eps=1e-6,
    )

    for token_idx, _, _, _, restored_ref in refs:
        _, _, _, restored = _decode_sparse_slot(
            k_cache, int(kv_slot_mapping[token_idx].item()), block_size
        )
        torch.testing.assert_close(restored, restored_ref, atol=0.08, rtol=0.06)


def test_sparse_attn_nope_rope_split():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_kv_compress_norm_rope_insert_sparse_attn,
    )

    torch.manual_seed(1)
    T = 4
    block_size = 2
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    token_to_req_indices = torch.zeros(T, device="cuda", dtype=torch.int32)
    slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    block_table = _make_block_table(T, block_size)
    state_cache = torch.randn(
        block_table.shape[1], block_size, SPARSE_HEAD_SIZE * 4, device="cuda"
    )
    weight = torch.randn(SPARSE_HEAD_SIZE, device="cuda", dtype=torch.float32)
    cache = _make_cos_sin_cache(T + 1)
    k_cache = _make_sparse_cache(2, block_size)

    fused_kv_compress_norm_rope_insert_sparse_attn(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        weight,
        cache,
        k_cache,
        kv_slot_mapping,
        block_size=block_size,
        kv_cache_block_size=block_size,
        compress_ratio=2,
        overlap=1,
    )
    token_idx, fp8_ref, _, rope_ref, _ = _sparse_cache_ref(
        state_cache,
        block_table,
        token_to_req_indices,
        positions,
        weight,
        cache,
        block_size=block_size,
        compress_ratio=2,
        overlap=1,
        eps=1e-6,
    )[0]
    fp8_bytes, _, rope, _ = _decode_sparse_slot(
        k_cache, int(kv_slot_mapping[token_idx].item()), block_size
    )

    assert torch.equal(fp8_bytes, fp8_ref)
    assert torch.equal(rope.to(torch.bfloat16), rope_ref)


def test_sparse_attn_scale_encoding():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_kv_compress_norm_rope_insert_sparse_attn,
    )

    torch.manual_seed(2)
    positions = torch.tensor([0], device="cuda", dtype=torch.int64)
    token_to_req_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    slot_mapping = torch.zeros(1, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.zeros(1, device="cuda", dtype=torch.int64)
    block_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    state_cache = torch.randn(1, 1, SPARSE_HEAD_SIZE * 4, device="cuda")
    weight = torch.randn(SPARSE_HEAD_SIZE, device="cuda", dtype=torch.float32)
    cache = _make_cos_sin_cache(1)
    k_cache = _make_sparse_cache(1, 1)

    fused_kv_compress_norm_rope_insert_sparse_attn(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        weight,
        cache,
        k_cache,
        kv_slot_mapping,
        block_size=1,
        kv_cache_block_size=1,
        compress_ratio=1,
        overlap=0,
    )
    _, _, scale_ref, _, _ = _sparse_cache_ref(
        state_cache,
        block_table,
        token_to_req_indices,
        positions,
        weight,
        cache,
        block_size=1,
        compress_ratio=1,
        overlap=0,
        eps=1e-6,
    )[0]
    _, scale_u8, _, _ = _decode_sparse_slot(k_cache, 0, 1)

    assert torch.equal(scale_u8, scale_ref)


def test_sparse_attn_cache_insert():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_kv_compress_norm_rope_insert_sparse_attn,
    )

    torch.manual_seed(3)
    T = 4
    block_size = 2
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    token_to_req_indices = torch.zeros(T, device="cuda", dtype=torch.int32)
    slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.tensor(
        [3, 2, 1, 0], device="cuda", dtype=torch.int64
    )
    block_table = _make_block_table(T, block_size, permute=True)
    state_cache = torch.randn(
        block_table.shape[1], block_size, SPARSE_HEAD_SIZE * 4, device="cuda"
    )
    weight = torch.randn(SPARSE_HEAD_SIZE, device="cuda", dtype=torch.float32)
    cache = _make_cos_sin_cache(T + 1)
    k_cache = _make_sparse_cache(2, block_size)

    fused_kv_compress_norm_rope_insert_sparse_attn(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        weight,
        cache,
        k_cache,
        kv_slot_mapping,
        block_size=block_size,
        kv_cache_block_size=block_size,
        compress_ratio=2,
        overlap=1,
    )
    refs = _sparse_cache_ref(
        state_cache,
        block_table,
        token_to_req_indices,
        positions,
        weight,
        cache,
        block_size=block_size,
        compress_ratio=2,
        overlap=1,
        eps=1e-6,
    )

    for token_idx, _, _, _, restored_ref in refs:
        _, _, _, restored = _decode_sparse_slot(
            k_cache, int(kv_slot_mapping[token_idx].item()), block_size
        )
        torch.testing.assert_close(restored, restored_ref, atol=0.08, rtol=0.06)


def test_sparse_attn_shape():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_kv_compress_norm_rope_insert_sparse_attn,
    )

    T = 2
    block_size = 2
    state_cache = torch.zeros(
        1, block_size, SPARSE_HEAD_SIZE * 4, device="cuda"
    )
    token_to_req_indices = torch.zeros(T, device="cuda", dtype=torch.int32)
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    block_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    weight = torch.ones(SPARSE_HEAD_SIZE, device="cuda", dtype=torch.float32)
    cache = _make_cos_sin_cache(T + 1)
    k_cache = _make_sparse_cache(1, block_size)

    out = fused_kv_compress_norm_rope_insert_sparse_attn(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        weight,
        cache,
        k_cache,
        kv_slot_mapping,
        block_size=block_size,
        kv_cache_block_size=block_size,
        compress_ratio=1,
        overlap=0,
    )

    assert out.shape == k_cache.shape
    assert out.dtype == torch.uint8


def test_indexer_fp8_quant():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_indexer_q_rope_quant,
    )

    torch.manual_seed(4)
    index_q = torch.randn(32, 64, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(32, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(33)
    weights = torch.randn(32, 64, device="cuda", dtype=torch.float32)

    out_fp8, weights_out = fused_indexer_q_rope_quant(
        index_q, cache, positions, weights
    )
    rotated = _rope_ref_q(index_q, positions, cache)
    scale = _fp8_scale_ref(rotated)
    ref_fp8 = torch.clamp(rotated / scale.unsqueeze(-1), -FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    )

    assert torch.allclose(
        out_fp8.float(), ref_fp8.float(), atol=1e-2, rtol=1e-2
    )
    assert torch.allclose(weights_out, weights / scale, atol=1e-2, rtol=1e-2)


def test_indexer_rope_correctness():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_indexer_q_rope_quant,
    )

    torch.manual_seed(5)
    index_q = torch.randn(16, 32, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(16, device="cuda", dtype=torch.int64) + 3
    cache = _make_cos_sin_cache(32)
    weights = torch.ones(16, 32, device="cuda", dtype=torch.float32)

    out_fp8, _ = fused_indexer_q_rope_quant(index_q, cache, positions, weights)
    rotated = _rope_ref_q(index_q, positions, cache)
    scale = _fp8_scale_ref(rotated)
    restored = out_fp8.float() * scale.unsqueeze(-1)

    torch.testing.assert_close(
        restored[..., :64], rotated[..., :64], atol=0.06, rtol=0.05
    )
    assert not torch.allclose(
        restored[..., 64:], index_q[..., 64:].float(), atol=0.06, rtol=0.05
    )


def test_indexer_weight_folding():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_indexer_q_rope_quant,
    )

    torch.manual_seed(6)
    index_q = torch.randn(8, 16, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(8, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(9)
    weights = torch.randn(8, 16, device="cuda", dtype=torch.float32)
    softmax_scale = 0.125
    head_scale = 0.5

    _, weights_out = fused_indexer_q_rope_quant(
        index_q,
        cache,
        positions,
        weights,
        softmax_scale=softmax_scale,
        head_scale=head_scale,
    )
    scale = _fp8_scale_ref(_rope_ref_q(index_q, positions, cache))

    assert torch.allclose(
        weights_out,
        weights * softmax_scale * head_scale / scale,
        atol=1e-2,
        rtol=1e-2,
    )


def test_indexer_single_block_scale():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_indexer_q_rope_quant,
    )

    torch.manual_seed(7)
    index_q = torch.randn(4, 8, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(4, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(5)
    weights = torch.ones(4, 8, device="cuda", dtype=torch.float32)

    _, weights_out = fused_indexer_q_rope_quant(
        index_q, cache, positions, weights
    )
    scale = _fp8_scale_ref(_rope_ref_q(index_q, positions, cache))

    torch.testing.assert_close(
        weights_out.reciprocal(), scale, atol=1e-3, rtol=1e-3
    )


def test_indexer_shape():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_indexer_q_rope_quant,
    )

    index_q = torch.randn(2, 4, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(2, device="cuda", dtype=torch.int64)
    cache = _make_cos_sin_cache(3)
    weights = torch.ones(2, 4, device="cuda", dtype=torch.float32)

    out_fp8, weights_out = fused_indexer_q_rope_quant(
        index_q, cache, positions, weights
    )

    assert out_fp8.shape == index_q.shape
    assert out_fp8.dtype == torch.float8_e4m3fn
    assert weights_out.shape == weights.shape


def _make_indexer_fp8_cache(num_blocks: int, block_size: int) -> torch.Tensor:
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        INDEXER_FP8_SCALE_DIM,
        INDEXER_FP8_TOKEN_STRIDE,
    )

    return torch.zeros(
        (
            num_blocks,
            block_size * INDEXER_FP8_TOKEN_STRIDE
            + block_size * INDEXER_FP8_SCALE_DIM,
        ),
        dtype=torch.uint8,
        device="cuda",
    )


def _decode_indexer_fp8_slot(
    cache: torch.Tensor, slot: int, block_size: int
) -> torch.Tensor:
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        INDEXER_FP8_SCALE_DIM,
        INDEXER_FP8_TOKEN_STRIDE,
    )

    block_idx = slot // block_size
    pos_in_block = slot % block_size
    row = cache[block_idx]
    data_base = pos_in_block * INDEXER_FP8_TOKEN_STRIDE
    scale_base = (
        block_size * INDEXER_FP8_TOKEN_STRIDE
        + pos_in_block * INDEXER_FP8_SCALE_DIM
    )
    fp8 = (
        row[data_base : data_base + INDEXER_FP8_TOKEN_STRIDE]
        .clone()
        .view(torch.float8_e4m3fn)
        .float()
    )
    scale = (
        row[scale_base : scale_base + INDEXER_FP8_SCALE_DIM]
        .clone()
        .view(torch.float32)
    )
    return fp8 * scale


def test_select_indexer_quant(monkeypatch):
    from batchgen_kernels.triton import v4_fused_compress_quant as mod

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    monkeypatch.delenv("BATCHGEN_V4_INDEXER_QUANT", raising=False)
    assert mod._indexer_quant_use_fp4(None) is True

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    assert mod._indexer_quant_use_fp4(None) is False

    monkeypatch.setenv("BATCHGEN_V4_INDEXER_QUANT", "mxfp4")
    assert mod._indexer_quant_use_fp4(None) is True
    monkeypatch.setenv("BATCHGEN_V4_INDEXER_QUANT", "fp8")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    assert mod._indexer_quant_use_fp4(None) is False

    assert mod._indexer_quant_use_fp4(True) is True
    assert mod._indexer_quant_use_fp4(False) is False


def test_indexer_fp8_compress_matches_eager():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_kv_compress_norm_rope_insert_indexer_fp8_attn,
    )

    torch.manual_seed(8)
    T = 8
    block_size = 4
    compress_ratio = 2
    overlap = 1
    block_table = _make_block_table(T, block_size)
    state_cache = 0.5 * torch.randn(
        block_table.shape[1], block_size, INDEXER_HEAD_SIZE * 4, device="cuda"
    )
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    token_to_req_indices = torch.zeros(T, device="cuda", dtype=torch.int32)
    slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    weight = torch.randn(INDEXER_HEAD_SIZE, device="cuda", dtype=torch.float32)
    cache = _make_cos_sin_cache(T + 1)
    k_cache = _make_indexer_fp8_cache(2, block_size)

    fused_kv_compress_norm_rope_insert_indexer_fp8_attn(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        weight,
        cache,
        k_cache,
        kv_slot_mapping,
        block_size=block_size,
        kv_cache_block_size=block_size,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )

    for token_idx, position in enumerate(positions.tolist()):
        if (position + 1) % compress_ratio != 0:
            continue
        restored = _decode_indexer_fp8_slot(
            k_cache, int(kv_slot_mapping[token_idx].item()), block_size
        )
        normed = _compress_norm_ref(
            state_cache,
            block_table,
            0,
            position,
            weight,
            head_dim=INDEXER_HEAD_SIZE,
            block_size=block_size,
            compress_ratio=compress_ratio,
            overlap=overlap,
            eps=1e-6,
        )
        rotated = _rope_ref(
            normed, (position // compress_ratio) * compress_ratio, cache
        )
        torch.testing.assert_close(restored, rotated, atol=0.1, rtol=0.1)


@requires_mxfp4
def test_indexer_mxfp4_roundtrip():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
    )

    torch.manual_seed(8)
    T = 8
    block_size = 4
    compress_ratio = 2
    overlap = 1
    block_table = _make_block_table(T, block_size)
    state_cache = 0.5 * torch.randn(
        block_table.shape[1], block_size, INDEXER_HEAD_SIZE * 4, device="cuda"
    )
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    token_to_req_indices = torch.zeros(T, device="cuda", dtype=torch.int32)
    slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    weight = torch.randn(INDEXER_HEAD_SIZE, device="cuda", dtype=torch.float32)
    cache = _make_cos_sin_cache(T + 1)
    k_cache = _make_mxfp4_cache(2, block_size)

    fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        weight,
        cache,
        k_cache,
        kv_slot_mapping,
        block_size=block_size,
        kv_cache_block_size=block_size,
        compress_ratio=compress_ratio,
        overlap=overlap,
    )

    for token_idx, position in enumerate(positions.tolist()):
        if (position + 1) % compress_ratio != 0:
            continue
        _, _, restored = _decode_mxfp4_slot(
            k_cache, int(kv_slot_mapping[token_idx].item()), block_size
        )
        normed = _compress_norm_ref(
            state_cache,
            block_table,
            0,
            position,
            weight,
            head_dim=INDEXER_HEAD_SIZE,
            block_size=block_size,
            compress_ratio=compress_ratio,
            overlap=overlap,
            eps=1e-6,
        )
        rotated = _rope_ref(
            normed, (position // compress_ratio) * compress_ratio, cache
        )
        torch.testing.assert_close(restored, rotated, atol=0.75, rtol=0.35)


@requires_mxfp4
def test_indexer_mxfp4_block32_scale():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
    )

    torch.manual_seed(9)
    positions = torch.tensor([0], device="cuda", dtype=torch.int64)
    token_to_req_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    slot_mapping = torch.zeros(1, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.zeros(1, device="cuda", dtype=torch.int64)
    block_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    state_cache = torch.randn(1, 1, INDEXER_HEAD_SIZE * 4, device="cuda")
    weight = torch.randn(INDEXER_HEAD_SIZE, device="cuda", dtype=torch.float32)
    cache = _make_cos_sin_cache(1)
    k_cache = _make_mxfp4_cache(1, 1)

    fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        weight,
        cache,
        k_cache,
        kv_slot_mapping,
        block_size=1,
        kv_cache_block_size=1,
        compress_ratio=1,
        overlap=0,
    )
    _, scale_u8, _ = _decode_mxfp4_slot(k_cache, 0, 1)
    normed = _compress_norm_ref(
        state_cache,
        block_table,
        0,
        0,
        weight,
        head_dim=INDEXER_HEAD_SIZE,
        block_size=1,
        compress_ratio=1,
        overlap=0,
        eps=1e-6,
    )
    rotated = _rope_ref(normed, 0, cache)
    _, scale_ref = _mxfp4_scale_ref(rotated)

    assert torch.equal(scale_u8, scale_ref)


@requires_mxfp4
def test_indexer_mxfp4_packed_format():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
    )

    positions = torch.tensor([0], device="cuda", dtype=torch.int64)
    token_to_req_indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    slot_mapping = torch.zeros(1, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.zeros(1, device="cuda", dtype=torch.int64)
    block_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
    state_cache = torch.zeros(1, 1, INDEXER_HEAD_SIZE * 4, device="cuda")
    state_cache[0, 0, :INDEXER_HEAD_SIZE] = 1.0
    weight = torch.ones(INDEXER_HEAD_SIZE, device="cuda", dtype=torch.float32)
    cache = torch.zeros(1, ROPE_DIM, device="cuda", dtype=torch.float32)
    cache[0, : ROPE_DIM // 2] = 1.0
    k_cache = _make_mxfp4_cache(1, 1)

    fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        weight,
        cache,
        k_cache,
        kv_slot_mapping,
        block_size=1,
        kv_cache_block_size=1,
        compress_ratio=1,
        overlap=0,
        rms_norm_eps=0.0,
    )
    packed, scale_u8, _ = _decode_mxfp4_slot(k_cache, 0, 1)

    assert torch.equal(packed, torch.full_like(packed, 0x66))
    assert torch.equal(scale_u8, torch.full_like(scale_u8, 125))


def test_benchmark():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_indexer_q_rope_quant,
        fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
        fused_kv_compress_norm_rope_insert_sparse_attn,
    )
    from tests.kernels.conftest import _bench

    torch.manual_seed(10)
    T = 8
    block_size = 4
    sparse_state = torch.randn(
        2, block_size, SPARSE_HEAD_SIZE * 4, device="cuda"
    )
    small_state = torch.randn(
        2, block_size, INDEXER_HEAD_SIZE * 4, device="cuda"
    )
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    token_to_req_indices = torch.zeros(T, device="cuda", dtype=torch.int32)
    slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    block_table = _make_block_table(T, block_size)
    weight_sparse = torch.randn(
        SPARSE_HEAD_SIZE, device="cuda", dtype=torch.float32
    )
    weight_small = torch.randn(
        INDEXER_HEAD_SIZE, device="cuda", dtype=torch.float32
    )
    cos_sin = _make_cos_sin_cache(T + 1)
    sparse_cache = _make_sparse_cache(2, block_size)
    mxfp4_cache = _make_mxfp4_cache(2, block_size)
    index_q = torch.randn(T, 16, 128, device="cuda", dtype=torch.bfloat16)
    index_weights = torch.randn(T, 16, device="cuda", dtype=torch.float32)

    sparse_ms = _bench(
        lambda: fused_kv_compress_norm_rope_insert_sparse_attn(
            sparse_state,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            weight_sparse,
            cos_sin,
            sparse_cache,
            kv_slot_mapping,
            block_size=block_size,
            kv_cache_block_size=block_size,
            compress_ratio=2,
            overlap=1,
        ),
        warmup=1,
        iters=3,
    )
    fp8_ms = _bench(
        lambda: fused_indexer_q_rope_quant(
            index_q,
            cos_sin,
            positions,
            index_weights,
        ),
        warmup=1,
        iters=3,
    )
    mxfp4_ms = _bench(
        lambda: fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
            small_state,
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            weight_small,
            cos_sin,
            mxfp4_cache,
            kv_slot_mapping,
            block_size=block_size,
            kv_cache_block_size=block_size,
            compress_ratio=2,
            overlap=1,
        ),
        warmup=1,
        iters=3,
    )
    print(
        f"\ncompress_quant sparse={sparse_ms:.3f} ms fp8={fp8_ms:.3f} ms mxfp4={mxfp4_ms:.3f} ms"
    )

    assert sparse_ms > 0
    assert fp8_ms > 0
    assert mxfp4_ms > 0


@requires_mxfp4
def test_all_three_variants_integration():
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_indexer_q_rope_quant,
        fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
        fused_kv_compress_norm_rope_insert_sparse_attn,
    )

    torch.manual_seed(11)
    T = 4
    block_size = 2
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    token_to_req_indices = torch.zeros(T, device="cuda", dtype=torch.int32)
    slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    block_table = _make_block_table(T, block_size, permute=True)
    cos_sin = _make_cos_sin_cache(T + 1)

    sparse_cache = _make_sparse_cache(2, block_size)
    sparse_state = torch.randn(
        2, block_size, SPARSE_HEAD_SIZE * 4, device="cuda"
    )
    sparse_weight = torch.randn(
        SPARSE_HEAD_SIZE, device="cuda", dtype=torch.float32
    )
    sparse_out = fused_kv_compress_norm_rope_insert_sparse_attn(
        sparse_state,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        sparse_weight,
        cos_sin,
        sparse_cache,
        kv_slot_mapping,
        block_size=block_size,
        kv_cache_block_size=block_size,
        compress_ratio=2,
        overlap=1,
    )

    index_q = torch.randn(T, 8, 128, device="cuda", dtype=torch.bfloat16)
    index_weights = torch.randn(T, 8, device="cuda", dtype=torch.float32)
    fp8_q, folded = fused_indexer_q_rope_quant(
        index_q, cos_sin, positions, index_weights
    )

    mxfp4_cache = _make_mxfp4_cache(2, block_size)
    mxfp4_state = torch.randn(
        2, block_size, INDEXER_HEAD_SIZE * 4, device="cuda"
    )
    mxfp4_weight = torch.randn(
        INDEXER_HEAD_SIZE, device="cuda", dtype=torch.float32
    )
    mxfp4_out = fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
        mxfp4_state,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        mxfp4_weight,
        cos_sin,
        mxfp4_cache,
        kv_slot_mapping,
        block_size=block_size,
        kv_cache_block_size=block_size,
        compress_ratio=2,
        overlap=1,
    )

    assert sparse_out.shape == sparse_cache.shape
    assert fp8_q.shape == index_q.shape
    assert folded.shape == index_weights.shape
    assert mxfp4_out.shape == mxfp4_cache.shape
    assert torch.isfinite(fp8_q.float()).all()
    assert torch.isfinite(folded).all()
    assert sparse_out.sum().item() > 0
    assert mxfp4_out.sum().item() > 0


# ---------------------------------------------------------------------------- #
#  Eager <-> fused-kernel bridge parity                                         #
#                                                                               #
#  The runtime decode path runs the EAGER compressor                            #
#  (DeepSeekV4Compressor.forward_decode in v4_compressor.py), while the fused   #
#  Triton write kernels (this module) are tested only against their own numpy-  #
#  style references. These bridge tests drive BOTH from identical projections   #
#  and assert the emitted compressed-KV row agrees within quant tolerance, so   #
#  swapping the eager path for the fused kernel is gated by a real equivalence  #
#  check.                                                                        #
#                                                                               #
#  Scope: OVERLAP=0 only. For overlap, the fused kernel uses a cross-chunk      #
#  window (prefill _overlap_transform semantics: prev-half ++ cur-half), while  #
#  eager forward_decode_batch pools only the current chunk's staged slots.      #
#  Those are intentionally different groupings, so an equality assertion would  #
#  be incorrect; overlap parity is out of scope here.                           #
# ---------------------------------------------------------------------------- #


def _populate_state_cache_from_eager(
    compressor,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    *,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """state_cache row layout: [kv0(H) | kv1(H) | score0(H) | score1(H)]; for
    OVERLAP=0 only half-0 is populated (kv=wkv, score=wgate+ape[slot])."""
    num_blocks = block_table.shape[1]
    state_cache = torch.zeros(
        num_blocks, block_size, head_dim * 4, device="cuda", dtype=torch.float32
    )
    state_width = head_dim * 2
    ratio = compressor.compress_ratio
    for token_idx, position in enumerate(positions.tolist()):
        kv = torch.nn.functional.linear(
            hidden_states[token_idx].unsqueeze(0), compressor.wkv_weight
        ).squeeze(0)
        gate = torch.nn.functional.linear(
            hidden_states[token_idx].unsqueeze(0), compressor.wgate_weight
        ).squeeze(0)
        slot = position % ratio
        score = gate + compressor.ape[slot]
        block = int(block_table[0, position // block_size].item())
        offset = position % block_size
        state_cache[block, offset, 0:head_dim] = kv
        state_cache[block, offset, state_width : state_width + head_dim] = score
    return state_cache


def _eager_emit(
    compressor,
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
) -> dict[int, torch.Tensor]:
    kv_state = torch.zeros(
        compressor.compress_ratio,
        compressor.coeff * head_dim,
        device="cuda",
        dtype=torch.float32,
    )
    score_state = torch.zeros_like(kv_state)
    emitted: dict[int, torch.Tensor] = {}
    for token_idx, position in enumerate(positions.tolist()):
        out, kv_state, score_state = compressor.forward_decode(
            hidden_states[token_idx : token_idx + 1],
            kv_state,
            score_state,
            positions[token_idx : token_idx + 1],
            cos_sin_cache,
        )
        if out.numel():
            emitted[token_idx] = out.squeeze(0).float()
    return emitted


def test_bridge_eager_matches_sparse_kernel():
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_kv_compress_norm_rope_insert_sparse_attn,
    )

    torch.manual_seed(20)
    hidden_size = 64
    head_dim = SPARSE_HEAD_SIZE
    rope_dim = ROPE_DIM
    ratio = 2
    num_chunks = 2
    T = ratio * num_chunks
    block_size = 4

    compressor = DeepSeekV4Compressor(
        hidden_size, head_dim, rope_dim, ratio, 1e-6, overlap=False
    ).cuda()
    hidden_states = torch.randn(
        T, hidden_size, device="cuda", dtype=torch.float32
    )
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(T + 1, rope_dim)
    block_table = _make_block_table(T, block_size)

    eager = _eager_emit(
        compressor, hidden_states, positions, cos_sin_cache, head_dim
    )
    assert sorted(eager.keys()) == [1, 3]

    state_cache = _populate_state_cache_from_eager(
        compressor,
        hidden_states,
        positions,
        block_table,
        head_dim=head_dim,
        block_size=block_size,
    )
    token_to_req_indices = torch.zeros(T, device="cuda", dtype=torch.int32)
    slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    k_cache = _make_sparse_cache(2, block_size)

    fused_kv_compress_norm_rope_insert_sparse_attn(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        compressor.norm.weight.contiguous(),
        cos_sin_cache,
        k_cache,
        kv_slot_mapping,
        block_size=block_size,
        kv_cache_block_size=block_size,
        compress_ratio=ratio,
        overlap=0,
    )

    fp8_atol, fp8_rtol = 0.1, 0.1
    for token_idx, eager_vec in eager.items():
        _, _, _, restored = _decode_sparse_slot(
            k_cache, int(kv_slot_mapping[token_idx].item()), block_size
        )
        torch.testing.assert_close(
            restored, eager_vec, atol=fp8_atol, rtol=fp8_rtol
        )


@requires_mxfp4
def test_bridge_eager_matches_indexer_mxfp4_kernel():
    from batchgen_kernels.attention.v4_compressor import DeepSeekV4Compressor
    from batchgen_kernels.triton.v4_fused_compress_quant import (
        fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
    )

    torch.manual_seed(21)
    hidden_size = 64
    head_dim = INDEXER_HEAD_SIZE
    rope_dim = ROPE_DIM
    ratio = 2
    num_chunks = 2
    T = ratio * num_chunks
    block_size = 4

    compressor = DeepSeekV4Compressor(
        hidden_size, head_dim, rope_dim, ratio, 1e-6, overlap=False
    ).cuda()
    mxfp4_amplitude_guard = 0.5
    hidden_states = mxfp4_amplitude_guard * torch.randn(
        T, hidden_size, device="cuda", dtype=torch.float32
    )
    positions = torch.arange(T, device="cuda", dtype=torch.int64)
    cos_sin_cache = _make_cos_sin_cache(T + 1, rope_dim)
    block_table = _make_block_table(T, block_size)

    eager = _eager_emit(
        compressor, hidden_states, positions, cos_sin_cache, head_dim
    )
    assert sorted(eager.keys()) == [1, 3]

    state_cache = _populate_state_cache_from_eager(
        compressor,
        hidden_states,
        positions,
        block_table,
        head_dim=head_dim,
        block_size=block_size,
    )
    token_to_req_indices = torch.zeros(T, device="cuda", dtype=torch.int32)
    slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    kv_slot_mapping = torch.arange(T, device="cuda", dtype=torch.int64)
    k_cache = _make_mxfp4_cache(2, block_size)

    fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
        state_cache,
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        compressor.norm.weight.contiguous(),
        cos_sin_cache,
        k_cache,
        kv_slot_mapping,
        block_size=block_size,
        kv_cache_block_size=block_size,
        compress_ratio=ratio,
        overlap=0,
    )

    mxfp4_atol, mxfp4_rtol = 0.75, 0.35
    for token_idx, eager_vec in eager.items():
        _, _, restored = _decode_mxfp4_slot(
            k_cache, int(kv_slot_mapping[token_idx].item()), block_size
        )
        torch.testing.assert_close(
            restored, eager_vec, atol=mxfp4_atol, rtol=mxfp4_rtol
        )
