# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

from __future__ import annotations

import torch
import triton
import triton.language as tl

SPARSE_HEAD_SIZE = 512
INDEXER_HEAD_SIZE = 128
ROPE_HEAD_DIM = 64
SPARSE_NOPE_HEAD_DIM = SPARSE_HEAD_SIZE - ROPE_HEAD_DIM
FP8_MAX = tl.constexpr(448.0)
SPARSE_QUANT_BLOCK = 64
SPARSE_TOKEN_STRIDE = SPARSE_NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2
SPARSE_SCALE_DIM = 8
INDEXER_FP8_TOKEN_STRIDE = INDEXER_HEAD_SIZE
INDEXER_FP8_SCALE_DIM = 4
MXFP4_BLOCK_SIZE = 32
INDEXER_MXFP4_TOKEN_STRIDE = INDEXER_HEAD_SIZE // 2
INDEXER_MXFP4_SCALE_DIM = INDEXER_HEAD_SIZE // MXFP4_BLOCK_SIZE


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _fused_kv_compress_norm_rope_insert_sparse_attn(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM_: tl.constexpr,
    FP8_MAX_: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)
    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )
    combined_mask = mask_pos[:, None] & mask[None, :]

    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)
    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )
    compressed_kv = tl.sum(kv * score, axis=0)

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM_
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM_ // 2
    N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
    N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK
    INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX_

    quant_input = normed.to(tl.bfloat16).to(tl.float32)
    quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
    block_absmax = tl.max(tl.abs(quant_2d), axis=1)
    block_absmax = tl.maximum(block_absmax, 1e-4)
    raw_scales = block_absmax * INV_FP8_MAX
    exponents = tl.ceil(tl.log2(raw_scales))
    inv_scales = tl.exp2(-exponents)
    x_scaled = quant_2d * tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
    x_fp8 = tl.clamp(x_scaled, -FP8_MAX_, FP8_MAX_).to(tl.float8e4nv)
    x_uint8 = tl.reshape(x_fp8.to(tl.uint8, bitcast=True), (TRITON_BLOCK_SIZE,))
    nope_mask = block < NOPE_HEAD_DIM
    tl.store(fp8_ptr + block, x_uint8, mask=nope_mask)

    scale_idx = tl.arange(0, N_QUANT_BLOCKS)
    encoded = tl.maximum(tl.minimum(exponents + 127.0, 255.0), 0.0)
    tl.store(
        scale_ptr + scale_idx,
        encoded.to(tl.uint8),
        mask=scale_idx < N_NOPE_BLOCKS,
    )
    tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))

    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    pair_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(pair_2d)

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)
    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(
        cache_base + HALF_ROPE + cs_idx,
        mask=is_rope_pair,
        other=0.0,
    )
    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    result = tl.interleave(new_even, new_odd)

    bf16_ptr = (fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))
    rope_local = block - NOPE_HEAD_DIM
    is_rope = (block >= NOPE_HEAD_DIM) & mask
    tl.store(bf16_ptr + rope_local, result.to(tl.bfloat16), mask=is_rope)


@triton.jit
def _fused_indexer_q_rope_quant(
    positions_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    cos_sin_cache_ptr,
    cos_sin_stride,
    half_rot_dim: tl.constexpr,
    index_q_fp8_ptr,
    index_q_fp8_stride0,
    index_q_fp8_stride1,
    head_dim: tl.constexpr,
    index_weights_ptr,
    index_weights_stride0,
    softmax_scale,
    head_scale,
    weights_out_ptr,
    weights_out_stride0,
):
    rot_dim: tl.constexpr = 2 * half_rot_dim
    nope_dim: tl.constexpr = head_dim - rot_dim

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    pos = tl.load(positions_ptr + tok_idx)
    offset = tl.arange(0, half_rot_dim)
    cache_base = cos_sin_cache_ptr + pos * cos_sin_stride
    cos = tl.load(cache_base + offset).to(tl.float32)
    sin = tl.load(cache_base + half_rot_dim + offset).to(tl.float32)

    base_ptr = (
        index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1
    )
    rot_base = base_ptr + nope_dim
    x_even = tl.load(rot_base + offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + offset * 2 + 1).to(tl.float32)
    r_even = (x_even * cos - x_odd * sin).to(tl.bfloat16).to(tl.float32)
    r_odd = (x_odd * cos + x_even * sin).to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
    if nope_dim > 0:
        nope_offset = tl.arange(0, nope_dim)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))

    log2_q_scale = tl.ceil(tl.log2(tl.maximum(amax, 1e-4) * (1.0 / FP8_MAX)))
    q_scale = tl.exp2(log2_q_scale)
    q_scale_inv = tl.exp2(-log2_q_scale)

    fp8_base = (
        index_q_fp8_ptr
        + tok_idx * index_q_fp8_stride0
        + head_idx * index_q_fp8_stride1
    )
    if nope_dim > 0:
        tl.store(
            fp8_base + nope_offset, (x_nope * q_scale_inv).to(tl.float8e4nv)
        )
    fp8_rot_base = fp8_base + nope_dim
    tl.store(
        fp8_rot_base + offset * 2, (r_even * q_scale_inv).to(tl.float8e4nv)
    )
    tl.store(
        fp8_rot_base + offset * 2 + 1, (r_odd * q_scale_inv).to(tl.float8e4nv)
    )

    weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride0 + head_idx
    )
    weights = weights.to(tl.float32) * q_scale_inv * softmax_scale * head_scale
    tl.store(
        weights_out_ptr + tok_idx * weights_out_stride0 + head_idx, weights
    )


@triton.jit
def _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM_: tl.constexpr,
    FP8_MAX_: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)
    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )
    combined_mask = mask_pos[:, None] & mask[None, :]

    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)
    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )
    compressed_kv = tl.sum(kv * score, axis=0)

    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM_
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM_ // 2
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    normed_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(normed_2d)

    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)
    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(
        cache_base + HALF_ROPE + cs_idx,
        mask=is_rope_pair,
        other=0.0,
    )
    new_even = (even * cos_v - odd * sin_v).to(tl.bfloat16).to(tl.float32)
    new_odd = (odd * cos_v + even * sin_v).to(tl.bfloat16).to(tl.float32)

    N_QUANT_BLOCKS: tl.constexpr = HEAD_SIZE // QUANT_BLOCK
    HALF_BLOCK: tl.constexpr = QUANT_BLOCK // 2
    tl.static_assert(TRITON_BLOCK_SIZE == HEAD_SIZE)
    tl.static_assert(HEAD_SIZE % QUANT_BLOCK == 0)
    tl.static_assert(TOKEN_STRIDE == HEAD_SIZE // 2)
    tl.static_assert(SCALE_DIM == N_QUANT_BLOCKS)

    even_2d = tl.reshape(new_even, (N_QUANT_BLOCKS, HALF_BLOCK))
    odd_2d = tl.reshape(new_odd, (N_QUANT_BLOCKS, HALF_BLOCK))
    amax = tl.maximum(
        tl.max(tl.abs(even_2d), axis=1),
        tl.max(tl.abs(odd_2d), axis=1),
    )
    amax = tl.maximum(amax, 6.0 * (2**-126))
    log2_ratio = tl.minimum(
        tl.maximum(tl.ceil(tl.log2(amax * (1.0 / 6.0))), -127.0), 127.0
    )
    inv_scale = tl.exp2(-log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)
    packed = _fp32x2_to_fp4x2(
        even_2d * tl.reshape(inv_scale, (N_QUANT_BLOCKS, 1)),
        odd_2d * tl.reshape(inv_scale, (N_QUANT_BLOCKS, 1)),
    )
    tl.store(
        val_ptr + tl.arange(0, TOKEN_STRIDE),
        tl.reshape(packed, (TOKEN_STRIDE,)),
    )
    tl.store(scale_ptr + tl.arange(0, SCALE_DIM), ue8m0)


def fused_kv_compress_norm_rope_insert_sparse_attn(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    *,
    block_size: int,
    kv_cache_block_size: int,
    compress_ratio: int,
    overlap: int,
    rms_norm_eps: float = 1e-6,
) -> torch.Tensor:
    assert state_cache.is_cuda and state_cache.ndim == 3
    assert state_cache.dtype in (torch.bfloat16, torch.float32)
    assert state_cache.shape[-1] == SPARSE_HEAD_SIZE * 4
    assert token_to_req_indices.is_cuda and token_to_req_indices.ndim == 1
    assert (
        positions.is_cuda
        and positions.ndim == 1
        and positions.dtype == torch.int64
    )
    assert slot_mapping.is_cuda and slot_mapping.ndim == 1
    assert block_table.is_cuda and block_table.ndim == 2
    assert rms_norm_weight.is_cuda and rms_norm_weight.shape == (
        SPARSE_HEAD_SIZE,
    )
    assert cos_sin_cache.is_cuda and cos_sin_cache.shape[-1] == ROPE_HEAD_DIM
    assert (
        k_cache.is_cuda and k_cache.dtype == torch.uint8 and k_cache.ndim == 2
    )
    assert kv_slot_mapping.is_cuda and kv_slot_mapping.ndim == 1
    assert state_cache.stride(-1) == 1 and cos_sin_cache.stride(-1) == 1

    num_tokens = positions.numel()
    if num_tokens == 0:
        return k_cache

    _fused_kv_compress_norm_rope_insert_sparse_attn[(num_tokens,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        k_cache,
        kv_slot_mapping,
        kv_cache_block_size,
        HEAD_SIZE=SPARSE_HEAD_SIZE,
        TRITON_BLOCK_SIZE=SPARSE_HEAD_SIZE,
        STATE_WIDTH=SPARSE_HEAD_SIZE * 2,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM_=ROPE_HEAD_DIM,
        FP8_MAX_=FP8_MAX,
        QUANT_BLOCK=SPARSE_QUANT_BLOCK,
        TOKEN_STRIDE=SPARSE_TOKEN_STRIDE,
        SCALE_DIM=SPARSE_SCALE_DIM,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        num_warps=4,
        num_stages=1,
    )
    return k_cache


def fused_indexer_q_rope_quant(
    index_q: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    index_weights: torch.Tensor,
    *,
    softmax_scale: float = 1.0,
    head_scale: float = 1.0,
    rope_dim: int = ROPE_HEAD_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert (
        index_q.is_cuda
        and index_q.ndim == 3
        and index_q.shape[-1] == INDEXER_HEAD_SIZE
    )
    assert index_q.dtype == torch.bfloat16
    assert (
        cos_sin_cache.is_cuda
        and cos_sin_cache.ndim == 2
        and cos_sin_cache.shape[-1] == rope_dim
    )
    assert (
        positions.is_cuda
        and positions.ndim == 1
        and positions.dtype == torch.int64
    )
    assert index_weights.is_cuda and index_weights.ndim == 2
    assert positions.shape[0] == index_q.shape[0] == index_weights.shape[0]
    assert index_q.shape[1] == index_weights.shape[1]
    assert rope_dim == ROPE_HEAD_DIM and rope_dim % 2 == 0
    assert index_q.stride(-1) == 1 and cos_sin_cache.stride(-1) == 1

    num_tokens, num_heads, _ = index_q.shape
    index_q_fp8 = torch.empty_like(index_q, dtype=torch.float8_e4m3fn)
    weights_out = torch.empty_like(index_weights, dtype=torch.float32)
    if num_tokens == 0:
        return index_q_fp8, weights_out

    _fused_indexer_q_rope_quant[(num_tokens, num_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        cos_sin_cache,
        cos_sin_cache.stride(0),
        rope_dim // 2,
        index_q_fp8,
        index_q_fp8.stride(0),
        index_q_fp8.stride(1),
        INDEXER_HEAD_SIZE,
        index_weights,
        index_weights.stride(0),
        softmax_scale,
        head_scale,
        weights_out,
        weights_out.stride(0),
        num_warps=1,
        num_stages=1,
    )
    return index_q_fp8, weights_out


def fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    *,
    block_size: int,
    kv_cache_block_size: int,
    compress_ratio: int,
    overlap: int,
    rms_norm_eps: float = 1e-6,
) -> torch.Tensor:
    assert state_cache.is_cuda and state_cache.ndim == 3
    assert state_cache.dtype in (torch.bfloat16, torch.float32)
    assert state_cache.shape[-1] == INDEXER_HEAD_SIZE * 4
    assert token_to_req_indices.is_cuda and token_to_req_indices.ndim == 1
    assert (
        positions.is_cuda
        and positions.ndim == 1
        and positions.dtype == torch.int64
    )
    assert slot_mapping.is_cuda and slot_mapping.ndim == 1
    assert block_table.is_cuda and block_table.ndim == 2
    assert rms_norm_weight.is_cuda and rms_norm_weight.shape == (
        INDEXER_HEAD_SIZE,
    )
    assert cos_sin_cache.is_cuda and cos_sin_cache.shape[-1] == ROPE_HEAD_DIM
    assert (
        k_cache.is_cuda and k_cache.dtype == torch.uint8 and k_cache.ndim == 2
    )
    assert kv_slot_mapping.is_cuda and kv_slot_mapping.ndim == 1
    assert state_cache.stride(-1) == 1 and cos_sin_cache.stride(-1) == 1

    num_tokens = positions.numel()
    if num_tokens == 0:
        return k_cache

    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn[(num_tokens,)](
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        token_to_req_indices,
        positions,
        slot_mapping,
        block_table,
        block_table.stride(0),
        block_size,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        k_cache,
        kv_slot_mapping,
        kv_cache_block_size,
        HEAD_SIZE=INDEXER_HEAD_SIZE,
        TRITON_BLOCK_SIZE=INDEXER_HEAD_SIZE,
        STATE_WIDTH=INDEXER_HEAD_SIZE * 2,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM_=ROPE_HEAD_DIM,
        FP8_MAX_=FP8_MAX,
        QUANT_BLOCK=MXFP4_BLOCK_SIZE,
        TOKEN_STRIDE=INDEXER_MXFP4_TOKEN_STRIDE,
        SCALE_DIM=INDEXER_MXFP4_SCALE_DIM,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        num_warps=1,
        num_stages=1,
    )
    return k_cache


__all__ = [
    "FP8_MAX",
    "INDEXER_FP8_SCALE_DIM",
    "INDEXER_FP8_TOKEN_STRIDE",
    "INDEXER_HEAD_SIZE",
    "INDEXER_MXFP4_SCALE_DIM",
    "INDEXER_MXFP4_TOKEN_STRIDE",
    "MXFP4_BLOCK_SIZE",
    "ROPE_HEAD_DIM",
    "SPARSE_HEAD_SIZE",
    "SPARSE_NOPE_HEAD_DIM",
    "SPARSE_QUANT_BLOCK",
    "SPARSE_SCALE_DIM",
    "SPARSE_TOKEN_STRIDE",
    "fused_indexer_q_rope_quant",
    "fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn",
    "fused_kv_compress_norm_rope_insert_sparse_attn",
]
