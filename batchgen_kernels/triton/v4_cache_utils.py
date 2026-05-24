# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
SCALE_DIM = 8
QUANT_BLOCK_SIZE = 64
FP8_MAX = 448.0
TOKEN_DATA_SIZE = NOPE_DIM + ROPE_DIM * 2
TOKEN_BYTES = TOKEN_DATA_SIZE + SCALE_DIM
SPARSE_PREFILL_TOPK_ALIGNMENT = 128


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [1, 2, 4]
        for ns in [1, 2, 3]
    ],
    key=["num_tokens"],
)
@triton.jit
def _quantize_and_insert_k_kernel(
    k_ptr,
    slot_mapping_ptr,
    cache_ptr,
    num_tokens,
    input_dim: tl.constexpr,
    fp8_dim: tl.constexpr,
    bf16_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid >= num_tokens:
        return

    slot_idx = tl.load(slot_mapping_ptr + pid)
    if slot_idx < 0:
        return

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size

    input_row_ptr = k_ptr + pid * input_dim
    cache_block_ptr = cache_ptr + block_idx.to(tl.int64) * block_stride
    token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
    token_scale_ptr = (
        cache_block_ptr
        + cache_block_size * token_data_size
        + pos_in_block * scale_dim
    )
    token_fp8_ptr = token_data_ptr
    token_bf16_ptr = token_data_ptr + fp8_dim

    for qblock_idx in tl.static_range(n_quant_blocks):
        qblock_start = qblock_idx * quant_block
        offsets = qblock_start + tl.arange(0, quant_block)
        x = tl.load(input_row_ptr + offsets).to(tl.float32)
        absmax = tl.max(tl.abs(x), axis=0)
        nonzero = absmax > 0.0
        safe_absmax = tl.where(nonzero, absmax, 1.0)
        exponent = tl.ceil(tl.log2(safe_absmax / fp8_max))
        scale = tl.where(nonzero, tl.exp2(exponent), 1.0)
        x_fp8 = tl.clamp(x / scale, -fp8_max, fp8_max).to(tl.float8e4nv)
        tl.store(token_fp8_ptr + offsets, x_fp8.to(tl.uint8, bitcast=True))
        encoded = tl.where(nonzero, exponent + 127.0, 0.0)
        encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
        tl.store(token_scale_ptr + qblock_idx, encoded.to(tl.uint8))

    tl.store(token_scale_ptr + n_quant_blocks, tl.zeros((), dtype=tl.uint8))

    bf16_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    for i in tl.static_range(bf16_dim // 16):
        offsets = i * 16 + tl.arange(0, 16)
        x = tl.load(input_row_ptr + fp8_dim + offsets)
        tl.store(bf16_ptr + offsets, x)


@triton.jit
def _dequantize_and_gather_k_kernel(
    out_ptr,
    indices_ptr,
    cache_ptr,
    num_rows,
    out_stride0,
    out_stride1,
    fp8_dim: tl.constexpr,
    bf16_dim: tl.constexpr,
    scale_dim: tl.constexpr,
    quant_block: tl.constexpr,
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,
    block_stride: tl.constexpr,
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,
):
    row = tl.program_id(0)

    if row >= num_rows:
        return

    output_row_ptr = out_ptr + row * out_stride0
    slot_idx = tl.load(indices_ptr + row)

    if slot_idx < 0:
        for qblock_idx in tl.static_range(n_quant_blocks):
            offsets = qblock_idx * quant_block + tl.arange(0, quant_block)
            tl.store(
                output_row_ptr + offsets * out_stride1,
                tl.zeros([quant_block], dtype=tl.bfloat16),
            )
        for i in tl.static_range(bf16_dim // 16):
            offsets = fp8_dim + i * 16 + tl.arange(0, 16)
            tl.store(
                output_row_ptr + offsets * out_stride1,
                tl.zeros([16], dtype=tl.bfloat16),
            )
        return

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size
    cache_block_ptr = cache_ptr + block_idx.to(tl.int64) * block_stride
    token_data_ptr = cache_block_ptr + pos_in_block * token_data_size
    token_scale_ptr = (
        cache_block_ptr
        + cache_block_size * token_data_size
        + pos_in_block * scale_dim
    )
    token_fp8_ptr = token_data_ptr
    token_bf16_ptr = token_data_ptr + fp8_dim

    for qblock_idx in tl.static_range(n_quant_blocks):
        offsets = qblock_idx * quant_block + tl.arange(0, quant_block)
        x_uint8 = tl.load(token_fp8_ptr + offsets)
        x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        encoded_scale = tl.load(token_scale_ptr + qblock_idx)
        scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
        x = x_fp8.to(tl.float32) * scale
        tl.store(output_row_ptr + offsets * out_stride1, x.to(tl.bfloat16))

    bf16_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    for i in tl.static_range(bf16_dim // 16):
        offsets = i * 16 + tl.arange(0, 16)
        x = tl.load(bf16_ptr + offsets)
        tl.store(output_row_ptr + (fp8_dim + offsets) * out_stride1, x)


@triton.jit
def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    is_valid_token_ptr,
    triton_block_size: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, triton_block_size):
        offset = i + tl.arange(0, triton_block_size)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        valid = local_idx >= 0
        block_indices = local_idx // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=mask & valid,
            other=0,
        )
        block_offsets = local_idx % block_size
        slot_ids = tl.where(
            valid, block_numbers * block_size + block_offsets, -1
        )
        tl.store(
            global_topk_indices_ptr
            + token_idx * global_topk_indices_stride
            + offset,
            slot_ids,
            mask=mask,
        )
        count += tl.sum(valid.to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


def _resolve_slot_mapping(
    block_table: torch.Tensor | None,
    token_positions: torch.Tensor | None,
    token_to_req_indices: torch.Tensor | None,
    slot_mapping: torch.Tensor | None,
    block_size: int,
) -> torch.Tensor:
    if slot_mapping is not None:
        return slot_mapping.contiguous()
    assert block_table is not None
    assert token_positions is not None
    assert token_to_req_indices is not None
    logical_block = torch.div(
        token_positions, block_size, rounding_mode="floor"
    )
    block_offset = token_positions.remainder(block_size)
    physical_block = block_table[
        token_to_req_indices.long(),
        logical_block.long(),
    ]
    return (
        physical_block.long() * block_size + block_offset.long()
    ).contiguous()


def quantize_and_insert_k(
    k_bf16: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor | None = None,
    token_positions: torch.Tensor | None = None,
    token_to_req_indices: torch.Tensor | None = None,
    slot_mapping: torch.Tensor | None = None,
    block_size: int = 64,
) -> torch.Tensor:
    assert k_bf16.is_cuda and cache.is_cuda
    assert k_bf16.dtype == torch.bfloat16
    assert cache.dtype == torch.uint8
    assert k_bf16.ndim == 2 and k_bf16.shape[1] == HEAD_DIM
    slot_mapping = _resolve_slot_mapping(
        block_table,
        token_positions,
        token_to_req_indices,
        slot_mapping,
        block_size,
    )
    assert slot_mapping.is_cuda
    if slot_mapping.numel() == 0:
        return cache
    _quantize_and_insert_k_kernel[(slot_mapping.numel(),)](
        k_bf16,
        slot_mapping,
        cache,
        slot_mapping.numel(),
        input_dim=HEAD_DIM,
        fp8_dim=NOPE_DIM,
        bf16_dim=ROPE_DIM,
        scale_dim=SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=cache.stride(0),
        fp8_max=FP8_MAX,
        n_quant_blocks=NOPE_DIM // QUANT_BLOCK_SIZE,
    )
    return cache


def dequantize_and_gather_k(
    cache: torch.Tensor,
    indices: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    assert cache.is_cuda and indices.is_cuda
    assert cache.dtype == torch.uint8
    flat_indices = indices.contiguous().view(-1)
    out = torch.empty(
        (flat_indices.numel(), HEAD_DIM),
        dtype=torch.bfloat16,
        device=cache.device,
    )
    if flat_indices.numel() == 0:
        return out.view(*indices.shape, HEAD_DIM)
    _dequantize_and_gather_k_kernel[(flat_indices.numel(),)](
        out,
        flat_indices,
        cache,
        flat_indices.numel(),
        out.stride(0),
        out.stride(1),
        fp8_dim=NOPE_DIM,
        bf16_dim=ROPE_DIM,
        scale_dim=SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=cache.stride(0),
        fp8_max=FP8_MAX,
        n_quant_blocks=NOPE_DIM // QUANT_BLOCK_SIZE,
        num_warps=4,
        num_stages=1,
    )
    return out.view(*indices.shape, HEAD_DIM)


def compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert topk_indices.is_cuda
    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
    topk_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )
    if num_tokens == 0:
        return global_topk_indices, topk_lens
    block = max(
        128, triton.next_power_of_2(max(1, min(topk_indices.shape[-1], 1024)))
    )
    _compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk_indices.shape[-1],
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        is_valid_token,
        triton_block_size=block,
    )
    return global_topk_indices, topk_lens


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    swa_indices: torch.Tensor,
    pad_to: int = SPARSE_PREFILL_TOPK_ALIGNMENT,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert topk_indices.is_cuda and swa_indices.is_cuda
    assert topk_indices.shape[0] == swa_indices.shape[0]
    total = topk_indices.shape[1] + swa_indices.shape[1]
    padded = ((total + pad_to - 1) // pad_to) * pad_to
    combined = torch.full(
        (topk_indices.shape[0], padded),
        -1,
        dtype=topk_indices.dtype,
        device=topk_indices.device,
    )
    lens = torch.empty(
        topk_indices.shape[0], dtype=torch.int32, device=topk_indices.device
    )

    for row in range(topk_indices.shape[0]):
        merged = torch.cat((topk_indices[row], swa_indices[row]))
        valid = merged[merged >= 0]
        if valid.numel() == 0:
            lens[row] = 0
            continue
        keep = torch.ones(valid.shape[0], dtype=torch.bool, device=valid.device)
        for i in range(valid.shape[0]):
            if i == 0:
                continue
            keep[i] = ~(valid[:i] == valid[i]).any()
        unique = valid[keep]
        combined[row, : unique.numel()] = unique
        lens[row] = unique.numel()

    return combined, lens


__all__ = [
    "FP8_MAX",
    "HEAD_DIM",
    "NOPE_DIM",
    "QUANT_BLOCK_SIZE",
    "ROPE_DIM",
    "SCALE_DIM",
    "SPARSE_PREFILL_TOPK_ALIGNMENT",
    "TOKEN_BYTES",
    "TOKEN_DATA_SIZE",
    "combine_topk_swa_indices",
    "compute_global_topk_indices_and_lens",
    "dequantize_and_gather_k",
    "quantize_and_insert_k",
]
