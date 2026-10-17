"""Dequantizing paged gather for the page-split FP8 indexer-K cache (Triton).

The page layout and the quantizer live in ``indexer_k_quant`` (torch only).
This module adds the interim READER: it gathers a dense logical token range
out of a page-split cache and dequantizes on output, so the existing bf16
scorer keeps working unchanged while the DeepGEMM ``fp8_paged_mqa_logits``
swap is prepared separately.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from batchgen_kernels.attention.dsa.indexer_k_quant import (
    SCALE_BYTES_PER_TOKEN,
    page_split_geometry,
    quantize_indexer_k_fp8,
)

__all__ = [
    "SCALE_BYTES_PER_TOKEN",
    "page_split_geometry",
    "quantize_indexer_k_fp8",
    "fused_dense_paged_gather_fp8_pagesplit",
]


@triton.jit
def _dense_gather_fp8_pagesplit_kernel(
    k_fp8_ptr,          # 1-byte view of the cache, indexed in bytes
    scale_f32_ptr,      # fp32 view of the SAME storage, indexed in fp32 units
    block_table_ptr,    # [B, max_pages_per_seq]
    out_ptr,            # [B, num_tokens, D]
    num_tokens,
    D: tl.constexpr,
    page_size: tl.constexpr,
    page_bytes: tl.constexpr,
    page_f32_stride: tl.constexpr,
    scale_f32_offset: tl.constexpr,
    max_pages_per_seq: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)

    t_start = pid_t * BLOCK_T
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    for ti in range(BLOCK_T):
        t = t_start + ti
        if t < num_tokens:
            logical_page = t // page_size
            page_offset = t - logical_page * page_size
            page_in_table = logical_page < max_pages_per_seq
            logical_page = tl.minimum(logical_page, max_pages_per_seq - 1)

            physical_page = tl.load(
                block_table_ptr + pid_b * max_pages_per_seq + logical_page,
            )
            # BatchGen's `-1` sentinel for an unallocated page-table slot:
            # zero the row instead of reading page 0, exactly as the bf16
            # gather does, so both paths stay byte-identical at OOB positions.
            valid = page_in_table & (physical_page >= 0)
            physical_page = tl.maximum(physical_page, 0)

            k_base = physical_page * page_bytes + page_offset * D
            vals = tl.load(k_fp8_ptr + k_base + d_offs, mask=d_mask, other=0.0)
            vals = vals.to(tl.float32)

            scale = tl.load(
                scale_f32_ptr
                + physical_page * page_f32_stride
                + scale_f32_offset
                + page_offset,
            )
            vals = vals * scale
            vals = tl.where(valid, vals, tl.zeros([BLOCK_D], dtype=tl.float32))

            dst = out_ptr + (pid_b * num_tokens + t) * D + d_offs
            tl.store(dst, vals, mask=d_mask)


def fused_dense_paged_gather_fp8_pagesplit(
    blocked_kv: torch.Tensor,
    block_table: torch.Tensor,
    num_tokens: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Gather + dequantize the dense logical range ``[0, num_tokens)``.

    Mirrors ``fused_dense_paged_gather`` for a page-split FP8 cache: same
    block-table semantics, same ``-1`` sentinel handling, same output shape
    ``[B, num_tokens, 1, data_dim]`` — but the values come back dequantized in
    ``out_dtype`` so the existing bf16 scorer consumes them unchanged.
    """

    _, page_bytes, data_dim, scale_region_offset = page_split_geometry(blocked_kv)
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    batch = int(block_table.shape[0])
    max_pages_per_seq = int(block_table.shape[1])

    flat_bytes = blocked_kv.reshape(-1)
    if flat_bytes.dtype is not torch.float8_e4m3fn:
        # A byte-typed carrier (uint8/int8) would be loaded as an INTEGER and
        # converted numerically, not reinterpreted, so the K region is always
        # read through its e4m3 view.
        flat_bytes = flat_bytes.view(torch.float8_e4m3fn)
    scale_view = flat_bytes.view(torch.float32)

    out = torch.empty(
        batch, num_tokens, data_dim, dtype=out_dtype, device=blocked_kv.device
    )

    block_d = triton.next_power_of_2(data_dim)
    total_work = batch * num_tokens
    if total_work <= 4096:
        block_t = 1
    elif total_work <= 32768:
        block_t = 8
    else:
        block_t = 32

    grid = (batch, triton.cdiv(num_tokens, block_t))
    _dense_gather_fp8_pagesplit_kernel[grid](
        flat_bytes,
        scale_view,
        block_table,
        out,
        num_tokens,
        D=data_dim,
        page_size=int(blocked_kv.shape[1]),
        page_bytes=page_bytes,
        page_f32_stride=page_bytes // SCALE_BYTES_PER_TOKEN,
        scale_f32_offset=scale_region_offset // SCALE_BYTES_PER_TOKEN,
        max_pages_per_seq=max_pages_per_seq,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
    )
    return out.view(batch, num_tokens, 1, data_dim)
