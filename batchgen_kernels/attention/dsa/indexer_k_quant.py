"""Page-split FP8 indexer-K quantization and page geometry (torch only).

The GLM-5.2 indexer (aux) cache stores its keys in the layout
`deep_gemm.fp8_paged_mqa_logits` reads, one page of uint8:

    [ page_size * data_dim   e4m3 K bytes ][ page_size * 4   fp32 scale bytes ]

Token ``o`` of a page owns ``[o * data_dim, (o + 1) * data_dim)`` of the K
region and its single fp32 scale at ``page_size * data_dim + o * 4``. One scale
covers the whole token (quantization block size == data_dim), which is what
``kv_cache_cast_to_fp8`` / SGLang's ``act_quant(block_size=128)`` produce.

This module deliberately imports nothing but torch: the quantizer is the
byte-exactness contract between the model, the host pages and DeepGEMM, so it
stays importable (and testable) without Triton or a GPU.
"""

from __future__ import annotations

from typing import Tuple

import torch

__all__ = [
    "SCALE_BYTES_PER_TOKEN",
    "FP8_MAX",
    "AMAX_FLOOR",
    "quantize_indexer_k_fp8",
    "page_split_geometry",
    "page_split_write_reference",
]

SCALE_BYTES_PER_TOKEN = 4
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4


def quantize_indexer_k_fp8(
    k: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize indexer K to e4m3 with one fp32 scale per token.

    Args:
        k: ``[..., data_dim]`` keys (bf16/fp16/fp32).

    Returns:
        ``(k_fp8, scales)``: ``k_fp8`` has ``k``'s shape in
        ``torch.float8_e4m3fn``; ``scales`` has shape ``k.shape[:-1]`` in
        ``torch.float32``.

    The amax floor and the reciprocal MULTIPLY are part of the contract: a
    divide rounds differently and loses byte identity with DeepGEMM's cast.
    """

    if k.dim() < 1 or k.shape[-1] == 0:
        raise ValueError("quantize_indexer_k_fp8 expects a non-empty last dim")
    amax = k.abs().float().amax(dim=-1, keepdim=True).clamp(AMAX_FLOOR)
    scales = amax / FP8_MAX
    k_fp8 = (k * (1.0 / scales)).to(torch.float8_e4m3fn)
    return k_fp8, scales.squeeze(-1).contiguous()


def page_split_geometry(blocked_kv: torch.Tensor) -> Tuple[int, int, int, int]:
    """``(num_pages, page_bytes, data_dim, scale_region_offset)`` of a cache.

    ``blocked_kv`` is the paged cache as the KV manager allocates it:
    ``[num_pages, page_size, 1, carrier_dim]`` with a 1-byte element type,
    where ``carrier_dim`` counts the token's TOTAL bytes (data + scale).
    """

    if blocked_kv.element_size() != 1:
        raise ValueError(
            "a page-split cache must have a 1-byte element type (got "
            f"{blocked_kv.dtype})"
        )
    if blocked_kv.dim() != 4:
        raise ValueError(
            "a page-split cache must be [num_pages, page_size, 1, carrier_dim]"
            f" (got {tuple(blocked_kv.shape)})"
        )
    if not blocked_kv.is_contiguous():
        raise ValueError("a page-split cache must be contiguous")
    num_pages, page_size, heads, carrier_dim = (
        int(blocked_kv.shape[0]),
        int(blocked_kv.shape[1]),
        int(blocked_kv.shape[2]),
        int(blocked_kv.shape[3]),
    )
    if heads != 1:
        raise ValueError(f"a page-split cache has a single K head (got {heads})")
    data_dim = carrier_dim - SCALE_BYTES_PER_TOKEN
    if data_dim <= 0:
        raise ValueError(
            f"carrier dim {carrier_dim} leaves no K data bytes per token"
        )
    page_bytes = page_size * carrier_dim
    if page_bytes % SCALE_BYTES_PER_TOKEN != 0:
        raise ValueError(
            f"page bytes {page_bytes} must be a multiple of "
            f"{SCALE_BYTES_PER_TOKEN} to address the scale region as fp32"
        )
    return num_pages, page_bytes, data_dim, page_size * data_dim


def page_split_write_reference(
    buf_u8: torch.Tensor,
    slots: torch.Tensor,
    k_fp8: torch.Tensor,
    scales: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Scatter quantized tokens into page-split pages (reference).

    ``buf_u8`` is ``[num_pages, page_bytes]`` uint8 and ``slots`` holds each
    token's ABSOLUTE physical slot (``physical_page * page_size + offset``).
    This is the oracle the host copy paths and the fused writer must match
    byte for byte.
    """

    if buf_u8.dim() != 2 or buf_u8.dtype is not torch.uint8:
        raise ValueError("buf_u8 must be a 2-D uint8 [num_pages, page_bytes]")
    data_dim = k_fp8.shape[-1]
    page_bytes = int(buf_u8.shape[1])
    if page_bytes != page_size * (data_dim + SCALE_BYTES_PER_TOKEN):
        raise ValueError(
            f"page bytes {page_bytes} do not match page_size={page_size} and "
            f"data_dim={data_dim}"
        )
    flat = buf_u8.view(-1)
    page_idx = (slots // page_size).to(torch.int64)
    offset = (slots % page_size).to(torch.int64)

    k_base = page_idx * page_bytes + offset * data_dim
    k_cols = torch.arange(data_dim, device=buf_u8.device)
    flat[(k_base[:, None] + k_cols[None, :]).reshape(-1)] = (
        k_fp8.reshape(-1, data_dim).view(torch.uint8).reshape(-1)
    )

    scale_base = (
        page_idx * page_bytes
        + page_size * data_dim
        + offset * SCALE_BYTES_PER_TOKEN
    )
    scale_cols = torch.arange(SCALE_BYTES_PER_TOKEN, device=buf_u8.device)
    flat[(scale_base[:, None] + scale_cols[None, :]).reshape(-1)] = (
        scales.reshape(-1).float().view(torch.uint8).reshape(-1)
    )
    return buf_u8
