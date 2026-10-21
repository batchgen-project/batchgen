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

from dataclasses import dataclass
from typing import Tuple

import torch

__all__ = [
    "SCALE_BYTES_PER_TOKEN",
    "FP8_MAX",
    "AMAX_FLOOR",
    "MAX_DATA_DIM",
    "WARP_SIZE",
    "quantize_indexer_k_fp8",
    "page_split_geometry",
    "page_split_write_reference",
    "page_split_cache_pages",
    "cache_is_page_split",
    "cache_data_dim",
    "PageSplitWritePlan",
    "plan_page_split_write",
    "NORM_ROPE_HEAD_DIM",
    "NORM_ROPE_ROPE_DIM",
    "NormRopeWritePlan",
    "plan_norm_rope_write",
    "indexer_k_norm_rope_reference",
    "hadamard_matrix_128",
]

SCALE_BYTES_PER_TOKEN = 4
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4
# One warp per token with at most 8 elements per lane, mirrored from
# `src/attention/fused_indexer_k_write_fp8.cu`.
WARP_SIZE = 32
MAX_DATA_DIM = 256

# The fused LayerNorm+RoPE+Hadamard+quantize+write kernel is specialized for
# the GLM-5.2 indexer geometry: one warp owns one 128-dim token (4 elements per
# lane) and the leading 64 dims carry interleaved RoPE (32 pairs).
NORM_ROPE_HEAD_DIM = 128
NORM_ROPE_ROPE_DIM = 64


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


def cache_is_page_split(blocked_kv: torch.Tensor, data_dim: int) -> bool:
    """True when this paged cache stores page-split FP8 pages.

    ``data_dim`` is the real head dim (elements of one token's keys). A
    page-split cache has a 1-byte element type and carries ``data_dim + 4``
    BYTES per token; an AoS cache carries ``data_dim`` elements of the key
    dtype. Any other combination is a mis-sized cache and is named here
    instead of being written through.
    """

    carrier = int(blocked_kv.shape[-1])
    element_size = int(blocked_kv.element_size())
    data_dim = int(data_dim)
    if element_size == 1:
        if carrier != data_dim + SCALE_BYTES_PER_TOKEN:
            raise ValueError(
                "a 1-byte indexer cache must be page-split, i.e. carry "
                f"{data_dim} K bytes plus {SCALE_BYTES_PER_TOKEN} scale bytes "
                f"per token, got a trailing dim of {carrier}"
            )
        return True
    if carrier != data_dim:
        raise ValueError(
            f"an AoS indexer cache must carry {data_dim} elements per token, "
            f"got a trailing dim of {carrier}"
        )
    return False


def cache_data_dim(blocked_kv: torch.Tensor) -> int:
    """Elements of K data per token in this paged cache.

    ``trailing dim - 4`` for a page-split cache (its trailing dim counts
    bytes, scale bytes included), the trailing dim itself for AoS.
    """

    carrier = int(blocked_kv.shape[-1])
    if int(blocked_kv.element_size()) == 1:
        return carrier - SCALE_BYTES_PER_TOKEN
    return carrier


def page_split_cache_pages(blocked_kv: torch.Tensor) -> torch.Tensor:
    """``[num_pages, page_bytes]`` uint8 view of a page-split paged cache.

    The fused writer and the host copy paths both address a page as flat
    bytes, so the byte view is derived once, here, from the cache the KV
    manager allocated (``[num_pages, page_size, 1, carrier_dim]``).
    """

    num_pages, page_bytes, _, _ = page_split_geometry(blocked_kv)
    return blocked_kv.view(torch.uint8).reshape(num_pages, page_bytes)


@dataclass(frozen=True)
class PageSplitWritePlan:
    """Resolved geometry of one fused quantize+write launch."""

    num_tokens: int
    num_pages: int
    page_size: int
    page_bytes: int
    data_dim: int
    scale_region_offset: int


def plan_page_split_write(
    blocked_kv: torch.Tensor,
    k_src: torch.Tensor,
    page_size_tokens: int,
) -> PageSplitWritePlan:
    """Validate a fused quantize+write against a cache and return its geometry.

    ``k_src`` is the bf16 indexer K of this step, either ``[N, data_dim]`` or
    any shape whose trailing dim is ``data_dim`` and whose remaining dims
    multiply to ``N`` (the decode writers hand over ``[N, 1, 1, data_dim]``).

    This is the whole precondition set of the CUDA kernel, expressed in torch
    so it is checked -- and tested -- without a GPU. Every failure names the
    two numbers that disagree instead of letting a launch write a wrong page.
    """

    num_pages, page_bytes, data_dim, scale_region_offset = page_split_geometry(
        blocked_kv
    )
    page_size = int(blocked_kv.shape[1])
    if int(page_size_tokens) != page_size:
        raise ValueError(
            f"page_size_tokens {int(page_size_tokens)} does not match the "
            f"cache's page size {page_size}"
        )
    if data_dim % WARP_SIZE != 0 or data_dim > MAX_DATA_DIM:
        raise ValueError(
            f"the fused page-split writer supports a data dim that is a "
            f"multiple of {WARP_SIZE} and at most {MAX_DATA_DIM}, got "
            f"{data_dim}"
        )
    if k_src.dtype is not torch.bfloat16:
        raise ValueError(
            f"the fused page-split writer quantizes from bfloat16, got "
            f"{k_src.dtype}"
        )
    if k_src.dim() < 1 or int(k_src.shape[-1]) != data_dim:
        raise ValueError(
            f"k_src trailing dim {tuple(k_src.shape)} does not match the "
            f"cache's {data_dim} K data elements per token"
        )
    num_tokens = k_src.numel() // data_dim if data_dim else 0
    if num_tokens * data_dim != k_src.numel():
        raise ValueError(
            f"k_src holds {k_src.numel()} elements, which is not a whole "
            f"number of {data_dim}-element tokens"
        )
    return PageSplitWritePlan(
        num_tokens=int(num_tokens),
        num_pages=int(num_pages),
        page_size=page_size,
        page_bytes=int(page_bytes),
        data_dim=int(data_dim),
        scale_region_offset=int(scale_region_offset),
    )


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


@dataclass(frozen=True)
class NormRopeWritePlan:
    """Resolved geometry of one fused norm+rope+quantize+write launch."""

    num_tokens: int
    num_pages: int
    page_size: int
    page_bytes: int
    data_dim: int
    rope_dim: int
    rope_pairs: int
    scale_region_offset: int
    norm_param_dtype: torch.dtype
    positions_dtype: torch.dtype


def plan_norm_rope_write(
    blocked_kv: torch.Tensor,
    k_raw: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    page_size_tokens: int,
) -> NormRopeWritePlan:
    """Validate a fused norm+rope+quantize+write and return its geometry.

    ``k_raw`` is the UNNORMALIZED ``wk`` output of this step, ``[N, 128]``
    bf16: the fused kernel owns the LayerNorm, the interleaved RoPE, the
    Hadamard rotation, the e4m3 quantization and the paged store that the
    un-fused chain spent four launches on.

    This is the kernel's whole precondition set expressed in torch, so every
    disagreement is named here -- without a GPU -- instead of reaching a
    launch that would write the wrong page.
    """

    base = plan_page_split_write(blocked_kv, k_raw, page_size_tokens)
    if base.data_dim != NORM_ROPE_HEAD_DIM:
        raise ValueError(
            f"the fused norm+rope writer is specialized for a head dim of "
            f"{NORM_ROPE_HEAD_DIM}, but the cache carries {base.data_dim}"
        )
    if k_raw.dim() != 2:
        raise ValueError(
            f"k_raw must be [N, {NORM_ROPE_HEAD_DIM}] (the wk output), got "
            f"{tuple(k_raw.shape)}"
        )
    if not k_raw.is_contiguous():
        raise ValueError(
            "k_raw must be contiguous; a re-layout here would allocate inside "
            "a captured graph"
        )

    if ln_weight.dtype is not ln_bias.dtype:
        raise ValueError(
            f"ln_weight and ln_bias must share a dtype, got {ln_weight.dtype} "
            f"and {ln_bias.dtype}"
        )
    if ln_weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(
            f"the LayerNorm parameters must be bfloat16 or float32, got "
            f"{ln_weight.dtype}"
        )
    for name, param in (("ln_weight", ln_weight), ("ln_bias", ln_bias)):
        if param.numel() != NORM_ROPE_HEAD_DIM:
            raise ValueError(
                f"{name} must hold {NORM_ROPE_HEAD_DIM} elements, got "
                f"{param.numel()}"
            )
        if not param.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    rope_pairs = NORM_ROPE_ROPE_DIM // 2
    for name, table in (("cos_cache", cos_cache), ("sin_cache", sin_cache)):
        if table.dim() != 2:
            raise ValueError(
                f"{name} must be [max_seq, rope_dim], got {tuple(table.shape)}"
            )
        if table.dtype is not torch.float32:
            raise ValueError(
                f"{name} must be float32 for graph capture, got {table.dtype}"
            )
        if not table.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if int(table.shape[1]) < rope_pairs:
            raise ValueError(
                f"{name} must hold at least {rope_pairs} interleaved pairs per "
                f"position, got {int(table.shape[1])}"
            )
    if cos_cache.shape != sin_cache.shape:
        raise ValueError(
            f"cos_cache {tuple(cos_cache.shape)} and sin_cache "
            f"{tuple(sin_cache.shape)} must have the same shape"
        )

    if positions.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"positions must be int32 or int64, got {positions.dtype}"
        )
    if not positions.is_contiguous():
        raise ValueError("positions must be contiguous")
    if positions.numel() != base.num_tokens:
        raise ValueError(
            f"positions holds {positions.numel()} entries for "
            f"{base.num_tokens} tokens"
        )

    return NormRopeWritePlan(
        num_tokens=base.num_tokens,
        num_pages=base.num_pages,
        page_size=base.page_size,
        page_bytes=base.page_bytes,
        data_dim=base.data_dim,
        rope_dim=NORM_ROPE_ROPE_DIM,
        rope_pairs=rope_pairs,
        scale_region_offset=base.scale_region_offset,
        norm_param_dtype=ln_weight.dtype,
        positions_dtype=positions.dtype,
    )


def hadamard_matrix_128(device: torch.device | None = None) -> torch.Tensor:
    """Natural-order (Sylvester) 128-point Hadamard matrix, unnormalized.

    ``H[e, f] = (-1) ** popcount(e & f)``. This is the transform both the
    16-thread x 8-element kernel (``hadamard_mult_thread`` +
    ``hadamard_mult_warp``) and the ported 32-lane x 4-element kernel realize;
    they differ only in the order the 7 butterfly stages are summed.
    """

    n = NORM_ROPE_HEAD_DIM
    idx = torch.arange(n, dtype=torch.int64, device=device)
    bits = (idx[:, None] & idx[None, :]).to(torch.int64)
    parity = torch.zeros_like(bits)
    for shift in range(7):
        parity = parity ^ ((bits >> shift) & 1)
    return torch.where(
        parity.bool(),
        torch.full_like(parity, -1, dtype=torch.float32),
        torch.ones_like(parity, dtype=torch.float32),
    )


def indexer_k_norm_rope_reference(
    k_raw: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    ln_eps: float,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    apply_hadamard: bool,
) -> torch.Tensor:
    """bf16 indexer K the fused kernel quantizes, as an explicit torch chain.

    Reproduces, stage for stage, what the fused kernel computes before its fp8
    epilogue: fp32 LayerNorm of the ``wk`` output, a round to bf16 (the
    un-fused chain materialized that tensor), interleaved RoPE on the leading
    ``rope_dim`` dims, the normalized 128-point Hadamard rotation, and a final
    round to bf16 -- the values whose amax becomes the per-token fp8 scale.

    This is the oracle for both the CPU contract tests and the on-device
    parity script; it is NOT bit-identical to the kernel, because the warp
    butterfly and the warp mean/variance reductions sum in a different order.
    """

    if k_raw.dim() != 2 or int(k_raw.shape[1]) != NORM_ROPE_HEAD_DIM:
        raise ValueError(
            f"k_raw must be [N, {NORM_ROPE_HEAD_DIM}], got "
            f"{tuple(k_raw.shape)}"
        )
    rope_dim = NORM_ROPE_ROPE_DIM
    rope_pairs = rope_dim // 2

    x = k_raw.float()
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    y = (x - mean) * torch.rsqrt(var + float(ln_eps))
    y = y * ln_weight.float() + ln_bias.float()
    y = y.to(torch.bfloat16).float()

    pos = positions.reshape(-1).to(torch.int64)
    cos = cos_cache.index_select(0, pos)[:, :rope_pairs]
    sin = sin_cache.index_select(0, pos)[:, :rope_pairs]
    rope = y[:, :rope_dim].reshape(-1, rope_pairs, 2)
    v0, v1 = rope[..., 0], rope[..., 1]
    rotated = torch.stack((v0 * cos - v1 * sin, v1 * cos + v0 * sin), dim=-1)
    y = torch.cat((rotated.reshape(-1, rope_dim), y[:, rope_dim:]), dim=-1)

    if apply_hadamard:
        h = hadamard_matrix_128(device=y.device)
        y = (y @ h) * (NORM_ROPE_HEAD_DIM ** -0.5)

    return y.to(torch.bfloat16)
