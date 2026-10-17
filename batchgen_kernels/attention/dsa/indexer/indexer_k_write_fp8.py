"""Fused indexer-K quantize + page-split cache write (AOT CUDA).

Source: ``batchgen_kernels/src/attention/fused_indexer_k_write_fp8.cu``,
registered as ``batchgen_kernels.attention._C_fused_indexer_k_write_fp8`` in
``setup.py`` and mirrored in ``_jit_registry.py``.

The kernel replaces, for a page-split FP8 indexer cache, the pair
"``quantize_indexer_k_fp8`` + bf16 paged token update": one launch per layer
quantizes the step's keys, writes them into the page's e4m3 K region and the
page's fp32 scale region, and -- when asked -- emits the SAME quantized bytes
as the two tensors the host KV region's append path takes (``k_fp8``,
``scales``). The host copy therefore moves exactly the bytes the GPU cache
holds, with no second quantization and no dequantize-requantize round trip.

Page geometry, the quantization contract and every precondition live in the
torch-only module ``batchgen_kernels.attention.dsa.indexer_k_quant`` so they
are checked (and unit-tested) without a GPU.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from batchgen_kernels import load_extension
from batchgen_kernels.attention.dsa.indexer_k_quant import (
    PageSplitWritePlan,
    plan_page_split_write,
)

__all__ = ["fused_indexer_k_write_fp8", "page_split_write_plan"]

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_extension(
            "batchgen_kernels.attention._C_fused_indexer_k_write_fp8"
        )
    return _MODULE


def page_split_write_plan(
    aux_cache: torch.Tensor,
    k_src: torch.Tensor,
    page_size_tokens: int,
) -> PageSplitWritePlan:
    """Re-exported planner, so callers can validate without launching."""

    return plan_page_split_write(aux_cache, k_src, page_size_tokens)


def fused_indexer_k_write_fp8(
    *,
    aux_cache: torch.Tensor,
    k_src: torch.Tensor,
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    token_indices: torch.Tensor,
    page_size_tokens: int,
    k_fp8_out: Optional[torch.Tensor] = None,
    scales_out: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
    emit_host_sources: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Quantize ``k_src`` to e4m3 and write it into a page-split cache.

    Args:
        aux_cache: the paged cache as the KV manager allocated it,
            ``[num_pages, page_size, 1, data_dim + 4]`` with a 1-byte element
            type (``data_dim`` e4m3 K bytes plus 4 fp32 scale bytes per token).
        k_src: this step's bf16 keys, ``[N, data_dim]`` or ``[N, 1, 1,
            data_dim]``; must be contiguous.
        page_table: ``[num_slots, max_pages_per_sequence]`` int32 GPU page
            table, read exactly as ``run_paged_kv_token_update_fused`` reads it.
        slot_indices: ``[N]`` int32 page-table row per token; ``-1`` skips the
            row (CUDA graph bucket padding).
        token_indices: ``[N]`` int32 absolute position of each new token.
        page_size_tokens: tokens per page; must equal ``aux_cache.shape[1]``.
        k_fp8_out: optional pre-allocated ``[N, ..., data_dim]`` e4m3 host-append
            source (a captured graph MUST pass its static buffer).
        scales_out: optional pre-allocated ``[N]`` fp32 host-append source.
        num_valid_tokens: optional ``[1]`` int32 device tensor; rows at or
            beyond it write nothing into the cache.
        emit_host_sources: when True and no buffers were passed, allocate them.

    Returns:
        ``(k_fp8, scales)`` -- the host-append sources, or ``(None, None)``
        when ``emit_host_sources`` is False and no buffers were passed.

    The host-append sources are written for EVERY row of the launch, including
    rows the cache write skips: they are row-local quantizations, and which
    rows reach the host is decided by the sequence set the append is called
    with. This matches the bf16 writer, which staged a full copy of all rows.
    """

    plan = plan_page_split_write(aux_cache, k_src, page_size_tokens)
    if not k_src.is_contiguous():
        raise ValueError(
            "fused_indexer_k_write_fp8 requires a contiguous k_src; a "
            "re-layout here would allocate inside a captured graph"
        )
    k_flat = k_src.view(plan.num_tokens, plan.data_dim)

    if k_fp8_out is None and emit_host_sources:
        k_fp8_out = torch.empty(
            plan.num_tokens,
            1,
            1,
            plan.data_dim,
            dtype=torch.float8_e4m3fn,
            device=k_src.device,
        )
    if scales_out is None and emit_host_sources:
        scales_out = torch.empty(
            plan.num_tokens, dtype=torch.float32, device=k_src.device
        )
    if k_fp8_out is not None and k_fp8_out.dtype is not torch.float8_e4m3fn:
        raise ValueError(
            "k_fp8_out must be a float8_e4m3fn tensor (got "
            f"{k_fp8_out.dtype}); the host region reads its bytes as e4m3"
        )
    if (k_fp8_out is None) != (scales_out is None):
        raise ValueError(
            "k_fp8_out and scales_out must be passed together: a page-split "
            "host append needs both the quantized keys and their scales"
        )

    pages = aux_cache.view(torch.uint8).reshape(plan.num_pages, plan.page_bytes)
    _module().fused_indexer_k_write_fp8(
        k_flat,
        page_table,
        slot_indices,
        token_indices,
        pages,
        int(page_size_tokens),
        None if k_fp8_out is None else k_fp8_out.view(
            plan.num_tokens, plan.data_dim
        ),
        scales_out,
        num_valid_tokens,
    )
    return k_fp8_out, scales_out
