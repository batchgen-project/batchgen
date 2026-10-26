"""Fused indexer-K LayerNorm + RoPE + Hadamard + quantize + write (AOT CUDA).

Source: ``batchgen_kernels/src/attention/fused_indexer_k_norm_rope_write_fp8.cu``,
registered as
``batchgen_kernels.attention._C_fused_indexer_k_norm_rope_write_fp8`` in
``setup.py`` and mirrored in ``_jit_registry.py``.

The kernel is a port of SGLang's ``fused_k_indexer_norm_rope_store``
(sgl-project/sglang @ ``e00930c5``, tag ``v0.5.21``). For a page-split FP8
indexer cache it collapses the four launches

    k_norm (ATen LayerNorm) -> fused RoPE+Hadamard -> fused quantize+write

into one: one warp per token normalizes the ``wk`` output, ropes the leading 64
dims, rotates all 128, quantizes to e4m3 with one fp32 scale per token and
writes the page's K region, the page's scale region and -- when asked -- the
two host-append staging tensors.

Page geometry, the quantization contract, the reference chain and every
precondition live in the torch-only module
``batchgen_kernels.attention.dsa.indexer_k_quant`` so they are checked (and
unit-tested) without a GPU.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from batchgen_kernels import load_extension
from batchgen_kernels.attention.dsa.indexer_k_quant import (
    NormRopeWritePlan,
    plan_norm_rope_write,
)

__all__ = [
    "fused_indexer_k_norm_rope_write_fp8",
    "norm_rope_write_plan",
]

_MODULE = None


def _module():
    global _MODULE
    if _MODULE is None:
        _MODULE = load_extension(
            "batchgen_kernels.attention._C_fused_indexer_k_norm_rope_write_fp8"
        )
    return _MODULE


def norm_rope_write_plan(
    aux_cache: torch.Tensor,
    k_raw: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    page_size_tokens: int,
) -> NormRopeWritePlan:
    """Re-exported planner, so callers can validate without launching."""

    return plan_norm_rope_write(
        aux_cache,
        k_raw,
        ln_weight,
        ln_bias,
        cos_cache,
        sin_cache,
        positions,
        page_size_tokens,
    )


def fused_indexer_k_norm_rope_write_fp8(
    *,
    aux_cache: torch.Tensor,
    k_raw: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    ln_eps: float,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    page_table: torch.Tensor,
    slot_indices: torch.Tensor,
    token_indices: torch.Tensor,
    page_size_tokens: int,
    apply_hadamard: bool,
    k_fp8_out: Optional[torch.Tensor] = None,
    scales_out: Optional[torch.Tensor] = None,
    num_valid_tokens: Optional[torch.Tensor] = None,
    emit_host_sources: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Normalize, rope, rotate, quantize and write this step's indexer keys.

    Args:
        aux_cache: the paged cache as the KV manager allocated it,
            ``[num_pages, page_size, 1, data_dim + 4]`` with a 1-byte element
            type.
        k_raw: the UNNORMALIZED ``wk`` output, ``[N, 128]`` bf16, contiguous.
        ln_weight / ln_bias: ``k_norm``'s gamma and beta, ``[128]`` bf16 or
            fp32 (whichever dtype the module holds).
        ln_eps: ``k_norm``'s epsilon.
        cos_cache / sin_cache: ``[max_seq, rope_dim]`` fp32 interleaved RoPE
            tables; only the first 32 pair entries of a row are read.
        positions: ``[N]`` int32/int64 absolute position of each token.
        page_table / slot_indices / token_indices / page_size_tokens: the GPU
            page-table lookup, identical to the un-fused writer's.
        apply_hadamard: run the normalized 128-point Hadamard rotation.
            Callers pass ``cache_applies_hadamard(aux_cache, data_dim)``:
            False on this page-split FP8 route (upstream v3.2 semantics,
            dropped on both q and k), True for a bf16 AoS indexer.
        k_fp8_out / scales_out: optional pre-allocated host-append sources (a
            captured graph MUST pass its static buffers).
        num_valid_tokens: optional ``[1]`` int32 device tensor; rows at or
            beyond it write nothing into the cache.
        emit_host_sources: when True and no buffers were passed, allocate them.

    Returns:
        ``(k_fp8, scales)`` -- the host-append sources, or ``(None, None)``
        when ``emit_host_sources`` is False and no buffers were passed.

    As in the un-fused writer, the host-append sources are written for EVERY
    row of the launch, including rows the cache write skips.
    """

    plan = plan_norm_rope_write(
        aux_cache,
        k_raw,
        ln_weight,
        ln_bias,
        cos_cache,
        sin_cache,
        positions,
        page_size_tokens,
    )

    if k_fp8_out is None and emit_host_sources:
        k_fp8_out = torch.empty(
            plan.num_tokens,
            1,
            1,
            plan.data_dim,
            dtype=torch.float8_e4m3fn,
            device=k_raw.device,
        )
    if scales_out is None and emit_host_sources:
        scales_out = torch.empty(
            plan.num_tokens, dtype=torch.float32, device=k_raw.device
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
    _module().fused_indexer_k_norm_rope_write_fp8(
        k_raw,
        ln_weight,
        ln_bias,
        cos_cache,
        sin_cache,
        positions,
        page_table,
        slot_indices,
        token_indices,
        pages,
        int(page_size_tokens),
        float(ln_eps),
        bool(apply_hadamard),
        None if k_fp8_out is None else k_fp8_out.view(
            plan.num_tokens, plan.data_dim
        ),
        scales_out,
        num_valid_tokens,
    )
    return k_fp8_out, scales_out
