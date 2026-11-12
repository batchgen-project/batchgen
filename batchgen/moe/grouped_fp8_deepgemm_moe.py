"""DeepGEMM m-grouped contiguous S1/S3 for the GLM-5 streamed prefill.

Adopted over the incumbent CuTe ptrs kernels for the prefill grouped MoE
GEMMs (external-dep ruling: a directly usable faster dependency is adopted;
the incumbent kernels stay in grouped_fp8_blockwise_moe.py untouched).

The contiguous API contract (pinned sgl deep_gemm build):
  symbol((a, sfa), (b, sfb), d, grouped_layout)
  - a: fp8 [m, K] K-major; sfa: fp32 [m, K/128]
  - b: fp8 [G, N, K] with b.stride(0) == N*K (dim0-strided views are
    rejected by the build's layout check), sfb: fp32 [G, N/128, K/128]
  - d: bf16 [m, N]
  - grouped_layout: int32 [m], row -> group index, -1 marks padding rows
  - every group's row segment must start/size-align to
    deep_gemm.get_m_alignment_for_contiguous_layout() (128 in the pinned
    build). The ragged dispatch is 64-aligned today and the constant is
    SHARED with decode paths, so the alignment is threaded through the
    dispatch/quant/capacity chain as a per-path parameter (not a global
    constant flip); until that lands, build_grouped_layout fails loud on a
    64-but-not-128-aligned layout.

The dense [G, N, K] operand exists because the weight ring allocates
field-major arenas: a frame's expert slots hold each weight field at a
uniform stride, and when the field's byte size is 256-aligned the stride
equals the field size, so consecutive slots form one dense region. This
module VERIFIES that from the per-expert tensors' data pointers and fails
loud otherwise - never a silent fallback.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch
from torch import Tensor

logger = logging.getLogger("batchgen.moe.fp8_deepgemm")

_QUANT_BLOCK = 128

_deepgemm = None


def _get_deepgemm():
    global _deepgemm
    if _deepgemm is None:
        import deep_gemm  # noqa: PLC0415 - optional heavy dep, loaded on use

        symbol = getattr(deep_gemm, "m_grouped_fp8_gemm_nt_contiguous", None)
        if not callable(symbol):
            raise RuntimeError(
                "deep_gemm.m_grouped_fp8_gemm_nt_contiguous is unavailable; "
                "the DeepGEMM grouped prefill path cannot run"
            )
        align_fn = getattr(
            deep_gemm, "get_m_alignment_for_contiguous_layout", None
        )
        alignment = int(align_fn()) if callable(align_fn) else 128
        _deepgemm = (symbol, alignment)
    return _deepgemm


def deepgemm_m_alignment() -> int:
    """Per-group row alignment the contiguous API requires."""
    return _get_deepgemm()[1]


def dense_group_view(members: Sequence[Tensor]) -> Tensor:
    """One dense [G, *shape] view over per-group tensors of one arena field.

    Every member must be a same-shaped, contiguous view of the SAME storage
    at a uniform element stride equal to the member's size (field-major ring
    arena with an aligned field size). Raises on any violation; a grouped
    GEMM silently reading a stale or foreign slot is never acceptable.
    """
    if not members:
        raise ValueError("dense_group_view needs at least one member")
    base = members[0]
    if not base.is_contiguous():
        raise ValueError("group member 0 is not contiguous")
    elem = base.element_size()
    step = base.numel()
    base_ptr = base.data_ptr()
    storage_ptr = base.untyped_storage().data_ptr()
    storage_bytes = base.untyped_storage().nbytes()
    for index, member in enumerate(members):
        if member.shape != base.shape or member.dtype != base.dtype:
            raise ValueError(f"group member {index} shape/dtype mismatch")
        if member.untyped_storage().data_ptr() != storage_ptr:
            raise ValueError(
                f"group member {index} lives in a different allocation; the "
                "ring was not field-major or the frame spans two arenas"
            )
        expected = base_ptr + index * step * elem
        if member.data_ptr() != expected:
            raise ValueError(
                f"group member {index} is not at the uniform dense stride "
                f"(offset {member.data_ptr() - base_ptr} B, expected "
                f"{index * step * elem} B); the field size is likely not "
                "alignment-exact"
            )
    end = base_ptr + len(members) * step * elem
    if end > storage_ptr + storage_bytes:
        raise ValueError("dense group view exceeds the arena")
    return base.as_strided(
        (len(members), *base.shape), (step, *base.stride())
    )


def build_grouped_layout(
    seqlens: Tensor,
    cu_seqlens: Tensor,
    total_rows: int,
    alignment: Optional[int] = None,
    validate: bool = True,
) -> Tensor:
    """int32 [total_rows] row->group map with -1 padding holes.

    ``cu_seqlens[g]`` is group ``g``'s first row and ``seqlens[g]`` its live
    row count in the reserved activation buffer; rows between
    ``cu_seqlens[g] + seqlens[g]`` and ``cu_seqlens[g + 1]`` are alignment
    padding and carry -1. Each group's START must sit on the DeepGEMM row
    alignment or the API rejects the layout. ``validate`` reads the layout
    back to the host to check that, which synchronizes the stream - callers
    on a steady-state path whose dispatch structurally produces the required
    alignment (it was given the same value) pass ``validate=False``.
    """
    if alignment is None:
        alignment = deepgemm_m_alignment()
    if seqlens.ndim != 1 or cu_seqlens.ndim != 1:
        raise ValueError("seqlens and cu_seqlens must be 1-D")
    if cu_seqlens.numel() != seqlens.numel() + 1:
        raise ValueError("cu_seqlens must have one more entry than seqlens")
    starts = cu_seqlens[:-1]
    ends = cu_seqlens[1:]
    if validate:
        # gpu-sync: allow layout validation off the steady-state path
        if bool((starts % alignment != 0).any()):
            raise ValueError(
                f"a group start is not {alignment}-row aligned; the ragged "
                "dispatch alignment does not match the GEMM backend"
            )
        if int(ends[-1]) > total_rows:
            raise ValueError("cu_seqlens exceeds the reserved rows")
    device = seqlens.device
    grouped = torch.full(
        (total_rows,), -1, dtype=torch.int32, device=device
    )
    groups = seqlens.numel()
    ids = torch.arange(groups, dtype=torch.int32, device=device)
    row = torch.arange(total_rows, dtype=cu_seqlens.dtype, device=device)
    # Row r belongs to group g when cu_seqlens[g] <= r < cu_seqlens[g]+seqlens[g].
    bucket = torch.bucketize(row, ends, right=True).clamp_(max=groups - 1)
    live = row < (starts + seqlens.to(cu_seqlens.dtype))[bucket]
    live &= row >= starts[bucket]
    grouped[live] = ids[bucket[live]]
    return grouped


def _scale_rows_first(scale: Tensor, rows: int) -> Tensor:
    """SFA as fp32 [rows, K/128] from the engine's [K/128, rows] buffer.

    Transpose shim: the incumbent act-quant kernels emit scales column-major
    for the CuTe kernels. The copy is small (fp32, K/128 columns) next to the
    GEMM it feeds; a row-major quant variant replaces it later.
    """
    if scale.ndim != 2:
        raise ValueError("activation scale must be 2-D")
    if scale.shape[1] < rows:
        raise ValueError("activation scale has fewer rows than the batch")
    return scale[:, :rows].t().contiguous()


def grouped_fp8_deepgemm_gemm(
    x_fp8: Tensor,
    x_scale: Tensor,
    b: Tensor,
    sfb: Tensor,
    *,
    output: Tensor,
    grouped_layout: Tensor,
) -> Tensor:
    """One m-grouped contiguous GEMM over a dense weight stack.

    ``b`` is the dense fp8 [G, N, K] stack (built once per layer with
    :func:`dense_group_view` over the field-major ring) and ``sfb`` its
    exact-extent fp32 [G, N/128, K/128] scales. ``output`` is bf16 [m, N]
    and ``grouped_layout`` the int32 [m] row map from
    :func:`build_grouped_layout` in the same group order.
    """
    symbol, _ = _get_deepgemm()
    rows = output.shape[0]
    if grouped_layout.shape[0] != rows:
        raise ValueError("grouped_layout rows must match the output rows")
    sfa = _scale_rows_first(x_scale, rows)
    symbol((x_fp8[:rows], sfa), (b, sfb), output, grouped_layout)
    return output
