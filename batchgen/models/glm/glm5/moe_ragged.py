# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""Compact ragged MoE dispatch layout for GLM-5 decode (M1a-2).

The 3D path reserved a per-expert slot table ``[E_local, mtp, dim]`` so that
expert ``e`` always started at row ``e * mtp``.  At mtp=2048 / E_local=32 that
is 1.51 GiB of dispatch + result buffer per rank, of which only
``num_global * topk`` rows are ever live.

This module replaces the per-expert stride with a compact ragged layout: expert
``e`` owns rows ``[cu_seqlens[e], cu_seqlens[e] + counts[e])`` of a single
``[capacity, dim]`` buffer, where ``capacity`` is the *total* worst case
``num_global * topk + E_local * (ALIGN - 1)`` — a static host constant, so
there is no per-expert overflow left to regrow.

Segment starts are aligned up to :data:`ROW_ALIGN` = 64 rows.  This is NOT for
the activation TMA (whose descriptor is re-based per expert and would accept
any row start); it is because the grouped GEMM addresses ``x_scale`` in the
same row space, at M-tile granularity:

    scale tile index = cu_seqlens[e] / TileM + local_tile

so ``cu_seqlens[e]`` must be divisible by every supported TileM (16/32/64).
Aligning to 64 satisfies all three and — critically — keeps the kernel edit
bit-identical for the existing padded callers, whose ``cu_seqlens[e] = e * mtp``
with mtp already a multiple of 64.  See ``m1a2_spec.md`` C1 option (a).

The alignment holes are never written and never read: the GEMM's per-expert TMA
extent is ``counts[e]``, so loads past it are hardware zero-filled and stores
past it are clipped.
"""

from __future__ import annotations

import torch

# Expert segment start alignment, in rows. Must be a multiple of the largest
# grouped-GEMM TileM (64) — see module docstring.
ROW_ALIGN = 64

# Blockwise FP8 quantization block along K.
QUANT_BLOCK = 128

# TileM selector handed to the grouped FP8 GEMM (`num_seq_per_group_avg`, which
# the compiled dispatch maps to TileM 16/32/64: <=16 -> 16, <=32 -> 32, else 64).
# The fixed336 decode workload routes 336 global rows at top-k 8 over 256
# experts — 10.5 rows per expert segment on average — so a 64-row M-tile leaves
# most of each tile empty.  An exact real-routing H200 sweep over layers
# 3/40/77 (three repetitions each) selected TileM=32 as the best static choice:
# 317.62 us rank-max full-window versus 318.87 us for TileM=16 and 324.17 us
# for TileM=64. This is a static, capture-time host constant, NOT a device->host
# readback of the real counts, so it stays CUDA-graph safe. ROW_ALIGN = 64
# divides 32, so segment starts remain tile-aligned and the padded callers
# (whose `cu_seqlens[e] = e * mtp`, mtp a multiple of 64) are unaffected.
GEMM_TILEM_AVG = 32

# Row capacity is rounded to this so the x_scale row stride (capacity * 4 B) is
# always 16 B aligned for the scale TMA descriptor.
_CAPACITY_BLOCK = 128

_ops_module = None


def _require_dispatch_module():
    """Load the ragged dispatch extension, hard-failing on a stale build."""
    from batchgen.moe.dispatch_scatter_3d import (
        require_dispatch_scatter_3d_kernels,
    )

    mod = require_dispatch_scatter_3d_kernels()
    if not hasattr(mod, "dispatch_scatter_ragged"):
        raise RuntimeError(
            "batchgen_kernels.moe._C_dispatch_scatter_3d has no "
            "`dispatch_scatter_ragged` — the compiled kernels predate the GLM-5 "
            "compact ragged MoE layout (M1a-2). Rebuild batchgen_kernels on this "
            "node; there is no padded-layout fallback."
        )
    return mod


def _require_ops_module():
    """Load the ragged FP8 quant extension, hard-failing on a stale build."""
    global _ops_module
    if _ops_module is not None:
        return _ops_module
    import batchgen_kernels

    mod = batchgen_kernels.load_extension("batchgen_kernels.moe._C_fp8_blockwise_ops")
    if not hasattr(mod, "act_quant_ragged"):
        raise RuntimeError(
            "batchgen_kernels.moe._C_fp8_blockwise_ops has no `act_quant_ragged` "
            "— the compiled kernels predate the GLM-5 compact ragged MoE layout "
            "(M1a-2). Rebuild batchgen_kernels on this node; there is no "
            "padded-layout fallback."
        )
    _ops_module = mod
    return mod


def require_ragged_kernels():
    """Load both compact dispatch/reduce and FP8 quantization extensions."""
    dispatch = _require_dispatch_module()
    if not hasattr(dispatch, "reduce_weighted_scatter_bf16_ordered"):
        raise RuntimeError(
            "batchgen_kernels.moe._C_dispatch_scatter_3d has no "
            "`reduce_weighted_scatter_bf16_ordered`"
        )
    return dispatch, _require_ops_module()


def ragged_row_capacity(max_global_tokens: int, topk: int, num_local_experts: int) -> int:
    """Static worst-case row count of the compact dispatch buffer.

    Every routed (token, expert) pair contributes at most one row, so the live
    rows are bounded by ``max_global_tokens * topk`` even if one rank owns every
    selected expert.  Each of the ``E_local`` segments then rounds its start up
    by at most ``ROW_ALIGN - 1``.
    """
    nk = int(max_global_tokens) * int(topk)
    raw = nk + int(num_local_experts) * (ROW_ALIGN - 1)
    return ((raw + _CAPACITY_BLOCK - 1) // _CAPACITY_BLOCK) * _CAPACITY_BLOCK


def make_quant_buffers(rows: int, dim: int, device: torch.device):
    """Persistent FP8 activation + scale buffers for one ragged GEMM stage.

    ``scale`` is transposed ``[dim/128, rows]`` — the grouped GEMM's x_scale
    layout — and MUST be zero-initialised, because ``act_quant_ragged`` only
    writes live rows and the GEMM still TMA-loads whole M-tiles.  A stale finite
    value in an alignment hole is harmless (it multiplies a hardware
    zero-filled activation row); a NaN from uninitialised memory would not be.
    """
    if dim % QUANT_BLOCK != 0:
        raise ValueError(f"ragged quant dim must be a multiple of {QUANT_BLOCK}, got {dim}")
    y = torch.empty(rows, dim, dtype=torch.uint8, device=device)
    scale = torch.zeros(dim // QUANT_BLOCK, rows, dtype=torch.float32, device=device)
    return y, scale


def dispatch_scatter_ragged(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    act_buffer: torch.Tensor,
    expert_start: int,
    num_local_experts: int,
    expert_counts: torch.Tensor,
    expert_counters: torch.Tensor,
    cu_seqlens: torch.Tensor,
    topk_pos: torch.Tensor,
):
    """Route tokens from flat ``[G, H]`` into the compact ragged buffer.

    Returns ``(expert_counts, cu_seqlens, topk_pos)``.  ``cu_seqlens`` is
    written on device (``[E_local+1]`` int32, 64-aligned segment starts) and
    feeds both :func:`act_quant_ragged` and the grouped GEMM.  ``topk_pos[i]``
    is the compact row of routed slot ``i`` (or -1 when non-local), so
    ``reduce_weighted_scatter`` is unchanged.
    """
    return _require_dispatch_module().dispatch_scatter_ragged(
        x,
        topk_indices,
        act_buffer,
        expert_start,
        num_local_experts,
        ROW_ALIGN,
        expert_counts,
        expert_counters,
        cu_seqlens,
        topk_pos,
    )


def act_quant_ragged(
    x: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    y: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """BF16 -> FP8 blockwise quantization over the live rows of ``x``.

    ``y``/``scale`` come from :func:`make_quant_buffers` and are written in
    place; ``scale`` is already in grouped-GEMM order, so no ``.t()`` follows.
    """
    _require_ops_module().act_quant_ragged(x, seqlens, cu_seqlens, y, scale)
