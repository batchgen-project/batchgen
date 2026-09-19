"""Compact ragged workspace helpers for GLM-5 grouped prefill."""

from __future__ import annotations

import torch

from batchgen.moe.dispatch_scatter_3d import (
    dispatch_scatter_ragged as _dispatch_scatter_ragged,
)


ROW_ALIGN = 64
QUANT_BLOCK = 128
_CAPACITY_BLOCK = 128

_ops_module = None


def _require_dispatch_module():
    from batchgen.moe.dispatch_scatter_3d import (
        require_dispatch_scatter_3d_kernels,
    )

    module = require_dispatch_scatter_3d_kernels()
    for symbol in (
        "dispatch_scatter_ragged",
        "reduce_weighted_scatter_bf16_ordered",
    ):
        if not hasattr(module, symbol):
            raise RuntimeError(
                "batchgen_kernels.moe._C_dispatch_scatter_3d has no "
                f"{symbol}; rebuild batchgen_kernels"
            )
    return module


def _require_ops_module():
    global _ops_module
    if _ops_module is not None:
        return _ops_module

    import batchgen_kernels

    module = batchgen_kernels.load_extension(
        "batchgen_kernels.moe._C_fp8_blockwise_ops"
    )
    if not hasattr(module, "act_quant_ragged"):
        raise RuntimeError(
            "batchgen_kernels.moe._C_fp8_blockwise_ops has no "
            "act_quant_ragged; rebuild batchgen_kernels"
        )
    _ops_module = module
    return module


def require_ragged_kernels():
    """Load every compact-ragged kernel required by grouped prefill."""
    return _require_dispatch_module(), _require_ops_module()


def ragged_row_capacity(
    max_global_tokens: int,
    topk: int,
    num_local_experts: int,
) -> int:
    """Return the static row capacity for 64-row-aligned expert segments."""
    live_rows = int(max_global_tokens) * int(topk)
    raw_rows = live_rows + int(num_local_experts) * (ROW_ALIGN - 1)
    return (
        (raw_rows + _CAPACITY_BLOCK - 1) // _CAPACITY_BLOCK
    ) * _CAPACITY_BLOCK


def make_quant_buffers(rows: int, dim: int, device: torch.device):
    """Allocate persistent FP8 activation and grouped-GEMM scale buffers."""
    if dim % QUANT_BLOCK != 0:
        raise ValueError(
            f"ragged quant dim must be a multiple of {QUANT_BLOCK}, got {dim}"
        )
    quantized = torch.empty(rows, dim, dtype=torch.uint8, device=device)
    scale = torch.zeros(
        dim // QUANT_BLOCK,
        rows,
        dtype=torch.float32,
        device=device,
    )
    return quantized, scale


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
    """Dispatch through the current compact-ragged wrapper API."""
    return _dispatch_scatter_ragged(
        x,
        topk_indices,
        act_buffer,
        expert_start,
        num_local_experts,
        expert_counts,
        expert_counters,
        topk_pos,
        cu_seqlens,
    )


def act_quant_ragged(
    x: torch.Tensor,
    seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    output: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Quantize live ragged rows into caller-owned output buffers."""
    _require_ops_module().act_quant_ragged(
        x,
        seqlens,
        cu_seqlens,
        output,
        scale,
    )


def reduce_weighted_scatter_bf16_ordered(
    expert_output: torch.Tensor,
    topk_pos: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    topk: int,
    *,
    output: torch.Tensor,
) -> torch.Tensor:
    """Call the ordered BF16 reducer without a module-import dependency."""
    dispatch, _ = require_ragged_kernels()
    return dispatch.reduce_weighted_scatter_bf16_ordered(
        expert_output,
        topk_pos,
        topk_indices,
        topk_weights,
        num_tokens,
        hidden_size,
        topk,
        output,
    )
