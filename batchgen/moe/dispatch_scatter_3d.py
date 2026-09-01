"""3D dispatch scatter + reduce kernels for strided MoE buffer layout.

Ported from GLM-5-FP8 layer_opt_pipeline (fp8_wgmma_pipeline.py).

Buffer layout: [E_local, max_tokens_padded, H] (3D strided).
Each expert e owns rows [e * mtp, (e+1) * mtp) in the flat [E*mtp, H] buffer.

dispatch_scatter_3d:
    Routes tokens from flat [G, H] into 3D [E*mtp, H] layout.
    Two-stage: count tokens per expert, then scatter with atomic counters.
    topk_pos stores absolute strided positions for reduce.

reduce_weighted_scatter:
    Weighted sum from 3D output back to flat [G, H] using topk_pos indices.
    FP32 accumulation, BF16 output. Template-specialized for K=2,4,8.
"""

import logging

import torch


_dispatch_reduce_module = None


# ──────────────────────────────────────────────────────────────────────────────
# CUDA Source Code (external .cu file)
# ──────────────────────────────────────────────────────────────────────────────
# Source: batchgen_kernels/src/moe/dispatch_scatter_3d.cu

# ──────────────────────────────────────────────────────────────────────────────
# Module Loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_dispatch_reduce_module():
    """Load the pre-compiled dispatch_scatter_3d + reduce_weighted_scatter module."""
    global _dispatch_reduce_module
    if _dispatch_reduce_module is not None:
        return _dispatch_reduce_module

    try:
        import batchgen_kernels
        _dispatch_reduce_module = batchgen_kernels.load_extension(
            "batchgen_kernels.moe._C_dispatch_scatter_3d"
        )
        logging.debug("Loaded pre-compiled dispatch_scatter_3d + reduce_weighted_scatter kernels")
        return _dispatch_reduce_module
    except Exception as e:
        logging.warning(f"Failed to load dispatch_scatter_3d kernels: {e}")
        return None


def require_dispatch_scatter_3d_kernels():
    """Load the dispatch/reduce extension once or fail closed.

    Grouped prefill calls this during configuration so the first token window
    never pays the development-loader/import-lock cost.
    """
    mod = _load_dispatch_reduce_module()
    if mod is None:
        raise RuntimeError("dispatch_scatter_3d kernels are unavailable")
    return mod


def dispatch_scatter_3d(
    x: torch.Tensor,
    topk_indices: torch.Tensor,
    act_buffer: torch.Tensor,
    expert_start: int,
    num_local_experts: int,
    max_tokens_padded: int,
    expert_counts: torch.Tensor,
    expert_counters: torch.Tensor,
    topk_pos: torch.Tensor,
):
    """Route tokens from flat [G, H] into 3D strided [E*mtp, H] buffer.

    Args:
        x: Input tokens [G, H] BF16
        topk_indices: Expert assignments [G, K] int32
        act_buffer: Pre-allocated 3D buffer [E_local * mtp, H] BF16
        expert_start: Global index of first local expert
        num_local_experts: Number of local experts
        max_tokens_padded: Stride per expert (mtp)
        expert_counts: Pre-allocated [E_local] int32 (zeroed internally)
        expert_counters: Pre-allocated [E_local] int32 (zeroed internally)
        topk_pos: Pre-allocated [G*K] int32 (set to strided positions)

    Returns:
        (expert_counts, topk_pos) — expert_counts[e] = tokens routed to expert e,
        topk_pos[i] = absolute row index in act_buffer (or -1 if non-local)
    """
    mod = _load_dispatch_reduce_module()
    return mod.dispatch_scatter_3d(
        x, topk_indices, act_buffer,
        expert_start, num_local_experts, max_tokens_padded,
        expert_counts, expert_counters, topk_pos,
    )


def reduce_weighted_scatter(
    expert_output: torch.Tensor,
    topk_pos: torch.Tensor,
    topk_weights: torch.Tensor,
    N: int,
    H: int,
    K: int,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """Weighted sum from 3D expert output back to flat [N, H].

    Args:
        expert_output: 3D strided buffer [E*mtp, H] BF16
        topk_pos: Strided positions [N*K] int32
        topk_weights: Routing weights [N, K] FP32
        N: Number of original tokens
        H: Hidden dimension
        K: Top-k value
        output: Pre-allocated output [N, H] BF16 (optional)

    Returns:
        output [N, H] BF16
    """
    mod = _load_dispatch_reduce_module()
    if output is None:
        output = torch.zeros(N, H, dtype=torch.bfloat16, device=expert_output.device)
    return mod.reduce_weighted_scatter(
        expert_output, topk_pos, topk_weights, N, H, K, output,
    )


def reduce_weighted_scatter_bf16_ordered(
    expert_output: torch.Tensor,
    topk_pos: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    N: int,
    H: int,
    K: int,
    output: torch.Tensor = None,
) -> torch.Tensor:
    """Reduce in ascending expert order with BF16 rounding at every add."""
    mod = _load_dispatch_reduce_module()
    kernel = getattr(mod, "reduce_weighted_scatter_bf16_ordered", None)
    if kernel is None:
        raise RuntimeError(
            "dispatch_scatter_3d extension lacks "
            "reduce_weighted_scatter_bf16_ordered"
        )
    if output is None:
        output = torch.empty(
            N, H, dtype=torch.bfloat16, device=expert_output.device
        )
    return kernel(
        expert_output,
        topk_pos,
        topk_indices,
        topk_weights,
        N,
        H,
        K,
        output,
    )
