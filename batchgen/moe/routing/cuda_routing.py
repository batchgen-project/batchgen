"""
CUDA routing kernels for GPT-OSS-120B MoE.

JIT-compiled CUDA extension providing gate, dispatch, and reduce kernels.
Compiled eagerly at import time so the first kernel call has no compilation stall.

Usage:
    from batchgen.moe.routing import (
        gate_topk_softmax_cuda,
        dispatch_count_gather_cuda,
        reduce_weighted_scatter_cuda,
    )
"""

import torch
from pathlib import Path
from torch.utils.cpp_extension import load

# ──────────────────────────────────────────────────────────────────────────────
# Compile at import time
# ──────────────────────────────────────────────────────────────────────────────

_csrc_dir = Path(__file__).parent / "csrc"

_cuda_ext = load(
    name="routing_cuda",
    sources=[
        str(_csrc_dir / "routing_extension.cc"),
        str(_csrc_dir / "gate_topk_softmax.cu"),
        str(_csrc_dir / "dispatch_count_gather.cu"),
        str(_csrc_dir / "reduce_weighted_scatter.cu"),
        str(_csrc_dir / "router_epilogue.cu"),
        str(_csrc_dir / "gate_sigmoid_topk.cu"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-std=c++17",
    ],
    verbose=False,
)


# ──────────────────────────────────────────────────────────────────────────────
# Python wrappers (matching Triton kernel signatures)
# ──────────────────────────────────────────────────────────────────────────────

def gate_topk_softmax_cuda(router_logits, topk_indices=None, topk_weights=None, k=4):
    """
    CUDA gate kernel: fused top-k selection + softmax.

    Args:
        router_logits: [N, E] FP32
        topk_indices: [N, K] int32 pre-allocated output (optional)
        topk_weights: [N, K] FP32 pre-allocated output (optional)
        k: top-k (default 4)

    Returns:
        topk_indices: [N, K] int32
        topk_weights: [N, K] FP32
    """
    ext = _cuda_ext
    N = router_logits.shape[0]
    device = router_logits.device

    # Kernel requires FP32 for numerical stability in softmax
    if router_logits.dtype != torch.float32:
        router_logits = router_logits.float()

    if topk_indices is None:
        topk_indices = torch.empty(N, k, dtype=torch.int32, device=device)
    if topk_weights is None:
        topk_weights = torch.empty(N, k, dtype=torch.float32, device=device)

    result = ext.gate_topk_softmax(router_logits, k, topk_indices, topk_weights)
    return result[0], result[1]


def gate_sigmoid_topk_cuda(
    router_logits, e_score_correction,
    k=8, routed_scaling_factor=2.5,
    topk_indices=None, topk_weights=None,
):
    """
    CUDA gate kernel: fused sigmoid + top-k + normalize + scale (K2.5).

    Algorithm:
        1. sigmoid(logits) → scores
        2. scores + e_score_correction → biased (for selection only)
        3. topk(biased, k) → indices
        4. gather raw sigmoid scores at indices → weights
        5. normalize weights, multiply by routed_scaling_factor

    Args:
        router_logits: [N, E] FP32
        e_score_correction: [E] FP32
        k: top-k (default 8)
        routed_scaling_factor: scaling factor (default 2.5)
        topk_indices: [N, K] int32 pre-allocated output (optional)
        topk_weights: [N, K] FP32 pre-allocated output (optional)

    Returns:
        topk_indices: [N, K] int32
        topk_weights: [N, K] FP32
    """
    ext = _cuda_ext
    N = router_logits.shape[0]
    device = router_logits.device

    if router_logits.dtype != torch.float32:
        router_logits = router_logits.float()
    if e_score_correction.dtype != torch.float32:
        e_score_correction = e_score_correction.float()

    if topk_indices is None:
        topk_indices = torch.empty(N, k, dtype=torch.int32, device=device)
    if topk_weights is None:
        topk_weights = torch.empty(N, k, dtype=torch.float32, device=device)

    result = ext.gate_sigmoid_topk(
        router_logits, e_score_correction, k, routed_scaling_factor,
        topk_indices, topk_weights,
    )
    return result[0], result[1]


def router_bias_cast_cuda(logits, bias, output):
    """Fused router epilogue: BF16 bias add + BF16→FP32 cast.

    Args:
        logits: [N, E] BF16 (router matmul output)
        bias: [E] BF16 (router bias, or empty tensor if no bias)
        output: [N, E] FP32 (pre-allocated output for gate kernel)
    """
    _cuda_ext.router_bias_cast(logits, bias, output)


def dispatch_count_gather_cuda(
    x, topk_indices,
    expert_start, num_local_experts,
    expert_counts=None, expert_offsets=None,
    expert_counters=None, dispatched_x=None, topk_pos=None,
):
    """
    CUDA dispatch kernel: count + prefix_sum + gather.

    Args:
        x: [N, H] BF16 token activations
        topk_indices: [N, K] int32 expert assignments
        expert_start: first local expert index
        num_local_experts: number of local experts
        expert_counts/offsets/counters/dispatched_x/topk_pos: pre-allocated (optional)

    Returns:
        dispatched_x: [max_dispatched, H] BF16
        expert_counts: [E_local] int32
        expert_offsets: [E_local+1] int32
        topk_pos: [N*K] int32
    """
    ext = _cuda_ext

    # Kernel requires int32 indices
    if topk_indices.dtype != torch.int32:
        topk_indices = topk_indices.to(torch.int32)

    N, K = topk_indices.shape
    H = x.shape[1]
    NK = N * K
    device = x.device
    E_local = num_local_experts

    # Allocate outputs if not pre-allocated
    if expert_counts is None:
        expert_counts = torch.zeros(E_local, dtype=torch.int32, device=device)
    else:
        expert_counts.zero_()

    if expert_offsets is None:
        expert_offsets = torch.empty(E_local + 1, dtype=torch.int32, device=device)

    if expert_counters is None:
        expert_counters = torch.zeros(E_local, dtype=torch.int32, device=device)
    else:
        expert_counters.zero_()

    if topk_pos is None:
        topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=device)

    if dispatched_x is None:
        dispatched_x = torch.empty(NK, H, dtype=x.dtype, device=device)

    result = ext.dispatch_count_gather(
        x, topk_indices,
        expert_start, num_local_experts,
        expert_counts, expert_offsets,
        expert_counters, dispatched_x, topk_pos,
    )
    return result[0], result[1], result[2], result[3]


def reduce_weighted_scatter_cuda(
    expert_output, topk_pos, topk_weights, N, H=None, K=4,
    output=None,
):
    """
    CUDA reduce kernel: weighted scatter-add.

    Args:
        expert_output: [total_dispatched, H] BF16
        topk_pos: [N*K] int32 (-1 for non-local)
        topk_weights: [N, K] FP32
        N: number of original tokens
        H: hidden size (auto-detected if None)
        K: top-k (default 4)
        output: [N, H] BF16 pre-allocated output (optional)

    Returns:
        output: [N, H] BF16
    """
    ext = _cuda_ext

    # Kernel requires FP32 weights for accumulation precision
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.float()

    if H is None:
        H = expert_output.shape[1]
    device = expert_output.device

    if output is None:
        output = torch.empty(N, H, dtype=torch.bfloat16, device=device)

    return ext.reduce_weighted_scatter(
        expert_output, topk_pos, topk_weights,
        N, H, K, output,
    )
