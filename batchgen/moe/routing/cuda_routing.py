"""
CUDA routing kernels for GPT-OSS-120B MoE.

JIT-compiled CUDA extension providing gate, dispatch, and reduce kernels.
These are CUDA equivalents of the Triton routing kernels for benchmarking.

Usage:
    from moe.gptoss.routing.cuda_routing import (
        gate_topk_softmax_cuda,
        dispatch_count_gather_cuda,
        reduce_weighted_scatter_cuda,
    )
"""

import os
import torch
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# JIT compilation
# ──────────────────────────────────────────────────────────────────────────────

_cuda_ext = None
_compile_error = None


def _get_cuda_ext():
    """Lazily compile and load the CUDA extension."""
    global _cuda_ext, _compile_error

    if _cuda_ext is not None:
        return _cuda_ext

    if _compile_error is not None:
        raise _compile_error

    try:
        from torch.utils.cpp_extension import load

        csrc_dir = Path(__file__).parent / "csrc"

        _cuda_ext = load(
            name="routing_cuda",
            sources=[
                str(csrc_dir / "routing_extension.cc"),
                str(csrc_dir / "gate_topk_softmax.cu"),
                str(csrc_dir / "dispatch_count_gather.cu"),
                str(csrc_dir / "reduce_weighted_scatter.cu"),
            ],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-std=c++17",
            ],
            verbose=False,
        )
        return _cuda_ext

    except Exception as e:
        _compile_error = RuntimeError(
            f"Failed to compile CUDA routing kernels: {e}\n"
            "Make sure CUDA toolkit is installed and nvcc is available."
        )
        raise _compile_error


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
    ext = _get_cuda_ext()
    N = router_logits.shape[0]
    device = router_logits.device

    if topk_indices is None:
        topk_indices = torch.empty(N, k, dtype=torch.int32, device=device)
    if topk_weights is None:
        topk_weights = torch.empty(N, k, dtype=torch.float32, device=device)

    result = ext.gate_topk_softmax(router_logits, k, topk_indices, topk_weights)
    return result[0], result[1]


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
    ext = _get_cuda_ext()
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
    ext = _get_cuda_ext()

    if H is None:
        H = expert_output.shape[1]
    device = expert_output.device

    if output is None:
        output = torch.empty(N, H, dtype=torch.bfloat16, device=device)

    return ext.reduce_weighted_scatter(
        expert_output, topk_pos, topk_weights,
        N, H, K, output,
    )
