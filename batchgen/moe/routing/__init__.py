"""CUDA routing kernels for MoE (gate, dispatch, reduce)."""

from .cuda_routing import (
    gate_topk_softmax_cuda,
    gate_sigmoid_topk_cuda,
    glm5_router_gemm_cuda,
    router_bias_cast_cuda,
    dispatch_count_gather_cuda,
    reduce_weighted_scatter_cuda,
    FusedGateContext,
)

__all__ = [
    "gate_topk_softmax_cuda",
    "gate_sigmoid_topk_cuda",
    "glm5_router_gemm_cuda",
    "router_bias_cast_cuda",
    "dispatch_count_gather_cuda",
    "reduce_weighted_scatter_cuda",
    "FusedGateContext",
]
