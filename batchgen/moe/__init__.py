# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

"""MoE (Mixture of Experts) module for BatchGen.

This module provides efficient MoE implementations for MXFP4 quantized models.

Components:
- token_dispatch: Token dispatch/combine for EP=1 and EP>1
- routing: Top-k expert routing with softmax
- fused_mxfp4_gemm: Fused dequant + GEMM kernels
- moe_mxfp4: Complete MoE forward with SwiGLU

Example:
    from batchgen.moe import moe_mxfp4_forward, LocalTokenDispatcher

    output = moe_mxfp4_forward(
        x, gate_weight, gate_bias,
        w1_packed, w1_scales, w1_bias,
        w2_packed, w2_scales, w2_bias,
        experts_per_token=4,
    )
"""

from .token_dispatch import (
    DispatchResult,
    TokenDispatcher,
    LocalTokenDispatcher,
    DistributedTokenDispatcher,
    create_token_dispatcher,
)

from .routing import (
    moe_routing,
    moe_routing_with_auxiliary_loss,
    MoERouter,
    compute_expert_load_stats,
)

from .fused_mxfp4_gemm import (
    fused_mxfp4_gemm,
    fused_mxfp4_grouped_gemm,
    fused_mxfp4_moe_gemm,
    fused_mxfp4_moe_gemm_sequential,
    fused_mxfp4_moe_gemm_from_list,
)

from .moe_mxfp4 import (
    moe_mxfp4_forward,
    moe_mxfp4_forward_reference,
    MoEMXFP4Layer,
    swiglu,
)

__all__ = [
    # Token dispatch
    'DispatchResult',
    'TokenDispatcher',
    'LocalTokenDispatcher',
    'DistributedTokenDispatcher',
    'create_token_dispatcher',
    # Routing
    'moe_routing',
    'moe_routing_with_auxiliary_loss',
    'MoERouter',
    'compute_expert_load_stats',
    # GEMM
    'fused_mxfp4_gemm',
    'fused_mxfp4_grouped_gemm',
    'fused_mxfp4_moe_gemm',
    'fused_mxfp4_moe_gemm_sequential',
    'fused_mxfp4_moe_gemm_from_list',
    # MoE
    'moe_mxfp4_forward',
    'moe_mxfp4_forward_reference',
    'MoEMXFP4Layer',
    'swiglu',
]
