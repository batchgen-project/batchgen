"""BatchGen Triton Kernels for optimized inference.

This package contains custom Triton kernels for:
- Fused MXFP4 dequantization + GEMM (for GPT-OSS-120B MoE layers)
- Other optimized operations

Usage:
    from batchgen.triton_kernels import fused_mxfp4_gemm, fused_mxfp4_mlp_forward
"""

from .fused_mxfp4_gemm import (
    fused_mxfp4_gemm,
    fused_mxfp4_mlp_forward,
)

__all__ = [
    'fused_mxfp4_gemm',
    'fused_mxfp4_mlp_forward',
]
