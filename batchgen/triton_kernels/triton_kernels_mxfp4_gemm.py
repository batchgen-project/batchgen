"""MXFP4 GEMM using OpenAI's triton_kernels package.

This module provides a wrapper around triton_kernels.matmul for single expert GEMM
with MXFP4 quantized weights. It serves as an alternative to the custom fused kernel.

Usage:
    from batchgen.triton_kernels.triton_kernels_mxfp4_gemm import triton_kernels_mxfp4_gemm

    # Single linear: out = x @ dequant(weight).T + bias
    out = triton_kernels_mxfp4_gemm(x, weight_packed, weight_scales, bias)

Requirements:
    pip install -e /path/to/triton/python/triton_kernels
"""

import torch
from typing import Optional

try:
    from triton_kernels.matmul import matmul, PrecisionConfig
    from triton_kernels.tensor import wrap_torch_tensor
    TRITON_KERNELS_AVAILABLE = True
except ImportError:
    TRITON_KERNELS_AVAILABLE = False


def triton_kernels_mxfp4_gemm(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single expert GEMM using triton_kernels.matmul.

    This function wraps OpenAI's triton_kernels.matmul which has native MXFP4 support.
    The uint8 weight tensor is automatically treated as FP4 when passed with scales.

    Args:
        x: Input activations [M, K] in BF16
        weight_packed: Packed FP4 weights [N, K//2] in uint8
        weight_scales: Scales [N, K//32] in uint8
        bias: Optional bias [N] in BF16

    Returns:
        Output [M, N] in BF16

    Note:
        triton_kernels expects column-major weights, so we transpose internally.
        The weight layout should be [N, K//2] where each byte contains 2 FP4 values.
    """
    if not TRITON_KERNELS_AVAILABLE:
        raise ImportError(
            "triton_kernels package not available. "
            "Install with: pip install -e /path/to/triton/python/triton_kernels"
        )

    # Save original shape for reshaping output
    original_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])  # [M, K]
    M, K = x_2d.shape

    # Handle 3D weight format: [N, K//32, 16] -> [N, K//2]
    if weight_packed.dim() == 3:
        N, G, B = weight_packed.shape
        weight_packed = weight_packed.view(N, G * B)

    N = weight_packed.shape[0]
    K_packed = weight_packed.shape[1]
    assert K == K_packed * 2, f"K mismatch: x has K={K}, weight has K={K_packed * 2}"

    # triton_kernels expects column-major weights with shape [K, N]
    # Our weights are [N, K//2] with packed FP4, so effective shape is [N, K]
    # We need to transpose to [K//2, N] for the packed representation
    # IMPORTANT: Do NOT call .contiguous() - transpose creates column-major view
    # which is required by triton_kernels (stride(-2) == 1)
    weight_T = weight_packed.T  # [K//2, N] uint8, column-major (strides: 1, K//2)

    # Transpose scales: [N, K//32] -> [K//32, N]
    # IMPORTANT: Use .contiguous() to make scales row-major (stride[-1] == 1)
    # This enables TMA (Tensor Memory Accelerator) in triton_kernels (matmul.py:327-332)
    # Without TMA, large tensors fail with ~33% error
    scales_T = weight_scales.T.contiguous()  # [K//32, N] uint8, row-major (strides: N, 1)

    # Wrap scales as triton_kernels Tensor
    scales_tensor = wrap_torch_tensor(scales_T)

    # Configure MXFP4 scales
    # b_mx_scale tells triton_kernels to treat weight as MXFP4 and apply scales
    pc = PrecisionConfig(b_mx_scale=scales_tensor)

    # Call triton_kernels.matmul
    # - x: [M, K] BF16
    # - weight_T: [K//2, N] uint8 (auto-detected as FP4 because dtype is uint8)
    # NOTE: Don't pass bias to matmul - triton_kernels has a type mismatch bug when
    # bias is BF16 but accumulator is FP32 (the else branch creates FP32 zeros)
    output = matmul(x_2d, weight_T, None, precision_config=pc)

    # Add bias manually (workaround for triton_kernels type mismatch bug)
    # Ensure proper dtype handling: output may be FP32 (accumulator), bias is BF16
    if bias is not None:
        # Convert output to match input dtype (BF16) before adding bias
        # This ensures numerical consistency with the expected output format
        if output.dtype != x.dtype:
            output = output.to(x.dtype)
        output = output + bias

    # Reshape output to match input batch dimensions
    output = output.view(*original_shape[:-1], N)

    return output


def triton_kernels_mxfp4_mlp_forward(
    x: torch.Tensor,
    gate_packed: torch.Tensor,
    gate_scales: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    up_packed: torch.Tensor,
    up_scales: torch.Tensor,
    up_bias: Optional[torch.Tensor],
    down_packed: torch.Tensor,
    down_scales: torch.Tensor,
    down_bias: Optional[torch.Tensor],
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    """Full MLP forward using triton_kernels for MXFP4 GEMM.

    Implements OpenAI SwiGLU: gate * sigmoid(alpha * gate) * (up + 1)

    Args:
        x: Input [*, hidden_size] in BF16
        gate_packed, gate_scales, gate_bias: Gate projection weights
        up_packed, up_scales, up_bias: Up projection weights
        down_packed, down_scales, down_bias: Down projection weights
        alpha: SwiGLU alpha parameter (default 1.702 for OpenAI)
        limit: Clamping limit (default 7.0)

    Returns:
        Output [*, hidden_size] in BF16
    """
    # Stage 1: Gate and Up projections
    gate_out = triton_kernels_mxfp4_gemm(x, gate_packed, gate_scales, gate_bias)
    up_out = triton_kernels_mxfp4_gemm(x, up_packed, up_scales, up_bias)

    # Stage 2: OpenAI SwiGLU activation
    # gate * sigmoid(alpha * gate) * (up + 1)
    gate_clamped = gate_out.clamp(max=limit)
    up_clamped = up_out.clamp(min=-limit, max=limit)

    glu = gate_clamped * torch.sigmoid(alpha * gate_clamped)
    intermediate = glu * (up_clamped + 1)

    # Stage 3: Down projection
    output = triton_kernels_mxfp4_gemm(intermediate, down_packed, down_scales, down_bias)

    return output
