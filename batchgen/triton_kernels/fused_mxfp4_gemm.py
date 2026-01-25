"""Fused MXFP4 Dequantization + GEMM Triton Kernels.

This module provides optimized Triton kernels for GPT-OSS-120B MLP layers that:
1. Dequantize MXFP4 weights on-the-fly during GEMM (no intermediate BF16 allocation)
2. Achieve significant speedup over unfused PyTorch path

MXFP4 Format:
- 2 FP4 values per uint8 byte (low nibble bits 0-3, high nibble bits 4-7)
- 32 FP4 values share one scale (uint8, exponent = scale - 127)
- FP4 lookup: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
- Dequant: value = fp4_lookup[nibble] * 2^exponent

Usage:
    from batchgen.triton_kernels.fused_mxfp4_gemm import fused_mxfp4_gemm

    # Single linear: out = x @ dequant(weight).T + bias
    out = fused_mxfp4_gemm(x, weight_packed, weight_scales, bias)
"""

import torch
import triton
import triton.language as tl
from typing import Optional


# =============================================================================
# Helper Functions (JIT compiled)
# =============================================================================

@triton.jit
def fp4_lookup(idx):
    """Lookup FP4 value from 4-bit index.

    FP4 table:
    0: 0.0,  1: 0.5,  2: 1.0,  3: 1.5,  4: 2.0,  5: 3.0,  6: 4.0,  7: 6.0
    8: -0.0, 9: -0.5, 10: -1.0, 11: -1.5, 12: -2.0, 13: -3.0, 14: -4.0, 15: -6.0
    """
    # Use bit manipulation for sign and magnitude
    # Sign: bit 3 (idx >= 8)
    sign = tl.where(idx >= 8, -1.0, 1.0)
    # Magnitude index: bits 0-2
    mag_idx = idx & 0x07

    # Magnitude lookup: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    mag = tl.where(mag_idx == 0, 0.0, 0.0)
    mag = tl.where(mag_idx == 1, 0.5, mag)
    mag = tl.where(mag_idx == 2, 1.0, mag)
    mag = tl.where(mag_idx == 3, 1.5, mag)
    mag = tl.where(mag_idx == 4, 2.0, mag)
    mag = tl.where(mag_idx == 5, 3.0, mag)
    mag = tl.where(mag_idx == 6, 4.0, mag)
    mag = tl.where(mag_idx == 7, 6.0, mag)

    return (sign * mag).to(tl.float32)


@triton.jit
def ldexp_triton(mantissa, exponent):
    """Compute mantissa * 2^exponent using bit manipulation.

    For float32: exponent stored in bits 23-30 (biased by 127)
    """
    # Clamp exponent to valid float32 range
    exp_clamped = tl.minimum(tl.maximum(exponent, -126), 127)

    # Create 2^exponent as float32
    # float32 bit layout: [sign(1)][exponent(8)][mantissa(23)]
    exp_bits = (exp_clamped + 127).to(tl.int32) << 23
    power_of_2 = exp_bits.to(tl.float32, bitcast=True)

    return mantissa * power_of_2


# =============================================================================
# Fused MXFP4 GEMM Kernel
# =============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_mxfp4_gemm_kernel(
    # Pointers
    lhs_ptr,           # [M, K] BF16 input activations
    rhs_packed_ptr,    # [N, K//2] uint8 packed FP4 weights
    rhs_scales_ptr,    # [N, K//32] uint8 scales
    bias_ptr,          # [N] BF16 bias (or nullptr)
    out_ptr,           # [M, N] BF16 output
    # Dimensions
    M, N, K,
    # Strides
    stride_lhs_m, stride_lhs_k,
    stride_rhs_n, stride_rhs_k,       # For packed: K//2 per row
    stride_scales_n, stride_scales_k, # For scales: K//32 per row
    stride_out_m, stride_out_n,
    # Flags
    HAS_BIAS: tl.constexpr,
    # Block sizes (BLOCK_K must be 32 for scale alignment)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused MXFP4 dequantization + GEMM.

    Computes: out = lhs @ dequant(rhs).T + bias

    Key insight: We dequantize rhs on-the-fly in the inner loop, avoiding
    the need to materialize the full BF16 weight matrix.

    BLOCK_K is fixed at 32 to match the MXFP4 scale block size (32 values per scale).
    This simplifies the kernel since each K block uses exactly one scale per row.

    Grid: (num_M_blocks, num_N_blocks)
    """
    # Block indices
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute block start positions
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    # Block offsets
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)

    # Initialize accumulator in FP32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop: iterate over K dimension in blocks of BLOCK_K (which equals 32)
    K_packed = K // 2  # Number of packed bytes per row
    BLOCK_K_HALF: tl.constexpr = BLOCK_K // 2  # 16 bytes per K block

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_block * BLOCK_K

        # Load scales for this K block (one scale covers exactly BLOCK_K=32 values)
        scale_idx = k_start // 32
        scale_ptrs = rhs_scales_ptr + offs_n * stride_scales_n + scale_idx * stride_scales_k
        scale_mask = offs_n < N
        scales = tl.load(scale_ptrs, mask=scale_mask, other=127)  # [BLOCK_N] uint8

        # Convert scale: exponent = scale - 127
        exponents = scales.to(tl.int32) - 127  # [BLOCK_N]

        # Load RHS packed tile [BLOCK_N, BLOCK_K//2]
        # Each byte contains 2 FP4 values (low nibble = even idx, high nibble = odd idx)
        offs_k_packed = k_start // 2 + tl.arange(0, BLOCK_K_HALF)
        rhs_ptrs = rhs_packed_ptr + offs_n[:, None] * stride_rhs_n + offs_k_packed[None, :] * stride_rhs_k
        rhs_mask = (offs_n[:, None] < N) & (offs_k_packed[None, :] < K_packed)
        rhs_packed = tl.load(rhs_ptrs, mask=rhs_mask, other=0)  # [BLOCK_N, 16] uint8

        # Extract FP4 indices from packed bytes
        idx_lo = (rhs_packed & 0x0F).to(tl.int32)          # Low nibble: even K positions
        idx_hi = ((rhs_packed >> 4) & 0x0F).to(tl.int32)   # High nibble: odd K positions

        # Lookup FP4 values
        val_lo = fp4_lookup(idx_lo)  # [BLOCK_N, 16] float32
        val_hi = fp4_lookup(idx_hi)  # [BLOCK_N, 16] float32

        # Broadcast exponents to match val_lo/val_hi shape [BLOCK_N, 16]
        exp_broadcast = exponents[:, None] + tl.zeros((1, BLOCK_K_HALF), dtype=tl.int32)

        # Apply ldexp: val * 2^exponent
        val_lo_scaled = ldexp_triton(val_lo, exp_broadcast)  # [BLOCK_N, 16] float32
        val_hi_scaled = ldexp_triton(val_hi, exp_broadcast)  # [BLOCK_N, 16] float32

        # NOTE: Keep weights in FP32 for precision (no BF16 conversion here)
        # This avoids TF32 precision loss in tl.dot

        # Load LHS even positions: lhs[:, k_start+0], lhs[:, k_start+2], ..., lhs[:, k_start+30]
        offs_k_even = k_start + tl.arange(0, BLOCK_K_HALF) * 2
        lhs_even_ptrs = lhs_ptr + offs_m[:, None] * stride_lhs_m + offs_k_even[None, :] * stride_lhs_k
        lhs_even_mask = (offs_m[:, None] < M) & (offs_k_even[None, :] < K)
        lhs_even = tl.load(lhs_even_ptrs, mask=lhs_even_mask, other=0.0)  # [BLOCK_M, 16]

        # Load LHS odd positions: lhs[:, k_start+1], lhs[:, k_start+3], ..., lhs[:, k_start+31]
        offs_k_odd = k_start + tl.arange(0, BLOCK_K_HALF) * 2 + 1
        lhs_odd_ptrs = lhs_ptr + offs_m[:, None] * stride_lhs_m + offs_k_odd[None, :] * stride_lhs_k
        lhs_odd_mask = (offs_m[:, None] < M) & (offs_k_odd[None, :] < K)
        lhs_odd = tl.load(lhs_odd_ptrs, mask=lhs_odd_mask, other=0.0)  # [BLOCK_M, 16]

        # Compute: lhs_even @ val_lo.T + lhs_odd @ val_hi.T
        # Use FP32 for inputs to avoid TF32 precision loss
        # val_lo_scaled.T: [16, BLOCK_N]
        acc += tl.dot(lhs_even.to(tl.float32), tl.trans(val_lo_scaled))
        acc += tl.dot(lhs_odd.to(tl.float32), tl.trans(val_hi_scaled))

    # Add bias if present
    if HAS_BIAS:
        bias_vals = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias_vals[None, :].to(tl.float32)

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


# =============================================================================
# Python Wrapper Functions
# =============================================================================

def fused_mxfp4_gemm(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused MXFP4 dequant + GEMM for a single linear layer.

    Computes: out = x @ dequant(weight).T + bias

    This avoids materializing the full BF16 weight matrix, saving memory
    and improving performance.

    Args:
        x: Input activations [*, K] in BF16
        weight_packed: Packed FP4 weights [N, K//2] or [N, K//32, 16] in uint8
        weight_scales: Scales [N, K//32] in uint8
        bias: Optional bias [N] in BF16

    Returns:
        Output [*, N] in BF16
    """
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

    # Ensure inputs are contiguous
    x_2d = x_2d.contiguous()
    weight_packed = weight_packed.contiguous()
    weight_scales = weight_scales.contiguous()

    # Ensure correct dtypes
    assert x_2d.dtype == torch.bfloat16, f"x must be BF16, got {x_2d.dtype}"
    assert weight_packed.dtype == torch.uint8, f"weight_packed must be uint8, got {weight_packed.dtype}"
    assert weight_scales.dtype == torch.uint8, f"weight_scales must be uint8, got {weight_scales.dtype}"

    # Allocate output
    output = torch.empty((M, N), dtype=torch.bfloat16, device=x_2d.device)

    # Grid: one block per (M_tile, N_tile)
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

    # Launch kernel
    fused_mxfp4_gemm_kernel[grid](
        x_2d, weight_packed, weight_scales,
        bias if bias is not None else x_2d,  # Pass dummy ptr if no bias
        output,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight_packed.stride(0), weight_packed.stride(1),
        weight_scales.stride(0), weight_scales.stride(1),
        output.stride(0), output.stride(1),
        HAS_BIAS=(bias is not None),
    )

    # Reshape output to match input batch dimensions
    output = output.view(*original_shape[:-1], N)

    return output


def fused_mxfp4_mlp_forward(
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
    """Full MLP forward with fused MXFP4 GEMM for gate, up, and down projections.

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
    # Stage 1: Gate and Up projections (fused dequant + GEMM)
    gate_out = fused_mxfp4_gemm(x, gate_packed, gate_scales, gate_bias)
    up_out = fused_mxfp4_gemm(x, up_packed, up_scales, up_bias)

    # Stage 2: OpenAI SwiGLU activation
    # gate * sigmoid(alpha * gate) * (up + 1)
    gate_clamped = gate_out.clamp(max=limit)
    up_clamped = up_out.clamp(min=-limit, max=limit)

    glu = gate_clamped * torch.sigmoid(alpha * gate_clamped)
    intermediate = glu * (up_clamped + 1)

    # Stage 3: Down projection (fused dequant + GEMM)
    output = fused_mxfp4_gemm(intermediate, down_packed, down_scales, down_bias)

    return output
