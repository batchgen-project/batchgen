"""Decoupled MXFP4 MoE: Batch Dequantization + BF16 Grouped GEMM.

This module implements a high-performance MoE layer by decoupling the MXFP4
dequantization from the grouped GEMM computation:

1. Batch Dequantization Kernel: Dequantize all 128 experts' weights in ONE
   highly parallel kernel launch (embarrassingly parallel, high bandwidth)

2. BF16 Grouped GEMM Kernel: Standard grouped GEMM on pre-dequantized BF16
   weights using optimal tensor core patterns (no FP4 lookup overhead)

This approach is 3-5x faster than the fused MXFP4 GEMM because:
- FP4 decoding uses E2M1 arithmetic (pure compute, no memory loads)
- Scale application is done once during dequant
- BF16 GEMM can use larger BLOCK_K (64/128 vs 32)
- Optimal tensor core utilization without dequant overhead in inner loop

Memory Requirements (GPT-OSS-120B, 128 experts):
- Per projection: 18.1 GB BF16 (128 experts × 141.6 MB each)
- With buffer reuse: Peak 18.1 GB (one buffer shared across projections)

Usage:
    from batchgen.moe.decoupled_mxfp4_moe import DecoupledMXFP4MoE

    # Replace existing MoE layer
    moe = DecoupledMXFP4MoE(config)
    moe.load_mxfp4_weights(gate_w, gate_s, up_w, up_s, down_w, down_s)
    output = moe(hidden_states)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from typing import List, Tuple, Optional
import time
import logging

# MXFP4 configuration (same as mxfp4_grouped_gemm.py)
MXFP4_BLOCK_SIZE = 32  # FP4 values per scale
MXFP4_PACKED_BLOCK_SIZE = 16  # Bytes per scale (32 values / 2 per byte)

# =============================================================================
# FP4 Decode Functions - Multiple Versions for Benchmarking
# =============================================================================
#
# FP4 is E2M1 format (2 exponent bits, 1 mantissa bit):
#   4-bit layout: [Sign][Exp1][Exp0][Mant]
#   - Bit 3: Sign (0=positive, 1=negative)
#   - Bits 1-2: Exponent (0-3)
#   - Bit 0: Mantissa (0 or 1)
#
# FP4 LUT (index → value):
#   0: 0.0,  1: 0.5,  2: 1.0,  3: 1.5,  4: 2.0,  5: 3.0,  6: 4.0,  7: 6.0
#   8: -0.0, 9: -0.5, 10: -1.0, 11: -1.5, 12: -2.0, 13: -3.0, 14: -4.0, 15: -6.0
#
# Available decode versions:
#   v1_sequential: 16 tl.where() (baseline, slowest)
#   v2_e2m1: E2M1 arithmetic (5-6 tl.where)
#   v3_binary_tree: Binary tree lookup (4 tl.where, optimized branching)
#   v4_branchless: Branchless bit manipulation (construct IEEE float directly)

# Version names for benchmarking
FP4_DECODE_VERSIONS = ["v1_sequential", "v2_e2m1", "v3_binary_tree", "v4_branchless", "v5_memopt", "v6_scale_transpose", "v7_fast_scale", "v8_ieee_pow2"]

# =============================================================================
# V1: Sequential 16 tl.where() - BASELINE (slowest)
# =============================================================================

@triton.jit
def _fp4_decode_v1_sequential(idx):
    """Original FP4 decode with 16 sequential tl.where() calls.

    This is the baseline/slowest implementation.
    16 sequential conditionals per FP4 value.

    Args:
        idx: 4-bit FP4 index (0-15)
    Returns:
        Decoded float32 values
    """
    val = tl.where(idx == 0, 0.0, 0.0)
    val = tl.where(idx == 1, 0.5, val)
    val = tl.where(idx == 2, 1.0, val)
    val = tl.where(idx == 3, 1.5, val)
    val = tl.where(idx == 4, 2.0, val)
    val = tl.where(idx == 5, 3.0, val)
    val = tl.where(idx == 6, 4.0, val)
    val = tl.where(idx == 7, 6.0, val)
    val = tl.where(idx == 8, -0.0, val)
    val = tl.where(idx == 9, -0.5, val)
    val = tl.where(idx == 10, -1.0, val)
    val = tl.where(idx == 11, -1.5, val)
    val = tl.where(idx == 12, -2.0, val)
    val = tl.where(idx == 13, -3.0, val)
    val = tl.where(idx == 14, -4.0, val)
    val = tl.where(idx == 15, -6.0, val)
    return val.to(tl.float32)


# =============================================================================
# V2: E2M1 Arithmetic Decode (5-6 tl.where)
# =============================================================================

@triton.jit
def _fp4_decode_v2_e2m1(idx):
    """Decode FP4 E2M1 format using pure arithmetic.

    E2M1 formula:
      exp=0 (subnormal): val = mant * 0.5  → {0.0, 0.5}
      exp>0 (normal):    val = (1 + mant*0.5) * 2^(exp-1)

    Uses 5-6 tl.where() calls (faster than v1_sequential).

    Args:
        idx: 4-bit FP4 index (0-15)
    Returns:
        Decoded float32 values
    """
    # Extract fields from 4-bit index
    abs_idx = idx & 0x7           # Remove sign bit (bits 0-2)
    exp = abs_idx >> 1            # Exponent: bits 1-2 (values 0-3)
    mant = abs_idx & 1            # Mantissa: bit 0 (0 or 1)

    # Convert to float for arithmetic
    mant_f = mant.to(tl.float32)

    # Compute 2^(exp-1) for normal values
    pow2 = tl.where(exp == 1, 1.0,
           tl.where(exp == 2, 2.0,
           tl.where(exp == 3, 4.0, 1.0)))

    # E2M1 decode
    val = tl.where(exp == 0,
                   mant_f * 0.5,                    # Subnormal: 0.0 or 0.5
                   (1.0 + mant_f * 0.5) * pow2)     # Normal: (1+M/2) * 2^(E-1)

    # Apply sign (bit 3)
    sign = (idx >> 3) & 1
    return tl.where(sign, -val, val)


# =============================================================================
# V3: Binary Tree Lookup (4 tl.where - optimized branching)
# =============================================================================

@triton.jit
def _fp4_decode_v3_binary_tree(idx):
    """Decode FP4 using binary tree lookup (4 tl.where calls).

    Binary search through values: O(log2(8)) = 3 comparisons for magnitude,
    plus 1 for sign = 4 total.

    FP4 absolute values (idx & 0x7):
      0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0, 6: 4.0, 7: 6.0

    Tree structure (split at median values):
      Level 1: idx < 4? → [0,1,2,3] vs [4,5,6,7]
      Level 2: idx < 2? or idx < 6? → further splits
      Level 3: final selection

    Args:
        idx: 4-bit FP4 index (0-15)
    Returns:
        Decoded float32 values
    """
    abs_idx = idx & 0x7  # Get absolute value index (0-7)

    # Binary tree lookup for absolute value (3 levels)
    # Level 1: split at 4 (values 0-3 vs 4-7)
    # Level 2: split at 2 and 6
    # Level 3: split at 1, 3, 5, 7

    # Use nested ternary pattern for binary tree
    # Left subtree (0-3): 0.0, 0.5, 1.0, 1.5
    # Right subtree (4-7): 2.0, 3.0, 4.0, 6.0

    val = tl.where(abs_idx < 4,
        # Left subtree: [0, 1, 2, 3] → [0.0, 0.5, 1.0, 1.5]
        tl.where(abs_idx < 2,
            tl.where(abs_idx == 0, 0.0, 0.5),     # 0→0.0, 1→0.5
            tl.where(abs_idx == 2, 1.0, 1.5)      # 2→1.0, 3→1.5
        ),
        # Right subtree: [4, 5, 6, 7] → [2.0, 3.0, 4.0, 6.0]
        tl.where(abs_idx < 6,
            tl.where(abs_idx == 4, 2.0, 3.0),     # 4→2.0, 5→3.0
            tl.where(abs_idx == 6, 4.0, 6.0)      # 6→4.0, 7→6.0
        )
    )

    # Apply sign (bit 3): 1 more tl.where
    sign = (idx >> 3) & 1
    return tl.where(sign, -val, val)


# =============================================================================
# V4: Branchless Bit Manipulation (construct IEEE float directly)
# =============================================================================

@triton.jit
def _fp4_decode_v4_branchless(idx):
    """Decode FP4 using branchless bit manipulation.

    Constructs IEEE 754 float32 representation directly from FP4 bits.
    Minimizes conditional branches by using arithmetic on bit fields.

    FP4 E2M1 layout: [S][E1][E0][M]
    IEEE 754 float32: [S][8-bit exp][23-bit mantissa]

    Strategy:
    - Handle subnormals (exp=0) and normals separately with select
    - Use bitcast for final conversion

    Args:
        idx: 4-bit FP4 index (0-15)
    Returns:
        Decoded float32 values
    """
    # Extract FP4 fields
    sign_bit = (idx >> 3) & 1          # Bit 3
    exp_field = (idx >> 1) & 0x3       # Bits 1-2 (0-3)
    mant_bit = idx & 1                 # Bit 0

    # Convert to int32 for bit operations
    sign_bit = sign_bit.to(tl.int32)
    exp_field = exp_field.to(tl.int32)
    mant_bit = mant_bit.to(tl.int32)

    # IEEE 754 float32 construction:
    # For E2M1 normal values (exp > 0):
    #   Value = (1 + M/2) * 2^(E-1)
    #   IEEE exp = 127 + (E-1) = 126 + E
    #   IEEE mantissa = M << 22 (1 mantissa bit in position 22)
    #
    # For E2M1 subnormal (exp = 0):
    #   Value = M * 0.5 = M * 2^(-1)
    #   If M=0: 0.0 (IEEE: all zeros except sign)
    #   If M=1: 0.5 (IEEE exp=126, mantissa=0)

    # Normal case: exp > 0
    ieee_exp_normal = (126 + exp_field) << 23  # Exponent field shifted
    ieee_mant_normal = mant_bit << 22          # Mantissa in bit 22
    ieee_normal = (sign_bit << 31) | ieee_exp_normal | ieee_mant_normal

    # Subnormal case: exp = 0
    # M=0 → 0.0: all zeros (or negative zero)
    # M=1 → 0.5: exp=126, mant=0
    ieee_half = (sign_bit << 31) | (126 << 23)  # 0.5 or -0.5
    ieee_zero = (sign_bit << 31)                 # 0.0 or -0.0
    ieee_subnormal = tl.where(mant_bit == 1, ieee_half, ieee_zero)

    # Select normal vs subnormal (1 branch)
    ieee_bits = tl.where(exp_field > 0, ieee_normal, ieee_subnormal)

    # Bitcast to float32
    return ieee_bits.to(tl.float32, bitcast=True)


# =============================================================================
# Legacy alias for backward compatibility
# =============================================================================

@triton.jit
def _fp4_e2m1_decode(idx):
    """Legacy alias for v2_e2m1 decode (default implementation)."""
    return _fp4_decode_v2_e2m1(idx)


@triton.jit
def _ldexp(mantissa, exponent):
    """Compute mantissa * 2^exponent using IEEE bit manipulation.

    WARNING: NCU profiling shows this function saturates integer ALU (86.7%)
    creating an artificial compute bottleneck for what should be a memory-bound
    dequantization kernel. Use _fast_scale for better performance.
    """
    exp_clamped = tl.minimum(tl.maximum(exponent, -126), 127)
    exp_bits = (exp_clamped + 127).to(tl.int32) << 23
    power_of_2 = exp_bits.to(tl.float32, bitcast=True)
    return mantissa * power_of_2


@triton.jit
def _fast_scale(mantissa, exponent):
    """Compute mantissa * 2^exponent using tl.exp2 (hardware-accelerated).

    This function uses the Special Function Unit (SFU) instead of the integer
    ALU for computing 2^exponent. SFU runs in parallel with ALU operations,
    which avoids the ALU saturation bottleneck seen with _ldexp.

    NCU profiling showed _ldexp saturates ALU at 86.7% while memory is only
    13.9% utilized. Using tl.exp2 should shift the kernel to memory-bound
    behavior with 50%+ HBM utilization.

    Args:
        mantissa: Float32 values to scale
        exponent: Integer exponent (scale factor from MXFP4 scales)

    Returns:
        mantissa * 2^exponent as float32
    """
    # tl.exp2 uses the SFU (Special Function Unit), not integer ALU
    # This frees the ALU pipeline and should eliminate the compute bottleneck
    scale = tl.exp2(exponent.to(tl.float32))
    return mantissa * scale


@triton.jit
def _scale_by_pow2(mantissa, exponent):
    """Compute mantissa * 2^exponent by constructing IEEE float directly.

    Alternative to _fast_scale that avoids tl.exp2 entirely. Instead of calling
    exp2, we construct 2^exponent directly by setting the IEEE 754 exponent field.

    IEEE 754 float32 format:
      - Sign: bit 31
      - Exponent: bits 23-30 (biased by 127)
      - Mantissa: bits 0-22

    For 2^exponent (a power of 2 with no fractional part):
      - Sign = 0
      - Exponent field = 127 + exponent
      - Mantissa = 0

    This uses only 2 integer ops (add, shift) + 1 multiply, compared to:
      - _ldexp: 9 integer ops (saturates ALU at 86.7%)
      - _fast_scale: tl.exp2 (may still have overhead)

    Args:
        mantissa: Float32 values to scale
        exponent: Integer exponent (scale factor from MXFP4 scales, typically -7 to 7)

    Returns:
        mantissa * 2^exponent as float32
    """
    # Construct 2^exponent directly as IEEE float32
    # Exponent field = 127 + exponent (IEEE bias)
    # No clamping needed: MXFP4 scales are uint8 biased by 127, so exponent is -127 to 128
    ieee_bits = ((127 + exponent).to(tl.int32) << 23).to(tl.uint32)
    scale = ieee_bits.to(tl.float32, bitcast=True)
    return mantissa * scale


# =============================================================================
# Batch MXFP4 Dequantization Kernel
# =============================================================================

@triton.jit
def batch_mxfp4_dequant_kernel(
    # Input pointers (arrays of pointers to expert weights)
    packed_ptrs,        # [num_experts] int64 pointers to packed FP4 [N, K//2]
    scale_ptrs,         # [num_experts] int64 pointers to scales [N, K//32]
    # Output buffer [num_experts, N, K] BF16
    output_ptr,
    # Dimensions
    N, K,               # Weight dimensions
    K_packed,           # K // 2
    K_scale,            # K // 32
    # Strides for packed weights [N, K//2]
    stride_packed_n, stride_packed_k,
    # Strides for scales [N, K//32]
    stride_scale_n, stride_scale_k,
    # Strides for output [num_experts, N, K]
    stride_out_e, stride_out_n, stride_out_k,
    # Stride for pointer arrays (should be 1 for contiguous)
    stride_ptrs,
    # Block sizes
    BLOCK_N: tl.constexpr,  # 64
    BLOCK_K: tl.constexpr,  # 32 (matches scale block)
):
    """Batch dequantize all experts' weights in parallel.

    Grid: (num_experts, cdiv(N, BLOCK_N), cdiv(K, BLOCK_K))
    - axis 0: expert index
    - axis 1: N-block index
    - axis 2: K-block index

    Each thread block dequantizes a [BLOCK_N, BLOCK_K] tile of one expert's weights.
    """
    expert_idx = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)
    k_block = tl.program_id(axis=2)

    # Cast expert_idx to int64 to prevent overflow when computing large strides
    # For [128, 13824, 5120] output: stride_out_e = 70,778,880 which exceeds int32 max
    expert_idx_64 = expert_idx.to(tl.int64)

    # Get base pointers for this expert (use stride_ptrs like working kernel)
    packed_base = tl.load(packed_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    scale_base = tl.load(scale_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))

    # Compute offsets for this tile
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    n_mask = offs_n < N
    k_mask = offs_k < K

    # Load scale for this K-block (one scale per 32 K values)
    # Scale shape: [N, K//32], we need scale at k_block for each N
    scale_ptrs_tile = scale_base + offs_n * stride_scale_n + k_block * stride_scale_k
    scales = tl.load(scale_ptrs_tile, mask=n_mask, other=127).to(tl.int32) - 127

    # Packed FP4 offsets: K//2 values, each byte has 2 FP4 values
    # For K=32 block, we have 16 packed bytes
    offs_k_packed = tl.arange(0, BLOCK_K // 2)
    k_packed_start = k_block * (BLOCK_K // 2)

    # Load packed FP4 weights [BLOCK_N, BLOCK_K//2]
    packed_ptrs_tile = packed_base + offs_n[:, None] * stride_packed_n + \
                       (k_packed_start + offs_k_packed[None, :]) * stride_packed_k
    packed_mask = n_mask[:, None] & (offs_k_packed[None, :] < K_packed - k_packed_start)
    packed = tl.load(packed_ptrs_tile, mask=packed_mask, other=0)

    # Unpack FP4: lo = even K indices, hi = odd K indices
    idx_lo = (packed & 0x0F).to(tl.int32)
    idx_hi = ((packed >> 4) & 0x0F).to(tl.int32)

    # Decode FP4 values using E2M1 arithmetic (pure compute, no memory loads)
    # This is the key optimization: arithmetic decode is much faster than
    # 16 sequential tl.where() or LUT memory loads
    val_lo = _fp4_e2m1_decode(idx_lo)  # [BLOCK_N, BLOCK_K//2]
    val_hi = _fp4_e2m1_decode(idx_hi)  # [BLOCK_N, BLOCK_K//2]

    # Apply scales: val * 2^scale
    # Broadcast scale [BLOCK_N] -> [BLOCK_N, BLOCK_K//2]
    exp_broadcast = scales[:, None] + tl.zeros((1, BLOCK_K // 2), dtype=tl.int32)
    val_lo_scaled = _ldexp(val_lo, exp_broadcast).to(tl.bfloat16)
    val_hi_scaled = _ldexp(val_hi, exp_broadcast).to(tl.bfloat16)

    # Interleave lo/hi to create contiguous [BLOCK_N, BLOCK_K] output
    # Using tl.join to combine [BLOCK_N, BLOCK_K//2] + [BLOCK_N, BLOCK_K//2] -> [BLOCK_N, BLOCK_K//2, 2]
    # Then reshape to [BLOCK_N, BLOCK_K] for contiguous memory access
    val_joined = tl.join(val_lo_scaled, val_hi_scaled)  # [BLOCK_N, BLOCK_K//2, 2]
    val_interleaved = tl.reshape(val_joined, (BLOCK_N, BLOCK_K))  # [BLOCK_N, BLOCK_K]

    # Single contiguous store (much faster than two strided stores)
    k_start = k_block * BLOCK_K
    offs_k_full = tl.arange(0, BLOCK_K)
    out_ptrs = output_ptr + expert_idx_64 * stride_out_e + \
               offs_n[:, None] * stride_out_n + \
               (k_start + offs_k_full[None, :]) * stride_out_k
    out_mask = n_mask[:, None] & ((k_start + offs_k_full[None, :]) < K)
    tl.store(out_ptrs, val_interleaved, mask=out_mask)


# =============================================================================
# Versioned Dequantization Kernels for Benchmarking
# =============================================================================
# Each kernel uses a different FP4 decode function for performance comparison.

@triton.jit
def batch_mxfp4_dequant_kernel_v1_sequential(
    packed_ptrs, scale_ptrs, output_ptr,
    N, K, K_packed, K_scale,
    stride_packed_n, stride_packed_k,
    stride_scale_n, stride_scale_k,
    stride_out_e, stride_out_n, stride_out_k,
    stride_ptrs,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """V1: 16 sequential tl.where() - baseline (slowest)."""
    expert_idx = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)
    k_block = tl.program_id(axis=2)
    expert_idx_64 = expert_idx.to(tl.int64)
    packed_base = tl.load(packed_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    scale_base = tl.load(scale_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    scale_ptrs_tile = scale_base + offs_n * stride_scale_n + k_block * stride_scale_k
    scales = tl.load(scale_ptrs_tile, mask=n_mask, other=127).to(tl.int32) - 127
    offs_k_packed = tl.arange(0, BLOCK_K // 2)
    k_packed_start = k_block * (BLOCK_K // 2)
    packed_ptrs_tile = packed_base + offs_n[:, None] * stride_packed_n + \
                       (k_packed_start + offs_k_packed[None, :]) * stride_packed_k
    packed_mask = n_mask[:, None] & (offs_k_packed[None, :] < K_packed - k_packed_start)
    packed = tl.load(packed_ptrs_tile, mask=packed_mask, other=0)
    idx_lo = (packed & 0x0F).to(tl.int32)
    idx_hi = ((packed >> 4) & 0x0F).to(tl.int32)
    # V1: Use sequential 16 tl.where() decode
    val_lo = _fp4_decode_v1_sequential(idx_lo)
    val_hi = _fp4_decode_v1_sequential(idx_hi)
    exp_broadcast = scales[:, None] + tl.zeros((1, BLOCK_K // 2), dtype=tl.int32)
    val_lo_scaled = _ldexp(val_lo, exp_broadcast).to(tl.bfloat16)
    val_hi_scaled = _ldexp(val_hi, exp_broadcast).to(tl.bfloat16)
    val_joined = tl.join(val_lo_scaled, val_hi_scaled)
    val_interleaved = tl.reshape(val_joined, (BLOCK_N, BLOCK_K))
    k_start = k_block * BLOCK_K
    offs_k_full = tl.arange(0, BLOCK_K)
    out_ptrs = output_ptr + expert_idx_64 * stride_out_e + \
               offs_n[:, None] * stride_out_n + \
               (k_start + offs_k_full[None, :]) * stride_out_k
    out_mask = n_mask[:, None] & ((k_start + offs_k_full[None, :]) < K)
    tl.store(out_ptrs, val_interleaved, mask=out_mask)


@triton.jit
def batch_mxfp4_dequant_kernel_v2_e2m1(
    packed_ptrs, scale_ptrs, output_ptr,
    N, K, K_packed, K_scale,
    stride_packed_n, stride_packed_k,
    stride_scale_n, stride_scale_k,
    stride_out_e, stride_out_n, stride_out_k,
    stride_ptrs,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """V2: E2M1 arithmetic decode (5-6 tl.where)."""
    expert_idx = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)
    k_block = tl.program_id(axis=2)
    expert_idx_64 = expert_idx.to(tl.int64)
    packed_base = tl.load(packed_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    scale_base = tl.load(scale_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    scale_ptrs_tile = scale_base + offs_n * stride_scale_n + k_block * stride_scale_k
    scales = tl.load(scale_ptrs_tile, mask=n_mask, other=127).to(tl.int32) - 127
    offs_k_packed = tl.arange(0, BLOCK_K // 2)
    k_packed_start = k_block * (BLOCK_K // 2)
    packed_ptrs_tile = packed_base + offs_n[:, None] * stride_packed_n + \
                       (k_packed_start + offs_k_packed[None, :]) * stride_packed_k
    packed_mask = n_mask[:, None] & (offs_k_packed[None, :] < K_packed - k_packed_start)
    packed = tl.load(packed_ptrs_tile, mask=packed_mask, other=0)
    idx_lo = (packed & 0x0F).to(tl.int32)
    idx_hi = ((packed >> 4) & 0x0F).to(tl.int32)
    # V2: Use E2M1 arithmetic decode
    val_lo = _fp4_decode_v2_e2m1(idx_lo)
    val_hi = _fp4_decode_v2_e2m1(idx_hi)
    exp_broadcast = scales[:, None] + tl.zeros((1, BLOCK_K // 2), dtype=tl.int32)
    val_lo_scaled = _ldexp(val_lo, exp_broadcast).to(tl.bfloat16)
    val_hi_scaled = _ldexp(val_hi, exp_broadcast).to(tl.bfloat16)
    val_joined = tl.join(val_lo_scaled, val_hi_scaled)
    val_interleaved = tl.reshape(val_joined, (BLOCK_N, BLOCK_K))
    k_start = k_block * BLOCK_K
    offs_k_full = tl.arange(0, BLOCK_K)
    out_ptrs = output_ptr + expert_idx_64 * stride_out_e + \
               offs_n[:, None] * stride_out_n + \
               (k_start + offs_k_full[None, :]) * stride_out_k
    out_mask = n_mask[:, None] & ((k_start + offs_k_full[None, :]) < K)
    tl.store(out_ptrs, val_interleaved, mask=out_mask)


@triton.jit
def batch_mxfp4_dequant_kernel_v3_binary_tree(
    packed_ptrs, scale_ptrs, output_ptr,
    N, K, K_packed, K_scale,
    stride_packed_n, stride_packed_k,
    stride_scale_n, stride_scale_k,
    stride_out_e, stride_out_n, stride_out_k,
    stride_ptrs,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """V3: Binary tree lookup (4 tl.where - optimized branching)."""
    expert_idx = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)
    k_block = tl.program_id(axis=2)
    expert_idx_64 = expert_idx.to(tl.int64)
    packed_base = tl.load(packed_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    scale_base = tl.load(scale_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    scale_ptrs_tile = scale_base + offs_n * stride_scale_n + k_block * stride_scale_k
    scales = tl.load(scale_ptrs_tile, mask=n_mask, other=127).to(tl.int32) - 127
    offs_k_packed = tl.arange(0, BLOCK_K // 2)
    k_packed_start = k_block * (BLOCK_K // 2)
    packed_ptrs_tile = packed_base + offs_n[:, None] * stride_packed_n + \
                       (k_packed_start + offs_k_packed[None, :]) * stride_packed_k
    packed_mask = n_mask[:, None] & (offs_k_packed[None, :] < K_packed - k_packed_start)
    packed = tl.load(packed_ptrs_tile, mask=packed_mask, other=0)
    idx_lo = (packed & 0x0F).to(tl.int32)
    idx_hi = ((packed >> 4) & 0x0F).to(tl.int32)
    # V3: Use binary tree lookup
    val_lo = _fp4_decode_v3_binary_tree(idx_lo)
    val_hi = _fp4_decode_v3_binary_tree(idx_hi)
    exp_broadcast = scales[:, None] + tl.zeros((1, BLOCK_K // 2), dtype=tl.int32)
    val_lo_scaled = _ldexp(val_lo, exp_broadcast).to(tl.bfloat16)
    val_hi_scaled = _ldexp(val_hi, exp_broadcast).to(tl.bfloat16)
    val_joined = tl.join(val_lo_scaled, val_hi_scaled)
    val_interleaved = tl.reshape(val_joined, (BLOCK_N, BLOCK_K))
    k_start = k_block * BLOCK_K
    offs_k_full = tl.arange(0, BLOCK_K)
    out_ptrs = output_ptr + expert_idx_64 * stride_out_e + \
               offs_n[:, None] * stride_out_n + \
               (k_start + offs_k_full[None, :]) * stride_out_k
    out_mask = n_mask[:, None] & ((k_start + offs_k_full[None, :]) < K)
    tl.store(out_ptrs, val_interleaved, mask=out_mask)


@triton.jit
def batch_mxfp4_dequant_kernel_v4_branchless(
    packed_ptrs, scale_ptrs, output_ptr,
    N, K, K_packed, K_scale,
    stride_packed_n, stride_packed_k,
    stride_scale_n, stride_scale_k,
    stride_out_e, stride_out_n, stride_out_k,
    stride_ptrs,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """V4: Branchless bit manipulation (construct IEEE float directly)."""
    expert_idx = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)
    k_block = tl.program_id(axis=2)
    expert_idx_64 = expert_idx.to(tl.int64)
    packed_base = tl.load(packed_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    scale_base = tl.load(scale_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    scale_ptrs_tile = scale_base + offs_n * stride_scale_n + k_block * stride_scale_k
    scales = tl.load(scale_ptrs_tile, mask=n_mask, other=127).to(tl.int32) - 127
    offs_k_packed = tl.arange(0, BLOCK_K // 2)
    k_packed_start = k_block * (BLOCK_K // 2)
    packed_ptrs_tile = packed_base + offs_n[:, None] * stride_packed_n + \
                       (k_packed_start + offs_k_packed[None, :]) * stride_packed_k
    packed_mask = n_mask[:, None] & (offs_k_packed[None, :] < K_packed - k_packed_start)
    packed = tl.load(packed_ptrs_tile, mask=packed_mask, other=0)
    idx_lo = (packed & 0x0F).to(tl.int32)
    idx_hi = ((packed >> 4) & 0x0F).to(tl.int32)
    # V4: Use branchless bit manipulation
    val_lo = _fp4_decode_v4_branchless(idx_lo)
    val_hi = _fp4_decode_v4_branchless(idx_hi)
    exp_broadcast = scales[:, None] + tl.zeros((1, BLOCK_K // 2), dtype=tl.int32)
    val_lo_scaled = _ldexp(val_lo, exp_broadcast).to(tl.bfloat16)
    val_hi_scaled = _ldexp(val_hi, exp_broadcast).to(tl.bfloat16)
    val_joined = tl.join(val_lo_scaled, val_hi_scaled)
    val_interleaved = tl.reshape(val_joined, (BLOCK_N, BLOCK_K))
    k_start = k_block * BLOCK_K
    offs_k_full = tl.arange(0, BLOCK_K)
    out_ptrs = output_ptr + expert_idx_64 * stride_out_e + \
               offs_n[:, None] * stride_out_n + \
               (k_start + offs_k_full[None, :]) * stride_out_k
    out_mask = n_mask[:, None] & ((k_start + offs_k_full[None, :]) < K)
    tl.store(out_ptrs, val_interleaved, mask=out_mask)


# =============================================================================
# V5: Memory-Optimized with BLOCK_K=64 (2x scale blocks per tile)
# =============================================================================
# Key optimizations:
# 1. BLOCK_K=64 reduces grid from 160 to 80 K-blocks (2x less overhead)
# 2. Load two scales per tile, apply to respective 32-value halves
# 3. Process packed weights in two phases (32 values each)
# 4. Uses v4_branchless decode (fastest)
#
# Expected improvement: ~1.3x from reduced grid overhead

@triton.jit
def batch_mxfp4_dequant_kernel_v5_memopt(
    packed_ptrs, scale_ptrs, output_ptr,
    N, K, K_packed, K_scale,
    stride_packed_n, stride_packed_k,
    stride_scale_n, stride_scale_k,
    stride_out_e, stride_out_n, stride_out_k,
    stride_ptrs,
    BLOCK_N: tl.constexpr,   # 128
    BLOCK_K: tl.constexpr,   # 64 (processes 2 scale blocks)
):
    """V5: Memory-optimized with BLOCK_K=64.

    This kernel processes 64 K values per tile instead of 32, which:
    - Reduces grid size by 2x (from ~2M to ~1M blocks)
    - Better amortizes kernel launch overhead
    - Improves cache locality for packed weights

    Each tile loads TWO scales (for K positions 0-31 and 32-63).
    """
    expert_idx = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)
    k_block = tl.program_id(axis=2)
    expert_idx_64 = expert_idx.to(tl.int64)

    packed_base = tl.load(packed_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    scale_base = tl.load(scale_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))

    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # With BLOCK_K=64, each tile spans 2 scale blocks
    # Scale block 0: covers K positions [0, 32)
    # Scale block 1: covers K positions [32, 64)
    scale_k_lo = k_block * 2       # Scale for first 32 K values
    scale_k_hi = k_block * 2 + 1   # Scale for second 32 K values

    # Load scales for both halves
    scale_ptrs_lo = scale_base + offs_n * stride_scale_n + scale_k_lo * stride_scale_k
    scale_ptrs_hi = scale_base + offs_n * stride_scale_n + scale_k_hi * stride_scale_k

    # Only load second scale if within bounds
    scales_lo = tl.load(scale_ptrs_lo, mask=n_mask, other=127).to(tl.int32) - 127
    hi_scale_mask = n_mask & (scale_k_hi < K_scale)
    scales_hi = tl.load(scale_ptrs_hi, mask=hi_scale_mask, other=127).to(tl.int32) - 127

    # === First half: K positions [k_block*64, k_block*64 + 32) ===
    # Packed offset: k_block * 32 (since 64 K values = 32 packed bytes)
    k_packed_start_lo = k_block * 32  # BLOCK_K=64, first half is 32 packed bytes
    offs_k_packed = tl.arange(0, 16)  # 16 bytes = 32 FP4 values

    packed_ptrs_lo = packed_base + offs_n[:, None] * stride_packed_n + \
                     (k_packed_start_lo + offs_k_packed[None, :]) * stride_packed_k
    packed_mask_lo = n_mask[:, None] & ((k_packed_start_lo + offs_k_packed[None, :]) < K_packed)
    packed_lo = tl.load(packed_ptrs_lo, mask=packed_mask_lo, other=0)

    # Unpack first half
    idx_lo_0 = (packed_lo & 0x0F).to(tl.int32)
    idx_hi_0 = ((packed_lo >> 4) & 0x0F).to(tl.int32)

    val_lo_0 = _fp4_decode_v4_branchless(idx_lo_0)
    val_hi_0 = _fp4_decode_v4_branchless(idx_hi_0)

    # Apply first scale (covers K positions 0-31)
    exp_lo = scales_lo[:, None] + tl.zeros((1, 16), dtype=tl.int32)
    val_lo_0_scaled = _ldexp(val_lo_0, exp_lo).to(tl.bfloat16)
    val_hi_0_scaled = _ldexp(val_hi_0, exp_lo).to(tl.bfloat16)

    # Interleave first half: [BLOCK_N, 32]
    val_joined_0 = tl.join(val_lo_0_scaled, val_hi_0_scaled)  # [BLOCK_N, 16, 2]
    val_first_half = tl.reshape(val_joined_0, (BLOCK_N, 32))

    # Store first half
    k_start = k_block * BLOCK_K
    offs_k_first = tl.arange(0, 32)
    out_ptrs_first = output_ptr + expert_idx_64 * stride_out_e + \
                     offs_n[:, None] * stride_out_n + \
                     (k_start + offs_k_first[None, :]) * stride_out_k
    out_mask_first = n_mask[:, None] & ((k_start + offs_k_first[None, :]) < K)
    tl.store(out_ptrs_first, val_first_half, mask=out_mask_first)

    # === Second half: K positions [k_block*64 + 32, k_block*64 + 64) ===
    k_packed_start_hi = k_packed_start_lo + 16  # Next 16 bytes

    packed_ptrs_hi = packed_base + offs_n[:, None] * stride_packed_n + \
                     (k_packed_start_hi + offs_k_packed[None, :]) * stride_packed_k
    packed_mask_hi = n_mask[:, None] & ((k_packed_start_hi + offs_k_packed[None, :]) < K_packed)
    packed_hi = tl.load(packed_ptrs_hi, mask=packed_mask_hi, other=0)

    # Unpack second half
    idx_lo_1 = (packed_hi & 0x0F).to(tl.int32)
    idx_hi_1 = ((packed_hi >> 4) & 0x0F).to(tl.int32)

    val_lo_1 = _fp4_decode_v4_branchless(idx_lo_1)
    val_hi_1 = _fp4_decode_v4_branchless(idx_hi_1)

    # Apply second scale (covers K positions 32-63)
    exp_hi = scales_hi[:, None] + tl.zeros((1, 16), dtype=tl.int32)
    val_lo_1_scaled = _ldexp(val_lo_1, exp_hi).to(tl.bfloat16)
    val_hi_1_scaled = _ldexp(val_hi_1, exp_hi).to(tl.bfloat16)

    # Interleave second half: [BLOCK_N, 32]
    val_joined_1 = tl.join(val_lo_1_scaled, val_hi_1_scaled)  # [BLOCK_N, 16, 2]
    val_second_half = tl.reshape(val_joined_1, (BLOCK_N, 32))

    # Store second half
    k_start_hi = k_start + 32
    offs_k_second = tl.arange(0, 32)
    out_ptrs_second = output_ptr + expert_idx_64 * stride_out_e + \
                      offs_n[:, None] * stride_out_n + \
                      (k_start_hi + offs_k_second[None, :]) * stride_out_k
    out_mask_second = n_mask[:, None] & ((k_start_hi + offs_k_second[None, :]) < K)
    tl.store(out_ptrs_second, val_second_half, mask=out_mask_second)


# =============================================================================
# V6: Scale Transpose - Coalesced Scale Loading
# =============================================================================
#
# Key optimization: Transpose scales from [N, K//32] to [K//32, N] at model load
# This enables coalesced memory access for scale loads:
#
# Before (N-major, v5):
#   scale_ptr = scale_base + offs_n * stride_scale_n + k_block * stride_scale_k
#   With stride_scale_n = K//32 = 160, this causes 128 scattered 4-byte loads
#   spaced 160 bytes apart → NO coalescing → 128 memory transactions
#
# After (K-major, v6):
#   scale_ptr = scale_base + k_block * stride_scale_k + offs_n * stride_scale_n
#   With stride_scale_n = 1, all 128 threads access contiguous addresses
#   → COALESCED → 4 memory transactions (128 bytes / 32-byte cache line)
#
# Expected improvement: ~1.5-1.8x faster scale loading → overall ~1.3x faster kernel

@triton.jit
def batch_mxfp4_dequant_kernel_v6_scale_transpose(
    packed_ptrs, scale_ptrs, output_ptr,
    N, K, K_packed, K_scale,
    stride_packed_n, stride_packed_k,
    stride_scale_k, stride_scale_n,  # NOTE: K-major order (stride_k is large, stride_n=1)
    stride_out_e, stride_out_n, stride_out_k,
    stride_ptrs,
    BLOCK_N: tl.constexpr,   # 128
    BLOCK_K: tl.constexpr,   # 64 (processes 2 scale blocks)
):
    """V6: Scale transpose for coalesced loading.

    Scale tensor is [K//32, N] (K-major) instead of [N, K//32] (N-major).
    This enables coalesced 128-byte loads for scale values.

    Each tile loads TWO scales (for K positions 0-31 and 32-63).

    Requirements:
    - Scale tensor must be transposed to [K//32, N] layout before calling
    - stride_scale_k = N (large stride across K dimension)
    - stride_scale_n = 1 (contiguous across N dimension)
    """
    expert_idx = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)
    k_block = tl.program_id(axis=2)
    expert_idx_64 = expert_idx.to(tl.int64)

    packed_base = tl.load(packed_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    scale_base = tl.load(scale_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))

    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # With BLOCK_K=64, each tile spans 2 scale blocks
    # Scale block 0: covers K positions [0, 32)
    # Scale block 1: covers K positions [32, 64)
    scale_k_lo = k_block * 2       # Scale for first 32 K values
    scale_k_hi = k_block * 2 + 1   # Scale for second 32 K values

    # Load scales for both halves - K-MAJOR LAYOUT for COALESCED access
    # With stride_scale_n = 1, offs_n gives contiguous addresses!
    scale_ptrs_lo = scale_base + scale_k_lo * stride_scale_k + offs_n * stride_scale_n
    scale_ptrs_hi = scale_base + scale_k_hi * stride_scale_k + offs_n * stride_scale_n

    # Only load second scale if within bounds
    scales_lo = tl.load(scale_ptrs_lo, mask=n_mask, other=127).to(tl.int32) - 127
    hi_scale_mask = n_mask & (scale_k_hi < K_scale)
    scales_hi = tl.load(scale_ptrs_hi, mask=hi_scale_mask, other=127).to(tl.int32) - 127

    # === First half: K positions [k_block*64, k_block*64 + 32) ===
    # Packed offset: k_block * 32 (since 64 K values = 32 packed bytes)
    k_packed_start_lo = k_block * 32  # BLOCK_K=64, first half is 32 packed bytes
    offs_k_packed = tl.arange(0, 16)  # 16 bytes = 32 FP4 values

    packed_ptrs_lo = packed_base + offs_n[:, None] * stride_packed_n + \
                     (k_packed_start_lo + offs_k_packed[None, :]) * stride_packed_k
    packed_mask_lo = n_mask[:, None] & ((k_packed_start_lo + offs_k_packed[None, :]) < K_packed)
    packed_lo = tl.load(packed_ptrs_lo, mask=packed_mask_lo, other=0)

    # Unpack first half
    idx_lo_0 = (packed_lo & 0x0F).to(tl.int32)
    idx_hi_0 = ((packed_lo >> 4) & 0x0F).to(tl.int32)

    val_lo_0 = _fp4_decode_v4_branchless(idx_lo_0)
    val_hi_0 = _fp4_decode_v4_branchless(idx_hi_0)

    # Apply first scale (covers K positions 0-31)
    exp_lo = scales_lo[:, None] + tl.zeros((1, 16), dtype=tl.int32)
    val_lo_0_scaled = _ldexp(val_lo_0, exp_lo).to(tl.bfloat16)
    val_hi_0_scaled = _ldexp(val_hi_0, exp_lo).to(tl.bfloat16)

    # Interleave first half: [BLOCK_N, 32]
    val_joined_0 = tl.join(val_lo_0_scaled, val_hi_0_scaled)  # [BLOCK_N, 16, 2]
    val_first_half = tl.reshape(val_joined_0, (BLOCK_N, 32))

    # Store first half
    k_start = k_block * BLOCK_K
    offs_k_first = tl.arange(0, 32)
    out_ptrs_first = output_ptr + expert_idx_64 * stride_out_e + \
                     offs_n[:, None] * stride_out_n + \
                     (k_start + offs_k_first[None, :]) * stride_out_k
    out_mask_first = n_mask[:, None] & ((k_start + offs_k_first[None, :]) < K)
    tl.store(out_ptrs_first, val_first_half, mask=out_mask_first)

    # === Second half: K positions [k_block*64 + 32, k_block*64 + 64) ===
    k_packed_start_hi = k_packed_start_lo + 16  # Next 16 bytes

    packed_ptrs_hi = packed_base + offs_n[:, None] * stride_packed_n + \
                     (k_packed_start_hi + offs_k_packed[None, :]) * stride_packed_k
    packed_mask_hi = n_mask[:, None] & ((k_packed_start_hi + offs_k_packed[None, :]) < K_packed)
    packed_hi = tl.load(packed_ptrs_hi, mask=packed_mask_hi, other=0)

    # Unpack second half
    idx_lo_1 = (packed_hi & 0x0F).to(tl.int32)
    idx_hi_1 = ((packed_hi >> 4) & 0x0F).to(tl.int32)

    val_lo_1 = _fp4_decode_v4_branchless(idx_lo_1)
    val_hi_1 = _fp4_decode_v4_branchless(idx_hi_1)

    # Apply second scale (covers K positions 32-63)
    exp_hi = scales_hi[:, None] + tl.zeros((1, 16), dtype=tl.int32)
    val_lo_1_scaled = _ldexp(val_lo_1, exp_hi).to(tl.bfloat16)
    val_hi_1_scaled = _ldexp(val_hi_1, exp_hi).to(tl.bfloat16)

    # Interleave second half: [BLOCK_N, 32]
    val_joined_1 = tl.join(val_lo_1_scaled, val_hi_1_scaled)  # [BLOCK_N, 16, 2]
    val_second_half = tl.reshape(val_joined_1, (BLOCK_N, 32))

    # Store second half
    k_start_hi = k_start + 32
    offs_k_second = tl.arange(0, 32)
    out_ptrs_second = output_ptr + expert_idx_64 * stride_out_e + \
                      offs_n[:, None] * stride_out_n + \
                      (k_start_hi + offs_k_second[None, :]) * stride_out_k
    out_mask_second = n_mask[:, None] & ((k_start_hi + offs_k_second[None, :]) < K)
    tl.store(out_ptrs_second, val_second_half, mask=out_mask_second)


# =============================================================================
# V7: Fast Scale - Uses tl.exp2 instead of _ldexp
# =============================================================================
#
# Key optimization: Replace _ldexp (which uses integer ALU for IEEE bit manipulation)
# with _fast_scale (which uses tl.exp2 on Special Function Unit).
#
# NCU profiling of v6 showed:
#   - ALU utilization: 86.7% (SATURATED - the bottleneck!)
#   - Memory throughput: 13.9% (UNDERUTILIZED)
#
# This is caused by _ldexp using integer ALU operations for bit manipulation.
# The dequant kernel is fundamentally memory-bound (low FLOPs/byte), but
# inefficient compute creates an artificial bottleneck.
#
# tl.exp2 uses the SFU (Special Function Unit), which:
#   1. Runs in parallel with ALU operations
#   2. Has ~16 ops/cycle throughput (vs ALU bottleneck)
#   3. Frees ALU for other operations
#
# Expected improvement: 1.7-2x faster (20 ms → 10-12 ms)
# Target: Shift from compute-bound (86% ALU) to memory-bound (50%+ HBM)

@triton.jit
def batch_mxfp4_dequant_kernel_v7_fast_scale(
    packed_ptrs, scale_ptrs, output_ptr,
    N, K, K_packed, K_scale,
    stride_packed_n, stride_packed_k,
    stride_scale_k, stride_scale_n,  # K-major order (same as v6)
    stride_out_e, stride_out_n, stride_out_k,
    stride_ptrs,
    BLOCK_N: tl.constexpr,   # 128
    BLOCK_K: tl.constexpr,   # 64 (processes 2 scale blocks)
):
    """V7: Fast scale using tl.exp2 (hardware-accelerated).

    Same as v6_scale_transpose but uses _fast_scale instead of _ldexp.
    This uses the Special Function Unit (SFU) for 2^exp computation instead
    of integer ALU bit manipulation, eliminating the ALU saturation bottleneck.

    Requirements:
    - Scale tensor must be transposed to [K//32, N] layout (same as v6)
    - stride_scale_k = N (large stride across K dimension)
    - stride_scale_n = 1 (contiguous across N dimension)
    """
    expert_idx = tl.program_id(axis=0)
    n_block = tl.program_id(axis=1)
    k_block = tl.program_id(axis=2)
    expert_idx_64 = expert_idx.to(tl.int64)

    packed_base = tl.load(packed_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    scale_base = tl.load(scale_ptrs + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))

    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # With BLOCK_K=64, each tile spans 2 scale blocks
    scale_k_lo = k_block * 2       # Scale for first 32 K values
    scale_k_hi = k_block * 2 + 1   # Scale for second 32 K values

    # Load scales - K-MAJOR LAYOUT for COALESCED access (same as v6)
    scale_ptrs_lo = scale_base + scale_k_lo * stride_scale_k + offs_n * stride_scale_n
    scale_ptrs_hi = scale_base + scale_k_hi * stride_scale_k + offs_n * stride_scale_n

    scales_lo = tl.load(scale_ptrs_lo, mask=n_mask, other=127).to(tl.int32) - 127
    hi_scale_mask = n_mask & (scale_k_hi < K_scale)
    scales_hi = tl.load(scale_ptrs_hi, mask=hi_scale_mask, other=127).to(tl.int32) - 127

    # === First half: K positions [k_block*64, k_block*64 + 32) ===
    k_packed_start_lo = k_block * 32
    offs_k_packed = tl.arange(0, 16)

    packed_ptrs_lo = packed_base + offs_n[:, None] * stride_packed_n + \
                     (k_packed_start_lo + offs_k_packed[None, :]) * stride_packed_k
    packed_mask_lo = n_mask[:, None] & ((k_packed_start_lo + offs_k_packed[None, :]) < K_packed)
    packed_lo = tl.load(packed_ptrs_lo, mask=packed_mask_lo, other=0)

    # Unpack first half
    idx_lo_0 = (packed_lo & 0x0F).to(tl.int32)
    idx_hi_0 = ((packed_lo >> 4) & 0x0F).to(tl.int32)

    val_lo_0 = _fp4_decode_v4_branchless(idx_lo_0)
    val_hi_0 = _fp4_decode_v4_branchless(idx_hi_0)

    # Apply first scale using _fast_scale (tl.exp2) instead of _ldexp
    exp_lo = scales_lo[:, None] + tl.zeros((1, 16), dtype=tl.int32)
    val_lo_0_scaled = _fast_scale(val_lo_0, exp_lo).to(tl.bfloat16)
    val_hi_0_scaled = _fast_scale(val_hi_0, exp_lo).to(tl.bfloat16)

    # Interleave first half
    val_joined_0 = tl.join(val_lo_0_scaled, val_hi_0_scaled)
    val_first_half = tl.reshape(val_joined_0, (BLOCK_N, 32))

    # Store first half
    k_start = k_block * BLOCK_K
    offs_k_first = tl.arange(0, 32)
    out_ptrs_first = output_ptr + expert_idx_64 * stride_out_e + \
                     offs_n[:, None] * stride_out_n + \
                     (k_start + offs_k_first[None, :]) * stride_out_k
    out_mask_first = n_mask[:, None] & ((k_start + offs_k_first[None, :]) < K)
    tl.store(out_ptrs_first, val_first_half, mask=out_mask_first)

    # === Second half: K positions [k_block*64 + 32, k_block*64 + 64) ===
    k_packed_start_hi = k_packed_start_lo + 16

    packed_ptrs_hi = packed_base + offs_n[:, None] * stride_packed_n + \
                     (k_packed_start_hi + offs_k_packed[None, :]) * stride_packed_k
    packed_mask_hi = n_mask[:, None] & ((k_packed_start_hi + offs_k_packed[None, :]) < K_packed)
    packed_hi = tl.load(packed_ptrs_hi, mask=packed_mask_hi, other=0)

    # Unpack second half
    idx_lo_1 = (packed_hi & 0x0F).to(tl.int32)
    idx_hi_1 = ((packed_hi >> 4) & 0x0F).to(tl.int32)

    val_lo_1 = _fp4_decode_v4_branchless(idx_lo_1)
    val_hi_1 = _fp4_decode_v4_branchless(idx_hi_1)

    # Apply second scale using _fast_scale (tl.exp2) instead of _ldexp
    exp_hi = scales_hi[:, None] + tl.zeros((1, 16), dtype=tl.int32)
    val_lo_1_scaled = _fast_scale(val_lo_1, exp_hi).to(tl.bfloat16)
    val_hi_1_scaled = _fast_scale(val_hi_1, exp_hi).to(tl.bfloat16)

    # Interleave second half
    val_joined_1 = tl.join(val_lo_1_scaled, val_hi_1_scaled)
    val_second_half = tl.reshape(val_joined_1, (BLOCK_N, 32))

    # Store second half
    k_start_hi = k_start + 32
    offs_k_second = tl.arange(0, 32)
    out_ptrs_second = output_ptr + expert_idx_64 * stride_out_e + \
                      offs_n[:, None] * stride_out_n + \
                      (k_start_hi + offs_k_second[None, :]) * stride_out_k
    out_mask_second = n_mask[:, None] & ((k_start_hi + offs_k_second[None, :]) < K)
    tl.store(out_ptrs_second, val_second_half, mask=out_mask_second)


# =============================================================================
# V8: IEEE Power-of-2 Construction (Alternative to tl.exp2)
# =============================================================================
#
# If v7_fast_scale doesn't provide expected speedup (NCU shows tl.exp2 has
# overhead), this version constructs 2^exponent directly via IEEE bit manipulation.
# Uses only 2 integer ops (add, shift) + 1 multiply, avoiding both _ldexp's
# 9-op overhead and any potential tl.exp2 overhead.

@triton.jit
def batch_mxfp4_dequant_kernel_v8_ieee_pow2(
    packed_ptrs, scale_ptrs, output_ptr,
    N, K, K_packed, K_scale,
    stride_packed_n, stride_packed_k,
    stride_scale_k, stride_scale_n,
    stride_out_e, stride_out_n, stride_out_k,
    stride_ptrs,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """V8: Direct IEEE 754 power-of-2 construction for scale application.

    Same structure as v7_fast_scale, but uses _scale_by_pow2 instead of _fast_scale.
    This constructs 2^exponent directly by setting the IEEE exponent field, using
    only 2 integer ops (add, shift) + 1 multiply.

    If NCU profiling shows v7's tl.exp2 still has significant overhead, this
    version may be faster by avoiding the exp2 instruction entirely.

    REQUIRES: Scales in K-major layout [K//32, N] (same as v6/v7)

    Grid: (num_experts, cdiv(N, BLOCK_N), cdiv(K, BLOCK_K))
          With BLOCK_K=64, K_blocks = K // 64 (half as many as v4)
    """
    # Program IDs
    expert_idx = tl.program_id(0)
    n_block = tl.program_id(1)
    k_block = tl.program_id(2)

    # Cast to int64 for large stride calculations (avoid int32 overflow)
    expert_idx_64 = expert_idx.to(tl.int64)

    # Get base pointers for this expert
    packed_base = tl.load(packed_ptrs + expert_idx * stride_ptrs).to(tl.uint64)
    scale_base = tl.load(scale_ptrs + expert_idx * stride_ptrs).to(tl.uint64)

    # Block offsets
    n_start = n_block * BLOCK_N
    k_start = k_block * BLOCK_K

    # N dimension offsets
    offs_n = n_start + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # =========================================================================
    # FIRST HALF: K positions 0-31 (scale block 0)
    # =========================================================================
    scale_k_lo = k_block * 2  # First scale block within this K-tile

    # Load scales for first half (K-major layout: [K//32, N])
    # scale_ptrs = scale_base + scale_k_lo * stride_scale_k + offs_n * stride_scale_n
    # With K-major: stride_scale_k = N, stride_scale_n = 1 -> COALESCED!
    scale_ptrs_lo = scale_base + scale_k_lo * stride_scale_k + offs_n * stride_scale_n
    scales_lo = tl.load(scale_ptrs_lo, mask=n_mask, other=127)

    # Load packed FP4 for first half: positions 0-31 -> packed bytes 0-15
    offs_k_lo = tl.arange(0, 16)  # 16 bytes = 32 FP4 values
    packed_ptrs_lo = packed_base + offs_n[:, None] * stride_packed_n + \
                     ((k_start // 2) + offs_k_lo[None, :]) * stride_packed_k
    packed_mask_lo = n_mask[:, None] & (((k_start // 2) + offs_k_lo[None, :]) < K_packed)
    packed_lo = tl.load(packed_ptrs_lo, mask=packed_mask_lo, other=0)

    # Unpack FP4 values
    idx_lo_0 = packed_lo & 0x0F
    idx_hi_0 = (packed_lo >> 4) & 0x0F

    # Decode using branchless IEEE construction
    val_lo_0 = _fp4_decode_v4_branchless(idx_lo_0)
    val_hi_0 = _fp4_decode_v4_branchless(idx_hi_0)

    # Apply first scale using _scale_by_pow2 (direct IEEE construction)
    exp_lo = (scales_lo - 127).to(tl.int32)[:, None]  # Unbias scale
    val_lo_0_scaled = _scale_by_pow2(val_lo_0, exp_lo).to(tl.bfloat16)
    val_hi_0_scaled = _scale_by_pow2(val_hi_0, exp_lo).to(tl.bfloat16)

    # Interleave first half: [lo_0, hi_0, lo_1, hi_1, ...]
    val_joined_0 = tl.join(val_lo_0_scaled, val_hi_0_scaled)
    val_first_half = tl.reshape(val_joined_0, (BLOCK_N, 32))

    # Store first half
    offs_k_first = tl.arange(0, 32)
    out_ptrs_first = output_ptr + expert_idx_64 * stride_out_e + \
                     offs_n[:, None] * stride_out_n + \
                     (k_start + offs_k_first[None, :]) * stride_out_k
    out_mask_first = n_mask[:, None] & ((k_start + offs_k_first[None, :]) < K)
    tl.store(out_ptrs_first, val_first_half, mask=out_mask_first)

    # =========================================================================
    # SECOND HALF: K positions 32-63 (scale block 1)
    # =========================================================================
    scale_k_hi = k_block * 2 + 1  # Second scale block within this K-tile

    # Load scales for second half
    scale_ptrs_hi = scale_base + scale_k_hi * stride_scale_k + offs_n * stride_scale_n
    scales_hi = tl.load(scale_ptrs_hi, mask=n_mask, other=127)

    # Load packed FP4 for second half: positions 32-63 -> packed bytes 16-31
    offs_k_hi = tl.arange(0, 16)
    packed_ptrs_hi = packed_base + offs_n[:, None] * stride_packed_n + \
                     ((k_start // 2 + 16) + offs_k_hi[None, :]) * stride_packed_k
    packed_mask_hi = n_mask[:, None] & (((k_start // 2 + 16) + offs_k_hi[None, :]) < K_packed)
    packed_hi = tl.load(packed_ptrs_hi, mask=packed_mask_hi, other=0)

    # Unpack FP4 values
    idx_lo_1 = packed_hi & 0x0F
    idx_hi_1 = (packed_hi >> 4) & 0x0F

    # Decode
    val_lo_1 = _fp4_decode_v4_branchless(idx_lo_1)
    val_hi_1 = _fp4_decode_v4_branchless(idx_hi_1)

    # Apply second scale using _scale_by_pow2 (direct IEEE construction)
    exp_hi = (scales_hi - 127).to(tl.int32)[:, None]
    val_lo_1_scaled = _scale_by_pow2(val_lo_1, exp_hi).to(tl.bfloat16)
    val_hi_1_scaled = _scale_by_pow2(val_hi_1, exp_hi).to(tl.bfloat16)

    # Interleave second half
    val_joined_1 = tl.join(val_lo_1_scaled, val_hi_1_scaled)
    val_second_half = tl.reshape(val_joined_1, (BLOCK_N, 32))

    # Store second half
    k_start_hi = k_start + 32
    offs_k_second = tl.arange(0, 32)
    out_ptrs_second = output_ptr + expert_idx_64 * stride_out_e + \
                      offs_n[:, None] * stride_out_n + \
                      (k_start_hi + offs_k_second[None, :]) * stride_out_k
    out_mask_second = n_mask[:, None] & ((k_start_hi + offs_k_second[None, :]) < K)
    tl.store(out_ptrs_second, val_second_half, mask=out_mask_second)


# Mapping from version name to kernel function (for Python wrapper)
_DEQUANT_KERNELS = {
    "v1_sequential": batch_mxfp4_dequant_kernel_v1_sequential,
    "v2_e2m1": batch_mxfp4_dequant_kernel_v2_e2m1,
    "v3_binary_tree": batch_mxfp4_dequant_kernel_v3_binary_tree,
    "v4_branchless": batch_mxfp4_dequant_kernel_v4_branchless,
    "v5_memopt": batch_mxfp4_dequant_kernel_v5_memopt,
    "v6_scale_transpose": batch_mxfp4_dequant_kernel_v6_scale_transpose,
    "v7_fast_scale": batch_mxfp4_dequant_kernel_v7_fast_scale,
    "v8_ieee_pow2": batch_mxfp4_dequant_kernel_v8_ieee_pow2,
}


def batch_mxfp4_dequant(
    packed_ptrs: torch.Tensor,    # [num_experts] int64
    scale_ptrs: torch.Tensor,     # [num_experts] int64
    output: torch.Tensor,         # [num_experts, N, K] BF16
    packed_ref: torch.Tensor,     # Reference tensor for strides [N, K//2]
    scale_ref: torch.Tensor,      # Reference tensor for strides [N, K//32] or [K//32, N] for v6
    BLOCK_N: int = 128,
    BLOCK_K: int = 32,
    version: str = "v2_e2m1",     # FP4 decode version
) -> None:
    """Batch dequantize all experts' MXFP4 weights into BF16 buffer.

    Args:
        packed_ptrs: Pointer array to packed FP4 weights [num_experts]
        scale_ptrs: Pointer array to scales [num_experts]
        output: Pre-allocated output buffer [num_experts, N, K] BF16
        packed_ref: Reference weight tensor for computing strides
        scale_ref: Reference scale tensor for computing strides
            - For v1-v5: [N, K//32] layout (N-major)
            - For v6/v7/v8: [K//32, N] layout (K-major, transposed)
        BLOCK_N: Tile size for N dimension (default 128, larger = fewer blocks)
        BLOCK_K: Tile size for K dimension (default 32, MUST match MXFP4 scale block)
                 For v5_memopt and v6_scale_transpose, this is automatically set to 64.
        version: FP4 decode version for benchmarking:
            - "v1_sequential": 16 sequential tl.where() (baseline, slowest)
            - "v2_e2m1": E2M1 arithmetic decode (5-6 tl.where)
            - "v3_binary_tree": Binary tree lookup (4 tl.where)
            - "v4_branchless": Branchless bit manipulation (IEEE float construction)
            - "v5_memopt": Memory-optimized with BLOCK_K=64 (2x fewer K-blocks)
            - "v6_scale_transpose": K-major scale layout for coalesced loading
              REQUIRES scales to be transposed to [K//32, N] layout!
            - "v7_fast_scale": Uses tl.exp2 (SFU) instead of _ldexp (ALU) for scaling.
              REQUIRES scales to be transposed to [K//32, N] layout (same as v6)!
              This eliminates ALU saturation bottleneck (86% → <40% ALU).
            - "v8_ieee_pow2": Direct IEEE 754 power-of-2 construction for scaling.
              REQUIRES scales to be transposed to [K//32, N] layout (same as v6/v7)!
              Uses only 2 integer ops (add, shift) + 1 multiply. Alternative if
              v7's tl.exp2 doesn't provide expected speedup.
    """
    num_experts = packed_ptrs.shape[0]
    N = output.shape[1]
    K = output.shape[2]
    K_packed = K // 2
    K_scale = K // 32

    # v5_memopt and later use BLOCK_K=64 (processes 2 scale blocks per tile)
    if version in ("v5_memopt", "v6_scale_transpose", "v7_fast_scale", "v8_ieee_pow2"):
        BLOCK_K = 64
        assert K % 64 == 0, f"K ({K}) must be divisible by 64 for {version}"

    # Grid: (num_experts, cdiv(N, BLOCK_N), cdiv(K, BLOCK_K))
    grid = (num_experts, triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))

    # Select kernel based on version (use default kernel for backward compatibility)
    if version == "default":
        kernel = batch_mxfp4_dequant_kernel
    else:
        kernel = _DEQUANT_KERNELS.get(version, batch_mxfp4_dequant_kernel_v2_e2m1)

    kernel[grid](
        packed_ptrs, scale_ptrs,
        output,
        N, K, K_packed, K_scale,
        packed_ref.stride(0), packed_ref.stride(1),
        scale_ref.stride(0), scale_ref.stride(1),
        output.stride(0), output.stride(1), output.stride(2),
        1,  # stride_ptrs (contiguous pointer array, same as working kernel)
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,  # More warps for larger tiles
    )


# =============================================================================
# BF16 Grouped GEMM Kernel
# =============================================================================

@triton.jit
def bf16_grouped_gemm_kernel_3d(
    # Input [E, M_max, K] BF16
    lhs_ptr,
    # Weight buffer [num_experts, N, K] BF16 (pre-dequantized)
    rhs_ptr,
    # Per-expert token counts [num_experts] int32
    expert_tokens_ptr,
    # Output [E, M_max, N] BF16
    output_ptr,
    # Dimensions
    M_max, N, K,
    # Strides for lhs [E, M_max, K]
    stride_lhs_e, stride_lhs_m, stride_lhs_k,
    # Strides for rhs [num_experts, N, K]
    stride_rhs_e, stride_rhs_n, stride_rhs_k,
    # Strides for output [E, M_max, N]
    stride_out_e, stride_out_m, stride_out_n,
    # Block sizes (constexpr for Triton)
    BLOCK_M: tl.constexpr,  # 64
    BLOCK_N: tl.constexpr,  # 64
    BLOCK_K: tl.constexpr,  # 32 (match working fused kernel)
):
    """BF16 grouped GEMM on pre-dequantized weights.

    Grid: (num_experts, cdiv(N, BLOCK_N))
    - axis 0: expert index
    - axis 1: N-block index

    Key advantage over fused MXFP4:
    - Pure BF16 operations → optimal tensor core utilization
    - No FP4 lookup or scale loads in inner loop

    This kernel loops over M-blocks internally to handle variable tokens per expert.
    Uses dynamic loop bounds (like the working fused MXFP4 kernel).
    """
    expert_idx = tl.program_id(axis=0)
    n_pid = tl.program_id(axis=1)

    # Cast expert_idx to int64 to prevent overflow when computing large strides
    # For weight_buffer [128, 13824, 5120]: stride_rhs_e = 70,778,880 which exceeds int32 max
    expert_idx_64 = expert_idx.to(tl.int64)

    # Early exit for empty experts
    gm = tl.load(expert_tokens_ptr + expert_idx).to(tl.int32)
    if gm == 0:
        return

    # Get base pointers for this expert (use expert_idx_64 for large stride multiplication)
    cur_lhs_ptr = lhs_ptr + expert_idx_64 * stride_lhs_e
    cur_rhs_ptr = rhs_ptr + expert_idx_64 * stride_rhs_e
    cur_out_ptr = output_ptr + expert_idx_64 * stride_out_e

    # N-block offset
    offs_n = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # Compute dynamic loop bounds (same pattern as working fused kernel)
    num_m_blocks = tl.cdiv(gm, BLOCK_M)
    num_k_blocks = K // BLOCK_K

    # Process each M-block (same structure as working fused_mxfp4_grouped_gemm_kernel_3d)
    for m_block in range(num_m_blocks):
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < gm

        # Initialize accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K-loop (matches working fused kernel pattern)
        for k_block in range(num_k_blocks):
            k_start = k_block * BLOCK_K
            offs_k = tl.arange(0, BLOCK_K)

            # Load LHS [BLOCK_M, BLOCK_K] and convert to float32 (like working kernel)
            lhs_ptrs = cur_lhs_ptr + offs_m[:, None] * stride_lhs_m + \
                       (k_start + offs_k[None, :]) * stride_lhs_k
            lhs = tl.load(lhs_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

            # Load RHS [BLOCK_N, BLOCK_K] and convert to float32
            rhs_ptrs = cur_rhs_ptr + offs_n[:, None] * stride_rhs_n + \
                       (k_start + offs_k[None, :]) * stride_rhs_k
            rhs = tl.load(rhs_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

            # Accumulate: lhs @ rhs.T (exact pattern from working kernel)
            acc += tl.dot(lhs.to(tl.bfloat16), tl.trans(rhs.to(tl.bfloat16)),
                          allow_tf32=False).to(tl.float32)

        # Store output [BLOCK_M, BLOCK_N]
        out_ptrs = cur_out_ptr + offs_m[:, None] * stride_out_m + \
                   offs_n[None, :] * stride_out_n
        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def bf16_grouped_gemm_3d(
    hidden_3d: torch.Tensor,          # [E, M_max, K] BF16
    weight_buffer: torch.Tensor,       # [num_experts, N, K] BF16
    expert_counts: torch.Tensor,       # [num_experts] int32
    N: int,
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,  # Use 32 like working fused kernel (smaller working set)
    num_warps: int = 8,
) -> torch.Tensor:
    """BF16 grouped GEMM on pre-dequantized weights.

    Args:
        hidden_3d: Input tensor [E, M_max, K] in BF16
        weight_buffer: Pre-dequantized weights [num_experts, N, K] in BF16
        expert_counts: Token counts per expert [num_experts]
        N: Output dimension
        BLOCK_M, BLOCK_N, BLOCK_K: Tile sizes
        num_warps: Number of warps per block

    Returns:
        output_3d: [E, M_max, N] in BF16
    """
    num_experts = hidden_3d.shape[0]
    M_max = hidden_3d.shape[1]
    K = hidden_3d.shape[2]
    device = hidden_3d.device

    # Ensure K is divisible by BLOCK_K
    assert K % BLOCK_K == 0, f"K ({K}) must be divisible by BLOCK_K ({BLOCK_K})"

    # Ensure expert_counts is int32
    if expert_counts.dtype != torch.int32:
        expert_counts = expert_counts.to(torch.int32)

    # Allocate output
    output_3d = torch.empty(num_experts, M_max, N, dtype=torch.bfloat16, device=device)

    # Grid: (num_experts, cdiv(N, BLOCK_N))
    grid = (num_experts, triton.cdiv(N, BLOCK_N))

    bf16_grouped_gemm_kernel_3d[grid](
        hidden_3d, weight_buffer, expert_counts, output_3d,
        M_max, N, K,
        hidden_3d.stride(0), hidden_3d.stride(1), hidden_3d.stride(2),
        weight_buffer.stride(0), weight_buffer.stride(1), weight_buffer.stride(2),
        output_3d.stride(0), output_3d.stride(1), output_3d.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )

    return output_3d


# =============================================================================
# Token Dispatch/Undispatch (Reused from mxfp4_grouped_gemm.py)
# =============================================================================

def moe_token_dispatch_3d(
    hidden_states: torch.Tensor,      # [batch*seq, hidden]
    topk_indices: torch.Tensor,       # [batch*seq, num_experts_per_tok]
    num_experts: int,
    M_max: int = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch tokens to 3D layout for grouped processing.

    Creates a [num_experts, M_max, hidden] tensor where each expert slice
    contains its assigned tokens (padded to M_max).

    Returns:
        hidden_3d: [num_experts, M_max, hidden] - tokens grouped by expert
        expert_counts: [num_experts] - actual token count per expert
    """
    device = hidden_states.device
    hidden_size = hidden_states.shape[-1]
    num_tokens = hidden_states.shape[0]
    num_experts_per_tok = topk_indices.shape[1]

    # Count tokens per expert
    flat_indices = topk_indices.view(-1)
    expert_counts = torch.bincount(flat_indices, minlength=num_experts)

    # Determine M_max if not specified
    if M_max is None:
        M_max = expert_counts.max().item()
        if M_max == 0:
            M_max = 1

    # Allocate 3D tensor
    hidden_3d = torch.zeros(num_experts, M_max, hidden_size,
                            dtype=hidden_states.dtype, device=device)

    # Fill in tokens for each expert
    # Create position indices within each expert's slice
    expert_positions = torch.zeros(num_experts, dtype=torch.int64, device=device)

    for k in range(num_experts_per_tok):
        expert_ids = topk_indices[:, k]  # [num_tokens]

        for e in range(num_experts):
            mask = expert_ids == e
            if mask.any():
                tokens = hidden_states[mask]
                num_to_add = tokens.shape[0]
                start_pos = expert_positions[e].item()
                end_pos = min(start_pos + num_to_add, M_max)
                actual_add = end_pos - start_pos
                hidden_3d[e, start_pos:end_pos] = tokens[:actual_add]
                expert_positions[e] += actual_add

    return hidden_3d, expert_counts.to(torch.int32)


def moe_token_undispatch_3d(
    output_3d: torch.Tensor,          # [num_experts, M_max, hidden]
    topk_indices: torch.Tensor,       # [batch*seq, num_experts_per_tok]
    topk_weights: torch.Tensor,       # [batch*seq, num_experts_per_tok]
    num_tokens: int,
) -> torch.Tensor:
    """Undispatch tokens from 3D layout back to original order with weighted sum.

    Args:
        output_3d: Expert outputs [num_experts, M_max, hidden]
        topk_indices: Which experts each token was sent to
        topk_weights: Routing weights for each expert selection
        num_tokens: Original number of tokens

    Returns:
        output: [num_tokens, hidden] - weighted sum of expert outputs
    """
    device = output_3d.device
    hidden_size = output_3d.shape[-1]
    num_experts = output_3d.shape[0]
    num_experts_per_tok = topk_indices.shape[1]

    output = torch.zeros(num_tokens, hidden_size, dtype=output_3d.dtype, device=device)

    # Track position within each expert's output
    expert_positions = torch.zeros(num_experts, dtype=torch.int64, device=device)

    for k in range(num_experts_per_tok):
        expert_ids = topk_indices[:, k]
        weights = topk_weights[:, k]

        for e in range(num_experts):
            mask = expert_ids == e
            if mask.any():
                num_tokens_for_expert = mask.sum().item()
                start_pos = expert_positions[e].item()
                end_pos = start_pos + num_tokens_for_expert

                expert_output = output_3d[e, start_pos:end_pos]
                token_weights = weights[mask].unsqueeze(-1)

                output[mask] += expert_output * token_weights
                expert_positions[e] = end_pos

    return output


# =============================================================================
# Decoupled MXFP4 MoE Module
# =============================================================================

class DecoupledMXFP4MoE(nn.Module):
    """MoE layer with decoupled MXFP4 dequantization + BF16 grouped GEMM.

    This module separates weight dequantization from GEMM computation for
    optimal performance. Dequantization runs once per forward pass in a
    highly parallel kernel, then grouped GEMM operates on BF16 weights.

    Performance:
    - Fused MXFP4 GEMM: ~73 ms (inline dequant kills tensor core efficiency)
    - Decoupled (this): ~15-25 ms (3-5x faster)
      - Batch dequant: ~5-10 ms (embarrassingly parallel)
      - BF16 grouped GEMM: ~10-15 ms (optimal tensor cores)

    Memory Requirements:
    - Per projection buffer: 18.1 GB (128 experts × 141.6 MB)
    - With buffer reuse: Peak 18.1 GB (one buffer shared across gate/up/down)

    Usage:
        moe = DecoupledMXFP4MoE(config)
        moe.load_mxfp4_weights(gate_w, gate_s, up_w, up_s, down_w, down_s, biases)
        output = moe(hidden_states)
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        buffer_mode: str = "per_projection",
        swiglu_alpha: float = 1.702,
        swiglu_limit: float = 7.0,
    ):
        """Initialize DecoupledMXFP4MoE.

        Args:
            num_experts: Number of experts (e.g., 128)
            num_experts_per_tok: Experts per token (e.g., 8)
            hidden_size: Model hidden dimension (e.g., 5120)
            intermediate_size: MLP intermediate dimension (e.g., 13824)
            buffer_mode: "per_projection" (18GB) or "all" (45GB)
            swiglu_alpha: SwiGLU alpha parameter
            swiglu_limit: Clamping limit for numerical stability
        """
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.buffer_mode = buffer_mode
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_limit = swiglu_limit

        # Router
        self.router = nn.Linear(hidden_size, num_experts, bias=True)

        # MXFP4 weights (loaded later)
        self.gate_weights: List[torch.Tensor] = None  # [num_experts] × [N, K//2]
        self.gate_scales: List[torch.Tensor] = None   # [num_experts] × [N, K//32]
        self.up_weights: List[torch.Tensor] = None
        self.up_scales: List[torch.Tensor] = None
        self.down_weights: List[torch.Tensor] = None
        self.down_scales: List[torch.Tensor] = None

        # Optional biases [num_experts, N]
        self.gate_biases: torch.Tensor = None
        self.up_biases: torch.Tensor = None
        self.down_biases: torch.Tensor = None

        # Pointer arrays (created after weight loading)
        self.gate_ptrs: torch.Tensor = None
        self.gate_scale_ptrs: torch.Tensor = None
        self.up_ptrs: torch.Tensor = None
        self.up_scale_ptrs: torch.Tensor = None
        self.down_ptrs: torch.Tensor = None
        self.down_scale_ptrs: torch.Tensor = None

        # BF16 buffers (pre-allocated for efficiency)
        self._bf16_buffer: torch.Tensor = None

    def _setup_pointer_arrays(self):
        """Create pointer arrays for batch dequantization kernel."""
        device = self.gate_weights[0].device

        self.gate_ptrs = torch.tensor(
            [w.data_ptr() for w in self.gate_weights], dtype=torch.int64, device=device)
        self.gate_scale_ptrs = torch.tensor(
            [s.data_ptr() for s in self.gate_scales], dtype=torch.int64, device=device)

        self.up_ptrs = torch.tensor(
            [w.data_ptr() for w in self.up_weights], dtype=torch.int64, device=device)
        self.up_scale_ptrs = torch.tensor(
            [s.data_ptr() for s in self.up_scales], dtype=torch.int64, device=device)

        self.down_ptrs = torch.tensor(
            [w.data_ptr() for w in self.down_weights], dtype=torch.int64, device=device)
        self.down_scale_ptrs = torch.tensor(
            [s.data_ptr() for s in self.down_scales], dtype=torch.int64, device=device)

    def _ensure_bf16_buffer(self, N: int, K: int, device: torch.device):
        """Ensure BF16 buffer is allocated with correct size."""
        required_size = (self.num_experts, N, K)
        if self._bf16_buffer is None or self._bf16_buffer.shape != required_size:
            self._bf16_buffer = torch.empty(
                required_size, dtype=torch.bfloat16, device=device)
        return self._bf16_buffer

    def load_mxfp4_weights(
        self,
        gate_weights: List[torch.Tensor],
        gate_scales: List[torch.Tensor],
        up_weights: List[torch.Tensor],
        up_scales: List[torch.Tensor],
        down_weights: List[torch.Tensor],
        down_scales: List[torch.Tensor],
        gate_biases: torch.Tensor = None,
        up_biases: torch.Tensor = None,
        down_biases: torch.Tensor = None,
    ):
        """Load MXFP4 quantized weights.

        Args:
            gate_weights: List of [intermediate_size, hidden_size//2] uint8
            gate_scales: List of [intermediate_size, hidden_size//32] uint8
            up_weights, up_scales: Same shapes as gate
            down_weights: List of [hidden_size, intermediate_size//2] uint8
            down_scales: List of [hidden_size, intermediate_size//32] uint8
            gate_biases: [num_experts, intermediate_size] BF16 (optional)
            up_biases, down_biases: Same pattern (optional)
        """
        self.gate_weights = gate_weights
        self.gate_scales = gate_scales
        self.up_weights = up_weights
        self.up_scales = up_scales
        self.down_weights = down_weights
        self.down_scales = down_scales
        self.gate_biases = gate_biases
        self.up_biases = up_biases
        self.down_biases = down_biases

        self._setup_pointer_arrays()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with decoupled dequant + BF16 grouped GEMM.

        Args:
            hidden_states: [batch, seq_len, hidden_size] or [batch*seq, hidden_size]

        Returns:
            output: Same shape as input
        """
        original_shape = hidden_states.shape
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_flat = hidden_states.view(-1, hidden_dim)
        else:
            hidden_flat = hidden_states
            batch_size, seq_len = hidden_flat.shape[0], 1

        num_tokens = hidden_flat.shape[0]
        device = hidden_flat.device

        # === Router ===
        router_logits = self.router(hidden_flat)
        topk_weights, topk_indices = torch.topk(
            router_logits, k=self.num_experts_per_tok, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)

        # === Token dispatch to 3D layout ===
        hidden_3d, expert_counts = moe_token_dispatch_3d(
            hidden_flat, topk_indices, self.num_experts)
        M_max = hidden_3d.shape[1]

        # === Gate projection ===
        # Dequantize gate weights
        gate_buffer = self._ensure_bf16_buffer(
            self.intermediate_size, self.hidden_size, device)
        batch_mxfp4_dequant(
            self.gate_ptrs, self.gate_scale_ptrs, gate_buffer,
            self.gate_weights[0], self.gate_scales[0])

        # BF16 grouped GEMM for gate
        gate_out = bf16_grouped_gemm_3d(
            hidden_3d, gate_buffer, expert_counts, self.intermediate_size)

        # Add bias if present
        if self.gate_biases is not None:
            gate_out = gate_out + self.gate_biases.unsqueeze(1)

        # === Up projection ===
        # Reuse buffer for up weights
        batch_mxfp4_dequant(
            self.up_ptrs, self.up_scale_ptrs, gate_buffer,
            self.up_weights[0], self.up_scales[0])

        up_out = bf16_grouped_gemm_3d(
            hidden_3d, gate_buffer, expert_counts, self.intermediate_size)

        if self.up_biases is not None:
            up_out = up_out + self.up_biases.unsqueeze(1)

        # === SwiGLU activation ===
        gate_clamped = gate_out.clamp(max=self.swiglu_limit)
        up_clamped = up_out.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        intermediate = gate_clamped * torch.sigmoid(
            self.swiglu_alpha * gate_clamped) * (up_clamped + 1)

        # === Down projection ===
        # Allocate new buffer for down (different dimensions)
        down_buffer = self._ensure_bf16_buffer(
            self.hidden_size, self.intermediate_size, device)
        batch_mxfp4_dequant(
            self.down_ptrs, self.down_scale_ptrs, down_buffer,
            self.down_weights[0], self.down_scales[0])

        output_3d = bf16_grouped_gemm_3d(
            intermediate, down_buffer, expert_counts, self.hidden_size)

        if self.down_biases is not None:
            output_3d = output_3d + self.down_biases.unsqueeze(1)

        # === Token undispatch ===
        output = moe_token_undispatch_3d(
            output_3d, topk_indices, topk_weights, num_tokens)

        # Reshape to original
        if len(original_shape) == 3:
            output = output.view(batch_size, seq_len, -1)

        return output


# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark_batch_dequant(
    num_experts: int,
    N: int,
    K: int,
    device: str = "cuda",
    warmup_iters: int = 3,
    bench_iters: int = 10,
    version: str = "v2_e2m1",
) -> float:
    """Benchmark batch MXFP4 dequantization kernel.

    Args:
        num_experts: Number of experts
        N: Output dimension (rows)
        K: Input dimension (columns)
        device: CUDA device
        warmup_iters: Warmup iterations
        bench_iters: Benchmark iterations
        version: FP4 decode version (v1_sequential, v2_e2m1, v3_binary_tree, v4_branchless)

    Returns:
        Time in milliseconds.
    """
    # Create test weights
    weights = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
               for _ in range(num_experts)]
    scales = [torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
              for _ in range(num_experts)]

    # Create pointer arrays
    weight_ptrs = torch.tensor([w.data_ptr() for w in weights],
                               dtype=torch.int64, device=device)
    scale_ptrs = torch.tensor([s.data_ptr() for s in scales],
                              dtype=torch.int64, device=device)

    # Allocate output
    output = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)

    # Warmup
    for _ in range(warmup_iters):
        batch_mxfp4_dequant(weight_ptrs, scale_ptrs, output, weights[0], scales[0], version=version)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        batch_mxfp4_dequant(weight_ptrs, scale_ptrs, output, weights[0], scales[0], version=version)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / bench_iters * 1000

    return elapsed


def benchmark_all_dequant_versions(
    num_experts: int = 128,
    N: int = 13824,
    K: int = 5120,
    device: str = "cuda",
    warmup_iters: int = 3,
    bench_iters: int = 10,
    print_results: bool = True,
) -> dict:
    """Benchmark all FP4 decode versions and compare performance.

    Args:
        num_experts: Number of experts (default 128 for GPT-OSS-120B)
        N: Output dimension (default 13824 intermediate_size)
        K: Input dimension (default 5120 hidden_size)
        device: CUDA device
        warmup_iters: Warmup iterations
        bench_iters: Benchmark iterations
        print_results: Whether to print comparison table

    Returns:
        Dictionary mapping version name to time in milliseconds.
    """
    # Create shared test data
    weights = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
               for _ in range(num_experts)]
    scales = [torch.randint(120, 134, (N, K // 32), dtype=torch.uint8, device=device)
              for _ in range(num_experts)]

    weight_ptrs = torch.tensor([w.data_ptr() for w in weights],
                               dtype=torch.int64, device=device)
    scale_ptrs = torch.tensor([s.data_ptr() for s in scales],
                              dtype=torch.int64, device=device)

    output = torch.empty(num_experts, N, K, dtype=torch.bfloat16, device=device)

    results = {}

    for version in FP4_DECODE_VERSIONS:
        # Warmup (compile kernel)
        for _ in range(warmup_iters):
            batch_mxfp4_dequant(weight_ptrs, scale_ptrs, output, weights[0], scales[0], version=version)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(bench_iters):
            batch_mxfp4_dequant(weight_ptrs, scale_ptrs, output, weights[0], scales[0], version=version)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / bench_iters * 1000

        results[version] = elapsed

    # Print results table
    if print_results:
        # Get GPU name
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"

        print(f"\n{'='*70}")
        print(f"FP4 Decode Version Comparison")
        print(f"{'='*70}")
        print(f"GPU: {gpu_name}")
        print(f"Config: {num_experts} experts, N={N}, K={K}")
        print(f"Output size: {num_experts * N * K * 2 / 1e9:.2f} GB BF16")
        print(f"{'='*70}")

        # Find best
        best_version = min(results, key=results.get)
        best_time = results[best_version]

        print(f"{'Version':<20} {'Time (ms)':<12} {'Speedup':<10} {'Notes'}")
        print(f"{'-'*70}")

        version_notes = {
            "v1_sequential": "16 tl.where() - baseline",
            "v2_e2m1": "E2M1 arithmetic (5-6 where)",
            "v3_binary_tree": "Binary tree (4 where)",
            "v4_branchless": "IEEE bitcast (2 where)",
            "v5_memopt": "BLOCK_K=64, 2x fewer K-blocks",
            "v6_scale_transpose": "K-major scales, coalesced loads",
            "v7_fast_scale": "tl.exp2 (SFU) instead of _ldexp (ALU)",
            "v8_ieee_pow2": "Direct IEEE pow2 (2 int ops + 1 mul)",
        }

        for version in FP4_DECODE_VERSIONS:
            time_ms = results[version]
            speedup = results["v1_sequential"] / time_ms if version != "v1_sequential" else 1.0
            marker = " <-- BEST" if version == best_version else ""
            notes = version_notes.get(version, "")
            print(f"{version:<20} {time_ms:<12.3f} {speedup:<10.2f}x {notes}{marker}")

        print(f"{'='*70}")
        print(f"Best: {best_version} at {best_time:.3f} ms")
        print(f"Speedup over baseline: {results['v1_sequential'] / best_time:.2f}x")
        print(f"{'='*70}\n")

    return results


def benchmark_bf16_grouped_gemm(
    num_experts: int,
    tokens_per_expert: int,
    N: int,
    K: int,
    device: str = "cuda",
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> float:
    """Benchmark BF16 grouped GEMM kernel.

    Returns time in milliseconds.
    """
    # Create test tensors
    hidden_3d = torch.randn(num_experts, tokens_per_expert, K,
                            dtype=torch.bfloat16, device=device)
    weight_buffer = torch.randn(num_experts, N, K,
                                dtype=torch.bfloat16, device=device)
    expert_counts = torch.full((num_experts,), tokens_per_expert,
                               dtype=torch.int32, device=device)

    # Warmup
    for _ in range(warmup_iters):
        _ = bf16_grouped_gemm_3d(hidden_3d, weight_buffer, expert_counts, N)
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(bench_iters):
        _ = bf16_grouped_gemm_3d(hidden_3d, weight_buffer, expert_counts, N)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / bench_iters * 1000

    return elapsed
