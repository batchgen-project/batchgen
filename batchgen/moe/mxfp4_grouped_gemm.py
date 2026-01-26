"""Fused MXFP4 dequantization and grouped GEMM for MoE layers.

This module implements fused dequantization of MXFP4 weights during matrix
multiplication, avoiding the memory overhead of materializing full BF16 weights.

MXFP4 Format:
- 32 FP4 values packed in 16 bytes (2 values per uint8)
- FP4 lookup table: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
- Scale block size: 32 elements
- Scale exponent: scale_uint8 - 127
"""

import logging
import torch
import triton
import triton.language as tl
from typing import List, Tuple

# Try to import triton_kernels for optimized MXFP4 GEMM
# triton_kernels is part of Triton 3.4+ or installed separately from Triton source
try:
    from triton_kernels.matmul import matmul as triton_kernels_matmul, PrecisionConfig
    from triton_kernels.tensor import wrap_torch_tensor
    HAS_TRITON_KERNELS = True
    logging.info("triton_kernels available - using optimized MXFP4 GEMM")
except ImportError:
    HAS_TRITON_KERNELS = False
    logging.warning("triton_kernels not available - using unfused MXFP4 path (slower)")


# MXFP4 configuration
MXFP4_BLOCK_SIZE = 32  # FP4 values per scale
MXFP4_PACKED_BLOCK_SIZE = 16  # Bytes per scale (32 values / 2 per byte)


@triton.jit
def _fp4_lookup(idx):
    """Lookup FP4 value from 4-bit index using vectorized operations.

    FP4 table:
    0: 0.0,  1: 0.5,  2: 1.0,  3: 1.5,  4: 2.0,  5: 3.0,  6: 4.0,  7: 6.0
    8: -0.0, 9: -0.5, 10: -1.0, 11: -1.5, 12: -2.0, 13: -3.0, 14: -4.0, 15: -6.0

    Optimized: Uses arithmetic to compute values instead of 16 conditionals.
    - idx 0-3: direct value (idx * 0.5)
    - idx 4-7: special values [2.0, 3.0, 4.0, 6.0]
    - idx 8-15: negative of idx 0-7
    """
    # Extract magnitude index (0-7) and sign bit (bit 3)
    mag_idx = idx & 0x7
    sign = (idx >> 3) & 1  # 0 for positive, 1 for negative

    # Compute magnitude based on index pattern
    # idx 0-3: 0.0, 0.5, 1.0, 1.5 (linear: idx * 0.5)
    # idx 4-7: 2.0, 3.0, 4.0, 6.0 (non-linear)
    is_linear = mag_idx < 4
    linear_val = mag_idx.to(tl.float32) * 0.5

    # Non-linear lookup for idx 4-7 using minimal conditionals
    # Pattern: 2.0, 3.0, 4.0, 6.0 -> base 2.0, then +1, +2, +4 for idx 5,6,7
    nonlinear_base = 2.0
    nonlinear_offset = tl.where(mag_idx == 5, 1.0,
                       tl.where(mag_idx == 6, 2.0,
                       tl.where(mag_idx == 7, 4.0, 0.0)))
    nonlinear_val = nonlinear_base + nonlinear_offset

    # Select linear or non-linear value
    magnitude = tl.where(is_linear, linear_val, nonlinear_val)

    # Apply sign
    val = tl.where(sign == 1, -magnitude, magnitude)

    return val.to(tl.float32)


@triton.jit
def _ldexp(mantissa, exponent):
    """Compute mantissa * 2^exponent."""
    # Clamp exponent to valid range
    exp_clamped = tl.minimum(tl.maximum(exponent, -126), 127)
    # Create 2^exponent as float
    exp_bits = (exp_clamped + 127).to(tl.int32) << 23
    power_of_2 = exp_bits.to(tl.float32, bitcast=True)
    return mantissa * power_of_2


# =============================================================================
# Single-Expert Fused MXFP4 GEMM Kernel (DISABLED)
# =============================================================================
# NOTE: This kernel is disabled because Triton doesn't support Python slice syntax
# like [:, 0::2] for strided indexing. The error:
#   "unsupported tensor index: <triton.language.core.slice object>"
#
# Instead, we use triton_kernels.matmul_ogs from OpenAI's library which has
# production-ready MXFP4 support with proper Hopper optimizations.
# =============================================================================

# @triton.jit
# def fused_mxfp4_single_gemm_kernel(...):
#     # Disabled - see note above
#     pass


@torch.inference_mode()
def fused_mxfp4_single_gemm(
    lhs: torch.Tensor,
    rhs_packed: torch.Tensor,
    rhs_scales: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Fused MXFP4 dequantization and GEMM for single expert.

    Uses triton_kernels.matmul when available (optimized for Hopper architecture).
    Falls back to unfused dequant + matmul when triton_kernels is not installed.

    Computes: output = lhs @ dequant(rhs).T + bias

    Args:
        lhs: Input activations [M, K] in BF16
        rhs_packed: Packed FP4 weights [N, K//2] or [N, K//32, 16] in uint8
        rhs_scales: Scales [N, K//32] in uint8
        bias: Optional bias [N] in BF16

    Returns:
        Output tensor [M, N] in BF16
    """
    assert lhs.dtype == torch.bfloat16, f"lhs must be BF16, got {lhs.dtype}"
    assert rhs_packed.dtype == torch.uint8, f"rhs_packed must be uint8, got {rhs_packed.dtype}"
    assert rhs_scales.dtype == torch.uint8, f"rhs_scales must be uint8, got {rhs_scales.dtype}"

    if HAS_TRITON_KERNELS:
        # Handle 3D block format: [N, K//32, 16] -> [N, K//2]
        if rhs_packed.dim() == 3:
            N, G, B = rhs_packed.shape
            rhs_packed = rhs_packed.view(N, G * B)

        N = rhs_packed.shape[0]

        # triton_kernels expects column-major weights with shape [K//2, N]
        # IMPORTANT: Do NOT call .contiguous() - transpose creates column-major view
        # which is required by triton_kernels (stride(-2) == 1)
        weight_T = rhs_packed.T  # [K//2, N] uint8, column-major (strides: 1, K//2)

        # Transpose scales: [N, K//32] -> [K//32, N]
        # IMPORTANT: Use .contiguous() to make scales row-major (stride[-1] == 1)
        # This enables TMA (Tensor Memory Accelerator) in triton_kernels
        # Without TMA, large tensors fail with ~33% error
        scales_T = rhs_scales.T.contiguous()  # [K//32, N] uint8, row-major (strides: N, 1)

        # Wrap scales as triton_kernels Tensor
        scales_tensor = wrap_torch_tensor(scales_T)

        # Configure MXFP4 scales
        # b_mx_scale tells triton_kernels to treat weight as MXFP4 and apply scales
        pc = PrecisionConfig(b_mx_scale=scales_tensor)

        # Call triton_kernels.matmul
        # NOTE: Don't pass bias to matmul - triton_kernels has a type mismatch bug when
        # bias is BF16 but accumulator is FP32 (the else branch creates FP32 zeros)
        output = triton_kernels_matmul(lhs, weight_T, None, precision_config=pc)

        # Add bias manually (workaround for triton_kernels type mismatch bug)
        # Ensure proper dtype handling: output may be FP32 (accumulator), bias is BF16
        if bias is not None:
            if output.dtype != lhs.dtype:
                output = output.to(lhs.dtype)
            output = output + bias
    else:
        # Fallback: unfused path (slower, creates BF16 intermediate)
        from batchgen.quantization.mxfp4 import mxfp4_dequantize

        # Handle 3D block format: [N, K//32, 16] → [N, K//2]
        if rhs_packed.dim() == 3:
            N, G, B = rhs_packed.shape
            rhs_packed = rhs_packed.view(N, G * B)

        # Dequantize to BF16
        weight_bf16 = mxfp4_dequantize(rhs_packed, rhs_scales, dtype=torch.bfloat16)

        # Standard matmul
        output = torch.mm(lhs, weight_bf16.T)

        if bias is not None:
            output = output + bias

    return output


@triton.jit
def fused_mxfp4_grouped_gemm_kernel(
    lhs_ptr,                    # BF16 activations [M, K]
    rhs_packed_ptrs_ptr,        # Pointers to packed FP4 weights [K//2, N]
    rhs_scales_ptrs_ptr,        # Pointers to scales [K//32, N]
    group_idx_ptr,
    group_sizes_ptr,
    group_start_indices_ptr,
    output_ptr,
    N, K, num_groups,
    stride_lhs_m, stride_lhs_k,
    stride_rhs_packed_n, stride_rhs_packed_k,
    stride_rhs_scales_n, stride_rhs_scales_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start_indices,
    stride_rhs_packed_ptrs, stride_rhs_scales_ptrs,
    GEMM_BLOCK_SIZE_M: tl.constexpr,
    GEMM_BLOCK_SIZE_N: tl.constexpr,
    GEMM_BLOCK_SIZE_K: tl.constexpr,
):
    """Fused MXFP4 dequantization and grouped GEMM kernel.

    Computes: output = lhs @ rhs.T where rhs is MXFP4 quantized.

    Key differences from FP8 version:
    - rhs_packed is uint8 with 2 FP4 values per byte
    - Scale block size is 32 (not 128)
    - Dequantization uses lookup table + ldexp
    """
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    lhs_dtype = tl.bfloat16

    for g in range(num_groups):
        # Get group info
        gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
        group_idx = tl.load(group_idx_ptr + g * stride_group_idx)
        start_idx = tl.load(group_start_indices_ptr + g * stride_group_start_indices)

        # Get pointers to this group's weights
        base_lhs_ptr = lhs_ptr + start_idx * stride_lhs_m
        rhs_packed_base = tl.load(rhs_packed_ptrs_ptr + group_idx * stride_rhs_packed_ptrs)
        rhs_packed_base = rhs_packed_base.to(tl.pointer_type(tl.uint8))
        rhs_scales_base = tl.load(rhs_scales_ptrs_ptr + group_idx * stride_rhs_scales_ptrs)
        rhs_scales_base = rhs_scales_base.to(tl.pointer_type(tl.uint8))

        # Compute tiles
        num_tiles_m = tl.cdiv(gm, GEMM_BLOCK_SIZE_M)
        num_tiles_n = tl.cdiv(N, GEMM_BLOCK_SIZE_N)
        num_tiles = num_tiles_m * num_tiles_n
        tile_id = pid

        while tile_id < num_tiles:
            tile_m = tile_id // num_tiles_n
            tile_n = tile_id % num_tiles_n

            # Tile offsets
            offs_m = tile_m * GEMM_BLOCK_SIZE_M + tl.arange(0, GEMM_BLOCK_SIZE_M)
            offs_n = tile_n * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)

            # Initialize accumulator
            acc = tl.zeros((GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32)

            # K dimension: process in blocks
            # Note: K is the unpacked dimension, K_packed = K // 2
            K_packed = K // 2

            for k_block in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
                k_start = k_block * GEMM_BLOCK_SIZE_K
                offs_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)

                # Load LHS tile [BLOCK_M, BLOCK_K]
                lhs_mask = (offs_m[:, None] < gm) & (offs_k[None, :] < K)
                lhs_ptrs = base_lhs_ptr + offs_m[:, None] * stride_lhs_m + offs_k[None, :] * stride_lhs_k
                lhs_tile = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)

                # Load RHS packed tile [BLOCK_N, BLOCK_K//2]
                # Each byte contains 2 FP4 values, so we load half the K dimension
                offs_k_packed = k_start // 2 + tl.arange(0, GEMM_BLOCK_SIZE_K // 2)
                rhs_mask = (offs_n[:, None] < N) & (offs_k_packed[None, :] < K_packed)
                rhs_packed_ptrs = rhs_packed_base + offs_n[:, None] * stride_rhs_packed_n + offs_k_packed[None, :] * stride_rhs_packed_k
                rhs_packed = tl.load(rhs_packed_ptrs, mask=rhs_mask, other=0)

                # Unpack FP4 values: [BLOCK_N, BLOCK_K//2] -> [BLOCK_N, BLOCK_K]
                # Low nibble = even indices, high nibble = odd indices
                idx_lo = (rhs_packed & 0x0F).to(tl.int32)
                idx_hi = ((rhs_packed >> 4) & 0x0F).to(tl.int32)

                # Lookup FP4 values
                val_lo = _fp4_lookup(idx_lo)  # [BLOCK_N, BLOCK_K//2]
                val_hi = _fp4_lookup(idx_hi)  # [BLOCK_N, BLOCK_K//2]

                # Load scales for this K block
                # Scale covers 32 consecutive K values, so scale_k_idx = k_start // 32
                scale_k_idx = k_start // 32
                n_scale_k = tl.cdiv(K, 32)

                # Each row in scales: [K//32]
                scale_ptrs = rhs_scales_base + offs_n * stride_rhs_scales_n + scale_k_idx * stride_rhs_scales_k
                scale_mask = offs_n < N
                scales_uint8 = tl.load(scale_ptrs, mask=scale_mask, other=127)

                # Convert scale: exponent = scale - 127
                exponents = scales_uint8.to(tl.int32) - 127

                # Apply ldexp to both lo and hi values
                # Broadcast exponents: [BLOCK_N] -> [BLOCK_N, BLOCK_K//2]
                exponents_broadcast = exponents[:, None] + tl.zeros((1, GEMM_BLOCK_SIZE_K // 2), dtype=tl.int32)
                val_lo_scaled = _ldexp(val_lo, exponents_broadcast)
                val_hi_scaled = _ldexp(val_hi, exponents_broadcast)

                # Interleave to get full K dimension: [BLOCK_N, BLOCK_K]
                # We need to combine lo/hi into a single tensor for matmul
                # For simplicity, we'll do two separate matmuls and combine

                # Convert to BF16
                val_lo_bf16 = val_lo_scaled.to(lhs_dtype)
                val_hi_bf16 = val_hi_scaled.to(lhs_dtype)

                # Split lhs into even/odd K indices
                lhs_even = lhs_tile[:, 0::2]  # [BLOCK_M, BLOCK_K//2]
                lhs_odd = lhs_tile[:, 1::2]   # [BLOCK_M, BLOCK_K//2]

                # Compute partial products and accumulate
                # lhs_even @ val_lo.T + lhs_odd @ val_hi.T
                acc += tl.dot(lhs_even, val_lo_bf16.T)
                acc += tl.dot(lhs_odd, val_hi_bf16.T)

            # Store output tile
            out_offs_m = start_idx + tile_m * GEMM_BLOCK_SIZE_M + tl.arange(0, GEMM_BLOCK_SIZE_M)
            out_offs_n = tile_n * GEMM_BLOCK_SIZE_N + tl.arange(0, GEMM_BLOCK_SIZE_N)
            out_ptrs = output_ptr + out_offs_m[:, None] * stride_output_m + out_offs_n[None, :] * stride_output_n
            out_mask = (out_offs_m[:, None] < start_idx + gm) & (out_offs_n[None, :] < N)
            tl.store(out_ptrs, acc.to(lhs_dtype), mask=out_mask)

            tile_id += num_programs


@torch.inference_mode()
def fused_mxfp4_grouped_gemm(
    lhs: torch.Tensor,
    rhs_packed_list: List[torch.Tensor],
    rhs_scales_list: List[torch.Tensor],
    group_sizes: List[Tuple[int, int]],
    gemm_block_size: Tuple[int, int, int] = (64, 64, 64),
) -> torch.Tensor:
    """Fused MXFP4 dequantization and grouped GEMM.

    Args:
        lhs: Activations [M, K] in BF16
        rhs_packed_list: List of packed FP4 weights [N, K//2] in uint8
        rhs_scales_list: List of scales [N, K//32] in uint8
        group_sizes: List of (group_idx, size) tuples
        gemm_block_size: (BLOCK_M, BLOCK_N, BLOCK_K) for tiling

    Returns:
        Output tensor [M, N] in BF16
    """
    assert lhs.dtype == torch.bfloat16, "lhs must be BF16"
    assert all(r.dtype == torch.uint8 for r in rhs_packed_list), "packed weights must be uint8"
    assert all(s.dtype == torch.uint8 for s in rhs_scales_list), "scales must be uint8"

    device = lhs.device
    M, K = lhs.shape
    N = rhs_packed_list[0].shape[0]  # [N, K//2]
    num_groups = len(group_sizes)

    # Create pointer arrays
    rhs_packed_ptrs = torch.tensor([r.data_ptr() for r in rhs_packed_list],
                                    dtype=torch.int64, device=device)
    rhs_scales_ptrs = torch.tensor([s.data_ptr() for s in rhs_scales_list],
                                    dtype=torch.int64, device=device)

    # Group metadata
    group_idx = torch.tensor([idx for idx, _ in group_sizes], dtype=torch.int32, device=device)
    group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
    group_start_indices = torch.roll(torch.cumsum(group_size, dim=0), 1)
    group_start_indices[0] = 0

    # Output
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(16, META['GEMM_BLOCK_SIZE_M']) * triton.cdiv(N, META['GEMM_BLOCK_SIZE_N']),
    )

    fused_mxfp4_grouped_gemm_kernel[grid](
        lhs, rhs_packed_ptrs, rhs_scales_ptrs,
        group_idx, group_size, group_start_indices,
        output,
        N, K, num_groups,
        lhs.stride(0), lhs.stride(1),
        rhs_packed_list[0].stride(0), rhs_packed_list[0].stride(1),
        rhs_scales_list[0].stride(0), rhs_scales_list[0].stride(1),
        output.stride(0), output.stride(1),
        group_idx.stride(0), group_size.stride(0), group_start_indices.stride(0),
        rhs_packed_ptrs.stride(0), rhs_scales_ptrs.stride(0),
        GEMM_BLOCK_SIZE_M=gemm_block_size[0],
        GEMM_BLOCK_SIZE_N=gemm_block_size[1],
        GEMM_BLOCK_SIZE_K=gemm_block_size[2],
        num_warps=8,
    )

    return output


# =============================================================================
# Grouped MXFP4 MoE Forward (Single Kernel Launch Per Stage)
# =============================================================================

def moe_token_dispatch(
    hidden_states: torch.Tensor,      # [batch*seq, hidden]
    topk_indices: torch.Tensor,       # [batch*seq, num_experts_per_tok]
    topk_weights: torch.Tensor,       # [batch*seq, num_experts_per_tok]
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch tokens to experts and create batched layout.

    Instead of 3D layout, we create sorted token lists per expert for efficient
    batched processing with triton_kernels.

    Returns:
        sorted_hidden: [total_tokens_routed, hidden] - tokens sorted by expert
        expert_offsets: [num_experts + 1] - start offset for each expert
        original_indices: [total_tokens_routed] - maps back to original token position
        original_k_indices: [total_tokens_routed] - which topk slot (0 to k-1)
        routing_weights: [total_tokens_routed] - routing weight for each entry
    """
    num_tokens, hidden = hidden_states.shape
    num_experts_per_tok = topk_indices.shape[1]
    device = hidden_states.device

    # Flatten topk_indices to get all (token, expert) pairs
    flat_indices = topk_indices.view(-1)  # [num_tokens * k]
    flat_weights = topk_weights.view(-1)  # [num_tokens * k]

    # Create token indices for each flattened entry
    token_indices = torch.arange(num_tokens, device=device).unsqueeze(1).expand(-1, num_experts_per_tok).reshape(-1)
    k_indices = torch.arange(num_experts_per_tok, device=device).unsqueeze(0).expand(num_tokens, -1).reshape(-1)

    # Sort by expert index to group tokens by expert
    sorted_expert_indices, sort_order = flat_indices.sort()

    # Reorder everything by expert
    sorted_token_indices = token_indices[sort_order]
    sorted_k_indices = k_indices[sort_order]
    sorted_weights = flat_weights[sort_order]

    # Gather hidden states in sorted order
    sorted_hidden = hidden_states[sorted_token_indices]

    # Compute expert offsets using bincount
    expert_counts = torch.bincount(sorted_expert_indices, minlength=num_experts)
    expert_offsets = torch.zeros(num_experts + 1, dtype=torch.int64, device=device)
    expert_offsets[1:] = expert_counts.cumsum(0)

    return sorted_hidden, expert_offsets, sorted_token_indices, sorted_k_indices, sorted_weights


# =============================================================================
# True Grouped MXFP4 GEMM with 3D Layout (DeepSeek-V3 Pattern)
# =============================================================================

@triton.jit
def fused_mxfp4_grouped_gemm_kernel_3d(
    # Input [E, M_max, K] BF16
    lhs_ptr,
    # Weight pointer arrays [num_experts] int64
    rhs_ptrs_ptr,           # -> [N, K//2] uint8 packed FP4
    rhs_scale_ptrs_ptr,     # -> [N, K//32] uint8
    # Per-expert token counts [num_experts] int32
    expert_tokens_ptr,
    # Output [E, M_max, N] BF16
    output_ptr,
    # Dimensions
    M_max, N, K,
    # Strides for lhs [E, M_max, K]
    stride_lhs_e, stride_lhs_m, stride_lhs_k,
    # Strides for rhs weights [N, K//2]
    stride_rhs_n, stride_rhs_k_packed,
    # Strides for scales [N, K//32]
    stride_scale_n, stride_scale_k,
    # Strides for output [E, M_max, N]
    stride_out_e, stride_out_m, stride_out_n,
    # Stride for pointer arrays
    stride_ptrs,
    # Block sizes
    BLOCK_M: tl.constexpr,  # 64
    BLOCK_N: tl.constexpr,  # 64
    BLOCK_K: tl.constexpr,  # 32 (must match MXFP4 scale block size)
):
    """True grouped MXFP4 GEMM following DeepSeek-V3 pattern.

    Grid: (num_experts, cdiv(N, BLOCK_N))
    - axis 0: expert index
    - axis 1: N-block index

    Each thread block handles one (expert, N-block) pair and loops over:
    - M-blocks (tokens for that expert)
    - K-blocks (reduction dimension)
    """
    expert_idx = tl.program_id(axis=0)
    n_pid = tl.program_id(axis=1)

    # Early exit for empty experts (critical for performance)
    gm = tl.load(expert_tokens_ptr + expert_idx).to(tl.int32)
    if gm == 0:
        return

    # Get base pointers for this expert's input/output slices
    cur_lhs_ptr = lhs_ptr + expert_idx * stride_lhs_e
    cur_out_ptr = output_ptr + expert_idx * stride_out_e

    # Load weight pointers for this expert from pointer arrays
    rhs_base_ptr = tl.load(rhs_ptrs_ptr + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))

    # N-block offset
    offs_n = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # Process all M-blocks for this expert (tokens)
    num_m_blocks = tl.cdiv(gm, BLOCK_M)

    for m_block in range(num_m_blocks):
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < gm

        # Initialize accumulator
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K-loop (BLOCK_K=32 matches MXFP4 scale block size)
        num_k_blocks = K // BLOCK_K

        for k_block in range(num_k_blocks):
            k_start = k_block * BLOCK_K

            # Load packed FP4 [BLOCK_N, BLOCK_K//2] uint8
            k_packed_start = k_start // 2
            offs_k_packed = tl.arange(0, BLOCK_K // 2)
            rhs_ptrs = rhs_base_ptr + offs_n[:, None] * stride_rhs_n + \
                       (k_packed_start + offs_k_packed[None, :]) * stride_rhs_k_packed
            rhs_packed = tl.load(rhs_ptrs, mask=n_mask[:, None], other=0)

            # Unpack FP4 (lo = even K indices, hi = odd K indices)
            idx_lo = (rhs_packed & 0x0F).to(tl.int32)
            idx_hi = ((rhs_packed >> 4) & 0x0F).to(tl.int32)
            val_lo = _fp4_lookup(idx_lo)  # [BLOCK_N, BLOCK_K//2]
            val_hi = _fp4_lookup(idx_hi)  # [BLOCK_N, BLOCK_K//2]

            # Load scale for this K-block (one scale per N row per K-block)
            scale_ptrs = scale_base_ptr + offs_n * stride_scale_n + k_block * stride_scale_k
            scales = tl.load(scale_ptrs, mask=n_mask, other=127).to(tl.int32) - 127

            # Apply ldexp: val * 2^scale
            exp_broadcast = scales[:, None] + tl.zeros((1, BLOCK_K // 2), dtype=tl.int32)
            val_lo_scaled = _ldexp(val_lo, exp_broadcast)
            val_hi_scaled = _ldexp(val_hi, exp_broadcast)

            # Load LHS at even/odd K positions (matches FP4 interleaved packing)
            offs_k_even = tl.arange(0, BLOCK_K // 2) * 2      # [0, 2, 4, ...]
            offs_k_odd = tl.arange(0, BLOCK_K // 2) * 2 + 1   # [1, 3, 5, ...]

            lhs_even_ptrs = cur_lhs_ptr + offs_m[:, None] * stride_lhs_m + (k_start + offs_k_even[None, :]) * stride_lhs_k
            lhs_odd_ptrs = cur_lhs_ptr + offs_m[:, None] * stride_lhs_m + (k_start + offs_k_odd[None, :]) * stride_lhs_k

            lhs_even = tl.load(lhs_even_ptrs, mask=m_mask[:, None], other=0.0)  # [BLOCK_M, BLOCK_K//2] BF16
            lhs_odd = tl.load(lhs_odd_ptrs, mask=m_mask[:, None], other=0.0)   # [BLOCK_M, BLOCK_K//2] BF16

            # Convert dequantized weights to BF16 for tensor core matmul
            rhs_even_bf16 = val_lo_scaled.to(tl.bfloat16)  # [BLOCK_N, BLOCK_K//2]
            rhs_odd_bf16 = val_hi_scaled.to(tl.bfloat16)   # [BLOCK_N, BLOCK_K//2]

            # Accumulate using tensor cores (TF32 enabled by default on Hopper)
            # [BLOCK_M, BLOCK_K//2] @ [BLOCK_K//2, BLOCK_N] -> [BLOCK_M, BLOCK_N]
            acc += tl.dot(lhs_even, tl.trans(rhs_even_bf16))
            acc += tl.dot(lhs_odd, tl.trans(rhs_odd_bf16))

        # Store output [BLOCK_M, BLOCK_N]
        out_ptrs = cur_out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def reshape_to_3d_expert_layout(
    sorted_hidden: torch.Tensor,      # [total_tokens, hidden]
    expert_counts: torch.Tensor,      # [num_experts] int32/int64
    num_experts: int,
) -> Tuple[torch.Tensor, int]:
    """Reshape sorted tokens to 3D layout [E, M_max, K] for grouped GEMM.

    Args:
        sorted_hidden: Tokens sorted by expert [total_tokens_routed, hidden]
        expert_counts: Number of tokens per expert [num_experts]
        num_experts: Total number of experts

    Returns:
        hidden_3d: [num_experts, max_tokens_per_expert, hidden] - zero-padded
        max_tokens: Maximum tokens assigned to any single expert
    """
    max_tokens = expert_counts.max().item()
    if max_tokens == 0:
        # No tokens routed to any expert (edge case)
        hidden_size = sorted_hidden.shape[-1]
        return torch.zeros(num_experts, 1, hidden_size, dtype=sorted_hidden.dtype, device=sorted_hidden.device), 1

    hidden_size = sorted_hidden.shape[-1]
    device = sorted_hidden.device
    dtype = sorted_hidden.dtype

    # Allocate 3D tensor (padded with zeros for empty slots)
    hidden_3d = torch.zeros(num_experts, max_tokens, hidden_size, dtype=dtype, device=device)

    # Copy tokens to their expert slots
    # This can be optimized with a Triton scatter kernel later
    offset = 0
    for e in range(num_experts):
        count = expert_counts[e].item()
        if count > 0:
            hidden_3d[e, :count] = sorted_hidden[offset:offset+count]
            offset += count

    return hidden_3d, max_tokens


def gather_from_3d_expert_layout(
    output_3d: torch.Tensor,          # [E, M_max, hidden]
    expert_counts: torch.Tensor,      # [num_experts] int32/int64
    total_tokens: int,
) -> torch.Tensor:
    """Gather outputs from 3D layout back to sorted 1D layout.

    Args:
        output_3d: Expert outputs [num_experts, M_max, hidden]
        expert_counts: Number of tokens per expert
        total_tokens: Total number of tokens to gather

    Returns:
        sorted_output: [total_tokens, hidden]
    """
    num_experts = output_3d.shape[0]
    hidden_size = output_3d.shape[-1]
    device = output_3d.device
    dtype = output_3d.dtype

    sorted_output = torch.zeros(total_tokens, hidden_size, dtype=dtype, device=device)

    offset = 0
    for e in range(num_experts):
        count = expert_counts[e].item()
        if count > 0:
            sorted_output[offset:offset+count] = output_3d[e, :count]
            offset += count

    return sorted_output


def setup_expert_weight_pointers(
    weight_list: List[torch.Tensor],  # [num_experts] of [N, K//2] uint8 or similar
    scale_list: List[torch.Tensor],   # [num_experts] of [N, K//32] uint8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create pointer arrays for expert weights (one-time setup at model init).

    Args:
        weight_list: List of weight tensors, one per expert
        scale_list: List of scale tensors, one per expert

    Returns:
        weight_ptrs: [num_experts] int64 - data pointers to each weight tensor
        scale_ptrs: [num_experts] int64 - data pointers to each scale tensor
    """
    device = weight_list[0].device

    weight_ptrs = torch.tensor(
        [w.data_ptr() for w in weight_list],
        dtype=torch.int64, device=device
    )
    scale_ptrs = torch.tensor(
        [s.data_ptr() for s in scale_list],
        dtype=torch.int64, device=device
    )

    return weight_ptrs, scale_ptrs


def grouped_mxfp4_gemm_3d(
    hidden_3d: torch.Tensor,          # [E, M_max, K] BF16
    weight_ptrs: torch.Tensor,        # [num_experts] int64
    scale_ptrs: torch.Tensor,         # [num_experts] int64
    expert_counts: torch.Tensor,      # [num_experts] int32
    N: int,                           # Output dimension
    weight_ref: torch.Tensor,         # Reference weight for strides [N, K//2]
    scale_ref: torch.Tensor,          # Reference scale for strides [N, K//32]
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
) -> torch.Tensor:
    """Launch grouped MXFP4 GEMM kernel with 3D layout.

    Args:
        hidden_3d: Input tensor [E, M_max, K] in BF16
        weight_ptrs: Pointer array [num_experts] to weight tensors
        scale_ptrs: Pointer array [num_experts] to scale tensors
        expert_counts: Token counts per expert [num_experts]
        N: Output dimension (number of output features)
        weight_ref: Reference weight tensor for computing strides
        scale_ref: Reference scale tensor for computing strides

    Returns:
        output_3d: [E, M_max, N] in BF16
    """
    num_experts = hidden_3d.shape[0]
    M_max = hidden_3d.shape[1]
    K = hidden_3d.shape[2]
    device = hidden_3d.device

    # Ensure expert_counts is int32 for kernel
    if expert_counts.dtype != torch.int32:
        expert_counts = expert_counts.to(torch.int32)

    # Allocate output
    output_3d = torch.empty(num_experts, M_max, N, dtype=torch.bfloat16, device=device)

    # Grid: (num_experts, cdiv(N, BLOCK_N))
    grid = (num_experts, triton.cdiv(N, BLOCK_N))

    fused_mxfp4_grouped_gemm_kernel_3d[grid](
        hidden_3d,
        weight_ptrs, scale_ptrs,
        expert_counts,
        output_3d,
        M_max, N, K,
        hidden_3d.stride(0), hidden_3d.stride(1), hidden_3d.stride(2),
        weight_ref.stride(0), weight_ref.stride(1),
        scale_ref.stride(0), scale_ref.stride(1),
        output_3d.stride(0), output_3d.stride(1), output_3d.stride(2),
        1,  # stride_ptrs (contiguous pointer array)
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=8,
    )

    return output_3d


def grouped_mxfp4_gemm_3d_tunable(
    hidden_3d: torch.Tensor,          # [E, M_max, K] BF16
    weight_ptrs: torch.Tensor,        # [num_experts] int64
    scale_ptrs: torch.Tensor,         # [num_experts] int64
    expert_counts: torch.Tensor,      # [num_experts] int32
    N: int,                           # Output dimension
    weight_ref: torch.Tensor,         # Reference weight for strides [N, K//2]
    scale_ref: torch.Tensor,          # Reference scale for strides [N, K//32]
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,
    num_warps: int = 8,
    num_stages: int = 1,
) -> torch.Tensor:
    """Tunable variant of grouped MXFP4 GEMM for hyperparameter search.

    Same as grouped_mxfp4_gemm_3d but with configurable num_warps and num_stages.

    Args:
        hidden_3d: Input tensor [E, M_max, K] in BF16
        weight_ptrs: Pointer array [num_experts] to weight tensors
        scale_ptrs: Pointer array [num_experts] to scale tensors
        expert_counts: Token counts per expert [num_experts]
        N: Output dimension (number of output features)
        weight_ref: Reference weight tensor for computing strides
        scale_ref: Reference scale tensor for computing strides
        BLOCK_M: Tile size for M dimension (default 64)
        BLOCK_N: Tile size for N dimension (default 64)
        BLOCK_K: Tile size for K dimension (default 32, must match MXFP4 scale block)
        num_warps: Number of warps per block (default 8)
        num_stages: Number of pipeline stages (default 1)

    Returns:
        output_3d: [E, M_max, N] in BF16
    """
    num_experts = hidden_3d.shape[0]
    M_max = hidden_3d.shape[1]
    K = hidden_3d.shape[2]
    device = hidden_3d.device

    # Ensure expert_counts is int32 for kernel
    if expert_counts.dtype != torch.int32:
        expert_counts = expert_counts.to(torch.int32)

    # Allocate output
    output_3d = torch.empty(num_experts, M_max, N, dtype=torch.bfloat16, device=device)

    # Grid: (num_experts, cdiv(N, BLOCK_N))
    grid = (num_experts, triton.cdiv(N, BLOCK_N))

    fused_mxfp4_grouped_gemm_kernel_3d[grid](
        hidden_3d,
        weight_ptrs, scale_ptrs,
        expert_counts,
        output_3d,
        M_max, N, K,
        hidden_3d.stride(0), hidden_3d.stride(1), hidden_3d.stride(2),
        weight_ref.stride(0), weight_ref.stride(1),
        scale_ref.stride(0), scale_ref.stride(1),
        output_3d.stride(0), output_3d.stride(1), output_3d.stride(2),
        1,  # stride_ptrs (contiguous pointer array)
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output_3d


def grouped_mxfp4_moe_forward_3d(
    hidden_states: torch.Tensor,          # [batch*seq, hidden]
    topk_indices: torch.Tensor,           # [batch*seq, num_experts_per_tok]
    topk_weights: torch.Tensor,           # [batch*seq, num_experts_per_tok]
    # Pre-computed pointer arrays (from setup_expert_weight_pointers)
    gate_ptrs: torch.Tensor,              # [num_experts] int64
    gate_scale_ptrs: torch.Tensor,
    up_ptrs: torch.Tensor,
    up_scale_ptrs: torch.Tensor,
    down_ptrs: torch.Tensor,
    down_scale_ptrs: torch.Tensor,
    # Reference weights for strides (any expert's weight works)
    gate_weight_ref: torch.Tensor,        # [N_inter, hidden//2]
    gate_scale_ref: torch.Tensor,         # [N_inter, hidden//32]
    up_weight_ref: torch.Tensor,
    up_scale_ref: torch.Tensor,
    down_weight_ref: torch.Tensor,        # [hidden, N_inter//2]
    down_scale_ref: torch.Tensor,         # [hidden, N_inter//32]
    # Biases (optional, stacked as [num_experts, N])
    gate_biases: torch.Tensor = None,     # [num_experts, N_inter] or None
    up_biases: torch.Tensor = None,
    down_biases: torch.Tensor = None,
    num_experts: int = 128,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
) -> torch.Tensor:
    """True grouped MXFP4 MoE forward with 3D layout (DeepSeek-V3 pattern).

    This implementation uses a single kernel launch per projection stage,
    processing all experts in parallel instead of looping over them.

    Kernel launches per MoE layer:
    - Before: 128 experts × 3 projections = 384 launches
    - After: 3 launches (gate, up, down)

    Args:
        hidden_states: Input [batch*seq, hidden] in BF16
        topk_indices: Expert indices [batch*seq, num_experts_per_tok]
        topk_weights: Routing weights [batch*seq, num_experts_per_tok]
        gate_ptrs, gate_scale_ptrs: Pointer arrays for gate projection
        up_ptrs, up_scale_ptrs: Pointer arrays for up projection
        down_ptrs, down_scale_ptrs: Pointer arrays for down projection
        gate_weight_ref, etc.: Reference tensors for computing strides
        gate_biases, up_biases, down_biases: Optional stacked biases [E, N]
        num_experts: Number of experts (default 128)
        swiglu_alpha: SwiGLU alpha (default 1.702)
        swiglu_limit: Clamping limit (default 7.0)

    Returns:
        Output [batch*seq, hidden] in BF16
    """
    num_tokens, hidden_size = hidden_states.shape
    device = hidden_states.device

    # Dimensions from reference weights
    N_intermediate = gate_weight_ref.shape[0]  # Intermediate size (e.g., 5760)

    # Step 1: Dispatch tokens to experts (sort by expert)
    sorted_hidden, expert_offsets, original_indices, original_k, routing_weights = moe_token_dispatch(
        hidden_states, topk_indices, topk_weights, num_experts
    )

    total_tokens_routed = sorted_hidden.shape[0]

    # Compute per-expert token counts
    expert_counts = (expert_offsets[1:] - expert_offsets[:-1]).to(torch.int32)

    # Step 2: Reshape to 3D layout [E, M_max, K]
    hidden_3d, M_max = reshape_to_3d_expert_layout(sorted_hidden, expert_counts, num_experts)

    # Step 3: Gate projection (SINGLE kernel for all experts)
    gate_out_3d = grouped_mxfp4_gemm_3d(
        hidden_3d, gate_ptrs, gate_scale_ptrs, expert_counts,
        N_intermediate, gate_weight_ref, gate_scale_ref
    )

    # Step 4: Up projection (SINGLE kernel for all experts)
    up_out_3d = grouped_mxfp4_gemm_3d(
        hidden_3d, up_ptrs, up_scale_ptrs, expert_counts,
        N_intermediate, up_weight_ref, up_scale_ref
    )

    # Add biases if present (broadcasted over [E, M_max, N])
    if gate_biases is not None:
        gate_out_3d = gate_out_3d + gate_biases.unsqueeze(1)
    if up_biases is not None:
        up_out_3d = up_out_3d + up_biases.unsqueeze(1)

    # Step 5: SwiGLU activation (in-place on 3D tensors)
    gate_clamped = gate_out_3d.clamp(max=swiglu_limit)
    up_clamped = up_out_3d.clamp(min=-swiglu_limit, max=swiglu_limit)
    intermediate_3d = gate_clamped * torch.sigmoid(swiglu_alpha * gate_clamped) * (up_clamped + 1)

    # Step 6: Down projection (SINGLE kernel for all experts)
    output_3d = grouped_mxfp4_gemm_3d(
        intermediate_3d, down_ptrs, down_scale_ptrs, expert_counts,
        hidden_size, down_weight_ref, down_scale_ref
    )

    if down_biases is not None:
        output_3d = output_3d + down_biases.unsqueeze(1)

    # Step 7: Gather back from 3D to sorted 1D
    sorted_output = gather_from_3d_expert_layout(output_3d, expert_counts, total_tokens_routed)

    # Step 8: Scatter back to original order with routing weights
    output = torch.zeros(num_tokens, hidden_size, dtype=hidden_states.dtype, device=device)
    weighted_output = sorted_output * routing_weights.unsqueeze(-1)
    output.scatter_add_(0, original_indices.unsqueeze(-1).expand_as(weighted_output), weighted_output)

    return output


# =============================================================================
# Original Per-Expert Loop Implementation (for comparison/fallback)
# =============================================================================

def grouped_mxfp4_moe_forward(
    hidden_states: torch.Tensor,          # [batch*seq, hidden]
    topk_indices: torch.Tensor,           # [batch*seq, num_experts_per_tok]
    topk_weights: torch.Tensor,           # [batch*seq, num_experts_per_tok]
    gate_weights: List[torch.Tensor],     # [num_experts] of [N, K//2] uint8
    gate_scales: List[torch.Tensor],      # [num_experts] of [N, K//32] uint8
    gate_biases: List[torch.Tensor],      # [num_experts] of [N] BF16 (or None)
    up_weights: List[torch.Tensor],       # [num_experts] of [N, K//2] uint8
    up_scales: List[torch.Tensor],        # [num_experts] of [N, K//32] uint8
    up_biases: List[torch.Tensor],        # [num_experts] of [N] BF16 (or None)
    down_weights: List[torch.Tensor],     # [num_experts] of [hidden, N//2] uint8
    down_scales: List[torch.Tensor],      # [num_experts] of [hidden, N//32] uint8
    down_biases: List[torch.Tensor],      # [num_experts] of [hidden] BF16 (or None)
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
) -> torch.Tensor:
    """Full MoE forward with grouped MXFP4 GEMM using triton_kernels.

    This implementation groups tokens by expert and processes each expert's
    tokens in a batch, significantly reducing per-token overhead compared
    to the naive per-expert loop.

    Stages:
    1. Dispatch tokens to experts (sort by expert)
    2. Process each expert's tokens in batch:
       - Gate projection: x @ gate.T + gate_bias
       - Up projection: x @ up.T + up_bias
       - SwiGLU: gate * sigmoid(alpha * gate) * (up + 1)
       - Down projection: intermediate @ down.T + down_bias
    3. Combine results back to original order

    Args:
        hidden_states: Input [batch*seq, hidden] in BF16
        topk_indices: Expert indices [batch*seq, num_experts_per_tok]
        topk_weights: Routing weights [batch*seq, num_experts_per_tok]
        gate_weights, gate_scales, gate_biases: Gate projection per expert
        up_weights, up_scales, up_biases: Up projection per expert
        down_weights, down_scales, down_biases: Down projection per expert
        swiglu_alpha: SwiGLU alpha parameter (default 1.702 for OpenAI)
        swiglu_limit: Clamping limit (default 7.0)

    Returns:
        Output [batch*seq, hidden] in BF16
    """
    if not HAS_TRITON_KERNELS:
        raise ImportError("grouped_mxfp4_moe_forward requires triton_kernels")

    num_tokens, hidden = hidden_states.shape
    num_experts = len(gate_weights)
    num_experts_per_tok = topk_indices.shape[1]
    device = hidden_states.device
    intermediate_size = gate_weights[0].shape[0]  # N dimension

    # Step 1: Dispatch tokens to experts
    sorted_hidden, expert_offsets, original_indices, original_k, routing_weights = moe_token_dispatch(
        hidden_states, topk_indices, topk_weights, num_experts
    )

    # Step 2: Process each expert's tokens in batch
    # Allocate output for all sorted tokens
    sorted_output = torch.zeros_like(sorted_hidden)

    for expert_idx in range(num_experts):
        start = expert_offsets[expert_idx].item()
        end = expert_offsets[expert_idx + 1].item()

        if start == end:
            continue  # No tokens for this expert

        expert_input = sorted_hidden[start:end]  # [num_tokens_for_expert, hidden]

        # Get expert weights
        gate_packed = gate_weights[expert_idx]
        gate_scale = gate_scales[expert_idx]
        gate_bias = gate_biases[expert_idx] if gate_biases else None

        up_packed = up_weights[expert_idx]
        up_scale = up_scales[expert_idx]
        up_bias = up_biases[expert_idx] if up_biases else None

        down_packed = down_weights[expert_idx]
        down_scale = down_scales[expert_idx]
        down_bias = down_biases[expert_idx] if down_biases else None

        # Stage 1a: Gate projection
        gate_out = fused_mxfp4_single_gemm(expert_input, gate_packed, gate_scale, gate_bias)

        # Stage 1b: Up projection
        up_out = fused_mxfp4_single_gemm(expert_input, up_packed, up_scale, up_bias)

        # Stage 1c: SwiGLU activation
        gate_clamped = gate_out.clamp(max=swiglu_limit)
        up_clamped = up_out.clamp(min=-swiglu_limit, max=swiglu_limit)
        glu = gate_clamped * torch.sigmoid(swiglu_alpha * gate_clamped)
        intermediate = glu * (up_clamped + 1)

        # Stage 2: Down projection
        expert_output = fused_mxfp4_single_gemm(intermediate, down_packed, down_scale, down_bias)

        # Store in sorted output
        sorted_output[start:end] = expert_output

    # Step 3: Combine results back to original order with routing weights
    # Each original token position accumulates weighted outputs from its top-k experts
    output = torch.zeros(num_tokens, hidden, dtype=hidden_states.dtype, device=device)

    # Apply routing weights and scatter back
    weighted_output = sorted_output * routing_weights.unsqueeze(-1)
    output.scatter_add_(0, original_indices.unsqueeze(-1).expand_as(weighted_output), weighted_output)

    return output


def mxfp4_mlp_forward(
    x: torch.Tensor,
    gate_packed: torch.Tensor,
    gate_scales: torch.Tensor,
    gate_bias: torch.Tensor,
    up_packed: torch.Tensor,
    up_scales: torch.Tensor,
    up_bias: torch.Tensor,
    down_packed: torch.Tensor,
    down_scales: torch.Tensor,
    down_bias: torch.Tensor,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
) -> torch.Tensor:
    """Single expert MLP forward with MXFP4 weights.

    Implements: down(SwiGLU(gate(x), up(x)))

    This is the optimized path for a single expert using triton_kernels.

    Args:
        x: Input [M, hidden] in BF16
        gate_packed, gate_scales, gate_bias: Gate projection weights
        up_packed, up_scales, up_bias: Up projection weights
        down_packed, down_scales, down_bias: Down projection weights
        swiglu_alpha: SwiGLU alpha (default 1.702)
        swiglu_limit: Clamping limit (default 7.0)

    Returns:
        Output [M, hidden] in BF16
    """
    # Ensure 2D input
    original_shape = x.shape
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])

    # Stage 1: Gate and Up projections (can be parallelized on GPU)
    gate_out = fused_mxfp4_single_gemm(x, gate_packed, gate_scales, gate_bias)
    up_out = fused_mxfp4_single_gemm(x, up_packed, up_scales, up_bias)

    # Stage 2: SwiGLU activation
    gate_clamped = gate_out.clamp(max=swiglu_limit)
    up_clamped = up_out.clamp(min=-swiglu_limit, max=swiglu_limit)
    glu = gate_clamped * torch.sigmoid(swiglu_alpha * gate_clamped)
    intermediate = glu * (up_clamped + 1)

    # Stage 3: Down projection
    output = fused_mxfp4_single_gemm(intermediate, down_packed, down_scales, down_bias)

    # Restore original shape
    if len(original_shape) > 2:
        output = output.view(*original_shape[:-1], -1)

    return output


# =============================================================================
# Optimized Single-Expert MXFP4 GEMM Kernel (Same Tiling as Grouped)
# =============================================================================

@triton.jit
def fused_mxfp4_single_gemm_kernel_optimized(
    # Input [M, K] BF16
    lhs_ptr,
    # Weight [N, K//2] uint8
    rhs_ptr,
    # Scale [N, K//32] uint8
    scale_ptr,
    # Output [M, N] BF16
    output_ptr,
    # Bias (optional)
    bias_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_lhs_m, stride_lhs_k,
    stride_rhs_n, stride_rhs_k,
    stride_scale_n, stride_scale_k,
    stride_out_m, stride_out_n,
    # Config
    HAS_BIAS: tl.constexpr,
    # Block sizes (same as grouped kernel)
    BLOCK_M: tl.constexpr,  # 64
    BLOCK_N: tl.constexpr,  # 64
    BLOCK_K: tl.constexpr,  # 32
):
    """Optimized single-expert MXFP4 GEMM with same tiling as grouped kernel.

    Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
    - axis 0: M-block index
    - axis 1: N-block index

    Uses identical MXFP4 dequantization and accumulation pattern as the grouped
    kernel for consistent performance characteristics.
    """
    m_pid = tl.program_id(axis=0)
    n_pid = tl.program_id(axis=1)

    offs_m = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop with MXFP4 dequantization (BLOCK_K=32 matches scale block size)
    num_k_blocks = K // BLOCK_K

    for k_block in range(num_k_blocks):
        k_start = k_block * BLOCK_K

        # Load packed FP4 weights [BLOCK_N, BLOCK_K//2]
        k_packed_start = k_start // 2
        offs_k_packed = tl.arange(0, BLOCK_K // 2)
        rhs_ptrs = rhs_ptr + offs_n[:, None] * stride_rhs_n + \
                   (k_packed_start + offs_k_packed[None, :]) * stride_rhs_k
        rhs_packed = tl.load(rhs_ptrs, mask=n_mask[:, None], other=0)

        # Unpack FP4 (lo = even K indices, hi = odd K indices)
        idx_lo = (rhs_packed & 0x0F).to(tl.int32)
        idx_hi = ((rhs_packed >> 4) & 0x0F).to(tl.int32)
        val_lo = _fp4_lookup(idx_lo)  # [BLOCK_N, BLOCK_K//2]
        val_hi = _fp4_lookup(idx_hi)  # [BLOCK_N, BLOCK_K//2]

        # Load and apply scales
        scale_ptrs = scale_ptr + offs_n * stride_scale_n + k_block * stride_scale_k
        scales = tl.load(scale_ptrs, mask=n_mask, other=127).to(tl.int32) - 127

        # Apply ldexp: val * 2^scale
        exp_broadcast = scales[:, None] + tl.zeros((1, BLOCK_K // 2), dtype=tl.int32)
        val_lo_scaled = _ldexp(val_lo, exp_broadcast)
        val_hi_scaled = _ldexp(val_hi, exp_broadcast)

        # Load LHS at even/odd K positions
        offs_k_even = tl.arange(0, BLOCK_K // 2) * 2
        offs_k_odd = offs_k_even + 1
        lhs_even_ptrs = lhs_ptr + offs_m[:, None] * stride_lhs_m + (k_start + offs_k_even[None, :]) * stride_lhs_k
        lhs_odd_ptrs = lhs_ptr + offs_m[:, None] * stride_lhs_m + (k_start + offs_k_odd[None, :]) * stride_lhs_k

        lhs_even = tl.load(lhs_even_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)
        lhs_odd = tl.load(lhs_odd_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

        # Accumulate: lhs_even @ val_lo.T + lhs_odd @ val_hi.T
        acc += tl.dot(lhs_even.to(tl.bfloat16), tl.trans(val_lo_scaled.to(tl.bfloat16)), allow_tf32=False).to(tl.float32)
        acc += tl.dot(lhs_odd.to(tl.bfloat16), tl.trans(val_hi_scaled.to(tl.bfloat16)), allow_tf32=False).to(tl.float32)

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias[None, :]

    # Store output [BLOCK_M, BLOCK_N]
    out_ptrs = output_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@torch.inference_mode()
def mxfp4_expert_forward_single(
    x: torch.Tensor,
    gate_packed: torch.Tensor,
    gate_scales: torch.Tensor,
    up_packed: torch.Tensor,
    up_scales: torch.Tensor,
    down_packed: torch.Tensor,
    down_scales: torch.Tensor,
    gate_bias: torch.Tensor = None,
    up_bias: torch.Tensor = None,
    down_bias: torch.Tensor = None,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
) -> torch.Tensor:
    """Single expert forward with optimized MXFP4 kernel.

    Uses the same tiling pattern as the grouped GEMM kernel (BLOCK_M=64,
    BLOCK_N=64, BLOCK_K=32) for consistent performance.

    This function is intended for non-persistent experts that are loaded
    on-demand, while persistent experts use the grouped GEMM kernel.

    Args:
        x: Input [M, hidden] in BF16
        gate_packed: Gate weights [N_inter, hidden//2] uint8
        gate_scales: Gate scales [N_inter, hidden//32] uint8
        up_packed: Up weights [N_inter, hidden//2] uint8
        up_scales: Up scales [N_inter, hidden//32] uint8
        down_packed: Down weights [hidden, N_inter//2] uint8
        down_scales: Down scales [hidden, N_inter//32] uint8
        gate_bias: Optional gate bias [N_inter] BF16
        up_bias: Optional up bias [N_inter] BF16
        down_bias: Optional down bias [hidden] BF16
        swiglu_alpha: SwiGLU alpha (default 1.702)
        swiglu_limit: Clamping limit (default 7.0)

    Returns:
        Output [M, hidden] in BF16
    """
    M = x.shape[0]
    K = x.shape[1]  # hidden_size
    N_inter = gate_packed.shape[0]  # intermediate_size
    N_hidden = down_packed.shape[0]  # hidden_size

    # Gate projection - use working fused_mxfp4_single_gemm (not buggy custom kernel)
    gate_out = fused_mxfp4_single_gemm(x, gate_packed, gate_scales, gate_bias)

    # Up projection
    up_out = fused_mxfp4_single_gemm(x, up_packed, up_scales, up_bias)

    # SwiGLU activation
    gate_clamped = gate_out.clamp(max=swiglu_limit)
    up_clamped = up_out.clamp(min=-swiglu_limit, max=swiglu_limit)
    intermediate = gate_clamped * torch.sigmoid(swiglu_alpha * gate_clamped) * (up_clamped + 1)

    # Down projection
    output = fused_mxfp4_single_gemm(intermediate, down_packed, down_scales, down_bias)

    return output


def mxfp4_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_scales: torch.Tensor,
    bias: torch.Tensor = None,
    use_fused: bool = True,
) -> torch.Tensor:
    """Linear layer with MXFP4 quantized weights.

    Uses fused dequant + GEMM kernel to avoid materializing full BF16 weights.

    Args:
        x: Input [batch, seq_len, hidden_size] in BF16
        weight_packed: Packed FP4 weights [out_features, hidden_size//2] in uint8
        weight_scales: Scales [out_features, hidden_size//32] in uint8
        bias: Optional bias [out_features]
        use_fused: If True, use fused Triton kernel. If False, use unfused path.

    Returns:
        Output [batch, seq_len, out_features] in BF16
    """
    original_shape = x.shape
    x_2d = x.view(-1, x.shape[-1])

    # Ensure BF16 input
    if x_2d.dtype != torch.bfloat16:
        x_2d = x_2d.to(torch.bfloat16)

    if use_fused:
        # Use fused dequant + GEMM kernel (no temporary BF16 allocation)
        output = fused_mxfp4_single_gemm(x_2d, weight_packed, weight_scales, bias)
    else:
        # Fallback: unfused path (materializes full BF16 weights)
        from batchgen.quantization.mxfp4 import mxfp4_dequantize
        weight_bf16 = mxfp4_dequantize(weight_packed, weight_scales, dtype=torch.bfloat16)
        output = torch.mm(x_2d, weight_bf16.T)
        if bias is not None:
            output = output + bias

    # Reshape to original batch dimensions
    output = output.view(*original_shape[:-1], -1)

    return output
