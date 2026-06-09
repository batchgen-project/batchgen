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
from typing import List, Tuple

import torch
import triton
import triton.language as tl

# Try to import triton_kernels for optimized MXFP4 GEMM
# triton_kernels is part of Triton 3.4+ or installed separately from Triton source
try:
    import triton_kernels.matmul as tk_matmul
    from triton_kernels.tensor import wrap_torch_tensor

    PrecisionConfig = tk_matmul.PrecisionConfig
    triton_kernels_matmul = tk_matmul.matmul
    HAS_TRITON_KERNELS = True
    logging.info("triton_kernels available - using optimized MXFP4 GEMM")
except ImportError:
    HAS_TRITON_KERNELS = False
    logging.warning(
        "triton_kernels not available - using unfused MXFP4 path (slower)"
    )


# MXFP4 configuration
MXFP4_BLOCK_SIZE = 32  # FP4 values per scale
MXFP4_PACKED_BLOCK_SIZE = 16  # Bytes per scale (32 values / 2 per byte)


@triton.jit
def _fp4_lookup(idx):
    """Lookup FP4 value from 4-bit index (LEGACY - slow, 16 tl.where calls).

    FP4 table:
    0: 0.0,  1: 0.5,  2: 1.0,  3: 1.5,  4: 2.0,  5: 3.0,  6: 4.0,  7: 6.0
    8: -0.0, 9: -0.5, 10: -1.0, 11: -1.5, 12: -2.0, 13: -3.0, 14: -4.0, 15: -6.0

    DEPRECATED: Use _fp4_decode_v4_branchless instead (2 tl.where, 8x faster).
    """
    # Positive values (idx 0-7)
    val = tl.where(idx == 0, 0.0, 0.0)
    val = tl.where(idx == 1, 0.5, val)
    val = tl.where(idx == 2, 1.0, val)
    val = tl.where(idx == 3, 1.5, val)
    val = tl.where(idx == 4, 2.0, val)
    val = tl.where(idx == 5, 3.0, val)
    val = tl.where(idx == 6, 4.0, val)
    val = tl.where(idx == 7, 6.0, val)

    # Negative values (idx 8-15)
    val = tl.where(idx == 8, -0.0, val)
    val = tl.where(idx == 9, -0.5, val)
    val = tl.where(idx == 10, -1.0, val)
    val = tl.where(idx == 11, -1.5, val)
    val = tl.where(idx == 12, -2.0, val)
    val = tl.where(idx == 13, -3.0, val)
    val = tl.where(idx == 14, -4.0, val)
    val = tl.where(idx == 15, -6.0, val)

    return val.to(tl.float32)


@triton.jit
def _fp4_decode_v4_branchless(idx):
    """Decode FP4 using branchless bit manipulation (FAST - 2 tl.where calls).

    Constructs IEEE 754 float32 representation directly from FP4 bits.
    This is ~8x faster than _fp4_lookup which uses 16 sequential tl.where calls.

    FP4 E2M1 layout: [S][E1][E0][M]
    IEEE 754 float32: [S][8-bit exp][23-bit mantissa]

    Args:
        idx: 4-bit FP4 index (0-15)
    Returns:
        Decoded float32 values
    """
    # Extract FP4 fields
    sign_bit = (idx >> 3) & 1  # Bit 3
    exp_field = (idx >> 1) & 0x3  # Bits 1-2 (0-3)
    mant_bit = idx & 1  # Bit 0

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
    ieee_mant_normal = mant_bit << 22  # Mantissa in bit 22
    ieee_normal = (sign_bit << 31) | ieee_exp_normal | ieee_mant_normal

    # Subnormal case: exp = 0
    # M=0 → 0.0: all zeros (or negative zero)
    # M=1 → 0.5: exp=126, mant=0
    ieee_half = (sign_bit << 31) | (126 << 23)  # 0.5 or -0.5
    ieee_zero = sign_bit << 31  # 0.0 or -0.0
    ieee_subnormal = tl.where(mant_bit == 1, ieee_half, ieee_zero)

    # Select normal vs subnormal (1 branch)
    ieee_bits = tl.where(exp_field > 0, ieee_normal, ieee_subnormal)

    # Bitcast to float32
    return ieee_bits.to(tl.float32, bitcast=True)


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
    assert rhs_packed.dtype == torch.uint8, (
        f"rhs_packed must be uint8, got {rhs_packed.dtype}"
    )
    assert rhs_scales.dtype == torch.uint8, (
        f"rhs_scales must be uint8, got {rhs_scales.dtype}"
    )

    if HAS_TRITON_KERNELS:
        # Handle 3D block format: [N, K//32, 16] -> [N, K//2]
        if rhs_packed.dim() == 3:
            N, G, B = rhs_packed.shape
            rhs_packed = rhs_packed.view(N, G * B)

        N = rhs_packed.shape[0]

        # triton_kernels expects column-major weights with shape [K//2, N]
        # IMPORTANT: Do NOT call .contiguous() - transpose creates column-major view
        # which is required by triton_kernels (stride(-2) == 1)
        weight_T = (
            rhs_packed.T
        )  # [K//2, N] uint8, column-major (strides: 1, K//2)

        # Transpose scales: [N, K//32] -> [K//32, N]
        # IMPORTANT: Use .contiguous() to make scales row-major (stride[-1] == 1)
        # This enables TMA (Tensor Memory Accelerator) in triton_kernels
        # Without TMA, large tensors fail with ~33% error
        scales_T = (
            rhs_scales.T.contiguous()
        )  # [K//32, N] uint8, row-major (strides: N, 1)

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
        weight_bf16 = mxfp4_dequantize(
            rhs_packed, rhs_scales, dtype=torch.bfloat16
        )

        # Standard matmul
        output = torch.mm(lhs, weight_bf16.T)

        if bias is not None:
            output = output + bias

    return output


@triton.jit
def fused_mxfp4_grouped_gemm_kernel(
    lhs_ptr,  # BF16 activations [M, K]
    rhs_packed_ptrs_ptr,  # Pointers to packed FP4 weights [K//2, N]
    rhs_scales_ptrs_ptr,  # Pointers to scales [K//32, N]
    group_idx_ptr,
    group_sizes_ptr,
    group_start_indices_ptr,
    output_ptr,
    N,
    K,
    num_groups,
    stride_lhs_m,
    stride_lhs_k,
    stride_rhs_packed_n,
    stride_rhs_packed_k,
    stride_rhs_scales_n,
    stride_rhs_scales_k,
    stride_output_m,
    stride_output_n,
    stride_group_idx,
    stride_group_sizes,
    stride_group_start_indices,
    stride_rhs_packed_ptrs,
    stride_rhs_scales_ptrs,
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
        start_idx = tl.load(
            group_start_indices_ptr + g * stride_group_start_indices
        )

        # Get pointers to this group's weights
        base_lhs_ptr = lhs_ptr + start_idx * stride_lhs_m
        rhs_packed_base = tl.load(
            rhs_packed_ptrs_ptr + group_idx * stride_rhs_packed_ptrs
        )
        rhs_packed_base = rhs_packed_base.to(tl.pointer_type(tl.uint8))
        rhs_scales_base = tl.load(
            rhs_scales_ptrs_ptr + group_idx * stride_rhs_scales_ptrs
        )
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
            offs_m = tile_m * GEMM_BLOCK_SIZE_M + tl.arange(
                0, GEMM_BLOCK_SIZE_M
            )
            offs_n = tile_n * GEMM_BLOCK_SIZE_N + tl.arange(
                0, GEMM_BLOCK_SIZE_N
            )

            # Initialize accumulator
            acc = tl.zeros(
                (GEMM_BLOCK_SIZE_M, GEMM_BLOCK_SIZE_N), dtype=tl.float32
            )

            # K dimension: process in blocks
            # Note: K is the unpacked dimension, K_packed = K // 2
            K_packed = K // 2

            for k_block in range(0, tl.cdiv(K, GEMM_BLOCK_SIZE_K)):
                k_start = k_block * GEMM_BLOCK_SIZE_K
                offs_k = k_start + tl.arange(0, GEMM_BLOCK_SIZE_K)

                # Load LHS tile [BLOCK_M, BLOCK_K]
                lhs_mask = (offs_m[:, None] < gm) & (offs_k[None, :] < K)
                lhs_ptrs = (
                    base_lhs_ptr
                    + offs_m[:, None] * stride_lhs_m
                    + offs_k[None, :] * stride_lhs_k
                )
                lhs_tile = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)

                # Load RHS packed tile [BLOCK_N, BLOCK_K//2]
                # Each byte contains 2 FP4 values, so we load half the K dimension
                offs_k_packed = k_start // 2 + tl.arange(
                    0, GEMM_BLOCK_SIZE_K // 2
                )
                rhs_mask = (offs_n[:, None] < N) & (
                    offs_k_packed[None, :] < K_packed
                )
                rhs_packed_ptrs = (
                    rhs_packed_base
                    + offs_n[:, None] * stride_rhs_packed_n
                    + offs_k_packed[None, :] * stride_rhs_packed_k
                )
                rhs_packed = tl.load(rhs_packed_ptrs, mask=rhs_mask, other=0)

                # Unpack FP4 values: [BLOCK_N, BLOCK_K//2] -> [BLOCK_N, BLOCK_K]
                # Low nibble = even indices, high nibble = odd indices
                # Use optimized v4_branchless decode (2 tl.where vs 16 for _fp4_lookup)
                idx_lo = (rhs_packed & 0x0F).to(tl.int32)
                idx_hi = ((rhs_packed >> 4) & 0x0F).to(tl.int32)

                # Decode FP4 values using fast branchless method
                val_lo = _fp4_decode_v4_branchless(
                    idx_lo
                )  # [BLOCK_N, BLOCK_K//2]
                val_hi = _fp4_decode_v4_branchless(
                    idx_hi
                )  # [BLOCK_N, BLOCK_K//2]

                # Load scales for this K block
                # Scale covers 32 consecutive K values, so scale_k_idx = k_start // 32
                scale_k_idx = k_start // 32
                n_scale_k = tl.cdiv(K, 32)

                # Each row in scales: [K//32]
                scale_ptrs = (
                    rhs_scales_base
                    + offs_n * stride_rhs_scales_n
                    + scale_k_idx * stride_rhs_scales_k
                )
                scale_mask = offs_n < N
                scales_uint8 = tl.load(scale_ptrs, mask=scale_mask, other=127)

                # Convert scale: exponent = scale - 127
                exponents = scales_uint8.to(tl.int32) - 127

                # Apply ldexp to both lo and hi values
                # Broadcast exponents: [BLOCK_N] -> [BLOCK_N, BLOCK_K//2]
                exponents_broadcast = exponents[:, None] + tl.zeros(
                    (1, GEMM_BLOCK_SIZE_K // 2), dtype=tl.int32
                )
                val_lo_scaled = _ldexp(val_lo, exponents_broadcast)
                val_hi_scaled = _ldexp(val_hi, exponents_broadcast)

                # Interleave lo/hi to get contiguous K dimension [BLOCK_N, BLOCK_K]
                # val_lo has K indices [0, 2, 4, ...], val_hi has [1, 3, 5, ...]
                # Use tl.join to create [BLOCK_N, BLOCK_K//2, 2] then reshape
                val_joined = tl.join(val_lo_scaled, val_hi_scaled)
                val_interleaved = tl.reshape(
                    val_joined, (GEMM_BLOCK_SIZE_N, GEMM_BLOCK_SIZE_K)
                )

                # Convert to BF16
                val_bf16 = val_interleaved.to(lhs_dtype)

                # Single full-size dot product (vs two half-size before)
                # [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
                acc += tl.dot(lhs_tile, val_bf16.T)

            # Store output tile
            out_offs_m = (
                start_idx
                + tile_m * GEMM_BLOCK_SIZE_M
                + tl.arange(0, GEMM_BLOCK_SIZE_M)
            )
            out_offs_n = tile_n * GEMM_BLOCK_SIZE_N + tl.arange(
                0, GEMM_BLOCK_SIZE_N
            )
            out_ptrs = (
                output_ptr
                + out_offs_m[:, None] * stride_output_m
                + out_offs_n[None, :] * stride_output_n
            )
            out_mask = (out_offs_m[:, None] < start_idx + gm) & (
                out_offs_n[None, :] < N
            )
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
    assert all(r.dtype == torch.uint8 for r in rhs_packed_list), (
        "packed weights must be uint8"
    )
    assert all(s.dtype == torch.uint8 for s in rhs_scales_list), (
        "scales must be uint8"
    )

    device = lhs.device
    M, K = lhs.shape
    N = rhs_packed_list[0].shape[0]  # [N, K//2]
    num_groups = len(group_sizes)

    # Create pointer arrays
    rhs_packed_ptrs = torch.tensor(
        [r.data_ptr() for r in rhs_packed_list],
        dtype=torch.int64,
        device=device,
    )
    rhs_scales_ptrs = torch.tensor(
        [s.data_ptr() for s in rhs_scales_list],
        dtype=torch.int64,
        device=device,
    )

    # Group metadata
    group_idx = torch.tensor(
        [idx for idx, _ in group_sizes], dtype=torch.int32, device=device
    )
    group_size = torch.tensor(
        [size for _, size in group_sizes], dtype=torch.int32, device=device
    )
    group_start_indices = torch.roll(torch.cumsum(group_size, dim=0), 1)
    group_start_indices[0] = 0

    # Output
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    # Launch kernel
    grid = lambda META: (
        triton.cdiv(16, META["GEMM_BLOCK_SIZE_M"])
        * triton.cdiv(N, META["GEMM_BLOCK_SIZE_N"]),
    )

    fused_mxfp4_grouped_gemm_kernel[grid](
        lhs,
        rhs_packed_ptrs,
        rhs_scales_ptrs,
        group_idx,
        group_size,
        group_start_indices,
        output,
        N,
        K,
        num_groups,
        lhs.stride(0),
        lhs.stride(1),
        rhs_packed_list[0].stride(0),
        rhs_packed_list[0].stride(1),
        rhs_scales_list[0].stride(0),
        rhs_scales_list[0].stride(1),
        output.stride(0),
        output.stride(1),
        group_idx.stride(0),
        group_size.stride(0),
        group_start_indices.stride(0),
        rhs_packed_ptrs.stride(0),
        rhs_scales_ptrs.stride(0),
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
    hidden_states: torch.Tensor,  # [batch*seq, hidden]
    topk_indices: torch.Tensor,  # [batch*seq, num_experts_per_tok]
    topk_weights: torch.Tensor,  # [batch*seq, num_experts_per_tok]
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
    token_indices = (
        torch.arange(num_tokens, device=device)
        .unsqueeze(1)
        .expand(-1, num_experts_per_tok)
        .reshape(-1)
    )
    k_indices = (
        torch.arange(num_experts_per_tok, device=device)
        .unsqueeze(0)
        .expand(num_tokens, -1)
        .reshape(-1)
    )

    # Sort by expert index to group tokens by expert
    sorted_expert_indices, sort_order = flat_indices.sort()

    # Reorder everything by expert
    sorted_token_indices = token_indices[sort_order]
    sorted_k_indices = k_indices[sort_order]
    sorted_weights = flat_weights[sort_order]

    # Gather hidden states in sorted order
    sorted_hidden = hidden_states[sorted_token_indices]

    # Compute expert offsets using bincount (requires int64)
    expert_counts = torch.bincount(
        sorted_expert_indices.to(torch.int64)
        if sorted_expert_indices.dtype != torch.int64
        else sorted_expert_indices,
        minlength=num_experts,
    )
    expert_offsets = torch.zeros(
        num_experts + 1, dtype=torch.int64, device=device
    )
    expert_offsets[1:] = expert_counts.cumsum(0)

    return (
        sorted_hidden,
        expert_offsets,
        sorted_token_indices,
        sorted_k_indices,
        sorted_weights,
    )


# =============================================================================
# True Grouped MXFP4 GEMM with 3D Layout (DeepSeek-V3 Pattern)
# =============================================================================


@triton.jit
def fused_mxfp4_grouped_gemm_kernel_3d(
    # Input [E, M_max, K] BF16
    lhs_ptr,
    # Weight pointer arrays [num_experts] int64
    rhs_ptrs_ptr,  # -> [N, K//2] uint8 packed FP4
    rhs_scale_ptrs_ptr,  # -> [N, K//32] uint8
    # Per-expert token counts [num_experts] int32
    expert_tokens_ptr,
    # Output [E, M_max, N] BF16
    output_ptr,
    # Dimensions
    M_max,
    N,
    K,
    # Strides for lhs [E, M_max, K]
    stride_lhs_e,
    stride_lhs_m,
    stride_lhs_k,
    # Strides for rhs weights [N, K//2]
    stride_rhs_n,
    stride_rhs_k_packed,
    # Strides for scales [N, K//32]
    stride_scale_n,
    stride_scale_k,
    # Strides for output [E, M_max, N]
    stride_out_e,
    stride_out_m,
    stride_out_n,
    # Stride for pointer arrays
    stride_ptrs,
    # Block sizes
    BLOCK_M: tl.constexpr,  # 64
    BLOCK_N: tl.constexpr,  # 64
    BLOCK_K: tl.constexpr,  # Ignored - kernel uses 32-wide K blocks internally
):
    """True grouped MXFP4 GEMM following DeepSeek-V3 pattern.

    Grid: (num_experts, cdiv(N, BLOCK_N))
    - axis 0: expert index
    - axis 1: N-block index

    Each thread block handles one (expert, N-block) pair and loops over:
    - M-blocks (tokens for that expert)
    - K-blocks (32-wide, matching MXFP4 scale granularity)

    The kernel processes 32 K values per iteration, which naturally aligns with
    MXFP4 scale blocks (one scale per 32 FP4 values). This avoids the complexity
    and bugs of combining multiple scale blocks per K-tile.
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
    rhs_base_ptr = tl.load(rhs_ptrs_ptr + expert_idx * stride_ptrs).to(
        tl.pointer_type(tl.uint8)
    )
    scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + expert_idx * stride_ptrs).to(
        tl.pointer_type(tl.uint8)
    )

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

        # K-loop: process 32 K values per iteration (one scale block)
        # MXFP4 scale granularity is 32, so this naturally aligns with scales
        # Using 32-wide K blocks is simpler and avoids the interleaving bug
        # that occurred with 64-wide blocks + tl.join
        num_k_blocks = K // 32  # Always use 32-wide K blocks

        for k_block in range(num_k_blocks):
            k_start = k_block * 32  # K starting position

            # ===== Load packed FP4 weights [BLOCK_N, 16] for 32 K values =====
            k_packed = k_start // 2  # Packed byte index (2 FP4 per byte)
            offs_k_packed = tl.arange(
                0, 16
            )  # 32 values / 2 per byte = 16 bytes
            rhs_ptrs = (
                rhs_base_ptr
                + offs_n[:, None] * stride_rhs_n
                + (k_packed + offs_k_packed[None, :]) * stride_rhs_k_packed
            )
            rhs_packed = tl.load(rhs_ptrs, mask=n_mask[:, None], other=0)

            # ===== Unpack FP4: extract lo/hi nibbles =====
            idx_lo = (rhs_packed & 0x0F).to(tl.int32)  # [BLOCK_N, 16]
            idx_hi = ((rhs_packed >> 4) & 0x0F).to(tl.int32)  # [BLOCK_N, 16]
            val_lo = _fp4_decode_v4_branchless(idx_lo)
            val_hi = _fp4_decode_v4_branchless(idx_hi)

            # ===== Load scale for this K block (one scale per 32 K values) =====
            scale_idx = k_block  # Direct mapping: k_block -> scale index
            scale_ptrs = (
                scale_base_ptr
                + offs_n * stride_scale_n
                + scale_idx * stride_scale_k
            )
            scales = (
                tl.load(scale_ptrs, mask=n_mask, other=127).to(tl.int32) - 127
            )

            # ===== Apply ldexp: value * 2^scale =====
            exp_broadcast = scales[:, None] + tl.zeros((1, 16), dtype=tl.int32)
            val_lo_scaled = _ldexp(val_lo, exp_broadcast)
            val_hi_scaled = _ldexp(val_hi, exp_broadcast)

            # ===== Interleave lo/hi nibbles: [BLOCK_N, 16] x 2 -> [BLOCK_N, 32] =====
            # tl.join stacks along innermost dim: [N,16] + [N,16] -> [N,16,2]
            # tl.reshape flattens to [N,32] with order [lo0,hi0,lo1,hi1,...] = [K0,K1,K2,...]
            # This is CORRECT for lo/hi nibble interleaving within a single scale block
            val_joined = tl.join(val_lo_scaled, val_hi_scaled)
            val_interleaved = tl.reshape(
                val_joined, (BLOCK_N, 32)
            )  # [BLOCK_N, 32]

            # ===== Load LHS tile [BLOCK_M, 32] =====
            offs_k = tl.arange(0, 32)
            lhs_ptrs = (
                cur_lhs_ptr
                + offs_m[:, None] * stride_lhs_m
                + (k_start + offs_k[None, :]) * stride_lhs_k
            )
            lhs_tile = tl.load(lhs_ptrs, mask=m_mask[:, None], other=0.0)

            # ===== GEMM: [BLOCK_M, 32] @ [32, BLOCK_N] -> [BLOCK_M, BLOCK_N] =====
            acc += tl.dot(
                lhs_tile.to(tl.bfloat16),
                tl.trans(val_interleaved.to(tl.bfloat16)),
                allow_tf32=False,
            ).to(tl.float32)

        # Store output [BLOCK_M, BLOCK_N]
        out_ptrs = (
            cur_out_ptr
            + offs_m[:, None] * stride_out_m
            + offs_n[None, :] * stride_out_n
        )
        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def reshape_to_3d_expert_layout(
    sorted_hidden: torch.Tensor,  # [total_tokens, hidden]
    expert_counts: torch.Tensor,  # [num_experts] int32/int64
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
    # Single CPU-GPU sync: read all expert counts at once
    counts_list = expert_counts.tolist()
    max_tokens = max(counts_list) if counts_list else 0
    if max_tokens == 0:
        # No tokens routed to any expert (edge case)
        hidden_size = sorted_hidden.shape[-1]
        return torch.zeros(
            num_experts,
            1,
            hidden_size,
            dtype=sorted_hidden.dtype,
            device=sorted_hidden.device,
        ), 1

    hidden_size = sorted_hidden.shape[-1]
    device = sorted_hidden.device
    dtype = sorted_hidden.dtype

    # Allocate 3D tensor (padded with zeros for empty slots)
    hidden_3d = torch.zeros(
        num_experts, max_tokens, hidden_size, dtype=dtype, device=device
    )

    # Copy tokens to their expert slots
    # This can be optimized with a Triton scatter kernel later
    offset = 0
    for e in range(num_experts):
        count = counts_list[e]
        if count > 0:
            hidden_3d[e, :count] = sorted_hidden[offset : offset + count]
            offset += count

    return hidden_3d, max_tokens


def gather_from_3d_expert_layout(
    output_3d: torch.Tensor,  # [E, M_max, hidden]
    expert_counts: torch.Tensor,  # [num_experts] int32/int64
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

    sorted_output = torch.zeros(
        total_tokens, hidden_size, dtype=dtype, device=device
    )

    # Single CPU-GPU sync: read all expert counts at once
    counts_list = expert_counts.tolist()
    offset = 0
    for e in range(num_experts):
        count = counts_list[e]
        if count > 0:
            sorted_output[offset : offset + count] = output_3d[e, :count]
            offset += count

    return sorted_output


def setup_expert_weight_pointers(
    weight_list: List[
        torch.Tensor
    ],  # [num_experts] of [N, K//2] uint8 or similar
    scale_list: List[torch.Tensor],  # [num_experts] of [N, K//32] uint8
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
        [w.data_ptr() for w in weight_list], dtype=torch.int64, device=device
    )
    scale_ptrs = torch.tensor(
        [s.data_ptr() for s in scale_list], dtype=torch.int64, device=device
    )

    return weight_ptrs, scale_ptrs


def grouped_mxfp4_gemm_3d(
    hidden_3d: torch.Tensor,  # [E, M_max, K] BF16
    weight_ptrs: torch.Tensor,  # [num_experts] int64
    scale_ptrs: torch.Tensor,  # [num_experts] int64
    expert_counts: torch.Tensor,  # [num_experts] int32
    N: int,  # Output dimension
    weight_ref: torch.Tensor,  # Reference weight for strides [N, K//2]
    scale_ref: torch.Tensor,  # Reference scale for strides [N, K//32]
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,  # Fixed at 32 to match MXFP4 scale granularity (ignored by kernel)
) -> torch.Tensor:
    """Launch grouped MXFP4 GEMM kernel with 3D layout.

    Note: The kernel internally uses 32-wide K blocks to match MXFP4 scale granularity.
    The BLOCK_K parameter is kept for API compatibility but is ignored.

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
    output_3d = torch.empty(
        num_experts, M_max, N, dtype=torch.bfloat16, device=device
    )

    # Grid: (num_experts, cdiv(N, BLOCK_N))
    grid = (num_experts, triton.cdiv(N, BLOCK_N))

    fused_mxfp4_grouped_gemm_kernel_3d[grid](
        hidden_3d,
        weight_ptrs,
        scale_ptrs,
        expert_counts,
        output_3d,
        M_max,
        N,
        K,
        hidden_3d.stride(0),
        hidden_3d.stride(1),
        hidden_3d.stride(2),
        weight_ref.stride(0),
        weight_ref.stride(1),
        scale_ref.stride(0),
        scale_ref.stride(1),
        output_3d.stride(0),
        output_3d.stride(1),
        output_3d.stride(2),
        1,  # stride_ptrs (contiguous pointer array)
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
    )

    return output_3d


def grouped_mxfp4_gemm_3d_tunable(
    hidden_3d: torch.Tensor,  # [E, M_max, K] BF16
    weight_ptrs: torch.Tensor,  # [num_experts] int64
    scale_ptrs: torch.Tensor,  # [num_experts] int64
    expert_counts: torch.Tensor,  # [num_experts] int32
    N: int,  # Output dimension
    weight_ref: torch.Tensor,  # Reference weight for strides [N, K//2]
    scale_ref: torch.Tensor,  # Reference scale for strides [N, K//32]
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,  # Fixed at 32 to match MXFP4 scale granularity (ignored by kernel)
    num_warps: int = 8,
    num_stages: int = 1,
) -> torch.Tensor:
    """Tunable variant of grouped MXFP4 GEMM for hyperparameter search.

    Same as grouped_mxfp4_gemm_3d but with configurable num_warps and num_stages.
    Note: BLOCK_K is ignored - the kernel always uses 32-wide K blocks.

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
        BLOCK_K: Ignored - kernel uses 32-wide K blocks to match MXFP4 scales
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
    output_3d = torch.empty(
        num_experts, M_max, N, dtype=torch.bfloat16, device=device
    )

    # Grid: (num_experts, cdiv(N, BLOCK_N))
    grid = (num_experts, triton.cdiv(N, BLOCK_N))

    fused_mxfp4_grouped_gemm_kernel_3d[grid](
        hidden_3d,
        weight_ptrs,
        scale_ptrs,
        expert_counts,
        output_3d,
        M_max,
        N,
        K,
        hidden_3d.stride(0),
        hidden_3d.stride(1),
        hidden_3d.stride(2),
        weight_ref.stride(0),
        weight_ref.stride(1),
        scale_ref.stride(0),
        scale_ref.stride(1),
        output_3d.stride(0),
        output_3d.stride(1),
        output_3d.stride(2),
        1,  # stride_ptrs (contiguous pointer array)
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return output_3d


def grouped_mxfp4_moe_forward_3d(
    hidden_states: torch.Tensor,  # [batch*seq, hidden]
    topk_indices: torch.Tensor,  # [batch*seq, num_experts_per_tok]
    topk_weights: torch.Tensor,  # [batch*seq, num_experts_per_tok]
    # Pre-computed pointer arrays (from setup_expert_weight_pointers)
    gate_ptrs: torch.Tensor,  # [num_experts] int64
    gate_scale_ptrs: torch.Tensor,
    up_ptrs: torch.Tensor,
    up_scale_ptrs: torch.Tensor,
    down_ptrs: torch.Tensor,
    down_scale_ptrs: torch.Tensor,
    # Reference weights for strides (any expert's weight works)
    gate_weight_ref: torch.Tensor,  # [N_inter, hidden//2]
    gate_scale_ref: torch.Tensor,  # [N_inter, hidden//32]
    up_weight_ref: torch.Tensor,
    up_scale_ref: torch.Tensor,
    down_weight_ref: torch.Tensor,  # [hidden, N_inter//2]
    down_scale_ref: torch.Tensor,  # [hidden, N_inter//32]
    # Biases (optional, stacked as [num_experts, N])
    gate_biases: torch.Tensor = None,  # [num_experts, N_inter] or None
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
    (
        sorted_hidden,
        expert_offsets,
        original_indices,
        original_k,
        routing_weights,
    ) = moe_token_dispatch(
        hidden_states, topk_indices, topk_weights, num_experts
    )

    total_tokens_routed = sorted_hidden.shape[0]

    # Compute per-expert token counts
    expert_counts = (expert_offsets[1:] - expert_offsets[:-1]).to(torch.int32)

    # Step 2: Reshape to 3D layout [E, M_max, K]
    hidden_3d, M_max = reshape_to_3d_expert_layout(
        sorted_hidden, expert_counts, num_experts
    )

    # Step 3: Gate projection (SINGLE kernel for all experts)
    gate_out_3d = grouped_mxfp4_gemm_3d(
        hidden_3d,
        gate_ptrs,
        gate_scale_ptrs,
        expert_counts,
        N_intermediate,
        gate_weight_ref,
        gate_scale_ref,
    )

    # Step 4: Up projection (SINGLE kernel for all experts)
    up_out_3d = grouped_mxfp4_gemm_3d(
        hidden_3d,
        up_ptrs,
        up_scale_ptrs,
        expert_counts,
        N_intermediate,
        up_weight_ref,
        up_scale_ref,
    )

    # Add biases if present (broadcasted over [E, M_max, N])
    if gate_biases is not None:
        gate_out_3d = gate_out_3d + gate_biases.unsqueeze(1)
    if up_biases is not None:
        up_out_3d = up_out_3d + up_biases.unsqueeze(1)

    # Step 5: SwiGLU activation (in-place on 3D tensors)
    gate_clamped = gate_out_3d.clamp(max=swiglu_limit)
    up_clamped = up_out_3d.clamp(min=-swiglu_limit, max=swiglu_limit)
    intermediate_3d = (
        gate_clamped
        * torch.sigmoid(swiglu_alpha * gate_clamped)
        * (up_clamped + 1)
    )

    # Step 6: Down projection (SINGLE kernel for all experts)
    output_3d = grouped_mxfp4_gemm_3d(
        intermediate_3d,
        down_ptrs,
        down_scale_ptrs,
        expert_counts,
        hidden_size,
        down_weight_ref,
        down_scale_ref,
    )

    if down_biases is not None:
        output_3d = output_3d + down_biases.unsqueeze(1)

    # Step 7: Gather back from 3D to sorted 1D
    sorted_output = gather_from_3d_expert_layout(
        output_3d, expert_counts, total_tokens_routed
    )

    # Step 8: Scatter back to original order with routing weights
    output = torch.zeros(
        num_tokens, hidden_size, dtype=hidden_states.dtype, device=device
    )
    weighted_output = sorted_output * routing_weights.unsqueeze(-1)
    output.scatter_add_(
        0,
        original_indices.unsqueeze(-1).expand_as(weighted_output),
        weighted_output,
    )

    return output


# =============================================================================
# Grouped MXFP4 MoE Forward with CUDA Routing Kernels
# =============================================================================


def grouped_mxfp4_moe_forward_cuda_routing(
    hidden_states: torch.Tensor,  # [batch*seq, hidden] BF16
    topk_indices: torch.Tensor,  # [batch*seq, num_experts_per_tok] int32
    topk_weights: torch.Tensor,  # [batch*seq, num_experts_per_tok] FP32
    # Pre-computed pointer arrays (from setup_expert_weight_pointers)
    gate_ptrs: torch.Tensor,
    gate_scale_ptrs: torch.Tensor,
    up_ptrs: torch.Tensor,
    up_scale_ptrs: torch.Tensor,
    down_ptrs: torch.Tensor,
    down_scale_ptrs: torch.Tensor,
    # Reference weights for strides
    gate_weight_ref: torch.Tensor,
    gate_scale_ref: torch.Tensor,
    up_weight_ref: torch.Tensor,
    up_scale_ref: torch.Tensor,
    down_weight_ref: torch.Tensor,
    down_scale_ref: torch.Tensor,
    # Optional biases
    gate_biases: torch.Tensor = None,
    up_biases: torch.Tensor = None,
    down_biases: torch.Tensor = None,
    num_experts: int = 128,
    expert_start: int = 0,
    num_local_experts: int = 128,
    swiglu_alpha: float = 1.702,
    swiglu_limit: float = 7.0,
    activation: str = "openai",
) -> torch.Tensor:
    """Grouped MXFP4 MoE forward with CUDA routing (dispatch + reduce).

    Replaces PyTorch sort/bincount dispatch and scatter_add_ reduce with
    fused CUDA kernels. The 3D GEMM stages are unchanged.

    Key differences from grouped_mxfp4_moe_forward_3d:
    - CUDA dispatch: count + prefix_sum + gather (3 sub-kernels vs PyTorch sort)
    - CUDA reduce: weighted scatter-add via topk_pos (vs scatter_add_ with index expansion)
    - No int64 conversions: all routing indices stay int32 throughout

    Args:
        hidden_states: Input [batch*seq, hidden] in BF16
        topk_indices: Expert indices [batch*seq, K] in int32 (from gate_topk_softmax_cuda)
        topk_weights: Routing weights [batch*seq, K] in FP32
        expert_start: First local expert index (for EP)
        num_local_experts: Number of local experts
        Other args: Same as grouped_mxfp4_moe_forward_3d
    """
    from batchgen.moe.routing import (
        dispatch_count_gather_cuda,
        reduce_weighted_scatter_cuda,
    )

    num_tokens, hidden_size = hidden_states.shape
    K = topk_indices.shape[1]
    device = hidden_states.device
    N_intermediate = gate_weight_ref.shape[0]

    # Step 1: CUDA dispatch (replaces moe_token_dispatch)
    dispatched_x, expert_counts, expert_offsets, topk_pos = (
        dispatch_count_gather_cuda(
            hidden_states,
            topk_indices,
            expert_start,
            num_local_experts,
        )
    )

    # Trim to actual dispatched tokens (sync consolidated with reshape_to_3d below)
    total_dispatched = int(expert_offsets[num_local_experts])
    dispatched_x = dispatched_x[:total_dispatched]

    # Step 2: Reshape to 3D layout for GEMM
    hidden_3d, M_max = reshape_to_3d_expert_layout(
        dispatched_x, expert_counts, num_local_experts
    )

    # Step 3: Gate projection
    gate_out_3d = grouped_mxfp4_gemm_3d(
        hidden_3d,
        gate_ptrs,
        gate_scale_ptrs,
        expert_counts,
        N_intermediate,
        gate_weight_ref,
        gate_scale_ref,
    )

    # Step 4: Up projection
    up_out_3d = grouped_mxfp4_gemm_3d(
        hidden_3d,
        up_ptrs,
        up_scale_ptrs,
        expert_counts,
        N_intermediate,
        up_weight_ref,
        up_scale_ref,
    )

    # Add biases if present
    if gate_biases is not None:
        gate_out_3d = gate_out_3d + gate_biases[:num_local_experts].unsqueeze(1)
    if up_biases is not None:
        up_out_3d = up_out_3d + up_biases[:num_local_experts].unsqueeze(1)

    gate_clamped = gate_out_3d.clamp(max=swiglu_limit)
    up_clamped = up_out_3d.clamp(min=-swiglu_limit, max=swiglu_limit)
    if activation == "v4_silu":
        intermediate_3d = (
            torch.nn.functional.silu(gate_clamped.float()) * up_clamped.float()
        ).to(hidden_states.dtype)
    else:
        intermediate_3d = (
            gate_clamped
            * torch.sigmoid(swiglu_alpha * gate_clamped)
            * (up_clamped + 1)
        )

    # Step 6: Down projection
    output_3d = grouped_mxfp4_gemm_3d(
        intermediate_3d,
        down_ptrs,
        down_scale_ptrs,
        expert_counts,
        hidden_size,
        down_weight_ref,
        down_scale_ref,
    )

    if down_biases is not None:
        output_3d = output_3d + down_biases[:num_local_experts].unsqueeze(1)

    # Step 7: Gather from 3D back to flat sorted layout
    sorted_output = gather_from_3d_expert_layout(
        output_3d, expert_counts, total_dispatched
    )

    # Step 8: CUDA reduce (replaces scatter_add_)
    output = reduce_weighted_scatter_cuda(
        sorted_output,
        topk_pos,
        topk_weights,
        num_tokens,
        hidden_size,
        K,
    )

    return output


# =============================================================================
# Original Per-Expert Loop Implementation (for comparison/fallback)
# =============================================================================


def grouped_mxfp4_moe_forward(
    hidden_states: torch.Tensor,  # [batch*seq, hidden]
    topk_indices: torch.Tensor,  # [batch*seq, num_experts_per_tok]
    topk_weights: torch.Tensor,  # [batch*seq, num_experts_per_tok]
    gate_weights: List[torch.Tensor],  # [num_experts] of [N, K//2] uint8
    gate_scales: List[torch.Tensor],  # [num_experts] of [N, K//32] uint8
    gate_biases: List[torch.Tensor],  # [num_experts] of [N] BF16 (or None)
    up_weights: List[torch.Tensor],  # [num_experts] of [N, K//2] uint8
    up_scales: List[torch.Tensor],  # [num_experts] of [N, K//32] uint8
    up_biases: List[torch.Tensor],  # [num_experts] of [N] BF16 (or None)
    down_weights: List[torch.Tensor],  # [num_experts] of [hidden, N//2] uint8
    down_scales: List[torch.Tensor],  # [num_experts] of [hidden, N//32] uint8
    down_biases: List[torch.Tensor],  # [num_experts] of [hidden] BF16 (or None)
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
    (
        sorted_hidden,
        expert_offsets,
        original_indices,
        original_k,
        routing_weights,
    ) = moe_token_dispatch(
        hidden_states, topk_indices, topk_weights, num_experts
    )

    # Step 2: Process each expert's tokens in batch
    # Allocate output for all sorted tokens
    sorted_output = torch.zeros_like(sorted_hidden)

    # Single CPU-GPU sync: read all offsets at once
    offsets_list = expert_offsets[: num_experts + 1].tolist()

    for expert_idx in range(num_experts):
        start = offsets_list[expert_idx]
        end = offsets_list[expert_idx + 1]

        if start == end:
            continue  # No tokens for this expert

        expert_input = sorted_hidden[
            start:end
        ]  # [num_tokens_for_expert, hidden]

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
        gate_out = fused_mxfp4_single_gemm(
            expert_input, gate_packed, gate_scale, gate_bias
        )

        # Stage 1b: Up projection
        up_out = fused_mxfp4_single_gemm(
            expert_input, up_packed, up_scale, up_bias
        )

        # Stage 1c: SwiGLU activation
        gate_clamped = gate_out.clamp(max=swiglu_limit)
        up_clamped = up_out.clamp(min=-swiglu_limit, max=swiglu_limit)
        glu = gate_clamped * torch.sigmoid(swiglu_alpha * gate_clamped)
        intermediate = glu * (up_clamped + 1)

        # Stage 2: Down projection
        expert_output = fused_mxfp4_single_gemm(
            intermediate, down_packed, down_scale, down_bias
        )

        # Store in sorted output
        sorted_output[start:end] = expert_output

    # Step 3: Combine results back to original order with routing weights
    # Each original token position accumulates weighted outputs from its top-k experts
    output = torch.zeros(
        num_tokens, hidden, dtype=hidden_states.dtype, device=device
    )

    # Apply routing weights and scatter back
    weighted_output = sorted_output * routing_weights.unsqueeze(-1)
    output.scatter_add_(
        0,
        original_indices.unsqueeze(-1).expand_as(weighted_output),
        weighted_output,
    )

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
    output = fused_mxfp4_single_gemm(
        intermediate, down_packed, down_scales, down_bias
    )

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
    M,
    N,
    K,
    # Strides
    stride_lhs_m,
    stride_lhs_k,
    stride_rhs_n,
    stride_rhs_k,
    stride_scale_n,
    stride_scale_k,
    stride_out_m,
    stride_out_n,
    # Config
    HAS_BIAS: tl.constexpr,
    # Block sizes (same as grouped kernel)
    BLOCK_M: tl.constexpr,  # 64
    BLOCK_N: tl.constexpr,  # 64
    BLOCK_K: tl.constexpr,  # 64 (processes 2 scale blocks per K-tile)
):
    """Optimized single-expert MXFP4 GEMM with same tiling as grouped kernel.

    Grid: (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N))
    - axis 0: M-block index
    - axis 1: N-block index

    Uses identical MXFP4 dequantization and accumulation pattern as the grouped
    kernel for consistent performance characteristics.

    BLOCK_K=64 optimization: Each K-tile spans 2 MXFP4 scale blocks (32 each).
    """
    m_pid = tl.program_id(axis=0)
    n_pid = tl.program_id(axis=1)

    offs_m = m_pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop with BLOCK_K=64 (processes 2 scale blocks per iteration)
    num_k_blocks = K // BLOCK_K

    for k_block in range(num_k_blocks):
        k_start = k_block * BLOCK_K

        # With BLOCK_K=64, we process 2 scale blocks:
        # - First 32 K values use scale_k_lo
        # - Second 32 K values use scale_k_hi
        scale_k_lo = k_block * 2  # Scale block index for K[0:32]
        scale_k_hi = k_block * 2 + 1  # Scale block index for K[32:64]

        # ===== FIRST HALF: K positions [k_start, k_start+32) =====
        k_packed_lo = k_start // 2
        offs_k_packed_lo = tl.arange(0, 16)
        rhs_ptrs_lo = (
            rhs_ptr
            + offs_n[:, None] * stride_rhs_n
            + (k_packed_lo + offs_k_packed_lo[None, :]) * stride_rhs_k
        )
        rhs_packed_lo = tl.load(rhs_ptrs_lo, mask=n_mask[:, None], other=0)

        idx_lo_lo = (rhs_packed_lo & 0x0F).to(tl.int32)
        idx_hi_lo = ((rhs_packed_lo >> 4) & 0x0F).to(tl.int32)
        val_lo_lo = _fp4_decode_v4_branchless(idx_lo_lo)
        val_hi_lo = _fp4_decode_v4_branchless(idx_hi_lo)

        scale_ptrs_lo = (
            scale_ptr + offs_n * stride_scale_n + scale_k_lo * stride_scale_k
        )
        scales_lo = (
            tl.load(scale_ptrs_lo, mask=n_mask, other=127).to(tl.int32) - 127
        )

        exp_broadcast_lo = scales_lo[:, None] + tl.zeros(
            (1, 16), dtype=tl.int32
        )
        val_lo_lo_scaled = _ldexp(val_lo_lo, exp_broadcast_lo)
        val_hi_lo_scaled = _ldexp(val_hi_lo, exp_broadcast_lo)

        val_joined_lo = tl.join(val_lo_lo_scaled, val_hi_lo_scaled)
        val_interleaved_lo = tl.reshape(val_joined_lo, (BLOCK_N, 32))

        # ===== SECOND HALF: K positions [k_start+32, k_start+64) =====
        k_packed_hi = (k_start + 32) // 2
        offs_k_packed_hi = tl.arange(0, 16)
        rhs_ptrs_hi = (
            rhs_ptr
            + offs_n[:, None] * stride_rhs_n
            + (k_packed_hi + offs_k_packed_hi[None, :]) * stride_rhs_k
        )
        rhs_packed_hi = tl.load(rhs_ptrs_hi, mask=n_mask[:, None], other=0)

        idx_lo_hi = (rhs_packed_hi & 0x0F).to(tl.int32)
        idx_hi_hi = ((rhs_packed_hi >> 4) & 0x0F).to(tl.int32)
        val_lo_hi = _fp4_decode_v4_branchless(idx_lo_hi)
        val_hi_hi = _fp4_decode_v4_branchless(idx_hi_hi)

        scale_ptrs_hi = (
            scale_ptr + offs_n * stride_scale_n + scale_k_hi * stride_scale_k
        )
        scales_hi = (
            tl.load(scale_ptrs_hi, mask=n_mask, other=127).to(tl.int32) - 127
        )

        exp_broadcast_hi = scales_hi[:, None] + tl.zeros(
            (1, 16), dtype=tl.int32
        )
        val_lo_hi_scaled = _ldexp(val_lo_hi, exp_broadcast_hi)
        val_hi_hi_scaled = _ldexp(val_hi_hi, exp_broadcast_hi)

        val_joined_hi = tl.join(val_lo_hi_scaled, val_hi_hi_scaled)
        val_interleaved_hi = tl.reshape(val_joined_hi, (BLOCK_N, 32))

        # ===== COMBINE: Concatenate both halves [BLOCK_N, 64] =====
        val_full = tl.join(val_interleaved_lo, val_interleaved_hi)
        val_interleaved = tl.reshape(val_full, (BLOCK_N, BLOCK_K))

        # Load LHS contiguously [BLOCK_M, 64]
        offs_k = tl.arange(0, BLOCK_K)
        lhs_ptrs = (
            lhs_ptr
            + offs_m[:, None] * stride_lhs_m
            + (k_start + offs_k[None, :]) * stride_lhs_k
        )
        lhs_tile = tl.load(
            lhs_ptrs, mask=m_mask[:, None], other=0.0
        )  # [BLOCK_M, BLOCK_K]

        # Single full-size dot product
        acc += tl.dot(
            lhs_tile.to(tl.bfloat16),
            tl.trans(val_interleaved.to(tl.bfloat16)),
            allow_tf32=False,
        ).to(tl.float32)

    # Add bias if present
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias[None, :]

    # Store output [BLOCK_M, BLOCK_N]
    out_ptrs = (
        output_ptr
        + offs_m[:, None] * stride_out_m
        + offs_n[None, :] * stride_out_n
    )
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
    BLOCK_N=64, BLOCK_K=64) for consistent performance.

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
    intermediate = (
        gate_clamped
        * torch.sigmoid(swiglu_alpha * gate_clamped)
        * (up_clamped + 1)
    )

    # Down projection
    output = fused_mxfp4_single_gemm(
        intermediate, down_packed, down_scales, down_bias
    )

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
        output = fused_mxfp4_single_gemm(
            x_2d, weight_packed, weight_scales, bias
        )
    else:
        # Fallback: unfused path (materializes full BF16 weights)
        from batchgen.quantization.mxfp4 import mxfp4_dequantize

        weight_bf16 = mxfp4_dequantize(
            weight_packed, weight_scales, dtype=torch.bfloat16
        )
        output = torch.mm(x_2d, weight_bf16.T)
        if bias is not None:
            output = output + bias

    # Reshape to original batch dimensions
    output = output.view(*original_shape[:-1], -1)

    return output
