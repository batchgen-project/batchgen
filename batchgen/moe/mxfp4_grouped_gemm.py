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

import torch
import triton
import triton.language as tl
from typing import List, Tuple


# MXFP4 configuration
MXFP4_BLOCK_SIZE = 32  # FP4 values per scale
MXFP4_PACKED_BLOCK_SIZE = 16  # Bytes per scale (32 values / 2 per byte)


@triton.jit
def _fp4_lookup(idx):
    """Lookup FP4 value from 4-bit index.

    FP4 table:
    0: 0.0,  1: 0.5,  2: 1.0,  3: 1.5,  4: 2.0,  5: 3.0,  6: 4.0,  7: 6.0
    8: -0.0, 9: -0.5, 10: -1.0, 11: -1.5, 12: -2.0, 13: -3.0, 14: -4.0, 15: -6.0
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
def _ldexp(mantissa, exponent):
    """Compute mantissa * 2^exponent."""
    # Clamp exponent to valid range
    exp_clamped = tl.minimum(tl.maximum(exponent, -126), 127)
    # Create 2^exponent as float
    exp_bits = (exp_clamped + 127).to(tl.int32) << 23
    power_of_2 = exp_bits.to(tl.float32, bitcast=True)
    return mantissa * power_of_2


# =============================================================================
# Single-Expert Fused MXFP4 GEMM Kernel
# =============================================================================

@triton.jit
def fused_mxfp4_single_gemm_kernel(
    lhs_ptr,                    # BF16 activations [M, K]
    rhs_packed_ptr,             # Packed FP4 weights [N, K//2] in uint8
    rhs_scales_ptr,             # Scales [N, K//32] in uint8
    output_ptr,                 # Output [M, N] in BF16
    M, N, K,
    stride_lhs_m, stride_lhs_k,
    stride_rhs_packed_n, stride_rhs_packed_k,
    stride_rhs_scales_n, stride_rhs_scales_k,
    stride_output_m, stride_output_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused MXFP4 dequantization and GEMM for single expert.

    Computes: output = lhs @ rhs.T where rhs is MXFP4 quantized.

    This is a simplified version without grouping overhead for single-expert inference.

    Args:
        lhs_ptr: Input activations [M, K] in BF16
        rhs_packed_ptr: Packed weights [N, K//2] in uint8 (2 FP4 values per byte)
        rhs_scales_ptr: Scales [N, K//32] in uint8
        output_ptr: Output tensor [M, N] in BF16
    """
    # 2D tiling: program_id(0) for M tiles, program_id(1) for N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Tile offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K dimension sizes
    K_packed = K // 2

    # Iterate over K dimension in blocks
    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k_block * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load LHS tile [BLOCK_M, BLOCK_K]
        lhs_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        lhs_ptrs = lhs_ptr + offs_m[:, None] * stride_lhs_m + offs_k[None, :] * stride_lhs_k
        lhs_tile = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)

        # Load RHS packed tile [BLOCK_N, BLOCK_K//2]
        offs_k_packed = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        rhs_mask = (offs_n[:, None] < N) & (offs_k_packed[None, :] < K_packed)
        rhs_packed_ptrs = rhs_packed_ptr + offs_n[:, None] * stride_rhs_packed_n + offs_k_packed[None, :] * stride_rhs_packed_k
        rhs_packed = tl.load(rhs_packed_ptrs, mask=rhs_mask, other=0)

        # Unpack FP4 nibbles
        idx_lo = (rhs_packed & 0x0F).to(tl.int32)
        idx_hi = ((rhs_packed >> 4) & 0x0F).to(tl.int32)

        # Lookup FP4 values
        val_lo = _fp4_lookup(idx_lo)
        val_hi = _fp4_lookup(idx_hi)

        # Load scales for this K block
        # Scale covers 32 consecutive K values, but we need scales for multiple K blocks
        # For BLOCK_K elements starting at k_start, we need scale at k_start // 32
        scale_k_idx = k_start // 32
        scale_ptrs = rhs_scales_ptr + offs_n * stride_rhs_scales_n + scale_k_idx * stride_rhs_scales_k
        scale_mask = offs_n < N
        scales_uint8 = tl.load(scale_ptrs, mask=scale_mask, other=127)

        # Convert scale: exponent = scale - 127
        exponents = scales_uint8.to(tl.int32) - 127

        # Broadcast exponents for ldexp: [BLOCK_N] -> [BLOCK_N, BLOCK_K//2]
        exponents_broadcast = exponents[:, None] + tl.zeros((1, BLOCK_K // 2), dtype=tl.int32)
        val_lo_scaled = _ldexp(val_lo, exponents_broadcast)
        val_hi_scaled = _ldexp(val_hi, exponents_broadcast)

        # Convert to BF16 for dot product
        val_lo_bf16 = val_lo_scaled.to(tl.bfloat16)
        val_hi_bf16 = val_hi_scaled.to(tl.bfloat16)

        # Split LHS into even/odd K indices
        lhs_even = lhs_tile[:, 0::2]  # [BLOCK_M, BLOCK_K//2]
        lhs_odd = lhs_tile[:, 1::2]   # [BLOCK_M, BLOCK_K//2]

        # Compute: lhs_even @ val_lo.T + lhs_odd @ val_hi.T
        # This handles the interleaved packing where low nibble = even K, high nibble = odd K
        acc += tl.dot(lhs_even, val_lo_bf16.T)
        acc += tl.dot(lhs_odd, val_hi_bf16.T)

    # Store output tile
    out_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


@torch.inference_mode()
def fused_mxfp4_single_gemm(
    lhs: torch.Tensor,
    rhs_packed: torch.Tensor,
    rhs_scales: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """Fused MXFP4 dequantization and GEMM for single expert.

    Computes: output = lhs @ dequant(rhs).T + bias

    Args:
        lhs: Input activations [M, K] in BF16
        rhs_packed: Packed FP4 weights [N, K//2] in uint8
        rhs_scales: Scales [N, K//32] in uint8
        bias: Optional bias [N] in BF16

    Returns:
        Output tensor [M, N] in BF16
    """
    assert lhs.dtype == torch.bfloat16, f"lhs must be BF16, got {lhs.dtype}"
    assert rhs_packed.dtype == torch.uint8, f"rhs_packed must be uint8, got {rhs_packed.dtype}"
    assert rhs_scales.dtype == torch.uint8, f"rhs_scales must be uint8, got {rhs_scales.dtype}"

    M, K = lhs.shape
    N = rhs_packed.shape[0]  # [N, K//2]

    # Verify dimensions
    assert rhs_packed.shape[1] == K // 2, f"Expected rhs_packed.shape[1]={K//2}, got {rhs_packed.shape[1]}"
    assert rhs_scales.shape[0] == N, f"Expected rhs_scales.shape[0]={N}, got {rhs_scales.shape[0]}"

    # Output tensor
    output = torch.empty((M, N), dtype=torch.bfloat16, device=lhs.device)

    # Tile sizes
    # BLOCK_K must equal 32 to match MXFP4 scale block size (1 scale per 32 K values)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32  # Must equal 32 to match scale block size

    # Grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Launch kernel
    fused_mxfp4_single_gemm_kernel[grid](
        lhs, rhs_packed, rhs_scales, output,
        M, N, K,
        lhs.stride(0), lhs.stride(1),
        rhs_packed.stride(0), rhs_packed.stride(1),
        rhs_scales.stride(0), rhs_scales.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )

    # Add bias
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
