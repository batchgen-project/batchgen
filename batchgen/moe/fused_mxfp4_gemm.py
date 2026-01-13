"""W4A16 Fused Dequant-GEMM for MXFP4 weights.

This implements fused dequantization and GEMM for GPT-OSS-120B's MXFP4 format:
- Activations: BF16 (16-bit, kept as-is)
- Weights: MXFP4 (4-bit, dequantized on-the-fly during GEMM)

MXFP4 Format:
- Block size: 32 FP4 values share one scale
- Packing: 2 FP4 values per uint8 byte (low nibble = first, high nibble = second)
- FP4 Lookup: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
- Scale: uint8 stored, exponent = scale - 127, dequant = fp4_value * 2^exponent
"""

import torch
import triton
import triton.language as tl
import logging


# FP4 lookup table values (indices 0-15)
# We encode these in the kernel directly for efficiency
FP4_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@triton.jit
def _fp4_lookup(idx):
    """Lookup FP4 value from 4-bit index using conditional logic."""
    # Positive values (idx 0-7)
    val = tl.where(idx == 0, 0.0,
           tl.where(idx == 1, 0.5,
           tl.where(idx == 2, 1.0,
           tl.where(idx == 3, 1.5,
           tl.where(idx == 4, 2.0,
           tl.where(idx == 5, 3.0,
           tl.where(idx == 6, 4.0,
           tl.where(idx == 7, 6.0,
           # Negative values (idx 8-15)
           tl.where(idx == 8, 0.0,  # -0.0 == 0.0 in float
           tl.where(idx == 9, -0.5,
           tl.where(idx == 10, -1.0,
           tl.where(idx == 11, -1.5,
           tl.where(idx == 12, -2.0,
           tl.where(idx == 13, -3.0,
           tl.where(idx == 14, -4.0,
           tl.where(idx == 15, -6.0, 0.0))))))))))))))))
    return val


@triton.jit
def _ldexp(val, exp):
    """Compute val * 2^exp using bit manipulation for float32."""
    # For BF16/FP32, we can use the formula: val * 2^exp
    # This is equivalent to torch.ldexp
    # We compute 2^exp as float and multiply
    two_pow_exp = tl.exp2(exp.to(tl.float32))
    return val * two_pow_exp


@triton.jit
def fused_mxfp4_gemm_kernel(
    # Activation (LHS): BF16 [M, K]
    lhs_ptr,
    # Weight (RHS): MXFP4 packed [N, K//2] as uint8
    rhs_packed_ptr,
    # Weight scales: [N, K//32] as uint8
    rhs_scales_ptr,
    # Output: BF16 [M, N]
    output_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_lhs_m, stride_lhs_k,
    stride_rhs_n, stride_rhs_k,  # Note: stride_rhs_k is for packed bytes (K//2)
    stride_scales_n, stride_scales_k,  # Note: stride_scales_k is for scales (K//32)
    stride_output_m, stride_output_n,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,  # Must be multiple of 32 (MXFP4 block size)
):
    """Fused MXFP4 dequant + GEMM kernel (W4A16).

    Computes: output = lhs @ rhs.T
    Where rhs is stored in MXFP4 format and dequantized on-the-fly.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Compute tile indices
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k

        # Load LHS tile [BLOCK_M, BLOCK_K] in BF16
        lhs_ptrs = lhs_ptr + offs_m[:, None] * stride_lhs_m + k_offs[None, :] * stride_lhs_k
        lhs_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0)
        lhs = lhs.to(tl.float32)

        # Load and dequantize RHS tile [BLOCK_N, BLOCK_K]
        # RHS is stored as packed uint8: [N, K//2]
        # Each byte contains 2 FP4 values

        # For each k position, we need to:
        # 1. Load the packed byte at position k//2
        # 2. Extract the appropriate nibble based on k%2
        # 3. Lookup FP4 value
        # 4. Apply scale from position k//32

        # Compute packed byte indices
        k_packed = k_offs // 2  # [BLOCK_K]
        k_nibble = k_offs % 2   # 0 = low nibble, 1 = high nibble

        # Load packed bytes [BLOCK_N, BLOCK_K//2] - but we load full BLOCK_K positions
        # We need to handle the packing carefully
        rhs_packed_ptrs = rhs_packed_ptr + offs_n[:, None] * stride_rhs_n + k_packed[None, :] * stride_rhs_k
        rhs_mask = (offs_n[:, None] < N) & (k_packed[None, :] < (K // 2))
        rhs_packed = tl.load(rhs_packed_ptrs, mask=rhs_mask, other=0)
        rhs_packed = rhs_packed.to(tl.uint8)

        # Extract nibbles based on position
        # Low nibble (k%2 == 0): packed & 0x0F
        # High nibble (k%2 == 1): (packed >> 4) & 0x0F
        is_high_nibble = (k_nibble[None, :] == 1)
        idx_lo = (rhs_packed & 0x0F).to(tl.int32)
        idx_hi = ((rhs_packed >> 4) & 0x0F).to(tl.int32)
        fp4_idx = tl.where(is_high_nibble, idx_hi, idx_lo)

        # Lookup FP4 values
        fp4_val = _fp4_lookup(fp4_idx)

        # Load scales [BLOCK_N, BLOCK_K//32]
        k_scale_idx = k_offs // 32
        scales_ptrs = rhs_scales_ptr + offs_n[:, None] * stride_scales_n + k_scale_idx[None, :] * stride_scales_k
        scales_mask = (offs_n[:, None] < N) & (k_scale_idx[None, :] < (K // 32))
        scales = tl.load(scales_ptrs, mask=scales_mask, other=127)
        scales = scales.to(tl.int32)

        # Compute exponent (scale - 127)
        exponent = scales - 127

        # Dequantize: fp4_val * 2^exponent
        rhs_dequant = _ldexp(fp4_val, exponent)

        # Matrix multiply: [BLOCK_M, BLOCK_K] @ [BLOCK_N, BLOCK_K].T -> [BLOCK_M, BLOCK_N]
        acc += tl.dot(lhs, rhs_dequant.trans(1, 0).to(tl.float32))

    # Store output
    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, acc.to(tl.bfloat16), mask=output_mask)


@triton.jit
def fused_mxfp4_grouped_gemm_kernel(
    # Activation (LHS): BF16 [M, K] - sorted by group
    lhs_ptr,
    # Weight pointers: list of MXFP4 packed tensors
    rhs_packed_ptrs_ptr,
    # Scale pointers: list of scale tensors
    rhs_scales_ptrs_ptr,
    # Group info
    group_idx_ptr,           # [num_groups] - which expert each group uses
    group_sizes_ptr,         # [num_groups] - size of each group
    group_start_indices_ptr, # [num_groups] - start index in LHS
    # Output: BF16 [M, N]
    output_ptr,
    # Dimensions
    N, K, num_groups,
    # Strides
    stride_lhs_m, stride_lhs_k,
    stride_rhs_n, stride_rhs_k,
    stride_scales_n, stride_scales_k,
    stride_output_m, stride_output_n,
    stride_group_idx, stride_group_sizes, stride_group_start,
    stride_rhs_ptrs, stride_scales_ptrs,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused MXFP4 dequant + grouped GEMM kernel for MoE.

    Each group uses a different expert's weights.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for g in range(num_groups):
        # Get group info
        gm = tl.load(group_sizes_ptr + g * stride_group_sizes)
        expert_idx = tl.load(group_idx_ptr + g * stride_group_idx)
        start_idx = tl.load(group_start_indices_ptr + g * stride_group_start)

        # Get weight pointers for this expert
        rhs_packed_ptr = tl.load(rhs_packed_ptrs_ptr + expert_idx * stride_rhs_ptrs).to(tl.pointer_type(tl.uint8))
        rhs_scales_ptr = tl.load(rhs_scales_ptrs_ptr + expert_idx * stride_scales_ptrs).to(tl.pointer_type(tl.uint8))

        # LHS base pointer for this group
        lhs_base = lhs_ptr + start_idx * stride_lhs_m
        output_base = output_ptr + start_idx * stride_output_m

        # Process tiles for this group
        num_tiles_m = tl.cdiv(gm, BLOCK_M)
        num_tiles_n = tl.cdiv(N, BLOCK_N)
        num_tiles = num_tiles_m * num_tiles_n

        tile_id = pid
        while tile_id < num_tiles:
            tile_m = tile_id // num_tiles_n
            tile_n = tile_id % num_tiles_n

            offs_m = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

            # Initialize accumulator
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Iterate over K
            for k_start in range(0, K, BLOCK_K):
                offs_k = k_start + tl.arange(0, BLOCK_K)

                # Load LHS tile
                lhs_ptrs = lhs_base + offs_m[:, None] * stride_lhs_m + offs_k[None, :] * stride_lhs_k
                lhs_mask = (offs_m[:, None] < gm) & (offs_k[None, :] < K)
                lhs = tl.load(lhs_ptrs, mask=lhs_mask, other=0.0).to(tl.float32)

                # Load and dequantize RHS
                k_packed = offs_k // 2
                k_nibble = offs_k % 2
                k_scale_idx = offs_k // 32

                rhs_packed_ptrs = rhs_packed_ptr + offs_n[:, None] * stride_rhs_n + k_packed[None, :] * stride_rhs_k
                rhs_mask = (offs_n[:, None] < N) & (k_packed[None, :] < (K // 2))
                rhs_packed = tl.load(rhs_packed_ptrs, mask=rhs_mask, other=0).to(tl.uint8)

                is_high_nibble = (k_nibble[None, :] == 1)
                idx_lo = (rhs_packed & 0x0F).to(tl.int32)
                idx_hi = ((rhs_packed >> 4) & 0x0F).to(tl.int32)
                fp4_idx = tl.where(is_high_nibble, idx_hi, idx_lo)
                fp4_val = _fp4_lookup(fp4_idx)

                scales_ptrs = rhs_scales_ptr + offs_n[:, None] * stride_scales_n + k_scale_idx[None, :] * stride_scales_k
                scales_mask = (offs_n[:, None] < N) & (k_scale_idx[None, :] < (K // 32))
                scales = tl.load(scales_ptrs, mask=scales_mask, other=127).to(tl.int32)
                exponent = scales - 127

                rhs_dequant = _ldexp(fp4_val, exponent)
                acc += tl.dot(lhs, rhs_dequant.trans(1, 0).to(tl.float32))

            # Store output
            output_ptrs = output_base + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
            output_mask = (offs_m[:, None] < gm) & (offs_n[None, :] < N)
            tl.store(output_ptrs, acc.to(tl.bfloat16), mask=output_mask)

            tile_id += num_programs


@torch.inference_mode()
def fused_mxfp4_gemm(
    lhs: torch.Tensor,           # [M, K] BF16
    rhs_packed: torch.Tensor,    # [N, K//2] uint8 (MXFP4 packed)
    rhs_scales: torch.Tensor,    # [N, K//32] uint8
) -> torch.Tensor:
    """Single MXFP4 W4A16 GEMM.

    Args:
        lhs: Activations [M, K] in BF16
        rhs_packed: MXFP4 packed weights [N, K//2] as uint8
        rhs_scales: MXFP4 scales [N, K//32] as uint8

    Returns:
        Output [M, N] in BF16
    """
    assert lhs.dtype == torch.bfloat16, f"LHS must be BF16, got {lhs.dtype}"
    assert rhs_packed.dtype == torch.uint8, f"RHS packed must be uint8, got {rhs_packed.dtype}"
    assert rhs_scales.dtype == torch.uint8, f"RHS scales must be uint8, got {rhs_scales.dtype}"

    M, K = lhs.shape
    N = rhs_packed.shape[0]

    assert rhs_packed.shape[1] == K // 2, f"RHS packed shape mismatch: {rhs_packed.shape[1]} != {K // 2}"
    assert rhs_scales.shape == (N, K // 32), f"RHS scales shape mismatch: {rhs_scales.shape} != {(N, K // 32)}"

    output = torch.empty((M, N), dtype=torch.bfloat16, device=lhs.device)

    # Block sizes (must be tuned for optimal performance)
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64  # Must be multiple of 32

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fused_mxfp4_gemm_kernel[grid](
        lhs,
        rhs_packed,
        rhs_scales,
        output,
        M, N, K,
        lhs.stride(0), lhs.stride(1),
        rhs_packed.stride(0), rhs_packed.stride(1),
        rhs_scales.stride(0), rhs_scales.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output


@torch.inference_mode()
def fused_mxfp4_grouped_gemm(
    lhs: torch.Tensor,                      # [M, K] BF16, sorted by group
    rhs_packed_list: list[torch.Tensor],    # List of [N, K//2] uint8 for each expert
    rhs_scales_list: list[torch.Tensor],    # List of [N, K//32] uint8 for each expert
    group_sizes: list[tuple[int, int]],     # [(expert_idx, group_size), ...]
) -> torch.Tensor:
    """Fused MXFP4 grouped GEMM for MoE.

    Args:
        lhs: Activations [M, K] in BF16, sorted by group
        rhs_packed_list: List of MXFP4 packed weights for each expert
        rhs_scales_list: List of MXFP4 scales for each expert
        group_sizes: List of (expert_idx, group_size) tuples

    Returns:
        Output [M, N] in BF16
    """
    assert lhs.dtype == torch.bfloat16
    device = lhs.device

    N = rhs_packed_list[0].shape[0]
    K = lhs.shape[1]
    M = lhs.shape[0]

    # Prepare group info tensors
    num_groups = len(group_sizes)
    group_idx = torch.tensor([idx for idx, _ in group_sizes], dtype=torch.int32, device=device)
    group_size = torch.tensor([size for _, size in group_sizes], dtype=torch.int32, device=device)
    group_start = torch.zeros(num_groups, dtype=torch.int32, device=device)
    group_start[1:] = torch.cumsum(group_size[:-1], dim=0)

    # Create pointer tensors
    rhs_packed_ptrs = torch.tensor([t.data_ptr() for t in rhs_packed_list], dtype=torch.int64, device=device)
    rhs_scales_ptrs = torch.tensor([t.data_ptr() for t in rhs_scales_list], dtype=torch.int64, device=device)

    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    grid = (num_sms,)

    fused_mxfp4_grouped_gemm_kernel[grid](
        lhs,
        rhs_packed_ptrs,
        rhs_scales_ptrs,
        group_idx, group_size, group_start,
        output,
        N, K, num_groups,
        lhs.stride(0), lhs.stride(1),
        rhs_packed_list[0].stride(0), rhs_packed_list[0].stride(1),
        rhs_scales_list[0].stride(0), rhs_scales_list[0].stride(1),
        output.stride(0), output.stride(1),
        group_idx.stride(0), group_size.stride(0), group_start.stride(0),
        rhs_packed_ptrs.stride(0), rhs_scales_ptrs.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return output


def test_fused_mxfp4_gemm():
    """Test the fused MXFP4 GEMM kernel."""
    from batchgen.quantization.mxfp4 import mxfp4_dequantize_reference

    torch.manual_seed(42)
    device = torch.device("cuda:0")

    M, N, K = 128, 256, 512

    # Create random activations
    lhs = torch.randn(M, K, dtype=torch.bfloat16, device=device)

    # Create random MXFP4 weights
    rhs_packed = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
    rhs_scales = torch.randint(100, 150, (N, K // 32), dtype=torch.uint8, device=device)

    # Reference: dequantize then GEMM
    rhs_dequant = mxfp4_dequantize_reference(rhs_packed, rhs_scales, dtype=torch.bfloat16)
    ref_output = lhs @ rhs_dequant.T

    # Fused kernel
    fused_output = fused_mxfp4_gemm(lhs, rhs_packed, rhs_scales)

    # Compare
    max_diff = (ref_output - fused_output).abs().max().item()
    mean_diff = (ref_output - fused_output).abs().mean().item()

    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")

    assert max_diff < 0.1, f"Max diff too large: {max_diff}"
    print("Test passed!")


if __name__ == "__main__":
    test_fused_mxfp4_gemm()
