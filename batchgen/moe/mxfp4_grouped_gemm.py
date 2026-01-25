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
