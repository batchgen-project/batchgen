"""Fused INT4 W4A16 dequantization and grouped GEMM for Kimi K2.5 MoE layers.

This module implements fused dequantization of INT4 weights during matrix
multiplication, avoiding the memory overhead of materializing full BF16 weights.

INT4 W4A16 Format (compressed-tensors pack-quantized, symmetric):
- 32 INT4 values per scale group (group_size=32), packed in 16 bytes
- Packing: 2 INT4 values per uint8 (low nibble first, high nibble second)
- INT4 range: signed [-8, +7] (4-bit two's complement)
- Scales: bf16, one per group of 32 elements
- Activations: BF16 (untouched — no activation quantization)

Key differences from MXFP4 (GPT-OSS):
- No FP4 lookup table — simple integer sign-extend
- No ldexp — direct bf16 multiply with bf16 scale
- Scales are bf16 (not uint8 exponents)
- Activations stay BF16 (not quantized to FP8)
- SiLU activation (not OpenAI SwiGLU with alpha/clamping)
"""

import logging
import torch
import triton
import triton.language as tl
from typing import List, Tuple

# Import shared MoE utilities (format-agnostic)
from batchgen.moe.mxfp4_grouped_gemm import (
    moe_token_dispatch,
    reshape_to_3d_expert_layout,
    gather_from_3d_expert_layout,
    setup_expert_weight_pointers,
)

# INT4 configuration
INT4_GROUP_SIZE = 32  # INT4 values per scale
INT4_PACKED_BLOCK_SIZE = 16  # Bytes per scale group (32 values / 2 per byte)


# =============================================================================
# Triton JIT Helpers
# =============================================================================

@triton.jit
def _int4_sign_extend(nibble):
    """Sign-extend unsigned 4-bit nibble [0,15] to signed [-8,+7].

    INT4 two's complement: 0-7 are positive, 8-15 represent -8 to -1.
    """
    return tl.where(nibble >= 8, nibble - 16, nibble)


# =============================================================================
# 3D Grouped INT4 W4A16 GEMM Kernel
# =============================================================================

@triton.jit
def fused_int4_grouped_gemm_kernel_3d(
    # Input [E, M_max, K] BF16
    lhs_ptr,
    # Weight pointer arrays [num_experts] int64
    rhs_ptrs_ptr,           # -> [N, K//2] uint8 packed INT4
    rhs_scale_ptrs_ptr,     # -> [N, K//32] bf16
    # Per-expert token counts [num_experts] int32
    expert_tokens_ptr,
    # Output [E, M_max, N] BF16
    output_ptr,
    # Dimensions
    M_max, N, K,
    # Strides for lhs [E, M_max, K]
    stride_lhs_e, stride_lhs_m, stride_lhs_k,
    # Strides for rhs weights [N, K//2] uint8
    stride_rhs_n, stride_rhs_k_packed,
    # Strides for scales [N, K//32] bf16
    stride_scale_n, stride_scale_k,
    # Strides for output [E, M_max, N]
    stride_out_e, stride_out_m, stride_out_n,
    # Stride for pointer arrays
    stride_ptrs,
    # Block sizes
    BLOCK_M: tl.constexpr,  # 64
    BLOCK_N: tl.constexpr,  # 64
    BLOCK_K: tl.constexpr,  # Ignored — kernel uses 32-wide K blocks internally
):
    """Grouped INT4 W4A16 GEMM following DeepSeek-V3 3D layout pattern.

    Grid: (num_experts, cdiv(N, BLOCK_N))
    - axis 0: expert index
    - axis 1: N-block index

    Each thread block handles one (expert, N-block) pair and loops over:
    - M-blocks (tokens for that expert)
    - K-blocks (32-wide, matching INT4 scale granularity)

    INT4 dequant is simpler than MXFP4:
    - Sign-extend nibble (1 tl.where) vs FP4 decode (2 tl.where + IEEE754 bitcast)
    - bf16 multiply vs ldexp (exponent bit manipulation)
    - bf16 scale direct load vs uint8→int32 exponent conversion
    """
    expert_idx = tl.program_id(axis=0)
    n_pid = tl.program_id(axis=1)

    # Early exit for empty experts
    gm = tl.load(expert_tokens_ptr + expert_idx).to(tl.int32)
    if gm == 0:
        return

    # Base pointers for this expert's input/output slices
    cur_lhs_ptr = lhs_ptr + expert_idx * stride_lhs_e
    cur_out_ptr = output_ptr + expert_idx * stride_out_e

    # Load weight and scale pointers for this expert
    rhs_base_ptr = tl.load(rhs_ptrs_ptr + expert_idx * stride_ptrs).to(tl.pointer_type(tl.uint8))
    scale_base_ptr = tl.load(rhs_scale_ptrs_ptr + expert_idx * stride_ptrs).to(tl.pointer_type(tl.bfloat16))

    # N-block offset
    offs_n = n_pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # Process all M-blocks for this expert
    num_m_blocks = tl.cdiv(gm, BLOCK_M)

    for m_block in range(num_m_blocks):
        offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < gm

        # FP32 accumulator for precision
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # K-loop: 32 K values per iteration (one INT4 scale group)
        num_k_blocks = K // 32

        for k_block in range(num_k_blocks):
            k_start = k_block * 32

            # ===== Load packed INT4 weights [BLOCK_N, 16] for 32 K values =====
            k_packed = k_start // 2  # Packed byte index
            offs_k_packed = tl.arange(0, 16)  # 32 values / 2 per byte = 16 bytes
            rhs_ptrs = rhs_base_ptr + offs_n[:, None] * stride_rhs_n + \
                       (k_packed + offs_k_packed[None, :]) * stride_rhs_k_packed
            rhs_packed = tl.load(rhs_ptrs, mask=n_mask[:, None], other=0)

            # ===== Extract and sign-extend nibbles =====
            lo = _int4_sign_extend((rhs_packed & 0x0F).to(tl.int32))   # [BLOCK_N, 16]
            hi = _int4_sign_extend(((rhs_packed >> 4) & 0x0F).to(tl.int32))  # [BLOCK_N, 16]

            # ===== Load bf16 scale (one per 32 K values, per N row) =====
            # Direct bf16 load — no uint8→exponent conversion needed
            scale_ptrs = scale_base_ptr + offs_n * stride_scale_n + k_block * stride_scale_k
            scale_val = tl.load(scale_ptrs, mask=n_mask, other=0.0).to(tl.bfloat16)

            # ===== Apply scale: int_val * bf16_scale → bf16 =====
            scale_broadcast = scale_val[:, None] + tl.zeros((1, 16), dtype=tl.bfloat16)
            lo_scaled = lo.to(tl.bfloat16) * scale_broadcast  # [BLOCK_N, 16]
            hi_scaled = hi.to(tl.bfloat16) * scale_broadcast  # [BLOCK_N, 16]

            # ===== Interleave lo/hi → [BLOCK_N, 32] =====
            # tl.join: [N,16] + [N,16] → [N,16,2]
            # tl.reshape: [N,16,2] → [N,32] with order [lo0,hi0,lo1,hi1,...] = [K0,K1,...]
            val_joined = tl.join(lo_scaled, hi_scaled)
            val_interleaved = tl.reshape(val_joined, (BLOCK_N, 32))

            # ===== Load LHS tile [BLOCK_M, 32] =====
            offs_k = tl.arange(0, 32)
            lhs_ptrs = cur_lhs_ptr + offs_m[:, None] * stride_lhs_m + \
                       (k_start + offs_k[None, :]) * stride_lhs_k
            lhs_tile = tl.load(lhs_ptrs, mask=m_mask[:, None], other=0.0)

            # ===== BF16 GEMM: [BLOCK_M, 32] @ [32, BLOCK_N] → accumulate =====
            acc += tl.dot(
                lhs_tile.to(tl.bfloat16),
                tl.trans(val_interleaved),
                allow_tf32=False,
            ).to(tl.float32)

        # Store output [BLOCK_M, BLOCK_N]
        out_ptrs = cur_out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


# =============================================================================
# Python Launchers
# =============================================================================

def grouped_int4_gemm_3d(
    hidden_3d: torch.Tensor,          # [E, M_max, K] BF16
    weight_ptrs: torch.Tensor,        # [num_experts] int64
    scale_ptrs: torch.Tensor,         # [num_experts] int64
    expert_counts: torch.Tensor,      # [num_experts] int32
    N: int,                           # Output dimension
    weight_ref: torch.Tensor,         # Reference weight for strides [N, K//2] uint8
    scale_ref: torch.Tensor,          # Reference scale for strides [N, K//32] bf16
    BLOCK_M: int = 64,
    BLOCK_N: int = 64,
    BLOCK_K: int = 32,  # Fixed at 32 to match INT4 scale granularity
) -> torch.Tensor:
    """Launch grouped INT4 W4A16 GEMM kernel with 3D layout.

    The kernel uses 32-wide K blocks internally to align with INT4 scale groups.

    Args:
        hidden_3d: Input tensor [E, M_max, K] in BF16
        weight_ptrs: Pointer array [num_experts] to packed weight tensors [N, K//2] uint8
        scale_ptrs: Pointer array [num_experts] to scale tensors [N, K//32] bf16
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

    if expert_counts.dtype != torch.int32:
        expert_counts = expert_counts.to(torch.int32)

    output_3d = torch.empty(num_experts, M_max, N, dtype=torch.bfloat16, device=device)

    # Grid: (num_experts, cdiv(N, BLOCK_N))
    grid = (num_experts, triton.cdiv(N, BLOCK_N))

    fused_int4_grouped_gemm_kernel_3d[grid](
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


@torch.inference_mode()
def fused_int4_single_gemm(
    lhs: torch.Tensor,
    rhs_packed: torch.Tensor,
    rhs_scales: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    """INT4 W4A16 dequantization + GEMM for single expert.

    Uses unfused path: dequant INT4→BF16 then standard BF16 matmul.
    This is used for non-persistent experts loaded on-demand.

    Computes: output = lhs @ dequant(rhs).T + bias

    Args:
        lhs: Input activations [M, K] in BF16
        rhs_packed: Packed INT4 weights [N, K//2] in uint8
        rhs_scales: Scales [N, K//32] in bf16
        bias: Optional bias [N] in BF16

    Returns:
        Output tensor [M, N] in BF16
    """
    assert lhs.dtype == torch.bfloat16, f"lhs must be BF16, got {lhs.dtype}"
    assert rhs_packed.dtype == torch.uint8, f"rhs_packed must be uint8, got {rhs_packed.dtype}"
    assert rhs_scales.dtype == torch.bfloat16, f"rhs_scales must be bf16, got {rhs_scales.dtype}"

    from batchgen.quantization.int4 import int4_dequantize

    # Dequantize INT4 → BF16
    weight_bf16 = int4_dequantize(rhs_packed, rhs_scales, dtype=torch.bfloat16)

    # Standard BF16 matmul
    output = torch.mm(lhs, weight_bf16.T)

    if bias is not None:
        output = output + bias

    return output


# =============================================================================
# INT4 MoE Forward (3D Layout, Single Kernel Per Stage)
# =============================================================================

def grouped_int4_moe_forward_3d(
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
    gate_weight_ref: torch.Tensor,        # [N_inter, hidden//2] uint8
    gate_scale_ref: torch.Tensor,         # [N_inter, hidden//32] bf16
    up_weight_ref: torch.Tensor,
    up_scale_ref: torch.Tensor,
    down_weight_ref: torch.Tensor,        # [hidden, N_inter//2] uint8
    down_scale_ref: torch.Tensor,         # [hidden, N_inter//32] bf16
    num_experts: int = 384,
) -> torch.Tensor:
    """Grouped INT4 W4A16 MoE forward with 3D layout (DeepSeek-V3 pattern).

    Single kernel launch per projection stage, processing all experts in parallel.

    Kernel launches per MoE layer:
    - Before (per-expert loop): 384 experts x 3 projections = 1152 launches
    - After (grouped 3D): 3 launches (gate, up, down)

    Activation: SiLU gating — silu(gate) * up (standard DeepSeek-V3 style).
    No clamping or custom SwiGLU alpha (that's GPT-OSS specific).

    Args:
        hidden_states: Input [batch*seq, hidden] in BF16
        topk_indices: Expert indices [batch*seq, num_experts_per_tok]
        topk_weights: Routing weights [batch*seq, num_experts_per_tok]
        gate_ptrs, gate_scale_ptrs: Pointer arrays for gate projection
        up_ptrs, up_scale_ptrs: Pointer arrays for up projection
        down_ptrs, down_scale_ptrs: Pointer arrays for down projection
        gate_weight_ref, etc.: Reference tensors for computing strides
        num_experts: Number of experts (default 384 for K2.5)

    Returns:
        Output [batch*seq, hidden] in BF16
    """
    num_tokens, hidden_size = hidden_states.shape
    device = hidden_states.device

    # Intermediate dimension from reference weights
    N_intermediate = gate_weight_ref.shape[0]

    # Step 1: Dispatch tokens to experts (sort by expert)
    sorted_hidden, expert_offsets, original_indices, original_k, routing_weights = moe_token_dispatch(
        hidden_states, topk_indices, topk_weights, num_experts
    )

    total_tokens_routed = sorted_hidden.shape[0]

    # Per-expert token counts
    expert_counts = (expert_offsets[1:] - expert_offsets[:-1]).to(torch.int32)

    # Step 2: Reshape to 3D layout [E, M_max, K]
    hidden_3d, M_max = reshape_to_3d_expert_layout(sorted_hidden, expert_counts, num_experts)

    # Step 3: Gate projection (SINGLE kernel for all experts)
    gate_out_3d = grouped_int4_gemm_3d(
        hidden_3d, gate_ptrs, gate_scale_ptrs, expert_counts,
        N_intermediate, gate_weight_ref, gate_scale_ref
    )

    # Step 4: Up projection (SINGLE kernel for all experts)
    up_out_3d = grouped_int4_gemm_3d(
        hidden_3d, up_ptrs, up_scale_ptrs, expert_counts,
        N_intermediate, up_weight_ref, up_scale_ref
    )

    # Step 5: SiLU gating activation (DeepSeek-V3 / K2.5 style)
    # silu(gate) * up — standard MoE activation, no clamping
    intermediate_3d = torch.nn.functional.silu(gate_out_3d) * up_out_3d

    # Step 6: Down projection (SINGLE kernel for all experts)
    output_3d = grouped_int4_gemm_3d(
        intermediate_3d, down_ptrs, down_scale_ptrs, expert_counts,
        hidden_size, down_weight_ref, down_scale_ref
    )

    # Step 7: Gather back from 3D to sorted 1D
    sorted_output = gather_from_3d_expert_layout(output_3d, expert_counts, total_tokens_routed)

    # Step 8: Scatter back to original order with routing weights
    output = torch.zeros(num_tokens, hidden_size, dtype=hidden_states.dtype, device=device)
    weighted_output = sorted_output * routing_weights.unsqueeze(-1)
    output.scatter_add_(0, original_indices.unsqueeze(-1).expand_as(weighted_output), weighted_output)

    return output


# =============================================================================
# Single-Expert INT4 MLP Forward (for non-persistent experts)
# =============================================================================

def int4_mlp_forward(
    x: torch.Tensor,
    gate_packed: torch.Tensor,
    gate_scales: torch.Tensor,
    up_packed: torch.Tensor,
    up_scales: torch.Tensor,
    down_packed: torch.Tensor,
    down_scales: torch.Tensor,
) -> torch.Tensor:
    """Single expert MLP forward with INT4 W4A16 weights.

    Implements: down(silu(gate(x)) * up(x))

    Uses unfused path: dequant + BF16 matmul per projection.

    Args:
        x: Input [M, hidden] in BF16
        gate_packed, gate_scales: Gate projection weights (INT4 packed + bf16 scales)
        up_packed, up_scales: Up projection weights
        down_packed, down_scales: Down projection weights

    Returns:
        Output [M, hidden] in BF16
    """
    original_shape = x.shape
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])

    # Gate and Up projections
    gate_out = fused_int4_single_gemm(x, gate_packed, gate_scales)
    up_out = fused_int4_single_gemm(x, up_packed, up_scales)

    # SiLU gating (DeepSeek-V3 / K2.5 style)
    intermediate = torch.nn.functional.silu(gate_out) * up_out

    # Down projection
    output = fused_int4_single_gemm(intermediate, down_packed, down_scales)

    if len(original_shape) > 2:
        output = output.view(*original_shape[:-1], -1)

    return output
