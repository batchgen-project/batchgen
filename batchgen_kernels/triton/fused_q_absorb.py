"""Fused q_absorb + query_states construction kernel.

Replaces 4 dispatches:
  1. torch.empty(query_states)           — allocation
  2. einsum('bhd,hdc->bhc', q_nope, q_absorb) — batched GEMV
  3. query_states[:,:,:,:kv_lora_rank] = absorbed  — indexed copy
  4. query_states[:,:,:,kv_lora_rank:] = q_pe      — indexed copy

With a single Triton kernel that:
  - Reads q_nope [bsz, H, D=128] and q_absorb [H, D=128, C=512]
  - Computes the batched dot product (absorbed = q_nope @ q_absorb)
  - Reads q_pe [bsz, H, 1, R=64]
  - Writes query_states [bsz, 1, H, C+R=576] directly in flash_mla layout

Grid: (H,) — one program per head, processes all batch elements.
At bsz=1: each program does a single [1, 128] @ [128, 512] dot product + copies 64 pe values.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_q_absorb_kernel(
    # Input pointers
    q_nope_ptr,     # [B, H, D] — q_nope squeezed (no seq dim)
    q_absorb_ptr,   # [H, D, C] — absorption weight
    q_pe_ptr,       # [B, H, 1, R] — q_pe (contiguous)
    # Output pointer
    out_ptr,        # [B, 1, H, C+R] — query_states in flash_mla layout
    # Dimensions
    B,
    H: tl.constexpr,
    D: tl.constexpr,     # qk_nope_head_dim = 128
    C: tl.constexpr,     # kv_lora_rank = 512
    R: tl.constexpr,     # qk_rope_head_dim = 64
    # Strides for q_nope [B, H, D]
    stride_qn_b, stride_qn_h, stride_qn_d,
    # Strides for q_absorb [H, D, C]
    stride_qa_h, stride_qa_d, stride_qa_c,
    # Strides for q_pe [B, H, 1, R]
    stride_qp_b, stride_qp_h, stride_qp_s, stride_qp_r,
    # Strides for output [B, 1, H, C+R]
    stride_out_b, stride_out_s, stride_out_h, stride_out_cr,
    # Block size
    BLOCK_B: tl.constexpr,
):
    """Fused q_absorb einsum + q_pe concatenation into query_states."""
    head_idx = tl.program_id(0)

    # Load absorption weight for this head: [D, C]
    # D=128, C=512 — fits in registers for small D
    d_range = tl.arange(0, D)
    c_range = tl.arange(0, C)
    w_offsets = head_idx * stride_qa_h + d_range[:, None] * stride_qa_d + c_range[None, :] * stride_qa_c
    w = tl.load(q_absorb_ptr + w_offsets)  # [D, C]

    # Process batch elements
    num_b_blocks = tl.cdiv(B, BLOCK_B)
    for b_block in range(num_b_blocks):
        b_start = b_block * BLOCK_B
        b_range = b_start + tl.arange(0, BLOCK_B)
        b_mask = b_range < B

        # Load q_nope for this head: [BLOCK_B, D]
        qn_offsets = b_range[:, None] * stride_qn_b + head_idx * stride_qn_h + d_range[None, :] * stride_qn_d
        qn = tl.load(q_nope_ptr + qn_offsets, mask=b_mask[:, None], other=0.0)  # [BLOCK_B, D]

        # Compute absorbed = q_nope @ q_absorb: [BLOCK_B, D] @ [D, C] -> [BLOCK_B, C]
        absorbed = tl.dot(qn, w)  # [BLOCK_B, C]

        # Write absorbed to output[:, :, head, :C]
        out_abs_offsets = (b_range[:, None] * stride_out_b
                          + 0 * stride_out_s
                          + head_idx * stride_out_h
                          + c_range[None, :] * stride_out_cr)
        tl.store(out_ptr + out_abs_offsets, absorbed, mask=b_mask[:, None])

        # Load q_pe for this head: [BLOCK_B, R]
        r_range = tl.arange(0, R)
        qp_offsets = b_range[:, None] * stride_qp_b + head_idx * stride_qp_h + 0 * stride_qp_s + r_range[None, :] * stride_qp_r
        qpe = tl.load(q_pe_ptr + qp_offsets, mask=b_mask[:, None], other=0.0)  # [BLOCK_B, R]

        # Write q_pe to output[:, :, head, C:C+R]
        out_pe_offsets = (b_range[:, None] * stride_out_b
                         + 0 * stride_out_s
                         + head_idx * stride_out_h
                         + (C + r_range[None, :]) * stride_out_cr)
        tl.store(out_ptr + out_pe_offsets, qpe, mask=b_mask[:, None])


def fused_q_absorb_query_states(
    q_nope: torch.Tensor,      # [bsz, num_heads, qk_nope_head_dim] — squeezed
    q_absorb: torch.Tensor,    # [num_heads, qk_nope_head_dim, kv_lora_rank]
    q_pe: torch.Tensor,        # [bsz, num_heads, 1, qk_rope_head_dim] — contiguous
    output: torch.Tensor = None,  # [bsz, 1, num_heads, kv_lora_rank + qk_rope_head_dim] — pre-allocated
) -> torch.Tensor:
    """Fused q_absorb einsum + q_pe concatenation.

    Replaces:
        query_states = torch.empty(bsz, H, 1, C+R)
        query_states[:,:,:,:C] = einsum('bhd,hdc->bhc', q_nope, q_absorb).view(...)
        query_states[:,:,:,C:] = q_pe
        query_states = query_states.view(bsz, 1, H, C+R)

    Args:
        q_nope: [bsz, H, D] BF16 — squeezed q_nope (no seq dim)
        q_absorb: [H, D, C] BF16 — absorption weight
        q_pe: [bsz, H, 1, R] BF16 — contiguous q_pe with RoPE applied

    Returns:
        query_states: [bsz, 1, H, C+R] BF16 in flash_mla input layout
    """
    B, H, D = q_nope.shape
    _, _, C = q_absorb.shape
    R = q_pe.shape[-1]

    assert q_absorb.shape == (H, D, C)
    assert q_pe.shape == (B, H, 1, R)
    assert q_pe.is_contiguous()

    if output is None:
        output = torch.empty(B, 1, H, C + R, dtype=q_nope.dtype, device=q_nope.device)

    fused_q_absorb_kernel[(H,)](
        q_nope, q_absorb, q_pe, output,
        B, H, D, C, R,
        q_nope.stride(0), q_nope.stride(1), q_nope.stride(2),
        q_absorb.stride(0), q_absorb.stride(1), q_absorb.stride(2),
        q_pe.stride(0), q_pe.stride(1), q_pe.stride(2), q_pe.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_B=max(1, min(B, 16)),
    )

    return output
