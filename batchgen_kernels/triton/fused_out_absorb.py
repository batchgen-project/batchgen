"""Fused out_absorb + transpose + contiguous + reshape kernel.

Replaces 3 dispatches:
  1. einsum('bqhc,hdc->bhqd', attn_out, out_absorb)  — batched GEMV
  2. .transpose(1,2).contiguous()                     — memory copy
  3. .reshape(bsz, o_proj_in_dim)                     — view

With a single Triton kernel that:
  - Reads attn_out [bsz, 1, H, C=512] and out_absorb [H, D=128, C=512]
  - Computes per-head: [bsz, C] @ [D, C]^T → [bsz, D]
  - Writes directly to flat [bsz, H*D=8192] layout (no transpose/copy)

Grid: (H,) — one program per head.
At bsz=1: each program does [1, 512] @ [128, 512]^T → [1, 128].
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_out_absorb_kernel(
    # Input pointers
    attn_out_ptr,    # [B, 1, H, C] — flash_mla output
    out_absorb_ptr,  # [H, D, C] — absorption weight
    # Output pointer
    out_ptr,         # [B, H*D] — flat layout for o_proj input
    # Dimensions
    B,
    H: tl.constexpr,
    D: tl.constexpr,     # v_head_dim = 128
    C: tl.constexpr,     # kv_lora_rank = 512
    # Strides for attn_out [B, 1, H, C]
    stride_ao_b, stride_ao_s, stride_ao_h, stride_ao_c,
    # Strides for out_absorb [H, D, C]
    stride_oa_h, stride_oa_d, stride_oa_c,
    # Strides for output [B, H*D]
    stride_out_b, stride_out_hd,
    # Block size
    BLOCK_B: tl.constexpr,
):
    """Fused out_absorb einsum + layout transform."""
    head_idx = tl.program_id(0)

    # Load weight for this head: [D, C] → we need [C, D] for matmul
    # Actually: out[b, h, d] = sum_c attn_out[b, 0, h, c] * out_absorb[h, d, c]
    # = attn_out[b, h, :] @ out_absorb[h, :, :]^T (transposed)
    # So load out_absorb[h] as [D, C], compute [B, C] @ [C, D] = [B, D]
    # Actually einsum is bqhc,hdc->bhqd meaning:
    #   result[b,h,q,d] = sum_c attn_out[b,q,h,c] * out_absorb[h,d,c]
    # With q=1: result[b,h,d] = sum_c attn_out[b,h,c] * out_absorb[h,d,c]
    # = attn_out[b,h,:] @ out_absorb[h,:,:]^T  where out_absorb is [D,C]

    d_range = tl.arange(0, D)
    c_range = tl.arange(0, C)

    # Load out_absorb[head, :, :] as [D, C]
    w_offsets = head_idx * stride_oa_h + d_range[:, None] * stride_oa_d + c_range[None, :] * stride_oa_c
    w = tl.load(out_absorb_ptr + w_offsets)  # [D, C]
    # Transpose to [C, D] for matmul
    wt = tl.trans(w)  # [C, D]

    num_b_blocks = tl.cdiv(B, BLOCK_B)
    for b_block in range(num_b_blocks):
        b_start = b_block * BLOCK_B
        b_range = b_start + tl.arange(0, BLOCK_B)
        b_mask = b_range < B

        # Load attn_out[b, 0, head, :] → [BLOCK_B, C]
        ao_offsets = (b_range[:, None] * stride_ao_b
                      + 0 * stride_ao_s
                      + head_idx * stride_ao_h
                      + c_range[None, :] * stride_ao_c)
        ao = tl.load(attn_out_ptr + ao_offsets, mask=b_mask[:, None], other=0.0)  # [BLOCK_B, C]

        # Compute: [BLOCK_B, C] @ [C, D] → [BLOCK_B, D]
        result = tl.dot(ao, wt)  # [BLOCK_B, D]

        # Write to flat output: out[b, head*D + d]
        out_offsets = (b_range[:, None] * stride_out_b
                       + (head_idx * D + d_range[None, :]) * stride_out_hd)
        tl.store(out_ptr + out_offsets, result, mask=b_mask[:, None])


def fused_out_absorb_reshape(
    attn_out: torch.Tensor,        # [bsz, 1, H, C=512]
    out_absorb: torch.Tensor,      # [H, D=128, C=512]
    output: torch.Tensor = None,   # [bsz, H*D=8192] pre-allocated
) -> torch.Tensor:
    """Fused out_absorb einsum + transpose + reshape.

    Replaces:
        out = einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
        out = out.transpose(1,2).contiguous()
        out = out.reshape(bsz, H*D)

    Returns:
        [bsz, H*D] BF16 — ready for o_proj linear
    """
    B = attn_out.shape[0]
    H, D, C = out_absorb.shape

    assert attn_out.shape == (B, 1, H, C), f"Expected [B,1,H,C], got {attn_out.shape}"

    if output is None:
        output = torch.empty(B, H * D, dtype=attn_out.dtype, device=attn_out.device)

    fused_out_absorb_kernel[(H,)](
        attn_out, out_absorb, output,
        B, H, D, C,
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
        out_absorb.stride(0), out_absorb.stride(1), out_absorb.stride(2),
        output.stride(0), output.stride(1),
        BLOCK_B=max(1, min(B, 16)),
    )

    return output
