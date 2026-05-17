"""Graph-friendly GLM-5 DSA query packing helpers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_flashmla_query_kernel(
    ABSORBED_ptr,  # [B, H, 512]
    ROPE_ptr,      # [B, H, 64]
    OUT_ptr,       # [B, 1, H, 576]
    NUM_VALID_TOKENS_ptr,  # [1] int32, optional by HAS_VALID_TOKENS
    B: tl.constexpr,
    H: tl.constexpr,
    HAS_VALID_TOKENS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    valid = offs < 576
    if HAS_VALID_TOKENS:
        num_valid_tokens = tl.load(NUM_VALID_TOKENS_ptr)
        if pid_b >= num_valid_tokens:
            tl.store(
                OUT_ptr + pid_b * H * 576 + pid_h * 576 + offs,
                tl.zeros([BLOCK_D], dtype=tl.float32),
                mask=valid,
            )
            return

    absorbed = tl.load(
        ABSORBED_ptr + pid_b * H * 512 + pid_h * 512 + offs,
        mask=offs < 512,
        other=0.0,
    )
    rope = tl.load(
        ROPE_ptr + pid_b * H * 64 + pid_h * 64 + (offs - 512),
        mask=(offs >= 512) & valid,
        other=0.0,
    )
    vals = tl.where(offs < 512, absorbed, rope)
    tl.store(
        OUT_ptr + pid_b * H * 576 + pid_h * 576 + offs,
        vals,
        mask=valid,
    )


def pack_flashmla_query_out(
    absorbed_q: torch.Tensor,
    q_rope: torch.Tensor,
    out: torch.Tensor,
    num_valid_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack ``[absorbed_nope, rope]`` into FlashMLA query shape ``[B,1,H,576]``."""
    if not absorbed_q.is_contiguous() or not q_rope.is_contiguous():
        raise ValueError("absorbed_q and q_rope must be contiguous")
    B, H, nope_dim = absorbed_q.shape
    if nope_dim != 512:
        raise ValueError(f"absorbed_q last dim must be 512, got {nope_dim}")
    if q_rope.shape != (B, H, 64):
        raise ValueError(f"q_rope must have shape {(B, H, 64)}, got {tuple(q_rope.shape)}")
    if out.shape != (B, 1, H, 576) or out.dtype != absorbed_q.dtype:
        raise ValueError(f"out must have shape {(B, 1, H, 576)} and dtype {absorbed_q.dtype}")
    if num_valid_tokens is not None:
        if num_valid_tokens.device != absorbed_q.device:
            raise ValueError("num_valid_tokens must be on the same device as absorbed_q")
        if num_valid_tokens.dtype != torch.int32:
            raise TypeError(f"num_valid_tokens must be int32, got {num_valid_tokens.dtype}")
        if num_valid_tokens.numel() != 1:
            raise ValueError(
                f"num_valid_tokens must contain one element, got {tuple(num_valid_tokens.shape)}"
            )
    _pack_flashmla_query_kernel[(B, H)](
        absorbed_q,
        q_rope,
        out,
        num_valid_tokens if num_valid_tokens is not None else absorbed_q,
        B=B,
        H=H,
        HAS_VALID_TOKENS=num_valid_tokens is not None,
        BLOCK_D=triton.next_power_of_2(576),
    )
    return out
