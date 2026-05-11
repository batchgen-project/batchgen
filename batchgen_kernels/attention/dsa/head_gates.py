"""Valid-row-aware GLM-5 DSA head-gate projection."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _head_gates_out_kernel(
    hidden_ptr,  # [B, K]
    weight_ptr,  # [H, K]
    out_ptr,  # [B, H]
    num_valid_tokens_ptr,  # [1] int32, optional by HAS_VALID_TOKENS
    hidden_stride_b,
    weight_stride_h,
    out_stride_b,
    B: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    SCALE: tl.constexpr,
    HAS_VALID_TOKENS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    if HAS_VALID_TOKENS:
        num_valid_tokens = tl.load(num_valid_tokens_ptr)
        if pid_b >= num_valid_tokens:
            tl.store(out_ptr + pid_b * out_stride_b + pid_h, 0.0)
            return

    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros((), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs
        mask = k < K
        hidden = tl.load(
            hidden_ptr + pid_b * hidden_stride_b + k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            weight_ptr + pid_h * weight_stride_h + k,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(hidden * weight, axis=0)

    tl.store(out_ptr + pid_b * out_stride_b + pid_h, acc * SCALE)


def head_gates_out(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    *,
    scale: float,
    num_valid_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``out = hidden @ weight.T * scale`` with padded rows zeroed."""
    if not hidden.is_contiguous():
        raise ValueError("hidden must be contiguous")
    if not weight.is_contiguous():
        raise ValueError("weight must be contiguous")
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            f"hidden and weight must be rank-2, got {tuple(hidden.shape)} and {tuple(weight.shape)}"
        )
    batch, hidden_size = hidden.shape
    num_heads, weight_hidden = weight.shape
    if weight_hidden != hidden_size:
        raise ValueError(
            f"weight hidden dimension {weight_hidden} does not match hidden {hidden_size}"
        )
    if out.shape != (batch, num_heads) or out.dtype != torch.float32:
        raise ValueError(
            f"out must be float32 with shape {(batch, num_heads)}, got {tuple(out.shape)} {out.dtype}"
        )
    if out.device != hidden.device or weight.device != hidden.device:
        raise ValueError("hidden, weight, and out must be on the same device")
    if num_valid_tokens is not None:
        if num_valid_tokens.device != hidden.device:
            raise ValueError("num_valid_tokens must be on the same device as hidden")
        if num_valid_tokens.dtype != torch.int32:
            raise TypeError(f"num_valid_tokens must be int32, got {num_valid_tokens.dtype}")
        if num_valid_tokens.numel() != 1:
            raise ValueError(
                f"num_valid_tokens must contain one element, got {tuple(num_valid_tokens.shape)}"
            )

    block_k = min(1024, triton.next_power_of_2(hidden_size))
    _head_gates_out_kernel[(batch, num_heads)](
        hidden,
        weight,
        out,
        num_valid_tokens if num_valid_tokens is not None else out,
        hidden.stride(0),
        weight.stride(0),
        out.stride(0),
        B=batch,
        H=num_heads,
        K=hidden_size,
        SCALE=float(scale),
        HAS_VALID_TOKENS=num_valid_tokens is not None,
        BLOCK_K=block_k,
    )
    return out
