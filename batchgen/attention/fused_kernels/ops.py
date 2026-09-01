"""
Fused CUDA attention kernels: RMSNorm, Add+RMSNorm, RoPE, QKV Split.

Pre-compiled via pip install -e batchgen_kernels/. Loaded lazily on first use.
"""

from typing import Optional

import torch

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        import batchgen_kernels
        _ext = batchgen_kernels.load_extension("batchgen_kernels.attention._C_fused_ops")
    return _ext


def preload_fused_attention_kernels():
    """Resolve the extension before a timed model forward."""
    return _get_ext()


def cuda_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Standalone RMSNorm. Single CUDA kernel launch.
    num_valid_tokens: optional 1-element int32 device tensor to skip padding rows."""
    return _get_ext().rmsnorm_forward(x, weight, eps, num_valid_tokens)


def cuda_add_rmsnorm(
    residual: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> tuple:
    """Fused residual add + RMSNorm. Residual modified in-place.
    Returns (normed, residual).
    num_valid_tokens: optional 1-element int32 device tensor to skip padding rows."""
    results = _get_ext().add_rmsnorm_forward(residual, hidden, weight, eps, num_valid_tokens)
    return results[0], results[1]


def cuda_rope(
    query: torch.Tensor,   # [B, S, num_heads, head_dim]
    key: torch.Tensor,     # [B, S, num_kv_heads, head_dim]
    cos: torch.Tensor,
    sin: torch.Tensor,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> tuple:
    """Fused RoPE for Q and K. Single CUDA kernel launch.
    Returns (q_rot, k_rot).
    num_valid_tokens: optional 1-element int32 device tensor to skip padding tokens."""
    head_dim = query.shape[-1]
    half_dim = head_dim // 2

    # Normalize cos/sin to [B, S, head_dim]
    if cos.dim() == 4:
        cos = cos.squeeze(2)
        sin = sin.squeeze(2)
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    results = _get_ext().rope_forward(query, key, cos, sin, half_dim, num_valid_tokens)
    return results[0], results[1]


def cuda_qkv_split(
    qkv: torch.Tensor,
    q_size: int,
    kv_size: int,
    num_valid_tokens: Optional[torch.Tensor] = None,
) -> tuple:
    """QKV split (allocating). Returns (q, k, v).
    num_valid_tokens: optional 1-element int32 device tensor to skip padding rows."""
    results = _get_ext().qkv_split_forward(qkv, q_size, kv_size, num_valid_tokens)
    return results[0], results[1], results[2]


def cuda_qkv_split_inplace(
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    q_size: int,
    kv_size: int,
    num_valid_tokens: Optional[torch.Tensor] = None,
):
    """QKV split (zero-alloc). Writes to pre-allocated output tensors.
    num_valid_tokens: optional 1-element int32 device tensor to skip padding rows."""
    _get_ext().qkv_split_inplace(qkv, q_out, k_out, v_out, q_size, kv_size, num_valid_tokens)
