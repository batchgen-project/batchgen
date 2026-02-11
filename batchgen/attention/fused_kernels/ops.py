"""
Fused CUDA attention kernels: RMSNorm, Add+RMSNorm, RoPE, QKV Split.

Compiled via torch.utils.cpp_extension.load() at import time.
"""

import torch
from pathlib import Path
from torch.utils.cpp_extension import load

_csrc_dir = Path(__file__).parent / "csrc"

_ext = load(
    name="attention_fused_cuda",
    sources=[
        str(_csrc_dir / "attention_extension.cc"),
        str(_csrc_dir / "rmsnorm.cu"),
        str(_csrc_dir / "rope.cu"),
        str(_csrc_dir / "qkv_split.cu"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    verbose=False,
)


def cuda_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Standalone RMSNorm. Single CUDA kernel launch."""
    return _ext.rmsnorm_forward(x, weight, eps)


def cuda_add_rmsnorm(
    residual: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple:
    """Fused residual add + RMSNorm. Residual modified in-place.
    Returns (normed, residual)."""
    results = _ext.add_rmsnorm_forward(residual, hidden, weight, eps)
    return results[0], results[1]


def cuda_rope(
    query: torch.Tensor,   # [B, S, num_heads, head_dim]
    key: torch.Tensor,     # [B, S, num_kv_heads, head_dim]
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple:
    """Fused RoPE for Q and K. Single CUDA kernel launch.
    Returns (q_rot, k_rot)."""
    head_dim = query.shape[-1]
    half_dim = head_dim // 2

    # Normalize cos/sin to [B, S, head_dim]
    if cos.dim() == 4:
        cos = cos.squeeze(2)
        sin = sin.squeeze(2)
    elif cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    results = _ext.rope_forward(query, key, cos, sin, half_dim)
    return results[0], results[1]


def cuda_qkv_split(
    qkv: torch.Tensor,
    q_size: int,
    kv_size: int,
) -> tuple:
    """QKV split (allocating). Returns (q, k, v)."""
    results = _ext.qkv_split_forward(qkv, q_size, kv_size)
    return results[0], results[1], results[2]


def cuda_qkv_split_inplace(
    qkv: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    q_size: int,
    kv_size: int,
):
    """QKV split (zero-alloc). Writes to pre-allocated output tensors."""
    _ext.qkv_split_inplace(qkv, q_out, k_out, v_out, q_size, kv_size)
