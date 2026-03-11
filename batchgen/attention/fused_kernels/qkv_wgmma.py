"""Fused QKV Projection WGMMA Kernel (requires SM90+).

Replaces cuBLAS GEMM + QKV split + RoPE with a single WGMMA kernel.
Compiled via torch.utils.cpp_extension.load at first use with runtime-detected SM arch
(same pattern as MoE WGMMA kernels).

Falls back gracefully on pre-SM90 GPUs (is_qkv_wgmma_available() returns False).
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Module loading (follows MoE WGMMA pattern: fused_wgmma_grouped.py)
# ──────────────────────────────────────────────────────────────────────────────

_module = None
_qkv_wgmma_available: Optional[bool] = None


def _check_wgmma_support() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)
    if cc[0] < 9:
        logger.debug(f"QKV WGMMA requires SM90+, found SM{cc[0]}{cc[1]}")
        return False
    return True


def _get_module():
    global _module

    if _module is not None:
        return _module

    try:
        import batchgen_kernels
        _module = batchgen_kernels.load_extension("batchgen_kernels.attention._C_qkv_wgmma")
        logger.info("Loaded pre-compiled QKV WGMMA kernel")
        return _module
    except Exception as e:
        logger.warning(f"Failed to load QKV WGMMA kernel: {e}")
        return None


def is_qkv_wgmma_available() -> bool:
    global _qkv_wgmma_available

    if _qkv_wgmma_available is not None:
        return _qkv_wgmma_available

    if not _check_wgmma_support():
        _qkv_wgmma_available = False
        return False

    if os.environ.get("BATCHGEN_DISABLE_WGMMA_QKV", "0") == "1":
        logger.info("QKV WGMMA kernel disabled by BATCHGEN_DISABLE_WGMMA_QKV")
        _qkv_wgmma_available = False
        return False

    _qkv_wgmma_available = _get_module() is not None
    return _qkv_wgmma_available


# ──────────────────────────────────────────────────────────────────────────────
# Python wrapper
# ──────────────────────────────────────────────────────────────────────────────

def cuda_qkv_wgmma(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    q_size: int = 4096,
    kv_size: int = 512,
    num_valid_tokens: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    head_dim: int = 64,
) -> tuple:
    """Fused QKV projection + split + optional RoPE via WGMMA.

    Args:
        input: [M, K] BF16 hidden states (must be 2D contiguous)
        weight: [N, K] BF16 packed QKV weight (nn.Linear convention)
        bias: [N] BF16 bias or None
        q_size: Q output columns (default 4096)
        kv_size: K/V output columns each (default 512)
        num_valid_tokens: 1-element int32 device tensor for CUDA graph padding skip
        rope_cos: [M, head_dim] BF16 precomputed cos values, or None
        rope_sin: [M, head_dim] BF16 precomputed sin values, or None
        head_dim: attention head dimension (default 64)

    Returns:
        (Q, K, V) as separate contiguous [M, *] BF16 tensors.
        When rope_cos/sin provided, Q and K have RoPE applied; V is unchanged.
    """
    mod = _get_module()
    assert mod is not None, "QKV WGMMA module not available"

    empty_bf16 = torch.empty(0, dtype=torch.bfloat16, device=input.device)
    if bias is None:
        bias = empty_bf16
    if rope_cos is None:
        rope_cos = empty_bf16
    if rope_sin is None:
        rope_sin = empty_bf16

    results = mod.qkv_wgmma_forward(
        input, weight, bias, q_size, kv_size, num_valid_tokens,
        rope_cos, rope_sin, head_dim,
    )
    return results[0], results[1], results[2]


def create_qkv_tma_desc(tensor: torch.Tensor, block_rows: int = 64, block_cols: int = 64) -> torch.Tensor:
    """Create TMA descriptor for a 2D BF16 tensor.

    Returns a CPU uint8 tensor of 128 bytes encoding the CUtensorMap.
    Must be called BEFORE CUDA graph capture — TMA descriptor creation
    is a CPU-side driver API call and is not capturable.

    Args:
        tensor: [rows, cols] BF16 CUDA tensor (must be contiguous, at fixed GPU address)
        block_rows: tile rows (BLOCK_M=64 for input, BLOCK_N=64 for weight)
        block_cols: tile cols (BLOCK_K=64)

    Returns:
        torch.Tensor of shape [128], dtype uint8, on CPU
    """
    mod = _get_module()
    assert mod is not None, "QKV WGMMA module not available"
    return mod.create_qkv_tma_desc(tensor, block_rows, block_cols)


def cuda_qkv_wgmma_inplace(
    Q_out: torch.Tensor,
    K_out: torch.Tensor,
    V_out: torch.Tensor,
    tma_desc_a: torch.Tensor,
    tma_desc_b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    head_dim: int = 64,
    num_valid_tokens: Optional[torch.Tensor] = None,
    M: int = 0,
    N: int = 0,
    K: int = 0,
    q_size: int = 0,
    kv_size: int = 0,
) -> None:
    """Inplace WGMMA for CUDA graph path.

    Uses pre-allocated output buffers and pre-built TMA descriptors.
    Safe to call during CUDA graph capture — no cuTensorMapEncodeTiled,
    no torch::empty inside C++.

    Args:
        Q_out: [max_M, q_size] BF16, pre-allocated output buffer
        K_out: [max_M, kv_size] BF16, pre-allocated output buffer
        V_out: [max_M, kv_size] BF16, pre-allocated output buffer
        tma_desc_a: CPU uint8[128], TMA descriptor for input buffer
        tma_desc_b: CPU uint8[128], TMA descriptor for weight
        bias: [N] BF16 or None
        rope_cos: [M, head_dim] BF16 or None
        rope_sin: [M, head_dim] BF16 or None
        head_dim: attention head dimension (64)
        num_valid_tokens: 1-element int32 device tensor, or None
        M, N, K: matrix dimensions
        q_size, kv_size: split sizes
    """
    mod = _get_module()
    assert mod is not None, "QKV WGMMA module not available"

    empty_bf16 = torch.empty(0, dtype=torch.bfloat16, device=Q_out.device)
    if bias is None:
        bias = empty_bf16
    if rope_cos is None:
        rope_cos = empty_bf16
    if rope_sin is None:
        rope_sin = empty_bf16

    mod.qkv_wgmma_forward_inplace(
        Q_out, K_out, V_out,
        tma_desc_a, tma_desc_b,
        bias, rope_cos, rope_sin, head_dim,
        num_valid_tokens, M, N, K, q_size, kv_size,
    )
