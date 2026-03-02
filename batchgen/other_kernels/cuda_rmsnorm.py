"""CUDA RMSNorm and fused Add+RMSNorm kernels.

Ported from batchgen_kernel_dev/attention/csrc/rmsnorm.cu.

Two kernels:
  rmsnorm_forward(input, weight, eps) -> output
  add_rmsnorm_forward(residual, hidden, weight, eps) -> (normed, residual)

Both use warp-shuffle block reduction, 256 threads/block, one block per row.
All launches use getCurrentCUDAStream for CUDA graph compatibility.
"""

import logging

import torch


_rmsnorm_module = None


# ──────────────────────────────────────────────────────────────────────────────
# CUDA Source Code (external .cu file)
# ──────────────────────────────────────────────────────────────────────────────
# Source: batchgen_kernels/src/common/cuda_rmsnorm.cu

# ──────────────────────────────────────────────────────────────────────────────
# Module Loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_rmsnorm_module():
    """Load the pre-compiled CUDA RMSNorm module."""
    global _rmsnorm_module
    if _rmsnorm_module is not None:
        return _rmsnorm_module

    try:
        import batchgen_kernels
        _rmsnorm_module = batchgen_kernels.load_extension(
            "batchgen_kernels.common._C_cuda_rmsnorm"
        )
        logging.info("Loaded pre-compiled CUDA RMSNorm kernels")
        return _rmsnorm_module
    except Exception as e:
        logging.warning(f"Failed to load CUDA RMSNorm kernels: {e}")
        return None


def cuda_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Standalone CUDA RMSNorm. Drop-in replacement for fused_rmsnorm."""
    mod = _load_rmsnorm_module()
    return mod.rmsnorm_forward(x, weight, eps)


def cuda_add_rmsnorm(
    residual: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> tuple:
    """Fused residual add + RMSNorm. Returns (normed, residual_updated).

    residual is modified in-place.
    """
    mod = _load_rmsnorm_module()
    return mod.add_rmsnorm_forward(residual, hidden, weight, eps)
