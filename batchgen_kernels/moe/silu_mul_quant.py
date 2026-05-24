"""CUDA fused SiLU(gate) * up + per-token FP8 E4M3 quantization.

Ported from sglang's silu_and_mul_masked_post_quant.cuh (contig path).
Uses torch.utils.cpp_extension.load_inline for JIT compilation.

Public API:
    fused_silu_mul_quant_cuda(gate, up) -> (quant_fp8 [T, D], scales [T])
"""

from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load_inline

_MODULE = None

CPP_SOURCE = r"""
#include <torch/extension.h>

void silu_mul_quant_cuda(
    torch::Tensor gate,
    torch::Tensor up,
    torch::Tensor output,
    torch::Tensor scales);
"""


def _get_module():
    global _MODULE
    if _MODULE is None:
        cuda_src_path = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            "src",
            "moe",
            "silu_mul_quant.cu",
        )
        with open(cuda_src_path) as f:
            cuda_source = f.read()
        _MODULE = load_inline(
            name="batchgen_silu_mul_quant_cuda",
            cpp_sources=CPP_SOURCE,
            cuda_sources=cuda_source,
            functions=["silu_mul_quant_cuda"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _MODULE


def fused_silu_mul_quant_cuda(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU(gate) * up with per-token FP8 E4M3 quantization.

    Args:
        gate: [T, intermediate] bfloat16 tensor.
        up:   [T, intermediate] bfloat16 tensor.

    Returns:
        quant_fp8: [T, intermediate] float8_e4m3fn tensor.
        scales:    [T] float32 per-token scales.
    """
    assert gate.ndim == 2, f"gate must be 2D, got {gate.ndim}D"
    assert up.ndim == 2, f"up must be 2D, got {up.ndim}D"
    assert gate.shape == up.shape, f"shape mismatch: {gate.shape} vs {up.shape}"

    T, D = gate.shape
    gate = gate.contiguous().to(torch.bfloat16)
    up = up.contiguous().to(torch.bfloat16)

    output = torch.empty(T, D, dtype=torch.float8_e4m3fn, device=gate.device)
    scales = torch.empty(T, dtype=torch.float32, device=gate.device)

    _get_module().silu_mul_quant_cuda(gate, up, output, scales)
    return output, scales


__all__ = ["fused_silu_mul_quant_cuda"]
