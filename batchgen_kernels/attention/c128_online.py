"""Streaming HCA compress-128 (ring_size=1, online softmax) — CUDA kernel.

Ported from sglang deepseek_v4/c128_online.cuh.  Uses ``load_inline`` for
JIT compilation (no setup.py entry needed).

Public API
----------
    c128_online_compress(kv_score_buffer, kv_score_input, indices) -> Tensor

Buffer convention
-----------------
``kv_score_buffer`` is a ``float32`` tensor of shape ``[N, head_dim * 3]``
holding per-slot running state laid out as ``[max(D) | sum(D) | kv(D)]``.
**Zero-initialise** the buffer before the first token of every 128-chunk;
the kernel detects ``sum == 0`` to distinguish first-token init from
mid-chunk update.

``kv_score_input`` is ``[B, head_dim * 2]`` laid out as ``[kv(D) | score(D)]``
where ``score`` already includes any positional bias (APE).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline

_MODULE = None

_CUDA_SRC_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "attention"
    / "c128_online.cu"
)

CPP_SOURCE = r"""
#include <torch/extension.h>

void c128_online_step(
    torch::Tensor kv_score_buffer,
    torch::Tensor kv_score_input,
    torch::Tensor output,
    torch::Tensor indices);
"""


def _get_module():
    global _MODULE
    if _MODULE is None:
        cuda_source = _CUDA_SRC_PATH.read_text()
        _MODULE = load_inline(
            name="batchgen_c128_online",
            cpp_sources=CPP_SOURCE,
            cuda_sources=cuda_source,
            functions=["c128_online_step"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _MODULE


def c128_online_compress(
    kv_score_buffer: torch.Tensor,
    kv_score_input: torch.Tensor,
    indices: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Streaming HCA compression (ring_size=1), low memory variant.

    Parameters
    ----------
    kv_score_buffer : Tensor [N, D*3], float32
        Per-slot running state ``[max | sum | kv]``.  Modified **in-place**.
        Zero the relevant slots before the first token of each 128-chunk.
    kv_score_input : Tensor [B, D*2], float32
        New tokens laid out as ``[kv | score]`` (score includes APE bias).
    indices : Tensor [B], int32
        Maps each input row to a buffer slot.
    out : Tensor [B, D], float32, optional
        Pre-allocated output.  Created if *None*.

    Returns
    -------
    Tensor [B, D], float32
        Current weighted-average compressed kv for each input.
    """
    if kv_score_input.numel() == 0:
        head_dim = (
            kv_score_input.shape[-1] // 2 if kv_score_input.dim() == 2 else 0
        )
        return kv_score_input.new_empty(0, head_dim)

    B = kv_score_input.shape[0]
    head_dim = kv_score_input.shape[1] // 2

    if out is None:
        out = torch.empty(
            B, head_dim, dtype=torch.float32, device=kv_score_input.device
        )

    _get_module().c128_online_step(
        kv_score_buffer.contiguous(),
        kv_score_input.contiguous(),
        out,
        indices.contiguous(),
    )
    return out


__all__ = ["c128_online_compress"]
