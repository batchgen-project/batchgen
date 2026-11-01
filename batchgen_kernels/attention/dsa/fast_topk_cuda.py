"""CUDA top-k helper specialized for GLM-5 DSA index_topk=2048.

The kernel is the AOT extension ``batchgen_kernels.attention._C_dsa_fast_topk``
(source ``batchgen_kernels/src/attention/dsa_fast_topk.cu``), built by
``setup.py`` and mirrored in ``_jit_registry.py``. It is imported on first use
and never compiled at runtime.
"""

from __future__ import annotations

import torch

_MODULE = None
_TOPK = 2048
_EXTENSION = "batchgen_kernels.attention._C_dsa_fast_topk"


def _get_module():
    global _MODULE
    if _MODULE is None:
        # Resolved at call time so tests can substitute the loader.
        from batchgen_kernels import load_extension

        _MODULE = load_extension(_EXTENSION, allow_dev_jit=False)
    return _MODULE


def fast_topk_2048_out(
    score: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
    num_valid_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    if score.ndim != 2:
        raise ValueError(f"score must have shape [B, N], got {tuple(score.shape)}")
    if score.dtype != torch.float32:
        raise TypeError(f"score must be float32, got {score.dtype}")
    if lengths.shape != (score.shape[0],):
        raise ValueError(
            f"lengths must have shape {(score.shape[0],)}, got {tuple(lengths.shape)}"
        )
    if lengths.dtype != torch.int32:
        raise TypeError(f"lengths must be int32, got {lengths.dtype}")
    if indices.shape != (score.shape[0], _TOPK):
        raise ValueError(
            f"indices must have shape {(score.shape[0], _TOPK)}, got {tuple(indices.shape)}"
        )
    if indices.dtype != torch.int32:
        raise TypeError(f"indices must be int32, got {indices.dtype}")
    if num_valid_tokens is not None:
        if num_valid_tokens.dtype != torch.int32:
            raise TypeError(f"num_valid_tokens must be int32, got {num_valid_tokens.dtype}")
        if num_valid_tokens.numel() != 1:
            raise ValueError(
                f"num_valid_tokens must contain one element, got {tuple(num_valid_tokens.shape)}"
            )
    if not score.is_cuda or not lengths.is_cuda or not indices.is_cuda:
        raise ValueError("score, lengths, and indices must be CUDA tensors")
    if num_valid_tokens is not None and not num_valid_tokens.is_cuda:
        raise ValueError("num_valid_tokens must be a CUDA tensor")
    _get_module().fast_topk_2048_out(
        score.contiguous(),
        lengths.contiguous(),
        indices,
        num_valid_tokens.contiguous() if num_valid_tokens is not None else None,
    )
    return indices


def fast_topk_2048(score: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    indices = torch.empty(score.shape[0], _TOPK, dtype=torch.int32, device=score.device)
    return fast_topk_2048_out(score, lengths, indices)
