from __future__ import annotations

import torch

try:
    from batchgen_kernels.attention.dsa.fast_topk_cuda import (
        fast_topk as _fast_topk,
    )

    _cuda_available = True
except Exception:
    _fast_topk = None
    _cuda_available = False

_SUPPORTED_K = (512, 1024, 2048)


def v4_topk(
    scores: torch.Tensor,
    k: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.ndim != 2:
        raise ValueError(
            f"scores must have shape [T, N], got {tuple(scores.shape)}"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if scores.shape[-1] < k:
        raise ValueError(f"k={k} exceeds scores.shape[-1]={scores.shape[-1]}")

    if (
        _cuda_available
        and scores.is_cuda
        and k in _SUPPORTED_K
        and scores.dtype == torch.float32
    ):
        try:
            lengths = torch.full(
                (scores.shape[0],),
                scores.shape[1],
                dtype=torch.int32,
                device=scores.device,
            )
            indices = _fast_topk(scores, lengths, k)
            values = scores.gather(1, indices.to(torch.int64))
            return values, indices.to(torch.int64)
        except Exception:
            pass

    return torch.topk(scores, k=k, dim=-1)


__all__ = ["v4_topk"]
