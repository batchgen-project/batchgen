from __future__ import annotations

import torch

_C = None
_LOAD_FAILED_EXC: Exception | None = None


def _load():
    global _C, _LOAD_FAILED_EXC
    if _C is not None:
        return _C
    if _LOAD_FAILED_EXC is not None:
        raise _LOAD_FAILED_EXC
    import batchgen_kernels

    try:
        _C = batchgen_kernels.load_extension(
            "batchgen_kernels.attention._C_fused_qk_rmsnorm"
        )
    except Exception as exc:
        _LOAD_FAILED_EXC = exc
        raise
    return _C


def fused_qk_rmsnorm_cuda(
    qr: torch.Tensor,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    mod = _load()
    qr_out, kv_out = mod.fused_qk_rmsnorm_forward(qr, kv, kv_weight, eps)
    return qr_out, kv_out


def is_cuda_backend_available() -> bool:
    try:
        _load()
        return True
    except Exception:
        return False
