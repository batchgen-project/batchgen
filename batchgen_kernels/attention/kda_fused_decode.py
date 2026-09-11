"""AOT Kimi-K3 decode fusion.

The kernel combines the three short-convolution updates, the one-token
delta-rule recurrence, and the per-head gated RMSNorm.  It is intentionally
K3-specific: local KDA head counts are the TP8/TP16/TP32 values 12/6/3,
head/value dimensions are 128, and the convolution width is four.

The wrapper is AOT-only in production.  A missing extension must fall back to
the established FLA path at the model seam rather than starting a per-worker
JIT build during decode.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from batchgen_kernels import load_extension

_ext = None

_SUPPORTED_HEADS = {3, 6, 12}
_HEAD_DIM = 128
_CONV_WIDTH = 4


def _get_ext():
    global _ext
    if _ext is None:
        module_name = "batchgen_kernels.attention._C_kda_fused_decode"
        try:
            _ext = load_extension(module_name, allow_dev_jit=False)
        except ImportError:
            # A staged AOT build lives outside the source package on remote
            # machines.  Keep production AOT-only: this only adds the explicit
            # artifact directory to the package search path, never JIT builds.
            ext_dir = os.environ.get("K3_FUSED_DECODE_EXT_DIR")
            if not ext_dir:
                raise
            import batchgen_kernels.attention as attention_pkg

            if ext_dir not in attention_pkg.__path__:
                attention_pkg.__path__.append(ext_dir)
            _ext = load_extension(module_name, allow_dev_jit=False)
    return _ext


def covered(
    mixed_qkv: torch.Tensor,
    forget_gate: torch.Tensor,
    beta: torch.Tensor,
    conv_q: torch.Tensor,
    conv_k: torch.Tensor,
    conv_v: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    weight_v: torch.Tensor,
    bias_q: Optional[torch.Tensor],
    bias_k: Optional[torch.Tensor],
    bias_v: Optional[torch.Tensor],
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    onorm_gate: torch.Tensor,
    onorm_weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
) -> bool:
    """Return whether the AOT kernel can consume these native K3 tensors."""
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] % (3 * _HEAD_DIM):
        return False
    heads = mixed_qkv.shape[1] // (3 * _HEAD_DIM)
    if heads not in _SUPPORTED_HEADS:
        return False
    batch = mixed_qkv.shape[0]
    segment = heads * _HEAD_DIM
    if (
        forget_gate.shape != (batch, segment)
        or beta.shape != (batch, heads)
        or onorm_gate.shape != (batch, segment)
        or state.ndim != 4
        or state.shape[1:] != (heads, _HEAD_DIM, _HEAD_DIM)
        or state_indices.shape != (batch,)
    ):
        return False
    if any(
        tensor.shape != (state.shape[0], segment, _CONV_WIDTH - 1)
        for tensor in (conv_q, conv_k, conv_v)
    ):
        return False
    if any(
        tensor.shape != (segment, _CONV_WIDTH)
        for tensor in (weight_q, weight_k, weight_v)
    ):
        return False
    if any(
        bias is not None and bias.shape != (segment,)
        for bias in (bias_q, bias_k, bias_v)
    ):
        return False
    if a_log.ndim != 1 or a_log.shape[0] < heads:
        return False
    if dt_bias.shape != (segment,) or onorm_weight.shape != (_HEAD_DIM,):
        return False
    if mixed_qkv.dtype is not torch.bfloat16:
        return False
    if any(
        tensor.dtype is not torch.bfloat16
        for tensor in (forget_gate, beta, conv_q, conv_k, conv_v, onorm_gate)
    ):
        return False
    if any(
        tensor.dtype is not torch.float32
        for tensor in (weight_q, weight_k, weight_v, a_log, dt_bias, onorm_weight)
    ):
        return False
    if any(bias is not None and bias.dtype is not torch.float32
           for bias in (bias_q, bias_k, bias_v)):
        return False
    if state.dtype is not torch.float32 or state_indices.dtype is not torch.int32:
        return False
    if any(
        tensor.stride(-1) != 1
        for tensor in (mixed_qkv, forget_gate, beta, onorm_gate,
                       weight_q, weight_k, weight_v)
    ):
        return False
    if any(tensor.stride(-1) != 1 for tensor in (conv_q, conv_k, conv_v)):
        return False
    if state.stride(-1) != 1 or state.stride(-2) != _HEAD_DIM:
        return False
    if state.stride(-3) != _HEAD_DIM * _HEAD_DIM:
        return False
    if not state_indices.is_contiguous():
        return False
    return all(
        tensor.device == mixed_qkv.device
        for tensor in (
            forget_gate, beta, conv_q, conv_k, conv_v,
            weight_q, weight_k, weight_v, a_log, dt_bias,
            onorm_gate, onorm_weight, state, state_indices,
        )
        if tensor is not None
    ) and all(
        bias is None or bias.device == mixed_qkv.device
        for bias in (bias_q, bias_k, bias_v)
    )


def kda_fused_decode(
    mixed_qkv: torch.Tensor,
    forget_gate: torch.Tensor,
    beta: torch.Tensor,
    conv_q: torch.Tensor,
    conv_k: torch.Tensor,
    conv_v: torch.Tensor,
    weight_q: torch.Tensor,
    weight_k: torch.Tensor,
    weight_v: torch.Tensor,
    bias_q: Optional[torch.Tensor],
    bias_k: Optional[torch.Tensor],
    bias_v: Optional[torch.Tensor],
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    onorm_gate: torch.Tensor,
    onorm_weight: torch.Tensor,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    scale: float,
    onorm_eps: float,
    lower_bound: Optional[float],
) -> torch.Tensor:
    """Run the fused decode step and return flattened BF16 head output.

    ``conv_*`` and ``state`` are updated in place.  Negative state indices are
    treated as CUDA-graph padding rows and produce zero output without
    touching either pool.
    """
    if not covered(
        mixed_qkv, forget_gate, beta, conv_q, conv_k, conv_v,
        weight_q, weight_k, weight_v, bias_q, bias_k, bias_v,
        a_log, dt_bias, onorm_gate, onorm_weight, state, state_indices,
    ):
        raise ValueError("unsupported tensors for K3 fused KDA decode")
    return _get_ext().kda_fused_decode_forward(
        mixed_qkv, forget_gate, beta,
        conv_q, conv_k, conv_v,
        weight_q, weight_k, weight_v,
        bias_q, bias_k, bias_v,
        a_log, dt_bias, onorm_gate, onorm_weight,
        state, state_indices,
        float(scale), float(onorm_eps),
        float(lower_bound) if lower_bound is not None else 0.0,
        lower_bound is not None,
    )


__all__ = ["covered", "kda_fused_decode"]
