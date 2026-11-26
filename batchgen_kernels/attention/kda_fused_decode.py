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

KERNEL_VERSIONS = (1, 2)
DEFAULT_VERSION = 2

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


def coverage_gap(
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
) -> Optional[str]:
    """The first failing coverage check as a message, or None when the AOT
    kernel can consume these native K3 tensors."""
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] % (3 * _HEAD_DIM):
        return (f"mixed_qkv shape {tuple(mixed_qkv.shape)} is not "
                f"[batch, heads * 3 * {_HEAD_DIM}]")
    heads = mixed_qkv.shape[1] // (3 * _HEAD_DIM)
    if heads not in _SUPPORTED_HEADS:
        return (f"local head count {heads} not in "
                f"{sorted(_SUPPORTED_HEADS)}")
    batch = mixed_qkv.shape[0]
    segment = heads * _HEAD_DIM
    if forget_gate.shape != (batch, segment):
        return (f"forget_gate shape {tuple(forget_gate.shape)} != "
                f"{(batch, segment)}")
    if beta.shape != (batch, heads):
        return f"beta shape {tuple(beta.shape)} != {(batch, heads)}"
    if onorm_gate.shape != (batch, segment):
        return (f"onorm_gate shape {tuple(onorm_gate.shape)} != "
                f"{(batch, segment)}")
    if state.ndim != 4 or state.shape[1:] != (heads, _HEAD_DIM, _HEAD_DIM):
        return (f"state shape {tuple(state.shape)} is not "
                f"[slots, {heads}, {_HEAD_DIM}, {_HEAD_DIM}]")
    if state_indices.shape != (batch,):
        return (f"state_indices shape {tuple(state_indices.shape)} != "
                f"{(batch,)}")
    for name, tensor in (("conv_q", conv_q), ("conv_k", conv_k),
                         ("conv_v", conv_v)):
        if tensor.shape != (state.shape[0], segment, _CONV_WIDTH - 1):
            return (f"{name} shape {tuple(tensor.shape)} != "
                    f"{(state.shape[0], segment, _CONV_WIDTH - 1)}")
    for name, tensor in (("weight_q", weight_q), ("weight_k", weight_k),
                         ("weight_v", weight_v)):
        if tensor.shape != (segment, _CONV_WIDTH):
            return (f"{name} shape {tuple(tensor.shape)} != "
                    f"{(segment, _CONV_WIDTH)}")
    for name, bias in (("bias_q", bias_q), ("bias_k", bias_k),
                       ("bias_v", bias_v)):
        if bias is not None and bias.shape != (segment,):
            return f"{name} shape {tuple(bias.shape)} != {(segment,)}"
    if a_log.ndim != 1 or a_log.shape[0] < heads:
        return (f"a_log shape {tuple(a_log.shape)} is not [>= {heads}] "
                f"1-D")
    if dt_bias.shape != (segment,):
        return f"dt_bias shape {tuple(dt_bias.shape)} != {(segment,)}"
    if onorm_weight.shape != (_HEAD_DIM,):
        return (f"onorm_weight shape {tuple(onorm_weight.shape)} != "
                f"{(_HEAD_DIM,)}")
    if mixed_qkv.dtype is not torch.bfloat16:
        return f"mixed_qkv dtype {mixed_qkv.dtype} is not bfloat16"
    for name, tensor in (("forget_gate", forget_gate), ("beta", beta),
                         ("conv_q", conv_q), ("conv_k", conv_k),
                         ("conv_v", conv_v), ("onorm_gate", onorm_gate)):
        if tensor.dtype is not torch.bfloat16:
            return f"{name} dtype {tensor.dtype} is not bfloat16"
    for name, tensor in (("weight_q", weight_q), ("weight_k", weight_k),
                         ("weight_v", weight_v), ("a_log", a_log),
                         ("dt_bias", dt_bias),
                         ("onorm_weight", onorm_weight)):
        if tensor.dtype is not torch.float32:
            return f"{name} dtype {tensor.dtype} is not float32"
    for name, bias in (("bias_q", bias_q), ("bias_k", bias_k),
                       ("bias_v", bias_v)):
        if bias is not None and bias.dtype is not torch.float32:
            return f"{name} dtype {bias.dtype} is not float32"
    if state.dtype is not torch.float32:
        return f"state dtype {state.dtype} is not float32"
    if state_indices.dtype is not torch.int32:
        return f"state_indices dtype {state_indices.dtype} is not int32"
    for name, tensor in (("mixed_qkv", mixed_qkv),
                         ("forget_gate", forget_gate), ("beta", beta),
                         ("onorm_gate", onorm_gate), ("weight_q", weight_q),
                         ("weight_k", weight_k), ("weight_v", weight_v),
                         ("conv_q", conv_q), ("conv_k", conv_k),
                         ("conv_v", conv_v)):
        if tensor.stride(-1) != 1:
            return (f"{name} last-axis stride {tensor.stride(-1)} != 1 "
                    f"(not row-contiguous)")
    if state.stride(-1) != 1 or state.stride(-2) != _HEAD_DIM:
        return (f"state inner strides {state.stride()} are not the dense "
                f"[..., {_HEAD_DIM}, 1] layout")
    if state.stride(-3) != _HEAD_DIM * _HEAD_DIM:
        return (f"state head stride {state.stride(-3)} != "
                f"{_HEAD_DIM * _HEAD_DIM}")
    # The kernel moves the state in float4 units.
    if state.stride(0) % 4 or state.data_ptr() % 16:
        return (f"state slot stride {state.stride(0)} or base address is "
                f"not float4-aligned")
    if not state_indices.is_contiguous():
        return "state_indices is not contiguous"
    named = (
        ("forget_gate", forget_gate), ("beta", beta), ("conv_q", conv_q),
        ("conv_k", conv_k), ("conv_v", conv_v), ("weight_q", weight_q),
        ("weight_k", weight_k), ("weight_v", weight_v), ("a_log", a_log),
        ("dt_bias", dt_bias), ("onorm_gate", onorm_gate),
        ("onorm_weight", onorm_weight), ("state", state),
        ("state_indices", state_indices), ("bias_q", bias_q),
        ("bias_k", bias_k), ("bias_v", bias_v),
    )
    for name, tensor in named:
        if tensor is not None and tensor.device != mixed_qkv.device:
            return (f"{name} device {tensor.device} != mixed_qkv device "
                    f"{mixed_qkv.device}")
    return None


def covered(*args, **kwargs) -> bool:
    """Return whether the AOT kernel can consume these native K3 tensors
    (see :func:`coverage_gap` for the first failing check)."""
    return coverage_gap(*args, **kwargs) is None


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
    version: int = DEFAULT_VERSION,
) -> torch.Tensor:
    """Run the fused decode step and return flattened BF16 head output.

    ``conv_*`` and ``state`` are updated in place.  Negative state indices are
    treated as CUDA-graph padding rows and produce zero output without
    touching either pool.  ``version`` 1 stages the recurrent state through
    shared memory in four chunks; version 2 keeps it in registers with every
    load issued up front.  Both compute the same recurrence.
    """
    if version not in KERNEL_VERSIONS:
        raise ValueError(f"unknown fused KDA decode version {version}")
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
        int(version),
    )


__all__ = ["covered", "kda_fused_decode"]
