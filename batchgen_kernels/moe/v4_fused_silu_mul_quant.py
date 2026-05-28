# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

from __future__ import annotations

import torch
import torch.nn.functional as F


def _quantize_per_token(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = x.abs().amax(dim=-1) / fp8_max
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x_fp8 = torch.clamp(
        x / safe_scale.unsqueeze(-1),
        min=-fp8_max,
        max=fp8_max,
    ).to(torch.float8_e4m3fn)
    return x_fp8, scale


def fused_silu_mul_quant(
    gate: torch.Tensor,
    up: torch.Tensor,
    swiglu_limit: float = 10.0,
    quantize: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    assert gate.ndim == 2
    assert up.ndim == 2
    assert gate.shape == up.shape

    gate_f32 = gate.float()
    up_f32 = up.float()
    if swiglu_limit > 0:
        gate_f32 = torch.clamp(gate_f32, max=swiglu_limit)
        up_f32 = torch.clamp(up_f32, min=-swiglu_limit, max=swiglu_limit)

    out = F.silu(gate_f32) * up_f32
    if quantize:
        return _quantize_per_token(out)
    return out.to(torch.bfloat16)


__all__ = ["fused_silu_mul_quant"]
