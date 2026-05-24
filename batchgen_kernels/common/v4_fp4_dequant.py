# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

from __future__ import annotations

from typing import Optional

import torch

FP4_E2M1_TABLE = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

BLOCK_SIZE = 32


def _to_packed_bytes(weight: torch.Tensor) -> torch.Tensor:
    if weight.element_size() == 1:
        return weight.contiguous().view(torch.uint8)
    return weight.contiguous().to(torch.uint8)


def dequant_fp4_e2m1(
    weight: torch.Tensor,
    scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Nibble-unpack E2M1 FP4 → lookup → scale (block=32) → dtype. Matches model.py exactly."""
    packed = _to_packed_bytes(weight)
    table = torch.tensor(
        FP4_E2M1_TABLE, dtype=torch.float32, device=packed.device
    )

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F

    unpacked_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    unpacked = torch.empty(
        unpacked_shape, dtype=torch.float32, device=packed.device
    )
    unpacked[..., 0::2] = table[low.long()]
    unpacked[..., 1::2] = table[high.long()]

    expanded_scale = (
        scale.to(torch.float32)
        .unsqueeze(-1)
        .expand(*scale.shape, BLOCK_SIZE)
        .reshape(*scale.shape[:-1], scale.shape[-1] * BLOCK_SIZE)
    )
    expanded_scale = expanded_scale[..., : unpacked.shape[-1]]

    return (unpacked * expanded_scale).to(dtype)
