from __future__ import annotations

import torch

_FP4_E2M1_TABLE_VALUES = (
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


def fp4_packed_bytes(weight: torch.Tensor) -> torch.Tensor:
    if weight.element_size() == 1:
        return weight.contiguous().view(torch.uint8)
    return weight.contiguous().to(torch.uint8)


def dequant_fp4_e2m1_weight(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    if scale is None:
        raise RuntimeError("DeepSeek-V4 FP4 weight is missing its E8M0 scale tensor.")
    packed = fp4_packed_bytes(weight)
    table = torch.tensor(
        _FP4_E2M1_TABLE_VALUES,
        dtype=torch.float32,
        device=packed.device,
    )
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    unpacked_shape = packed.shape[:-1] + (packed.shape[-1] * 2,)
    unpacked = torch.empty(unpacked_shape, dtype=torch.float32, device=packed.device)
    unpacked[..., 0::2] = table[low.long()]
    unpacked[..., 1::2] = table[high.long()]
    expanded_scale = (
        scale.to(torch.float32)
        .unsqueeze(-1)
        .expand(*scale.shape, 32)
        .reshape(*scale.shape[:-1], scale.shape[-1] * 32)
    )
    expanded_scale = expanded_scale[..., : unpacked.shape[-1]]
    return (unpacked * expanded_scale).to(dtype)
