"""Grouped MXFP4 MoE wiring for DeepSeek-V4 on sm120.

DEFAULT = ``v4_mega3_moe_forward``: the validated, fastest fused grouped path
(stage1+SwiGLU kernel + stage2+scatter kernel). Set
``BATCHGEN_V4_RAGGED_FALLBACK=1`` to force the ragged parity/debug fallback.
Inside mega3, set ``BATCHGEN_V4_MEGA_USE_NATIVE=1`` to use the native CUDA
implementation instead of the Triton implementation.
"""

from __future__ import annotations

import os

import torch

_V4_STAGE1_GROUPED_CFG = {
    "block_m": 16,
    "block_n": 256,
    "block_k": 256,
    "num_warps": 4,
    "num_stages": 1,
}

_V4_STAGE2_GROUPED_CFG = dict(_V4_STAGE1_GROUPED_CFG)


def setup_v4_expert_weight_pointers(
    expert_weights: list[dict[str, torch.Tensor]],
    *,
    global_expert_count: int | None = None,
) -> dict[str, object]:
    """Canonicalize resident expert weights into reusable ragged/mega bundles."""
    from batchgen.moe.v4_ragged_moe_sm120 import prepare_ragged_weight_bundle

    if not expert_weights:
        raise ValueError("expert_weights must be non-empty")
    if global_expert_count is None:
        global_expert_count = len(expert_weights)
    if global_expert_count < len(expert_weights):
        raise ValueError("global_expert_count must be >= resident expert count")

    required = (
        "w1.weight",
        "w1.scale",
        "w3.weight",
        "w3.scale",
        "w2.weight",
        "w2.scale",
    )
    first = expert_weights[0]
    device = first["w1.weight"].device
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)

    for expert in expert_weights:
        for name in required:
            if name not in expert:
                raise KeyError(name)
            tensor = expert[name]
            ref = first[name]
            if tensor.device != device:
                raise ValueError(f"{name} must be on device {device}")
            if tensor.shape != ref.shape:
                raise ValueError(f"{name} shape must match first expert")
            if tensor.stride() != ref.stride():
                raise ValueError(f"{name} stride must match first expert")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous for ragged staging")
            if name.endswith(".weight") and tensor.element_size() != 1:
                raise ValueError(f"{name} must be byte-packed FP4")
            if name.endswith(".scale") and tensor.element_size() not in (1, 4):
                raise ValueError(f"{name} scale must be E8M0/uint8 or float32")
            if name.endswith(".scale") and tensor.element_size() == 1:
                if tensor.dtype != torch.uint8 and tensor.dtype != e8m0_dtype:
                    raise ValueError(
                        f"{name} 1-byte scale must be uint8 or E8M0"
                    )
            if (
                name.endswith(".scale")
                and tensor.element_size() == 4
                and tensor.dtype != torch.float32
            ):
                raise ValueError(f"{name} 4-byte scale must be float32")

    ragged_bundle = prepare_ragged_weight_bundle(expert_weights)
    return {
        "ragged_bundle": ragged_bundle,
        "mega_bundle": ragged_bundle,
        "global_expert_count": int(global_expert_count),
    }


def _v4_grouped_mxfp4_moe_forward_3d_ptrs_legacy(
    token_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    weight_ptrs: dict[str, object],
    owned_start: int,
    owned_count: int,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    del (
        token_states,
        topk_weights,
        topk_indices,
        weight_ptrs,
        owned_start,
        owned_count,
        swiglu_limit,
    )
    raise NotImplementedError(
        "Legacy grouped MoE path removed — use ragged kernel"
    )


def v4_grouped_mxfp4_moe_forward_3d_ptrs(
    token_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    weight_ptrs: dict[str, object],
    owned_start: int,
    owned_count: int,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    # Debug/parity escape hatch: force the ragged fallback instead of mega3.
    if os.environ.get("BATCHGEN_V4_RAGGED_FALLBACK", "0") == "1":
        from batchgen.moe.v4_ragged_moe_sm120 import (
            v4_grouped_mxfp4_moe_forward_ragged_ptrs,
        )

        return v4_grouped_mxfp4_moe_forward_ragged_ptrs(
            token_states,
            topk_weights,
            topk_indices,
            weight_ptrs,
            owned_start,
            owned_count,
            swiglu_limit,
        )

    # Default production path; mega3 can internally opt into native CUDA via
    # BATCHGEN_V4_MEGA_USE_NATIVE=1.
    from batchgen.moe.v4_mega3_moe_sm120 import v4_mega3_moe_forward

    return v4_mega3_moe_forward(
        token_states,
        topk_weights,
        topk_indices,
        weight_ptrs,
        owned_start,
        owned_count,
        swiglu_limit,
    )


def v4_slot_moe_forward(
    token_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    weight_ptrs: dict[str, object],
    owned_start: int,
    owned_count: int,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    from batchgen.moe.v4_ragged_moe_sm120 import (
        v4_grouped_mxfp4_moe_forward_ragged_ptrs,
    )

    # Debug/parity escape hatch: force the ragged fallback instead of mega3.
    if os.environ.get("BATCHGEN_V4_RAGGED_FALLBACK", "0") == "1":
        return v4_grouped_mxfp4_moe_forward_ragged_ptrs(
            token_states,
            topk_weights,
            topk_indices,
            weight_ptrs,
            owned_start,
            owned_count,
            swiglu_limit,
        )

    # Default production path; mega3 can internally opt into native CUDA via
    # BATCHGEN_V4_MEGA_USE_NATIVE=1.
    from batchgen.moe.v4_mega3_moe_sm120 import v4_mega3_moe_forward

    return v4_mega3_moe_forward(
        token_states,
        topk_weights,
        topk_indices,
        weight_ptrs,
        owned_start,
        owned_count,
        swiglu_limit,
    )


def v4_grouped_mxfp4_moe_forward_qat(
    token_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    weight_ptrs: dict[str, object],
    owned_start: int,
    owned_count: int,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    del (
        token_states,
        topk_weights,
        topk_indices,
        weight_ptrs,
        owned_start,
        owned_count,
        swiglu_limit,
    )
    raise NotImplementedError(
        "Legacy grouped MoE path removed — use ragged kernel"
    )
