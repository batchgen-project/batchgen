from __future__ import annotations

import logging

import torch

import batchgen_kernels

_MODULE = None
_AVAILABLE = None


def _has_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return major == 12


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    _MODULE = batchgen_kernels.load_extension("batchgen_kernels.moe._C_mega_moe_sm120")
    return _MODULE


def is_mega_moe_sm120_available() -> bool:
    global _AVAILABLE
    if _AVAILABLE is not None:
        return _AVAILABLE
    if not _has_sm120():
        _AVAILABLE = False
        return False
    try:
        _load_module()
        _AVAILABLE = True
    except Exception as exc:  # pragma: no cover - import/build failure path
        logging.warning("Failed to load native sm120 mega MoE kernel: %s", exc)
        _AVAILABLE = False
    return _AVAILABLE


def mega_moe_sm120_forward(
    hidden_states: torch.Tensor,
    slot_token_ids: torch.Tensor,
    slot_weights: torch.Tensor,
    block_experts: torch.Tensor,
    block_slot_starts: torch.Tensor,
    block_rows: torch.Tensor,
    num_blocks: torch.Tensor,
    expt_hist: torch.Tensor,
    stage1_weight: torch.Tensor,
    stage1_scale: torch.Tensor,
    stage2_weight: torch.Tensor,
    stage2_scale: torch.Tensor,
    output: torch.Tensor,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    mod = _load_module()
    mod.mega_moe_sm120_forward_cuda(
        hidden_states,
        slot_token_ids,
        slot_weights,
        block_experts,
        block_slot_starts,
        block_rows,
        num_blocks,
        expt_hist,
        stage1_weight,
        stage1_scale,
        stage2_weight,
        stage2_scale,
        output,
        float(swiglu_limit),
    )
    return output


__all__ = ["is_mega_moe_sm120_available", "mega_moe_sm120_forward"]
