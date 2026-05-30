# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""Single-expert INT4 W4A16 WGMMA kernels for K2.5 non-persistent experts.

Processes ONE expert at a time with fused INT4 dequant + WGMMA matmul.
Used in the per-expert loop for non-persistent (host-offloaded) experts.

Stage 1: gate + up + SiLU fused (2D grid: N_tiles × M_tiles)
Stage 2: down projection (2D grid: K_tiles × M_tiles)
"""

import logging
import torch

_single_expert_module = None
_arch = None


def _get_arch() -> str:
    """Cached device arch ('sm90a' / 'sm100')."""
    global _arch
    if _arch is None:
        import batchgen_kernels
        _arch = batchgen_kernels.get_device_arch()
    return _arch


def _get_single_expert_module():
    """Load the pre-compiled single-expert INT4 WGMMA CUDA module."""
    global _single_expert_module
    if _single_expert_module is not None:
        return _single_expert_module

    import batchgen_kernels
    _single_expert_module = batchgen_kernels.load_extension("batchgen_kernels.moe._C_single_expert_int4_wgmma")
    logging.info("[WGMMA] Loaded pre-compiled single-expert INT4 WGMMA module")
    return _single_expert_module


def single_expert_int4_forward(
    hidden: torch.Tensor,
    gate_packed: torch.Tensor,
    gate_scale: torch.Tensor,
    up_packed: torch.Tensor,
    up_scale: torch.Tensor,
    down_packed: torch.Tensor,
    down_scale: torch.Tensor,
) -> torch.Tensor:
    """Run single-expert INT4 stage1 (gate+up+SiLU) + stage2 (down).

    Args:
        hidden: [M, K=7168] bf16
        gate_packed: [N=2048, K//2] uint8/int32 (packed INT4)
        gate_scale: [N=2048, K//32] bf16
        up_packed: [N=2048, K//2] uint8/int32
        up_scale: [N=2048, K//32] bf16
        down_packed: [K=7168, N//2] uint8/int32
        down_scale: [K=7168, N//32] bf16

    Returns:
        output: [M, K=7168] bf16
    """
    # SM100 (Blackwell): the WGMMA INT4 .cu is not built — use the Triton MLP.
    if _get_arch() == "sm100":
        from batchgen_kernels.triton.fused_int4_grouped_silu import int4_expert_mlp
        gp = gate_packed.view(torch.uint8) if gate_packed.dtype == torch.int32 else gate_packed
        upp = up_packed.view(torch.uint8) if up_packed.dtype == torch.int32 else up_packed
        dp = down_packed.view(torch.uint8) if down_packed.dtype == torch.int32 else down_packed
        return int4_expert_mlp(
            hidden, gp, gate_scale, upp, up_scale, dp, down_scale, group_size=32,
        )

    mod = _get_single_expert_module()
    empty_bias = torch.empty(0, dtype=torch.bfloat16, device=hidden.device)

    # Cast packed weights to uint8 view if stored as int32
    # The kernel reads raw bytes via uint8_t* pointers
    if gate_packed.dtype == torch.int32:
        gate_packed = gate_packed.view(torch.uint8)
        up_packed = up_packed.view(torch.uint8)
        down_packed = down_packed.view(torch.uint8)

    # Stage 1: gate + up + SiLU → [M, N_intermediate]
    intermediate = mod.int4_single_expert_stage1(
        hidden, gate_packed, gate_scale, up_packed, up_scale,
        empty_bias, empty_bias,
    )

    # Stage 2: down → [M, K]
    output = mod.int4_single_expert_stage2(
        intermediate, down_packed, down_scale, empty_bias,
    )

    return output
