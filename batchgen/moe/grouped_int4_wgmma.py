# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
# ---------------------------------------------------------------------------- #

"""Grouped INT4 W4A16 MoE WGMMA kernels for K2.5 decode.

Self-contained module with embedded CUDA source. No external dependencies
beyond PyTorch and CUDA toolkit.

Stage 1: gate + up + SiLU fused. Stage 2: down projection.
All persistent experts processed in 2 kernel launches (vs 336 per-expert).

Architecture:
- 2 warpgroups: WG0 (producer: TMA A-load + INT4 decode B), WG1 (math: WGMMA)
- INT4 offset decode (nibble - 8) with BF16 scales
- TMA for A-matrix loads, manual decode for B-matrix
- 128B swizzle layout for WGMMA consumption
"""

import logging
import torch
from torch.utils.cpp_extension import load
import batchgen_kernels
from typing import Tuple, List, Optional

# Lazy-loaded CUDA module
_wgmma_module = None

def _get_wgmma_module():
    """Build and cache the grouped INT4 WGMMA CUDA module."""
    global _wgmma_module
    if _wgmma_module is not None:
        return _wgmma_module

    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)
    arch = f"-arch=sm_{cc[0]}{cc[1]}a"
    cuda_flags = ["-std=c++17", arch, "-O3", "--ptxas-options=-v", "-lineinfo"]

    _src_dir = batchgen_kernels.get_src_dir()
    _wgmma_module = load(
        name="grouped_int4_moe_wgmma_v1",
        sources=[str(_src_dir / "moe" / "grouped_int4_wgmma_ext.cu")],
        extra_cuda_cflags=cuda_flags,
        verbose=False,
    )
    logging.info("[WGMMA] Built grouped INT4 WGMMA module")
    return _wgmma_module


def setup_expert_weight_pointers(
    experts: list,
    experts_per_rank: int,
    routed_expert_start_idx: int,
    device: torch.device,
) -> dict:
    """Build pointer arrays for grouped WGMMA from persistent expert weights.

    Args:
        experts: nn.ModuleList of wrapped experts (KimiK25ExpertWrapper or None)
        experts_per_rank: Number of local experts per rank
        routed_expert_start_idx: Global index of first local expert
        device: GPU device

    Returns:
        Dict with pointer tensors and stride info.
    """
    E = experts_per_rank
    gate_ptrs = torch.zeros(E, dtype=torch.int64, device=device)
    gate_scale_ptrs = torch.zeros(E, dtype=torch.int64, device=device)
    up_ptrs = torch.zeros(E, dtype=torch.int64, device=device)
    up_scale_ptrs = torch.zeros(E, dtype=torch.int64, device=device)
    down_ptrs = torch.zeros(E, dtype=torch.int64, device=device)
    down_scale_ptrs = torch.zeros(E, dtype=torch.int64, device=device)

    for local_e in range(E):
        global_e = routed_expert_start_idx + local_e
        wrapper = experts[global_e]
        if wrapper is None:
            continue

        module = wrapper.module if hasattr(wrapper, 'module') else wrapper

        gate_ptrs[local_e] = module.int4_gate_packed.data_ptr()
        gate_scale_ptrs[local_e] = module.int4_gate_scale.data_ptr()
        up_ptrs[local_e] = module.int4_up_packed.data_ptr()
        up_scale_ptrs[local_e] = module.int4_up_scale.data_ptr()
        down_ptrs[local_e] = module.int4_down_packed.data_ptr()
        down_scale_ptrs[local_e] = module.int4_down_scale.data_ptr()

    # Get strides from the first valid expert
    first_expert = experts[routed_expert_start_idx]
    module = first_expert.module if hasattr(first_expert, 'module') else first_expert

    # Packed weights are int32 (4 bytes/elem), kernel uses uint8* (byte addressing)
    # Must convert element stride → byte stride for packed weights
    stride_gate_weight_n = module.int4_gate_packed.stride(0) * module.int4_gate_packed.element_size()
    stride_gate_scale_n = module.int4_gate_scale.stride(0)  # bf16 elements, kernel indexes as bf16*
    stride_down_weight_n = module.int4_down_packed.stride(0) * module.int4_down_packed.element_size()
    stride_down_scale_n = module.int4_down_scale.stride(0)  # bf16 elements

    return {
        "gate_ptrs": gate_ptrs,
        "gate_scale_ptrs": gate_scale_ptrs,
        "up_ptrs": up_ptrs,
        "up_scale_ptrs": up_scale_ptrs,
        "down_ptrs": down_ptrs,
        "down_scale_ptrs": down_scale_ptrs,
        "stride_gate_weight_n": stride_gate_weight_n,
        "stride_gate_scale_n": stride_gate_scale_n,
        "stride_down_weight_n": stride_down_weight_n,
        "stride_down_scale_n": stride_down_scale_n,
    }


def grouped_int4_moe_forward(
    hidden: torch.Tensor,
    expert_offsets: torch.Tensor,
    weight_ptrs: dict,
    N_intermediate: int,
    K: int,
) -> torch.Tensor:
    """Run grouped INT4 MoE stage1 + stage2.

    Args:
        hidden: Sorted tokens [total_tokens, K] bf16 (contiguous by expert)
        expert_offsets: [E+1] int32 cumulative token offsets
        weight_ptrs: Dict from setup_expert_weight_pointers()
        N_intermediate: MoE intermediate size (2048)
        K: Hidden size (7168)

    Returns:
        output: [total_tokens, K] bf16
    """
    mod = _get_wgmma_module()

    total_tokens = hidden.shape[0]
    if total_tokens == 0:
        return torch.empty(0, K, dtype=torch.bfloat16, device=hidden.device)

    # TMA requires gmem_rows >= BLOCK_M. Pad to BLOCK_M when total_tokens
    # is too small (e.g. EP decode with few tokens routed locally).
    _BLOCK_M = 64
    if total_tokens < _BLOCK_M:
        padded = torch.zeros(
            _BLOCK_M, K, dtype=hidden.dtype, device=hidden.device
        )
        padded[:total_tokens] = hidden
        hidden = padded
        total_tokens = _BLOCK_M

    max_m_tiles = (total_tokens + 63) // 64
    empty_bias = torch.empty(0, dtype=torch.int64, device=hidden.device)

    # Stage 1: gate + up + SiLU
    intermediate = mod.grouped_int4_moe_stage1(
        hidden, expert_offsets,
        weight_ptrs["gate_ptrs"], weight_ptrs["gate_scale_ptrs"],
        weight_ptrs["up_ptrs"], weight_ptrs["up_scale_ptrs"],
        empty_bias, empty_bias,
        N_intermediate,
        weight_ptrs["stride_gate_weight_n"],
        weight_ptrs["stride_gate_scale_n"],
        max_m_tiles,
    )

    # Stage 2: down projection
    output = mod.grouped_int4_moe_stage2(
        intermediate, expert_offsets,
        weight_ptrs["down_ptrs"], weight_ptrs["down_scale_ptrs"],
        empty_bias,
        K,
        weight_ptrs["stride_down_weight_n"],
        weight_ptrs["stride_down_scale_n"],
        max_m_tiles,
    )

    return output
