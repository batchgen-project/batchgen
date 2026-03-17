"""Marlin grouped MoE Stage 1 kernel wrapper.

v12c m_block_size_8 Marlin W4A16 for decode (M<=8 per expert).
Compiles via load_inline on first call. Targets SM90a (H20/H100).

Usage:
    from batchgen.moe.marlin_grouped_moe import marlin_grouped_stage1
    intermediate = marlin_grouped_stage1(
        sorted_hidden, expert_offsets,
        gate_ptrs, gate_scale_ptrs, up_ptrs, up_scale_ptrs,
        N, K, workspace)
"""

import logging
import os
from pathlib import Path

import torch

_module = None
_warned = False


def _load_module():
    """Compile Marlin grouped GEMM kernel via load_inline."""
    global _module
    if _module is not None:
        return _module

    cu_path = Path(__file__).parent / "marlin_grouped_gemm.cu"
    cuda_src = cu_path.read_text()

    launcher_code = r"""
#include <torch/extension.h>

void grouped_marlin_gemm(
    torch::Tensor A, torch::Tensor B_ptrs, torch::Tensor C_ptrs,
    torch::Tensor scales_ptrs, torch::Tensor expert_offsets,
    int num_experts, int prob_n, int prob_k,
    torch::Tensor workspace, int num_matrices, int n_tiles,
    bool use_atomic_add, bool use_fp32_reduce);

void silu_mul(torch::Tensor gate, torch::Tensor up, torch::Tensor out);
"""

    from torch.utils.cpp_extension import load_inline

    _module = load_inline(
        name="marlin_grouped_gemm",
        cpp_sources=[launcher_code],
        cuda_sources=[cuda_src],
        functions=["grouped_marlin_gemm", "silu_mul"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "-arch=sm_90a",
            "--use_fast_math",
            "-lineinfo",
            "-DUSE_BF16_COMPUTE",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        ],
        verbose=False,
    )
    return _module


def is_marlin_available() -> bool:
    """Check if Marlin grouped GEMM is available."""
    try:
        _load_module()
        return True
    except Exception:
        return False


def marlin_grouped_stage1(
    sorted_hidden: torch.Tensor,
    expert_offsets: torch.Tensor,
    gate_marlin_ptrs: torch.Tensor,
    gate_scale_ptrs: torch.Tensor,
    up_marlin_ptrs: torch.Tensor,
    up_scale_ptrs: torch.Tensor,
    N: int,
    K: int,
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Marlin grouped Stage 1: gate + up + SiLU for M<=8 decode.

    Accepts flat contiguous sorted activations with cumulative expert_offsets.

    Args:
        sorted_hidden: [total_dispatched, K] BF16 sorted activations (contiguous)
        expert_offsets: [num_experts+1] int32 cumulative token offsets
        gate/up_marlin_ptrs: [E] int64 device pointers to Marlin weights
        gate/up_scale_ptrs: [E] int64 device pointers to Marlin scales
        N: intermediate_size (output width per expert)
        K: hidden_size (input width)
        workspace: pre-allocated int32 lock buffer

    Returns:
        [total_dispatched, N] BF16 intermediate activations (after SiLU)
    """
    global _warned
    if not _warned:
        logging.info("[Marlin] Using Marlin W4A16 grouped GEMM for decode S1")
        _warned = True

    mod = _load_module()
    E = expert_offsets.shape[0] - 1
    num_matrices = 2 * E
    total_tokens = sorted_hidden.shape[0]
    device = sorted_hidden.device
    dtype = sorted_hidden.dtype

    # Build interleaved pointer arrays: [gate_0..gate_E-1, up_0..up_E-1]
    B_ptrs = torch.cat([gate_marlin_ptrs, up_marlin_ptrs])
    scales_ptrs = torch.cat([gate_scale_ptrs, up_scale_ptrs])

    # Output buffers
    C_gate = torch.empty(total_tokens, N, dtype=dtype, device=device)
    C_up = torch.empty(total_tokens, N, dtype=dtype, device=device)

    # Build C_ptrs: point directly into gate and up output rows
    C_ptrs = torch.zeros(num_matrices, dtype=torch.int64, device=device)
    bytes_per_row = N * 2  # BF16 = 2 bytes
    for e in range(E):
        row_start = expert_offsets[e].item()
        C_ptrs[e] = C_gate.data_ptr() + row_start * bytes_per_row
        C_ptrs[E + e] = C_up.data_ptr() + row_start * bytes_per_row

    n_tiles = N // 256  # TN=16, tile_size=16 → 256 cols per CTA

    # Launch grouped GEMM (single kernel for all 2E matrices)
    mod.grouped_marlin_gemm(
        sorted_hidden, B_ptrs, C_ptrs, scales_ptrs,
        expert_offsets, E, N, K, workspace,
        num_matrices, n_tiles, False, True,
    )

    # SiLU: silu(gate) * up
    silu_out = torch.empty_like(C_gate)
    mod.silu_mul(C_gate, C_up, silu_out)

    return silu_out


def marlin_grouped_stage1_from_3d(
    dispatched_x_3d: torch.Tensor,
    expert_counts: torch.Tensor,
    mtp: int,
    gate_marlin_ptrs: torch.Tensor,
    gate_scale_ptrs: torch.Tensor,
    up_marlin_ptrs: torch.Tensor,
    up_scale_ptrs: torch.Tensor,
    N: int,
    K: int,
    workspace: torch.Tensor,
    intermediate_3d: torch.Tensor = None,
) -> torch.Tensor:
    """Marlin grouped Stage 1 from 3D strided buffer layout.

    Handles the 3D strided → flat contiguous conversion for compatibility
    with the production WGMMA dispatch path. Copy overhead ~1us for decode.

    Args:
        dispatched_x_3d: [E*mtp, K] BF16 3D strided buffer
        expert_counts: [E] int32 tokens per expert
        mtp: max_tokens_padded (stride between expert slabs)
        intermediate_3d: [E*mtp, N] BF16 output buffer to write into (optional)
            If provided, writes Marlin output back into 3D strided format for S2.
        (other args same as marlin_grouped_stage1)

    Returns:
        If intermediate_3d is provided: intermediate_3d (modified in-place)
        Otherwise: [total_dispatched, N] flat BF16 intermediate
    """
    E = expert_counts.shape[0]
    device = dispatched_x_3d.device
    dtype = dispatched_x_3d.dtype

    # Build cumulative offsets and copy valid tokens to flat buffer
    total_tokens = expert_counts.sum().item()
    if total_tokens == 0:
        if intermediate_3d is not None:
            return intermediate_3d
        return torch.empty(0, N, dtype=dtype, device=device)

    flat_hidden = torch.empty(total_tokens, K, dtype=dtype, device=device)
    expert_offsets = torch.zeros(E + 1, dtype=torch.int32, device=device)

    offset = 0
    for e in range(E):
        cnt = expert_counts[e].item()
        if cnt > 0:
            src_start = e * mtp
            flat_hidden[offset:offset + cnt] = dispatched_x_3d[src_start:src_start + cnt]
        expert_offsets[e + 1] = offset + cnt
        offset += cnt

    # Run Marlin on flat data
    flat_intermediate = marlin_grouped_stage1(
        flat_hidden, expert_offsets,
        gate_marlin_ptrs, gate_scale_ptrs,
        up_marlin_ptrs, up_scale_ptrs,
        N, K, workspace,
    )

    # Copy results back to 3D strided format if needed
    if intermediate_3d is not None:
        offset = 0
        for e in range(E):
            cnt = expert_counts[e].item()
            if cnt > 0:
                dst_start = e * mtp
                intermediate_3d[dst_start:dst_start + cnt] = flat_intermediate[offset:offset + cnt]
            offset += cnt
        return intermediate_3d

    return flat_intermediate
