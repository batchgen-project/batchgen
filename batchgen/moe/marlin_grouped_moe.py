"""Marlin grouped MoE Stage 1 kernel wrapper — zero per-step overhead.

v12c m_block_size_8 Marlin W4A16 for decode (M<=8 per expert).
All buffers and pointer arrays pre-computed at init time.
Per-step forward: 2 kernel launches (GEMM + SiLU), zero Python loops or allocations.
"""

import logging
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
    torch::Tensor scales_ptrs,
    torch::Tensor expert_starts, torch::Tensor expert_counts,
    int num_experts, int prob_n, int prob_k,
    torch::Tensor workspace, int num_matrices, int n_tiles);

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
    try:
        _load_module()
        return True
    except Exception:
        return False


def marlin_grouped_stage1_3d_inplace(
    dispatched_x_3d: torch.Tensor,
    intermediate_3d: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_starts: torch.Tensor,
    B_ptrs: torch.Tensor,
    scales_ptrs: torch.Tensor,
    C_ptrs: torch.Tensor,
    gate_buf: torch.Tensor,
    up_buf: torch.Tensor,
    N: int,
    K: int,
    workspace: torch.Tensor,
) -> None:
    """Zero-overhead Marlin S1: 2 kernel launches, zero Python loops/allocations.

    Args:
        dispatched_x_3d: [E*mtp, K] BF16 — input (read via expert_starts stride)
        intermediate_3d: [E*mtp, N] BF16 — output (SiLU result written here)
        expert_counts: [E] int32 GPU — actual tokens per expert (from dispatch, per-step)
        expert_starts: [E] int32 GPU — row offsets (pre-computed: arange(E)*mtp, init-time)
        B_ptrs: [2E] int64 — gate+up weight pointers (init-time)
        scales_ptrs: [2E] int64 — gate+up scale pointers (init-time)
        C_ptrs: [2E] int64 — output pointers: [gate_buf rows..., up_buf rows...] (init-time)
        gate_buf: [E*mtp, N] BF16 — pre-allocated gate output buffer (init-time)
        up_buf: [E*mtp, N] BF16 — pre-allocated up output buffer (init-time)
        N, K: dimensions
        workspace: [locks] int32 (init-time)
    """
    global _warned
    if not _warned:
        logging.info("[Marlin] Using Marlin W4A16 grouped GEMM for decode S1")
        _warned = True

    mod = _load_module()
    E = expert_counts.shape[0]
    n_tiles = N // 256

    # Launch 1: Grouped GEMM — all 2E matrices in one kernel
    mod.grouped_marlin_gemm(
        dispatched_x_3d, B_ptrs, C_ptrs, scales_ptrs,
        expert_starts, expert_counts,
        E, N, K, workspace, 2 * E, n_tiles,
    )

    # Launch 2: SiLU element-wise on entire buffer (padding is harmless)
    mod.silu_mul(gate_buf, up_buf, intermediate_3d)
