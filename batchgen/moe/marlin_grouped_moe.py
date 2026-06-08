"""Marlin grouped MoE Stage 1 kernel wrapper — zero per-step overhead.

Two kernel variants:
- M8 (v12c): mma_trans, MBLOCK=8, decode M<=8. 80 regs, 32% occ, ~179us.
- M16 (v14): standard mma, MBLOCK=16, CTA M-tiling for any M. 130 regs, ~318us.
  Grid: num_matrices × max_m_tiles × n_tiles. GPU-side expert_counts dispatch.

Both use GROUP_BLOCKS=2 (gs=32, K2.5 native).
All buffers and pointer arrays pre-computed at init time.
Per-step forward: 2 kernel launches (GEMM + SiLU), zero Python loops or allocations.
"""

import logging

import torch

import batchgen_kernels.moe._C_marlin_grouped_gemm as _module

_warned_m8 = False
_warned_m16 = False


def _load_module():
    """Return pre-compiled Marlin grouped GEMM kernel from batchgen_kernels."""
    return _module


def is_marlin_available() -> bool:
    return _module is not None


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
    compact_stride: int = 16,
) -> None:
    """M8 path: Marlin S1 for decode (M<=8 per expert).

    Writes gate+up to compact buffers, then scatter SiLU to intermediate.
    """
    global _warned_m8
    if not _warned_m8:
        logging.info("[Marlin] Using M8 Marlin W4A16 grouped GEMM for decode S1")
        _warned_m8 = True

    mod = _load_module()
    E = expert_counts.shape[0]
    n_tiles = N // 256
    mtp = intermediate_3d.shape[0] // E

    # Launch 1: Grouped GEMM — all 2E matrices in one kernel
    mod.grouped_marlin_gemm(
        dispatched_x_3d, B_ptrs, C_ptrs, scales_ptrs,
        expert_starts, expert_counts,
        E, N, K, workspace, 2 * E, n_tiles,
    )

    # Launch 2: SiLU with scatter — compact gate/up → mtp-strided intermediate
    mod.silu_mul_scatter(
        gate_buf, up_buf, intermediate_3d, expert_counts,
        E, compact_stride, mtp, N,
    )


def marlin_grouped_stage1_fused(
    dispatched_x_3d: torch.Tensor,
    intermediate_3d: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_starts: torch.Tensor,
    gate_B_ptrs: torch.Tensor,
    gate_scales_ptrs: torch.Tensor,
    up_B_ptrs: torch.Tensor,
    up_scales_ptrs: torch.Tensor,
    C_ptrs: torch.Tensor,
    N: int,
    K: int,
    workspace: torch.Tensor,
    max_m_tiles: int,
    mtp: int,
    num_experts: int,
) -> None:
    """Fused S1: gate+up+SiLU in single kernel. No temp buffer.

    Each CTA does two sequential K-reductions (gate then up) for the same
    (expert, m_tile, n_tile). Gate result stored in SMEM, fused with SiLU
    in the write-back.

    Args:
        dispatched_x_3d: [E*mtp, K] BF16 input
        intermediate_3d: [E*mtp, N] BF16 output (SiLU(gate) * up written here)
        expert_counts: [E] int32 GPU
        expert_starts: [E] int32 GPU (= arange(E) * mtp)
        gate_B_ptrs: [E] int64 gate weight pointers
        gate_scales_ptrs: [E] int64 gate scale pointers
        up_B_ptrs: [E] int64 up weight pointers
        up_scales_ptrs: [E] int64 up scale pointers
        C_ptrs: [E] int64 output pointers (into intermediate at mtp stride)
        N, K: dimensions
        workspace: [locks] int32
        max_m_tiles: ceil(min(num_global, mtp) / 16)
        mtp: max tokens padded per expert
        num_experts: E
    """
    global _warned_m16
    if not _warned_m16:
        logging.info("[Marlin] Using fused M16 Marlin S1 (gate+up+SiLU, single kernel)")
        _warned_m16 = True

    mod = _load_module()
    n_tiles = N // 256

    mod.grouped_marlin_gemm_m16_s1(
        dispatched_x_3d,
        gate_B_ptrs, up_B_ptrs, C_ptrs,
        gate_scales_ptrs, up_scales_ptrs,
        expert_starts, expert_counts,
        num_experts, N, K, workspace, n_tiles, max_m_tiles,
    )


def single_expert_marlin_decode(
    x: torch.Tensor,
    gate_qw: torch.Tensor, gate_scale: torch.Tensor,
    up_qw: torch.Tensor, up_scale: torch.Tensor,
    down_qw: torch.Tensor, down_scale: torch.Tensor,
    N: int, K: int,
) -> torch.Tensor:
    """Grouped-Marlin W4A16 decode for ONE expert (E=1), for streamed/offloaded experts.

    The fused decode kernel is the same one the persistent path uses; here it runs over a
    single expert with `expert_starts=[0]`, `expert_counts=[t]`. The weight tensors come
    straight from the weight-buffer slot in **Marlin layout** — the kernel only reads their
    `.data_ptr()` (tensor .shape is ignored), so no Marlin->raw transform / reshape is needed.

    Args:
        x: [t, K] BF16 gathered tokens routed to this expert.
        *_qw / *_scale: the expert's Marlin INT4 packed weights (int32) + BF16 scales.
        N: moe_intermediate_size; K: hidden_size.
    Returns: [t, K] BF16.
    """
    mod = _load_module()
    device = x.device
    t = x.shape[0]
    x = x.contiguous()

    def _p(tensor):
        return torch.tensor([tensor.data_ptr()], dtype=torch.int64, device=device)

    gate_B, up_B, down_B = _p(gate_qw), _p(up_qw), _p(down_qw)
    gate_sB, up_sB, down_sB = _p(gate_scale), _p(up_scale), _p(down_scale)
    expert_starts = torch.zeros(1, dtype=torch.int32, device=device)
    expert_counts = torch.tensor([t], dtype=torch.int32, device=device)
    intermediate = torch.empty(t, N, dtype=torch.bfloat16, device=device)
    expert_out = torch.empty(t, K, dtype=torch.bfloat16, device=device)
    s1_C, s3_C = _p(intermediate), _p(expert_out)
    s1_ws = torch.zeros(N // 256 + 17, dtype=torch.int32, device=device)
    s3_ws = torch.zeros(K // 256 + 17, dtype=torch.int32, device=device)
    max_m_tiles = (t + 15) // 16

    # Stage 1: fused gate + up + SiLU -> intermediate [t, N]
    mod.grouped_marlin_gemm_m16_s1(
        x, gate_B, up_B, s1_C, gate_sB, up_sB,
        expert_starts, expert_counts, 1, N, K, s1_ws, N // 256, max_m_tiles)
    # Stage 3: down -> expert_out [t, K]
    mod.grouped_marlin_gemm_m16(
        intermediate, down_B, s3_C, down_sB, expert_starts, expert_counts,
        1, K, N, s3_ws, 1, K // 256, max_m_tiles)
    return expert_out


def marlin_grouped_stage1_unified(
    dispatched_x_3d: torch.Tensor,
    intermediate_3d: torch.Tensor,
    up_buf: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_starts: torch.Tensor,
    B_ptrs: torch.Tensor,
    scales_ptrs: torch.Tensor,
    C_ptrs: torch.Tensor,
    N: int,
    K: int,
    workspace: torch.Tensor,
    max_m_tiles: int,
    compact_stride: int,
    mtp: int,
    num_experts: int,
) -> None:
    """Unified M16 path: Marlin S1 for any M via CTA M-tiling.

    Gate output writes directly to intermediate (mtp stride).
    Up output writes to compact up_buf (compact_stride).
    Dual-stride SiLU fuses: intermediate = SiLU(gate_in_intermediate) * up_from_buf.

    Args:
        dispatched_x_3d: [E*mtp, K] BF16 input
        intermediate_3d: [E*mtp, N] BF16 output (gate writes here, SiLU in-place)
        up_buf: [E*compact_stride, N] BF16 temp buffer for up projection
        expert_counts: [E] int32 GPU — actual tokens per expert
        expert_starts: [E] int32 GPU — input row offsets (arange(E)*mtp)
        B_ptrs: [2E] int64 — gate+up weight pointers
        scales_ptrs: [2E] int64 — gate+up scale pointers
        C_ptrs: [2E] int64 — gate ptrs into intermediate, up ptrs into up_buf
        N, K: dimensions
        workspace: [locks] int32
        max_m_tiles: pigeonhole upper bound on M-tiles per expert
        compact_stride: up_buf rows per expert (= max_m_tiles * 16)
        mtp: gate stride in intermediate (max_tokens_padded)
        num_experts: E
    """
    global _warned_m16
    if not _warned_m16:
        logging.info("[Marlin] Using M16 Marlin W4A16 grouped GEMM (CTA M-tiling, any M)")
        _warned_m16 = True

    mod = _load_module()
    n_tiles = N // 256
    num_matrices = 2 * num_experts

    # Launch 1: M16 grouped GEMM with CTA M-tiling
    # Grid: num_matrices × max_m_tiles × n_tiles
    # Each CTA processes 16 rows, early-exits if beyond expert_counts[expert]
    mod.grouped_marlin_gemm_m16(
        dispatched_x_3d, B_ptrs, C_ptrs, scales_ptrs,
        expert_starts, expert_counts,
        num_experts, N, K, workspace, num_matrices, n_tiles, max_m_tiles,
    )

    # Launch 2: Dual-stride SiLU
    # Reads gate from intermediate (mtp stride), up from up_buf (compact_stride)
    # Writes SiLU(gate) * up in-place to intermediate
    mod.silu_mul_dual_stride(
        intermediate_3d, up_buf, expert_counts,
        num_experts, mtp, compact_stride, N,
    )
