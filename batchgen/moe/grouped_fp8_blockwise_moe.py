"""FP8 Blockwise Grouped GEMM for MoE — CuTe persistent kernel wrapper.

Provides S1 (gate+up+SiLU) and S3 (down) grouped GEMM functions for
FP8 blockwise-scaled MoE layers. Uses pre-allocated reserved buffers
with uniform mtp-stride layout [E * mtp, dim].

Architecture: persistent 3-WG CuTe kernel, adaptive TileM (16/32/64),
TileN=128, TileK=128, 8-stage TMA pipeline, FastDivmod tile scheduling.

Usage:
    from batchgen.moe.grouped_fp8_blockwise_moe import (
        grouped_fp8_blockwise_s1_silu,
        grouped_fp8_blockwise_s3,
    )
"""

import logging
import torch
from torch import Tensor
from typing import Optional

logger = logging.getLogger("batchgen.moe.fp8_blockwise")

_warned_import = False
_warned_fused_s1 = False

_arch = None


def _get_arch() -> str:
    """Cached device arch ("sm100" / "sm90a" / ...) via batchgen_kernels."""
    global _arch
    if _arch is None:
        import batchgen_kernels as _bk
        _arch = _bk.get_device_arch()
    return _arch


def _grouped_fp8_blockwise_gemm_sm100(
    x_fp8: Tensor,
    weight_3d: Tensor,
    x_scale: Tensor,
    w_scale_3d: Tensor,
    output: Optional[Tensor] = None,
) -> Tensor:
    """SM100 (Blackwell) fallback for the FP8 blockwise grouped GEMM.

    The compiled SM90a CuTe kernel is unavailable on sm_100, and cuBLAS in
    torch 2.9+cu129 does not yet support 1x128/128x128 blockwise FP8 scaling
    (the heuristic returns CUBLAS_STATUS_NOT_SUPPORTED). Only row-wise FP8
    GEMM is supported, so we emulate deepseek-style blockwise scaling exactly
    by splitting the contraction dim K into 128-wide blocks and issuing one
    row-wise ``torch._scaled_mm`` per block, accumulating partials in fp32.

    Within a single K-block the activation scale is constant per token row
    (1x128) and the weight scale is constant per 128-output-row block
    (128x128, expanded here to per-output-column), so the row-wise GEMM is
    numerically identical to true blockwise scaling for that block.

    Processes the full uniform ``mtp`` reserved rows for every expert so the
    control flow is static (CUDA-graph compatible — no data-dependent shapes
    or host syncs on ``seqlens``). Padding rows produce values in output rows
    that downstream gather ignores.
    """
    E, N, K = weight_3d.shape
    g = 128
    assert K % g == 0, f"FP8 sm100 GEMM requires K (={K}) multiple of 128"
    assert N % g == 0, f"FP8 sm100 GEMM requires N (={N}) multiple of 128"
    EM = x_fp8.shape[0]
    assert EM % E == 0, f"x_fp8 rows (={EM}) not divisible by E (={E})"
    mtp = EM // E
    nblk = K // g
    assert x_scale.shape[0] >= nblk, (
        f"x_scale dim0 (={x_scale.shape[0]}) < K/128 (={nblk})")
    assert w_scale_3d.shape[1] == N // g, (
        f"w_scale dim1 (={w_scale_3d.shape[1]}) != N/128 (={N // g})")
    assert w_scale_3d.shape[2] >= nblk, (
        f"w_scale dim2 (={w_scale_3d.shape[2]}) < K/128 (={nblk})")

    if output is None:
        output = torch.empty((EM, N), dtype=torch.bfloat16, device=x_fp8.device)

    for e in range(E):
        start = e * mtp
        x_e = x_fp8[start:start + mtp]            # [mtp, K] fp8
        w_e = weight_3d[e]                        # [N, K] fp8
        xs_e = x_scale[:, start:start + mtp]      # [>=nblk, mtp] f32 (transposed)
        ws_e = w_scale_3d[e]                      # [N/128, >=nblk] f32
        acc = torch.zeros((mtp, N), dtype=torch.float32, device=x_fp8.device)
        for j in range(nblk):
            a_blk = x_e[:, j * g:(j + 1) * g]            # [mtp, 128] row-major view
            b_blk = w_e[:, j * g:(j + 1) * g].t()        # [128, N] col-major view
            sa = xs_e[j].contiguous().view(mtp, 1)       # [mtp, 1] act scale
            sb = ws_e[:, j].repeat_interleave(g)[:N].contiguous().view(1, N)
            o = torch._scaled_mm(
                a_blk, b_blk, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
            acc += o.float()
        output[start:start + mtp] = acc.to(torch.bfloat16)
    return output


def _get_kernel():
    """Load the compiled FP8 blockwise GEMM kernel."""
    global _warned_import
    try:
        from batchgen_kernels.moe._C_fp8_blockwise_gemm import (
            fp8_blockwise_grouped_gemm,
        )
        return fp8_blockwise_grouped_gemm
    except ImportError:
        if not _warned_import:
            _warned_import = True
            logger.warning(
                "FP8 blockwise grouped GEMM kernel not available "
                "(batchgen_kernels.moe._C_fp8_blockwise_gemm). "
                "Falling back to Triton implementation."
            )
        return None


def _get_fused_s1_kernel():
    """Load the compiled fused S1 kernel (gate+up+SiLU)."""
    global _warned_fused_s1
    try:
        from batchgen_kernels.moe._C_fp8_blockwise_gemm import (
            fp8_blockwise_fused_s1,
        )
        return fp8_blockwise_fused_s1
    except ImportError:
        if not _warned_fused_s1:
            _warned_fused_s1 = True
            logger.warning(
                "FP8 fused S1 kernel not available — falling back to 2× GEMM + SiLU"
            )
        return None


def grouped_fp8_blockwise_gemm(
    x_fp8: Tensor,
    weight_3d: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    x_scale: Tensor,
    w_scale_3d: Tensor,
    num_seq_per_group_avg: int,
    output: Optional[Tensor] = None,
    tma_desc: Optional[Tensor] = None,
) -> Tensor:
    """Single FP8 blockwise grouped GEMM.

    Args:
        x_fp8:      [E*mtp, K] fp8 — activations in reserved buffer
        weight_3d:  [E, N, K] fp8 — pre-stacked expert weights
        seqlens:    [E] int32 — actual tokens per expert
        cu_seqlens: [E+1] int32 — [0, mtp, 2*mtp, ..., E*mtp]
        x_scale:    [K/128, E*mtp] f32 — transposed, uniform mtp stride
        w_scale_3d: [E, N/128, (K/128+3)//4*4] f32 — K-dim padded to 4
        num_seq_per_group_avg: int — controls TileM selection (16/32/64)
        output:     [E*mtp, N] bf16 — pre-allocated output (optional)
        tma_desc:   cached TMA descriptors (optional, for reuse)

    Returns:
        [E*mtp, N] bf16 output
    """
    if _get_arch() == "sm100":
        return _grouped_fp8_blockwise_gemm_sm100(
            x_fp8, weight_3d, x_scale, w_scale_3d, output)

    kernel = _get_kernel()
    if kernel is None:
        raise RuntimeError(
            "FP8 blockwise grouped GEMM kernel not compiled. "
            "Rebuild batchgen_kernels with SM90a support."
        )

    # TileM=48 not supported (mtp multiple of 64 not divisible by 48)
    if 33 <= num_seq_per_group_avg <= 48:
        num_seq_per_group_avg = 64

    return kernel(
        x_fp8, weight_3d, seqlens, cu_seqlens,
        x_scale, w_scale_3d,
        num_seq_per_group_avg,
        output, tma_desc,
    )


def grouped_fp8_blockwise_s1_silu(
    x_fp8: Tensor,
    x_scale: Tensor,
    gate_w3d: Tensor,
    up_w3d: Tensor,
    gate_ws3d: Tensor,
    up_ws3d: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    num_seq_per_group_avg: int,
    gate_out: Optional[Tensor] = None,
    up_out: Optional[Tensor] = None,
) -> Tensor:
    """S1: gate + up projection with SiLU activation.

    Computes: silu(gate_proj(x)) * up_proj(x)

    Args:
        x_fp8:      [E*mtp, K] fp8 — quantized activations
        x_scale:    [K/128, E*mtp] f32 — transposed activation scales
        gate_w3d:   [E, N, K] fp8 — gate projection weights
        up_w3d:     [E, N, K] fp8 — up projection weights
        gate_ws3d:  [E, N/128, K/128_pad4] f32 — gate weight scales
        up_ws3d:    [E, N/128, K/128_pad4] f32 — up weight scales
        seqlens:    [E] int32
        cu_seqlens: [E+1] int32
        num_seq_per_group_avg: int
        gate_out:   [E*mtp, N] bf16 — pre-allocated (optional)
        up_out:     [E*mtp, N] bf16 — pre-allocated (optional)

    Returns:
        [E*mtp, N] bf16 — silu(gate) * up
    """
    gate_result = grouped_fp8_blockwise_gemm(
        x_fp8, gate_w3d, seqlens, cu_seqlens,
        x_scale, gate_ws3d, num_seq_per_group_avg,
        output=gate_out,
    )

    up_result = grouped_fp8_blockwise_gemm(
        x_fp8, up_w3d, seqlens, cu_seqlens,
        x_scale, up_ws3d, num_seq_per_group_avg,
        output=up_out,
    )

    # Fused SiLU: silu(gate) * up
    return torch.nn.functional.silu(gate_result) * up_result


def grouped_fp8_blockwise_fused_s1(
    x_fp8: Tensor,
    x_scale: Tensor,
    gate_w3d: Tensor,
    up_w3d: Tensor,
    gate_ws3d: Tensor,
    up_ws3d: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    num_seq_per_group_avg: int,
    output: Optional[Tensor] = None,
) -> Tensor:
    """Fused S1: gate GEMM + up GEMM + SiLU in single kernel launch.

    Two-phase CuTe persistent kernel (v19). Gate result stays in SMEM,
    SiLU applied in the epilogue. 1.75× faster than 2× GEMM + SiLU at decode.

    Falls back to grouped_fp8_blockwise_s1_silu if fused kernel unavailable.

    Args:
        x_fp8:      [E*mtp, K] fp8 — quantized activations
        x_scale:    [K/128, E*mtp] f32 — transposed activation scales
        gate_w3d:   [E, N, K] fp8 — gate projection weights
        up_w3d:     [E, N, K] fp8 — up projection weights
        gate_ws3d:  [E, N/128, K/128_pad4] f32 — gate weight scales
        up_ws3d:    [E, N/128, K/128_pad4] f32 — up weight scales
        seqlens:    [E] int32
        cu_seqlens: [E+1] int32
        num_seq_per_group_avg: int
        output:     [E*mtp, N] bf16 — pre-allocated (optional)

    Returns:
        [E*mtp, N] bf16 — silu(gate) * up
    """
    kernel = _get_fused_s1_kernel()
    if kernel is not None:
        return kernel(
            x_fp8, gate_w3d, up_w3d,
            seqlens, cu_seqlens, x_scale,
            gate_ws3d, up_ws3d,
            num_seq_per_group_avg,
            output,
        )
    # Fallback: 2× GEMM + SiLU
    return grouped_fp8_blockwise_s1_silu(
        x_fp8, x_scale, gate_w3d, up_w3d,
        gate_ws3d, up_ws3d, seqlens, cu_seqlens,
        num_seq_per_group_avg,
    )


def grouped_fp8_blockwise_s3(
    x_fp8: Tensor,
    x_scale: Tensor,
    down_w3d: Tensor,
    down_ws3d: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    num_seq_per_group_avg: int,
    output: Optional[Tensor] = None,
) -> Tensor:
    """S3: down projection.

    Computes: down_proj(x)

    Args:
        x_fp8:      [E*mtp, N] fp8 — quantized intermediate
        x_scale:    [N/128, E*mtp] f32 — transposed scales
        down_w3d:   [E, K, N] fp8 — down projection weights
        down_ws3d:  [E, K/128, N/128_pad4] f32 — weight scales
        seqlens:    [E] int32
        cu_seqlens: [E+1] int32
        num_seq_per_group_avg: int
        output:     [E*mtp, K] bf16 — pre-allocated (optional)

    Returns:
        [E*mtp, K] bf16
    """
    return grouped_fp8_blockwise_gemm(
        x_fp8, down_w3d, seqlens, cu_seqlens,
        x_scale, down_ws3d, num_seq_per_group_avg,
        output=output,
    )
