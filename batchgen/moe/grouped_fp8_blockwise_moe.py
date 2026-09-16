"""FP8 Blockwise Grouped GEMM for MoE — CuTe persistent kernel wrapper.

Provides S1 (gate+up+SiLU) and S3 (down) grouped GEMM functions for
FP8 blockwise-scaled MoE layers. Uses pre-allocated reserved buffers
[M, dim] where expert e owns rows starting at cu_seqlens[e]: either the
uniform mtp-stride layout (cu_seqlens[e] = e * mtp) or a compact ragged
layout with 64-aligned cu_seqlens (see dispatch_scatter_ragged and
batchgen_kernels.moe._C_fp8_blockwise_ops.act_quant_ragged).

x_scale tiles are addressed per expert as cu_seqlens[e] / TileM. The kernel
traps on device if any cu_seqlens[e] is not TileM-aligned or an expert's
tiled span exceeds x_scale's columns. Host checks reject mismatched devices,
dtypes, ranks, non-contiguous tensors, and shapes before any launch.

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

_MODULE_NAME = "batchgen_kernels.moe._C_fp8_blockwise_gemm"

# Loaded once on first use; None after a failed import.
_module = None
_module_loaded = False
_warned_gemm = False
_warned_fused_s1 = False
_warned_ptrs = False
_warned_fused_s1_ptrs = False


def _get_module():
    """Load the compiled FP8 blockwise extension once (None if not built).

    Only ImportError is treated as "not available"; JIT/build/runtime errors
    propagate so they are not mistaken for a missing kernel.
    """
    global _module, _module_loaded
    if not _module_loaded:
        try:
            import batchgen_kernels
            _module = batchgen_kernels.load_extension(_MODULE_NAME)
        except ImportError as e:
            _module = None
            logger.warning(
                "FP8 blockwise kernel extension not available (%s): %s",
                _MODULE_NAME, e,
            )
        _module_loaded = True
    return _module


def _get_kernel():
    """Return the compiled FP8 blockwise grouped GEMM kernel, or None."""
    global _warned_gemm
    module = _get_module()
    kernel = getattr(module, "fp8_blockwise_grouped_gemm", None)
    if kernel is None and module is not None and not _warned_gemm:
        _warned_gemm = True
        logger.warning(
            "FP8 blockwise grouped GEMM symbol missing from %s", _MODULE_NAME
        )
    return kernel


def _get_fused_s1_kernel():
    """Return the compiled fused S1 kernel (gate+up+SiLU), or None."""
    global _warned_fused_s1
    module = _get_module()
    kernel = getattr(module, "fp8_blockwise_fused_s1", None)
    if kernel is None and module is not None and not _warned_fused_s1:
        _warned_fused_s1 = True
        logger.warning(
            "FP8 fused S1 symbol missing from %s", _MODULE_NAME
        )
    return kernel


def _get_ptrs_kernel():
    """Return grouped GEMM for independent expert weight addresses."""
    global _warned_ptrs
    module = _get_module()
    kernel = getattr(module, "fp8_blockwise_grouped_gemm_ptrs", None)
    if kernel is None and module is not None and not _warned_ptrs:
        _warned_ptrs = True
        logger.warning(
            "FP8 pointer-array grouped GEMM symbol missing from %s",
            _MODULE_NAME,
        )
    return kernel


def _get_fused_s1_ptrs_kernel():
    """Return fused S1 for independently allocated expert weights."""
    global _warned_fused_s1_ptrs
    module = _get_module()
    kernel = getattr(module, "fp8_blockwise_fused_s1_ptrs", None)
    if kernel is None and module is not None and not _warned_fused_s1_ptrs:
        _warned_fused_s1_ptrs = True
        logger.warning(
            "FP8 pointer-array fused S1 symbol missing from %s", _MODULE_NAME
        )
    return kernel


def require_grouped_fp8_blockwise_ptr_kernels():
    """Load both pointer-array entry points or fail before serving."""
    grouped = _get_ptrs_kernel()
    fused_s1 = _get_fused_s1_ptrs_kernel()
    if grouped is None or fused_s1 is None:
        missing = []
        if grouped is None:
            missing.append("fp8_blockwise_grouped_gemm_ptrs")
        if fused_s1 is None:
            missing.append("fp8_blockwise_fused_s1_ptrs")
        raise RuntimeError(
            "FP8 pointer-array grouped kernels are incomplete: "
            + ", ".join(missing)
        )
    return grouped, fused_s1


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
        x_fp8:      [M, K] fp8 — activations in reserved buffer (K % 128 == 0)
        weight_3d:  [E, N, K] fp8 — pre-stacked expert weights (N % 128 == 0)
        seqlens:    [E] int32 — actual tokens per expert
        cu_seqlens: [E+1] int32 — expert row offsets, each a multiple of the
                    selected TileM: uniform [0, mtp, ..., E*mtp] or ragged
                    64-aligned offsets
        x_scale:    [K/128, M_pad] f32 — transposed; M_pad <= M and
                    M_pad % TileM == 0
        w_scale_3d: [E, N/128, (K/128+3)//4*4] f32 — K-dim padded to 4
        num_seq_per_group_avg: int — controls TileM selection (16/32/64)
        output:     [M, N] bf16 — pre-allocated output (optional)
        tma_desc:   caller-owned 64-byte-aligned TMA scratch [2*E, 128]
                    (optional; descriptors are refreshed for current offsets)

    Returns:
        [M, N] bf16 output
    """
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


def grouped_fp8_blockwise_gemm_ptrs(
    x_fp8: Tensor,
    weight_prototype: Tensor,
    weight_ptrs: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    x_scale: Tensor,
    w_scale_prototype: Tensor,
    w_scale_ptrs: Tensor,
    num_seq_per_group_avg: int,
    output: Optional[Tensor] = None,
    tma_desc: Optional[Tensor] = None,
    tiles: Optional[Tensor] = None,
    cu_tiles: Optional[Tensor] = None,
) -> Tensor:
    """Grouped FP8 GEMM over independent core-engine weight allocations.

    ``weight_prototype`` and ``w_scale_prototype`` provide only shape, stride,
    dtype, and a valid descriptor seed. The CUDA preparation kernel replaces
    their addresses with ``weight_ptrs[e]`` / ``w_scale_ptrs[e]`` for every
    expert before launching the persistent grouped GEMM.
    """
    kernel = _get_ptrs_kernel()
    if kernel is None:
        raise RuntimeError(
            "FP8 pointer-array grouped GEMM kernel not compiled. Rebuild "
            "batchgen_kernels with SM90a support."
        )
    if 33 <= num_seq_per_group_avg <= 48:
        num_seq_per_group_avg = 64
    return kernel(
        x_fp8,
        weight_prototype,
        weight_ptrs,
        seqlens,
        cu_seqlens,
        x_scale,
        w_scale_prototype,
        w_scale_ptrs,
        num_seq_per_group_avg,
        output,
        tma_desc,
        tiles,
        cu_tiles,
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

    Falls back to grouped_fp8_blockwise_s1_silu if fused kernel unavailable
    and ``output`` is None; raises RuntimeError if ``output`` is supplied,
    since the allocating fallback cannot honor the persistent output buffer.
    Accepts the same uniform or ragged cu_seqlens / x_scale layouts as
    :func:`grouped_fp8_blockwise_gemm` (E*mtp below reads as M / M_pad).

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
    if output is not None:
        raise RuntimeError(
            "FP8 fused S1 kernel (fp8_blockwise_fused_s1) not available in "
            f"{_MODULE_NAME}, but a pre-allocated output buffer was supplied; "
            "the 2× GEMM + SiLU fallback allocates a new tensor and cannot "
            "write into it. Rebuild batchgen_kernels with the fused S1 kernel."
        )
    # Fallback: 2× GEMM + SiLU
    return grouped_fp8_blockwise_s1_silu(
        x_fp8, x_scale, gate_w3d, up_w3d,
        gate_ws3d, up_ws3d, seqlens, cu_seqlens,
        num_seq_per_group_avg,
    )


def grouped_fp8_blockwise_fused_s1_ptrs(
    x_fp8: Tensor,
    x_scale: Tensor,
    gate_w_prototype: Tensor,
    gate_weight_ptrs: Tensor,
    up_w_prototype: Tensor,
    up_weight_ptrs: Tensor,
    gate_ws_prototype: Tensor,
    gate_scale_ptrs: Tensor,
    up_ws_prototype: Tensor,
    up_scale_ptrs: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    num_seq_per_group_avg: int,
    output: Optional[Tensor] = None,
    tma_desc: Optional[Tensor] = None,
    tiles: Optional[Tensor] = None,
    cu_tiles: Optional[Tensor] = None,
) -> Tensor:
    """Fused gate+up+SiLU over streamed expert pointer arrays."""
    kernel = _get_fused_s1_ptrs_kernel()
    if kernel is None:
        raise RuntimeError(
            "FP8 pointer-array fused S1 kernel not compiled. Rebuild "
            "batchgen_kernels with SM90a support."
        )
    if 33 <= num_seq_per_group_avg <= 48:
        num_seq_per_group_avg = 64
    return kernel(
        x_fp8,
        gate_w_prototype,
        gate_weight_ptrs,
        up_w_prototype,
        up_weight_ptrs,
        seqlens,
        cu_seqlens,
        x_scale,
        gate_ws_prototype,
        gate_scale_ptrs,
        up_ws_prototype,
        up_scale_ptrs,
        num_seq_per_group_avg,
        output,
        tma_desc,
        tiles,
        cu_tiles,
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


def grouped_fp8_blockwise_s3_ptrs(
    x_fp8: Tensor,
    x_scale: Tensor,
    down_w_prototype: Tensor,
    down_weight_ptrs: Tensor,
    down_ws_prototype: Tensor,
    down_scale_ptrs: Tensor,
    seqlens: Tensor,
    cu_seqlens: Tensor,
    num_seq_per_group_avg: int,
    output: Optional[Tensor] = None,
    tma_desc: Optional[Tensor] = None,
    tiles: Optional[Tensor] = None,
    cu_tiles: Optional[Tensor] = None,
) -> Tensor:
    """S3 down projection for streamed, independently allocated experts."""
    return grouped_fp8_blockwise_gemm_ptrs(
        x_fp8,
        down_w_prototype,
        down_weight_ptrs,
        seqlens,
        cu_seqlens,
        x_scale,
        down_ws_prototype,
        down_scale_ptrs,
        num_seq_per_group_avg,
        output=output,
        tma_desc=tma_desc,
        tiles=tiles,
        cu_tiles=cu_tiles,
    )
