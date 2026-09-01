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

_KERNEL_MODULE_NAME = "batchgen_kernels.moe._C_fp8_blockwise_gemm"
_kernel_module = None
_warned_import = False
_warned_fused_s1 = False
_warned_ptrs = False
_warned_fused_s1_ptrs = False


def _get_kernel_module():
    global _kernel_module
    if _kernel_module is None:
        import batchgen_kernels
        _kernel_module = batchgen_kernels.load_extension(_KERNEL_MODULE_NAME)
    return _kernel_module


def _get_kernel():
    """Load the compiled FP8 blockwise GEMM kernel."""
    global _warned_import
    try:
        module = _get_kernel_module()
    except ImportError:
        module = None
    kernel = getattr(module, "fp8_blockwise_grouped_gemm", None)
    if kernel is None:
        if not _warned_import:
            _warned_import = True
            logger.warning(
                "FP8 blockwise grouped GEMM kernel not available "
                f"({_KERNEL_MODULE_NAME}). Calls requiring it will fail."
            )
        return None
    return kernel


def _get_fused_s1_kernel():
    """Load the compiled fused S1 kernel (gate+up+SiLU)."""
    global _warned_fused_s1
    try:
        module = _get_kernel_module()
    except ImportError:
        module = None
    kernel = getattr(module, "fp8_blockwise_fused_s1", None)
    if kernel is None:
        if not _warned_fused_s1:
            _warned_fused_s1 = True
            logger.warning(
                "FP8 fused S1 kernel not available — falling back to 2× GEMM + "
                "SiLU only when no persistent output buffer was supplied"
            )
        return None
    return kernel


def _get_ptrs_kernel():
    """Load the grouped GEMM ABI for independent expert weight addresses."""
    global _warned_ptrs
    try:
        module = _get_kernel_module()
    except ImportError:
        module = None
    kernel = getattr(module, "fp8_blockwise_grouped_gemm_ptrs", None)
    if kernel is None and not _warned_ptrs:
        _warned_ptrs = True
        logger.warning(
            "FP8 pointer-array grouped GEMM kernel not available "
            f"({_KERNEL_MODULE_NAME})"
        )
    return kernel


def _get_fused_s1_ptrs_kernel():
    """Load fused S1 for independently allocated expert weights."""
    global _warned_fused_s1_ptrs
    try:
        module = _get_kernel_module()
    except ImportError:
        module = None
    kernel = getattr(module, "fp8_blockwise_fused_s1_ptrs", None)
    if kernel is None and not _warned_fused_s1_ptrs:
        _warned_fused_s1_ptrs = True
        logger.warning(
            "FP8 pointer-array fused S1 kernel not available "
            f"({_KERNEL_MODULE_NAME})"
        )
    return kernel


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
    kernel = _get_kernel()
    if kernel is None:
        raise RuntimeError(
            "FP8 blockwise grouped GEMM kernel not compiled. "
            "Rebuild batchgen_kernels with SM90a support, or enable "
            "BATCHGEN_KERNELS_DEV=1 for a source checkout."
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

    Falls back to grouped_fp8_blockwise_s1_silu if the fused kernel is
    unavailable and no persistent ``output`` buffer was supplied.

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
            "FP8 fused S1 kernel not compiled, but output= was supplied. "
            "The unfused fallback cannot populate the caller's persistent buffer. "
            "Rebuild batchgen_kernels with SM90a support, or enable "
            "BATCHGEN_KERNELS_DEV=1 for a source checkout."
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
    )
