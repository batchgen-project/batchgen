"""Re-export shim: the canonical home moved to ``batchgen_kernels.triton``.

The Triton kernel definitions and their launch wrappers that used to live in
this module now live in :mod:`batchgen_kernels.triton.w8a8_grouped_gemm_stage_2`,
which is where the Triton version-management, registry and test conventions are
kept.  The relocation was byte-for-byte: no kernel body, signature, autotune
config or launch wrapper was changed, so compiled behaviour is unchanged.

This module stays as a thin re-export so existing call sites keep working.
Prefer importing from ``batchgen_kernels.triton.w8a8_grouped_gemm_stage_2`` in
new code.
"""

from batchgen_kernels.triton.w8a8_grouped_gemm_stage_2 import (  # noqa: F401
    fused_dequant_grouped_gemm_fp8_fp8_fp32_kernel,
    fused_dequant_grouped_gemm_fp8_fp8_fp32_triton,
    fused_dequant_grouped_gemm_fp8_fp8_kernel_optimized,
    fused_dequant_grouped_gemm_fp8_fp8_triton_optimized,
)
