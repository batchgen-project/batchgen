"""Re-export shim: the canonical home moved to ``batchgen_kernels.triton``.

The Triton kernel definitions and their launch wrappers that used to live in
this module now live in :mod:`batchgen_kernels.triton.fused_grouped_dequant_gemm`,
which is where the Triton version-management, registry and test conventions are
kept.  The relocation was byte-for-byte: no kernel body, signature, autotune
config or launch wrapper was changed, so compiled behaviour is unchanged.

This module stays as a thin re-export so existing call sites keep working.
Prefer importing from ``batchgen_kernels.triton.fused_grouped_dequant_gemm`` in
new code.
"""

from batchgen_kernels.triton.fused_grouped_dequant_gemm import (  # noqa: F401
    HAS_TENSOR_DESC,
    fused_dequant_grouped_gemm_bf16_fp8_kernel,
    fused_dequant_grouped_gemm_bf16_fp8_kernel_fp32,
    fused_dequant_grouped_gemm_bf16_fp8_kernel_v2,
    fused_dequant_grouped_gemm_bf16_fp8_triton,
    fused_dequant_grouped_gemm_bf16_fp8_triton_v2,
    fused_dequant_grouped_gemm_fp8_fp8_kernel,
    fused_dequant_grouped_gemm_fp8_fp8_triton,
    fused_dequant_grouped_gemm_fp8_tma,
    fused_dequant_grouped_gemm_fp8_tma_kernel,
    supports_tma,
)
