"""Re-export shim: the canonical home moved to ``batchgen_kernels.triton``.

The Triton kernel definitions and their launch wrappers that used to live in
this module now live in :mod:`batchgen_kernels.triton.w8a8_grouped_gemm_stage_1`,
which is where the Triton version-management, registry and test conventions are
kept.  The relocation was byte-for-byte: no kernel body, signature, autotune
config or launch wrapper was changed, so compiled behaviour is unchanged.

This module stays as a thin re-export so existing call sites keep working.
Prefer importing from ``batchgen_kernels.triton.w8a8_grouped_gemm_stage_1`` in
new code.

Only the module's functions and its immutable ``HAS_TENSOR_DESC`` probe are
re-exported.  The private one-shot allocator state (``_allocator_set``) is
deliberately not re-exported: it is mutated through a ``global`` in the
canonical module, and a shim-level alias would be a stale snapshot.
"""

from batchgen_kernels.triton.w8a8_grouped_gemm_stage_1 import (  # noqa: F401
    HAS_TENSOR_DESC,
    fused_fp8_moe_baseline_optimized_v2,
    fused_fp8_moe_bf16_epilogue_kernel,
    fused_fp8_moe_parallel_experts_kernel_no_activation,
    fused_fp8_moe_parallel_experts_kernel_optimized,
    fused_fp8_moe_parallel_experts_kernel_tma,
    fused_fp8_moe_persistent_descriptor_kernel_v2,
    fused_fp8_moe_persistent_kernel,
    fused_fp8_moe_pipelined_kernel,
    fused_fp8_moe_stage_1_baseline_v2,
    fused_fp8_moe_stage_1_bf16_epilogue,
    fused_fp8_moe_stage_1_no_activation,
    fused_fp8_moe_stage_1_optimized,
    fused_fp8_moe_stage_1_persistent,
    fused_fp8_moe_stage_1_persistent_v2,
    fused_fp8_moe_stage_1_pipelined,
    fused_fp8_moe_stage_1_tma,
    supports_tma,
)
