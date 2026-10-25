"""Re-export shim: the canonical home moved to ``batchgen_kernels.triton``.

The Triton kernel definitions and their launch wrappers that used to live in
this module now live in :mod:`batchgen_kernels.triton.fused_dequant_moe`, which
is where the Triton version-management, registry and test conventions are kept.
The relocation was byte-for-byte: no kernel body, signature, autotune config or
launch wrapper was changed, so compiled behaviour is unchanged.

This module stays as a thin re-export so existing call sites keep working.
Prefer importing from ``batchgen_kernels.triton.fused_dequant_moe`` in new code.
"""

from batchgen_kernels.triton.fused_dequant_moe import (  # noqa: F401
    fused_dequant_weighted_moe_stage_1,
    fused_dequant_weighted_moe_stage_1_kernel,
    fused_fp8_moe_stage_1,
    fused_fp8_moe_stage_1_bak,
    fused_fp8_moe_stage_1_kernel,
    fused_fp8_moe_stage_1_kernel_v2,
)
