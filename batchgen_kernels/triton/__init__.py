"""Triton kernel collection for BatchGen inference.

Re-exports all public Triton kernels from their respective modules.
"""

from batchgen_kernels.triton.rmsnorm import fused_rmsnorm, FusedRMSNorm
from batchgen_kernels.triton.add_rmsnorm import fused_add_rmsnorm
from batchgen_kernels.triton.kv_cache import (
    run_paged_kv_token_update,
    run_paged_kv_token_update_fused,
)
from batchgen_kernels.triton.moe_weighted_sum import (
    moe_weighted_sum_triton,
    moe_weighted_sum_v3,
    moe_weighted_sum_triton_v2,
)
from batchgen_kernels.triton.fp8_quantize import (
    compressed_kv_bf16_to_fp8_per_token,
    per_token_blocked_quantize_bf16_to_fp8,
    compressed_kv_fp8_to_bf16_per_token,
    deepseek_v3_dequantization,
    dequant_compressed_kv_per_token_with_length,
    dequant_compressed_kv_per_token_with_length_v2,
)
from batchgen_kernels.triton.fused_rmsnorm_rope import (
    fused_rmsnorm_rope,
    fused_rmsnorm_rope_cache_update,
    fused_rmsnorm_rope_cache_update_with_q,
    fused_rmsnorm_rope_with_q,
    fused_rmsnorm_rope_with_q_native,
    fused_rmsnorm_rope_cache_update_with_q_return_new_kv,
)
from batchgen_kernels.triton.fused_dequant_gemm import fused_fp8_bf16_gemm
from batchgen_kernels.triton.fused_q_absorb import fused_q_absorb_query_states
from batchgen_kernels.triton.fused_out_absorb import fused_out_absorb_reshape
from batchgen_kernels.triton.int4_grouped_gemm import (
    int4_grouped_gemm,
    int4_moe_grouped_gemm,
)
