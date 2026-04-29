"""DSA (Dense Sparse Attention) indexer and absorb kernels for GLM-5 decode.

Requires H20 (SM90a) with WGMMA + TMA support.
All kernels use runtime CUDA compilation (load_inline) or Triton JIT.
"""

__all__ = []

# WP3: Fused paged gather (Triton)
try:
    from batchgen_kernels.attention.dsa.fused_paged_gather import (
        fused_dense_paged_gather,
        fused_paged_gather,
        fused_indexer_gather,
    )
    __all__ += ["fused_dense_paged_gather", "fused_paged_gather", "fused_indexer_gather"]
except (ImportError, Exception):
    pass

# Unified selected-KV selector (Triton)
try:
    from batchgen_kernels.attention.dsa.fused_unified_selector import (
        fused_select_mla_kv_bf16,
        fused_select_mla_kv_bf16_out,
    )
    __all__ += ["fused_select_mla_kv_bf16", "fused_select_mla_kv_bf16_out"]
except (ImportError, Exception):
    pass

# Synthetic selected-KV FlashMLA block-table fill (Triton)
try:
    from batchgen_kernels.attention.dsa.selected_block_table import (
        make_selected_block_table,
    )
    __all__ += ["make_selected_block_table"]
except (ImportError, Exception):
    pass

# WP2: Fused indexer KV proj (CUDA WGMMA)
try:
    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
        build_module,
        FP8IndexerWeightsCUDA,
        cuda_wk_proj_rmsnorm,
    )
    __all__ += ["build_module", "FP8IndexerWeightsCUDA", "cuda_wk_proj_rmsnorm"]
except (ImportError, Exception):
    pass

# WP4: Fused indexer scoring (CUDA WGMMA + CUDA RoPE/Hadamard + Triton)
try:
    from batchgen_kernels.attention.dsa.fused_indexer_score import (
        FP8WqbWeightsCUDA,
        fused_score_pipeline,
    )
    __all__ += ["FP8WqbWeightsCUDA", "fused_score_pipeline"]
except (ImportError, Exception):
    pass

# WP5: FP8 absorb (Triton WGMMA)
try:
    from batchgen_kernels.attention.dsa.fp8_absorb import (
        FP8AbsorbWeights,
        fp8_q_absorb,
        fp8_out_absorb,
    )
    __all__ += ["FP8AbsorbWeights", "fp8_q_absorb", "fp8_out_absorb"]
except (ImportError, Exception):
    pass
