"""GLM-5 family JIT coverage declarations.

Pure data, importable without torch: the deep_gemm backend's static
call-site contract (scan roots, wrapper callees, non-kernel attributes, and
the pinned call-site table the repo-wide scanner holds the tree against) and
the family's operator slots. Registered with the shared
``batchgen_kernels.jit_site_registry`` at import, through the
``batchgen.models.jit_coverage`` seam.
"""

from batchgen_kernels.jit_site_registry import (
    BackendInventory,
    OperatorSlot,
    register_backend_inventory,
    register_model_type_family,
    register_slot,
)

_FA3 = "batchgen/attention/mla/fa3_backend.py"
_FLASHMLA = "batchgen/attention/mla/flashmla_backend.py"
_SELECTOR = "batchgen/attention/dsa/glm5_decode_selector.py"
_GLM5 = "batchgen/models/glm/glm5/"
_SCORER = "batchgen_kernels/attention/dsa/paged_mqa_scorer.py"
_UNFUSED_EXPERT = ("expert_gate_up", "expert_down", "shared_gate_up_unfused",
                   "shared_down_unfused")

DEEP_GEMM_INVENTORY = register_backend_inventory(BackendInventory(
    backend="deep_gemm",
    paths=(
        "batchgen/models/glm",
        "batchgen/attention/dsa",
        "batchgen/attention/mla/fa3_backend.py",
        "batchgen/attention/mla/flashmla_backend.py",
        "batchgen/gemm/w8a8_deepgemm.py",
        "batchgen/moe",
        "batchgen_kernels/attention/dsa",
    ),
    callees=(
        "w8a8_deepgemm", "w8a16_gemm", "_gemm", "gemm",
        "_fp8_linear_from_quantized", "indexer_wk_proj_deepgemm_out",
        "run_deepgemm_masked_moe", "masked_gemm", "deep_gemm_paged_mqa_logits",
        "paged_mqa_logits_schedule_meta", "grouped_fp8_deepgemm_gemm",
    ),
    non_kernels=("get_num_sms", "set_pdl", "get_pdl",
                 "get_m_alignment_for_contiguous_layout"),
    call_sites={
        # Prefill attention (dense FA3 path; GLM-5.2/5.3 for prompts up to
        # index_topk), q_a, q_b, kv_a, kv_b, o_proj in call order.
        (_FA3, "mla_prefill_flashattention3_w8a16_deepgemm_prepacked",
         "_gemm"): (5, ("q_a", "q_b", "kv_a", "kv_b", "o_proj")),
        (_FA3, "_mla_prefill_single_sequence_chunked", "gemm"):
            (5, ("kv_a", "kv_b", "q_a", "q_b", "o_proj")),
        (_FA3, "w8a16_gemm", "deep_gemm.fp8_gemm_nt"): (1, "wrapper"),
        (_FA3, "mla_prefill_flashattention3_w8a16_deepgemm", "w8a16_gemm"):
            (5, "unreachable: GLM-5 prefill requires prepack_mode"),
        (_FA3, "mla_prefill_w8a16_deepgemm", "w8a16_gemm"):
            (5, "unreachable: not bound by the GLM-5 manager"),
        (_FA3, "mla_prefill_flashattention3_fused_dequant", "w8a16_gemm"):
            (4, "unreachable: not bound by the GLM-5 manager"),
        (_FLASHMLA, "mla_decoding_flashmla_attn_mode_3_bf16", "w8a8_deepgemm"):
            (4, "unreachable: use_dense_mla only"),
        (_FLASHMLA, "mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv",
         "w8a8_deepgemm"): (4, "unreachable: use_dense_mla only"),
        (_FLASHMLA, "mla_decoding_flashmla_attn_mode_3_fp8_kv_bf16_attn",
         "w8a8_deepgemm"): (4, "unreachable: use_dense_mla only"),
        # Eager decode attention and the per-layer DSA graph's projections.
        (_SELECTOR, "build_glm5_dsa_flashmla_inputs", "w8a8_deepgemm"):
            (4, ("q_a", "q_b_wq_b", "q_b", "kv_a")),
        (_SELECTOR, "build_glm5_dsa_graph_segment_inputs", "w8a8_deepgemm"):
            (3, ("q_a", "q_b", "kv_a")),
        ("batchgen/gemm/w8a8_deepgemm.py", "w8a8_deepgemm",
         "deep_gemm.fp8_gemm_nt"): (1, "wrapper"),
        ("batchgen/gemm/w8a8_deepgemm.py", "w8a8_deepgemm", "masked_gemm"):
            (1, "wrapper"),
        # Decode graph segments: masked one-group GEMMs and the DeepGEMM wk.
        (_GLM5 + "cuda_graph_segments.py", "Glm5FullDsaAttnSegment.forward",
         "w8a8_deepgemm"): (3, ("masked:qkv_a", "masked:q_b",
                                "masked:q_b_wq_b", "masked:o_proj")),
        (_GLM5 + "cuda_graph_segments.py", "Glm5FullDsaAttnSegment.forward",
         "indexer_wk_proj_deepgemm_out"): (1, ("indexer_wk_graph",)),
        (_GLM5 + "reuse_topk_segment.py", "Glm5ReuseTopkAttnSegment.forward",
         "w8a8_deepgemm"): (3, ("masked:qkv_a", "masked:q_b",
                                "masked:o_proj")),
        # Indexer and MLP modules.
        (_GLM5 + "model.py", "Glm5Indexer.compute_indexer_kv", "w8a16_gemm"):
            (1, ("indexer_wk",)),
        (_GLM5 + "model.py", "Glm5Indexer.score_and_select", "w8a16_gemm"):
            (1, "unreachable: reference scorer, no GLM-5 caller"),
        (_GLM5 + "model.py", "Glm5Indexer.score_and_select_relu_gated",
         "w8a16_gemm"): (1, "unreachable: reference scorer, no GLM-5 caller"),
        (_GLM5 + "model.py", "Glm5Expert.deepgemm_forward", "w8a16_gemm"):
            (3, _UNFUSED_EXPERT),
        (_GLM5 + "model.py", "Glm5MLP._forward_slice", "w8a16_gemm"):
            (3, ("dense_mlp_gate_up", "dense_mlp_down")),
        # MoE.
        (_GLM5 + "moe_cuda_graph_segments.py",
         "Glm5MoEGraphSegment._forward_deepep", "run_deepgemm_masked_moe"):
            (1, ("masked:moe_gate_up", "masked:moe_down")),
        # Grouped contiguous prefill MoE (the DEFAULT S1/S3 backend since the
        # m-grouped contiguous adoption): the wrapper resolves the kernel
        # with getattr, which the scanner reads as this deep_gemm call site.
        ("batchgen/moe/grouped_fp8_deepgemm_moe.py", "_get_deepgemm",
         "deep_gemm.m_grouped_fp8_gemm_nt_contiguous"): (1, "wrapper"),
        (_GLM5 + "model.py", "Glm5MoE._forward_prefill_grouped_impl",
         "grouped_fp8_deepgemm_gemm"): (3, ("contiguous:moe_gate_up",
                                            "contiguous:moe_down")),
        (_GLM5 + "moe_deepgemm_masked.py", "run_deepgemm_masked_moe",
         "masked_gemm"): (2, "wrapper"),
        (_GLM5 + "moe_deepgemm_masked.py", "masked_gemm_fn",
         "deep_gemm.fp8_m_grouped_gemm_nt_masked"): (1, "wrapper"),
        # Attention wrappers: eager decode and the legacy DSA graph's o_proj.
        (_GLM5 + "wrappers.py", "GLM5ExpertWrapper._forward_unfused",
         "w8a16_gemm"): (3, _UNFUSED_EXPERT),
        (_GLM5 + "wrappers.py", "GLM5ExpertWrapper._forward_fused_gate_up",
         "deep_gemm.fp8_gemm_nt"): (2, ("shared_gate_up_fused",
                                        "shared_down_fused")),
        (_GLM5 + "wrappers.py", "GLM5AttnWrapper._forward_decode_dsa_eager",
         "w8a8_deepgemm"): (1, ("o_proj",)),
        (_GLM5 + "wrappers.py", "GLM5AttnWrapper._project_dsa_attn_heads",
         "w8a8_deepgemm"): (1, ("o_proj",)),
        # Packed sparse prefill (GLM-5.2/5.3).
        (_GLM5 + "sparse_prefill.py", "_fp8_linear_from_quantized",
         "deep_gemm.fp8_gemm_nt"): (1, "wrapper"),
        (_GLM5 + "sparse_prefill.py",
         "_compute_indexer_kv_from_quantized_hidden",
         "_fp8_linear_from_quantized"): (1, ("indexer_wk",)),
        (_GLM5 + "sparse_prefill.py", "select_packed_glm52_topk",
         "_fp8_linear_from_quantized"): (1, ("indexer_wq_b",)),
        (_GLM5 + "sparse_prefill.py", "glm52_sparse_prefill_prepacked",
         "_fp8_linear_from_quantized"): (3, ("q_a", "kv_a", "q_b")),
        (_GLM5 + "sparse_prefill.py", "glm52_sparse_prefill_prepacked",
         "w8a16_gemm"): (1, ("o_proj",)),
        (_GLM5 + "sparse_prefill.py", "_score_packed_indexer_topk_chunk",
         "deep_gemm.fp8_mqa_logits"): (1, ("mqa_logits",)),
        (_GLM5 + "sparse_prefill.py", "validate_glm52_sparse_prefill_runtime",
         "deep_gemm.fp8_mqa_logits"): (1, ("mqa_logits",)),
        # The DeepGEMM wk projection and the paged-MQA scorer.
        ("batchgen_kernels/attention/dsa/indexer_wk_proj_deepgemm.py",
         "indexer_wk_proj_deepgemm_out", "deep_gemm.fp8_gemm_nt"):
            (1, "wrapper"),
        (_SCORER, "deep_gemm_paged_mqa_logits",
         "deep_gemm.fp8_paged_mqa_logits"): (1, "wrapper"),
        (_SCORER, "paged_mqa_logits_schedule_meta",
         "deep_gemm.get_paged_mqa_logits_metadata"): (1, "wrapper"),
        (_SCORER, "paged_mqa_score_and_topk", "deep_gemm_paged_mqa_logits"):
            (1, ("paged_mqa",)),
        (_SCORER, "paged_mqa_score_and_topk",
         "paged_mqa_logits_schedule_meta"): (1, ("paged_mqa",)),
        (_SCORER, "paged_mqa_score_and_topk_prepared_out",
         "deep_gemm_paged_mqa_logits"): (1, ("paged_mqa",)),
        (_SCORER, "paged_mqa_step_prologue_out",
         "paged_mqa_logits_schedule_meta"): (1, ("paged_mqa",)),
        (_SCORER, "warm_paged_mqa_logits", "deep_gemm_paged_mqa_logits"):
            (1, ("paged_mqa",)),
        (_SCORER, "warm_paged_mqa_logits", "paged_mqa_logits_schedule_meta"):
            (1, ("paged_mqa",)),
    },
))


# Phase 1: the one slot with >=2 validated backends. No speculative entries.
PREFILL_GROUPED_GEMM_SLOT = register_slot(OperatorSlot(
    family="glm5",
    slot="prefill_grouped_gemm",
    backends={
        # The adopted m-grouped contiguous DeepGEMM backend; its key space
        # is the contiguous site family of the deep_gemm inventory.
        "deepgemm": ("jit", "deep_gemm"),
        # The incumbent CuTe pointer kernels: AOT, nothing compiles.
        "cute_ptrs": ("aot",),
    },
    default="deepgemm",
))



# Runtime model types of this family (prefix match in the preflight).
register_model_type_family("glm_moe_dsa", "glm5")
