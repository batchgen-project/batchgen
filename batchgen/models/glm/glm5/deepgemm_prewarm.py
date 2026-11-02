# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""GLM-5 DeepGEMM prewarm: compile every DeepGEMM kernel a served GLM-5
batch can select before the worker serves, then report any later compile.

DeepGEMM JIT-compiles a kernel per template-argument set and picks the set on
the host from the call's shape (``batchgen_kernels.deepgemm_sm90_configs``
ports that selection). The entries below are added to the three phases of
the GLM-5 Triton prewarm (``triton_prewarm``), so the node-local partition,
the barrier and the every-rank load pass are the same; each entry launches
one representative shape with fresh zero tensors in the serving call's
argument layout.

Site table (values are GLM-5.2's; each is derived from the config fields
in ``dense_sites`` / ``masked_dense_sites`` / ``masked_moe_sites``):

====================  =========  =====  =====  ======  ===================  ========  ========
site                  kernel     N      K      groups  M or expected_m      A scales  phase
====================  =========  =====  =====  ======  ===================  ========  ========
q_a                   normal     2048   6144   1       prefill tokens,      K-major   prefill,
                                                       eager decode rows              decode
kv_a                  normal     576    6144   1       same                 K-major   prefill,
                                                                                      decode
q_b                   normal     16384  2048   1       same                 K-major   prefill,
                                                                                      decode
q_b_wq_b (merged)     normal     20480  2048   1       eager decode rows    K-major   decode
kv_b                  normal     28672  512    1       prefill tokens       K-major   prefill
o_proj                normal     6144   16384  1       prefill tokens,      K-major   prefill,
                                                       eager decode rows              decode
indexer_wk            normal     128    6144   1       prefill tokens       K-major   prefill
indexer_wk (graph)    normal     128    6144   1       decode bucket        TMA       decode
indexer_wq_b          normal     4096   2048   1       prefill tokens       K-major   prefill
dense_mlp_gate_up     normal     12288  6144   1       prefill tokens,      K-major   prefill,
                                                       decode rows                    decode
dense_mlp_down        normal     6144   12288  1       same                 K-major   prefill,
                                                                                      decode
expert_gate_up        normal     2048   6144   1       prefill tokens,      K-major   prefill,
(unfused)                                              decode rows                    decode
expert_down           normal     6144   2048   1       same                 K-major   prefill,
(unfused)                                                                             decode
shared_gate_up        normal     4096   6144   1       decode rows          TMA       decode
(fused)
shared_down (fused)   normal     6144   2048   1       decode rows          TMA       decode
qkv_a (masked)        masked     2624   6144   1       = bucket             TMA       geometry
q_b (masked)          masked     16384  2048   1       = bucket             TMA       geometry
q_b_wq_b (masked)     masked     20480  2048   1       = bucket             TMA       geometry
o_proj (masked)       masked     6144   16384  1       = bucket             TMA       geometry
moe_gate_up           masked     4096   6144   32      expected_m_per_      MN-major  geometry
                                                       expert(bucket)                 (DeepEP)
moe_down              masked     6144   2048   32      same                 MN-major  geometry
                                                                                      (DeepEP)
mqa_logits            fp8_mqa_logits: 32 heads x 128, compressed logits    prefill
paged_mqa             metadata + fp8_paged_mqa_logits: align(batch, 32)    geometry
                                                                           (page split)
====================  =========  =====  =====  ======  ===================  ========  ========

"K-major" scales are the row-major ``[M, K / 128]`` ones ``act_quant``
returns; DeepGEMM transposes them (``transpose_fp32``, keyed on
``ceil(K / 128)``) before the GEMM, so a K-major entry also compiles that
transpose. "TMA" scales are already MN-major and TMA-aligned
(``scale_tma_aligned=True``). The merged ``[q_b | wq_b]`` operand exists on
the indexer layers when ``qb_wqb_merge_policy`` is on; the reuse layers keep
the plain q_b. The unfused expert sites are the streamed prefill shared
expert and the non-resident routed experts of EP offloading; the fused shared
expert is the persistent decode one. ``CALL_SITES`` maps every DeepGEMM call
in the GLM-5 serving sources to these sites, and a CPU test fails when a
call appears that the table does not name.

Row ranges:

  * Normal kernels: every m in ``[1, max_rows]``, where ``max_rows`` is the
    larger of the prefill micro-batch token cap and the longest prompt (a
    prompt longer than the cap is its own micro-batch). Eager decode rows are
    assumed to stay below it. ``normal_representative_ms`` collapses the
    range to one m per kernel key; at GLM-5.2's shapes the key set stops
    growing well below the cap.
  * Masked GEMMs: ``expected_m`` is the captured bucket (or, for the MoE,
    ``expected_m_per_expert`` of it). Buckets never exceed the GPU page
    table's slot capacity (``Glm5DecodeGeometry.max_decode_rows``), so every
    row count up to it is covered; this is why these entries run in the
    geometry phase, which reads the KV managers.
  * Paged MQA: every ``align(batch, 32)`` up to that slot capacity, graph
    buckets and eager batches alike. An eager batch above the slot capacity
    is not covered.

Every entry allocates fresh tensors, except the MoE masked GEMM's B operands:
those are the resident ``[E, N, K]`` routed stacks, handed over read only
with every ``masked_m`` zero, so the kernel loads no tile of them (a fresh
copy would be close to a GiB per projection).
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from batchgen_kernels import deepgemm_sm90_configs as _dg
from batchgen_kernels.triton_prewarm import PrewarmEntry


logger = logging.getLogger(__name__)

LOG_TAG = "[GLM5_DEEPGEMM_WARMUP]"

# FP8 blockwise scale block of every GLM-5 GEMM.
SCALE_BLOCK = 128

# How a dense site hands DeepGEMM its operands.
VIA_W8A16 = "w8a16"      # fa3_backend.w8a16_gemm: act_quant inside, K-major
VIA_W8A8 = "w8a8"        # act_quant, then w8a8_deepgemm's unmasked path
VIA_TMA = "tma"          # TMA-aligned quantize, then deep_gemm.fp8_gemm_nt
VIA_WK = "wk"            # TMA-aligned act_quant, indexer_wk_proj_deepgemm_out

# Row extent of one expert's slab in the MoE masked entries: the masked
# kernel's block M, the alignment the DeepEP receive stride keeps.
MOE_MASKED_ROWS = 64

# fp32 scale bytes per token of the page-split FP8 indexer cache
# (``indexer_k_quant.SCALE_BYTES_PER_TOKEN``).
PAGE_SPLIT_SCALE_BYTES = 4

# fp8_mqa_logits entry: DeepGEMM rounds the query rows to 4, and the logits
# width is a multiple of 256 (``sparse_prefill._score_chunk_rows``).
MQA_ROWS = 4
MQA_SCORE_WIDTH = 256

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASE_GEOMETRY = "geometry"


class LateDeepGemmCompileError(RuntimeError):
    """DeepGEMM compiled a kernel after the prewarm under strict mode."""


# ============================================================================
#  Site table (pure)
# ============================================================================


@dataclass(frozen=True)
class DenseSite:
    """One ``fp8_gemm_nt`` call shape."""

    name: str
    n: int
    k: int
    via: str


@dataclass(frozen=True)
class MaskedSite:
    """One ``fp8_m_grouped_gemm_nt_masked`` call shape."""

    name: str
    n: int
    k: int
    groups: int


def dense_sites(shapes, phase: str, *, merge_q_b_wq_b: bool = True,
                wk_deepgemm: bool = True) -> List[DenseSite]:
    """The Normal-kernel sites a phase reaches, in table order.

    ``shapes`` is a ``triton_prewarm.Glm5PrewarmShapes``.
    """
    s = shapes
    expert_gate_up = DenseSite("expert_gate_up", s.expert_intermediate_size,
                               s.hidden_size, VIA_W8A16)
    expert_down = DenseSite("expert_down", s.hidden_size,
                            s.expert_intermediate_size, VIA_W8A16)
    dense_mlp = [
        DenseSite("dense_mlp_gate_up", s.intermediate_size, s.hidden_size,
                  VIA_W8A16),
        DenseSite("dense_mlp_down", s.hidden_size, s.intermediate_size,
                  VIA_W8A16),
    ]
    if phase == PHASE_PREFILL:
        sites = [
            DenseSite("q_a", s.q_lora_rank, s.hidden_size, VIA_W8A16),
            DenseSite("kv_a", s.kv_dim, s.hidden_size, VIA_W8A16),
            DenseSite("q_b", s.q_b_cols, s.q_lora_rank, VIA_W8A16),
            DenseSite("kv_b", s.kv_b_rows, s.kv_lora_rank, VIA_W8A16),
            DenseSite("o_proj", s.hidden_size, s.attn_out_dim, VIA_W8A16),
            DenseSite("indexer_wk", s.index_head_dim, s.hidden_size,
                      VIA_W8A16),
        ]
        if s.sparse_prefill:
            sites.append(DenseSite("indexer_wq_b", s.wq_b_cols, s.q_lora_rank,
                                   VIA_W8A16))
        # The streamed prefill shared expert runs the unfused chain.
        shared_gate_up = DenseSite("shared_gate_up_unfused",
                                   s.shared_intermediate_size, s.hidden_size,
                                   VIA_W8A16)
        shared_down = DenseSite("shared_down_unfused", s.hidden_size,
                                s.shared_intermediate_size, VIA_W8A16)
        return sites + dense_mlp + [shared_gate_up, shared_down]
    if phase != PHASE_DECODE:
        raise ValueError(f"no dense DeepGEMM sites in phase {phase!r}")
    sites = [
        DenseSite("q_a", s.q_lora_rank, s.hidden_size, VIA_W8A8),
        DenseSite("kv_a", s.kv_dim, s.hidden_size, VIA_W8A8),
        DenseSite("q_b", s.q_b_cols, s.q_lora_rank, VIA_W8A8),
    ]
    if merge_q_b_wq_b:
        sites.append(DenseSite("q_b_wq_b", s.q_b_cols + s.wq_b_cols,
                               s.q_lora_rank, VIA_W8A8))
    sites.append(DenseSite("o_proj", s.hidden_size, s.attn_out_dim, VIA_W8A8))
    if wk_deepgemm:
        sites.append(DenseSite("indexer_wk_graph", s.index_head_dim,
                               s.hidden_size, VIA_WK))
    sites += dense_mlp
    sites += [
        expert_gate_up,
        expert_down,
        DenseSite("shared_gate_up_fused", 2 * s.shared_intermediate_size,
                  s.hidden_size, VIA_TMA),
        DenseSite("shared_down_fused", s.hidden_size,
                  s.shared_intermediate_size, VIA_TMA),
    ]
    return sites


def masked_dense_sites(shapes, *, merge_q_b_wq_b: bool = True
                       ) -> List[MaskedSite]:
    """The decode graph's one-group masked GEMMs (``expected_m`` = bucket)."""
    s = shapes
    sites = [
        MaskedSite("qkv_a", s.qkv_a_width, s.hidden_size, 1),
        MaskedSite("q_b", s.q_b_cols, s.q_lora_rank, 1),
    ]
    if merge_q_b_wq_b:
        sites.append(MaskedSite("q_b_wq_b", s.q_b_cols + s.wq_b_cols,
                                s.q_lora_rank, 1))
    sites.append(MaskedSite("o_proj", s.hidden_size, s.attn_out_dim, 1))
    return sites


def masked_moe_sites(shapes, local_experts: int) -> List[MaskedSite]:
    """The DeepEP decode MoE's two masked GEMMs over the local experts."""
    s = shapes
    groups = int(local_experts)
    return [
        MaskedSite("moe_gate_up", 2 * s.expert_intermediate_size,
                   s.hidden_size, groups),
        MaskedSite("moe_down", s.hidden_size, s.expert_intermediate_size,
                   groups),
    ]


def expected_m_per_expert(*, local_rows: int, world_size: int, topk: int,
                          num_experts_global: int) -> int:
    """``moe_deepgemm_masked.expected_m_per_expert`` (imported there lazily
    to keep this module free of the MoE stack); the test pins the two."""
    pairs = int(local_rows) * int(world_size) * int(topk)
    return (pairs + int(num_experts_global)) // int(num_experts_global)


def paged_mqa_batches(max_rows: int) -> List[int]:
    """One batch per ``align(batch, 32)`` value up to ``max_rows``."""
    top = (int(max_rows) + 31) // 32 * 32
    return list(range(32, top + 1, 32))


@lru_cache(maxsize=None)
def _normal_reps(n: int, k: int, max_rows: int, num_sms: int) -> Tuple[int, ...]:
    return tuple(_dg.normal_representative_ms(n, k, max_rows, num_sms))


# ============================================================================
#  Inventory: every DeepGEMM call in the GLM-5 serving sources
# ============================================================================

# ``(path, enclosing function, callee) -> (call count, what it reaches)``.
# "wrapper" marks the call inside a helper whose own call sites are listed;
# "unreachable: ..." names why GLM-5 serving never runs it. The test scans
# ``INVENTORY_PATHS`` for calls of ``deep_gemm.<kernel>`` and of
# ``INVENTORY_CALLEES`` and requires this table to match exactly.
INVENTORY_PATHS = (
    "batchgen/models/glm",
    "batchgen/attention/dsa",
    "batchgen/attention/mla/fa3_backend.py",
    "batchgen/attention/mla/flashmla_backend.py",
    "batchgen/gemm/w8a8_deepgemm.py",
    "batchgen_kernels/attention/dsa",
)
INVENTORY_CALLEES = (
    "w8a8_deepgemm", "w8a16_gemm", "_gemm", "gemm",
    "_fp8_linear_from_quantized", "indexer_wk_proj_deepgemm_out",
    "run_deepgemm_masked_moe", "masked_gemm", "deep_gemm_paged_mqa_logits",
    "paged_mqa_logits_schedule_meta",
)
# ``deep_gemm`` attributes that launch no JIT kernel.
INVENTORY_NON_KERNELS = ("get_num_sms", "set_pdl", "get_pdl")

_FA3 = "batchgen/attention/mla/fa3_backend.py"
_FLASHMLA = "batchgen/attention/mla/flashmla_backend.py"
_SELECTOR = "batchgen/attention/dsa/glm5_decode_selector.py"
_GLM5 = "batchgen/models/glm/glm5/"
_SCORER = "batchgen_kernels/attention/dsa/paged_mqa_scorer.py"
_UNFUSED_EXPERT = ("expert_gate_up", "expert_down", "shared_gate_up_unfused",
                   "shared_down_unfused")

CALL_SITES: Dict[Tuple[str, str, str], Tuple[int, Any]] = {
    # Prefill attention (dense FA3 path; GLM-5.2/5.3 for prompts up to
    # index_topk), q_a, q_b, kv_a, kv_b, o_proj in call order.
    (_FA3, "mla_prefill_flashattention3_w8a16_deepgemm_prepacked", "_gemm"):
        (5, ("q_a", "q_b", "kv_a", "kv_b", "o_proj")),
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
     "w8a8_deepgemm"): (3, ("masked:qkv_a", "masked:q_b", "masked:q_b_wq_b",
                            "masked:o_proj")),
    (_GLM5 + "cuda_graph_segments.py", "Glm5FullDsaAttnSegment.forward",
     "indexer_wk_proj_deepgemm_out"): (1, ("indexer_wk_graph",)),
    (_GLM5 + "reuse_topk_segment.py", "Glm5ReuseTopkAttnSegment.forward",
     "w8a8_deepgemm"): (3, ("masked:qkv_a", "masked:q_b", "masked:o_proj")),
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
    (_GLM5 + "moe_cuda_graph_segments.py", "Glm5MoEGraphSegment._forward_deepep",
     "run_deepgemm_masked_moe"): (1, ("masked:moe_gate_up", "masked:moe_down")),
    (_GLM5 + "moe_deepgemm_masked.py", "run_deepgemm_masked_moe",
     "masked_gemm"): (2, "wrapper"),
    (_GLM5 + "wrappers.py", "GLM5ExpertWrapper._forward_unfused", "w8a16_gemm"):
        (3, _UNFUSED_EXPERT),
    (_GLM5 + "wrappers.py", "GLM5ExpertWrapper._forward_fused_gate_up",
     "deep_gemm.fp8_gemm_nt"): (2, ("shared_gate_up_fused",
                                    "shared_down_fused")),
    # Attention wrappers: eager decode and the legacy DSA graph's o_proj.
    (_GLM5 + "wrappers.py", "GLM5AttnWrapper._forward_decode_dsa_eager",
     "w8a8_deepgemm"): (1, ("o_proj",)),
    (_GLM5 + "wrappers.py", "GLM5AttnWrapper._project_dsa_attn_heads",
     "w8a8_deepgemm"): (1, ("o_proj",)),
    # Packed sparse prefill (GLM-5.2/5.3).
    (_GLM5 + "sparse_prefill.py", "_fp8_linear_from_quantized",
     "deep_gemm.fp8_gemm_nt"): (1, "wrapper"),
    (_GLM5 + "sparse_prefill.py", "_compute_indexer_kv_from_quantized_hidden",
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
     "indexer_wk_proj_deepgemm_out", "deep_gemm.fp8_gemm_nt"): (1, "wrapper"),
    (_SCORER, "deep_gemm_paged_mqa_logits", "deep_gemm.fp8_paged_mqa_logits"):
        (1, "wrapper"),
    (_SCORER, "paged_mqa_logits_schedule_meta",
     "deep_gemm.get_paged_mqa_logits_metadata"): (1, "wrapper"),
    (_SCORER, "paged_mqa_score_and_topk", "deep_gemm_paged_mqa_logits"):
        (1, ("paged_mqa",)),
    (_SCORER, "paged_mqa_score_and_topk", "paged_mqa_logits_schedule_meta"):
        (1, ("paged_mqa",)),
    (_SCORER, "paged_mqa_score_and_topk_prepared_out",
     "deep_gemm_paged_mqa_logits"): (1, ("paged_mqa",)),
    (_SCORER, "paged_mqa_step_prologue_out", "paged_mqa_logits_schedule_meta"):
        (1, ("paged_mqa",)),
    (_SCORER, "warm_paged_mqa_logits", "deep_gemm_paged_mqa_logits"):
        (1, ("paged_mqa",)),
    (_SCORER, "warm_paged_mqa_logits", "paged_mqa_logits_schedule_meta"):
        (1, ("paged_mqa",)),
}


def table_site_names() -> set:
    """Every site name the table defines, as ``CALL_SITES`` refers to them."""
    class _All:
        """Shapes with every optional site on."""

        def __getattr__(self, name):
            return 1

    shapes = _All()
    names = {site.name for phase in (PHASE_PREFILL, PHASE_DECODE)
             for site in dense_sites(shapes, phase)}
    names.update("masked:" + site.name for site in masked_dense_sites(shapes))
    names.update("masked:" + site.name for site in masked_moe_sites(shapes, 1))
    names.update(("mqa_logits", "paged_mqa"))
    return names


# ============================================================================
#  Entry runners: fresh tensors in the serving call's layout
# ============================================================================


def _valid_count(rows: int, device):
    return torch.full((1,), int(rows), dtype=torch.int32, device=device)


def _weight(n: int, k: int, device):
    """B operand: FP8 ``[N, K]`` and fp32 ``[ceil(N/128), ceil(K/128)]``
    block scales, both contiguous, as every GLM-5 weight is held."""
    weight = torch.zeros((n, k), dtype=torch.float8_e4m3fn, device=device)
    scale = torch.ones((-(-n // SCALE_BLOCK), -(-k // SCALE_BLOCK)),
                       dtype=torch.float32, device=device)
    return weight, scale


def _run_dense(site: DenseSite, rows: int, device,
               fns: Optional[Dict[str, Callable]] = None) -> None:
    """One Normal-kernel call of ``site`` at ``rows`` rows. ``fns`` replaces
    the serving functions (tests): ``gemm`` and, where the site quantizes
    outside the GEMM wrapper, ``quant``."""
    fns = dict(fns or {})
    x = torch.zeros((int(rows), site.k), dtype=torch.bfloat16, device=device)
    weight, scale = _weight(site.n, site.k, device)
    if site.via == VIA_W8A16:
        gemm = fns.get("gemm")
        if gemm is None:
            from batchgen.attention.mla.fa3_backend import w8a16_gemm as gemm
        gemm(weight, scale, x)
        return
    if site.via == VIA_W8A8:
        quant, gemm = fns.get("quant"), fns.get("gemm")
        if quant is None:
            from batchgen.attention.mla.fa3_backend import act_quant as quant
        if gemm is None:
            from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm as gemm
        x_fp8, x_scale = quant(x)
        gemm(x_fp8, x_scale, weight, scale)
        return
    out = torch.empty((int(rows), site.n), dtype=torch.bfloat16, device=device)
    if site.via == VIA_TMA:
        quant, gemm = fns.get("quant"), fns.get("gemm")
        if quant is None:
            from batchgen_kernels.triton.fp8_quantize import (
                per_token_blocked_quantize_bf16_to_fp8_1d as quant,
            )
        if gemm is None:
            import deep_gemm

            gemm = deep_gemm.fp8_gemm_nt
        x_fp8, x_scale = quant(x, scale_tma_aligned=True)
        gemm((x_fp8, x_scale), (weight, scale), out)
        return
    if site.via == VIA_WK:
        quant, gemm, wrap = fns.get("quant"), fns.get("gemm"), fns.get("wrap")
        if quant is None:
            from batchgen.attention.mla.fa3_backend import act_quant as quant
        if gemm is None or wrap is None:
            from batchgen_kernels.attention.dsa.indexer_wk_proj_deepgemm import (
                IndexerWkDeepGemmWeights,
                indexer_wk_proj_deepgemm_out,
            )
            gemm = gemm or indexer_wk_proj_deepgemm_out
            wrap = wrap or IndexerWkDeepGemmWeights
        x_fp8, x_scale = quant(x, num_valid_tokens=_valid_count(rows, device),
                               scale_tma_aligned=True)
        gemm(x_fp8, x_scale, wrap(weight, scale), out)
        return
    raise ValueError(f"unknown DeepGEMM site layout {site.via!r}")


def _run_masked_dense(site: MaskedSite, expected_m: int, device,
                      fns: Optional[Dict[str, Callable]] = None) -> None:
    """The decode graph's masked call: ``rows = expected_m = bucket``, the
    TMA-aligned valid-count cast, ``w8a8_deepgemm``'s masked path."""
    fns = dict(fns or {})
    quant, gemm = fns.get("quant"), fns.get("gemm")
    if quant is None:
        from batchgen.attention.mla.fa3_backend import act_quant as quant
    if gemm is None:
        from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm as gemm
    rows = int(expected_m)
    valid = _valid_count(rows, device)
    x = torch.zeros((rows, site.k), dtype=torch.bfloat16, device=device)
    x_fp8, x_scale = quant(x, num_valid_tokens=valid, scale_tma_aligned=True)
    weight, scale = _weight(site.n, site.k, device)
    out = torch.empty((rows, site.n), dtype=torch.bfloat16, device=device)
    gemm(x_fp8, x_scale, weight, scale, out=out, num_valid_tokens=valid,
         expected_m=rows)


def _run_masked_moe(weight, weight_scale, expected_m: int, device,
                    fn: Optional[Callable] = None) -> None:
    """One MoE masked GEMM against the resident ``[E, N, K]`` stack, laid
    out as ``run_deepgemm_masked_moe`` hands it: A ``[E, S, K]`` e4m3, its
    scales the (1, 2) transpose of a contiguous ``[E, K/128, S]`` block, D
    ``[E, S, N]`` bf16. Every ``masked_m`` is zero, so no tile is loaded."""
    if fn is None:
        from .moe_deepgemm_masked import masked_gemm_fn
        fn = masked_gemm_fn()
    groups, n, k = (int(v) for v in weight.shape)
    rows = MOE_MASKED_ROWS
    x = torch.zeros((groups, rows, k), dtype=torch.float8_e4m3fn, device=device)
    x_scale = torch.ones((groups, k // SCALE_BLOCK, rows), dtype=torch.float32,
                         device=device).transpose(1, 2)
    out = torch.empty((groups, rows, n), dtype=torch.bfloat16, device=device)
    masked_m = torch.zeros((groups,), dtype=torch.int32, device=device)
    fn((x, x_scale), (weight, weight_scale), out, masked_m, int(expected_m))


def _run_paged_mqa(batch: int, num_heads: int, head_dim: int, page_size: int,
                   device, fn: Optional[Callable] = None) -> None:
    """``get_paged_mqa_logits_metadata`` and ``fp8_paged_mqa_logits`` at one
    ``align(batch, 32)`` through the scorer's own warm helper, on a one-page
    page-split scratch cache (the template keys on the batch alignment and
    fixed model values, not on the cache or the context length)."""
    if fn is None:
        from batchgen_kernels.attention.dsa.paged_mqa_scorer import (
            warm_paged_mqa_logits as fn,
        )
    cache = torch.zeros((1, int(page_size), 1,
                         int(head_dim) + PAGE_SPLIT_SCALE_BYTES),
                        dtype=torch.uint8, device=device)
    fn(aux_blocked_k=cache, batch=int(batch), num_heads=int(num_heads),
       index_head_dim=int(head_dim), max_pages_per_seq=1,
       max_seqlen=int(page_size), page_size=int(page_size))


def _run_mqa_logits(num_heads: int, head_dim: int, device,
                    fn: Optional[Callable] = None) -> None:
    """The packed sparse prefill's scorer as ``_score_packed_indexer_topk_
    chunk`` calls it: e4m3 q ``[rows, H, D]``, e4m3 k ``[T, D]`` with fp32
    per-token scales ``[T]``, fp32 weights ``[rows, H]``, int32 ranges and
    compressed logits (``max_seqlen_k`` a multiple of 256)."""
    if fn is None:
        import deep_gemm

        fn = deep_gemm.fp8_mqa_logits
    q = torch.zeros((MQA_ROWS, int(num_heads), int(head_dim)),
                    dtype=torch.float8_e4m3fn, device=device)
    k = torch.zeros((MQA_SCORE_WIDTH, int(head_dim)),
                    dtype=torch.float8_e4m3fn, device=device)
    k_scale = torch.ones((MQA_SCORE_WIDTH,), dtype=torch.float32, device=device)
    weights = torch.zeros((MQA_ROWS, int(num_heads)), dtype=torch.float32,
                          device=device)
    starts = torch.zeros((MQA_ROWS,), dtype=torch.int32, device=device)
    ends = torch.ones((MQA_ROWS,), dtype=torch.int32, device=device)
    fn(q, (k, k_scale), weights, starts, ends,
       max_seqlen_k=MQA_SCORE_WIDTH, clean_logits=False)


# ============================================================================
#  Entry lists (deterministic: identical on every rank)
# ============================================================================


def _dense_entries(sites: Sequence[DenseSite], max_rows: int, num_sms: int,
                   device) -> List[PrewarmEntry]:
    """One entry per distinct ``(via, N, K)`` and representative m.

    The first entry of each predicted kernel key comes first, the entries
    whose key an earlier entry already compiles after it, so the node-local
    ranks do not compile one kernel twice at the same time.
    """
    first: List[PrewarmEntry] = []
    rest: List[PrewarmEntry] = []
    seen_sites = set()
    seen_keys = set()
    for site in sites:
        shape = (site.via, site.n, site.k)
        if shape in seen_sites:
            continue
        seen_sites.add(shape)
        for rows in _normal_reps(site.n, site.k, int(max_rows), int(num_sms)):
            key = _dg.best_config(rows, site.n, site.k,
                                  num_sms=num_sms).kernel_key()
            entry = PrewarmEntry(
                f"deepgemm/{site.via}/{site.n}x{site.k}/m{rows}",
                partial(_run_dense, site, rows, device),
            )
            if key in seen_keys:
                rest.append(entry)
            else:
                seen_keys.add(key)
                first.append(entry)
    return first + rest


def prefill_entries(shapes, device, *, max_rows: int,
                    num_sms: int) -> List[PrewarmEntry]:
    """Every prefill Normal site over ``[1, max_rows]`` and, on the packed
    sparse prefill, the indexer's ``fp8_mqa_logits``."""
    entries = _dense_entries(dense_sites(shapes, PHASE_PREFILL), max_rows,
                             num_sms, device)
    if shapes.sparse_prefill:
        entries.append(PrewarmEntry(
            f"deepgemm/mqa_logits/h{shapes.index_n_heads}"
            f"d{shapes.index_head_dim}",
            partial(_run_mqa_logits, shapes.index_n_heads,
                    shapes.index_head_dim, device),
        ))
    return entries


def decode_entries(shapes, device, *, max_rows: int, num_sms: int,
                   merge_q_b_wq_b: bool, wk_deepgemm: bool
                   ) -> List[PrewarmEntry]:
    """Every decode Normal site over ``[1, max_rows]``."""
    sites = dense_sites(shapes, PHASE_DECODE, merge_q_b_wq_b=merge_q_b_wq_b,
                        wk_deepgemm=wk_deepgemm)
    return _dense_entries(sites, max_rows, num_sms, device)


@dataclass(frozen=True)
class MoeStacks:
    """One routed layer's resident masked-GEMM B operands (read only)."""

    gate_up: Any
    gate_up_scale: Any
    down: Any
    down_scale: Any


def geometry_entries(shapes, device, *, max_decode_rows: int, num_sms: int,
                     merge_q_b_wq_b: bool, page_split: bool,
                     moe_stacks: Optional[MoeStacks] = None,
                     world_size: int = 1, topk: int = 1,
                     page_size: int = 64) -> List[PrewarmEntry]:
    """Masked GEMMs over every bucket up to ``max_decode_rows`` and the
    paged-MQA scorer over every 32-aligned batch up to it.

    ``moe_stacks`` (DeepEP only) adds the MoE masked GEMMs; its group count
    is the stack's leading dimension, the rank's local experts.
    """
    rows = range(1, int(max_decode_rows) + 1)
    entries: List[PrewarmEntry] = []
    for site in masked_dense_sites(shapes, merge_q_b_wq_b=merge_q_b_wq_b):
        for em in _dg.masked_representative_expected_ms(
                site.n, site.k, site.groups, rows, num_sms):
            entries.append(PrewarmEntry(
                f"deepgemm/masked/{site.n}x{site.k}/g1/em{em}",
                partial(_run_masked_dense, site, em, device),
            ))
    if moe_stacks is not None:
        groups = int(moe_stacks.gate_up.shape[0])
        expected = {expected_m_per_expert(
            local_rows=r, world_size=world_size, topk=topk,
            num_experts_global=groups * int(world_size)) for r in rows}
        for site, weight, scale in zip(
                masked_moe_sites(shapes, groups),
                (moe_stacks.gate_up, moe_stacks.down),
                (moe_stacks.gate_up_scale, moe_stacks.down_scale)):
            for em in _dg.masked_representative_expected_ms(
                    site.n, site.k, site.groups, expected, num_sms):
                entries.append(PrewarmEntry(
                    f"deepgemm/masked/{site.n}x{site.k}/g{groups}/em{em}",
                    partial(_run_masked_moe, weight, scale, em, device),
                ))
    if page_split:
        entries.extend(
            PrewarmEntry(
                f"deepgemm/paged_mqa/b{batch}",
                partial(_run_paged_mqa, batch, shapes.index_n_heads,
                        shapes.index_head_dim, page_size, device),
            )
            for batch in paged_mqa_batches(max_decode_rows)
        )
    return entries


# ============================================================================
#  Reading the live model and runtime (read only)
# ============================================================================


def deepgemm_num_sms() -> int:
    """The SM count DeepGEMM's heuristics see (``set_num_sms`` included)."""
    import deep_gemm

    return int(deep_gemm.get_num_sms())


_VERSION_CHECKED = False


def check_version(rank) -> None:
    """Log, once per process, whether the installed DeepGEMM is the version
    the config port reproduces. A mismatch still runs the prewarm."""
    global _VERSION_CHECKED
    if _VERSION_CHECKED:
        return
    _VERSION_CHECKED = True
    installed = _dg.installed_version()
    if installed != _dg.SUPPORTED_DEEPGEMM_VERSION:
        logger.warning(
            "%s rank=%s installed DeepGEMM %s, config model written for %s: "
            "the prewarm runs but may miss kernels this version selects",
            LOG_TAG, rank, installed, _dg.SUPPORTED_DEEPGEMM_VERSION,
        )


def moe_stacks(model) -> Optional[MoeStacks]:
    """The first routed layer's fused ``[E, 2I, H]`` / ``[E, H, I]`` stacks,
    when it is loaded in the DeepEP masked layout; None otherwise."""
    layers = getattr(getattr(model, "model", None), "layers", None) or []
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if not getattr(mlp, "fp8_gate_up_fused", False):
            continue
        gate_up = getattr(mlp, "fp8_gate_up_w3d", None)
        down = getattr(mlp, "fp8_down_w3d", None)
        if gate_up is None or down is None:
            continue
        return MoeStacks(
            gate_up=gate_up.view(torch.float8_e4m3fn),
            gate_up_scale=mlp.fp8_gate_up_ws3d,
            down=down.view(torch.float8_e4m3fn),
            down_scale=mlp.fp8_down_ws3d,
        )
    return None


# ============================================================================
#  Late-compile visibility
# ============================================================================


def jit_cache_dir() -> Path:
    """DeepGEMM's kernel cache: ``$DG_JIT_CACHE_DIR/cache``, by default
    ``~/.deep_gemm/cache``. Shared by the ranks of a node."""
    root = os.environ.get("DG_JIT_CACHE_DIR") or os.path.join(
        os.path.expanduser("~"), ".deep_gemm")
    return Path(root) / "cache"


def cache_kernel_count(path: Optional[Path] = None) -> int:
    """Compiled kernels in the cache: its ``kernel.*`` directories."""
    path = jit_cache_dir() if path is None else Path(path)
    try:
        names = os.listdir(path)
    except FileNotFoundError:
        return 0
    return sum(1 for name in names
               if name.startswith("kernel.") and (path / name).is_dir())


_BASELINE: Optional[int] = None


def record_baseline(rank) -> int:
    """The cache's kernel count once the geometry phase is done; later
    compiles are measured against it."""
    global _BASELINE
    _BASELINE = cache_kernel_count()
    logger.info("%s rank=%s DeepGEMM cache %s holds %d kernels after the "
                "prewarm", LOG_TAG, rank, jit_cache_dir(), _BASELINE)
    return _BASELINE


def check_late_compiles(rank, where: str, strict: bool = False) -> int:
    """Kernels DeepGEMM compiled since the baseline (0 before the first
    geometry phase). Logged at WARNING; ``strict`` raises instead. The cache
    is node-wide, so the count includes every rank of the node."""
    global _BASELINE
    if _BASELINE is None:
        return 0
    count = cache_kernel_count()
    late = count - _BASELINE
    if late <= 0:
        return 0
    message = (f"DeepGEMM compiled {late} kernels after the prewarm "
               f"(cache {jit_cache_dir()}: {_BASELINE} -> {count}, seen at "
               f"{where})")
    _BASELINE = count
    logger.warning("%s rank=%s %s", LOG_TAG, rank, message)
    if strict:
        raise LateDeepGemmCompileError(message)
    return late
