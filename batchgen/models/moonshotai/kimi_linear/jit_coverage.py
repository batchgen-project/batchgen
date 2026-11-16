# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Kimi-Linear family (Kimi-K3 / K3.1) JIT coverage declarations.

Pure data, importable without torch, registered at import (the family's
``config`` imports this module, as ``glm5/config.py`` does for GLM-5):

* ``DEEP_GEMM_INVENTORY`` -- this family's part of the shared ``deep_gemm``
  call-site contract (``batchgen_kernels.jit_site_registry``). The paged-MQA
  scorer the K3.1 decode reaches lives in ``batchgen_kernels/attention/dsa``
  and is pinned by the GLM-5 part, which owns that scan root; the one direct
  DeepGEMM call in this family's own sources is the sparse prefill scorer.
  The repo-wide scanner holds this tree to the table below, so a new
  DeepGEMM call here fails CI until it is named and prewarmed.
* ``K31_DSA_JIT_KERNELS`` -- every JIT kernel a served Kimi-K3.1 batch
  reaches beyond Kimi-K3's, with the startup prewarm site(s) that compile its
  serving keys (``dsa_prewarm`` and ``triton_prewarm`` entry-name prefixes).
  The Triton names are what the late-compile recorder reports; the DeepGEMM
  names are the ``kernel.<family>.<hash>`` cache families the DeepGEMM
  late-compile guard reports. A late compile of a kernel listed here is a
  prewarm coverage defect against this table.
"""

from dataclasses import dataclass
from typing import Tuple

from batchgen_kernels.jit_site_registry import (
    BackendInventory,
    register_backend_inventory,
)

FAMILY = "kimi_linear"

_KIMI_LINEAR = "batchgen/models/moonshotai/kimi_linear"

# Prewarm entry-name prefixes (the entry names are ``<prefix>/<shape>``).
SITE_KDA_O_NORM = "kda_o_norm"
SITE_DSA_PREFILL_SELECT = "dsa_prefill_select"
SITE_DSA_SELECTED_POSITIONS = "dsa_selected_positions"
SITE_DSA_PAGED_MQA_PREPARE = "dsa_paged_mqa_prepare"
SITE_DSA_PAGED_MQA = "dsa_paged_mqa"

DEEP_GEMM_INVENTORY = register_backend_inventory(BackendInventory(
    backend="deep_gemm",
    paths=(_KIMI_LINEAR,),
    callees=(),
    non_kernels=("get_num_sms", "set_pdl", "get_pdl",
                 "get_m_alignment_for_contiguous_layout"),
    call_sites={
        # The causal prefill indexer scorer (packed and prefix-pool forms).
        (_KIMI_LINEAR + "/dsa_sparse_prefill.py", "_score_and_select_prefill",
         "deep_gemm.fp8_mqa_logits"): (1, (SITE_DSA_PREFILL_SELECT,)),
    },
    family=FAMILY,
))


PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"


@dataclass(frozen=True)
class JitKernelSlot:
    """One JIT kernel and the prewarm sites that compile its serving keys."""

    kernel: str
    backend: str          # "triton" | "deep_gemm"
    phases: Tuple[str, ...]
    sites: Tuple[str, ...]
    reached_by: str


K31_DSA_JIT_KERNELS: Tuple[JitKernelSlot, ...] = (
    JitKernelSlot(
        "layer_norm_gated_fwd_kernel", "triton", (PHASE_PREFILL,),
        (SITE_KDA_O_NORM,),
        "fla FusedRMSNormGated, the KDA o_norm over a whole prefill "
        "micro-batch; a prompt longer than the micro-batch cap is a "
        "micro-batch of its own, so the row range runs to the longest prompt",
    ),
    JitKernelSlot(
        "_per_token_blocked_quantize_1d_kernel", "triton", (PHASE_PREFILL,),
        (SITE_DSA_PREFILL_SELECT,),
        "act_quant of the index keys and the index queries in "
        "dsa_sparse_prefill._score_and_select_prefill",
    ),
    JitKernelSlot(
        "sm90_fp8_mqa_logits", "deep_gemm", (PHASE_PREFILL,),
        (SITE_DSA_PREFILL_SELECT,),
        "deep_gemm.fp8_mqa_logits in _score_and_select_prefill",
    ),
    JitKernelSlot(
        "_transform_selected_positions_kernel", "triton", (PHASE_DECODE,),
        (SITE_DSA_SELECTED_POSITIONS,),
        "dsa_decode.sparse_mla_attention: eager step (no valid count, "
        "index_topk wide) and graph step (valid count, graph score width)",
    ),
    JitKernelSlot(
        "_prepare_paged_mqa_inputs_kernel", "triton", (PHASE_DECODE,),
        (SITE_DSA_PAGED_MQA_PREPARE,),
        "paged_mqa_score_and_topk_out in the graph step "
        "(dsa_decode_graph_safe), keyed on the aux graph page-table width",
    ),
    JitKernelSlot(
        "sm90_paged_mqa_logits_metadata", "deep_gemm", (PHASE_DECODE,),
        (SITE_DSA_PAGED_MQA,),
        "get_paged_mqa_logits_metadata, templated on align(batch, 32), in "
        "the eager and graph scorers",
    ),
    JitKernelSlot(
        "sm90_fp8_paged_mqa_logits", "deep_gemm", (PHASE_DECODE,),
        (SITE_DSA_PAGED_MQA,),
        "fp8_paged_mqa_logits in the eager and graph scorers",
    ),
)
