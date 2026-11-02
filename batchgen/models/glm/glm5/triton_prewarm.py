# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""GLM-5 Triton prewarm: compile every reachable serving key before it is
needed, then log (or, under ``batchgen_debug.glm5_triton_strict``, refuse)
any later compilation. Each phase also carries the DeepGEMM entries of
``deepgemm_prewarm``, which documents their site table.

Triton specialises a ``@triton.jit`` function on its constexpr values, every
argument's dtype, and the integer class of every runtime int ("== 1",
"divisible by 16", other) and pointer alignment. The entries below drive the
serving wrappers (``act_quant``, the fused norms, the RoPE/KV-store kernel,
the DSA transform, scorer and KV writes, the router GEMM) with fresh,
zero-filled tensors whose dtypes, strides and pointer offsets are the ones
the serving call sites hand them. Nothing reads or writes a live KV pool,
page table or indexer cache; only the page-table WIDTHS and page sizes of the
GPU KV managers are read, because several decode kernels bake them as
constexprs.

Three phases, each once per process:

  * prefill (``configure_prefill``): blocked FP8 activation casts at every
    prefill GEMM width, the shared expert's SiLU*up cast, and on the packed
    sparse prefill (GLM-5.2 / 5.3) the top-k offset kernel and the kv_b
    weight dequant;
  * decode (``configure_decoding``): the decode graph's and eager decode's
    casts, the q_a norm (fused or not, per ``norm_quant_fuse_policy``), the
    eager RMSNorm+RoPE, the head gates, the split-K router and, with
    DeepEP, the masked SiLU*up cast;
  * geometry (``prewarm_decode_graphs``, right before CUDA-graph capture):
    the kernels whose keys depend on the page-table widths and the graph's
    score span. It re-runs when that geometry changes. The late-compile
    guard is armed at its end.

Each phase partitions its entries over the node-local ranks, which share one
Triton cache directory; after a barrier every rank runs every entry, loading
the binaries into its own in-memory kernel cache.

The pure class model (``Glm5PrewarmShapes``, ``quant_key``,
``Glm5DecodeGeometry``, ...) is what the CPU tests check the witness sets
against. Importable without Triton: every GPU dependency is imported inside
the function that launches it.
"""

import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from batchgen_kernels import triton_prewarm as _core
from batchgen_kernels.triton_prewarm import (  # noqa: F401 - re-exported
    DISARMED,
    LOG,
    STRICT,
    LateTritonCompileError,
    PrewarmEntry,
    int_class,
)


logger = logging.getLogger(__name__)

LOG_TAG = "[GLM5_TRITON_WARMUP]"

# Batch-level ``batchgen_debug`` key that turns a late compile into an error.
STRICT_DEBUG_KEY = "glm5_triton_strict"

# Row counts that reach every (rows, align4(rows)) class pair a blocked FP8
# cast can see: 1 -> ("1", "any"), 13 -> ("any", "d16"), 16 -> ("d16", "d16"),
# 17 -> ("any", "any"). align4(rows) is the row stride of the TMA-aligned
# MN-major scales; 13 is the one row count below 16 whose align4 is 16.
ROW_WITNESSES = (1, 13, 16, 17)

# Per-expert row extents of the DeepEP receive buffers. The masked SiLU*up
# cast reads its scales MN-major, so the extent is the scales' group stride:
# a multiple of the 4-float TMA alignment, either divisible by 16 or not.
MASKED_ROW_STRIDE_WITNESSES = (16, 20)

# Widths of the ACTIVE page table (``_GPUPageTableManager.gpu_table``), which
# eager decode hands the paged KV write as a runtime int.
PAGE_TABLE_COL_WITNESSES = (1, 16, 17)

# FP8 blockwise scale block of every GLM-5 activation cast.
SCALE_BLOCK = 128

# Packed sparse-attention prefill (``sparse_prefill.should_use_glm52_sparse_prefill``).
SPARSE_PREFILL_MODEL_TYPES = ("glm_moe_dsa_5_2", "glm_moe_dsa_5_3")

PHASE_PREFILL = "prefill"
PHASE_DECODE = "decode"
PHASE_GEOMETRY = "geometry"


class Glm5PrewarmGeometryError(RuntimeError):
    """The GPU KV managers do not expose the geometry the decode keys need."""


# ============================================================================
#  Shapes and integer classes (pure)
# ============================================================================


def next_power_of_2(value: int) -> int:
    """``triton.next_power_of_2`` for positive ints."""
    value = int(value)
    if value < 1:
        raise ValueError(f"next_power_of_2 needs a positive int, got {value}")
    return 1 << (value - 1).bit_length()


def align4(rows: int) -> int:
    """Row extent of the TMA-aligned MN-major scale allocation."""
    return (int(rows) + 3) // 4 * 4


def _require(config, name: str):
    value = getattr(config, name, None)
    if value is None:
        raise ValueError(f"GLM-5 Triton prewarm: model config has no {name}")
    return value


@dataclass(frozen=True)
class Glm5PrewarmShapes:
    """The model dimensions that select Triton keys."""

    hidden_size: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    num_heads: int
    intermediate_size: int
    shared_intermediate_size: int
    expert_intermediate_size: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    num_routed_experts: int
    max_position_embeddings: int
    sparse_prefill: bool

    @classmethod
    def from_config(cls, config) -> "Glm5PrewarmShapes":
        return cls(
            hidden_size=int(_require(config, "hidden_size")),
            q_lora_rank=int(_require(config, "q_lora_rank")),
            kv_lora_rank=int(_require(config, "kv_lora_rank")),
            qk_nope_head_dim=int(_require(config, "qk_nope_head_dim")),
            qk_rope_head_dim=int(_require(config, "qk_rope_head_dim")),
            v_head_dim=int(_require(config, "v_head_dim")),
            num_heads=int(_require(config, "num_attention_heads")),
            intermediate_size=int(_require(config, "intermediate_size")),
            shared_intermediate_size=(
                int(_require(config, "moe_intermediate_size"))
                * int(_require(config, "n_shared_experts"))
            ),
            expert_intermediate_size=int(
                _require(config, "moe_intermediate_size")
            ),
            index_n_heads=int(_require(config, "index_n_heads")),
            index_head_dim=int(_require(config, "index_head_dim")),
            index_topk=int(_require(config, "index_topk")),
            num_routed_experts=int(_require(config, "n_routed_experts")),
            max_position_embeddings=int(
                getattr(config, "max_position_embeddings", 0) or 0
            ),
            sparse_prefill=(
                getattr(config, "model_type", None) in SPARSE_PREFILL_MODEL_TYPES
            ),
        )

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def kv_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    @property
    def qkv_a_width(self) -> int:
        """Columns of the decode graph's fused q_a | kv_a GEMM output."""
        return self.q_lora_rank + self.kv_dim

    @property
    def attn_out_dim(self) -> int:
        return self.num_heads * self.v_head_dim

    @property
    def q_b_cols(self) -> int:
        return self.num_heads * self.q_head_dim

    @property
    def wq_b_cols(self) -> int:
        return self.index_n_heads * self.index_head_dim

    @property
    def kv_b_rows(self) -> int:
        return self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)


def quant_scale_strides(rows: int, dim: int, tma: bool) -> Tuple[int, int]:
    """``(stride_m, stride_b)`` of the scales ``per_token_blocked_quantize_
    bf16_to_fp8_1d`` allocates: plain ``[rows, blocks]``, or the transposed
    view of a ``[blocks, align4(rows)]`` allocation."""
    if tma:
        return 1, align4(rows)
    return -(-int(dim) // SCALE_BLOCK), 1


def quant_key(rows: int, dim: int, has_valid: bool, tma: bool) -> tuple:
    """What selects a ``_per_token_blocked_quantize_1d_kernel`` binary.

    ``dim`` fixes every constexpr and the input/output strides; the rest is
    the class of the runtime row count and of the two scale strides.
    """
    stride_m, stride_b = quant_scale_strides(rows, dim, tma)
    return (int(dim), bool(has_valid), bool(tma), int_class(rows),
            int_class(stride_m), int_class(stride_b))


def silu_mul_quant_key(rows: int, dim: int) -> tuple:
    """``_silu_mul_per_token_blocked_quantize_1d_kernel``: always the
    TMA-aligned scales, so the row count and align4(rows) select the key."""
    return (int(dim), int_class(rows), int_class(align4(rows)))


def rmsnorm_quant_key(rows: int, hidden: int, has_valid: bool,
                      tma: bool) -> tuple:
    """``_rmsnorm_fp8_quant_kernel``: the row count only sizes the grid."""
    stride_m, stride_b = quant_scale_strides(rows, hidden, tma)
    return (int(hidden), bool(has_valid), int_class(stride_m),
            int_class(stride_b))


@dataclass(frozen=True)
class QuantCombo:
    """One (width, valid-token gate, scale layout) of the blocked FP8 cast."""

    dim: int
    has_valid: bool
    tma: bool

    @property
    def name(self) -> str:
        layout = (
            "valid_tma" if self.has_valid and self.tma
            else "tma" if self.tma
            else "valid" if self.has_valid
            else "plain"
        )
        return f"act_quant/{layout}/d{self.dim}"


def plain_quant_dims(shapes: Glm5PrewarmShapes, phase: str) -> List[int]:
    """Widths cast with plain ``act_quant(x)`` (no valid count, row-major
    scales): every ``w8a16_gemm`` input, eager decode's attention casts, and
    on the packed sparse prefill its indexer q/k casts."""
    dims = {
        shapes.hidden_size,               # q_a/kv_a, dense MLP gate/up, indexer wk
        shapes.q_lora_rank,               # q_b
        shapes.attn_out_dim,              # o_proj
        shapes.intermediate_size,         # dense MLP down
        shapes.shared_intermediate_size,  # unfused shared-expert down
    }
    if phase == PHASE_PREFILL:
        dims.add(shapes.kv_lora_rank)     # kv_b over the normed latent
        if shapes.sparse_prefill:
            dims.add(shapes.index_head_dim)
    return sorted(dims)


def quant_combos(shapes: Glm5PrewarmShapes, phase: str) -> List[QuantCombo]:
    """Every blocked FP8 cast configuration a phase can reach."""
    combos = [QuantCombo(dim, False, False)
              for dim in plain_quant_dims(shapes, phase)]
    # The fused shared expert's x cast: TMA-aligned scales, no valid count.
    combos.append(QuantCombo(shapes.hidden_size, False, True))
    if phase == PHASE_DECODE:
        # The decode graph's casts: qkv_a input, q_a, o_proj input.
        combos.extend(
            QuantCombo(dim, True, True)
            for dim in sorted({shapes.hidden_size, shapes.q_lora_rank,
                               shapes.attn_out_dim})
        )
    return combos


def router_row_witnesses(row_range: Tuple[int, int]) -> Tuple[int, ...]:
    """Row counts covering every split-K router key. The split count is one
    value across the range and the rows and the partials' split stride
    ``rows * n_experts`` are runtime ints, so the keys differ only in the
    row count's class: the first row of each class the range contains."""
    lo, hi = max(int(row_range[0]), 1), int(row_range[1])
    return tuple(_core.class_representatives(range(lo, min(hi, lo + 16) + 1)))


def gather_block_t(batch: int, num_tokens: int) -> int:
    """``BLOCK_T`` of ``fused_dense_paged_gather`` (eager bf16-AoS scorer)."""
    total = int(batch) * int(num_tokens)
    if total <= 4096:
        return 1
    if total <= 32768:
        return 8
    return 32


def eager_score_seqlens(index_topk: int) -> Tuple[int, int]:
    """Score spans of the eager bf16-AoS scorer: it only runs for batches
    with a row longer than ``index_topk``, so the span exceeds it."""
    return int(index_topk) + 1, int(index_topk) + 16


@dataclass(frozen=True)
class Glm5DecodeGeometry:
    """Page-table widths and page sizes of the decode graph's KV managers."""

    primary_width: int
    aux_width: int
    primary_page_size: int
    aux_page_size: int
    aux_page_split: bool
    max_position_embeddings: int = 0
    # Slot capacity of the primary GPU page table: no captured bucket exceeds
    # it, and it bounds the DeepGEMM geometry entries' row counts.
    max_decode_rows: int = 0

    @property
    def graph_max_seqlen(self) -> int:
        """The score span the decode graph is captured with: the worker's
        ``_glm5_dsa_graph_score_capacity_tokens``."""
        capacities = [self.primary_width * self.primary_page_size,
                      self.aux_width * self.aux_page_size]
        if self.max_position_embeddings > 0:
            capacities.append(self.max_position_embeddings)
        capacity = min(capacities)
        if capacity <= 0:
            raise Glm5PrewarmGeometryError(
                f"GLM-5 decode geometry has no score capacity: {self}"
            )
        return capacity

    @property
    def paged_mqa_block_p(self) -> int:
        """``BLOCK_P`` of the paged-MQA input preparation (page-split aux)."""
        return min(256, next_power_of_2(self.aux_width))

    @property
    def score_block_s(self) -> int:
        """``BLOCK_S`` of the bf16-AoS paged scorer at the graph span."""
        return min(128, next_power_of_2(self.graph_max_seqlen))

    def key(self) -> tuple:
        return (self.primary_width, self.aux_width, self.primary_page_size,
                self.aux_page_size, self.aux_page_split,
                self.max_position_embeddings, self.max_decode_rows)


# ============================================================================
#  Unreachable kernels (section D): predicates logged at prewarm time
# ============================================================================


_RETIRED_NOTE = (
    "retired Glm5DsaAttnSegment kernels (_pack_flashmla_query_kernel, "
    "_unified_select_kernel, _fill_sequential_block_table_kernel, "
    "_paged_gather_kernel) and _fused_paged_score_kernel have no caller"
)


def describe_section_d(predicates: Dict[str, Any]) -> List[Tuple[int, str]]:
    """``(level, message)`` per predicate that keeps a kernel off the
    prewarm list. WARNING marks a predicate that makes such a kernel
    reachable in this configuration."""
    lines: List[Tuple[int, str]] = [(logging.INFO, _RETIRED_NOTE)]
    if "absorb_container" in predicates:
        container = predicates["absorb_container"]
        if container is None:
            lines.append((logging.INFO, "absorb_container=None: decode absorb "
                          "weights are not built on this instance"))
        elif container == "BF16AbsorbWeights":
            lines.append((logging.INFO, "absorb_container=BF16AbsorbWeights: "
                          "_absorb_gemv_kernel and _absorb_fp8_wgmma_kernel "
                          "are unreachable"))
        else:
            lines.append((logging.WARNING, f"absorb_container={container}: "
                          "_absorb_gemv_kernel/_absorb_fp8_wgmma_kernel are "
                          "reachable and NOT prewarmed"))
    if predicates.get("moe_3d_ready") is not None:
        if predicates["moe_3d_ready"]:
            lines.append((logging.INFO, "use_3d_moe and _fp8_blockwise_ready "
                          "on every MoE layer: _scatter_weight_reduce_kernel "
                          "and fused_dequant_grouped_gemm_fp8_tma_kernel are "
                          "unreachable"))
        else:
            lines.append((logging.WARNING, "a MoE layer lacks use_3d_moe and "
                          "_fp8_blockwise_ready: the _triton_compute fallback "
                          "(_scatter_weight_reduce_kernel, "
                          "fused_dequant_grouped_gemm_fp8_tma_kernel) is "
                          "reachable and NOT prewarmed"))
    if "use_dense_mla" in predicates:
        if predicates["use_dense_mla"]:
            lines.append((logging.WARNING, "use_dense_mla=True: the "
                          "flashmla_backend decode kernels are reachable and "
                          "NOT prewarmed"))
        else:
            lines.append((logging.INFO, "use_dense_mla=False: the "
                          "flashmla_backend decode kernels are unreachable"))
    if "index_topk" in predicates:
        topk = int(predicates["index_topk"])
        level = logging.INFO if topk == 2048 else logging.WARNING
        verdict = ("unreachable" if topk == 2048
                   else "reachable and NOT prewarmed")
        lines.append((level, f"index_topk={topk}: _topk_from_scores_kernel "
                      f"is {verdict}"))
    if predicates.get("hidden_size") is not None:
        hidden = int(predicates["hidden_size"])
        level = logging.INFO if hidden <= 8192 else logging.WARNING
        verdict = ("unreachable" if hidden <= 8192
                   else "reachable and NOT prewarmed")
        lines.append((level, f"hidden_size={hidden}: _rmsnorm_kernel_large "
                      f"is {verdict}"))
    if predicates.get("enable_deepep"):
        lines.append((logging.INFO, "enable_deepep=True: "
                      "_silu_mul_masked_group_quant_kernel is keyed on the "
                      "DeepEP receive buffers' per-expert row extent; "
                      f"prewarmed on extents {MASKED_ROW_STRIDE_WITNESSES}"))
    if "aux_page_split" in predicates:
        if predicates["aux_page_split"]:
            lines.append((logging.INFO, "aux cache page-split FP8: the "
                          "bf16-AoS scorer, its aux KV write and the eager "
                          "dense gather are unreachable"))
        else:
            lines.append((logging.INFO, "aux cache bf16 AoS: "
                          "_prepare_paged_mqa_inputs_kernel is unreachable"))
    if predicates.get("eager_width_constexpr"):
        lines.append((logging.INFO, "eager decode (over-bucket batches, "
                      "glm5_dsa_mode=eager, graph compare) passes the ACTIVE "
                      "page-table width, which follows the longest sequence, "
                      "as a constexpr to _transform_selected_positions_kernel "
                      "and _dense_paged_gather_kernel; those keys are "
                      "unbounded and NOT prewarmed"))
    return lines


def log_section_d(predicates: Dict[str, Any], rank) -> None:
    for level, message in describe_section_d(predicates):
        logger.log(level, "%s rank=%s %s", LOG_TAG, rank, message)


def _absorb_container_name(model) -> Optional[str]:
    for layer in _layers(model):
        container = getattr(getattr(layer, "self_attn", None),
                            "_fp8_absorb_weights", None)
        if container is not None:
            return type(container).__name__
    return None


def _moe_3d_ready(model) -> Optional[bool]:
    flags = [
        bool(getattr(mlp, "use_3d_moe", False))
        and bool(getattr(mlp, "_fp8_blockwise_ready", False))
        for mlp in (getattr(layer, "mlp", None) for layer in _layers(model))
        if mlp is not None and hasattr(mlp, "use_3d_moe")
    ]
    return all(flags) if flags else None


def decode_section_d_predicates(model, config) -> Dict[str, Any]:
    return {
        "absorb_container": _absorb_container_name(model),
        "moe_3d_ready": _moe_3d_ready(model),
        "use_dense_mla": bool(getattr(config, "use_dense_mla", False)),
        "index_topk": int(getattr(config, "index_topk", 2048)),
        "hidden_size": getattr(config, "hidden_size", None),
        "enable_deepep": bool(getattr(config, "enable_deepep", False)),
    }


# ============================================================================
#  Compile recorder and late-compile guard, bound to GLM-5
# ============================================================================


def strict_requested(debug) -> bool:
    """Whether a ``batchgen_debug`` dict sets ``glm5_triton_strict``."""
    return _core.strict_requested(debug, (STRICT_DEBUG_KEY,))


def _batch_debug() -> dict:
    """The served batch's ``batchgen_debug``: the dict the GLM-5 wrappers
    read their batch-level flags from (``wrappers._glm5_dsa_debug_dict``)."""
    try:
        from batchgen.models.wrappers.attention import AttnWrapperBase
    except ImportError:
        return {}
    debug = getattr(AttnWrapperBase, "batchgen_debug", None) or {}
    return debug if isinstance(debug, dict) else {}


def get_recorder() -> _core.TritonCompileRecorder:
    """The process-wide recorder (Triton's hooks are process-wide)."""
    return _core.get_recorder(
        log_tag=LOG_TAG, debug_source=_batch_debug,
        strict_keys=(STRICT_DEBUG_KEY,),
    )


# ============================================================================
#  Entry runners: fresh tensors through the serving wrappers
# ============================================================================


def _zeros(shape, dtype, device):
    return torch.zeros(shape, dtype=dtype, device=device)


def _valid_count(rows: int, device):
    return torch.full((1,), int(rows), dtype=torch.int32, device=device)


def _run_act_quant(combo: QuantCombo, device, rows_list=ROW_WITNESSES,
                   fn: Optional[Callable] = None) -> None:
    if fn is None:
        from batchgen.attention.mla.fa3_backend import act_quant as fn
    for rows in rows_list:
        x = _zeros((rows, combo.dim), torch.bfloat16, device)
        fn(
            x,
            num_valid_tokens=_valid_count(rows, device) if combo.has_valid else None,
            scale_tma_aligned=combo.tma,
        )


def _run_silu_mul_quant(dim: int, device, rows_list=ROW_WITNESSES,
                        fn: Optional[Callable] = None) -> None:
    if fn is None:
        from batchgen_kernels.triton.fp8_quantize import (
            silu_mul_per_token_blocked_quantize_bf16_to_fp8_1d as fn,
        )
    for rows in rows_list:
        fn(_zeros((rows, 2 * dim), torch.bfloat16, device),
           scale_tma_aligned=True)


def _run_offset_topk(topk: int, device, rows: int = 1,
                     fn: Optional[Callable] = None) -> None:
    """``offset_packed_topk_indices_`` over one score chunk; ``n_elements``
    is a multiple of ``topk`` and therefore always divisible by 16."""
    if fn is None:
        from .sparse_prefill import offset_packed_topk_indices_ as fn
    fn(_zeros((rows, topk), torch.int32, device),
       _zeros((rows,), torch.int32, device))


def _run_weight_dequant(rows: int, cols: int, dtype, device,
                        fn: Optional[Callable] = None) -> None:
    """The packed sparse prefill's per-layer kv_b dequant, at the live
    weight's shape and dtype."""
    if fn is None:
        from batchgen.attention.mla.flashmla_backend import (
            deepseek_v3_dequantization as fn,
        )
    weight = _zeros((rows, cols), dtype, device)
    scale = torch.ones(
        (-(-rows // SCALE_BLOCK), -(-cols // SCALE_BLOCK)),
        dtype=torch.float32, device=device,
    )
    fn(weight, scale)


def _run_q_a_norm(shapes: Glm5PrewarmShapes, weight, eps: float, fused: bool,
                  device, rows_list=ROW_WITNESSES,
                  fn: Optional[Callable] = None) -> None:
    """The decode graph's q_a norm: its input is the q_a column slice of the
    fused qkv_a output, so its row stride is the qkv_a width."""
    if fn is None:
        if fused:
            from batchgen_kernels.triton.rmsnorm_fp8_quant_fused import (
                fused_rmsnorm_fp8_quant as fn,
            )
        else:
            from batchgen_kernels.triton.rmsnorm import fused_rmsnorm as fn
    for rows in rows_list:
        qkv_a = _zeros((rows, shapes.qkv_a_width), torch.bfloat16, device)
        q_a = qkv_a[:, : shapes.q_lora_rank]
        out = torch.empty((rows, shapes.q_lora_rank), dtype=torch.bfloat16,
                          device=device)
        if fused:
            fn(q_a, weight, eps, out=out,
               num_valid_tokens=_valid_count(rows, device),
               scale_tma_aligned=True)
        else:
            fn(q_a, weight, eps, out=out)


def _rope_tables(shapes: Glm5PrewarmShapes, device):
    """FP32 cos/sin rows, the dtype of ``Glm5RotaryEmbedding``'s cache; the
    entries read position 0 only."""
    cos = _zeros((1, shapes.qk_rope_head_dim), torch.float32, device)
    return cos, torch.zeros_like(cos)


def _run_rope_eager(shapes: Glm5PrewarmShapes, weight, eps: float, device,
                    rows_list=ROW_WITNESSES,
                    fn: Optional[Callable] = None) -> None:
    """``glm5_decode_selector``'s RMSNorm+RoPE: a dense GEMM output, the
    contiguous q_pe, and no mask, split source or cache store."""
    if fn is None:
        from batchgen.attention.mla.fused_rmsnorm_rope import (
            fused_rmsnorm_rope_with_q_native as fn,
        )
    cos, sin = _rope_tables(shapes, device)
    for rows in rows_list:
        fn(
            _zeros((rows, 1, shapes.kv_dim), torch.bfloat16, device),
            _zeros((rows, shapes.num_heads, 1, shapes.qk_rope_head_dim),
                   torch.bfloat16, device),
            cos, sin,
            _zeros((rows, 1), torch.int64, device),
            weight, shapes.kv_lora_rank, shapes.qk_rope_head_dim, eps=eps,
        )


def _run_rope_graph(shapes: Glm5PrewarmShapes, weight, eps: float,
                    q_row_width: int, page_table_width: int, page_size: int,
                    cache_dtype, device, rows_list=ROW_WITNESSES,
                    fn: Optional[Callable] = None) -> None:
    """The decode graph's RMSNorm+RoPE with the row mask, the strided q_pe
    source (a column slice of the q_b output, row stride ``q_row_width``)
    and the folded primary-cache store into a one-page scratch cache."""
    if fn is None:
        from batchgen.attention.mla.fused_rmsnorm_rope import (
            fused_rmsnorm_rope_with_q_native as fn,
        )
    cos, sin = _rope_tables(shapes, device)
    for rows in rows_list:
        qkv_a = _zeros((rows, shapes.qkv_a_width), torch.bfloat16, device)
        q_out = _zeros((rows, q_row_width), torch.bfloat16, device)
        q_heads = q_out[:, : shapes.q_b_cols].unflatten(
            -1, (shapes.num_heads, shapes.q_head_dim)
        )
        fn(
            qkv_a[:, shapes.q_lora_rank:].view(rows, 1, shapes.kv_dim),
            _zeros((rows, shapes.num_heads, 1, shapes.qk_rope_head_dim),
                   torch.bfloat16, device),
            cos, sin,
            _zeros((rows, 1), torch.int64, device),
            weight, shapes.kv_lora_rank, shapes.qk_rope_head_dim, eps=eps,
            out=_zeros((rows, shapes.kv_dim), torch.bfloat16,
                       device).view(rows, 1, -1),
            row_valid_mask=torch.ones(rows, dtype=torch.bool, device=device),
            q_pe_src=q_heads[..., shapes.qk_nope_head_dim:].unsqueeze(2),
            cache_tensor=_zeros((1, page_size, 1, shapes.kv_dim), cache_dtype,
                                device),
            page_table=_zeros((1, page_table_width), torch.int32, device),
            slot_indices=_zeros((rows,), torch.int32, device),
            token_indices=_zeros((rows,), torch.int32, device),
            page_size_tokens=int(page_size),
        )


def _run_head_gates(shapes: Glm5PrewarmShapes, weight, rows: int, device,
                    fn: Optional[Callable] = None) -> None:
    if fn is None:
        from batchgen_kernels.attention.dsa.head_gates import (
            head_gates_out as fn,
        )
    fn(
        _zeros((rows, shapes.hidden_size), torch.bfloat16, device),
        weight,
        torch.empty((rows, shapes.index_n_heads), dtype=torch.float32,
                    device=device),
        scale=(shapes.index_n_heads ** -0.5) * (shapes.index_head_dim ** -0.5),
        num_valid_tokens=_valid_count(rows, device),
    )


def _run_router(shapes: Glm5PrewarmShapes, rows: int, device,
                fn: Optional[Callable] = None) -> None:
    """The graphed split-K router: all-gathered (or local) rows against the
    bf16 copy of the gate weight, into FP32 logits."""
    if fn is None:
        from batchgen_kernels.triton.glm5_router_gemm import (
            glm5_router_gemm as fn,
        )
    fn(
        _zeros((rows, shapes.hidden_size), torch.bfloat16, device),
        _zeros((shapes.num_routed_experts, shapes.hidden_size), torch.bfloat16,
               device),
        torch.empty((rows, shapes.num_routed_experts), dtype=torch.float32,
                    device=device),
    )


def _run_masked_act_quant(inter: int, row_stride: int, device,
                          fn: Optional[Callable] = None) -> None:
    """The DeepEP masked chain's SiLU*up cast, laid out as
    ``run_deepgemm_masked_moe`` hands it: gate and up as the two column
    halves of one ``[E, S, 2I]`` buffer and the scales as the (1, 2)
    transpose of a contiguous ``[E, I/128, S]`` block. One expert with no
    live rows: the launch compiles the key and touches no row."""
    if fn is None:
        from .moe_deepgemm_masked import masked_act_quant_fn
        fn = masked_act_quant_fn()
    gate_up = _zeros((1, row_stride, 2 * inter), torch.bfloat16, device)
    fn(
        gate_up[:, :, :inter],
        gate_up[:, :, inter:],
        torch.empty((1, row_stride, inter), dtype=torch.float8_e4m3fn,
                    device=device),
        torch.empty((1, inter // SCALE_BLOCK, row_stride), dtype=torch.float32,
                    device=device).transpose(1, 2),
        _zeros((1,), torch.int32, device),
        group_size=SCALE_BLOCK,
    )


def _run_selected_positions(geometry: Glm5DecodeGeometry, topk: int, device,
                            fn: Optional[Callable] = None) -> None:
    """The decode graph's logical-to-physical top-k transform: slot indices
    and the valid count given, int32 indices, the primary graph width."""
    if fn is None:
        from batchgen_kernels.attention.dsa.selected_page_table import (
            transform_selected_positions_out as fn,
        )
    fn(
        _zeros((1, geometry.primary_width), torch.int32, device),
        _zeros((1,), torch.int32, device),
        _zeros((1, topk), torch.int32, device),
        torch.empty((1, topk), dtype=torch.int32, device=device),
        torch.empty((1,), dtype=torch.int32, device=device),
        page_size=geometry.primary_page_size,
        primary_slot_indices=_zeros((1,), torch.int32, device),
        num_valid_tokens=_valid_count(1, device),
    )


def _run_paged_mqa_prepare(geometry: Glm5DecodeGeometry, device,
                           fn: Optional[Callable] = None,
                           next_n: Optional[int] = None) -> None:
    """The page-split scorer's step prologue launch (the Triton half of
    ``paged_mqa_step_prologue_out``) at the aux graph width."""
    if fn is None:
        from batchgen_kernels.attention.dsa.paged_mqa_scorer import (
            prepare_paged_mqa_inputs_out as fn,
        )
    if next_n is None:
        from batchgen_kernels.attention.dsa.paged_mqa_logits import (
            PAGED_MQA_NEXT_N as next_n,
        )
    width = geometry.aux_width
    fn(
        _zeros((1, width), torch.int32, device),
        _zeros((1,), torch.int32, device),
        _zeros((1,), torch.int32, device),
        torch.empty((1, width), dtype=torch.int32, device=device),
        torch.empty((1, next_n), dtype=torch.int32, device=device),
        num_valid_tokens=_valid_count(1, device),
        deep_gemm_context_lens_out=torch.empty((1, next_n), dtype=torch.int32,
                                               device=device),
    )


def _run_paged_score(shapes: Glm5PrewarmShapes, geometry: Glm5DecodeGeometry,
                     rows: int, aux_dtype, device,
                     fn: Optional[Callable] = None) -> None:
    """The decode graph's bf16-AoS scorer and top-k at the graph span. Every
    row has length 0, so the kernel only stores -inf and the top-k reads it."""
    if fn is None:
        from batchgen_kernels.attention.dsa.fused_indexer_score import (
            fused_paged_score_and_topk_with_slots_out as fn,
        )
    span = geometry.graph_max_seqlen
    fn(
        _zeros((rows, shapes.index_n_heads, shapes.index_head_dim),
               torch.bfloat16, device),
        _zeros((1, geometry.aux_page_size, 1, shapes.index_head_dim),
               aux_dtype, device),
        _zeros((1, geometry.aux_width), torch.int32, device),
        _zeros((rows,), torch.int32, device),
        _zeros((rows, shapes.index_n_heads), torch.float32, device),
        _zeros((rows,), torch.int32, device),
        torch.empty((rows, span), dtype=torch.float32, device=device),
        torch.empty((rows, shapes.index_topk), dtype=torch.int32,
                    device=device),
        topk=shapes.index_topk,
        page_size=geometry.aux_page_size,
        max_seqlen=span,
        num_valid_tokens=_valid_count(rows, device),
    )


def _run_kv_write(elements: int, page_size: int, rows_list, cols_list,
                  has_valid: bool, dtype, device,
                  fn: Optional[Callable] = None) -> None:
    """``run_paged_kv_token_update_fused`` into a one-page scratch cache of
    the live layout ``[pages, page_size, 1, elements]``."""
    if fn is None:
        from batchgen_kernels.triton.kv_cache import (
            run_paged_kv_token_update_fused as fn,
        )
    for rows in rows_list:
        tokens = _zeros((rows, elements), dtype, device)
        index = _zeros((rows,), torch.int32, device)
        for cols in cols_list:
            fn(
                k_cache=_zeros((1, page_size, 1, elements), dtype, device),
                k_tokens=tokens,
                page_table=_zeros((1, cols), torch.int32, device),
                slot_indices=index,
                token_indices=index,
                page_size_tokens=int(page_size),
                num_valid_tokens=_valid_count(rows, device) if has_valid else None,
            )


def _run_eager_score(shapes: Glm5PrewarmShapes, device,
                     rows_list=(1, 16, 17),
                     fn: Optional[Callable] = None) -> None:
    """Eager bf16-AoS scoring (``fused_score_pipeline``'s score and top-k)
    over a dense ``[rows, span, head_dim]`` gather."""
    if fn is None:
        from batchgen_kernels.attention.dsa.fused_indexer_score import (
            fused_score_and_topk as fn,
        )
    for rows in rows_list:
        for span in eager_score_seqlens(shapes.index_topk):
            fn(
                _zeros((rows, shapes.index_n_heads, shapes.index_head_dim),
                       torch.bfloat16, device),
                _zeros((rows, span, shapes.index_head_dim), torch.bfloat16,
                       device),
                _zeros((rows, shapes.index_n_heads), torch.float32, device),
                _zeros((rows,), torch.int32, device),
                min(shapes.index_topk, span),
            )


# ============================================================================
#  Entry lists (deterministic: identical on every rank)
# ============================================================================


@dataclass(frozen=True)
class Glm5DecodeWeights:
    """Resident weights the decode entries hand the wrappers (read only)."""

    q_a_norm_weight: Any
    q_a_norm_eps: float
    kv_a_norm_weight: Any
    kv_a_norm_eps: float
    indexer_weights_proj: Any = None


def prefill_entries(shapes: Glm5PrewarmShapes, device,
                    kv_b_dtype=None) -> List[PrewarmEntry]:
    entries = [
        PrewarmEntry(combo.name, partial(_run_act_quant, combo, device))
        for combo in quant_combos(shapes, PHASE_PREFILL)
    ]
    entries.append(PrewarmEntry(
        f"silu_mul_quant/d{shapes.shared_intermediate_size}",
        partial(_run_silu_mul_quant, shapes.shared_intermediate_size, device),
    ))
    if shapes.sparse_prefill:
        entries.append(PrewarmEntry(
            f"offset_topk/k{shapes.index_topk}",
            partial(_run_offset_topk, shapes.index_topk, device),
        ))
        if kv_b_dtype is not None:
            entries.append(PrewarmEntry(
                f"kv_b_dequant/{shapes.kv_b_rows}x{shapes.kv_lora_rank}",
                partial(_run_weight_dequant, shapes.kv_b_rows,
                        shapes.kv_lora_rank, kv_b_dtype, device),
            ))
    return entries


def decode_entries(shapes: Glm5PrewarmShapes, weights: Glm5DecodeWeights, *,
                   buckets: Optional[Sequence[int]], deepep: bool,
                   router_row_range: Tuple[int, int],
                   fuse_q_a_norm_quant: bool, device) -> List[PrewarmEntry]:
    """Weight- and bucket-keyed decode entries.

    ``buckets`` is the decode graph's capture bucket list. The head gates
    take the row count as a runtime int, so without the list they are
    covered by ``ROW_WITNESSES``. The split-K router's keys do not depend
    on the bucket at all (``router_row_witnesses``), and with DeepEP the
    masked SiLU*up cast is keyed on the receive buffers' per-expert row
    extent (``MASKED_ROW_STRIDE_WITNESSES``).
    """
    entries = [
        PrewarmEntry(combo.name, partial(_run_act_quant, combo, device))
        for combo in quant_combos(shapes, PHASE_DECODE)
    ]
    entries.append(PrewarmEntry(
        f"silu_mul_quant/d{shapes.shared_intermediate_size}",
        partial(_run_silu_mul_quant, shapes.shared_intermediate_size, device),
    ))
    entries.append(PrewarmEntry(
        "rmsnorm_fp8_quant/q_a" if fuse_q_a_norm_quant else "rmsnorm/q_a",
        partial(_run_q_a_norm, shapes, weights.q_a_norm_weight,
                weights.q_a_norm_eps, fuse_q_a_norm_quant, device),
    ))
    entries.append(PrewarmEntry(
        "rmsnorm_rope/eager",
        partial(_run_rope_eager, shapes, weights.kv_a_norm_weight,
                weights.kv_a_norm_eps, device),
    ))
    if weights.indexer_weights_proj is not None:
        if buckets:
            rows_list = sorted({int(b) for b in buckets if int(b) > 0})
            prefix = "head_gates/b"
        else:
            rows_list, prefix = list(ROW_WITNESSES), "head_gates/rows"
        entries.extend(
            PrewarmEntry(f"{prefix}{rows}",
                         partial(_run_head_gates, shapes,
                                 weights.indexer_weights_proj, rows, device))
            for rows in rows_list
        )
    entries.extend(
        PrewarmEntry(f"router_split_k/rows{rows}",
                     partial(_run_router, shapes, rows, device))
        for rows in router_row_witnesses(router_row_range)
    )
    if deepep:
        entries.extend(
            PrewarmEntry(
                f"masked_act_quant/i{shapes.expert_intermediate_size}"
                f"/s{stride}",
                partial(_run_masked_act_quant, shapes.expert_intermediate_size,
                        stride, device),
            )
            for stride in MASKED_ROW_STRIDE_WITNESSES
        )
    return entries


def geometry_entries(shapes: Glm5PrewarmShapes, weights: Glm5DecodeWeights,
                     geometry: Glm5DecodeGeometry, *,
                     buckets: Optional[Sequence[int]], q_row_widths,
                     primary_dtype, aux_dtype, device) -> List[PrewarmEntry]:
    """Entries keyed on the page-table widths and the graph score span."""
    entries = [
        PrewarmEntry(
            f"rmsnorm_rope/graph/q{int(width)}/w{geometry.primary_width}",
            partial(_run_rope_graph, shapes, weights.kv_a_norm_weight,
                    weights.kv_a_norm_eps, int(width), geometry.primary_width,
                    geometry.primary_page_size, primary_dtype, device),
        )
        for width in sorted({int(w) for w in q_row_widths})
    ]
    entries.append(PrewarmEntry(
        f"selected_positions/graph/w{geometry.primary_width}",
        partial(_run_selected_positions, geometry, shapes.index_topk, device),
    ))
    # Eager decode's primary write: no valid count, the active table width.
    entries.append(PrewarmEntry(
        f"kv_write/primary_eager/e{shapes.kv_dim}",
        partial(_run_kv_write, shapes.kv_dim, geometry.primary_page_size,
                PAGE_TABLE_COL_WITNESSES, PAGE_TABLE_COL_WITNESSES, False,
                primary_dtype, device),
    ))
    if geometry.aux_page_split:
        entries.append(PrewarmEntry(
            f"paged_mqa_prepare/w{geometry.aux_width}",
            partial(_run_paged_mqa_prepare, geometry, device),
        ))
        return entries
    if buckets:
        rows_list = sorted({int(b) for b in buckets if int(b) > 0})
        prefix = "paged_score/b"
    else:
        rows_list, prefix = list(ROW_WITNESSES), "paged_score/rows"
    entries.extend(
        PrewarmEntry(
            f"{prefix}{rows}/w{geometry.aux_width}/s{geometry.graph_max_seqlen}",
            partial(_run_paged_score, shapes, geometry, rows, aux_dtype,
                    device),
        )
        for rows in rows_list
    )
    entries.append(PrewarmEntry(
        f"kv_write/aux_graph/w{geometry.aux_width}",
        partial(_run_kv_write, shapes.index_head_dim, geometry.aux_page_size,
                PAGE_TABLE_COL_WITNESSES, (geometry.aux_width,), True,
                aux_dtype, device),
    ))
    entries.append(PrewarmEntry(
        "kv_write/aux_eager",
        partial(_run_kv_write, shapes.index_head_dim, geometry.aux_page_size,
                PAGE_TABLE_COL_WITNESSES, PAGE_TABLE_COL_WITNESSES, False,
                aux_dtype, device),
    ))
    entries.append(PrewarmEntry(
        "eager_score", partial(_run_eager_score, shapes, device),
    ))
    return entries


# ============================================================================
#  Reading the live model and KV managers (read only)
# ============================================================================


def _layers(model) -> list:
    layers = getattr(getattr(model, "model", None), "layers", None)
    return list(layers) if layers is not None else []


def _resident(tensor) -> bool:
    return (tensor is not None and not tensor.is_meta and tensor.numel() > 0
            and tensor.is_cuda)


def _attn_module(layer):
    wrapper = getattr(layer, "self_attn", None)
    return getattr(wrapper, "module", wrapper)


def decode_weights(model) -> Optional[Glm5DecodeWeights]:
    """The q_a/kv_a norm weights of the first layer and the first indexer's
    head-gate projection; None when a norm weight is not resident."""
    layers = _layers(model)
    if not layers:
        return None
    attn = _attn_module(layers[0])
    q_norm = getattr(attn, "q_a_layernorm", None)
    kv_norm = getattr(attn, "kv_a_layernorm", None)
    q_weight = getattr(q_norm, "weight", None)
    kv_weight = getattr(kv_norm, "weight", None)
    if not (_resident(q_weight) and _resident(kv_weight)):
        return None
    gates = None
    for layer in layers:
        indexer = getattr(_attn_module(layer), "indexer", None)
        weight = getattr(getattr(indexer, "weights_proj", None), "weight", None)
        if _resident(weight):
            gates = weight.data
            break
    return Glm5DecodeWeights(
        q_a_norm_weight=q_weight, q_a_norm_eps=float(q_norm.eps),
        kv_a_norm_weight=kv_weight, kv_a_norm_eps=float(kv_norm.eps),
        indexer_weights_proj=gates,
    )


def _graph_table_width(manager, role: str) -> int:
    table = getattr(manager, "_gpu_page_table_manager", None)
    width = getattr(table, "graph_max_pages_per_sequence", None)
    if width is None:
        raise Glm5PrewarmGeometryError(
            f"the {role} GPU KV manager exposes no CUDA-graph page-table width"
        )
    return int(width)


def _page_table_slots(manager, role: str) -> int:
    table = getattr(manager, "_gpu_page_table_manager", None)
    slots = getattr(table, "max_slots", None)
    if slots is None:
        raise Glm5PrewarmGeometryError(
            f"the {role} GPU KV manager exposes no page-table slot capacity"
        )
    return int(slots)


def _aux_is_page_split(aux_manager, index_head_dim: int) -> bool:
    """``aux_cache_is_page_split`` on a meta probe of the aux config, as
    ``aux_manager_applies_hadamard`` resolves the format."""
    from batchgen_kernels.attention.dsa.indexer_k_quant import (
        cache_is_page_split,
    )

    config = aux_manager.config
    probe = torch.empty((0, 0, 0, int(config.k_head_dim)),
                        dtype=config.kv_dtype, device="meta")
    return bool(cache_is_page_split(probe, int(index_head_dim)))


def decode_geometry(gpu_manager, shapes: Glm5PrewarmShapes
                    ) -> Tuple[Glm5DecodeGeometry, Any, Any]:
    """``(geometry, primary kv dtype, aux kv dtype)`` of the worker's KV
    managers, as ``_setup_cuda_graphs`` reads them."""
    primary = getattr(gpu_manager, "primary", gpu_manager)
    aux = getattr(gpu_manager, "auxiliary", None)
    if primary is None or aux is None:
        raise Glm5PrewarmGeometryError(
            "GLM-5 decode geometry needs the primary and auxiliary GPU KV "
            "managers"
        )
    geometry = Glm5DecodeGeometry(
        primary_width=_graph_table_width(primary, "primary"),
        aux_width=_graph_table_width(aux, "auxiliary"),
        primary_page_size=int(primary.config.page_size_tokens),
        aux_page_size=int(aux.config.page_size_tokens),
        aux_page_split=_aux_is_page_split(aux, shapes.index_head_dim),
        max_position_embeddings=shapes.max_position_embeddings,
        max_decode_rows=_page_table_slots(primary, "primary"),
    )
    return geometry, primary.config.kv_dtype, aux.config.kv_dtype


# ============================================================================
#  Phases
# ============================================================================


_DONE_PHASES: set = set()
_GEOMETRY_KEY: Optional[tuple] = None


def _restore_guard(guard, mode: str) -> None:
    if mode == STRICT:
        guard.arm(strict=True)
    elif mode == LOG:
        guard.arm()
    else:
        guard.disarm()


def _run_phase(phase: str, entries, *, part_index: int, part_count: int,
               barrier, rank, arm_after: bool) -> None:
    """Partitioned run with the guard disarmed, so the deliberate compiles
    of a phase are never reported as late."""
    from .deepgemm_prewarm import cache_kernel_count

    recorder = get_recorder()
    active = recorder if recorder.install() else None
    previous = recorder.guard.armed_mode
    recorder.guard.disarm()
    start = time.perf_counter()
    mark = active.mark() if active is not None else 0
    deepgemm_before = cache_kernel_count()
    try:
        with torch.inference_mode():
            _core.run_prewarm(
                entries, recorder=active, part_index=part_index,
                part_count=part_count, barrier=barrier, rank=rank,
                log_tag=f"{LOG_TAG} {phase}",
            )
    finally:
        torch.cuda.empty_cache()
        if arm_after and active is not None:
            recorder.guard.arm()
        else:
            _restore_guard(recorder.guard, previous)
    if active is not None:
        for kernel, (compiled, loaded) in sorted(
                active.per_kernel(mark).items()):
            logger.debug(
                "%s rank=%s %s variants kernel=%s compiled=%d loaded=%d",
                LOG_TAG, rank, phase, kernel, compiled, loaded,
            )
    logger.info("%s rank=%s %s entries=%d total wall=%.1fs DeepGEMM cache "
                "kernels %d -> %d", LOG_TAG, rank, phase, len(entries),
                time.perf_counter() - start, deepgemm_before,
                cache_kernel_count())
    if arm_after and active is not None:
        logger.info(
            "%s rank=%s late-compile guard armed: later compiles log at "
            "WARNING; batchgen_debug.%s makes them raise",
            LOG_TAG, rank, STRICT_DEBUG_KEY,
        )


def checkpoint_fp8_dtype(config):
    """Element type of the checkpoint's blockwise FP8 weights, or None.

    The prefill weights are streamed, so the module parameters still carry
    their construction dtype at configure time; the checkpoint's
    quantization config is what the streamed kv_b weight will be.
    """
    quant = getattr(config, "quantization_config", None) or {}
    if not isinstance(quant, dict) or quant.get("quant_method") != "fp8":
        return None
    if quant.get("fmt", "e4m3") != "e4m3":
        return None
    return torch.float8_e4m3fn


def prewarm_glm5_prefill(config, *, max_rows: int, part_index=0,
                         part_count=1, barrier=None, rank=0) -> bool:
    """Phase 1, once per process. The prefill weights are streamed, so no
    entry reads them: every tensor is a fresh scratch. ``max_rows`` bounds
    the DeepGEMM GEMMs' row count (``deepgemm_prewarm``)."""
    if PHASE_PREFILL in _DONE_PHASES:
        return False
    from . import deepgemm_prewarm as dg

    dg.check_version(rank)
    shapes = Glm5PrewarmShapes.from_config(config)
    device = torch.device("cuda", torch.cuda.current_device())
    kv_b_dtype = checkpoint_fp8_dtype(config) if shapes.sparse_prefill else None
    if shapes.sparse_prefill and kv_b_dtype is None:
        logger.warning("%s rank=%s the checkpoint is not blockwise e4m3 FP8; "
                       "the sparse prefill kv_b dequant is not prewarmed",
                       LOG_TAG, rank)
    entries = prefill_entries(shapes, device, kv_b_dtype=kv_b_dtype)
    entries += dg.prefill_entries(shapes, device, max_rows=max_rows,
                                  num_sms=dg.deepgemm_num_sms())
    _run_phase(PHASE_PREFILL, entries, part_index=part_index,
               part_count=part_count, barrier=barrier, rank=rank,
               arm_after=False)
    _DONE_PHASES.add(PHASE_PREFILL)
    return True


def _router_row_range() -> Tuple[int, int]:
    from batchgen_kernels.triton.glm5_router_gemm_config import (
        ROUTER_SPLIT_K_MAX_ROWS,
        ROUTER_SPLIT_K_MIN_ROWS,
    )

    return ROUTER_SPLIT_K_MIN_ROWS, ROUTER_SPLIT_K_MAX_ROWS


def prewarm_glm5_decode(model, config, *, decode_buckets, max_rows: int,
                        part_index=0, part_count=1, barrier=None,
                        rank=0) -> bool:
    """Phase 2, once per process, after the decode model's fused kernels
    are initialised. ``max_rows`` bounds the DeepGEMM GEMMs' row count."""
    if PHASE_DECODE in _DONE_PHASES:
        return False
    from . import deepgemm_prewarm as dg
    from .indexer_wk_backend import glm5_indexer_wk_is_deepgemm
    from .norm_quant_fuse_policy import glm5_norm_quant_fuse_site_enabled
    from .qb_wqb_merge_policy import glm5_qb_wqb_merge_enabled

    shapes = Glm5PrewarmShapes.from_config(config)
    log_section_d(decode_section_d_predicates(model, config), rank)
    weights = decode_weights(model)
    if weights is None:
        logger.warning("%s rank=%s decode norm weights are not resident; the "
                       "decode phase is not prewarmed", LOG_TAG, rank)
        return False
    entries = decode_entries(
        shapes, weights, buckets=decode_buckets,
        deepep=bool(getattr(config, "enable_deepep", False)),
        router_row_range=_router_row_range(),
        fuse_q_a_norm_quant=glm5_norm_quant_fuse_site_enabled("q_a"),
        device=weights.q_a_norm_weight.device,
    )
    dg.check_version(rank)
    entries += dg.decode_entries(
        shapes, weights.q_a_norm_weight.device, max_rows=max_rows,
        num_sms=dg.deepgemm_num_sms(),
        merge_q_b_wq_b=glm5_qb_wqb_merge_enabled(),
        wk_deepgemm=glm5_indexer_wk_is_deepgemm(),
    )
    _run_phase(PHASE_DECODE, entries, part_index=part_index,
               part_count=part_count, barrier=barrier, rank=rank,
               arm_after=False)
    _DONE_PHASES.add(PHASE_DECODE)
    return True


def prewarm_glm5_geometry(model, config, gpu_manager, *, decode_buckets,
                          world_size: int = 1, part_index=0, part_count=1,
                          barrier=None, rank=0) -> bool:
    """Phase 3: once per page-table geometry. Arms the late-compile guard
    and records the DeepGEMM cache's kernel count.

    Every rank must reach the same decision, or the phase barrier hangs;
    the geometry comes from the KV configuration, which all ranks share.
    ``world_size`` is the EP world of the DeepEP MoE's ``expected_m``.
    """
    global _GEOMETRY_KEY
    from . import deepgemm_prewarm as dg
    from .qb_wqb_merge_policy import glm5_qb_wqb_merge_enabled

    shapes = Glm5PrewarmShapes.from_config(config)
    try:
        geometry, primary_dtype, aux_dtype = decode_geometry(gpu_manager,
                                                             shapes)
        capacity = geometry.graph_max_seqlen
    except Glm5PrewarmGeometryError as exc:
        logger.warning("%s rank=%s geometry phase not run: %s", LOG_TAG, rank,
                       exc)
        return False
    if geometry.key() == _GEOMETRY_KEY:
        return False
    weights = decode_weights(model)
    if weights is None:
        logger.warning("%s rank=%s decode norm weights are not resident; the "
                       "geometry phase is not run", LOG_TAG, rank)
        return False
    previous = _GEOMETRY_KEY
    logger.info(
        "%s rank=%s geometry primary_width=%d aux_width=%d page_sizes=%d/%d "
        "aux_page_split=%s graph_max_seqlen=%d max_decode_rows=%d%s",
        LOG_TAG, rank, geometry.primary_width, geometry.aux_width,
        geometry.primary_page_size, geometry.aux_page_size,
        geometry.aux_page_split, capacity, geometry.max_decode_rows,
        "" if previous is None else f" (changed from {previous})",
    )
    log_section_d({"aux_page_split": geometry.aux_page_split,
                   "eager_width_constexpr": True}, rank)
    # q_b output row widths whose rope half the graph's RMSNorm+RoPE reads:
    # the separate q_b GEMM, and the merged [q_b | wq_b] GEMM of the indexer
    # layers when the merge is on.
    q_row_widths = {shapes.q_b_cols}
    if glm5_qb_wqb_merge_enabled():
        q_row_widths.add(shapes.q_b_cols + shapes.wq_b_cols)
    entries = geometry_entries(
        shapes, weights, geometry, buckets=decode_buckets,
        q_row_widths=q_row_widths, primary_dtype=primary_dtype,
        aux_dtype=aux_dtype, device=weights.q_a_norm_weight.device,
    )
    dg.check_version(rank)
    stacks = None
    if bool(getattr(config, "enable_deepep", False)):
        stacks = dg.moe_stacks(model)
        if stacks is None:
            logger.warning(
                "%s rank=%s enable_deepep is set but no MoE layer holds the "
                "fused masked-GEMM stacks; the DeepEP masked GEMMs are NOT "
                "prewarmed", LOG_TAG, rank,
            )
    entries += dg.geometry_entries(
        shapes, weights.q_a_norm_weight.device,
        max_decode_rows=geometry.max_decode_rows,
        num_sms=dg.deepgemm_num_sms(),
        merge_q_b_wq_b=glm5_qb_wqb_merge_enabled(),
        page_split=geometry.aux_page_split, moe_stacks=stacks,
        world_size=world_size,
        topk=int(getattr(config, "num_experts_per_tok", 1) or 1),
        page_size=geometry.aux_page_size,
    )
    _run_phase(PHASE_GEOMETRY, entries, part_index=part_index,
               part_count=part_count, barrier=barrier, rank=rank,
               arm_after=True)
    dg.record_baseline(rank)
    _GEOMETRY_KEY = geometry.key()
    return True
