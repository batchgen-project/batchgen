"""CUDA-graph capturable GLM-5 DSA decode segments.

This module contains the first production integration bridge for the GLM-5
BF16 DSA graph path.  The segment starts at the already validated graph-safe
boundary:

    q_a, q_nope/q_rope, auxiliary indexer pages, primary MLA pages
      -> fused indexer score/top-k
      -> selected logical-to-physical token transform
      -> FA3 over a page-size-1 view of the resident KV (no copy)
      -> q/out absorb

``Glm5FullDsaAttnSegment`` additionally specializes an ``all_short`` graph
lifetime: when every admissible row's KV budget fits in ``index_topk`` the
whole indexer scoring chain and the transform are skipped and FA3 reads the
resident page-size-``page_size`` cache directly.

It deliberately does not change the default GLM-5 decode path.  Callers must
explicitly construct and register this segment, and unsupported production
preconditions should fast-fail before replay rather than silently falling back.
"""

from __future__ import annotations

import logging

from dataclasses import dataclass, fields
from typing import Dict, Optional

import torch

try:
    from flash_attn_interface import flash_attn_with_kvcache as _fa3_with_kvcache
except ImportError:  # pragma: no cover - resolved by the node's FA3 install
    _fa3_with_kvcache = None

from batchgen.attention.mla.fa3_backend import act_quant
from batchgen.attention.mla.fused_rmsnorm_rope import (
    fused_rmsnorm_rope_with_q_native as _fused_rmsnorm_rope,
)
from batchgen.attention.dsa.aux_indexer_k_writer import (
    aux_cache_data_dim,
    aux_cache_is_page_split,
    write_aux_indexer_k,
)
from batchgen.attention.dsa.sparse_decode_mla import (
    prepare_sparse_flash_mla_decode_tensor_metadata,
    prepare_sparse_flash_mla_decode_inputs,
    run_prepared_sparse_flash_mla_decode,
)
from batchgen.attention.dsa.unified_selector import select_mla_kv_for_flashmla_bf16_out
from batchgen.attention.mla.packed_fp8_sparse_decode import (
    guard_sparse_indices_against_empty_rows,
    prepare_sparse_fp8_mla_metadata,
    run_sparse_fp8_mla_decode,
)
from batchgen.attention.mla.primary_kv_record import (
    prepare_primary_mla_kv,
    primary_cache_is_packed_fp8,
    primary_record_dim,
    primary_record_dtype,
)
from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm
from batchgen_kernels.attention.dsa.fp8_absorb import (
    FP8AbsorbWeights,
    fp8_out_absorb_out,
    fp8_q_absorb_out,
)
from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
    cuda_wk_proj_gemm_only_out,
    make_fp8_activation_scratch,
)
from batchgen_kernels.attention.dsa.fused_indexer_score import (
    FP8WqbWeightsCUDA,
    cuda_wq_b_proj_out,
    fused_paged_score_and_topk_with_slots_out,
    rope_hadamard_q_out,
)
from batchgen_kernels.attention.dsa.head_gates import head_gates_out
from batchgen_kernels.attention.dsa.paged_mqa_logits import PAGED_MQA_NEXT_N
from batchgen_kernels.attention.dsa.paged_mqa_scorer import (
    paged_mqa_score_and_topk_out,
    warm_paged_mqa_logits,
)
from batchgen_kernels.attention.dsa.query_pack import pack_flashmla_query_out
from batchgen_kernels.attention.dsa.selected_page_table import (
    transform_selected_positions_out,
)
from batchgen_kernels.triton.kv_cache import run_paged_kv_token_update_fused
from batchgen_kernels.triton.rmsnorm import fused_rmsnorm


@dataclass
class _Glm5DsaSegmentBuffers:
    q_x_fp8: torch.Tensor
    q_x_scale: torch.Tensor
    q_tma_desc: torch.Tensor
    q_flat: torch.Tensor
    q_index: torch.Tensor
    # Scorer buffers. Exactly one of the two groups is allocated per bucket,
    # decided by the aux cache's format: `agg_scores` for the retired bf16
    # Triton scorer, the DeepGEMM trio for a page-split FP8 cache.
    agg_scores: torch.Tensor
    q_index_fp8: torch.Tensor
    aux_block_table: torch.Tensor
    aux_context_lens: torch.Tensor
    top_k_indices: torch.Tensor
    selected_mla_kv: torch.Tensor
    selected_lengths: torch.Tensor
    row_modes: torch.Tensor
    absorbed_q: torch.Tensor
    query_states: torch.Tensor
    attn_heads: torch.Tensor
    prepared_flashmla: object


@dataclass
class _Glm5FullDsaSegmentBuffers:
    valid_mask: torch.Tensor
    aux_valid_mask: torch.Tensor
    row_indices: torch.Tensor
    valid_rows_bf16: torch.Tensor
    valid_rows_ones: torch.Tensor
    valid_rows_zeros: torch.Tensor
    safe_slot_zeros: torch.Tensor
    skip_slot_neg_ones: torch.Tensor
    safe_seqlen_zeros: torch.Tensor
    kv_primary_slot_indices: torch.Tensor
    kv_aux_slot_indices: torch.Tensor
    safe_primary_slot_indices: torch.Tensor
    safe_aux_slot_indices: torch.Tensor
    safe_cache_seqlens: torch.Tensor
    qkv_a: torch.Tensor
    q_a: torch.Tensor
    q_a_normed: torch.Tensor
    q_flat: torch.Tensor
    q_nope: torch.Tensor
    q_rope_4d: torch.Tensor
    new_compressed_kv: torch.Tensor
    indexer_k_raw: torch.Tensor
    indexer_k_x_fp8: torch.Tensor
    indexer_k_x_scale: torch.Tensor
    indexer_k_tma_desc: torch.Tensor
    q_x_fp8: torch.Tensor
    q_x_scale: torch.Tensor
    q_tma_desc: torch.Tensor
    q_flat_indexer: torch.Tensor
    q_index: torch.Tensor
    head_gates: torch.Tensor
    positions_expanded: torch.Tensor
    # Scorer buffers, one group per aux cache format (see
    # ``_make_scorer_buffers``): ``agg_scores`` is the retired bf16 Triton
    # scorer's dense aggregate, the other three are what
    # ``deep_gemm.fp8_paged_mqa_logits`` reads.
    agg_scores: torch.Tensor
    q_index_fp8: torch.Tensor
    aux_block_table: torch.Tensor
    aux_context_lens: torch.Tensor
    top_k_indices: torch.Tensor
    # Physical page-size-1 token IDs consumed as the FA3 ``page_table``; the
    # BF16 selected-KV slab (``selected_mla_kv``) and the FlashMLA query pack
    # (``query_states`` / ``prepared_flashmla``) are gone with the gather.
    selected_token_ids: torch.Tensor
    selected_lengths: torch.Tensor
    absorbed_q: torch.Tensor
    attn_heads: torch.Tensor
    # Packed-FP8 primary cache only (1-element placeholders otherwise, like
    # the scorer groups above):
    #   query_packed       [B, H, 576] bf16 -- FlashMLA takes ONE 576-wide
    #                      query, not FA3's split q / qv pair, so the absorbed
    #                      NoPE half and the RoPE half are assembled here. The
    #                      absorb kernel hardcodes a contiguous [B, H, N]
    #                      output (fp8_absorb.py:244), so this cannot be a
    #                      strided view of the two halves; see the cost note
    #                      in forward().
    #   primary_k_packed   [B, 1, 1, 656] uint8 -- the quantizer's destination,
    #                      before the invalid-row select.
    #   packed_zeros_row   [1, 1, 1, 656] uint8 -- the zero record an invalid
    #                      row is selected to (`valid_rows_zeros` stays bf16:
    #                      it is also the zero source of the bf16 row-mask
    #                      select).
    query_packed: torch.Tensor
    primary_k_packed: torch.Tensor
    packed_zeros_row: torch.Tensor


@dataclass
class _Glm5FullDsaSegmentOutputs:
    primary_k_tensor: torch.Tensor
    # The aux tensor the host append must receive: bf16 keys for an AoS aux
    # cache, e4m3 keys for a page-split one.
    indexer_k_tensor: torch.Tensor
    attn_output: torch.Tensor
    # fp32 per-token scales, allocated only for a page-split aux cache (an AoS
    # host region rejects scales, so None is the correct value there).
    indexer_k_scale: torch.Tensor | None = None


# Static-buffer footprint: capture runs largest-first, so the full per-row
# buffer set is allocated once for the largest bucket and every smaller bucket
# is a leading-dim slice view of that base. Views are safe because exactly one
# bucket's graph replays per step and every non-constant field is fully
# rewritten (out=/copy_) for rows [0, bucket) before it is read within that
# replay; constant fields are never written. One field class cannot be
# views and is rebuilt per bucket:
#   - FP8 activation scratch: its TMA descriptor bakes pointer + global rows.
# (The former ``prepared_flashmla`` rebuild is gone with the FlashMLA call:
# FA3 takes its inputs directly, so nothing bakes shapes at setup time.)
_GLM5_DSA_REBUILT_FIELDS = frozenset({
    "indexer_k_x_fp8",
    "indexer_k_x_scale",
    "indexer_k_tma_desc",
    "q_x_fp8",
    "q_x_scale",
    "q_tma_desc",
})
_GLM5_DSA_VIEW_FIELDS = frozenset({
    "valid_mask",
    "aux_valid_mask",
    "row_indices",
    "valid_rows_bf16",
    "valid_rows_ones",
    "valid_rows_zeros",
    "safe_slot_zeros",
    "skip_slot_neg_ones",
    "safe_seqlen_zeros",
    "kv_primary_slot_indices",
    "kv_aux_slot_indices",
    "safe_primary_slot_indices",
    "safe_aux_slot_indices",
    "safe_cache_seqlens",
    "qkv_a",
    "q_a",
    "q_a_normed",
    "q_flat",
    "q_nope",
    "q_rope_4d",
    "new_compressed_kv",
    "indexer_k_raw",
    "q_flat_indexer",
    "q_index",
    "head_gates",
    "positions_expanded",
    "agg_scores",
    "q_index_fp8",
    "aux_block_table",
    "aux_context_lens",
    "top_k_indices",
    "selected_token_ids",
    "selected_lengths",
    "absorbed_q",
    "attn_heads",
    "query_packed",
    "primary_k_packed",
})
# Constant across buckets, so never a leading-dim view: one zero record that
# every bucket's invalid-row select reads from.
_GLM5_DSA_SHARED_FIELDS = frozenset({
    "packed_zeros_row",
})

# Fail loudly at import time if the dataclass gains a field this split does
# not name: an unclassified field would silently become a stale full-size
# allocation or, worse, a wrongly-shaped view.
_split_delta = (
    set(f.name for f in fields(_Glm5FullDsaSegmentBuffers))
    ^ (
        _GLM5_DSA_VIEW_FIELDS
        | _GLM5_DSA_REBUILT_FIELDS
        | _GLM5_DSA_SHARED_FIELDS
    )
)
if _split_delta:
    raise AssertionError(
        f"_Glm5FullDsaSegmentBuffers fields not covered by (or extraneous in) "
        f"the view/rebuild split: {sorted(_split_delta)}"
    )


def _warm_paged_mqa_scorer(
    *,
    page_split: bool,
    aux_blocked_k: torch.Tensor,
    bucket_size: int,
    num_index_heads: int,
    index_head_dim: int,
    max_aux_pages_per_seq: int,
    max_seqlen: int,
    aux_page_size: int,
) -> None:
    """JIT-compile the DeepGEMM scorer for this bucket before capture.

    ``get_paged_mqa_logits_metadata`` is templated on ``align(batch, 32)``, so
    every captured bucket is a distinct instantiation and must be warmed
    separately; the logits kernel itself compiles once. Static buffers are
    built before capture on the production path, which makes this the natural
    warm point (first-batch rule).
    """

    if not page_split or not torch.cuda.is_available():
        return
    if torch.cuda.is_current_stream_capturing():
        # Reached only if a capture entered `forward` without pre-built
        # statics. Compiling would then happen inside the capture; warn rather
        # than warm, so the cause is visible in the log instead of showing up
        # as capture-time latency.
        logging.warning(
            "GLM-5.2 DSA scorer: static buffers for bucket %d were created "
            "DURING capture, so the DeepGEMM paged-MQA JIT could not be warmed "
            "first; build statics before capture",
            bucket_size,
        )
        return
    warm_paged_mqa_logits(
        aux_blocked_k=aux_blocked_k,
        batch=bucket_size,
        num_heads=num_index_heads,
        index_head_dim=index_head_dim,
        max_pages_per_seq=max_aux_pages_per_seq,
        max_seqlen=max_seqlen,
        page_size=aux_page_size,
    )


def _make_scorer_buffers(
    *,
    page_split: bool,
    bucket_size: int,
    max_seqlen: int,
    num_index_heads: int,
    index_head_dim: int,
    max_aux_pages_per_seq: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(agg_scores, q_index_fp8, aux_block_table, aux_context_lens)``.

    The two scorers need disjoint statics, and only one of them can ever run
    for a given aux cache, so the other group is a zero-byte placeholder rather
    than a real allocation. The placeholder still carries ``bucket_size`` as
    its leading dim (with a zero-extent trailing dim): every field here is a
    ``_GLM5_DSA_VIEW_FIELDS`` member, which smaller buckets slice as
    ``base[:bucket]``, so a placeholder must obey the same "leading dim is the
    bucket" invariant as the live group -- a 1-element placeholder sliced to
    ``[:2]`` silently stays 1 row and becomes a wrongly-shaped view. That matters
    for footprint too: ``agg_scores`` is a dense
    ``[bucket, max_seqlen]`` fp32 aggregate, which at production bucket and
    context is tens of MB charged against the same memory the KV pages come
    from, and the DeepGEMM scorer never writes it (it hands the top-k the
    logits DeepGEMM allocated).

    * ``q_index_fp8``: ``[bucket, next_n=1, heads, head_dim]`` e4m3, the only
      ``q`` layout ``fp8_paged_mqa_logits`` accepts.
    * ``aux_block_table``: the aux page table's rows gathered into BATCH order.
      DeepGEMM indexes the block table by batch row and has no slot
      indirection, which BatchGen's slot-major table needs resolving first.
    * ``aux_context_lens``: the TRUE 2-D ``[bucket, next_n]`` per-row lengths,
      the top-k's bound. DeepGEMM is handed a guarded copy derived in the
      same preparation launch (``guard_deep_gemm_context_lens``).
    """

    if not page_split:
        return (
            torch.empty(bucket_size, max_seqlen, dtype=torch.float32, device=device),
            torch.empty(bucket_size, 0, 0, 0, dtype=torch.float8_e4m3fn, device=device),
            torch.empty(bucket_size, 0, dtype=torch.int32, device=device),
            torch.empty(bucket_size, 0, dtype=torch.int32, device=device),
        )
    return (
        torch.empty(bucket_size, 0, dtype=torch.float32, device=device),
        torch.empty(
            bucket_size,
            PAGED_MQA_NEXT_N,
            num_index_heads,
            index_head_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        torch.empty(
            bucket_size, max_aux_pages_per_seq, dtype=torch.int32, device=device
        ),
        torch.empty(bucket_size, PAGED_MQA_NEXT_N, dtype=torch.int32, device=device),
    )


def _make_packed_primary_buffers(
    *,
    packed: bool,
    bucket_size: int,
    num_heads: int,
    source_dim: int,
    record_dim: int,
    record_dtype: torch.dtype,
    device: torch.device,
):
    """The three buffers a packed-FP8 primary cache needs, or placeholders.

    Allocated as a group, like the scorer's two format-specific groups, so a
    bf16 AoS capture pays nothing for the packed path and a packed capture
    cannot silently fall back to a half-built buffer set.

    * ``query_packed``: ``[bucket, heads, source_dim]`` bf16, the single
      576-wide query FlashMLA takes. NOT a view over ``absorbed_q`` and
      ``q_rope_4d``: the absorb kernel writes a contiguous ``[B, H, N]``
      output (``fp8_absorb.py:244`` computes ``b*H*N + h*N + n`` with no
      output strides), so writing the two halves straight into a 576-wide
      buffer would need that kernel to learn output strides -- a change to a
      kernel every MLA model shares, deliberately not made here.
    * ``primary_k_packed``: ``[bucket, 1, 1, record_dim]`` of the cache dtype,
      the quantizer's destination before invalid rows are selected out.
    * ``packed_zeros_row``: ``[1, 1, 1, record_dim]`` zeros, the record an
      invalid row is selected to. An all-zero packed record is a VALID zero
      token -- scale bits 0.0f, data 0x00 -- which is also what a freshly
      allocated page holds, so a skipped row and an untouched slot are the
      same bytes.
    """

    if not packed:
        return (
            torch.empty(bucket_size, 0, dtype=torch.bfloat16, device=device),
            torch.empty(bucket_size, 0, dtype=record_dtype, device=device),
            torch.empty(0, dtype=record_dtype, device=device),
        )
    return (
        torch.empty(
            bucket_size, num_heads, source_dim, dtype=torch.bfloat16, device=device
        ),
        torch.empty(
            bucket_size, 1, 1, record_dim, dtype=record_dtype, device=device
        ),
        torch.zeros(1, 1, 1, record_dim, dtype=record_dtype, device=device),
    )


class Glm5DsaAttnSegment:
    """Graph-capturable GLM-5 DSA attention subsegment.

    The segment uses the unified selected-KV API: every row writes a fixed
    ``index_topk`` selected-token buffer, while ``selected_lengths`` carries the
    runtime valid length into FlashMLA. This keeps short, boundary, mixed, and
    long decode rows on the same graph-safe path.
    """

    def __init__(
        self,
        *,
        primary_blocked_k: torch.Tensor,
        aux_blocked_k: torch.Tensor,
        primary_page_table: torch.Tensor,
        aux_page_table: torch.Tensor,
        wq_b_weights: FP8WqbWeightsCUDA,
        absorb_weights: FP8AbsorbWeights,
        cuda_module,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        max_seqlen: int,
        index_topk: int = 2048,
        page_size: int = 64,
        aux_page_size: int | None = None,
        num_index_heads: int = 32,
        num_attn_heads: int = 64,
        q_lora_rank: int = 2048,
        index_head_dim: int = 128,
        q_nope_head_dim: int = 192,
        kv_lora_rank: int = 512,
        attn_out_dim: int = 256,
        softmax_scale: float | None = None,
        shared_buffers: Optional[Dict[int, _Glm5DsaSegmentBuffers]] = None,
    ) -> None:
        if primary_blocked_k.ndim != 4:
            raise ValueError(
                "primary_blocked_k must have shape [num_pages, page_size, "
                f"num_kv_heads, kv_dim], got {tuple(primary_blocked_k.shape)}"
            )
        if primary_blocked_k.shape[1] != page_size:
            raise ValueError(
                f"primary_blocked_k page size {primary_blocked_k.shape[1]} != {page_size}"
            )
        if primary_blocked_k.shape[2] != 1:
            raise ValueError("GLM-5 DSA BF16 path expects one MLA KV head")
        if aux_blocked_k.ndim != 4:
            raise ValueError(
                "aux_blocked_k must have shape [num_pages, page_size, 1, index_head_dim], "
                f"got {tuple(aux_blocked_k.shape)}"
            )
        if aux_blocked_k.shape[2] != 1:
            raise ValueError("GLM-5 DSA aux cache expects one indexer KV head")
        if primary_page_table.ndim != 2 or aux_page_table.ndim != 2:
            raise ValueError(
                "GLM-5 DSA graph requires 2-D primary/aux page tables, "
                f"got {tuple(primary_page_table.shape)} and {tuple(aux_page_table.shape)}"
            )
        if primary_page_table.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"primary_page_table must be int32/int64, got {primary_page_table.dtype}")
        if aux_page_table.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"aux_page_table must be int32/int64, got {aux_page_table.dtype}")
        if primary_page_table.device != primary_blocked_k.device:
            raise ValueError("primary_page_table and primary_blocked_k must share device")
        if aux_page_table.device != aux_blocked_k.device:
            raise ValueError("aux_page_table and aux_blocked_k must share device")

        self.primary_blocked_k = primary_blocked_k
        self.aux_blocked_k = aux_blocked_k
        self.primary_page_table = primary_page_table
        self.aux_page_table = aux_page_table
        self.wq_b_weights = wq_b_weights
        self.absorb_weights = absorb_weights
        self.cuda_module = cuda_module
        self.cos_table = cos_table
        self.sin_table = sin_table
        self.max_seqlen = max_seqlen
        self.index_topk = index_topk
        self.page_size = page_size
        self.aux_page_size = page_size if aux_page_size is None else aux_page_size
        self.num_index_heads = num_index_heads
        self.num_attn_heads = num_attn_heads
        self.q_lora_rank = q_lora_rank
        self.index_head_dim = index_head_dim
        self.q_nope_head_dim = q_nope_head_dim
        self.kv_lora_rank = kv_lora_rank
        self.attn_out_dim = attn_out_dim
        self.kv_dim = primary_blocked_k.shape[3]
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.kv_dim**-0.5
        self.max_pages_per_seq = primary_page_table.shape[1]
        self.max_aux_pages_per_seq = aux_page_table.shape[1]
        self._uses_shared_buffers = shared_buffers is not None
        self._buffers = shared_buffers if shared_buffers is not None else {}
        self._attn_head_outputs: Dict[int, torch.Tensor] = {}
        self._flashmla_metadata_specs: Dict[
            int,
            tuple[tuple[int, ...], torch.dtype, tuple[int, ...], torch.dtype],
        ] = {}

        # NOT migrated to the packed FP8 primary record, and it must not
        # pretend to be: this segment GATHERS the selected KV into a synthetic
        # page-size-N slab and attends over it with dense bf16 FlashMLA
        # (`select_mla_kv_for_flashmla_bf16_out` + the gathered-KV wrapper).
        # A packed cache's 656-byte records would be gathered as if they were
        # 656 bf16 elements -- a silently wrong slab, not a crash -- so the
        # format is rejected at construction. The migrated path is
        # `Glm5FullDsaAttnSegment` / `Glm5ReuseTopkAttnSegment`.
        if int(primary_blocked_k.element_size()) == 1:
            raise ValueError(
                "GLM-5 legacy DSA segment cannot read a packed FP8 primary "
                f"MLA cache (trailing dim {self.kv_dim}, "
                f"{primary_blocked_k.dtype}): it gathers bf16 KV into a "
                "synthetic page slab. Use the full DSA segment for a packed "
                "cache, or run with the bf16 profile (glm52_packed_kv off)."
            )
        if self.kv_dim != self.kv_lora_rank + 64:
            raise ValueError(
                f"GLM-5 DSA selected KV dim must be kv_lora_rank + 64, got {self.kv_dim}"
            )
        if aux_blocked_k.shape[1] != self.aux_page_size:
            raise ValueError(
                f"aux_blocked_k page size {aux_blocked_k.shape[1]} != {self.aux_page_size}"
            )
        # See the full segment: both aux formats are accepted here, and a
        # mis-sized cache is named by the shared discriminator.
        self.aux_is_page_split = aux_cache_is_page_split(
            aux_blocked_k, self.index_head_dim
        )
        if cos_table.dtype != torch.float32 or sin_table.dtype != torch.float32:
            raise TypeError("cos_table and sin_table must be float32 for graph capture")
        if not cos_table.is_contiguous() or not sin_table.is_contiguous():
            raise ValueError("cos_table and sin_table must be contiguous")

    def _padding_selected_length(self) -> int:
        return min(int(self.max_seqlen), int(self.index_topk))

    def _flashmla_tensor_metadata_specs(
        self,
        bucket_size: int,
    ) -> tuple[tuple[int, ...], torch.dtype, tuple[int, ...], torch.dtype]:
        cached = self._flashmla_metadata_specs.get(bucket_size)
        if cached is not None:
            return cached
        lengths = torch.full(
            (bucket_size,),
            self._padding_selected_length(),
            dtype=torch.int32,
            device=self.primary_blocked_k.device,
        )
        tile_scheduler_metadata, num_splits = prepare_sparse_flash_mla_decode_tensor_metadata(
            lengths,
            self.num_attn_heads,
        )
        spec = (
            tuple(tile_scheduler_metadata.shape),
            tile_scheduler_metadata.dtype,
            tuple(num_splits.shape),
            num_splits.dtype,
        )
        self._flashmla_metadata_specs[bucket_size] = spec
        return spec

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        tile_shape, tile_dtype, num_splits_shape, num_splits_dtype = (
            self._flashmla_tensor_metadata_specs(bucket_size)
        )
        return {
            "q_a": TensorSpec(("batch_size", self.q_lora_rank), torch.bfloat16),
            "q_nope": TensorSpec(
                ("batch_size", self.num_attn_heads, self.q_nope_head_dim),
                torch.bfloat16,
            ),
            "q_rope": TensorSpec(("batch_size", self.num_attn_heads, 64), torch.bfloat16),
            "head_gates": TensorSpec(("batch_size", self.num_index_heads), torch.float32),
            "cache_seqlens": TensorSpec(
                ("batch_size",), torch.int32, fill_value=float(self.max_seqlen)
            ),
            "positions_expanded": TensorSpec(
                ("batch_size", self.num_index_heads),
                torch.int64,
                fill_value=float(self.max_seqlen - 1),
            ),
            "primary_slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=0),
            "aux_slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=0),
            "num_valid_tokens": TensorSpec((1,), torch.int32, fill_value=float(bucket_size)),
            "flashmla_tile_scheduler_metadata": TensorSpec(tile_shape, tile_dtype),
            "flashmla_num_splits": TensorSpec(num_splits_shape, num_splits_dtype),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "attn_heads": TensorSpec(
                ("batch_size", 1, self.num_attn_heads, self.attn_out_dim),
                torch.bfloat16,
            ),
            "top_k_indices": TensorSpec(("batch_size", self.index_topk), torch.int32),
            "selected_lengths": TensorSpec(("batch_size",), torch.int32),
            "selected_mla_kv": TensorSpec(
                ("batch_size", self.index_topk, 1, self.kv_dim), torch.bfloat16
            ),
            "absorbed_q": TensorSpec(
                ("batch_size", self.num_attn_heads, self.kv_lora_rank),
                torch.bfloat16,
            ),
            "query_states": TensorSpec(
                ("batch_size", 1, self.num_attn_heads, self.kv_dim),
                torch.bfloat16,
            ),
            "raw_attn_out": TensorSpec(
                ("batch_size", 1, self.num_attn_heads, self.kv_lora_rank),
                torch.bfloat16,
            ),
        }

    def setup_static_buffers(self, bucket_size: int) -> None:
        # Warm first and for EVERY bucket, including the ones that end up as
        # views of a larger base: the DeepGEMM metadata kernel is templated on
        # align(bucket, 32), so each bucket is its own JIT instantiation and a
        # view-built bucket would otherwise compile during capture.
        self._warm_scorer_jit(bucket_size)
        if bucket_size in self._buffers:
            self._setup_static_output_buffers(bucket_size)
            return
        device = self.primary_blocked_k.device
        q_x_fp8, q_x_scale, q_tma_desc = make_fp8_activation_scratch(
            bucket_size,
            self.q_lora_rank,
            self.cuda_module,
            device=device,
        )
        q_flat = torch.empty(
            bucket_size,
            self.num_index_heads * self.index_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        q_index = torch.empty(
            bucket_size,
            self.num_index_heads,
            self.index_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        agg_scores, q_index_fp8, aux_block_table, aux_context_lens = (
            _make_scorer_buffers(
                page_split=self.aux_is_page_split,
                bucket_size=bucket_size,
                max_seqlen=self.max_seqlen,
                num_index_heads=self.num_index_heads,
                index_head_dim=self.index_head_dim,
                max_aux_pages_per_seq=self.max_aux_pages_per_seq,
                device=device,
            )
        )
        top_k_indices = torch.empty(
            bucket_size,
            self.index_topk,
            dtype=torch.int32,
            device=device,
        )
        selected_mla_kv = torch.empty(
            bucket_size,
            self.index_topk,
            1,
            self.kv_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        selected_lengths = torch.empty(bucket_size, dtype=torch.int32, device=device)
        selected_lengths.fill_(self.index_topk)
        row_modes = torch.empty(bucket_size, dtype=torch.int32, device=device)
        absorbed_q = torch.empty(
            bucket_size,
            self.num_attn_heads,
            self.kv_lora_rank,
            dtype=torch.bfloat16,
            device=device,
        )
        query_states = torch.empty(
            bucket_size,
            1,
            self.num_attn_heads,
            self.kv_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        attn_heads = torch.empty(
            bucket_size,
            1,
            self.num_attn_heads,
            self.attn_out_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        prepared_flashmla = prepare_sparse_flash_mla_decode_inputs(
            query_states,
            selected_mla_kv,
            selected_lengths,
            self.num_attn_heads,
            self.softmax_scale,
            head_dim_v=self.kv_lora_rank,
            page_size=self.page_size,
        )
        self._buffers[bucket_size] = _Glm5DsaSegmentBuffers(
            q_x_fp8=q_x_fp8,
            q_x_scale=q_x_scale,
            q_tma_desc=q_tma_desc,
            q_flat=q_flat,
            q_index=q_index,
            agg_scores=agg_scores,
            q_index_fp8=q_index_fp8,
            aux_block_table=aux_block_table,
            aux_context_lens=aux_context_lens,
            top_k_indices=top_k_indices,
            selected_mla_kv=selected_mla_kv,
            selected_lengths=selected_lengths,
            row_modes=row_modes,
            absorbed_q=absorbed_q,
            query_states=query_states,
            attn_heads=attn_heads,
            prepared_flashmla=prepared_flashmla,
        )
        self._setup_static_output_buffers(bucket_size)

    def _warm_scorer_jit(self, bucket_size: int) -> None:
        _warm_paged_mqa_scorer(
            page_split=self.aux_is_page_split,
            aux_blocked_k=self.aux_blocked_k,
            bucket_size=bucket_size,
            num_index_heads=self.num_index_heads,
            index_head_dim=self.index_head_dim,
            max_aux_pages_per_seq=self.max_aux_pages_per_seq,
            max_seqlen=self.max_seqlen,
            aux_page_size=self.aux_page_size,
        )

    def _score_and_topk(
        self,
        buffers: _Glm5DsaSegmentBuffers,
        *,
        aux_slot_indices: torch.Tensor,
        head_gates: torch.Tensor,
        cache_seqlens: torch.Tensor,
        num_valid_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score the aux cache and select top-k, dispatching on CACHE FORMAT.

        Page-split FP8 (GLM-5.2) goes to ``deep_gemm.fp8_paged_mqa_logits``,
        which computes the ReLU-gated GLM-5 aggregation directly from the paged
        cache. A bf16 AoS cache (GLM-5, 5.1, 5.3) keeps the retired no-ReLU
        Triton scorer. There is no fallback between them: the FP8 cache is
        unreadable by the Triton kernel and the bf16 cache is unreadable by
        DeepGEMM, so a mis-detected format fails loud inside the callee.
        """

        if self.aux_is_page_split:
            return paged_mqa_score_and_topk_out(
                q=buffers.q_index,
                aux_blocked_k=self.aux_blocked_k,
                aux_page_table=self.aux_page_table,
                aux_slot_indices=aux_slot_indices,
                head_gates=head_gates,
                cache_seqlens=cache_seqlens,
                q_fp8=buffers.q_index_fp8,
                block_table=buffers.aux_block_table,
                context_lens=buffers.aux_context_lens,
                top_k_indices=buffers.top_k_indices,
                topk=self.index_topk,
                max_seqlen=self.max_seqlen,
                index_head_dim=self.index_head_dim,
                page_size=self.aux_page_size,
                num_valid_tokens=num_valid_tokens,
            )
        return fused_paged_score_and_topk_with_slots_out(
            buffers.q_index,
            self.aux_blocked_k,
            self.aux_page_table,
            aux_slot_indices,
            head_gates,
            cache_seqlens,
            buffers.agg_scores,
            buffers.top_k_indices,
            topk=self.index_topk,
            page_size=self.aux_page_size,
            max_seqlen=self.max_seqlen,
            num_valid_tokens=num_valid_tokens,
        )

    def initialize_static_inputs(
        self,
        static_inputs: Dict[str, torch.Tensor],
        bucket_size: int,
    ) -> None:
        selected_lengths = torch.full(
            (bucket_size,),
            self._padding_selected_length(),
            dtype=torch.int32,
            device=self.primary_blocked_k.device,
        )
        tile_scheduler_metadata, num_splits = prepare_sparse_flash_mla_decode_tensor_metadata(
            selected_lengths,
            self.num_attn_heads,
        )
        static_inputs["flashmla_tile_scheduler_metadata"].copy_(
            tile_scheduler_metadata,
            non_blocking=True,
        )
        static_inputs["flashmla_num_splits"].copy_(num_splits, non_blocking=True)

    def _setup_static_output_buffers(self, bucket_size: int) -> None:
        if not self._uses_shared_buffers or bucket_size in self._attn_head_outputs:
            return
        self._attn_head_outputs[bucket_size] = torch.empty(
            bucket_size,
            1,
            self.num_attn_heads,
            self.attn_out_dim,
            dtype=torch.bfloat16,
            device=self.primary_blocked_k.device,
        )

    def release_static_buffers(self, bucket_size: int) -> None:
        self._buffers.pop(bucket_size, None)
        self._attn_head_outputs.pop(bucket_size, None)

    def forward(
        self,
        *,
        q_a: torch.Tensor,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        head_gates: torch.Tensor,
        cache_seqlens: torch.Tensor,
        positions_expanded: torch.Tensor,
        primary_slot_indices: torch.Tensor,
        aux_slot_indices: torch.Tensor,
        num_valid_tokens: torch.Tensor,
        flashmla_tile_scheduler_metadata: torch.Tensor,
        flashmla_num_splits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = q_a.shape[0]
        buffers = self._buffers.get(batch_size)
        if buffers is None:
            self.setup_static_buffers(batch_size)
            buffers = self._buffers[batch_size]

        cuda_wq_b_proj_out(
            q_a,
            self.wq_b_weights,
            self.cuda_module,
            buffers.q_x_fp8,
            buffers.q_x_scale,
            buffers.q_tma_desc,
            buffers.q_flat,
        )
        rope_hadamard_q_out(
            buffers.q_flat.view(batch_size, self.num_index_heads, self.index_head_dim),
            self.cos_table,
            self.sin_table,
            positions_expanded.view(-1),
            buffers.q_index,
        )
        self._score_and_topk(
            buffers,
            aux_slot_indices=aux_slot_indices,
            head_gates=head_gates,
            cache_seqlens=cache_seqlens,
            num_valid_tokens=num_valid_tokens,
        )
        select_mla_kv_for_flashmla_bf16_out(
            self.primary_blocked_k,
            self.primary_page_table,
            cache_seqlens,
            buffers.top_k_indices,
            self.page_size,
            buffers.selected_mla_kv,
            buffers.selected_lengths,
            None,
            buffers.row_modes,
            index_topk=self.index_topk,
            return_indices=False,
            primary_slot_indices=primary_slot_indices,
        )
        fp8_q_absorb_out(q_nope, self.absorb_weights, buffers.absorbed_q)
        pack_flashmla_query_out(buffers.absorbed_q, q_rope, buffers.query_states)
        attn_out = run_prepared_sparse_flash_mla_decode(
            buffers.prepared_flashmla,
            tile_scheduler_metadata=flashmla_tile_scheduler_metadata,
            num_splits=flashmla_num_splits,
        )
        attn_heads = self._attn_head_outputs.get(batch_size, buffers.attn_heads)
        fp8_out_absorb_out(attn_out, self.absorb_weights, attn_heads)

        return {
            "attn_heads": attn_heads,
            "top_k_indices": buffers.top_k_indices,
            "selected_lengths": buffers.selected_lengths,
            "selected_mla_kv": buffers.selected_mla_kv,
            "absorbed_q": buffers.absorbed_q,
            "query_states": buffers.query_states,
            "raw_attn_out": attn_out,
        }


class Glm5FullDsaAttnSegment:
    """Graph-capturable GLM-5 DSA attention from hidden states through o_proj."""

    def __init__(
        self,
        *,
        wrapper,
        primary_blocked_k: torch.Tensor,
        aux_blocked_k: torch.Tensor,
        primary_page_table: torch.Tensor,
        aux_page_table: torch.Tensor,
        wq_b_weights: FP8WqbWeightsCUDA,
        absorb_weights: FP8AbsorbWeights,
        cuda_module,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
        max_seqlen: int,
        index_topk: int = 2048,
        page_size: int = 64,
        aux_page_size: int | None = None,
        all_short: bool = False,
        shared_buffers: Optional[Dict[int, _Glm5FullDsaSegmentBuffers]] = None,
    ) -> None:
        self.wrapper = wrapper
        self.attn = wrapper.module
        self.layer_idx = int(wrapper.layer_idx)
        self.primary_blocked_k = primary_blocked_k
        self.aux_blocked_k = aux_blocked_k
        self.primary_page_table = primary_page_table
        self.aux_page_table = aux_page_table
        self.wq_b_weights = wq_b_weights
        self.absorb_weights = absorb_weights
        self.cuda_module = cuda_module
        self.cos_table = cos_table.contiguous()
        self.sin_table = sin_table.contiguous()
        self.max_seqlen = int(max_seqlen)
        self.index_topk = int(index_topk)
        self.page_size = int(page_size)
        self.aux_page_size = int(aux_page_size if aux_page_size is not None else page_size)
        # Graph-lifetime specialization: True only when every admissible row's
        # page-aligned KV budget is <= index_topk for the whole lifetime of
        # this capture, so no row can ever need top-k selection. Decided by
        # the caller (worker) per bucket; default False keeps the general
        # transform + page-size-1 FA3 path, which covers every row.
        self.all_short = bool(all_short)
        self._uses_shared_buffers = shared_buffers is not None
        self._buffers = shared_buffers if shared_buffers is not None else {}
        self._outputs: Dict[int, _Glm5FullDsaSegmentOutputs] = {}

        if self.primary_blocked_k.ndim != 4 or self.primary_blocked_k.shape[2] != 1:
            raise ValueError(
                "primary_blocked_k must have shape [num_pages, page_size, 1, kv_dim], "
                f"got {tuple(primary_blocked_k.shape)}"
            )
        if self.aux_blocked_k.ndim != 4 or self.aux_blocked_k.shape[2] != 1:
            raise ValueError(
                "aux_blocked_k must have shape [num_pages, page_size, 1, index_dim], "
                f"got {tuple(aux_blocked_k.shape)}"
            )
        if self.primary_page_table.ndim != 2 or self.aux_page_table.ndim != 2:
            raise ValueError("primary and aux page tables must be rank-2 tensors")
        if self.primary_blocked_k.shape[1] != self.page_size:
            raise ValueError("primary_blocked_k page size does not match page_size")
        if self.aux_blocked_k.shape[1] != self.aux_page_size:
            raise ValueError("aux_blocked_k page size does not match aux_page_size")
        # Accepts both primary formats and names a mis-sized cache: bf16 AoS
        # keeps `kv_lora_rank + qk_rope_head_dim` elements per token, FlashMLA's
        # packed FP8 record keeps 656 BYTES (e4m3 NoPE, fp32 block scales, bf16
        # RoPE). Read off the allocated cache, never from a model name -- the
        # same rule the aux cache uses just above.
        self.primary_source_dim = int(
            self.attn.kv_lora_rank + self.attn.qk_rope_head_dim
        )
        self.primary_is_packed = primary_cache_is_packed_fp8(
            self.primary_blocked_k, self.primary_source_dim
        )
        # Shape/dtype of the record buffers this segment allocates and of the
        # tensor it hands to the host append; the composing layer segment
        # declares its own spec from these.
        self.primary_record_dim = primary_record_dim(self.primary_blocked_k)
        self.primary_record_dtype = primary_record_dtype(self.primary_blocked_k)
        # Accepts both aux formats and names a mis-sized cache: bf16 AoS keeps
        # `index_head_dim` elements per token, page-split FP8 keeps
        # `index_head_dim + 4` BYTES (the fp32 per-token scale included).
        self.aux_is_page_split = aux_cache_is_page_split(
            self.aux_blocked_k, self.attn.indexer.index_head_dim
        )
        # Shape/dtype of the aux tensor this segment hands to the host append;
        # the composing layer segment declares its own spec from these.
        self.aux_host_kv_dim = aux_cache_data_dim(self.aux_blocked_k)
        self.aux_host_kv_dtype = (
            torch.float8_e4m3fn if self.aux_is_page_split else torch.bfloat16
        )
        # A packed-FP8 primary cache is read by sparse FlashMLA on BOTH paths,
        # so FA3 is not a requirement of this segment; a bf16 AoS cache has no
        # other reader, so its absence is still fatal. Checked per format
        # rather than unconditionally: demanding FA3 on the packed path would
        # make an unrelated missing dependency look like a GLM-5.2 failure.
        if not self.primary_is_packed and _fa3_with_kvcache is None:
            raise RuntimeError(
                "GLM-5 DSA graph requires flash_attn_interface "
                "(FlashAttention-3): flash_attn_with_kvcache is unavailable"
            )
        # Per-bucket sparse metadata, built once per bucket at setup time and
        # replayed unchanged: in sparse mode FlashMLA's metadata kernel
        # substitutes `topk` for every row's length and never reads
        # cache_seqlens, so its output is step-invariant. See
        # packed_fp8_sparse_decode note 1.
        self._sparse_metadata: Dict[int, object] = {}

    def _padding_selected_length(self) -> int:
        return min(int(self.max_seqlen), int(self.index_topk))

    def _warm_scorer_jit(self, bucket_size: int) -> None:
        indexer = self.attn.indexer
        _warm_paged_mqa_scorer(
            page_split=self.aux_is_page_split,
            aux_blocked_k=self.aux_blocked_k,
            bucket_size=bucket_size,
            num_index_heads=indexer.index_n_heads,
            index_head_dim=indexer.index_head_dim,
            max_aux_pages_per_seq=int(self.aux_page_table.shape[1]),
            max_seqlen=self.max_seqlen,
            aux_page_size=self.aux_page_size,
        )

    def _score_and_topk(
        self,
        buffers: _Glm5FullDsaSegmentBuffers,
        *,
        aux_slot_indices: torch.Tensor,
        head_gates: torch.Tensor,
        cache_seqlens: torch.Tensor,
        num_valid_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score the aux cache and select top-k, dispatching on CACHE FORMAT.

        Page-split FP8 (GLM-5.2) goes to ``deep_gemm.fp8_paged_mqa_logits``,
        which computes the ReLU-gated GLM-5 aggregation -- the training-time
        formulation -- directly from the paged cache. A bf16 AoS cache (GLM-5,
        5.1, 5.3) keeps the retired no-ReLU Triton scorer, because that is the
        formulation those profiles were qualified with. Neither kernel can read
        the other's cache, so there is no fallback edge: a mis-sized cache is
        named by the format discriminator at construction.
        """

        if self.aux_is_page_split:
            return paged_mqa_score_and_topk_out(
                q=buffers.q_index,
                aux_blocked_k=self.aux_blocked_k,
                aux_page_table=self.aux_page_table,
                aux_slot_indices=aux_slot_indices,
                head_gates=head_gates,
                cache_seqlens=cache_seqlens,
                q_fp8=buffers.q_index_fp8,
                block_table=buffers.aux_block_table,
                context_lens=buffers.aux_context_lens,
                top_k_indices=buffers.top_k_indices,
                topk=self.index_topk,
                max_seqlen=self.max_seqlen,
                index_head_dim=self.attn.indexer.index_head_dim,
                page_size=self.aux_page_size,
                num_valid_tokens=num_valid_tokens,
            )
        return fused_paged_score_and_topk_with_slots_out(
            buffers.q_index,
            self.aux_blocked_k,
            self.aux_page_table,
            aux_slot_indices,
            head_gates,
            cache_seqlens,
            buffers.agg_scores,
            buffers.top_k_indices,
            topk=self.index_topk,
            page_size=self.aux_page_size,
            max_seqlen=self.max_seqlen,
            num_valid_tokens=num_valid_tokens,
        )

    def _prepare_sparse_metadata(self, bucket_size: int) -> None:
        """Build this bucket's sparse FP8 metadata once, outside any capture.

        In sparse mode FlashMLA's metadata kernel substitutes ``topk`` for
        every row's length and never reads ``cache_seqlens``
        (``csrc/smxx/get_mla_metadata.cu:30``), so the result depends only on
        (bucket_size, topk) plus arch constants and is identical on every
        replay. Computing it here keeps a kernel launch AND an allocation out
        of the captured region, and keeps the metadata off the step's critical
        path entirely -- it is not recomputed per step the way the dense
        metadata would have to be.
        """

        if not self.primary_is_packed:
            return
        if bucket_size in self._sparse_metadata:
            return
        if torch.cuda.is_current_stream_capturing():
            # Reached only if a capture entered `forward` without pre-built
            # statics. Allocating and launching here would bake graph-private
            # metadata buffers; refuse loudly instead of capturing something
            # that replays against freed memory.
            raise RuntimeError(
                f"GLM-5.2 packed decode: sparse metadata for bucket "
                f"{bucket_size} was requested DURING capture; build static "
                "buffers before capture"
            )
        self._sparse_metadata[bucket_size] = prepare_sparse_fp8_mla_metadata(
            batch_size=bucket_size,
            topk=int(self.index_topk),
            num_heads_q=int(self.attn.num_heads),
            num_heads_k=1,
            num_q_tokens_per_q_seq=1,
            device=self.primary_blocked_k.device,
        )

    def _run_sparse_fp8(
        self,
        buffers: _Glm5FullDsaSegmentBuffers,
        batch_size: int,
    ) -> torch.Tensor:
        """Sparse FP8 FlashMLA decode over the packed cache.

        The ONE reader of a packed primary cache, used by both the selected
        and the all-short path: the pin has no dense FP8 MLA decode kernel, so
        "attend to the whole short context" is expressed as "select every
        token of it", which `transform_selected_positions_out` already
        produces for a row whose length is within the selection budget (it
        falls back to identity offsets there).

        `query_packed` is the 576-wide concatenation the kernel takes, written
        in place by the absorb GEMV and the RoPE kernel -- its two halves ARE
        the absorbed-NoPE and RoPE buffers, so there is no extra copy.
        """

        metadata = self._sparse_metadata.get(batch_size)
        if metadata is None:
            raise RuntimeError(
                f"GLM-5.2 packed decode: no sparse metadata for bucket "
                f"{batch_size}; setup_static_buffers must run before capture"
            )
        # Keep every row's softmax denominator non-zero. A padding row is all
        # -1 and would otherwise come back NaN (packed_fp8_sparse_decode
        # note 3); its output is masked away later, but a NaN must not be
        # produced in the first place.
        guard_sparse_indices_against_empty_rows(
            buffers.selected_token_ids, buffers.safe_slot_zeros
        )
        return run_sparse_fp8_mla_decode(
            query=buffers.query_packed.unsqueeze(1),
            packed_blocked_k=self.primary_blocked_k,
            indices=buffers.selected_token_ids.unsqueeze(1),
            # Shape-checked at the pybind boundary, never read by the sparse
            # kernel (note 2). A slot-major slice is passed rather than a
            # dedicated placeholder so the call carries real, correctly-typed
            # data; a pin bump must re-verify that it stays unread, because
            # these rows are slot-major, not batch-major.
            block_table=self.primary_page_table[:batch_size],
            cache_seqlens=buffers.selected_lengths,
            metadata=metadata,
            softmax_scale=float(self.attn.softmax_scale),
            head_dim_v=int(self.attn.kv_lora_rank),
        )

    def _run_all_short_fa3(
        self,
        buffers: _Glm5FullDsaSegmentBuffers,
    ) -> torch.Tensor:
        """Dense MLA decode straight over the resident page-size-N KV.

        Valid only for an ``all_short`` graph: every row attends to its whole
        context, so no selection is needed and the page table / slot indices
        of the real cache are used as-is (no copy, no top-k).

        bf16 AoS caches only -- a packed cache has no dense FP8 reader.
        """
        if _fa3_with_kvcache is None:
            raise RuntimeError("FlashAttention-3 is unavailable")
        attn = self.attn
        return _fa3_with_kvcache(
            q=buffers.q_rope_4d.squeeze(2).unsqueeze(1),
            k_cache=self.primary_blocked_k[..., attn.kv_lora_rank :],
            v_cache=self.primary_blocked_k[..., : attn.kv_lora_rank],
            qv=buffers.absorbed_q.unsqueeze(1),
            page_table=self.primary_page_table,
            cache_batch_idx=buffers.safe_primary_slot_indices,
            cache_seqlens=buffers.safe_cache_seqlens,
            softmax_scale=float(attn.softmax_scale),
            causal=True,
            num_splits=0,
            return_softmax_lse=False,
        )

    def _run_selected_fa3(
        self,
        buffers: _Glm5FullDsaSegmentBuffers,
    ) -> torch.Tensor:
        """Sparse MLA decode over the selected PHYSICAL token IDs.

        The KV cache is re-viewed at page size 1 so the per-row selected
        token IDs act directly as the FA3 page table: the selected KV is
        never materialized. ``cache_batch_idx`` is intentionally absent —
        the token IDs are already absolute, so there is no slot indirection.
        """
        if _fa3_with_kvcache is None:
            raise RuntimeError("FlashAttention-3 is unavailable")
        attn = self.attn
        flat_kv = self.primary_blocked_k.view(
            -1,
            1,
            1,
            self.primary_blocked_k.shape[-1],
        )
        return _fa3_with_kvcache(
            q=buffers.q_rope_4d.squeeze(2).unsqueeze(1),
            k_cache=flat_kv[..., attn.kv_lora_rank :],
            v_cache=flat_kv[..., : attn.kv_lora_rank],
            qv=buffers.absorbed_q.unsqueeze(1),
            page_table=buffers.selected_token_ids,
            cache_seqlens=buffers.selected_lengths,
            softmax_scale=float(attn.softmax_scale),
            causal=True,
            num_splits=0,
            return_softmax_lse=False,
        )

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.attn.hidden_size),
                torch.bfloat16,
            ),
            "position_ids": TensorSpec(
                ("batch_size", 1),
                torch.int64,
                fill_value=float(self.max_seqlen - 1),
            ),
            "cache_seqlens": TensorSpec(
                ("batch_size",),
                torch.int32,
                fill_value=float(self.max_seqlen),
            ),
            "primary_slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=-1),
            "aux_slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=-1),
            "num_valid_tokens": TensorSpec((1,), torch.int32, fill_value=float(bucket_size)),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        specs = {
            "attn_output": TensorSpec(
                ("batch_size", 1, self.attn.hidden_size),
                torch.bfloat16,
            ),
            "primary_k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.primary_record_dim),
                self.primary_record_dtype,
            ),
            "indexer_k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.aux_host_kv_dim),
                self.aux_host_kv_dtype,
            ),
        }
        if self.aux_is_page_split:
            specs["indexer_k_scale"] = TensorSpec(
                ("batch_size",), torch.float32,
            )
        return specs

    def setup_static_buffers(self, bucket_size: int) -> None:
        # Warm first and for EVERY bucket, including the ones that end up as
        # views of a larger base: the DeepGEMM metadata kernel is templated on
        # align(bucket, 32), so each bucket is its own JIT instantiation and a
        # view-built bucket would otherwise compile during capture.
        self._warm_scorer_jit(bucket_size)
        self._prepare_sparse_metadata(bucket_size)
        if bucket_size in self._buffers:
            self._setup_static_output_buffers(bucket_size)
            return
        if self._buffers and bucket_size < max(self._buffers):
            # A larger base set already exists (capture runs largest-first, so
            # the first captured bucket allocated the full set): build this
            # bucket as leading-dim slice views plus the per-bucket rebuilt
            # fields.
            self._buffers[bucket_size] = self._make_bucket_view_buffers(bucket_size)
            self._setup_static_output_buffers(bucket_size)
            return
        if self._buffers:
            # A bucket larger than the current base: allocate a fresh full set
            # (already-captured graphs keep their baked pointers). Correct but
            # forfeits the max-bucket-view footprint for the existing sets —
            # the production capture path is largest-first, so reaching this
            # means the capture order regressed.
            logging.warning(
                "GLM-5 DSA statics: bucket %d requested after smaller base %d; "
                "allocating a full set (largest-first capture restores the "
                "max-bucket-view footprint)",
                bucket_size,
                max(self._buffers),
            )
        device = self.primary_blocked_k.device
        attn = self.attn
        indexer = attn.indexer
        index_dim = indexer.index_head_dim
        kv_dim = attn.kv_lora_rank + attn.qk_rope_head_dim

        indexer_k_x_fp8, indexer_k_x_scale, indexer_k_tma_desc = make_fp8_activation_scratch(
            bucket_size,
            attn.hidden_size,
            self.cuda_module,
            device=device,
        )
        q_x_fp8, q_x_scale, q_tma_desc = make_fp8_activation_scratch(
            bucket_size,
            attn.q_lora_rank,
            self.cuda_module,
            device=device,
        )

        selected_token_ids = torch.empty(
            bucket_size,
            self.index_topk,
            dtype=torch.int32,
            device=device,
        )
        selected_lengths = torch.empty(bucket_size, dtype=torch.int32, device=device)
        selected_lengths.fill_(self._padding_selected_length())
        (
            query_packed,
            primary_k_packed,
            packed_zeros_row,
        ) = _make_packed_primary_buffers(
            packed=self.primary_is_packed,
            bucket_size=bucket_size,
            num_heads=attn.num_heads,
            source_dim=self.primary_source_dim,
            record_dim=self.primary_record_dim,
            record_dtype=self.primary_record_dtype,
            device=device,
        )
        agg_scores, q_index_fp8, aux_block_table, aux_context_lens = (
            _make_scorer_buffers(
                page_split=self.aux_is_page_split,
                bucket_size=bucket_size,
                max_seqlen=self.max_seqlen,
                num_index_heads=indexer.index_n_heads,
                index_head_dim=index_dim,
                max_aux_pages_per_seq=int(self.aux_page_table.shape[1]),
                device=device,
            )
        )
        qkv_a = torch.empty(
            bucket_size,
            attn.q_lora_rank + kv_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        self._buffers[bucket_size] = _Glm5FullDsaSegmentBuffers(
            valid_mask=torch.empty(bucket_size, dtype=torch.bool, device=device),
            aux_valid_mask=torch.empty(bucket_size, dtype=torch.bool, device=device),
            row_indices=torch.arange(bucket_size, dtype=torch.int32, device=device),
            valid_rows_bf16=torch.empty(bucket_size, dtype=torch.bfloat16, device=device),
            valid_rows_ones=torch.ones(bucket_size, dtype=torch.bfloat16, device=device),
            valid_rows_zeros=torch.zeros(bucket_size, dtype=torch.bfloat16, device=device),
            safe_slot_zeros=torch.zeros(bucket_size, dtype=torch.int32, device=device),
            skip_slot_neg_ones=torch.full((bucket_size,), -1, dtype=torch.int32, device=device),
            safe_seqlen_zeros=torch.zeros(bucket_size, dtype=torch.int32, device=device),
            kv_primary_slot_indices=torch.empty(bucket_size, dtype=torch.int32, device=device),
            kv_aux_slot_indices=torch.empty(bucket_size, dtype=torch.int32, device=device),
            safe_primary_slot_indices=torch.empty(bucket_size, dtype=torch.int32, device=device),
            safe_aux_slot_indices=torch.empty(bucket_size, dtype=torch.int32, device=device),
            safe_cache_seqlens=torch.empty(bucket_size, dtype=torch.int32, device=device),
            qkv_a=qkv_a,
            q_a=qkv_a[:, : attn.q_lora_rank],
            q_a_normed=torch.empty(
                bucket_size,
                attn.q_lora_rank,
                dtype=torch.bfloat16,
                device=device,
            ),
            q_flat=torch.empty(
                bucket_size,
                attn.num_heads * attn.q_head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            q_nope=torch.empty(
                bucket_size,
                attn.num_heads,
                attn.qk_nope_head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            q_rope_4d=torch.empty(
                bucket_size,
                attn.num_heads,
                1,
                attn.qk_rope_head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            new_compressed_kv=qkv_a[:, attn.q_lora_rank :].view(
                bucket_size, 1, kv_dim
            ),
            indexer_k_raw=torch.empty(bucket_size, index_dim, dtype=torch.bfloat16, device=device),
            indexer_k_x_fp8=indexer_k_x_fp8,
            indexer_k_x_scale=indexer_k_x_scale,
            indexer_k_tma_desc=indexer_k_tma_desc,
            q_x_fp8=q_x_fp8,
            q_x_scale=q_x_scale,
            q_tma_desc=q_tma_desc,
            q_flat_indexer=torch.empty(
                bucket_size,
                indexer.index_n_heads * index_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            q_index=torch.empty(
                bucket_size,
                indexer.index_n_heads,
                index_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            head_gates=torch.empty(bucket_size, indexer.index_n_heads, dtype=torch.float32, device=device),
            positions_expanded=torch.empty(bucket_size, indexer.index_n_heads, dtype=torch.int64, device=device),
            agg_scores=agg_scores,
            q_index_fp8=q_index_fp8,
            aux_block_table=aux_block_table,
            aux_context_lens=aux_context_lens,
            top_k_indices=torch.empty(bucket_size, self.index_topk, dtype=torch.int32, device=device),
            selected_token_ids=selected_token_ids,
            selected_lengths=selected_lengths,
            query_packed=query_packed,
            primary_k_packed=primary_k_packed,
            packed_zeros_row=packed_zeros_row,
            absorbed_q=torch.empty(
                bucket_size,
                attn.num_heads,
                attn.kv_lora_rank,
                dtype=torch.bfloat16,
                device=device,
            ),
            attn_heads=torch.empty(
                bucket_size,
                1,
                attn.num_heads,
                attn.v_head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        self._setup_static_output_buffers(bucket_size)

    def _base_bucket(self) -> int:
        return max(self._buffers)

    def _make_bucket_view_buffers(self, bucket_size: int) -> _Glm5FullDsaSegmentBuffers:
        base_bucket = self._base_bucket()
        base = self._buffers[base_bucket]
        attn = self.attn
        view_kwargs = {
            name: getattr(base, name)[:bucket_size] for name in _GLM5_DSA_VIEW_FIELDS
        }
        # FP8 activation scratch TMA descriptors bake pointer + global rows:
        # re-encode per bucket (the scratch itself is tiny).
        indexer_k_x_fp8, indexer_k_x_scale, indexer_k_tma_desc = make_fp8_activation_scratch(
            bucket_size,
            attn.hidden_size,
            self.cuda_module,
            device=self.primary_blocked_k.device,
        )
        q_x_fp8, q_x_scale, q_tma_desc = make_fp8_activation_scratch(
            bucket_size,
            attn.q_lora_rank,
            self.cuda_module,
            device=self.primary_blocked_k.device,
        )
        return _Glm5FullDsaSegmentBuffers(
            indexer_k_x_fp8=indexer_k_x_fp8,
            indexer_k_x_scale=indexer_k_x_scale,
            indexer_k_tma_desc=indexer_k_tma_desc,
            q_x_fp8=q_x_fp8,
            q_x_scale=q_x_scale,
            q_tma_desc=q_tma_desc,
            **{
                name: getattr(base, name)
                for name in _GLM5_DSA_SHARED_FIELDS
            },
            **view_kwargs,
        )

    def _setup_static_output_buffers(self, bucket_size: int) -> None:
        if bucket_size in self._outputs:
            return
        if self._outputs and bucket_size < max(self._outputs):
            base = self._outputs[max(self._outputs)]
            self._outputs[bucket_size] = _Glm5FullDsaSegmentOutputs(
                primary_k_tensor=base.primary_k_tensor[:bucket_size],
                indexer_k_tensor=base.indexer_k_tensor[:bucket_size],
                attn_output=base.attn_output[:bucket_size],
                indexer_k_scale=(
                    None
                    if base.indexer_k_scale is None
                    else base.indexer_k_scale[:bucket_size]
                ),
            )
            return
        device = self.primary_blocked_k.device
        attn = self.attn
        self._outputs[bucket_size] = _Glm5FullDsaSegmentOutputs(
            # The record this cache stores, not the latent it came from: this
            # tensor is BOTH the GPU-cache store source and the tensor the
            # host append stages, so its shape/dtype must be the cache's.
            primary_k_tensor=torch.empty(
                bucket_size,
                1,
                1,
                self.primary_record_dim,
                dtype=self.primary_record_dtype,
                device=device,
            ),
            indexer_k_tensor=torch.empty(
                bucket_size,
                1,
                1,
                self.aux_host_kv_dim,
                dtype=self.aux_host_kv_dtype,
                device=device,
            ),
            attn_output=torch.empty(bucket_size, attn.hidden_size, dtype=torch.bfloat16, device=device),
            indexer_k_scale=(
                torch.empty(bucket_size, dtype=torch.float32, device=device)
                if self.aux_is_page_split
                else None
            ),
        )

    def initialize_static_inputs(
        self,
        static_inputs: Dict[str, torch.Tensor],
        bucket_size: int,
    ) -> None:
        static_inputs["hidden_states"].zero_()
        static_inputs["position_ids"].zero_()
        static_inputs["cache_seqlens"].zero_()
        static_inputs["primary_slot_indices"].fill_(-1)
        static_inputs["aux_slot_indices"].fill_(-1)
        # FA3 can illegal-access during graph capture with an all-zero
        # selected-length schedule. Capture one safe dummy row; replay overwrites
        # this scalar with the real local batch size before graph launch.
        static_inputs["num_valid_tokens"].fill_(1)
        static_inputs["cache_seqlens"][:1].fill_(1)
        static_inputs["primary_slot_indices"][:1].fill_(0)
        static_inputs["aux_slot_indices"][:1].fill_(0)

    def release_static_buffers(self, bucket_size: int) -> None:
        self._buffers.pop(bucket_size, None)
        self._outputs.pop(bucket_size, None)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        primary_slot_indices: torch.Tensor,
        aux_slot_indices: torch.Tensor,
        num_valid_tokens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        attn = self.attn
        indexer = attn.indexer
        batch_size = hidden_states.shape[0]
        buffers = self._buffers.get(batch_size)
        if buffers is None:
            self.setup_static_buffers(batch_size)
            buffers = self._buffers[batch_size]
        outputs = self._outputs.get(batch_size)
        if outputs is None:
            self._setup_static_output_buffers(batch_size)
            outputs = self._outputs[batch_size]

        if num_valid_tokens is None:
            torch.ge(primary_slot_indices, 0, out=buffers.valid_mask)
        else:
            torch.lt(buffers.row_indices, num_valid_tokens, out=buffers.valid_mask)
            torch.ge(primary_slot_indices, 0, out=buffers.aux_valid_mask)
            torch.logical_and(buffers.valid_mask, buffers.aux_valid_mask, out=buffers.valid_mask)
        torch.ge(aux_slot_indices, 0, out=buffers.aux_valid_mask)
        torch.logical_and(buffers.valid_mask, buffers.aux_valid_mask, out=buffers.valid_mask)
        torch.where(
            buffers.valid_mask,
            primary_slot_indices,
            buffers.skip_slot_neg_ones,
            out=buffers.kv_primary_slot_indices,
        )
        torch.where(
            buffers.valid_mask,
            aux_slot_indices,
            buffers.skip_slot_neg_ones,
            out=buffers.kv_aux_slot_indices,
        )
        torch.where(
            buffers.valid_mask,
            primary_slot_indices,
            buffers.safe_slot_zeros,
            out=buffers.safe_primary_slot_indices,
        )
        torch.where(
            buffers.valid_mask,
            aux_slot_indices,
            buffers.safe_slot_zeros,
            out=buffers.safe_aux_slot_indices,
        )
        torch.where(
            buffers.valid_mask,
            cache_seqlens,
            buffers.safe_seqlen_zeros,
            out=buffers.safe_cache_seqlens,
        )
        torch.where(
            buffers.valid_mask,
            buffers.valid_rows_ones,
            buffers.valid_rows_zeros,
            out=buffers.valid_rows_bf16,
        )

        hidden_flat = hidden_states.view(batch_size, attn.hidden_size).contiguous()
        hidden_fp8, hidden_scale = act_quant(
            hidden_flat,
            num_valid_tokens=num_valid_tokens,
            scale_tma_aligned=num_valid_tokens is not None,
        )
        fused_qkv_a_weight = getattr(self.wrapper, "_fp8_qkv_a_proj", None)
        fused_qkv_a_scale = getattr(self.wrapper, "_fp8_qkv_a_scale", None)
        if fused_qkv_a_weight is None or fused_qkv_a_scale is None:
            raise RuntimeError(
                f"Layer {self.layer_idx}: GLM-5 graph requires fused Q-A/KV-A weights"
            )
        w8a8_deepgemm(
            hidden_fp8,
            hidden_scale,
            fused_qkv_a_weight,
            fused_qkv_a_scale,
            out=buffers.qkv_a,
            num_valid_tokens=num_valid_tokens,
            expected_m=batch_size,
        )
        # Masked DeepGEMM leaves rows >= num_valid_tokens unwritten, so this
        # buffer carries stale rows. They are not masked here: every consumer
        # of a masked-GEMM output is row-local (fused_rmsnorm, act_quant, RoPE,
        # the absorb GEMVs, FA3) and every write to state shared across rows is
        # already gated on the same row validity -- see the invariant note on
        # `attn_output`'s mask below, which is the one mask that must stay.
        q_a_normed = fused_rmsnorm(
            buffers.q_a,
            attn.q_a_layernorm.weight,
            attn.q_a_layernorm.eps,
            out=buffers.q_a_normed,
        )
        q_a_fp8, q_a_scale = act_quant(
            q_a_normed,
            num_valid_tokens=num_valid_tokens,
            scale_tma_aligned=num_valid_tokens is not None,
        )
        w8a8_deepgemm(
            q_a_fp8,
            q_a_scale,
            attn.q_b_proj.weight,
            self.wrapper.weight_dequant_scale["q_b_proj.weight_scale_inv"],
            out=buffers.q_flat,
            num_valid_tokens=num_valid_tokens,
            expected_m=batch_size,
        )
        q_view = buffers.q_flat.view(batch_size, 1, attn.num_heads, attn.q_head_dim).transpose(1, 2)
        buffers.q_nope.copy_(q_view[..., : attn.qk_nope_head_dim].squeeze(2).contiguous())
        buffers.q_rope_4d.copy_(q_view[..., attn.qk_nope_head_dim :].contiguous())

        offload_kv = _fused_rmsnorm_rope(
            buffers.new_compressed_kv,
            buffers.q_rope_4d,
            self.cos_table,
            self.sin_table,
            position_ids,
            attn.kv_a_layernorm.weight,
            attn.kv_lora_rank,
            attn.qk_rope_head_dim,
            eps=attn.kv_a_layernorm.eps,
        )
        # This tensor leaves the segment (the whole-model graph stages it for
        # host KV offload) and is also the GPU cache's store source, so it is
        # the one place an invalid row's bytes would escape -- and the ONE
        # place the cache's record is produced. Select instead of copy: a
        # select costs the same single launch, writes the exact zeros the
        # pre-mask path produced for invalid rows, and -- unlike a multiply,
        # where bf16 NaN * 0 is NaN -- it cannot carry a non-finite stale row
        # through.
        if self.primary_is_packed:
            # Quantize ONCE, into a scratch, then select invalid rows to the
            # zero record. Two launches rather than one because the select's
            # source cannot alias its destination; both move only
            # batch * 656 bytes.
            prepare_primary_mla_kv(
                primary_blocked_k=self.primary_blocked_k,
                compressed_kv=offload_kv.view(batch_size, -1),
                out=buffers.primary_k_packed,
                source_dim=self.primary_source_dim,
            )
            torch.where(
                buffers.valid_mask.view(batch_size, 1, 1, 1),
                buffers.primary_k_packed,
                buffers.packed_zeros_row,
                out=outputs.primary_k_tensor,
            )
        else:
            torch.where(
                buffers.valid_mask.view(batch_size, 1, 1, 1),
                offload_kv.view(batch_size, 1, 1, -1),
                buffers.valid_rows_zeros[:1],
                out=outputs.primary_k_tensor,
            )
        token_indices = position_ids.view(batch_size).to(dtype=torch.int32)
        run_paged_kv_token_update_fused(
            k_cache=self.primary_blocked_k,
            k_tokens=outputs.primary_k_tensor.view(batch_size, -1),
            page_table=self.primary_page_table,
            slot_indices=buffers.kv_primary_slot_indices,
            token_indices=token_indices,
            page_size_tokens=self.page_size,
            num_valid_tokens=num_valid_tokens,
        )

        cuda_wk_proj_gemm_only_out(
            hidden_flat,
            self.wrapper._indexer_cuda_weights,
            self.cuda_module,
            buffers.indexer_k_x_fp8,
            buffers.indexer_k_x_scale,
            buffers.indexer_k_tma_desc,
            buffers.indexer_k_raw,
            num_valid_tokens=num_valid_tokens,
        )
        k_normed = indexer.k_norm(buffers.indexer_k_raw)
        indexer_k_tensor = indexer._fused_rope_hadamard_or_fallback(
            k_normed.unsqueeze(1),
            position_ids.view(batch_size),
            max_seqlen=self.max_seqlen,
        ).unsqueeze(2)
        # Page-split aux cache: ONE launch quantizes the keys, writes the page's
        # e4m3 K region and fp32 scale region, and fills the two static host
        # staging buffers with the same bytes (no extra copy, no second
        # quantization). A bf16 AoS cache keeps the copy + paged update pair.
        if self.aux_is_page_split:
            write_aux_indexer_k(
                aux_blocked_k=self.aux_blocked_k,
                indexer_k=indexer_k_tensor,
                page_table=self.aux_page_table,
                slot_indices=buffers.kv_aux_slot_indices,
                token_indices=token_indices,
                page_size_tokens=self.aux_page_size,
                k_fp8_out=outputs.indexer_k_tensor,
                scales_out=outputs.indexer_k_scale,
                num_valid_tokens=num_valid_tokens,
            )
        else:
            outputs.indexer_k_tensor.copy_(indexer_k_tensor)
            write_aux_indexer_k(
                aux_blocked_k=self.aux_blocked_k,
                indexer_k=outputs.indexer_k_tensor,
                page_table=self.aux_page_table,
                slot_indices=buffers.kv_aux_slot_indices,
                token_indices=token_indices,
                page_size_tokens=self.aux_page_size,
                num_valid_tokens=num_valid_tokens,
            )

        if not self.all_short:
            head_gates_out(
                hidden_flat,
                indexer.weights_proj.weight.data,
                buffers.head_gates,
                scale=(indexer.index_n_heads ** -0.5) * (indexer.index_head_dim ** -0.5),
                num_valid_tokens=num_valid_tokens,
            )
            buffers.positions_expanded.copy_(
                position_ids.view(batch_size, 1).expand(batch_size, indexer.index_n_heads)
            )
            cuda_wq_b_proj_out(
                q_a_normed,
                self.wq_b_weights,
                self.cuda_module,
                buffers.q_x_fp8,
                buffers.q_x_scale,
                buffers.q_tma_desc,
                buffers.q_flat_indexer,
                num_valid_tokens=num_valid_tokens,
            )
            rope_hadamard_q_out(
                buffers.q_flat_indexer.view(batch_size, indexer.index_n_heads, indexer.index_head_dim),
                self.cos_table,
                self.sin_table,
                buffers.positions_expanded.view(-1),
                buffers.q_index,
            )
            self._score_and_topk(
                buffers,
                aux_slot_indices=buffers.safe_aux_slot_indices,
                head_gates=buffers.head_gates,
                cache_seqlens=buffers.safe_cache_seqlens,
                num_valid_tokens=num_valid_tokens,
            )

        fp8_q_absorb_out(
            buffers.q_nope,
            self.absorb_weights,
            buffers.absorbed_q,
            num_valid_tokens=num_valid_tokens,
        )
        if self.primary_is_packed:
            # A packed cache has no dense FP8 decode kernel in the pin, so the
            # all-short path is served by the SAME sparse kernel: it selects
            # every token of the row, which is exactly what the transform
            # already produces for a row within the selection budget (it falls
            # back to identity offsets). The transform therefore runs on both
            # paths here, unlike the bf16 branch below where all-short skips it.
            transform_selected_positions_out(
                self.primary_page_table,
                buffers.safe_cache_seqlens,
                buffers.top_k_indices,
                buffers.selected_token_ids,
                buffers.selected_lengths,
                page_size=self.page_size,
                primary_slot_indices=buffers.safe_primary_slot_indices,
                num_valid_tokens=num_valid_tokens,
            )
            # FlashMLA takes ONE 576-wide query; FA3 took the split
            # q/qv pair. Assembled here rather than written in place by the
            # producing kernels, because the absorb kernel hardcodes a
            # contiguous [B, H, N] output. Cost: batch * heads * 576 * 2 B
            # read+written per layer-step -- at TP 8 (heads_local = 8) and
            # batch 64 that is 0.59 MB/layer, ~46 MB/step, which is well
            # under the attention time this path is meant to save. Removing it
            # needs output strides on fp8_q_absorb_out and on the fused
            # RoPE kernel; deferred (see the slice report).
            buffers.query_packed[:, :, : attn.kv_lora_rank].copy_(
                buffers.absorbed_q
            )
            buffers.query_packed[:, :, attn.kv_lora_rank :].copy_(
                buffers.q_rope_4d.squeeze(2)
            )
            attn_out = self._run_sparse_fp8(buffers, batch_size)
        elif self.all_short:
            attn_out = self._run_all_short_fa3(buffers)
        else:
            transform_selected_positions_out(
                self.primary_page_table,
                buffers.safe_cache_seqlens,
                buffers.top_k_indices,
                buffers.selected_token_ids,
                buffers.selected_lengths,
                page_size=self.page_size,
                primary_slot_indices=buffers.safe_primary_slot_indices,
                num_valid_tokens=num_valid_tokens,
            )
            attn_out = self._run_selected_fa3(buffers)
        fp8_out_absorb_out(
            attn_out,
            self.absorb_weights,
            buffers.attn_heads,
            num_valid_tokens=num_valid_tokens,
        )
        attn_heads_flat = buffers.attn_heads.reshape(batch_size, attn.num_heads * attn.v_head_dim)
        attn_output_fp8, attn_output_scale = act_quant(
            attn_heads_flat,
            num_valid_tokens=num_valid_tokens,
            scale_tma_aligned=num_valid_tokens is not None,
        )
        w8a8_deepgemm(
            attn_output_fp8,
            attn_output_scale,
            attn.o_proj.weight,
            self.wrapper.weight_dequant_scale["o_proj.weight_scale_inv"],
            out=outputs.attn_output,
            num_valid_tokens=num_valid_tokens,
            expected_m=batch_size,
        )
        # The one load-bearing row mask in this segment. Everything above is
        # row-local or already row-gated, but `attn_output` leaves the segment
        # into the decoder layer's residual add and from there into the MoE
        # router, whose gate top-k and ragged dispatch DO mix rows: an invalid
        # row's stale activations would produce stale expert choices and inflate
        # the per-expert counts. It also covers the rows that are inside
        # num_valid_tokens but have no cache slot -- the masked GEMMs compute
        # those rows, so no num_valid_tokens gate zeroes them, and `valid_mask`
        # is what makes them inert.
        outputs.attn_output.mul_(buffers.valid_rows_bf16.view(batch_size, 1))

        graph_outputs = {
            "attn_output": outputs.attn_output.view(batch_size, 1, attn.hidden_size),
            "primary_k_tensor": outputs.primary_k_tensor,
            "indexer_k_tensor": outputs.indexer_k_tensor,
        }
        if outputs.indexer_k_scale is not None:
            graph_outputs["indexer_k_scale"] = outputs.indexer_k_scale
        return graph_outputs


def make_glm5_dsa_graph_segment_name(layer_idx: int) -> str:
    return f"glm5_layer_{layer_idx}_dsa_attn"


def make_glm5_full_dsa_graph_segment_name(layer_idx: int) -> str:
    return f"glm5_layer_{layer_idx}_full_dsa_attn"
