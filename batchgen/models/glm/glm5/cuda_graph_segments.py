"""CUDA-graph capturable GLM-5 DSA decode segments.

This module contains the first production integration bridge for the GLM-5
BF16 DSA graph path.  The segment starts at the already validated graph-safe
boundary:

    q_a, q_nope/q_rope, auxiliary indexer pages, primary MLA pages
      -> fused indexer score/top-k
      -> BF16 selected-KV gather
      -> FlashMLA dense decode over selected pages
      -> q/out absorb

It deliberately does not change the default GLM-5 decode path.  Callers must
explicitly construct and register this segment, and unsupported production
preconditions should fast-fail before replay rather than silently falling back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields as dataclass_fields
from typing import Dict, Optional

import torch
from batchgen.attention.mla.fa3_backend import act_quant
from batchgen.attention.mla.fused_rmsnorm_rope import (
    fused_rmsnorm_rope_with_q_native as _fused_rmsnorm_rope,
)
from batchgen.attention.dsa.sparse_decode_mla import (
    prepare_sparse_flash_mla_decode_tensor_metadata,
    prepare_sparse_flash_mla_decode_inputs,
    run_prepared_sparse_flash_mla_decode,
)
from batchgen.attention.dsa.unified_selector import select_mla_kv_for_flashmla_bf16_out
from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm
from batchgen_kernels.attention.dsa.fp8_absorb import (
    FP8AbsorbWeights,
    fp8_out_absorb_out,
    fp8_q_absorb_out,
)
from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
    _BLOCK_K as _FP8_SCRATCH_BLOCK_K,
    _BLOCK_M as _FP8_SCRATCH_BLOCK_M,
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
from batchgen_kernels.attention.dsa.query_pack import pack_flashmla_query_out
from batchgen_kernels.triton.kv_cache import run_paged_kv_token_update_fused


@dataclass
class _Glm5DsaSegmentBuffers:
    q_x_fp8: torch.Tensor
    q_x_scale: torch.Tensor
    q_tma_desc: torch.Tensor
    q_flat: torch.Tensor
    q_index: torch.Tensor
    agg_scores: torch.Tensor
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
    q_a: torch.Tensor
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
    agg_scores: torch.Tensor
    top_k_indices: torch.Tensor
    selected_mla_kv: torch.Tensor
    selected_lengths: torch.Tensor
    row_modes: torch.Tensor
    absorbed_q: torch.Tensor
    query_states: torch.Tensor
    attn_heads: torch.Tensor
    prepared_flashmla: object


@dataclass
class _Glm5FullDsaSegmentOutputs:
    primary_k_tensor: torch.Tensor
    indexer_k_tensor: torch.Tensor
    attn_output: torch.Tensor


# ---------------------------------------------------------------------------
# Per-bucket buffer VIEWS (allocate once at the largest bucket)
#
# Capture runs largest-first (batchgen_worker.py: `sorted(capture_buckets,
# reverse=True)`), so the first setup_static_buffers call carries the biggest
# bucket. Every later bucket takes leading-dim slices of that one allocation
# instead of a full fresh set. Slices share storage with the base tensor and
# capture happens AFTER setup, so the pointers a graph bakes at capture time
# are the base pointers — replaying bucket N writes into the base storage,
# which is exactly what its own views alias. Buckets are replayed one at a
# time, so cross-bucket aliasing of scratch is not a hazard (it is the same
# reuse the buffers already get across the 21 layers sharing this dict).
#
# Every field below has dim 0 == bucket size, and every initializer used for
# them is slice-invariant: torch.empty (uninitialized), torch.arange
# (arange(n)[:m] == arange(m)), torch.ones / torch.zeros / torch.full /
# Tensor.fill_ (constant). No field needs re-initialization after slicing.
_FULL_DSA_BUCKET_DIM_FIELDS = (
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
    "q_a",
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
    "top_k_indices",
    "selected_mla_kv",
    "selected_lengths",
    "row_modes",
    "absorbed_q",
    "query_states",
    "attn_heads",
)

# Fields that must be rebuilt per bucket rather than sliced:
#   * the two FP8 activation scratch triples — dim 0 is max(bucket, _BLOCK_M),
#     and the TMA descriptor bakes BOTH the base pointer and the global row
#     count, so the descriptor is re-encoded over the sliced view;
#   * prepared_flashmla — the tile-scheduler metadata and the synthetic
#     block table are sized by batch.
_FULL_DSA_REBUILT_FIELDS = (
    "indexer_k_x_fp8",
    "indexer_k_x_scale",
    "indexer_k_tma_desc",
    "q_x_fp8",
    "q_x_scale",
    "q_tma_desc",
    "prepared_flashmla",
)


def _assert_dsa_buffer_field_coverage(covered, where: str) -> None:
    """Fail loudly if the buffer dataclass grew a field the view path ignores."""
    expected = {f.name for f in dataclass_fields(_Glm5FullDsaSegmentBuffers)}
    covered = set(covered)
    missing = sorted(expected - covered)
    unknown = sorted(covered - expected)
    if missing or unknown:
        raise RuntimeError(
            f"{where}: _Glm5FullDsaSegmentBuffers field coverage is stale — "
            f"missing={missing} unknown={unknown}. Add each new field to the "
            "sliced set or to the rebuilt/placeholder set."
        )


_assert_dsa_buffer_field_coverage(
    _FULL_DSA_BUCKET_DIM_FIELDS + _FULL_DSA_REBUILT_FIELDS,
    "cuda_graph_segments",
)


def _slice_fp8_activation_scratch(
    base_x_fp8: torch.Tensor,
    base_x_scale: torch.Tensor,
    batch_size: int,
    module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-bucket view of a scratch pair made by ``make_fp8_activation_scratch``.

    Mirrors that helper exactly — same row padding, same TMA box dims — but the
    storage comes from the largest bucket's scratch instead of a fresh
    allocation. The dim-0 slice starts at offset 0, so the view keeps the base
    ``data_ptr``; only the descriptor's global row count narrows to this
    bucket's padded batch (which is what ``_validate_projection_out_buffers``
    checks the tensor against).
    """
    padded_batch = max(batch_size, _FP8_SCRATCH_BLOCK_M)
    if padded_batch > base_x_fp8.shape[0] or batch_size > base_x_scale.shape[0]:
        raise ValueError(
            f"FP8 activation scratch base too small: need "
            f"({padded_batch}, {batch_size}), have "
            f"({base_x_fp8.shape[0]}, {base_x_scale.shape[0]})"
        )
    x_fp8 = base_x_fp8[:padded_batch]
    x_scale = base_x_scale[:batch_size]
    a_tma_desc = module.create_tma_desc(
        x_fp8,
        padded_batch,
        x_fp8.shape[1],
        _FP8_SCRATCH_BLOCK_M,
        _FP8_SCRATCH_BLOCK_K,
    )
    return x_fp8, x_scale, a_tma_desc


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

        if self.kv_dim != self.kv_lora_rank + 64:
            raise ValueError(
                f"GLM-5 DSA selected KV dim must be kv_lora_rank + 64, got {self.kv_dim}"
            )
        if aux_blocked_k.shape[1] != self.aux_page_size:
            raise ValueError(
                f"aux_blocked_k page size {aux_blocked_k.shape[1]} != {self.aux_page_size}"
            )
        if aux_blocked_k.shape[3] != self.index_head_dim:
            raise ValueError(
                f"aux_blocked_k head dim {aux_blocked_k.shape[3]} != {self.index_head_dim}"
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
        agg_scores = torch.empty(
            bucket_size,
            self.max_seqlen,
            dtype=torch.float32,
            device=device,
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
        fused_paged_score_and_topk_with_slots_out(
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
        self._uses_shared_buffers = shared_buffers is not None
        self._buffers = shared_buffers if shared_buffers is not None else {}
        self._outputs: Dict[int, _Glm5FullDsaSegmentOutputs] = {}
        self._flashmla_metadata_specs: Dict[int, tuple[tuple[int, ...], torch.dtype, tuple[int, ...], torch.dtype]] = {}

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
        if self.primary_blocked_k.shape[3] != self.attn.kv_lora_rank + self.attn.qk_rope_head_dim:
            raise ValueError("primary_blocked_k last dimension does not match GLM-5 compressed KV")
        if self.aux_blocked_k.shape[3] != self.attn.indexer.index_head_dim:
            raise ValueError("aux_blocked_k last dimension does not match GLM-5 indexer K")

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
            self.attn.num_heads,
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
            "flashmla_tile_scheduler_metadata": TensorSpec(tile_shape, tile_dtype),
            "flashmla_num_splits": TensorSpec(num_splits_shape, num_splits_dtype),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "attn_output": TensorSpec(
                ("batch_size", 1, self.attn.hidden_size),
                torch.bfloat16,
            ),
            "primary_k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.primary_blocked_k.shape[3]),
                torch.bfloat16,
            ),
            "indexer_k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.aux_blocked_k.shape[3]),
                torch.bfloat16,
            ),
        }

    def _base_buffers_for(self, bucket_size: int):
        """Largest already-allocated bucket that can back ``bucket_size`` views.

        The base is read from ``self._buffers`` rather than an instance
        attribute on purpose: the 21 indexer layers SHARE one buffers dict
        (the worker's ``shared_dsa_buffers``), so whichever segment instance
        allocated a bucket first owns the storage for all of them.
        """
        if not self._buffers:
            return None
        base_bucket = max(self._buffers)
        if base_bucket < bucket_size:
            return None
        return self._buffers[base_bucket]

    def setup_static_buffers(self, bucket_size: int) -> None:
        if bucket_size in self._buffers:
            self._setup_static_output_buffers(bucket_size)
            return
        base = self._base_buffers_for(bucket_size)
        if base is None:
            if self._buffers:
                logging.warning(
                    "GLM-5 DSA segment: bucket %d is larger than every "
                    "allocated bucket %s — capture is not running "
                    "largest-first, allocating a second full buffer set",
                    bucket_size,
                    sorted(self._buffers),
                )
            # Base allocation. Graph capture happens AFTER this call, so every
            # pointer a graph bakes belongs to this storage (or to a view of
            # it), and this set stays alive as long as any bucket's views do.
            self._buffers[bucket_size] = self._allocate_static_buffers(bucket_size)
        else:
            self._buffers[bucket_size] = self._view_static_buffers(base, bucket_size)
        self._setup_static_output_buffers(bucket_size)

    def _view_static_buffers(
        self,
        base: _Glm5FullDsaSegmentBuffers,
        bucket_size: int,
    ) -> _Glm5FullDsaSegmentBuffers:
        """Build a bucket's buffer set as leading-dim slices of ``base``."""
        attn = self.attn
        buffers = {
            name: getattr(base, name)[:bucket_size]
            for name in _FULL_DSA_BUCKET_DIM_FIELDS
        }
        (
            buffers["indexer_k_x_fp8"],
            buffers["indexer_k_x_scale"],
            buffers["indexer_k_tma_desc"],
        ) = _slice_fp8_activation_scratch(
            base.indexer_k_x_fp8,
            base.indexer_k_x_scale,
            bucket_size,
            self.cuda_module,
        )
        (
            buffers["q_x_fp8"],
            buffers["q_x_scale"],
            buffers["q_tma_desc"],
        ) = _slice_fp8_activation_scratch(
            base.q_x_fp8,
            base.q_x_scale,
            bucket_size,
            self.cuda_module,
        )
        # Recomputed on the SLICED views: blocked_k is a reshape of
        # selected_mla_kv (same storage) and cache_seqlens is selected_lengths
        # itself, so the prepared object stays wired to this bucket's views.
        buffers["prepared_flashmla"] = prepare_sparse_flash_mla_decode_inputs(
            buffers["query_states"],
            buffers["selected_mla_kv"],
            buffers["selected_lengths"],
            attn.num_heads,
            float(attn.softmax_scale),
            head_dim_v=attn.kv_lora_rank,
            page_size=self.page_size,
        )
        return _Glm5FullDsaSegmentBuffers(**buffers)

    def _allocate_static_buffers(self, bucket_size: int) -> _Glm5FullDsaSegmentBuffers:
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

        selected_mla_kv = torch.empty(
            bucket_size,
            self.index_topk,
            1,
            kv_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        selected_lengths = torch.empty(bucket_size, dtype=torch.int32, device=device)
        selected_lengths.fill_(self._padding_selected_length())
        query_states = torch.empty(
            bucket_size,
            1,
            attn.num_heads,
            kv_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        prepared_flashmla = prepare_sparse_flash_mla_decode_inputs(
            query_states,
            selected_mla_kv,
            selected_lengths,
            attn.num_heads,
            float(attn.softmax_scale),
            head_dim_v=attn.kv_lora_rank,
            page_size=self.page_size,
        )
        return _Glm5FullDsaSegmentBuffers(
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
            q_a=torch.empty(bucket_size, attn.q_lora_rank, dtype=torch.bfloat16, device=device),
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
            new_compressed_kv=torch.empty(bucket_size, 1, kv_dim, dtype=torch.bfloat16, device=device),
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
            agg_scores=torch.empty(bucket_size, self.max_seqlen, dtype=torch.float32, device=device),
            top_k_indices=torch.empty(bucket_size, self.index_topk, dtype=torch.int32, device=device),
            selected_mla_kv=selected_mla_kv,
            selected_lengths=selected_lengths,
            row_modes=torch.empty(bucket_size, dtype=torch.int32, device=device),
            absorbed_q=torch.empty(
                bucket_size,
                attn.num_heads,
                attn.kv_lora_rank,
                dtype=torch.bfloat16,
                device=device,
            ),
            query_states=query_states,
            attn_heads=torch.empty(
                bucket_size,
                1,
                attn.num_heads,
                attn.v_head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            prepared_flashmla=prepared_flashmla,
        )

    def _setup_static_output_buffers(self, bucket_size: int) -> None:
        if bucket_size in self._outputs:
            return
        device = self.primary_blocked_k.device
        attn = self.attn
        kv_dim = attn.kv_lora_rank + attn.qk_rope_head_dim
        self._outputs[bucket_size] = _Glm5FullDsaSegmentOutputs(
            primary_k_tensor=torch.empty(bucket_size, 1, 1, kv_dim, dtype=torch.bfloat16, device=device),
            indexer_k_tensor=torch.empty(
                bucket_size,
                1,
                1,
                attn.indexer.index_head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            attn_output=torch.empty(bucket_size, attn.hidden_size, dtype=torch.bfloat16, device=device),
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
        # FlashMLA can illegal-access during graph capture with an all-zero
        # selected-length schedule. Capture one safe dummy row; replay overwrites
        # this scalar with the real local batch size before graph launch.
        static_inputs["num_valid_tokens"].fill_(1)
        static_inputs["cache_seqlens"][:1].fill_(1)
        static_inputs["primary_slot_indices"][:1].fill_(0)
        static_inputs["aux_slot_indices"][:1].fill_(0)
        selected_lengths = torch.ones(
            (bucket_size,),
            dtype=torch.int32,
            device=self.primary_blocked_k.device,
        )
        tile_scheduler_metadata, num_splits = prepare_sparse_flash_mla_decode_tensor_metadata(
            selected_lengths,
            self.attn.num_heads,
        )
        static_inputs["flashmla_tile_scheduler_metadata"].copy_(
            tile_scheduler_metadata,
            non_blocking=True,
        )
        static_inputs["flashmla_num_splits"].copy_(num_splits, non_blocking=True)

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
        flashmla_tile_scheduler_metadata: torch.Tensor,
        flashmla_num_splits: torch.Tensor,
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
        valid_rows_bf16_4d = buffers.valid_rows_bf16.view(batch_size, 1, 1, 1)

        hidden_flat = hidden_states.view(batch_size, attn.hidden_size).contiguous()
        hidden_fp8, hidden_scale = act_quant(
            hidden_flat,
            num_valid_tokens=num_valid_tokens,
            scale_tma_aligned=num_valid_tokens is not None,
        )
        w8a8_deepgemm(
            hidden_fp8,
            hidden_scale,
            attn.q_a_proj.weight,
            self.wrapper.weight_dequant_scale["q_a_proj.weight_scale_inv"],
            out=buffers.q_a,
            num_valid_tokens=num_valid_tokens,
            expected_m=batch_size,
        )
        buffers.q_a.mul_(buffers.valid_rows_bf16.view(batch_size, 1))
        q_a_normed = attn.q_a_layernorm(buffers.q_a).contiguous()
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
        buffers.q_flat.mul_(buffers.valid_rows_bf16.view(batch_size, 1))
        q_view = buffers.q_flat.view(batch_size, 1, attn.num_heads, attn.q_head_dim).transpose(1, 2)
        buffers.q_nope.copy_(q_view[..., : attn.qk_nope_head_dim].squeeze(2).contiguous())
        buffers.q_rope_4d.copy_(q_view[..., attn.qk_nope_head_dim :].contiguous())

        w8a8_deepgemm(
            hidden_fp8,
            hidden_scale,
            attn.kv_a_proj_with_mqa.weight,
            self.wrapper.weight_dequant_scale["kv_a_proj_with_mqa.weight_scale_inv"],
            out=buffers.new_compressed_kv.view(batch_size, -1),
            num_valid_tokens=num_valid_tokens,
            expected_m=batch_size,
        )
        buffers.new_compressed_kv.mul_(buffers.valid_rows_bf16.view(batch_size, 1, 1))
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
        outputs.primary_k_tensor.copy_(offload_kv.view(batch_size, 1, 1, -1))
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
        outputs.indexer_k_tensor.copy_(indexer_k_tensor)
        run_paged_kv_token_update_fused(
            k_cache=self.aux_blocked_k,
            k_tokens=outputs.indexer_k_tensor.view(batch_size, -1),
            page_table=self.aux_page_table,
            slot_indices=buffers.kv_aux_slot_indices,
            token_indices=token_indices,
            page_size_tokens=self.aux_page_size,
            num_valid_tokens=num_valid_tokens,
        )

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
        fused_paged_score_and_topk_with_slots_out(
            buffers.q_index,
            self.aux_blocked_k,
            self.aux_page_table,
            buffers.safe_aux_slot_indices,
            buffers.head_gates,
            buffers.safe_cache_seqlens,
            buffers.agg_scores,
            buffers.top_k_indices,
            topk=self.index_topk,
            page_size=self.aux_page_size,
            max_seqlen=self.max_seqlen,
            num_valid_tokens=num_valid_tokens,
        )
        select_mla_kv_for_flashmla_bf16_out(
            self.primary_blocked_k,
            self.primary_page_table,
            buffers.safe_cache_seqlens,
            buffers.top_k_indices,
            self.page_size,
            buffers.selected_mla_kv,
            buffers.selected_lengths,
            None,
            buffers.row_modes,
            index_topk=self.index_topk,
            return_indices=False,
            primary_slot_indices=buffers.safe_primary_slot_indices,
            num_valid_tokens=num_valid_tokens,
        )

        fp8_q_absorb_out(
            buffers.q_nope,
            self.absorb_weights,
            buffers.absorbed_q,
            num_valid_tokens=num_valid_tokens,
        )
        pack_flashmla_query_out(
            buffers.absorbed_q,
            buffers.q_rope_4d.squeeze(2),
            buffers.query_states,
            num_valid_tokens=num_valid_tokens,
        )
        buffers.query_states.mul_(valid_rows_bf16_4d)
        attn_out = run_prepared_sparse_flash_mla_decode(
            buffers.prepared_flashmla,
            tile_scheduler_metadata=flashmla_tile_scheduler_metadata,
            num_splits=flashmla_num_splits,
        )
        fp8_out_absorb_out(
            attn_out,
            self.absorb_weights,
            buffers.attn_heads,
            num_valid_tokens=num_valid_tokens,
        )
        buffers.attn_heads.mul_(valid_rows_bf16_4d)
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
        outputs.attn_output.mul_(buffers.valid_rows_bf16.view(batch_size, 1))

        return {
            "attn_output": outputs.attn_output.view(batch_size, 1, attn.hidden_size),
            "primary_k_tensor": outputs.primary_k_tensor,
            "indexer_k_tensor": outputs.indexer_k_tensor,
        }


def make_glm5_dsa_graph_segment_name(layer_idx: int) -> str:
    return f"glm5_layer_{layer_idx}_dsa_attn"


def make_glm5_full_dsa_graph_segment_name(layer_idx: int) -> str:
    return f"glm5_layer_{layer_idx}_full_dsa_attn"
