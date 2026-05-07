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

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

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
    cuda_wk_proj_gemm_only_out,
    make_fp8_activation_scratch,
)
from batchgen_kernels.attention.dsa.fused_indexer_score import (
    FP8WqbWeightsCUDA,
    cuda_wq_b_proj_out,
    fused_paged_score_and_topk_with_slots_out,
    rope_hadamard_q_out,
)
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
    valid_rows_bf16: torch.Tensor
    valid_rows_ones: torch.Tensor
    valid_rows_zeros: torch.Tensor
    safe_cache_seqlens: torch.Tensor
    padding_cache_seqlens: torch.Tensor
    safe_primary_slot_indices: torch.Tensor
    safe_aux_slot_indices: torch.Tensor
    zero_slot_indices: torch.Tensor
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

    def setup_static_buffers(self, bucket_size: int) -> None:
        if bucket_size in self._buffers:
            self._setup_static_output_buffers(bucket_size)
            return
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
        self._buffers[bucket_size] = _Glm5FullDsaSegmentBuffers(
            valid_mask=torch.empty(bucket_size, dtype=torch.bool, device=device),
            valid_rows_bf16=torch.empty(bucket_size, dtype=torch.bfloat16, device=device),
            valid_rows_ones=torch.ones(bucket_size, dtype=torch.bfloat16, device=device),
            valid_rows_zeros=torch.zeros(bucket_size, dtype=torch.bfloat16, device=device),
            safe_cache_seqlens=torch.empty(bucket_size, dtype=torch.int32, device=device),
            padding_cache_seqlens=torch.full(
                (bucket_size,),
                self._padding_selected_length(),
                dtype=torch.int32,
                device=device,
            ),
            safe_primary_slot_indices=torch.empty(bucket_size, dtype=torch.int32, device=device),
            safe_aux_slot_indices=torch.empty(bucket_size, dtype=torch.int32, device=device),
            zero_slot_indices=torch.zeros(bucket_size, dtype=torch.int32, device=device),
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
        self._setup_static_output_buffers(bucket_size)

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
        static_inputs["position_ids"].fill_(self.max_seqlen - 1)
        static_inputs["cache_seqlens"].fill_(self.max_seqlen)
        static_inputs["primary_slot_indices"].fill_(-1)
        static_inputs["aux_slot_indices"].fill_(-1)
        selected_lengths = torch.full(
            (bucket_size,),
            self._padding_selected_length(),
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

        torch.ge(primary_slot_indices, 0, out=buffers.valid_mask)
        torch.where(
            buffers.valid_mask,
            buffers.valid_rows_ones,
            buffers.valid_rows_zeros,
            out=buffers.valid_rows_bf16,
        )
        torch.where(
            buffers.valid_mask,
            cache_seqlens,
            buffers.padding_cache_seqlens,
            out=buffers.safe_cache_seqlens,
        )
        torch.where(
            buffers.valid_mask,
            primary_slot_indices,
            buffers.zero_slot_indices,
            out=buffers.safe_primary_slot_indices,
        )
        torch.where(
            buffers.valid_mask,
            aux_slot_indices,
            buffers.zero_slot_indices,
            out=buffers.safe_aux_slot_indices,
        )
        valid_rows_bf16_4d = buffers.valid_rows_bf16.view(batch_size, 1, 1, 1)

        hidden_flat = hidden_states.view(batch_size, attn.hidden_size).contiguous()
        hidden_fp8, hidden_scale = act_quant(hidden_flat)
        w8a8_deepgemm(
            hidden_fp8,
            hidden_scale,
            attn.q_a_proj.weight,
            self.wrapper.weight_dequant_scale["q_a_proj.weight_scale_inv"],
            out=buffers.q_a,
        )
        q_a_normed = attn.q_a_layernorm(buffers.q_a).contiguous()
        q_a_fp8, q_a_scale = act_quant(q_a_normed)
        w8a8_deepgemm(
            q_a_fp8,
            q_a_scale,
            attn.q_b_proj.weight,
            self.wrapper.weight_dequant_scale["q_b_proj.weight_scale_inv"],
            out=buffers.q_flat,
        )
        q_view = buffers.q_flat.view(batch_size, 1, attn.num_heads, attn.q_head_dim).transpose(1, 2)
        buffers.q_nope.copy_(q_view[..., : attn.qk_nope_head_dim].squeeze(2).contiguous())
        buffers.q_rope_4d.copy_(q_view[..., attn.qk_nope_head_dim :].contiguous())

        w8a8_deepgemm(
            hidden_fp8,
            hidden_scale,
            attn.kv_a_proj_with_mqa.weight,
            self.wrapper.weight_dequant_scale["kv_a_proj_with_mqa.weight_scale_inv"],
            out=buffers.new_compressed_kv.view(batch_size, -1),
        )
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
            slot_indices=primary_slot_indices,
            token_indices=token_indices,
            page_size_tokens=self.page_size,
        )

        cuda_wk_proj_gemm_only_out(
            hidden_flat,
            self.wrapper._indexer_cuda_weights,
            self.cuda_module,
            buffers.indexer_k_x_fp8,
            buffers.indexer_k_x_scale,
            buffers.indexer_k_tma_desc,
            buffers.indexer_k_raw,
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
            slot_indices=aux_slot_indices,
            token_indices=token_indices,
            page_size_tokens=self.aux_page_size,
        )

        gates = F.linear(hidden_flat.float(), indexer.weights_proj.weight.data.float())
        gates = gates * ((indexer.index_n_heads ** -0.5) * (indexer.index_head_dim ** -0.5))
        buffers.head_gates.copy_(gates.to(torch.float32))
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
        )
        buffers.selected_mla_kv.mul_(valid_rows_bf16_4d)

        fp8_q_absorb_out(buffers.q_nope, self.absorb_weights, buffers.absorbed_q)
        pack_flashmla_query_out(
            buffers.absorbed_q,
            buffers.q_rope_4d.squeeze(2),
            buffers.query_states,
        )
        buffers.query_states.mul_(valid_rows_bf16_4d)
        attn_out = run_prepared_sparse_flash_mla_decode(
            buffers.prepared_flashmla,
            tile_scheduler_metadata=flashmla_tile_scheduler_metadata,
            num_splits=flashmla_num_splits,
        )
        fp8_out_absorb_out(attn_out, self.absorb_weights, buffers.attn_heads)
        buffers.attn_heads.mul_(valid_rows_bf16_4d)
        attn_heads_flat = buffers.attn_heads.reshape(batch_size, attn.num_heads * attn.v_head_dim)
        attn_output_fp8, attn_output_scale = act_quant(attn_heads_flat)
        w8a8_deepgemm(
            attn_output_fp8,
            attn_output_scale,
            attn.o_proj.weight,
            self.wrapper.weight_dequant_scale["o_proj.weight_scale_inv"],
            out=outputs.attn_output,
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
