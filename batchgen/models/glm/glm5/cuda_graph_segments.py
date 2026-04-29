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
from typing import Dict

import torch

from batchgen.attention.dsa.sparse_decode_mla import (
    prepare_sparse_flash_mla_decode_inputs,
    run_prepared_sparse_flash_mla_decode,
)
from batchgen.attention.dsa.unified_selector import select_mla_kv_for_flashmla_bf16_out
from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen_kernels.attention.dsa.fp8_absorb import (
    FP8AbsorbWeights,
    fp8_out_absorb_out,
    fp8_q_absorb_out,
)
from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
    make_fp8_activation_scratch,
)
from batchgen_kernels.attention.dsa.fused_indexer_score import (
    FP8WqbWeightsCUDA,
    cuda_wq_b_proj_out,
    fused_paged_score_and_topk_out,
    rope_hadamard_q_out,
)
from batchgen_kernels.attention.dsa.query_pack import pack_flashmla_query_out


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


class Glm5DsaAttnSegment:
    """Graph-capturable GLM-5 DSA attention subsegment.

    The segment is full-indexer-only in this first integration slice.  Dense
    short rows and mixed dispatch remain outside graph routing until their
    production policy is separately validated.
    """

    def __init__(
        self,
        *,
        primary_blocked_k: torch.Tensor,
        aux_blocked_k: torch.Tensor,
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

        self.primary_blocked_k = primary_blocked_k
        self.aux_blocked_k = aux_blocked_k
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
        self.max_pages_per_seq = (max_seqlen + page_size - 1) // page_size
        self.max_aux_pages_per_seq = (
            max_seqlen + self.aux_page_size - 1
        ) // self.aux_page_size
        self._buffers: Dict[int, _Glm5DsaSegmentBuffers] = {}

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

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
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
            "primary_page_table": TensorSpec(
                ("batch_size", self.max_pages_per_seq), torch.int32, fill_value=0
            ),
            "aux_page_table": TensorSpec(
                ("batch_size", self.max_aux_pages_per_seq), torch.int32, fill_value=0
            ),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "attn_heads": TensorSpec(
                ("batch_size", 1, self.num_attn_heads, self.attn_out_dim),
                torch.bfloat16,
            ),
            "top_k_indices": TensorSpec(("batch_size", self.index_topk), torch.int64),
            "selected_lengths": TensorSpec(("batch_size",), torch.int32),
            "selected_mla_kv": TensorSpec(
                ("batch_size", self.index_topk, 1, self.kv_dim), torch.bfloat16
            ),
        }

    def setup_static_buffers(self, bucket_size: int) -> None:
        if bucket_size in self._buffers:
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
            dtype=torch.long,
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

    def forward(
        self,
        *,
        q_a: torch.Tensor,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        head_gates: torch.Tensor,
        cache_seqlens: torch.Tensor,
        positions_expanded: torch.Tensor,
        primary_page_table: torch.Tensor,
        aux_page_table: torch.Tensor,
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
        fused_paged_score_and_topk_out(
            buffers.q_index,
            self.aux_blocked_k,
            aux_page_table,
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
            primary_page_table,
            cache_seqlens,
            buffers.top_k_indices,
            self.page_size,
            buffers.selected_mla_kv,
            buffers.selected_lengths,
            None,
            buffers.row_modes,
            index_topk=self.index_topk,
            return_indices=False,
        )
        fp8_q_absorb_out(q_nope, self.absorb_weights, buffers.absorbed_q)
        pack_flashmla_query_out(buffers.absorbed_q, q_rope, buffers.query_states)
        attn_out = run_prepared_sparse_flash_mla_decode(buffers.prepared_flashmla)
        fp8_out_absorb_out(attn_out, self.absorb_weights, buffers.attn_heads)

        return {
            "attn_heads": buffers.attn_heads,
            "top_k_indices": buffers.top_k_indices,
            "selected_lengths": buffers.selected_lengths,
            "selected_mla_kv": buffers.selected_mla_kv,
        }


def make_glm5_dsa_graph_segment_name(layer_idx: int) -> str:
    return f"glm5_layer_{layer_idx}_dsa_attn"
