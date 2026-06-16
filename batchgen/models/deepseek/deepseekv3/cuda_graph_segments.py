"""CUDA-graph capturable DeepSeek-R1 / DeepSeek-V3 whole-model decode segments.

This is a faithful clone of the GLM-5 segment trio
(``glm/glm5/moe_cuda_graph_segments.py`` + ``glm/glm5/whole_model_cuda_graph_segments.py``)
plus the Kimi-K2.5 inlined-MLA attention segment
(``moonshotai/kimi_k25/cuda_graph_segments.py``) with the R1-specific swaps:

  1. R1MoEGraphSegment routing: GLM-5's ``glm5_router_gemm_cuda`` +
     ``gate_sigmoid_topk_cuda`` are REPLACED by R1's group-limited gate, called
     verbatim as ``self.moe.gate(all_tokens)`` (``DeepseekV3MoE``'s ``MoEGate``).
     The rest of the MoE pipeline (all_gather -> gate -> GLM-5 padding mask ->
     dispatch_scatter_3d -> FP8 blockwise S1/S3 -> reduce_weighted_scatter ->
     all_reduce -> slice + shared expert) is identical to GLM-5.

  2. R1AttnSegment: MLA (no DSA indexer, no aux KV). Single page_table +
     slot_indices. FP8 MLA decode ops inlined from
     ``flashmla_backend.mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv``.
     kv_b_proj is pre-dequantized into static q_absorb/out_absorb at construction
     (NOT per step inside the graph). KV write uses the static page_table +
     slot_indices (K25's ``run_paged_kv_token_update_fused``), not the eager
     ``update_layer_decode_new_token``.

  3. R1WholeModelSegment: clones ``Glm5WholeModelSegment`` but uses
     R1AttnSegment + R1MoEGraphSegment per layer, DROPS all aux-KV plumbing, runs
     ``DenseMLP.forward`` eagerly for layers < first_k_dense_replace, and gates
     the in-graph lm_head behind ``capture_lm_head``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.models.wrappers.attention import AttnWrapperBase
from batchgen.moe.dispatch_scatter_3d import dispatch_scatter_3d, reduce_weighted_scatter
from batchgen.moe.grouped_fp8_blockwise_moe import (
    grouped_fp8_blockwise_fused_s1,
    grouped_fp8_blockwise_s3,
)

logger = logging.getLogger(__name__)


def _act_quant_3d(x: torch.Tensor, seqlens: torch.Tensor):
    from batchgen_kernels.moe._C_fp8_blockwise_ops import act_quant_3d

    return act_quant_3d(x, seqlens)


def make_r1_moe_graph_segment_name(layer_idx: int) -> str:
    return f"r1_layer_{layer_idx}_moe"


def make_r1_attn_graph_segment_name(layer_idx: int) -> str:
    return f"r1_layer_{layer_idx}_attn"


def make_r1_whole_model_graph_segment_name() -> str:
    return "r1_whole_model"


# ============================================================================
# R1 MoE graph buffer pool  (clone of Glm5MoEGraphBufferPool)
# ============================================================================
#
# R1 swap vs GLM-5: routing is the eager group-limited gate, so router_logits /
# topk_indices / topk_weights are produced by ``self.moe.gate(...)`` (its
# intermediate allocations live in the CUDA-graph memory pool). Only the masked
# topk buffers + the padding sentinels are kept here, since GLM-5's in-graph
# padding mask writes them via ``torch.where(out=...)``.


@dataclass
class _R1MoEGraphBuffers:
    padded: torch.Tensor
    all_tokens: torch.Tensor
    topk_masked_indices: torch.Tensor
    topk_masked_weights: torch.Tensor
    topk_negative_ones: torch.Tensor
    topk_zero_weights: torch.Tensor
    rank_ids: torch.Tensor
    local_pos: torch.Tensor
    expert_counts: torch.Tensor
    expert_counters: torch.Tensor
    topk_pos: torch.Tensor
    dispatched_x: torch.Tensor
    intermediate: torch.Tensor
    expert_out: torch.Tensor
    routed_global_output: torch.Tensor
    local_moe_output: torch.Tensor
    cu_seqlens: torch.Tensor
    max_tokens_padded: int


class R1MoEGraphBufferPool:
    """Shared static buffers for all R1 MoE graph segments on one rank."""

    _MTP_BLOCK = 128

    def __init__(
        self,
        *,
        world_size: int,
        hidden_size: int,
        num_experts_per_tok: int,
        num_local_experts: int,
        intermediate_size: int,
        device: torch.device,
        bucket_sizes: List[int],
        base_mtp: int,
    ) -> None:
        if not bucket_sizes:
            raise ValueError("R1 MoE graph requires at least one bucket size")
        self.world_size = int(world_size)
        self.hidden_size = int(hidden_size)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.num_local_experts = int(num_local_experts)
        self.intermediate_size = int(intermediate_size)
        self.device = device
        self.bucket_sizes = sorted({int(b) for b in bucket_sizes})
        self.base_mtp = int(base_mtp)
        self._base: Dict[str, torch.Tensor] = {}
        self._views: Dict[int, _R1MoEGraphBuffers] = {}

    def setup(self) -> None:
        if self._base:
            return

        max_bucket = max(self.bucket_sizes)
        max_global = self.world_size * max_bucket
        mtp = max(self.base_mtp, self._round_up(max_global, self._MTP_BLOCK))
        rows = self.num_local_experts * mtp
        nk = max_global * self.num_experts_per_tok
        d = self.device
        h = self.hidden_size
        n = self.intermediate_size
        k = self.num_experts_per_tok

        b = self._base
        b["padded"] = torch.zeros(max_bucket, h, dtype=torch.bfloat16, device=d)
        b["all_tokens"] = torch.zeros(max_global, h, dtype=torch.bfloat16, device=d)
        b["topk_masked_indices"] = torch.empty(max_global, k, dtype=torch.int32, device=d)
        b["topk_masked_weights"] = torch.empty(max_global, k, dtype=torch.float32, device=d)
        b["topk_negative_ones"] = torch.full((max_global, k), -1, dtype=torch.int32, device=d)
        b["topk_zero_weights"] = torch.zeros(max_global, k, dtype=torch.float32, device=d)
        b["expert_counts"] = torch.zeros(self.num_local_experts, dtype=torch.int32, device=d)
        b["expert_counters"] = torch.zeros(self.num_local_experts, dtype=torch.int32, device=d)
        b["topk_pos"] = torch.full((nk,), -1, dtype=torch.int32, device=d)
        b["dispatched_x"] = torch.zeros(rows, h, dtype=torch.bfloat16, device=d)
        b["intermediate"] = torch.empty(rows, n, dtype=torch.bfloat16, device=d)
        b["expert_out"] = torch.empty(rows, h, dtype=torch.bfloat16, device=d)
        b["routed_global_output"] = torch.empty(max_global, h, dtype=torch.bfloat16, device=d)
        b["local_moe_output"] = torch.empty(max_bucket, h, dtype=torch.bfloat16, device=d)
        b["cu_seqlens"] = torch.arange(
            0,
            (self.num_local_experts + 1) * mtp,
            mtp,
            dtype=torch.int32,
            device=d,
        )

        total_bytes = sum(t.nelement() * t.element_size() for t in b.values())
        logger.info(
            "R1MoEGraphBufferPool: allocated %.2f GiB "
            "(max_bucket=%d, world_size=%d, mtp=%d, rows=%d)",
            total_bytes / (1024**3),
            max_bucket,
            self.world_size,
            mtp,
            rows,
        )

        for bucket_size in self.bucket_sizes:
            self._create_view(bucket_size, mtp)

    def get(self, bucket_size: int) -> _R1MoEGraphBuffers:
        self.setup()
        return self._views[int(bucket_size)]

    def _create_view(self, bucket_size: int, mtp: int) -> None:
        global_rows = self.world_size * bucket_size
        nk = global_rows * self.num_experts_per_tok
        rows = self.num_local_experts * mtp
        b = self._base
        positions = torch.arange(global_rows, dtype=torch.int64, device=self.device)
        self._views[bucket_size] = _R1MoEGraphBuffers(
            padded=b["padded"][:bucket_size],
            all_tokens=b["all_tokens"][:global_rows],
            topk_masked_indices=b["topk_masked_indices"][:global_rows],
            topk_masked_weights=b["topk_masked_weights"][:global_rows],
            topk_negative_ones=b["topk_negative_ones"][:global_rows],
            topk_zero_weights=b["topk_zero_weights"][:global_rows],
            rank_ids=positions // bucket_size,
            local_pos=positions % bucket_size,
            expert_counts=b["expert_counts"],
            expert_counters=b["expert_counters"],
            topk_pos=b["topk_pos"][:nk],
            dispatched_x=b["dispatched_x"][:rows],
            intermediate=b["intermediate"][:rows],
            expert_out=b["expert_out"][:rows],
            routed_global_output=b["routed_global_output"][:global_rows],
            local_moe_output=b["local_moe_output"][:bucket_size],
            cu_seqlens=b["cu_seqlens"],
            max_tokens_padded=mtp,
        )

    @staticmethod
    def _round_up(value: int, block: int) -> int:
        return ((value + block - 1) // block) * block

    def release(self) -> None:
        self._views.clear()
        self._base.clear()


# ============================================================================
# R1 MoE graph segment  (clone of Glm5MoEGraphSegment)
# ============================================================================


class R1MoEGraphSegment:
    """Graph-capturable full DeepSeek-R1 MoE decode module segment."""

    def __init__(
        self,
        moe,
        pool: R1MoEGraphBufferPool,
        comm,
        *,
        world_size: int,
        rank: int,
        device: torch.device,
    ) -> None:
        self.moe = moe
        self.pool = pool
        self.comm = comm
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.device = device
        self.hidden_size = int(moe.hidden_size)
        self.num_experts_per_tok = int(moe.num_experts_per_tok)
        self.num_local_experts = int(moe.experts_per_rank)
        self.expert_start = int(moe.routed_expert_start_idx)
        self.intermediate_size = int(moe.moe_intermediate_size)

        if not getattr(moe, "_fp8_blockwise_ready", False):
            raise RuntimeError(
                f"Layer {moe.layer_idx}: R1 MoE graph requires FP8 blockwise weights"
            )
        if comm is None:
            raise RuntimeError(f"Layer {moe.layer_idx}: R1 MoE graph requires EP communicator")
        self._validate_shared_expert_graph_safe()

    def _validate_shared_expert_graph_safe(self) -> None:
        # R1's shared expert is a persistent FP8 DeepseekV3Expert; its weights are
        # resident (no per-forward load/free), so the graph can call it eagerly.
        shared = getattr(self.moe, "shared_experts", None)
        if shared is None:
            raise RuntimeError(
                f"Layer {self.moe.layer_idx}: R1 MoE graph requires a shared expert module"
            )

    def setup_static_buffers(self, bucket_size: int) -> None:
        if hasattr(self.comm, "disabled"):
            self.comm.disabled = False
        self.pool.setup()

    def release_static_buffers(self, bucket_size: int) -> None:
        self.pool.release()

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "padded": TensorSpec(("batch_size", self.hidden_size), torch.bfloat16),
            "rank_token_counts": TensorSpec((self.world_size,), torch.int64),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "moe_output": TensorSpec(
                ("batch_size", self.hidden_size),
                torch.bfloat16,
            ),
        }

    def forward(
        self,
        *,
        padded: torch.Tensor,
        rank_token_counts: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        bucket_size = padded.shape[0]
        bufs = self.pool.get(bucket_size)
        global_rows = self.world_size * bucket_size

        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                bufs.all_tokens,
                padded,
                stream=torch.cuda.current_stream(self.device),
            )

        # R1 swap: group-limited gate called verbatim (DeepseekV3MoE.MoEGate).
        # It is fixed-shape per bucket (global_rows tokens) and @torch.inference_mode;
        # its intermediate allocations are captured into the graph memory pool.
        topk_indices, topk_weights = self.moe.gate(bufs.all_tokens)

        # GLM-5 in-graph padding mask (kept exactly) — what makes R1's 0-token
        # DP-rank guard graph-safe: padded rows get expert -1 / weight 0 so they
        # never inflate expert_counts.
        valid_per_row = rank_token_counts[bufs.rank_ids]
        padding_mask = bufs.local_pos >= valid_per_row
        padding_mask_2d = padding_mask.unsqueeze(1).expand_as(bufs.topk_masked_indices)
        torch.where(
            padding_mask_2d,
            bufs.topk_negative_ones,
            topk_indices,
            out=bufs.topk_masked_indices,
        )
        torch.where(
            padding_mask_2d,
            bufs.topk_zero_weights,
            topk_weights,
            out=bufs.topk_masked_weights,
        )

        bufs.dispatched_x.zero_()
        expert_counts, topk_pos = dispatch_scatter_3d(
            bufs.all_tokens,
            bufs.topk_masked_indices,
            bufs.dispatched_x,
            self.expert_start,
            self.num_local_experts,
            bufs.max_tokens_padded,
            bufs.expert_counts,
            bufs.expert_counters,
            bufs.topk_pos,
        )

        self._fp8_blockwise_gemm_3d(bufs, expert_counts)

        bufs.routed_global_output.zero_()
        routed_global_output = reduce_weighted_scatter(
            bufs.expert_out,
            topk_pos,
            bufs.topk_masked_weights,
            global_rows,
            self.hidden_size,
            self.num_experts_per_tok,
            output=bufs.routed_global_output,
        )

        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                routed_global_output,
                op=dist.ReduceOp.SUM,
                stream=torch.cuda.current_stream(self.device),
            )

        start = self.rank * bucket_size
        bufs.local_moe_output.copy_(routed_global_output[start:start + bucket_size])
        # R1 swap: shared expert add is eager inside the segment
        # (DeepseekV3MoE.shared_expert_forward -> self.shared_experts(identity)).
        bufs.local_moe_output.add_(self.moe.shared_expert_forward(padded))
        return {"moe_output": bufs.local_moe_output}

    def _fp8_blockwise_gemm_3d(
        self,
        bufs: _R1MoEGraphBuffers,
        expert_counts: torch.Tensor,
    ) -> None:
        e = self.num_local_experts
        h = self.hidden_size
        n = self.intermediate_size
        mtp = bufs.max_tokens_padded
        seqlens = expert_counts[:e]
        avg = max(mtp // max(e, 1), 1)

        x_3d = bufs.dispatched_x.view(e, mtp, h)
        x_quant_3d, x_scale_3d = _act_quant_3d(x_3d, seqlens)
        x_quant = x_quant_3d.view(e * mtp, h)
        x_scale_t = x_scale_3d.view(e * mtp, -1).t().contiguous()

        s1_result = grouped_fp8_blockwise_fused_s1(
            x_quant.view(torch.float8_e4m3fn),
            x_scale_t,
            self.moe.fp8_gate_w3d.view(torch.float8_e4m3fn),
            self.moe.fp8_up_w3d.view(torch.float8_e4m3fn),
            self.moe.fp8_gate_ws3d,
            self.moe.fp8_up_ws3d,
            seqlens,
            bufs.cu_seqlens,
            avg,
            output=bufs.intermediate,
        )
        inter_quant_3d, inter_scale_3d = _act_quant_3d(
            s1_result.view(e, mtp, n),
            seqlens,
        )
        inter_quant = inter_quant_3d.view(e * mtp, n)
        inter_scale_t = inter_scale_3d.view(e * mtp, -1).t().contiguous()

        grouped_fp8_blockwise_s3(
            inter_quant.view(torch.float8_e4m3fn),
            inter_scale_t,
            self.moe.fp8_down_w3d.view(torch.float8_e4m3fn),
            self.moe.fp8_down_ws3d,
            seqlens,
            bufs.cu_seqlens,
            avg,
            output=bufs.expert_out,
        )


# ============================================================================
# R1 MLA attention segment  (structural clone of K25AttnSegment, FP8 ops from
# flashmla_backend.mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv)
# ============================================================================


class R1AttnSegment:
    """DeepSeek-R1 MLA attention block as a single CUDA-graph-capturable segment.

    Covers: input_layernorm -> FP8 Q/KV projections -> KV norm+RoPE -> paged KV
            write (static page_table) -> q absorb -> FlashMLA -> out absorb ->
            FP8 o_proj -> residual + post_attn_norm.

    Inputs:  hidden_states [B,1,H], cache_seqlens [B], page_table [B,max_pages],
             slot_indices [B]
    Outputs: normed [B,1,H] (MoE/MLP input), residual [B,1,H],
             k_tensor [B,1,1,kv_dim]

    R1 vs K25: single page_table + slot_indices (no DSA indexer / aux KV); FP8
    projections (w8a8_deepgemm + act_quant) instead of BF16 F.linear; kv_b_proj
    is pre-dequantized into static q_absorb/out_absorb at construction.
    """

    def __init__(
        self,
        decoder_layer,
        attn_wrapper,
        layer_idx: int,
        max_seq_len: int,
        max_pages_per_seq: int,
        page_size_tokens: int,
    ) -> None:
        # Unwrap the actual attention module (DeepSeekAttnWrapper.module).
        self.attn_wrapper = attn_wrapper
        self.attn_mod = attn_wrapper.module if hasattr(attn_wrapper, "module") else attn_wrapper
        self.layer_idx = int(layer_idx)
        self.max_seq_len = int(max_seq_len)
        self.max_pages_per_seq = int(max_pages_per_seq)
        self.page_size_tokens = int(page_size_tokens)

        # FP8 weight dequant scales (resident dict on the wrapper).
        self.weight_scale = getattr(attn_wrapper, "weight_dequant_scale", None)
        if self.weight_scale is None:
            raise RuntimeError(
                f"Layer {self.layer_idx}: R1 MLA graph requires the attn wrapper's "
                "weight_dequant_scale (FP8 projection scales)"
            )

        # Layer norms (owned by the decoder layer).
        self.input_ln_weight = decoder_layer.input_layernorm.weight
        self.input_ln_eps = decoder_layer.input_layernorm.variance_epsilon
        self.post_ln_weight = decoder_layer.post_attention_layernorm.weight
        self.post_ln_eps = decoder_layer.post_attention_layernorm.variance_epsilon

        attn = self.attn_mod
        self.hidden_size = int(attn.hidden_size)
        self.num_heads = int(attn.num_heads)              # 128
        self.q_lora_rank = int(attn.q_lora_rank)          # 1536
        self.kv_lora_rank = int(attn.kv_lora_rank)        # 512
        self.qk_nope_head_dim = int(attn.qk_nope_head_dim)  # 128
        self.qk_rope_head_dim = int(attn.qk_rope_head_dim)  # 64
        self.v_head_dim = int(attn.v_head_dim)            # 128
        self.q_head_dim = int(attn.q_head_dim)            # 192
        self.kv_dim = self.kv_lora_rank + self.qk_rope_head_dim  # 576
        self.softmax_scale = attn.softmax_scale

        # Cached fused functions (avoid repeated lookups).
        self._fused_add_rmsnorm = decoder_layer._get_fused_add_rmsnorm_fn()
        from batchgen.models.deepseek.deepseekv3.model import RMSNorm

        self._fused_rmsnorm = RMSNorm._get_fused_fn()

        # Pre-dequantize kv_b_proj into static q_absorb / out_absorb at
        # construction (R1's weights are FP8; the eager decode dequantizes per
        # step at flashmla_backend.py:1339-1344 — we must NOT do that in-graph).
        # Prefer DeepseekV3Attention.initialize()'s precomputed absorb tensors;
        # fall back to dequantizing kv_b_proj here.
        q_absorb = getattr(attn, "q_absorb", None)
        out_absorb = getattr(attn, "out_absorb", None)
        if q_absorb is None or out_absorb is None:
            from batchgen.attention.mla.flashmla_backend import deepseek_v3_dequantization

            kv_b_proj = deepseek_v3_dequantization(
                attn.kv_b_proj.weight.data,
                self.weight_scale["kv_b_proj.weight_scale_inv"],
            ).view(self.num_heads, -1, self.kv_lora_rank)
            q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
            out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]
        self.q_absorb = q_absorb.contiguous()
        self.out_absorb = out_absorb.contiguous()

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "cache_seqlens": TensorSpec(
                ("batch_size",), torch.int32, fill_value=1
            ),
            "page_table": TensorSpec(
                ("batch_size", self.max_pages_per_seq), torch.int32, fill_value=0
            ),
            "slot_indices": TensorSpec(
                ("batch_size",), torch.int32, fill_value=0
            ),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "normed": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "residual": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "k_tensor": TensorSpec(
                ("batch_size", 1, 1, self.kv_dim), torch.bfloat16
            ),
        }

    def _rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        if self._fused_rmsnorm is not None:
            return self._fused_rmsnorm(x, weight, eps)
        h = x.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        return (weight * (h * torch.rsqrt(variance + eps))).to(x.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Inlined FP8 MLA attention with static page_table for graph capture.

        Mirrors mla_decoding_flashmla_attn_mode_3_bf16_with_pagekv except:
        - KV write uses the static page_table + slot_indices (K25 pattern), not
          gpu_paged_kv_manager.update_layer_decode_new_token.
        - q_absorb / out_absorb are the pre-dequantized static tensors (no
          per-step kv_b_proj dequant).
        - No host syncs / .item() (the eager RoPE-overflow .max().item() check
          is dropped; positions are clamped instead, K25-style).
        """
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata
        from batchgen.attention.mla.fa3_backend import act_quant
        from batchgen.gemm.w8a8_deepgemm import w8a8_deepgemm
        from batchgen.attention.mla.fused_rmsnorm_rope import fused_rmsnorm_rope_with_q
        from batchgen_kernels.triton.kv_cache import run_paged_kv_token_update_fused

        B = hidden_states.shape[0]
        attn = self.attn_mod
        gpu_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        ws = self.weight_scale

        # position = cache_seqlens - 1 (this token's slot, 0-indexed). Clamp to
        # avoid host syncs on padded rows (K25-style).
        q_position_ids = (cache_seqlens - 1).clamp(min=0).unsqueeze(1).to(torch.int64)

        # === Pre-attn RMSNorm ===
        residual = hidden_states
        normed = self._rmsnorm(hidden_states, self.input_ln_weight, self.input_ln_eps)
        normed_sq = normed.squeeze(1)  # [B, H]

        # === FP8 Q + KV projections (w8a8_deepgemm) ===
        hs_fp8, hs_scale = act_quant(normed_sq)
        q = w8a8_deepgemm(hs_fp8, hs_scale, attn.q_a_proj.weight, ws["q_a_proj.weight_scale_inv"])
        new_compressed_kv = w8a8_deepgemm(
            hs_fp8, hs_scale, attn.kv_a_proj_with_mqa.weight,
            ws["kv_a_proj_with_mqa.weight_scale_inv"],
        ).view(B, 1, -1)
        q = attn.q_a_layernorm(q)
        q_fp8, q_scale = act_quant(q)
        q = w8a8_deepgemm(q_fp8, q_scale, attn.q_b_proj.weight, ws["q_b_proj.weight_scale_inv"])

        # === Q reshape + split ===
        q = q.view(B, 1, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = q_pe.contiguous()

        # === RoPE cos/sin (pre-extended to max_seq_len during init) ===
        cos, sin = attn.rotary_emb(q_pe, seq_len=self.max_seq_len)

        # === Fused KV norm + RoPE on both KV and Q ===
        offload_kv = fused_rmsnorm_rope_with_q(
            new_compressed_kv, q_pe, cos, sin,
            q_position_ids, attn.kv_a_layernorm.weight,
            self.kv_lora_rank, self.qk_rope_head_dim,
        )
        k_tensor = offload_kv.view(B, 1, 1, offload_kv.size(-1))

        # === KV write — use STATIC page_table + slot_indices (K25 pattern) ===
        # Fetch the K-cache tensor WITHOUT requiring the active page table to be
        # built: at whole-model capture time it is not, and this segment uses the
        # static page_table input (not the manager's), so the page-table check in
        # get_layer_kv_with_page_table must be bypassed.
        blocked_k = gpu_kv_manager.get_layer_k_cache(self.layer_idx)
        run_paged_kv_token_update_fused(
            k_cache=blocked_k,
            k_tokens=k_tensor.view(B, -1),
            page_table=page_table,
            slot_indices=slot_indices,
            token_indices=q_position_ids.squeeze(-1).to(torch.int32),
            page_size_tokens=self.page_size_tokens,
        )

        # === Q absorb + query states construction ===
        qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        query_states = torch.empty(
            B, self.num_heads, 1, qk_head_dim,
            dtype=blocked_k.dtype, device=hidden_states.device,
        )
        q_nope_sq = q_nope.squeeze(2)
        query_states[:, :, :, : self.kv_lora_rank] = torch.einsum(
            "bhd,hdc->bhc", q_nope_sq, self.q_absorb
        ).view(B, self.num_heads, 1, self.kv_lora_rank)
        query_states[:, :, :, self.kv_lora_rank :] = q_pe
        query_states = query_states.view(B, 1, self.num_heads, qk_head_dim)

        # === FlashMLA — use STATIC page_table; metadata computed in-graph ===
        tile_scheduler_metadata, num_splits = get_mla_metadata(
            cache_seqlens, self.num_heads, 1
        )
        attn_out, _ = flash_mla_with_kvcache(
            query_states, blocked_k,
            page_table,
            cache_seqlens,
            self.kv_lora_rank,  # head_dim_v = 512
            tile_scheduler_metadata, num_splits,
            self.softmax_scale, True,
        )

        # === Output absorb + FP8 O_proj ===
        attn_output = torch.einsum("bqhc,hdc->bhqd", attn_out, self.out_absorb)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(B, self.num_heads * self.v_head_dim)
        ao_fp8, ao_scale = act_quant(attn_output)
        attn_output = w8a8_deepgemm(
            ao_fp8, ao_scale, attn.o_proj.weight, ws["o_proj.weight_scale_inv"]
        )
        attn_output = attn_output.view(B, 1, -1)

        # === Post-attn: residual add + RMSNorm ===
        if self._fused_add_rmsnorm is not None:
            normed_out, residual_out = self._fused_add_rmsnorm(
                residual, attn_output,
                self.post_ln_weight, self.post_ln_eps,
            )
        else:
            combined = residual + attn_output
            residual_out = combined
            normed_out = self._rmsnorm(combined, self.post_ln_weight, self.post_ln_eps)

        return {"normed": normed_out, "residual": residual_out, "k_tensor": k_tensor}


# ============================================================================
# R1 whole-model graph segment  (clone of Glm5WholeModelSegment, aux-KV dropped)
# ============================================================================


class R1WholeModelSegment:
    """Graph-capturable DeepSeek-R1 decode forward for global padded buckets.

    Clones Glm5WholeModelSegment but:
      (a) per layer runs R1AttnSegment + R1MoEGraphSegment;
      (b) NO aux-KV plumbing (single primary MLA KV);
      (c) layers < first_k_dense_replace run DenseMLP.forward eagerly (no MoE);
      (d) static inputs: input_ids, cache_seqlens, position_ids, page_table,
          slot_indices, rank_token_counts (NO aux_slot_indices);
      (e) lm_head in-graph behind ``capture_lm_head`` (False -> output
          hidden_states instead of logits);
      (f) run_model_with_probes eager-reference kept for the compare gate.
    """

    def __init__(
        self,
        *,
        model,
        device: torch.device,
        world_size: int,
        rank: int,
        max_pages_per_seq: int,
        vocab_size: int,
        hidden_size: int,
        max_bucket_size: int,
        max_seqlen: int,
        page_size_tokens: int,
        moe_pool: R1MoEGraphBufferPool,
        comm,
        capture_lm_head: bool = True,
        include_embedding: bool = True,
        compare_probe_layers: Iterable[int] | None = None,
    ) -> None:
        if not include_embedding:
            raise NotImplementedError(
                "R1 whole-model graph currently captures input_ids -> embedding"
            )
        if max_pages_per_seq <= 0:
            raise ValueError("max_pages_per_seq must be positive")
        if max_bucket_size <= 0:
            raise ValueError("max_bucket_size must be positive")
        if max_seqlen <= 0:
            raise ValueError("max_seqlen must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")

        self.model = model
        self.device = device
        self.world_size = int(world_size)
        self.rank = int(rank)
        self.max_pages_per_seq = int(max_pages_per_seq)
        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.max_bucket_size = int(max_bucket_size)
        self.max_seqlen = int(max_seqlen)
        self.page_size_tokens = int(page_size_tokens)
        self.moe_pool = moe_pool
        self.comm = comm
        self.capture_lm_head = bool(capture_lm_head)
        self.include_embedding = bool(include_embedding)

        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            raise ValueError("R1WholeModelSegment requires model.model.layers")
        self.num_layers = len(layers)
        if self.num_layers <= 0:
            raise ValueError("R1WholeModelSegment requires at least one decoder layer")
        self.first_k_dense_replace = int(model.config.first_k_dense_replace)

        probes = sorted({int(layer_idx) for layer_idx in (compare_probe_layers or [])})
        invalid_probes = [
            layer_idx for layer_idx in probes if layer_idx < 0 or layer_idx >= self.num_layers
        ]
        if invalid_probes:
            raise ValueError(
                f"R1 whole-model probe layers out of range: {invalid_probes}; "
                f"num_layers={self.num_layers}"
            )
        self.compare_probe_layers = tuple(probes)
        self._compare_probe_layer_set = set(self.compare_probe_layers)

        # Build per-layer segments: R1AttnSegment for every layer; R1MoEGraphSegment
        # only for MoE layers (>= first_k_dense_replace). Dense layers run
        # DenseMLP.forward eagerly inside the loop (no MoE segment).
        self.attn_segments: List[R1AttnSegment] = []
        self.moe_segments: List[Optional[R1MoEGraphSegment]] = []
        for layer_idx, layer in enumerate(layers):
            self.attn_segments.append(
                R1AttnSegment(
                    decoder_layer=layer,
                    attn_wrapper=layer.self_attn,
                    layer_idx=layer_idx,
                    max_seq_len=self.max_seqlen,
                    max_pages_per_seq=self.max_pages_per_seq,
                    page_size_tokens=self.page_size_tokens,
                )
            )
            if layer_idx >= self.first_k_dense_replace:
                self.moe_segments.append(
                    R1MoEGraphSegment(
                        layer.mlp,
                        self.moe_pool,
                        self.comm,
                        world_size=self.world_size,
                        rank=self.rank,
                        device=self.device,
                    )
                )
            else:
                self.moe_segments.append(None)

        # KV offload buffers: one [num_layers, max_bucket, 1, 1, kv_dim] bank
        # (single primary MLA KV; no aux). Filled in-graph via _copy_primary_kv,
        # staged by the worker after replay.
        self.primary_kv_dim = self.attn_segments[0].kv_dim
        self._kv_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self._kv_key_buffer: torch.Tensor | None = None
        self.primary_kv_offload_buffers: list[dict[str, torch.Tensor | None]] | None = None
        self._no_v_cache = True
        self._capture_inputs: dict[str, torch.Tensor] | None = None

    def set_capture_inputs(self, **inputs: torch.Tensor) -> None:
        required = set(self.get_static_input_specs(self.max_bucket_size))
        missing = required - set(inputs)
        if missing:
            raise ValueError(f"missing R1 whole-model capture inputs: {sorted(missing)}")
        unknown = set(inputs) - required
        if unknown:
            raise ValueError(f"unknown R1 whole-model capture inputs: {sorted(unknown)}")
        self._capture_inputs = dict(inputs)

    def initialize_static_inputs(
        self,
        static_inputs: Mapping[str, torch.Tensor],
        bucket_size: int,
    ) -> None:
        input_specs = self.get_static_input_specs(bucket_size)
        for name, target in static_inputs.items():
            spec = input_specs.get(name)
            if spec is not None:
                target.fill_(spec.fill_value)

        # Fill-value capture (generic K2.5/GPT-OSS warmup flow): when the worker
        # did not call set_capture_inputs, the static buffers keep their fill
        # values. R1's plain-MLA KV write uses page_table/slot_indices = 0 (page 0)
        # during warmup, the same as K2.5's per-layer attn capture.
        if self._capture_inputs is None:
            return

        for name, source in self._capture_inputs.items():
            target = static_inputs[name]
            if name == "rank_token_counts":
                if tuple(source.shape) != tuple(target.shape):
                    raise ValueError(
                        f"R1 whole-model capture input {name} shape {tuple(source.shape)} "
                        f"does not match static shape {tuple(target.shape)} for bucket {bucket_size}"
                    )
                target.copy_(source.to(device=target.device, dtype=target.dtype), non_blocking=True)
                continue
            if source.shape[0] > target.shape[0]:
                raise ValueError(
                    f"R1 whole-model capture input {name} batch dim {source.shape[0]} "
                    f"exceeds static shape {tuple(target.shape)} for bucket {bucket_size}"
                )
            if source.shape[1:] != target.shape[1:]:
                raise ValueError(
                    f"R1 whole-model capture input {name} trailing shape "
                    f"{tuple(source.shape[1:])} does not match static shape "
                    f"{tuple(target.shape[1:])} for bucket {bucket_size}"
                )
            if source.shape[0] > 0:
                target[: source.shape[0]].copy_(
                    source.to(device=target.device, dtype=target.dtype),
                    non_blocking=True,
                )

    def setup_static_buffers(self, bucket_size: int) -> None:
        if bucket_size > self.max_bucket_size:
            raise ValueError(
                f"bucket_size {bucket_size} exceeds max_bucket_size {self.max_bucket_size}"
            )
        if self._kv_buffers is not None:
            return

        alloc_size = self.max_bucket_size
        self._kv_key_buffer = torch.zeros(
            self.num_layers,
            alloc_size,
            1,
            1,
            self.primary_kv_dim,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._kv_buffers = [
            {"key": self._kv_key_buffer[layer_idx], "value": None}
            for layer_idx in range(self.num_layers)
        ]
        self.primary_kv_offload_buffers = self._kv_buffers
        # Propagate setup to the layer MoE segments (allocates the shared pool).
        for moe_segment in self.moe_segments:
            if moe_segment is not None:
                moe_segment.setup_static_buffers(bucket_size)

    def release_static_buffers(self, bucket_size: int) -> None:
        for moe_segment in self.moe_segments:
            if moe_segment is not None:
                moe_segment.release_static_buffers(bucket_size)
        self._kv_buffers = None
        self._kv_key_buffer = None
        self.primary_kv_offload_buffers = None
        self._capture_inputs = None

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "input_ids": TensorSpec(("batch_size", 1), torch.int64, fill_value=0),
            "cache_seqlens": TensorSpec(
                ("batch_size",), torch.int32, fill_value=1
            ),
            "position_ids": TensorSpec(("batch_size", 1), torch.int64, fill_value=0),
            "page_table": TensorSpec(
                ("batch_size", self.max_pages_per_seq), torch.int32, fill_value=0
            ),
            "slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=0),
            "rank_token_counts": TensorSpec((self.world_size,), torch.int64, fill_value=0),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        specs = {
            "hidden_states": TensorSpec(("batch_size", self.hidden_size), torch.bfloat16),
        }
        if self.capture_lm_head:
            specs["logits"] = TensorSpec(("batch_size", self.vocab_size), torch.bfloat16)
        for layer_idx in self.compare_probe_layers:
            specs[self._probe_output_name(layer_idx)] = TensorSpec(
                ("batch_size", self.hidden_size), torch.bfloat16
            )
        return specs

    def _copy_primary_kv(self, layer_idx: int, k_tensor: torch.Tensor, _v_tensor=None) -> None:
        if self._kv_buffers is None:
            raise RuntimeError("R1 whole-model graph KV buffers are not initialized")
        k_tensor = self._normalize_k_tensor(k_tensor, self.primary_kv_dim)
        self._kv_buffers[int(layer_idx)]["key"][: k_tensor.shape[0]].copy_(k_tensor)

    @staticmethod
    def _normalize_k_tensor(k_tensor: torch.Tensor, expected_dim: int) -> torch.Tensor:
        if k_tensor.dim() == 3:
            k_tensor = k_tensor.unsqueeze(2)
        if k_tensor.dim() != 4:
            raise RuntimeError(
                f"R1 whole-model graph expected 3-D/4-D KV tensor, got {tuple(k_tensor.shape)}"
            )
        if k_tensor.shape[-1] != expected_dim:
            raise RuntimeError(
                f"R1 whole-model graph KV dim mismatch: got {k_tensor.shape[-1]}, "
                f"expected {expected_dim}"
            )
        return k_tensor

    def _set_moe_bucket_state(self, bucket_size: int, rank_token_counts: torch.Tensor) -> None:
        from batchgen.models.deepseek.deepseekv3.model import DeepseekV3MoE

        DeepseekV3MoE._rank_token_counts = rank_token_counts
        for layer in self.model.model.layers:
            mlp = getattr(layer, "mlp", None)
            if isinstance(mlp, DeepseekV3MoE):
                mlp.set_num_tokens_per_rank(int(bucket_size))

    @staticmethod
    def _probe_output_name(layer_idx: int) -> str:
        return f"probe_layer_{int(layer_idx):03d}_hidden"

    def run_model_with_probes(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        slot_indices: torch.Tensor,
        rank_token_counts: torch.Tensor,
        use_layer_segments: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        bucket_size = int(input_ids.shape[0])
        hidden_states = self.model.model.embed_tokens(input_ids)  # [B, 1, H]
        outputs: dict[str, torch.Tensor] = {}

        if use_layer_segments is not False:
            for layer_idx, layer in enumerate(self.model.model.layers):
                attn_out = self.attn_segments[layer_idx].forward(
                    hidden_states=hidden_states,
                    cache_seqlens=cache_seqlens,
                    page_table=page_table,
                    slot_indices=slot_indices,
                )
                normed = attn_out["normed"]        # [B, 1, H] (post-attn norm)
                residual = attn_out["residual"]    # [B, 1, H]
                self._copy_primary_kv(layer_idx, attn_out["k_tensor"], None)

                moe_segment = self.moe_segments[layer_idx]
                if moe_segment is not None:
                    # MoE layer: segment expects/returns [B, H] (no q_len dim).
                    moe_out = moe_segment.forward(
                        padded=normed.view(bucket_size, self.hidden_size),
                        rank_token_counts=rank_token_counts,
                    )["moe_output"].view(bucket_size, 1, self.hidden_size)
                else:
                    # Dense layer (< first_k_dense_replace): DenseMLP.forward eager.
                    moe_out = layer.mlp(normed)
                hidden_states = residual + moe_out

                if layer_idx in self._compare_probe_layer_set:
                    outputs[self._probe_output_name(layer_idx)] = hidden_states[:, -1, :]
        else:
            # Eager reference path (no layer segments): plain decoder layers.
            for layer_idx, layer in enumerate(self.model.model.layers):
                layer_output = layer(hidden_states, layer_idx=layer_idx)
                hidden_states = (
                    layer_output[0] if isinstance(layer_output, tuple) else layer_output
                )
                if layer_idx in self._compare_probe_layer_set:
                    outputs[self._probe_output_name(layer_idx)] = hidden_states[:, -1, :]

        hidden_states = self.model.model.norm(hidden_states)
        last_hidden = hidden_states[:, -1, :]
        outputs["hidden_states"] = last_hidden
        if self.capture_lm_head:
            logits = self.model.lm_head(hidden_states)
            outputs["logits"] = logits[:, -1, :]
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        position_ids: torch.Tensor,
        page_table: torch.Tensor,
        slot_indices: torch.Tensor,
        rank_token_counts: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        bucket_size = int(input_ids.shape[0])
        self._set_moe_bucket_state(bucket_size, rank_token_counts)

        old_cache_seqlens = AttnWrapperBase.cache_seqlens
        old_position_ids = AttnWrapperBase.position_ids
        old_max_seqlen = AttnWrapperBase.max_seqlen
        old_kv_cb = AttnWrapperBase.kv_append_callback
        try:
            AttnWrapperBase.cache_seqlens = cache_seqlens
            AttnWrapperBase.position_ids = position_ids
            AttnWrapperBase.max_seqlen = self.max_seqlen
            AttnWrapperBase.kv_append_callback = self._copy_primary_kv
            outputs = self.run_model_with_probes(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_seqlens=cache_seqlens,
                page_table=page_table,
                slot_indices=slot_indices,
                rank_token_counts=rank_token_counts,
            )
        finally:
            AttnWrapperBase.cache_seqlens = old_cache_seqlens
            AttnWrapperBase.position_ids = old_position_ids
            AttnWrapperBase.max_seqlen = old_max_seqlen
            AttnWrapperBase.kv_append_callback = old_kv_cb

        return outputs


def compare_r1_whole_model_graph_logits(
    *,
    eager_logits: torch.Tensor,
    graph_logits: torch.Tensor,
    eager_hidden_states: torch.Tensor | None = None,
    graph_hidden_states: torch.Tensor | None = None,
    eager_probe_hidden_states: Mapping[str, torch.Tensor] | None = None,
    graph_probe_hidden_states: Mapping[str, torch.Tensor] | None = None,
    eager_tokens: torch.Tensor | None = None,
    graph_tokens: torch.Tensor | None = None,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict[str, object]:
    """Compare eager and graph logits without changing decode control flow.

    Faithful clone of compare_glm5_whole_model_graph_logits.
    """
    if eager_logits.shape != graph_logits.shape:
        return {
            "ok": False,
            "shape_match": False,
            "eager_shape": tuple(int(dim) for dim in eager_logits.shape),
            "graph_shape": tuple(int(dim) for dim in graph_logits.shape),
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "argmax_mismatch": -1,
            "token_mismatch": -1,
        }

    eager_f = eager_logits.detach().to(torch.float32)
    graph_f = graph_logits.detach().to(torch.float32)
    diff = (eager_f - graph_f).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    logits_ok = bool(torch.allclose(eager_f, graph_f, atol=atol, rtol=rtol))
    argmax_mismatch = int(
        (torch.argmax(eager_f, dim=-1) != torch.argmax(graph_f, dim=-1)).sum().item()
    )

    token_mismatch = 0
    token_shape_match = True
    if eager_tokens is not None and graph_tokens is not None:
        token_shape_match = eager_tokens.shape == graph_tokens.shape
        if token_shape_match:
            token_mismatch = int((eager_tokens != graph_tokens).sum().item())
        else:
            token_mismatch = -1

    hidden_ok = True
    hidden_shape_match = True
    hidden_max_abs = 0.0
    hidden_mean_abs = 0.0
    if eager_hidden_states is not None or graph_hidden_states is not None:
        if eager_hidden_states is None or graph_hidden_states is None:
            hidden_ok = False
            hidden_shape_match = False
            hidden_max_abs = float("inf")
            hidden_mean_abs = float("inf")
        else:
            hidden_shape_match = eager_hidden_states.shape == graph_hidden_states.shape
            if hidden_shape_match:
                eager_h = eager_hidden_states.detach().to(torch.float32)
                graph_h = graph_hidden_states.detach().to(torch.float32)
                hidden_diff = (eager_h - graph_h).abs()
                hidden_max_abs = float(hidden_diff.max().item()) if hidden_diff.numel() else 0.0
                hidden_mean_abs = float(hidden_diff.mean().item()) if hidden_diff.numel() else 0.0
                hidden_ok = bool(torch.allclose(eager_h, graph_h, atol=atol, rtol=rtol))
            else:
                hidden_ok = False
                hidden_max_abs = float("inf")
                hidden_mean_abs = float("inf")

    probe_first_mismatch = ""
    probe_max_abs = 0.0
    probe_mean_abs = 0.0
    probe_shape_match = True
    probe_ok = True
    if eager_probe_hidden_states is not None or graph_probe_hidden_states is not None:
        eager_probe_hidden_states = eager_probe_hidden_states or {}
        graph_probe_hidden_states = graph_probe_hidden_states or {}
        if set(eager_probe_hidden_states) != set(graph_probe_hidden_states):
            probe_ok = False
            probe_shape_match = False
            probe_first_mismatch = "probe_key_set"
            probe_max_abs = float("inf")
            probe_mean_abs = float("inf")
        else:
            for name in sorted(eager_probe_hidden_states):
                eager_probe = eager_probe_hidden_states[name]
                graph_probe = graph_probe_hidden_states[name]
                if eager_probe.shape != graph_probe.shape:
                    probe_ok = False
                    probe_shape_match = False
                    probe_first_mismatch = name
                    probe_max_abs = float("inf")
                    probe_mean_abs = float("inf")
                    break
                eager_p = eager_probe.detach().to(torch.float32)
                graph_p = graph_probe.detach().to(torch.float32)
                probe_diff = (eager_p - graph_p).abs()
                cur_max = float(probe_diff.max().item()) if probe_diff.numel() else 0.0
                cur_mean = float(probe_diff.mean().item()) if probe_diff.numel() else 0.0
                probe_max_abs = max(probe_max_abs, cur_max)
                probe_mean_abs = max(probe_mean_abs, cur_mean)
                cur_ok = bool(torch.allclose(eager_p, graph_p, atol=atol, rtol=rtol))
                if not cur_ok and not probe_first_mismatch:
                    probe_first_mismatch = name
                    probe_ok = False

    return {
        "ok": bool(
            logits_ok
            and hidden_ok
            and probe_ok
            and argmax_mismatch == 0
            and token_shape_match
            and token_mismatch == 0
        ),
        "shape_match": True,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "hidden_shape_match": hidden_shape_match,
        "hidden_max_abs": hidden_max_abs,
        "hidden_mean_abs": hidden_mean_abs,
        "probe_shape_match": probe_shape_match,
        "probe_first_mismatch": probe_first_mismatch,
        "probe_max_abs": probe_max_abs,
        "probe_mean_abs": probe_mean_abs,
        "argmax_mismatch": argmax_mismatch,
        "token_mismatch": token_mismatch,
        "token_shape_match": token_shape_match,
    }
