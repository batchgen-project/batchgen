"""CUDA Graph capturable segments for Kimi K2.5 decode.

Segments:
  K25WholeModelSegment: embedding → 61 decoder layers (MLA + MoE) → final norm → lm_head

K2.5 specifics vs GPT-OSS:
  - MLA attention (q_a→norm→q_b→absorb→FlashMLA→out_absorb→o_proj) instead of GQA
  - 3D strided MoE buffers (dispatch_scatter_3d) via KimiK25MoEBufferManager
  - Layer 0: dense MLP, layers 1-60: MoE (384 routed + 1 shared expert)
  - Shared expert runs on same stream during graph capture (no async overlap)
  - fused_rmsnorm_rope_with_q for combined KV norm+RoPE+cache update

MLA forward is INLINED (not delegated to decoding_attn_mode_3_bf16) because:
  - CUDA graph requires static tensor addresses — the gpu_paged_kv_manager's internal
    block_table may be reallocated. We use the static page_table input instead.
  - Same approach as GPT-OSS WholeModelSegment (see cuda_graph_segments.py lines 880-900).
  - Zero overhead: same kernels, same number of launches, just different page_table pointer.

Dynamic metadata (cache_seqlens, page_table) are static-address input buffers
whose contents are updated before each replay. KV cache is at fixed GPU addresses.
NCCL collectives use PyNccl (ctypes) for graph compatibility.
"""

import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.models.wrappers.attention import AttnWrapperBase

logger = logging.getLogger(__name__)


class K25WholeModelSegment:
    """Captures the entire K2.5 decode forward pass in a single CUDA graph.

    Embedding → 61 decoder layers (MLA attention + MoE/dense) → final RMSNorm → lm_head.

    All layers run eagerly within the graph (no per-layer graph replay).
    NCCL collectives (all_gather + all_reduce per MoE layer) are captured.
    KV cache writes use static page_table/slot_indices inputs (not gpu_manager's state).

    Inputs:  input_ids [B, 1] int64, cache_seqlens [B] int32,
             page_table [B, max_pages] int32, slot_indices [B] int32
    Outputs: logits [B, vocab_size] bfloat16
    """

    def __init__(
        self,
        model,  # KimiK25ForCausalLM instance
        device: torch.device,
        max_pages_per_seq: int,
        vocab_size: int,
        hidden_size: int,
        max_bucket_size: int = 16,
    ):
        self.model = model
        self.device = device
        self.max_pages_per_seq = max_pages_per_seq
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_bucket_size = max_bucket_size

        layers = model.model.layers
        self.num_layers = len(layers)

        # MLA dimensions (from first attention layer)
        attn0 = layers[0].self_attn
        attn_mod = attn0.module if hasattr(attn0, 'module') else attn0
        self.kv_lora_rank = attn_mod.kv_lora_rank        # 512
        self.qk_rope_head_dim = attn_mod.qk_rope_head_dim  # 64
        self.qk_nope_head_dim = attn_mod.qk_nope_head_dim  # 128
        self.v_head_dim = attn_mod.v_head_dim              # 128
        self.num_heads = attn_mod.num_heads                # 64
        self.q_head_dim = attn_mod.q_head_dim              # 192
        self.q_lora_rank = attn_mod.q_lora_rank            # 1536
        self.kv_dim = self.kv_lora_rank + self.qk_rope_head_dim  # 576
        self.max_seq_len = model.model.config.max_position_embeddings
        self.softmax_scale = attn_mod.softmax_scale

        # Per-layer KV buffers for host offloading (allocated in setup)
        self._kv_buffers = None
        # K2.5 MLA has no separate V cache (compressed KV only)
        self._no_v_cache = True

    def setup_static_buffers(self, bucket_size: int) -> None:
        """Allocate KV offload buffers and set capture mode flags."""
        if self._kv_buffers is None:
            alloc_size = self.max_bucket_size
            self._kv_buffers = []
            for _ in range(self.num_layers):
                self._kv_buffers.append({
                    "key": torch.zeros(
                        alloc_size, 1, 1, self.kv_dim,
                        dtype=torch.bfloat16, device=self.device,
                    ),
                    # K2.5 MLA has no separate V cache (compressed KV only).
                    # Placeholder for decode loop KV callback compatibility.
                    "value": torch.zeros(
                        alloc_size, 1, 1, self.kv_dim,
                        dtype=torch.bfloat16, device=self.device,
                    ),
                })

        # Set capture mode flag
        for layer in self.model.model.layers:
            layer._graph_capture_mode = True

        # Enable NCCL communicators on MoE layers
        for layer in self.model.model.layers:
            moe = layer.mlp
            if hasattr(moe, 'comm') and moe.comm is not None:
                moe.comm.disabled = False

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "input_ids": TensorSpec(("batch_size", 1), torch.int64, fill_value=0),
            "cache_seqlens": TensorSpec(("batch_size",), torch.int32, fill_value=0),
            "page_table": TensorSpec(
                ("batch_size", self.max_pages_per_seq), torch.int32, fill_value=0
            ),
            "slot_indices": TensorSpec(("batch_size",), torch.int32, fill_value=-1),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "logits": TensorSpec(("batch_size", self.vocab_size), torch.bfloat16),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Full decode forward: embed → layers → norm → lm_head.

        MLA attention is INLINED to use static page_table/slot_indices inputs.
        All ops run on the current CUDA stream (graph-captured).
        """
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata
        from batchgen_kernels.triton.fused_rmsnorm_rope import fused_rmsnorm_rope_with_q
        from batchgen.kv_cache.gpu_kv_kernels import run_paged_kv_token_update_fused

        B = input_ids.shape[0]
        model = self.model.model  # KimiK25Model
        gpu_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        page_size_tokens = gpu_kv_manager.config.page_size_tokens

        # === Embedding ===
        hidden_states = model.embed_tokens(input_ids)  # [B, 1, H]

        # === Position IDs (shared across all layers) ===
        q_position_ids = (cache_seqlens - 1).clamp(min=0).unsqueeze(1).to(torch.int64)  # [B, 1]

        # === Decoder layers ===
        for layer_idx, decoder_layer in enumerate(model.layers):
            attn_wrapper = decoder_layer.self_attn
            attn_mod = attn_wrapper.module if hasattr(attn_wrapper, 'module') else attn_wrapper

            # =====================================================
            # MLA Attention (INLINED — uses static page_table)
            # =====================================================

            # Pre-attn RMSNorm
            residual = hidden_states
            normed = decoder_layer.input_layernorm(hidden_states)

            # Q projections: hidden → q_a → norm → q_b
            normed_sq = normed.squeeze(1)  # [B, H]
            q = F.linear(normed_sq, attn_mod.q_a_proj.weight)
            new_compressed_kv = F.linear(normed_sq, attn_mod.kv_a_proj_with_mqa.weight).view(B, 1, -1)
            q = attn_mod.q_a_layernorm(q)
            q = F.linear(q, attn_mod.q_b_proj.weight)

            # Q reshape + split into q_nope [B,H,128] and q_pe [B,H,1,64]
            q = q.view(B, 1, self.num_heads, self.q_head_dim).transpose(1, 2)
            q_nope, q_pe = torch.split(
                q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            q_pe = q_pe.contiguous()

            # RoPE cos/sin
            cos, sin = attn_mod.rotary_emb(q_pe, seq_len=self.max_seq_len)

            # Fused KV norm + RoPE + Q RoPE (existing production kernel)
            offload_kv = fused_rmsnorm_rope_with_q(
                new_compressed_kv,
                q_pe,
                cos,
                sin,
                q_position_ids,
                attn_mod.kv_a_layernorm.weight,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )

            # KV tensor for host offloading callback
            k_tensor = offload_kv.view(B, 1, 1, offload_kv.size(-1))

            # Copy KV to static offload buffer (baked into graph)
            if self._kv_buffers is not None:
                self._kv_buffers[layer_idx]["key"][:B].copy_(k_tensor[:B])

            # KV write to GPU paged cache — use STATIC page_table + slot_indices
            # (NOT gpu_kv_manager.update_layer_decode_new_token which reads manager state)
            blocked_k, _, _ = gpu_kv_manager.get_layer_kv_with_page_table(layer_idx)
            run_paged_kv_token_update_fused(
                k_cache=blocked_k,
                k_tokens=k_tensor.view(B, -1),
                page_table=page_table,      # static input
                slot_indices=slot_indices,   # static input
                token_indices=q_position_ids.squeeze(-1).to(torch.int32),
                page_size_tokens=page_size_tokens,
            )

            # Q absorb: q_nope × kv_b_proj → query_states
            kv_b_proj = attn_mod.kv_b_proj.weight.data.view(
                self.num_heads, -1, self.kv_lora_rank
            )
            q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :]
            out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :]

            qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
            query_states = torch.empty(
                B, self.num_heads, 1, qk_head_dim,
                dtype=blocked_k.dtype, device=self.device,
            )
            q_nope_sq = q_nope.squeeze(2)
            query_states[:, :, :, :self.kv_lora_rank] = torch.einsum(
                "bhd,hdc->bhc", q_nope_sq, q_absorb
            ).view(B, self.num_heads, 1, self.kv_lora_rank)
            query_states[:, :, :, self.kv_lora_rank:] = q_pe
            query_states = query_states.view(B, 1, self.num_heads, qk_head_dim)

            # FlashMLA attention — use STATIC page_table as block_table
            tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, 128, 1)

            attn_out, _ = flash_mla_with_kvcache(
                query_states,
                blocked_k,
                page_table,             # static input (NOT manager's block_table)
                cache_seqlens,
                self.kv_lora_rank,      # 512 = head_dim_v
                tile_scheduler_metadata,
                num_splits,
                self.softmax_scale,
                True,                   # causal
            )

            # Output absorb + o_proj
            attn_output = torch.einsum('bqhc,hdc->bhqd', attn_out, out_absorb)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(B, self.num_heads * self.v_head_dim)
            attn_output = F.linear(attn_output, attn_mod.o_proj.weight)
            attn_output = attn_output.view(B, 1, -1)

            # =====================================================
            # Post-attn: residual add + RMSNorm
            # =====================================================
            fused_add_norm = decoder_layer._get_fused_add_rmsnorm_fn()
            if fused_add_norm is not None:
                normed, residual = fused_add_norm(
                    residual, attn_output,
                    decoder_layer.post_attention_layernorm.weight,
                    decoder_layer.post_attention_layernorm.variance_epsilon,
                )
            else:
                hidden_states = residual + attn_output
                residual = hidden_states
                normed = decoder_layer.post_attention_layernorm(hidden_states)

            # =====================================================
            # MoE / Dense MLP
            # =====================================================
            moe = decoder_layer.mlp
            if hasattr(moe, 'comm') and moe.comm is not None:
                # MoE layer — graph-compatible path (no separate stream)
                mlp_out = self._moe_forward_graph(moe, normed)
            else:
                # Dense MLP (layer 0) or fallback
                mlp_out = moe(normed)

            # residual += mlp_out
            hidden_states = residual + mlp_out

        # === Final norm + lm_head ===
        hidden_states = model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)  # [B, 1, vocab]
        logits = logits.squeeze(1)  # [B, vocab]

        return {"logits": logits}

    def _moe_forward_graph(
        self,
        moe,  # KimiK25MoE instance
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Graph-compatible MoE forward without separate stream for shared expert.

        Differences from KimiK25MoE._forward_decode:
        - Shared expert runs on SAME stream (required for CUDA graph capture)
        - No dynamic buffer resize (pre-sized to max bucket)
        - Uses same comm, dispatch, WGMMA, reduce path
        """
        import torch.distributed as dist
        from batchgen.moe.dispatch_scatter_3d import (
            dispatch_scatter_3d,
            reduce_weighted_scatter,
        )

        buf = moe.__class__._buf
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        identity = hidden_states
        num_tokens, H = hidden_states.shape
        device = moe.device or hidden_states.device
        topk = moe.top_k
        num_global = moe.world_size * moe.num_tokens_per_rank

        # 1) AllGather into reserved buffer
        all_tokens = buf.all_tokens[:num_global]
        padded = buf.padded
        padded.zero_()
        if num_tokens > 0:
            padded[:num_tokens] = hidden_states

        with moe.comm.change_state(enable=True):
            moe.comm.all_gather(
                all_tokens, padded,
                stream=torch.cuda.current_stream(device),
            )

        # 2) Shared expert on SAME stream (graph-compatible, no async overlap)
        shared_out = moe.shared_experts(identity)

        # 3) CUDA gate
        topk_idx, topk_weight = moe.gate(all_tokens.view(num_global, 1, H))

        # 4) 3D dispatch scatter
        expert_counts, topk_pos = dispatch_scatter_3d(
            all_tokens, topk_idx.to(torch.int32),
            buf.dispatched_x,
            moe.routed_expert_start_idx, moe.experts_per_rank,
            buf.max_tokens_padded,
            buf.expert_counts, buf.expert_counters,
            buf.topk_pos[:num_global * topk],
        )

        # 5) Expert compute: grouped INT4 WGMMA inplace
        if getattr(moe, '_use_grouped_wgmma', False) \
                and buf.tma_dispatched is not None:
            mod = moe._wgmma_mod
            w = moe._moe_weights
            mtp = buf.max_tokens_padded
            _BLOCK_M = 64
            avg_per_expert = (num_global * topk + buf.E_local - 1) // buf.E_local
            max_m_tiles = (min(avg_per_expert * 2, mtp) + _BLOCK_M - 1) // _BLOCK_M
            max_m_tiles = max(max_m_tiles, 1)
            N = moe.moe_intermediate_size
            K = H

            mod.grouped_int4_moe_stage1_inplace(
                buf.dispatched_x, buf.intermediate, buf.tma_dispatched,
                expert_counts,
                w["_ptr_gate"], w["_ptr_gate_scale"],
                w["_ptr_up"], w["_ptr_up_scale"],
                buf.empty_bias, buf.empty_bias,
                N, K // 2, K // 32, max_m_tiles, mtp,
            )

            mod.grouped_int4_moe_stage2_inplace(
                buf.intermediate, buf.expert_out, buf.tma_intermediate,
                expert_counts,
                w["_ptr_down"], w["_ptr_down_scale"],
                buf.empty_bias,
                K, N // 2, N // 32, max_m_tiles, mtp,
            )

        # 6) Reduce
        result_buf = buf.result_buffer[:num_global]
        global_results = reduce_weighted_scatter(
            buf.expert_out, topk_pos, topk_weight,
            num_global, H, topk, output=result_buf,
        )

        # 7) AllReduce
        with moe.comm.change_state(enable=True):
            moe.comm.all_reduce(
                global_results, op=dist.ReduceOp.SUM,
                stream=torch.cuda.current_stream(device),
            )

        # 8) Extract local slice + add shared expert
        start = moe.rank * moe.num_tokens_per_rank
        end = start + num_tokens
        out = global_results[start:end] + shared_out
        return out.view(*orig_shape)
