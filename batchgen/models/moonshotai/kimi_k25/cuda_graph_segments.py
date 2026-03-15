"""CUDA Graph capturable segments for Kimi K2.5 decode.

Segments:
  K25WholeModelSegment: embedding → 61 decoder layers (MLA + MoE) → final norm → lm_head

K2.5 specifics vs GPT-OSS:
  - MLA attention (q_a→norm→q_b→absorb→FlashMLA→out_absorb→o_proj) instead of GQA
  - 3D strided MoE buffers (dispatch_scatter_3d) via KimiK25MoEBufferManager
  - Layer 0: dense MLP, layers 1-60: MoE (384 routed + 1 shared expert)
  - Shared expert runs on same stream during graph capture (no async overlap)
  - fused_rmsnorm_rope_with_q for combined KV norm+RoPE+cache update

Dynamic metadata (cache_seqlens, page_table) are static-address input buffers
whose contents are updated before each replay. KV cache is at fixed GPU addresses.
NCCL collectives use PyNccl (ctypes) for graph compatibility.
"""

import logging
from typing import Dict, Optional

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.models.wrappers.attention import AttnWrapperBase

logger = logging.getLogger(__name__)


class K25WholeModelSegment:
    """Captures the entire K2.5 decode forward pass in a single CUDA graph.

    Embedding → 61 decoder layers (MLA attention + MoE/dense) → final RMSNorm → lm_head.

    All layers run eagerly within the graph (no per-layer graph replay).
    NCCL collectives (all_gather + all_reduce per MoE layer) are captured.
    KV cache writes happen at fixed GPU addresses via gpu_paged_kv_manager.

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

        # KV dimensions for offload buffers
        attn0 = layers[0].self_attn
        if hasattr(attn0, 'module'):
            attn_mod = attn0.module
        else:
            attn_mod = attn0
        self.kv_lora_rank = attn_mod.kv_lora_rank
        self.qk_rope_head_dim = attn_mod.qk_rope_head_dim
        self.kv_dim = self.kv_lora_rank + self.qk_rope_head_dim  # 576
        self.max_seq_len = model.model.config.max_position_embeddings

        # Per-layer KV buffers for host offloading (allocated in setup)
        self._kv_buffers = None

    def setup_static_buffers(self, bucket_size: int) -> None:
        """Allocate KV offload buffers and set capture mode flags.

        Called once per bucket size during graph capture.
        """
        # Allocate KV buffers once at max_bucket_size
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
                    # Empty tensor for compatibility with decode loop KV callback.
                    "value": torch.zeros(
                        alloc_size, 1, 1, self.kv_dim,
                        dtype=torch.bfloat16, device=self.device,
                    ),
                })

        # Set capture mode flag so layers know graph is being captured
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

        All ops run on the current CUDA stream (graph-captured).
        """
        B = input_ids.shape[0]
        model = self.model.model  # KimiK25Model

        # === Embedding ===
        hidden_states = model.embed_tokens(input_ids)  # [B, 1, H]

        # === Pre-compute position IDs from cache_seqlens ===
        # position_ids = cache_seqlens - 1 (0-indexed position of the new decode token)
        q_position_ids = (cache_seqlens - 1).clamp(min=0).unsqueeze(1).to(torch.int64)  # [B, 1]

        gpu_kv_manager = AttnWrapperBase.gpu_paged_kv_manager

        # === Decoder layers ===
        for layer_idx, decoder_layer in enumerate(model.layers):
            attn_wrapper = decoder_layer.self_attn
            if hasattr(attn_wrapper, 'module'):
                attn_mod = attn_wrapper.module
            else:
                attn_mod = attn_wrapper

            # =====================================================
            # MLA Attention (inline for graph compatibility)
            # =====================================================

            # Pre-attn RMSNorm
            residual = hidden_states
            normed = decoder_layer.input_layernorm(hidden_states)

            # Call the bound decode function (either optimized or original)
            # This function handles: q projections, KV norm+RoPE, cache update,
            # FlashMLA attention, absorb einsums, o_proj
            attn_output, k_tensor = attn_mod.decoding_attn_mode_3_bf16(
                normed,
                q_position_ids,
                cache_seqlens,
                self.max_seq_len,  # Fixed max for RoPE cos/sin cache coverage
                None,  # weight_scale (BF16, not needed)
                gpu_kv_manager,
                layer_idx,
                None,  # batch_slice
            )

            # Copy KV to static offload buffer (baked into graph for host offloading)
            if self._kv_buffers is not None and k_tensor is not None:
                self._kv_buffers[layer_idx]["key"][:B].copy_(k_tensor[:B])

            # Post-attn: residual add + RMSNorm (fused when available)
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
            if hasattr(moe, '_forward_decode_graph'):
                # Graph-compatible MoE forward (no separate stream for shared expert)
                mlp_out = moe._forward_decode_graph(normed)
            elif hasattr(moe, '_forward_decode') and hasattr(moe, 'comm') and moe.comm is not None:
                # MoE layer — use graph-compatible path
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
            from batchgen.moe.fused_int4_wgmma_grouped import _load_int4_grouped_module
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
