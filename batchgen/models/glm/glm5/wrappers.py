# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""GLM-5 specific wrappers for BatchGen execution.

Standalone wrappers for GLM-5 FP8 model — no cross-model imports.
- GLM5ExpertWrapper: Expert wrapper with FP8 dequantization
- GLM5AttnWrapper: Attention wrapper with FP8 dequant + DSA integration

Key differences from DeepSeek:
- kv_a_proj_with_mqa naming (same)
- DSA indexer integration in attention wrapper
- Different MLA dimensions (qk_nope=192, v_head=256)
- rope_interleave=True
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase


def glm5_fp8_dequantization(
    weight_data_fp8: torch.Tensor,
    weight_scale_inv_fp32: torch.Tensor,
    block_size=(128, 128),
) -> torch.Tensor:
    """Blockwise FP8 dequantization (standalone, no cross-model import)."""
    rows, cols = weight_data_fp8.shape
    block_rows, block_cols = block_size
    n_block_rows = rows // block_rows
    n_block_cols = cols // block_cols
    weight_4d = weight_data_fp8.reshape(
        n_block_rows, block_rows, n_block_cols, block_cols
    ).to(torch.float32)
    scale_4d = weight_scale_inv_fp32.unsqueeze(1).unsqueeze(-1)
    dequantized_4d = weight_4d * scale_4d
    return dequantized_4d.reshape(rows, cols).to(torch.bfloat16)


class GLM5ExpertWrapper(ExpertWrapperBase):
    """Expert wrapper for GLM-5 (BF16 routed experts).

    GLM-5 routed experts are BF16, not FP8. Uses standard nn.Linear forward.
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        expert_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = False,
        weight_dequant_scale: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            persistent
        )
        self.weight_dequant_scale = weight_dequant_scale or {}
        self.cached_gate = None
        self.cached_up = None
        self.cached_down = None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """BF16 experts need no dequantization — pass through."""
        return weights_dict

    def _register_fp8_weights(self):
        self.cached_gate = self.module.gate_proj.weight.data
        self.cached_up = self.module.up_proj.weight.data
        self.cached_down = self.module.down_proj.weight.data

    def _unregister_fp8_weights(self):
        self.cached_gate = None
        self.cached_up = None
        self.cached_down = None

    def _forward_impl(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.module(hidden_states)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.persistent:
            weights = self.load_weights(self.module_key)
            for name, param in self.module.named_parameters():
                param.data = weights[name]
        else:
            self.module.gate_proj.weight.data = self.cached_gate
            self.module.up_proj.weight.data = self.cached_up
            self.module.down_proj.weight.data = self.cached_down

        result = self.micro_batch_forward(hidden_states, "expert")

        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        return result


class GLM5AttnWrapper(AttnWrapperBase):
    """Attention wrapper with FP8 dequant + DSA for GLM-5.

    Key differences from DeepSeek:
    - kv_a_proj_with_mqa (same naming)
    - DSA indexer integration: prefill populates auxiliary cache,
      decode uses indexer scoring for sparse attention
    - MLA dims: qk_nope=192, v_head=256, q_lora_rank=2048
    """

    def __init__(
        self,
        module: nn.Module,
        layer_idx: int,
        core_engine,
        engine_config,
        model_config,
        persistent: bool = True,
        weight_dequant_scale: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent, weight_dequant_scale
        )
        # FP8 weight caching for GLM-5 MLA
        self.fp8_q_a_proj = None
        self.fp8_q_b_proj = None
        self.fp8_kv_a_proj = None
        self.fp8_kv_b_proj = None
        self.fp8_o_proj = None

    def _register_fp8_weights(self):
        """Cache FP8 attention weights. GLM-5 uses kv_a_proj_with_mqa."""
        self.fp8_q_a_proj = self.module.q_a_proj.weight.data
        self.fp8_q_b_proj = self.module.q_b_proj.weight.data
        self.fp8_kv_a_proj = self.module.kv_a_proj_with_mqa.weight.data
        self.fp8_kv_b_proj = self.module.kv_b_proj.weight.data
        self.fp8_o_proj = self.module.o_proj.weight.data

    def _unregister_fp8_weights(self):
        self.fp8_q_a_proj = None
        self.fp8_q_b_proj = None
        self.fp8_kv_a_proj = None
        self.fp8_kv_b_proj = None
        self.fp8_o_proj = None

    def dequantize_weights(
        self, weights_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return FP8 weights unchanged — deepgemm handles FP8 directly."""
        return weights_dict

    def _forward_prefill(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Prefill forward with DSA auxiliary cache population.

        1. Standard MLA prefill via FA3 (full attention)
        2. Compute indexer K and write to auxiliary cache
        """
        if self.prepack_mode:
            hidden_states_2d = hidden_states.squeeze(0)
            attn_output, offload_kv = self.module.prefill_attn_w8a16_prepacked(
                hidden_states_2d,
                self.position_ids.to(hidden_states_2d.device),
                self.prepack_cu_seqlens.to(hidden_states_2d.device),
                self.prepack_max_seqlen,
                self.prepack_num_sequences,
                self.weight_dequant_scale
            )

            # DSA: compute indexer K and offload to auxiliary cache
            gpu_paged_kv_manager_aux = AttnWrapperBase.gpu_paged_kv_manager_aux
            if gpu_paged_kv_manager_aux is not None and hasattr(self.module, 'indexer'):
                indexer_kv = self.module.indexer.compute_indexer_kv(
                    hidden_states_2d.unsqueeze(0)
                )
                # indexer_kv: [1, total_tokens, 1, index_dim]
                self._offload_prepacked_indexer_kv(indexer_kv.squeeze(0))

            self._offload_prepacked_kv(offload_kv)
            attn_output = attn_output.unsqueeze(0)
            return (attn_output, None, None)
        else:
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)
            attn_output, offload_kv = self.module.prefill_attn_w8a16(
                hidden_states, attention_mask, position_ids,
                self.weight_dequant_scale
            )
            return (attn_output, None, offload_kv)

    def _offload_prepacked_indexer_kv(self, offload_kv: torch.Tensor):
        """Offload indexer KV cache per-sequence to auxiliary host memory."""
        cu_seqlens = self.prepack_cu_seqlens
        num_sequences = self.prepack_num_sequences
        global_sequence_ids = self.cur_batch

        for seq_idx in range(num_sequences):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx
            seq_kv = offload_kv[start_idx:end_idx].unsqueeze(0).unsqueeze(2)
            seq_global_id = [global_sequence_ids[seq_idx]]
            self.core_engine.host_paged_kv_worker_view_aux.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )

    def _offload_prepacked_kv(self, offload_kv: torch.Tensor):
        """Offload KV cache per-sequence to host memory."""
        cu_seqlens = self.prepack_cu_seqlens
        num_sequences = self.prepack_num_sequences
        global_sequence_ids = self.cur_batch

        for seq_idx in range(num_sequences):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx
            seq_kv = offload_kv[start_idx:end_idx].unsqueeze(0).unsqueeze(2)
            seq_global_id = [global_sequence_ids[seq_idx]]
            self.core_engine.host_paged_kv_worker_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_kv,
                v_tensor=None,
                sequence_lengths=[seq_len],
            )

    def _forward_decode(self, hidden_states: torch.Tensor, **kwargs) -> Tuple:
        """Decode forward with DSA sparse attention.

        For DSA models:
        1. Compute indexer K for new token, write to auxiliary cache
        2. Score all cached tokens, select top-K
        3. Gather MLA KV at top-K positions from primary cache
        4. Run FlashMLA on sparse subset

        Falls back to standard full-cache FlashMLA when DSA is not active.
        """
        past_key_states = AttnWrapperBase.past_key_states
        position_ids = AttnWrapperBase.position_ids
        cache_seqlens = AttnWrapperBase.cache_seqlens
        max_seqlen = AttnWrapperBase.max_seqlen
        gpu_paged_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        gpu_paged_kv_manager_aux = AttnWrapperBase.gpu_paged_kv_manager_aux

        if gpu_paged_kv_manager is not None:
            # DSA: update auxiliary (indexer) cache with new token
            if gpu_paged_kv_manager_aux is not None and hasattr(self.module, 'indexer'):
                indexer = self.module.indexer
                # Step 1: Compute indexer K for new token and write to aux cache
                indexer_kv = indexer.compute_indexer_kv(hidden_states)
                # indexer_kv: [batch, 1, 1, index_dim]
                gpu_paged_kv_manager_aux.update_layer_decode_new_token(
                    k_tensor=indexer_kv.squeeze(1),  # [batch, 1, index_dim]
                    v_tensor=None,
                    sequence_lengths=cache_seqlens,
                    layer_idx=self.layer_idx,
                )

                # Step 2: Score all cached tokens, select top-K
                # Compute q_a from MLA compression (shared with indexer)
                q_a = self.module.q_a_proj(hidden_states)
                q_a = self.module.q_a_layernorm(q_a)

                indexer_blocked_k, _, idx_block_table = \
                    gpu_paged_kv_manager_aux.get_layer_kv_with_page_table(self.layer_idx)
                updated_seqlens = cache_seqlens + 1
                page_size = gpu_paged_kv_manager_aux.config.page_size_tokens
                top_k_indices = indexer.score_and_select_paged(
                    q_a, hidden_states,
                    indexer_blocked_k, idx_block_table,
                    updated_seqlens, page_size,
                )

                # Step 3: Sparse gather MLA KV at top-K positions
                from batchgen.attention.dsa.sparse_gather import sparse_gather_from_paged_kv
                mla_blocked_k, _, mla_block_table = \
                    gpu_paged_kv_manager.get_layer_kv_with_page_table(self.layer_idx)
                mla_page_size = gpu_paged_kv_manager.config.page_size_tokens
                sparse_mla_kv = sparse_gather_from_paged_kv(
                    mla_blocked_k, mla_block_table, top_k_indices, mla_page_size,
                )
                # sparse_mla_kv: [batch, topk, 1, 576]
                # TODO: Pass sparse_mla_kv to a sparse FlashMLA decode function
                # For now, fall through to full-cache path for correctness

            # Standard BF16 paged KV path (full-cache attention)
            attn_output, k_tensor = self.module.decoding_attn_mode_3_bf16(
                hidden_states,
                position_ids,
                cache_seqlens,
                max_seqlen,
                self.weight_dequant_scale,
                gpu_paged_kv_manager,
                self.layer_idx,
                None
            )
            if AttnWrapperBase.kv_append_callback is not None:
                AttnWrapperBase.kv_append_callback(self.layer_idx, k_tensor, None)
            return (attn_output, None, None)
        else:
            # FP8 KV cache with tensor references
            attention_mask = AttnWrapperBase.attention_mask
            scale = AttnWrapperBase.scale
            layer_past_key = past_key_states[self.layer_idx] if past_key_states else None
            layer_scale = scale[self.layer_idx] if scale else None
            attn_output, updated_past_key, updated_scale = self.module.decoding_attn_mode_3_fp8(
                hidden_states,
                layer_past_key,
                None,
                attention_mask,
                position_ids,
                layer_scale,
                cache_seqlens,
                max_seqlen,
                self.weight_dequant_scale
            )
            return (attn_output, updated_past_key, updated_scale)
