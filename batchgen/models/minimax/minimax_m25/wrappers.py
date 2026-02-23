# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""MiniMax-M2.5 wrappers for BatchGen execution.

Provides wrappers for MiniMax-M2.5 with FP8 quantization:
- MiniMaxM25ExpertWrapper: Expert wrapper with FP8 block-wise dequantization
- MiniMaxM25AttnWrapper: Attention wrapper with GQA + QK norm + partial RoPE
"""

import logging
import math
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from batchgen.models.wrappers import ExpertWrapperBase, AttnWrapperBase
from batchgen.quantization.fp8e4m3 import deepseek_v3_dequantization


class MiniMaxM25ExpertWrapper(ExpertWrapperBase):
    """Expert wrapper with FP8 dequantization for MiniMax-M2.5.

    Reuses DeepSeek-V3's FP8 block-wise [128,128] dequantization.
    Expert FFN: gate_proj [1536, 3072], up_proj [1536, 3072], down_proj [3072, 1536]
    Activation: standard SwiGLU (silu(gate) * up)
    """

    def __init__(self, module, layer_idx, expert_idx, core_engine, engine_config,
                 model_config, persistent=False, weight_dequant_scale=None):
        super().__init__(
            module, layer_idx, expert_idx, core_engine, engine_config, model_config,
            persistent
        )
        self.weight_dequant_scale = weight_dequant_scale or {}
        self.fp8_gate = None
        self.fp8_up = None
        self.fp8_down = None

    def dequantize_weights(self, weights_dict):
        """Dequantize FP8 weights using block-wise scale factors."""
        result = {}
        for name, weight in weights_dict.items():
            scale_key = f"{name}_scale_inv"
            if scale_key in self.weight_dequant_scale:
                result[name] = deepseek_v3_dequantization(
                    weight, self.weight_dequant_scale[scale_key]
                )
            else:
                result[name] = weight
        return result

    def _register_fp8_weights(self):
        self.fp8_gate = self.module.gate_proj.weight.data
        self.fp8_up = self.module.up_proj.weight.data
        self.fp8_down = self.module.down_proj.weight.data

    def _unregister_fp8_weights(self):
        self.fp8_gate = None
        self.fp8_up = None
        self.fp8_down = None

    def _forward_impl(self, hidden_states):
        """Forward using deepgemm kernel (FP8 weight × BF16 activation)."""
        return self.module.deepgemm_forward(hidden_states, self.weight_dequant_scale)

    def forward(self, hidden_states):
        rank = self.get_rank_safe()
        if not self.persistent:
            weights = self.load_weights(self.module_key)
            for name, param in self.module.named_parameters():
                param.data = weights[name]
        else:
            self.module.gate_proj.weight.data = self.fp8_gate
            self.module.up_proj.weight.data = self.fp8_up
            self.module.down_proj.weight.data = self.fp8_down

        result = self._forward_impl(hidden_states)

        if not self.persistent:
            torch.cuda.current_stream(
                self.engine_config.Basic_Config.device_torch
            ).synchronize()
            self.free_weights(self.module_key)
            self.clear_weights()

        return result


class MiniMaxM25AttnWrapper(AttnWrapperBase):
    """Attention wrapper for MiniMax-M2.5 GQA with QK norm + partial RoPE.

    M2.5 attention is BF16 (not quantized). Only expert FFN weights are FP8.
    """

    def __init__(self, module, layer_idx, core_engine, engine_config,
                 model_config, persistent=True, weight_dequant_scale=None):
        super().__init__(
            module, layer_idx, core_engine, engine_config, model_config,
            persistent, weight_dequant_scale
        )

    def dequantize_weights(self, weights_dict):
        """M2.5 attention is BF16 — no dequantization needed."""
        return weights_dict

    def _forward_prefill(self, hidden_states, **kwargs):
        """Prefill forward using GQA FlashAttention."""
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if self.prepack_mode:
            hidden_states_2d = hidden_states.squeeze(0)
            attn_output, k_for_cache, v_for_cache = self.module.prefill_attn_gqa_prepacked(
                hidden_states_2d,
                self.position_ids.to(hidden_states_2d.device),
                self.prepack_cu_seqlens.to(hidden_states_2d.device),
                self.prepack_max_seqlen,
                self.prepack_num_sequences,
            )
            self._offload_prepacked_kv_gqa(k_for_cache, v_for_cache)
            attn_output = attn_output.unsqueeze(0)
            return (attn_output, None, None)
        else:
            attn_output, k_for_cache, v_for_cache = self.module.prefill_attn_gqa(
                hidden_states, attention_mask, position_ids,
            )
            return (attn_output, None, (k_for_cache, v_for_cache))

    def _offload_prepacked_kv_gqa(self, k_cache, v_cache):
        """Offload GQA KV cache per-sequence to host memory."""
        cu_seqlens = self.prepack_cu_seqlens
        num_sequences = self.prepack_num_sequences
        global_sequence_ids = self.cur_batch

        for seq_idx in range(num_sequences):
            start_idx = cu_seqlens[seq_idx].item()
            end_idx = cu_seqlens[seq_idx + 1].item()
            seq_len = end_idx - start_idx

            seq_k = k_cache[start_idx:end_idx].unsqueeze(0)
            seq_v = v_cache[start_idx:end_idx].unsqueeze(0)
            seq_global_id = [global_sequence_ids[seq_idx]]

            self.core_engine.host_paged_kv_worker_view.async_offload_layer_kv_to_host(
                layer_idx=self.layer_idx,
                sequence_ids=seq_global_id,
                k_tensor=seq_k,
                v_tensor=seq_v,
                sequence_lengths=[seq_len],
            )

    def _forward_decode(self, hidden_states, **kwargs):
        """Decode forward using GQA FlashAttention with paged KV cache."""
        position_ids = AttnWrapperBase.position_ids
        cache_seqlens = AttnWrapperBase.cache_seqlens
        max_seqlen = AttnWrapperBase.max_seqlen
        gpu_paged_kv_manager = AttnWrapperBase.gpu_paged_kv_manager

        if gpu_paged_kv_manager is not None:
            attn_output, k_tensor, v_tensor = self.module.decoding_attn_gqa_paged(
                hidden_states, position_ids, cache_seqlens, max_seqlen,
                gpu_paged_kv_manager, self.layer_idx,
            )
            if AttnWrapperBase.kv_append_callback is not None:
                AttnWrapperBase.kv_append_callback(self.layer_idx, k_tensor, v_tensor)
            return (attn_output, None, None)
        else:
            # Fallback
            past_key_states = AttnWrapperBase.past_key_states
            past_value_states = AttnWrapperBase.past_value_states
            attention_mask = AttnWrapperBase.attention_mask

            attn_output, attn_weights, present = self.module(
                hidden_states, attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=(
                    past_key_states[self.layer_idx] if past_key_states else None,
                    past_value_states[self.layer_idx] if past_value_states else None,
                ),
                use_cache=True,
            )
            return (attn_output, present[0] if present else None, present[1] if present else None)
