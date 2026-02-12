"""CUDA Graph capturable segment for GPT-OSS-120B decode: full attention block.

Captures the entire attention block in one graph:
  RMSNorm → QKV proj → split → reshape → RoPE → KV write → FA → O_proj
  → residual add + post-attn RMSNorm

Dynamic metadata (cache_seqlens, page_table) are static-address input buffers
whose contents are updated before each replay. KV cache and cos/sin tables
are at fixed GPU addresses.
"""

import logging
from typing import Dict

import torch

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.models.wrappers.attention import AttnWrapperBase

logger = logging.getLogger(__name__)


class FullAttnSegment:
    """Full attention block as a single CUDA-graph-capturable segment.

    Inputs:  hidden_states [B, 1, hidden_size], cache_seqlens [B] int32,
             page_table [B, max_pages_per_seq] int32
    Outputs: normed [B, 1, hidden_size] (MoE input), residual [B, 1, hidden_size]
    """

    def __init__(self, decoder_layer, attn_wrapper, layer_idx: int, max_seq_len: int,
                 max_pages_per_seq: int):
        # Pre-attn: RMSNorm
        self.ln_weight = decoder_layer.input_layernorm.weight
        self.ln_eps = decoder_layer.input_layernorm.eps

        # Pre-attn: QKV proj
        self.qkv_proj = attn_wrapper.module.qkv_proj

        # Dimensions
        self.hidden_size = decoder_layer.hidden_size
        self.q_size = attn_wrapper.q_size
        self.kv_size = attn_wrapper.kv_size
        self.num_heads = attn_wrapper.num_heads
        self.num_kv_heads = attn_wrapper.num_kv_heads
        self.head_dim = attn_wrapper.head_dim

        # Mid: RoPE
        self.rotary_emb = attn_wrapper.module.rotary_emb
        self.max_seq_len = max_seq_len

        # Mid: attention
        self.o_proj = attn_wrapper.module.o_proj
        self.scale = attn_wrapper.scale
        self.sliding_window = attn_wrapper.sliding_window
        self.sinks = attn_wrapper.sinks
        self.layer_idx = layer_idx

        # Post-attn: RMSNorm
        self.post_ln_weight = decoder_layer.post_attention_layernorm.weight
        self.post_ln_eps = decoder_layer.post_attention_layernorm.eps

        # Page table dimensions for static buffer
        self.max_pages_per_seq = max_pages_per_seq

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
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "normed": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "residual": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
        }

    def forward(
        self, hidden_states: torch.Tensor, cache_seqlens: torch.Tensor,
        page_table: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        from batchgen.attention.fused_kernels import (
            cuda_rmsnorm, cuda_qkv_split, cuda_rope, cuda_add_rmsnorm,
        )
        from batchgen.attention.gqa.fa_decode import gqa_decode_fa

        B = hidden_states.shape[0]

        # === Pre-attn: RMSNorm → QKV proj → split → reshape ===
        normed = cuda_rmsnorm(hidden_states, self.ln_weight, self.ln_eps)
        qkv = self.qkv_proj(normed)
        query, key, value = cuda_qkv_split(qkv, self.q_size, self.kv_size)
        query = query.view(B, 1, self.num_heads, self.head_dim)
        key = key.view(B, 1, self.num_kv_heads, self.head_dim)
        value = value.view(B, 1, self.num_kv_heads, self.head_dim)

        # === RoPE ===
        current_pos = cache_seqlens - 1
        cos, sin = self.rotary_emb(value.transpose(1, 2), seq_len=self.max_seq_len)
        cos = cos[current_pos].unsqueeze(1)
        sin = sin[current_pos].unsqueeze(1)
        query, key = cuda_rope(query, key, cos, sin)

        # === KV write ===
        gpu_kv_manager = AttnWrapperBase.gpu_paged_kv_manager
        gpu_kv_manager.update_layer_decode_new_token(
            k_tensor=key, v_tensor=value,
            sequence_lengths=current_pos,
            layer_idx=self.layer_idx,
        )

        # === FlashAttention ===
        k_cache, v_cache, _ = gpu_kv_manager.get_layer_kv_with_page_table(
            self.layer_idx
        )
        attn_out, _ = gqa_decode_fa(
            q=query, k_cache=k_cache, v_cache=v_cache,
            cache_seqlens=cache_seqlens, block_table=page_table,
            sinks=self.sinks, softmax_scale=self.scale,
            sliding_window=self.sliding_window,
        )

        # === O_proj ===
        attn_out = attn_out.view(B, 1, self.num_heads * self.head_dim)
        attn_out = self.o_proj(attn_out)

        # === Post-attn: residual add + RMSNorm ===
        normed, residual = cuda_add_rmsnorm(
            hidden_states, attn_out, self.post_ln_weight, self.post_ln_eps
        )
        return {"normed": normed, "residual": residual}
