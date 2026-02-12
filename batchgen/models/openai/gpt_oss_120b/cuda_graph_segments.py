"""CUDA Graph capturable segments for GPT-OSS-120B decode.

Pre-attn graph:   RMSNorm → QKV proj → QKV split → reshape → Q, K, V
Eager middle:     RoPE → KV write → FlashAttention → O_proj  (via wrapper)
Post-attn graph:  residual add + post-attn RMSNorm → normed, residual
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

from batchgen.cuda_graph.graph_manager import TensorSpec

logger = logging.getLogger(__name__)


class PreAttnSegment:
    """Capturable segment: RMSNorm → QKV proj → QKV split → reshape.

    Input:  hidden_states  [B, 1, hidden_size]
    Output: query [B, 1, num_heads, head_dim],
            key   [B, 1, num_kv_heads, head_dim],
            value [B, 1, num_kv_heads, head_dim]
    """

    def __init__(self, decoder_layer, attn_wrapper):
        # RMSNorm params (from decoder layer)
        self.ln_weight = decoder_layer.input_layernorm.weight
        self.ln_eps = decoder_layer.input_layernorm.eps

        # QKV proj (from raw attention module via wrapper)
        self.qkv_proj = attn_wrapper.module.qkv_proj

        # Dimensions
        self.hidden_size = decoder_layer.hidden_size        # 2880
        self.q_size = attn_wrapper.q_size                   # 4096
        self.kv_size = attn_wrapper.kv_size                 # 512
        self.num_heads = attn_wrapper.num_heads             # 64
        self.num_kv_heads = attn_wrapper.num_kv_heads       # 8
        self.head_dim = attn_wrapper.head_dim               # 64

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "query": TensorSpec(
                ("batch_size", 1, self.num_heads, self.head_dim), torch.bfloat16
            ),
            "key": TensorSpec(
                ("batch_size", 1, self.num_kv_heads, self.head_dim), torch.bfloat16
            ),
            "value": TensorSpec(
                ("batch_size", 1, self.num_kv_heads, self.head_dim), torch.bfloat16
            ),
        }

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        from batchgen.attention.fused_kernels import cuda_rmsnorm, cuda_qkv_split

        # RMSNorm
        normed = cuda_rmsnorm(hidden_states, self.ln_weight, self.ln_eps)

        # QKV projection
        qkv = self.qkv_proj(normed)

        # QKV split
        query, key, value = cuda_qkv_split(qkv, self.q_size, self.kv_size)

        # Reshape
        B = hidden_states.shape[0]
        query = query.view(B, 1, self.num_heads, self.head_dim)
        key = key.view(B, 1, self.num_kv_heads, self.head_dim)
        value = value.view(B, 1, self.num_kv_heads, self.head_dim)

        return {"query": query, "key": key, "value": value}


class PostAttnSegment:
    """Capturable segment: fused residual add + post-attention RMSNorm.

    Inputs:  attn_residual [B, 1, hidden_size], attn_output [B, 1, hidden_size]
    Outputs: normed [B, 1, hidden_size] (MoE input), residual [B, 1, hidden_size]
    """

    def __init__(self, decoder_layer):
        self.ln_weight = decoder_layer.post_attention_layernorm.weight
        self.ln_eps = decoder_layer.post_attention_layernorm.eps
        self.hidden_size = decoder_layer.hidden_size  # 2880

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "attn_residual": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "attn_output": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
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
        self, attn_residual: torch.Tensor, attn_output: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        from batchgen.attention.fused_kernels import cuda_add_rmsnorm

        normed, residual = cuda_add_rmsnorm(
            attn_residual, attn_output, self.ln_weight, self.ln_eps
        )
        return {"normed": normed, "residual": residual}
