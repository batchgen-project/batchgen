"""CUDA Graph capturable segments for GPT-OSS-120B decode.

Two segments per decoder layer, with KV write + FlashAttention running eagerly
between them:

  Pre-attn segment (graph):  input_layernorm → packed QKV proj → QKV split → RoPE
  Eager middle:              KV cache write → FlashAttention
  Post-attn segment (graph): O_proj → residual_add

FlashAttention with paged KV cache is excluded from graphs because:
- Dynamic workspace allocation based on max(cache_seqlens)
- Variable page table access patterns in continuous batching

The MoE block runs eagerly (outside both segments).
Host KV append callback runs after the eager middle.
"""

import logging
from typing import Dict

import torch
import torch.nn as nn

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.attention.fused_kernels import cuda_rmsnorm, cuda_qkv_split, cuda_rope

logger = logging.getLogger(__name__)


class GptOssPreAttnSegment:
    """Capturable segment: RMSNorm → QKV proj → QKV split → RoPE.

    Args:
        attn_wrapper: The GptOssAttnWrapper for this layer.
        input_layernorm: The RMSNorm before attention.
        layer_idx: Decoder layer index (0-35).
    """

    def __init__(
        self,
        attn_wrapper,
        input_layernorm: nn.Module,
        layer_idx: int,
    ):
        self.attn_module = attn_wrapper.module
        self.input_layernorm = input_layernorm
        self.layer_idx = layer_idx

        self.num_heads = attn_wrapper.num_heads          # 64
        self.num_kv_heads = attn_wrapper.num_kv_heads    # 8
        self.head_dim = attn_wrapper.head_dim             # 64
        self.hidden_size = 2880
        self.q_size = attn_wrapper.q_size                # 4096
        self.kv_size = attn_wrapper.kv_size              # 512

        # Pre-fetch RoPE cos/sin to avoid CPU-GPU sync inside graph
        rotary_emb = self.attn_module.rotary_emb
        self._rope_cos = rotary_emb.cos_cached  # [max_pos, head_dim]
        self._rope_sin = rotary_emb.sin_cached  # [max_pos, head_dim]

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "cache_seqlens": TensorSpec(
                ("batch_size",), torch.int32, fill_value=1
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """RMSNorm → QKV proj → QKV split → RoPE.

        Args:
            hidden_states: [bucket_size, 1, hidden_size]
            cache_seqlens: [bucket_size] int32

        Returns:
            query: [bucket_size, 1, num_heads, head_dim]
            key: [bucket_size, 1, num_kv_heads, head_dim]
            value: [bucket_size, 1, num_kv_heads, head_dim]
        """
        batch = hidden_states.shape[0]

        # 1. Input LayerNorm
        normed = cuda_rmsnorm(hidden_states, self.input_layernorm.weight, self.input_layernorm.eps)

        # 2. Packed QKV Projection + Split
        qkv = self.attn_module.qkv_proj(normed)
        query, key, value = cuda_qkv_split(qkv, self.q_size, self.kv_size)

        query = query.view(batch, 1, self.num_heads, self.head_dim)
        key = key.view(batch, 1, self.num_kv_heads, self.head_dim)
        value = value.view(batch, 1, self.num_kv_heads, self.head_dim)

        # 3. RoPE
        current_token_position = cache_seqlens - 1
        cos_pos = self._rope_cos[current_token_position].unsqueeze(1).to(query.dtype)
        sin_pos = self._rope_sin[current_token_position].unsqueeze(1).to(query.dtype)
        query, key = cuda_rope(query, key, cos_pos, sin_pos)

        return {"query": query, "key": key, "value": value}


class GptOssPostAttnSegment:
    """Capturable segment: O_proj only.

    The residual add + post-attention layernorm is handled eagerly via
    cuda_add_rmsnorm to match the main branch's fused kernel exactly.

    Args:
        attn_wrapper: The GptOssAttnWrapper for this layer.
    """

    def __init__(self, attn_wrapper):
        self.attn_module = attn_wrapper.module
        self.num_heads = attn_wrapper.num_heads      # 64
        self.head_dim = attn_wrapper.head_dim         # 64
        self.hidden_size = 2880

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "attn_output": TensorSpec(
                ("batch_size", 1, self.num_heads, self.head_dim), torch.bfloat16
            ),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "o_proj_output": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
        }

    def forward(
        self,
        attn_output: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """O_proj only.

        Args:
            attn_output: [bucket_size, 1, num_heads, head_dim]

        Returns:
            o_proj_output: [bucket_size, 1, hidden_size]
        """
        batch = attn_output.shape[0]
        attn_output = attn_output.view(batch, 1, self.num_heads * self.head_dim)
        o_proj_output = self.attn_module.o_proj(attn_output)
        return {"o_proj_output": o_proj_output}
