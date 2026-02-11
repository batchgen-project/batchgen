"""CUDA Graph capturable segments for GPT-OSS-120B decode.

Implements CapturableSegment protocol for the attention block of each decoder
layer. The segment captures:
    input_layernorm → packed QKV proj → QKV split → RoPE → KV cache write → FlashAttn → sink → O_proj → residual_add

Uses fused CUDA kernels (cuda_rmsnorm, cuda_qkv_split, cuda_rope) to minimize
kernel launches inside the captured graph.

The MoE block runs eagerly (outside the graph).
Host KV append callback runs after graph replay.
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from batchgen.cuda_graph.graph_manager import TensorSpec
from batchgen.attention.gqa import gqa_decode_fa
from batchgen.attention.fused_kernels import cuda_rmsnorm, cuda_qkv_split, cuda_rope

logger = logging.getLogger(__name__)


class GptOssAttentionSegment:
    """CapturableSegment for one GPT-OSS-120B decoder layer's attention block.

    During graph capture, all tensor inputs have shapes determined by the
    bucket size. Between replays, inputs are updated via `.copy_()` into
    the static buffers managed by CUDAGraphManager.

    The segment holds references to the attention module's weights and the
    paged KV cache tensors (which have static addresses throughout decode).

    Args:
        attn_wrapper: The GptOssAttnWrapper for this layer.
        input_layernorm: The RMSNorm before attention.
        layer_idx: Decoder layer index (0-35).
        gpu_paged_kv_manager: The GPU paged KV cache manager.
    """

    def __init__(
        self,
        attn_wrapper,  # GptOssAttnWrapper
        input_layernorm: nn.Module,
        layer_idx: int,
        gpu_paged_kv_manager,
        max_position_embeddings: int = 131072,
    ):
        self.wrapper = attn_wrapper
        self.attn_module = attn_wrapper.module  # GptOssAttention
        self.input_layernorm = input_layernorm
        self.layer_idx = layer_idx
        self.gpu_kv_manager = gpu_paged_kv_manager

        # Architecture constants
        self.num_heads = attn_wrapper.num_heads          # 64
        self.num_kv_heads = attn_wrapper.num_kv_heads    # 8
        self.head_dim = attn_wrapper.head_dim             # 64
        self.hidden_size = 2880  # config.hidden_size
        self.q_size = attn_wrapper.q_size                # 4096
        self.kv_size = attn_wrapper.kv_size              # 512
        self.scale = attn_wrapper.scale
        self.sinks = attn_wrapper.sinks
        self.is_sliding = attn_wrapper.is_sliding
        self.sliding_window = attn_wrapper.sliding_window

        # Pre-fetch the full RoPE cos/sin cache to avoid Python-int seq_len
        # dependency inside the graph. The rotary_emb caches cos/sin up to
        # max_position_embeddings at init time.
        rotary_emb = self.attn_module.rotary_emb
        self._rope_cos = rotary_emb.cos_cached  # [max_pos, head_dim]
        self._rope_sin = rotary_emb.sin_cached  # [max_pos, head_dim]

    def get_static_input_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "residual": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "cache_seqlens": TensorSpec(
                ("batch_size",), torch.int32, fill_value=1
            ),
        }

    def get_static_output_specs(self, bucket_size: int) -> Dict[str, TensorSpec]:
        return {
            "hidden_states": TensorSpec(
                ("batch_size", 1, self.hidden_size), torch.bfloat16
            ),
            "key": TensorSpec(
                ("batch_size", 1, self.num_kv_heads, self.head_dim),
                torch.bfloat16,
            ),
            "value": TensorSpec(
                ("batch_size", 1, self.num_kv_heads, self.head_dim),
                torch.bfloat16,
            ),
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Execute the attention block.

        This is the function captured by the CUDA graph. All inputs have
        static shapes (padded to bucket size). The KV cache tensors
        (k_cache, v_cache, block_table) are accessed via the gpu_kv_manager
        reference — their GPU addresses are stable.

        Args:
            hidden_states: [bucket_size, 1, hidden_size] — pre-residual input
            residual: [bucket_size, 1, hidden_size] — residual for skip connection
            cache_seqlens: [bucket_size] int32 — current context lengths

        Returns:
            Dict with:
                hidden_states: [bucket_size, 1, hidden_size] — after attn + residual
                key: [bucket_size, 1, num_kv_heads, head_dim] — for host KV append
                value: [bucket_size, 1, num_kv_heads, head_dim] — for host KV append
        """
        batch = hidden_states.shape[0]

        # === 1. Input LayerNorm (fused CUDA kernel) ===
        normed = cuda_rmsnorm(hidden_states, self.input_layernorm.weight, self.input_layernorm.eps)

        # === 2. Packed QKV Projection + Fused Split ===
        qkv = self.attn_module.qkv_proj(normed)
        query, key, value = cuda_qkv_split(qkv, self.q_size, self.kv_size)

        # Reshape: [batch, 1, num_heads, head_dim]
        query = query.view(batch, 1, self.num_heads, self.head_dim)
        key = key.view(batch, 1, self.num_kv_heads, self.head_dim)
        value = value.view(batch, 1, self.num_kv_heads, self.head_dim)

        # === 3. RoPE (fused CUDA kernel) ===
        current_token_position = cache_seqlens - 1  # [batch]

        # Pre-fetched cos/sin cache indexed by position
        cos_pos = self._rope_cos[current_token_position].unsqueeze(1).to(query.dtype)
        sin_pos = self._rope_sin[current_token_position].unsqueeze(1).to(query.dtype)

        query, key = cuda_rope(query, key, cos_pos, sin_pos)

        # === 4. KV Cache Write ===
        self.gpu_kv_manager.update_layer_decode_new_token(
            k_tensor=key,
            v_tensor=value,
            sequence_lengths=current_token_position,
            layer_idx=self.layer_idx,
        )

        # === 5. FlashAttention with paged KV cache ===
        k_cache, v_cache, page_table = self.gpu_kv_manager.get_layer_kv_with_page_table(
            self.layer_idx
        )

        attn_output, _ = gqa_decode_fa(
            q=query,            # [batch, 1, num_heads, head_dim]
            k_cache=k_cache,    # [num_pages, page_size, num_kv_heads, head_dim]
            v_cache=v_cache,    # [num_pages, page_size, num_kv_heads, head_dim]
            cache_seqlens=cache_seqlens,
            block_table=page_table,
            sinks=self.sinks,
            softmax_scale=self.scale,
            sliding_window=self.sliding_window,
        )

        # === 6. Output Projection ===
        attn_output = attn_output.view(batch, 1, self.num_heads * self.head_dim)
        attn_output = self.attn_module.o_proj(attn_output)

        # === 7. Residual Add ===
        hidden_states_out = residual + attn_output

        return {
            "hidden_states": hidden_states_out,
            "key": key,
            "value": value,
        }
