# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  Licensed under the Apache License, Version 2.0 (the "License");              #
#  you may not use this file except in compliance with the License.             #
#                                                                               #
#  You may obtain a copy of the License at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/LICENSE-2.0                   #
#                                                                               #
#  Unless required by applicable law or agreed to in writing, software          #
#  distributed under the License is distributed on an "AS IS" BASIS,            #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.     #
#  See the License for the specific language governing permissions and          #
#  limitations under the License.                                               #
# ---------------------------------------------------------------------------- #

"""Kimi K2.5 model definition following BatchGen flat design pattern.

This module defines the core model structure for Kimi K2.5 (DeepSeek-V3 architecture
with K2.5-specific hyperparameters). Weight loading, quantization, and inference
optimization are handled by separate wrappers and parameter server.

Architecture:
    - 61 transformer layers (3 dense + 58 MoE)
    - MLA attention with 64 heads, kv_lora_rank=512
    - 384 routed experts + 1 shared expert per MoE layer
    - INT4 W4A16 quantization (routed experts only)
    - RoPE theta=50000, YaRN scaling
    - RMSNorm (eps=1e-6)

Reference: DeepSeek-V3 architecture with K2.5 modifications.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import KimiK25Config


# ============================================================================
# RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# ============================================================================
# MLA Attention (Multi-head Latent Attention)
# ============================================================================

class KimiK25Attention(nn.Module):
    """MLA attention for K2.5.

    Multi-head Latent Attention with compressed KV cache:
        - Q projection: q_a (1536) → q_b (64*192=12,288)
        - KV projection: kv_a (576) → kv_b (64*256=16,384)
        - Compressed KV cache: 576-dim instead of separate K/V
        - RoPE on qk_rope_dim=64, non-rope on qk_nope_dim=128
        - 64 attention heads, v_head_dim=128

    This is a structural stub — actual forward pass (FlashAttention, RoPE,
    KV cache) is handled by KimiK25AttnWrapper for prefill/decode dispatch.
    """

    def __init__(self, config: KimiK25Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads  # 64
        self.q_lora_rank = config.q_lora_rank  # 1536
        self.kv_lora_rank = config.kv_lora_rank  # 512
        self.qk_nope_head_dim = config.qk_nope_head_dim  # 128
        self.qk_rope_head_dim = config.qk_rope_head_dim  # 64
        self.v_head_dim = config.v_head_dim  # 128
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 192

        # Q projection with low-rank compression
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=1e-6)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        # KV projection with MQA-style compression
        # Output: kv_lora_rank (512) for compressed KV + qk_rope_head_dim (64) for K RoPE
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=1e-6)
        # Output: num_heads * (qk_nope_head_dim + v_head_dim) = 64 * (128 + 128) = 16,384
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLA forward pass (stub).

        The actual implementation (FlashAttention-2, RoPE, KV cache) is handled
        by KimiK25AttnWrapper which intercepts this call.

        This stub defines the projection structure only.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            attn_output: [batch, seq, hidden_size]
        """
        # This forward is a placeholder — wrappers intercept for optimized attention
        # The projections (q_a/q_b/kv_a/kv_b/o) are used by wrappers
        raise NotImplementedError(
            "KimiK25Attention.forward() is a stub. "
            "Use KimiK25AttnWrapper for actual attention computation."
        )


# ============================================================================
# Dense MLP (First 3 Layers)
# ============================================================================

class DenseMLP(nn.Module):
    """Dense FFN for first 3 K2.5 layers (non-MoE).

    Standard SwiGLU-style MLP:
        gate_out = SiLU(gate_proj(x))
        up_out = up_proj(x)
        output = down_proj(gate_out * up_out)

    No expert routing — always active for all tokens.
    """

    def __init__(self, config: KimiK25Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size  # 18432

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Dense MLP forward.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            output: [batch, seq, hidden_size]
        """
        gate_out = self.act_fn(self.gate_proj(hidden_states))
        up_out = self.up_proj(hidden_states)
        return self.down_proj(gate_out * up_out)


# ============================================================================
# MoE Layer (384 Routed + 1 Shared Expert)
# ============================================================================

class KimiK25MoE(nn.Module):
    """MoE layer with 384 routed + 1 shared expert.

    Routing: Simple top-8 (n_group=1, no group-based selection)
    Experts: INT4 W4A16 quantization (routed), BF16 (shared)
    Kernels: Fused gate+up projection via grouped GEMM

    This is a structural stub — actual expert forward (INT4 dequant + grouped GEMM)
    is handled by KimiK25ExpertWrapper for prefill/decode dispatch.
    """

    def __init__(self, config: KimiK25Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts  # 384
        self.top_k = config.num_experts_per_tok  # 8
        self.moe_intermediate_size = config.moe_intermediate_size  # 2048

        # Router (BF16)
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)

        # Routed experts (stubs — wrappers handle INT4 dequant + GEMM)
        # Each expert: gate_proj (7168 → 2048), up_proj (7168 → 2048), down_proj (2048 → 7168)
        # Actual forward handled by KimiK25ExpertWrapper
        # We define structure only (nn.ModuleList for weight storage)
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                'gate_proj': nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False),
                'up_proj': nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False),
                'down_proj': nn.Linear(self.moe_intermediate_size, self.hidden_size, bias=False),
            })
            for _ in range(self.num_experts)
        ])

        # Shared expert (BF16, always active)
        self.shared_experts = nn.ModuleDict({
            'gate_proj': nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False),
            'up_proj': nn.Linear(self.hidden_size, self.moe_intermediate_size, bias=False),
            'down_proj': nn.Linear(self.moe_intermediate_size, self.hidden_size, bias=False),
        })
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE forward pass (stub).

        The actual implementation (top-k routing, INT4 dequant, grouped GEMM) is
        handled by KimiK25ExpertWrapper which intercepts this call.

        This stub defines the router and expert structure only.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            output: [batch, seq, hidden_size]
        """
        # This forward is a placeholder — wrappers intercept for optimized MoE
        # The gate and expert weights are used by wrappers
        raise NotImplementedError(
            "KimiK25MoE.forward() is a stub. "
            "Use KimiK25ExpertWrapper for actual expert computation."
        )


# ============================================================================
# Decoder Layer
# ============================================================================

class KimiK25DecoderLayer(nn.Module):
    """Single K2.5 transformer layer with MLA attention + MoE/dense FFN.

    Pre-norm architecture:
        1. LayerNorm → Attention → Residual
        2. LayerNorm → MLP/MoE → Residual

    First 3 layers use DenseMLP, remaining 58 layers use KimiK25MoE.
    """

    def __init__(self, config: KimiK25Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-norm for attention
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = KimiK25Attention(config, layer_idx)

        # Pre-norm for MLP/MoE
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # First 3 layers: dense MLP, remaining: MoE
        if layer_idx < config.first_k_dense_replace:
            self.mlp = DenseMLP(config)
        else:
            self.mlp = KimiK25MoE(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Layer forward with pre-norm + residual connections.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            hidden_states: [batch, seq, hidden_size]
        """
        # Pre-norm attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # Pre-norm MoE/FFN + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# Main Model
# ============================================================================

class KimiK25Model(nn.Module):
    """Kimi K2.5 main model (flat design).

    Architecture:
        - 61 transformer layers
        - MLA attention (kv_lora_rank=512, q_lora_rank=1536, 64 heads)
        - First 3 layers: dense MLP (18432 intermediate)
        - Remaining 58 layers: 384 routed experts + 1 shared expert
        - RMSNorm (eps=1e-6), SiLU activation
        - INT4 W4A16 quantization (handled by wrappers)

    This model follows the BatchGen flat design pattern:
        - Single class with embed_tokens, layers, norm, unembedding
        - No outer wrapper (unlike HuggingFace's ForCausalLM nesting)
        - Wrappers handle optimization (quantization, KV cache, kernels)
        - model.py defines structure only
    """

    def __init__(self, config: KimiK25Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Token embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer layers (61 total: 3 dense + 58 MoE)
        self.layers = nn.ModuleList([
            KimiK25DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Final layer norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Unembedding (lm_head)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: Token IDs [batch, seq] (text-only mode)
            inputs_embeds: Pre-computed embeddings [batch, seq, hidden_size]
                (multimodal mode with vision tokens already replaced)

        Returns:
            logits: [batch, seq, vocab_size]

        Note:
            When inputs_embeds is provided (multimodal mode), input_ids is ignored.
            This enables vision-language models to inject vision embeddings directly.
        """
        # Embedding lookup (skip if inputs_embeds provided)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Final norm + unembedding
        hidden_states = self.norm(hidden_states)
        logits = self.unembedding(hidden_states)

        return logits
