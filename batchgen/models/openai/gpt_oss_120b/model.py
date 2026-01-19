# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
#  you may not use this file except in compliance with the license.             #
#                                                                               #
#  you may obtain a copy of the license at                                      #
#                                                                               #
#                  http://www.apache.org/licenses/license-2.0                   #
#                                                                               #
#  unless required by applicable law or agreed to in writing, software          #
#  distributed under the license is distributed on an "as is" basis,            #
#  without warranties or conditions of any kind, either express or implied.     #
#  see the license for the specific language governing permissions and          #
#  limitations under the license.                                               #
# ---------------------------------------------------------------------------- #

"""GPT-OSS-120B model implementation following OpenAI's reference architecture.

Key features:
- 36 layers, hidden_size=2880, head_dim=64
- GQA: 64 attention heads, 8 KV heads (8:1 ratio)
- 128 experts, Top-4 routing with softmax normalization
- Alternating sliding (128 tokens) / full attention per layer
- YaRN RoPE with theta=150000, factor=32
- SwiGLU activation: (x_glu * sigmoid(alpha * x_glu)) * (x_linear + 1)
- Sink tokens: learnable per-head parameters in attention

Reference: https://github.com/openai/gpt-oss
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_gpt_oss import GptOssConfig


# ============================================================================
# SwiGLU Activation (OpenAI's exact formula)
# ============================================================================

def swiglu(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0) -> torch.Tensor:
    """SwiGLU activation with OpenAI's formula.

    The input x is interleaved: [gate0, up0, gate1, up1, ...]
    - x_glu: gating values (even indices)
    - x_linear: linear values (odd indices)

    Formula: (x_glu * sigmoid(alpha * x_glu)) * (x_linear + 1)
    with input clamping at ±limit.

    Args:
        x: Input tensor with interleaved gate/up values
        alpha: Sigmoid scaling factor (default: 1.702)
        limit: Clamping limit for inputs (default: 7.0)

    Returns:
        Activated tensor with half the last dimension
    """
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the INPUT values (not output)
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note: add extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


# ============================================================================
# RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(input_dtype)


# ============================================================================
# YaRN Rotary Position Embedding
# ============================================================================

class YaRNRotaryEmbedding(nn.Module):
    """YaRN Rotary Position Embedding for extended context.

    Implements YaRN (Yet another RoPE extensioN) with:
    - theta=150000
    - factor=32
    - beta_fast=32, beta_slow=1

    Reference: https://arxiv.org/abs/2309.00071
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 150000.0,
        scaling_factor: float = 32.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        device: torch.device = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self._compute_inv_freq(device)
        self._set_cos_sin_cache(max_position_embeddings, device, torch.get_default_dtype())

    def _compute_inv_freq(self, device: torch.device):
        """Compute inverse frequencies with YaRN interpolation."""
        freq = self.base ** (
            torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim
        )

        if self.scaling_factor > 1.0:
            # YaRN concentration
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0

            d_half = self.dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.original_max_position_embeddings / (self.beta_slow * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.original_max_position_embeddings / (self.beta_fast * 2 * math.pi))
                / math.log(self.base)
            )

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
            self.concentration = concentration
        else:
            inv_freq = 1.0 / freq
            self.concentration = 1.0

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Build cos/sin cache for positions."""
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * self.concentration).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.concentration).to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin embeddings for the given sequence length."""
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(1)  # [seq, 1, dim]
    sin = sin.unsqueeze(1)  # [seq, 1, dim]

    q1, q2 = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
    k1, k2 = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]

    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)

    return q_rot, k_rot


# ============================================================================
# Attention Block with Sink Tokens
# ============================================================================

class GptOssAttention(nn.Module):
    """Grouped Query Attention with sink tokens and alternating sliding/full attention.

    Key features:
    - 64 query heads, 8 KV heads (GQA with 8:1 ratio)
    - Head dim = 64
    - Sink tokens: learnable per-head parameters added to attention softmax
    - Alternating sliding window (128 tokens on even layers) / full attention (odd layers)
    """

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Determine if this layer uses sliding window
        self.is_sliding = config.is_sliding_attention(layer_idx)
        self.sliding_window = config.sliding_window if self.is_sliding else 0

        # Sink tokens: learnable per-head parameters
        self.sinks = nn.Parameter(torch.empty(self.num_heads, dtype=torch.bfloat16))

        # Projections
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # Attention scale
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # RoPE
        self.rotary_emb = YaRNRotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=config.rope_scaling.get("factor", 32.0),
            original_max_position_embeddings=config.rope_scaling.get(
                "original_max_position_embeddings", 4096
            ),
            beta_fast=config.rope_scaling.get("beta_fast", 32.0),
            beta_slow=config.rope_scaling.get("beta_slow", 1.0),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Get RoPE embeddings
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        # Apply RoPE
        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        else:
            cos = cos[:q_len]
            sin = sin[:q_len]

        # For GQA: reshape Q to [batch, kv_heads, q_mult, seq, head_dim]
        query_states = query_states.view(
            bsz, self.num_kv_heads, self.num_key_value_groups, q_len, self.head_dim
        )
        query_states = query_states.transpose(2, 3)  # [batch, kv_heads, seq, q_mult, head_dim]

        # Apply RoPE to Q and K
        query_flat = query_states.view(bsz * self.num_kv_heads, q_len * self.num_key_value_groups, self.head_dim)
        key_flat = key_states.view(bsz * self.num_kv_heads, q_len, self.head_dim)

        # Simple RoPE application
        cos_expanded = cos.unsqueeze(0).expand(bsz, -1, -1)  # [batch, seq, dim]
        sin_expanded = sin.unsqueeze(0).expand(bsz, -1, -1)

        query_states = query_states.view(bsz, self.num_heads, q_len, self.head_dim)
        key_states = key_states.view(bsz, self.num_kv_heads, q_len, self.head_dim)

        # Transpose for RoPE: [batch, seq, heads, head_dim]
        q_for_rope = query_states.transpose(1, 2).contiguous()
        k_for_rope = key_states.transpose(1, 2).contiguous()

        # Apply rotary embeddings
        q_for_rope = q_for_rope.view(bsz, q_len, self.num_heads, self.head_dim)
        k_for_rope = k_for_rope.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        cos_q = cos_expanded.unsqueeze(2)  # [batch, seq, 1, dim]
        sin_q = sin_expanded.unsqueeze(2)

        q1, q2 = q_for_rope[..., :self.head_dim//2], q_for_rope[..., self.head_dim//2:]
        k1, k2 = k_for_rope[..., :self.head_dim//2], k_for_rope[..., self.head_dim//2:]

        cos_half = cos_q[..., :self.head_dim//2]
        sin_half = sin_q[..., :self.head_dim//2]

        q_rot = torch.cat([q1 * cos_half - q2 * sin_half, q2 * cos_half + q1 * sin_half], dim=-1)
        k_rot = torch.cat([k1 * cos_half - k2 * sin_half, k2 * cos_half + k1 * sin_half], dim=-1)

        query_states = q_rot.transpose(1, 2)  # [batch, heads, seq, head_dim]
        key_states = k_rot.transpose(1, 2)    # [batch, kv_heads, seq, head_dim]
        value_states = value_states           # [batch, kv_heads, seq, head_dim]

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for GQA
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        # Create causal mask
        kv_len = key_states.shape[-2]
        causal_mask = torch.triu(
            torch.ones((q_len, kv_len), dtype=torch.bool, device=attn_weights.device),
            diagonal=kv_len - q_len + 1
        )

        # Apply sliding window mask if applicable
        if self.is_sliding and self.sliding_window > 0:
            row_idx = torch.arange(q_len, device=attn_weights.device).unsqueeze(1)
            col_idx = torch.arange(kv_len, device=attn_weights.device).unsqueeze(0)
            offset = kv_len - q_len
            distance = col_idx - (row_idx + offset)
            sliding_mask = distance < -self.sliding_window
            causal_mask = causal_mask | sliding_mask

        attn_weights = attn_weights.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Add sink tokens to attention
        # Sinks are added as an extra column in the attention scores
        # Shape: [batch, heads, seq, 1]
        sink_scores = self.sinks.view(1, self.num_heads, 1, 1).expand(bsz, -1, q_len, 1)
        attn_weights_with_sinks = torch.cat([attn_weights, sink_scores], dim=-1)

        # Softmax with sinks
        attn_probs = F.softmax(attn_weights_with_sinks, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Remove sink column before matmul with values
        attn_probs = attn_probs[..., :-1]

        # Apply attention to values
        attn_output = torch.matmul(attn_probs, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# ============================================================================
# Expert Module (Single Expert FFN with SwiGLU)
# ============================================================================

class GptOssExpert(nn.Module):
    """Single expert FFN with OpenAI's SwiGLU activation.

    Uses fused gate+up projection (mlp1) followed by SwiGLU and down projection (mlp2).
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.swiglu_limit = config.swiglu_limit

        # Fused gate+up projection (output is 2x intermediate_size for SwiGLU interleaving)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        # Down projection
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with OpenAI's SwiGLU activation."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)

        # Interleave gate and up for SwiGLU
        # Create interleaved tensor: [gate0, up0, gate1, up1, ...]
        interleaved = torch.stack([gate, up], dim=-1).view(*gate.shape[:-1], -1)

        # Apply OpenAI's SwiGLU
        hidden = swiglu(interleaved, alpha=1.702, limit=self.swiglu_limit)

        return self.down_proj(hidden)


# ============================================================================
# MoE Layer (Mixture of Experts)
# ============================================================================

class GptOssMoE(nn.Module):
    """Mixture of Experts layer with Top-4 routing.

    128 experts total, Top-4 selected per token with softmax-normalized weights.
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Router
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([GptOssExpert(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with Top-K expert routing."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # Compute routing logits
        router_logits = self.router(hidden_states_flat)  # [batch*seq, num_experts]

        # Select Top-K experts
        topk_weights, topk_indices = torch.topk(router_logits, k=self.num_experts_per_tok, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)

        # Initialize output
        output = torch.zeros_like(hidden_states_flat)

        # Route tokens to experts
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (topk_indices == i).any(dim=-1)
            if expert_mask.any():
                # Get weights for this expert
                expert_weights = torch.where(
                    topk_indices == i,
                    topk_weights,
                    torch.zeros_like(topk_weights),
                ).sum(dim=-1)

                # Process tokens
                expert_input = hidden_states_flat[expert_mask]
                expert_output = expert(expert_input)
                output[expert_mask] += expert_output * expert_weights[expert_mask].unsqueeze(-1)

        return output.view(batch_size, seq_len, hidden_dim)


# ============================================================================
# Decoder Layer
# ============================================================================

class GptOssDecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GptOssAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GptOssMoE(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        residual = hidden_states

        # Pre-norm + attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Pre-norm + MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights, present_key_value


# ============================================================================
# Main Model
# ============================================================================

class GptOssModel(nn.Module):
    """GPT-OSS-120B transformer model (OpenAI-style, no HuggingFace dependencies)."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Debug: Log config values before creating embedding
        import logging
        logging.info(f"GptOssModel.__init__: Creating embedding with vocab_size={config.vocab_size}, hidden_size={config.hidden_size}, padding_idx={self.padding_idx}")

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Debug: Log actual embedding shape
        logging.info(f"GptOssModel.__init__: embed_tokens.weight.shape={self.embed_tokens.weight.shape}")

        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache += (layer_outputs[2],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return (hidden_states, next_cache, all_hidden_states, all_self_attns)


class GptOss(nn.Module):
    """GPT-OSS-120B model with language modeling head.

    Instantiate with BatchGen config:
        model = GptOss(config)
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config

        # Debug: Log config values
        import logging
        logging.info(f"GptOss.__init__: vocab_size={config.vocab_size}, hidden_size={config.hidden_size}")

        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Debug: Log actual parameter shapes
        logging.info(f"GptOss.__init__: embed_tokens.weight.shape={self.model.embed_tokens.weight.shape}")
        logging.info(f"GptOss.__init__: lm_head.weight.shape={self.lm_head.weight.shape}")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        return (loss, logits, outputs[1], outputs[2], outputs[3])

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
        }
