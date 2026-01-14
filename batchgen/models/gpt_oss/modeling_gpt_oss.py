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

"""GPT-OSS-120B model implementation for BatchGen.

Based on OpenAI's reference PyTorch implementation at:
https://github.com/openai/gpt-oss

Key architectural features:
- 36 layers, hidden_size=2880, head_dim=64
- GQA: 64 attention heads, 8 KV heads (8:1 ratio)
- 128 experts, Top-4 routing with softmax normalization
- MXFP4 quantized expert weights
- Alternating sliding (128 tokens) / full attention
- YaRN RoPE with theta=150000, factor=32
- SwiGLU activation with clamping at +/-7.0
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_gpt_oss import GptOssConfig

logger = logging.get_logger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def yarn_find_correction_dim(
    num_rotations: int, dim: int, base: float = 10000, max_position_embeddings: int = 2048
) -> float:
    """Inverse dim formula to find dim based on number of rotations."""
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(
    low_rot: float, high_rot: float, dim: int, base: float = 10000, max_position_embeddings: int = 2048
) -> Tuple[int, int]:
    """Find dim range bounds based on rotations."""
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    """Get magnitude scale for YaRN."""
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(low: int, high: int, dim: int) -> torch.Tensor:
    """Create linear ramp mask for YaRN frequency interpolation."""
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
    return torch.clamp(linear_func, 0, 1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ============================================================================
# Model Components
# ============================================================================

class GptOssRMSNorm(nn.Module):
    """RMSNorm layer as used in GPT-OSS."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class GptOssYaRNRotaryEmbedding(nn.Module):
    """YaRN Rotary Position Embedding for GPT-OSS.

    Implements YaRN (Yet another RoPE extensioN) with:
    - theta=150000
    - factor=32
    - beta_fast=32, beta_slow=1
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 150000.0,
        device: torch.device = None,
        scaling_factor: float = 32.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.max_seq_len_cached = None

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device if device is not None else torch.device("cpu"),
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Build cos/sin cache with YaRN frequency interpolation."""
        self.max_seq_len_cached = seq_len
        dim = self.dim

        # Compute frequency bands
        freq_extra = 1.0 / (self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        # Find interpolation range
        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )

        # Create interpolation mask
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(device=device, dtype=torch.float32)

        # Interpolate frequencies
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build position embeddings
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        # Apply magnitude scaling
        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin embeddings for the given sequence length."""
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class GptOssAttention(nn.Module):
    """Grouped Query Attention with alternating sliding/full attention.

    GPT-OSS uses:
    - 64 query heads, 8 KV heads (GQA with 8:1 ratio)
    - Head dim = 64
    - Alternating sliding window (128 tokens) / full attention per layer
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
        self.sliding_window = config.sliding_window if self.is_sliding else None

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

        # RoPE
        self.rotary_emb = GptOssYaRNRotaryEmbedding(
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
            cos = cos[position_ids].unsqueeze(1)
            sin = sin[position_ids].unsqueeze(1)
        else:
            cos = cos[:q_len].unsqueeze(0).unsqueeze(0)
            sin = sin[:q_len].unsqueeze(0).unsqueeze(0)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # Repeat KV for GQA
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)

        # Compute attention
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Apply sliding window mask if applicable
        if self.is_sliding and self.sliding_window is not None:
            causal_mask = self._create_sliding_window_mask(
                q_len, key_states.shape[-2], self.sliding_window, hidden_states.device
            )
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads for GQA."""
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

    def _create_sliding_window_mask(
        self, q_len: int, kv_len: int, window_size: int, device: torch.device
    ) -> torch.Tensor:
        """Create sliding window attention mask."""
        row_idx = torch.arange(q_len, device=device).unsqueeze(1)
        col_idx = torch.arange(kv_len, device=device).unsqueeze(0)

        # Positions outside the sliding window get -inf
        offset = kv_len - q_len
        distance = col_idx - (row_idx + offset)
        mask = torch.where(
            (distance > 0) | (distance < -window_size),
            torch.tensor(float("-inf"), device=device),
            torch.tensor(0.0, device=device),
        )
        return mask.unsqueeze(0).unsqueeze(0)


class GptOssExpert(nn.Module):
    """Single expert FFN with SwiGLU activation.

    Uses clamped SwiGLU: clamp(silu(gate) * up, -limit, limit)
    Default limit = 7.0
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.swiglu_limit = config.swiglu_limit

        # SwiGLU: gate and up projections combined
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with clamped SwiGLU activation."""
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # SwiGLU with clamping
        hidden = torch.clamp(F.silu(gate) * up, min=-self.swiglu_limit, max=self.swiglu_limit)
        return self.down_proj(hidden)


class GptOssMoE(nn.Module):
    """Mixture of Experts layer with Top-4 routing.

    128 experts total, Top-4 selected per token.
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

        # Route tokens to experts (simple loop implementation)
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


class GptOssDecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.input_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GptOssAttention(config, layer_idx)
        self.post_attention_layernorm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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


class GptOssModel(PreTrainedModel):
    """GPT-OSS transformer model."""

    config_class = GptOssConfig
    base_model_prefix = "model"

    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = GptOssRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        # Create causal attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        attention_mask = self._prepare_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
        )

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

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        inputs_embeds: torch.Tensor,
        past_key_values_length: int,
    ) -> torch.Tensor:
        """Prepare 4D causal attention mask."""
        batch_size, seq_length = input_shape
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

        # Create causal mask
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=dtype, device=device) * float("-inf"),
            diagonal=1,
        )

        # Expand for past key values
        if past_key_values_length > 0:
            causal_mask = torch.cat(
                [
                    torch.zeros((seq_length, past_key_values_length), dtype=dtype, device=device),
                    causal_mask,
                ],
                dim=-1,
            )

        # Expand for batch and heads
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Combine with padding mask
        if attention_mask.dim() == 2:
            expanded_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_length, attention_mask.shape[-1]
            )
            inverted_mask = 1.0 - expanded_mask.float()
            causal_mask = causal_mask + inverted_mask.masked_fill(
                inverted_mask.bool(), float("-inf")
            )

        return causal_mask.to(dtype)


class GptOssForCausalLM(PreTrainedModel):
    """GPT-OSS model with language modeling head."""

    config_class = GptOssConfig
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: GptOssConfig):
        super().__init__(config)
        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
