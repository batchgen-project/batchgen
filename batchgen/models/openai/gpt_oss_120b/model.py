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

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_gpt_oss import GptOssConfig

# CUDA routing kernels (fused gate + dispatch + reduce)
try:
    from batchgen.moe.routing import gate_topk_softmax_cuda
    _HAS_CUDA_ROUTING = True
except ImportError:
    _HAS_CUDA_ROUTING = False

# WGMMA single-expert kernels (per-expert WGMMA, highest priority decode path)
try:
    from batchgen.moe.fused_wgmma_expert import (
        fused_mxfp4_expert_forward,
        wgmma_single_expert_moe_forward_cuda_routing,
        is_wgmma_available,
    )
    if os.environ.get("BATCHGEN_DISABLE_WGMMA_SINGLE", "0") == "1":
        _HAS_WGMMA_SINGLE = False
        print("[WGMMA single] disabled by BATCHGEN_DISABLE_WGMMA_SINGLE", flush=True)
    else:
        _HAS_WGMMA_SINGLE = is_wgmma_available()
        if _HAS_WGMMA_SINGLE:
            print("[WGMMA single] available (confirmed correct, highest priority)", flush=True)
        else:
            print("[WGMMA single] not available (SM90 required)", flush=True)
except Exception as e:
    _HAS_WGMMA_SINGLE = False
    print(f"[WGMMA single] failed to load: {e}", flush=True)

_WGMMA_SINGLE_LOGGED = False

# WGMMA grouped kernels (fused gate+up+SwiGLU + down, 1D+offsets layout)
# NOTE: Disabled by default — produces gibberish in production (routing/layout bug).
# Use BATCHGEN_ENABLE_WGMMA_GROUPED=1 to explicitly enable for debugging.
try:
    from batchgen.moe.fused_wgmma_grouped import (
        fused_mxfp4_grouped_moe_forward_cuda_routing,
        is_grouped_wgmma_available,
    )
    if os.environ.get("BATCHGEN_ENABLE_WGMMA_GROUPED", "0") == "1":
        _HAS_WGMMA_GROUPED = is_grouped_wgmma_available()
        if _HAS_WGMMA_GROUPED:
            print("[WGMMA grouped] explicitly enabled via BATCHGEN_ENABLE_WGMMA_GROUPED", flush=True)
        else:
            print("[WGMMA grouped] requested but not available (SM90 required)", flush=True)
    else:
        _HAS_WGMMA_GROUPED = False
        print("[WGMMA grouped] disabled by default (use BATCHGEN_ENABLE_WGMMA_GROUPED=1 to enable)", flush=True)
except Exception as e:
    import traceback
    _HAS_WGMMA_GROUPED = False
    print(f"[WGMMA grouped] failed to load: {e}", flush=True)
    traceback.print_exc()

_WGMMA_GROUPED_LOGGED = False  # one-time invocation log
_COMPARE_GROUPED = os.environ.get("BATCHGEN_COMPARE_GROUPED", "0") == "1"
_COMPARE_COUNT = 0
_COMPARE_MAX = int(os.environ.get("BATCHGEN_COMPARE_MAX", "5"))
_FULL_COMPARE = os.environ.get("BATCHGEN_DEBUG_FULL_COMPARE", "0") == "1"
_FULL_COMPARE_COUNT = 0
_FULL_COMPARE_MAX = 300


# ============================================================================
# Decode Layer Timing Infrastructure
# ============================================================================

@dataclass
class LayerTimingStats:
    """Per-layer timing statistics for decode."""
    layer_idx: int
    attn_ms: float = 0.0
    moe_ms: float = 0.0
    moe_router_ms: float = 0.0
    moe_dispatch_ms: float = 0.0
    moe_gemm_ms: float = 0.0
    moe_combine_ms: float = 0.0


class DecodeLayerTiming:
    """Global timing collector for decode layers.

    Enable with: export BATCHGEN_LAYER_TIMING=1

    Usage:
        DecodeLayerTiming.start_layer(layer_idx)
        # ... do attention ...
        DecodeLayerTiming.record_attn(elapsed_ms)
        # ... do MoE ...
        DecodeLayerTiming.record_moe(elapsed_ms)
        DecodeLayerTiming.end_layer()

        # At end of forward pass:
        DecodeLayerTiming.print_summary()
    """
    enabled: bool = os.environ.get("BATCHGEN_LAYER_TIMING", "0") == "1"
    layer_stats: List[LayerTimingStats] = []
    current_layer: Optional[LayerTimingStats] = None
    _iteration_count: int = 0

    @classmethod
    def start_layer(cls, layer_idx: int):
        if cls.enabled:
            cls.current_layer = LayerTimingStats(layer_idx=layer_idx)

    @classmethod
    def record_attn(cls, elapsed_ms: float):
        if cls.enabled and cls.current_layer:
            cls.current_layer.attn_ms = elapsed_ms

    @classmethod
    def record_moe(cls, elapsed_ms: float):
        if cls.enabled and cls.current_layer:
            cls.current_layer.moe_ms = elapsed_ms

    @classmethod
    def record_moe_component(cls, component: str, elapsed_ms: float):
        if cls.enabled and cls.current_layer:
            setattr(cls.current_layer, f"moe_{component}_ms", elapsed_ms)

    @classmethod
    def end_layer(cls):
        if cls.enabled and cls.current_layer:
            cls.layer_stats.append(cls.current_layer)
            cls.current_layer = None

    @classmethod
    def print_summary(cls):
        """Print timing summary and reset stats."""
        if not cls.layer_stats:
            return

        cls._iteration_count += 1

        # Calculate totals
        total_attn = sum(s.attn_ms for s in cls.layer_stats)
        total_moe = sum(s.moe_ms for s in cls.layer_stats)
        total_router = sum(s.moe_router_ms for s in cls.layer_stats)
        total_gemm = sum(s.moe_gemm_ms for s in cls.layer_stats)
        total_combine = sum(s.moe_combine_ms for s in cls.layer_stats)
        total_time = total_attn + total_moe

        num_layers = len(cls.layer_stats)

        print(f"\n=== Decode Timing (iter {cls._iteration_count}, {num_layers} layers) ===")
        print(f"Total: {total_time:.2f} ms ({1000/total_time:.1f} tokens/sec)")
        print(f"  Attention: {total_attn:.2f} ms ({100*total_attn/total_time:.1f}%)")
        print(f"  MoE:       {total_moe:.2f} ms ({100*total_moe/total_time:.1f}%)")
        if total_router > 0 or total_gemm > 0:
            print(f"    Router:  {total_router:.2f} ms")
            print(f"    GEMM:    {total_gemm:.2f} ms")
            print(f"    Combine: {total_combine:.2f} ms")

        # Per-layer breakdown (first 3 layers)
        print(f"\nPer-layer (first 3):")
        for s in cls.layer_stats[:3]:
            print(f"  L{s.layer_idx}: attn={s.attn_ms:.2f}ms, moe={s.moe_ms:.2f}ms")

        cls.layer_stats.clear()

    @classmethod
    def reset(cls):
        """Reset all stats."""
        cls.layer_stats.clear()
        cls.current_layer = None
        cls._iteration_count = 0


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
            # CRITICAL: beta_fast (32.0) is used for LOW (interpolation boundary)
            #           beta_slow (1.0) is used for HIGH (extrapolation boundary)
            # This matches OpenAI's ntk_beta=32 for low, ntk_alpha=1 for high
            low = (
                d_half
                * math.log(self.original_max_position_embeddings / (self.beta_fast * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.original_max_position_embeddings / (self.beta_slow * 2 * math.pi))
                / math.log(self.base)
            )
            # Sanity check: low < high (same assertion as OpenAI reference)
            assert 0 < low < high < d_half - 1, f"YaRN params invalid: low={low}, high={high}, d_half={d_half}"

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
        # NOTE: Must be zeros (not empty) - uninitialized values corrupt attention
        self.sinks = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.bfloat16))

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

        # Router (OpenAI's gate uses bias=True)
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=True)

        # Experts
        self.experts = nn.ModuleList([GptOssExpert(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with Top-K expert routing."""
        timing_enabled = DecodeLayerTiming.enabled
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)

        # ========== ROUTER TIMING ==========
        if timing_enabled:
            torch.cuda.synchronize()
            router_start = time.perf_counter()

        # Compute routing logits
        router_logits = self.router(hidden_states_flat)  # [batch*seq, num_experts]

        # Select Top-K experts with fused CUDA gate (topk + softmax in one kernel)
        if _HAS_CUDA_ROUTING:
            topk_indices, topk_weights = gate_topk_softmax_cuda(
                router_logits, k=self.num_experts_per_tok
            )

        else:
            topk_weights, topk_indices = torch.topk(router_logits, k=self.num_experts_per_tok, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)

        if timing_enabled:
            torch.cuda.synchronize()
            DecodeLayerTiming.record_moe_component("router", (time.perf_counter() - router_start) * 1000)

        # Initialize output
        output = torch.zeros_like(hidden_states_flat)

        # ========== GEMM TIMING (per-expert loop) ==========
        if timing_enabled:
            torch.cuda.synchronize()
            gemm_start = time.perf_counter()

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

        if timing_enabled:
            torch.cuda.synchronize()
            DecodeLayerTiming.record_moe_component("gemm", (time.perf_counter() - gemm_start) * 1000)

        return output.view(batch_size, seq_len, hidden_dim)


# ============================================================================
# Quantized MoE Layer (MXFP4 with Grouped GEMM)
# ============================================================================

class GptOssMoEQuantized(nn.Module):
    """MoE layer with MXFP4 quantized experts and grouped GEMM execution.

    This class provides hybrid execution:
    - Persistent experts (weights in VRAM): Use grouped GEMM (3 kernel launches total)
    - Non-persistent experts (loaded on-demand): Use optimized single-expert kernel

    The grouped GEMM approach reduces kernel launches from 128×3=384 to just 3,
    achieving ~5x speedup for persistent experts.

    Weight storage:
    - gate_weights: [num_experts] list of [intermediate_size, hidden_size//2] uint8
    - gate_scales: [num_experts] list of [intermediate_size, hidden_size//32] uint8
    - up_weights, up_scales: Same shapes as gate
    - down_weights: [num_experts] list of [hidden_size, intermediate_size//2] uint8
    - down_scales: [num_experts] list of [hidden_size, intermediate_size//32] uint8
    - gate_biases: [num_experts, intermediate_size] BF16 (optional)
    - up_biases: [num_experts, intermediate_size] BF16 (optional)
    - down_biases: [num_experts, hidden_size] BF16 (optional)
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Router (same as original)
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=True)

        # MXFP4 quantized weights - initialized as None, populated during weight loading
        # Stored as lists of tensors (one per expert) for flexibility
        self.gate_weights = None  # List[Tensor[N_inter, hidden//2]]
        self.gate_scales = None   # List[Tensor[N_inter, hidden//32]]
        self.up_weights = None
        self.up_scales = None
        self.down_weights = None  # List[Tensor[hidden, N_inter//2]]
        self.down_scales = None

        # Optional biases (stacked as [num_experts, N])
        self.gate_biases = None
        self.up_biases = None
        self.down_biases = None

        # Pointer arrays for grouped GEMM (set after weight loading)
        self.gate_ptrs = None
        self.gate_scale_ptrs = None
        self.up_ptrs = None
        self.up_scale_ptrs = None
        self.down_ptrs = None
        self.down_scale_ptrs = None

        # Persistent expert mask (which experts are in VRAM)
        # True = persistent (use grouped GEMM), False = non-persistent (use single kernel)
        self.persistent_mask = None  # [num_experts] bool tensor

        # SwiGLU parameters
        self.swiglu_alpha = 1.702
        self.swiglu_limit = getattr(config, 'swiglu_limit', 7.0)

    def setup_pointer_arrays(self):
        """Create pointer arrays for grouped GEMM. Call after loading weights."""
        from batchgen.moe.mxfp4_grouped_gemm import setup_expert_weight_pointers

        if self.gate_weights is None:
            raise RuntimeError("Weights not loaded. Call setup_pointer_arrays() after loading weights.")

        # Ensure all weight/scale tensors are contiguous before capturing pointers.
        # The grouped kernel reads via raw pointers with stride = shape[1], so
        # non-contiguous tensors would cause incorrect data access.
        self.gate_weights = [w.contiguous() for w in self.gate_weights]
        self.gate_scales = [s.contiguous() for s in self.gate_scales]
        self.up_weights = [w.contiguous() for w in self.up_weights]
        self.up_scales = [s.contiguous() for s in self.up_scales]
        self.down_weights = [w.contiguous() for w in self.down_weights]
        self.down_scales = [s.contiguous() for s in self.down_scales]

        # Create pointer arrays (now guaranteed contiguous)
        self.gate_ptrs, self.gate_scale_ptrs = setup_expert_weight_pointers(
            self.gate_weights, self.gate_scales
        )
        self.up_ptrs, self.up_scale_ptrs = setup_expert_weight_pointers(
            self.up_weights, self.up_scales
        )
        self.down_ptrs, self.down_scale_ptrs = setup_expert_weight_pointers(
            self.down_weights, self.down_scales
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Hybrid forward: grouped GEMM for persistent, single for non-persistent."""
        from batchgen.moe.mxfp4_grouped_gemm import (
            grouped_mxfp4_moe_forward_3d,
            mxfp4_expert_forward_single,
        )
        if _HAS_CUDA_ROUTING:
            from batchgen.moe.mxfp4_grouped_gemm import grouped_mxfp4_moe_forward_cuda_routing

        timing_enabled = DecodeLayerTiming.enabled
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        # ========== ROUTER TIMING ==========
        if timing_enabled:
            torch.cuda.synchronize()
            router_start = time.perf_counter()

        # Compute routing logits
        router_logits = self.router(hidden_flat)

        # Select Top-K experts with fused CUDA gate (topk + softmax in one kernel)
        if _HAS_CUDA_ROUTING:
            topk_indices, topk_weights = gate_topk_softmax_cuda(
                router_logits, k=self.num_experts_per_tok
            )
        else:
            topk_weights, topk_indices = torch.topk(router_logits, k=self.num_experts_per_tok, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)

        if timing_enabled:
            torch.cuda.synchronize()
            DecodeLayerTiming.record_moe_component("router", (time.perf_counter() - router_start) * 1000)

        # Initialize output
        output = torch.zeros_like(hidden_flat)

        # ========== GEMM TIMING ==========
        if timing_enabled:
            torch.cuda.synchronize()
            gemm_start = time.perf_counter()

        # Check if pointer arrays are set up
        if self.gate_ptrs is None:
            self.setup_pointer_arrays()

        # Determine persistent vs non-persistent experts
        if self.persistent_mask is not None:
            has_non_persistent = not self.persistent_mask.all()
        else:
            # Default: all experts are persistent
            has_non_persistent = False

        # === Part A: Grouped GEMM for all experts (or persistent only) ===
        if not has_non_persistent:
            # Comparison diagnostic: run both WGMMA and Triton, compare outputs
            global _COMPARE_COUNT
            run_compare = (_COMPARE_GROUPED and _HAS_WGMMA_GROUPED
                           and _HAS_CUDA_ROUTING and _COMPARE_COUNT < _COMPARE_MAX)

            if _HAS_WGMMA_SINGLE and _HAS_CUDA_ROUTING:
                # WGMMA single-expert: per-expert WGMMA + CUDA routing (highest priority)
                global _WGMMA_SINGLE_LOGGED
                if not _WGMMA_SINGLE_LOGGED:
                    print("[WGMMA single] MoE forward using per-expert WGMMA path", flush=True)
                    _WGMMA_SINGLE_LOGGED = True
                output = wgmma_single_expert_moe_forward_cuda_routing(
                    hidden_flat, topk_indices, topk_weights,
                    self.gate_weights, self.gate_scales,
                    self.up_weights, self.up_scales,
                    self.down_weights, self.down_scales,
                    gate_biases=self.gate_biases,
                    up_biases=self.up_biases,
                    down_biases=self.down_biases,
                    num_experts=self.num_experts,
                    num_local_experts=self.num_experts,
                )
            elif _HAS_WGMMA_GROUPED and _HAS_CUDA_ROUTING and not run_compare:
                # WGMMA grouped: fast path (4 kernel launches)
                global _WGMMA_GROUPED_LOGGED
                if not _WGMMA_GROUPED_LOGGED:
                    print("[WGMMA grouped] MoE forward using grouped path (4 launches)", flush=True)
                    _WGMMA_GROUPED_LOGGED = True
                _debug_lists = None
                if os.environ.get("BATCHGEN_DEBUG_GROUPED", "0") == "1":
                    _debug_lists = (
                        self.gate_weights, self.gate_scales,
                        self.up_weights, self.up_scales,
                        self.down_weights, self.down_scales,
                    )
                output = fused_mxfp4_grouped_moe_forward_cuda_routing(
                    hidden_flat, topk_indices, topk_weights,
                    self.gate_ptrs, self.gate_scale_ptrs,
                    self.up_ptrs, self.up_scale_ptrs,
                    self.down_ptrs, self.down_scale_ptrs,
                    self.gate_weights[0], self.gate_scales[0],
                    self.down_weights[0], self.down_scales[0],
                    num_experts=self.num_experts,
                    num_local_experts=self.num_experts,
                    _debug_weight_lists=_debug_lists,
                )
            elif run_compare:
                # DIAGNOSTIC: run both paths and compare
                _COMPARE_COUNT += 1
                torch.cuda.synchronize()

                # Print context on first comparison
                if _COMPARE_COUNT == 1:
                    gw = self.gate_weights[0]
                    gs = self.gate_scales[0]
                    dw = self.down_weights[0]
                    ds = self.down_scales[0]
                    print(f"[COMPARE] Context: input={hidden_flat.shape} "
                          f"experts={self.num_experts} topk={topk_indices.shape[1]}",
                          flush=True)
                    print(f"  gate_w={gw.shape} stride={gw.stride()} contig={gw.is_contiguous()} "
                          f"gate_s={gs.shape} stride={gs.stride()} contig={gs.is_contiguous()}",
                          flush=True)
                    print(f"  down_w={dw.shape} stride={dw.stride()} contig={dw.is_contiguous()} "
                          f"down_s={ds.shape} stride={ds.stride()} contig={ds.is_contiguous()}",
                          flush=True)
                    print(f"  gate_w.shape[0]={gw.shape[0]} (N_intermediate) "
                          f"gate_w.shape[1]={gw.shape[1]} (s1_stride_weight_n) "
                          f"gate_s.shape[1]={gs.shape[1]} (s1_stride_scale_n)",
                          flush=True)
                    print(f"  down_w.shape[1]={dw.shape[1]} (s2_stride_weight_n) "
                          f"down_s.shape[1]={ds.shape[1]} (s2_stride_scale_n)",
                          flush=True)

                # 1) WGMMA grouped output
                wgmma_out = fused_mxfp4_grouped_moe_forward_cuda_routing(
                    hidden_flat, topk_indices, topk_weights,
                    self.gate_ptrs, self.gate_scale_ptrs,
                    self.up_ptrs, self.up_scale_ptrs,
                    self.down_ptrs, self.down_scale_ptrs,
                    self.gate_weights[0], self.gate_scales[0],
                    self.down_weights[0], self.down_scales[0],
                    num_experts=self.num_experts,
                    num_local_experts=self.num_experts,
                )
                torch.cuda.synchronize()

                # 2) Triton grouped output (known correct)
                triton_out = grouped_mxfp4_moe_forward_cuda_routing(
                    hidden_flat, topk_indices, topk_weights,
                    self.gate_ptrs, self.gate_scale_ptrs,
                    self.up_ptrs, self.up_scale_ptrs,
                    self.down_ptrs, self.down_scale_ptrs,
                    self.gate_weights[0], self.gate_scales[0],
                    self.up_weights[0], self.up_scales[0],
                    self.down_weights[0], self.down_scales[0],
                    self.gate_biases, self.up_biases, self.down_biases,
                    num_experts=self.num_experts,
                    num_local_experts=self.num_experts,
                    swiglu_alpha=self.swiglu_alpha,
                    swiglu_limit=self.swiglu_limit,
                )
                torch.cuda.synchronize()

                # 3) Compare
                diff = (wgmma_out.float() - triton_out.float()).abs()
                ref_abs = triton_out.float().abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                max_ref = ref_abs.max().item()
                rel_err = (diff / (ref_abs + 1e-8)).max().item()

                # Check with BF16 WGMMA tolerance
                tol = 1e-5 + 1.6e-2 * ref_abs
                n_fail = (diff > tol).sum().item()
                n_total = diff.numel()
                fail_pct = n_fail / n_total * 100

                # Output range stats
                w_min = wgmma_out.float().min().item()
                w_max = wgmma_out.float().max().item()
                w_nan = torch.isnan(wgmma_out).sum().item()
                w_inf = torch.isinf(wgmma_out).sum().item()
                t_min = triton_out.float().min().item()
                t_max = triton_out.float().max().item()

                print(f"[COMPARE #{_COMPARE_COUNT}] "
                      f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} "
                      f"rel_err={rel_err:.6f} fail={n_fail}/{n_total} ({fail_pct:.4f}%)",
                      flush=True)
                print(f"  WGMMA range: [{w_min:.4f}, {w_max:.4f}] nan={w_nan} inf={w_inf} "
                      f"Triton range: [{t_min:.4f}, {t_max:.4f}] "
                      f"ref_max={max_ref:.4f}", flush=True)

                if max_diff > 1.0:
                    # Large divergence — dump more detail
                    nonzero_w = (wgmma_out.float().abs() > 1e-8).sum().item()
                    nonzero_t = (triton_out.float().abs() > 1e-8).sum().item()
                    all_zero_w = (wgmma_out == 0).all().item()
                    print(f"  WARNING: large divergence! "
                          f"nonzero WGMMA={nonzero_w} Triton={nonzero_t} "
                          f"total_elements={n_total} all_zero_w={all_zero_w}",
                          flush=True)

                # Use Triton output (known correct) for model
                output = triton_out

            elif _HAS_CUDA_ROUTING:
                output = grouped_mxfp4_moe_forward_cuda_routing(
                    hidden_flat, topk_indices, topk_weights,
                    self.gate_ptrs, self.gate_scale_ptrs,
                    self.up_ptrs, self.up_scale_ptrs,
                    self.down_ptrs, self.down_scale_ptrs,
                    self.gate_weights[0], self.gate_scales[0],
                    self.up_weights[0], self.up_scales[0],
                    self.down_weights[0], self.down_scales[0],
                    self.gate_biases, self.up_biases, self.down_biases,
                    num_experts=self.num_experts,
                    num_local_experts=self.num_experts,
                    swiglu_alpha=self.swiglu_alpha,
                    swiglu_limit=self.swiglu_limit,
                )
            else:
                output = grouped_mxfp4_moe_forward_3d(
                    hidden_flat, topk_indices, topk_weights,
                    self.gate_ptrs, self.gate_scale_ptrs,
                    self.up_ptrs, self.up_scale_ptrs,
                    self.down_ptrs, self.down_scale_ptrs,
                    self.gate_weights[0], self.gate_scales[0],
                    self.up_weights[0], self.up_scales[0],
                    self.down_weights[0], self.down_scales[0],
                    self.gate_biases, self.up_biases, self.down_biases,
                    num_experts=self.num_experts,
                    swiglu_alpha=self.swiglu_alpha,
                    swiglu_limit=self.swiglu_limit,
                )
        else:
            # Hybrid execution: grouped for persistent, single for non-persistent
            persistent_experts = self.persistent_mask.nonzero(as_tuple=True)[0]
            non_persistent_experts = (~self.persistent_mask).nonzero(as_tuple=True)[0]

            # Part A: Grouped GEMM for persistent experts
            # Create mask for tokens routed to persistent experts only
            persistent_routing_mask = torch.zeros_like(topk_indices, dtype=torch.bool)
            for pe in persistent_experts:
                persistent_routing_mask |= (topk_indices == pe)

            if persistent_routing_mask.any():
                # Use grouped GEMM for persistent experts
                # Note: grouped_mxfp4_moe_forward_3d handles all experts but only
                # processes tokens routed to experts in the pointer arrays
                persistent_output = grouped_mxfp4_moe_forward_3d(
                    hidden_flat, topk_indices, topk_weights,
                    self.gate_ptrs, self.gate_scale_ptrs,
                    self.up_ptrs, self.up_scale_ptrs,
                    self.down_ptrs, self.down_scale_ptrs,
                    self.gate_weights[0], self.gate_scales[0],
                    self.up_weights[0], self.up_scales[0],
                    self.down_weights[0], self.down_scales[0],
                    self.gate_biases, self.up_biases, self.down_biases,
                    num_experts=self.num_experts,
                    swiglu_alpha=self.swiglu_alpha,
                    swiglu_limit=self.swiglu_limit,
                )
                output += persistent_output

            # Part B: Single-expert kernel for non-persistent experts
            for expert_idx in non_persistent_experts.tolist():
                expert_mask = (topk_indices == expert_idx).any(dim=-1)
                if expert_mask.any():
                    expert_input = hidden_flat[expert_mask]

                    # Use WGMMA single-expert if available, else Triton
                    if _HAS_WGMMA_SINGLE:
                        expert_output = fused_mxfp4_expert_forward(
                            expert_input,
                            self.gate_weights[expert_idx], self.gate_scales[expert_idx],
                            self.up_weights[expert_idx], self.up_scales[expert_idx],
                            self.down_weights[expert_idx], self.down_scales[expert_idx],
                            gate_bias=self.gate_biases[expert_idx] if self.gate_biases is not None else None,
                            up_bias=self.up_biases[expert_idx] if self.up_biases is not None else None,
                            down_bias=self.down_biases[expert_idx] if self.down_biases is not None else None,
                        )
                    else:
                        expert_output = mxfp4_expert_forward_single(
                            expert_input,
                            self.gate_weights[expert_idx], self.gate_scales[expert_idx],
                            self.up_weights[expert_idx], self.up_scales[expert_idx],
                            self.down_weights[expert_idx], self.down_scales[expert_idx],
                            self.gate_biases[expert_idx] if self.gate_biases is not None else None,
                            self.up_biases[expert_idx] if self.up_biases is not None else None,
                            self.down_biases[expert_idx] if self.down_biases is not None else None,
                            swiglu_alpha=self.swiglu_alpha,
                            swiglu_limit=self.swiglu_limit,
                        )

                    # Get routing weight for this expert
                    expert_weight = torch.where(
                        topk_indices[expert_mask] == expert_idx,
                        topk_weights[expert_mask],
                        torch.zeros_like(topk_weights[expert_mask])
                    ).sum(dim=-1)

                    output[expert_mask] += expert_output * expert_weight.unsqueeze(-1)

        if timing_enabled:
            torch.cuda.synchronize()
            DecodeLayerTiming.record_moe_component("gemm", (time.perf_counter() - gemm_start) * 1000)

        return output.view(batch_size, seq_len, hidden_dim)


# ============================================================================
# EP-Enabled MoE Layer (Expert Parallelism with AllGather/AllReduce)
# ============================================================================

class GptOssMoE_EP(nn.Module):
    """EP-enabled MoE for GPT-OSS-120B with MXFP4 quantization.

    Distributes 128 experts across multiple ranks using Expert Parallelism:
    - Each rank owns 128 // world_size experts
    - Communication: AllGather tokens → Route globally → Process local experts → AllReduce

    Based on DeepSeek's DeepseekV3MoE_Decoding_FP8 pattern (modeling_deepseek_v3.py:1757-2066).
    """

    def __init__(self, config: GptOssConfig, comm=None):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok  # 4
        self.hidden_size = config.hidden_size
        self.comm = comm

        # Import distributed after checking availability
        import torch.distributed as dist

        # Distributed metadata
        if not dist.is_initialized():
            self.rank, self.world_size = 0, 1
        else:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.experts_per_rank = 128 // self.world_size
        self.total_experts = 128
        self.routed_expert_start_idx = self.rank * self.experts_per_rank
        self.routed_expert_end_idx = (self.rank + 1) * self.experts_per_rank

        # Router (replicated across all ranks - routes to all 128 experts)
        self.router = nn.Linear(config.hidden_size, 128, bias=True)

        # Experts placeholder - only local experts will be non-None
        # Populated by Parallel_Strategy_Manager._swap_to_ep_moe()
        self.experts = nn.ModuleList([None] * 128)

        # Communication setup
        self.device = torch.device("cuda", self.rank % torch.cuda.device_count())
        self.num_tokens_per_rank = None

        # EP offloading flag (set by Parallel_Strategy_Manager)
        self.enable_ep_offloading = False

        # SwiGLU parameters
        self.swiglu_alpha = getattr(config, 'swiglu_alpha', 1.702)
        self.swiglu_limit = getattr(config, 'swiglu_limit', 7.0)

        # Grouped WGMMA pointer arrays (set by Parallel_Strategy_Manager)
        self.gate_ptrs = None
        self.gate_scale_ptrs = None
        self.up_ptrs = None
        self.up_scale_ptrs = None
        self.down_ptrs = None
        self.down_scale_ptrs = None
        self.gate_weight_ref = None
        self.gate_scale_ref = None
        self.down_weight_ref = None
        self.down_scale_ref = None
        self._use_grouped_wgmma = False
        self._grouped_logged = False

    def init_num_tokens(self, num_tokens_per_rank: int):
        """Initialize communication buffers for given batch size.

        Pre-allocates all buffers to avoid per-forward allocation overhead.
        """
        self.num_tokens_per_rank = num_tokens_per_rank
        global_num_tokens = num_tokens_per_rank * self.world_size
        K = self.num_experts_per_tok
        hidden_size = self.config.hidden_size

        # Pre-allocate index tensors (following DeepSeek pattern)
        self.token_idx_buffer = torch.arange(
            global_num_tokens, dtype=torch.int64, device=self.device
        ).repeat_interleave(K)
        self.topk_pos_buffer = torch.arange(
            K, dtype=torch.int64, device=self.device
        ).repeat(global_num_tokens)

        # Pre-allocate communication buffers
        self.all_tokens_buffer = torch.zeros(
            (global_num_tokens, hidden_size), device=self.device, dtype=torch.bfloat16
        )
        self.padded_hidden_buffer = torch.zeros(
            (num_tokens_per_rank, hidden_size), device=self.device, dtype=torch.bfloat16
        )
        self.global_results_buffer = torch.zeros(
            (global_num_tokens, hidden_size), device=self.device, dtype=torch.bfloat16
        )

        # Pre-allocate expert counts buffer
        self.expert_counts_buffer = torch.zeros(128, dtype=torch.int32, device=self.device)

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int):
        """Update num_tokens_per_rank for dynamic batch size.

        Reallocates buffers only when size changes.
        """
        if num_tokens_per_rank == self.num_tokens_per_rank:
            return  # No reallocation needed

        # Reallocate all buffers with new size
        self.init_num_tokens(num_tokens_per_rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with EP communication."""
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # Use loop-based execution (simpler, works with MXFP4 wrappers)
        out = self.moe_infer_loop_with_offloading(hidden_states)

        return out.view(*orig_shape)

    @torch.inference_mode()
    def moe_infer_loop_with_offloading(self, x: torch.Tensor) -> torch.Tensor:
        """Loop-based expert execution with AllGather/AllReduce.

        Following DeepSeek's moe_infer_loop_with_offloading (lines 1933-2066):
        1. AllGather tokens from all ranks
        2. Route globally (router logits for all 128 experts)
        3. Loop through local experts, process tokens
        4. AllReduce to combine results
        5. Extract local rank's results

        Args:
            x: Input tensor [num_tokens, hidden_size]

        Returns:
            Output tensor [num_tokens, hidden_size]
        """
        import torch.distributed as dist

        num_tokens, hidden_size = x.shape
        device = x.device

        # Safety check
        if self.num_tokens_per_rank is None:
            raise RuntimeError("num_tokens_per_rank not set. Call init_num_tokens() first.")

        if num_tokens > self.num_tokens_per_rank:
            raise RuntimeError(
                f"MoE buffer overflow: num_tokens={num_tokens} > num_tokens_per_rank={self.num_tokens_per_rank}"
            )

        # ---- 1) AllGather: Collect tokens from all ranks ----
        # Reuse pre-allocated buffers (avoid per-forward allocation)
        all_tokens = self.all_tokens_buffer
        all_tokens.zero_()

        # Pad local tokens to num_tokens_per_rank
        padded_hidden_states = self.padded_hidden_buffer
        padded_hidden_states.zero_()
        if num_tokens > 0:
            padded_hidden_states[:num_tokens] = x

        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                all_tokens,
                padded_hidden_states,
                stream=torch.cuda.default_stream(self.device)
            )

        # ---- 2) Router: Compute routing for ALL global tokens ----
        router_logits = self.router(all_tokens)  # [global_tokens, 128]
        if _HAS_CUDA_ROUTING:
            topk_indices, topk_weights = gate_topk_softmax_cuda(
                router_logits, k=self.num_experts_per_tok
            )

        else:
            topk_weights, topk_indices = torch.topk(
                router_logits, k=self.num_experts_per_tok, dim=-1
            )
            topk_weights = F.softmax(topk_weights, dim=-1)

        # ---- 3) Process local experts ----
        num_global_tokens = all_tokens.shape[0]
        K = self.num_experts_per_tok

        # Use pre-allocated output buffer (BF16 to avoid dtype conversion in loop)
        global_results = self.global_results_buffer
        global_results.zero_()

        # Comparison diagnostic: compare WGMMA grouped vs per-expert loop
        global _COMPARE_COUNT, _FULL_COMPARE_COUNT
        run_compare = (_COMPARE_GROUPED and self._use_grouped_wgmma
                       and _HAS_CUDA_ROUTING and _COMPARE_COUNT < _COMPARE_MAX)

        if self._use_grouped_wgmma and _HAS_CUDA_ROUTING and not run_compare:
            # Grouped WGMMA: 4 kernel launches for all local experts
            if not self._grouped_logged:
                logging.info(
                    f"[WGMMA grouped] EP rank {self.rank}: using grouped path "
                    f"(experts [{self.routed_expert_start_idx}, "
                    f"{self.routed_expert_start_idx + self.experts_per_rank}), 4 launches)"
                )
                self._grouped_logged = True

            _debug_lists = None
            if os.environ.get("BATCHGEN_DEBUG_GROUPED", "0") == "1":
                if hasattr(self, '_local_gate_weights'):
                    _debug_lists = (
                        self._local_gate_weights, self._local_gate_scales,
                        self._local_up_weights, self._local_up_scales,
                        self._local_down_weights, self._local_down_scales,
                    )
            _need_internals = (_FULL_COMPARE
                               and _FULL_COMPARE_COUNT < _FULL_COMPARE_MAX)
            _result = fused_mxfp4_grouped_moe_forward_cuda_routing(
                all_tokens, topk_indices, topk_weights,
                self.gate_ptrs, self.gate_scale_ptrs,
                self.up_ptrs, self.up_scale_ptrs,
                self.down_ptrs, self.down_scale_ptrs,
                self.gate_weight_ref, self.gate_scale_ref,
                self.down_weight_ref, self.down_scale_ref,
                num_experts=self.total_experts,
                expert_start=self.routed_expert_start_idx,
                num_local_experts=self.experts_per_rank,
                _debug_weight_lists=_debug_lists,
                _return_internals=_need_internals,
            )
            if _need_internals and isinstance(_result, tuple):
                _grouped_output, _sorted_output, _topk_pos, _expert_offsets = _result
                global_results[:num_global_tokens] = _grouped_output
            else:
                if isinstance(_result, tuple):
                    global_results[:num_global_tokens] = _result[0]
                else:
                    global_results[:num_global_tokens] = _result
        elif run_compare:
            # DIAGNOSTIC: stage-by-stage comparison of grouped vs single-expert
            _COMPARE_COUNT += 1
            torch.cuda.synchronize()

            from batchgen.moe.fused_wgmma_grouped import (
                fused_mxfp4_grouped_stage1, fused_mxfp4_grouped_stage2,
            )
            from batchgen.moe.fused_wgmma_expert import _load_mxfp4_module
            from batchgen.moe.routing import dispatch_count_gather_cuda, reduce_weighted_scatter_cuda

            mod_single = _load_mxfp4_module()

            # ─── 1. POINTER VALIDATION ───
            if _COMPARE_COUNT == 1:
                gw = self.gate_weight_ref
                gs = self.gate_scale_ref
                dw = self.down_weight_ref
                ds = self.down_scale_ref
                print(f"[COMPARE] EP rank {self.rank}: input={all_tokens.shape} "
                      f"experts=[{self.routed_expert_start_idx}, "
                      f"{self.routed_expert_start_idx + self.experts_per_rank}) "
                      f"topk={topk_indices.shape[1]}", flush=True)
                print(f"  gate_w={gw.shape} stride={gw.stride()} "
                      f"gate_s={gs.shape} stride={gs.stride()}", flush=True)
                print(f"  down_w={dw.shape} stride={dw.stride()} "
                      f"down_s={ds.shape} stride={ds.stride()}", flush=True)

                ptr_ok = True
                for e_local in range(min(3, self.experts_per_rank)):
                    global_e = self.routed_expert_start_idx + e_local
                    w = self.experts[global_e]
                    checks = [
                        ("gate_w", self.gate_ptrs, w.mxfp4_gate_packed),
                        ("gate_s", self.gate_scale_ptrs, w.mxfp4_gate_scales),
                        ("up_w", self.up_ptrs, w.mxfp4_up_packed),
                        ("up_s", self.up_scale_ptrs, w.mxfp4_up_scales),
                        ("down_w", self.down_ptrs, w.mxfp4_down_packed),
                        ("down_s", self.down_scale_ptrs, w.mxfp4_down_scales),
                    ]
                    for name, ptr_arr, tensor in checks:
                        ptr_val = ptr_arr[e_local].item()
                        expected = tensor.data_ptr()
                        if ptr_val != expected:
                            print(f"  PTR MISMATCH e{e_local} {name}: "
                                  f"ptr={ptr_val} expected={expected} "
                                  f"tensor.shape={tensor.shape}", flush=True)
                            ptr_ok = False
                if ptr_ok:
                    print(f"  All pointers OK ({min(3, self.experts_per_rank)} experts checked)",
                          flush=True)

            # ─── 2. DISPATCH ───
            dispatched_x, expert_counts, expert_offsets, topk_pos = \
                dispatch_count_gather_cuda(
                    all_tokens, topk_indices,
                    self.routed_expert_start_idx, self.experts_per_rank,
                )
            total = expert_offsets[self.experts_per_rank].item()
            dispatched_x = dispatched_x[:total]

            if total > 0:
                N_inter = self.gate_weight_ref.shape[0]
                hidden = dispatched_x.shape[1]
                s1_sw = self.gate_weight_ref.shape[1]
                s1_ss = self.gate_scale_ref.shape[1]
                s2_sw = self.down_weight_ref.shape[1]
                s2_ss = self.down_scale_ref.shape[1]
                offsets_cpu = expert_offsets.cpu()
                empty_bias = torch.empty(0, dtype=torch.bfloat16,
                                         device=dispatched_x.device)

                if _COMPARE_COUNT == 1:
                    print(f"  N_inter={N_inter} hidden={hidden} "
                          f"s1_stride_w={s1_sw} s1_stride_s={s1_ss} "
                          f"s2_stride_w={s2_sw} s2_stride_s={s2_ss}",
                          flush=True)

                # ─── 3. STAGE 1 COMPARISON ───
                grouped_s1 = fused_mxfp4_grouped_stage1(
                    dispatched_x, expert_offsets,
                    self.gate_ptrs, self.gate_scale_ptrs,
                    self.up_ptrs, self.up_scale_ptrs,
                    N_inter, s1_sw, s1_ss,
                )
                ref_s1 = torch.zeros_like(grouped_s1)
                for e_local in range(self.experts_per_rank):
                    s = offsets_cpu[e_local].item()
                    e = offsets_cpu[e_local + 1].item()
                    if e <= s:
                        continue
                    global_e = self.routed_expert_start_idx + e_local
                    w = self.experts[global_e]
                    tok = dispatched_x[s:e].contiguous()
                    ref_s1[s:e] = mod_single.mxfp4_moe_stage1(
                        tok,
                        w.mxfp4_gate_packed, w.mxfp4_gate_scales,
                        w.mxfp4_up_packed, w.mxfp4_up_scales,
                        empty_bias, empty_bias,
                    )
                torch.cuda.synchronize()
                s1d = (grouped_s1.float() - ref_s1.float()).abs()
                print(f"[S1 COMPARE #{_COMPARE_COUNT}] rank {self.rank}: "
                      f"max_diff={s1d.max():.6f} mean={s1d.mean():.6f}",
                      flush=True)
                print(f"  grouped: [{grouped_s1.float().min():.4f}, "
                      f"{grouped_s1.float().max():.4f}]  "
                      f"single: [{ref_s1.float().min():.4f}, "
                      f"{ref_s1.float().max():.4f}]", flush=True)

                # ─── 4. STAGE 2 COMPARISON (using grouped S1 as input) ───
                grouped_s2 = fused_mxfp4_grouped_stage2(
                    grouped_s1, expert_offsets,
                    self.down_ptrs, self.down_scale_ptrs,
                    hidden, s2_sw, s2_ss,
                )
                ref_s2 = torch.zeros_like(grouped_s2)
                for e_local in range(self.experts_per_rank):
                    s = offsets_cpu[e_local].item()
                    e = offsets_cpu[e_local + 1].item()
                    if e <= s:
                        continue
                    global_e = self.routed_expert_start_idx + e_local
                    w = self.experts[global_e]
                    tok = grouped_s1[s:e].contiguous()
                    ref_s2[s:e] = mod_single.mxfp4_moe_stage2(
                        tok,
                        w.mxfp4_down_packed, w.mxfp4_down_scales,
                        empty_bias,
                    )
                torch.cuda.synchronize()
                s2d = (grouped_s2.float() - ref_s2.float()).abs()
                print(f"[S2 COMPARE #{_COMPARE_COUNT}] rank {self.rank}: "
                      f"max_diff={s2d.max():.6f} mean={s2d.mean():.6f}",
                      flush=True)
                print(f"  grouped: [{grouped_s2.float().min():.4f}, "
                      f"{grouped_s2.float().max():.4f}]  "
                      f"single: [{ref_s2.float().min():.4f}, "
                      f"{ref_s2.float().max():.4f}]", flush=True)

                # ─── 5. REDUCE + use reference output ───
                output = reduce_weighted_scatter_cuda(
                    ref_s2, topk_pos, topk_weights,
                    num_global_tokens, hidden, topk_indices.shape[1],
                )
                global_results[:num_global_tokens] = output
            else:
                pass  # no tokens dispatched
        else:
            # Per-expert loop fallback
            # Flat view of expert assignments
            flat_expert_idx = topk_indices.view(-1)  # [global_tokens * K]

            # Use pre-allocated index tensors
            token_indices = self.token_idx_buffer
            topk_positions = self.topk_pos_buffer

            # Pre-compute expert token counts ONCE to avoid per-expert .any() sync
            # This reduces GPU→CPU syncs from 32 per layer to 1 per layer
            expert_counts = self.expert_counts_buffer
            expert_counts.zero_()
            expert_counts.scatter_add_(
                0, flat_expert_idx.to(torch.int64),
                torch.ones_like(flat_expert_idx, dtype=torch.int32)
            )
            # Single CPU transfer for all counts
            expert_counts_cpu = expert_counts.cpu()

            # Loop through local experts only
            for local_e in range(self.experts_per_rank):
                global_e = self.routed_expert_start_idx + local_e

                # Check token count without GPU sync (already on CPU)
                if expert_counts_cpu[global_e].item() == 0:
                    continue

                # Find tokens routed to this expert
                mask = flat_expert_idx == global_e

                # Get token indices and topk positions for this expert
                expert_token_idx = token_indices[mask]
                expert_topk_pos = topk_positions[mask]

                # Gather tokens for this expert
                tokens_for_expert = all_tokens[expert_token_idx]

                # Call expert forward (wrapper handles MXFP4 dequant)
                expert = self.experts[global_e]
                if expert is None:
                    continue  # Expert not loaded on this rank (shouldn't happen)

                expert_output = expert(tokens_for_expert)

                # Get weights for these tokens at these topk positions
                expert_weights = topk_weights[expert_token_idx, expert_topk_pos]

                # Weighted accumulation into results
                weighted_output = (expert_output * expert_weights.unsqueeze(-1)).to(global_results.dtype)
                global_results.index_add_(0, expert_token_idx, weighted_output)

        # ── Full-pipeline comparison: grouped reduce vs same-kernel index_add (pre-AllReduce) ──
        if (_FULL_COMPARE and self._use_grouped_wgmma
                and _FULL_COMPARE_COUNT < _FULL_COMPARE_MAX):
            _FULL_COMPARE_COUNT += 1
            torch.cuda.synchronize()

            hidden = all_tokens.shape[1]

            # ── Method A: Re-reduce using per-expert-loop-style accumulation ──
            # Uses SAME sorted_output from grouped kernel (exact same data),
            # but accumulates per-expert using mask + index_add_ (like the for-loop path)
            ref_from_sorted = torch.zeros(num_global_tokens, hidden,
                                          dtype=torch.bfloat16, device=self.device)

            # _sorted_output was set above when _need_internals=True
            has_sorted_output = ('_sorted_output' in dir() or
                                 '_sorted_output' in locals())
            try:
                _sorted_output_check = _sorted_output
                has_sorted_output = _sorted_output_check is not None
            except NameError:
                has_sorted_output = False

            if has_sorted_output:
                # Per-expert accumulation using sorted_output + topk_pos
                _tpos_2d = _topk_pos.view(num_global_tokens, K)
                for k_slot in range(K):
                    valid = _tpos_2d[:, k_slot] >= 0
                    if valid.any():
                        pos = _tpos_2d[valid, k_slot].long()
                        expert_out = _sorted_output[pos]
                        wt = topk_weights[valid, k_slot:k_slot+1]
                        weighted = (expert_out.float() * wt).to(torch.bfloat16)
                        valid_idx = torch.where(valid)[0]
                        ref_from_sorted.index_add_(0, valid_idx, weighted)
                has_sorted = True
            else:
                has_sorted = False
                _sorted_output = None  # ensure defined

            # ── Method B: Per-expert loop with expert.forward() (original reference) ──
            ref_from_loop = torch.zeros(num_global_tokens, hidden,
                                        dtype=torch.bfloat16, device=self.device)
            flat_expert_idx = topk_indices.view(-1)
            _token_indices = self.token_idx_buffer
            _topk_positions_buf = self.topk_pos_buffer

            # Also collect per-expert RAW output comparison (Method D)
            _d_per_expert = []  # (global_e, M, max_diff_raw)

            for local_e in range(self.experts_per_rank):
                global_e = self.routed_expert_start_idx + local_e
                mask = flat_expert_idx == global_e
                if not mask.any():
                    continue
                expert_token_idx = _token_indices[mask]
                expert_topk_pos = _topk_positions_buf[mask]
                tokens_for_expert = all_tokens[expert_token_idx]

                expert = self.experts[global_e]
                if expert is None:
                    continue
                expert_output = expert(tokens_for_expert)
                expert_weights_val = topk_weights[expert_token_idx, expert_topk_pos]
                weighted_output = (expert_output * expert_weights_val.unsqueeze(-1)).to(torch.bfloat16)
                ref_from_loop.index_add_(0, expert_token_idx, weighted_output)

                # Method D: Compare RAW expert output with sorted_output
                if has_sorted:
                    M_expert = expert_output.shape[0]
                    # Get sorted_output positions for these tokens+slots
                    flat_pos = expert_token_idx * K + expert_topk_pos
                    sorted_pos = _topk_pos[flat_pos].long()
                    valid_pos = sorted_pos >= 0
                    if valid_pos.any():
                        sorted_vals = _sorted_output[sorted_pos[valid_pos]]
                        expert_vals = expert_output[valid_pos]
                        raw_diff = (sorted_vals.float() - expert_vals.float()).abs()
                        max_raw = raw_diff.max().item()
                        mean_raw = raw_diff.mean().item()
                        _d_per_expert.append((global_e, M_expert, max_raw, mean_raw))
                        if max_raw > 0.01:
                            # Show details for first few bad tokens
                            per_token_raw = raw_diff.max(dim=1).values
                            _, worst_raw = per_token_raw.topk(min(3, per_token_raw.shape[0]))
                            for idx in worst_raw:
                                i = idx.item()
                                pos_i = sorted_pos[valid_pos][i].item()
                                print(f"  [RAW D] expert {global_e} local_token {i}: "
                                      f"sorted_output[{pos_i}] range="
                                      f"[{sorted_vals[i].float().min():.4f},{sorted_vals[i].float().max():.4f}] "
                                      f"expert_output range="
                                      f"[{expert_vals[i].float().min():.4f},{expert_vals[i].float().max():.4f}] "
                                      f"max_diff={per_token_raw[i]:.6f}", flush=True)

            torch.cuda.synchronize()

            grouped_slice = global_results[:num_global_tokens]

            # Compare A: grouped reduce vs index_add on SAME sorted_output
            if has_sorted:
                diff_A = (grouped_slice.float() - ref_from_sorted.float()).abs()
                max_A = diff_A.max().item()
                mean_A = diff_A.mean().item()
                bad_A = (diff_A.max(dim=1).values > 0.01).sum().item()
            else:
                max_A = -1.0
                mean_A = -1.0
                bad_A = -1

            # Compare B: grouped reduce vs expert.forward() loop
            diff_B = (grouped_slice.float() - ref_from_loop.float()).abs()
            max_B = diff_B.max().item()
            mean_B = diff_B.mean().item()
            bad_B = (diff_B.max(dim=1).values > 0.01).sum().item()

            # Compare C: index_add on sorted_output vs expert.forward() loop
            if has_sorted:
                diff_C = (ref_from_sorted.float() - ref_from_loop.float()).abs()
                max_C = diff_C.max().item()
                mean_C = diff_C.mean().item()
                bad_C = (diff_C.max(dim=1).values > 0.01).sum().item()
            else:
                max_C = -1.0
                mean_C = -1.0
                bad_C = -1

            g_nonzero = (grouped_slice != 0).any(dim=1).sum().item()

            print(f"\n[FULL COMPARE #{_FULL_COMPARE_COUNT}] rank {self.rank} "
                  f"(expert_start={self.routed_expert_start_idx}):", flush=True)
            print(f"  A) grouped_reduce vs index_add(sorted_output): "
                  f"max={max_A:.6f} mean={mean_A:.6f} bad={bad_A}", flush=True)
            print(f"  B) grouped_reduce vs expert.forward() loop:    "
                  f"max={max_B:.6f} mean={mean_B:.6f} bad={bad_B}", flush=True)
            print(f"  C) index_add(sorted_output) vs expert.forward(): "
                  f"max={max_C:.6f} mean={mean_C:.6f} bad={bad_C}", flush=True)
            print(f"  grouped: nonzero={g_nonzero} "
                  f"range=[{grouped_slice.float().min():.4f}, {grouped_slice.float().max():.4f}]",
                  flush=True)

            if max_B > 0.01:
                per_token = diff_B.max(dim=1).values
                _, worst = per_token.topk(min(3, num_global_tokens))
                for idx in worst:
                    i = idx.item()
                    print(f"  worst token {i}: B_diff={per_token[i]:.4f} "
                          f"grouped=[{grouped_slice[i].float().min():.4f},{grouped_slice[i].float().max():.4f}] "
                          f"loop=[{ref_from_loop[i].float().min():.4f},{ref_from_loop[i].float().max():.4f}] "
                          f"experts={topk_indices[i].cpu().tolist()}", flush=True)

            # Method D summary: per-expert RAW output comparison
            if _d_per_expert:
                d_max_overall = max(d[2] for d in _d_per_expert)
                d_bad = sum(1 for d in _d_per_expert if d[2] > 0.01)
                print(f"  D) per-expert RAW sorted_output vs expert.forward(): "
                      f"overall_max={d_max_overall:.6f} bad_experts={d_bad}/{len(_d_per_expert)}",
                      flush=True)
                for ge, m, mx, mn in _d_per_expert:
                    if mx > 0.01 or len(_d_per_expert) <= 5:
                        print(f"    expert {ge}: M={m} max_raw_diff={mx:.6f} mean={mn:.6f}",
                              flush=True)
            print(flush=True)

        # ---- 4) AllReduce: Combine results from all ranks ----
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                global_results,
                op=dist.ReduceOp.SUM,
                stream=torch.cuda.default_stream(self.device)
            )

        # ---- 5) Extract results for local tokens ----
        start_token_idx = self.rank * self.num_tokens_per_rank
        end_token_idx = start_token_idx + num_tokens

        return global_results[start_token_idx:end_token_idx].to(x.dtype)


# ============================================================================
# Prefill MoE Layer (MXFP4 with CuTe Dequant + torch.matmul)
# ============================================================================

class GptOssMoEPrefill(nn.Module):
    """MoE layer for prefill using CuTe dequant + per-expert torch.matmul.

    This implements sequential per-expert processing:
    1. For each activated expert:
       a. CuTe dequant gate weight → BF16 buffer
       b. torch.matmul(tokens, gate_weight.T) → gate output
       c. CuTe dequant up weight → BF16 buffer
       d. torch.matmul(tokens, up_weight.T) → up output
       e. SwiGLU(gate, up) → intermediate
       f. CuTe dequant down weight → BF16 buffer
       g. torch.matmul(intermediate, down_weight.T) → expert output
       h. Accumulate weighted output

    Memory usage:
    - Single-expert BF16 buffer: ~142 MB (reused across projections)
      - gate/up: [intermediate_size, hidden_size] = [13824, 5120] × 2 bytes = 141.6 MB
      - down: [hidden_size, intermediate_size] = [5120, 13824] × 2 bytes = 141.6 MB

    Note: This class requires K-major scales layout [K//32, N] for the CuTe kernel.
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Router (same as original)
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=True)

        # MXFP4 quantized weights - stored as lists of tensors (one per expert)
        # Packed weights: [N, K//2] uint8
        # N-major scales: [N, K//32] uint8 (same as decode path)
        self.gate_weights = None  # List[Tensor[intermediate_size, hidden_size//2]]
        self.gate_scales = None   # List[Tensor[intermediate_size, hidden_size//32]]
        self.up_weights = None
        self.up_scales = None
        self.down_weights = None  # List[Tensor[hidden_size, intermediate_size//2]]
        self.down_scales = None   # List[Tensor[intermediate_size//32, hidden_size]]

        # Optional biases
        self.gate_biases = None  # [num_experts, intermediate_size]
        self.up_biases = None
        self.down_biases = None

        # Reusable BF16 buffer for dequantized weights (allocated on first forward)
        # Size: max(gate, up, down) = max(13824×5120, 5120×13824) = 141.6 MB
        self._bf16_buffer = None
        self._buffer_shape = None

        # SwiGLU parameters
        self.swiglu_alpha = 1.702
        self.swiglu_limit = getattr(config, 'swiglu_limit', 7.0)

    def _get_bf16_buffer(self, shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """Get or allocate BF16 buffer for dequantized weights."""
        if self._bf16_buffer is None or self._buffer_shape != shape or self._bf16_buffer.device != device:
            self._bf16_buffer = torch.empty(shape, dtype=torch.bfloat16, device=device)
            self._buffer_shape = shape
        return self._bf16_buffer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass with fused WGMMA kernels (or CuTe fallback)."""
        import os
        from batchgen.moe.fused_wgmma_expert import (
            fused_mxfp4_expert_forward_from_dict,
            is_wgmma_available,
        )

        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)  # [total_tokens, hidden_size]

        # Compute routing logits
        router_logits = self.router(hidden_flat)  # [total_tokens, num_experts]

        # Select Top-K experts per token
        if _HAS_CUDA_ROUTING:
            topk_indices, topk_weights = gate_topk_softmax_cuda(
                router_logits, k=self.num_experts_per_tok
            )

        else:
            topk_weights, topk_indices = torch.topk(
                router_logits, k=self.num_experts_per_tok, dim=-1
            )  # [total_tokens, top_k]
            topk_weights = F.softmax(topk_weights, dim=-1)

        # Initialize output
        output = torch.zeros_like(hidden_flat)

        # Find all unique experts that are activated
        unique_experts = topk_indices.unique().tolist()

        # Check if we can use fused WGMMA kernels
        # BATCHGEN_PREFILL_USE_CUTE=1 forces CuTe fallback for debugging
        use_fused = (
            is_wgmma_available()
            and os.environ.get("BATCHGEN_PREFILL_USE_CUTE", "0") != "1"
        )

        # Debug: compare fused vs CuTe for first expert in first layer
        debug_prefill = os.environ.get("BATCHGEN_DEBUG_PREFILL", "0") == "1"

        # Process each activated expert
        for expert_idx in unique_experts:
            # Find tokens routed to this expert
            expert_mask = (topk_indices == expert_idx).any(dim=-1)  # [total_tokens]
            if not expert_mask.any():
                continue

            expert_input = hidden_flat[expert_mask].contiguous()  # [num_tokens, hidden_size]

            # Get routing weight for this expert
            expert_weights = torch.where(
                topk_indices[expert_mask] == expert_idx,
                topk_weights[expert_mask],
                torch.zeros_like(topk_weights[expert_mask])
            ).sum(dim=-1)  # [num_tokens]

            if use_fused:
                # === Fused WGMMA path ===
                # Build weights dict (same format as decode path)
                weights = {
                    "gate_proj.weight": self.gate_weights[expert_idx],
                    "gate_proj.weight_scales": self.gate_scales[expert_idx],
                    "up_proj.weight": self.up_weights[expert_idx],
                    "up_proj.weight_scales": self.up_scales[expert_idx],
                    "down_proj.weight": self.down_weights[expert_idx],
                    "down_proj.weight_scales": self.down_scales[expert_idx],
                }
                if self.gate_biases is not None:
                    weights["gate_proj.bias"] = self.gate_biases[expert_idx]
                if self.up_biases is not None:
                    weights["up_proj.bias"] = self.up_biases[expert_idx]
                if self.down_biases is not None:
                    weights["down_proj.bias"] = self.down_biases[expert_idx]

                expert_output = fused_mxfp4_expert_forward_from_dict(expert_input, weights)

                # Debug: detect NaN in fused output (first 3 experts only)
                if debug_prefill and expert_idx in unique_experts[:3]:
                    with torch.no_grad():
                        has_nan = expert_output.isnan().any().item()
                        has_inf = expert_output.isinf().any().item()
                        input_has_nan = expert_input.isnan().any().item()
                        if has_nan or has_inf:
                            # Find first NaN/Inf position
                            nan_mask = expert_output.isnan() | expert_output.isinf()
                            flat_idx = nan_mask.view(-1).int().argmax().item()
                            row = flat_idx // expert_output.shape[-1]
                            col = flat_idx % expert_output.shape[-1]
                            print(f"[PREFILL DEBUG] Expert {expert_idx}: M={expert_input.shape[0]} - "
                                  f"{'NaN' if has_nan else 'Inf'} at [{row}, {col}]")
                            if not input_has_nan:
                                print(f"  Input VALID: std={expert_input.float().std().item():.4f}, "
                                      f"max={expert_input.abs().max().item():.4f}")
                            else:
                                print(f"  Input has NaN (propagated from previous layer)")
                            print(f"  col%32={col%32}, col%64={col%64} (scale/tile boundary check)")
                            # Check total NaN count
                            nan_count = nan_mask.sum().item()
                            total = expert_output.numel()
                            print(f"  NaN count: {nan_count}/{total} ({100*nan_count/total:.1f}%)")
                            # Check scale values for this expert
                            gate_s = weights['gate_proj.weight_scales']
                            print(f"  Gate scales: min={gate_s.min().item()}, max={gate_s.max().item()}, "
                                  f"count>=250: {(gate_s >= 250).sum().item()}")
            else:
                # === CuTe fallback path ===
                expert_output = self._forward_expert_cute(expert_input, expert_idx)

            # Accumulate weighted output
            output[expert_mask] += expert_output * expert_weights.unsqueeze(-1)

        return output.view(batch_size, seq_len, hidden_dim)

    def _forward_expert_cute(self, expert_input: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Fallback path using CuTe dequant + torch.matmul."""
        from batchgen.moe.cute_mxfp4_dequant import mxfp4_dequant_single_expert_cute

        num_tokens = expert_input.shape[0]

        # Scales are stored N-major [N, K//32], CuTe needs K-major [K//32, N]
        gate_scales_kmajor = self.gate_scales[expert_idx].T.contiguous()
        up_scales_kmajor = self.up_scales[expert_idx].T.contiguous()
        down_scales_kmajor = self.down_scales[expert_idx].T.contiguous()

        # === Gate projection ===
        gate_buffer = self._get_bf16_buffer(
            (self.intermediate_size, self.hidden_size),
            expert_input.device
        )
        mxfp4_dequant_single_expert_cute(
            self.gate_weights[expert_idx],
            gate_scales_kmajor,
            gate_buffer,
        )
        gate_out = torch.matmul(expert_input, gate_buffer.T)
        if self.gate_biases is not None:
            gate_out = gate_out + self.gate_biases[expert_idx]

        # === Up projection ===
        up_buffer = self._get_bf16_buffer(
            (self.intermediate_size, self.hidden_size),
            expert_input.device
        )
        mxfp4_dequant_single_expert_cute(
            self.up_weights[expert_idx],
            up_scales_kmajor,
            up_buffer,
        )
        up_out = torch.matmul(expert_input, up_buffer.T)
        if self.up_biases is not None:
            up_out = up_out + self.up_biases[expert_idx]

        # === SwiGLU activation ===
        interleaved = torch.stack([gate_out, up_out], dim=-1).view(num_tokens, -1)
        intermediate = swiglu(interleaved, alpha=self.swiglu_alpha, limit=self.swiglu_limit)

        # === Down projection ===
        down_buffer = self._get_bf16_buffer(
            (self.hidden_size, self.intermediate_size),
            expert_input.device
        )
        mxfp4_dequant_single_expert_cute(
            self.down_weights[expert_idx],
            down_scales_kmajor,
            down_buffer,
        )
        down_out = torch.matmul(intermediate, down_buffer.T)
        if self.down_biases is not None:
            down_out = down_out + self.down_biases[expert_idx]

        return down_out


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
        debug_layer = os.environ.get("BATCHGEN_DEBUG_LAYER", "0") == "1"
        timing_enabled = DecodeLayerTiming.enabled

        # Start layer timing
        DecodeLayerTiming.start_layer(self.layer_idx)

        residual = hidden_states

        # Pre-norm + attention
        hidden_states = self.input_layernorm(hidden_states)

        if debug_layer and self.layer_idx < 3:
            with torch.no_grad():
                print(f"[L{self.layer_idx}] after input_layernorm: std={hidden_states.float().std().item():.4f}, max={hidden_states.abs().max().item():.4f}")

        # ========== ATTENTION TIMING ==========
        if timing_enabled:
            torch.cuda.synchronize()
            attn_start = time.perf_counter()

        hidden_states, attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        if timing_enabled:
            torch.cuda.synchronize()
            DecodeLayerTiming.record_attn((time.perf_counter() - attn_start) * 1000)

        if debug_layer and self.layer_idx < 3:
            with torch.no_grad():
                print(f"[L{self.layer_idx}] after attention (before residual): std={hidden_states.float().std().item():.4f}, max={hidden_states.abs().max().item():.4f}")

        hidden_states = residual + hidden_states

        if debug_layer and self.layer_idx < 3:
            with torch.no_grad():
                print(f"[L{self.layer_idx}] after attn+residual: std={hidden_states.float().std().item():.4f}, max={hidden_states.abs().max().item():.4f}")

        # Pre-norm + MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if debug_layer and self.layer_idx < 3:
            with torch.no_grad():
                print(f"[L{self.layer_idx}] after post_attn_layernorm: std={hidden_states.float().std().item():.4f}, max={hidden_states.abs().max().item():.4f}")

        # ========== MoE TIMING ==========
        if timing_enabled:
            torch.cuda.synchronize()
            moe_start = time.perf_counter()

        hidden_states = self.mlp(hidden_states)

        if timing_enabled:
            torch.cuda.synchronize()
            DecodeLayerTiming.record_moe((time.perf_counter() - moe_start) * 1000)

        if debug_layer and self.layer_idx < 3:
            with torch.no_grad():
                print(f"[L{self.layer_idx}] after MLP (before residual): std={hidden_states.float().std().item():.4f}, max={hidden_states.abs().max().item():.4f}")

        hidden_states = residual + hidden_states

        if debug_layer and self.layer_idx < 3:
            with torch.no_grad():
                print(f"[L{self.layer_idx}] after MLP+residual (final): std={hidden_states.float().std().item():.4f}, max={hidden_states.abs().max().item():.4f}")

        # End layer timing
        DecodeLayerTiming.end_layer()

        return hidden_states, attn_weights, present_key_value


# ============================================================================
# Main Model
# ============================================================================

class GptOssModel(nn.Module):
    """GPT-OSS-120B transformer model (OpenAI-style, no HuggingFace dependencies).

    Note: Due to BatchGen's config_torch_module_initializer() memory optimization,
    parameters are initially created with placeholder shape [1]. The actual weights
    are loaded later via Parallel_Strategy_Manager._load_model_skeleton() using
    direct assignment (param.data = tensor).
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

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

            # Debug: Track hidden state statistics per layer
            if os.environ.get("BATCHGEN_DEBUG_HIDDEN", "0") == "1":
                with torch.no_grad():
                    h_min = hidden_states.min().item()
                    h_max = hidden_states.max().item()
                    h_mean = hidden_states.float().mean().item()
                    h_std = hidden_states.float().std().item()
                    has_nan = torch.isnan(hidden_states).any().item()
                    has_inf = torch.isinf(hidden_states).any().item()
                    print(f"[Layer {idx}] hidden: min={h_min:.4f}, max={h_max:.4f}, mean={h_mean:.6f}, std={h_std:.4f}, nan={has_nan}, inf={has_inf}")

                    # Check sequence convergence at layers 0, 17, 35 (first, middle, last)
                    if idx in [0, 17, 35] and hidden_states.shape[0] >= 3:
                        # Compare first 3 sequences at last position (for decode: seq_len=1)
                        h0 = hidden_states[0, -1, :8].tolist()  # seq 0, last pos, first 8 dims
                        h1 = hidden_states[1, -1, :8].tolist()  # seq 1
                        h2 = hidden_states[2, -1, :8].tolist()  # seq 2
                        print(f"[Layer {idx}] seq0[:8]: {[f'{v:.2f}' for v in h0]}")
                        print(f"[Layer {idx}] seq1[:8]: {[f'{v:.2f}' for v in h1]}")
                        print(f"[Layer {idx}] seq2[:8]: {[f'{v:.2f}' for v in h2]}")
                        # Check if sequences are identical
                        diff_01 = (hidden_states[0, -1] - hidden_states[1, -1]).abs().max().item()
                        diff_02 = (hidden_states[0, -1] - hidden_states[2, -1]).abs().max().item()
                        print(f"[Layer {idx}] max_diff: seq0-seq1={diff_01:.4f}, seq0-seq2={diff_02:.4f}")

            if use_cache:
                next_cache += (layer_outputs[2],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # Debug: Check normalized hidden states before lm_head
        if os.environ.get("BATCHGEN_DEBUG_HIDDEN", "0") == "1":
            with torch.no_grad():
                h_std = hidden_states.float().std().item()
                h_max = hidden_states.abs().max().item()
                print(f"[After final norm] std={h_std:.4f}, max_abs={h_max:.4f}")
                if hidden_states.shape[0] >= 3:
                    h0 = hidden_states[0, -1, :8].tolist()
                    h1 = hidden_states[1, -1, :8].tolist()
                    h2 = hidden_states[2, -1, :8].tolist()
                    print(f"[After norm] seq0[:8]: {[f'{v:.4f}' for v in h0]}")
                    print(f"[After norm] seq1[:8]: {[f'{v:.4f}' for v in h1]}")
                    print(f"[After norm] seq2[:8]: {[f'{v:.4f}' for v in h2]}")
                    diff_01 = (hidden_states[0, -1] - hidden_states[1, -1]).abs().max().item()
                    diff_02 = (hidden_states[0, -1] - hidden_states[2, -1]).abs().max().item()
                    print(f"[After norm] max_diff: seq0-seq1={diff_01:.6f}, seq0-seq2={diff_02:.6f}")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Print timing summary if enabled
        DecodeLayerTiming.print_summary()

        return (hidden_states, next_cache, all_hidden_states, all_self_attns)


class GptOss(nn.Module):
    """GPT-OSS-120B model with language modeling head.

    Instantiate with BatchGen config:
        model = GptOss(config)

    Note: Due to BatchGen's config_torch_module_initializer() memory optimization,
    parameters are initially created with placeholder shape [1]. The actual weights
    are loaded later via Parallel_Strategy_Manager._load_model_skeleton().
    """

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        self.model = GptOssModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor, ...], CausalLMOutputWithPast]:
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

        # Debug logging for logits analysis
        if os.environ.get("BATCHGEN_DEBUG_LOGITS", "0") == "1":
            with torch.no_grad():
                # Get stats for last token position (for autoregressive generation)
                last_logits = logits[:, -1, :]  # [batch, vocab_size]
                top_vals, top_ids = torch.topk(last_logits, k=10, dim=-1)
                print(f"\n[LOGITS DEBUG] Shape: {logits.shape}")
                print(f"[LOGITS DEBUG] Last token logits: min={last_logits.min():.4f}, max={last_logits.max():.4f}, mean={last_logits.mean():.4f}")
                print(f"[LOGITS DEBUG] Top-10 token IDs: {top_ids[0].tolist()}")
                print(f"[LOGITS DEBUG] Top-10 logit values: {[f'{v:.4f}' for v in top_vals[0].tolist()]}")
                # Check for common newline tokens
                print(f"[LOGITS DEBUG] Token 10 (LF) logit: {last_logits[0, 10].item():.4f}")
                print(f"[LOGITS DEBUG] Token 13 (CR) logit: {last_logits[0, 13].item():.4f}")
                # Check if logits are uniform (std close to 0)
                std = last_logits.std().item()
                print(f"[LOGITS DEBUG] Logits std: {std:.4f} (uniform if ~0)")

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
            past_key_values=outputs[1],
            hidden_states=outputs[2],
            attentions=outputs[3],
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
