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
# Default: enabled on SM90+ (Hopper). Use BATCHGEN_DISABLE_WGMMA_GROUPED=1 to force-disable.
try:
    from batchgen.moe.fused_wgmma_grouped import (
        fused_mxfp4_grouped_moe_forward_cuda_routing,
        is_grouped_wgmma_available,
    )
    if os.environ.get("BATCHGEN_DISABLE_WGMMA_GROUPED", "0") == "1":
        _HAS_WGMMA_GROUPED = False
        print("[WGMMA grouped] disabled via BATCHGEN_DISABLE_WGMMA_GROUPED", flush=True)
    else:
        _HAS_WGMMA_GROUPED = is_grouped_wgmma_available()
        if _HAS_WGMMA_GROUPED:
            print("[WGMMA grouped] enabled (SM90 detected)", flush=True)
        else:
            print("[WGMMA grouped] not available (SM90 required)", flush=True)
except Exception as e:
    import traceback
    _HAS_WGMMA_GROUPED = False
    print(f"[WGMMA grouped] failed to load: {e}", flush=True)
    traceback.print_exc()

_WGMMA_GROUPED_LOGGED = False  # one-time invocation log


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
        from batchgen.attention.fused_kernels import cuda_rmsnorm
        return cuda_rmsnorm(x, self.weight, self.eps)


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

        # Projections — packed QKV: single GEMM instead of 3
        self.q_size = self.num_heads * self.head_dim        # 4096
        self.kv_size = self.num_kv_heads * self.head_dim    # 512
        self.qkv_proj = nn.Linear(
            self.hidden_size,
            self.q_size + 2 * self.kv_size,  # 5120
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # Attention scale
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # RoPE — shared instance assigned by GptOssModel.__init__()
        self.rotary_emb = None

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

        # Project Q, K, V (packed QKV)
        qkv = self.qkv_proj(hidden_states)
        from batchgen.attention.fused_kernels import cuda_qkv_split
        query_states, key_states, value_states = cuda_qkv_split(qkv, self.q_size, self.kv_size)

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

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
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

class _ExpertPlaceholder:
    """Lightweight placeholder for expert slots in GptOssMoE.experts.

    Avoids creating 4608 nn.Module objects during model init. Supports arbitrary
    attribute setting (used by _load_expert_module to attach mxfp4_* tensors).
    Replaced by GptOssExpertWrapper during _config_expert_module().
    """
    pass


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

        # Experts — use plain list of lightweight placeholders instead of nn.ModuleList
        # to avoid creating 4608 nn.Module objects (128 experts × 36 layers).
        # These are replaced by GptOssExpertWrapper during _config_expert_module().
        self.experts = [_ExpertPlaceholder() for _ in range(self.num_experts)]

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
# Unified Decode MoE Layer
# ============================================================================

class GptOssMoEDecode(nn.Module):
    """Unified decode MoE for GPT-OSS-120B.

    Handles both single-GPU and EP (Expert Parallelism) modes with a single
    forward path:
      1. [EP only] AllGather tokens from all ranks
      2. Router → topk_indices, topk_weights
      3. Grouped kernel for persistent experts (weights in VRAM)
      4. Single-expert kernel for non-persistent experts one-by-one (if any)
      5. [EP only] AllReduce results

    Supports two weight formats:
      - "mxfp4": MXFP4 quantized weights with fused WGMMA kernels
      - "bf16": Pre-dequantized BF16 weights with grouped BF16 GEMM

    The model instance does not need to know about ep_with_offloading or
    pre_dequantize_weights flags. The Parallel_Strategy_Manager configures
    persistent/non-persistent expert lists and weight format at setup time.
    """

    def __init__(self, config: GptOssConfig, ep_enabled: bool = False, comm=None):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size

        # EP configuration
        self.ep_enabled = ep_enabled
        self.comm = comm

        if ep_enabled:
            import torch.distributed as dist
            if not dist.is_initialized():
                self.rank, self.world_size = 0, 1
            else:
                self.rank = dist.get_rank()
                self.world_size = dist.get_world_size()
        else:
            self.rank, self.world_size = 0, 1

        # Expert topology (set by Parallel_Strategy_Manager)
        self.total_experts = config.num_local_experts  # 128
        self.expert_start = 0  # first local expert global index
        self.num_local_experts = config.num_local_experts  # experts on this rank

        # Persistent / non-persistent expert indices (set by PSM)
        self.persistent_expert_indices = []   # global indices, weights in VRAM
        self.non_persistent_expert_indices = []  # global indices, loaded on-demand

        # Weight format: "mxfp4" or "bf16"
        self.weight_format = "mxfp4"

        # Router
        self.router = nn.Linear(config.hidden_size, self.total_experts, bias=True)

        # Expert wrappers for non-persistent experts (single-expert forward)
        self.experts = nn.ModuleList([None] * self.total_experts)

        # ---- MXFP4 grouped kernel pointer arrays (persistent experts) ----
        self.gate_ptrs = None
        self.gate_scale_ptrs = None
        self.up_ptrs = None
        self.up_scale_ptrs = None
        self.down_ptrs = None
        self.down_scale_ptrs = None
        self.gate_bias_ptrs = None
        self.up_bias_ptrs = None
        self.down_bias_ptrs = None
        # Reference tensors for stride computation
        self.gate_weight_ref = None
        self.gate_scale_ref = None
        self.down_weight_ref = None
        self.down_scale_ref = None

        # ---- MXFP4 per-expert weight lists (non-persistent experts) ----
        self.gate_weights = None  # List[Tensor] indexed by global expert idx
        self.gate_scales = None
        self.up_weights = None
        self.up_scales = None
        self.down_weights = None
        self.down_scales = None
        self.gate_biases = None
        self.up_biases = None
        self.down_biases = None

        # ---- BF16 grouped kernel pointer arrays (placeholder) ----
        # TODO: Set up when grouped BF16 kernel is ported
        self.bf16_gate_ptrs = None
        self.bf16_up_ptrs = None
        self.bf16_down_ptrs = None

        # ---- EP buffers (allocated lazily via init_num_tokens) ----
        self.num_tokens_per_rank = None
        self.device = torch.device("cuda", self.rank % torch.cuda.device_count()) if ep_enabled else None

    def init_num_tokens(self, num_tokens_per_rank: int):
        """Initialize EP communication buffers. Only needed when ep_enabled=True."""
        if not self.ep_enabled:
            return

        self.num_tokens_per_rank = num_tokens_per_rank
        global_num_tokens = num_tokens_per_rank * self.world_size
        K = self.num_experts_per_tok
        hidden_size = self.hidden_size

        self.token_idx_buffer = torch.arange(
            global_num_tokens, dtype=torch.int64, device=self.device
        ).repeat_interleave(K)
        self.topk_pos_buffer = torch.arange(
            K, dtype=torch.int64, device=self.device
        ).repeat(global_num_tokens)

        self.all_tokens_buffer = torch.zeros(
            (global_num_tokens, hidden_size), device=self.device, dtype=torch.bfloat16
        )
        self.padded_hidden_buffer = torch.zeros(
            (num_tokens_per_rank, hidden_size), device=self.device, dtype=torch.bfloat16
        )
        self.global_results_buffer = torch.zeros(
            (global_num_tokens, hidden_size), device=self.device, dtype=torch.bfloat16
        )

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int):
        """Update num_tokens_per_rank for dynamic batch size."""
        if num_tokens_per_rank == self.num_tokens_per_rank:
            return
        self.init_num_tokens(num_tokens_per_rank)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        if len(orig_shape) == 3:
            hidden_states = hidden_states.view(-1, orig_shape[-1])

        if self.ep_enabled:
            out = self._forward_ep(hidden_states)
        else:
            out = self._forward_local(hidden_states)

        return out.view(*orig_shape)

    def _forward_local(self, hidden_flat: torch.Tensor) -> torch.Tensor:
        """Single-GPU forward: route → grouped persistent → single non-persistent."""
        # Route
        router_logits = self.router(hidden_flat)
        if _HAS_CUDA_ROUTING:
            topk_indices, topk_weights = gate_topk_softmax_cuda(
                router_logits, k=self.num_experts_per_tok
            )
        else:
            topk_weights, topk_indices = torch.topk(router_logits, k=self.num_experts_per_tok, dim=-1)
            topk_weights = F.softmax(topk_weights, dim=-1)

        output = torch.zeros_like(hidden_flat)

        # Phase 1: Grouped kernel for persistent experts
        num_persistent = len(self.persistent_expert_indices)
        if num_persistent > 0:
            output = self._grouped_forward(
                hidden_flat, topk_indices, topk_weights,
                expert_start=self.expert_start,
                num_local_experts=num_persistent,
            )

        # Phase 2: Single-expert kernel for non-persistent experts
        if self.non_persistent_expert_indices:
            self._single_expert_forward(
                hidden_flat, topk_indices, topk_weights, output,
            )

        return output

    @torch.inference_mode()
    def _forward_ep(self, x: torch.Tensor) -> torch.Tensor:
        """EP forward: AllGather → route → grouped persistent → single non-persistent → AllReduce."""
        import torch.distributed as dist

        num_tokens = x.shape[0]

        if self.num_tokens_per_rank is None:
            raise RuntimeError("num_tokens_per_rank not set. Call init_num_tokens() first.")
        if num_tokens > self.num_tokens_per_rank:
            raise RuntimeError(
                f"MoE buffer overflow: num_tokens={num_tokens} > num_tokens_per_rank={self.num_tokens_per_rank}"
            )

        # 1) AllGather
        all_tokens = self.all_tokens_buffer
        all_tokens.zero_()
        padded = self.padded_hidden_buffer
        padded.zero_()
        if num_tokens > 0:
            padded[:num_tokens] = x

        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                all_tokens, padded,
                stream=torch.cuda.default_stream(self.device)
            )

        # 2) Route
        router_logits = self.router(all_tokens)
        if _HAS_CUDA_ROUTING:
            topk_indices, topk_weights = gate_topk_softmax_cuda(
                router_logits, k=self.num_experts_per_tok
            )
        else:
            topk_weights, topk_indices = torch.topk(
                router_logits, k=self.num_experts_per_tok, dim=-1
            )
            topk_weights = F.softmax(topk_weights, dim=-1)

        # 3) Process local experts
        global_results = self.global_results_buffer
        global_results.zero_()
        num_global_tokens = all_tokens.shape[0]

        # Phase 1: Grouped kernel for persistent experts
        num_persistent = len(self.persistent_expert_indices)
        if num_persistent > 0:
            global_results[:num_global_tokens] = self._grouped_forward(
                all_tokens, topk_indices, topk_weights,
                expert_start=self.expert_start,
                num_local_experts=num_persistent,
            )

        # Phase 2: Single-expert kernel for non-persistent experts
        if self.non_persistent_expert_indices:
            self._single_expert_forward(
                all_tokens, topk_indices, topk_weights,
                global_results[:num_global_tokens],
            )

        # 4) AllReduce
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                global_results,
                op=dist.ReduceOp.SUM,
                stream=torch.cuda.default_stream(self.device)
            )

        # 5) Extract local rank slice
        start = self.rank * self.num_tokens_per_rank
        return global_results[start:start + num_tokens].to(x.dtype)

    def _grouped_forward(
        self,
        hidden_flat: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_start: int,
        num_local_experts: int,
    ) -> torch.Tensor:
        """Grouped kernel for persistent experts."""
        if self.weight_format == "mxfp4":
            if not (_HAS_WGMMA_GROUPED and _HAS_CUDA_ROUTING):
                raise RuntimeError(
                    "Grouped WGMMA MXFP4 kernel not available. "
                    "Requires SM90 (Hopper) and CUDA routing."
                )
            return fused_mxfp4_grouped_moe_forward_cuda_routing(
                hidden_flat, topk_indices, topk_weights,
                self.gate_ptrs, self.gate_scale_ptrs,
                self.up_ptrs, self.up_scale_ptrs,
                self.down_ptrs, self.down_scale_ptrs,
                self.gate_weight_ref, self.gate_scale_ref,
                self.down_weight_ref, self.down_scale_ref,
                num_experts=self.total_experts,
                expert_start=expert_start,
                num_local_experts=num_local_experts,
                gate_bias_ptrs=self.gate_bias_ptrs,
                up_bias_ptrs=self.up_bias_ptrs,
                down_bias_ptrs=self.down_bias_ptrs,
            )
        elif self.weight_format == "bf16":
            # Placeholder: grouped BF16 kernel to be ported from
            # batchgen_kernels/moe/gptoss/grouped_bf16_moe_wgmma.py
            raise NotImplementedError(
                "Grouped BF16 MoE kernel not yet ported. "
                "See batchgen_kernels/moe/gptoss/grouped_bf16_moe_wgmma.py"
            )
        else:
            raise ValueError(f"Unknown weight_format: {self.weight_format}")

    def _single_expert_forward(
        self,
        hidden_flat: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Process non-persistent experts one by one, accumulating into output."""
        # Precompute which experts have tokens with a single CPU-GPU sync
        # instead of one sync per expert from expert_mask.any()
        active_experts_set = set(topk_indices.flatten().tolist())

        for expert_idx in self.non_persistent_expert_indices:
            if expert_idx not in active_experts_set:
                continue
            expert_mask = (topk_indices == expert_idx).any(dim=-1)

            expert_input = hidden_flat[expert_mask]

            if self.weight_format == "mxfp4":
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
                    expert = self.experts[expert_idx]
                    expert_output = expert(expert_input)
            elif self.weight_format == "bf16":
                # BF16 single expert: use wrapper forward (torch.mm path)
                expert = self.experts[expert_idx]
                expert_output = expert(expert_input)
            else:
                raise ValueError(f"Unknown weight_format: {self.weight_format}")

            # Weighted accumulation
            expert_weight = torch.where(
                topk_indices[expert_mask] == expert_idx,
                topk_weights[expert_mask],
                torch.zeros_like(topk_weights[expert_mask])
            ).sum(dim=-1)

            output[expert_mask] += expert_output * expert_weight.unsqueeze(-1)


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

        # Persistent / non-persistent expert indices (set by PSM)
        self.persistent_expert_indices = []
        self.non_persistent_expert_indices = []

        # Grouped WGMMA pointer arrays for persistent experts (set by PSM)
        self.gate_ptrs = None
        self.gate_scale_ptrs = None
        self.up_ptrs = None
        self.up_scale_ptrs = None
        self.down_ptrs = None
        self.down_scale_ptrs = None
        self.gate_bias_ptrs = None
        self.up_bias_ptrs = None
        self.down_bias_ptrs = None
        self.gate_weight_ref = None
        self.gate_scale_ref = None
        self.down_weight_ref = None
        self.down_scale_ref = None

    def _get_bf16_buffer(self, shape: Tuple[int, int], device: torch.device) -> torch.Tensor:
        """Get or allocate BF16 buffer for dequantized weights."""
        if self._bf16_buffer is None or self._buffer_shape != shape or self._bf16_buffer.device != device:
            self._bf16_buffer = torch.empty(shape, dtype=torch.bfloat16, device=device)
            self._buffer_shape = shape
        return self._bf16_buffer

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass: grouped WGMMA for persistent experts, per-expert loop for the rest."""
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
            )
            topk_weights = F.softmax(topk_weights, dim=-1)

        # Initialize output
        output = torch.zeros_like(hidden_flat)

        # Phase 1: Grouped WGMMA for persistent experts
        num_persistent = len(self.persistent_expert_indices)
        if num_persistent > 0 and _HAS_WGMMA_GROUPED and _HAS_CUDA_ROUTING and self.gate_ptrs is not None:
            output = fused_mxfp4_grouped_moe_forward_cuda_routing(
                hidden_flat, topk_indices, topk_weights,
                self.gate_ptrs, self.gate_scale_ptrs,
                self.up_ptrs, self.up_scale_ptrs,
                self.down_ptrs, self.down_scale_ptrs,
                self.gate_weight_ref, self.gate_scale_ref,
                self.down_weight_ref, self.down_scale_ref,
                num_experts=self.num_experts,
                num_local_experts=num_persistent,
                gate_bias_ptrs=self.gate_bias_ptrs,
                up_bias_ptrs=self.up_bias_ptrs,
                down_bias_ptrs=self.down_bias_ptrs,
            )
            # If all experts are persistent, we're done
            if not self.non_persistent_expert_indices:
                return output.view(batch_size, seq_len, hidden_dim)

        # Phase 2: Per-expert loop for non-persistent experts (or all if grouped unavailable)
        if self.non_persistent_expert_indices:
            loop_experts = self.non_persistent_expert_indices
        else:
            # Grouped WGMMA not available — fall back to per-expert for all
            loop_experts = topk_indices.unique().tolist()

        # Check if we can use fused WGMMA single-expert kernels
        use_fused = (
            is_wgmma_available()
            and os.environ.get("BATCHGEN_PREFILL_USE_CUTE", "0") != "1"
        )

        for expert_idx in loop_experts:
            expert_mask = (topk_indices == expert_idx).any(dim=-1)
            if not expert_mask.any():
                continue

            expert_input = hidden_flat[expert_mask].contiguous()

            expert_weights = torch.where(
                topk_indices[expert_mask] == expert_idx,
                topk_weights[expert_mask],
                torch.zeros_like(topk_weights[expert_mask])
            ).sum(dim=-1)

            if use_fused:
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
            else:
                expert_output = self._forward_expert_cute(expert_input, expert_idx)

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
    """Single transformer decoder layer.

    Supports optional CUDA graph mode for decode phase. When
    ``cuda_graph_manager`` is set, two segments are replayed from captured
    graphs with KV write + FlashAttention running eagerly between them:
      pre_attn graph:  RMSNorm → QKV proj → QKV split → RoPE
      eager middle:    KV cache write → FlashAttention
      post_attn graph: O_proj → residual add
    The MoE block always runs eagerly.
    """

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = GptOssAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GptOssMoE(config)

        # CUDA graph support (set externally after model init)
        self.cuda_graph_manager = None
        self._pre_attn_segment_name = None
        self._post_attn_segment_name = None

    def enable_cuda_graph(self, manager, pre_attn_name: str, post_attn_name: str):
        """Enable CUDA graph mode for this layer.

        Args:
            manager: CUDAGraphManager with segments pre-captured.
            pre_attn_name: Segment name for RMSNorm → QKV proj → split → reshape.
            post_attn_name: Segment name for residual add + post-attn RMSNorm.
        """
        self.cuda_graph_manager = manager
        self._pre_attn_segment_name = pre_attn_name
        self._post_attn_segment_name = post_attn_name

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

        # ========== ATTENTION BLOCK ==========
        # attn_residual: pre-attention hidden_states (for fused residual add + norm)
        # For CUDA graph path: post_attn segment does residual add, so attn_residual=None
        attn_residual = None
        use_graph = (self.cuda_graph_manager is not None
                     and self._pre_attn_segment_name is not None)

        if use_graph:
            batch_size = hidden_states.shape[0]

            if batch_size > 0:
                try:
                    # --- Pre-attn graph: RMSNorm → QKV proj → split → reshape ---
                    pre_out = self.cuda_graph_manager.replay(
                        self._pre_attn_segment_name, batch_size,
                        hidden_states=hidden_states,
                    )
                    query = pre_out["query"]
                    key = pre_out["key"]
                    value = pre_out["value"]

                    # --- Eager middle via wrapper: RoPE → KV write → FA → O_proj ---
                    attn_output = self.self_attn._forward_decode_mid(
                        query, key, value,
                    )

                    # --- Post-attn graph: residual add + post-attn RMSNorm ---
                    post_out = self.cuda_graph_manager.replay(
                        self._post_attn_segment_name, batch_size,
                        attn_residual=hidden_states,
                        attn_output=attn_output,
                    )
                    # Post-attn graph already fused residual add + norm.
                    # hidden_states = normed MoE input, residual for MoE residual add.
                    # Set attn_residual = "done" sentinel to skip the norm block below.
                    hidden_states = post_out["normed"]
                    residual = post_out["residual"]
                    attn_residual = "graph_done"

                except (ValueError, RuntimeError) as e:
                    logging.warning(
                        f"Layer {self.layer_idx}: CUDA graph replay failed ({e}), "
                        "falling back to eager execution"
                    )
                    hidden_states, attn_residual = self._forward_attn_eager(
                        hidden_states, attention_mask, position_ids,
                        past_key_value, output_attentions, use_cache,
                        debug_layer, timing_enabled,
                    )
            else:
                hidden_states, attn_residual = self._forward_attn_eager(
                    hidden_states, attention_mask, position_ids,
                    past_key_value, output_attentions, use_cache,
                    debug_layer, timing_enabled,
                )
        else:
            hidden_states, attn_residual = self._forward_attn_eager(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions, use_cache,
                debug_layer, timing_enabled,
            )

        # ========== MoE BLOCK (always eager) ==========
        # Post-attn norm: skip if post-attn graph already computed it
        if attn_residual == "graph_done":
            # hidden_states = normed, residual = residual — both set by post-attn graph
            pass
        elif debug_layer and self.layer_idx < 3:
            # Unfused path for debug visibility
            if attn_residual is not None:
                hidden_states = attn_residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            with torch.no_grad():
                print(f"[L{self.layer_idx}] after post_attn_layernorm: std={hidden_states.float().std().item():.4f}, max={hidden_states.abs().max().item():.4f}")
        elif attn_residual is not None:
            # Eager path: fuse residual add + post-attn layernorm (1 kernel)
            from batchgen.attention.fused_kernels import cuda_add_rmsnorm
            hidden_states, residual = cuda_add_rmsnorm(
                attn_residual, hidden_states,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
            )
        else:
            # Fallback (shouldn't happen in normal flow)
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)

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

        return hidden_states, None, None

    def _forward_attn_eager(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor]],
        output_attentions: bool,
        use_cache: bool,
        debug_layer: bool = False,
        timing_enabled: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eager attention forward (original path).

        Returns:
            (attn_output, residual) — caller uses cuda_add_rmsnorm to fuse
            the residual add + post-attention layernorm.
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if debug_layer and self.layer_idx < 3:
            with torch.no_grad():
                print(f"[L{self.layer_idx}] after input_layernorm: std={hidden_states.float().std().item():.4f}, max={hidden_states.abs().max().item():.4f}")

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

        return hidden_states, residual


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

        # Create shared RoPE instance (identical for all 36 layers — avoids 35× redundant
        # CPU trig computation on [131072, 64] tensors during init)
        self._shared_rotary_emb = YaRNRotaryEmbedding(
            dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=config.rope_scaling.get("factor", 32.0),
            original_max_position_embeddings=config.rope_scaling.get(
                "original_max_position_embeddings", 4096
            ),
            beta_fast=config.rope_scaling.get("beta_fast", 32.0),
            beta_slow=config.rope_scaling.get("beta_slow", 1.0),
        )

        self.layers = nn.ModuleList(
            [GptOssDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Assign shared RoPE to all attention layers
        for layer in self.layers:
            layer.self_attn.rotary_emb = self._shared_rotary_emb

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
