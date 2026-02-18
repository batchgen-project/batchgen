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

"""Kimi K2.5 model definition following BatchGen design pattern.

Architecture:
    - 61 transformer layers (1 dense + 60 MoE)
    - MLA attention with 64 heads, kv_lora_rank=512
    - 384 routed experts + 1 shared expert per MoE layer
    - INT4 W4A16 quantization (routed experts only)
    - Shared YaRN RoPE (single instance across all layers)
    - RMSNorm (eps=1e-6)

Design:
    - KimiK25ForCausalLM (outer): .model + .lm_head (worker-compatible)
    - KimiK25Model (inner): .embed_tokens, .layers, .norm
    - Wrappers handle optimized forward (INT4 dequant, FlashAttention, KV cache)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Tuple

from dataclasses import dataclass
from batchgen.layers.rotary_embedding import YarnRotaryEmbedding


@dataclass
class _CausalLMOutput:
    """Minimal output container with .logits attribute for worker compatibility."""
    logits: torch.Tensor


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

    Structural definition — actual forward (FlashAttention, RoPE, KV cache)
    is handled by KimiK25AttnWrapper.

    The rotary_emb attribute is assigned externally by KimiK25Model to share
    a single YarnRotaryEmbedding instance across all layers.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads  # 64
        self.q_lora_rank = config.q_lora_rank  # 1536
        self.kv_lora_rank = config.kv_lora_rank  # 512
        self.qk_nope_head_dim = config.qk_nope_head_dim  # 128
        self.qk_rope_head_dim = config.qk_rope_head_dim  # 64
        self.v_head_dim = config.v_head_dim  # 128
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 192
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        # Q projection with low-rank compression
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        # KV projection with MQA-style compression
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        # RoPE — assigned by KimiK25Model (shared across layers)
        self.rotary_emb = None

        # Softmax scales for MLA (materialized and unmaterialized KV)
        self.qkv_materialized_softmax_scale = self.q_head_dim ** -0.5
        self.qkv_unmaterialized_softmax_scale = (self.kv_lora_rank + self.qk_rope_head_dim) ** -0.5
        if config.rope_scaling is not None:
            mscale_all_dim = config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = _yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.qkv_materialized_softmax_scale *= mscale * mscale
                self.qkv_unmaterialized_softmax_scale *= mscale * mscale
        self.softmax_scale = self.qkv_materialized_softmax_scale

    def initialize(self):
        """Pre-compute absorbed projections for decode phase."""
        if getattr(self.config, 'phase', None) == "decode":
            kv_b_proj = self.kv_b_proj.weight.view(
                self.num_heads, -1, self.kv_lora_rank
            )
            self.q_absorb = kv_b_proj[:, : self.qk_nope_head_dim, :]
            self.out_absorb = kv_b_proj[:, self.qk_nope_head_dim :, :]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "KimiK25Attention.forward() is structural. "
            "Use KimiK25AttnWrapper for actual attention computation."
        )


def _yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


# ============================================================================
# Expert MLP
# ============================================================================

class KimiK25Expert(nn.Module):
    """Single expert FFN with SiLU gating.

    Used for both routed experts (INT4 W4A16, weights managed by wrappers)
    and shared experts (BF16, weights loaded directly).
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Dense MLP (Layer 0)
# ============================================================================

class DenseMLP(nn.Module):
    """Dense FFN for K2.5 layer 0 (non-MoE).

    Uses larger intermediate_size (18432) than MoE experts (2048).
    """

    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


# ============================================================================
# MoE Gate (Router)
# ============================================================================

class MoEGate(nn.Module):
    """MoE router with sigmoid scoring and top-k selection.

    K2.5 specifics:
        - Sigmoid scoring (not softmax)
        - n_group=1, topk_group=1 (no group-based selection)
        - routed_scaling_factor=2.5
        - e_score_correction_bias for noaux_tc routing
    """

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok  # 8
        self.n_routed_experts = config.n_routed_experts  # 384
        self.routed_scaling_factor = config.routed_scaling_factor  # 2.5
        self.scoring_func = config.scoring_func  # "sigmoid"
        self.topk_method = config.topk_method  # "noaux_tc"
        self.n_group = config.n_group  # 1
        self.topk_group = config.topk_group  # 1
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(
            torch.empty(self.n_routed_experts, config.hidden_size)
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @torch.inference_mode()
    def warmup(self):
        pass

    def forward(self, hidden_states: torch.Tensor):
        """Standard routing forward.

        Args:
            hidden_states: [batch, seq, hidden_size]

        Returns:
            topk_idx: [total_tokens, top_k] — selected expert indices
            topk_weight: [total_tokens, top_k] — normalized + scaled weights
        """
        if hidden_states.dim() == 2:
            num_tokens, h = hidden_states.shape
        else:
            bsz, seq_len, h = hidden_states.shape
            num_tokens = bsz * seq_len
        hidden_states = hidden_states.view(-1, h)

        # Early return for zero tokens (can happen on some ranks during decode)
        if num_tokens == 0:
            return (
                torch.empty(0, self.top_k, dtype=torch.long, device=hidden_states.device),
                torch.empty(0, self.top_k, dtype=hidden_states.dtype, device=hidden_states.device),
            )

        logits = F.linear(hidden_states.float(), self.weight.float(), None)
        scores = logits.sigmoid()

        # Top-k with noaux_tc correction bias
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        # Group-based selection (n_group=1 makes this a simple topk)
        group_scores = (
            scores_for_choice.view(num_tokens, self.n_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(num_tokens, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        # Normalize and scale
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight

    @torch.inference_mode()
    @torch.compile(mode="max-autotune", backend="inductor")
    def decoding_forward(self, hidden_states):
        """Decode-optimized routing (torch.compiled)."""
        if hidden_states.dim() == 2:
            num_tokens, h = hidden_states.shape
        else:
            bsz, seq_len, h = hidden_states.shape
            num_tokens = bsz * seq_len
        hidden_states = hidden_states.view(-1, h)

        logits = F.linear(hidden_states.float(), self.weight.float(), None)
        scores = logits.sigmoid()

        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(num_tokens, self.n_group, -1)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(
            group_scores, k=self.topk_group, dim=-1, sorted=False
        )[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(num_tokens, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)

        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight.to(hidden_states.dtype)


# ============================================================================
# MoE Layer (384 Routed + 1 Shared Expert)
# ============================================================================

class KimiK25MoE(nn.Module):
    """MoE layer with 384 routed + 1 shared expert.

    In EP mode (decode), only local experts are instantiated; rest are None.
    Wrappers handle the actual forward pass (INT4 dequant, routing execution).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.n_routed_experts  # 384
        self.top_k = config.num_experts_per_tok  # 8
        self.moe_intermediate_size = config.moe_intermediate_size  # 2048
        self.num_experts_per_tok = config.num_experts_per_tok

        # Router
        self.gate = MoEGate(config)

        # Routed experts — EP mode creates None placeholders for non-local experts
        ep_size = getattr(config, 'ep_size', 1)
        if ep_size > 1 and dist.is_initialized():
            rank = dist.get_rank()
            experts_per_rank = self.num_experts // ep_size
            start = rank * experts_per_rank
            end = start + experts_per_rank
            self.experts = nn.ModuleList([
                KimiK25Expert(self.hidden_size, self.moe_intermediate_size)
                if start <= i < end else None
                for i in range(self.num_experts)
            ])
        else:
            self.experts = nn.ModuleList([
                KimiK25Expert(self.hidden_size, self.moe_intermediate_size)
                for _ in range(self.num_experts)
            ])

        # Shared expert (BF16, always active)
        n_shared = getattr(config, 'n_shared_experts', 1)
        self.shared_experts = KimiK25Expert(
            self.hidden_size,
            self.moe_intermediate_size * n_shared,
        )

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MoE forward with loop-based expert dispatch.

        Routes tokens to top-k experts via the gate, dispatches to each expert
        individually (compatible with INT4 wrappers), and adds shared expert output.
        """
        identity = hidden_states
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        num_tokens, hidden_size = hidden_states.shape
        device = hidden_states.device

        # Gate routing
        topk_idx, topk_weight = self.gate(identity)
        # topk_idx: [num_tokens, top_k], topk_weight: [num_tokens, top_k]

        K = self.top_k
        flat_expert_idx = topk_idx.view(-1)  # [num_tokens * K]
        flat_weights = topk_weight.view(-1)  # [num_tokens * K]
        token_indices = torch.arange(num_tokens, device=device).repeat_interleave(K)
        topk_positions = torch.arange(K, device=device).repeat(num_tokens)

        # Accumulate weighted expert outputs
        results = torch.zeros(num_tokens, hidden_size, device=device, dtype=torch.float32)

        for expert_idx, expert in enumerate(self.experts):
            if expert is None:
                continue

            # Find tokens routed to this expert
            mask = flat_expert_idx == expert_idx
            if not mask.any():
                continue

            expert_token_idx = token_indices[mask]
            expert_topk_pos = topk_positions[mask]
            tokens_for_expert = hidden_states[expert_token_idx]

            expert_output = expert(tokens_for_expert)

            expert_weights = topk_weight[expert_token_idx, expert_topk_pos]
            weighted_output = expert_output.float() * expert_weights.unsqueeze(-1)
            results.index_add_(0, expert_token_idx, weighted_output)

        results = results.to(hidden_states.dtype)

        # Add shared expert output
        results = results + self.shared_experts(identity.view(-1, hidden_size))

        return results.view(*orig_shape)


# ============================================================================
# Decoder Layer
# ============================================================================

class KimiK25DecoderLayer(nn.Module):
    """Single K2.5 transformer layer with pre-norm architecture.

    Layer 0: dense MLP. Layers 1-60: MoE with 384 routed + 1 shared expert.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = KimiK25Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_idx < config.first_k_dense_replace:
            self.mlp = DenseMLP(config)
        else:
            self.mlp = KimiK25MoE(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # Pre-norm attention + residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden_states=hidden_states)
        hidden_states = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        hidden_states = residual + hidden_states

        # Pre-norm MoE/FFN + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return (hidden_states, None, None)


# ============================================================================
# Inner Model
# ============================================================================

class KimiK25Model(nn.Module):
    """Kimi K2.5 transformer model (inner, no lm_head).

    Contains embed_tokens, layers, norm. A single shared YarnRotaryEmbedding
    instance is created and assigned to all attention layers to avoid
    duplicated cos/sin caches on GPU (~2.4 GiB savings).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Shared RoPE — single instance for all 61 attention layers
        rope_scaling = config.rope_scaling or {}
        self._shared_rotary_emb = YarnRotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=rope_scaling.get("factor", 1.0),
            original_max_position_embeddings=rope_scaling.get("original_max_position_embeddings", 4096),
            beta_fast=rope_scaling.get("beta_fast", 32.0),
            beta_slow=rope_scaling.get("beta_slow", 1.0),
        )

        self.layers = nn.ModuleList([
            KimiK25DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])

        # Assign shared RoPE to all attention layers
        for layer in self.layers:
            layer.self_attn.rotary_emb = self._shared_rotary_emb

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds")
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            layer_output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        hidden_states = self.norm(hidden_states)
        return hidden_states


# ============================================================================
# Outer Wrapper (worker-compatible)
# ============================================================================

class KimiK25ForCausalLM(nn.Module):
    """Kimi K2.5 model with language modeling head.

    Provides .model and .lm_head attributes expected by batchgen_worker.py:
        - self.model.layers[i]  (via KimiK25Model)
        - self.lm_head          (output projection)
    """

    def __init__(self, config, comm=None):
        super().__init__()
        self.config = config
        self.model = KimiK25Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )
        logits = self.lm_head(hidden_states)
        return _CausalLMOutput(logits=logits)

    def eval(self):
        return super().eval()
