# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                             #
# ---------------------------------------------------------------------------- #

"""GLM-5 (GlmMoeDsaForCausalLM) model implementation.

Clean standalone model following BatchGen flat model spec.

Architecture:
- 78 layers, hidden_size=6144, vocab=154880
- MLA attention (q_lora_rank=2048, kv_lora_rank=512, qk_nope=192, v_head=256)
- DSA indexer (32 heads, 128 head_dim, top-2048)
- MoE: 256 routed experts + 1 shared, top-8, n_group=1, sigmoid scoring
- First 3 layers are dense (no MoE)
- FP8 E4M3 quantization, [128,128] block
- rope_interleave=True, rope_theta=1M, no scaling
"""

import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_glm5 import Glm5Config


# ============================================================================
# RMSNorm
# ============================================================================

class Glm5RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from batchgen.attention.fused_kernels import cuda_rmsnorm
        return cuda_rmsnorm(x, self.weight, self.eps)


# ============================================================================
# Rotary Position Embedding (base, no YaRN)
# ============================================================================

class Glm5RotaryEmbedding(nn.Module):
    """Base RoPE with theta=1M, no scaling."""

    def __init__(self, dim: int, max_position_embeddings: int = 202752, base: float = 1000000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_position_embeddings, torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.dtype)
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def apply_rotary_pos_emb_interleaved(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE with interleaved pattern (rope_interleave=True).

    Interleaved: even indices get cos/sin, odd indices get -sin/cos.
    q[..., 0::2] and q[..., 1::2] form the pairs.
    """
    cos = cos.unsqueeze(1)  # [seq, 1, dim]
    sin = sin.unsqueeze(1)  # [seq, 1, dim]

    # Interleaved: pair (x[0], x[1]), (x[2], x[3]), ...
    q1, q2 = q[..., 0::2], q[..., 1::2]
    k1, k2 = k[..., 0::2], k[..., 1::2]

    # cos/sin have shape [seq, 1, rope_dim] where rope_dim = 2 * dim//2
    # We need half the cos/sin for the pairs
    cos_half = cos[..., : cos.shape[-1] // 2]
    sin_half = sin[..., : sin.shape[-1] // 2]

    q_rot = torch.stack([q1 * cos_half - q2 * sin_half, q2 * cos_half + q1 * sin_half], dim=-1)
    q_rot = q_rot.flatten(-2)
    k_rot = torch.stack([k1 * cos_half - k2 * sin_half, k2 * cos_half + k1 * sin_half], dim=-1)
    k_rot = k_rot.flatten(-2)

    return q_rot, k_rot


def apply_rotary_pos_emb_split(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE with split pattern (standard DeepSeek style)."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q1, q2 = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
    k1, k2 = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]
    q_rot = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1)
    return q_rot, k_rot


# ============================================================================
# DSA Indexer
# ============================================================================

class Glm5Indexer(nn.Module):
    """Lightning Indexer for DSA (DeepSeek Sparse Attention).

    GLM-5 indexer components:
    - wk: K projection from hidden_states [index_dim, hidden_size] (FP8)
    - wq_b: Q B-projection from q_a intermediate [index_dim, q_lora_rank] (FP8)
    - k_norm: RMSNorm on K output (with bias) [index_dim]
    - weights_proj: Per-head importance scoring [index_n_heads, hidden_size] (BF16)

    The Q path shares q_a from the main MLA attention, then applies wq_b.
    The K path: hidden_states -> wk -> k_norm -> indexer K.
    Scoring: importance scores from weights_proj modulate Q@K scoring.
    """

    def __init__(self, config: Glm5Config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.index_n_heads = config.index_n_heads
        self.index_head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.index_dim = config.index_n_heads * config.index_head_dim  # 4096

        # K path: hidden -> wk -> k_norm
        self.wk = nn.Linear(config.hidden_size, self.index_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.index_dim)  # Has weight + bias

        # Q path: q_a (from main attention) -> wq_b
        self.wq_b = nn.Linear(config.q_lora_rank, self.index_dim, bias=False)

        # Importance scoring
        self.weights_proj = nn.Linear(config.hidden_size, self.index_n_heads, bias=False)

    def compute_indexer_kv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute indexer K for cache storage.

        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            indexer_k: [batch, seq_len, 1, index_dim] shaped for paged KV manager
        """
        k = self.wk(hidden_states)
        k = self.k_norm(k)
        return k.unsqueeze(2)  # [batch, seq_len, 1, index_dim]

    def score_and_select(
        self,
        q_a: torch.Tensor,
        hidden_states: torch.Tensor,
        cached_k: torch.Tensor,
        cache_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Score cached tokens and select top-K.

        Args:
            q_a: [batch, 1, q_lora_rank] — q_a intermediate from main attention
            hidden_states: [batch, 1, hidden_size] — for importance scoring
            cached_k: [batch, max_seqlen, index_dim] — gathered indexer K cache
            cache_seqlens: [batch] — valid lengths

        Returns:
            top_k_indices: [batch, index_topk]
        """
        batch_size = q_a.shape[0]
        max_seqlen = cached_k.shape[1]

        # Q from shared q_a intermediate
        q = self.wq_b(q_a)  # [batch, 1, index_dim]
        q = q.view(batch_size, self.index_n_heads, self.index_head_dim)

        # Reshape cached K
        cached_k = cached_k.view(batch_size, max_seqlen, self.index_n_heads, self.index_head_dim)
        cached_k = cached_k.permute(0, 2, 1, 3)  # [batch, n_heads, max_seqlen, head_dim]

        # Q @ K^T
        q = q.unsqueeze(2)  # [batch, n_heads, 1, head_dim]
        scores = torch.matmul(q, cached_k.transpose(-2, -1))  # [batch, n_heads, 1, max_seqlen]
        scores = scores.squeeze(2)  # [batch, n_heads, max_seqlen]

        # Importance weighting
        importance = self.weights_proj(hidden_states.squeeze(1))  # [batch, n_heads]
        scores = scores * importance.unsqueeze(-1)  # [batch, n_heads, max_seqlen]

        # Mask invalid positions
        position_indices = torch.arange(max_seqlen, device=scores.device).unsqueeze(0)
        mask = position_indices >= cache_seqlens.unsqueeze(1)
        scores.masked_fill_(mask.unsqueeze(1), float("-inf"))

        # Aggregate across heads and select top-K
        aggregated = scores.sum(dim=1)  # [batch, max_seqlen]
        effective_topk = min(self.index_topk, max_seqlen)
        _, top_k_indices = torch.topk(aggregated, effective_topk, dim=-1)

        return top_k_indices

    def score_and_select_paged(
        self,
        q_a: torch.Tensor,
        hidden_states: torch.Tensor,
        indexer_blocked_k: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        page_size: int = 64,
    ) -> torch.Tensor:
        """Score cached tokens from paged cache and select top-K.

        Gathers indexer K from paged cache into contiguous tensor,
        then delegates to score_and_select.

        Args:
            q_a: [batch, 1, q_lora_rank] — q_a intermediate from main attention
            hidden_states: [batch, 1, hidden_size] — for importance scoring
            indexer_blocked_k: [num_pages, page_size, 1, index_dim] — paged indexer cache
            block_table: [batch, max_num_pages_per_seq] — page mapping
            cache_seqlens: [batch] — valid lengths
            page_size: tokens per page

        Returns:
            top_k_indices: [batch, index_topk] — absolute token positions
        """
        from batchgen.attention.dsa.indexer import _gather_all_from_paged_cache

        max_seqlen = int(cache_seqlens.max().item())
        # gathered: [batch, max_seqlen, 1, index_dim]
        gathered_k = _gather_all_from_paged_cache(
            indexer_blocked_k, block_table, cache_seqlens, page_size, max_seqlen
        )
        # Squeeze num_k_heads dim: [batch, max_seqlen, index_dim]
        gathered_k = gathered_k.squeeze(2)
        return self.score_and_select(q_a, hidden_states, gathered_k, cache_seqlens)


# ============================================================================
# MLA Attention
# ============================================================================

class Glm5MLA(nn.Module):
    """Multi-head Latent Attention with DSA indexer.

    MLA computation:
    1. Compress Q: hidden -> q_a_proj -> q_a_layernorm -> q_b_proj -> split(q_nope, q_rope)
    2. Compress KV: hidden -> kv_a_proj_with_mqa -> split(kv_a, k_rope) -> kv_a_layernorm -> kv_b_proj -> split(k_nope, v)
    3. Apply RoPE (interleaved) to q_rope and k_rope
    4. Attention: (q_nope || q_rope) @ (k_nope || k_rope)^T -> softmax -> @ v
    5. Output: o_proj

    For decode with DSA: use indexer to select top-K cached tokens, then attend only to those.
    """

    def __init__(self, config: Glm5Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim  # 256
        self.compressed_kv_dim = config.kv_lora_rank + config.qk_rope_head_dim  # 576

        # Q path: hidden -> q_a_proj -> q_a_layernorm -> q_b_proj
        self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_layernorm = Glm5RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )

        # KV path: hidden -> kv_a_proj_with_mqa -> split -> kv_a_layernorm -> kv_b_proj
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, config.kv_lora_rank + config.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = Glm5RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)

        # RoPE
        self.rotary_emb = None  # Shared instance, set by Glm5Model.__init__

        # Softmax scale
        self.softmax_scale = self.q_head_dim ** -0.5

        # DSA indexer
        self.indexer = Glm5Indexer(config, layer_idx)

        # Absorbed projections for decode (set by initialize())
        self.q_absorb = None
        self.out_absorb = None

    def initialize(self):
        """Pre-compute absorbed projections for decode phase."""
        if hasattr(self.config, 'phase') and self.config.phase == "decode":
            kv_b = self.kv_b_proj.weight.view(
                self.num_heads, -1, self.kv_lora_rank
            )
            self.q_absorb = kv_b[:, : self.qk_nope_head_dim, :]
            self.out_absorb = kv_b[:, self.qk_nope_head_dim :, :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """MLA forward (reference implementation, replaced by wrapper at runtime)."""
        bsz, q_len, _ = hidden_states.size()

        # Q path
        q_a = self.q_a_proj(hidden_states)
        q_a = self.q_a_layernorm(q_a)
        q = self.q_b_proj(q_a)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)

        # KV path
        kv_a_with_rope = self.kv_a_proj_with_mqa(hidden_states)
        kv_a, k_rope = torch.split(
            kv_a_with_rope, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv_a = self.kv_a_layernorm(kv_a)
        kv_b = self.kv_b_proj(kv_a)
        kv_b = kv_b.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        kv_b = kv_b.transpose(1, 2)

        k_nope, v = torch.split(kv_b, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # RoPE
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_rope = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        cos, sin = self.rotary_emb(hidden_states, seq_len=q_len)
        if position_ids is not None:
            cos = cos[position_ids].squeeze(0)
            sin = sin[position_ids].squeeze(0)
        else:
            cos = cos[:q_len]
            sin = sin[:q_len]

        # Apply interleaved RoPE
        q_rope, k_rope = apply_rotary_pos_emb_interleaved(q_rope, k_rope, cos, sin)

        # Concatenate nope + rope
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.softmax_scale

        # Causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, None


# ============================================================================
# Expert Module
# ============================================================================

class Glm5Expert(nn.Module):
    """Single expert FFN: gate_proj + up_proj -> SiLU gate -> down_proj."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    @torch.inference_mode()
    def deepgemm_forward(self, x, scale):
        """FP8 forward using DeepGEMM single-expert GEMM.

        Used by GLM5ExpertWrapper for non-persistent experts (loaded one-by-one)
        and shared experts. w8a16_gemm quantizes BF16 activations to FP8,
        then runs fp8_gemm_nt against FP8 weights with block-wise scales.
        """
        from batchgen.attention.mla.fa3_backend import w8a16_gemm
        up = w8a16_gemm(self.up_proj.weight.data, scale['up_proj.weight_scale_inv'], x)
        gate = w8a16_gemm(self.gate_proj.weight.data, scale['gate_proj.weight_scale_inv'], x)
        intermediate = F.silu(gate) * up
        return w8a16_gemm(self.down_proj.weight.data, scale['down_proj.weight_scale_inv'], intermediate)


# ============================================================================
# MoE Gate
# ============================================================================

class Glm5MoEGate(nn.Module):
    """MoE routing gate with sigmoid scoring and n_group=1 (simple top-K).

    Includes e_score_correction_bias for score correction.
    """

    def __init__(self, config: Glm5Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor

        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = nn.Parameter(torch.zeros(config.n_routed_experts))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Returns:
            topk_weights: [batch*seq, num_experts_per_tok]
            topk_indices: [batch*seq, num_experts_per_tok]
        """
        bsz_seq = hidden_states.shape[0]

        # Sigmoid scoring (n_group=1: score all experts directly)
        logits = F.linear(hidden_states, self.weight)  # [bsz_seq, num_experts]
        scores = torch.sigmoid(logits)

        # Apply score correction bias
        scores = scores + self.e_score_correction_bias.unsqueeze(0)

        # Simple top-K (n_group=1, no group-based routing)
        topk_weights, topk_indices = torch.topk(
            scores, k=self.num_experts_per_tok, dim=-1
        )

        # Normalize
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Scale
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights, topk_indices


# ============================================================================
# MoE Layer (Prefill)
# ============================================================================

class Glm5MoE(nn.Module):
    """MoE layer for prefill: per-expert sequential processing."""

    def __init__(self, config: Glm5Config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size

        self.gate = Glm5MoEGate(config)
        self.experts = nn.ModuleList([
            Glm5Expert(config.hidden_size, config.moe_intermediate_size)
            for _ in range(self.num_experts)
        ])

        # Shared expert
        self.shared_experts = Glm5Expert(config.hidden_size, config.moe_intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)

        topk_weights, topk_indices = self.gate(hidden_flat)

        # Shared expert
        shared_output = self.shared_experts(hidden_flat)

        # Routed experts
        output = torch.zeros_like(hidden_flat)
        for i, expert in enumerate(self.experts):
            expert_mask = (topk_indices == i).any(dim=-1)
            if not expert_mask.any():
                continue
            expert_input = hidden_flat[expert_mask]
            expert_output = expert(expert_input)
            expert_weight = torch.where(
                topk_indices[expert_mask] == i,
                topk_weights[expert_mask],
                torch.zeros_like(topk_weights[expert_mask])
            ).sum(dim=-1)
            output[expert_mask] += expert_output * expert_weight.unsqueeze(-1)

        output = output + shared_output
        return output.view(batch_size, seq_len, hidden_dim)


# ============================================================================
# MoE Layer (Decode with EP)
# ============================================================================

class Glm5MoEDecode(nn.Module):
    """MoE layer for decode with EP and grouped FP8 GEMM.

    Modeled on DeepseekV3MoE_Decoding_FP8. Two execution paths:
    - moe_infer_allgather_allreduce_bf16_acc: grouped FP8 GEMM for all-persistent
    - moe_infer_loop_with_offloading: loop-based for mixed persistent/non-persistent
    """

    def __init__(self, config: Glm5Config, comm=None):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.comm = comm

        import torch.distributed as dist
        if not dist.is_initialized():
            self.rank, self.world_size = 0, 1
        else:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.experts_per_rank = config.n_routed_experts // self.world_size
        self.total_experts = self.world_size * self.experts_per_rank
        self.routed_expert_start_idx = self.rank * self.experts_per_rank
        self.routed_expert_end_idx = (self.rank + 1) * self.experts_per_rank

        self.gate = Glm5MoEGate(config)
        self.experts = nn.ModuleList([None] * self.total_experts)
        self.shared_experts = Glm5Expert(config.hidden_size, config.moe_intermediate_size)

        self.device = torch.device("cuda", self.rank % torch.cuda.device_count())
        self.num_tokens_per_rank = None
        self.enable_ep_offloading = False
        self.num_persistent_local_experts = self.experts_per_rank  # default: all persistent

    def init_num_tokens(self, num_tokens_per_rank: int):
        self.num_tokens_per_rank = num_tokens_per_rank
        self.max_num_tokens_per_rank = num_tokens_per_rank
        global_num_tokens = num_tokens_per_rank * self.world_size
        K = self.num_experts_per_tok
        self.token_idx = torch.arange(
            global_num_tokens, dtype=torch.int32, device=self.device
        ).repeat_interleave(K)
        self.topk_pos = torch.arange(
            K, dtype=torch.int32, device=self.device
        ).repeat(global_num_tokens)

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int):
        if num_tokens_per_rank == self.num_tokens_per_rank:
            return
        if hasattr(self, 'max_num_tokens_per_rank') and num_tokens_per_rank > self.max_num_tokens_per_rank:
            self.max_num_tokens_per_rank = num_tokens_per_rank
        self.num_tokens_per_rank = num_tokens_per_rank
        global_num_tokens = num_tokens_per_rank * self.world_size
        K = self.num_experts_per_tok
        self.token_idx = torch.arange(
            global_num_tokens, dtype=torch.int32, device=self.device
        ).repeat_interleave(K)
        self.topk_pos = torch.arange(
            K, dtype=torch.int32, device=self.device
        ).repeat(global_num_tokens)

    def init(self, micro_batch_size):
        """Collect FP8 weight tensors and build pointer arrays for grouped GEMM.

        Only collects from persistent experts (first num_persistent_local_experts
        in the local range). Non-persistent experts use the loop path.
        """
        self.gate_list, self.up_list, self.down_list = [], [], []
        self.gate_scale_list, self.up_scale_list, self.down_scale_list = [], [], []
        n_persistent = getattr(self, 'num_persistent_local_experts', self.experts_per_rank)
        persistent_end = self.routed_expert_start_idx + n_persistent
        for e in range(self.routed_expert_start_idx, persistent_end):
            wrapper = self.experts[e]
            self.gate_list.append(wrapper.cached_gate)
            self.up_list.append(wrapper.cached_up)
            self.down_list.append(wrapper.cached_down)
            self.gate_scale_list.append(
                wrapper.weight_dequant_scale['gate_proj.weight_scale_inv'])
            self.up_scale_list.append(
                wrapper.weight_dequant_scale['up_proj.weight_scale_inv'])
            self.down_scale_list.append(
                wrapper.weight_dequant_scale['down_proj.weight_scale_inv'])

        self.gate_ptrs_ptr = torch.tensor(
            [r.data_ptr() for r in self.gate_list], dtype=torch.int64, device=self.device)
        self.up_ptrs_ptr = torch.tensor(
            [r.data_ptr() for r in self.up_list], dtype=torch.int64, device=self.device)
        self.down_ptrs_ptr = torch.tensor(
            [r.data_ptr() for r in self.down_list], dtype=torch.int64, device=self.device)
        self.gate_scale_ptrs_ptr = torch.tensor(
            [s.data_ptr() for s in self.gate_scale_list], dtype=torch.int64, device=self.device)
        self.up_scale_ptrs_ptr = torch.tensor(
            [s.data_ptr() for s in self.up_scale_list], dtype=torch.int64, device=self.device)
        self.down_scale_ptrs_ptr = torch.tensor(
            [s.data_ptr() for s in self.down_scale_list], dtype=torch.int64, device=self.device)

    def cleanup(self):
        for attr in ('gate_list', 'up_list', 'down_list',
                      'gate_scale_list', 'up_scale_list', 'down_scale_list',
                      'gate_ptrs_ptr', 'up_ptrs_ptr', 'down_ptrs_ptr',
                      'gate_scale_ptrs_ptr', 'up_scale_ptrs_ptr', 'down_scale_ptrs_ptr'):
            setattr(self, attr, None)

    def grouped_dequant_moe_fp8(self, x, eids, expert_counts, expert_offsets,
                                num_local_experts=None):
        """Grouped FP8 GEMM: gate+up+SiLU → down, same as DeepSeek.

        Args:
            num_local_experts: Number of experts in the pointer arrays. Defaults to
                num_persistent_local_experts (set by init()). Must match the length
                of gate_list/up_list/down_list.
        """
        from batchgen.attention.mla.fa3_backend import act_quant
        from batchgen.gemm.w8a8_grouped_gemm_stage_1 import fused_fp8_moe_stage_1_tma
        from batchgen.moe.fused_grouped_dequant_gemm import fused_dequant_grouped_gemm_fp8_tma
        from mgn_kernel import compact_expert_data

        if num_local_experts is None:
            num_local_experts = len(self.gate_list)

        actual_num_tokens = expert_offsets[-1]
        if isinstance(actual_num_tokens, torch.Tensor):
            actual_num_tokens = actual_num_tokens.item()
        if actual_num_tokens == 0:
            return torch.empty((0, x.shape[1] if x.dim() > 1 else self.hidden_size),
                               device=x.device, dtype=torch.bfloat16)

        expert_counts = expert_counts.to(torch.int32)
        group_size, activated_group_idx, group_start_indices, num_active_experts = \
            compact_expert_data(expert_counts)

        if isinstance(num_active_experts, torch.Tensor):
            num_active_val = num_active_experts.item()
        else:
            num_active_val = int(num_active_experts)
        if num_active_val == 0:
            return torch.empty((0, x.shape[1] if x.dim() > 1 else self.hidden_size),
                               device=x.device, dtype=torch.bfloat16)

        x_sliced = x[:actual_num_tokens]
        x_quant, x_scale = act_quant(x_sliced)

        intermediate = fused_fp8_moe_stage_1_tma(
            x_quant, x_scale,
            self.gate_list, self.gate_ptrs_ptr,
            self.up_list, self.up_ptrs_ptr,
            self.gate_scale_list, self.gate_scale_ptrs_ptr,
            self.up_scale_list, self.up_scale_ptrs_ptr,
            group_size, activated_group_idx, group_start_indices,
            num_active_experts, num_local_experts,
        )

        intermediate, intermediate_scale = act_quant(intermediate)

        return fused_dequant_grouped_gemm_fp8_tma(
            intermediate, intermediate_scale,
            self.down_list, self.down_ptrs_ptr,
            self.down_scale_list, self.down_scale_ptrs_ptr,
            group_size, activated_group_idx, group_start_indices,
            num_active_experts,
        )

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        identity = hidden_states

        if getattr(self, 'enable_ep_offloading', False):
            out = self.moe_infer_loop_with_offloading(hidden_states)
        else:
            out = self.moe_infer_allgather_allreduce_bf16_acc(hidden_states)

        out = out + self.shared_experts(identity)
        return out.view(*orig_shape)

    @torch.inference_mode()
    def moe_infer_allgather_allreduce_bf16_acc(self, x: torch.Tensor) -> torch.Tensor:
        """Grouped FP8 GEMM path: AllGather → Gate → Dispatch → GroupedGEMM → Scatter → AllReduce."""
        import torch.distributed as dist
        from mgn_kernel import fused_moe_token_dispatch
        # scatter_weight_reduce_optimized is imported from DeepSeek's modeling file
        from batchgen.models.deepseek.deepseekv3.modeling_deepseek_v3 import scatter_weight_reduce_optimized

        num_tokens, hidden_size = x.shape
        device = x.device

        if num_tokens > self.num_tokens_per_rank:
            raise RuntimeError(
                f"MoE buffer overflow: num_tokens={num_tokens} > "
                f"num_tokens_per_rank={self.num_tokens_per_rank}"
            )

        # 1) AllGather
        all_tokens = torch.zeros(
            (self.world_size * self.num_tokens_per_rank, self.hidden_size),
            device=self.device, dtype=torch.bfloat16,
        )
        if x.shape[0] < self.num_tokens_per_rank:
            padded_hidden_states = torch.zeros(
                (self.num_tokens_per_rank, hidden_size),
                device=self.device, dtype=x.dtype,
            )
            padded_hidden_states[:x.shape[0]] = x
        else:
            padded_hidden_states = x

        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                all_tokens, padded_hidden_states,
                stream=torch.cuda.default_stream(self.device),
            )

        # 2) Gate (GLM-5: sigmoid scoring)
        global_x = all_tokens
        topk_weight, topk_idx = self.gate(global_x)
        # gate returns (weights, indices); fused_moe_token_dispatch expects int32 indices
        topk_idx = topk_idx.to(torch.int32)
        # topk_weight must be float32 for scatter_weight_reduce_optimized
        topk_weight = topk_weight.to(torch.float32)

        # 3) Dispatch tokens to local experts
        input_x, input_eids, global_indices, token_topk_pos, expert_counts, expert_offsets = \
            fused_moe_token_dispatch(
                global_x, topk_idx, self.token_idx, self.topk_pos,
                self.routed_expert_start_idx, self.routed_expert_end_idx,
            )

        # 4) Grouped FP8 GEMM
        res = self.grouped_dequant_moe_fp8(
            input_x, input_eids, expert_counts, expert_offsets,
        )

        # 5) Scatter + weighted reduce
        global_results = scatter_weight_reduce_optimized(
            res, global_indices, token_topk_pos, topk_weight,
            self.num_tokens_per_rank * self.world_size, self.num_experts_per_tok,
        )
        global_results = global_results.to(torch.bfloat16)

        # 6) AllReduce
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                global_results, op=dist.ReduceOp.SUM,
                stream=torch.cuda.default_stream(self.device),
            )

        # 7) Extract local slice
        start_token_ids = self.rank * self.num_tokens_per_rank
        return global_results[start_token_ids:start_token_ids + num_tokens].to(x.dtype)

    @torch.inference_mode()
    def moe_infer_loop_with_offloading(self, x: torch.Tensor) -> torch.Tensor:
        """Hybrid EP path: grouped FP8 GEMM for persistent + loop for non-persistent.

        Like K2.5 pattern: fused_moe_token_dispatch sorts tokens by local expert index,
        so expert_offsets cleanly splits persistent vs non-persistent slices.
        """
        import torch.distributed as dist
        from mgn_kernel import fused_moe_token_dispatch
        from batchgen.models.deepseek.deepseekv3.modeling_deepseek_v3 import scatter_weight_reduce_optimized

        num_tokens, hidden_size = x.shape
        device = x.device
        n_persistent = self.num_persistent_local_experts

        if num_tokens > self.num_tokens_per_rank:
            raise RuntimeError(
                f"MoE buffer overflow: num_tokens={num_tokens} > "
                f"num_tokens_per_rank={self.num_tokens_per_rank}"
            )

        # 1) AllGather
        all_tokens = torch.zeros(
            (self.world_size * self.num_tokens_per_rank, self.hidden_size),
            device=self.device, dtype=torch.bfloat16,
        )
        padded = torch.zeros(
            (self.num_tokens_per_rank, hidden_size),
            device=self.device, dtype=x.dtype,
        )
        if num_tokens > 0:
            padded[:num_tokens] = x

        with self.comm.change_state(enable=True):
            self.comm.all_gather(
                all_tokens, padded,
                stream=torch.cuda.default_stream(self.device),
            )

        # 2) Gate
        global_x = all_tokens
        topk_weight, topk_idx = self.gate(global_x)
        topk_idx = topk_idx.to(torch.int32)
        topk_weight = topk_weight.to(torch.float32)

        # 3) Dispatch tokens to ALL local experts (persistent + non-persistent)
        input_x, input_eids, global_indices, token_topk_pos, expert_counts, expert_offsets = \
            fused_moe_token_dispatch(
                global_x, topk_idx, self.token_idx, self.topk_pos,
                self.routed_expert_start_idx, self.routed_expert_end_idx,
            )

        offsets_cpu = expert_offsets.tolist()
        actual_total = offsets_cpu[-1]

        if actual_total == 0:
            # No tokens dispatched to this rank
            global_results = torch.zeros(
                self.num_tokens_per_rank * self.world_size, hidden_size,
                device=self.device, dtype=torch.bfloat16,
            )
        else:
            res = input_x.new_empty(actual_total, hidden_size, dtype=torch.bfloat16)

            # Phase 1: Grouped FP8 GEMM for persistent experts
            persistent_end = offsets_cpu[n_persistent]
            if n_persistent > 0 and persistent_end > 0:
                persistent_counts = expert_counts[:n_persistent]
                persistent_offsets = expert_offsets[:n_persistent + 1]
                persistent_res = self.grouped_dequant_moe_fp8(
                    input_x[:persistent_end], input_eids[:persistent_end],
                    persistent_counts, persistent_offsets,
                )
                res[:persistent_end] = persistent_res[:persistent_end]

            # Phase 2: Loop for non-persistent experts
            for local_e in range(n_persistent, self.experts_per_rank):
                start_off = offsets_cpu[local_e]
                end_off = offsets_cpu[local_e + 1]
                if start_off == end_off:
                    continue
                global_e = self.routed_expert_start_idx + local_e
                res[start_off:end_off] = self.experts[global_e](input_x[start_off:end_off])

            # 4) Scatter + weighted reduce
            global_results = scatter_weight_reduce_optimized(
                res, global_indices, token_topk_pos, topk_weight,
                self.num_tokens_per_rank * self.world_size, self.num_experts_per_tok,
            )
            global_results = global_results.to(torch.bfloat16)

        # 5) AllReduce
        with self.comm.change_state(enable=True):
            self.comm.all_reduce(
                global_results, op=dist.ReduceOp.SUM,
                stream=torch.cuda.default_stream(self.device),
            )

        # 6) Extract local slice
        start = self.rank * self.num_tokens_per_rank
        return global_results[start:start + num_tokens].to(x.dtype)


# ============================================================================
# Dense MLP (for first 3 layers)
# ============================================================================

class Glm5MLP(nn.Module):
    """Dense MLP for the first k layers (no MoE routing)."""

    def __init__(self, config: Glm5Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# Decoder Layer
# ============================================================================

class Glm5DecoderLayer(nn.Module):
    """Single transformer decoder layer."""

    def __init__(self, config: Glm5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.input_layernorm = Glm5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Glm5MLA(config, layer_idx)
        self.post_attention_layernorm = Glm5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Dense MLP for first k layers, MoE for rest
        is_dense = layer_idx < config.first_k_dense_replace
        if is_dense:
            self.mlp = Glm5MLP(config)
        else:
            self.mlp = Glm5MoE(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Pre-norm attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, present = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Pre-norm MoE/FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attn_weights, present


# ============================================================================
# Main Model
# ============================================================================

class Glm5Model(nn.Module):
    """GLM-5 transformer (no CausalLM wrapper verbosity)."""

    def __init__(self, config: Glm5Config):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Shared RoPE instance
        self._shared_rotary_emb = Glm5RotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.layers = nn.ModuleList(
            [Glm5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        # Assign shared RoPE
        for layer in self.layers:
            layer.self_attn.rotary_emb = self._shared_rotary_emb

        self.norm = Glm5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values is not None else None
            hidden_states, _, _ = layer(
                hidden_states, attention_mask, position_ids, past_kv, use_cache,
            )

        hidden_states = self.norm(hidden_states)
        return (hidden_states,)


class Glm5ForCausalLM(nn.Module):
    """GLM-5 model with language modeling head.

    Flat structure:
    - model.embed_tokens: nn.Embedding
    - model.layers: nn.ModuleList of Glm5DecoderLayer
    - model.norm: Glm5RMSNorm
    - lm_head: nn.Linear
    """

    def __init__(self, config: Glm5Config):
        super().__init__()
        self.config = config
        self.model = Glm5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits)
