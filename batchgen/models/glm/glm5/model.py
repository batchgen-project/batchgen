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

import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decode_utils import clamp_token_indices_to_seqlens
# GLM-5 uses BatchGen's internal config (a plain BaseModelConfig dataclass), not
# an HF transformers.PretrainedConfig — matching every other model (kimi, etc.).
# Imported under the historical name `Glm5Config` so the __init__ type hints below
# need no churn; only attribute reads are used, so any config object works.
from .config import GLM5Config as Glm5Config, dsa_layer_skips_topk


# ============================================================================
# Compact ragged MoE decode kernels (M1a-2). Started as a port of minimax-m25's
# padded [E, mtp, dim] path; the buffer layout is now compact (moe_ragged.py).
# ============================================================================
# The MoE decode path uses `dispatch_scatter_ragged` (moe_ragged.py) +
# `grouped_fp8_blockwise_*` + `reduce_weighted_scatter`. Falls back to
# `_triton_compute` only if the dispatch/reduce kernels aren't available at
# all; there is no padded-layout fallback inside the ragged path.

try:
    from batchgen.moe.grouped_fp8_blockwise_moe import (
        grouped_fp8_blockwise_fused_s1,
        grouped_fp8_blockwise_fused_s1_ptrs,
        grouped_fp8_blockwise_s3,
        grouped_fp8_blockwise_s3_ptrs,
        require_grouped_fp8_blockwise_ptr_kernels,
    )
    _GLM5_HAS_FP8_BLOCKWISE = True
except ImportError:
    _GLM5_HAS_FP8_BLOCKWISE = False

try:
    from batchgen.moe.dispatch_scatter_3d import (
        reduce_weighted_scatter,
        reduce_weighted_scatter_bf16_ordered,
    )
    _GLM5_HAS_DISPATCH_3D = True
except ImportError:
    _GLM5_HAS_DISPATCH_3D = False

from .moe_ragged import (
    GEMM_TILEM_AVG as _GLM5_MOE_GEMM_TILEM_AVG,
    act_quant_ragged as _glm5_act_quant_ragged,
    dispatch_scatter_ragged as _glm5_dispatch_scatter_ragged,
    make_quant_buffers as _glm5_make_quant_buffers,
    ragged_row_capacity as _glm5_ragged_row_capacity,
    require_ragged_kernels as _glm5_require_ragged_kernels,
)

# DEAD as of M1a-2 (compact ragged MoE dispatch): the dispatch/result buffers are
# no longer sized by a per-expert stride, so this knob controls nothing. It is
# still defined only because `batchgen_worker.py` imports it and forwards it as
# `Glm5MoEGraphBufferPool(base_mtp=...)`, which is likewise ignored. Delete both
# together in the follow-up that is allowed to touch the worker.
_GLM5_3D_MTP = int(os.environ.get("BATCHGEN_GLM5_3D_MTP", "4096"))
_GLM5_MOE_CUDA_GRAPH_ENV = "BATCHGEN_GLM5_MOE_CUDA_GRAPH"
_GLM5_MOE_GRAPH_COMPARE_ENV = "BATCHGEN_GLM5_MOE_GRAPH_COMPARE"
_GLM5_MOE_ROUTER_MODE_ENV = "BATCHGEN_GLM5_MOE_ROUTER_MODE"
_GLM5_PREFILL_GROUPED_TOKEN_WINDOW = 16_384
_GLM5_PREFILL_GROUPED_TILEM_AVG = 64


def _debug_flag_enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _glm5_moe_cuda_graph_required() -> bool:
    mode = _glm5_moe_debug_mode()
    if mode == "eager":
        return False
    if mode == "graph":
        return True
    return os.environ.get(_GLM5_MOE_CUDA_GRAPH_ENV, "0") == "1"


def _glm5_moe_debug_dict() -> dict:
    try:
        from batchgen.models.wrappers.attention import AttnWrapperBase
    except ImportError:
        return {}
    debug = getattr(AttnWrapperBase, "batchgen_debug", None) or {}
    return debug if isinstance(debug, dict) else {}


def _glm5_moe_debug_mode() -> Optional[str]:
    value = _glm5_moe_debug_dict().get("glm5_moe_mode")
    if not isinstance(value, str):
        return None
    mode = value.strip().lower()
    return mode if mode in {"graph", "eager"} else None


def _glm5_moe_router_mode() -> str:
    """Router GEMM implementation for eager GLM-5 MoE decode.

    Default ``custom`` keeps graph/eager on the row-stable GLM router kernel.
    ``cublas`` restores the historical eager router for trajectory A/B tests.
    """
    value = _glm5_moe_debug_dict().get("glm5_moe_router_mode")
    if not isinstance(value, str):
        value = os.environ.get(_GLM5_MOE_ROUTER_MODE_ENV, "")
    mode = value.strip().lower()
    return mode if mode in {"custom", "cublas"} else "custom"


def _record_glm5_moe_dispatch(
    path: str,
    *,
    layer_idx: int,
    bsz: int,
    reason: str,
) -> None:
    try:
        from batchgen.models.wrappers.attention import AttnWrapperBase
    except ImportError:
        return
    AttnWrapperBase.record_glm5_dispatch(
        kind="moe",
        path=path,
        layer_idx=layer_idx,
        bsz=bsz,
        reason=reason,
    )


def _glm5_moe_graph_compare_active() -> bool:
    if _glm5_moe_debug_mode() == "eager":
        return False
    debug = _glm5_moe_debug_dict()
    return (
        _debug_flag_enabled(debug.get("glm5_moe_graph_compare"))
        or os.environ.get(_GLM5_MOE_GRAPH_COMPARE_ENV, "0") == "1"
    )


def _glm5_moe_graph_compare_layer_enabled(layer_idx: int) -> bool:
    if not _glm5_moe_graph_compare_active():
        return False
    debug = _glm5_moe_debug_dict()
    layers = debug.get("glm5_moe_graph_compare_layers")
    if layers is None:
        layers = os.environ.get("BATCHGEN_GLM5_MOE_GRAPH_COMPARE_LAYERS", "3")
    if layers in ("all", "*"):
        return True
    if isinstance(layers, int):
        return layer_idx == layers
    if isinstance(layers, str):
        return str(layer_idx) in {
            item.strip() for item in layers.split(",") if item.strip()
        }
    if isinstance(layers, (list, tuple, set)):
        try:
            return layer_idx in {int(item) for item in layers}
        except (TypeError, ValueError):
            logging.warning(
                "Ignoring invalid glm5_moe_graph_compare_layers=%r; defaulting to layer 3",
                layers,
            )
            return layer_idx == 3
    return layer_idx == 3


def _glm5_moe_graph_compare_fail_on_mismatch() -> bool:
    debug = _glm5_moe_debug_dict()
    return _debug_flag_enabled(debug.get("glm5_moe_graph_compare_fail_on_mismatch"))


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
        self.max_seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        # Keep the cache in FP32. Each position's cos/sin is a transcendental
        # quantized once at cache build, then reused for every attention Q.K
        # across 78 layers and all decode steps. Casting down to BF16 here
        # bakes ~2^-7 rounding per (pos, dim) into every subsequent dot
        # product, which compounds over long prompts and has been traced to
        # repetition / immediate-EOS pathology on GLM-5-FP8. Reference impls
        # cache cos/sin in FP32 explicitly for numerical stability; the cast
        # to x.dtype happens in forward() at use time.
        del dtype
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def forward(self, x: torch.Tensor, seq_len: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cos_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        # Return FP32 cos/sin. PyTorch dtype promotion will upcast the BF16
        # query/key during rotate (t * cos, rotate_half(t) * sin), computing
        # the rotation in FP32 and casting back to BF16 at the downstream
        # assignment. Casting cos/sin down to x.dtype here would bake BF16
        # rounding into every position and defeat the FP32 cache.
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
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

# Resolve hadamard kernels once at import time (triggers JIT compilation)
try:
    from batchgen.other_kernels.hadamard_transform import hadamard_transform as _hadamard_cuda_fn
except (ImportError, Exception):
    _hadamard_cuda_fn = None

# Fused RoPE+Hadamard kernel — validated: 99/99 tests passed, 16.5x speedup over separate ops.
try:
    from batchgen.other_kernels.hadamard_transform import fused_rope_hadamard as _fused_rope_hadamard_fn
except (ImportError, Exception):
    _fused_rope_hadamard_fn = None

_hadamard_matrix_cache: Dict[Tuple, torch.Tensor] = {}


def _get_hadamard_matrix(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return cached Hadamard matrix H/sqrt(dim) via Sylvester construction."""
    key = (dim, device, dtype)
    if key not in _hadamard_matrix_cache:
        H = torch.tensor([[1.0]], device=device, dtype=dtype)
        while H.shape[0] < dim:
            H = torch.cat([torch.cat([H, H], dim=1),
                           torch.cat([H, -H], dim=1)], dim=0)
        _hadamard_matrix_cache[key] = H * (dim ** -0.5)
    return _hadamard_matrix_cache[key]


def _hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Hadamard transform with 1/sqrt(dim) scaling. x last dim must be power of 2."""
    if _hadamard_cuda_fn is not None:
        return _hadamard_cuda_fn(x.contiguous(), scale=x.shape[-1] ** -0.5)
    dim = x.shape[-1]
    assert dim & (dim - 1) == 0, f"Hadamard requires power-of-2 dim, got {dim}"
    H = _get_hadamard_matrix(dim, x.device, x.dtype)
    return x @ H


class Glm5Indexer(nn.Module):
    """GLM-5 DSA indexer.

    Uses MQA (Multi-Query Attention) pattern for scoring:
    - K is single-head: hidden_states -> wk [hidden_size -> head_dim=128] -> k_norm
      -> RoPE(first 64 dims) -> Hadamard transform -> cache
    - Q is multi-head: q_a -> wq_b [q_lora_rank -> n_heads*head_dim=4096] -> reshape
      -> RoPE(first 64 dims) -> Hadamard transform
    - Scoring: Q[n_heads, head_dim] @ K[1, head_dim]^T (K broadcast across heads)
    - Head gates: weights_proj[hidden_size -> n_heads] modulate per-head scores
    - Aggregate across heads -> top-K selection

    The K cached per token is only head_dim=128 (not n_heads*head_dim=4096).
    """

    def __init__(self, config: Glm5Config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.index_n_heads = config.index_n_heads   # 32
        self.index_head_dim = config.index_head_dim  # 128
        self.index_topk = config.index_topk          # 2048
        self.rope_head_dim = config.qk_rope_head_dim  # 64 — first 64 dims get RoPE
        self.softmax_scale = config.index_head_dim ** -0.5

        # K path: hidden -> wk -> k_norm -> RoPE -> Hadamard (MQA: single head, dim=128)
        self.wk = nn.Linear(config.hidden_size, config.index_head_dim, bias=False)
        self.k_norm = nn.LayerNorm(config.index_head_dim)  # Has weight + bias

        # Q path: q_a (from main attention) -> wq_b (multi-head, dim=4096) -> RoPE -> Hadamard
        self.wq_b = nn.Linear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            bias=False,
        )

        # Per-head importance scoring
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)

        # RoPE — assigned externally after construction (shares main attention's rotary_emb)
        self.rotary_emb: Optional[nn.Module] = None

    def _apply_rope_to_k(self, k: torch.Tensor, positions: torch.Tensor, max_seqlen: Optional[int] = None) -> torch.Tensor:
        """Apply RoPE to first rope_head_dim dims of K [batch, seq, head_dim].

        positions: [batch] or [batch, seq] integer position IDs.
        max_seqlen: if provided, avoids CPU-GPU sync from int(positions.max()).
        """
        k_rope = k[..., :self.rope_head_dim]
        k_nope = k[..., self.rope_head_dim:]
        seq_len = max_seqlen if max_seqlen is not None else int(positions.max()) + 1
        cos, sin = self.rotary_emb(k_rope, seq_len)
        # Index cos/sin by position: positions may be [batch] (decode) or [batch, seq] (prefill)
        cos = cos[positions]  # [batch, ...rope_dim*2]
        sin = sin[positions]
        if cos.dim() == 2:
            cos = cos.unsqueeze(1)  # [batch, 1, rope_dim*2]
            sin = sin.unsqueeze(1)
        k_rope = k_rope.unsqueeze(2)  # [batch, seq, 1, rope_dim]
        k_rope, _ = apply_rotary_pos_emb_interleaved(k_rope, k_rope, cos, sin)
        k_rope = k_rope.squeeze(2)
        return torch.cat([k_rope, k_nope], dim=-1)

    def _apply_rope_to_q(self, q: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to first rope_head_dim dims of Q [batch, n_heads, head_dim].

        positions: [batch] integer position IDs (decode only).
        """
        q_rope = q[..., :self.rope_head_dim]
        q_nope = q[..., self.rope_head_dim:]
        seq_len = int(positions.max()) + 1
        cos, sin = self.rotary_emb(q_rope.view(-1, 1, self.rope_head_dim), seq_len)
        cos = cos[positions].unsqueeze(1)  # [batch, 1, rope_dim*2]
        sin = sin[positions].unsqueeze(1)
        q_rope = q_rope.unsqueeze(2)  # [batch, n_heads, 1, rope_dim]
        q_rope, _ = apply_rotary_pos_emb_interleaved(q_rope, q_rope, cos, sin)
        q_rope = q_rope.squeeze(2)
        return torch.cat([q_rope, q_nope], dim=-1)

    def _fused_rope_hadamard_or_fallback(
        self, k: torch.Tensor, positions: torch.Tensor, max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """Fused interleaved RoPE + Hadamard, falling back to separate ops."""
        if _fused_rope_hadamard_fn is not None:
            seq_len = max_seqlen if max_seqlen is not None else int(positions.max()) + 1
            cos, sin = self.rotary_emb(k, seq_len)
            return _fused_rope_hadamard_fn(
                k.to(torch.bfloat16), cos.float(), sin.float(),
                positions.reshape(-1), scale=k.shape[-1] ** -0.5,
            )
        k = self._apply_rope_to_k(k, positions, max_seqlen=max_seqlen)
        return _hadamard_transform(k.to(torch.bfloat16)).to(k.dtype)

    def compute_indexer_kv(
        self, hidden_states: torch.Tensor, positions: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute indexer K for cache storage.

        Pipeline: wk -> k_norm -> RoPE(first 64 dims) -> Hadamard -> cache

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            positions: [batch, seq_len] or [seq_len] — token positions for RoPE.
                       If None, RoPE and Hadamard are skipped (backwards compat).
            max_seqlen: max sequence length (int) to avoid CPU-GPU sync in RoPE.

        Returns:
            indexer_k: [batch, seq_len, 1, head_dim] shaped for paged KV manager
                       head_dim=128 (single MQA head)
        """
        if hasattr(self, 'wk_scale'):
            from batchgen.attention.mla.fa3_backend import w8a16_gemm
            k = w8a16_gemm(self.wk.weight.data, self.wk_scale, hidden_states)
        else:
            k = self.wk(hidden_states)   # [batch, seq_len, head_dim=128]
        k = self.k_norm(k)

        if positions is not None and self.rotary_emb is not None:
            k = self._fused_rope_hadamard_or_fallback(k, positions, max_seqlen=max_seqlen)

        return k.unsqueeze(2)  # [batch, seq_len, 1, head_dim]

    def score_and_select(
        self,
        q_a: torch.Tensor,
        hidden_states: torch.Tensor,
        cached_k: torch.Tensor,
        cache_seqlens: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """Score cached tokens via MQA Q@K and select top-K.

        K is already RoPE'd + Hadamard'd in cache.
        Q gets: wq_b -> reshape -> RoPE -> Hadamard -> score against cached K.

        Args:
            q_a: [batch, 1, q_lora_rank] — q_a intermediate from main attention
            hidden_states: [batch, 1, hidden_size] — for head gate scoring
            cached_k: [batch, max_seqlen, head_dim] — gathered indexer K (128-dim, with RoPE+Hadamard)
            cache_seqlens: [batch] — valid lengths
            positions: [batch] or [batch, 1] — current token positions for Q RoPE
            max_seqlen: max sequence length (int) to avoid CPU-GPU sync

        Returns:
            top_k_indices: [batch, index_topk]
        """
        batch_size = q_a.shape[0]
        if max_seqlen is None:
            max_seqlen = cached_k.shape[1]

        # Q from shared q_a intermediate: [batch, 1, q_lora_rank] -> [batch, n_heads, head_dim]
        if hasattr(self, 'wq_b_scale'):
            from batchgen.attention.mla.fa3_backend import w8a16_gemm
            q = w8a16_gemm(self.wq_b.weight.data, self.wq_b_scale, q_a)
        else:
            q = self.wq_b(q_a)  # [batch, 1, n_heads * head_dim]
        q = q.view(batch_size, self.index_n_heads, self.index_head_dim)

        # Apply RoPE + Hadamard to Q (must match cached K processing)
        if positions is not None and self.rotary_emb is not None:
            if _fused_rope_hadamard_fn is not None:
                cos, sin = self.rotary_emb(q.view(-1, 1, self.rope_head_dim), max_seqlen)
                # Reshape [B, n_heads, 128] → [B*n_heads, 128], expand positions
                B = q.shape[0]
                q_flat = q.reshape(-1, self.index_head_dim)
                pos_expanded = positions.reshape(-1).repeat_interleave(self.index_n_heads)
                q = _fused_rope_hadamard_fn(
                    q_flat.to(torch.bfloat16), cos.float(), sin.float(),
                    pos_expanded, scale=self.index_head_dim ** -0.5,
                ).reshape(B, self.index_n_heads, self.index_head_dim).to(q.dtype)
            else:
                q = self._apply_rope_to_q(q, positions)
                q = _hadamard_transform(q.to(torch.bfloat16)).to(q.dtype)

        # MQA: K is [batch, max_seqlen, head_dim] — no per-head dim
        # Q @ K^T: [batch, n_heads, 1, head_dim] @ [batch, 1, head_dim, max_seqlen]
        q = q.unsqueeze(2)  # [batch, n_heads, 1, head_dim]
        cached_k_t = cached_k.transpose(1, 2).unsqueeze(1)  # [batch, 1, head_dim, max_seqlen]
        scores = torch.matmul(q, cached_k_t)  # [batch, n_heads, 1, max_seqlen]
        scores = scores.squeeze(2)  # [batch, n_heads, max_seqlen]

        # Head gate weighting (reference: weights * n_heads^-0.5 * softmax_scale)
        head_gates = self.weights_proj(hidden_states.squeeze(1))  # [batch, n_heads]
        head_gates = head_gates.float() * (self.index_n_heads ** -0.5) * self.softmax_scale
        scores = scores * head_gates.unsqueeze(-1)  # [batch, n_heads, max_seqlen]

        # Mask invalid positions
        position_indices = torch.arange(max_seqlen, device=scores.device).unsqueeze(0)
        mask = position_indices >= cache_seqlens.unsqueeze(1)
        scores.masked_fill_(mask.unsqueeze(1), float("-inf"))

        # Aggregate across heads and select top-K
        aggregated = scores.sum(dim=1)  # [batch, max_seqlen]
        # Clamp topk: index_topk is fixed hyperparameter, max_seqlen is Python int.
        # Per-sequence masking (-inf) handles varying lengths within the batch.
        effective_topk = min(self.index_topk, max_seqlen)
        _, top_k_indices = torch.topk(aggregated, effective_topk, dim=-1)
        return clamp_token_indices_to_seqlens(top_k_indices, cache_seqlens)

    def score_and_select_relu_gated(
        self,
        q_a: torch.Tensor,
        hidden_states: torch.Tensor,
        cached_k: torch.Tensor,
        cache_seqlens: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """ReLU-gated scoring: score = (relu(Q·K * softmax_scale) * head_gates).sum(heads).

        Two differences vs ``score_and_select``:

        1. ``F.relu`` is applied to per-head scores BEFORE head-weighting. Without
           ReLU, negative per-head scores cancel positive ones from other heads
           during the sum, distorting the top-K selection vs the training-time
           formulation of GLM-5's indexer.
        2. ``softmax_scale`` multiplies the Q·K product directly (not folded into
           ``head_gates``). Equivalent when ReLU is absent, but with ReLU the
           ordering matters because scaling the input changes which values clip
           to zero.

        Also masks ``positions >= cache_seqlens`` with ``-inf`` BEFORE the sum
        rather than after, so padded positions can't leak any post-ReLU energy
        into the aggregate.

        Args, returns: same as ``score_and_select``.
        """
        batch_size = q_a.shape[0]
        if max_seqlen is None:
            max_seqlen = cached_k.shape[1]

        # Q from shared q_a intermediate (same as score_and_select).
        if hasattr(self, 'wq_b_scale'):
            from batchgen.attention.mla.fa3_backend import w8a16_gemm
            q = w8a16_gemm(self.wq_b.weight.data, self.wq_b_scale, q_a)
        else:
            q = self.wq_b(q_a)
        q = q.view(batch_size, self.index_n_heads, self.index_head_dim)

        # RoPE + Hadamard on Q (same as score_and_select).
        if positions is not None and self.rotary_emb is not None:
            if _fused_rope_hadamard_fn is not None:
                cos, sin = self.rotary_emb(q.view(-1, 1, self.rope_head_dim), max_seqlen)
                B = q.shape[0]
                q_flat = q.reshape(-1, self.index_head_dim)
                pos_expanded = positions.reshape(-1).repeat_interleave(self.index_n_heads)
                q = _fused_rope_hadamard_fn(
                    q_flat.to(torch.bfloat16), cos.float(), sin.float(),
                    pos_expanded, scale=self.index_head_dim ** -0.5,
                ).reshape(B, self.index_n_heads, self.index_head_dim).to(q.dtype)
            else:
                q = self._apply_rope_to_q(q, positions)
                q = _hadamard_transform(q.to(torch.bfloat16)).to(q.dtype)

        # Score in float for numerical stability.
        q_f = q.float()                                     # [B, H, D]
        k_f = cached_k.float()                              # [B, T, D]
        # Q·K^T: [B, H, T] = einsum('bhd,btd->bht', q, k)
        scores = torch.einsum("bhd,btd->bht", q_f, k_f) * self.softmax_scale

        # Mask invalid positions BEFORE ReLU so they become 0 after F.relu(-inf).
        # Using a very-negative (not -inf) sentinel would still pass ReLU as 0; -inf
        # is the safe choice because subsequent ops are sums (not softmaxes).
        position_indices = torch.arange(max_seqlen, device=scores.device).unsqueeze(0)
        mask = position_indices >= cache_seqlens.unsqueeze(1)    # [B, T]
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))

        # Per-head ReLU clips negative scores to 0 before aggregation.
        scores = F.relu(scores)

        # Head gates: weights_proj(hidden) * n_heads^-0.5 (softmax_scale already folded in above).
        head_gates = self.weights_proj(hidden_states.squeeze(1)).float()  # [B, H]
        head_gates = head_gates * (self.index_n_heads ** -0.5)

        # Weighted sum over heads: [B, T]
        aggregated = torch.einsum("bht,bh->bt", scores, head_gates)

        effective_topk = min(self.index_topk, max_seqlen)
        _, top_k_indices = torch.topk(aggregated, effective_topk, dim=-1)
        return clamp_token_indices_to_seqlens(top_k_indices, cache_seqlens)

    def score_and_select_paged(
        self,
        q_a: torch.Tensor,
        hidden_states: torch.Tensor,
        indexer_blocked_k: torch.Tensor,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        indexer_kv_manager,
        page_size: int = 64,
        positions: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> torch.Tensor:
        """Score cached tokens from paged cache and select top-K.

        Gathers indexer K from paged cache into contiguous tensor using
        cached gather indices (reused across layers within a decode step),
        then delegates to score_and_select.

        Args:
            q_a: [batch, 1, q_lora_rank] — q_a intermediate from main attention
            hidden_states: [batch, 1, hidden_size] — for head gate scoring
            indexer_blocked_k: [num_pages, page_size, 1, head_dim] — paged indexer cache
            block_table: [batch, max_num_pages_per_seq] — page mapping
            cache_seqlens: [batch] — valid lengths
            indexer_kv_manager: paged KV manager providing page-table versioning
            page_size: tokens per page
            positions: [batch] — current token positions for Q RoPE
            max_seqlen: max sequence length (int) to avoid CPU-GPU sync

        Returns:
            top_k_indices: [batch, index_topk] — absolute token positions
        """
        batch_size = block_table.shape[0]
        if max_seqlen is None:
            raise RuntimeError("GLM-5 DSA paged scoring requires explicit max_seqlen")
        num_k_heads = indexer_blocked_k.shape[2]
        k_head_dim = indexer_blocked_k.shape[3]

        from batchgen_kernels.attention.dsa import fused_dense_paged_gather

        # Gather dense logical range [0, max_seqlen) without building token indices.
        gathered = fused_dense_paged_gather(
            indexer_blocked_k,
            block_table,
            max_seqlen,
            page_size,
        ).view(batch_size, max_seqlen, num_k_heads, k_head_dim)

        gathered_k = gathered.squeeze(2)

        # WP4: Fused scoring pipeline (CUDA WGMMA wq_b + RoPE + Hadamard + scoring + topk)
        if hasattr(self, '_fused_score_weights') and self._fused_score_weights is not None:
            from batchgen_kernels.attention.dsa.fused_indexer_score import fused_score_pipeline
            # Get RoPE cos/sin tables
            seq_len = max_seqlen if max_seqlen is not None else int(positions.max()) + 1
            cos, sin = self.rotary_emb(
                gathered_k[:1, :1, :self.rope_head_dim],  # dummy for dtype/device
                seq_len,
            )
            top_k_indices, _ = fused_score_pipeline(
                q_a=q_a.squeeze(1),                      # [B, q_lora_rank]
                hidden_states=hidden_states.squeeze(1),   # [B, hidden_size]
                cached_k=gathered_k,                      # [B, max_seqlen, 128]
                cache_seqlens=cache_seqlens.int(),
                wq_b_weights=self._fused_score_weights,
                weights_proj_weight=self.weights_proj.weight.data,  # [32, 6144]
                cos_table=cos.to(torch.bfloat16),
                sin_table=sin.to(torch.bfloat16),
                positions=positions,
                module=self._fused_score_module,
                n_heads=self.index_n_heads,
                head_dim=self.index_head_dim,
                rope_dim=self.rope_head_dim,
                topk=min(self.index_topk, max_seqlen),
            )
            return clamp_token_indices_to_seqlens(top_k_indices, cache_seqlens)
        else:
            raise RuntimeError(
                f"[layer {self.layer_idx}] GLM-5 DSA selector requires WP4 fused "
                "indexer scoring; PyTorch fallback is disabled"
            )


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

        # DSA indexer — structurally absent when config.use_dense_mla is True
        # (the `indexer` attribute is then NOT set, so hasattr(...) routes to the
        # dense-MLA path — unchanged for DeepSeek-V3 / Kimi style configs).
        # For DSA configs the attribute always exists, but GLM-5.2 "shared" layers
        # carry no indexer weights: they reuse the previous full layer's top-k
        # indices, so self.indexer is None on those layers (GLM-5: never a shared
        # layer, so indexer is built on every layer — bit-identical).
        self.skip_topk = dsa_layer_skips_topk(config, layer_idx)
        self.next_skip_topk = dsa_layer_skips_topk(config, layer_idx + 1)
        if not getattr(config, "use_dense_mla", False):
            self.indexer = None if self.skip_topk else Glm5Indexer(config, layer_idx)

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


class _Glm5ExpertPlaceholder:
    """Lightweight placeholder for expert slots in Glm5MoE.

    Avoids creating 19200 nn.Module objects during model init. Replaced by
    GLM5ExpertWrapper during _config_expert_module().
    """
    pass


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
        """Route tokens to experts (DeepSeek-V3 noaux_tc semantics).

        The `e_score_correction_bias` is used for SELECTION only; the
        returned top-K weights are the RAW sigmoid scores at the selected
        indices, not the biased values. This matches the dedicated CUDA
        `gate_sigmoid_topk_kernel` used by the decode path and the HF reference
        modeling for DeepSeek-V3 / glm_moe_dsa. Using biased
        scores as weights (the previous behavior) shifts the expert mixing
        coefficients and injects drift at every MoE layer, which for GLM-5
        shows up as a small bias at the final-position logits.

        Returns:
            topk_weights: [batch*seq, num_experts_per_tok]
            topk_indices: [batch*seq, num_experts_per_tok]
        """
        # Router logits in FP32 — HF computes this GEMM in FP32 explicitly
        # (GlmMoeDsaTopkRouter.forward does F.linear(x.type(fp32), w.type(fp32))).
        # bf16 router_logits shifts the sigmoid+bias decision boundary enough to
        # re-order top-K expert selection for scores within ~1% of each other,
        # and that drift compounds across 75 MoE layers.
        logits = F.linear(hidden_states.float(), self.weight.float())  # [bsz_seq, num_experts]
        scores = torch.sigmoid(logits)

        # Bias is for SELECTION only — do not mutate `scores` itself
        biased = scores + self.e_score_correction_bias.float().unsqueeze(0)

        # Simple top-K on biased scores (n_group=1, no group-based routing)
        _, topk_indices = torch.topk(biased, k=self.num_experts_per_tok, dim=-1)

        # Gather RAW (un-biased) sigmoid scores at the selected indices
        topk_weights = scores.gather(-1, topk_indices)

        # Normalize over raw weights. The `+ 1e-20` matches the dedicated CUDA
        # gate_sigmoid_topk_kernel (gate_sigmoid_topk.cu:119) for byte-parity
        # between prefill (this Python path) and decode (CUDA kernel).
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        # Scale
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights, topk_indices


# ============================================================================
# MoE Layer (Prefill)
# ============================================================================


# ============================================================================
# Scatter + Weighted Reduce (Triton kernel, copied from DeepSeek V3)
# ============================================================================

import triton
import triton.language as tl


@triton.jit
def _scatter_weight_reduce_kernel(
    res_ptr, nnz_indices_ptr, topk_weight_ptr, output_ptr,
    num_tokens, num_experts_per_tok, hidden_size, nnz,
    BLOCK_SIZE_H: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if token_idx >= num_tokens:
        return
    h_offset = tl.program_id(1) * BLOCK_SIZE_H
    h_indices = h_offset + tl.arange(0, BLOCK_SIZE_H)
    h_mask = h_indices < hidden_size
    accumulator = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)
    for k in range(num_experts_per_tok):
        mapping_offset = token_idx * num_experts_per_tok + k
        nnz_idx = tl.load(nnz_indices_ptr + mapping_offset)
        is_valid = (nnz_idx >= 0) & (nnz_idx < nnz)
        weight = tl.load(topk_weight_ptr + mapping_offset)
        safe_nnz_idx = tl.where(is_valid, nnz_idx, 0)
        res_offset = safe_nnz_idx * hidden_size + h_indices
        load_mask = h_mask & is_valid
        res_vals = tl.load(res_ptr + res_offset, mask=load_mask, other=0.0)
        res_vals_fp32 = res_vals.to(tl.float32)
        weighted = tl.where(is_valid, res_vals_fp32 * weight, 0.0)
        accumulator += weighted
    output_offset = token_idx * hidden_size + h_indices
    tl.store(output_ptr + output_offset, accumulator, mask=h_mask)


def _build_inverse_mapping(global_indices, token_topk_pos, num_tokens, num_experts_per_tok):
    mapping = torch.full((num_tokens, num_experts_per_tok), -1,
                         dtype=torch.int64, device=global_indices.device)
    if global_indices.numel() == 0:
        return mapping
    mapping[global_indices, token_topk_pos] = torch.arange(
        len(global_indices), dtype=torch.int64, device=global_indices.device)
    return mapping


def scatter_weight_reduce_optimized(res, global_indices, token_topk_pos,
                                    topk_weight, num_tokens, num_experts_per_tok):
    assert topk_weight.dtype == torch.float32
    nnz, hidden_size = res.shape
    if nnz == 0:
        return torch.zeros((num_tokens, hidden_size), device=res.device, dtype=torch.float32)
    nnz_indices = _build_inverse_mapping(
        global_indices[:nnz], token_topk_pos[:nnz], num_tokens, num_experts_per_tok)
    output = torch.zeros((num_tokens, hidden_size), device=res.device, dtype=torch.float32)
    if num_tokens == 0:
        return output
    BLOCK_SIZE_H = min(triton.next_power_of_2(hidden_size), 256)
    grid = (num_tokens, triton.cdiv(hidden_size, BLOCK_SIZE_H))
    _scatter_weight_reduce_kernel[grid](
        res, nnz_indices, topk_weight, output,
        num_tokens, num_experts_per_tok, hidden_size, nnz,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
    )
    return output


# ============================================================================
# Compact ragged MoE Buffer Manager (M1a-2)
# ============================================================================

class Glm5MoE3DBuffers:
    """Pre-allocated buffers for GLM-5 MoE decode on the compact ragged layout.

    One instance per model, shared across all 75 MoE layers.

    Was the K2.5 per-expert 3D slot table ``[E_local, mtp, H]`` (1.51 GiB at
    mtp=2048 / E_local=32 for dispatch + result). M1a-2 replaces the per-expert
    stride with one compact row space of ``capacity`` rows, where

        capacity = round_up(max_global_bsz * topk + E_local * 63, 128)

    is the *total* worst case: every routed (token, expert) pair is at most one
    row, plus at most 63 rows per segment of start alignment. Because that bound
    is static there is no per-expert capacity left to overflow — see
    :meth:`resize_if_needed`, which asserts instead of regrowing.

    Layout contract (see ``moe_ragged.py``): expert ``e`` owns rows
    ``[cu_seqlens[e], cu_seqlens[e] + expert_counts[e])``; ``cu_seqlens`` is
    written on device by ``dispatch_scatter_ragged`` with 64-row-aligned starts
    so the grouped GEMM can index ``x_scale`` in the same row space.
    """

    def __init__(
        self,
        E_local: int,
        max_global_bsz: int,
        H: int,
        N_inter: int,
        topk: int,
        num_tokens_per_rank: int,
        device: torch.device,
    ):
        self.E_local = E_local
        self.H = H
        self.N_inter = N_inter
        self.topk = topk
        self.max_global_bsz = max_global_bsz
        self.num_tokens_per_rank = num_tokens_per_rank
        self.device = device

        NK = max_global_bsz * topk
        self.capacity = _glm5_ragged_row_capacity(max_global_bsz, topk, E_local)

        self.all_tokens = torch.zeros(max_global_bsz, H, dtype=torch.bfloat16, device=device)
        self.padded = torch.zeros(num_tokens_per_rank, H, dtype=torch.bfloat16, device=device)
        self.expert_counts = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.expert_counters = torch.zeros(E_local, dtype=torch.int32, device=device)
        self.cu_seqlens = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
        self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=device)
        self.result_buffer = torch.empty(max_global_bsz, H, dtype=torch.bfloat16, device=device)
        self.local_result_buffer = torch.empty(
            num_tokens_per_rank, H, dtype=torch.bfloat16, device=device
        )
        self._alloc_row_buffers()

        total_bytes = 0
        for t in (self.all_tokens, self.padded, self.expert_counts, self.expert_counters,
                  self.cu_seqlens, self.topk_pos, self.dispatched_x, self.intermediate,
                  self.expert_out, self.result_buffer, self.local_result_buffer,
                  self.x_fp8, self.x_scale,
                  self.inter_fp8, self.inter_scale):
            total_bytes += t.nelement() * t.element_size()
        logging.info(
            f"[Glm5MoE3DBuffers] ragged: E_local={E_local}, capacity={self.capacity} rows "
            f"(NK={NK} + align pad), H={H}, N_inter={N_inter}, "
            f"total={total_bytes / (1024**3):.2f} GiB"
        )

    def _alloc_row_buffers(self):
        """(Re)allocate everything indexed by the compact row space."""
        cap, H, N_inter, device = self.capacity, self.H, self.N_inter, self.device
        self.dispatched_x = torch.zeros(cap, H, dtype=torch.bfloat16, device=device)
        self.intermediate = torch.empty(cap, N_inter, dtype=torch.bfloat16, device=device)
        self.expert_out = torch.zeros(cap, H, dtype=torch.bfloat16, device=device)
        # FP8 staging for S1/S3. Persistent rather than per-call `torch.empty`
        # so nothing of this size is allocated inside a CUDA-graph capture, and
        # so the zero-init of the scale buffers happens exactly once (the GEMM
        # TMA-loads whole M-tiles, including alignment holes act_quant_ragged
        # never writes; a stale finite scale there multiplies a hardware
        # zero-filled activation row, an uninitialised NaN would not).
        self.x_fp8, self.x_scale = _glm5_make_quant_buffers(cap, H, device)
        self.inter_fp8, self.inter_scale = _glm5_make_quant_buffers(cap, N_inter, device)

    def resize_if_needed(self, global_bsz: int, num_tokens_per_rank: int = None):
        """Resize the comm/routing buffers, and the compact row space with them.

        The 3D path's per-expert `grew_mtp` regrow is gone — the compact layout
        has no per-expert capacity to overflow. What remains is a job-boundary
        resize: the compact bound is a pure function of
        ``(max_global_bsz, topk, E_local)``, so it can only change when the
        planner raises the batch between jobs, which is the same event that
        already reallocates the comm buffers. A *dispatch-time* overflow is
        still a hard failure (TORCH_CHECK inside ``dispatch_scatter_ragged``).

        NOTE vs m1a2_spec.md C5, which asked for a hard assert here on the
        grounds that the total bound is static: it is static only within a job.
        ``Glm5MoE.set_num_tokens_per_rank`` (called per batch job from
        ``batchgen_worker.py``) raises ``global_bsz``, so an assert would turn
        the documented n32 -> n128 -> n2048 job sequence into a crash.
        """
        grew_comm = global_bsz > self.max_global_bsz
        # `padded` (per-rank all-gather send buffer) is sized at creation-time
        # num_tokens_per_rank; the class-level buffer outlives the batch job, so
        # a later job with more per-rank tokens overruns it (bug_log 2026-08-13:
        # `padded[:16]` on a 2-row buffer after an n32 -> n128 job sequence).
        grew_padded = (
            num_tokens_per_rank is not None
            and num_tokens_per_rank > self.padded.shape[0]
        )
        grew_local_result = (
            num_tokens_per_rank is not None
            and num_tokens_per_rank > self.local_result_buffer.shape[0]
        )

        if not grew_comm and not grew_padded and not grew_local_result:
            return

        if grew_padded or grew_local_result:
            logging.info(
                f"[Glm5MoE3DBuffers] Resizing local buffers: "
                f"padded={self.padded.shape[0]}, result={self.local_result_buffer.shape[0]} "
                f"-> {num_tokens_per_rank}")
            if grew_padded:
                self.padded = torch.zeros(
                    num_tokens_per_rank,
                    self.H,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
            self.local_result_buffer = torch.empty(
                num_tokens_per_rank,
                self.H,
                dtype=torch.bfloat16,
                device=self.device,
            )

        if grew_comm:
            needed = _glm5_ragged_row_capacity(global_bsz, self.topk, self.E_local)
            logging.info(
                f"[Glm5MoE3DBuffers] Resizing comm buffers: {self.max_global_bsz} -> {global_bsz}")
            self.max_global_bsz = global_bsz
            NK = global_bsz * self.topk
            self.all_tokens = torch.zeros(global_bsz, self.H, dtype=torch.bfloat16, device=self.device)
            self.topk_pos = torch.full((NK,), -1, dtype=torch.int32, device=self.device)
            self.result_buffer = torch.empty(global_bsz, self.H, dtype=torch.bfloat16, device=self.device)
            if needed > self.capacity:
                logging.warning(
                    f"[Glm5MoE3DBuffers] Regrowing compact MoE rows: "
                    f"{self.capacity} -> {needed} (global_bsz={global_bsz}). Sizing "
                    f"the first job at the planner's max batch avoids this realloc.")
                self.capacity = needed
                self._alloc_row_buffers()


class Glm5PrefillMoEBuffers:
    """Bounded compact workspace shared by all grouped prefill MoE layers."""

    def __init__(
        self,
        num_experts: int,
        token_window: int,
        hidden_size: int,
        intermediate_size: int,
        topk: int,
        device: torch.device,
    ):
        self.num_experts = int(num_experts)
        self.token_window = int(token_window)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.topk = int(topk)
        self.device = device
        self.capacity = _glm5_ragged_row_capacity(
            self.token_window, self.topk, self.num_experts
        )

        self.expert_counts = torch.zeros(
            self.num_experts, dtype=torch.int32, device=device
        )
        self.expert_counters = torch.zeros_like(self.expert_counts)
        self.cu_seqlens = torch.zeros(
            self.num_experts + 1, dtype=torch.int32, device=device
        )
        self.topk_pos = torch.empty(
            self.token_window * self.topk, dtype=torch.int32, device=device
        )
        self.dispatched_x = torch.empty(
            self.capacity, self.hidden_size, dtype=torch.bfloat16, device=device
        )
        self.intermediate = torch.empty(
            self.capacity,
            self.intermediate_size,
            dtype=torch.bfloat16,
            device=device,
        )
        self.expert_out = torch.empty_like(self.dispatched_x)
        self.x_fp8, self.x_scale = _glm5_make_quant_buffers(
            self.capacity, self.hidden_size, device
        )
        self.inter_fp8, self.inter_scale = _glm5_make_quant_buffers(
            self.capacity, self.intermediate_size, device
        )

        # Device descriptor preparation still runs for every token window, but
        # its storage is persistent: no allocator calls occur in the MoE loop.
        self.s1_tma_desc = torch.empty(
            self.num_experts * 6, 128, dtype=torch.uint8, device=device
        )
        self.s3_tma_desc = torch.empty(
            self.num_experts * 4, 128, dtype=torch.uint8, device=device
        )
        self.tiles = torch.empty(
            self.num_experts, dtype=torch.int32, device=device
        )
        self.cu_tiles = torch.empty(
            self.num_experts + 1, dtype=torch.int32, device=device
        )

        tensors = (
            self.expert_counts,
            self.expert_counters,
            self.cu_seqlens,
            self.topk_pos,
            self.dispatched_x,
            self.intermediate,
            self.expert_out,
            self.x_fp8,
            self.x_scale,
            self.inter_fp8,
            self.inter_scale,
            self.s1_tma_desc,
            self.s3_tma_desc,
            self.tiles,
            self.cu_tiles,
        )
        total_bytes = sum(t.nelement() * t.element_size() for t in tensors)
        logging.info(
            "[GLM5_GROUPED_PREFILL] workspace: E=%d window=%d capacity=%d "
            "H=%d N=%d total=%.2f GiB",
            self.num_experts,
            self.token_window,
            self.capacity,
            self.hidden_size,
            self.intermediate_size,
            total_bytes / (1024**3),
        )


# ============================================================================
# MoE Layer (Decode with EP) — standalone nn.Module (K2.5 pattern)
# ============================================================================

def _glm5_moe_3d_blockwise_supported(
    experts_per_rank: int,
    num_persistent_local_experts: int,
    enable_ep_offloading: bool,
) -> bool:
    """3D blockwise MoE requires every local routed expert resident on GPU."""
    return (
        int(num_persistent_local_experts) == int(experts_per_rank)
        and not bool(enable_ep_offloading)
    )


class Glm5MoE(nn.Module):
    """GLM-5 MoE layer (unified prefill + EP decode).

    Standalone module (no MoEBase inheritance). Follows K2.5 pattern:
        forward() dispatches to _forward_prefill() or _forward_decode() by config.phase.

    Decode path:
        gate()                       — sigmoid + bias + topk (CUDA kernel with fallback)
        expert_compute_persistent()  — WGMMA FP8 pipeline (zero CPU-GPU sync)
        expert_compute_mixed()       — WGMMA persistent + loop non-persistent (one .tolist())
        shared_expert_forward()      — BF16 SwiGLU shared expert
    """

    # Shared across all instances (all 75 MoE layers)
    _wgmma_modules = None       # (wgmma_mod, fast_mod, dr_mod)
    _buf = None                 # WGMMAMoEBuffers instance (unified: GEMM + comm buffers)
    _wgmma_next_layer_id = 0    # Counter for layer registration

    # K2.5 3D-MoE path (minimax parity).
    _3d_buf: Optional[Glm5MoE3DBuffers] = None
    _warned_k25_path = False
    _warned_gemm_3d = False
    _warned_partial_3d_disabled = False
    _rank_token_counts: Optional[torch.Tensor] = None  # [world_size] real token count per rank — mask padding before dispatch
    _routed_moe_stream: Optional[torch.cuda.Stream] = None

    # Pure-DP grouped prefill. The 512-slot core-engine ring owns two complete
    # 256-expert FP8 layers; this class only owns bounded activation/workspace
    # storage and one reusable pointer table.
    _prefill_buf: Optional[Glm5PrefillMoEBuffers] = None
    _prefill_ptrs_pinned: Optional[torch.Tensor] = None
    _prefill_ptrs_dev: Optional[torch.Tensor] = None
    _prefill_ring_pending = None  # (completion event, module keys, core engine)
    _prefill_shared_pending = None  # (completion event, module key, core engine)
    _prefill_grouped_logged = False

    def __init__(self, config: Glm5Config, layer_idx: int = -1, comm=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.layer_idx = layer_idx
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

        self.device = torch.device("cuda", self.rank % torch.cuda.device_count())
        self.num_tokens_per_rank = None
        self._gate_w_fp32 = None
        if Glm5MoE._routed_moe_stream is None and torch.cuda.is_available():
            Glm5MoE._routed_moe_stream = torch.cuda.Stream(device=self.device)
        self.enable_ep_offloading = False
        self.num_persistent_local_experts = self.experts_per_rank

        self.use_wgmma_fp8 = os.environ.get("BATCHGEN_USE_WGMMA_FP8", "0") == "1"
        self.use_3d_moe = True

        self.gate = Glm5MoEGate(config)
        self.experts = [_Glm5ExpertPlaceholder() for _ in range(self.total_experts)]
        self.shared_experts = Glm5Expert(config.hidden_size, config.moe_intermediate_size)
        self._fp8_blockwise_ready = False
        self._moe_cuda_graph_manager = None
        self._moe_cuda_graph_segment_name = None
        self._moe_cuda_graph_segment = None
        self._moe_cuda_graph_bucketing = None
        self._moe_cuda_graph_required = False
        self._prefill_grouped_enabled = False
        self._prefill_prepared_keys = None
        self._prefill_weight_prototypes = None
        self._prefill_release_event = None
        self._prefill_shared_key = None
        self._prefill_shared_release_event = None

    @classmethod
    def init_prefill_grouped_buffers(
        cls,
        config: Glm5Config,
        device: torch.device,
    ) -> None:
        """Allocate and compile the fixed-window grouped prefill workspace."""
        if not _GLM5_HAS_FP8_BLOCKWISE or not _GLM5_HAS_DISPATCH_3D:
            raise RuntimeError(
                "GLM-5 grouped prefill requires FP8 grouped GEMM and compact "
                "dispatch/reduce kernels"
            )
        require_grouped_fp8_blockwise_ptr_kernels()
        _glm5_require_ragged_kernels()
        cls._prefill_buf = Glm5PrefillMoEBuffers(
            num_experts=config.n_routed_experts,
            token_window=_GLM5_PREFILL_GROUPED_TOKEN_WINDOW,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            topk=config.num_experts_per_tok,
            device=device,
        )
        cls._prefill_ptrs_pinned = torch.empty(
            6, config.n_routed_experts, dtype=torch.int64, pin_memory=True
        )
        cls._prefill_ptrs_dev = torch.empty(
            6, config.n_routed_experts, dtype=torch.int64, device=device
        )
        cls._prefill_ring_pending = None
        cls._prefill_shared_pending = None
        cls._prefill_grouped_logged = False

    @classmethod
    def retire_prefill_grouped_weights(cls) -> None:
        """Release the previous layer's ring slots after its CUDA event."""
        pending = cls._prefill_ring_pending
        if pending is not None:
            event, keys, core_engine = pending
            event.synchronize()
            for key in keys:
                core_engine.free_weights_buffer(key)
            cls._prefill_ring_pending = None
        shared_pending = cls._prefill_shared_pending
        if shared_pending is not None:
            event, key, core_engine = shared_pending
            event.synchronize()
            core_engine.free_weights_buffer(key)
            cls._prefill_shared_pending = None

    @classmethod
    def reset_prefill_grouped_state(cls) -> None:
        """Retire live slots, then drop prefill-only HBM before a phase flip."""
        cls.retire_prefill_grouped_weights()
        cls._prefill_buf = None
        cls._prefill_ptrs_pinned = None
        cls._prefill_ptrs_dev = None
        cls._prefill_ring_pending = None
        cls._prefill_shared_pending = None
        cls._prefill_grouped_logged = False

    # ── Token count management (called by PSM) ──

    @classmethod
    def init_buffer_manager(cls, num_tokens_per_rank: int, world_size: int,
                            hidden_size: int, device: torch.device):
        """Initialize shared WGMMAMoEBuffers. Called once by PSM.

        WGMMAMoEBuffers is the unified buffer manager — it contains both
        GEMM buffers (act_buf, gate_out, etc.) and comm buffers (all_tokens,
        padded_hidden_states) that MoEBufferManager used to provide separately.
        Actual buffer creation is deferred to _lazy_init_wgmma_bufs() since it
        needs weight pointers from the first layer's init().
        """
        # Store params for deferred creation in _lazy_init_wgmma_bufs
        cls._deferred_buf_params = (num_tokens_per_rank, world_size, hidden_size, device)

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

        # K2.5 3D-MoE buffer allocation (shared across all 75 MoE layers).
        if (
            self.use_3d_moe
            and _GLM5_HAS_DISPATCH_3D
            and Glm5MoE._3d_buf is None
            and _glm5_moe_3d_blockwise_supported(
                self.experts_per_rank,
                self.num_persistent_local_experts,
                self.enable_ep_offloading,
            )
        ):
            Glm5MoE._3d_buf = Glm5MoE3DBuffers(
                E_local=self.experts_per_rank,
                max_global_bsz=global_num_tokens,
                H=self.hidden_size,
                N_inter=self.config.moe_intermediate_size,
                topk=K,
                num_tokens_per_rank=num_tokens_per_rank,
                device=self.device,
            )

    def set_num_tokens_per_rank(self, num_tokens_per_rank: int):
        if num_tokens_per_rank == self.num_tokens_per_rank:
            return
        self.num_tokens_per_rank = num_tokens_per_rank
        if hasattr(self, 'max_num_tokens_per_rank') and num_tokens_per_rank > self.max_num_tokens_per_rank:
            self.max_num_tokens_per_rank = num_tokens_per_rank
        global_num_tokens = num_tokens_per_rank * self.world_size
        K = self.num_experts_per_tok
        self.token_idx = torch.arange(
            global_num_tokens, dtype=torch.int32, device=self.device
        ).repeat_interleave(K)
        self.topk_pos = torch.arange(
            K, dtype=torch.int32, device=self.device
        ).repeat(global_num_tokens)

        # Resize 3D buffers if global token count exceeded
        if self.use_3d_moe and Glm5MoE._3d_buf is not None:
            buf = Glm5MoE._3d_buf
            buf.resize_if_needed(global_num_tokens)
            # The send buffer MUST match num_tokens_per_rank exactly — NCCL
            # all_gather sends `input.numel()` elements per rank. If padded
            # stays at its initial size (e.g. 128) while num_tokens_per_rank
            # shrinks to 8, every rank sends 128*H elements instead of 8*H,
            # and the `all_tokens[:num_global]` slice ends up holding only
            # rank 0's contribution (padded with zeros) — ranks 1..15's
            # tokens land past the slice end and are never read, collapsing
            # MoE compute to near-zero input. Mirrors MiniMaxM25MoE.
            if buf.padded.shape[0] != num_tokens_per_rank:
                buf.padded = torch.zeros(
                    num_tokens_per_rank, buf.H,
                    dtype=torch.bfloat16, device=buf.device,
                )
                buf.local_result_buffer = torch.empty(
                    num_tokens_per_rank,
                    buf.H,
                    dtype=torch.bfloat16,
                    device=buf.device,
                )
                buf.num_tokens_per_rank = num_tokens_per_rank

    # ── Weight pointer setup (called by PSM) ──

    def init(self, micro_batch_size):
        """Collect FP8 weight tensors and build pointer arrays for grouped GEMM."""
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

        # WGMMA: build shared modules (once across all layers)
        if self.use_wgmma_fp8:
            if Glm5MoE._wgmma_modules is None:
                from batchgen.moe.fp8_wgmma_pipeline import build_all_modules
                Glm5MoE._wgmma_modules = build_all_modules()
            self._wgmma_layer_id = Glm5MoE._wgmma_next_layer_id
            Glm5MoE._wgmma_next_layer_id += 1

        # K2.5 3D-MoE weight stacking (per-layer) — mirror
        # MiniMaxM25MoE._init_fp8_blockwise_weights at model.py:874-926.
        if self.use_3d_moe and _GLM5_HAS_FP8_BLOCKWISE:
            if not _glm5_moe_3d_blockwise_supported(
                self.experts_per_rank,
                n_persistent,
                self.enable_ep_offloading,
            ):
                self._fp8_blockwise_ready = False
                if not Glm5MoE._warned_partial_3d_disabled:
                    logging.warning(
                        "[Glm5MoE] 3D FP8 MoE disabled: persistent experts "
                        "%s/%s, ep_offloading=%s. Falling back to mixed expert "
                        "decode; GLM-5 MoE/whole-model CUDA graph requires all "
                        "local experts to be persistent.",
                        n_persistent,
                        self.experts_per_rank,
                        self.enable_ep_offloading,
                    )
                    Glm5MoE._warned_partial_3d_disabled = True
                return
            self._init_fp8_blockwise_weights()

    def _init_fp8_blockwise_weights(self):
        """Stack per-expert FP8 weights into 3D tensors for blockwise GEMM.

        Mirrors MiniMaxM25MoE._init_fp8_blockwise_weights (model.py:874-926).
        GLM-5 shapes: hidden_size=6144, moe_intermediate_size=2048, both
        divisible by 128 → fits CuTe alignment.
        """
        E = self.experts_per_rank
        K = self.hidden_size                       # 6144
        N = self.config.moe_intermediate_size      # 2048
        scale_block = 128

        k_blocks = K // scale_block
        n_blocks = N // scale_block
        k_blocks_pad4 = (k_blocks + 3) // 4 * 4
        n_blocks_pad4 = (n_blocks + 3) // 4 * 4

        if not (
            len(self.gate_list) == E
            and len(self.up_list) == E
            and len(self.down_list) == E
            and len(self.gate_scale_list) == E
            and len(self.up_scale_list) == E
            and len(self.down_scale_list) == E
        ):
            raise RuntimeError(
                "GLM-5 3D FP8 MoE requires all local experts to be resident "
                f"before stacking weights; experts_per_rank={E}, "
                f"gate/up/down={len(self.gate_list)}/{len(self.up_list)}/{len(self.down_list)}, "
                "use mixed expert decode for partial-persistent single-node configs."
            )

        # GLM-5 has ~4x larger MoE projections than minimax (6144x2048 vs
        # 3072x1536). Each MoE-layer weight has THREE live references:
        #   1. placeholder attr  — expert.module.fp8_{gate,up,down}
        #   2. wrapper cache     — wrapper.cached_{gate,up,down}
        #   3. local list        — self.{gate,up,down}_list[i]
        # torch.stack() would allocate a brand-new 576 MB contiguous tensor
        # per projection while all three references still keep the
        # originals alive → ~43 GB of duplicate fp8 weights per rank
        # (75 MoE layers x 576 MB) and OOMs the 95 GB H20.
        # Fix: allocate the stacked tensor empty, copy each expert in
        # one at a time, and rebind ALL THREE references so the original
        # per-expert allocation becomes refcount=0 and the CUDA allocator
        # reclaims it before the next expert's copy.
        start = self.routed_expert_start_idx
        gate_shape = self.gate_list[0].shape
        self.fp8_gate_w3d = torch.empty(
            (E, *gate_shape), dtype=self.gate_list[0].dtype, device=self.device)
        for i in range(E):
            self.fp8_gate_w3d[i].copy_(self.gate_list[i])
            view = self.fp8_gate_w3d[i]
            wrapper = self.experts[start + i]
            wrapper.cached_gate = view
            if hasattr(wrapper.module, 'fp8_gate'):
                wrapper.module.fp8_gate = view
            self.gate_list[i] = view
        self.gate_ptrs_ptr = torch.tensor(
            [r.data_ptr() for r in self.gate_list],
            dtype=torch.int64, device=self.device)

        up_shape = self.up_list[0].shape
        self.fp8_up_w3d = torch.empty(
            (E, *up_shape), dtype=self.up_list[0].dtype, device=self.device)
        for i in range(E):
            self.fp8_up_w3d[i].copy_(self.up_list[i])
            view = self.fp8_up_w3d[i]
            wrapper = self.experts[start + i]
            wrapper.cached_up = view
            if hasattr(wrapper.module, 'fp8_up'):
                wrapper.module.fp8_up = view
            self.up_list[i] = view
        self.up_ptrs_ptr = torch.tensor(
            [r.data_ptr() for r in self.up_list],
            dtype=torch.int64, device=self.device)

        down_shape = self.down_list[0].shape
        self.fp8_down_w3d = torch.empty(
            (E, *down_shape), dtype=self.down_list[0].dtype, device=self.device)
        for i in range(E):
            self.fp8_down_w3d[i].copy_(self.down_list[i])
            view = self.fp8_down_w3d[i]
            wrapper = self.experts[start + i]
            wrapper.cached_down = view
            if hasattr(wrapper.module, 'fp8_down'):
                wrapper.module.fp8_down = view
            self.down_list[i] = view
        self.down_ptrs_ptr = torch.tensor(
            [r.data_ptr() for r in self.down_list],
            dtype=torch.int64, device=self.device)

        self.fp8_gate_ws3d = torch.zeros(
            E, n_blocks, k_blocks_pad4,
            dtype=torch.float32, device=self.device)
        for i, s in enumerate(self.gate_scale_list):
            self.fp8_gate_ws3d[i, :, :k_blocks] = s

        self.fp8_up_ws3d = torch.zeros(
            E, n_blocks, k_blocks_pad4,
            dtype=torch.float32, device=self.device)
        for i, s in enumerate(self.up_scale_list):
            self.fp8_up_ws3d[i, :, :k_blocks] = s

        self.fp8_down_ws3d = torch.zeros(
            E, k_blocks, n_blocks_pad4,
            dtype=torch.float32, device=self.device)
        for i, s in enumerate(self.down_scale_list):
            self.fp8_down_ws3d[i, :, :n_blocks] = s

        self._fp8_blockwise_ready = True

        if not getattr(Glm5MoE, '_warned_weights_stacked', False):
            logging.info(
                f"[Glm5MoE] FP8 blockwise 3D weights ready: "
                f"gate={list(self.fp8_gate_w3d.shape)}, "
                f"down={list(self.fp8_down_w3d.shape)}, "
                f"gate_scale={list(self.fp8_gate_ws3d.shape)}"
            )
            Glm5MoE._warned_weights_stacked = True

    def cleanup(self):
        for attr in ('gate_list', 'up_list', 'down_list',
                      'gate_scale_list', 'up_scale_list', 'down_scale_list',
                      'gate_ptrs_ptr', 'up_ptrs_ptr', 'down_ptrs_ptr',
                      'gate_scale_ptrs_ptr', 'up_scale_ptrs_ptr', 'down_scale_ptrs_ptr'):
            setattr(self, attr, None)

    def enable_moe_cuda_graph(
        self,
        manager,
        segment_name: str,
        segment,
        bucketing,
        *,
        graph_output_required: bool = False,
    ) -> None:
        self._moe_cuda_graph_manager = manager
        self._moe_cuda_graph_segment_name = segment_name
        self._moe_cuda_graph_segment = segment
        self._moe_cuda_graph_bucketing = bucketing
        self._moe_cuda_graph_required = graph_output_required

    # ── Forward ──

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if getattr(self.config, 'phase', 'decode') == 'decode':
            return self._forward_decode(hidden_states)
        return self._forward_prefill(hidden_states)

    @torch.inference_mode()
    def _forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """EP decode: AllGather → Gate → Expert Compute → AllReduce → Extract + Shared."""
        # K2.5 3D-MoE path (opt-in) — routes through validated minimax/kimi
        # pattern: dispatch_scatter_ragged + grouped_fp8_blockwise_* + reduce_weighted_scatter.
        if (self.use_3d_moe and self._fp8_blockwise_ready and
                Glm5MoE._3d_buf is not None and _GLM5_HAS_DISPATCH_3D):
            debug_mode = _glm5_moe_debug_mode()
            compare = (
                False if debug_mode == "eager"
                else _glm5_moe_graph_compare_layer_enabled(self.layer_idx)
            )
            graph_required = (
                debug_mode == "graph"
                or (
                    debug_mode != "eager"
                    and (
                        _glm5_moe_cuda_graph_required()
                        or getattr(self, "_moe_cuda_graph_required", False)
                    )
                )
            )
            if compare:
                _record_glm5_moe_dispatch(
                    "eager",
                    layer_idx=self.layer_idx,
                    bsz=hidden_states.shape[0],
                    reason="graph compare returns eager output",
                )
                return self._forward_decode_3d_graph_compare(hidden_states)
            if graph_required:
                if self._moe_cuda_graph_exceeds_max_bucket():
                    if not getattr(self, "_moe_cuda_graph_over_bucket_warned", False):
                        logging.warning(
                            "Layer %d: GLM-5 MoE CUDA graph requested but "
                            "num_tokens_per_rank=%s exceeds max graph bucket; "
                            "using eager MoE for this decode batch",
                            self.layer_idx,
                            self.num_tokens_per_rank,
                        )
                        self._moe_cuda_graph_over_bucket_warned = True
                    _record_glm5_moe_dispatch(
                        "eager",
                        layer_idx=self.layer_idx,
                        bsz=hidden_states.shape[0],
                        reason="graph requested but rank bucket exceeded",
                    )
                    return self._forward_decode_3d(hidden_states)
                _record_glm5_moe_dispatch(
                    "graph",
                    layer_idx=self.layer_idx,
                    bsz=hidden_states.shape[0],
                    reason="full-module graph replay",
                )
                return self._forward_decode_3d_graph(hidden_states)
            if debug_mode == "eager":
                reason = "debug mode requested eager"
            else:
                reason = "graph not requested"
            _record_glm5_moe_dispatch(
                "eager",
                layer_idx=self.layer_idx,
                bsz=hidden_states.shape[0],
                reason=reason,
            )
            return self._forward_decode_3d(hidden_states)

        _record_glm5_moe_dispatch(
            "eager",
            layer_idx=self.layer_idx,
            bsz=hidden_states.shape[0],
            reason="3d MoE graph path unavailable",
        )
        from contextlib import nullcontext as _nullctx
        from batchgen.timing import get_decode_timer

        dt = get_decode_timer()

        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        identity = hidden_states
        num_tokens, hidden_size = hidden_states.shape

        if num_tokens > self.num_tokens_per_rank:
            raise RuntimeError(
                f"MoE buffer overflow: num_tokens={num_tokens} > "
                f"num_tokens_per_rank={self.num_tokens_per_rank}"
            )

        buf = self._buf
        ntp = self.num_tokens_per_rank

        # 1) AllGather
        with (dt.timed("allgather", 0) if dt else _nullctx()):
            G = self.world_size * ntp
            all_tokens = buf.all_tokens[:G] if buf is not None else torch.zeros(
                G, hidden_size, device=self.device, dtype=torch.bfloat16)

            if buf is not None:
                all_tokens.zero_()
                padded = buf.padded_hidden_states[:ntp]
                padded.zero_()
                padded[:num_tokens] = hidden_states
            else:
                padded = torch.zeros(ntp, hidden_size, device=self.device, dtype=hidden_states.dtype)
                padded[:num_tokens] = hidden_states

            with self.comm.change_state(enable=True):
                self.comm.all_gather(
                    all_tokens, padded,
                    stream=torch.cuda.current_stream(self.device),
                )

        # 2) Gate
        with (dt.timed("routing", 0) if dt else _nullctx()):
            global_x = all_tokens
            topk_idx, topk_weight = self._gate_decode(global_x)

        # 3+4) Expert compute
        all_persistent = (self.num_persistent_local_experts == self.experts_per_rank)
        if all_persistent and not self.enable_ep_offloading:
            with (dt.timed("wgmma_pipeline", 0) if dt else _nullctx()):
                global_results = self.expert_compute_persistent(
                    global_x, topk_idx, topk_weight)
        else:
            with (dt.timed("mixed_compute", 0) if dt else _nullctx()):
                global_results = self.expert_compute_mixed(
                    global_x, topk_idx, topk_weight)

        # 5) AllReduce
        with (dt.timed("allreduce", 0) if dt else _nullctx()):
            with self.comm.change_state(enable=True):
                self.comm.all_reduce(
                    global_results, op=dist.ReduceOp.SUM,
                    stream=torch.cuda.current_stream(self.device),
                )

        # 6) Extract local slice + shared expert
        start = self.rank * ntp
        out = global_results[start:start + num_tokens].to(hidden_states.dtype)
        out = out + self.shared_expert_forward(identity)
        return out.view(*orig_shape)

    def _moe_cuda_graph_exceeds_max_bucket(self) -> bool:
        if self._moe_cuda_graph_bucketing is None or self.num_tokens_per_rank is None:
            return False
        try:
            self._moe_cuda_graph_bucketing.get_padded_size(int(self.num_tokens_per_rank))
        except ValueError:
            return True
        return False

    def _moe_cuda_graph_available(self) -> bool:
        if not (
            self._moe_cuda_graph_manager is not None
            and self._moe_cuda_graph_segment_name is not None
            and self._moe_cuda_graph_segment is not None
            and self._moe_cuda_graph_bucketing is not None
            and self.num_tokens_per_rank is not None
            and self.num_tokens_per_rank > 0
        ):
            return False
        try:
            return self._moe_cuda_graph_manager.has_graph(
                self._moe_cuda_graph_segment_name,
                self.num_tokens_per_rank,
            )
        except ValueError:
            return False

    @torch.inference_mode()
    def _forward_decode_3d_graph(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._moe_cuda_graph_available():
            raise RuntimeError(
                f"Layer {self.layer_idx}: GLM-5 MoE CUDA graph requested but not captured"
            )

        orig_shape = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
        num_tokens, _ = hidden_flat.shape
        ntp = self.num_tokens_per_rank
        if ntp is None or ntp <= 0:
            raise RuntimeError(f"Layer {self.layer_idx}: num_tokens_per_rank is not initialized")
        if num_tokens > ntp:
            raise RuntimeError(
                f"MoE graph buffer overflow: num_tokens={num_tokens} > "
                f"num_tokens_per_rank={ntp}"
            )

        bucket = self._moe_cuda_graph_bucketing.get_padded_size(ntp)
        bufs = self._moe_cuda_graph_segment.pool.get(bucket)
        padded = bufs.padded
        padded.zero_()
        if num_tokens > 0:
            padded[:num_tokens].copy_(hidden_flat)

        rank_counts = Glm5MoE._rank_token_counts
        if rank_counts is None:
            if not hasattr(self, "_moe_graph_rank_counts_full"):
                self._moe_graph_rank_counts_full = torch.empty(
                    self.world_size,
                    dtype=torch.int64,
                    device=self.device,
                )
            self._moe_graph_rank_counts_full.fill_(ntp)
            rank_counts = self._moe_graph_rank_counts_full
        elif rank_counts.dtype != torch.int64:
            rank_counts = rank_counts.to(torch.int64)

        graph_out = self._moe_cuda_graph_manager.replay(
            self._moe_cuda_graph_segment_name,
            bucket,
            padded=padded,
            rank_token_counts=rank_counts,
        )
        moe_output = graph_out.get("moe_output")
        if moe_output is None:
            raise RuntimeError(
                f"Layer {self.layer_idx}: GLM-5 MoE graph replay did not return "
                "full-module 'moe_output'"
            )

        if num_tokens == 0:
            return torch.empty(orig_shape, device=self.device, dtype=hidden_states.dtype)
        return moe_output[:num_tokens].to(hidden_flat.dtype).view(*orig_shape)

    @torch.inference_mode()
    def _forward_decode_3d_graph_compare(self, hidden_states: torch.Tensor) -> torch.Tensor:
        eager_out = self._forward_decode_3d(hidden_states)
        if not self._moe_cuda_graph_available():
            logging.warning(
                "[GLM5_MOE_GRAPH_COMPARE][L%d] graph unavailable; returning eager output",
                self.layer_idx,
            )
            return eager_out

        graph_out = self._forward_decode_3d_graph(hidden_states)
        diff = (graph_out.to(torch.float32) - eager_out.to(torch.float32)).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
        atol = float(os.environ.get("BATCHGEN_GLM5_MOE_GRAPH_COMPARE_ATOL", "1e-2"))
        ok = max_abs <= atol
        logging.info(
            "[GLM5_MOE_GRAPH_COMPARE][L%d][rank=%d][boundary=full_module] "
            "%s max_abs=%.6g mean_abs=%.6g "
            "shape=%s ntp=%s",
            self.layer_idx,
            self.rank,
            "OK" if ok else "FAIL",
            max_abs,
            mean_abs,
            tuple(int(x) for x in eager_out.shape),
            self.num_tokens_per_rank,
        )
        if not ok and _glm5_moe_graph_compare_fail_on_mismatch():
            raise RuntimeError(
                f"GLM-5 MoE graph compare mismatch at layer {self.layer_idx}: "
                f"max_abs={max_abs:.6g} > atol={atol:.6g}"
            )
        return eager_out

    # ── K2.5 3D MoE decode path (minimax/Kimi parity) ──

    @torch.inference_mode()
    def _forward_decode_3d(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """K2.5 pattern decode: AllGather → Gate → 3D dispatch → blockwise FP8 GEMM → weighted scatter → AllReduce → Shared.

        Mirrors MiniMaxM25MoE.moe_infer_allgather_allreduce_bf16_acc
        (model.py:1180-1287). Only difference is GLM-5's `shared_experts`
        addition at the end (minimax has no shared expert).
        """
        import torch.distributed as dist
        from contextlib import nullcontext as _nullctx
        from batchgen.timing import get_decode_timer

        dt = get_decode_timer()
        li = self.layer_idx

        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        identity = hidden_states
        num_tokens, hidden_size = hidden_states.shape

        if num_tokens > self.num_tokens_per_rank:
            raise RuntimeError(
                f"MoE buffer overflow: num_tokens={num_tokens} > "
                f"num_tokens_per_rank={self.num_tokens_per_rank}"
            )

        ntp = self.num_tokens_per_rank
        num_global = ntp * self.world_size
        topk = self.num_experts_per_tok
        buf = Glm5MoE._3d_buf
        buf.resize_if_needed(num_global, num_tokens_per_rank=ntp)

        if not getattr(Glm5MoE, '_warned_k25_path', False):
            logging.warning(
                "[Glm5MoE] HOT PATH: dispatch_scatter_ragged + reduce_weighted_scatter "
                "(compact ragged MoE)")
            Glm5MoE._warned_k25_path = True

        # 1) AllGather
        all_tokens = buf.all_tokens[:num_global]
        # Slice the send buffer to exactly ntp rows: ncclAllGather sends
        # input.numel(), so an oversized `padded` (left by a bigger previous
        # batch job) would rank-stride the gather wider than consumers read.
        with (dt.timed("moe_pad_copy", li) if dt else _nullctx()):
            padded = buf.padded[:ntp]
            padded.zero_()
            if num_tokens > 0:
                padded[:num_tokens] = hidden_states
        with (dt.timed("allgather", li) if dt else _nullctx()):
            with self.comm.change_state(enable=True):
                self.comm.all_gather(
                    all_tokens, padded,
                    stream=torch.cuda.current_stream(self.device),
                )

        current_stream = torch.cuda.current_stream(self.device)
        routed_stream = type(self)._routed_moe_stream
        if routed_stream is None:
            raise RuntimeError("GLM-5 routed MoE side stream is not initialized")
        routed_stream.wait_stream(current_stream)
        with (dt.timed("shared_expert", li) if dt else _nullctx()):
            shared_output = self.shared_expert_forward(identity)

        with torch.cuda.stream(routed_stream):
            # 2) Gate (returns int32 topk_idx, fp32 topk_weight)
            with (dt.timed("routing", li) if dt else _nullctx()):
                topk_idx, topk_weight = self._gate_decode(all_tokens)

            # Mask padding tokens so they don't inflate expert_counts nor pollute
            # grouped GEMM compute.
            rank_counts = Glm5MoE._rank_token_counts
            if rank_counts is not None:
                with (dt.timed("routing_pad_mask", li) if dt else _nullctx()):
                    positions = torch.arange(num_global, device=self.device)
                    rank_ids = positions // ntp
                    local_pos = positions % ntp
                    max_valid = rank_counts[rank_ids]
                    padding_mask = local_pos >= max_valid
                    padding_mask_2d = padding_mask.unsqueeze(1).expand_as(topk_idx)
                    topk_idx = torch.where(
                        padding_mask_2d,
                        torch.full_like(topk_idx, -1),
                        topk_idx,
                    )
                    topk_weight = torch.where(
                        padding_mask_2d,
                        torch.zeros_like(topk_weight),
                        topk_weight,
                    )

            # 3) Compact ragged dispatch
            with (dt.timed("dispatch", li) if dt else _nullctx()):
                expert_counts, cu_seqlens, topk_pos = _glm5_dispatch_scatter_ragged(
                    all_tokens, topk_idx.to(torch.int32),
                    buf.dispatched_x,
                    self.routed_expert_start_idx, self.experts_per_rank,
                    buf.expert_counts, buf.expert_counters,
                    buf.cu_seqlens,
                    buf.topk_pos[:num_global * topk],
                )

            # 4) FP8 blockwise GEMM on the compact buffer
            self._fp8_blockwise_gemm_3d(buf, expert_counts, cu_seqlens)

            # 5) Weighted scatter reduce
            result_buf = buf.result_buffer[:num_global]
            with (dt.timed("scatter_reduce", li) if dt else _nullctx()):
                global_results = reduce_weighted_scatter(
                    buf.expert_out, topk_pos, topk_weight,
                    num_global, hidden_size, topk,
                    output=result_buf,
                )

            # 6) Reduce directly into the rank-local rows.
            with (dt.timed("reduce_scatter", li) if dt else _nullctx()):
                with self.comm.change_state(enable=True):
                    self.comm.reduce_scatter(
                        buf.local_result_buffer[:ntp],
                        global_results,
                        stream=torch.cuda.current_stream(self.device),
                    )

        current_stream.wait_stream(routed_stream)

        # 7) Slice real local rows + add shared expert
        if num_tokens == 0:
            return torch.empty(orig_shape, device=self.device, dtype=hidden_states.dtype)
        out = buf.local_result_buffer[:num_tokens].to(hidden_states.dtype)
        out = out + shared_output
        return out.view(*orig_shape)

    def _fp8_blockwise_gemm_3d(self, buf, expert_counts, cu_seqlens):
        """FP8 blockwise grouped GEMM on the compact ragged buffer (in-place).

        Reads buf.dispatched_x, writes buf.expert_out. Every staging tensor is a
        persistent buffer, so this stage allocates nothing per call and the
        scale tensors are already in the grouped GEMM's transposed layout (the
        3D path needed a `[rows, K/128] -> [K/128, rows]` `.t().contiguous()`
        after each quant).

        There is no Triton fallback: `moe_ragged` hard-fails if the compiled
        kernels predate the compact layout.
        """
        from contextlib import nullcontext as _nullctx
        from batchgen.timing import get_decode_timer

        dt = get_decode_timer()
        li = self.layer_idx

        if not getattr(Glm5MoE, '_warned_gemm_3d', False):
            logging.warning(
                "[Glm5MoE] HOT PATH: _fp8_blockwise_gemm_3d (compact ragged, "
                f"capacity={buf.capacity} rows)")
            Glm5MoE._warned_gemm_3d = True

        E = self.experts_per_rank
        seqlens = expert_counts[:E]
        avg = _GLM5_MOE_GEMM_TILEM_AVG

        with (dt.timed("moe_act_quant", li) if dt else _nullctx()):
            _glm5_act_quant_ragged(
                buf.dispatched_x, seqlens, cu_seqlens, buf.x_fp8, buf.x_scale)

        # S1: gate + up + SiLU → BF16 intermediate
        with (dt.timed("grouped_gemm_s1", li) if dt else _nullctx()):
            s1_result = grouped_fp8_blockwise_fused_s1(
                buf.x_fp8.view(torch.float8_e4m3fn), buf.x_scale,
                self.fp8_gate_w3d.view(torch.float8_e4m3fn),
                self.fp8_up_w3d.view(torch.float8_e4m3fn),
                self.fp8_gate_ws3d, self.fp8_up_ws3d,
                seqlens, cu_seqlens, avg,
                output=buf.intermediate,
            )
        with (dt.timed("moe_act_quant", li) if dt else _nullctx()):
            _glm5_act_quant_ragged(
                s1_result, seqlens, cu_seqlens, buf.inter_fp8, buf.inter_scale)

        # S3: down projection — write straight into the shared expert_out
        # buffer (kernel supports output=). Stale rows beyond each expert's
        # count are never read: reduce_weighted_scatter only visits this step's
        # topk_pos slots.
        with (dt.timed("grouped_gemm_s3", li) if dt else _nullctx()):
            grouped_fp8_blockwise_s3(
                buf.inter_fp8.view(torch.float8_e4m3fn), buf.inter_scale,
                self.fp8_down_w3d.view(torch.float8_e4m3fn),
                self.fp8_down_ws3d,
                seqlens, cu_seqlens, avg,
                output=buf.expert_out,
            )

    # ── Gate + Expert Compute ──

    def _gate_decode(self, x: torch.Tensor):
        """Sigmoid gating with e_score_correction — fused CUDA kernel.

        Single CUDA launch for sigmoid + e_score_correction bias + top-k +
        normalize + scale. Replaces the 6-op PyTorch eager path
        (F.linear → sigmoid → +bias → topk → gather → /sum → ×scale) with
        one kernel after a graph-stable BF16 router GEMM. The router GEMM uses
        a fixed per-row accumulation order so valid rows do not drift when CUDA
        graph buckets include rank padding.
        """
        from contextlib import nullcontext as _nullctx
        from batchgen.moe.routing import gate_sigmoid_topk_cuda
        from batchgen.timing import get_decode_timer
        dt = get_decode_timer()
        with (dt.timed("router_gemm", 0) if dt else _nullctx()):
            if _glm5_moe_router_mode() == "custom_gemm":
                # Original fused kernel: one block per expert, serial N-row
                # loop — O(N) wall (98.8 ms/step at num_global~1448). Kept for
                # CUDA-graph bucket-M-independence work only.
                from batchgen.moe.routing import glm5_router_gemm_cuda
                router_logits = glm5_router_gemm_cuda(
                    x,
                    self.gate.weight,
                )
            else:
                # FP32 cuBLAS with the gate weight cast once and cached:
                # fp32-grade logits (maxabs ~2e-6 vs fp64), 100% top-8
                # agreement with the custom kernel, 0.14 ms/call at N=1448.
                if self._gate_w_fp32 is None:
                    self._gate_w_fp32 = self.gate.weight.float()
                router_logits = F.linear(x.float(), self._gate_w_fp32)
        with (dt.timed("gate_topk", 0) if dt else _nullctx()):
            return gate_sigmoid_topk_cuda(
                router_logits,
                self.gate.e_score_correction_bias.float(),
                k=self.num_experts_per_tok,
                routed_scaling_factor=self.gate.routed_scaling_factor,
            )

    def expert_compute_persistent(self, global_x, topk_idx, topk_weight):
        """All-persistent expert compute. Zero CPU-GPU sync on WGMMA path."""
        if self.use_wgmma_fp8:
            # WGMMA path: dispatch_scatter_3d → WGMMA pipeline → reduce (zero sync)
            bufs = Glm5MoE._buf
            if bufs is None:
                bufs = self._lazy_init_wgmma_bufs()

            if self._wgmma_layer_id not in bufs._layer_weights:
                bufs.register_layer_weights(
                    self._wgmma_layer_id,
                    self.gate_list, self.gate_scale_list,
                    self.up_list, self.up_scale_list,
                    self.down_list, self.down_scale_list)

            return bufs.forward(self._wgmma_layer_id, global_x, topk_idx, topk_weight)
        else:
            # Triton fallback path: dispatch → grouped FP8 GEMM → scatter
            return self._triton_compute(global_x, topk_idx, topk_weight)

    def _triton_compute(self, global_x, topk_idx, topk_weight):
        """Triton dispatch + grouped FP8 GEMM + scatter reduce."""
        from mgn_kernel import fused_moe_token_dispatch

        input_x, input_eids, global_indices, token_topk_pos, expert_counts, expert_offsets = \
            fused_moe_token_dispatch(
                global_x, topk_idx, self.token_idx, self.topk_pos,
                self.routed_expert_start_idx, self.routed_expert_end_idx,
            )

        res = self._grouped_dequant_moe_fp8(
            input_x, input_eids, expert_counts, expert_offsets,
        )

        global_results = scatter_weight_reduce_optimized(
            res, global_indices, token_topk_pos, topk_weight,
            self.num_tokens_per_rank * self.world_size, self.num_experts_per_tok,
        )
        return global_results.to(torch.bfloat16)

    def expert_compute_mixed(self, global_x, topk_idx, topk_weight):
        """Mixed path: WGMMA for persistent + loop for non-persistent.

        One .tolist() sync for expert_offsets (acceptable on opt-in path).
        """
        from mgn_kernel import fused_moe_token_dispatch

        n_persistent = self.num_persistent_local_experts
        num_tokens = global_x.shape[0]
        hidden_size = global_x.shape[1]

        # Dispatch
        input_x, input_eids, global_indices, token_topk_pos, expert_counts, expert_offsets = \
            fused_moe_token_dispatch(
                global_x, topk_idx, self.token_idx, self.topk_pos,
                self.routed_expert_start_idx, self.routed_expert_end_idx,
            )

        offsets_cpu = expert_offsets.tolist()
        actual_total = offsets_cpu[-1]

        if actual_total == 0:
            return torch.zeros(
                num_tokens, hidden_size,
                device=self.device, dtype=torch.bfloat16,
            )

        res = input_x.new_empty(actual_total, hidden_size, dtype=torch.bfloat16)

        # Phase 1: Grouped FP8 GEMM for persistent experts
        persistent_end = offsets_cpu[n_persistent]
        if n_persistent > 0 and persistent_end > 0:
            persistent_counts = expert_counts[:n_persistent]
            persistent_offsets = expert_offsets[:n_persistent + 1]
            persistent_res = self._grouped_dequant_moe_fp8(
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

        # Scatter + weighted reduce
        global_results = scatter_weight_reduce_optimized(
            res, global_indices, token_topk_pos, topk_weight,
            self.num_tokens_per_rank * self.world_size, self.num_experts_per_tok,
        )
        return global_results.to(torch.bfloat16)

    def shared_expert_forward(self, identity: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(identity)

    def _prefill_prepare_weights(self) -> None:
        """Acquire one full FP8 expert layer from the async core-engine ring."""
        if not self._prefill_grouped_enabled:
            return
        if getattr(self.config, "phase", "decode") == "decode":
            return
        if self._prefill_prepared_keys is not None:
            return

        cls = type(self)
        if (
            cls._prefill_buf is None
            or cls._prefill_ptrs_pinned is None
            or cls._prefill_ptrs_dev is None
        ):
            raise RuntimeError(
                "GLM-5 grouped prefill is enabled without initialized workspace"
            )

        cls.retire_prefill_grouped_weights()
        stage = cls._prefill_ptrs_pinned
        keys = []
        prototypes = None
        core_engine = self.experts[0].core_engine
        shared_key = None
        try:
            for expert_idx, expert in enumerate(self.experts):
                if expert.persistent or not expert.is_fp8:
                    raise RuntimeError(
                        "GLM-5 grouped prefill requires every routed expert to "
                        "be nonpersistent FP8"
                    )
                weights = expert.load_weights_pinned()
                gate = weights["gate_proj.weight"]
                up = weights["up_proj.weight"]
                down = weights["down_proj.weight"]
                gate_scale = expert.weight_dequant_scale[
                    "gate_proj.weight_scale_inv"
                ]
                up_scale = expert.weight_dequant_scale[
                    "up_proj.weight_scale_inv"
                ]
                down_scale = expert.weight_dequant_scale[
                    "down_proj.weight_scale_inv"
                ]
                stage[0, expert_idx] = gate.data_ptr()
                stage[1, expert_idx] = gate_scale.data_ptr()
                stage[2, expert_idx] = up.data_ptr()
                stage[3, expert_idx] = up_scale.data_ptr()
                stage[4, expert_idx] = down.data_ptr()
                stage[5, expert_idx] = down_scale.data_ptr()
                keys.append(expert.module_key)
                if prototypes is None:
                    prototypes = (
                        gate,
                        gate_scale,
                        up,
                        up_scale,
                        down,
                        down_scale,
                    )

            shared = self.shared_experts
            if shared.persistent or not shared.is_fp8:
                raise RuntimeError(
                    "GLM-5 grouped prefill requires a nonpersistent FP8 shared expert"
                )
            shared_weights = shared.load_weights_pinned()
            shared.cached_gate = shared_weights["gate_proj.weight"]
            shared.cached_up = shared_weights["up_proj.weight"]
            shared.cached_down = shared_weights["down_proj.weight"]
            shared_key = shared.module_key
            cls._prefill_ptrs_dev.copy_(stage, non_blocking=True)
        except Exception:
            for key in keys:
                core_engine.free_weights_buffer(key)
            if shared_key is not None:
                self.shared_experts.core_engine.free_weights_buffer(shared_key)
            raise

        self._prefill_prepared_keys = keys
        self._prefill_weight_prototypes = prototypes
        self._prefill_shared_key = shared_key

    @torch.inference_mode()
    def _forward_prefill_grouped(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run all 256 routed experts through compact pointer-array GEMMs."""
        cls = type(self)
        if self._prefill_prepared_keys is None:
            self._prefill_prepare_weights()
        keys = self._prefill_prepared_keys
        prototypes = self._prefill_weight_prototypes
        if keys is None or prototypes is None:
            raise RuntimeError("GLM-5 grouped prefill weights were not prepared")

        buf = cls._prefill_buf
        ptrs = cls._prefill_ptrs_dev
        if buf is None or ptrs is None:
            raise RuntimeError("GLM-5 grouped prefill workspace was released")

        orig_shape = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
        num_tokens, hidden_size = hidden_flat.shape
        topk = self.num_experts_per_tok
        topk_weights, topk_indices = self.gate(hidden_flat)
        topk_indices_i32 = topk_indices.to(torch.int32)
        output = torch.empty_like(hidden_flat)

        gate_w, gate_scale, up_w, up_scale, down_w, down_scale = prototypes
        for start in range(0, num_tokens, buf.token_window):
            end = min(start + buf.token_window, num_tokens)
            window_tokens = end - start
            counts, cu_seqlens, topk_pos = _glm5_dispatch_scatter_ragged(
                hidden_flat[start:end],
                topk_indices_i32[start:end],
                buf.dispatched_x,
                0,
                buf.num_experts,
                buf.expert_counts,
                buf.expert_counters,
                buf.cu_seqlens,
                buf.topk_pos[: window_tokens * topk],
            )
            _glm5_act_quant_ragged(
                buf.dispatched_x,
                counts,
                cu_seqlens,
                buf.x_fp8,
                buf.x_scale,
            )
            grouped_fp8_blockwise_fused_s1_ptrs(
                buf.x_fp8.view(torch.float8_e4m3fn),
                buf.x_scale,
                gate_w,
                ptrs[0],
                up_w,
                ptrs[2],
                gate_scale,
                ptrs[1],
                up_scale,
                ptrs[3],
                counts,
                cu_seqlens,
                _GLM5_PREFILL_GROUPED_TILEM_AVG,
                output=buf.intermediate,
                tma_desc=buf.s1_tma_desc,
                tiles=buf.tiles,
                cu_tiles=buf.cu_tiles,
            )
            _glm5_act_quant_ragged(
                buf.intermediate,
                counts,
                cu_seqlens,
                buf.inter_fp8,
                buf.inter_scale,
            )
            grouped_fp8_blockwise_s3_ptrs(
                buf.inter_fp8.view(torch.float8_e4m3fn),
                buf.inter_scale,
                down_w,
                ptrs[4],
                down_scale,
                ptrs[5],
                counts,
                cu_seqlens,
                _GLM5_PREFILL_GROUPED_TILEM_AVG,
                output=buf.expert_out,
                tma_desc=buf.s3_tma_desc,
                tiles=buf.tiles,
                cu_tiles=buf.cu_tiles,
            )
            reduce_weighted_scatter_bf16_ordered(
                buf.expert_out,
                topk_pos,
                topk_indices_i32[start:end],
                topk_weights[start:end],
                window_tokens,
                hidden_size,
                topk,
                output=output[start:end],
            )

        if self._prefill_release_event is None:
            self._prefill_release_event = torch.cuda.Event()
        self._prefill_release_event.record(torch.cuda.current_stream(self.device))
        cls._prefill_ring_pending = (
            self._prefill_release_event,
            keys,
            self.experts[0].core_engine,
        )
        self._prefill_prepared_keys = None
        self._prefill_weight_prototypes = None

        shared = self.shared_experts
        shared_output = shared._forward_impl(hidden_flat)
        if self._prefill_shared_release_event is None:
            self._prefill_shared_release_event = torch.cuda.Event()
        self._prefill_shared_release_event.record(
            torch.cuda.current_stream(self.device)
        )
        cls._prefill_shared_pending = (
            self._prefill_shared_release_event,
            self._prefill_shared_key,
            shared.core_engine,
        )
        self._prefill_shared_key = None
        shared.cached_gate = shared.cached_up = shared.cached_down = None

        if not cls._prefill_grouped_logged:
            cls._prefill_grouped_logged = True
            logging.info(
                "[GLM5_GROUPED_PREFILL] active: E=%d window=%d chunks=%d "
                "pointer_table=device grouped_s1+s3 ordered_bf16_reduce",
                buf.num_experts,
                buf.token_window,
                (num_tokens + buf.token_window - 1) // buf.token_window,
            )
        return (output + shared_output).view(*orig_shape)

    def _forward_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Prefill: per-expert loop (no EP).

        Accumulation dtype: default BF16 (matches HF ``GlmMoeDsaNaiveMoe`` which
        uses ``torch.zeros_like(hidden_states)`` keeping input dtype). Opt into
        the old FP32 accumulate path with ``BATCHGEN_GLM5_MOE_FP32_ACCUM=1``.

        Resolve each expert's token indices once. Reusing those indices for the
        input, router weights, and output accumulation avoids repeating
        dynamic-shape boolean indexing, which otherwise forces a GPU-to-CPU
        size synchronization for every indexed tensor.
        """
        import os as _os_moe
        if (
            getattr(self, "_prefill_grouped_enabled", False)
            and _os_moe.environ.get("BATCHGEN_GLM5_MOE_FP32_ACCUM", "0") != "1"
        ):
            return self._forward_prefill_grouped(hidden_states)
        _moe_fp32 = _os_moe.environ.get("BATCHGEN_GLM5_MOE_FP32_ACCUM", "0") == "1"
        orig_shape = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_states.shape[-1])
        identity = hidden_flat

        topk_weights, topk_indices = self.gate(hidden_flat)

        # Precompute per-expert activity in a single D2H sync so the skip
        # check below doesn't fire an .any() implicit-sync per expert per
        # layer (~1200 syncs/prefill on 16 local experts * 75 layers).
        _active = torch.zeros(self.total_experts, dtype=torch.bool, device=topk_indices.device)
        _active[topk_indices.reshape(-1)] = True
        _active_cpu = _active.tolist()

        if _moe_fp32:
            output = torch.zeros_like(hidden_flat, dtype=torch.float32)
        else:
            output = torch.zeros_like(hidden_flat)
        for i, expert in enumerate(self.experts):
            if isinstance(expert, _Glm5ExpertPlaceholder):
                continue
            if not _active_cpu[i]:
                continue
            expert_mask = (topk_indices == i).any(dim=-1)
            token_idx = expert_mask.nonzero(as_tuple=False).squeeze(-1)
            expert_input = hidden_flat.index_select(0, token_idx)
            expert_output = expert(expert_input)
            selected_topk_indices = topk_indices.index_select(0, token_idx)
            selected_topk_weights = topk_weights.index_select(0, token_idx)
            expert_weight = torch.where(
                selected_topk_indices == i,
                selected_topk_weights,
                torch.zeros_like(selected_topk_weights),
            ).sum(dim=-1)
            if _moe_fp32:
                weighted = expert_output.float() * expert_weight.unsqueeze(-1)
            else:
                weighted = expert_output * expert_weight.unsqueeze(-1).to(expert_output.dtype)
            output.index_add_(0, token_idx, weighted)

        if _moe_fp32:
            output = output.to(hidden_flat.dtype)
        output = output + self.shared_experts(identity)
        return output.view(*orig_shape)

    # ── Internal helpers ──

    def _lazy_init_wgmma_bufs(self):
        """Lazy-init WGMMAMoEBuffers on first forward."""
        from batchgen.moe.fp8_wgmma_pipeline import WGMMAMoEBuffers, DEFAULT_MTP
        wgmma_mod, fast_mod, dr_mod = Glm5MoE._wgmma_modules
        num_global_tokens = self.num_tokens_per_rank * self.world_size
        bufs = WGMMAMoEBuffers(
            wgmma_mod, fast_mod, dr_mod,
            len(self.gate_list), DEFAULT_MTP,
            self.hidden_size, self.config.moe_intermediate_size,
            self.gate_list, self.gate_scale_list,
            self.up_list, self.up_scale_list,
            self.down_list, self.down_scale_list,
            self.routed_expert_start_idx,
            self.num_experts_per_tok,
            num_global_tokens,
            num_tokens_per_rank=self.num_tokens_per_rank,
            device=self.device)
        Glm5MoE._buf = bufs
        return bufs

    def _grouped_dequant_moe_fp8(self, x, eids, expert_counts, expert_offsets,
                                  num_local_experts=None):
        """Grouped FP8 GEMM: gate+up+SiLU → down (Triton fallback path)."""
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
        if hasattr(self, 'gate_scale'):
            from batchgen.attention.mla.fa3_backend import w8a16_gemm
            gate = w8a16_gemm(self.gate_proj.weight.data, self.gate_scale, x)
            up = w8a16_gemm(self.up_proj.weight.data, self.up_scale, x)
            return w8a16_gemm(self.down_proj.weight.data, self.down_scale, F.silu(gate) * up)
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
            self.mlp = Glm5MoE(config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        from contextlib import nullcontext as _nullctx
        from batchgen.timing import get_decode_timer

        # self_attn is the GLM5AttnWrapper; the config (with .phase) lives on
        # the wrapped module.
        _attn_cfg = getattr(getattr(self.self_attn, 'module', self.self_attn), 'config', None)
        _is_decode = getattr(_attn_cfg, 'phase', 'decode') == 'decode'
        dt = get_decode_timer() if _is_decode else None
        li = self.layer_idx

        # Acquire this layer's full 256-expert pointer table at layer entry,
        # while the GPU may still be draining layer L-1. Retiring L-1 here
        # releases ring slots early enough for the core-engine H2D worker to
        # fill layer L+1 while this layer's attention runs.
        if (
            not _is_decode
            and isinstance(self.mlp, Glm5MoE)
            and self.mlp._prefill_grouped_enabled
        ):
            self.mlp._prefill_prepare_weights()

        # Pre-norm attention
        residual = hidden_states
        with (dt.timed("input_norm", li) if dt else _nullctx()):
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights, present = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # Fused residual add + RMSNorm (saves one HBM pass of [B, 6144])
        from batchgen.attention.fused_kernels import cuda_add_rmsnorm
        with (dt.timed("add_rmsnorm", li) if dt else _nullctx()):
            hidden_states, residual = cuda_add_rmsnorm(
                residual, hidden_states,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
            )

        # MoE/FFN
        with (dt.timed("dense_mlp", li)
              if dt and isinstance(self.mlp, Glm5MLP) else _nullctx()):
            hidden_states = self.mlp(hidden_states)
        with (dt.timed("residual_add", li) if dt else _nullctx()):
            hidden_states = residual + hidden_states

        if (
            not _is_decode
            and isinstance(self.mlp, Glm5MoE)
            and self.mlp._prefill_grouped_enabled
            and self.layer_idx == self.mlp.config.num_hidden_layers - 1
        ):
            Glm5MoE.retire_prefill_grouped_weights()

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

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        # Shared RoPE instance
        self._shared_rotary_emb = Glm5RotaryEmbedding(
            dim=config.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

        self.layers = nn.ModuleList(
            [Glm5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        # Assign shared RoPE to attention and indexer
        for layer in self.layers:
            layer.self_attn.rotary_emb = self._shared_rotary_emb
            # DSA layers have an `indexer` attribute; GLM-5.2 shared layers set it
            # to None (no indexer weights), so guard the deref.
            if getattr(layer.self_attn, 'indexer', None) is not None:
                layer.self_attn.indexer.rotary_emb = self._shared_rotary_emb

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
        from contextlib import nullcontext as _nullctx
        from batchgen.timing import get_decode_timer

        dt = (get_decode_timer()
              if getattr(self.config, 'phase', 'decode') == 'decode' else None)

        if inputs_embeds is None:
            with (dt.timed("embed", 0) if dt else _nullctx()):
                inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for idx, layer in enumerate(self.layers):
            past_kv = past_key_values[idx] if past_key_values is not None else None
            hidden_states, _, _ = layer(
                hidden_states, attention_mask, position_ids, past_kv, use_cache,
            )
        with (dt.timed("final_norm", 0) if dt else _nullctx()):
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
        from contextlib import nullcontext as _nullctx
        from batchgen.timing import get_decode_timer

        dt = (get_decode_timer()
              if getattr(self.config, 'phase', 'decode') == 'decode' else None)
        with (dt.timed("lm_head", 0) if dt else _nullctx()):
            logits = self.lm_head(hidden_states)
        from types import SimpleNamespace
        return SimpleNamespace(logits=logits)
