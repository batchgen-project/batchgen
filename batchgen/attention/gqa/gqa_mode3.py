# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

"""GQA Mode 3 attention: GPU paged KV cache decode.

Mode 3 = Prefill with Offloading + Pure GPU Decoding
- KV cache stays entirely on GPU (paged)
- No HtoD transfers during decode
- Works for any world_size

For GQA (GPT-OSS), no special preprocessing is needed - just offload
the computed K, V tensors to paged cache.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple

from .fa_decode import gqa_decode_fa
from .sink_correction import apply_sink_correction

# Import timing from model module (lazy to avoid circular import)
def _get_timing():
    try:
        from batchgen.models.openai.gpt_oss_120b.model import DECODE_TIMING
        return DECODE_TIMING
    except ImportError:
        return None


def gqa_decoding_mode_3_bf16(
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    # Weights
    qkv_weight: torch.Tensor,
    qkv_bias: Optional[torch.Tensor],
    out_weight: torch.Tensor,
    out_bias: Optional[torch.Tensor],
    norm_weight: torch.Tensor,
    norm_eps: float,
    sinks: torch.Tensor,
    # RoPE
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    # KV cache manager
    gpu_paged_kv_manager,
    layer_idx: int,
    # Config
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sm_scale: float,
    # Optional
    batch_slice: Optional[Tuple[int, int]] = None,
    sliding_window: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GQA Mode 3 decode: GPU paged KV cache.

    Flow:
    1. RMSNorm on hidden states
    2. Compute Q, K, V projections
    3. Apply RoPE to Q, K
    4. Store K, V to GPU paged cache (offload)
    5. Retrieve historical K, V with page table
    6. Flash attention with paged KV
    7. Apply sink correction
    8. Output projection

    No special preprocessing - just offload the computed K, V to paged cache.

    Args:
        hidden_states: Input tensor [batch, hidden_size]
        position_ids: Position IDs for RoPE [batch]
        cache_seqlens: Current sequence lengths [batch]
        max_seqlen: Maximum sequence length in batch
        qkv_weight: QKV projection weight [qkv_dim, hidden_size]
        qkv_bias: QKV projection bias [qkv_dim] or None
        out_weight: Output projection weight [hidden_size, q_dim]
        out_bias: Output projection bias [hidden_size] or None
        norm_weight: RMSNorm weight [hidden_size]
        norm_eps: RMSNorm epsilon
        sinks: Attention sink values [num_q_heads]
        rope_cos: RoPE cosine values [max_pos, head_dim//2]
        rope_sin: RoPE sine values [max_pos, head_dim//2]
        gpu_paged_kv_manager: GPU paged KV cache manager
        layer_idx: Layer index
        batch_slice: Optional (start, end) for micro-batching
        num_q_heads: Number of query heads
        num_kv_heads: Number of KV heads
        head_dim: Head dimension
        sm_scale: Softmax scale (1/sqrt(head_dim))
        sliding_window: Sliding window size (0 = full attention)

    Returns:
        Tuple of (attn_output, k_new, v_new)
        - attn_output: [batch, hidden_size]
        - k_new: New key tensor for this decode step
        - v_new: New value tensor for this decode step
    """
    timing = _get_timing()
    batch_size = hidden_states.shape[0]
    hidden_size = hidden_states.shape[-1]

    # 1. RMSNorm
    if timing and timing.enabled:
        with timing.time("attn.rms_norm"):
            t = _rms_norm(hidden_states, norm_weight, norm_eps)
    else:
        t = _rms_norm(hidden_states, norm_weight, norm_eps)

    # 2. QKV projection
    if timing and timing.enabled:
        with timing.time("attn.qkv_proj"):
            qkv = F.linear(t, qkv_weight, qkv_bias)
    else:
        qkv = F.linear(t, qkv_weight, qkv_bias)

    # Split into Q, K, V
    q_dim = num_q_heads * head_dim
    k_dim = num_kv_heads * head_dim
    v_dim = num_kv_heads * head_dim

    q = qkv[..., :q_dim]
    k = qkv[..., q_dim:q_dim + k_dim]
    v = qkv[..., q_dim + k_dim:q_dim + k_dim + v_dim]

    # Reshape to [batch, 1, heads, head_dim] for flash attention
    q = q.view(batch_size, 1, num_q_heads, head_dim)
    k = k.view(batch_size, 1, num_kv_heads, head_dim)
    v = v.view(batch_size, 1, num_kv_heads, head_dim)

    # 3. Apply RoPE
    if timing and timing.enabled:
        with timing.time("attn.rope"):
            q, k = _apply_rope(q, k, rope_cos, rope_sin, position_ids)
    else:
        q, k = _apply_rope(q, k, rope_cos, rope_sin, position_ids)

    # Keep original k, v for return
    k_new = k.squeeze(1)  # [batch, num_kv_heads, head_dim]
    v_new = v.squeeze(1)

    # 4. Offload: Store new K, V to GPU paged cache
    # Manager expects [batch, seq_len=1, num_kv_heads, head_dim]
    if timing and timing.enabled:
        with timing.time("attn.kv_update"):
            gpu_paged_kv_manager.update_layer_decode_new_token(
                k_tensor=k,  # [batch, 1, num_kv_heads, head_dim]
                v_tensor=v,
                sequence_lengths=cache_seqlens,
                layer_idx=layer_idx,
                batch_slice=batch_slice,
            )
    else:
        gpu_paged_kv_manager.update_layer_decode_new_token(
            k_tensor=k,
            v_tensor=v,
            sequence_lengths=cache_seqlens,
            layer_idx=layer_idx,
            batch_slice=batch_slice,
        )

    # 5. Get blocked historical KV and page table
    if timing and timing.enabled:
        with timing.time("attn.kv_fetch"):
            blocked_k, blocked_v, block_table = gpu_paged_kv_manager.get_layer_kv_with_page_table(
                layer_idx=layer_idx
            )
    else:
        blocked_k, blocked_v, block_table = gpu_paged_kv_manager.get_layer_kv_with_page_table(
            layer_idx=layer_idx
        )

    # Apply batch slice to block_table for micro-batching (like DeepSeek)
    if batch_slice is not None:
        start_idx, end_idx = batch_slice
        block_table = block_table[start_idx:end_idx]
        cache_seqlens_slice = cache_seqlens[start_idx:end_idx]
    else:
        cache_seqlens_slice = cache_seqlens

    # 6. Flash attention with paged KV
    if not hasattr(gqa_decoding_mode_3_bf16, '_logged'):
        print(f"[batchgen_decode] gqa_decoding_mode_3_bf16 CALLED (gqa_mode3.py)", flush=True)
        gqa_decoding_mode_3_bf16._logged = True
    if timing and timing.enabled:
        with timing.time("attn.flash_attn"):
            attn_output, lse = gqa_decode_fa(
                q,  # [batch, 1, num_q_heads, head_dim]
                blocked_k,  # [num_blocks, page_size, num_kv_heads, head_dim]
                blocked_v,
                cache_seqlens=cache_seqlens_slice,
                block_table=block_table,
                sinks=sinks,
                softmax_scale=sm_scale,
                sliding_window=sliding_window if sliding_window > 0 else None,
            )
    else:
        attn_output, lse = gqa_decode_fa(
            q,
            blocked_k,
            blocked_v,
            cache_seqlens=cache_seqlens_slice,
            block_table=block_table,
            sinks=sinks,
            softmax_scale=sm_scale,
            sliding_window=sliding_window if sliding_window > 0 else None,
        )

    # 7. Reshape and output projection
    # attn_output: [batch, 1, num_q_heads, head_dim] -> [batch, q_dim]
    attn_output = attn_output.view(batch_size, -1)

    # 8. Output projection
    if timing and timing.enabled:
        with timing.time("attn.out_proj"):
            attn_output = F.linear(attn_output, out_weight, out_bias)
    else:
        attn_output = F.linear(attn_output, out_weight, out_bias)

    return attn_output, k_new, v_new


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm implementation."""
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(dtype)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Q and K.

    Args:
        q: Query tensor [batch, seq_len, num_heads, head_dim]
        k: Key tensor [batch, seq_len, num_kv_heads, head_dim]
        cos: Cosine values [max_pos, head_dim//2] or [batch, head_dim//2] if pre-indexed
        sin: Sine values [max_pos, head_dim//2] or [batch, head_dim//2] if pre-indexed
        position_ids: Position IDs [batch]

    Returns:
        Tuple of (q_rotated, k_rotated)
    """
    # Get cos/sin for current positions
    # If cos/sin are already position-indexed (shape [batch, head_dim//2]), skip indexing
    batch_size = position_ids.shape[0]
    if cos.shape[0] != batch_size:
        # cos: [max_pos, head_dim//2] -> index to [batch, head_dim//2]
        cos = cos[position_ids]
        sin = sin[position_ids]

    # Expand for broadcast: [batch, 1, 1, head_dim//2]
    cos = cos.unsqueeze(1).unsqueeze(2)
    sin = sin.unsqueeze(1).unsqueeze(2)

    # Apply rotation
    q_rotated = _rotate_half(q, cos, sin)
    k_rotated = _rotate_half(k, cos, sin)

    return q_rotated, k_rotated


def _rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to tensor.

    x: [..., head_dim]
    cos, sin: [..., head_dim//2]
    """
    # Split x into two halves
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]

    # Rotate
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin,
    ], dim=-1)

    return x_rotated.to(x.dtype)
