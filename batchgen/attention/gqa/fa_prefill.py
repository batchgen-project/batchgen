"""GQA prefill using flash-attention with optional sink correction.

Uses flash_attn_varlen_func for variable-length (unpadded) sequences.
Supports both FA2 (Ampere) and FA3 (Hopper).

SINK TOKEN CONFIGURATION:
- USE_VANILLA_FOR_SINKS=False (default): Use FlashAttention with sigmoid post-correction
- USE_VANILLA_FOR_SINKS=True: Use vanilla PyTorch with inline softmax_with_sinks

Set BATCHGEN_VANILLA_SINKS=1 environment variable to enable vanilla path.
"""

import os
import torch
from typing import Optional, Tuple

# Note: We check env var at runtime in the function, not at import time,
# because worker processes may import modules before env vars are set.

# Detect which flash attention version is available
_USE_FA3 = False
_flash_varlen_func = None
_flash_attn_forward = None  # FA3 low-level API that returns LSE

try:
    from flash_attn_interface import flash_attn_varlen_func as _fa3_varlen_func
    from flash_attn_interface import _flash_attn_forward as _fa3_forward
    _USE_FA3 = True
    _flash_varlen_func = _fa3_varlen_func
    _flash_attn_forward = _fa3_forward
except ImportError:
    pass

if _flash_varlen_func is None:
    try:
        from flash_attn import flash_attn_varlen_func as _fa2_varlen_func
        _flash_varlen_func = _fa2_varlen_func
    except ImportError:
        pass


def gqa_prefill_fa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GQA prefill using flash-attention with optional sink correction.

    Args:
        q: Query tensor, unpadded (total_q, nheads, headdim)
        k: Key tensor, unpadded (total_k, nheads_kv, headdim)
        v: Value tensor, unpadded (total_k, nheads_kv, headdim)
        cu_seqlens_q: Cumulative sequence lengths for Q (batch + 1,)
        cu_seqlens_k: Cumulative sequence lengths for K (batch + 1,)
        max_seqlen_q: Maximum sequence length in Q batch
        max_seqlen_k: Maximum sequence length in K batch
        sinks: Optional per-head sink values (nheads,)
        softmax_scale: Scale factor for QK^T (default: 1/sqrt(headdim))
        sliding_window: Optional sliding window size for local attention

    Returns:
        Tuple of:
            - output: Attention output (total_q, nheads, headdim)
            - lse: Log-sum-exp values or None if sinks not provided
    """
    # Use vanilla PyTorch path for sinks when configured (correct inline softmax)
    # Check env var at runtime (not import time) for worker process compatibility
    use_vanilla_for_sinks = os.environ.get("BATCHGEN_VANILLA_SINKS", "0") == "1"
    use_vanilla = (sinks is not None and use_vanilla_for_sinks)

    # Debug: Log which attention path is being used
    if os.environ.get("BATCHGEN_DEBUG_SINK", "0") == "1":
        print(f"[GQA PREFILL] BATCHGEN_VANILLA_SINKS={os.environ.get('BATCHGEN_VANILLA_SINKS', '0')}, sinks={sinks is not None}, use_vanilla={use_vanilla}")

    if use_vanilla:
        print(f"[GQA PREFILL] >>> ENTERING VANILLA ATTENTION PATH <<<")
        from .gqa_attention import gqa_attention_prefill

        # Convert varlen format to padded batch format for vanilla attention
        # q: (total_q, nheads, headdim) -> padded: (batch, nheads, max_seqlen, headdim)
        batch = len(cu_seqlens_q) - 1
        nheads = q.shape[1]
        nheads_kv = k.shape[1]
        headdim = q.shape[2]

        if softmax_scale is None:
            softmax_scale = headdim ** -0.5

        # Pad each sequence to max_seqlen
        q_padded = torch.zeros(batch, nheads, max_seqlen_q, headdim, dtype=q.dtype, device=q.device)
        k_padded = torch.zeros(batch, nheads_kv, max_seqlen_k, headdim, dtype=k.dtype, device=k.device)
        v_padded = torch.zeros(batch, nheads_kv, max_seqlen_k, headdim, dtype=v.dtype, device=v.device)

        for i in range(batch):
            q_start = cu_seqlens_q[i].item()
            q_end = cu_seqlens_q[i + 1].item()
            q_len = q_end - q_start
            # varlen: (total, heads, dim) -> padded: (batch, heads, seq, dim)
            q_padded[i, :, :q_len, :] = q[q_start:q_end].permute(1, 0, 2)

            k_start = cu_seqlens_k[i].item()
            k_end = cu_seqlens_k[i + 1].item()
            k_len = k_end - k_start
            k_padded[i, :, :k_len, :] = k[k_start:k_end].permute(1, 0, 2)
            v_padded[i, :, :k_len, :] = v[k_start:k_end].permute(1, 0, 2)

        # Run vanilla prefill with correct inline softmax_with_sinks
        output_padded = gqa_attention_prefill(
            query=q_padded,
            key=k_padded,
            value=v_padded,
            sinks=sinks,
            scale=softmax_scale,
            sliding_window=sliding_window,
        )

        # Convert back to varlen format: (batch, heads, seq, dim) -> (total, heads, dim)
        output_list = []
        for i in range(batch):
            q_start = cu_seqlens_q[i].item()
            q_end = cu_seqlens_q[i + 1].item()
            q_len = q_end - q_start
            # padded: (batch, heads, seq, dim) -> varlen: (seq, heads, dim)
            output_list.append(output_padded[i, :, :q_len, :].permute(1, 0, 2))

        output = torch.cat(output_list, dim=0)
        return output, None

    if _flash_varlen_func is None:
        raise ImportError("Neither flash_attn_interface (FA3) nor flash_attn (FA2) is available")

    # Set up window_size parameter
    # Flash attention uses (window_size_left, window_size_right)
    # For causal with sliding window: (sliding_window - 1, 0)
    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)  # No windowing

    # Default scale
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    lse = None

    if _USE_FA3:
        # Use _flash_attn_forward to get LSE (flash_attn_varlen_func doesn't return it)
        # Call pattern matches FlashAttnVarlenFunc.forward
        output, lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None, None,  # k_new, v_new
            None,  # qv
            None,  # out (let it allocate)
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # cu_seqlens_k_new
            None,  # seqused_q
            None,  # seqused_k
            max_seqlen_q,
            max_seqlen_k,
            None, None, None,  # page_table, kv_batch_idx, leftpad_k
            None, None, None,  # rotary_cos, rotary_sin, seqlens_rotary
            None, None, None,  # q_descale, k_descale, v_descale
            softmax_scale,
            causal=True,
            window_size=window_size,
        )
    else:
        # FA2 needs return_softmax_lse=True to get LSE
        if sinks is not None:
            result = _flash_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
                return_softmax_lse=True,
            )
            if isinstance(result, tuple):
                output = result[0]
                lse = result[1]
            else:
                output = result
        else:
            output = _flash_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=window_size,
            )

    # Apply sink correction if sinks provided
    if sinks is not None and lse is not None:
        from .sink_correction import apply_sink_correction
        output = apply_sink_correction(output, lse, sinks)

    return output, lse
