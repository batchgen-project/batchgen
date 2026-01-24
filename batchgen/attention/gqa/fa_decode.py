"""GQA decode using flash-attention with paged KV cache and sink correction.

Uses flash_attn_with_kvcache for efficient decode with paged attention.
Supports both FA2 (Ampere) and FA3 (Hopper).

SINK TOKEN CONFIGURATION:
- USE_VANILLA_FOR_SINKS=False (default): Use FlashAttention with numerically stable
  sigmoid post-correction (fast, handles padding correctly)
- USE_VANILLA_FOR_SINKS=True: Use vanilla PyTorch decode with inline softmax_with_sinks
  (slower, for debugging only)

Set BATCHGEN_VANILLA_SINKS=1 environment variable to enable vanilla path.
"""

import os
import torch
from typing import Optional, Tuple

# Note: We check env var at runtime in the function, not at import time,
# because worker processes may import modules before env vars are set.

# Detect which flash attention version is available
_USE_FA3 = False
_flash_with_kvcache = None

try:
    from flash_attn_interface import flash_attn_with_kvcache as _fa3_with_kvcache
    _USE_FA3 = True
    _flash_with_kvcache = _fa3_with_kvcache
except ImportError:
    pass

if _flash_with_kvcache is None:
    try:
        from flash_attn import flash_attn_with_kvcache as _fa2_with_kvcache
        _flash_with_kvcache = _fa2_with_kvcache
    except ImportError:
        pass


def _gather_paged_kv_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather paged KV cache into contiguous format for vanilla attention.

    Args:
        k_cache: Paged key cache (num_blocks, page_size, nheads_kv, headdim)
        v_cache: Paged value cache (num_blocks, page_size, nheads_kv, headdim)
        block_table: Page table mapping (batch, max_blocks_per_seq) int32
        cache_seqlens: Current sequence lengths (batch,) int32

    Returns:
        Tuple of:
            - k_contig: Contiguous key cache (batch, max_seqlen, nheads_kv, headdim)
            - v_contig: Contiguous value cache (batch, max_seqlen, nheads_kv, headdim)
    """
    batch = block_table.shape[0]
    max_blocks = block_table.shape[1]
    page_size = k_cache.shape[1]
    nheads_kv = k_cache.shape[2]
    headdim = k_cache.shape[3]

    max_seqlen = int(cache_seqlens.max().item())

    # Pre-allocate contiguous buffers
    k_contig = torch.zeros(batch, max_seqlen, nheads_kv, headdim,
                           dtype=k_cache.dtype, device=k_cache.device)
    v_contig = torch.zeros(batch, max_seqlen, nheads_kv, headdim,
                           dtype=v_cache.dtype, device=v_cache.device)

    # Gather pages for each batch element
    for b in range(batch):
        seqlen = int(cache_seqlens[b].item())
        num_blocks_needed = (seqlen + page_size - 1) // page_size
        pos = 0
        for block_idx in range(num_blocks_needed):
            page_id = block_table[b, block_idx].item()
            copy_len = min(page_size, seqlen - pos)
            k_contig[b, pos:pos+copy_len] = k_cache[page_id, :copy_len]
            v_contig[b, pos:pos+copy_len] = v_cache[page_id, :copy_len]
            pos += copy_len

    return k_contig, v_contig


def gqa_decode_fa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GQA decode using flash-attention with paged KV cache and sink correction.

    Args:
        q: Query tensor (batch, seqlen_q, nheads, headdim)
            For standard decode, seqlen_q = 1
        k_cache: Paged key cache (num_blocks, page_size, nheads_kv, headdim)
        v_cache: Paged value cache (num_blocks, page_size, nheads_kv, headdim)
        cache_seqlens: Current sequence lengths (batch,) int32
        block_table: Page table mapping (batch, max_blocks_per_seq) int32
        sinks: Optional per-head sink values (nheads,)
        softmax_scale: Scale factor for QK^T (default: 1/sqrt(headdim))
        sliding_window: Optional sliding window size for local attention

    Returns:
        Tuple of:
            - output: Attention output (batch, seqlen_q, nheads, headdim)
            - lse: Log-sum-exp values or None if sinks not provided
    """
    # Use vanilla PyTorch path for sinks when configured (correct inline softmax)
    # Check env var at runtime (not import time) for worker process compatibility
    use_vanilla_for_sinks = os.environ.get("BATCHGEN_VANILLA_SINKS", "0") == "1"
    use_vanilla = (sinks is not None and use_vanilla_for_sinks)

    # Debug: Log which attention path is being used
    if os.environ.get("BATCHGEN_DEBUG_SINK", "0") == "1":
        print(f"[GQA DECODE] BATCHGEN_VANILLA_SINKS={os.environ.get('BATCHGEN_VANILLA_SINKS', '0')}, sinks={sinks is not None}, use_vanilla={use_vanilla}")

    if use_vanilla:
        from .gqa_attention import gqa_attention_decode

        # Gather paged KV cache into contiguous format
        k_contig, v_contig = _gather_paged_kv_cache(
            k_cache, v_cache, block_table, cache_seqlens
        )

        # Convert from FA format to vanilla format
        # FA: (batch, seqlen, nheads, headdim) -> Vanilla: (batch, nheads, seqlen, headdim)
        q_vanilla = q.permute(0, 2, 1, 3)  # [batch, nheads, 1, headdim]
        k_vanilla = k_contig.permute(0, 2, 1, 3)  # [batch, nheads_kv, max_seqlen, headdim]
        v_vanilla = v_contig.permute(0, 2, 1, 3)  # [batch, nheads_kv, max_seqlen, headdim]

        # Run vanilla decode with correct inline softmax_with_sinks
        output_vanilla = gqa_attention_decode(
            query=q_vanilla,
            key=k_vanilla,
            value=v_vanilla,
            sinks=sinks,
            scale=softmax_scale,
            sliding_window=sliding_window,
            cache_seqlens=cache_seqlens,
        )

        # Convert back to FA format
        output = output_vanilla.permute(0, 2, 1, 3)  # [batch, seqlen_q, nheads, headdim]
        return output, None

    # FlashAttention path
    if _flash_with_kvcache is None:
        raise ImportError("Neither flash_attn_interface (FA3) nor flash_attn (FA2) is available")

    # Set up window_size parameter
    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)

    # Default scale
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    lse = None

    # FA3 uses page_table, FA2 uses block_table
    page_table_kwarg = "page_table" if _USE_FA3 else "block_table"

    if sinks is not None:
        # Need LSE for sink correction
        result = _flash_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            return_softmax_lse=True,
            **{page_table_kwarg: block_table},
        )
        # Handle return value - could be (output, lse) or (output, lse, ...)
        if isinstance(result, tuple):
            output = result[0]
            lse = result[1]
        else:
            output = result

        # DEBUG: Print raw LSE from FlashAttention before sink correction
        if os.environ.get("BATCHGEN_DEBUG_SINK", "0") == "1":
            import torch as _torch
            with _torch.no_grad():
                if lse is not None:
                    print(f"\n[FA DECODE] Raw LSE from FlashAttention:")
                    print(f"[FA DECODE] LSE shape={lse.shape}, dtype={lse.dtype}")
                    print(f"[FA DECODE] LSE min={lse.min().item():.4f}, max={lse.max().item():.4f}, mean={lse.float().mean().item():.4f}")
                    print(f"[FA DECODE] Output shape={output.shape}, dtype={output.dtype}")
                    print(f"[FA DECODE] Output (pre-sink) min={output.min().item():.4f}, max={output.max().item():.4f}")
                    # Check for NaN/Inf
                    has_nan = _torch.isnan(lse).any().item()
                    has_inf = _torch.isinf(lse).any().item()
                    print(f"[FA DECODE] LSE has NaN={has_nan}, has Inf={has_inf}")
                else:
                    print(f"[FA DECODE] WARNING: LSE is None despite return_softmax_lse=True!")
    else:
        output = _flash_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
            **{page_table_kwarg: block_table},
        )

    # Apply sink correction if sinks provided
    if sinks is not None and lse is not None:
        from .sink_correction import apply_sink_correction
        output = apply_sink_correction(output, lse, sinks)

    return output, lse


def gqa_decode_fa_contiguous(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GQA decode with contiguous (non-paged) KV cache.

    Use this when KV cache is stored contiguously without page tables.

    Args:
        q: Query tensor (batch, seqlen_q, nheads, headdim)
        k_cache: Contiguous key cache (batch, max_seqlen, nheads_kv, headdim)
        v_cache: Contiguous value cache (batch, max_seqlen, nheads_kv, headdim)
        cache_seqlens: Current sequence lengths (batch,) int32
        sinks: Optional per-head sink values (nheads,)
        softmax_scale: Scale factor for QK^T
        sliding_window: Optional sliding window size

    Returns:
        Tuple of:
            - output: Attention output (batch, seqlen_q, nheads, headdim)
            - lse: Log-sum-exp values or None if sinks not provided
    """
    # Use vanilla PyTorch path for sinks when configured (correct inline softmax)
    # Check env var at runtime (not import time) for worker process compatibility
    use_vanilla_for_sinks = os.environ.get("BATCHGEN_VANILLA_SINKS", "0") == "1"
    use_vanilla = (sinks is not None and use_vanilla_for_sinks)

    if use_vanilla:
        from .gqa_attention import gqa_attention_decode

        # Convert from FA format to vanilla format
        # FA: (batch, seqlen, nheads, headdim) -> Vanilla: (batch, nheads, seqlen, headdim)
        q_vanilla = q.permute(0, 2, 1, 3)  # [batch, nheads, 1, headdim]
        k_vanilla = k_cache.permute(0, 2, 1, 3)  # [batch, nheads_kv, max_seqlen, headdim]
        v_vanilla = v_cache.permute(0, 2, 1, 3)  # [batch, nheads_kv, max_seqlen, headdim]

        # Run vanilla decode with correct inline softmax_with_sinks
        output_vanilla = gqa_attention_decode(
            query=q_vanilla,
            key=k_vanilla,
            value=v_vanilla,
            sinks=sinks,
            scale=softmax_scale,
            sliding_window=sliding_window,
            cache_seqlens=cache_seqlens,
        )

        # Convert back to FA format
        output = output_vanilla.permute(0, 2, 1, 3)  # [batch, seqlen_q, nheads, headdim]
        return output, None

    # FlashAttention path
    if _flash_with_kvcache is None:
        raise ImportError("Neither flash_attn_interface (FA3) nor flash_attn (FA2) is available")

    if sliding_window is not None and sliding_window > 0:
        window_size = (sliding_window - 1, 0)
    else:
        window_size = (-1, -1)

    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    lse = None

    if sinks is not None:
        result = _flash_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
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
        output = _flash_with_kvcache(
            q,
            k_cache,
            v_cache,
            cache_seqlens=cache_seqlens,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=window_size,
        )

    if sinks is not None and lse is not None:
        from .sink_correction import apply_sink_correction
        output = apply_sink_correction(output, lse, sinks)

    return output, lse
