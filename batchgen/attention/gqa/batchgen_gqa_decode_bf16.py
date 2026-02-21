"""GQA decode with automatic backend selection: custom WGMMA kernel or FA3.

On SM90+ GPUs (H20/H100), uses the custom WGMMA+TMA decode kernel for
head_dim=64/80/128. Falls back to FlashAttention 3 on other architectures
or if the custom kernel is unavailable.
"""

import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_custom_kernel = None
_custom_kernel_checked = False
_backend_logged = False


def _check_custom_kernel():
    global _custom_kernel, _custom_kernel_checked
    if _custom_kernel_checked:
        return _custom_kernel is not None
    _custom_kernel_checked = True

    if not torch.cuda.is_available():
        print("[batchgen_decode] CUDA not available, using FA3 fallback", flush=True)
        return False

    device_name = torch.cuda.get_device_name()
    print(f"[batchgen_decode] GPU: {device_name}", flush=True)

    # Custom WGMMA decode kernel is optimized for H20 only
    if "H20" not in device_name:
        print(f"[batchgen_decode] Not H20 ({device_name}), using FA3 fallback", flush=True)
        return False

    try:
        from batchgen_kernels.attention.decode import attention_decode_bf16
        _custom_kernel = attention_decode_bf16
        print("[batchgen_decode] Custom WGMMA decode kernel loaded successfully", flush=True)
        return True
    except Exception as e:
        print(f"[batchgen_decode] Failed to load custom kernel: {e}", flush=True)
        print("[batchgen_decode] Falling back to FA3", flush=True)
        return False


def batchgen_gqa_decode_bf16(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cache_seqlens: torch.Tensor,
    block_table: torch.Tensor,
    sinks: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """GQA decode with automatic backend selection.

    Same signature as gqa_decode_fa() — drop-in replacement.

    On SM90+ with the custom kernel available, uses the WGMMA+TMA decode
    kernel. Otherwise falls back to FlashAttention 3/2.

    Args:
        q: Query tensor (batch, seqlen_q=1, nheads, headdim) BF16
        k_cache: Paged key cache (num_blocks, page_size, nheads_kv, headdim) BF16
        v_cache: Paged value cache (num_blocks, page_size, nheads_kv, headdim) BF16
        cache_seqlens: Current sequence lengths (batch,) INT32
        block_table: Page table mapping (batch, max_blocks_per_seq) INT32
        sinks: Optional per-head sink values (nheads,)
        softmax_scale: Scale factor for QK^T (default: 1/sqrt(headdim))
        sliding_window: Optional sliding window size for local attention

    Returns:
        (output, lse) where:
            output: (batch, seqlen_q=1, nheads, headdim) BF16
            lse: Log-sum-exp values or None
    """
    global _backend_logged

    if _check_custom_kernel():
        if not _backend_logged:
            print(f"[batchgen_decode] Using custom WGMMA kernel "
                  f"(q={list(q.shape)}, headdim={q.shape[-1]}, "
                  f"sliding_window={sliding_window})", flush=True)
            _backend_logged = True

        # Shape adaptation: FA format -> custom kernel format
        # Q: [batch, 1, nheads, headdim] -> [batch, nheads, headdim]
        q_3d = q.squeeze(1)

        sw = sliding_window if sliding_window is not None and sliding_window > 0 else 0

        output_3d, lse = _custom_kernel(
            q=q_3d,
            kcache=k_cache,
            vcache=v_cache,
            block_ids=block_table,
            num_seq_kvcache=cache_seqlens,
            sliding_window=sw,
        )

        # Output: [batch, nheads, headdim] -> [batch, 1, nheads, headdim]
        output = output_3d.unsqueeze(1)

        # LSE shape: kernel returns [batch, nheads], sink correction expects [batch, nheads, seqlen=1]
        if lse is not None and sinks is not None:
            lse = lse.unsqueeze(-1)  # [batch, nheads] -> [batch, nheads, 1]
            from .sink_correction import apply_sink_correction
            output = apply_sink_correction(output, lse, sinks)

        return output, lse

    # Fallback to FA3/FA2
    if not _backend_logged:
        print(f"[batchgen_decode] Using FA3 fallback "
              f"(q={list(q.shape)}, headdim={q.shape[-1]})", flush=True)
        _backend_logged = True

    from .fa_decode import gqa_decode_fa
    return gqa_decode_fa(
        q, k_cache, v_cache, cache_seqlens, block_table,
        sinks=sinks, softmax_scale=softmax_scale,
        sliding_window=sliding_window,
    )
