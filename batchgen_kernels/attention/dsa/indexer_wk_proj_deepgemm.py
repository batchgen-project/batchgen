# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  Copyright (c) 2025-2026 BatchGen Team                                            #
# ---------------------------------------------------------------------------- #

"""Indexer wk projection, DeepGEMM dense-FP8 version.

Version management: this file is a NEW version of the projection next to the
custom WGMMA kernel in ``fused_indexer_kv_proj_cuda.py``, which is left
untouched and stays selectable (see
``batchgen/models/glm/glm5/indexer_wk_backend.py``).

What changes versus the custom kernel:

  * The custom kernel owns its activation quantize -- a PER-ROW BF16->FP8 cast
    of the layer's hidden states, with one fp32 scale per row, feeding a TMA
    descriptor it also owns. This version has no quantize of its own: it
    consumes the 128-BLOCK-scaled FP8 hidden that the same layer's qkv_a
    projection already produces (``act_quant(..., scale_tma_aligned=True)``),
    so the indexer's private quantize launch disappears from the step.
  * The GEMM itself is DeepGEMM's dense FP8 NT GEMM, used as installed (no
    fork, no patch), the same entry point the GLM-5 shared expert and
    ``batchgen.gemm.w8a8_deepgemm``'s unmasked path call.

Row validity: the dense GEMM computes ALL M rows of the activation, not just
the first ``num_valid_tokens``. It needs no mask of its own because
``act_quant``/``per_token_blocked_quantize_bf16_to_fp8_1d`` already store EXACT
ZEROS into the FP8 rows at or past ``num_valid_tokens`` (and 1e-12 into their
scales), so those rows' products are exact zeros -- the same bytes the custom
kernel's ``valid_m`` epilogue wrote there.

Scale layout: the activation scale is consumed in the MN-major, TMA-aligned
layout ``act_quant(scale_tma_aligned=True)`` produces, i.e. logical
``[M, K/128]`` with ``stride == (1, align4(M))``. That is the layout DeepGEMM
wants, so nothing transposes or copies it per layer. The weight keeps the
checkpoint's own ``[N, K]`` e4m3 bytes and its ``[ceil(N/128), K/128]`` fp32
block scales, exactly as the shared-expert GEMMs pass ``weight_scale_inv``
straight through.
"""

import torch

import deep_gemm

_SCALE_BLOCK = 128

# DeepGEMM's unmasked FP8 path in this repo is always driven with the UE8M0
# re-cast disabled (see batchgen/gemm/w8a8_deepgemm.py), and that is the
# configuration the 1b-v probe timed and numerically validated at these dims.
_DISABLE_UE8M0_CAST = True


class IndexerWkDeepGemmWeights:
    """The indexer wk weight in DeepGEMM's dense-FP8 NT operand layout.

    Built once per layer at init (never inside a capture) from the checkpoint's
    own FP8 weight and block scales -- no dequantize/requantize round trip, so
    the weight bytes the GEMM reads are the checkpoint's.
    """

    def __init__(self, wk_fp8: torch.Tensor, wk_scale_inv: torch.Tensor) -> None:
        if wk_fp8.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "indexer wk DeepGEMM weight must be float8_e4m3fn, got "
                f"{wk_fp8.dtype}; the DeepGEMM wk backend requires an FP8 "
                "checkpoint"
            )
        if wk_fp8.dim() != 2:
            raise ValueError(
                f"indexer wk weight must be rank 2 [N, K], got {tuple(wk_fp8.shape)}"
            )
        if wk_scale_inv.dtype != torch.float32:
            raise TypeError(
                f"indexer wk block scales must be fp32, got {wk_scale_inv.dtype}"
            )
        N, K = wk_fp8.shape
        if K % _SCALE_BLOCK != 0:
            raise ValueError(
                f"indexer wk K={K} must be a multiple of {_SCALE_BLOCK} for "
                "block-scaled FP8"
            )
        expected_scale = (
            (N + _SCALE_BLOCK - 1) // _SCALE_BLOCK,
            K // _SCALE_BLOCK,
        )
        if tuple(wk_scale_inv.shape) != expected_scale:
            raise ValueError(
                f"indexer wk block scales must have shape {expected_scale} for "
                f"a [{N}, {K}] weight, got {tuple(wk_scale_inv.shape)}"
            )
        if wk_fp8.device != wk_scale_inv.device:
            raise ValueError(
                "indexer wk weight and its block scales must be on the same "
                f"device, got {wk_fp8.device} and {wk_scale_inv.device}"
            )
        self.w_fp8 = wk_fp8.contiguous()
        self.w_scale = wk_scale_inv.contiguous()
        self.N = int(N)
        self.K = int(K)


def indexer_wk_proj_deepgemm_out(
    hidden_fp8: torch.Tensor,
    hidden_scale: torch.Tensor,
    weights: IndexerWkDeepGemmWeights,
    out: torch.Tensor,
) -> torch.Tensor:
    """``out[:] = hidden @ weights.wk.T`` in dense FP8, into a static buffer.

    ``hidden_fp8`` / ``hidden_scale`` are the SHARED blocked-FP8 activation
    (the qkv_a projection's operand), not a private quantize of this path.
    ``out`` is the caller's preallocated bf16 ``[M, N]`` buffer and is written
    in place -- the same buffer the custom kernel's ``_out`` variant wrote.
    """
    M, K = hidden_fp8.shape
    if hidden_fp8.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"indexer wk activation must be float8_e4m3fn, got {hidden_fp8.dtype}"
        )
    if K != weights.K:
        raise ValueError(
            f"indexer wk activation K={K} does not match the weight's K={weights.K}"
        )
    if out.shape != (M, weights.N) or out.dtype != torch.bfloat16:
        raise ValueError(
            f"indexer wk out must be bf16 with shape {(M, weights.N)}, got "
            f"{tuple(out.shape)} {out.dtype}"
        )
    deep_gemm.fp8_gemm_nt(
        (hidden_fp8, hidden_scale),
        (weights.w_fp8, weights.w_scale),
        out,
        disable_ue8m0_cast=_DISABLE_UE8M0_CAST,
    )
    return out
