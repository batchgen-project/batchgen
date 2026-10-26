"""Python entry point for the fused add+RMSNorm+FP8-quantize CUDA kernel.

The kernel lives in ``src/attention/csrc/add_rmsnorm_fp8_quant.cu`` and is a NEW
version beside ``src/attention/csrc/rmsnorm.cu``'s
``add_rmsnorm_bf16_6144_kernel``, which is untouched and still reachable through
``batchgen.attention.fused_kernels.cuda_add_rmsnorm``. See the ``.cu``'s header
for the statement-level byte-identity argument against that kernel and against
``_per_token_blocked_quantize_1d_kernel``.
"""

from typing import Optional, Tuple

import torch

_ext = None


def _get_ext():
    global _ext
    if _ext is None:
        import batchgen_kernels
        _ext = batchgen_kernels.load_extension(
            "batchgen_kernels.attention._C_add_rmsnorm_fp8_quant"
        )
    return _ext


def preload_add_rmsnorm_fp8_quant_kernel():
    """Resolve the extension before capture, so no JIT happens inside a graph."""
    return _get_ext()


def cuda_add_rmsnorm_fp8_quant(
    residual: torch.Tensor,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    norm_num_valid_tokens: Optional[torch.Tensor] = None,
    quant_num_valid_tokens: Optional[torch.Tensor] = None,
    row_valid_mask: Optional[torch.Tensor] = None,
    scale_tma_aligned: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused residual add + RMSNorm + per-token 128-block FP8 cast.

    ``residual`` is updated in place. Returns
    ``(normed, residual, x_fp8, x_scale)`` where ``normed``/``residual`` are
    bit-for-bit ``cuda_add_rmsnorm``'s and ``x_fp8``/``x_scale`` are
    bit-for-bit ``per_token_blocked_quantize_bf16_to_fp8_1d(normed, ...)``'s.

    The two validity gates are separate because the GLM-5 call sites disagree:
    the norm is launched over EVERY row (padding included) while its quantize
    gates padding rows to exact-zero FP8 bytes and 1e-12 scales.

    ``norm_num_valid_tokens``
        Skip the norm/residual work for rows at or past the count, exactly as
        ``cuda_add_rmsnorm``'s ``num_valid_tokens`` does. Requires
        ``quant_num_valid_tokens`` to be the SAME tensor: a row whose
        ``normed_out`` the kernel never wrote has no defined value to quantize.
    ``quant_num_valid_tokens``
        Rows at or past the count get exact-zero FP8 bytes and 1e-12 scales.
    ``row_valid_mask``
        ``cuda_add_rmsnorm``'s per-row PREDICATE on ``hidden`` (never a
        multiplicative factor), unchanged.
    ``scale_tma_aligned``
        Allocate the scales as the MN-major view DeepGEMM consumes without a
        repack, as ``act_quant(..., scale_tma_aligned=True)`` does.

    Only hidden_size 6144 bf16 is supported; anything else raises from the
    extension rather than rerouting to the unfused pair.
    """
    results = _get_ext().add_rmsnorm_fp8_quant_forward(
        residual,
        hidden,
        weight,
        eps,
        norm_num_valid_tokens,
        quant_num_valid_tokens,
        row_valid_mask,
        scale_tma_aligned,
    )
    return results[0], results[1], results[2], results[3]
