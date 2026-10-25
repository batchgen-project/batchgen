"""RMSNorm with a per-token 128-block FP8 (e4m3) cast in the same launch.

NEW KERNEL VERSION. It does NOT replace anything: ``triton/rmsnorm.py``'s
``_rmsnorm_kernel`` and ``triton/fp8_quantize.py``'s
``_per_token_blocked_quantize_1d_kernel`` stay in the tree, stay callable, and
stay selected whenever ``GLM5_NORM_QUANT_FUSE`` is ``False``.

WHY: GLM-5 decode normalizes ``q_a`` and then immediately casts the result to
blocked FP8 for the ``q_b`` GEMM. The norm already holds the normalized row in
registers when it stores it, so the cast is a second launch and a second full
HBM round trip over data that was already resident.

===============================================================================
BYTE IDENTITY
===============================================================================

**The norm half is ``_rmsnorm_kernel`` copied verbatim** -- the same 1D
``[BLOCK_SIZE]`` tile, the same ``tl.sum(x_fp32 * x_fp32, axis=0)`` over that
tile, the same ``/ hidden_size`` with ``hidden_size`` as a RUNTIME argument
(a ``tl.constexpr`` power-of-two divisor could be folded into an exact
reciprocal multiply, which is a different instruction), the same
``1.0 / tl.sqrt(variance + eps)``, the same ``x_fp32 * inv_rms * w_fp32``
expression order, and the same ``out.to(x.dtype)`` store.

Two things about that are load-bearing and easy to break:

1. **The tile stays 1D for the sum.** ``tl.sum`` is a tree reduction, so the
   same 2048 values summed as a ``[2048]`` tile and as a ``[16, 128]`` tile
   associate differently and need not give the same float. Only the quantize's
   per-block ``amax`` sees the 2D shape -- and ``max`` is exactly associative
   and commutative, so its tree is free.
2. **The launch passes no ``num_warps``.** ``tl.sum``'s tree depends on the
   warp count, so the fused launch must take the same default
   ``_rmsnorm_kernel``'s launch takes, which passes neither ``num_warps`` nor
   ``num_stages``. Adding a tuned ``num_warps`` here would change the BF16 norm
   output and therefore the FP8 bytes derived from it.

**The quantize half is ``_per_token_blocked_quantize_1d_kernel`` copied
statement for statement**, over the BF16-ROUNDED normalized value -- the same
bits the standalone kernel would have loaded back from global memory, not the
unrounded FP32 accumulator:

    q_bf16 = out.to(x.dtype)          # the value stored to O_ptr
    q_float = q_bf16.to(tl.float32)   # what the standalone kernel loaded
    amax = tl.max(tl.abs(q_float), axis=1)
    amax = tl.maximum(amax, FP8_E4M3_MIN_NORMAL)
    scale = tl.maximum(amax * FP8_SAFE_MAX_INV, EPSILON)
    q = q_float / scale
    q = tl.minimum(q, FP8_SAFE_MAX); q = tl.maximum(q, -FP8_SAFE_MAX)
    q = tl.where(tl.abs(q) < 1e30, q, 0.0)

including the constants' spelling (``1.0 / 448.0`` as a Python division, not a
hand-written decimal), the ``HAS_VALID_TOKENS`` branch's exact-zero FP8 row and
``EPSILON`` scales, and the ``block_valid`` mask on the scale store.

``dim % 128 == 0`` is required, which makes ``next_power_of_2(dim) ==
NUM_BLOCKS_P2 * 128``: the 1D tile reshapes onto the quantize's 2D tile with
the identical index mapping, and the 1D tile's masked-out tail lands exactly on
the 2D tile's padded block rows -- where the standalone kernel also loaded
zeros and also suppressed the scale store.

The runtime bitwise gate is ``tests/gpu/test_glm52_norm_quant_fuse_bitwise.py``;
``tests/test_glm52_norm_quant_fuse.py`` pins this file against the two source
kernels on CPU.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_fp8_quant_kernel(
    X_ptr,                  # [num_rows, hidden_size] bf16 input
    W_ptr,                  # [hidden_size] bf16 weight
    O_ptr,                  # [num_rows, hidden_size] bf16 normed output
    Q_ptr,                  # [num_rows, dim] float8_e4m3fn output
    S_ptr,                  # [num_rows, NUM_BLOCKS] fp32 block scales
    num_valid_tokens_ptr,
    stride_x,               # row stride of X
    stride_o,               # row stride of O
    q_stride_m, q_stride_d,
    s_stride_m, s_stride_b,
    hidden_size,            # RUNTIME, not constexpr -- see the module docstring
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,        # next_power_of_2(hidden_size)
    dim: tl.constexpr,               # == hidden_size, for the quantize indexing
    QUANT_BLOCK: tl.constexpr,       # 128 -- fp8 blockwise scale block
    NUM_BLOCKS: tl.constexpr,        # dim // QUANT_BLOCK
    NUM_BLOCKS_P2: tl.constexpr,     # pow-2 pad of NUM_BLOCKS
    HAS_VALID_TOKENS: tl.constexpr,
):
    """One program per row: RMSNorm, then the per-token 128-block FP8 cast.

    The two halves are the unfused kernels' bodies; see the module docstring
    for the statement-level correspondence and for the two invariants (1D sum
    tile, default ``num_warps``) that keep the BF16 output bit-identical.
    """
    # The reshape below is only index-preserving when the 1D tile and the 2D
    # quantize tile cover the same padded width.
    tl.static_assert(
        BLOCK_SIZE == NUM_BLOCKS_P2 * QUANT_BLOCK,
        "fused norm/quant requires next_pow2(dim) == NUM_BLOCKS_P2 * 128",
    )

    FP8_SAFE_MAX: tl.constexpr = 448.0
    FP8_SAFE_MAX_INV: tl.constexpr = 1.0 / 448.0
    FP8_E4M3_MIN_NORMAL: tl.constexpr = 1.52587890625e-05
    EPSILON: tl.constexpr = 1e-12

    # ---- _rmsnorm_kernel, verbatim ---------------------------------------
    row_idx = tl.program_id(0)
    row_start_x = row_idx * stride_x
    row_start_o = row_idx * stride_o

    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < hidden_size

    x = tl.load(X_ptr + row_start_x + cols, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)

    sum_sq = x_fp32 * x_fp32

    variance = tl.sum(sum_sq, axis=0) / hidden_size

    inv_rms = 1.0 / tl.sqrt(variance + eps)

    w = tl.load(W_ptr + cols, mask=mask, other=1.0)
    w_fp32 = w.to(tl.float32)

    out = x_fp32 * inv_rms * w_fp32
    out_typed = out.to(x.dtype)
    tl.store(O_ptr + row_start_o + cols, out_typed, mask=mask)

    # ---- _per_token_blocked_quantize_1d_kernel, over out_typed -----------
    pid_m = row_idx.to(tl.int64)

    block_ids = tl.arange(0, NUM_BLOCKS_P2)
    col_in_block = tl.arange(0, QUANT_BLOCK).to(tl.int64)

    cols_2d = block_ids[:, None].to(tl.int64) * QUANT_BLOCK + col_in_block[None, :]
    col_mask = cols_2d < dim

    q_stride_m_i64 = tl.cast(q_stride_m, tl.int64)
    s_stride_m_i64 = tl.cast(s_stride_m, tl.int64)

    q_offsets = pid_m * q_stride_m_i64 + cols_2d * q_stride_d
    block_valid = block_ids < NUM_BLOCKS
    s_offsets = pid_m * s_stride_m_i64 + block_ids.to(tl.int64) * s_stride_b

    if HAS_VALID_TOKENS:
        num_valid_tokens = tl.load(num_valid_tokens_ptr)
        if pid_m >= num_valid_tokens:
            # The BF16 norm above already ran for this row, because
            # `_rmsnorm_kernel` has no validity gate and normalizes every row.
            # Only the FP8 half is gated, and it writes what the standalone
            # quantize's HAS_VALID_TOKENS branch writes.
            tl.store(Q_ptr + q_offsets,
                     tl.zeros([NUM_BLOCKS_P2, QUANT_BLOCK], dtype=tl.float32),
                     mask=col_mask)
            tl.store(S_ptr + s_offsets,
                     EPSILON + tl.zeros([NUM_BLOCKS_P2], dtype=tl.float32),
                     mask=block_valid)
            return

    # The BF16-rounded normalized value, reshaped onto the quantize's tile.
    # Reading it from registers instead of global memory is the whole point of
    # the fusion; `out_typed` is bit-for-bit what O_ptr now holds.
    q_bf16 = tl.reshape(out_typed, (NUM_BLOCKS_P2, QUANT_BLOCK))
    q_float = q_bf16.to(tl.float32)

    amax = tl.max(tl.abs(q_float), axis=1)
    amax = tl.maximum(amax, FP8_E4M3_MIN_NORMAL)
    # SGLang formula: scale = amax * (1 / 448). Matches the 3D-grid kernel.
    scale = tl.maximum(amax * FP8_SAFE_MAX_INV, EPSILON)

    q_scaled = q_float / scale[:, None]
    q_scaled = tl.minimum(q_scaled, FP8_SAFE_MAX)
    q_scaled = tl.maximum(q_scaled, -FP8_SAFE_MAX)
    is_finite = tl.abs(q_scaled) < 1e30
    q_scaled = tl.where(is_finite, q_scaled, 0.0)

    tl.store(Q_ptr + q_offsets, q_scaled, mask=col_mask)
    tl.store(S_ptr + s_offsets, scale, mask=block_valid)


def fused_rmsnorm_fp8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    out: torch.Tensor | None = None,
    num_valid_tokens: torch.Tensor | None = None,
    scale_tma_aligned: bool = False,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNorm + per-token 128-block FP8 cast in one launch.

    Returns ``(normed, x_fp8, x_scale)``:

    * ``normed`` is bit-for-bit ``fused_rmsnorm(x, weight, eps, out=out)``;
    * ``x_fp8`` / ``x_scale`` are bit-for-bit
      ``per_token_blocked_quantize_bf16_to_fp8_1d(normed, num_valid_tokens=...,
      scale_tma_aligned=...)``.

    Unsupported configurations FAIL LOUD. There is no reroute to the unfused
    pair: a silent reroute would put the eliminated launch back in the graph
    while the switch still reported it gone.
    """
    if x.dtype != torch.bfloat16:
        raise TypeError(
            f"fused_rmsnorm_fp8_quant requires a bf16 input, got {x.dtype}"
        )
    if block_size != 128:
        raise ValueError(
            "fused_rmsnorm_fp8_quant is the 128-block version, got "
            f"block_size={block_size}"
        )

    orig_shape = x.shape
    hidden_size = orig_shape[-1]
    x_2d = x if x.dim() == 2 else x.reshape(-1, hidden_size)
    num_rows = x_2d.shape[0]

    if hidden_size % block_size != 0:
        raise ValueError(
            "fused_rmsnorm_fp8_quant requires hidden_size % 128 == 0 so the "
            "norm tile and the quantize tile share an index mapping; got "
            f"hidden_size={hidden_size}"
        )
    block_tile = triton.next_power_of_2(hidden_size)
    if block_tile > 8192:
        # `_rmsnorm_kernel_large`'s two-pass form is a different reduction and
        # therefore a different number; it has no fused variant.
        raise ValueError(
            "fused_rmsnorm_fp8_quant only covers the single-pass norm "
            f"(hidden_size <= 8192); got hidden_size={hidden_size}"
        )
    if weight.shape != (hidden_size,):
        raise ValueError(
            f"weight shape mismatch: {tuple(weight.shape)} vs ({hidden_size},)"
        )

    if num_valid_tokens is not None:
        if num_valid_tokens.device != x.device:
            raise ValueError("num_valid_tokens must be on the same device as x")
        if num_valid_tokens.dtype != torch.int32:
            raise TypeError(
                f"num_valid_tokens must be int32, got {num_valid_tokens.dtype}"
            )
        if num_valid_tokens.numel() != 1:
            raise ValueError(
                "num_valid_tokens must contain one element, got "
                f"{tuple(num_valid_tokens.shape)}"
            )

    # Last, so every shape/dtype/layout contract above is testable without a
    # GPU; the kernel itself of course needs one.
    if not (x.is_cuda and weight.is_cuda):
        raise ValueError("fused_rmsnorm_fp8_quant inputs must be on CUDA")

    if out is None:
        out = torch.empty_like(x)
    elif out.dtype != x.dtype or out.shape != orig_shape:
        raise ValueError(
            f"out must be a {x.dtype} tensor of shape {tuple(orig_shape)}, got "
            f"{out.dtype}/{tuple(out.shape)}"
        )
    out_2d = out if out.dim() == 2 else out.reshape(-1, hidden_size)

    num_blocks = hidden_size // block_size
    num_blocks_p2 = 1
    while num_blocks_p2 < num_blocks:
        num_blocks_p2 *= 2

    x_fp8 = torch.empty(
        (num_rows, hidden_size), device=x.device, dtype=torch.float8_e4m3fn
    )
    if scale_tma_aligned:
        aligned_m = ((num_rows + 3) // 4) * 4
        x_scale = torch.empty(
            (num_blocks, aligned_m), device=x.device, dtype=torch.float32
        ).transpose(0, 1)[:num_rows, :]
    else:
        x_scale = torch.empty(
            (num_rows, num_blocks), device=x.device, dtype=torch.float32
        )

    if num_rows == 0:
        return out.reshape(orig_shape), x_fp8, x_scale

    # No num_warps / num_stages: `_rmsnorm_kernel`'s launch passes neither, and
    # `tl.sum`'s reduction tree depends on the warp count. Tuning them here
    # would change the BF16 norm output -- and the FP8 bytes with it.
    _rmsnorm_fp8_quant_kernel[(num_rows,)](
        x_2d, weight, out_2d,
        x_fp8, x_scale,
        num_valid_tokens if num_valid_tokens is not None else x_scale,
        x_2d.stride(0), out_2d.stride(0),
        x_fp8.stride(0), x_fp8.stride(1),
        x_scale.stride(0), x_scale.stride(1),
        hidden_size,
        eps=eps,
        BLOCK_SIZE=block_tile,
        dim=hidden_size,
        QUANT_BLOCK=block_size,
        NUM_BLOCKS=num_blocks,
        NUM_BLOCKS_P2=num_blocks_p2,
        HAS_VALID_TOKENS=num_valid_tokens is not None,
    )
    return out.reshape(orig_shape), x_fp8, x_scale
