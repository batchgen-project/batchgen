# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  copyright (c) EfficientMoE team 2025                                         #
#                                                                               #
#  licensed under the apache license, version 2.0 (the "license");              #
# ---------------------------------------------------------------------------- #

"""Triton Block-Attention-Residual mixer for Kimi-K3.

The two-kernel decomposition is adapted from SGLang 0.5.18's
``sglang/srt/layers/attn_residual.py`` (Apache-2.0): one CTA scans the hidden
axis for each ``(token, residual-row)`` score, then one CTA per hidden chunk
forms the softmax-weighted output.  Unlike the eager fallback, it never builds
an fp32 ``(token, residual-row, hidden)`` tensor and does not enumerate token
chunks from Python.

This module is imported lazily by :mod:`block_residual` only for supported CUDA
inputs, so the model's import-light CPU tests do not require Triton.
"""

import torch
import triton
import triton.language as tl


_BLOCK_H = 1024
_MAX_ROWS = 16


@triton.jit
def _score_kernel(
    prefix_ptr,
    bank_ptr,
    cw_ptr,
    scores_ptr,
    num_bank_rows,
    eps,
    stride_prefix_token,
    stride_bank_token,
    stride_bank_row,
    stride_score_token,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Scan H once for one ``(token, bank-or-prefix row)`` score."""
    # Form row offsets in 64-bit.  Exact-64K K3 prefill scans 65,536 local
    # tokens against a bank whose token stride is 8 x 7168 elements, so a
    # 32-bit ``token * stride`` wraps negative from token 37,450 and reads
    # rows before this rank's slice of the shared bank buffer.
    token = tl.program_id(0).to(tl.int64)
    row = tl.program_id(1)
    if row > num_bank_rows:
        return

    # Match the eager reference's overflow behavior.  Computing dot(v, w)
    # before RMS normalization can overflow to inf for a still-finite BF16
    # activation; multiplying that inf by rrms=0 then creates NaN.  The eager
    # path normalizes each value first, so scan H twice and form the dot only
    # from ``value * rrms``.
    sumsq = 0.0
    for h0 in tl.static_range(0, H, BLOCK_H):
        offsets = h0 + tl.arange(0, BLOCK_H)
        if row < num_bank_rows:
            value = tl.load(
                bank_ptr
                + token * stride_bank_token
                + row * stride_bank_row
                + offsets
            ).to(tl.float32)
        else:
            value = tl.load(
                prefix_ptr + token * stride_prefix_token + offsets
            ).to(tl.float32)
        sumsq += tl.sum(value * value)

    rrms = 1.0 / tl.sqrt(sumsq / H + eps)
    dotv = 0.0
    for h0 in tl.static_range(0, H, BLOCK_H):
        offsets = h0 + tl.arange(0, BLOCK_H)
        if row < num_bank_rows:
            value = tl.load(
                bank_ptr
                + token * stride_bank_token
                + row * stride_bank_row
                + offsets
            ).to(tl.float32)
        else:
            value = tl.load(
                prefix_ptr + token * stride_prefix_token + offsets
            ).to(tl.float32)
        coefficient = tl.load(cw_ptr + offsets)
        dotv += tl.sum((value * rrms) * coefficient)

    # ``sumsq`` may overflow to +inf even though every BF16 input is finite.
    # PyTorch's eager reference then has rrms=0 and normalizes every value to
    # zero, so the score is exactly zero.  Triton's reassociation of the dot
    # expression may still form an overflowing intermediate and produce NaN;
    # restore the eager result explicitly for this one well-defined case.
    dotv = tl.where(rrms == 0.0, 0.0, dotv)
    tl.store(
        scores_ptr + token * stride_score_token + row,
        dotv,
    )


@triton.jit
def _combine_kernel(
    prefix_ptr,
    bank_ptr,
    scores_ptr,
    output_ptr,
    num_bank_rows,
    stride_prefix_token,
    stride_bank_token,
    stride_bank_row,
    stride_score_token,
    stride_output_token,
    BLOCK_H: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    """Softmax the row scores and form one output hidden-axis chunk."""
    token = tl.program_id(0).to(tl.int64)
    hidden_block = tl.program_id(1)

    row_offsets = tl.arange(0, MAX_ROWS)
    valid_rows = row_offsets <= num_bank_rows
    raw_scores = tl.load(
        scores_ptr + token * stride_score_token + row_offsets,
        mask=valid_rows,
        other=float("-inf"),
    )
    score_max = tl.max(raw_scores, axis=0)
    score_exp = tl.where(valid_rows, tl.exp(raw_scores - score_max), 0.0)
    probabilities = score_exp / tl.sum(score_exp, axis=0)

    hidden_offsets = hidden_block * BLOCK_H + tl.arange(0, BLOCK_H)
    accumulator = tl.zeros([BLOCK_H], tl.float32)
    for row in range(0, num_bank_rows + 1):
        if row < num_bank_rows:
            value = tl.load(
                bank_ptr
                + token * stride_bank_token
                + row * stride_bank_row
                + hidden_offsets
            ).to(tl.float32)
        else:
            value = tl.load(
                prefix_ptr + token * stride_prefix_token + hidden_offsets
            ).to(tl.float32)
        probability = tl.sum(
            tl.where(row_offsets == row, probabilities, 0.0), axis=0
        )
        accumulator += probability * value

    tl.store(
        output_ptr + token * stride_output_token + hidden_offsets,
        accumulator.to(output_ptr.dtype.element_ty),
    )


def supports_attn_residual_triton(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
) -> bool:
    """Return whether the production K3 shape can use the Triton path."""
    return (
        prefix_sum.is_cuda
        and not torch.is_grad_enabled()
        and prefix_sum.ndim == 2
        and block_residual.ndim == 3
        and prefix_sum.shape[0] == block_residual.shape[0]
        and prefix_sum.shape[1] == 7168
        and block_residual.shape[2] == 7168
        and 0 < block_residual.shape[1] < _MAX_ROWS
        and prefix_sum.stride(1) == 1
        and block_residual.stride(2) == 1
        and prefix_sum.device == block_residual.device
        and prefix_sum.dtype == block_residual.dtype
    )


def score_weight(proj, norm) -> torch.Tensor:
    """Fold the current fp32 ``norm.weight * proj.weight`` vector.

    Do not retain this across forwards.  K3 installs skeleton tensors through
    the parameter server during startup, and a folded tensor computed before
    the final parameter contents are visible can remain permanently stale even
    though the source Parameters are correct by admission time.  Re-folding
    7,168 elements is negligible beside scanning every token and guarantees
    that the fused path observes the same weights as the eager oracle.
    """
    return (
        norm.weight.float() * proj.weight.squeeze(0).float()
    ).contiguous()


def mix_attn_residual_triton(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj,
    norm,
) -> torch.Tensor:
    """Return the pre-norm K3 depth mixture for a supported CUDA input."""
    if not supports_attn_residual_triton(prefix_sum, block_residual):
        raise ValueError("unsupported Kimi-K3 attention-residual Triton shape")

    num_tokens, hidden = prefix_sum.shape
    if num_tokens == 0:
        return prefix_sum
    num_bank_rows = block_residual.shape[1]
    coefficients = score_weight(proj, norm)

    scores = torch.empty(
        (num_tokens, _MAX_ROWS),
        dtype=torch.float32,
        device=prefix_sum.device,
    )
    _score_kernel[(num_tokens, num_bank_rows + 1)](
        prefix_sum,
        block_residual,
        coefficients,
        scores,
        num_bank_rows,
        norm.variance_epsilon,
        prefix_sum.stride(0),
        block_residual.stride(0),
        block_residual.stride(1),
        scores.stride(0),
        H=hidden,
        BLOCK_H=_BLOCK_H,
        num_warps=8,
    )

    output = torch.empty_like(prefix_sum)
    _combine_kernel[(num_tokens, hidden // _BLOCK_H)](
        prefix_sum,
        block_residual,
        scores,
        output,
        num_bank_rows,
        prefix_sum.stride(0),
        block_residual.stride(0),
        block_residual.stride(1),
        scores.stride(0),
        output.stride(0),
        BLOCK_H=_BLOCK_H,
        MAX_ROWS=_MAX_ROWS,
        num_warps=4,
    )

    return output


def warmup_attn_residual_triton(model) -> int:
    """Compile both kernels and enumerate every depth-mix site at startup."""
    pairs = []
    for layer in model.layers:
        pairs.extend(
            (
                (layer.self_attention_res_proj, layer.self_attention_res_norm),
                (layer.mlp_res_proj, layer.mlp_res_norm),
            )
        )
    pairs.append((model.output_attn_res_proj, model.output_attn_res_norm))

    with torch.inference_mode():
        proj, norm = pairs[0]
        hidden = proj.weight.shape[1]
        prefix = torch.zeros(
            (1, hidden),
            dtype=proj.weight.dtype,
            device=proj.weight.device,
        )
        bank = torch.zeros(
            (1, 1, hidden),
            dtype=prefix.dtype,
            device=prefix.device,
        )
        mix_attn_residual_triton(prefix, bank, proj, norm)
        torch.cuda.synchronize(prefix.device)
    return len(pairs)
