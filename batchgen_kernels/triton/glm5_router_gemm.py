"""GLM-5 BF16 tensor-core router GEMM with FP32 output."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from batchgen_kernels.triton.glm5_router_gemm_config import (
    ROUTER_BLOCK_K,
    ROUTER_BLOCK_M,
    ROUTER_BLOCK_N,
    RouterGemmShapeError,
    select_router_split_k,
    validate_router_split_k,
)


@triton.jit
def _glm5_router_gemm_kernel(
    x,
    weight,
    output,
    m,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = tl.arange(0, block_k)
    x_ptrs = x + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = weight + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_start in range(0, k, block_k):
        k_mask = offs_k < k - k_start
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < m) & k_mask[None, :],
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(offs_n[:, None] < n) & k_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x_tile, tl.trans(w_tile), out_dtype=tl.float32)
        x_ptrs += block_k * stride_xk
        w_ptrs += block_k * stride_wk

    tl.store(
        output + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


@triton.jit
def _glm5_router_gemm_split_k_kernel(
    x,
    weight,
    partials,
    m,
    n: tl.constexpr,
    k: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wn: tl.constexpr,
    stride_wk: tl.constexpr,
    stride_ps,
    stride_pm: tl.constexpr,
    stride_pn: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    k_per_split: tl.constexpr,
    blocks_per_split: tl.constexpr,
):
    """One partial ``block_m x block_n`` tile over this program's K chunk.

    The chunk is ``[pid_s * k_per_split, (pid_s + 1) * k_per_split)``, a
    function of ``k`` and the split count only. Within the chunk the block_k
    steps run in increasing K order, exactly as the serial-K kernel walks the
    same range, so this program's accumulator is the serial kernel's partial
    sum over that range. ``m`` appears only in the row mask, never in a bound,
    so no row's arithmetic can change with the bucket size.

    ``stride_ps`` is ``m * n``, so it is a runtime int like ``m``: as a
    constexpr it would compile one binary per row count in the split-K range.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    offs_k = pid_s * k_per_split + tl.arange(0, block_k)
    x_ptrs = x + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = weight + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((block_m, block_n), dtype=tl.float32)

    # blocks_per_split * block_k == k_per_split, and k_per_split times the
    # split count == k, both enforced on the host, so the K mask below never
    # trims a step and the trip count is the same for every split.
    for kb in range(blocks_per_split):
        k_mask = (offs_k + kb * block_k) < k
        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < m) & k_mask[None, :],
            other=0.0,
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(offs_n[:, None] < n) & k_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(x_tile, tl.trans(w_tile), out_dtype=tl.float32)
        x_ptrs += block_k * stride_xk
        w_ptrs += block_k * stride_wk

    tl.store(
        partials
        + pid_s * stride_ps
        + offs_m[:, None] * stride_pm
        + offs_n[None, :] * stride_pn,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
    )


@triton.jit
def _glm5_router_split_k_reduce_kernel(
    partials,
    output,
    m,
    n: tl.constexpr,
    stride_ps,
    stride_pm: tl.constexpr,
    stride_pn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    split_k: tl.constexpr,
):
    """Sum the K-splits in a fixed order, split 0 upwards.

    No atomics and no cross-program accumulation: each program owns its output
    tile and reads every partial for it, so the result is reproducible and the
    operation order is the same for every launch.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    mask = (offs_m[:, None] < m) & (offs_n[None, :] < n)
    p_ptrs = partials + offs_m[:, None] * stride_pm + offs_n[None, :] * stride_pn

    acc = tl.load(p_ptrs, mask=mask, other=0.0)
    for s in range(1, split_k):
        acc += tl.load(p_ptrs + s * stride_ps, mask=mask, other=0.0)

    tl.store(
        output + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc,
        mask=mask,
    )


def glm5_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output: torch.Tensor,
    *,
    split_k: Optional[int] = None,
    partials: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute ``hidden_states @ router_weight.T`` into preallocated FP32 output.

    Shapes in the row range the split-K path targets are computed as a fixed
    partition of the K axis followed by a fixed-order reduction; every other
    shape keeps walking K serially in one program. Both paths produce, for a
    given row, a result that depends only on that row's activations, on K, and
    on the split count the policy picked for the launch -- never on how many
    padded rows shared the launch, and never on another row's values.

    The policy picks that count from the row count alone, which the graphed
    router path holds static per captured bucket, so one count and one
    reduction order are baked into each bucket's graph at capture time -- and
    it is the SAME count for every bucket in the range, so a row's logits are
    bitwise identical whether its launch carried the padded bucket or only the
    valid rows. The last float32 bits can still differ across the
    serial-K/split-K boundary, i.e. between a row in that range and the same
    row outside it; the exactness bound each count has to meet is asserted per
    count in tests/test_glm5_router_gemm_cuda.py.

    Args:
        split_k: force a split count. ``None`` applies the shape policy, ``1``
            forces the serial-K kernel. A count that cannot partition K into
            equal whole-block_k chunks raises
            :class:`~batchgen_kernels.triton.glm5_router_gemm_config.RouterGemmShapeError`.
        partials: optional preallocated ``(split_k, m, n)`` float32 scratch for
            the partial tiles. Allocated per call when omitted; inside a CUDA
            graph capture that allocation is made once and baked into the
            graph, like any other intermediate in the captured region.
    """
    m, k = hidden_states.shape
    n = router_weight.shape[0]

    if split_k is None:
        split_k = select_router_split_k(m, k)
    else:
        validate_router_split_k(k, split_k)

    if split_k == 1:
        _glm5_router_gemm_kernel[
            (triton.cdiv(m, ROUTER_BLOCK_M), triton.cdiv(n, ROUTER_BLOCK_N))
        ](
            hidden_states,
            router_weight,
            output,
            m,
            n,
            k,
            hidden_states.stride(0),
            hidden_states.stride(1),
            router_weight.stride(0),
            router_weight.stride(1),
            output.stride(0),
            output.stride(1),
            block_m=ROUTER_BLOCK_M,
            block_n=ROUTER_BLOCK_N,
            block_k=ROUTER_BLOCK_K,
            num_warps=4,
            num_stages=3,
        )
        return output

    k_per_split = k // split_k
    blocks_per_split = k_per_split // ROUTER_BLOCK_K
    if partials is None:
        partials = torch.empty(
            (split_k, m, n),
            dtype=torch.float32,
            device=hidden_states.device,
        )
    elif tuple(partials.shape) != (split_k, m, n) or partials.dtype != torch.float32:
        raise RouterGemmShapeError(
            f"router GEMM partials buffer must be float32 with shape "
            f"{(split_k, m, n)}, got {partials.dtype} {tuple(partials.shape)}"
        )

    _glm5_router_gemm_split_k_kernel[
        (triton.cdiv(m, ROUTER_BLOCK_M), triton.cdiv(n, ROUTER_BLOCK_N), split_k)
    ](
        hidden_states,
        router_weight,
        partials,
        m,
        n,
        k,
        hidden_states.stride(0),
        hidden_states.stride(1),
        router_weight.stride(0),
        router_weight.stride(1),
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        block_m=ROUTER_BLOCK_M,
        block_n=ROUTER_BLOCK_N,
        block_k=ROUTER_BLOCK_K,
        k_per_split=k_per_split,
        blocks_per_split=blocks_per_split,
        num_warps=4,
        num_stages=3,
    )

    _glm5_router_split_k_reduce_kernel[
        (triton.cdiv(m, ROUTER_BLOCK_M), triton.cdiv(n, ROUTER_BLOCK_N))
    ](
        partials,
        output,
        m,
        n,
        partials.stride(0),
        partials.stride(1),
        partials.stride(2),
        output.stride(0),
        output.stride(1),
        block_m=ROUTER_BLOCK_M,
        block_n=ROUTER_BLOCK_N,
        split_k=split_k,
        num_warps=4,
    )
    return output
