"""GLM-5 BF16 tensor-core router GEMM with FP32 output."""

import torch
import triton
import triton.language as tl


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


def glm5_router_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Compute ``hidden_states @ router_weight.T`` into preallocated FP32 output."""
    m, k = hidden_states.shape
    n = router_weight.shape[0]
    _glm5_router_gemm_kernel[
        (triton.cdiv(m, 16), triton.cdiv(n, 64))
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
        block_m=16,
        block_n=64,
        block_k=64,
        num_warps=4,
        num_stages=3,
    )
    return output
