"""BF16 fused MoE (grouped GEMM) — Triton port of the classic vLLM fused_moe.

Structure (per MoE block, all on GPU, CUDA-graph safe — no host syncs):
  1. moe_align_block_size: sort (token, expert-slot) pairs by expert, pad each
     expert's segment to BLOCK_M (torch impl: bincount/cumsum/stable-argsort —
     graph-capturable; can be upgraded to the CUDA align kernel later).
  2. stage1 grouped GEMM: C1 = x @ W13[e]^T for each token's expert e
     (W13 = [gate; up] packed (E, 2N, K)).
  3. silu_and_mul: y = silu(C1[:, :N]) * C1[:, N:] (fp32 internally).
  4. stage2 grouped GEMM: C3 = y @ W2[e]^T (W2 = down (E, H, N)).
  5. weighted reduce over top-k via the production
     batchgen_kernels.triton.moe_weighted_sum kernel.

vs vLLM's kernel: BF16-only (no quant/bias/LoRA/TMA paths), top_k is a runtime
arg, and routed-weight multiply is deferred to moe_weighted_sum.

Usage:
    from batchgen_kernels.triton.fused_moe_bf16 import fused_moe_bf16
    out = fused_moe_bf16(x, w13, w2, topk_weights, topk_ids)
"""

import torch
import triton
import triton.language as tl

from batchgen_kernels.triton.moe_weighted_sum import moe_weighted_sum_triton

# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)
_CONFIG_LARGE = (64, 64, 32, 8, 4, 3)
_CONFIG_SMALL = (16, 64, 32, 8, 4, 3)


@triton.jit
def _fused_moe_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    top_k,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """Grouped GEMM: C[token_slot] = A[token_slot // top_k] @ B[expert]^T.

    A: (num_tokens_a, K); B: (E, N, K); C: (EM_padded_rows, N) addressed by
    sorted token-slot ids. Token slots >= num_valid_tokens are padding.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            k_rem = K - k * BLOCK_SIZE_K
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < k_rem),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    acc = accumulator.to(c_ptr.dtype.element_ty)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _silu_and_mul_kernel(
    x_ptr,
    y_ptr,
    N,
    stride_xm,
    stride_ym,
    BLOCK_N: tl.constexpr,
):
    """y = silu(x[:, :N]) * x[:, N:] (fp32 internally), row-wise over (M, 2N)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    g = tl.load(x_ptr + pid_m * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + pid_m * stride_xm + N + offs, mask=mask, other=0.0).to(tl.float32)
    y = g / (1.0 + tl.exp(-g)) * u
    tl.store(y_ptr + pid_m * stride_ym + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort token-slots by expert and pad segments to block_size (GPU-only).

    Args:
        topk_ids: (M, top_k) expert indices per token.
        block_size: GEMM M-tile the segments are padded to.
        num_experts: total routed experts E.

    Returns:
        sorted_token_ids: (max_padded,) int32 token-slot ids (slot =
            token_row * top_k + k), padded slots filled with numel.
        expert_ids: (max_blocks,) int32 expert per BLOCK_M segment.
        num_tokens_post_padded: (1,) int32 total padded slots.
    """
    device = topk_ids.device
    top_k = topk_ids.shape[1]
    flat = topk_ids.flatten().to(torch.int64)
    numel = flat.numel()

    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + (block_size - 1)).div(block_size, rounding_mode="floor") * block_size
    padded_ends = torch.cumsum(padded, 0)
    padded_offsets = padded_ends - padded
    counts_ends = torch.cumsum(counts, 0)
    counts_offsets = counts_ends - counts

    order = torch.argsort(flat, stable=True)
    sorted_experts = flat[order]
    rank = torch.arange(numel, device=device) - counts_offsets[sorted_experts]

    max_padded = numel + (num_experts + 1) * (block_size - 1)
    sorted_ids = torch.full((max_padded,), numel, dtype=torch.int32, device=device)
    sorted_ids[padded_offsets[sorted_experts] + rank] = order.to(torch.int32)

    max_blocks = triton.cdiv(max_padded, block_size)
    block_starts = torch.arange(max_blocks, device=device) * block_size
    expert_ids = torch.searchsorted(padded_ends, block_starts, right=True).to(torch.int32)
    num_tokens_post_padded = padded_ends[-1].to(torch.int32).reshape(1)
    return sorted_ids, expert_ids, num_tokens_post_padded


def _pick_config(m_slots: int):
    return _CONFIG_SMALL if m_slots <= 1024 else _CONFIG_LARGE


def _stage_gemm(a, b, c, sorted_ids, expert_ids, num_post, top_k, num_valid, config):
    E, N, K = b.shape
    BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, warps, stages = config
    EM = sorted_ids.numel()
    grid = (triton.cdiv(EM, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _fused_moe_gemm[grid](
        a,
        b,
        c,
        sorted_ids,
        expert_ids,
        num_post,
        N,
        K,
        EM,
        num_valid,
        top_k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_M,
        EVEN_K=(K % BLOCK_K == 0),
        num_warps=warps,
        num_stages=stages,
    )


def fused_moe_bf16(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Fused BF16 MoE forward.

    Args:
        x: (M, K) bf16/fp16/fp32 hidden states.
        w13: (E, 2N, K) packed [gate; up] expert weights (gate first).
        w2: (E, H, N) down-projection expert weights.
        topk_weights: (M, top_k) fp32 routing weights (already renormalized
            and scaled, as produced by KimiMoEGate).
        topk_ids: (M, top_k) expert indices.

    Returns:
        (M, H) tensor in x.dtype: sum_k topk_weights[m, k] * expert(x[m]).
    """
    assert x.dim() == 2 and w13.dim() == 3 and w2.dim() == 3
    M, K = x.shape
    E, N2, _ = w13.shape
    N = N2 // 2
    H = w2.shape[1]
    top_k = topk_ids.shape[1]
    assert w13.shape[2] == K and w2.shape[2] == N and w2.shape[0] == E

    config = _pick_config(M * top_k)
    block_m = config[0]
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, block_m, E)

    c1 = torch.empty((M * top_k, 2 * N), device=x.device, dtype=x.dtype)
    _stage_gemm(x, w13, c1, sorted_ids, expert_ids, num_post, top_k, M * top_k, config)

    c2 = torch.empty((M * top_k, N), device=x.device, dtype=x.dtype)
    grid_act = (M * top_k, triton.cdiv(N, 1024))
    _silu_and_mul_kernel[grid_act](
        c1, c2, N, c1.stride(0), c2.stride(0), BLOCK_N=1024, num_warps=4
    )

    c3 = torch.empty((M * top_k, H), device=x.device, dtype=x.dtype)
    _stage_gemm(c2, w2, c3, sorted_ids, expert_ids, num_post, 1, M * top_k, config)

    out = moe_weighted_sum_triton(
        c3.view(M, top_k, H), topk_weights.to(torch.float32)
    )
    return out.to(x.dtype)
