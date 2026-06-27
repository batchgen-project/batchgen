"""BatchGen drop-in replacement for SGLang's fused_marlin_moe.

Same signature/semantics as sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe:
takes hidden [M, H] + topk_ids/topk_weights, runs the routed INT4-marlin MoE, returns
[M, H]. The ONLY difference vs SGLang is the GEMM kernel — everything else (the
moe_align compact dispatch, the per-expert-contiguous block layout, the weighted
combine) mirrors SGLang so a benchmark isolates the GEMM, not the dispatch.

Flow:
  moe_align_block_size(topk_ids, block=16, E) -> sorted_token_ids/expert_ids/num_post
  gather hidden by (sorted_token_ids // topk) -> compact A [num_post, H]
  BatchGen GEMM S1 (w13 concat, N=2*inter_pr) -> silu_mul_split -> S2 -> BG GEMM S3 (down)
  scatter-combine compact out back to [M, H] with topk_weights (sum over topk)

expert_starts (per-expert first row) = exclusive cumsum of ceil(count/16)*16, which
matches moe_align's block-padded layout; BatchGen's grouped GEMM reads count rows from
expert_starts[e], and moe_align puts each expert's valid tokens first in its range.

Kernel variant via env BATCHGEN_KIMI_TP_MARLIN_V3/_V2 (default v1).
"""
import os

import torch

from batchgen.moe.marlin_grouped_moe import _load_module

_BLOCK = 16  # MBLOCK of the BatchGen grouped marlin kernel


def _moe_align():
    from sglang.srt.layers.moe.fused_moe_triton.moe_align_block_size import (
        moe_align_block_size,
    )
    return moe_align_block_size


def batchgen_fused_marlin_moe(
    hidden_states: torch.Tensor,        # [M, H] bf16
    w13: torch.Tensor,                  # [E, H//16, 4*inter_pr] int32 (concat gate|up marlin)
    w2: torch.Tensor,                   # [E, inter_pr//16, 2*H] int32 (down marlin)
    w13_scale: torch.Tensor,            # [E, H//gs, 2*inter_pr] bf16
    w2_scale: torch.Tensor,             # [E, inter_pr//gs, H] bf16
    topk_ids: torch.Tensor,             # [M, topk] int
    topk_weights: torch.Tensor,         # [M, topk] float
    inter_pr: int,
    routed_scaling_factor=None,
) -> torch.Tensor:
    mod = _load_module()
    use_v3 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V3", "0") == "1"
    use_v2 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V2", "0") == "1"

    M, H = hidden_states.shape
    E = w13.shape[0]
    topk = topk_ids.shape[1]
    dev = hidden_states.device

    moe_align_block_size = _moe_align()
    sorted_ids, expert_ids, num_post_t = moe_align_block_size(
        topk_ids.to(torch.int32), _BLOCK, E)
    num_post = sorted_ids.shape[0]               # block-padded capacity (>= num_tokens_post_padded)

    # Per-expert valid counts + block-padded starts (matches moe_align's layout).
    counts = torch.bincount(topk_ids.reshape(-1).to(torch.int64), minlength=E).to(torch.int32)
    padded = ((counts + _BLOCK - 1) // _BLOCK) * _BLOCK
    starts = torch.zeros(E, dtype=torch.int32, device=dev)
    starts[1:] = torch.cumsum(padded, 0)[:-1].to(torch.int32)

    # ---- gather: compact A[row] = hidden[token], token = sorted_ids[row] // topk ----
    Mtopk = M * topk
    valid = sorted_ids < Mtopk                   # padding rows have sentinel >= M*topk
    flat = sorted_ids.clamp(0, Mtopk - 1)
    token_idx = (flat // topk)                   # [num_post] token per compact row
    A = hidden_states.index_select(0, token_idx.to(torch.int64))   # [num_post, H]
    A = A * valid.unsqueeze(1)                    # zero padding rows (bias-free -> harmless)

    # ---- BatchGen GEMM: S1 (gate|up) -> silu -> S3 (down) over the compact buffer ----
    idx = torch.arange(E, dtype=torch.int64, device=dev)
    n_tiles_s1 = (2 * inter_pr) // 256 if (2 * inter_pr) >= 256 else 1
    n_tiles_s3 = H // 256
    s1_B = w13.data_ptr() + idx * (H // 16) * (4 * inter_pr) * 4
    s1_S = w13_scale.data_ptr() + idx * (H // 32) * (2 * inter_pr) * 2
    s3_B = w2.data_ptr() + idx * (inter_pr // 16) * (2 * H) * 4
    s3_S = w2_scale.data_ptr() + idx * (inter_pr // 32) * H * 2

    gateup = torch.empty(num_post, 2 * inter_pr, dtype=torch.bfloat16, device=dev)
    inter = torch.empty(num_post, inter_pr, dtype=torch.bfloat16, device=dev)
    out_c = torch.empty(num_post, H, dtype=torch.bfloat16, device=dev)
    s1_C = gateup.data_ptr() + idx * 0           # placeholder; per-expert C ptrs below
    # C pointers are per-expert into the compact buffers at `starts` (row offset).
    s1_C = gateup.data_ptr() + starts.to(torch.int64) * (2 * inter_pr * 2)
    inter_C = inter.data_ptr() + starts.to(torch.int64) * (inter_pr * 2)
    s3_C = out_c.data_ptr() + starts.to(torch.int64) * (H * 2)

    max_count = int(counts.max().item())
    max_m_tiles = max(1, (max_count + 15) // 16)
    s1_ws = torch.zeros(E * (n_tiles_s1 + 17), dtype=torch.int32, device=dev)
    s3_ws = torch.zeros(E * (n_tiles_s3 + 17), dtype=torch.int32, device=dev)

    if use_v3:
        mod.grouped_marlin_tp_s1_v3(A, s1_B, inter_C, s1_S, starts, counts,
                                    E, 2 * inter_pr, H, s1_ws, E, n_tiles_s1, max_m_tiles, inter_pr)
        if max_count <= 8:
            mod.grouped_marlin_tp_s3_v3(inter, s3_B, s3_C, s3_S, starts, counts,
                                        E, H, inter_pr, s3_ws, E, n_tiles_s3, max_m_tiles, H)
        else:
            mod.grouped_marlin_tp_s3(inter, s3_B, s3_C, s3_S, starts, counts,
                                     E, H, inter_pr, s3_ws, E, n_tiles_s3, max_m_tiles)
    elif use_v2:
        mod.grouped_marlin_tp_s1(A, s1_B, inter_C, s1_S, starts, counts,
                                 E, 2 * inter_pr, H, s1_ws, E, n_tiles_s1, max_m_tiles, inter_pr)
        mod.grouped_marlin_tp_s3(inter, s3_B, s3_C, s3_S, starts, counts,
                                 E, H, inter_pr, s3_ws, E, n_tiles_s3, max_m_tiles)
    else:
        mod.grouped_marlin_gemm_m16(A, s1_B, s1_C, s1_S, starts, counts,
                                    E, 2 * inter_pr, H, s1_ws, E, n_tiles_s1, max_m_tiles)
        mod.silu_mul_split(gateup, inter, counts, E, max_m_tiles * 16, num_post // E if E else 0, inter_pr)
        mod.grouped_marlin_gemm_m16(inter, s3_B, s3_C, s3_S, starts, counts,
                                    E, H, inter_pr, s3_ws, E, n_tiles_s3, max_m_tiles)

    # ---- scatter-combine: out[token] += out_c[row] * topk_weight[row] (sum over topk) ----
    w_flat = topk_weights.reshape(-1).to(torch.float32)
    w_row = w_flat.index_select(0, flat.to(torch.int64)) * valid.to(torch.float32)  # [num_post]
    if routed_scaling_factor is not None:
        w_row = w_row * routed_scaling_factor
    contrib = out_c.to(torch.float32) * w_row.unsqueeze(1)
    output = torch.zeros(M, H, dtype=torch.float32, device=dev)
    output.index_add_(0, token_idx.to(torch.int64), contrib)
    return output.to(torch.bfloat16)
