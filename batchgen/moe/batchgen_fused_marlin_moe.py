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
from sgl_kernel import shuffle_rows, moe_sum_reduce  # fast CUDA gather + fp32-accum weighted reduce

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
    hybrid_bsz: int = 1,                # M < hybrid_bsz -> delegate the GEMM to SGLang's
                                        # moe_wna16_marlin (faster at tiny M). Default 1 =>
                                        # always BatchGen, so the harness isolates the BG GEMM.
) -> torch.Tensor:
    mod = _load_module()
    use_v3 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V3", "0") == "1"
    use_v2 = os.environ.get("BATCHGEN_KIMI_TP_MARLIN_V2", "0") == "1"

    M, H = hidden_states.shape
    E = w13.shape[0]
    topk = topk_ids.shape[1]
    dev = hidden_states.device

    # M-dispatch: tiny M -> SGLang's own marlin GEMM (its in-kernel gather/scatter beats
    # ours there). Same SGLang TP path, only the S1/S3 GEMM differs by M.
    if M < hybrid_bsz:
        from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
            fused_marlin_moe as _sgl_fmm,
        )
        return _sgl_fmm(
            hidden_states=hidden_states, w1=w13, w2=w2,
            w1_scale=w13_scale, w2_scale=w2_scale,
            gating_output=topk_weights, topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32), global_num_experts=E,
            num_bits=4, is_k_full=True, inplace=False,
            routed_scaling_factor=routed_scaling_factor,
        )

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
    # No padding mask: the BatchGen GEMM reads only expert_counts[e] valid rows per expert
    # (m_start>=count early-exit in marlin_grouped_gemm.cu); padding rows are never read.
    Mtopk = M * topk
    is_valid = sorted_ids < Mtopk                # padding rows carry sentinel >= M*topk
    token_idx = torch.where(is_valid, sorted_ids, torch.zeros_like(sorted_ids)) // topk
    A = shuffle_rows(hidden_states, token_idx.to(torch.int32), (num_post, H))   # [num_post, H]

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

    # ---- weighted combine: un-permute out_c -> token-major, weight, fp32-accum sum ----
    # inv_map[g] = compact row holding flattened assignment g (g = token*topk + slot).
    # K2.5 is no-drop routing, so moe_align emits each of the Mtopk assignments EXACTLY
    # ONCE -> a bijection of valid rows onto [0, Mtopk); padding rows route to dummy slot
    # Mtopk and are dropped. (A capacity/drop path would leave a slot unwritten -> garbage.)
    g_idx = torch.where(is_valid, sorted_ids.to(torch.int64),
                        torch.full_like(sorted_ids, Mtopk, dtype=torch.int64))
    inv = torch.empty(Mtopk + 1, dtype=torch.int32, device=dev)
    inv.scatter_(0, g_idx, torch.arange(num_post, dtype=torch.int32, device=dev))
    inv_map = inv[:Mtopk]

    out_tok = shuffle_rows(out_c, inv_map, (Mtopk, H)).view(M, topk, H)
    out_tok = out_tok * topk_weights.view(M, topk, 1).to(torch.bfloat16)   # topk_weight (×2.5) here
    output = torch.empty(M, H, dtype=torch.bfloat16, device=dev)
    moe_sum_reduce(out_tok, output,
                   routed_scaling_factor if routed_scaling_factor is not None else 1.0)
    return output
