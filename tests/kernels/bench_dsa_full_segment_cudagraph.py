"""Benchmark synthetic full GLM-5 DSA attention CUDA graph segment."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from batchgen.attention.dsa.sparse_decode_mla import (
    prepare_sparse_flash_mla_decode_inputs,
    run_prepared_sparse_flash_mla_decode,
)
from batchgen.attention.dsa.unified_selector import (
    select_mla_kv_for_flashmla_bf16,
    select_mla_kv_for_flashmla_bf16_out,
)
from batchgen_kernels.attention.dsa.fp8_absorb import (
    FP8AbsorbWeights,
    fp8_out_absorb,
    fp8_out_absorb_out,
    fp8_q_absorb,
    fp8_q_absorb_out,
)
from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import build_module, make_fp8_activation_scratch
from batchgen_kernels.attention.dsa.fused_indexer_score import (
    FP8WqbWeightsCUDA,
    cuda_wq_b_proj,
    cuda_wq_b_proj_out,
    fused_score_and_topk,
    fused_score_and_topk_out,
    rope_hadamard_q,
    rope_hadamard_q_out,
)
from batchgen_kernels.attention.dsa.query_pack import pack_flashmla_query_out


PAGE_SIZE = 64
INDEX_H = 32
ATTN_H = 64
INDEX_D = 128
D_QK = 576
D_V = 512
Q_NOPE = 192
OUT_D = 256
H_KV = 1


def _rope_tables(max_pos: int, rope_dim: int = 64):
    theta = 1000000.0
    freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device="cuda").float() / rope_dim))
    t = torch.arange(max_pos, device="cuda").float()
    angles = t[:, None] * freqs[None, :]
    return torch.cos(angles).repeat(1, 2).contiguous(), torch.sin(angles).repeat(1, 2).contiguous()


def _make_primary_cache(batch_size: int, max_seqlen: int):
    pages_per_seq = (max_seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch_size * pages_per_seq
    blocked_k = torch.randn(total_pages, PAGE_SIZE, H_KV, D_QK, device="cuda", dtype=torch.bfloat16)
    page_table = torch.arange(total_pages, device="cuda", dtype=torch.int32).view(batch_size, pages_per_seq)
    return blocked_k, page_table


def _make_paths(batch_size: int, max_seqlen: int, topk: int, module):
    q_rank = 2048
    softmax_scale = D_QK**-0.5
    q_a = (torch.randn(batch_size, q_rank, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    wq_b = (torch.randn(INDEX_H * INDEX_D, q_rank, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    aux_cached_k = (torch.randn(batch_size, max_seqlen, INDEX_D, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    head_gates = torch.randn(batch_size, INDEX_H, device="cuda", dtype=torch.float32)
    cache_seqlens = torch.full((batch_size,), max_seqlen, device="cuda", dtype=torch.int32)
    positions = torch.randint(0, max_seqlen, (batch_size,), device="cuda", dtype=torch.int64)
    positions_expanded = positions.repeat_interleave(INDEX_H).contiguous()
    cos, sin = _rope_tables(max_seqlen)
    primary_blocked_k, primary_page_table = _make_primary_cache(batch_size, max_seqlen)
    q_weights = FP8WqbWeightsCUDA(wq_b, module)

    q_nope = (torch.randn(batch_size, ATTN_H, Q_NOPE, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    q_rope = (torch.randn(batch_size, ATTN_H, 64, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    q_absorb_w = (torch.randn(ATTN_H, Q_NOPE, D_V, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    out_absorb_w = (torch.randn(ATTN_H, OUT_D, D_V, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    absorb_weights = FP8AbsorbWeights(q_absorb_w, out_absorb_w)

    q_x_fp8, q_x_scale, q_tma = make_fp8_activation_scratch(batch_size, q_rank, module, device="cuda")
    q_flat = torch.empty(batch_size, INDEX_H * INDEX_D, device="cuda", dtype=torch.bfloat16)
    q_index = torch.empty(batch_size, INDEX_H, INDEX_D, device="cuda", dtype=torch.bfloat16)
    agg = torch.empty(batch_size, max_seqlen, device="cuda", dtype=torch.float32)
    topk_indices = torch.empty(batch_size, topk, device="cuda", dtype=torch.long)
    selected = torch.empty(batch_size, topk, H_KV, D_QK, device="cuda", dtype=torch.bfloat16)
    selected_lengths = torch.empty(batch_size, device="cuda", dtype=torch.int32)
    selected_lengths.fill_(topk)
    row_modes = torch.empty(batch_size, device="cuda", dtype=torch.int32)
    absorbed_q = torch.empty(batch_size, ATTN_H, D_V, device="cuda", dtype=torch.bfloat16)
    query_states = torch.empty(batch_size, 1, ATTN_H, D_QK, device="cuda", dtype=torch.bfloat16)
    final_out = torch.empty(batch_size, 1, ATTN_H, OUT_D, device="cuda", dtype=torch.bfloat16)
    prepared = prepare_sparse_flash_mla_decode_inputs(
        query_states,
        selected,
        selected_lengths,
        ATTN_H,
        softmax_scale,
        head_dim_v=D_V,
        page_size=PAGE_SIZE,
    )

    def allocating_path():
        q = cuda_wq_b_proj(q_a, q_weights, module).view(batch_size, INDEX_H, INDEX_D)
        q = rope_hadamard_q(q, cos, sin, positions)
        indices = fused_score_and_topk(q, aux_cached_k, head_gates, cache_seqlens, topk=topk)
        selected_alloc, selected_lengths_alloc, _, _ = select_mla_kv_for_flashmla_bf16(
            primary_blocked_k,
            primary_page_table,
            cache_seqlens,
            indices,
            index_topk=topk,
            page_size=PAGE_SIZE,
            return_indices=False,
        )
        absorbed = fp8_q_absorb(q_nope, absorb_weights)
        pack_flashmla_query_out(absorbed, q_rope, query_states)
        prepared_alloc = prepare_sparse_flash_mla_decode_inputs(
            query_states,
            selected_alloc,
            selected_lengths_alloc,
            ATTN_H,
            softmax_scale,
            head_dim_v=D_V,
            page_size=PAGE_SIZE,
        )
        attn = run_prepared_sparse_flash_mla_decode(prepared_alloc)
        return fp8_out_absorb(attn, absorb_weights)

    def out_path():
        cuda_wq_b_proj_out(q_a, q_weights, module, q_x_fp8, q_x_scale, q_tma, q_flat)
        rope_hadamard_q_out(q_flat.view(batch_size, INDEX_H, INDEX_D), cos, sin, positions_expanded, q_index)
        fused_score_and_topk_out(q_index, aux_cached_k, head_gates, cache_seqlens, agg, topk_indices, topk=topk)
        select_mla_kv_for_flashmla_bf16_out(
            primary_blocked_k,
            primary_page_table,
            cache_seqlens,
            topk_indices,
            PAGE_SIZE,
            selected,
            selected_lengths,
            None,
            row_modes,
            index_topk=topk,
            return_indices=False,
        )
        fp8_q_absorb_out(q_nope, absorb_weights, absorbed_q)
        pack_flashmla_query_out(absorbed_q, q_rope, query_states)
        attn = run_prepared_sparse_flash_mla_decode(prepared)
        fp8_out_absorb_out(attn, absorb_weights, final_out)
        return final_out

    return allocating_path, out_path


def _make_graph_path(out_path):
    for _ in range(3):
        out_path()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_path()

    def replay():
        graph.replay()

    return replay


def _time_cuda(fn, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def _summary(times: list[float]) -> tuple[float, float, float]:
    ordered = sorted(times)
    return statistics.median(times), ordered[int(0.9 * (len(ordered) - 1))], min(times)


def _comparison(base: float, candidate: float, name: str) -> str:
    if candidate <= base:
        return f"{name} {base / candidate:.2f}x faster"
    return f"allocating_eager {candidate / base:.2f}x faster"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[2048])
    parser.add_argument("--topk", type=int, default=2048)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(20260429)
    module = build_module()
    print("b,seqlen,path,median_ms,p90_ms,min_ms,comparison")
    for batch_size in args.batch_sizes:
        for max_seqlen in args.seq_lens:
            topk = min(args.topk, max_seqlen)
            allocating_path, out_path = _make_paths(batch_size, max_seqlen, topk, module)
            graph_path = _make_graph_path(out_path)
            allocating = _summary(_time_cuda(allocating_path, warmup=args.warmup, iters=args.iters))
            out_eager = _summary(_time_cuda(out_path, warmup=args.warmup, iters=args.iters))
            graph = _summary(_time_cuda(graph_path, warmup=args.warmup, iters=args.iters))
            out_cmp = _comparison(allocating[0], out_eager[0], "out_eager")
            graph_cmp = _comparison(allocating[0], graph[0], "cudagraph")
            for path, summary, cmp_label in (
                ("allocating_eager", allocating, out_cmp),
                ("out_eager", out_eager, out_cmp),
                ("cudagraph", graph, graph_cmp),
            ):
                print(
                    f"b={batch_size},seqlen={max_seqlen},{path},"
                    f"median_ms={summary[0]:.4f},p90_ms={summary[1]:.4f},"
                    f"min_ms={summary[2]:.4f},{cmp_label}"
                )


if __name__ == "__main__":
    main()
