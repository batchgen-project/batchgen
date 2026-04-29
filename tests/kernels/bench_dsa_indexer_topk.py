"""Benchmark GLM-5 DSA custom score+topk against torch.topk reference."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from batchgen_kernels.attention.dsa.fused_indexer_score import (
    _fused_score_kernel,
    fused_score_and_topk,
    fused_score_and_topk_out,
)


def _make_inputs(batch_size: int, max_seqlen: int, n_heads: int, head_dim: int):
    q = torch.randn(batch_size, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    cached_k = torch.randn(batch_size, max_seqlen, head_dim, device="cuda", dtype=torch.bfloat16)
    head_gates = torch.randn(batch_size, n_heads, device="cuda", dtype=torch.float32)
    cache_seqlens = torch.full((batch_size,), max_seqlen, device="cuda", dtype=torch.int32)
    return q, cached_k, head_gates, cache_seqlens


def _score_only(q, cached_k, head_gates, cache_seqlens):
    batch_size, n_heads, head_dim = q.shape
    max_seqlen = cached_k.shape[1]
    agg = torch.empty(batch_size, max_seqlen, dtype=torch.float32, device=q.device)
    block_s = min(128, triton.next_power_of_2(max_seqlen))
    _fused_score_kernel[(triton.cdiv(max_seqlen, block_s), batch_size)](
        q,
        cached_k,
        head_gates,
        cache_seqlens,
        agg,
        max_seqlen,
        B=batch_size,
        n_heads=n_heads,
        head_dim=head_dim,
        BLOCK_S=block_s,
        BLOCK_D=head_dim,
    )
    return agg


def _torch_topk_path(q, cached_k, head_gates, cache_seqlens, topk: int):
    agg = _score_only(q, cached_k, head_gates, cache_seqlens)
    _, indices = torch.topk(agg, topk, dim=-1)
    return indices


def _make_graph_path(q, cached_k, head_gates, cache_seqlens, topk: int):
    batch_size = q.shape[0]
    max_seqlen = cached_k.shape[1]
    agg = torch.empty(batch_size, max_seqlen, dtype=torch.float32, device=q.device)
    out = torch.empty(batch_size, topk, dtype=torch.long, device=q.device)

    def run():
        return fused_score_and_topk_out(
            q,
            cached_k,
            head_gates,
            cache_seqlens,
            agg,
            out,
            topk=topk,
        )

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    def replay():
        graph.replay()
        return out

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


def _label(base: float, candidate: float, name: str) -> str:
    if candidate <= base:
        return f"{name} {base / candidate:.2f}x faster"
    return f"torch_topk {candidate / base:.2f}x faster"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--include-cudagraph", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    print("b,seqlen,path,median_ms,p90_ms,min_ms,comparison")
    for batch_size in args.batch_sizes:
        for max_seqlen in args.seq_lens:
            topk = min(args.topk, max_seqlen)
            q, cached_k, head_gates, cache_seqlens = _make_inputs(
                batch_size,
                max_seqlen,
                args.n_heads,
                args.head_dim,
            )
            torch_times = _time_cuda(
                lambda: _torch_topk_path(q, cached_k, head_gates, cache_seqlens, topk),
                warmup=args.warmup,
                iters=args.iters,
            )
            custom_times = _time_cuda(
                lambda: fused_score_and_topk(q, cached_k, head_gates, cache_seqlens, topk=topk),
                warmup=args.warmup,
                iters=args.iters,
            )
            torch_summary = _summary(torch_times)
            custom_summary = _summary(custom_times)
            comparison = _label(torch_summary[0], custom_summary[0], "custom_topk")
            for path, summary in (("torch_topk", torch_summary), ("custom_topk", custom_summary)):
                print(
                    f"b={batch_size},seqlen={max_seqlen},{path},"
                    f"median_ms={summary[0]:.4f},p90_ms={summary[1]:.4f},"
                    f"min_ms={summary[2]:.4f},{comparison}"
                )

            if args.include_cudagraph:
                graph_fn = _make_graph_path(q, cached_k, head_gates, cache_seqlens, topk)
                graph_times = _time_cuda(graph_fn, warmup=args.warmup, iters=args.iters)
                graph_summary = _summary(graph_times)
                graph_comparison = _label(custom_summary[0], graph_summary[0], "custom_topk_cudagraph")
                print(
                    f"b={batch_size},seqlen={max_seqlen},custom_topk_cudagraph,"
                    f"median_ms={graph_summary[0]:.4f},p90_ms={graph_summary[1]:.4f},"
                    f"min_ms={graph_summary[2]:.4f},{graph_comparison}"
                )


if __name__ == "__main__":
    main()
