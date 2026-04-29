"""Benchmark GLM-5 DSA FP8 q_absorb/out_absorb CUDA graph replay."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from batchgen_kernels.attention.dsa.fp8_absorb import (
    FP8AbsorbWeights,
    fp8_out_absorb,
    fp8_out_absorb_out,
    fp8_q_absorb,
    fp8_q_absorb_out,
)


def _make_paths(batch_size: int):
    n_heads = 64
    qk_nope = 192
    v_dim = 512
    out_dim = 256
    q_nope = (torch.randn(batch_size, n_heads, qk_nope, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    attn_out = (torch.randn(batch_size, 1, n_heads, v_dim, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    q_absorb_w = (torch.randn(n_heads, qk_nope, v_dim, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    out_absorb_w = (torch.randn(n_heads, out_dim, v_dim, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    weights = FP8AbsorbWeights(q_absorb_w, out_absorb_w)
    q_graph = torch.empty(batch_size, n_heads, v_dim, device="cuda", dtype=torch.bfloat16)
    out_graph = torch.empty(batch_size, 1, n_heads, out_dim, device="cuda", dtype=torch.bfloat16)

    def allocating_path():
        return fp8_q_absorb(q_nope, weights), fp8_out_absorb(attn_out, weights)

    def out_path():
        fp8_q_absorb_out(q_nope, weights, q_graph)
        fp8_out_absorb_out(attn_out, weights, out_graph)
        return q_graph, out_graph

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
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(20260429)
    print("b,path,median_ms,p90_ms,min_ms,comparison")
    for batch_size in args.batch_sizes:
        allocating_path, out_path = _make_paths(batch_size)
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
                f"b={batch_size},{path},median_ms={summary[0]:.4f},"
                f"p90_ms={summary[1]:.4f},min_ms={summary[2]:.4f},{cmp_label}"
            )


if __name__ == "__main__":
    main()
