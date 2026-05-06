"""Benchmark GLM-5 DSA Q/K projection out-buffer path and CUDA graph replay."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
    FP8IndexerWeightsCUDA,
    build_module,
    cuda_wk_proj_gemm_only,
    cuda_wk_proj_gemm_only_out,
    make_fp8_activation_scratch,
)
from batchgen_kernels.attention.dsa.fused_indexer_score import (
    FP8WqbWeightsCUDA,
    cuda_wq_b_proj,
    cuda_wq_b_proj_out,
)


def _make_case(batch_size: int):
    q_rank = 2048
    hidden_size = 6144
    q_out_dim = 4096
    k_out_dim = 128
    q_a = (torch.randn(batch_size, q_rank, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    hidden = (torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    wq_b = (torch.randn(q_out_dim, q_rank, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    wk = (torch.randn(k_out_dim, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    return q_a, hidden, wq_b, wk


def _make_out_path(batch_size: int, module):
    q_a, hidden, wq_b, wk = _make_case(batch_size)
    q_weights = FP8WqbWeightsCUDA(wq_b, module)
    k_weights = FP8IndexerWeightsCUDA(wk, module)
    q_x_fp8, q_x_scale, q_tma = make_fp8_activation_scratch(
        batch_size,
        q_a.shape[1],
        module,
        device="cuda",
    )
    k_x_fp8, k_x_scale, k_tma = make_fp8_activation_scratch(
        batch_size,
        hidden.shape[1],
        module,
        device="cuda",
    )
    q_out = torch.empty(batch_size, q_weights.N, device="cuda", dtype=torch.bfloat16)
    k_out = torch.empty(batch_size, k_weights.N, device="cuda", dtype=torch.bfloat16)

    def out_path():
        cuda_wq_b_proj_out(q_a, q_weights, module, q_x_fp8, q_x_scale, q_tma, q_out)
        cuda_wk_proj_gemm_only_out(hidden, k_weights, module, k_x_fp8, k_x_scale, k_tma, k_out)
        return q_out, k_out

    def allocating_path():
        return cuda_wq_b_proj(q_a, q_weights, module), cuda_wk_proj_gemm_only(hidden, k_weights, module)

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
    module = build_module()
    print("b,path,median_ms,p90_ms,min_ms,comparison")
    for batch_size in args.batch_sizes:
        allocating_path, out_path = _make_out_path(batch_size, module)
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
