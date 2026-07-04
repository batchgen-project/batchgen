#!/usr/bin/env python3
"""Benchmark the DeepSeek-V4 MXFP4 grouped-MoE paths.

Measures:
- GPU kernel time via CUDA events (captures route_pack GPU ops + Triton kernels)
- Wall time via perf_counter + synchronize

Example:
  python -m batchgen.moe.bench_v4_mega3_moe --tokens 64 --iters 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def _build_case(tokens: int):
    from batchgen.moe.v4_slot_moe_sm120 import setup_v4_expert_weight_pointers

    torch.manual_seed(4000 + tokens)
    hidden, inter, n_experts, topk = 4096, 2048, 32, 6
    swiglu_limit = 10.0
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device="cuda") * 0.5

    def _rand_fp4(out_dim: int, in_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        packed = torch.randint(
            0,
            256,
            (out_dim, in_dim // 2),
            dtype=torch.uint8,
            device="cuda",
        )
        scale = torch.randint(
            120,
            132,
            (out_dim, in_dim // 32),
            dtype=torch.uint8,
            device="cuda",
        )
        return packed.view(torch.float4_e2m1fn_x2).contiguous(), scale.contiguous()

    weight_dicts = []
    for _ in range(n_experts):
        rw = {}
        for name, out_dim, in_dim in (
            ("w1", inter, hidden),
            ("w2", hidden, inter),
            ("w3", inter, hidden),
        ):
            rw[f"{name}.weight"], rw[f"{name}.scale"] = _rand_fp4(
                out_dim, in_dim
            )
        weight_dicts.append(rw)

    logits = torch.randn(tokens, n_experts, device="cuda")
    topk_weights, topk_indices = torch.topk(
        torch.softmax(logits.float(), dim=-1), topk, dim=-1
    )
    staged = setup_v4_expert_weight_pointers(weight_dicts)
    return x, topk_weights, topk_indices.to(torch.int64), staged, n_experts, swiglu_limit


def _bench_cuda_us(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _bench_wall_us(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def _maybe_load_sglang_reference(tokens: int) -> float | None:
    path = Path(__file__).resolve().parents[2] / "logs/sglang_v4_flash_decode_rows.jsonl"
    if not path.exists():
        return None
    best = None
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("size") == tokens:
                median = row.get("median_us")
                if median is not None:
                    best = float(median)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[64])
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    from batchgen.moe.v4_mega3_moe_sm120 import v4_mega3_moe_forward
    from batchgen.moe.v4_ragged_moe_sm120 import (
        v4_grouped_mxfp4_moe_forward_ragged_ptrs,
    )

    print(f"device={torch.cuda.get_device_name()} iters={args.iters} warmup={args.warmup}")
    print(
        f"{'tokens':>8} | {'path':>8} | {'kernel_us':>10} | {'wall_us':>10} | {'sglang_ref_us':>13}"
    )
    print("-" * 64)
    for tokens in args.tokens:
        x, topk_weights, topk_indices, staged, n_experts, lim = _build_case(tokens)

        def run_ragged():
            return v4_grouped_mxfp4_moe_forward_ragged_ptrs(
                x, topk_weights, topk_indices, staged, 0, n_experts, lim
            )

        def run_mega3():
            return v4_mega3_moe_forward(
                x, topk_weights, topk_indices, staged, 0, n_experts, lim
            )

        for name, fn in (("ragged", run_ragged), ("mega3", run_mega3)):
            kernel_us = _bench_cuda_us(fn, args.iters, args.warmup)
            wall_us = _bench_wall_us(fn, args.iters, args.warmup)
            sglang_ref = _maybe_load_sglang_reference(tokens)
            sglang_str = f"{sglang_ref:.1f}" if sglang_ref is not None else "n/a"
            print(
                f"{tokens:8d} | {name:>8} | {kernel_us:10.1f} | {wall_us:10.1f} | {sglang_str:>13}"
            )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    main()
