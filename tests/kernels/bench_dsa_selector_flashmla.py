"""Benchmark GLM-5 BF16 DSA selector/FlashMLA-input prep on CUDA.

This is a synthetic microbenchmark for the pre-integration gate. It compares:

1. current GLM hot path:
   - all-short rows gather only ``max_seqlen`` selected tokens;
   - mixed/all-long rows gather 2048 selected tokens.
2. fused unified selector:
   - every row gathers 2048 selected tokens and uses ``selected_lengths`` to
     tell FlashMLA the valid prefix.

The benchmark includes the sparse gather kernel and FlashMLA input preparation,
and optionally includes the FlashMLA invocation itself.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from batchgen.attention.dsa.sparse_decode_mla import (
    prepare_sparse_flash_mla_decode_inputs,
    run_prepared_sparse_flash_mla_decode,
)
from batchgen.attention.dsa.sparse_gather import sparse_gather_from_paged_kv
from batchgen.attention.dsa.unified_selector import select_mla_kv_for_flashmla_bf16
from batchgen.attention.dsa.unified_selector import select_mla_kv_for_flashmla_bf16_out
from batchgen.models.glm.glm5.decode_utils import build_clamped_dense_token_indices


INDEX_TOPK = 2048
PAGE_SIZE = 64
H_Q = 64
H_KV = 1
D_QK = 576
D_V = 512
SOFTMAX_SCALE = D_QK**-0.5


@dataclass(frozen=True)
class BenchCase:
    name: str
    seqlens: tuple[int, ...]


def _make_cases(batch_size: int) -> list[BenchCase]:
    def take(values: list[int]) -> tuple[int, ...]:
        return tuple(values[i % len(values)] for i in range(batch_size))

    return [
        BenchCase("short_512", take([512])),
        BenchCase("short_boundary", take([2048])),
        BenchCase("long_4096", take([4096])),
        BenchCase("mixed_boundary", take([64, 2047, 2048, 2049])),
        BenchCase("mixed_long", take([512, 2049, 4096, 8192])),
    ]


def _make_query(batch_size: int) -> torch.Tensor:
    query = torch.randn(
        batch_size,
        1,
        H_Q,
        D_QK,
        device="cuda",
        dtype=torch.bfloat16,
    )
    return (query / 10).clamp(-1.0, 1.0)


def _make_paged_cache(batch_size: int, max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    pages_per_seq = math.ceil(max_tokens / PAGE_SIZE)
    total_pages = batch_size * pages_per_seq
    logical = torch.randn(
        batch_size,
        max_tokens,
        H_KV,
        D_QK,
        device="cuda",
        dtype=torch.bfloat16,
    )
    logical = (logical / 10).clamp(-1.0, 1.0)
    page_table = torch.randperm(
        total_pages,
        device="cuda",
        dtype=torch.int32,
    ).view(batch_size, pages_per_seq)
    blocked_k = torch.empty(
        total_pages,
        PAGE_SIZE,
        H_KV,
        D_QK,
        device="cuda",
        dtype=torch.bfloat16,
    )
    for row in range(batch_size):
        for logical_page in range(pages_per_seq):
            physical_page = int(page_table[row, logical_page].item())
            start = logical_page * PAGE_SIZE
            end = min(start + PAGE_SIZE, max_tokens)
            blocked_k[physical_page].zero_()
            blocked_k[physical_page, : end - start] = logical[row, start:end]
    return blocked_k, page_table


def _long_topk_indices(batch_size: int, max_tokens: int) -> torch.Tensor:
    base = torch.arange(INDEX_TOPK, device="cuda", dtype=torch.long)
    indices = torch.empty(batch_size, INDEX_TOPK, device="cuda", dtype=torch.long)
    for row in range(batch_size):
        if row % 3 == 0:
            indices[row] = base
        elif row % 3 == 1:
            indices[row] = torch.randperm(max_tokens, device="cuda")[:INDEX_TOPK]
        else:
            stride = max(1, max_tokens // INDEX_TOPK)
            indices[row] = (base * stride).clamp(max=max_tokens - 1)
    return indices


def _previous_indices(
    cache_seqlens: torch.Tensor,
    max_seqlen: int,
    long_topk_indices: torch.Tensor,
) -> torch.Tensor:
    short_mask = cache_seqlens <= INDEX_TOPK
    if bool(short_mask.all().item()):
        return build_clamped_dense_token_indices(
            cache_seqlens,
            max_seqlen,
            cache_seqlens.device,
        )
    if not bool(short_mask.any().item()):
        return long_topk_indices
    out = torch.empty(
        cache_seqlens.numel(),
        INDEX_TOPK,
        device=cache_seqlens.device,
        dtype=torch.long,
    )
    out[short_mask] = build_clamped_dense_token_indices(
        cache_seqlens[short_mask],
        INDEX_TOPK,
        cache_seqlens.device,
    )
    out[~short_mask] = long_topk_indices[~short_mask]
    return out


def _fixed_indices(
    cache_seqlens: torch.Tensor,
    long_topk_indices: torch.Tensor,
) -> torch.Tensor:
    short_mask = cache_seqlens <= INDEX_TOPK
    dense = build_clamped_dense_token_indices(
        cache_seqlens,
        INDEX_TOPK,
        cache_seqlens.device,
    )
    return torch.where(short_mask.unsqueeze(1), dense, long_topk_indices)


def _selector_prepare(
    query_states: torch.Tensor,
    blocked_k: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    token_indices: torch.Tensor,
    *,
    unified: bool,
):
    if unified:
        selected, selected_lengths, _, _ = select_mla_kv_for_flashmla_bf16(
            blocked_k,
            page_table,
            cache_seqlens,
            token_indices,
            index_topk=INDEX_TOPK,
            page_size=PAGE_SIZE,
            return_indices=False,
        )
    else:
        selected = sparse_gather_from_paged_kv(
            blocked_k,
            page_table,
            token_indices,
            PAGE_SIZE,
        )
        selected_lengths = torch.clamp(cache_seqlens, max=token_indices.shape[1]).to(
            dtype=torch.int32,
        )
    return prepare_sparse_flash_mla_decode_inputs(
        query_states,
        selected,
        selected_lengths,
        H_Q,
        SOFTMAX_SCALE,
        head_dim_v=D_V,
        page_size=PAGE_SIZE,
    )


def _selector_only(
    blocked_k: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    token_indices: torch.Tensor,
):
    return select_mla_kv_for_flashmla_bf16(
        blocked_k,
        page_table,
        cache_seqlens,
        token_indices,
        index_topk=INDEX_TOPK,
        page_size=PAGE_SIZE,
        return_indices=False,
    )


def _make_selector_graph_fn(
    blocked_k: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    token_indices: torch.Tensor,
):
    selected = torch.empty(
        token_indices.shape[0],
        INDEX_TOPK,
        H_KV,
        D_QK,
        device="cuda",
        dtype=torch.bfloat16,
    )
    selected_lengths = torch.empty(
        token_indices.shape[0],
        device="cuda",
        dtype=torch.int32,
    )
    row_modes = torch.empty(
        token_indices.shape[0],
        device="cuda",
        dtype=torch.int32,
    )

    def run_out():
        return select_mla_kv_for_flashmla_bf16_out(
            blocked_k,
            page_table,
            cache_seqlens,
            token_indices,
            PAGE_SIZE,
            selected,
            selected_lengths,
            None,
            row_modes,
            index_topk=INDEX_TOPK,
            return_indices=False,
        )

    for _ in range(3):
        run_out()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_out()

    def replay():
        graph.replay()
        return selected, selected_lengths, None, row_modes

    return replay


def _time_cuda(fn, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))
    return times_ms


def _summarize(times_ms: list[float]) -> tuple[float, float, float, float, float]:
    return (
        statistics.median(times_ms),
        sorted(times_ms)[int(0.5 * (len(times_ms) - 1))],
        sorted(times_ms)[int(0.9 * (len(times_ms) - 1))],
        sorted(times_ms)[int(0.99 * (len(times_ms) - 1))],
        min(times_ms),
    )


def _compare_label(previous_ms: float, fixed_ms: float) -> str:
    if fixed_ms <= previous_ms:
        return f"fused_unified {previous_ms / fixed_ms:.2f}x faster"
    return f"current_glm_hot {fixed_ms / previous_ms:.2f}x faster"


def _bench_rope_hadamard(batch_size: int, max_seqlen: int, warmup: int, iters: int) -> None:
    try:
        from batchgen.other_kernels.hadamard_transform import fused_rope_hadamard
    except Exception as exc:
        print(f"rope_hadamard,b={batch_size},skipped,{exc}")
        return

    x = torch.randn(batch_size * 32, 128, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(max_seqlen, 64, device="cuda", dtype=torch.float32)
    sin = torch.randn(max_seqlen, 64, device="cuda", dtype=torch.float32)
    positions = torch.randint(0, max_seqlen, (batch_size * 32,), device="cuda")
    times = _time_cuda(
        lambda: fused_rope_hadamard(x, cos, sin, positions, scale=128**-0.5),
        warmup=warmup,
        iters=iters,
    )
    median_ms, p50_ms, p90_ms, p99_ms, min_ms = _summarize(times)
    print(
        "rope_hadamard,"
        f"b={batch_size},median_ms={median_ms:.4f},p50_ms={p50_ms:.4f},"
        f"p90_ms={p90_ms:.4f},p99_ms={p99_ms:.4f},min_ms={min_ms:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--include-flashmla", action="store_true")
    parser.add_argument("--include-rope-hadamard", action="store_true")
    parser.add_argument("--include-cudagraph", action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    print(
        "case,b,path,median_ms,p50_ms,p90_ms,p99_ms,min_ms,comparison"
    )
    for batch_size in args.batch_sizes:
        for case in _make_cases(batch_size):
            cache_seqlens = torch.tensor(case.seqlens, device="cuda", dtype=torch.int32)
            max_tokens = max(max(case.seqlens), INDEX_TOPK)
            blocked_k, page_table = _make_paged_cache(batch_size, max_tokens)
            query_states = _make_query(batch_size)
            long_topk = _long_topk_indices(batch_size, max_tokens)
            previous_indices = _previous_indices(cache_seqlens, max(case.seqlens), long_topk)
            fixed_indices = _fixed_indices(cache_seqlens, long_topk)

            def previous_prepare():
                return _selector_prepare(
                    query_states,
                    blocked_k,
                    page_table,
                    cache_seqlens,
                    previous_indices,
                    unified=False,
                )

            def fixed_prepare():
                return _selector_prepare(
                    query_states,
                    blocked_k,
                    page_table,
                    cache_seqlens,
                    fixed_indices,
                    unified=True,
                )

            if args.include_flashmla:
                def previous_fn():
                    return run_prepared_sparse_flash_mla_decode(previous_prepare())

                def fixed_fn():
                    return run_prepared_sparse_flash_mla_decode(fixed_prepare())
            else:
                previous_fn = previous_prepare
                fixed_fn = fixed_prepare

            previous_times = _time_cuda(previous_fn, warmup=args.warmup, iters=args.iters)
            fixed_times = _time_cuda(fixed_fn, warmup=args.warmup, iters=args.iters)
            previous_summary = _summarize(previous_times)
            fixed_summary = _summarize(fixed_times)
            comparison = _compare_label(previous_summary[0], fixed_summary[0])
            for path, summary in (
                ("current_glm_hot", previous_summary),
                ("fused_unified", fixed_summary),
            ):
                print(
                    f"{case.name},b={batch_size},{path},"
                    f"median_ms={summary[0]:.4f},p50_ms={summary[1]:.4f},"
                    f"p90_ms={summary[2]:.4f},p99_ms={summary[3]:.4f},"
                    f"min_ms={summary[4]:.4f},{comparison}"
                )

            if args.include_cudagraph:
                eager_selector_times = _time_cuda(
                    lambda: _selector_only(
                        blocked_k,
                        page_table,
                        cache_seqlens,
                        fixed_indices,
                    ),
                    warmup=args.warmup,
                    iters=args.iters,
                )
                graph_selector_fn = _make_selector_graph_fn(
                    blocked_k,
                    page_table,
                    cache_seqlens,
                    fixed_indices,
                )
                graph_selector_times = _time_cuda(
                    graph_selector_fn,
                    warmup=args.warmup,
                    iters=args.iters,
                )
                eager_selector_summary = _summarize(eager_selector_times)
                graph_selector_summary = _summarize(graph_selector_times)
                graph_comparison = _compare_label(
                    eager_selector_summary[0],
                    graph_selector_summary[0],
                )
                for path, summary in (
                    ("fused_selector_eager", eager_selector_summary),
                    ("fused_selector_cudagraph", graph_selector_summary),
                ):
                    print(
                        f"{case.name},b={batch_size},{path},"
                        f"median_ms={summary[0]:.4f},p50_ms={summary[1]:.4f},"
                        f"p90_ms={summary[2]:.4f},p99_ms={summary[3]:.4f},"
                        f"min_ms={summary[4]:.4f},{graph_comparison}"
                    )

        if args.include_rope_hadamard:
            _bench_rope_hadamard(batch_size, 8192, args.warmup, args.iters)


if __name__ == "__main__":
    main()
