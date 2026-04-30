"""Benchmark manager-compatible GLM-5 DSA CUDA graph segment."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from batchgen.cuda_graph import BatchSizeBucketing, CUDAGraphManager
from batchgen.models.glm.glm5.cuda_graph_segments import (
    Glm5DsaAttnSegment,
    make_glm5_dsa_graph_segment_name,
)
from batchgen_kernels.attention.dsa.fp8_absorb import FP8AbsorbWeights
from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import build_module
from batchgen_kernels.attention.dsa.fused_indexer_score import FP8WqbWeightsCUDA


PAGE_SIZE = 64
INDEX_HEADS = 32
ATTN_HEADS = 64
INDEX_DIM = 128
Q_RANK = 2048
Q_NOPE = 192
KV_DIM = 576
KV_LORA = 512
ATTN_OUT = 256


def _rope_tables(max_pos: int, rope_dim: int = 64):
    theta = 1000000.0
    freqs = 1.0 / (
        theta ** (torch.arange(0, rope_dim, 2, device="cuda").float() / rope_dim)
    )
    t = torch.arange(max_pos, device="cuda").float()
    angles = t[:, None] * freqs[None, :]
    return (
        torch.cos(angles).repeat(1, 2).contiguous(),
        torch.sin(angles).repeat(1, 2).contiguous(),
    )


def _make_primary_cache(batch_size: int, max_seqlen: int):
    pages_per_seq = (max_seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch_size * pages_per_seq
    blocked_k = (
        torch.randn(
            total_pages,
            PAGE_SIZE,
            1,
            KV_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    ).contiguous()
    page_table = torch.arange(
        total_pages,
        device="cuda",
        dtype=torch.int32,
    ).view(batch_size, pages_per_seq)
    return blocked_k, page_table


def _make_aux_cache(batch_size: int, max_seqlen: int):
    pages_per_seq = (max_seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch_size * pages_per_seq
    blocked_k = (
        torch.randn(
            total_pages,
            PAGE_SIZE,
            1,
            INDEX_DIM,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.1
    ).contiguous()
    page_table = torch.arange(
        total_pages,
        device="cuda",
        dtype=torch.int32,
    ).view(batch_size, pages_per_seq)
    return blocked_k, page_table


def _make_inputs(
    batch_size: int,
    max_seqlen: int,
):
    positions = torch.randint(
        0,
        max_seqlen,
        (batch_size,),
        device="cuda",
        dtype=torch.int64,
    )
    return {
        "q_a": (
            torch.randn(batch_size, Q_RANK, device="cuda", dtype=torch.bfloat16) * 0.1
        ).contiguous(),
        "q_nope": (
            torch.randn(batch_size, ATTN_HEADS, Q_NOPE, device="cuda", dtype=torch.bfloat16)
            * 0.1
        ).contiguous(),
        "q_rope": (
            torch.randn(batch_size, ATTN_HEADS, 64, device="cuda", dtype=torch.bfloat16)
            * 0.1
        ).contiguous(),
        "head_gates": torch.randn(
            batch_size,
            INDEX_HEADS,
            device="cuda",
            dtype=torch.float32,
        ).contiguous(),
        "cache_seqlens": torch.full(
            (batch_size,),
            max_seqlen,
            device="cuda",
            dtype=torch.int32,
        ),
        "positions_expanded": positions[:, None]
        .expand(batch_size, INDEX_HEADS)
        .contiguous(),
        "primary_slot_indices": torch.arange(batch_size, device="cuda", dtype=torch.int32),
        "aux_slot_indices": torch.arange(batch_size, device="cuda", dtype=torch.int32),
    }


def _make_segment(batch_size: int, max_seqlen: int, index_topk: int, module):
    primary_blocked_k, primary_page_table = _make_primary_cache(batch_size, max_seqlen)
    aux_blocked_k, aux_page_table = _make_aux_cache(batch_size, max_seqlen)
    cos, sin = _rope_tables(max_seqlen + 8)
    wq_b = (
        torch.randn(INDEX_HEADS * INDEX_DIM, Q_RANK, device="cuda", dtype=torch.bfloat16)
        * 0.01
    ).contiguous()
    q_absorb = (
        torch.randn(ATTN_HEADS, Q_NOPE, KV_LORA, device="cuda", dtype=torch.bfloat16)
        * 0.01
    ).contiguous()
    out_absorb = (
        torch.randn(ATTN_HEADS, ATTN_OUT, KV_LORA, device="cuda", dtype=torch.bfloat16)
        * 0.01
    ).contiguous()
    segment = Glm5DsaAttnSegment(
        primary_blocked_k=primary_blocked_k,
        aux_blocked_k=aux_blocked_k,
        primary_page_table=primary_page_table,
        aux_page_table=aux_page_table,
        wq_b_weights=FP8WqbWeightsCUDA(wq_b, module),
        absorb_weights=FP8AbsorbWeights(q_absorb, out_absorb),
        cuda_module=module,
        cos_table=cos,
        sin_table=sin,
        max_seqlen=max_seqlen,
        index_topk=index_topk,
        page_size=PAGE_SIZE,
        softmax_scale=KV_DIM**-0.5,
    )
    return segment, primary_page_table, aux_page_table


def _time_cuda(fn, *, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--topk", type=int, default=2048)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(20260430)
    module = build_module()
    print("b,seqlen,topk,path,median_ms,p90_ms,min_ms,comparison")

    for batch_size in args.batch_sizes:
        for max_seqlen in args.seq_lens:
            topk = min(args.topk, max_seqlen)
            segment, primary_page_table, aux_page_table = _make_segment(
                batch_size,
                max_seqlen,
                topk,
                module,
            )
            inputs = _make_inputs(batch_size, max_seqlen)
            segment_name = make_glm5_dsa_graph_segment_name(0)
            manager = CUDAGraphManager(
                BatchSizeBucketing([batch_size]),
                device=torch.device("cuda"),
            )
            manager.register_segment(segment_name, segment)
            manager.warmup_and_capture_all()

            def eager_segment():
                segment.forward(**inputs)

            def graph_segment():
                manager.replay(segment_name, batch_size, **inputs)

            eager = _summary(_time_cuda(eager_segment, warmup=args.warmup, iters=args.iters))
            graph = _summary(_time_cuda(graph_segment, warmup=args.warmup, iters=args.iters))
            speedup = eager[0] / graph[0]
            for path, summary in (("eager_segment", eager), ("cudagraph", graph)):
                cmp_label = (
                    "baseline"
                    if path == "eager_segment"
                    else f"cudagraph {speedup:.2f}x faster"
                )
                print(
                    f"b={batch_size},seqlen={max_seqlen},topk={topk},{path},"
                    f"median_ms={summary[0]:.4f},p90_ms={summary[1]:.4f},"
                    f"min_ms={summary[2]:.4f},{cmp_label}"
                )


if __name__ == "__main__":
    main()
