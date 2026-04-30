"""Sanity tests for GLM-5 DSA custom fused score+topk."""

from __future__ import annotations

import pytest
import torch

from batchgen_kernels.attention.dsa.fused_indexer_score import (
    fused_score_and_topk,
    fused_score_and_topk_out,
    fused_paged_score_and_topk_out,
)
from batchgen_kernels.attention.dsa.fast_topk_cuda import fast_topk_2048


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused DSA indexer top-k tests require CUDA",
)


def _reference_scores(
    q: torch.Tensor,
    cached_k: torch.Tensor,
    head_gates: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> torch.Tensor:
    scores = torch.einsum("bhd,bsd->bhs", q.float(), cached_k.float())
    scores = (scores * head_gates.float().unsqueeze(-1)).sum(dim=1)
    pos = torch.arange(cached_k.shape[1], device=cached_k.device)
    scores = scores.masked_fill(pos.unsqueeze(0) >= cache_seqlens.unsqueeze(1), -float("inf"))
    return scores


def _make_paged_k(batch_size: int, max_seqlen: int, head_dim: int, page_size: int):
    pages_per_seq = (max_seqlen + page_size - 1) // page_size
    total_pages = batch_size * pages_per_seq
    blocked_k = (
        torch.randn(
            total_pages,
            page_size,
            1,
            head_dim,
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
    dense_k = blocked_k.view(batch_size, pages_per_seq * page_size, head_dim)[:, :max_seqlen]
    return blocked_k, page_table, dense_k.contiguous()


@pytest.mark.parametrize(
    ("batch_size", "max_seqlen", "topk"),
    [
        (1, 256, 32),
        (2, 512, 64),
        (4, 2048, 128),
    ],
)
def test_fused_score_and_topk_matches_torch_topk_values(batch_size, max_seqlen, topk):
    torch.cuda.set_device(0)
    torch.manual_seed(20260429 + batch_size + max_seqlen + topk)
    n_heads = 4
    head_dim = 32
    q = torch.randn(
        batch_size,
        n_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    cached_k = torch.randn(
        batch_size,
        max_seqlen,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    head_gates = torch.randn(batch_size, n_heads, device="cuda", dtype=torch.float32)
    cache_seqlens = torch.tensor(
        [max(topk, max_seqlen - 17 - i * 11) for i in range(batch_size)],
        device="cuda",
        dtype=torch.int32,
    )

    actual_indices = fused_score_and_topk(
        q,
        cached_k,
        head_gates,
        cache_seqlens,
        topk=topk,
    )
    ref_scores = _reference_scores(q, cached_k, head_gates, cache_seqlens)
    ref_values, _ = torch.topk(ref_scores, topk, dim=-1)
    actual_values = torch.gather(ref_scores, 1, actual_indices.long())

    torch.testing.assert_close(
        torch.sort(actual_values, dim=-1).values,
        torch.sort(ref_values, dim=-1).values,
        atol=2e-2,
        rtol=2e-2,
    )


def test_fused_score_and_topk_cuda_graph_replay_matches_eager():
    torch.cuda.set_device(0)
    torch.manual_seed(20260429)
    batch_size = 2
    max_seqlen = 512
    topk = 64
    n_heads = 4
    head_dim = 32
    q = torch.randn(
        batch_size,
        n_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    cached_k = torch.randn(
        batch_size,
        max_seqlen,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    head_gates = torch.randn(batch_size, n_heads, device="cuda", dtype=torch.float32)
    cache_seqlens = torch.tensor([max_seqlen, max_seqlen - 37], device="cuda", dtype=torch.int32)

    eager = fused_score_and_topk(q, cached_k, head_gates, cache_seqlens, topk=topk)
    agg = torch.empty(batch_size, max_seqlen, device="cuda", dtype=torch.float32)
    graph_out = torch.empty(batch_size, topk, device="cuda", dtype=torch.long)

    for _ in range(3):
        fused_score_and_topk_out(
            q,
            cached_k,
            head_gates,
            cache_seqlens,
            agg,
            graph_out,
            topk=topk,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_score_and_topk_out(
            q,
            cached_k,
            head_gates,
            cache_seqlens,
            agg,
            graph_out,
            topk=topk,
        )
    graph.replay()
    torch.cuda.synchronize()

    ref_scores = _reference_scores(q, cached_k, head_gates, cache_seqlens)
    torch.testing.assert_close(
        torch.sort(torch.gather(ref_scores, 1, graph_out.long()), dim=-1).values,
        torch.sort(torch.gather(ref_scores, 1, eager.long()), dim=-1).values,
        atol=2e-2,
        rtol=2e-2,
    )


@pytest.mark.parametrize(
    ("batch_size", "max_seqlen", "topk", "page_size"),
    [
        (1, 256, 32, 64),
        (3, 512, 64, 64),
        (7, 1024, 128, 64),
    ],
)
def test_fused_paged_score_and_topk_matches_dense_scorer(
    batch_size,
    max_seqlen,
    topk,
    page_size,
):
    torch.cuda.set_device(0)
    torch.manual_seed(20260430 + batch_size + max_seqlen + topk)
    n_heads = 4
    head_dim = 32
    q = (
        torch.randn(batch_size, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        * 0.1
    ).contiguous()
    aux_blocked_k, aux_page_table, dense_k = _make_paged_k(
        batch_size,
        max_seqlen,
        head_dim,
        page_size,
    )
    head_gates = torch.randn(batch_size, n_heads, device="cuda", dtype=torch.float32)
    cache_seqlens = torch.tensor(
        [max(topk, max_seqlen - 13 - i * 7) for i in range(batch_size)],
        device="cuda",
        dtype=torch.int32,
    )

    dense_topk = fused_score_and_topk(q, dense_k, head_gates, cache_seqlens, topk=topk)
    agg = torch.empty(batch_size, max_seqlen, device="cuda", dtype=torch.float32)
    paged_topk = torch.empty(batch_size, topk, device="cuda", dtype=torch.long)
    fused_paged_score_and_topk_out(
        q,
        aux_blocked_k,
        aux_page_table,
        head_gates,
        cache_seqlens,
        agg,
        paged_topk,
        topk=topk,
        page_size=page_size,
        max_seqlen=max_seqlen,
    )
    ref_scores = _reference_scores(q, dense_k, head_gates, cache_seqlens)
    torch.testing.assert_close(
        torch.sort(torch.gather(ref_scores, 1, paged_topk.long()), dim=-1).values,
        torch.sort(torch.gather(ref_scores, 1, dense_topk.long()), dim=-1).values,
        atol=2e-2,
        rtol=2e-2,
    )


def test_fast_topk_2048_cuda_matches_torch_topk_values():
    torch.cuda.set_device(0)
    torch.manual_seed(20260430)
    batch_size = 3
    max_seqlen = 4096
    scores = torch.randn(batch_size, max_seqlen, device="cuda", dtype=torch.float32)
    lengths = torch.tensor([4096, 3072, 2057], device="cuda", dtype=torch.int32)
    pos = torch.arange(max_seqlen, device="cuda")
    masked_scores = scores.masked_fill(pos.unsqueeze(0) >= lengths.unsqueeze(1), -float("inf"))

    actual = fast_topk_2048(masked_scores, lengths)
    ref_values, _ = torch.topk(masked_scores, 2048, dim=-1)
    actual_values = torch.gather(masked_scores, 1, actual.long())

    torch.testing.assert_close(
        torch.sort(actual_values, dim=-1).values,
        torch.sort(ref_values, dim=-1).values,
        atol=0,
        rtol=0,
    )


def test_fused_paged_score_and_topk_cuda_graph_replay_matches_eager():
    torch.cuda.set_device(0)
    torch.manual_seed(20260430)
    batch_size = 2
    max_seqlen = 512
    topk = 64
    page_size = 64
    n_heads = 4
    head_dim = 32
    q = (
        torch.randn(batch_size, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        * 0.1
    ).contiguous()
    aux_blocked_k, aux_page_table, _ = _make_paged_k(
        batch_size,
        max_seqlen,
        head_dim,
        page_size,
    )
    head_gates = torch.randn(batch_size, n_heads, device="cuda", dtype=torch.float32)
    cache_seqlens = torch.tensor(
        [max_seqlen, max_seqlen - 37],
        device="cuda",
        dtype=torch.int32,
    )
    agg = torch.empty(batch_size, max_seqlen, device="cuda", dtype=torch.float32)
    eager = torch.empty(batch_size, topk, device="cuda", dtype=torch.long)
    graph_out = torch.empty(batch_size, topk, device="cuda", dtype=torch.long)
    fused_paged_score_and_topk_out(
        q,
        aux_blocked_k,
        aux_page_table,
        head_gates,
        cache_seqlens,
        agg,
        eager,
        topk=topk,
        page_size=page_size,
        max_seqlen=max_seqlen,
    )

    def run_graph_path():
        fused_paged_score_and_topk_out(
            q,
            aux_blocked_k,
            aux_page_table,
            head_gates,
            cache_seqlens,
            agg,
            graph_out,
            topk=topk,
            page_size=page_size,
            max_seqlen=max_seqlen,
        )

    for _ in range(3):
        run_graph_path()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_graph_path()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_out, eager)
