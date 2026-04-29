"""Sanity tests for GLM-5 DSA custom fused score+topk."""

from __future__ import annotations

import pytest
import torch

from batchgen_kernels.attention.dsa.fused_indexer_score import (
    fused_score_and_topk,
    fused_score_and_topk_out,
)


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
    actual_values = torch.gather(ref_scores, 1, actual_indices)

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

    torch.testing.assert_close(graph_out, eager)
