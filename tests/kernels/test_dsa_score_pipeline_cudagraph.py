"""CUDA graph sanity for GLM-5 DSA q projection + score/top-k slice."""

from __future__ import annotations

import pytest
import torch

from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
    build_module,
    make_fp8_activation_scratch,
)
from batchgen_kernels.attention.dsa.fused_indexer_score import (
    FP8WqbWeightsCUDA,
    cuda_wq_b_proj,
    cuda_wq_b_proj_out,
    fused_score_and_topk,
    fused_score_and_topk_out,
    rope_hadamard_q,
    rope_hadamard_q_out,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSA score pipeline CUDA graph tests require CUDA",
)


def _rope_tables(max_pos: int, rope_dim: int = 64):
    theta = 1000000.0
    freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device="cuda").float() / rope_dim))
    t = torch.arange(max_pos, device="cuda").float()
    angles = t[:, None] * freqs[None, :]
    cos = torch.cos(angles).repeat(1, 2).contiguous()
    sin = torch.sin(angles).repeat(1, 2).contiguous()
    return cos, sin


def _selected_values(scores: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.gather(scores, 1, indices)


def _reference_scores(
    q: torch.Tensor,
    cached_k: torch.Tensor,
    head_gates: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> torch.Tensor:
    scores = torch.einsum("bhd,bsd->bhs", q.float(), cached_k.float())
    scores = (scores * head_gates.float().unsqueeze(-1)).sum(dim=1)
    pos = torch.arange(cached_k.shape[1], device=cached_k.device)
    return scores.masked_fill(pos.unsqueeze(0) >= cache_seqlens.unsqueeze(1), -float("inf"))


def test_dsa_q_projection_score_topk_cuda_graph_replay():
    torch.cuda.set_device(0)
    torch.manual_seed(20260429)
    module = build_module()

    batch_size = 2
    n_heads = 32
    head_dim = 128
    q_rank = 2048
    max_seqlen = 1024
    topk = 128

    q_a = (torch.randn(batch_size, q_rank, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    wq_b = (torch.randn(n_heads * head_dim, q_rank, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    cached_k = (torch.randn(batch_size, max_seqlen, head_dim, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    head_gates = torch.randn(batch_size, n_heads, device="cuda", dtype=torch.float32)
    cache_seqlens = torch.tensor([max_seqlen, max_seqlen - 17], device="cuda", dtype=torch.int32)
    positions = torch.tensor([123, 997], device="cuda", dtype=torch.int64)
    positions_expanded = positions.repeat_interleave(n_heads).contiguous()
    cos, sin = _rope_tables(max_pos=2048)
    q_weights = FP8WqbWeightsCUDA(wq_b, module)

    q_eager = cuda_wq_b_proj(q_a, q_weights, module).view(batch_size, n_heads, head_dim)
    q_eager = rope_hadamard_q(q_eager, cos, sin, positions)
    eager_topk = fused_score_and_topk(
        q_eager,
        cached_k,
        head_gates,
        cache_seqlens,
        topk=topk,
    )

    q_x_fp8, q_x_scale, q_tma = make_fp8_activation_scratch(
        batch_size,
        q_rank,
        module,
        device="cuda",
    )
    q_flat = torch.empty(batch_size, n_heads * head_dim, device="cuda", dtype=torch.bfloat16)
    q_rope = torch.empty(batch_size, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    agg = torch.empty(batch_size, max_seqlen, device="cuda", dtype=torch.float32)
    graph_topk = torch.empty(batch_size, topk, device="cuda", dtype=torch.long)

    def run_graph_path():
        cuda_wq_b_proj_out(q_a, q_weights, module, q_x_fp8, q_x_scale, q_tma, q_flat)
        rope_hadamard_q_out(
            q_flat.view(batch_size, n_heads, head_dim),
            cos,
            sin,
            positions_expanded,
            q_rope,
        )
        fused_score_and_topk_out(
            q_rope,
            cached_k,
            head_gates,
            cache_seqlens,
            agg,
            graph_topk,
            topk=topk,
        )

    for _ in range(3):
        run_graph_path()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_graph_path()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(q_rope, q_eager, atol=0, rtol=0)
    ref_scores = _reference_scores(q_eager, cached_k, head_gates, cache_seqlens)
    torch.testing.assert_close(
        torch.sort(_selected_values(ref_scores, graph_topk), dim=-1).values,
        torch.sort(_selected_values(ref_scores, eager_topk), dim=-1).values,
        atol=2e-2,
        rtol=2e-2,
    )
