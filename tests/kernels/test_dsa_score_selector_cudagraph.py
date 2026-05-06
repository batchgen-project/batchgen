"""CUDA graph sanity for GLM-5 DSA score/top-k + selected-KV gather slice."""

from __future__ import annotations

import pytest
import torch

from batchgen.attention.dsa.unified_selector import select_mla_kv_for_flashmla_bf16_out
from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
    build_module,
    make_fp8_activation_scratch,
)
from batchgen_kernels.attention.dsa.fused_indexer_score import (
    FP8WqbWeightsCUDA,
    cuda_wq_b_proj_out,
    fused_score_and_topk_out,
    rope_hadamard_q_out,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSA score+selector CUDA graph tests require CUDA",
)


PAGE_SIZE = 64
H_KV = 1
D_QK = 576


def _rope_tables(max_pos: int, rope_dim: int = 64):
    theta = 1000000.0
    freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device="cuda").float() / rope_dim))
    t = torch.arange(max_pos, device="cuda").float()
    angles = t[:, None] * freqs[None, :]
    return torch.cos(angles).repeat(1, 2).contiguous(), torch.sin(angles).repeat(1, 2).contiguous()


def _make_primary_cache(batch_size: int, max_seqlen: int):
    pages_per_seq = (max_seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch_size * pages_per_seq
    blocked_k = torch.randn(
        total_pages,
        PAGE_SIZE,
        H_KV,
        D_QK,
        device="cuda",
        dtype=torch.bfloat16,
    )
    page_table = torch.arange(total_pages, device="cuda", dtype=torch.int32).view(batch_size, pages_per_seq)
    return blocked_k, page_table


def test_dsa_score_selector_cuda_graph_replay_matches_eager():
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
    aux_cached_k = (torch.randn(batch_size, max_seqlen, head_dim, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    head_gates = torch.randn(batch_size, n_heads, device="cuda", dtype=torch.float32)
    cache_seqlens = torch.full((batch_size,), max_seqlen, device="cuda", dtype=torch.int32)
    positions = torch.tensor([max_seqlen - 3, max_seqlen - 1], device="cuda", dtype=torch.int64)
    positions_expanded = positions.repeat_interleave(n_heads).contiguous()
    cos, sin = _rope_tables(max_pos=max_seqlen + 8)
    primary_blocked_k, primary_page_table = _make_primary_cache(batch_size, max_seqlen)
    q_weights = FP8WqbWeightsCUDA(wq_b, module)

    q_x_fp8, q_x_scale, q_tma = make_fp8_activation_scratch(
        batch_size,
        q_rank,
        module,
        device="cuda",
    )
    q_flat = torch.empty(batch_size, n_heads * head_dim, device="cuda", dtype=torch.bfloat16)
    q_rope = torch.empty(batch_size, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    agg = torch.empty(batch_size, max_seqlen, device="cuda", dtype=torch.float32)
    topk_indices = torch.empty(batch_size, topk, device="cuda", dtype=torch.long)
    selected = torch.empty(batch_size, topk, H_KV, D_QK, device="cuda", dtype=torch.bfloat16)
    selected_lengths = torch.empty(batch_size, device="cuda", dtype=torch.int32)
    row_modes = torch.empty(batch_size, device="cuda", dtype=torch.int32)

    def run_segment():
        cuda_wq_b_proj_out(q_a, q_weights, module, q_x_fp8, q_x_scale, q_tma, q_flat)
        rope_hadamard_q_out(q_flat.view(batch_size, n_heads, head_dim), cos, sin, positions_expanded, q_rope)
        fused_score_and_topk_out(
            q_rope,
            aux_cached_k,
            head_gates,
            cache_seqlens,
            agg,
            topk_indices,
            topk=topk,
        )
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

    run_segment()
    eager_selected = selected.clone()
    eager_lengths = selected_lengths.clone()
    eager_modes = row_modes.clone()

    for _ in range(3):
        run_segment()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_segment()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(selected, eager_selected, atol=0, rtol=0)
    torch.testing.assert_close(selected_lengths, eager_lengths, atol=0, rtol=0)
    torch.testing.assert_close(row_modes, eager_modes, atol=0, rtol=0)
