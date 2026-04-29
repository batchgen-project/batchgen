"""Synthetic full GLM-5 DSA attention segment CUDA graph sanity."""

from __future__ import annotations

import pytest
import torch

from batchgen.attention.dsa.sparse_decode_mla import (
    prepare_sparse_flash_mla_decode_inputs,
    run_prepared_sparse_flash_mla_decode,
)
from batchgen.attention.dsa.unified_selector import select_mla_kv_for_flashmla_bf16_out
from batchgen_kernels.attention.dsa.fp8_absorb import (
    FP8AbsorbWeights,
    fp8_out_absorb_out,
    fp8_q_absorb_out,
)
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
from batchgen_kernels.attention.dsa.query_pack import pack_flashmla_query_out


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSA full-segment CUDA graph tests require CUDA",
)


PAGE_SIZE = 64
INDEX_H = 32
ATTN_H = 64
INDEX_D = 128
D_QK = 576
D_V = 512
Q_NOPE = 192
OUT_D = 256
H_KV = 1


def _require_flash_mla():
    return pytest.importorskip("flash_mla")


def _rope_tables(max_pos: int, rope_dim: int = 64):
    theta = 1000000.0
    freqs = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, device="cuda").float() / rope_dim))
    t = torch.arange(max_pos, device="cuda").float()
    angles = t[:, None] * freqs[None, :]
    return torch.cos(angles).repeat(1, 2).contiguous(), torch.sin(angles).repeat(1, 2).contiguous()


def _make_primary_cache(batch_size: int, max_seqlen: int):
    pages_per_seq = (max_seqlen + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch_size * pages_per_seq
    blocked_k = torch.randn(total_pages, PAGE_SIZE, H_KV, D_QK, device="cuda", dtype=torch.bfloat16)
    page_table = torch.arange(total_pages, device="cuda", dtype=torch.int32).view(batch_size, pages_per_seq)
    return blocked_k, page_table


def test_synthetic_full_dsa_attention_segment_cuda_graph_replay():
    _require_flash_mla()
    torch.cuda.set_device(0)
    torch.manual_seed(20260429)
    module = build_module()

    batch_size = 2
    q_rank = 2048
    max_seqlen = 1024
    topk = 128
    softmax_scale = D_QK**-0.5

    q_a = (torch.randn(batch_size, q_rank, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    wq_b = (torch.randn(INDEX_H * INDEX_D, q_rank, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    aux_cached_k = (torch.randn(batch_size, max_seqlen, INDEX_D, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    head_gates = torch.randn(batch_size, INDEX_H, device="cuda", dtype=torch.float32)
    cache_seqlens = torch.full((batch_size,), max_seqlen, device="cuda", dtype=torch.int32)
    positions = torch.tensor([max_seqlen - 5, max_seqlen - 1], device="cuda", dtype=torch.int64)
    positions_expanded = positions.repeat_interleave(INDEX_H).contiguous()
    cos, sin = _rope_tables(max_pos=max_seqlen + 8)
    primary_blocked_k, primary_page_table = _make_primary_cache(batch_size, max_seqlen)
    q_weights = FP8WqbWeightsCUDA(wq_b, module)

    q_nope = (torch.randn(batch_size, ATTN_H, Q_NOPE, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    q_rope = (torch.randn(batch_size, ATTN_H, 64, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    q_absorb_w = (torch.randn(ATTN_H, Q_NOPE, D_V, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    out_absorb_w = (torch.randn(ATTN_H, OUT_D, D_V, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    absorb_weights = FP8AbsorbWeights(q_absorb_w, out_absorb_w)

    q_x_fp8, q_x_scale, q_tma = make_fp8_activation_scratch(batch_size, q_rank, module, device="cuda")
    q_flat = torch.empty(batch_size, INDEX_H * INDEX_D, device="cuda", dtype=torch.bfloat16)
    q_index = torch.empty(batch_size, INDEX_H, INDEX_D, device="cuda", dtype=torch.bfloat16)
    agg = torch.empty(batch_size, max_seqlen, device="cuda", dtype=torch.float32)
    topk_indices = torch.empty(batch_size, topk, device="cuda", dtype=torch.long)
    selected = torch.empty(batch_size, topk, H_KV, D_QK, device="cuda", dtype=torch.bfloat16)
    selected_lengths = torch.empty(batch_size, device="cuda", dtype=torch.int32)
    selected_lengths.fill_(topk)
    row_modes = torch.empty(batch_size, device="cuda", dtype=torch.int32)
    absorbed_q = torch.empty(batch_size, ATTN_H, D_V, device="cuda", dtype=torch.bfloat16)
    query_states = torch.empty(batch_size, 1, ATTN_H, D_QK, device="cuda", dtype=torch.bfloat16)
    final_out = torch.empty(batch_size, 1, ATTN_H, OUT_D, device="cuda", dtype=torch.bfloat16)

    prepared = prepare_sparse_flash_mla_decode_inputs(
        query_states,
        selected,
        selected_lengths,
        ATTN_H,
        softmax_scale,
        head_dim_v=D_V,
        page_size=PAGE_SIZE,
    )

    def run_segment():
        cuda_wq_b_proj_out(q_a, q_weights, module, q_x_fp8, q_x_scale, q_tma, q_flat)
        rope_hadamard_q_out(q_flat.view(batch_size, INDEX_H, INDEX_D), cos, sin, positions_expanded, q_index)
        fused_score_and_topk_out(q_index, aux_cached_k, head_gates, cache_seqlens, agg, topk_indices, topk=topk)
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
        fp8_q_absorb_out(q_nope, absorb_weights, absorbed_q)
        pack_flashmla_query_out(absorbed_q, q_rope, query_states)
        attn_out = run_prepared_sparse_flash_mla_decode(prepared)
        fp8_out_absorb_out(attn_out, absorb_weights, final_out)

    run_segment()
    expected = final_out.clone()

    for _ in range(3):
        run_segment()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_segment()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(final_out, expected, atol=0, rtol=0)
