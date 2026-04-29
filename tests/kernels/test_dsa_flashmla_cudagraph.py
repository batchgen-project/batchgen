"""CUDA graph sanity for prepared BF16 selected-KV FlashMLA decode."""

from __future__ import annotations

import pytest
import torch

from batchgen.attention.dsa.sparse_decode_mla import (
    prepare_sparse_flash_mla_decode_inputs,
    run_prepared_sparse_flash_mla_decode,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSA FlashMLA CUDA graph tests require CUDA",
)


H_Q = 64
H_KV = 1
D_QK = 576
D_V = 512
INDEX_TOPK = 2048


def _require_flash_mla():
    return pytest.importorskip("flash_mla")


def _attention_reference(query_states, selected_mla_kv, selected_lengths, softmax_scale):
    out = torch.empty(
        query_states.shape[0],
        1,
        H_Q,
        D_V,
        device=query_states.device,
        dtype=torch.float32,
    )
    for row in range(query_states.shape[0]):
        valid = int(selected_lengths[row].item())
        q = query_states[row, 0].float()
        kv = selected_mla_kv[row, :valid, 0].float()
        scores = torch.matmul(q, kv.transpose(0, 1)) * softmax_scale
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
        out[row, 0] = torch.matmul(probs, kv[:, :D_V])
    return out.to(query_states.dtype)


def test_prepared_flashmla_decode_cuda_graph_replay_matches_eager_and_reference():
    _require_flash_mla()
    torch.cuda.set_device(0)
    torch.manual_seed(20260429)
    batch_size = 2
    selected_lengths = torch.tensor([1536, INDEX_TOPK], device="cuda", dtype=torch.int32)
    query_states = (torch.randn(batch_size, 1, H_Q, D_QK, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    selected_mla_kv = (
        torch.randn(batch_size, INDEX_TOPK, H_KV, D_QK, device="cuda", dtype=torch.bfloat16) * 0.1
    ).contiguous()
    selected_mla_kv[0, int(selected_lengths[0].item()):] = 0
    softmax_scale = D_QK**-0.5

    prepared = prepare_sparse_flash_mla_decode_inputs(
        query_states,
        selected_mla_kv,
        selected_lengths,
        H_Q,
        softmax_scale,
        head_dim_v=D_V,
    )
    eager = run_prepared_sparse_flash_mla_decode(prepared)
    ref = _attention_reference(query_states, selected_mla_kv, selected_lengths, softmax_scale)
    torch.testing.assert_close(eager, ref, atol=2e-2, rtol=2e-2)

    for _ in range(3):
        graph_out = run_prepared_sparse_flash_mla_decode(prepared)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = run_prepared_sparse_flash_mla_decode(prepared)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(graph_out, eager, atol=0, rtol=0)
