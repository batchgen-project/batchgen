"""CUDA graph sanity for GLM-5 DSA FP8 q_absorb/out_absorb kernels."""

from __future__ import annotations

import pytest
import torch

from batchgen_kernels.attention.dsa.fp8_absorb import (
    FP8AbsorbWeights,
    fp8_out_absorb,
    fp8_out_absorb_out,
    fp8_q_absorb,
    fp8_q_absorb_out,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSA absorb CUDA graph tests require CUDA",
)


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_dsa_absorb_out_matches_eager(batch_size: int):
    torch.cuda.set_device(0)
    torch.manual_seed(20260429 + batch_size)
    n_heads = 64
    qk_nope = 192
    v_dim = 512
    out_dim = 256

    q_nope = (torch.randn(batch_size, n_heads, qk_nope, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    attn_out = (torch.randn(batch_size, 1, n_heads, v_dim, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    q_absorb_w = (torch.randn(n_heads, qk_nope, v_dim, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    out_absorb_w = (torch.randn(n_heads, out_dim, v_dim, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    weights = FP8AbsorbWeights(q_absorb_w, out_absorb_w)

    q_eager = fp8_q_absorb(q_nope, weights)
    out_eager = fp8_out_absorb(attn_out, weights)
    q_out = torch.empty_like(q_eager)
    out_out = torch.empty_like(out_eager)

    fp8_q_absorb_out(q_nope, weights, q_out)
    fp8_out_absorb_out(attn_out, weights, out_out)
    torch.testing.assert_close(q_out, q_eager, atol=0, rtol=0)
    torch.testing.assert_close(out_out, out_eager, atol=0, rtol=0)

    q_ref = torch.einsum("bhd,hdc->bhc", q_nope.float(), q_absorb_w.float()).to(torch.bfloat16)
    out_ref = torch.einsum("bqhc,hdc->bqhd", attn_out.float(), out_absorb_w.float()).to(torch.bfloat16)
    assert _calc_diff(q_out, q_ref) < 1e-2
    assert _calc_diff(out_out, out_ref) < 1e-2


def test_dsa_absorb_out_cuda_graph_replay_matches_eager():
    torch.cuda.set_device(0)
    torch.manual_seed(20260430)
    batch_size = 4
    n_heads = 64
    qk_nope = 192
    v_dim = 512
    out_dim = 256

    q_nope = (torch.randn(batch_size, n_heads, qk_nope, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    attn_out = (torch.randn(batch_size, 1, n_heads, v_dim, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    q_absorb_w = (torch.randn(n_heads, qk_nope, v_dim, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    out_absorb_w = (torch.randn(n_heads, out_dim, v_dim, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    weights = FP8AbsorbWeights(q_absorb_w, out_absorb_w)
    q_graph = torch.empty(batch_size, n_heads, v_dim, device="cuda", dtype=torch.bfloat16)
    out_graph = torch.empty(batch_size, 1, n_heads, out_dim, device="cuda", dtype=torch.bfloat16)

    def run_graph_path():
        fp8_q_absorb_out(q_nope, weights, q_graph)
        fp8_out_absorb_out(attn_out, weights, out_graph)

    for _ in range(3):
        run_graph_path()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_graph_path()

    new_q = (torch.randn_like(q_nope) * 0.1).contiguous()
    new_attn = (torch.randn_like(attn_out) * 0.1).contiguous()
    q_nope.copy_(new_q)
    attn_out.copy_(new_attn)
    q_expected = fp8_q_absorb(q_nope, weights)
    out_expected = fp8_out_absorb(attn_out, weights)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(q_graph, q_expected, atol=0, rtol=0)
    torch.testing.assert_close(out_graph, out_expected, atol=0, rtol=0)
