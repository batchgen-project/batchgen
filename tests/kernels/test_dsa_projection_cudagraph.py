"""CUDA graph sanity for GLM-5 DSA Q/K projection kernels."""

from __future__ import annotations

import pytest
import torch

from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
    FP8IndexerWeightsCUDA,
    build_module,
    cuda_wk_proj_gemm_only,
    cuda_wk_proj_gemm_only_out,
    make_fp8_activation_scratch,
)
from batchgen_kernels.attention.dsa.fused_indexer_score import (
    FP8WqbWeightsCUDA,
    cuda_wq_b_proj,
    cuda_wq_b_proj_out,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSA projection CUDA graph tests require CUDA",
)


def _calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double()
    y = y.double()
    denom = (x * x + y * y).sum()
    if denom == 0:
        return 0.0
    return float(1 - 2 * (x * y).sum() / denom)


def _dequant_weight(weights: FP8IndexerWeightsCUDA) -> torch.Tensor:
    out = torch.empty(weights.N, weights.K, dtype=torch.float32, device=weights.w_fp8.device)
    for n_tile in range(weights.N // 32):
        ns = n_tile * 32
        ne = ns + 32
        for kb in range(weights.w_scale.shape[1]):
            ks = kb * weights.block_k
            ke = min(ks + weights.block_k, weights.K)
            out[ns:ne, ks:ke] = weights.w_fp8[ns:ne, ks:ke].float() * weights.w_scale[n_tile, kb]
    return out


def _fp8_linear_reference(x: torch.Tensor, weights: FP8IndexerWeightsCUDA, module) -> torch.Tensor:
    x_fp8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    x_scale = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    module.run_act_quant(x, x_fp8, x_scale)
    x_dequant = x_fp8.float() * x_scale[:, None]
    return torch.nn.functional.linear(x_dequant, _dequant_weight(weights)).to(torch.bfloat16)


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_dsa_qk_projection_out_matches_eager_and_fp8_reference(batch_size: int):
    torch.cuda.set_device(0)
    torch.manual_seed(20260429 + batch_size)
    module = build_module()

    q_rank = 2048
    hidden_size = 6144
    q_out_dim = 4096
    k_out_dim = 128

    q_a = (torch.randn(batch_size, q_rank, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    hidden = (torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    wq_b = (torch.randn(q_out_dim, q_rank, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    wk = (torch.randn(k_out_dim, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()

    q_weights = FP8WqbWeightsCUDA(wq_b, module)
    k_weights = FP8IndexerWeightsCUDA(wk, module)

    q_eager = cuda_wq_b_proj(q_a, q_weights, module)
    k_eager = cuda_wk_proj_gemm_only(hidden, k_weights, module)

    q_x_fp8, q_x_scale, q_tma = make_fp8_activation_scratch(
        batch_size,
        q_rank,
        module,
        device="cuda",
    )
    k_x_fp8, k_x_scale, k_tma = make_fp8_activation_scratch(
        batch_size,
        hidden_size,
        module,
        device="cuda",
    )
    q_out = torch.empty_like(q_eager)
    k_out = torch.empty_like(k_eager)

    cuda_wq_b_proj_out(q_a, q_weights, module, q_x_fp8, q_x_scale, q_tma, q_out)
    cuda_wk_proj_gemm_only_out(hidden, k_weights, module, k_x_fp8, k_x_scale, k_tma, k_out)

    torch.testing.assert_close(q_out, q_eager, atol=0, rtol=0)
    torch.testing.assert_close(k_out, k_eager, atol=0, rtol=0)
    assert _calc_diff(q_out, _fp8_linear_reference(q_a, q_weights, module)) < 1e-3
    assert _calc_diff(k_out, _fp8_linear_reference(hidden, k_weights, module)) < 1e-3


def test_dsa_qk_projection_out_cuda_graph_replay_reads_updated_inputs():
    torch.cuda.set_device(0)
    torch.manual_seed(20260430)
    module = build_module()

    batch_size = 4
    q_rank = 2048
    hidden_size = 6144
    q_out_dim = 4096
    k_out_dim = 128

    q_a = (torch.randn(batch_size, q_rank, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    hidden = (torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1).contiguous()
    wq_b = (torch.randn(q_out_dim, q_rank, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()
    wk = (torch.randn(k_out_dim, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.01).contiguous()

    q_weights = FP8WqbWeightsCUDA(wq_b, module)
    k_weights = FP8IndexerWeightsCUDA(wk, module)
    q_x_fp8, q_x_scale, q_tma = make_fp8_activation_scratch(
        batch_size,
        q_rank,
        module,
        device="cuda",
    )
    k_x_fp8, k_x_scale, k_tma = make_fp8_activation_scratch(
        batch_size,
        hidden_size,
        module,
        device="cuda",
    )
    q_graph = torch.empty(batch_size, q_out_dim, device="cuda", dtype=torch.bfloat16)
    k_graph = torch.empty(batch_size, k_out_dim, device="cuda", dtype=torch.bfloat16)

    def run_graph_path():
        cuda_wq_b_proj_out(q_a, q_weights, module, q_x_fp8, q_x_scale, q_tma, q_graph)
        cuda_wk_proj_gemm_only_out(hidden, k_weights, module, k_x_fp8, k_x_scale, k_tma, k_graph)

    for _ in range(3):
        run_graph_path()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_graph_path()

    new_q_a = (torch.randn_like(q_a) * 0.1).contiguous()
    new_hidden = (torch.randn_like(hidden) * 0.1).contiguous()
    q_a.copy_(new_q_a)
    hidden.copy_(new_hidden)
    q_expected = cuda_wq_b_proj(q_a, q_weights, module)
    k_expected = cuda_wk_proj_gemm_only(hidden, k_weights, module)

    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(q_graph, q_expected, atol=0, rtol=0)
    torch.testing.assert_close(k_graph, k_expected, atol=0, rtol=0)


def test_dsa_qk_projection_out_captures_on_non_default_stream():
    torch.cuda.set_device(0)
    torch.manual_seed(20260431)
    module = build_module()

    batch_size = 2
    q_rank = 2048
    hidden_size = 6144
    q_out_dim = 4096
    k_out_dim = 128

    q_a = (
        torch.randn(batch_size, q_rank, device="cuda", dtype=torch.bfloat16) * 0.1
    ).contiguous()
    hidden = (
        torch.randn(batch_size, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.1
    ).contiguous()
    wq_b = (
        torch.randn(q_out_dim, q_rank, device="cuda", dtype=torch.bfloat16) * 0.01
    ).contiguous()
    wk = (
        torch.randn(k_out_dim, hidden_size, device="cuda", dtype=torch.bfloat16) * 0.01
    ).contiguous()

    q_weights = FP8WqbWeightsCUDA(wq_b, module)
    k_weights = FP8IndexerWeightsCUDA(wk, module)
    q_x_fp8, q_x_scale, q_tma = make_fp8_activation_scratch(
        batch_size,
        q_rank,
        module,
        device="cuda",
    )
    k_x_fp8, k_x_scale, k_tma = make_fp8_activation_scratch(
        batch_size,
        hidden_size,
        module,
        device="cuda",
    )
    q_graph = torch.empty(batch_size, q_out_dim, device="cuda", dtype=torch.bfloat16)
    k_graph = torch.empty(batch_size, k_out_dim, device="cuda", dtype=torch.bfloat16)

    def run_graph_path():
        cuda_wq_b_proj_out(q_a, q_weights, module, q_x_fp8, q_x_scale, q_tma, q_graph)
        cuda_wk_proj_gemm_only_out(hidden, k_weights, module, k_x_fp8, k_x_scale, k_tma, k_graph)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            run_graph_path()
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(capture_stream):
        with torch.cuda.graph(graph):
            run_graph_path()
    torch.cuda.current_stream().wait_stream(capture_stream)

    q_expected = cuda_wq_b_proj(q_a, q_weights, module)
    k_expected = cuda_wk_proj_gemm_only(hidden, k_weights, module)
    q_graph.zero_()
    k_graph.zero_()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        graph.replay()
    capture_stream.synchronize()

    torch.testing.assert_close(q_graph, q_expected, atol=0, rtol=0)
    torch.testing.assert_close(k_graph, k_expected, atol=0, rtol=0)
