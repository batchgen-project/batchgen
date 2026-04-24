"""Unit test: confirm the hoisted-TMA-descriptor path in
`cuda_wk_proj_gemm_only` produces bit-exact output vs the fresh-alloc path,
and that the cached activation TMA descriptor is reused across calls
(no new 128 B HtoD per step).

Fresh-alloc reference is inlined here so the test does not depend on any
prior revision of the file.

Run (on H20): `pytest tests/kernels/test_wk_proj_tma_hoist.py -x -s`
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="requires Hopper (sm_90a)",
)


def _fresh_alloc_reference(hidden_states, weights, module):
    """Old path: fresh x_fp8 alloc + create_tma_desc per call."""
    hidden_states = hidden_states.contiguous()
    B, K = hidden_states.shape
    N = weights.N
    x_fp8 = torch.empty(B, K, dtype=torch.float8_e4m3fn, device=hidden_states.device)
    x_scale = torch.empty(B, dtype=torch.float32, device=hidden_states.device)
    module.run_act_quant(hidden_states, x_fp8, x_scale)
    B_padded = max(B, 64)
    if B < 64:
        x_fp8_padded = torch.zeros(B_padded, K, dtype=torch.float8_e4m3fn, device=x_fp8.device)
        x_fp8_padded[:B] = x_fp8
        x_fp8 = x_fp8_padded
    a_tma_desc = module.create_tma_desc(x_fp8, B_padded, K, 64, 128)
    return module.indexer_kv_proj_gemm_only(
        a_tma_desc, weights.tma_desc,
        weights.w_scale, x_scale,
        B, N, K,
    )


@pytest.fixture(scope="module")
def _module_and_weights():
    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
        build_module,
        FP8IndexerWeightsCUDA,
    )
    torch.manual_seed(0)
    device = torch.device("cuda")
    N, K = 128, 7168
    wk = (torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.1)
    module = build_module()
    weights = FP8IndexerWeightsCUDA(wk, module)
    return module, weights


@pytest.mark.parametrize("B", [1, 8, 32, 64, 65, 128])
def test_hoisted_matches_fresh_alloc(_module_and_weights, B):
    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
        cuda_wk_proj_gemm_only,
    )
    module, weights = _module_and_weights
    torch.manual_seed(B)
    hidden = torch.randn(B, weights.K, dtype=torch.bfloat16, device="cuda") * 0.5

    out_ref = _fresh_alloc_reference(hidden, weights, module)
    out_new = cuda_wk_proj_gemm_only(hidden, weights, module)
    torch.cuda.synchronize()

    # Both paths run the same GEMM kernel on the same inputs; act_quant is
    # deterministic. Expect bit-exact match.
    assert torch.equal(out_ref, out_new), (
        f"B={B}: max abs diff = {(out_ref - out_new).abs().max().item()}"
    )


def test_desc_is_reused_across_calls(_module_and_weights):
    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
        cuda_wk_proj_gemm_only,
    )
    module, weights = _module_and_weights

    # Force fresh weights so the buffer is unallocated at the start.
    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
        FP8IndexerWeightsCUDA,
    )
    wk = weights.w_fp8.to(torch.float32).to(torch.bfloat16)  # round-trip placeholder; only pointers matter
    w2 = FP8IndexerWeightsCUDA.__new__(FP8IndexerWeightsCUDA)
    # Reuse the already-quantized weights; only exercise _ensure_act_buf.
    w2.w_fp8 = weights.w_fp8
    w2.w_scale = weights.w_scale
    w2.block_k = weights.block_k
    w2.N = weights.N
    w2.K = weights.K
    w2.tma_desc = weights.tma_desc
    w2._act_buf_B_padded = 0
    w2.x_fp8_buf = None
    w2.x_scale_buf = None
    w2.a_tma_desc = None

    hidden = torch.randn(64, w2.K, dtype=torch.bfloat16, device="cuda") * 0.5

    _ = cuda_wk_proj_gemm_only(hidden, w2, module)
    first_desc_ptr = w2.a_tma_desc.data_ptr()
    first_buf_ptr = w2.x_fp8_buf.data_ptr()

    # Call again at various smaller / equal B; descriptor + buffer must be reused.
    for B in [1, 8, 32, 64]:
        h = torch.randn(B, w2.K, dtype=torch.bfloat16, device="cuda") * 0.5
        _ = cuda_wk_proj_gemm_only(h, w2, module)
        assert w2.a_tma_desc.data_ptr() == first_desc_ptr, \
            f"TMA desc rebuilt at B={B} — hoist failed"
        assert w2.x_fp8_buf.data_ptr() == first_buf_ptr, \
            f"x_fp8 buf reallocated at B={B} — hoist failed"


def test_grow_when_B_padded_exceeds_cached(_module_and_weights):
    from batchgen_kernels.attention.dsa.fused_indexer_kv_proj_cuda import (
        cuda_wk_proj_gemm_only,
        FP8IndexerWeightsCUDA,
    )
    module, weights = _module_and_weights
    w2 = FP8IndexerWeightsCUDA.__new__(FP8IndexerWeightsCUDA)
    w2.w_fp8 = weights.w_fp8
    w2.w_scale = weights.w_scale
    w2.block_k = weights.block_k
    w2.N = weights.N
    w2.K = weights.K
    w2.tma_desc = weights.tma_desc
    w2._act_buf_B_padded = 0
    w2.x_fp8_buf = None
    w2.x_scale_buf = None
    w2.a_tma_desc = None

    _ = cuda_wk_proj_gemm_only(torch.randn(64, w2.K, dtype=torch.bfloat16, device="cuda"), w2, module)
    first = w2.a_tma_desc.data_ptr()

    # Larger B forces grow → new descriptor expected.
    _ = cuda_wk_proj_gemm_only(torch.randn(128, w2.K, dtype=torch.bfloat16, device="cuda"), w2, module)
    assert w2.a_tma_desc.data_ptr() != first, "expected TMA desc rebuild when B_padded grows"
    assert w2._act_buf_B_padded == 128
