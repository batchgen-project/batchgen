"""The GLM-5 3D MoE CUDA-ops chain must not depend on pre-zeroed buffers.

The decode path no longer memsets ``dispatched_x`` (eager + whole-model graph
segment) or the reduce output. These tests pin the kernel semantics that make
that removal safe, by running the exact eager/graph call chain
(``dispatch_scatter_3d`` -> ``act_quant_3d`` -> fused S1 -> ``act_quant_3d``
-> S3 -> ``reduce_weighted_scatter``) twice — once over zero-filled buffers,
once over NaN-poisoned buffers — and requiring bitwise-identical per-token
outputs:

- the grouped GEMM's per-expert TMA descriptors clip loads and stores to
  ``seqlens[e]`` (rows past the count are hardware-zero-filled on load);
- ``act_quant_3d`` returns at ``token >= count``;
- ``reduce_weighted_scatter`` writes every output element unconditionally.

GPU kernels only; every test skips cleanly without CUDA or the extensions.
"""

import importlib

import pytest
import torch


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


@pytest.fixture(scope="module")
def chain_mods():
    _require_cuda()
    try:
        from batchgen.moe.dispatch_scatter_3d import (
            dispatch_scatter_3d,
            reduce_weighted_scatter,
        )
        from batchgen.moe.grouped_fp8_blockwise_moe import (
            grouped_fp8_blockwise_fused_s1,
            grouped_fp8_blockwise_s3,
        )
    except ImportError as exc:  # noqa: BLE001
        pytest.skip(f"3D MoE wrappers unavailable: {exc}")
        raise
    try:
        ops = importlib.import_module("batchgen_kernels.moe._C_fp8_blockwise_ops")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"FP8 blockwise ops unavailable: {exc}")
        raise
    if not hasattr(ops, "act_quant_3d"):
        pytest.skip("extension predates act_quant_3d")
    # The chain also needs the dispatch/reduce and grouped GEMM extensions;
    # exercise their loaders so a missing build skips instead of failing.
    try:
        from batchgen.moe.dispatch_scatter_3d import require_dispatch_scatter_3d_kernels
        require_dispatch_scatter_3d_kernels()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dispatch_scatter_3d extension unavailable: {exc}")
        raise
    return {
        "dispatch": dispatch_scatter_3d,
        "reduce": reduce_weighted_scatter,
        "fused_s1": grouped_fp8_blockwise_fused_s1,
        "s3": grouped_fp8_blockwise_s3,
        "act_quant_3d": ops.act_quant_3d,
    }


# E=5 with expert 4 kept empty: zero-count experts must also be safe.
_E = 5
_MTP = 128
_H = 256   # hidden (K of S1), multiple of 128
_N = 256   # intermediate, multiple of 128
_G = 40    # global tokens
_K = 8     # top-k (reduce template supports 2/4/8)


def _topk_indices(device):
    """Deterministic assignment: <= 2*G rows per expert (< mtp), expert 4
    empty, odd tokens carry one non-local (-1) slot."""
    idx = torch.empty(_G, _K, dtype=torch.int32)
    for i in range(_G):
        for j in range(_K):
            if j < 4:
                idx[i, j] = i % 4
            else:
                idx[i, j] = (i + j) % 4
        if i % 2 == 1:
            idx[i, 7] = -1
    return idx.to(device)


def _weights(generator, device):
    k_blocks = _H // 128
    n_blocks = _N // 128
    k_pad4 = (k_blocks + 3) // 4 * 4
    n_pad4 = (n_blocks + 3) // 4 * 4

    def w(n, k):
        return (
            torch.randn(_E, n, k, generator=generator)
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
            .to(device)
        )

    def ws(n_b, k_p4):
        return (
            torch.rand(_E, n_b, k_p4, generator=generator)
            .mul_(0.1)
            .add_(0.01)
            .to(device)
        )

    return {
        "gate_w": w(_N, _H), "up_w": w(_N, _H), "down_w": w(_H, _N),
        "gate_ws": ws(n_blocks, k_pad4), "up_ws": ws(n_blocks, k_pad4),
        "down_ws": ws(k_blocks, n_pad4),
    }


def _run_chain(mods, wts, tokens, topk_idx, topk_weight, poison):
    """Run the eager/graph 3D MoE chain; `poison` pre-fills every
    intermediate buffer with NaN instead of zeros."""
    device = tokens.device
    fill = float("nan") if poison else 0.0
    rows = _E * _MTP

    dispatched = torch.full((rows, _H), fill, dtype=torch.bfloat16, device=device)
    expert_counts = torch.empty(_E, dtype=torch.int32, device=device)
    expert_counters = torch.empty(_E, dtype=torch.int32, device=device)
    topk_pos = torch.empty(_G * _K, dtype=torch.int32, device=device)

    expert_counts, topk_pos = mods["dispatch"](
        tokens, topk_idx, dispatched, 0, _E, _MTP,
        expert_counts, expert_counters, topk_pos,
    )

    seqlens = expert_counts[:_E]
    cu_seqlens = torch.arange(0, (_E + 1) * _MTP, _MTP, dtype=torch.int32, device=device)
    avg = max(_MTP // _E, 1)

    x_quant_3d, x_scale_3d = mods["act_quant_3d"](dispatched.view(_E, _MTP, _H), seqlens)
    x_quant = x_quant_3d.view(rows, _H)
    x_scale_t = x_scale_3d.view(rows, -1).t().contiguous()

    s1_out = torch.full((rows, _N), fill, dtype=torch.bfloat16, device=device)
    s1_res = mods["fused_s1"](
        x_quant.view(torch.float8_e4m3fn), x_scale_t,
        wts["gate_w"].view(torch.float8_e4m3fn), wts["up_w"].view(torch.float8_e4m3fn),
        wts["gate_ws"], wts["up_ws"], seqlens, cu_seqlens, avg, output=s1_out,
    )

    inter_quant_3d, inter_scale_3d = mods["act_quant_3d"](
        s1_res.view(_E, _MTP, _N), seqlens)
    inter_quant = inter_quant_3d.view(rows, _N)
    inter_scale_t = inter_scale_3d.view(rows, -1).t().contiguous()

    expert_out = torch.full((rows, _H), fill, dtype=torch.bfloat16, device=device)
    mods["s3"](
        inter_quant.view(torch.float8_e4m3fn), inter_scale_t,
        wts["down_w"].view(torch.float8_e4m3fn), wts["down_ws"],
        seqlens, cu_seqlens, avg, output=expert_out,
    )

    result = torch.full((_G, _H), fill, dtype=torch.bfloat16, device=device)
    mods["reduce"](expert_out, topk_pos, topk_weight, _G, _H, _K, output=result)
    torch.cuda.synchronize()
    return expert_counts.cpu(), result


def test_chain_is_bitwise_identical_without_pre_zero(chain_mods):
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randn(_G, _H, generator=generator).mul_(0.1).to(torch.bfloat16).to(device)
    topk_idx = _topk_indices(device)
    topk_weight = torch.rand(_G, _K, generator=generator).to(device)
    topk_weight = torch.where(topk_idx >= 0, topk_weight, torch.zeros_like(topk_weight))
    wts = _weights(generator, device)

    counts_zero, out_zero = _run_chain(chain_mods, wts, tokens, topk_idx, topk_weight, poison=False)
    counts_nan, out_nan = _run_chain(chain_mods, wts, tokens, topk_idx, topk_weight, poison=True)

    assert torch.equal(counts_zero, counts_nan)
    assert int(counts_zero[4]) == 0  # the empty expert stayed empty
    assert not torch.isnan(out_nan.float()).any(), "NaN leaked from unwritten buffer rows"
    assert torch.equal(out_zero, out_nan), (
        "outputs differ when intermediate buffers start as NaN garbage; "
        "a kernel in the chain is consuming rows past expert_counts"
    )


def test_reduce_overwrites_poisoned_output(chain_mods):
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(11)
    rows = _E * _MTP
    expert_out = torch.randn(rows, _H, generator=generator).to(torch.bfloat16).to(device)
    topk_idx = _topk_indices(device)
    topk_weight = torch.rand(_G, _K, generator=generator).to(device)

    # Build valid positions the way dispatch would (row < count per expert).
    tokens = torch.randn(_G, _H, generator=generator).to(torch.bfloat16).to(device)
    dispatched = torch.zeros(rows, _H, dtype=torch.bfloat16, device=device)
    expert_counts = torch.empty(_E, dtype=torch.int32, device=device)
    expert_counters = torch.empty(_E, dtype=torch.int32, device=device)
    topk_pos = torch.empty(_G * _K, dtype=torch.int32, device=device)
    _, topk_pos = chain_mods["dispatch"](
        tokens, topk_idx, dispatched, 0, _E, _MTP,
        expert_counts, expert_counters, topk_pos,
    )

    out_zero = torch.zeros(_G, _H, dtype=torch.bfloat16, device=device)
    chain_mods["reduce"](expert_out, topk_pos, topk_weight, _G, _H, _K, output=out_zero)
    out_nan = torch.full((_G, _H), float("nan"), dtype=torch.bfloat16, device=device)
    chain_mods["reduce"](expert_out, topk_pos, topk_weight, _G, _H, _K, output=out_nan)
    torch.cuda.synchronize()

    assert not torch.isnan(out_nan.float()).any()
    assert torch.equal(out_zero, out_nan)
