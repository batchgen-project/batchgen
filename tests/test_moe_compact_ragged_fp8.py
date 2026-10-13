"""Compact ragged MoE primitives: dispatch, FP8 quant, and grouped GEMM layout.

Covers batchgen_kernels >= 0.4.2:
  * ``dispatch_scatter_ragged`` counts, 64-aligned ``cu_seqlens``, and row mapping.
  * ``act_quant_ragged`` byte/scale parity with padded ``act_quant_3d`` on valid
    rows (zero-token experts, counts 1/63/64/65), zero-input pad rows, and
    untouched rows outside every span.
  * Grouped GEMM / fused S1 ragged-vs-uniform parity across TileM selection
    boundaries, plus an independent FP32 dequant reference so a shared x_scale
    addressing bug cannot pass as parity.
  * Host boundary checks reject malformed inputs before any launch.
  * The device alignment trap, isolated in a subprocess.

GPU kernels only; every test skips cleanly without CUDA or the extensions.
"""

import importlib
import subprocess
import sys
import textwrap

import pytest
import torch

ALIGN = 64
BLOCK = 128
COUNTS = [0, 1, 63, 64, 65, 2, 0]


def _align64(v: int) -> int:
    return (v + ALIGN - 1) // ALIGN * ALIGN


def _ragged_cu(counts):
    cu = [0]
    for c in counts:
        cu.append(cu[-1] + _align64(c))
    return cu


def _pad4(v: int) -> int:
    return (v + 3) // 4 * 4


def _ragged_capacity(num_tokens: int, top_k: int, num_experts: int) -> int:
    nk = num_tokens * top_k
    return (nk + 63 * min(num_experts, nk)) // 64 * 64


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _import_ext(name: str, required_symbol: str):
    _require_cuda()
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - any load failure means unavailable
        pytest.skip(f"{name} unavailable: {exc}")
    if not hasattr(mod, required_symbol):
        pytest.skip(f"{name} predates batchgen_kernels 0.4.2 (no {required_symbol})")
    return mod


@pytest.fixture(scope="module")
def dispatch_mod():
    _require_cuda()
    try:
        from batchgen.moe.dispatch_scatter_3d import _load_dispatch_reduce_module
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"batchgen.moe.dispatch_scatter_3d unavailable: {exc}")
    mod = _load_dispatch_reduce_module()
    if mod is None:
        pytest.skip("dispatch_scatter_3d extension unavailable")
    if not hasattr(mod, "dispatch_scatter_ragged"):
        pytest.skip("dispatch_scatter_3d extension predates batchgen_kernels 0.4.2")
    return mod


@pytest.fixture(scope="module")
def ops_mod():
    return _import_ext("batchgen_kernels.moe._C_fp8_blockwise_ops", "act_quant_ragged")


@pytest.fixture(scope="module")
def gemm_mod():
    return _import_ext("batchgen_kernels.moe._C_fp8_blockwise_gemm", "fp8_blockwise_fused_s1")


# ──────────────────────────────────────────────────────────────────────────────
# K3 API preservation
# ──────────────────────────────────────────────────────────────────────────────

def test_dispatch_wrapper_symbols_preserved():
    try:
        from batchgen.moe import dispatch_scatter_3d as wrapper
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"batchgen.moe.dispatch_scatter_3d unavailable: {exc}")
    for name in (
        "dispatch_scatter_3d",
        "reduce_weighted_scatter",
        "reduce_weighted_scatter_fp32",
        "dispatch_scatter_ragged",
    ):
        assert callable(getattr(wrapper, name, None)), name


def test_dispatch_extension_symbols_preserved(dispatch_mod):
    for name in (
        "dispatch_scatter_3d",
        "reduce_weighted_scatter",
        "reduce_weighted_scatter_fp32",
        "dispatch_scatter_ragged",
    ):
        assert callable(getattr(dispatch_mod, name, None)), name


def test_reduce_fp32_k16_consumes_ragged_positions(dispatch_mod):
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(3)
    n, h, k, rows = 5, 37, 16, 200
    expert_out = torch.randn(rows, h, generator=gen).to(torch.bfloat16).to(dev)
    pos = torch.randint(-1, rows, (n * k,), generator=gen, dtype=torch.int32).to(dev)
    weights = torch.rand(n, k, generator=gen).to(dev)
    out = torch.empty(n, h, dtype=torch.float32, device=dev)

    dispatch_mod.reduce_weighted_scatter_fp32(expert_out, pos, weights, n, h, k, out)
    torch.cuda.synchronize()

    vals = expert_out.float()[pos.clamp(min=0).long()].view(n, k, h)
    mask = (pos >= 0).view(n, k, 1).float()
    ref = (vals * weights.view(n, k, 1) * mask).sum(dim=1)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# dispatch_scatter_ragged
# ──────────────────────────────────────────────────────────────────────────────

def _make_topk(counts, expert_start, top_k, extra_tokens, gen):
    local = []
    for e, c in enumerate(counts):
        local += [expert_start + e] * c
    n_tokens = (len(local) + top_k - 1) // top_k + extra_tokens
    total = n_tokens * top_k
    non_local = [-1, 0, expert_start - 1, expert_start + len(counts), 1_000_000]
    fill = [non_local[i % len(non_local)] for i in range(total - len(local))]
    flat = torch.tensor(local + fill, dtype=torch.int32)
    flat = flat[torch.randperm(total, generator=gen)]
    return flat.view(n_tokens, top_k)


@pytest.mark.parametrize("hidden", [128, 131])
def test_dispatch_scatter_ragged_counts_prefix_and_rows(dispatch_mod, hidden):
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)
    counts = COUNTS
    num_experts, expert_start, top_k = len(counts), 10, 4

    topk = _make_topk(counts, expert_start, top_k, extra_tokens=3, gen=gen).to(dev)
    n_tokens = topk.size(0)
    x = torch.randn(n_tokens, hidden, generator=gen).to(torch.bfloat16).to(dev)

    capacity = _ragged_capacity(n_tokens, top_k, num_experts)
    expected_cu = _ragged_cu(counts)
    assert capacity % ALIGN == 0 and capacity >= expected_cu[-1]

    act = torch.zeros(capacity, hidden, dtype=torch.bfloat16, device=dev)
    expert_counts = torch.full((num_experts,), -5, dtype=torch.int32, device=dev)
    expert_counters = torch.full((num_experts,), -5, dtype=torch.int32, device=dev)
    topk_pos = torch.full((n_tokens * top_k,), -7, dtype=torch.int32, device=dev)
    cu_seqlens = torch.full((num_experts + 1,), -9, dtype=torch.int32, device=dev)

    ret = dispatch_mod.dispatch_scatter_ragged(
        x, topk, act, expert_start, num_experts,
        expert_counts, expert_counters, topk_pos, cu_seqlens,
    )
    torch.cuda.synchronize()

    assert ret[0].data_ptr() == expert_counts.data_ptr()
    assert ret[1].data_ptr() == cu_seqlens.data_ptr()
    assert ret[2].data_ptr() == topk_pos.data_ptr()
    assert expert_counts.cpu().tolist() == counts
    assert cu_seqlens.cpu().tolist() == expected_cu

    flat = topk.view(-1).cpu()
    pos = topk_pos.cpu()
    local = (flat >= expert_start) & (flat < expert_start + num_experts)
    assert bool((pos[~local] == -1).all())
    for e, c in enumerate(counts):
        rows = sorted(pos[flat == expert_start + e].tolist())
        assert rows == list(range(expected_cu[e], expected_cu[e] + c)), e

    token_ids = (torch.arange(n_tokens * top_k) // top_k)[local].to(dev)
    assert torch.equal(act[pos[local].long().to(dev)], x[token_ids])


def test_dispatch_scatter_ragged_empty_input(dispatch_mod):
    dev = torch.device("cuda")
    num_experts, top_k, hidden = 7, 4, 131
    x = torch.empty(0, hidden, dtype=torch.bfloat16, device=dev)
    topk = torch.empty(0, top_k, dtype=torch.int32, device=dev)
    act = torch.empty(0, hidden, dtype=torch.bfloat16, device=dev)
    counts = torch.full((num_experts,), -1, dtype=torch.int32, device=dev)
    counters = torch.full_like(counts, -1)
    positions = torch.empty(0, dtype=torch.int32, device=dev)
    cu = torch.full((num_experts + 1,), -1, dtype=torch.int32, device=dev)

    dispatch_mod.dispatch_scatter_ragged(
        x, topk, act, 10, num_experts, counts, counters, positions, cu,
    )
    torch.cuda.synchronize()
    assert counts.cpu().tolist() == [0] * num_experts
    assert counters.cpu().tolist() == [0] * num_experts
    assert cu.cpu().tolist() == [0] * (num_experts + 1)


def test_dispatch_scatter_ragged_tight_capacity(dispatch_mod):
    dev = torch.device("cuda")
    num_experts, expert_start, hidden = 5, 20, 131
    topk = torch.arange(
        expert_start, expert_start + num_experts, dtype=torch.int32, device=dev,
    ).view(num_experts, 1)
    x = torch.arange(num_experts * hidden, dtype=torch.float32, device=dev).view(
        num_experts, hidden,
    ).to(torch.bfloat16)
    capacity = _ragged_capacity(num_experts, 1, num_experts)
    assert capacity == num_experts * ALIGN
    act = torch.full((capacity, hidden), -1.0, dtype=torch.bfloat16, device=dev)
    counts = torch.empty(num_experts, dtype=torch.int32, device=dev)
    counters = torch.empty_like(counts)
    positions = torch.empty(num_experts, dtype=torch.int32, device=dev)
    cu = torch.empty(num_experts + 1, dtype=torch.int32, device=dev)

    dispatch_mod.dispatch_scatter_ragged(
        x, topk, act, expert_start, num_experts,
        counts, counters, positions, cu,
    )
    torch.cuda.synchronize()
    expected_cu = list(range(0, capacity + 1, ALIGN))
    assert cu.cpu().tolist() == expected_cu
    assert positions.cpu().tolist() == expected_cu[:-1]
    assert torch.equal(act[positions.long()], x)


# ──────────────────────────────────────────────────────────────────────────────
# act_quant_ragged
# ──────────────────────────────────────────────────────────────────────────────

def _to_ragged(x3, counts, cu, max_rows, pad_value):
    k = x3.size(-1)
    out = torch.full((max_rows, k), pad_value, dtype=x3.dtype, device=x3.device)
    for e, c in enumerate(counts):
        out[cu[e]:cu[e] + c] = x3[e, :c]
    return out


@pytest.mark.parametrize("k", [256, 300, 6144])
def test_act_quant_ragged_matches_act_quant_3d(ops_mod, k):
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(1)
    counts = COUNTS
    num_experts, mtp = len(counts), 128
    num_blocks = (k + BLOCK - 1) // BLOCK

    row_gain = torch.empty(num_experts, mtp, 1).uniform_(0.05, 20.0, generator=gen)
    x3 = (torch.randn(num_experts, mtp, k, generator=gen) * row_gain).to(torch.bfloat16).to(dev)
    seqlens = torch.tensor(counts, dtype=torch.int32, device=dev)
    y3, s3 = ops_mod.act_quant_3d(x3, seqlens)

    cu = _ragged_cu(counts)
    max_rows = cu[-1] + ALIGN + 5  # tail rows outside every expert span
    # Pad rows hold inf: reading them would poison the scale and bytes.
    x_rag = _to_ragged(x3, counts, cu, max_rows, float("inf"))
    cu_t = torch.tensor(cu, dtype=torch.int32, device=dev)
    y = torch.full((max_rows, k), 0xAB, dtype=torch.uint8, device=dev)
    scale = torch.full((num_blocks, max_rows), -7.0, dtype=torch.float32, device=dev)

    ret = ops_mod.act_quant_ragged(x_rag, seqlens, cu_t, y, scale)
    torch.cuda.synchronize()
    assert ret[0].data_ptr() == y.data_ptr() and ret[1].data_ptr() == scale.data_ptr()

    zero_x = torch.zeros(1, 1, k, dtype=torch.bfloat16, device=dev)
    _, zero_s = ops_mod.act_quant_3d(zero_x, torch.ones(1, dtype=torch.int32, device=dev))
    zero_col = zero_s[0, 0].view(num_blocks, 1)

    for e, c in enumerate(counts):
        if c:
            assert torch.equal(y[cu[e]:cu[e] + c], y3[e, :c]), e
            assert torch.equal(scale[:, cu[e]:cu[e] + c], s3[e, :c].t()), e
        pad = slice(cu[e] + c, cu[e] + _align64(c))
        n_pad = pad.stop - pad.start
        if n_pad:
            assert bool((y[pad] == 0).all()), e
            assert torch.equal(scale[:, pad], zero_col.expand(num_blocks, n_pad)), e

    assert bool((y[cu[-1]:] == 0xAB).all())
    assert bool((scale[:, cu[-1]:] == -7.0).all())


# ──────────────────────────────────────────────────────────────────────────────
# Grouped GEMM / fused S1: ragged vs uniform
# ──────────────────────────────────────────────────────────────────────────────

def _fp8_weights(num_experts, n, k, gen, dev):
    w = (torch.randn(num_experts, n, k, generator=gen) * 2.0).clamp(-6.0, 6.0)
    ws = torch.empty(num_experts, n // BLOCK, _pad4(k // BLOCK)).uniform_(0.01, 0.02, generator=gen)
    return w.to(torch.float8_e4m3fn).to(dev), ws.to(dev)


def _dequant_rows(w, ws, e, x_q, x_s):
    """FP32 reference: (W_e * ws_e) @ (x_q * x_s) for the given rows."""
    n, k = w.shape[1], w.shape[2]
    w_scale = ws[e, :, :k // BLOCK].repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)
    w_dq = w[e].float() * w_scale[:n, :k]
    x_dq = x_q.float() * x_s.t().repeat_interleave(BLOCK, 1)[:, :k]
    return x_dq @ w_dq.t()


@pytest.mark.parametrize("fused_s1", [False, True])
@pytest.mark.parametrize("avg", [16, 17, 32, 33, 64])
@pytest.mark.parametrize("dims", [(256, 128), (1280, 1152)])
def test_grouped_gemm_ragged_matches_uniform(ops_mod, gemm_mod, dims, avg, fused_s1):
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(2)
    k, n = dims
    counts = COUNTS
    num_experts, mtp = len(counts), 128

    row_gain = torch.empty(num_experts, mtp, 1).uniform_(0.25, 4.0, generator=gen)
    x3 = (torch.randn(num_experts, mtp, k, generator=gen) * row_gain).to(torch.bfloat16).to(dev)
    seqlens = torch.tensor(counts, dtype=torch.int32, device=dev)

    # Uniform layout (pre-existing contract).
    y3, s3 = ops_mod.act_quant_3d(x3, seqlens)
    xq_u = y3.view(num_experts * mtp, k).view(torch.float8_e4m3fn)
    xs_u = s3.view(num_experts * mtp, -1).t().contiguous()
    cu_u = torch.arange(0, (num_experts + 1) * mtp, mtp, dtype=torch.int32, device=dev)

    # Compact ragged layout; m_pad leaves one 64-row tail beyond cu[E].
    cu = _ragged_cu(counts)
    rows = cu[-1] + ALIGN
    x_rag = _to_ragged(x3, counts, cu, rows, 0.0)
    cu_r = torch.tensor(cu, dtype=torch.int32, device=dev)
    xq_r = torch.empty(rows, k, dtype=torch.float8_e4m3fn, device=dev)
    xs_r = torch.empty(k // BLOCK, rows, dtype=torch.float32, device=dev)
    ops_mod.act_quant_ragged(x_rag, seqlens, cu_r, xq_r, xs_r)

    out_u = torch.empty(num_experts * mtp, n, dtype=torch.bfloat16, device=dev)
    out_r = torch.empty(rows, n, dtype=torch.bfloat16, device=dev)
    if fused_s1:
        wg, wsg = _fp8_weights(num_experts, n, k, gen, dev)
        wu, wsu = _fp8_weights(num_experts, n, k, gen, dev)
        gemm_mod.fp8_blockwise_fused_s1(xq_u, wg, wu, seqlens, cu_u, xs_u, wsg, wsu, avg, out_u)
        gemm_mod.fp8_blockwise_fused_s1(xq_r, wg, wu, seqlens, cu_r, xs_r, wsg, wsu, avg, out_r)
    else:
        w, ws = _fp8_weights(num_experts, n, k, gen, dev)
        gemm_mod.fp8_blockwise_grouped_gemm(xq_u, w, seqlens, cu_u, xs_u, ws, avg, out_u)
        gemm_mod.fp8_blockwise_grouped_gemm(xq_r, w, seqlens, cu_r, xs_r, ws, avg, out_r)
    torch.cuda.synchronize()

    for e, c in enumerate(counts):
        if not c:
            continue
        got = out_r[cu[e]:cu[e] + c]
        assert torch.equal(got, out_u[e * mtp:e * mtp + c]), (e, avg)

        x_q, x_s = xq_r[cu[e]:cu[e] + c], xs_r[:, cu[e]:cu[e] + c]
        if fused_s1:
            gate = _dequant_rows(wg, wsg, e, x_q, x_s)
            up = _dequant_rows(wu, wsu, e, x_q, x_s)
            ref = torch.nn.functional.silu(gate) * up
            torch.testing.assert_close(got.float(), ref, rtol=5e-2, atol=2e-2)
        else:
            ref = _dequant_rows(w, ws, e, x_q, x_s)
            torch.testing.assert_close(got.float(), ref, rtol=2e-2, atol=5e-3)


def test_compact_ragged_pipeline_cuda_graph_replay(dispatch_mod, ops_mod, gemm_mod):
    dev = torch.device("cuda")
    rows, k, n, num_experts = 128, 256, 128, 2
    topk = torch.cat((
        torch.zeros(1, dtype=torch.int32, device=dev),
        torch.ones(63, dtype=torch.int32, device=dev),
    )).view(64, 1)
    x = torch.randn(64, k, dtype=torch.bfloat16, device=dev)
    act = torch.empty(rows, k, dtype=torch.bfloat16, device=dev)
    counts = torch.empty(num_experts, dtype=torch.int32, device=dev)
    counters = torch.empty_like(counts)
    positions = torch.empty(64, dtype=torch.int32, device=dev)
    cu = torch.empty(num_experts + 1, dtype=torch.int32, device=dev)
    xq = torch.empty(rows, k, dtype=torch.float8_e4m3fn, device=dev)
    xs = torch.empty(k // BLOCK, rows, dtype=torch.float32, device=dev)
    weight = torch.full((num_experts, n, k), 0.25, device=dev).to(torch.float8_e4m3fn)
    ws = torch.full((num_experts, n // BLOCK, 4), 0.01, device=dev)
    out = torch.empty(rows, n, dtype=torch.bfloat16, device=dev)

    def run_pipeline():
        dispatch_mod.dispatch_scatter_ragged(
            x, topk, act, 0, num_experts,
            counts, counters, positions, cu,
        )
        ops_mod.act_quant_ragged(act, counts, cu, xq, xs)
        gemm_mod.fp8_blockwise_grouped_gemm(
            xq, weight, counts, cu, xs, ws, 32, out,
        )

    run_pipeline()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_pipeline()

    graph.replay()
    torch.cuda.synchronize()
    assert counts.cpu().tolist() == [1, 63]
    assert cu.cpu().tolist() == [0, 64, 128]
    valid_positions = positions.long()
    assert bool(torch.isfinite(out[valid_positions]).all())
    assert bool((out[valid_positions] != 0).any())

    x.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out[valid_positions], torch.zeros_like(out[valid_positions]))


# ──────────────────────────────────────────────────────────────────────────────
# Malformed host inputs
# ──────────────────────────────────────────────────────────────────────────────

def _small_dispatch_args(dev):
    n_tokens, top_k, hidden, num_experts = 6, 2, 16, 3
    return dict(
        x=torch.zeros(n_tokens, hidden, dtype=torch.bfloat16, device=dev),
        topk_indices=torch.zeros(n_tokens, top_k, dtype=torch.int32, device=dev),
        act_buffer=torch.zeros(64 * 3, hidden, dtype=torch.bfloat16, device=dev),
        expert_start=0,
        num_local_experts=num_experts,
        expert_counts=torch.zeros(num_experts, dtype=torch.int32, device=dev),
        expert_counters=torch.zeros(num_experts, dtype=torch.int32, device=dev),
        topk_pos=torch.zeros(n_tokens * top_k, dtype=torch.int32, device=dev),
        cu_seqlens=torch.zeros(num_experts + 1, dtype=torch.int32, device=dev),
    )


_DISPATCH_BAD = {
    "x_float32": lambda a: a.update(x=a["x"].float()),
    "x_cpu": lambda a: a.update(x=a["x"].cpu()),
    "topk_int64": lambda a: a.update(topk_indices=a["topk_indices"].long()),
    "topk_noncontiguous": lambda a: a.update(
        topk_indices=torch.zeros(2, 6, dtype=torch.int32, device="cuda").t()),
    "act_hidden_mismatch": lambda a: a.update(act_buffer=a["act_buffer"][:, :8].contiguous()),
    "act_capacity_short": lambda a: a.update(act_buffer=a["act_buffer"][:12]),
    "counts_cpu": lambda a: a.update(expert_counts=a["expert_counts"].cpu()),
    "counts_size": lambda a: a.update(expert_counts=a["expert_counts"][:2]),
    "cu_size": lambda a: a.update(cu_seqlens=a["cu_seqlens"][:3]),
    "topk_pos_size": lambda a: a.update(topk_pos=a["topk_pos"][:-1]),
    "expert_start_negative": lambda a: a.update(expert_start=-1),
    "num_experts_zero": lambda a: a.update(num_local_experts=0),
}


@pytest.mark.parametrize("case", sorted(_DISPATCH_BAD))
def test_dispatch_scatter_ragged_rejects_malformed(dispatch_mod, case):
    args = _small_dispatch_args(torch.device("cuda"))
    _DISPATCH_BAD[case](args)
    with pytest.raises(RuntimeError):
        dispatch_mod.dispatch_scatter_ragged(
            args["x"], args["topk_indices"], args["act_buffer"],
            args["expert_start"], args["num_local_experts"],
            args["expert_counts"], args["expert_counters"],
            args["topk_pos"], args["cu_seqlens"],
        )
    torch.cuda.synchronize()


def _small_quant_args(dev):
    max_rows, k, num_experts = 128, 300, 2
    return dict(
        x=torch.zeros(max_rows, k, dtype=torch.bfloat16, device=dev),
        seqlens=torch.tensor([3, 1], dtype=torch.int32, device=dev),
        cu_seqlens=torch.tensor([0, 64, 128], dtype=torch.int32, device=dev),
        y=torch.zeros(max_rows, k, dtype=torch.uint8, device=dev),
        scale=torch.zeros(3, max_rows, dtype=torch.float32, device=dev),
    )


_QUANT_BAD = {
    "x_float16": lambda a: a.update(x=a["x"].half()),
    "x_cpu": lambda a: a.update(x=a["x"].cpu()),
    "x_noncontiguous": lambda a: a.update(
        x=torch.zeros(300, 128, dtype=torch.bfloat16, device="cuda").t()),
    "seqlens_int64": lambda a: a.update(seqlens=a["seqlens"].long()),
    "seqlens_empty": lambda a: a.update(seqlens=a["seqlens"][:0]),
    "cu_size": lambda a: a.update(cu_seqlens=a["cu_seqlens"][:2]),
    "cu_cpu": lambda a: a.update(cu_seqlens=a["cu_seqlens"].cpu()),
    "y_float16": lambda a: a.update(y=a["y"].half()),
    "y_rows": lambda a: a.update(y=a["y"][:64]),
    "scale_transposed": lambda a: a.update(scale=a["scale"].t().contiguous()),
    "scale_blocks": lambda a: a.update(scale=a["scale"][:2]),
    "scale_rows": lambda a: a.update(scale=a["scale"][:, :64].contiguous()),
}


@pytest.mark.parametrize("case", sorted(_QUANT_BAD))
def test_act_quant_ragged_rejects_malformed(ops_mod, case):
    args = _small_quant_args(torch.device("cuda"))
    _QUANT_BAD[case](args)
    with pytest.raises(RuntimeError):
        ops_mod.act_quant_ragged(
            args["x"], args["seqlens"], args["cu_seqlens"], args["y"], args["scale"])
    torch.cuda.synchronize()


def _small_gemm_args(dev):
    num_experts, k, n, rows = 2, 256, 128, 128
    fp8 = torch.float8_e4m3fn
    return dict(
        x=torch.zeros(rows, k, dtype=fp8, device=dev),
        weight=torch.zeros(num_experts, n, k, dtype=fp8, device=dev),
        seqlens=torch.tensor([3, 1], dtype=torch.int32, device=dev),
        cu_seqlens=torch.tensor([0, 64, 128], dtype=torch.int32, device=dev),
        x_scale=torch.ones(k // BLOCK, rows, dtype=torch.float32, device=dev),
        w_scale=torch.ones(num_experts, n // BLOCK, _pad4(k // BLOCK), dtype=torch.float32, device=dev),
        avg=64,
        output=torch.empty(rows, n, dtype=torch.bfloat16, device=dev),
    )


_GEMM_BAD = {
    "x_uint8": lambda a: a.update(x=a["x"].view(torch.uint8)),
    "x_cpu": lambda a: a.update(x=a["x"].cpu()),
    "x_noncontiguous": lambda a: a.update(
        x=torch.zeros(256, 128, dtype=torch.float8_e4m3fn, device="cuda").t()),
    "weight_k_mismatch": lambda a: a.update(weight=a["weight"][:, :, :128].contiguous()),
    "k_not_block_multiple": lambda a: a.update(
        x=a["x"][:, :200].contiguous(), weight=a["weight"][:, :, :200].contiguous(),
        x_scale=a["x_scale"][:1].contiguous()),
    "seqlens_int64": lambda a: a.update(seqlens=a["seqlens"].long()),
    "seqlens_size": lambda a: a.update(seqlens=a["seqlens"][:1]),
    "cu_size": lambda a: a.update(cu_seqlens=a["cu_seqlens"][:2]),
    "cu_cpu": lambda a: a.update(cu_seqlens=a["cu_seqlens"].cpu()),
    "x_scale_blocks": lambda a: a.update(x_scale=a["x_scale"][:1].contiguous()),
    "x_scale_exceeds_rows": lambda a: a.update(
        x_scale=torch.ones(2, 192, dtype=torch.float32, device="cuda")),
    "x_scale_not_tile_aligned": lambda a: a.update(x_scale=a["x_scale"][:, :100].contiguous()),
    "w_scale_not_pad4": lambda a: a.update(
        w_scale=torch.ones(2, 1, 8, dtype=torch.float32, device="cuda")),
    "w_scale_groups": lambda a: a.update(w_scale=a["w_scale"][:1]),
    "output_shape": lambda a: a.update(output=a["output"][:64]),
    "output_dtype": lambda a: a.update(output=a["output"].float()),
}


def _call_gemm(gemm_mod, a, fused):
    if fused:
        return gemm_mod.fp8_blockwise_fused_s1(
            a["x"], a["weight"], a.get("up_weight", a["weight"]), a["seqlens"],
            a["cu_seqlens"], a["x_scale"], a["w_scale"], a.get("up_w_scale", a["w_scale"]),
            a["avg"], a["output"])
    return gemm_mod.fp8_blockwise_grouped_gemm(
        a["x"], a["weight"], a["seqlens"], a["cu_seqlens"], a["x_scale"], a["w_scale"],
        a["avg"], a["output"], a.get("tma_desc"))


@pytest.mark.parametrize("fused_s1", [False, True])
@pytest.mark.parametrize("case", sorted(_GEMM_BAD))
def test_grouped_gemm_rejects_malformed(gemm_mod, case, fused_s1):
    args = _small_gemm_args(torch.device("cuda"))
    _GEMM_BAD[case](args)
    with pytest.raises(RuntimeError):
        _call_gemm(gemm_mod, args, fused_s1)
    torch.cuda.synchronize()


@pytest.mark.parametrize("case", ["up_weight_shape", "up_w_scale_shape"])
def test_fused_s1_rejects_mismatched_up(gemm_mod, case):
    args = _small_gemm_args(torch.device("cuda"))
    if case == "up_weight_shape":
        args["up_weight"] = torch.zeros(3, 128, 256, dtype=torch.float8_e4m3fn, device="cuda")
    else:
        args["up_w_scale"] = torch.ones(2, 2, 4, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError):
        _call_gemm(gemm_mod, args, fused=True)
    torch.cuda.synchronize()


def test_grouped_gemm_rejects_malformed_tma_desc(gemm_mod):
    args = _small_gemm_args(torch.device("cuda"))
    args["tma_desc"] = torch.empty(2, 128, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError):
        _call_gemm(gemm_mod, args, fused=False)
    torch.cuda.synchronize()


def test_grouped_gemm_refreshes_caller_tma_scratch(gemm_mod):
    args = _small_gemm_args(torch.device("cuda"))
    args["tma_desc"] = torch.zeros(4, 128, dtype=torch.uint8, device="cuda")
    _call_gemm(gemm_mod, args, fused=False)
    torch.cuda.synchronize()
    assert int(torch.count_nonzero(args["tma_desc"]).item()) > 0


def test_grouped_gemm_rejects_misaligned_tma_scratch(gemm_mod):
    args = _small_gemm_args(torch.device("cuda"))
    storage = torch.empty(4 * 128 + 1, dtype=torch.uint8, device="cuda")
    args["tma_desc"] = storage[1:].view(4, 128)
    assert args["tma_desc"].is_contiguous()
    with pytest.raises(RuntimeError, match="64-byte aligned"):
        _call_gemm(gemm_mod, args, fused=False)


@pytest.mark.parametrize("fused_s1", [False, True])
def test_grouped_gemm_rejects_more_than_256_groups(gemm_mod, fused_s1):
    dev = torch.device("cuda")
    groups, rows, k, n = 257, 64, 128, 128
    args = dict(
        x=torch.empty(rows, k, dtype=torch.float8_e4m3fn, device=dev),
        weight=torch.empty(groups, n, k, dtype=torch.float8_e4m3fn, device=dev),
        seqlens=torch.zeros(groups, dtype=torch.int32, device=dev),
        cu_seqlens=torch.zeros(groups + 1, dtype=torch.int32, device=dev),
        x_scale=torch.empty(k // BLOCK, rows, dtype=torch.float32, device=dev),
        w_scale=torch.empty(groups, n // BLOCK, 4, dtype=torch.float32, device=dev),
        avg=16,
        output=torch.empty(rows, n, dtype=torch.bfloat16, device=dev),
    )
    with pytest.raises(RuntimeError, match="num_group"):
        _call_gemm(gemm_mod, args, fused_s1)


def test_grouped_gemm_rejects_cross_device(gemm_mod):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices")
    args = _small_gemm_args(torch.device("cuda:0"))
    args["cu_seqlens"] = args["cu_seqlens"].to("cuda:1")
    with pytest.raises(RuntimeError):
        _call_gemm(gemm_mod, args, fused=False)


# ──────────────────────────────────────────────────────────────────────────────
# Device alignment trap (isolated: a trap poisons the CUDA context)
# ──────────────────────────────────────────────────────────────────────────────

_TRAP_CHILD = textwrap.dedent(
    """
    import sys
    import torch
    from batchgen_kernels.moe import _C_fp8_blockwise_gemm as gm

    fused = sys.argv[1] == "fused"
    layout = sys.argv[2]
    dev = "cuda"
    k, n, rows = 256, 128, 256
    fp8 = torch.float8_e4m3fn
    x = torch.zeros(rows, k, dtype=fp8, device=dev)
    w = torch.zeros(2, n, k, dtype=fp8, device=dev)
    seqlens = torch.tensor([1, 1], dtype=torch.int32, device=dev)
    cu = torch.tensor([0, 65, 130], dtype=torch.int32, device=dev)
    if layout == "exceeds":
        cu = torch.tensor([0, 256, 256], dtype=torch.int32, device=dev)
    elif layout == "overlap":
        cu = torch.tensor([0, 0, 64], dtype=torch.int32, device=dev)
    elif layout == "final_out_of_bounds":
        cu = torch.tensor([0, 64, 272], dtype=torch.int32, device=dev)
    elif layout == "negative_cu":
        cu = torch.tensor([-16, 64, 128], dtype=torch.int32, device=dev)
    elif layout == "negative_seqlen":
        seqlens = torch.tensor([-1, 1], dtype=torch.int32, device=dev)
    elif layout == "large_negative_seqlen":
        seqlens = torch.tensor([-33, 1], dtype=torch.int32, device=dev)
    xs = torch.ones(k // 128, rows, dtype=torch.float32, device=dev)
    ws = torch.ones(2, n // 128, 4, dtype=torch.float32, device=dev)
    out = torch.empty(rows, n, dtype=torch.bfloat16, device=dev)
    print("READY", flush=True)
    if fused:
        gm.fp8_blockwise_fused_s1(x, w, w, seqlens, cu, xs, ws, ws, 16, out)
    else:
        gm.fp8_blockwise_grouped_gemm(x, w, seqlens, cu, xs, ws, 16, out)
    torch.cuda.synchronize()
    print("NO_TRAP", flush=True)
    """
)


@pytest.mark.parametrize("variant", ["gemm", "fused"])
@pytest.mark.parametrize("layout", [
    "misaligned", "exceeds", "overlap", "final_out_of_bounds",
    "negative_cu", "negative_seqlen", "large_negative_seqlen",
])
def test_grouped_gemm_device_layout_trap(gemm_mod, variant, layout):
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _TRAP_CHILD, variant, layout],
            capture_output=True, text=True, timeout=300,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"malformed device layout hung instead of trapping: {exc}")
    assert "READY" in proc.stdout, proc.stderr[-4000:]
    assert "NO_TRAP" not in proc.stdout, "malformed device layout did not trap"
    assert proc.returncode != 0
