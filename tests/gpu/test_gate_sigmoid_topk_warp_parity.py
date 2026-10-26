# ---------------------------------------------------------------------------- #
#  BatchGen — GLM-5.2                                                           #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""REMOTE-ONLY GPU gate: the one-warp-per-row sigmoid top-k is byte-identical
to the block-per-row kernel it replaces on the GLM-5.2 DeepEP decode path.

Run on a GPU machine (single device is enough), never on the dev box:

    pytest tests/gpu/test_gate_sigmoid_topk_warp_parity.py

"Byte-identical" is checked with ``torch.equal`` on the raw outputs -- same
expert ids in the same route order (ties included), same FP32 weight bits --
never with a tolerance and never after sorting. The int64 form must equal the
block kernel's int32 ids widened, which is the layout the DeepEP exchange
reads.

  A  GLM-5.2 geometry (E=256, top-8, ungrouped noaux_tc, scale 2.5) at EVERY
     row count 1..256 (this covers the CUDA-graph decode buckets 8/32/64/128),
     int32 and int64 ids, several logit/bias scales.
  B  Ties: saturated sigmoids (many experts scoring exactly 1.0), all-equal
     rows, exact duplicates placed in the same lane, across lanes and across
     the k-th boundary, and biased-score ties between different raw scores.
  C  Other top-k / expert counts the kernel accepts (k 2/4/8/16, E not a
     multiple of 32) and strided logit rows.
  D  Rows with some NaN logits but at least k finite ones (the block kernel
     skips NaN candidates; so must this one).
  E  Both padding descriptions across CUDA-graph replays with the counts
     mutated on the device, including the DeepEP single-rank-block geometry.
  F  The launch shape itself: 32 threads per CTA, at most 64 registers per
     thread and no shared memory, read from the profiler's launch record, so a
     regression that re-inflates the CTA fails here.
"""

import json
import os
import tempfile

import pytest
import torch

from batchgen.moe.routing import gate_sigmoid_topk_cuda, gate_sigmoid_topk_warp_cuda

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="gate_sigmoid_topk is a CUDA kernel"
)

E_GLM, K_GLM, SCALE = 256, 8, 2.5


def _block(logits, bias, k, **kw):
    n = logits.shape[0]
    idx = torch.full((n, k), -7, dtype=torch.int32, device=logits.device)
    w = torch.full((n, k), -7.0, dtype=torch.float32, device=logits.device)
    gate_sigmoid_topk_cuda(logits, bias, k=k, routed_scaling_factor=SCALE,
                           topk_indices=idx, topk_weights=w, **kw)
    return idx, w


def _warp(logits, bias, k, idx_dtype, **kw):
    n = logits.shape[0]
    idx = torch.full((n, k), -7, dtype=idx_dtype, device=logits.device)
    w = torch.full((n, k), -7.0, dtype=torch.float32, device=logits.device)
    out = gate_sigmoid_topk_warp_cuda(logits, bias, k=k, routed_scaling_factor=SCALE,
                                      topk_indices=idx, topk_weights=w, **kw)
    assert out[0] is idx and out[1] is w
    return idx, w


def _assert_identical(logits, bias, k, what, **kw):
    ref_idx, ref_w = _block(logits, bias, k, **kw)
    for dt in (torch.int32, torch.int64):
        idx, w = _warp(logits, bias, k, dt, **kw)
        torch.cuda.synchronize()
        assert torch.equal(idx, ref_idx.to(dt)), f"{what}: ids differ ({dt})"
        assert torch.equal(w.view(torch.int32), ref_w.view(torch.int32)), \
            f"{what}: weight bits differ ({dt})"
    return ref_idx, ref_w


def _inputs(n, e, *, seed, logit_scale=1.0, bias_scale=1.0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(n, e, device="cuda", generator=gen) * logit_scale
    bias = torch.randn(e, device="cuda", generator=gen) * bias_scale
    return logits, bias


# --- A: GLM-5.2 geometry, every row count ------------------------------------
@pytest.mark.parametrize("logit_scale,bias_scale", [
    (1.0, 1.0),      # wide bias: selection differs from the raw-score order
    (3.0, 0.05),     # realistic: bias a small correction on spread logits
    (0.25, 0.01),    # tightly packed scores: near-ties everywhere
])
def test_glm52_geometry_every_row_count(logit_scale, bias_scale):
    for n in range(1, 257):
        logits, bias = _inputs(n, E_GLM, seed=1000 + n,
                               logit_scale=logit_scale, bias_scale=bias_scale)
        _assert_identical(logits, bias, K_GLM, f"N={n}")


# --- B: ties -----------------------------------------------------------------
def test_saturated_sigmoids_tie_to_the_lower_expert_id():
    n = 64
    logits, _ = _inputs(n, E_GLM, seed=3)
    # logits >= 20 saturate the FP32 sigmoid to exactly 1.0: 40 tied experts
    # per row, scattered across lanes and slots.
    gen = torch.Generator(device="cuda").manual_seed(4)
    for r in range(n):
        hot = torch.randperm(E_GLM, generator=gen, device="cuda")[:40]
        logits[r, hot] = 20.0 + torch.rand(40, generator=gen, device="cuda") * 50
    zero_bias = torch.zeros(E_GLM, device="cuda")
    idx, _ = _assert_identical(logits, zero_bias, K_GLM, "saturated, zero bias")
    # control: the tie break really is exercised (ascending ids among the 1.0s)
    assert torch.all(idx[:, 1:] > idx[:, :-1])
    _assert_identical(logits, torch.full((E_GLM,), 0.125, device="cuda"),
                      K_GLM, "saturated, constant bias")


def test_all_equal_rows_select_experts_zero_to_k():
    logits = torch.zeros(16, E_GLM, device="cuda")
    bias = torch.zeros(E_GLM, device="cuda")
    idx, w = _assert_identical(logits, bias, K_GLM, "all equal")
    assert torch.equal(idx, torch.arange(K_GLM, dtype=torch.int32,
                                         device="cuda").expand(16, K_GLM))


@pytest.mark.parametrize("positions", [
    (5, 37, 69, 101),       # same lane (5), four different slots
    (3, 4, 30, 31),         # adjacent lanes, same slot
    (0, 255, 128, 127),     # first, last, and the half boundary
    (7, 8, 200, 201, 202, 203, 204, 205, 206),  # 9-way tie across the k-th route
])
def test_exact_duplicates_tie_by_expert_id(positions):
    n = 32
    logits, bias = _inputs(n, E_GLM, seed=17, logit_scale=0.1, bias_scale=0.0)
    bias.zero_()
    pos = torch.tensor(positions, device="cuda")
    logits[:, pos] = 6.0           # identical top values for every row
    idx, _ = _assert_identical(logits, bias, K_GLM, f"dups {positions}")
    lead = sorted(positions)[:K_GLM]
    assert idx[0, :len(lead)].tolist() == lead


def test_biased_ties_between_different_raw_scores():
    # Two experts whose biased scores are bit-equal while their raw sigmoids
    # differ: selection must break the tie by id, weights must still gather
    # the raw scores.
    n = 8
    logits = torch.full((n, E_GLM), -4.0, device="cuda")
    bias = torch.zeros(E_GLM, device="cuda")
    logits[:, 10] = 0.0          # sigmoid 0.5
    logits[:, 3] = 2.0           # sigmoid ~0.8808
    s3 = torch.sigmoid(torch.tensor(2.0, device="cuda"))
    bias[10] = (s3 - 0.5)        # biased[10] lands on (or next to) biased[3]
    _assert_identical(logits, bias, K_GLM, "biased tie")


# --- C: other shapes ------------------------------------------------------------
@pytest.mark.parametrize("e,k", [(256, 2), (256, 4), (256, 16), (200, 8), (64, 8),
                                 (33, 16), (16, 16), (8, 8)])
def test_other_expert_counts_and_topk(e, k):
    for n in (1, 7, 64, 129):
        logits, bias = _inputs(n, e, seed=31 * e + k + n)
        _assert_identical(logits, bias, k, f"E={e} k={k} N={n}")


def test_strided_logit_rows():
    n = 64
    wide = torch.randn(n, E_GLM + 96, device="cuda")
    bias = torch.randn(E_GLM, device="cuda") * 0.1
    strided = wide[:, :E_GLM]
    assert not strided.is_contiguous()
    _assert_identical(strided, bias, K_GLM, "strided rows")
    # control: the contiguous copy gives the same answer, a stride-blind read not
    idx_s, _ = _warp(strided, bias, K_GLM, torch.int64)
    idx_c, _ = _warp(strided.contiguous(), bias, K_GLM, torch.int64)
    blind = wide.flatten()[: n * E_GLM].view(n, E_GLM)
    idx_b, _ = _warp(blind, bias, K_GLM, torch.int64)
    assert torch.equal(idx_s, idx_c) and not torch.equal(idx_s, idx_b)


# --- D: NaN logits ----------------------------------------------------------------
def test_rows_with_some_nan_logits():
    n = 32
    logits, bias = _inputs(n, E_GLM, seed=99)
    gen = torch.Generator(device="cuda").manual_seed(5)
    for r in range(n):
        bad = torch.randperm(E_GLM, generator=gen, device="cuda")[: 8 * (r % 8)]
        logits[r, bad] = float("nan")
    _assert_identical(logits, bias, K_GLM, "partial NaN rows")


# --- E: padding across graph replays ---------------------------------------------
def _graph(fn):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


@pytest.mark.parametrize("bucket", [8, 32, 64, 128, 256])
def test_deepep_single_rank_block_across_graph_replays(bucket):
    """The DeepEP decode geometry: one rank block of ``bucket`` rows, live up to
    rank_token_counts[rank]; int64 ids straight into the exchange buffer."""
    logits, bias = _inputs(bucket, E_GLM, seed=bucket, logit_scale=3.0, bias_scale=0.05)
    counts = torch.full((8,), bucket, dtype=torch.int64, device="cuda")
    rank = 5
    mine = counts[rank:rank + 1]
    ll_idx = torch.empty(bucket, K_GLM, dtype=torch.int64, device="cuda")
    w = torch.empty(bucket, K_GLM, dtype=torch.float32, device="cuda")
    ref_idx = torch.empty(bucket, K_GLM, dtype=torch.int32, device="cuda")
    ref_w = torch.empty_like(w)

    def run_warp():
        gate_sigmoid_topk_warp_cuda(logits, bias, k=K_GLM, routed_scaling_factor=SCALE,
                                    topk_indices=ll_idx, topk_weights=w,
                                    rank_token_counts=mine, rows_per_rank=bucket)

    def run_block():
        gate_sigmoid_topk_cuda(logits, bias, k=K_GLM, routed_scaling_factor=SCALE,
                               topk_indices=ref_idx, topk_weights=ref_w,
                               rank_token_counts=mine, rows_per_rank=bucket)

    g_warp, g_block = _graph(run_warp), _graph(run_block)
    for live in (bucket, 1, bucket // 2 + 1, 0, bucket - 1, bucket):
        ll_idx.fill_(-7)
        w.fill_(-7.0)
        ref_idx.fill_(-9)
        ref_w.fill_(-9.0)
        counts[rank] = live
        g_warp.replay()
        g_block.replay()
        torch.cuda.synchronize()
        assert torch.equal(ll_idx, ref_idx.to(torch.int64)), f"ids differ at live={live}"
        assert torch.equal(w.view(torch.int32), ref_w.view(torch.int32)), \
            f"weights differ at live={live}"
        assert torch.all(ll_idx[live:] == -1) and torch.all(w[live:] == 0)


def test_multi_rank_blocks_and_num_valid_tokens_match_the_block_kernel():
    world, rpr = 4, 32
    n = world * rpr
    logits, bias = _inputs(n, E_GLM, seed=77)
    counts = torch.tensor([3, 32, 0, 17], dtype=torch.int64, device="cuda")
    _assert_identical(logits, bias, K_GLM, "rank blocks",
                      rank_token_counts=counts, rows_per_rank=rpr)
    for live in (n, 0, 1, 77):
        valid = torch.tensor([live], dtype=torch.int32, device="cuda")
        _assert_identical(logits, bias, K_GLM, f"num_valid_tokens={live}",
                          num_valid_tokens=valid)


def test_fails_closed_on_mismatched_outputs():
    logits, bias = _inputs(16, E_GLM, seed=1)
    with pytest.raises(RuntimeError, match="contiguous"):
        gate_sigmoid_topk_warp_cuda(
            logits, bias, k=K_GLM,
            topk_indices=torch.empty(16, 2 * K_GLM, dtype=torch.int64,
                                     device="cuda")[:, ::2],
            topk_weights=torch.empty(16, K_GLM, device="cuda"))
    with pytest.raises(RuntimeError, match="contiguous"):
        gate_sigmoid_topk_warp_cuda(
            logits, bias, k=K_GLM,
            topk_indices=torch.empty(15, K_GLM, dtype=torch.int64, device="cuda"),
            topk_weights=torch.empty(16, K_GLM, device="cuda"))
    with pytest.raises(RuntimeError, match="must equal N"):
        gate_sigmoid_topk_warp_cuda(
            logits, bias, k=K_GLM,
            rank_token_counts=torch.tensor([4, 4], dtype=torch.int64, device="cuda"),
            rows_per_rank=9)


# --- F: the footprint ----------------------------------------------------------------
def test_launch_footprint_fits_beside_a_deepgemm_cta():
    """One 32-thread CTA per row at <= 64 registers per thread (<= 2,048
    registers per CTA) and no shared memory."""
    from torch.profiler import ProfilerActivity, profile

    n = 64
    logits, bias = _inputs(n, E_GLM, seed=2)
    idx = torch.empty(n, K_GLM, dtype=torch.int64, device="cuda")
    w = torch.empty(n, K_GLM, device="cuda")
    gate_sigmoid_topk_warp_cuda(logits, bias, k=K_GLM, topk_indices=idx, topk_weights=w)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        gate_sigmoid_topk_warp_cuda(logits, bias, k=K_GLM, topk_indices=idx, topk_weights=w)
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.json")
        prof.export_chrome_trace(path)
        events = json.load(open(path))["traceEvents"]
    launches = [e for e in events if e.get("cat") == "kernel"
                and "gate_sigmoid_topk_warp_kernel" in e.get("name", "")]
    assert len(launches) == 1, [e.get("name") for e in events if e.get("cat") == "kernel"]
    args = launches[0]["args"]
    assert list(args["block"]) == [32, 1, 1]
    assert list(args["grid"]) == [n, 1, 1]
    regs = int(args["registers per thread"])
    assert regs <= 64, f"{regs} registers per thread x 32 threads > 2,048 per CTA"
    assert int(args["shared memory"]) <= 1024
