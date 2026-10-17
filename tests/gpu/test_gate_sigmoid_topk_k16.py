# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""REMOTE-ONLY GPU gate for ``gate_sigmoid_topk`` at K3 geometry.

Run on a GPU machine (single device is enough), never on the dev box:

    pytest tests/gpu/test_gate_sigmoid_topk_k16.py

What the CPU contracts cannot see, and this file pins:

  A  E=896 / top_k=16 against the eager ``KimiMoEGate.select_experts`` recipe
     (sigmoid scores, FP32 correction bias, ungrouped top-k, renormalize,
     scale).  Compared as a dense [N, E] weight vector so the kernel's
     descending order versus ``torch.topk``'s is not a false failure.
  B  Strided router-logit rows — the K3 fused front hands the kernel the
     leading ``num_experts`` columns of a ``[N, num_experts + latent]`` FP32
     GEMM output, so the kernel must honour ``stride(0)``, not assume ``E``.
  C  A device-resident ``num_valid_tokens`` scalar mutated between CUDA-graph
     replays: rows at or beyond it must come out index ``-1`` / weight ``0``,
     and the live rows must be unaffected.
  D  K=8 / E=384 regression: the K2.5 and GLM-5 callers pass contiguous logits
     and no valid-token scalar, and must match the eager reference within the
     established tolerance at their own geometry.
  E  The optional latent epilogue: with ``latent_out``/``latent_offset`` the
     kernel also casts the latent suffix of the same fused rows to BF16, which
     must equal ``fused_out[:, num_experts:].to(torch.bfloat16)``, must leave
     routing untouched, must zero padding rows, and must fail closed when the
     requested columns do not fit inside the row stride.
  F  The EP all-gathered padding shape (``rank_token_counts`` +
     ``rows_per_rank``): live rows are a per-rank prefix inside each rank's row
     block, NOT a prefix of N, so this case pins the result against the
     ``torch.where`` masking the GLM-5 MoE graph used to run as separate
     launches, across CUDA-graph replays with mutated counts, and pins the
     fail-closed checks for a geometry that does not tile N exactly.
"""

import pytest
import torch

from batchgen.moe.routing import gate_sigmoid_topk_cuda

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="gate_sigmoid_topk is a CUDA kernel"
)

SCALE = 2.5


def _reference(logits, bias, k):
    """The eager ``KimiMoEGate.select_experts`` recipe, ungrouped."""
    scores = logits.float().sigmoid()
    _, idx = torch.topk(scores + bias.float(), k=k, dim=-1, sorted=True)
    weight = scores.gather(1, idx)
    weight = weight / (weight.sum(dim=-1, keepdim=True) + 1e-20)
    return idx.to(torch.int32), weight * SCALE


def _dense(idx, weight, num_experts):
    """Scatter [N, K] routes into [N, E] so route ORDER cannot affect equality."""
    out = torch.zeros(
        idx.shape[0], num_experts, dtype=torch.float32, device=idx.device
    )
    live = idx >= 0
    out.scatter_(
        1, idx.clamp(min=0).long(), torch.where(live, weight, torch.zeros_like(weight))
    )
    return out


def _inputs(num_tokens, num_experts, *, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device, generator=gen
    )
    bias = torch.randn(
        num_experts, dtype=torch.float32, device=device, generator=gen
    )
    return logits, bias


def test_k16_e896_matches_the_eager_selection():
    num_tokens, num_experts, k = 96, 896, 16
    logits, bias = _inputs(num_tokens, num_experts, seed=11)

    idx, weight = gate_sigmoid_topk_cuda(
        logits, bias, k=k, routed_scaling_factor=SCALE
    )
    ref_idx, ref_weight = _reference(logits, bias, k)

    assert idx.dtype == torch.int32 and tuple(idx.shape) == (num_tokens, k)
    assert weight.dtype == torch.float32
    # Exact expert SET: selection is discrete, so a mismatch is a real bug and
    # not a tolerance question.
    assert torch.equal(
        idx.sort(dim=-1).values, ref_idx.sort(dim=-1).values
    ), "K16/E896 selected a different expert set than the eager gate"
    torch.testing.assert_close(
        _dense(idx, weight, num_experts),
        _dense(ref_idx, ref_weight, num_experts),
        rtol=1e-5,
        atol=2e-6,
    )
    # Renormalize + scale: every row's weights sum to the scaling factor.
    torch.testing.assert_close(
        weight.sum(dim=-1),
        torch.full((num_tokens,), SCALE, device=weight.device),
        rtol=1e-5,
        atol=1e-5,
    )


def test_strided_router_logit_rows_are_read_with_the_row_stride():
    num_tokens, num_experts, latent, k = 64, 896, 512, 16
    fused_out = torch.randn(
        num_tokens, num_experts + latent, dtype=torch.float32, device="cuda"
    )
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")
    strided = fused_out[:, :num_experts]
    assert strided.stride(0) == num_experts + latent and not strided.is_contiguous()

    idx, weight = gate_sigmoid_topk_cuda(
        strided, bias, k=k, routed_scaling_factor=SCALE
    )
    ref_idx, ref_weight = gate_sigmoid_topk_cuda(
        strided.contiguous(), bias, k=k, routed_scaling_factor=SCALE
    )

    assert torch.equal(idx, ref_idx)
    assert torch.equal(weight, ref_weight)
    # MANDATORY control: a stride-blind kernel would read the latent columns of
    # earlier rows, so the packed-copy answer must differ from reading the wide
    # buffer as if it were [N, E]-contiguous.
    as_if_contiguous = fused_out.flatten()[: num_tokens * num_experts].view(
        num_tokens, num_experts
    )
    wrong_idx, _ = gate_sigmoid_topk_cuda(
        as_if_contiguous, bias, k=k, routed_scaling_factor=SCALE
    )
    assert not torch.equal(wrong_idx, ref_idx)


def test_device_num_valid_tokens_masks_padding_across_graph_replays():
    num_tokens, num_experts, k = 32, 896, 16
    logits, bias = _inputs(num_tokens, num_experts, seed=7)
    idx = torch.empty(num_tokens, k, dtype=torch.int32, device="cuda")
    weight = torch.empty(num_tokens, k, dtype=torch.float32, device="cuda")
    valid = torch.zeros(1, dtype=torch.int32, device="cuda")

    def run():
        gate_sigmoid_topk_cuda(
            logits, bias, k=k, routed_scaling_factor=SCALE,
            topk_indices=idx, topk_weights=weight, num_valid_tokens=valid,
        )

    # Warm up on a side stream, then capture: the valid count must live only on
    # the device, so one captured graph serves every live-row count.
    valid.fill_(num_tokens)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    ref_idx, ref_weight = _reference(logits, bias, k)
    for live in (num_tokens, 1, 17, 0, num_tokens):
        idx.fill_(-7)
        weight.fill_(-7.0)
        valid.fill_(live)
        graph.replay()
        torch.cuda.synchronize()

        assert torch.equal(
            idx[:live].sort(dim=-1).values, ref_idx[:live].sort(dim=-1).values
        ), f"live rows changed at num_valid_tokens={live}"
        torch.testing.assert_close(
            _dense(idx[:live], weight[:live], num_experts),
            _dense(ref_idx[:live], ref_weight[:live], num_experts),
            rtol=1e-5,
            atol=2e-6,
        )
        assert torch.equal(
            idx[live:], torch.full_like(idx[live:], -1)
        ), f"padding indices not -1 at num_valid_tokens={live}"
        assert torch.equal(
            weight[live:], torch.zeros_like(weight[live:])
        ), f"padding weights not 0 at num_valid_tokens={live}"


def test_latent_epilogue_casts_the_same_fused_rows_without_touching_routing():
    num_tokens, num_experts, latent, k = 64, 896, 512, 16
    fused_out = torch.randn(
        num_tokens, num_experts + latent, dtype=torch.float32, device="cuda"
    )
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")
    router_logits = fused_out[:, :num_experts]
    latent_out = torch.full(
        (num_tokens, latent), -7.0, dtype=torch.bfloat16, device="cuda"
    )

    idx, weight = gate_sigmoid_topk_cuda(
        router_logits, bias, k=k, routed_scaling_factor=SCALE,
        latent_out=latent_out, latent_offset=num_experts,
    )
    ref_idx, ref_weight = gate_sigmoid_topk_cuda(
        router_logits, bias, k=k, routed_scaling_factor=SCALE
    )

    # The epilogue is exactly the copy the K3 graph used to run separately.
    assert torch.equal(latent_out, fused_out[:, num_experts:].to(torch.bfloat16))
    # ...and it must not perturb the selection or the weights.
    assert torch.equal(idx, ref_idx)
    assert torch.equal(weight, ref_weight)


def _legacy_rank_masked(idx, weight, counts, rows_per_rank):
    """The masking the GLM-5 MoE graph ran as four separate launches.

    ``rank_token_counts[rank_ids]`` gather, ``local_pos >= valid_per_row``
    compare, then one ``torch.where`` per output. Kept here as the oracle the
    folded kernel epilogue must reproduce bit for bit.
    """
    n = idx.shape[0]
    positions = torch.arange(n, dtype=torch.int64, device=idx.device)
    valid_per_row = counts[positions // rows_per_rank]
    padding = (positions % rows_per_rank) >= valid_per_row
    padding_2d = padding.unsqueeze(1).expand_as(idx)
    return (
        torch.where(padding_2d, torch.full_like(idx, -1), idx),
        torch.where(padding_2d, torch.zeros_like(weight), weight),
    )


def test_rank_token_counts_masks_per_rank_padding_across_graph_replays():
    # GLM-5 EP geometry: world_size rank blocks of bucket_size rows each, with
    # an independent live-token prefix per rank.
    world_size, rows_per_rank, num_experts, k = 4, 8, 384, 8
    num_tokens = world_size * rows_per_rank
    logits, bias = _inputs(num_tokens, num_experts, seed=23)
    idx = torch.empty(num_tokens, k, dtype=torch.int32, device="cuda")
    weight = torch.empty(num_tokens, k, dtype=torch.float32, device="cuda")
    counts = torch.full((world_size,), rows_per_rank, dtype=torch.int64, device="cuda")

    def run():
        gate_sigmoid_topk_cuda(
            logits, bias, k=k, routed_scaling_factor=SCALE,
            topk_indices=idx, topk_weights=weight,
            rank_token_counts=counts, rows_per_rank=rows_per_rank,
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    # The unmasked routing the kernel must still produce for live rows.
    full_idx = torch.empty_like(idx)
    full_weight = torch.empty_like(weight)
    gate_sigmoid_topk_cuda(
        logits, bias, k=k, routed_scaling_factor=SCALE,
        topk_indices=full_idx, topk_weights=full_weight,
    )
    torch.cuda.synchronize()

    # Ragged, zero, and full per-rank occupancies, including a rank that is
    # entirely padding between two live ranks.
    for live in (
        [rows_per_rank] * world_size,
        [3, 8, 0, 5],
        [0, 0, 0, 0],
        [1, 0, 7, 2],
        [rows_per_rank] * world_size,
    ):
        idx.fill_(-7)
        weight.fill_(-7.0)
        counts.copy_(torch.tensor(live, dtype=torch.int64, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()

        want_idx, want_weight = _legacy_rank_masked(
            full_idx, full_weight, counts, rows_per_rank
        )
        # Bitwise, not approximate: the kernel epilogue replaces a `where`, so
        # live rows must be untouched and padding rows exactly -1 / +0.
        assert torch.equal(idx, want_idx), f"indices differ at live={live}"
        assert torch.equal(weight, want_weight), f"weights differ at live={live}"


def test_rank_token_counts_fails_closed():
    world_size, rows_per_rank, num_experts, k = 2, 8, 384, 8
    num_tokens = world_size * rows_per_rank
    logits, bias = _inputs(num_tokens, num_experts, seed=5)
    counts = torch.full((world_size,), rows_per_rank, dtype=torch.int64, device="cuda")
    valid = torch.tensor([4], dtype=torch.int32, device="cuda")

    # Both padding descriptions at once: ambiguous, so it must not silently pick.
    with pytest.raises(ValueError, match="mutually exclusive"):
        gate_sigmoid_topk_cuda(
            logits, bias, k=k, routed_scaling_factor=SCALE,
            num_valid_tokens=valid,
            rank_token_counts=counts, rows_per_rank=rows_per_rank,
        )
    # A geometry that does not tile N exactly would index past the counts.
    with pytest.raises(RuntimeError, match="must equal N"):
        gate_sigmoid_topk_cuda(
            logits, bias, k=k, routed_scaling_factor=SCALE,
            rank_token_counts=counts, rows_per_rank=rows_per_rank + 1,
        )
    # int32 counts would be read as int64 pairs.
    with pytest.raises(RuntimeError, match="must be int64"):
        gate_sigmoid_topk_cuda(
            logits, bias, k=k, routed_scaling_factor=SCALE,
            rank_token_counts=counts.to(torch.int32), rows_per_rank=rows_per_rank,
        )


def test_latent_epilogue_replays_with_device_valid_count_in_a_cuda_graph():
    num_tokens, num_experts, latent, k = 32, 896, 512, 16
    fused_out = torch.randn(
        num_tokens, num_experts + latent, dtype=torch.float32, device="cuda"
    )
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")
    router_logits = fused_out[:, :num_experts]
    idx = torch.empty(num_tokens, k, dtype=torch.int32, device="cuda")
    weight = torch.empty(num_tokens, k, dtype=torch.float32, device="cuda")
    latent_out = torch.empty(num_tokens, latent, dtype=torch.bfloat16, device="cuda")
    valid = torch.zeros(1, dtype=torch.int32, device="cuda")

    def run():
        gate_sigmoid_topk_cuda(
            router_logits, bias, k=k, routed_scaling_factor=SCALE,
            topk_indices=idx, topk_weights=weight, num_valid_tokens=valid,
            latent_out=latent_out, latent_offset=num_experts,
        )

    valid.fill_(num_tokens)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()

    ref_idx, ref_weight = _reference(router_logits, bias, k)
    expected_latent = fused_out[:, num_experts:].to(torch.bfloat16)
    for live in (num_tokens, 1, 17, 0, num_tokens):
        idx.fill_(-7)
        weight.fill_(-7.0)
        latent_out.fill_(-7.0)
        valid.fill_(live)
        graph.replay()
        torch.cuda.synchronize()

        assert torch.equal(
            idx[:live].sort(dim=-1).values, ref_idx[:live].sort(dim=-1).values
        ), f"live routes changed at num_valid_tokens={live}"
        torch.testing.assert_close(
            _dense(idx[:live], weight[:live], num_experts),
            _dense(ref_idx[:live], ref_weight[:live], num_experts),
            rtol=1e-5,
            atol=2e-6,
        )
        assert torch.equal(
            latent_out[:live], expected_latent[:live]
        ), f"live latent rows wrong at num_valid_tokens={live}"
        assert torch.equal(
            idx[live:], torch.full_like(idx[live:], -1)
        ), f"padding indices not -1 at num_valid_tokens={live}"
        assert torch.equal(
            weight[live:], torch.zeros_like(weight[live:])
        ), f"padding weights not 0 at num_valid_tokens={live}"
        assert torch.equal(
            latent_out[live:], torch.zeros_like(latent_out[live:])
        ), f"padding latent rows not 0 at num_valid_tokens={live}"


def test_latent_epilogue_zeroes_padding_rows_from_the_device_valid_count():
    num_tokens, num_experts, latent, k = 32, 896, 512, 16
    fused_out = torch.randn(
        num_tokens, num_experts + latent, dtype=torch.float32, device="cuda"
    )
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")
    router_logits = fused_out[:, :num_experts]
    idx = torch.empty(num_tokens, k, dtype=torch.int32, device="cuda")
    weight = torch.empty(num_tokens, k, dtype=torch.float32, device="cuda")
    latent_out = torch.empty(num_tokens, latent, dtype=torch.bfloat16, device="cuda")
    valid = torch.zeros(1, dtype=torch.int32, device="cuda")
    expected = fused_out[:, num_experts:].to(torch.bfloat16)

    for live in (num_tokens, 1, 19, 0):
        latent_out.fill_(-7.0)
        valid.fill_(live)
        gate_sigmoid_topk_cuda(
            router_logits, bias, k=k, routed_scaling_factor=SCALE,
            topk_indices=idx, topk_weights=weight, num_valid_tokens=valid,
            latent_out=latent_out, latent_offset=num_experts,
        )
        torch.cuda.synchronize()

        assert torch.equal(
            latent_out[:live], expected[:live]
        ), f"live latent rows wrong at num_valid_tokens={live}"
        # The zero-padded decode activation makes the bias-free fused GEMM emit
        # zero rows, so padded latent rows must be deterministically zero.
        assert torch.equal(
            latent_out[live:], torch.zeros_like(latent_out[live:])
        ), f"padding latent rows not 0 at num_valid_tokens={live}"


def test_latent_epilogue_fails_closed_on_offsets_outside_the_row_stride():
    num_tokens, num_experts, latent, k = 8, 896, 512, 16
    fused_out = torch.randn(
        num_tokens, num_experts + latent, dtype=torch.float32, device="cuda"
    )
    bias = torch.randn(num_experts, dtype=torch.float32, device="cuda")
    router_logits = fused_out[:, :num_experts]
    latent_out = torch.empty(num_tokens, latent, dtype=torch.bfloat16, device="cuda")

    def run(logits, offset, out=latent_out):
        gate_sigmoid_topk_cuda(
            logits, bias, k=k, routed_scaling_factor=SCALE,
            latent_out=out, latent_offset=offset,
        )

    # Overlapping the router columns would read the logits back as latent.
    with pytest.raises(RuntimeError):
        run(router_logits, num_experts - 1)
    # Running past the end of the row would read the next row's logits.
    with pytest.raises(RuntimeError):
        run(router_logits, num_experts + 1)
    # A genuinely contiguous [N, E] buffer has no latent columns at all.
    with pytest.raises(RuntimeError):
        run(router_logits.contiguous(), num_experts)
    # A row count that does not match the logits cannot be indexed row-wise.
    with pytest.raises(RuntimeError):
        run(
            router_logits,
            num_experts,
            torch.empty(num_tokens + 1, latent, dtype=torch.bfloat16, device="cuda"),
        )
    # FP32 output would silently reinterpret the BF16 stores.
    with pytest.raises(RuntimeError):
        run(
            router_logits,
            num_experts,
            torch.empty(num_tokens, latent, dtype=torch.float32, device="cuda"),
        )


def test_k8_e384_regression_for_the_k25_and_glm5_callers():
    num_tokens, num_experts, k = 128, 384, 8
    logits, bias = _inputs(num_tokens, num_experts, seed=3)

    idx, weight = gate_sigmoid_topk_cuda(
        logits, bias, k=k, routed_scaling_factor=SCALE
    )
    ref_idx, ref_weight = _reference(logits, bias, k)

    assert torch.equal(idx.sort(dim=-1).values, ref_idx.sort(dim=-1).values)
    torch.testing.assert_close(
        _dense(idx, weight, num_experts),
        _dense(ref_idx, ref_weight, num_experts),
        rtol=1e-5,
        atol=2e-6,
    )
