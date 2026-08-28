# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
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
