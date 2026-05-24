# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _make_inputs(
    tokens: int,
    experts: int,
    hidden_size: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    hidden_states = torch.randn(
        tokens,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    gate_weight = torch.randn(
        experts,
        hidden_size,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    bias = torch.randn(
        experts,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    return hidden_states, gate_weight, bias


def _direct_sqrtsoftplus_topk(
    hidden_states: torch.Tensor,
    gate_weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    topk: int = 6,
    route_scale: float = 1.0,
    norm_topk_prob: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = F.linear(hidden_states.float(), gate_weight.float())
    scores = F.softplus(scores).sqrt()
    select_scores = scores + bias.float().unsqueeze(0)
    topk_indices = torch.topk(select_scores, k=topk, dim=-1).indices
    topk_weights = scores.gather(-1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (
            topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        )
    return topk_weights * route_scale, topk_indices


@pytest.mark.parametrize("tokens", [1, 4, 32, 128, 1024])
@pytest.mark.parametrize("experts", [256, 384])
def test_sqrtsoftplus_matches_pytorch(tokens, experts):
    from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk

    hidden_states, gate_weight, bias = _make_inputs(
        tokens, experts, 128, seed=tokens + experts
    )

    weights, indices = sqrtsoftplus_topk(hidden_states, gate_weight, bias)
    expected_weights, expected_indices = _direct_sqrtsoftplus_topk(
        hidden_states, gate_weight, bias
    )

    assert torch.equal(indices, expected_indices)
    assert torch.allclose(weights, expected_weights, atol=1e-4)


def test_with_bias():
    from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk

    hidden_states = torch.zeros(32, 1, device="cuda", dtype=torch.bfloat16)
    gate_weight = torch.zeros(256, 1, device="cuda", dtype=torch.bfloat16)
    bias = torch.linspace(-2.0, 2.0, 256, device="cuda", dtype=torch.float32)

    _, indices = sqrtsoftplus_topk(hidden_states, gate_weight, bias)
    expected = torch.topk(bias.unsqueeze(0).expand(32, -1), k=6, dim=-1).indices

    assert torch.equal(indices, expected)


def test_normalization():
    from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk

    hidden_states, gate_weight, bias = _make_inputs(32, 256, 128, seed=3)
    weights, _ = sqrtsoftplus_topk(
        hidden_states,
        gate_weight,
        bias,
        topk=6,
        norm_topk_prob=True,
    )

    expected = torch.ones(32, device="cuda", dtype=weights.dtype)
    assert torch.allclose(weights.sum(dim=-1), expected, atol=1e-4)


@pytest.mark.parametrize("route_scale", [1.5, 2.5])
def test_route_scale(route_scale):
    from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk

    hidden_states, gate_weight, bias = _make_inputs(32, 256, 128, seed=11)
    base_weights, base_indices = sqrtsoftplus_topk(
        hidden_states, gate_weight, bias, route_scale=1.0
    )
    scaled_weights, scaled_indices = sqrtsoftplus_topk(
        hidden_states,
        gate_weight,
        bias,
        route_scale=route_scale,
    )

    assert torch.equal(scaled_indices, base_indices)
    assert torch.allclose(scaled_weights, base_weights * route_scale, atol=1e-4)


def test_large_negative():
    from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk

    hidden_states = torch.full(
        (32, 1), -100.0, device="cuda", dtype=torch.float32
    )
    gate_weight = torch.ones(256, 1, device="cuda", dtype=torch.float32)
    bias = torch.zeros(256, device="cuda", dtype=torch.float32)

    weights, _ = sqrtsoftplus_topk(
        hidden_states,
        gate_weight,
        bias,
        norm_topk_prob=False,
    )

    assert torch.isfinite(weights).all()
    assert torch.allclose(weights, torch.zeros_like(weights), atol=1e-4)


def test_zero_input():
    from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk

    hidden_states = torch.zeros(32, 1, device="cuda", dtype=torch.float32)
    gate_weight = torch.ones(256, 1, device="cuda", dtype=torch.float32)
    bias = torch.zeros(256, device="cuda", dtype=torch.float32)

    weights, _ = sqrtsoftplus_topk(
        hidden_states,
        gate_weight,
        bias,
        norm_topk_prob=False,
    )

    expected = torch.full_like(
        weights, F.softplus(torch.zeros((), device="cuda")).sqrt()
    )
    assert torch.allclose(weights, expected, atol=1e-4)


def test_large_positive():
    from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk

    hidden_states = torch.full(
        (32, 1), 100.0, device="cuda", dtype=torch.float32
    )
    gate_weight = torch.ones(256, 1, device="cuda", dtype=torch.float32)
    bias = torch.zeros(256, device="cuda", dtype=torch.float32)

    weights, _ = sqrtsoftplus_topk(
        hidden_states,
        gate_weight,
        bias,
        norm_topk_prob=False,
    )

    expected = torch.full_like(weights, 10.0)
    assert torch.allclose(weights, expected, atol=1e-4)


def test_all_equal_scores():
    from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk

    hidden_states = torch.zeros(32, 16, device="cuda", dtype=torch.bfloat16)
    gate_weight = torch.zeros(256, 16, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(256, device="cuda", dtype=torch.float32)

    weights, indices = sqrtsoftplus_topk(hidden_states, gate_weight, bias)

    assert weights.shape == (32, 6)
    assert indices.shape == (32, 6)
    assert (indices >= 0).all()
    assert (indices < 256).all()
    assert torch.allclose(
        weights, torch.full_like(weights, 1.0 / 6.0), atol=1e-4
    )


@pytest.mark.parametrize("tokens", [1, 128, 1024])
def test_flash_shape(tokens):
    from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk

    hidden_states, gate_weight, bias = _make_inputs(
        tokens, 256, 4096, seed=tokens
    )
    weights, indices = sqrtsoftplus_topk(hidden_states, gate_weight, bias)

    assert weights.shape == (tokens, 6)
    assert indices.shape == (tokens, 6)


@pytest.mark.parametrize("tokens", [1, 128, 1024])
def test_pro_shape(tokens):
    from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk

    hidden_states, gate_weight, bias = _make_inputs(
        tokens, 384, 7168, seed=tokens + 100
    )
    weights, indices = sqrtsoftplus_topk(hidden_states, gate_weight, bias)

    assert weights.shape == (tokens, 6)
    assert indices.shape == (tokens, 6)


@pytest.mark.parametrize("tokens", [128, 1024, 4096])
@pytest.mark.parametrize("experts", [256, 384])
def test_benchmark(tokens, experts):
    from batchgen_kernels.moe.v4_sqrtsoftplus_topk import sqrtsoftplus_topk
    from tests.kernels.conftest import _bench

    hidden_states, gate_weight, bias = _make_inputs(
        tokens, experts, 256, seed=tokens + experts + 17
    )

    weights, indices = sqrtsoftplus_topk(hidden_states, gate_weight, bias)
    expected_weights, expected_indices = _direct_sqrtsoftplus_topk(
        hidden_states, gate_weight, bias
    )

    python_ms = _bench(sqrtsoftplus_topk, hidden_states, gate_weight, bias)
    direct_ms = _bench(
        _direct_sqrtsoftplus_topk, hidden_states, gate_weight, bias
    )

    assert torch.equal(indices, expected_indices)
    assert torch.allclose(weights, expected_weights, atol=1e-4)
    print(
        f"\nK9 routing T={tokens} E={experts}: "
        f"python={python_ms:.3f} ms direct={direct_ms:.3f} ms"
    )
