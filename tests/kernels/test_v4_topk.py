# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _run_topk(
    scores: torch.Tensor,
    k: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    from batchgen_kernels.attention.dsa.v4_topk import v4_topk

    return v4_topk(scores, k)


@pytest.mark.parametrize("T", [1, 32, 128, 1024])
@pytest.mark.parametrize("N", [1024, 4096])
def test_topk_matches_pytorch(T, N):
    torch.manual_seed(T * 10 + N)
    scores = torch.randn(T, N, device="cuda", dtype=torch.float32)

    values, indices = _run_topk(scores, 512)
    expected_values, expected_indices = torch.topk(scores, k=512, dim=-1)

    torch.testing.assert_close(values, expected_values)
    assert torch.equal(indices, expected_indices)


def test_topk_values_sorted_desc():
    torch.manual_seed(1)
    scores = torch.randn(128, 4096, device="cuda", dtype=torch.float32)

    values, _ = _run_topk(scores, 512)

    assert torch.all(values[:, :-1] >= values[:, 1:])


def test_topk_indices_valid():
    torch.manual_seed(2)
    scores = torch.randn(128, 4096, device="cuda", dtype=torch.float32)

    _, indices = _run_topk(scores, 512)

    assert ((indices >= 0) & (indices < scores.shape[-1])).all()


def test_k_equals_n():
    torch.manual_seed(3)
    scores = torch.randn(32, 512, device="cuda", dtype=torch.float32)

    values, indices = _run_topk(scores, 512)
    expected_values, expected_indices = torch.topk(scores, k=512, dim=-1)

    torch.testing.assert_close(values, expected_values)
    assert torch.equal(indices, expected_indices)


def test_k_equals_1():
    torch.manual_seed(4)
    scores = torch.randn(128, 4096, device="cuda", dtype=torch.float32)

    values, indices = _run_topk(scores, 1)
    expected_values, expected_indices = scores.max(dim=-1, keepdim=True)

    torch.testing.assert_close(values, expected_values)
    assert torch.equal(indices, expected_indices)


def test_all_equal_scores():
    scores = torch.ones(32, 1024, device="cuda", dtype=torch.float32)

    values, indices = _run_topk(scores, 512)
    sorted_indices = torch.sort(indices, dim=-1).values

    assert torch.equal(values, torch.ones_like(values))
    assert torch.all(sorted_indices[:, 1:] > sorted_indices[:, :-1])


def test_negative_scores():
    torch.manual_seed(5)
    scores = -torch.rand(128, 4096, device="cuda", dtype=torch.float32)

    values, indices = _run_topk(scores, 512)
    expected_values, expected_indices = torch.topk(scores, k=512, dim=-1)

    torch.testing.assert_close(values, expected_values)
    assert torch.equal(indices, expected_indices)


def test_single_token():
    torch.manual_seed(6)
    scores = torch.randn(1, 4096, device="cuda", dtype=torch.float32)

    values, indices = _run_topk(scores, 512)
    expected_values, expected_indices = torch.topk(scores, k=512, dim=-1)

    torch.testing.assert_close(values, expected_values)
    assert torch.equal(indices, expected_indices)


def test_flash_n1024_k512():
    torch.manual_seed(7)
    scores = torch.randn(128, 1024, device="cuda", dtype=torch.float32)

    values, indices = _run_topk(scores, 512)

    assert values.shape == (128, 512)
    assert indices.shape == (128, 512)


def test_pro_n4096_k512():
    torch.manual_seed(8)
    scores = torch.randn(128, 4096, device="cuda", dtype=torch.float32)

    values, indices = _run_topk(scores, 512)

    assert values.shape == (128, 512)
    assert indices.shape == (128, 512)


def test_benchmark():
    from tests.kernels.conftest import _bench

    torch.manual_seed(9)
    scores = torch.randn(1024, 4096, device="cuda", dtype=torch.float32)

    fused_ms = _bench(_run_topk, scores, 512)
    reference_ms = _bench(torch.topk, scores, 512, -1)
    print(
        f"\nK6 benchmark T=1024 N=4096 k=512 fused={fused_ms:.3f} ms pytorch={reference_ms:.3f} ms"
    )

    assert fused_ms > 0
    assert reference_ms > 0
