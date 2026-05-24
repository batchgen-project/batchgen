# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _make_inputs(
    T: int, hidden: int, seed: int = 0, batch: int = 1, hc_mult: int = 4
):
    torch.manual_seed(seed)
    hidden_states = torch.randn(
        batch, T, hc_mult, hidden, device="cuda", dtype=torch.bfloat16
    )
    fn_weight = torch.randn(
        (2 + hc_mult) * hc_mult,
        hc_mult * hidden,
        device="cuda",
        dtype=torch.float32,
    )
    scale = torch.randn(3, device="cuda", dtype=torch.float32)
    base = torch.randn(
        (2 + hc_mult) * hc_mult, device="cuda", dtype=torch.float32
    )
    return hidden_states, fn_weight, scale, base


def _make_split_inputs(T: int, seed: int = 0, batch: int = 1, hc_mult: int = 4):
    torch.manual_seed(seed)
    mixes = torch.randn(
        batch, T, (2 + hc_mult) * hc_mult, device="cuda", dtype=torch.float32
    )
    scale = torch.randn(3, device="cuda", dtype=torch.float32)
    base = torch.randn(
        (2 + hc_mult) * hc_mult, device="cuda", dtype=torch.float32
    )
    return mixes, scale, base


def _ref_pre(
    hidden_states: torch.Tensor,
    fn_weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
    rms_norm_eps: float = 1e-6,
):
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashDecoderLayer,
    )

    ctx = SimpleNamespace(
        hc_mult=hc_mult,
        hc_sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
        rms_norm_eps=rms_norm_eps,
    )
    return DeepSeekV4FlashDecoderLayer._hc_pre(
        ctx, hidden_states, fn_weight, scale, base
    )


def _ref_post(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
):
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashDecoderLayer,
    )

    return DeepSeekV4FlashDecoderLayer._hc_post(
        SimpleNamespace(), hidden_states, residual, post, comb
    )


def test_sinkhorn_doubly_stochastic():
    from batchgen_kernels.common.v4_hyper_connections import hc_split

    mixes, scale, base = _make_split_inputs(T=1, seed=0)
    _, _, comb = hc_split(mixes, scale, base, 4, 20, 1e-6)

    rows = comb[0, 0].sum(dim=-1)
    cols = comb[0, 0].sum(dim=-2)
    ones = torch.ones(4, device="cuda", dtype=comb.dtype)

    assert torch.allclose(rows, ones, atol=1e-3)
    assert torch.allclose(cols, ones, atol=1e-3)


@pytest.mark.parametrize("T", [1, 32, 128])
@pytest.mark.parametrize("hidden", [4096, 7168])
def test_hc_pre_matches_ref(T, hidden):
    from batchgen_kernels.common.v4_hyper_connections import hc_pre

    hidden_states, fn_weight, scale, base = _make_inputs(
        T, hidden, seed=T + hidden
    )

    reduced, post, comb = hc_pre(
        hidden_states, fn_weight, scale, base, 4, 20, 1e-6, 1e-6
    )
    ref_reduced, ref_post_out, ref_comb = _ref_pre(
        hidden_states, fn_weight, scale, base
    )

    assert torch.allclose(reduced, ref_reduced, atol=1e-3)
    assert torch.allclose(post, ref_post_out, atol=1e-3)
    assert torch.allclose(comb, ref_comb, atol=1e-3)


@pytest.mark.parametrize("T", [1, 32, 128])
def test_hc_post_matches_ref(T):
    from batchgen_kernels.common.v4_hyper_connections import hc_post

    hidden = 4096
    hc_mult = 4
    torch.manual_seed(100 + T)
    hidden_states = torch.randn(
        1, T, hidden, device="cuda", dtype=torch.bfloat16
    )
    residual = torch.randn(
        1, T, hc_mult, hidden, device="cuda", dtype=torch.bfloat16
    )
    post = torch.randn(1, T, hc_mult, device="cuda", dtype=torch.float32)
    comb = torch.softmax(
        torch.randn(1, T, hc_mult, hc_mult, device="cuda", dtype=torch.float32),
        dim=-1,
    )

    output = hc_post(hidden_states, residual, post, comb)
    ref_output = _ref_post(hidden_states, residual, post, comb)

    assert torch.allclose(output, ref_output, atol=1e-3)


def test_hc_split_sigmoid_softmax_sinkhorn():
    from batchgen_kernels.common.v4_hyper_connections import hc_split
    from batchgen.models.deepseek.deepseekv4_flash.model import _hc_split

    mixes, scale, base = _make_split_inputs(T=32, seed=1)

    pre, post, comb = hc_split(mixes, scale, base, 4, 20, 1e-6)
    ref_pre, ref_post_out, ref_comb = _hc_split(mixes, scale, base, 4, 20, 1e-6)

    assert torch.allclose(pre, ref_pre, atol=1e-3)
    assert torch.allclose(post, ref_post_out, atol=1e-3)
    assert torch.allclose(comb, ref_comb, atol=1e-3)


def test_pre_reduces_hc_mult():
    import torch.nn.functional as F

    from batchgen_kernels.common.v4_hyper_connections import hc_pre, hc_split

    hidden_states, fn_weight, scale, base = _make_inputs(
        T=32, hidden=4096, seed=2
    )
    reduced, _, _ = hc_pre(
        hidden_states, fn_weight, scale, base, 4, 20, 1e-6, 1e-6
    )

    flat = hidden_states.flatten(2).float()
    mixes = F.linear(flat, fn_weight) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + 1e-6
    )
    pre, _, _ = hc_split(mixes, scale, base, 4, 20, 1e-6)
    expected = torch.sum(
        pre.unsqueeze(-1) * flat.view(hidden_states.shape), dim=2
    ).to(hidden_states.dtype)

    assert reduced.shape == (1, 32, 4096)
    assert torch.allclose(reduced, expected, atol=1e-3)


def test_post_reconstruction():
    from batchgen_kernels.common.v4_hyper_connections import hc_post

    torch.manual_seed(3)
    hidden_states = torch.randn(
        1, 32, 4096, device="cuda", dtype=torch.bfloat16
    )
    residual = torch.randn(1, 32, 4, 4096, device="cuda", dtype=torch.bfloat16)
    post = torch.randn(1, 32, 4, device="cuda", dtype=torch.float32)
    comb = torch.randn(1, 32, 4, 4, device="cuda", dtype=torch.float32)

    output = hc_post(hidden_states, residual, post, comb)
    expected = (
        post.unsqueeze(-1) * hidden_states.unsqueeze(-2)
        + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    ).to(hidden_states.dtype)

    assert output.shape == (1, 32, 4, 4096)
    assert torch.allclose(output, expected, atol=1e-3)


def test_sinkhorn_convergence():
    from batchgen_kernels.common.v4_hyper_connections import hc_split

    mixes, scale, base = _make_split_inputs(T=1, seed=4)

    _, _, comb1 = hc_split(mixes, scale, base, 4, 1, 1e-6)
    _, _, comb5 = hc_split(mixes, scale, base, 4, 5, 1e-6)
    _, _, comb20 = hc_split(mixes, scale, base, 4, 20, 1e-6)

    def _dev(x):
        rows = (x[0, 0].sum(dim=-1) - 1).abs().max()
        cols = (x[0, 0].sum(dim=-2) - 1).abs().max()
        return torch.maximum(rows, cols)

    dev1 = _dev(comb1)
    dev5 = _dev(comb5)
    dev20 = _dev(comb20)

    assert dev20 <= dev5 + 1e-6
    assert dev5 <= dev1 + 1e-6
    assert dev20.item() < 1e-3


def test_single_token():
    from batchgen_kernels.common.v4_hyper_connections import hc_post, hc_pre

    hidden_states, fn_weight, scale, base = _make_inputs(
        T=1, hidden=4096, seed=5
    )
    reduced, post, comb = hc_pre(
        hidden_states, fn_weight, scale, base, 4, 20, 1e-6, 1e-6
    )
    ref_reduced, ref_post_out, ref_comb = _ref_pre(
        hidden_states, fn_weight, scale, base
    )
    output = hc_post(reduced, hidden_states, post, comb)
    ref_output = _ref_post(ref_reduced, hidden_states, ref_post_out, ref_comb)

    assert torch.allclose(reduced, ref_reduced, atol=1e-3)
    assert torch.allclose(post, ref_post_out, atol=1e-3)
    assert torch.allclose(comb, ref_comb, atol=1e-3)
    assert torch.allclose(output, ref_output, atol=1e-3)


def test_all_zero_hidden():
    from batchgen_kernels.common.v4_hyper_connections import hc_pre

    _, fn_weight, scale, base = _make_inputs(T=32, hidden=4096, seed=6)
    hidden_states = torch.zeros(
        1, 32, 4, 4096, device="cuda", dtype=torch.bfloat16
    )

    reduced, post, comb = hc_pre(
        hidden_states, fn_weight, scale, base, 4, 20, 1e-6, 1e-6
    )
    ref_reduced, ref_post_out, ref_comb = _ref_pre(
        hidden_states, fn_weight, scale, base
    )

    assert torch.count_nonzero(reduced).item() == 0
    assert torch.allclose(reduced, ref_reduced, atol=1e-3)
    assert torch.allclose(post, ref_post_out, atol=1e-3)
    assert torch.allclose(comb, ref_comb, atol=1e-3)


def test_comb_shape():
    from batchgen_kernels.common.v4_hyper_connections import hc_split

    mixes, scale, base = _make_split_inputs(T=8, seed=7)
    _, _, comb = hc_split(mixes, scale, base, 4, 20, 1e-6)

    assert comb.shape == (1, 8, 4, 4)


def test_sinkhorn_zero_iters():
    from batchgen_kernels.common.v4_hyper_connections import hc_split

    mixes, scale, base = _make_split_inputs(T=32, seed=8)
    pre, post, comb = hc_split(mixes, scale, base, 4, 0, 1e-6)

    expected_pre = torch.sigmoid(mixes[..., :4] * scale[0] + base[:4]) + 1e-6
    expected_post = 2 * torch.sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
    comb_base = base[8:].view(4, 4)
    expected_comb = mixes[..., 8:].view(1, 32, 4, 4)
    expected_comb = (
        torch.softmax(expected_comb * scale[2] + comb_base, dim=-1) + 1e-6
    )
    expected_comb = expected_comb / (
        expected_comb.sum(dim=-2, keepdim=True) + 1e-6
    )

    assert torch.allclose(pre, expected_pre, atol=1e-3)
    assert torch.allclose(post, expected_post, atol=1e-3)
    assert torch.allclose(comb, expected_comb, atol=1e-3)


@pytest.mark.parametrize("T", [1, 32, 128, 1024])
def test_flash_shape(T):
    from batchgen_kernels.common.v4_hyper_connections import hc_post, hc_pre

    hidden_states, fn_weight, scale, base = _make_inputs(T, 4096, seed=9 + T)
    reduced, post, comb = hc_pre(
        hidden_states, fn_weight, scale, base, 4, 20, 1e-6, 1e-6
    )
    output = hc_post(reduced, hidden_states, post, comb)

    assert reduced.shape == (1, T, 4096)
    assert post.shape == (1, T, 4)
    assert comb.shape == (1, T, 4, 4)
    assert output.shape == (1, T, 4, 4096)


@pytest.mark.parametrize("T", [1, 32, 128, 1024])
def test_pro_shape(T):
    from batchgen_kernels.common.v4_hyper_connections import hc_post, hc_pre

    hidden_states, fn_weight, scale, base = _make_inputs(T, 7168, seed=19 + T)
    reduced, post, comb = hc_pre(
        hidden_states, fn_weight, scale, base, 4, 20, 1e-6, 1e-6
    )
    output = hc_post(reduced, hidden_states, post, comb)

    assert reduced.shape == (1, T, 7168)
    assert post.shape == (1, T, 4)
    assert comb.shape == (1, T, 4, 4)
    assert output.shape == (1, T, 4, 7168)


@pytest.mark.parametrize("T", [1, 128, 1024])
@pytest.mark.parametrize("hidden", [4096, 7168])
def test_benchmark(T, hidden):
    from batchgen_kernels.common.v4_hyper_connections import hc_post, hc_pre
    from tests.kernels.conftest import _bench, disable_tf32

    hidden_states, fn_weight, scale, base = _make_inputs(
        T, hidden, seed=1000 + T + hidden
    )

    def standalone():
        reduced, post, comb = hc_pre(
            hidden_states, fn_weight, scale, base, 4, 20, 1e-6, 1e-6
        )
        return hc_post(reduced, hidden_states, post, comb)

    def reference():
        reduced, post, comb = _ref_pre(hidden_states, fn_weight, scale, base)
        return _ref_post(reduced, hidden_states, post, comb)

    with disable_tf32():
        standalone_ms = _bench(standalone)
        reference_ms = _bench(reference)
    print(
        f"\nK12 benchmark T={T} hidden={hidden} standalone={standalone_ms:.3f} ms ref={reference_ms:.3f} ms"
    )

    assert standalone_ms > 0
    assert reference_ms > 0
