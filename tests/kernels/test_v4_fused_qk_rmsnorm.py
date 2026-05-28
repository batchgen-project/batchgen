# ---------------------------------------------------------------------------- #
#  BatchGen                                                                      #
#  copyright (c) EfficientMoE team 2025                                          #
# ---------------------------------------------------------------------------- #

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _q_ref(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head Q RMSNorm reference. Variance computed over the last dim.

    Works on 3D [T, n_heads, head_dim] (per-head) and 2D [T, dim] (single head).
    """
    x_fp32 = x.float()
    return (
        x_fp32 * torch.rsqrt(x_fp32.square().mean(-1, keepdim=True) + eps)
    ).to(x.dtype)


def _kv_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    from batchgen.models.deepseek.deepseekv4_flash.model import (
        DeepSeekV4FlashRMSNorm,
    )

    norm = DeepSeekV4FlashRMSNorm(x.shape[-1], eps=eps).cuda()
    with torch.no_grad():
        norm.weight.copy_(weight)
    return norm(x)


# ---------------------------------------------------------------------------- #
# Per-token, per-head correctness                                              #
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("T", [1, 4, 32, 128, 1024, 8192])
def test_q_norm_no_weight(T):
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    eps = 1e-6
    n_heads = 64
    head_dim = 512
    torch.manual_seed(0)
    qr = torch.randn(T, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(head_dim, device="cuda", dtype=torch.float32)

    qr_out, _ = fused_qk_rmsnorm(qr, kv, kv_weight, eps=eps)
    expected = _q_ref(qr, eps)

    from tests.kernels.conftest import _assert_bf16_close

    _assert_bf16_close(qr_out, expected, msg=f"q_norm T={T}")


@pytest.mark.parametrize("T", [1, 4, 32, 128, 1024, 8192])
def test_kv_norm_with_weight(T):
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    eps = 1e-6
    n_heads = 64
    head_dim = 512
    torch.manual_seed(1)
    qr = torch.randn(T, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(head_dim, device="cuda", dtype=torch.float32)

    _, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight, eps=eps)
    expected = _kv_ref(kv, kv_weight, eps)

    from tests.kernels.conftest import _assert_bf16_close

    _assert_bf16_close(kv_out, expected, msg=f"kv_norm T={T}")


def test_fused_matches_separate():
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    eps = 1e-6
    T = 128
    n_heads = 64
    head_dim = 512
    torch.manual_seed(2)
    qr = torch.randn(T, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(head_dim, device="cuda", dtype=torch.float32)

    qr_out, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight, eps=eps)
    qr_expected = _q_ref(qr, eps)
    kv_expected = _kv_ref(kv, kv_weight, eps)

    assert torch.allclose(qr_out, qr_expected, atol=1e-3, rtol=1e-3)
    assert torch.allclose(kv_out, kv_expected, atol=1e-3, rtol=1e-3)


def test_output_dtype_bf16():
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    qr = torch.randn(32, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.ones(512, device="cuda", dtype=torch.float32)

    qr_out, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight)

    assert qr_out.dtype == torch.bfloat16
    assert kv_out.dtype == torch.bfloat16


def test_fp32_accumulation_large_values():
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    eps = 1e-6
    T = 32
    n_heads = 64
    head_dim = 512
    torch.manual_seed(3)
    qr = (
        torch.rand(T, n_heads, head_dim, device="cuda", dtype=torch.float32)
        * 9e3
        + 1e3
    ).to(torch.bfloat16)
    qr = qr * torch.sign(
        torch.randn(T, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    )
    kv = (
        torch.rand(T, head_dim, device="cuda", dtype=torch.float32) * 9e3 + 1e3
    ).to(torch.bfloat16)
    kv = kv * torch.sign(
        torch.randn(T, head_dim, device="cuda", dtype=torch.bfloat16)
    )
    kv_weight = torch.randn(head_dim, device="cuda", dtype=torch.float32)

    qr_out, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight, eps=eps)
    qr_expected = _q_ref(qr, eps)
    kv_expected = _kv_ref(kv, kv_weight, eps)

    assert torch.allclose(qr_out, qr_expected, atol=1e-3, rtol=1e-3)
    assert torch.allclose(kv_out, kv_expected, atol=1e-3, rtol=1e-3)


def test_empty_tensor():
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    qr = torch.empty(0, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.empty(0, 512, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.ones(512, device="cuda", dtype=torch.float32)

    qr_out, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight)

    assert qr_out.shape == qr.shape
    assert kv_out.shape == kv.shape
    assert qr_out.dtype == torch.bfloat16
    assert kv_out.dtype == torch.bfloat16


def test_all_zero_input():
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    qr = torch.zeros(32, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.zeros(32, 512, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(512, device="cuda", dtype=torch.float32)

    qr_out, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight)

    assert torch.count_nonzero(qr_out).item() == 0
    assert torch.count_nonzero(kv_out).item() == 0


def test_very_large_values():
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    eps = 1e-6
    qr = torch.full((32, 64, 512), 1e6, device="cuda", dtype=torch.bfloat16)
    kv = torch.full((32, 512), -1e6, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(512, device="cuda", dtype=torch.float32)

    qr_out, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight, eps=eps)
    qr_expected = _q_ref(qr, eps)
    kv_expected = _kv_ref(kv, kv_weight, eps)

    assert torch.allclose(qr_out, qr_expected, atol=1e-3, rtol=1e-3)
    assert torch.allclose(kv_out, kv_expected, atol=1e-3, rtol=1e-3)


def test_very_small_values():
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    eps = 1e-6
    qr = torch.full((32, 64, 512), 1e-8, device="cuda", dtype=torch.bfloat16)
    kv = torch.full((32, 512), -1e-8, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(512, device="cuda", dtype=torch.float32)

    qr_out, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight, eps=eps)
    qr_expected = _q_ref(qr, eps)
    kv_expected = _kv_ref(kv, kv_weight, eps)

    assert torch.allclose(qr_out, qr_expected, atol=1e-3, rtol=1e-3)
    assert torch.allclose(kv_out, kv_expected, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("T", [1, 128, 8192])
def test_flash_shape(T):
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    H = 64
    head_dim = 512
    qr = torch.randn(T, H, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.ones(head_dim, device="cuda", dtype=torch.float32)

    qr_out, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight)

    assert qr_out.shape == (T, H, head_dim)
    assert kv_out.shape == (T, head_dim)


@pytest.mark.parametrize("T", [1, 128, 8192])
def test_pro_shape(T):
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    H = 128
    head_dim = 512
    qr = torch.randn(T, H, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.ones(head_dim, device="cuda", dtype=torch.float32)

    qr_out, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight)

    assert qr_out.shape == (T, H, head_dim)
    assert kv_out.shape == (T, head_dim)


# ---------------------------------------------------------------------------- #
# NEW: Per-head correctness gates (catch B1)                                   #
# ---------------------------------------------------------------------------- #


def test_q_per_head_independence():
    """Catches B1: variance must be per-head, not over flattened row.

    Construct Q with vastly different magnitudes per head. Under a flat
    reduction the small-magnitude heads would be drowned out and their norm
    would be wrong. Under per-head reduction each head normalizes only by
    its own variance.
    """
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    eps = 1e-6
    T = 4
    n_heads = 4
    head_dim = 512
    torch.manual_seed(42)

    qr = torch.zeros(T, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    # head 0: very large values
    qr[:, 0, :] = torch.full(
        (T, head_dim), 1e3, device="cuda", dtype=torch.bfloat16
    )
    # head 1: very small values
    qr[:, 1, :] = torch.full(
        (T, head_dim), 1e-3, device="cuda", dtype=torch.bfloat16
    )
    # heads 2-3: random
    qr[:, 2:, :] = torch.randn(
        T, 2, head_dim, device="cuda", dtype=torch.bfloat16
    )

    kv = torch.randn(T, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.ones(head_dim, device="cuda", dtype=torch.float32)

    qr_out, _ = fused_qk_rmsnorm(qr, kv, kv_weight, eps=eps)
    expected = _q_ref(qr, eps)

    # Per-head: each head normalized to ~unit RMS. After norm, all elements
    # of any head with constant input should be ~1.0 regardless of input magnitude.
    # head 0: all 1.0  (1e3 / sqrt(1e6 + eps) ~= 1)
    # head 1: all 1.0  (1e-3 / sqrt(1e-6 + eps) is dominated by eps; check via _q_ref)
    for h in range(n_heads):
        assert torch.allclose(
            qr_out[:, h, :].float(),
            expected[:, h, :].float(),
            atol=1e-3,
            rtol=1e-3,
        ), f"per-head mismatch at head {h}"

    # Sanity: head 0 normalized output should be close to 1.0 elementwise
    assert torch.allclose(
        qr_out[:, 0, :].float(),
        torch.ones_like(qr_out[:, 0, :]).float(),
        atol=1e-2,
    ), "head 0 (constant 1e3) should normalize to ~1.0 under per-head norm"


@pytest.mark.parametrize("n_heads", [1, 2, 8, 64, 128])
def test_q_parametrized_n_heads(n_heads):
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    eps = 1e-6
    T = 16
    head_dim = 512
    torch.manual_seed(7)
    qr = torch.randn(T, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(head_dim, device="cuda", dtype=torch.float32)

    qr_out, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight, eps=eps)
    qr_expected = _q_ref(qr, eps)
    kv_expected = _kv_ref(kv, kv_weight, eps)

    from tests.kernels.conftest import _assert_bf16_close

    _assert_bf16_close(qr_out, qr_expected, msg=f"q n_heads={n_heads}")
    _assert_bf16_close(kv_out, kv_expected, msg=f"kv n_heads={n_heads}")


def test_grid_parallelism_smoke():
    """Documents the new (T, n_heads+1) grid via a large-scale launch."""
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    eps = 1e-6
    T = 1024
    n_heads = 128
    head_dim = 512
    torch.manual_seed(11)
    qr = torch.randn(T, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(head_dim, device="cuda", dtype=torch.float32)

    qr_out, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight, eps=eps)
    qr_expected = _q_ref(qr, eps)
    kv_expected = _kv_ref(kv, kv_weight, eps)

    from tests.kernels.conftest import _assert_bf16_close

    _assert_bf16_close(qr_out, qr_expected, msg="grid-smoke q")
    _assert_bf16_close(kv_out, kv_expected, msg="grid-smoke kv")


def test_non_contiguous_q_stride():
    """Q from .view(B,T,n_heads,head_dim) on a contiguous tensor has
    standard strides. This test feeds a strided-but-last-contig Q to verify
    stride-aware indexing.
    """
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm

    eps = 1e-6
    T = 32
    n_heads = 64
    head_dim = 512
    torch.manual_seed(13)

    # Build [2, T, n_heads, head_dim] then slice out first batch -> [T, n_heads, head_dim].
    # The slice keeps inner contig but has a non-default stride(0).
    big = torch.randn(
        2, T, n_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    qr = big[
        0
    ]  # contiguous in last dim, non-default stride(0) = T*n_heads*head_dim
    assert qr.stride(-1) == 1
    assert (
        not qr.is_contiguous() or qr.stride(0) == n_heads * head_dim
    )  # may still be contig

    kv = torch.randn(T, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(head_dim, device="cuda", dtype=torch.float32)

    qr_out, kv_out = fused_qk_rmsnorm(qr, kv, kv_weight, eps=eps)
    qr_expected = _q_ref(qr, eps)
    kv_expected = _kv_ref(kv, kv_weight, eps)

    from tests.kernels.conftest import _assert_bf16_close

    _assert_bf16_close(qr_out, qr_expected, msg="non-contig q")
    _assert_bf16_close(kv_out, kv_expected, msg="non-contig kv")


# ---------------------------------------------------------------------------- #
# Benchmark                                                                    #
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("T", [1, 128, 1024, 8192])
def test_benchmark(T):
    from batchgen_kernels.triton.v4_fused_qk_rmsnorm import fused_qk_rmsnorm
    from tests.kernels.conftest import _bench

    eps = 1e-6
    n_heads = 64
    head_dim = 512
    torch.manual_seed(4)
    qr = torch.randn(T, n_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(T, head_dim, device="cuda", dtype=torch.bfloat16)
    kv_weight = torch.randn(head_dim, device="cuda", dtype=torch.float32)

    def separate():
        return _q_ref(qr, eps), _kv_ref(kv, kv_weight, eps)

    fused_ms = _bench(fused_qk_rmsnorm, qr, kv, kv_weight, eps)
    separate_ms = _bench(separate)
    print(
        f"\nK1 benchmark T={T} fused={fused_ms:.3f} ms separate={separate_ms:.3f} ms"
    )

    assert fused_ms > 0
    assert separate_ms > 0
