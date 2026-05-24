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


def _ref_silu_mul(
    gate: torch.Tensor,
    up: torch.Tensor,
    swiglu_limit: float = 10.0,
) -> torch.Tensor:
    gate_f32 = gate.float()
    up_f32 = up.float()
    if swiglu_limit > 0:
        gate_f32 = torch.clamp(gate_f32, max=swiglu_limit)
        up_f32 = torch.clamp(up_f32, min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate_f32) * up_f32


def _dequantize_per_token(
    x_fp8: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    return x_fp8.float() * scale.unsqueeze(-1)


@pytest.mark.parametrize("T", [1, 32, 128, 1024])
@pytest.mark.parametrize("inter", [2048, 3072])
def test_silu_mul_matches_pytorch(T, inter):
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import fused_silu_mul_quant

    torch.manual_seed(T * 10000 + inter)
    gate = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)

    out = fused_silu_mul_quant(gate, up)
    expected = _ref_silu_mul(gate, up).to(torch.bfloat16)

    assert torch.allclose(out.float(), expected.float(), atol=1e-2, rtol=1e-2)


def test_gate_clamp_max10():
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import fused_silu_mul_quant

    gate = torch.linspace(
        -20.0, 20.0, 128, device="cuda", dtype=torch.float32
    ).repeat(4, 1)
    up = torch.ones_like(gate)

    out = fused_silu_mul_quant(gate, up, swiglu_limit=10.0)

    assert out.float().max().item() <= F.silu(torch.tensor(10.0)).item() + 1e-3


def test_up_clamp_range():
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import fused_silu_mul_quant

    gate = torch.ones(8, 128, device="cuda", dtype=torch.float32)
    up = torch.linspace(
        -20.0, 20.0, 128, device="cuda", dtype=torch.float32
    ).repeat(8, 1)

    out = fused_silu_mul_quant(gate, up, swiglu_limit=10.0)
    expected = F.silu(torch.ones_like(up)) * up.clamp(-10.0, 10.0)

    assert torch.equal(out, expected.to(torch.bfloat16))


def test_post_quant_fp8():
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import fused_silu_mul_quant

    T = 128
    inter = 2048
    torch.manual_seed(0)
    gate = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)

    out_fp8, scale = fused_silu_mul_quant(gate, up, quantize=True)
    restored = _dequantize_per_token(out_fp8, scale)
    expected = _ref_silu_mul(gate, up)

    assert out_fp8.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.float32
    assert scale.shape == (T,)
    assert torch.allclose(restored, expected, atol=0.05, rtol=0.05)


def test_matches_model_ref():
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import fused_silu_mul_quant

    T = 32
    inter = 2048
    torch.manual_seed(1)
    gate = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)

    out = fused_silu_mul_quant(gate, up, swiglu_limit=10.0)
    expected = _ref_silu_mul(gate, up, swiglu_limit=10.0).to(torch.bfloat16)

    assert torch.equal(out, expected)


def test_input_bf16_output():
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import fused_silu_mul_quant

    gate = torch.randn(32, 2048, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(32, 2048, device="cuda", dtype=torch.bfloat16)

    out = fused_silu_mul_quant(gate, up)

    assert out.dtype == torch.bfloat16


def test_gate_zero_output_zero():
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import fused_silu_mul_quant

    gate = torch.zeros(32, 2048, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(32, 2048, device="cuda", dtype=torch.bfloat16)

    out = fused_silu_mul_quant(gate, up)

    assert torch.count_nonzero(out).item() == 0


def test_up_zero_output_zero():
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import fused_silu_mul_quant

    gate = torch.randn(32, 2048, device="cuda", dtype=torch.bfloat16)
    up = torch.zeros(32, 2048, device="cuda", dtype=torch.bfloat16)

    out = fused_silu_mul_quant(gate, up)

    assert torch.count_nonzero(out).item() == 0


def test_no_clamping_limit_zero():
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import fused_silu_mul_quant

    torch.manual_seed(2)
    gate = torch.randn(32, 2048, device="cuda", dtype=torch.bfloat16) * 20
    up = torch.randn(32, 2048, device="cuda", dtype=torch.bfloat16) * 20

    out = fused_silu_mul_quant(gate, up, swiglu_limit=0.0)
    expected = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)

    assert torch.equal(out, expected)


@pytest.mark.parametrize("T", [1, 128, 1024])
def test_flash_inter_size(T):
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import fused_silu_mul_quant

    inter = 2048
    gate = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)

    out = fused_silu_mul_quant(gate, up)

    assert out.shape == (T, inter)


@pytest.mark.parametrize("T", [1, 128, 1024])
def test_pro_inter_size(T):
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import fused_silu_mul_quant

    inter = 3072
    gate = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)

    out = fused_silu_mul_quant(gate, up)

    assert out.shape == (T, inter)


@pytest.mark.parametrize("T", [128, 1024])
@pytest.mark.parametrize("inter", [2048, 3072])
def test_benchmark(T, inter):
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import fused_silu_mul_quant
    from tests.kernels.conftest import _bench

    torch.manual_seed(T * 1000 + inter)
    gate = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(T, inter, device="cuda", dtype=torch.bfloat16)

    def separate():
        return _ref_silu_mul(gate, up).to(torch.bfloat16)

    fused_ms = _bench(fused_silu_mul_quant, gate, up)
    separate_ms = _bench(separate)
    print(
        f"\nK14 benchmark T={T} inter={inter} fused={fused_ms:.3f} ms separate={separate_ms:.3f} ms"
    )

    assert fused_ms > 0
    assert separate_ms > 0
