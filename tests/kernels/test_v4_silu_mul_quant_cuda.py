import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _ref_silu_mul_quant(
    gate: torch.Tensor, up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    out = F.silu(gate.float()) * up.float()
    scale = out.abs().amax(dim=-1) / fp8_max
    safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x_fp8 = torch.clamp(
        out / safe_scale.unsqueeze(-1), min=-fp8_max, max=fp8_max
    ).to(torch.float8_e4m3fn)
    return x_fp8, safe_scale


@pytest.mark.parametrize("T", [128, 1024])
@pytest.mark.parametrize("D", [2048, 3072])
def test_fused_silu_mul_quant_cuda_vs_baseline(T, D):
    from batchgen_kernels.moe.silu_mul_quant import fused_silu_mul_quant_cuda

    torch.manual_seed(T * 10000 + D)
    gate = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)

    out_cuda, scales_cuda = fused_silu_mul_quant_cuda(gate, up)
    out_ref, scales_ref = _ref_silu_mul_quant(gate, up)

    cuda_deq = out_cuda.float() * scales_cuda.unsqueeze(-1)
    ref_deq = out_ref.float() * scales_ref.unsqueeze(-1)

    torch.testing.assert_close(cuda_deq, ref_deq, atol=0.05, rtol=0.01)


@pytest.mark.parametrize("T", [128, 1024])
@pytest.mark.parametrize("D", [2048, 3072])
def test_fused_silu_mul_quant_cuda_vs_pytorch_baseline(T, D):
    from batchgen_kernels.moe.silu_mul_quant import fused_silu_mul_quant_cuda
    from batchgen_kernels.moe.v4_fused_silu_mul_quant import (
        fused_silu_mul_quant,
    )

    torch.manual_seed(T * 10000 + D)
    gate = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)

    out_cuda, scales_cuda = fused_silu_mul_quant_cuda(gate, up)
    out_ref, scales_ref = fused_silu_mul_quant(
        gate, up, swiglu_limit=0.0, quantize=True
    )

    cuda_deq = out_cuda.float() * scales_cuda.unsqueeze(-1)
    ref_deq = out_ref.float() * scales_ref.unsqueeze(-1)

    torch.testing.assert_close(cuda_deq, ref_deq, atol=0.05, rtol=0.01)


@pytest.mark.parametrize("T", [128, 1024])
@pytest.mark.parametrize("D", [2048, 3072])
def test_output_shapes_and_dtypes(T, D):
    from batchgen_kernels.moe.silu_mul_quant import fused_silu_mul_quant_cuda

    gate = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(T, D, device="cuda", dtype=torch.bfloat16)

    out, scales = fused_silu_mul_quant_cuda(gate, up)

    assert out.shape == (T, D)
    assert out.dtype == torch.float8_e4m3fn
    assert scales.shape == (T,)
    assert scales.dtype == torch.float32
    assert (scales > 0).all()
