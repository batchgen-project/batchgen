"""Triton SiTU-and-multiply vs the eager SituAndMul on CUDA (skipped without a GPU)."""
import pytest
import torch
import torch.nn.functional as F

model = pytest.importorskip("batchgen.models.moonshotai.kimi_linear.model")  # needs fla
SituAndMul = model.SituAndMul


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("linear_beta", [None, 25.0])
@pytest.mark.parametrize("rows,d", [(1, 2048), (7, 1536), (64, 3072), (301, 1024)])
def test_situ_triton_matches_eager(linear_beta, rows, d):
    from batchgen.models.moonshotai.kimi_linear.situ_triton import situ_triton_available
    if not situ_triton_available():
        pytest.skip("triton unavailable")
    torch.manual_seed(0)
    act = SituAndMul(beta=4.0, linear_beta=linear_beta)
    x = (torch.randn(rows, 2 * d, device="cuda") * 3).to(torch.bfloat16)
    got = act(x)
    ref = act._forward_eager(x)
    assert got.shape == ref.shape and got.dtype == ref.dtype
    # fp32 math on both sides, one bf16 rounding; tanh formulations may differ by an ulp
    diff = (got.float() - ref.float()).abs()
    tol = ref.float().abs().max().item() * 2 ** -7 + 1e-3
    assert diff.max().item() <= tol, diff.max().item()
    assert (got != ref).float().mean().item() < 0.02


def test_situ_eager_path_on_cpu():
    act = SituAndMul(beta=4.0, linear_beta=25.0)
    x = torch.randn(3, 8)
    assert torch.equal(act(x), act._forward_eager(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_gate_up_matches_two_gemms():
    from types import SimpleNamespace
    KimiMLP = model.KimiMLP
    cfg = SimpleNamespace(hidden_size=64, intermediate_size=48, hidden_act="situ",
                          activation_situ_beta=4.0, activation_situ_linear_beta=25.0)
    mlp = KimiMLP(cfg).to("cuda", torch.bfloat16)
    x = torch.randn(9, 64, device="cuda", dtype=torch.bfloat16)
    ref = mlp.down_proj(mlp.act_fn(torch.cat([mlp.gate_proj(x), mlp.up_proj(x)], dim=-1)))
    got = mlp._ffn(x)
    assert mlp.gate_proj.weight.data_ptr() == mlp._gate_up_fused.data_ptr()
    assert torch.allclose(got.float(), ref.float(), atol=2e-2, rtol=2e-2)
    # the projections still work on their own (views of the slab)
    assert torch.allclose(mlp.gate_proj(x).float(), F.linear(x, mlp._gate_up_fused[:48]).float())
