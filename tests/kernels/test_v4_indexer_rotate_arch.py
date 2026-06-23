from __future__ import annotations

import pytest
import torch

import batchgen_kernels.attention.v4_compressor as comp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def test_fp8_fake_quant_blockwise_finite_and_inplace():
    torch.manual_seed(0)
    q = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    before = q.clone()
    comp._fp8_fake_quant_blockwise(q, 32)
    assert q.shape == before.shape
    assert torch.isfinite(q.float()).all()
    assert not torch.equal(q, before)


def test_maybe_rotate_uses_fp8_fallback_on_sm90(monkeypatch):
    monkeypatch.setattr(comp, "_supports_mxfp4", lambda: False)

    called = {"fp8": 0}
    real = comp._fp8_fake_quant_blockwise

    def spy(q, bs):
        called["fp8"] += 1
        return real(q, bs)

    monkeypatch.setattr(comp, "_fp8_fake_quant_blockwise", spy)

    compressor = comp.DeepSeekV4Compressor(
        64, 128, 64, 4, 1e-6, overlap=False, rotate=True
    ).cuda()
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    out = compressor._maybe_rotate(x)

    assert called["fp8"] == 1
    assert out.shape == x.shape
    assert torch.isfinite(out.float()).all()


def test_maybe_rotate_uses_fp4_on_sm120(monkeypatch):
    if torch.cuda.get_device_capability()[0] < 12:
        pytest.skip("fp4_act_quant requires sm120")
    pytest.importorskip("tilelang", reason="fp4_act_quant requires tilelang")
    monkeypatch.setattr(comp, "_supports_mxfp4", lambda: True)

    fp8_calls = {"n": 0}
    monkeypatch.setattr(
        comp,
        "_fp8_fake_quant_blockwise",
        lambda q, bs: fp8_calls.__setitem__("n", fp8_calls["n"] + 1),
    )

    compressor = comp.DeepSeekV4Compressor(
        64, 128, 64, 4, 1e-6, overlap=False, rotate=True
    ).cuda()
    x = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    out = compressor._maybe_rotate(x)

    assert fp8_calls["n"] == 0
    assert out.shape == x.shape
    assert torch.isfinite(out.float()).all()
