from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from batchgen.models.glm.glm5.model import Glm5MLP


@pytest.mark.parametrize("fp8_path", [False, True])
def test_dense_mlp_long_prefill_matches_full_forward(monkeypatch, fp8_path):
    torch.manual_seed(7)
    mlp = Glm5MLP(SimpleNamespace(hidden_size=8, intermediate_size=12))
    x = torch.randn(1, 262_145, 8)

    if fp8_path:
        from batchgen.attention.mla import fa3_backend

        for name in ("gate_scale", "up_scale", "down_scale"):
            setattr(mlp, name, torch.ones(1))
        monkeypatch.setattr(
            fa3_backend,
            "w8a16_gemm",
            lambda weight, _scale, activation: F.linear(activation, weight),
        )

    expected = mlp._forward_slice(x)
    actual = mlp(x)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
