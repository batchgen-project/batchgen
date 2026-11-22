"""``act_quant`` over zero rows launches no kernel.

A bf16 input reaches the legacy Triton fallback only when it has no rows; no
warm-up exercises that specialisation, so launching it would load a kernel
after readiness on the first rank with an empty batch.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from batchgen.quantization import act_quant as mod  # noqa: E402


class _NoLaunch:
    def __getitem__(self, grid):
        raise AssertionError(f"kernel launched over grid {grid} for an input with no rows")


@pytest.mark.parametrize("shape, tma, scale_shape", [
    ((0, 256), False, (0, 2)),
    ((0, 300), False, (0, 3)),
    ((0, 256), True, (0, 2)),
    ((0, 4, 256), False, (0, 4, 2)),
])
def test_zero_rows_return_empty_outputs_without_a_kernel_launch(monkeypatch, shape, tma, scale_shape):
    monkeypatch.setattr(mod, "act_quant_kernel_2d", _NoLaunch())
    x = torch.empty(shape, dtype=torch.bfloat16)
    y, scale = mod.act_quant(x, scale_tma_aligned=tma)
    assert y.shape == x.shape and y.dtype == torch.float8_e4m3fn
    assert tuple(scale.shape) == scale_shape and scale.dtype == torch.float32
