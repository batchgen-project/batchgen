import ast
from pathlib import Path

import pytest
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]


def _isolated_rmsnorm():
    path = (
        ROOT
        / "batchgen"
        / "models"
        / "moonshotai"
        / "kimi_linear"
        / "model.py"
    )
    tree = ast.parse(path.read_text())
    klass = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "KimiRMSNorm"
    )
    module = ast.Module(body=[klass], type_ignores=[])
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["KimiRMSNorm"]


@pytest.mark.parametrize("shape", [(2, 17), (3, 5, 17), (65, 17)])
def test_resident_prefill_rmsnorm_tile_is_exact(shape):
    KimiRMSNorm = _isolated_rmsnorm()
    torch.manual_seed(260820)
    norm = KimiRMSNorm(shape[-1], eps=1e-6)
    norm.weight.data.copy_(torch.randn_like(norm.weight))
    x = torch.randn(shape, dtype=torch.bfloat16)

    expected = norm(x)
    norm._resident_prefill_token_tile = 8
    actual = norm(x)

    assert torch.equal(actual, expected)


def test_resident_prefill_rmsnorm_tile_keeps_shape_and_dtype():
    KimiRMSNorm = _isolated_rmsnorm()
    norm = KimiRMSNorm(17).to(dtype=torch.bfloat16)
    norm._resident_prefill_token_tile = 8
    x = torch.randn((2, 33, 17), dtype=torch.bfloat16)

    actual = norm(x)

    assert actual.shape == x.shape
    assert actual.dtype == x.dtype
