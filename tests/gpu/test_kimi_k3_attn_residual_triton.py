# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                          #
#  copyright (c) EfficientMoE team 2025                                        #
#  Licensed under the Apache License, Version 2.0                              #
# ---------------------------------------------------------------------------- #
"""GPU parity for the Kimi-K3 Triton Block-Attention-Residual mixer."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest
import torch


if os.environ.get("K3_GPU_STAGE") == "1" and not torch.cuda.is_available():
    raise RuntimeError("K3_GPU_STAGE=1 requires a visible CUDA device")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="staged Kimi-K3 CUDA test"
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = (
    ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"
)
ALIAS = "_k3_attn_residual_gpu"
package = types.ModuleType(ALIAS)
package.__path__ = [str(PACKAGE_DIR)]
sys.modules.setdefault(ALIAS, package)
block_residual = importlib.import_module(f"{ALIAS}.block_residual")
attn_residual_triton = importlib.import_module(
    f"{ALIAS}.attn_residual_triton"
)


class _Norm(torch.nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(hidden))
        self.variance_epsilon = 1e-5
        self._resident_prefill_token_tile = None


def _inputs(tokens: int, bank_rows: int, dtype=torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(
        20260902 + tokens * 17 + bank_rows
    )
    prefix = torch.randn(
        (tokens, 7168), generator=generator, device="cuda", dtype=dtype
    ).mul_(0.1)
    # Allocate the full production bank, then narrow it.  This preserves the
    # larger token stride used by BlockResidualBuffer and catches kernels that
    # incorrectly assume a tightly packed [T, nvb, H] view.
    full_bank = torch.randn(
        (tokens, 8, 7168), generator=generator, device="cuda", dtype=dtype
    ).mul_(0.1)
    bank = full_bank[:, :bank_rows, :]
    proj = torch.nn.Linear(7168, 1, bias=False, device="cuda", dtype=dtype)
    norm = _Norm(7168).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        proj.weight.normal_(mean=0.0, std=0.02, generator=generator)
        norm.weight.normal_(mean=1.0, std=0.05, generator=generator)
    return prefix, bank, proj, norm


@pytest.mark.parametrize(
    "tokens,bank_rows",
    [
        (1, 1),
        (3, 8),
        (7, 4),
        (16, 8),
        (61, 1),
        (64, 8),
        (257, 4),
        (1024, 8),
        (8192, 8),
    ],
)
def test_triton_mixer_bf16_parity(tokens, bank_rows):
    prefix, bank, proj, norm = _inputs(tokens, bank_rows)
    with torch.inference_mode():
        expected = block_residual._apply_attn_res_eager(
            prefix, bank, proj, norm
        )
        actual = attn_residual_triton.mix_attn_residual_triton(
            prefix, bank, proj, norm
        )
    torch.cuda.synchronize()

    difference = (actual.float() - expected.float()).abs()
    tolerance = 1e-5 + 1.6e-2 * expected.float().abs()
    num_fail = int((difference > tolerance).sum().item())
    numel = actual.numel()
    fail_fraction = num_fail / numel
    print(
        f"T={tokens} nvb={bank_rows} max_abs={difference.max().item():.8g} "
        f"fail={num_fail}/{numel} ({fail_fraction:.3e})"
    )
    assert num_fail == 0 or fail_fraction < 1e-4


def test_triton_mixer_score_weight_tracks_current_parameter_values():
    prefix, bank, proj, norm = _inputs(7, 4)
    with torch.inference_mode():
        first = attn_residual_triton.mix_attn_residual_triton(
            prefix, bank, proj, norm
        )
        proj.weight.add_(0.25)
        second = attn_residual_triton.mix_attn_residual_triton(
            prefix, bank, proj, norm
        )
        expected = block_residual._apply_attn_res_eager(
            prefix, bank, proj, norm
        )
    assert not torch.equal(first, second)
    torch.testing.assert_close(second, expected, atol=1e-5, rtol=1.6e-2)


def test_triton_mixer_normalizes_before_overflowing_score_dot():
    prefix, bank, proj, norm = _inputs(4, 1)
    with torch.no_grad():
        prefix.fill_(1e38)
        bank.fill_(1e38)
        # Make a pre-normalization value*coefficient product overflow too.
        # The eager path first obtains rrms=0, so its normalized score remains
        # exactly zero and the two equal rows still average to the input.
        proj.weight.fill_(4)
        norm.weight.fill_(1)

    with torch.inference_mode():
        expected = block_residual._apply_attn_res_eager(
            prefix, bank, proj, norm
        )
        actual = attn_residual_triton.mix_attn_residual_triton(
            prefix, bank, proj, norm
        )
    torch.cuda.synchronize()

    assert torch.isfinite(expected).all()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
