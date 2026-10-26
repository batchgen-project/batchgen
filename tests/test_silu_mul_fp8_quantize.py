"""Fused SiLU(gate) * up + per-token 128-block FP8 quantize (Triton).

``silu_mul_per_token_blocked_quantize_bf16_to_fp8_1d`` replaces the
``F.silu(gate) * up`` pair plus ``per_token_blocked_quantize_bf16_to_fp8_1d``
that feeds a down-projection GEMM. Its contract is bit-identity with that
unfused sequence on the same bf16 [gate | up] rows, in both scale layouts
(row-major, and the MN-major TMA-aligned layout DeepGEMM consumes without a
transpose). GPU-only; a skip on the node is a gate failure.
"""

import pytest
import torch
import torch.nn.functional as F

INTER = 2048          # GLM-5.2 shared-expert intermediate size
ROWS = (1, 3, 4, 7, 8, 16, 31, 32, 61, 64, 96, 128, 256)


def _cuda_ready():
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    try:
        import triton  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"triton unavailable: {exc!r}"
    return None


_skip = pytest.mark.skipif(_cuda_ready() is not None, reason=str(_cuda_ready()))


def _mismatch(a, b):
    a8, b8 = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    return int((a8 != b8).sum())


def _gate_up_input(rows, seed, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, 2 * INTER, generator=g, device="cuda") * scale
    return x.to(torch.bfloat16)


@_skip
@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("rows", ROWS)
def test_silu_mul_quantize_is_bit_identical_to_the_unfused_pair(rows, aligned):
    from batchgen_kernels.triton.fp8_quantize import (
        per_token_blocked_quantize_bf16_to_fp8_1d,
        silu_mul_per_token_blocked_quantize_bf16_to_fp8_1d,
    )
    for seed, scale in ((rows, 1.0), (rows + 1000, 8.0), (rows + 2000, 0.02)):
        gate_up = _gate_up_input(rows, seed, scale)
        # exact zeros, a whole zero 128-block, and large-magnitude gates
        gate_up[0, :7] = 0
        gate_up[-1, 128:256] = 0
        gate_up[-1, INTER + 128:INTER + 256] = 0
        gate_up[0, 300:310] = torch.tensor([90.0, -90.0, 30.0, -30.0, 15.0,
                                            -15.0, 1e-30, -1e-30, 2.0, -2.0],
                                           dtype=torch.bfloat16, device="cuda")
        ref_h = F.silu(gate_up[:, :INTER]) * gate_up[:, INTER:]
        ref_q, ref_s = per_token_blocked_quantize_bf16_to_fp8_1d(
            ref_h.contiguous(), scale_tma_aligned=aligned)
        q, s = silu_mul_per_token_blocked_quantize_bf16_to_fp8_1d(
            gate_up, scale_tma_aligned=aligned)
        assert q.shape == ref_q.shape and s.shape == ref_s.shape
        assert s.stride() == ref_s.stride()
        nq, ns = _mismatch(q, ref_q), _mismatch(s, ref_s)
        assert nq == 0 and ns == 0, (
            f"rows={rows} seed={seed} aligned={aligned}: {nq} fp8 / {ns} scale "
            "elements differ from F.silu(g) * u -> quantize")
        if aligned:
            # the layout DeepGEMM consumes without a transpose
            assert s.stride(0) == 1 and s.stride(1) == (rows + 3) // 4 * 4


@_skip
def test_silu_mul_quantize_rejects_inputs_it_cannot_reproduce():
    from batchgen_kernels.triton.fp8_quantize import (
        silu_mul_per_token_blocked_quantize_bf16_to_fp8_1d as fn,
    )
    x = _gate_up_input(4, 0)
    with pytest.raises(TypeError):
        fn(x.float())
    with pytest.raises(ValueError):
        fn(x[:, :-2].contiguous())
    with pytest.raises(ValueError):
        fn(x[:, : 2 * 192].contiguous())       # half width not a 128 multiple
    with pytest.raises(ValueError):
        fn(x.t())
    with pytest.raises(ValueError):
        fn(x[:0])
