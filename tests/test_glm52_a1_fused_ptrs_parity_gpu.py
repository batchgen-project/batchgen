"""A1 GPU gate: the pointer-form S1 on a FUSED stack equals the contiguous
form on per-projection stacks, value for value.

This is the one correctness claim the fused layout rests on. The CuTe grouped
kernels derive each expert's weight address from a hardcoded ``N * K`` group
stride (``fp8_blockwise_gemm.cu:60``; the scales at ``:67``), so the contiguous
kernel CANNOT read a half of a fused ``[E, 2I, H]`` block -- it would address
expert ``e`` at ``e * I * H`` instead of ``e * 2I * H``. The ragged/prefill S1
on the DeepEP configuration therefore switches to
``grouped_fp8_blockwise_fused_s1_ptrs``, which takes every expert's address
explicitly.

The test builds ONE set of random weights, lays it out both ways, and requires
the two kernels to agree. A silent stride error shows up here as a mismatch on
every expert except the first -- which is exactly the failure the fused layout
would otherwise introduce, and exactly what a shape-only test would miss.

Also pinned here: the contiguous kernel on a fused stack is NOT quietly wrong
but simply not offered -- the loader leaves ``fp8_gate_w3d`` None under the
fused layout, so there is no tensor to hand it. That part is covered on CPU in
``test_glm52_a1_fused_gate_up_layout.py``; this file is the numerical half.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU"
)

E, H, I = 4, 512, 256
SCALE_BLOCK = 128
FP8 = torch.float8_e4m3fn
# Per-element relative agreement. The two kernels run the same arithmetic on
# the same bytes in the same order; only the weight ADDRESSING differs, so the
# expectation is bit-identity and the tolerance is a guard, not a budget.
DIFF_THRESHOLD = 1e-6


def _quant_weight(ref):
    """Blockwise FP8 weights plus their [E, N/128, K/128] f32 scales."""
    e, n, k = ref.shape
    blocks = ref.reshape(
        e, n // SCALE_BLOCK, SCALE_BLOCK, k // SCALE_BLOCK, SCALE_BLOCK
    )
    amax = blocks.abs().amax(dim=(2, 4)).clamp(min=1e-10)
    scale = amax / torch.finfo(FP8).max
    q = (blocks / scale[:, :, None, :, None]).to(FP8)
    return q.reshape(e, n, k), scale.contiguous()


def _quant_rows(x):
    """Per-128-group row quantization, in the CuTe S1's [K/128, M] form."""
    m, k = x.shape
    g = x.reshape(m, k // SCALE_BLOCK, SCALE_BLOCK)
    amax = g.abs().amax(dim=2).clamp(min=1e-10)
    scale = amax / torch.finfo(FP8).max
    q = (g / scale[:, :, None]).to(FP8).reshape(m, k)
    return q, scale.t().contiguous()


@pytest.fixture(scope="module")
def layer():
    from batchgen.moe.grouped_fp8_blockwise_moe import (
        require_grouped_fp8_blockwise_ptr_kernels,
    )

    require_grouped_fp8_blockwise_ptr_kernels()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(20261022)
    rows_per_expert = 32
    m = E * rows_per_expert

    def rand(*shape):
        return torch.randn(*shape, generator=gen, device=dev,
                           dtype=torch.float32) * 0.1

    gate_q, gate_s = _quant_weight(rand(E, I, H))
    up_q, up_s = _quant_weight(rand(E, I, H))
    x_q, x_s = _quant_rows(rand(m, H).to(torch.bfloat16).float())

    seqlens = torch.full((E,), rows_per_expert, dtype=torch.int32, device=dev)
    cu = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.cumsum(seqlens, 0)
    return dict(dev=dev, m=m, rows=rows_per_expert, x_q=x_q, x_s=x_s,
                gate_q=gate_q, gate_s=gate_s, up_q=up_q, up_s=up_s,
                seqlens=seqlens, cu=cu)


def _contiguous_s1(L):
    from batchgen.moe.grouped_fp8_blockwise_moe import (
        grouped_fp8_blockwise_fused_s1,
    )

    out = torch.zeros(L["m"], I, dtype=torch.bfloat16, device=L["dev"])
    # The CuTe scale stacks pad the trailing block axis to 4; at these shapes
    # K/128 = 4 already, so the pad is the identity.
    grouped_fp8_blockwise_fused_s1(
        L["x_q"], L["x_s"], L["gate_q"], L["up_q"], L["gate_s"], L["up_s"],
        L["seqlens"], L["cu"], L["rows"], output=out,
    )
    torch.cuda.synchronize()
    return out


def _fused_ptrs_s1(L):
    from batchgen.moe.grouped_fp8_blockwise_moe import (
        grouped_fp8_blockwise_fused_s1_ptrs,
    )

    dev = L["dev"]
    # The loader's layout: gate rows [0, I), up rows [I, 2I), one block.
    gate_up = torch.empty(E, 2 * I, H, dtype=FP8, device=dev)
    gate_up[:, :I, :] = L["gate_q"]
    gate_up[:, I:, :] = L["up_q"]
    nb = I // SCALE_BLOCK
    gate_up_s = torch.empty(
        E, 2 * nb, H // SCALE_BLOCK, dtype=torch.float32, device=dev)
    gate_up_s[:, :nb, :] = L["gate_s"]
    gate_up_s[:, nb:, :] = L["up_s"]

    gate_rows = [gate_up[e, :I, :] for e in range(E)]
    up_rows = [gate_up[e, I:, :] for e in range(E)]
    gate_s_rows = [gate_up_s[e, :nb, :] for e in range(E)]
    up_s_rows = [gate_up_s[e, nb:, :] for e in range(E)]
    ptr = lambda rs: torch.tensor(  # noqa: E731
        [r.data_ptr() for r in rs], dtype=torch.int64, device=dev)

    out = torch.zeros(L["m"], I, dtype=torch.bfloat16, device=dev)
    grouped_fp8_blockwise_fused_s1_ptrs(
        L["x_q"], L["x_s"],
        gate_rows[0], ptr(gate_rows),
        up_rows[0], ptr(up_rows),
        gate_s_rows[0], ptr(gate_s_rows),
        up_s_rows[0], ptr(up_s_rows),
        L["seqlens"], L["cu"], L["rows"], output=out,
    )
    torch.cuda.synchronize()
    return out, gate_up


def test_the_pointer_form_on_a_fused_stack_matches_the_contiguous_form(layer):
    want = _contiguous_s1(layer)
    got, _ = _fused_ptrs_s1(layer)

    assert torch.isfinite(got).all(), "the fused/pointer S1 produced non-finite"
    num = (got.float() - want.float()).abs().max().item()
    den = want.float().abs().max().item()
    rel = num / max(den, 1e-12)
    print(f"PTRS-vs-CONTIGUOUS max|d|={num:.3e} rel={rel:.3e}")
    assert rel < DIFF_THRESHOLD, rel


def test_the_agreement_holds_on_every_expert_not_just_the_first(layer):
    """The stride bug this guards against leaves expert 0 correct and corrupts
    every later expert, so a whole-tensor norm alone could hide it."""
    want = _contiguous_s1(layer)
    got, _ = _fused_ptrs_s1(layer)
    rows = layer["rows"]
    for e in range(E):
        lo, hi = e * rows, (e + 1) * rows
        d = (got[lo:hi].float() - want[lo:hi].float()).abs().max().item()
        den = max(want[lo:hi].float().abs().max().item(), 1e-12)
        print(f"  expert {e}: rel={d / den:.3e}")
        assert d / den < DIFF_THRESHOLD, (e, d / den)


def test_the_fused_halves_are_the_addresses_the_kernel_was_given(layer):
    """Address arithmetic on device memory, so the pointer arrays cannot be
    right by accident: the halves are I * H elements apart and contiguous."""
    _got, gate_up = _fused_ptrs_s1(layer)
    for e in range(E):
        gate, up = gate_up[e, :I, :], gate_up[e, I:, :]
        assert gate.is_contiguous() and up.is_contiguous()
        assert up.data_ptr() - gate.data_ptr() == I * H * gate_up.element_size()
        assert gate_up[e].data_ptr() - gate_up[0].data_ptr() == (
            e * 2 * I * H * gate_up.element_size()
        )
