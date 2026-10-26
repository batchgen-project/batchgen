"""GLM-5 norm -> FP8-quantize fusion — the BITWISE gate (GPU tier).

This is the gate POIS's 2026-10-25 precision ruling asks for: the fused
kernels' FP8 bytes and per-128-block scales must equal the standalone
quantize's output BIT FOR BIT, on valid rows and padding rows alike, and the
BF16 norm output must equal the unfused norm's.

Run it on the machine that serves GLM-5.2, from the worktree, with the decode
runtime's own interpreter::

    pytest -s tests/test_glm52_norm_quant_fuse_bitwise_gpu.py

The CUDA half needs ``batchgen_kernels.attention._C_add_rmsnorm_fp8_quant``
built (``pip install -e batchgen_kernels/``, or ``BATCHGEN_KERNELS_DEV=1`` for
the JIT path registered in ``_jit_registry.py``).

What "bitwise" means here
-------------------------
* FP8: compared as ``uint8``, so a differing exponent/mantissa byte fails even
  where the dequantized value would round to the same float.
* Scales: compared as ``int32`` bit patterns as well as by value, so a -0.0 or
  a one-ULP difference cannot hide behind ``==``.
* BF16: compared as ``uint16``.

Nothing here uses a tolerance. A tolerance-passing-but-not-identical result is
exactly the outcome this file exists to reject.

Falsifiers this file is designed to catch
-----------------------------------------
1. The fused kernel quantizes the unrounded FP32 accumulator instead of the
   BF16-rounded value (would differ on many bytes).
2. ``1.0f / 448.0f`` instead of ``static_cast<float>(1.0 / 448.0)`` for the
   scale constant (one-ULP scale differences on some rows).
3. The quantize divide using an IEEE divide (``__fdiv_rn`` / ``div.rn.f32``)
   or an approximate one (``__fdividef`` / ``div.approx.f32``) instead of the
   ``div.full.f32`` the reference Triton kernel emits. This one is not
   hypothetical: the first version of the fused CUDA kernel used ``__fdiv_rn``
   and this gate failed 4 of 74 cases, each a SINGLE e4m3 byte, every one a
   quotient sitting on the subnormal rounding boundary 3.5*2^-9 -- where
   ``div.full.f32``'s <=2 ULP (deterministic, but not correctly rounded) result
   lands on the other side of a representable fp8 value. A more accurate
   divide is still a failure here: the reference's bytes are the contract.
   The Triton kernel's own divide must not be "fixed" to match the CUDA one
   either; the old kernel is the ground truth and stays untouched.
4. The Triton launch picking a tuned ``num_warps`` (changes the sum tree, so
   the BF16 norm moves and every FP8 byte derived from it moves with it).
5. The padding-row branch writing something other than 0x00 / 1e-12.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

_BLOCK = 128
_EPSILON = 1e-12


# --------------------------------------------------------------------------- #
# Bitwise comparators
# --------------------------------------------------------------------------- #

def _assert_fp8_identical(got, want, what):
    assert got.dtype == want.dtype == torch.float8_e4m3fn, (got.dtype, want.dtype)
    assert got.shape == want.shape, (got.shape, want.shape)
    g = got.contiguous().view(torch.uint8)
    w = want.contiguous().view(torch.uint8)
    if not torch.equal(g, w):
        diff = (g != w).nonzero()
        raise AssertionError(
            f"{what}: {diff.shape[0]} of {g.numel()} FP8 BYTES differ; first at "
            f"{diff[0].tolist()} got 0x{int(g.flatten()[0]):02x} class, "
            f"sample got={g[tuple(diff[0].tolist())]} want={w[tuple(diff[0].tolist())]}"
        )


def _assert_f32_identical(got, want, what):
    assert got.dtype == want.dtype == torch.float32
    assert got.shape == want.shape, (got.shape, want.shape)
    g = got.contiguous().view(torch.int32)
    w = want.contiguous().view(torch.int32)
    if not torch.equal(g, w):
        diff = (g != w).nonzero()
        idx = tuple(diff[0].tolist())
        raise AssertionError(
            f"{what}: {diff.shape[0]} of {g.numel()} fp32 bit patterns differ; "
            f"first at {idx}: got {got.contiguous()[idx].item()!r} "
            f"want {want.contiguous()[idx].item()!r}"
        )


def _assert_bf16_identical(got, want, what):
    assert got.dtype == want.dtype == torch.bfloat16
    g = got.contiguous().view(torch.uint16)
    w = want.contiguous().view(torch.uint16)
    if not torch.equal(g, w):
        diff = (g != w).nonzero()
        raise AssertionError(
            f"{what}: {diff.shape[0]} of {g.numel()} BF16 words differ; first at "
            f"{diff[0].tolist()}"
        )


def _num_valid(n, device):
    return torch.tensor([n], dtype=torch.int32, device=device)


# --------------------------------------------------------------------------- #
# Triton: fused_rmsnorm_fp8_quant vs fused_rmsnorm + act_quant  (the q_a site)
# --------------------------------------------------------------------------- #

_Q_A_SHAPES = [
    # (rows, dim). 2048 is GLM-5.2's q_lora_rank, the production width.
    (1, 2048),
    (3, 2048),          # rows % 4 != 0 exercises the TMA scale padding
    (16, 2048),
    (256, 2048),
    (2048, 2048),
    (7, 512),           # narrow, NUM_BLOCKS not a power of two (4)
    (5, 2560),          # NUM_BLOCKS = 20 -> padded to 32
]


@pytest.mark.parametrize("rows,dim", _Q_A_SHAPES)
@pytest.mark.parametrize("gated", [False, True])
def test_triton_fused_norm_quant_is_bitwise_identical(rows, dim, gated):
    from batchgen_kernels.triton.rmsnorm import fused_rmsnorm
    from batchgen_kernels.triton.fp8_quantize import (
        per_token_blocked_quantize_bf16_to_fp8_1d,
    )
    from batchgen_kernels.triton.rmsnorm_fp8_quant_fused import (
        fused_rmsnorm_fp8_quant,
    )

    dev = "cuda"
    torch.manual_seed(1234 + rows + dim)
    x = (torch.randn(rows, dim, device=dev) * 1.7).to(torch.bfloat16)
    weight = (torch.randn(dim, device=dev) * 0.4 + 1.0).to(torch.bfloat16)
    eps = 1e-5
    nvt = _num_valid(max(1, rows // 2), dev) if gated else None

    # Reference: the two launches, exactly as the segment issued them.
    ref_normed = fused_rmsnorm(x, weight, eps, out=torch.empty_like(x))
    ref_fp8, ref_scale = per_token_blocked_quantize_bf16_to_fp8_1d(
        ref_normed, num_valid_tokens=nvt, scale_tma_aligned=nvt is not None
    )

    got_normed, got_fp8, got_scale = fused_rmsnorm_fp8_quant(
        x, weight, eps, out=torch.empty_like(x),
        num_valid_tokens=nvt, scale_tma_aligned=nvt is not None,
    )

    _assert_bf16_identical(got_normed, ref_normed, "q_a normed")
    _assert_fp8_identical(got_fp8, ref_fp8, "q_a fp8")
    _assert_f32_identical(got_scale, ref_scale, "q_a scale")
    assert got_scale.stride() == ref_scale.stride()


@pytest.mark.parametrize("valid", [0, 1, 3, 4, 15, 16, 17])
def test_triton_fused_norm_quant_num_valid_edges(valid):
    """Every padding boundary the decode bucket can present, including 0 and a
    count equal to the row count (no padding at all)."""
    from batchgen_kernels.triton.rmsnorm import fused_rmsnorm
    from batchgen_kernels.triton.fp8_quantize import (
        per_token_blocked_quantize_bf16_to_fp8_1d,
    )
    from batchgen_kernels.triton.rmsnorm_fp8_quant_fused import (
        fused_rmsnorm_fp8_quant,
    )

    dev, rows, dim = "cuda", 17, 2048
    torch.manual_seed(99 + valid)
    # Deliberately large magnitudes so the clamp and the e4m3 range matter.
    x = (torch.randn(rows, dim, device=dev) * 60.0).to(torch.bfloat16)
    weight = (torch.randn(dim, device=dev) * 2.0).to(torch.bfloat16)
    nvt = _num_valid(valid, dev)

    ref_normed = fused_rmsnorm(x, weight, 1e-5, out=torch.empty_like(x))
    ref_fp8, ref_scale = per_token_blocked_quantize_bf16_to_fp8_1d(
        ref_normed, num_valid_tokens=nvt, scale_tma_aligned=True
    )
    got_normed, got_fp8, got_scale = fused_rmsnorm_fp8_quant(
        x, weight, 1e-5, out=torch.empty_like(x),
        num_valid_tokens=nvt, scale_tma_aligned=True,
    )

    _assert_bf16_identical(got_normed, ref_normed, f"normed (valid={valid})")
    _assert_fp8_identical(got_fp8, ref_fp8, f"fp8 (valid={valid})")
    _assert_f32_identical(got_scale, ref_scale, f"scale (valid={valid})")

    # And the padding contract itself, not just agreement with the reference.
    if valid < rows:
        pad_bytes = got_fp8[valid:].contiguous().view(torch.uint8)
        assert torch.equal(pad_bytes, torch.zeros_like(pad_bytes)), (
            "padding rows must carry exact-zero FP8 BYTES"
        )
        pad_scale = got_scale[valid:].contiguous()
        assert torch.equal(
            pad_scale, torch.full_like(pad_scale, _EPSILON)
        ), "padding rows must carry the 1e-12 scale"
    # The BF16 norm has no validity gate and never had one: every row is
    # normalized, padding included.
    assert torch.isfinite(got_normed.float()).all()


def test_triton_fused_norm_quant_rejects_unsupported_widths():
    from batchgen_kernels.triton.rmsnorm_fp8_quant_fused import (
        fused_rmsnorm_fp8_quant,
    )
    dev = "cuda"
    with pytest.raises(ValueError, match="hidden_size % 128 == 0"):
        fused_rmsnorm_fp8_quant(
            torch.zeros(4, 200, device=dev, dtype=torch.bfloat16),
            torch.zeros(200, device=dev, dtype=torch.bfloat16),
        )
    with pytest.raises(ValueError, match="single-pass"):
        fused_rmsnorm_fp8_quant(
            torch.zeros(2, 16384, device=dev, dtype=torch.bfloat16),
            torch.zeros(16384, device=dev, dtype=torch.bfloat16),
        )
    with pytest.raises(TypeError, match="bf16"):
        fused_rmsnorm_fp8_quant(
            torch.zeros(4, 2048, device=dev, dtype=torch.float16),
            torch.zeros(2048, device=dev, dtype=torch.float16),
        )


# --------------------------------------------------------------------------- #
# CUDA: cuda_add_rmsnorm_fp8_quant vs cuda_add_rmsnorm + act_quant
# (the qkv_a and shared-expert sites; hidden_size 6144)
# --------------------------------------------------------------------------- #

_LAYER_SHAPES = [1, 2, 3, 4, 7, 64, 255, 256, 1024, 2048]


@pytest.mark.parametrize("rows", _LAYER_SHAPES)
@pytest.mark.parametrize(
    "gated,masked",
    [
        (False, False),   # the shared-expert site: no quant gate, no row mask
        (True, False),    # the qkv_a site: quant gate, no row mask
        (False, True),    # post-attention norm with the attn_output row mask
        (True, True),
    ],
)
def test_cuda_fused_add_norm_quant_is_bitwise_identical(rows, gated, masked):
    from batchgen.attention.fused_kernels import cuda_add_rmsnorm
    from batchgen.attention.mla.fa3_backend import act_quant
    from batchgen_kernels.attention.add_rmsnorm_fp8_quant import (
        cuda_add_rmsnorm_fp8_quant,
    )

    dev, dim = "cuda", 6144
    torch.manual_seed(4321 + rows)
    residual = (torch.randn(rows, dim, device=dev) * 0.9).to(torch.bfloat16)
    hidden = (torch.randn(rows, dim, device=dev) * 0.4).to(torch.bfloat16)
    weight = (torch.randn(dim, device=dev) * 0.3 + 1.0).to(torch.bfloat16)
    eps = 1e-5
    nvt = _num_valid(max(1, rows // 2), dev) if gated else None
    row_mask = None
    if masked:
        row_mask = torch.ones(rows, dtype=torch.bfloat16, device=dev)
        row_mask[rows // 2:] = 0.0

    # Reference: the unfused norm, then the standalone quantize of its output.
    ref_residual = residual.clone()
    ref_normed, ref_residual_out = cuda_add_rmsnorm(
        ref_residual, hidden, weight, eps, row_valid_mask=row_mask
    )
    ref_fp8, ref_scale = act_quant(
        ref_normed,
        num_valid_tokens=nvt,
        scale_tma_aligned=True,
    )

    got_residual = residual.clone()
    got_normed, got_residual_out, got_fp8, got_scale = cuda_add_rmsnorm_fp8_quant(
        got_residual, hidden, weight, eps,
        quant_num_valid_tokens=nvt,
        row_valid_mask=row_mask,
        scale_tma_aligned=True,
    )

    _assert_bf16_identical(got_normed, ref_normed, "layer normed")
    _assert_bf16_identical(got_residual_out, ref_residual_out, "layer residual")
    _assert_fp8_identical(got_fp8, ref_fp8, "layer fp8")
    _assert_f32_identical(got_scale, ref_scale, "layer scale")
    assert got_scale.stride() == ref_scale.stride()


@pytest.mark.parametrize("valid", [0, 1, 3, 4, 63, 64, 65])
def test_cuda_fused_add_norm_quant_num_valid_edges(valid):
    from batchgen.attention.fused_kernels import cuda_add_rmsnorm
    from batchgen.attention.mla.fa3_backend import act_quant
    from batchgen_kernels.attention.add_rmsnorm_fp8_quant import (
        cuda_add_rmsnorm_fp8_quant,
    )

    dev, rows, dim = "cuda", 65, 6144
    torch.manual_seed(777 + valid)
    residual = (torch.randn(rows, dim, device=dev) * 40.0).to(torch.bfloat16)
    hidden = (torch.randn(rows, dim, device=dev) * 40.0).to(torch.bfloat16)
    weight = (torch.randn(dim, device=dev) * 3.0).to(torch.bfloat16)
    nvt = _num_valid(valid, dev)

    ref_residual = residual.clone()
    ref_normed, _ = cuda_add_rmsnorm(ref_residual, hidden, weight, 1e-5)
    ref_fp8, ref_scale = act_quant(
        ref_normed, num_valid_tokens=nvt, scale_tma_aligned=True
    )

    got_residual = residual.clone()
    got_normed, _, got_fp8, got_scale = cuda_add_rmsnorm_fp8_quant(
        got_residual, hidden, weight, 1e-5,
        quant_num_valid_tokens=nvt, scale_tma_aligned=True,
    )

    _assert_bf16_identical(got_normed, ref_normed, f"normed (valid={valid})")
    _assert_fp8_identical(got_fp8, ref_fp8, f"fp8 (valid={valid})")
    _assert_f32_identical(got_scale, ref_scale, f"scale (valid={valid})")

    if valid < rows:
        pad_bytes = got_fp8[valid:].contiguous().view(torch.uint8)
        assert torch.equal(pad_bytes, torch.zeros_like(pad_bytes))
        pad_scale = got_scale[valid:].contiguous()
        assert torch.equal(pad_scale, torch.full_like(pad_scale, _EPSILON))


def test_cuda_fused_contiguous_scale_layout_matches_too():
    """``scale_tma_aligned=False`` is the plain [M, blocks] contiguous form."""
    from batchgen.attention.fused_kernels import cuda_add_rmsnorm
    from batchgen.attention.mla.fa3_backend import act_quant
    from batchgen_kernels.attention.add_rmsnorm_fp8_quant import (
        cuda_add_rmsnorm_fp8_quant,
    )

    dev, rows, dim = "cuda", 37, 6144
    torch.manual_seed(5)
    residual = (torch.randn(rows, dim, device=dev)).to(torch.bfloat16)
    hidden = (torch.randn(rows, dim, device=dev)).to(torch.bfloat16)
    weight = (torch.randn(dim, device=dev) * 0.5 + 1.0).to(torch.bfloat16)

    ref_res = residual.clone()
    ref_normed, _ = cuda_add_rmsnorm(ref_res, hidden, weight, 1e-5)
    ref_fp8, ref_scale = act_quant(ref_normed, scale_tma_aligned=False)

    got_res = residual.clone()
    got_normed, _, got_fp8, got_scale = cuda_add_rmsnorm_fp8_quant(
        got_res, hidden, weight, 1e-5, scale_tma_aligned=False
    )
    _assert_bf16_identical(got_normed, ref_normed, "normed")
    _assert_fp8_identical(got_fp8, ref_fp8, "fp8")
    _assert_f32_identical(got_scale, ref_scale, "scale")
    assert got_scale.is_contiguous() and got_scale.shape == (rows, dim // _BLOCK)


def test_cuda_fused_kernel_norm_gate_requires_the_same_quant_gate():
    """A norm-skipped row's ``normed_out`` has no defined value to quantize, so
    the combination is refused rather than given an invented answer."""
    from batchgen_kernels.attention.add_rmsnorm_fp8_quant import (
        cuda_add_rmsnorm_fp8_quant,
    )
    dev, rows, dim = "cuda", 8, 6144
    residual = torch.zeros(rows, dim, device=dev, dtype=torch.bfloat16)
    hidden = torch.zeros(rows, dim, device=dev, dtype=torch.bfloat16)
    weight = torch.ones(dim, device=dev, dtype=torch.bfloat16)
    nvt = _num_valid(4, dev)

    with pytest.raises(RuntimeError, match="SAME tensor"):
        cuda_add_rmsnorm_fp8_quant(
            residual, hidden, weight, 1e-5, norm_num_valid_tokens=nvt
        )
    with pytest.raises(RuntimeError, match="SAME tensor"):
        cuda_add_rmsnorm_fp8_quant(
            residual, hidden, weight, 1e-5,
            norm_num_valid_tokens=nvt,
            quant_num_valid_tokens=_num_valid(4, dev),  # equal value, other tensor
        )
    # The same tensor on both gates is accepted.
    cuda_add_rmsnorm_fp8_quant(
        residual, hidden, weight, 1e-5,
        norm_num_valid_tokens=nvt, quant_num_valid_tokens=nvt,
    )


def test_cuda_fused_kernel_refuses_other_widths_instead_of_falling_back():
    from batchgen_kernels.attention.add_rmsnorm_fp8_quant import (
        cuda_add_rmsnorm_fp8_quant,
    )
    dev, rows, dim = "cuda", 4, 4096
    with pytest.raises(RuntimeError, match="hidden-size-6144"):
        cuda_add_rmsnorm_fp8_quant(
            torch.zeros(rows, dim, device=dev, dtype=torch.bfloat16),
            torch.zeros(rows, dim, device=dev, dtype=torch.bfloat16),
            torch.ones(dim, device=dev, dtype=torch.bfloat16),
            1e-5,
        )
    with pytest.raises(RuntimeError, match="bf16"):
        cuda_add_rmsnorm_fp8_quant(
            torch.zeros(rows, 6144, device=dev, dtype=torch.float16),
            torch.zeros(rows, 6144, device=dev, dtype=torch.float16),
            torch.ones(6144, device=dev, dtype=torch.float16),
            1e-5,
        )


def test_cuda_fused_kernel_handles_the_3d_layer_shape():
    """The layer segment hands it [batch, 1, 6144] tensors, not [batch, 6144];
    the FP8 output is the flat [batch, 6144] the GEMM consumes."""
    from batchgen.attention.fused_kernels import cuda_add_rmsnorm
    from batchgen.attention.mla.fa3_backend import act_quant
    from batchgen_kernels.attention.add_rmsnorm_fp8_quant import (
        cuda_add_rmsnorm_fp8_quant,
    )

    dev, rows, dim = "cuda", 12, 6144
    torch.manual_seed(31)
    residual = torch.randn(rows, 1, dim, device=dev).to(torch.bfloat16)
    hidden = torch.randn(rows, 1, dim, device=dev).to(torch.bfloat16)
    weight = (torch.randn(dim, device=dev) * 0.4 + 1.0).to(torch.bfloat16)

    ref_res = residual.clone()
    ref_normed, _ = cuda_add_rmsnorm(ref_res, hidden, weight, 1e-5)
    ref_fp8, ref_scale = act_quant(
        ref_normed.view(rows, dim), scale_tma_aligned=True
    )

    got_res = residual.clone()
    got_normed, _, got_fp8, got_scale = cuda_add_rmsnorm_fp8_quant(
        got_res, hidden, weight, 1e-5, scale_tma_aligned=True
    )
    assert got_normed.shape == (rows, 1, dim)
    assert got_fp8.shape == (rows, dim)
    _assert_bf16_identical(got_normed, ref_normed, "normed")
    _assert_fp8_identical(got_fp8, ref_fp8, "fp8")
    _assert_f32_identical(got_scale, ref_scale, "scale")


# --------------------------------------------------------------------------- #
# The seam the shared expert consumes
# --------------------------------------------------------------------------- #

def test_the_shared_expert_accepts_the_handed_down_pair_unchanged():
    """A pre-quantized x must drive the same gate/up GEMM operands.

    Rather than build a whole expert, this pins the wrapper's validation: the
    accepted pair is the one the fused kernel produces, and a contiguous scale
    (the layout DeepGEMM would misread) is refused.
    """
    from batchgen_kernels.attention.add_rmsnorm_fp8_quant import (
        cuda_add_rmsnorm_fp8_quant,
    )

    dev, rows, dim = "cuda", 20, 6144
    torch.manual_seed(42)
    residual = torch.randn(rows, dim, device=dev).to(torch.bfloat16)
    hidden = torch.randn(rows, dim, device=dev).to(torch.bfloat16)
    weight = torch.ones(dim, device=dev, dtype=torch.bfloat16)

    _, _, x_fp8, x_scale = cuda_add_rmsnorm_fp8_quant(
        residual.clone(), hidden, weight, 1e-5, scale_tma_aligned=True
    )
    aligned = ((rows + 3) // 4) * 4
    assert x_scale.stride() == (1, aligned)
    assert x_fp8.shape == (rows, dim)
    assert x_scale.shape == (rows, dim // _BLOCK)
