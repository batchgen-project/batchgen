"""The fused indexer-K norm+rope+quantize+write contract, without a GPU.

``fused_indexer_k_norm_rope_write_fp8`` is a port of SGLang's
``fused_k_indexer_norm_rope_store`` (sgl-project/sglang @ ``e00930c5``, tag
``v0.5.21``). It replaces three launches of the GLM-5.2 indexer K tail -- the
``k_norm`` LayerNorm, the fused RoPE+Hadamard and the fused quantize+write --
with one, so what it computes has to be pinned against the chain it replaces
rather than against itself.

Everything here runs on the torch-only contract module: the planner that the
CUDA host side mirrors check for check, and the reference chain that is the
oracle for the on-device parity script. The reference is validated against an
INDEPENDENT un-fused chain built from ``nn.LayerNorm`` and an explicit
Hadamard matrix, so a mistake in the ported stage order shows up here.

``batchgen_kernels.attention.dsa.indexer_k_quant`` is torch-only, but its
PACKAGE imports Triton-backed kernels on purpose, so the module is loaded
directly from its file (same shim as
``test_indexer_k_page_split_write_contract.py``).
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_QUANT_PATH = (
    _REPO_ROOT / "batchgen_kernels" / "attention" / "dsa" / "indexer_k_quant.py"
)


def _load_quant_module():
    name = "_indexer_k_norm_rope_contract"
    spec = importlib.util.spec_from_file_location(name, _QUANT_PATH)
    module = importlib.util.module_from_spec(spec)
    # Registered before exec because @dataclass resolves its own module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_quant = _load_quant_module()

AMAX_FLOOR = _quant.AMAX_FLOOR
FP8_MAX = _quant.FP8_MAX
NORM_ROPE_HEAD_DIM = _quant.NORM_ROPE_HEAD_DIM
NORM_ROPE_ROPE_DIM = _quant.NORM_ROPE_ROPE_DIM
hadamard_matrix_128 = _quant.hadamard_matrix_128
indexer_k_norm_rope_reference = _quant.indexer_k_norm_rope_reference
page_split_write_reference = _quant.page_split_write_reference
plan_norm_rope_write = _quant.plan_norm_rope_write
quantize_indexer_k_fp8 = _quant.quantize_indexer_k_fp8

DATA_DIM = NORM_ROPE_HEAD_DIM
ROPE_DIM = NORM_ROPE_ROPE_DIM
ROPE_PAIRS = ROPE_DIM // 2
PAGE_SIZE = 64
MAX_SEQ = 256
LN_EPS = 1e-5


def _cache(num_pages: int = 4) -> torch.Tensor:
    return torch.zeros(
        num_pages, PAGE_SIZE, 1, DATA_DIM + 4, dtype=torch.float8_e4m3fn
    )


def _rope_tables(generator: torch.Generator) -> tuple:
    angle = torch.rand(MAX_SEQ, ROPE_DIM, generator=generator) * (2 * math.pi)
    return angle.cos().contiguous(), angle.sin().contiguous()


def _norm_params(dtype: torch.dtype = torch.bfloat16) -> tuple:
    g = torch.Generator().manual_seed(7)
    weight = (1.0 + 0.1 * torch.randn(DATA_DIM, generator=g)).to(dtype)
    bias = (0.05 * torch.randn(DATA_DIM, generator=g)).to(dtype)
    return weight.contiguous(), bias.contiguous()


def _bf16_ordinal(t: torch.Tensor) -> torch.Tensor:
    """Monotone integer key of a bf16 tensor; adjacent values differ by 1."""

    bits = t.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    magnitude = bits & 0x7FFF
    return torch.where(bits & 0x8000 != 0, 0x8000 - magnitude, 0x8000 + magnitude)


def _assert_within_one_bf16_ulp(got: torch.Tensor, want: torch.Tensor) -> int:
    """Two fp32 orderings of one transform may only disagree on a bf16 tie.

    Returns how many elements landed on a different bf16 value, so a caller
    can also pin that the rate stays in the rounding-tie regime instead of
    hiding a real stage-order mistake behind a loose tolerance.
    """

    delta = (_bf16_ordinal(got) - _bf16_ordinal(want)).abs()
    worst = int(delta.max().item())
    assert worst <= 1, (
        f"a result differs by {worst} bf16 ulps, which is more than a "
        f"rounding tie between two summation orders"
    )
    return int(torch.count_nonzero(delta).item())


def _plan_kwargs(**overrides) -> dict:
    g = torch.Generator().manual_seed(3)
    cos, sin = _rope_tables(g)
    weight, bias = _norm_params()
    kwargs = {
        "blocked_kv": _cache(),
        "k_raw": torch.randn(8, DATA_DIM, generator=g).to(torch.bfloat16),
        "ln_weight": weight,
        "ln_bias": bias,
        "cos_cache": cos,
        "sin_cache": sin,
        "positions": torch.arange(8, dtype=torch.int64),
        "page_size_tokens": PAGE_SIZE,
    }
    kwargs.update(overrides)
    return kwargs


# --------------------------------------------------------------------------
# Hadamard identity
# --------------------------------------------------------------------------


def test_hadamard_matrix_is_the_natural_order_sylvester_transform():
    h = hadamard_matrix_128()
    assert h.shape == (DATA_DIM, DATA_DIM)
    assert set(h.unique().tolist()) == {-1.0, 1.0}
    # H[e, f] = (-1) ** popcount(e & f): check the definition directly on a
    # few entries, then the global orthogonality it implies.
    for e, f in ((0, 0), (1, 1), (3, 3), (5, 3), (127, 127), (96, 48)):
        sign = -1.0 if bin(e & f).count("1") % 2 else 1.0
        assert h[e, f].item() == sign
    torch.testing.assert_close(
        h @ h.t() / DATA_DIM, torch.eye(DATA_DIM), atol=0.0, rtol=0.0
    )


def test_normalized_hadamard_preserves_the_dot_product():
    """Why dropping the rotation on BOTH q and k is logit-preserving."""
    g = torch.Generator().manual_seed(11)
    h = hadamard_matrix_128() * (DATA_DIM ** -0.5)
    q = torch.randn(4, DATA_DIM, generator=g, dtype=torch.float64)
    k = torch.randn(4, DATA_DIM, generator=g, dtype=torch.float64)
    rotated = (q.float() @ h).double() @ ((k.float() @ h).double()).t()
    torch.testing.assert_close(rotated, q @ k.t(), atol=1e-4, rtol=1e-5)


# --------------------------------------------------------------------------
# Reference chain == the un-fused chain it replaces
# --------------------------------------------------------------------------


def _unfused_chain(
    k_raw: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    apply_hadamard: bool,
) -> torch.Tensor:
    """LayerNorm -> interleaved RoPE -> Hadamard, built independently.

    Deliberately written against ``nn.LayerNorm`` (the module the model holds)
    and an explicit matrix multiply, mirroring what
    ``cuda_graph_segments.py`` ran as three launches.
    """

    norm = torch.nn.LayerNorm(DATA_DIM, eps=LN_EPS)
    with torch.no_grad():
        norm.weight.copy_(weight.float())
        norm.bias.copy_(bias.float())
    normed = norm(k_raw.float()).to(torch.bfloat16)

    out = normed.float().clone()
    pos = positions.reshape(-1).to(torch.int64)
    for row, p in enumerate(pos.tolist()):
        for pair in range(ROPE_PAIRS):
            c = cos[p, pair].item()
            s = sin[p, pair].item()
            v0 = out[row, 2 * pair].item()
            v1 = out[row, 2 * pair + 1].item()
            out[row, 2 * pair] = v0 * c - v1 * s
            out[row, 2 * pair + 1] = v1 * c + v0 * s
    if apply_hadamard:
        out = (out @ hadamard_matrix_128()) * (DATA_DIM ** -0.5)
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("apply_hadamard", (True, False))
@pytest.mark.parametrize("batch", (1, 8, 32, 64, 128))
def test_reference_matches_the_unfused_chain(apply_hadamard, batch):
    g = torch.Generator().manual_seed(100 + batch)
    cos, sin = _rope_tables(g)
    weight, bias = _norm_params()
    k_raw = (0.5 * torch.randn(batch, DATA_DIM, generator=g)).to(torch.bfloat16)
    positions = torch.randint(0, MAX_SEQ, (batch,), generator=g, dtype=torch.int64)

    got = indexer_k_norm_rope_reference(
        k_raw, weight, bias, LN_EPS, cos, sin, positions, apply_hadamard
    )
    want = _unfused_chain(
        k_raw, weight, bias, cos, sin, positions, apply_hadamard
    )
    assert got.dtype is torch.bfloat16
    if not apply_hadamard:
        # Without the butterfly both chains sum in the same order.
        torch.testing.assert_close(
            got.float(), want.float(), atol=0.0, rtol=0.0
        )
        return
    # With it, the matrix multiply and the 7-stage butterfly accumulate
    # differently in fp32; the rounding to bf16 absorbs all but a few ties.
    mismatched = _assert_within_one_bf16_ulp(got, want)
    assert mismatched <= max(1, got.numel() // 1000), (
        f"{mismatched}/{got.numel()} elements moved by a bf16 ulp, far above "
        f"the rounding-tie rate"
    )


def test_rope_only_touches_the_leading_rope_dims():
    g = torch.Generator().manual_seed(5)
    cos, sin = _rope_tables(g)
    weight, bias = _norm_params()
    k_raw = torch.randn(4, DATA_DIM, generator=g).to(torch.bfloat16)
    zero_pos = torch.zeros(4, dtype=torch.int64)

    # position 0 of an all-ones cos / all-zeros sin table is the identity
    # rotation, so the Hadamard-free result must equal the plain LayerNorm.
    identity_cos = torch.ones_like(cos)
    identity_sin = torch.zeros_like(sin)
    got = indexer_k_norm_rope_reference(
        k_raw, weight, bias, LN_EPS, identity_cos, identity_sin, zero_pos, False
    )
    want = _unfused_chain(
        k_raw, weight, bias, identity_cos, identity_sin, zero_pos, False
    )
    torch.testing.assert_close(got.float(), want.float(), atol=0.0, rtol=0.0)

    # With a real table the trailing dims are untouched and the leading ones
    # are not.
    rotated = indexer_k_norm_rope_reference(
        k_raw, weight, bias, LN_EPS, cos, sin, torch.full((4,), 9), False
    )
    torch.testing.assert_close(
        rotated[:, ROPE_DIM:].float(), got[:, ROPE_DIM:].float(),
        atol=0.0, rtol=0.0,
    )
    assert not torch.equal(rotated[:, :ROPE_DIM], got[:, :ROPE_DIM])


# --------------------------------------------------------------------------
# fp8 epilogue and the page bytes
# --------------------------------------------------------------------------


def test_the_amax_floor_and_reciprocal_contract_survives_the_fusion():
    """A row that is exactly zero must take the amax floor, not divide by 0."""
    weight = torch.zeros(DATA_DIM, dtype=torch.bfloat16)
    bias = torch.zeros(DATA_DIM, dtype=torch.bfloat16)
    g = torch.Generator().manual_seed(13)
    cos, sin = _rope_tables(g)
    k_raw = torch.randn(3, DATA_DIM, generator=g).to(torch.bfloat16)
    keys = indexer_k_norm_rope_reference(
        k_raw, weight, bias, LN_EPS, cos, sin, torch.zeros(3, dtype=torch.int64),
        True,
    )
    assert torch.count_nonzero(keys) == 0
    _, scales = quantize_indexer_k_fp8(keys)
    torch.testing.assert_close(
        scales, torch.full((3,), AMAX_FLOOR / FP8_MAX), atol=0.0, rtol=0.0
    )


def test_adversarial_rows_cannot_poison_a_neighbour_scale():
    g = torch.Generator().manual_seed(17)
    cos, sin = _rope_tables(g)
    weight, bias = _norm_params()
    k_raw = (0.5 * torch.randn(4, DATA_DIM, generator=g)).to(torch.bfloat16)
    k_raw[1] = float("nan")
    k_raw[2] = 1e4
    keys = indexer_k_norm_rope_reference(
        k_raw, weight, bias, LN_EPS, cos, sin, torch.arange(4), True
    )
    _, scales = quantize_indexer_k_fp8(keys)
    # Rows 0 and 3 are finite and independent of their neighbours: one warp
    # owns one token, so no reduction crosses rows.
    clean = indexer_k_norm_rope_reference(
        k_raw[[0, 3]], weight, bias, LN_EPS, cos, sin, torch.tensor([0, 3]), True
    )
    _, clean_scales = quantize_indexer_k_fp8(clean)
    torch.testing.assert_close(
        scales[[0, 3]], clean_scales, atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize(
    "slots", ([0], [PAGE_SIZE - 1], [PAGE_SIZE], [PAGE_SIZE + 1],
              [PAGE_SIZE - 1, PAGE_SIZE, 2 * PAGE_SIZE - 1, 2 * PAGE_SIZE])
)
def test_page_boundary_slots_land_in_the_right_region(slots):
    g = torch.Generator().manual_seed(23)
    cos, sin = _rope_tables(g)
    weight, bias = _norm_params()
    n = len(slots)
    k_raw = (0.5 * torch.randn(n, DATA_DIM, generator=g)).to(torch.bfloat16)
    keys = indexer_k_norm_rope_reference(
        k_raw, weight, bias, LN_EPS, cos, sin, torch.arange(n), True
    )
    k_fp8, scales = quantize_indexer_k_fp8(keys)

    buf = torch.zeros(4, PAGE_SIZE * (DATA_DIM + 4), dtype=torch.uint8)
    page_split_write_reference(
        buf, torch.tensor(slots, dtype=torch.int64), k_fp8, scales, PAGE_SIZE
    )
    for row, slot in enumerate(slots):
        page, offset = divmod(slot, PAGE_SIZE)
        k_bytes = buf[page, offset * DATA_DIM:(offset + 1) * DATA_DIM]
        torch.testing.assert_close(
            k_bytes, k_fp8[row].view(torch.uint8), atol=0.0, rtol=0.0
        )
        base = PAGE_SIZE * DATA_DIM + offset * 4
        got_scale = buf[page, base:base + 4].clone().view(torch.float32)
        torch.testing.assert_close(
            got_scale, scales[row:row + 1], atol=0.0, rtol=0.0
        )


# --------------------------------------------------------------------------
# Planner: every precondition the CUDA host side checks
# --------------------------------------------------------------------------


def test_plan_resolves_the_geometry():
    plan = plan_norm_rope_write(**_plan_kwargs())
    assert plan.num_tokens == 8
    assert plan.data_dim == DATA_DIM
    assert plan.rope_dim == ROPE_DIM
    assert plan.rope_pairs == ROPE_PAIRS
    assert plan.page_size == PAGE_SIZE
    assert plan.page_bytes == PAGE_SIZE * (DATA_DIM + 4)
    assert plan.scale_region_offset == PAGE_SIZE * DATA_DIM
    assert plan.norm_param_dtype is torch.bfloat16
    assert plan.positions_dtype is torch.int64


@pytest.mark.parametrize("param_dtype", (torch.bfloat16, torch.float32))
def test_plan_accepts_both_layernorm_parameter_dtypes(param_dtype):
    weight, bias = _norm_params(param_dtype)
    plan = plan_norm_rope_write(
        **_plan_kwargs(ln_weight=weight, ln_bias=bias)
    )
    assert plan.norm_param_dtype is param_dtype


@pytest.mark.parametrize("pos_dtype", (torch.int32, torch.int64))
def test_plan_accepts_both_position_dtypes(pos_dtype):
    plan = plan_norm_rope_write(
        **_plan_kwargs(positions=torch.arange(8, dtype=pos_dtype))
    )
    assert plan.positions_dtype is pos_dtype


def test_plan_rejects_a_head_dim_the_kernel_is_not_specialized_for():
    cache = torch.zeros(4, PAGE_SIZE, 1, 64 + 4, dtype=torch.float8_e4m3fn)
    g = torch.Generator().manual_seed(1)
    with pytest.raises(ValueError, match="specialized for a head dim of 128"):
        plan_norm_rope_write(
            **_plan_kwargs(
                blocked_kv=cache,
                k_raw=torch.randn(8, 64, generator=g).to(torch.bfloat16),
            )
        )


def test_plan_rejects_a_non_2d_k_raw():
    g = torch.Generator().manual_seed(1)
    with pytest.raises(ValueError, match=r"k_raw must be \[N, 128\]"):
        plan_norm_rope_write(
            **_plan_kwargs(
                k_raw=torch.randn(8, 1, 1, DATA_DIM, generator=g).to(
                    torch.bfloat16
                )
            )
        )


def test_plan_rejects_a_noncontiguous_k_raw():
    g = torch.Generator().manual_seed(1)
    wide = torch.randn(8, 2 * DATA_DIM, generator=g).to(torch.bfloat16)
    with pytest.raises(ValueError, match="k_raw must be contiguous"):
        plan_norm_rope_write(**_plan_kwargs(k_raw=wide[:, :DATA_DIM]))


def test_plan_rejects_mismatched_layernorm_parameter_dtypes():
    weight, bias = _norm_params(torch.bfloat16)
    with pytest.raises(ValueError, match="must share a dtype"):
        plan_norm_rope_write(
            **_plan_kwargs(ln_weight=weight, ln_bias=bias.float())
        )


def test_plan_rejects_an_unsupported_layernorm_parameter_dtype():
    weight, bias = _norm_params(torch.bfloat16)
    with pytest.raises(ValueError, match="bfloat16 or float32"):
        plan_norm_rope_write(
            **_plan_kwargs(
                ln_weight=weight.half(), ln_bias=bias.half()
            )
        )


def test_plan_rejects_wrongly_sized_layernorm_parameters():
    weight, bias = _norm_params()
    with pytest.raises(ValueError, match="ln_bias must hold 128 elements"):
        plan_norm_rope_write(
            **_plan_kwargs(ln_weight=weight, ln_bias=bias[:64].contiguous())
        )


def test_plan_rejects_a_non_fp32_rope_table():
    g = torch.Generator().manual_seed(1)
    cos, sin = _rope_tables(g)
    with pytest.raises(ValueError, match="cos_cache must be float32"):
        plan_norm_rope_write(
            **_plan_kwargs(cos_cache=cos.to(torch.bfloat16), sin_cache=sin)
        )


def test_plan_rejects_a_rope_table_with_too_few_pairs():
    g = torch.Generator().manual_seed(1)
    cos, sin = _rope_tables(g)
    narrow_cos = cos[:, : ROPE_PAIRS - 1].contiguous()
    narrow_sin = sin[:, : ROPE_PAIRS - 1].contiguous()
    with pytest.raises(ValueError, match="at least 32 interleaved pairs"):
        plan_norm_rope_write(
            **_plan_kwargs(cos_cache=narrow_cos, sin_cache=narrow_sin)
        )


def test_plan_rejects_rope_tables_of_different_shapes():
    g = torch.Generator().manual_seed(1)
    cos, sin = _rope_tables(g)
    with pytest.raises(ValueError, match="must have the same shape"):
        plan_norm_rope_write(
            **_plan_kwargs(cos_cache=cos, sin_cache=sin[: MAX_SEQ // 2])
        )


def test_plan_rejects_a_float_positions_tensor():
    with pytest.raises(ValueError, match="positions must be int32 or int64"):
        plan_norm_rope_write(**_plan_kwargs(positions=torch.zeros(8)))


def test_plan_rejects_a_positions_length_mismatch():
    with pytest.raises(ValueError, match="holds 4 entries for 8 tokens"):
        plan_norm_rope_write(
            **_plan_kwargs(positions=torch.arange(4, dtype=torch.int64))
        )


def test_plan_rejects_a_page_size_that_disagrees_with_the_cache():
    with pytest.raises(ValueError, match="does not match the cache's page size"):
        plan_norm_rope_write(**_plan_kwargs(page_size_tokens=PAGE_SIZE // 2))


def test_plan_rejects_a_non_bf16_k_raw():
    g = torch.Generator().manual_seed(1)
    with pytest.raises(ValueError, match="quantizes from bfloat16"):
        plan_norm_rope_write(
            **_plan_kwargs(k_raw=torch.randn(8, DATA_DIM, generator=g))
        )

