# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""KimiMLP's token-tiled FFN is BIT-EXACT against the unchunked body.

Under test: ``KimiMLP.forward`` in
``batchgen/models/moonshotai/kimi_linear/model.py``, which tiles the dense
layer-0 MLP and the MoE shared expert over tokens so the SiTU fp32 island stops
scaling with S (batchgen_design/model_support/kimi_k3/PREFILL_MEMORY_AUDIT.md,
fix 1: 98.998 GiB -> 6.19 GiB in layer 0).

The gate is ``torch.equal``, not a tolerance. K3's top-16 router is
discontinuous, so a 1-ulp seam anywhere upstream of it becomes an O(1) logit
difference (tests/gpu/test_kimi_k3_kda_fla_parity.py::
test_E_kernel_seam_amplification); a memory optimisation that moves the last
bit is not an option. The reference is ``KimiMLP._ffn`` — the pre-chunking body
verbatim — applied to the whole input in one call.

WHAT THE CPU TESTS PIN, AND WHAT THEY DO NOT.  Tiling is exact if and only if
each op is invariant to how many rows are handed to it at once. That is a
property of the *backend*, not of the tiling, and torch's CPU backend does not
have it unconditionally — MEASURED here, on arm64 with 6 threads:

  * fp32 ``torch.sigmoid`` is NOT element-count-invariant. Splitting a
    (391, 384) fp32 tensor into 7 row-blocks moves one block by 1 ulp
    (1.49e-08), because ``at::parallel_for`` re-cuts the element range and the
    vectorized/scalar tail lands on different elements. Single-threaded it is
    invariant; in bf16 it is invariant at 6 threads too.
  * fp32 ``F.linear`` at M=1 is not row-block invariant either (a 1-row GEMM is
    a GEMV). This is why ``forward`` splits into EVEN tiles and never emits a
    degenerate tail — ``test_tiles_are_even``.

THE fp32 ARM OF THE SWEEP IS LOAD-BEARING, and not as a secondary dtype: it is
the only arm that has the resolution to see a degenerate tile. MEASURED — with
``forward`` mutated back to `fixed width + ragged remainder`, the suite goes
red on exactly four cases, all fp32 (``[65-dtype0-situ]``, ``[65-dtype0-silu]``,
``[257-dtype0-situ]``, ``[257-dtype0-silu]``, i.e. num_tokens = k*TILE+1); every
bf16 case at the same token counts stays green, because bf16's 8-bit mantissa
rounds the GEMV/GEMM difference away. Production is bf16. Do not "simplify" the
sweep down to the production dtype — that deletes the only detector for the
trap the even split exists to avoid.

So the ``torch.equal`` sweeps run single-threaded, where the CPU backend IS
element-count invariant and any failure is therefore the tiling's fault: index
arithmetic, a dropped or duplicated row, the ragged final tile, output dtype or
shape. ``test_bf16_exact_at_native_thread_count`` then re-runs the production
dtype at the machine's real thread count, and
``test_backend_invariance_notes`` records the two backend measurements above so
nobody deletes the single-thread pin without knowing what it is for. The GPU
answer — CUDA elementwise kernels apply the same functor to every element and
these GEMM shapes never reach split-K, so both should be size-invariant — is
asserted by ``test_chunked_equals_unchunked_cuda``, which skips off-GPU.

CPU-only, no ``import batchgen``: ``model.py`` is loaded by file path with
``fla`` and ``.config`` stubbed, because the FFN path touches neither.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
import weakref
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
KL_PKG_DIR = ROOT / "batchgen" / "models" / "moonshotai" / "kimi_linear"


# --------------------------------------------------------------------------- #
#  Module loading (file-path, no batchgen import)                              #
# --------------------------------------------------------------------------- #
def _load_by_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_stubs() -> None:
    """Stub exactly what ``model.py`` imports at module scope and the FFN never
    calls: the fla kernels (KDA) and ``.config`` (which would pull in
    ``batchgen.config``, i.e. the JIT core-engine build)."""
    for name in ("fla", "fla.modules", "fla.ops", "fla.ops.kda"):
        sys.modules.setdefault(name, types.ModuleType(name))
    fla_modules = sys.modules["fla.modules"]
    for attr in ("FusedRMSNormGated", "ShortConvolution"):
        setattr(fla_modules, attr, type(attr, (), {}))
    fla_kda = sys.modules["fla.ops.kda"]
    for attr in ("chunk_kda", "fused_recurrent_kda"):
        setattr(fla_kda, attr, lambda *a, **k: None)

    if "_kl" not in sys.modules:
        pkg = types.ModuleType("_kl")
        pkg.__path__ = [str(KL_PKG_DIR)]
        sys.modules["_kl"] = pkg
    if "_kl.config" not in sys.modules:
        cfg_mod = types.ModuleType("_kl.config")
        cfg_mod.KimiLinearConfig = type("KimiLinearConfig", (), {})
        sys.modules["_kl.config"] = cfg_mod
    if "_kl.block_residual" not in sys.modules:
        _load_by_path("_kl.block_residual", KL_PKG_DIR / "block_residual.py")


@pytest.fixture(scope="module")
def M():
    _install_stubs()
    return _load_by_path("_kl.model", KL_PKG_DIR / "model.py")


NATIVE_THREADS = torch.get_num_threads()


@pytest.fixture(autouse=True)
def single_thread():
    """See the module docstring: multithreaded fp32 ``sigmoid`` on CPU is not
    element-count invariant, which is a torch-CPU property and not something
    the tiling can fix or is allowed to hide. Pin one thread so these tests
    measure the tiling."""
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(NATIVE_THREADS)


# --------------------------------------------------------------------------- #
#  Fixtures                                                                    #
# --------------------------------------------------------------------------- #
# Narrow enough to run in a second on CPU, wide enough that the projections are
# real GEMMs (K=256) rather than a dot product the backend cannot get wrong.
HIDDEN = 256
INTERMEDIATE = 384
TILE = 64          # stands in for the production _FFN_TOKEN_TILE = 8192


def _config(act: str):
    """KimiMLP reads only these four fields (plus build_activation's getattrs).
    K3's real numbers: situ, beta=4.0, linear_beta=25.0."""
    return types.SimpleNamespace(
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        hidden_act=act,
        activation_situ_beta=4.0 if act == "situ" else None,
        activation_situ_linear_beta=25.0 if act == "situ" else None,
    )


def _mlp(M, act: str, dtype: torch.dtype, hidden=HIDDEN, inter=INTERMEDIATE,
         device="cpu"):
    torch.manual_seed(20260808)
    cfg = _config(act)
    cfg.hidden_size, cfg.intermediate_size = hidden, inter
    return M.KimiMLP(cfg).to(device=device, dtype=dtype).eval()


# Token counts around the tile width. 4*TILE+1 and 2*TILE+3 are the ones that
# matter: a fixed-width split would end them on a 1-row and a 3-row tile. 391
# is the ragged case that first exposed the multithreaded-sigmoid property.
TOKEN_COUNTS = [1, 7, TILE - 1, TILE, TILE + 1, TILE + 7,
                2 * TILE, 2 * TILE + 3, 4 * TILE + 1, 5 * TILE - 1, 391]


# --------------------------------------------------------------------------- #
#  T1 — chunked == unchunked, bitwise                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("act", ["situ", "silu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
def test_chunked_equals_unchunked_2d(M, monkeypatch, act, dtype, num_tokens):
    monkeypatch.setattr(M, "_FFN_TOKEN_TILE", TILE)
    mlp = _mlp(M, act, dtype)
    x = torch.randn(num_tokens, HIDDEN, generator=torch.Generator().manual_seed(
        num_tokens)).to(dtype)

    with torch.inference_mode():
        ref = mlp._ffn(x)          # the pre-chunking body, one call
        got = mlp(x)               # tiled

    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert torch.equal(got, ref), (
        "chunked FFN differs from the unchunked body: act={} dtype={} "
        "num_tokens={} max|delta|={}".format(
            act, dtype, num_tokens, (got.float() - ref.float()).abs().max()))


@pytest.mark.parametrize("act", ["situ", "silu"])
@pytest.mark.parametrize("num_tokens", [TILE + 1, 4 * TILE + 1, 391])
def test_chunked_equals_unchunked_3d(M, monkeypatch, act, num_tokens):
    """Prefill hands the MLP a 3-D ``(1, S, H)`` tensor (batchgen_worker
    :7068), and the MoE shared expert gets the same ``identity``. The tiling
    flattens; the output shape and the bits must both survive."""
    monkeypatch.setattr(M, "_FFN_TOKEN_TILE", TILE)
    mlp = _mlp(M, act, torch.bfloat16)
    x = torch.randn(1, num_tokens, HIDDEN,
                    generator=torch.Generator().manual_seed(7)).bfloat16()

    with torch.inference_mode():
        ref = mlp._ffn(x)
        got = mlp(x)

    assert got.shape == (1, num_tokens, HIDDEN)
    assert torch.equal(got, ref)


def test_batch_gt_one(M, monkeypatch):
    """(B, S, H) with B>1: tokens are counted across the batch, so the tiles
    cut through the flattened row axis and not the sequence axis."""
    monkeypatch.setattr(M, "_FFN_TOKEN_TILE", TILE)
    mlp = _mlp(M, "situ", torch.float32)
    x = torch.randn(3, TILE + 5, HIDDEN,
                    generator=torch.Generator().manual_seed(11))

    with torch.inference_mode():
        ref = mlp._ffn(x)
        got = mlp(x)

    assert got.shape == (3, TILE + 5, HIDDEN)
    assert torch.equal(got, ref)


def test_short_input_takes_the_untiled_path(M):
    """<= one tile must be the ORIGINAL call — one `_ffn` on the whole input,
    no output buffer, no copy. Decode and CUDA-graph capture run through here,
    and the branch is on shape only, so capture stays shape-static."""
    mlp = _mlp(M, "situ", torch.float32)
    x = torch.randn(17, HIDDEN, generator=torch.Generator().manual_seed(3))
    calls = []
    real_ffn = mlp._ffn
    mlp._ffn = lambda t: (calls.append(tuple(t.shape)), real_ffn(t))[1]

    with torch.inference_mode():
        got = mlp(x)

    assert calls == [(17, HIDDEN)]
    assert torch.equal(got, real_ffn(x))


def test_resident_prefill_uses_smaller_tile_only_for_long_inputs(M, monkeypatch):
    monkeypatch.setattr(M, "_FFN_TOKEN_TILE", TILE)
    mlp = _mlp(M, "situ", torch.bfloat16)
    mlp._resident_prefill_token_tile = 8
    calls = []
    real_ffn = mlp._ffn
    mlp._ffn = lambda t: (calls.append(t.shape[0]), real_ffn(t))[1]

    short = torch.randn(8, HIDDEN).bfloat16()
    long = torch.randn(4 * TILE, HIDDEN).bfloat16()
    with torch.inference_mode():
        short_out = mlp(short)
        long_out = mlp(long)

    assert calls[0] == 8
    assert max(calls[1:]) <= 8
    assert torch.equal(short_out, real_ffn(short))
    assert torch.equal(long_out, real_ffn(long))

    calls.clear()
    ragged = torch.randn(4 * TILE + 1, HIDDEN).bfloat16()
    with torch.inference_mode():
        ragged_out = mlp(ragged)
    assert max(calls) > 8
    assert torch.equal(ragged_out, real_ffn(ragged))


@pytest.mark.parametrize("shape", [(4 * TILE + 1, HIDDEN),
                                   (1, 4 * TILE + 1, HIDDEN)])
def test_forward_into_may_alias_input(M, monkeypatch, shape):
    """The streamed-prefill seam reuses its dead MoE input as FFN output."""
    monkeypatch.setattr(M, "_FFN_TOKEN_TILE", TILE)
    mlp = _mlp(M, "situ", torch.bfloat16)
    source = torch.randn(
        *shape, generator=torch.Generator().manual_seed(20260902)
    ).bfloat16()

    with torch.inference_mode():
        expected = mlp(source.clone())
        alias = source.clone()
        result = mlp.forward_into(alias, alias)

    assert result.data_ptr() == alias.data_ptr()
    assert result.shape == expected.shape
    assert torch.equal(result, expected)


def test_forward_into_honors_resident_tile(M, monkeypatch):
    monkeypatch.setattr(M, "_FFN_TOKEN_TILE", TILE)
    mlp = _mlp(M, "situ", torch.bfloat16)
    mlp._resident_prefill_token_tile = 8
    source = torch.randn(4 * TILE, HIDDEN).bfloat16()
    calls = []
    real_ffn = mlp._ffn
    mlp._ffn = lambda t: (calls.append(t.shape[0]), real_ffn(t))[1]

    with torch.inference_mode():
        expected = mlp(source.clone())
        calls.clear()
        alias = source.clone()
        result = mlp.forward_into(alias, alias)

    assert max(calls) <= 8
    assert torch.equal(result, expected)


def test_forward_into_rejects_mismatched_output(M):
    mlp = _mlp(M, "situ", torch.bfloat16)
    x = torch.randn(7, HIDDEN).bfloat16()
    with pytest.raises(ValueError, match="matching tensors"):
        mlp.forward_into(x, torch.empty(8, HIDDEN, dtype=x.dtype))


def test_bf16_exact_at_native_thread_count(M, monkeypatch):
    """The production dtype, at the machine's real thread count — i.e. without
    the single-thread pin the fp32 sweeps need. MEASURED exact for every token
    count on arm64/6 threads."""
    monkeypatch.setattr(M, "_FFN_TOKEN_TILE", TILE)
    torch.set_num_threads(NATIVE_THREADS)
    mlp = _mlp(M, "situ", torch.bfloat16)
    for num_tokens in TOKEN_COUNTS:
        x = torch.randn(num_tokens, HIDDEN,
                        generator=torch.Generator().manual_seed(num_tokens)
                        ).bfloat16()
        with torch.inference_mode():
            assert torch.equal(mlp(x), mlp._ffn(x)), num_tokens


# --------------------------------------------------------------------------- #
#  T2 — the even-split rule                                                    #
# --------------------------------------------------------------------------- #
def _tile_sizes(num_tokens: int, tile: int):
    """The boundary arithmetic in KimiMLP.forward, transcribed."""
    n = math.ceil(num_tokens / tile)
    bounds = [(i * num_tokens) // n for i in range(n + 1)]
    return [b - a for a, b in zip(bounds, bounds[1:])]


def test_tiles_are_even(M):
    """No tile is ever a degenerate GEMM. A fixed-width split would emit a
    1-row tail at num_tokens = k*tile + 1; the even split keeps every tile at
    tile/2 or wider, never drops a row, and never over-runs the tile width."""
    tile = M._FFN_TOKEN_TILE
    for num_tokens in (tile + 1, 2 * tile + 1, 16 * tile + 1, 16 * tile + 3,
                       3 * tile - 1, 131072, 131069, 78039):
        sizes = _tile_sizes(num_tokens, tile)
        assert sum(sizes) == num_tokens
        assert min(sizes) >= tile // 2, (num_tokens, min(sizes))
        assert max(sizes) <= tile, (num_tokens, max(sizes))
        assert max(sizes) - min(sizes) <= 1


def test_previous_tile_is_freed_before_the_next(M, monkeypatch):
    """No tile's output is co-live with the next tile's peak.

    Tiling bounds the FFN island only if the loop actually holds ONE tile at a
    time. Python rebinds ``y`` on the next iteration's assignment — i.e. AFTER
    ``self._ffn(...)`` for that tile has already peaked — so without the
    explicit ``del`` the previous ``(tile, hidden)`` output sits underneath
    every peak (+0.109 GiB at K3 scale). Weakrefs, not byte counting: this
    asserts the liveness directly and is backend-independent.
    """
    monkeypatch.setattr(M, "_FFN_TOKEN_TILE", TILE)
    mlp = _mlp(M, "situ", torch.bfloat16)
    num_tokens = 4 * TILE + 1
    x = torch.randn(num_tokens, HIDDEN,
                    generator=torch.Generator().manual_seed(13)).bfloat16()

    real_ffn = mlp._ffn
    refs, alive_at_entry = [], []

    def spy(t):
        alive_at_entry.append(sum(r() is not None for r in refs))
        y = real_ffn(t)
        refs.append(weakref.ref(y))
        return y

    mlp._ffn = spy
    with torch.inference_mode():
        got = mlp(x)

    assert len(refs) == math.ceil(num_tokens / TILE) == 5
    assert alive_at_entry == [0] * len(refs), (
        "a previous tile's output was still alive when the next tile started: "
        "{} — the `del y` in KimiMLP.forward was removed".format(alive_at_entry))
    assert torch.equal(got, real_ffn(x))


def test_backend_invariance_notes():
    """The two backend measurements the module docstring rests on. Printed,
    never asserted: they are properties of this torch build, and the tiling is
    written so production never depends on either (even tiles avoid the
    degenerate GEMM; the GPU avoids the threading artifact entirely)."""
    torch.set_num_threads(NATIVE_THREADS)
    g = torch.Generator().manual_seed(1)

    x = torch.randn(64, 1024, generator=g)
    w = torch.randn(1536, 1024, generator=g) / 32
    full = torch.nn.functional.linear(x, w)
    m1 = all(torch.equal(torch.nn.functional.linear(x[i:i + 1], w), full[i])
             for i in range(x.shape[0]))
    m8 = all(torch.equal(torch.nn.functional.linear(x[i:i + 8], w), full[i:i + 8])
             for i in range(0, x.shape[0], 8))

    t = torch.randn(391, 384, generator=g)
    bounds = [(i * 391) // 7 for i in range(8)]
    sig = torch.sigmoid(t)
    sig_ok = all(torch.equal(torch.sigmoid(t[s:e]), sig[s:e])
                 for s, e in zip(bounds, bounds[1:]))
    t16 = t.bfloat16()
    sig16 = torch.sigmoid(t16.float())
    sig16_ok = all(torch.equal(torch.sigmoid(t16[s:e].float()), sig16[s:e])
                   for s, e in zip(bounds, bounds[1:]))

    print("\n[backend, {} threads] fp32 linear row-block exact: M=1 {}, M=8 {}"
          "\n[backend] fp32 sigmoid element-count invariant: {}  (bf16-sourced: {})"
          .format(NATIVE_THREADS, m1, m8, sig_ok, sig16_ok))


# --------------------------------------------------------------------------- #
#  T3 — the GPU answer (skips off-GPU; run it on the node)                     #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("inter,num_tokens", [
    # MoE shared expert (2 x moe_intermediate_size 3072)
    (6144, 8192 + 1), (6144, 2 * 8192 + 517), (6144, 3 * 8192),
    # LAYER 0's DENSE MLP — the width fix 1 actually exists for, and a
    # different cuBLAS regime (N = 2*33792 = 67,584). 6144 passing does not
    # imply 33792 passes; both must be run. Unchunked reference transient is
    # 24 * num_tokens * inter = 6.6 GiB and 13.8 GiB for these two cases.
    (33792, 8192 + 1), (33792, 16384), (33792, 2 * 8192 + 517),
])
def test_chunked_equals_unchunked_cuda(M, inter, num_tokens):
    """Real K3 widths (hidden 7168), production dtype, production tile. This is
    the assertion the CPU cannot make: that CUDA's elementwise kernels and these
    GEMM shapes are both invariant to how many rows arrive at once. num_tokens
    = 8192+1 is the degenerate-tail trap the even split exists to avoid."""
    mlp = _mlp(M, "situ", torch.bfloat16, hidden=7168, inter=inter,
               device="cuda")
    x = torch.randn(1, num_tokens, 7168, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(5)
                    ).bfloat16()

    with torch.inference_mode():
        ref = mlp._ffn(x)
        got = mlp(x)
        mlp._resident_prefill_token_tile = 512
        resident_got = mlp(x)

    assert torch.equal(got, ref), (
        "CUDA: chunked FFN differs from the unchunked body at inter={} "
        "num_tokens={} max|delta|={}".format(
            inter, num_tokens, (got.float() - ref.float()).abs().max()))
    assert torch.equal(resident_got, ref), (
        "CUDA: resident-prefill FFN tile differs from the unchunked body at "
        "inter={} num_tokens={} max|delta|={}".format(
            inter, num_tokens,
            (resident_got.float() - ref.float()).abs().max()))
