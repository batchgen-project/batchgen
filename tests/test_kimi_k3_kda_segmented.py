# ---------------------------------------------------------------------------- #
#  BatchGen — Kimi-K3                                                           #
#  copyright (c) EfficientMoE team 2025                                         #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Segmented KDA prefill is BIT-EXACT against the single-call sweep.

Under test: ``_kda_segment_plan`` and ``_kda_chunk_segments`` in
``batchgen/models/moonshotai/kimi_linear/serving_modules.py``, which run
``chunk_kda`` over a packed prefill in token segments and thread the recurrent
state between them through the KDA state pool
(``batchgen_design/model_support/kimi_k3/PREFILL_MEMORY_AUDIT.md``, fix 5:
51.0 GiB per KDA layer -> 18.0 GiB + 4.5 GiB per segment at S=131,072 /
T=16,384).

The gate is ``torch.equal``, not a tolerance, for the output AND for the final
recurrent states left in the pool. K3's top-16 router is discontinuous, so a
1-ulp seam upstream of it becomes an O(1) logit difference
(tests/gpu/test_kimi_k3_kda_fla_parity.py::test_E_kernel_seam_amplification).

WHY IT CAN BE EXACT.  A chunked linear-attention delta rule is chunk-local
except for one fp32 state that crosses chunk boundaries. If every segment cut
lands on a chunk boundary OF THE SEQUENCE IT FALLS IN, each segment's chunk
grid is exactly the restriction of the unsegmented grid: identical chunks,
identical per-chunk arithmetic, and the only thing that crosses the cut is that
same fp32 state, now via an fp32 round-trip through the pool instead of a
register. Nothing is re-associated and no reduction is re-cut.

WHAT THESE CPU TESTS PIN, AND WHAT THEY DO NOT.

  PINNED — the segmentation bookkeeping, which is where the bugs are: cut
  placement and chunk alignment, the sequence range and rebased cu_seqlens
  handed to each segment, the slot slice, the state hand-off through the pool
  (including the PARTIAL state written for a sequence that straddles a cut),
  and the stitching of segment outputs. ``_cpu_chunk_kda`` below is a
  transcription of ``fla/ops/kda/naive.py::naive_chunk_kda`` unrolled to one
  chunk at a time so it accepts a ragged tail; it is chunk-local in exactly the
  way the Triton kernel is, so a misplaced cut changes its answer
  (``test_control_misaligned_cuts_break_exactness`` proves it does).

  NOT PINNED — that fla's Triton kernels are themselves invariant to the number
  of tokens per launch. That is argued from the 0.5.2 source, not measured
  here: every autotune ``key=`` on the KDA forward path
  (``chunk_delta_h``, ``kda/gate``, ``kda/chunk_intra``, ``kda/wy_fast``,
  ``gla/chunk``) lists only head/dim/BT constants and never T, so all segments
  select the same config; ``h0``/``ht`` are fp32 both ways
  (``chunk_delta_h.py`` L707 ``new_zeros(..., dtype=torch.float32)``, L137
  ``b_h1 += tl.load(p_h0_1).to(tl.float32)``, L314 ``tl.store(p_ht, b_h1)``);
  and the ``FlashKDABackend`` verifier keys off flags, not shapes. A GPU run has
  to confirm it.

CPU-only, no ``import batchgen``: ``serving_modules.py`` is loaded by file path
with ``.wrappers`` stubbed, because the real package ``__init__`` JIT-builds a
CUDA extension and a sibling CPU test replaces ``sys.modules["batchgen"]`` with
a stub of its own.

Run: python -m pytest tests/test_kimi_k3_kda_segmented.py -q
"""

import importlib
import random
import sys
import types
from pathlib import Path

import pytest
import torch

_KL_DIR = (Path(__file__).resolve().parents[1] / "batchgen" / "models"
           / "moonshotai" / "kimi_linear")


def _load_serving_modules(alias="_kl_kdaseg"):
    """Import ``kimi_linear/serving_modules.py`` as ``alias.serving_modules``
    without running the real package ``__init__``. Its only package-relative
    import is ``.wrappers``, which it uses in the MoE path only."""
    if alias not in sys.modules:
        pkg = types.ModuleType(alias)
        pkg.__path__ = [str(_KL_DIR)]
        sys.modules[alias] = pkg
        wrappers = types.ModuleType(alias + ".wrappers")
        wrappers.KimiLinearExpertWrapper = type(
            "KimiLinearExpertWrapper", (), {})
        sys.modules[alias + ".wrappers"] = wrappers
    return importlib.import_module(alias + ".serving_modules")


SM = _load_serving_modules()

# The reference is a Python chunk loop, so every op sees the same shapes in the
# segmented and unsegmented runs. Single-threaded anyway: torch's CPU backend is
# not unconditionally element-count invariant (see
# tests/test_kimi_linear_ffn_chunk.py), and a threading artefact here would be
# indistinguishable from a real segmentation bug.
torch.set_num_threads(1)

BT = SM._KDA_CHUNK_SIZE

# The chunk size fla ACTUALLY uses, written out rather than read from the
# module under test. `_KDA_CHUNK_SIZE` is the whole exactness argument — every
# cut has to land on fla's own chunk grid — so a test that derives its
# expectation from that same constant cannot detect the constant being wrong.
# MEASURED at the source: fla 0.5.2 `fla/ops/kda/chunk.py`
#   chunk_size = kwargs.pop("chunk_size", 64)
# and `kda_prefill_serving` never passes `chunk_size`.
FLA_CHUNK_SIZE = 64

H, D = 2, 8


# --------------------------------------------------------------------------- #
#  CPU stand-in for fla.ops.kda.chunk_kda                                       #
# --------------------------------------------------------------------------- #
def _kda_gate(g, A_log, dt_bias, lower_bound):
    """fla/ops/kda/gate.py: lower_bound * sigmoid(exp(A_log) * (g + dt_bias))."""
    g = g.float() + dt_bias.view(H, D)
    return lower_bound * torch.sigmoid(A_log.view(H, 1).float().exp() * g)


def _l2norm(x):
    return x / x.float().pow(2).sum(-1, keepdim=True).sqrt()


def _cpu_chunk_kda(q, k, v, g, beta, A_log, dt_bias, use_qk_l2norm_in_kernel,
                   use_gate_in_kernel, use_beta_sigmoid_in_kernel, lower_bound,
                   initial_state, output_final_state, cu_seqlens):
    """naive_chunk_kda, per sequence, one chunk at a time (ragged tail allowed).

    Same call signature as ``fla.ops.kda.chunk_kda`` for the arguments
    ``kda_prefill_serving`` actually passes.
    """
    assert use_qk_l2norm_in_kernel and use_gate_in_kernel
    assert use_beta_sigmoid_in_kernel and output_final_state
    assert q.shape[0] == 1
    scale = q.shape[-1] ** -0.5
    bounds = cu_seqlens.tolist()

    o = torch.zeros(1, q.shape[1], H, D, dtype=v.dtype)
    final = torch.zeros_like(initial_state)
    for n in range(len(bounds) - 1):
        lo, hi = bounds[n], bounds[n + 1]
        qs = _l2norm(q[0, lo:hi].float()).transpose(0, 1) * scale  # (H, L, D)
        ks = _l2norm(k[0, lo:hi].float()).transpose(0, 1)
        vs = v[0, lo:hi].float().transpose(0, 1)
        gs = _kda_gate(g[0, lo:hi], A_log, dt_bias, lower_bound).transpose(0, 1)
        bs = torch.sigmoid(beta[0, lo:hi].float()).transpose(0, 1)  # (H, L)

        S = initial_state[n].float().clone()  # (H, D, D)
        for c in range(0, hi - lo, BT):
            sl = slice(c, min(c + BT, hi - lo))
            bt = sl.stop - sl.start
            q_i, k_i, v_i, b_i = qs[:, sl], ks[:, sl], vs[:, sl], bs[:, sl]
            g_i = gs[:, sl].cumsum(-2)  # chunk-local cumsum

            A = torch.zeros(H, bt, bt)
            for i in range(bt):
                A[..., i] = torch.einsum(
                    'hcd,hd->hc', k_i * (g_i - g_i[:, i:i + 1]).exp(), k_i[:, i])
            A = A * b_i[..., None]
            A = -A.masked_fill(torch.ones(bt, bt, dtype=torch.bool).triu(0), 0)
            for i in range(1, bt):
                A[..., i, :i] = (A[..., i, :i].clone()
                                 + (A[..., i, :, None].clone()
                                    * A[..., :, :i].clone()).sum(-2))
            A = (A + torch.eye(bt)) * b_i[..., None, :]

            w = A @ (g_i.exp() * k_i)
            u = A @ v_i

            Aqk = torch.zeros(H, bt, bt)
            for j in range(bt):
                Aqk[..., j] = torch.einsum(
                    'hcd,hd->hc', q_i * (g_i - g_i[:, j:j + 1]).exp(), k_i[:, j])
            Aqk = Aqk.masked_fill(torch.ones(bt, bt, dtype=torch.bool).triu(1), 0)

            vv = u - w @ S
            o[0, lo + sl.start:lo + sl.stop] = (
                ((q_i * g_i.exp()) @ S + Aqk @ vv).transpose(0, 1).to(v.dtype))
            S = S * g_i[:, -1].exp()[..., None]
            S = S + ((g_i[:, -1:] - g_i).exp() * k_i).transpose(-1, -2) @ vv
        final[n] = S
    return o, final


_KERNEL_KWARGS = dict(
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    lower_bound=-5.0,
    output_final_state=True,
)


def _inputs(seed, cu, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    total = cu[-1]

    def r(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    def a(*shape):
        return r(*shape).to(dtype)

    return dict(
        q=a(1, total, H, D), k=a(1, total, H, D), v=a(1, total, H, D),
        f=a(1, total, H, D), beta=a(1, total, H),
        cu_seqlens=torch.tensor(cu, dtype=torch.int32),
        slot_ids=torch.tensor(
            random.Random(seed).sample(range(8), len(cu) - 1),
            dtype=torch.int32),
        kwargs=dict(A_log=r(H), dt_bias=r(H * D), **_KERNEL_KWARGS),
    )


def _sweep(inp, segment_tokens, pool_seed=0):
    """One full KDA sweep; returns (output, pool after the sweep)."""
    gen = torch.Generator().manual_seed(pool_seed)
    pool = torch.randn(8, H, D, D, generator=gen, dtype=torch.float32)
    o = SM._kda_chunk_segments(
        _cpu_chunk_kda, inp["q"], inp["k"], inp["v"], inp["f"], inp["beta"],
        inp["cu_seqlens"], inp["slot_ids"], pool, inp["kwargs"], segment_tokens)
    return o, pool


# --------------------------------------------------------------------------- #
#  the plan                                                                     #
# --------------------------------------------------------------------------- #
CASES = [
    ([0, 512], 128),               # one sequence, cuts strictly inside it
    ([0, 137, 320, 400], 128),     # 3 ragged sequences, cuts inside and across
    ([0, 64, 512, 576, 1000], 256),
    ([0, 1000], 64),               # segment == chunk size
    ([0, 33, 70, 91], 64),         # sequences shorter than a segment
]


@pytest.mark.parametrize("cu,seg", CASES)
def test_plan_tiles_the_range_and_respects_sequences(cu, seg):
    plan = SM._kda_segment_plan(cu, seg)
    total = cu[-1]

    assert plan[0][0] == 0 and plan[-1][1] == total
    for (a, b, lo, hi, bounds), (na, *_) in zip(plan, plan[1:] + [(total,)]):
        assert a < b <= a + seg, \
            "segment [{}, {}) is empty or over budget".format(a, b)
        assert b == na, "segments do not tile the range"
        # every cut is a sequence boundary or a chunk boundary of its sequence
        if b != total:
            j = max(i for i in range(len(cu)) if cu[i] <= b)
            # FLA_CHUNK_SIZE, not BT: the assertion must hold against fla's
            # real grid even if _KDA_CHUNK_SIZE is mis-set.
            assert b == cu[j] or (b - cu[j]) % FLA_CHUNK_SIZE == 0, (
                "cut {} is {} tokens into sequence {} — not a multiple of the "
                "chunk_kda chunk size {}".format(
                    b, b - cu[j], j, FLA_CHUNK_SIZE))
        # the rebased cu_seqlens must be exactly the clamped intersections
        assert bounds == [min(max(c, a), b) - a for c in cu[lo:hi + 1]]
        assert bounds[0] == 0 and bounds[-1] == b - a
        assert all(y > x for x, y in zip(bounds, bounds[1:])), \
            "a segment declared a zero-length sequence"

    # every sequence is covered exactly once, in order
    for i in range(len(cu) - 1):
        pieces = [(max(a, cu[i]), min(b, cu[i + 1]))
                  for a, b, lo, hi, _ in plan if lo <= i < hi]
        assert pieces[0][0] == cu[i] and pieces[-1][1] == cu[i + 1]
        assert all(x[1] == y[0] for x, y in zip(pieces, pieces[1:]))


def test_plan_is_a_single_segment_when_the_batch_fits():
    assert SM._kda_segment_plan([0, 100, 300], 512) == [
        (0, 300, 0, 2, [0, 100, 300])]


def test_plan_rejects_a_misaligned_segment_size():
    with pytest.raises(ValueError, match="multiple of"):
        SM._kda_segment_plan([0, 512], 100)
    with pytest.raises(ValueError, match="multiple of"):
        SM._kda_segment_plan([0, 512], 0)


@pytest.mark.parametrize("cu", [
    [0, 0, 512],            # leading
    [0, 512, 512],          # trailing
    [0, 64, 64, 128],       # INTERIOR — invisible to any coverage check that
    [0, 128, 128, 128, 256],  # only inspects the first and last segment
    [0, 300, 200],          # non-monotonic outright
])
def test_plan_refuses_to_drop_a_zero_length_sequence(cu):
    """A dropped sequence would silently leave its pool state stale. The conv1d
    kernel cannot handle a zero-length sequence either, so this is a hard
    failure rather than a special case.

    The interior cases are the ones that matter: the segments on either side of
    a zero-length sequence still tile the token axis perfectly and still start
    at sequence 0 and end at the last sequence, so only a check on the INPUT
    catches them."""
    with pytest.raises(ValueError, match="strictly increasing"):
        SM._kda_segment_plan(cu, 128)


def test_chunk_size_constant_matches_flas_own():
    """``_KDA_CHUNK_SIZE`` is load-bearing for exactness, not a tuning knob.

    Every other test in this file constructs its expectations from it, so none
    of them can see it being wrong — MEASURED: setting it to 32 leaves all 46
    of them green while producing cuts that land mid-chunk in the real kernel.
    This is the one assertion that pins it, against a literal.
    """
    assert SM._KDA_CHUNK_SIZE == FLA_CHUNK_SIZE, (
        "_KDA_CHUNK_SIZE={} but fla chunks at {}: segment cuts would land "
        "inside a chunk, re-cutting the gate cumsum and the WY transform, and "
        "the sweep would no longer be bit-exact"
        .format(SM._KDA_CHUNK_SIZE, FLA_CHUNK_SIZE))
    assert SM.KDA_PREFILL_SEGMENT_TOKENS % FLA_CHUNK_SIZE == 0

    try:                                    # exact when fla is installed
        import inspect

        from fla.ops.kda import chunk as fla_chunk
        src = inspect.getsource(fla_chunk)
        assert 'chunk_size = kwargs.pop("chunk_size", {})'.format(
            FLA_CHUNK_SIZE) in src or \
            "chunk_size = kwargs.pop('chunk_size', {})".format(
                FLA_CHUNK_SIZE) in src, (
            "fla's chunk_kda default chunk size is no longer {}"
            .format(FLA_CHUNK_SIZE))
    except ImportError:
        pass                                # CPU dev box: the literal stands


def test_bias_free_output_projection_reuses_caller_storage_bit_exactly():
    gen = torch.Generator().manual_seed(20260902)
    linear = torch.nn.Linear(7, 11, bias=False)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(11, 7, generator=gen))
    x = torch.randn(13, 7, generator=gen)
    out = torch.empty(13, 11)

    # ``out=`` GEMMs are inference-only, matching the worker's prefill loop.
    with torch.inference_mode():
        expected = linear(x)
        actual = SM._linear_no_bias_into(linear, x, out)

    assert actual.data_ptr() == out.data_ptr()
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("seed", range(20))
def test_plan_invariants_on_random_batches(seed):
    rng = random.Random(seed)
    cu, t = [0], 0
    for _ in range(rng.randint(1, 6)):
        t += rng.randint(1, 400)
        cu.append(t)
    seg = BT * rng.randint(1, 5)
    plan = SM._kda_segment_plan(cu, seg)
    assert [p[0] for p in plan][1:] == [p[1] for p in plan][:-1]
    assert plan[-1][1] == cu[-1]
    for a, b, lo, hi, bounds in plan:
        assert 0 < b - a <= seg
        assert bounds == [min(max(c, a), b) - a for c in cu[lo:hi + 1]]
        if b != cu[-1]:
            j = max(i for i in range(len(cu)) if cu[i] <= b)
            assert b == cu[j] or (b - cu[j]) % BT == 0


# --------------------------------------------------------------------------- #
#  the sweep                                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cu,seg", CASES)
def test_segmented_sweep_is_bit_identical(cu, seg, dtype):
    inp = _inputs(hash((tuple(cu), seg)) & 0xFFFF, cu, dtype)
    ref_o, ref_pool = _sweep(inp, None)
    seg_o, seg_pool = _sweep(inp, seg)

    n_segments = len(SM._kda_segment_plan(cu, seg))
    assert torch.equal(seg_o, ref_o), (
        "{} segments of {} tokens over cu={} moved the output: max_abs={:.3e}"
        .format(n_segments, seg, cu,
                (seg_o.float() - ref_o.float()).abs().max().item()))
    assert torch.equal(seg_pool, ref_pool), (
        "final recurrent states differ: max_abs={:.3e}"
        .format((seg_pool - ref_pool).abs().max().item()))


@pytest.mark.parametrize("cu", [[0, 512], [0, 137, 320, 400]])
def test_answer_is_invariant_to_the_segment_size(cu):
    """Any chunk-aligned segment size gives the same bits — the strongest form
    of the claim, and the one a future retune of KDA_PREFILL_SEGMENT_TOKENS
    depends on."""
    inp = _inputs(23, cu)
    ref, ref_pool = _sweep(inp, None)
    for seg in (BT, 2 * BT, 3 * BT, 5 * BT, 64 * BT):
        o, pool = _sweep(inp, seg)
        assert torch.equal(o, ref) and torch.equal(pool, ref_pool), (
            "segment size {} changed the answer".format(seg))


def test_the_exactness_claim_is_not_vacuous():
    """The bit-exact cases above must actually exercise both hazards."""
    cu, seg = [0, 137, 320, 400], 128
    plan = SM._kda_segment_plan(cu, seg)
    assert len(plan) > 1, "nothing is being segmented"
    assert any(not any(c == b for c in cu) for _, b, _, _, _ in plan[:-1]), (
        "no cut falls strictly inside a sequence — the state hand-off is "
        "never exercised")
    assert any(hi - lo > 1 for _, _, lo, hi, _ in plan), (
        "no segment spans a sequence boundary — the cu_seqlens rebasing is "
        "never exercised")
    straddling = [i for i in range(len(cu) - 1)
                  if sum(1 for _, _, lo, hi, _ in plan if lo <= i < hi) > 1]
    assert straddling, "no sequence is written to the pool more than once"


def test_control_misaligned_cuts_break_exactness():
    """Drop the chunk alignment and the same driver stops being exact.

    Without this the ``torch.equal`` above could be passing because the
    reference is insensitive to where the cuts land, which would make the
    alignment logic — the whole substance of the fix — untested.
    """
    cu, seg = [0, 400], 100
    inp = _inputs(7, cu)
    ref_o, ref_pool = _sweep(inp, None)

    saved = SM._KDA_CHUNK_SIZE
    SM._KDA_CHUNK_SIZE = 1  # cuts now land at 100 / 200 / 300, mid-chunk
    try:
        bad_o, bad_pool = _sweep(inp, seg)
    finally:
        SM._KDA_CHUNK_SIZE = saved

    assert not torch.equal(bad_o, ref_o), (
        "cutting mid-chunk produced the identical output — the reference is "
        "not chunk-local and proves nothing about alignment")
    assert not torch.equal(bad_pool, ref_pool)
    # ... and it is a rounding-level difference, i.e. the recurrence itself is
    # still correct; alignment is about bits, not about being wrong.
    assert (bad_o - ref_o).abs().max().item() < 1e-3


def test_control_a_leaked_recurrence_is_visible():
    """Two sequences sharing a pool slot must NOT give the reference answer."""
    cu, seg = [0, 192, 384], 128
    inp = _inputs(11, cu)
    ref_o, _ = _sweep(inp, None)
    inp["slot_ids"] = torch.tensor([3, 3], dtype=torch.int32)
    leaked_o, _ = _sweep(inp, seg)
    assert not torch.equal(leaked_o, ref_o)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_output_buffer_matches_v(dtype):
    cu = [0, 137, 320]
    o, _ = _sweep(_inputs(3, cu, dtype), 128)
    assert o.shape == (1, cu[-1], H, D) and o.dtype == dtype
    assert o.is_contiguous(), (
        "the stitched output must be contiguous — o_norm/o_proj reshape it")
