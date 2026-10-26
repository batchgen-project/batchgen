"""The decode sampler's one-sample draw must not sync, and must not change.

On the host-decided (decode step) path every sampled row used to be drawn
with `torch.multinomial(probs, 1)`. For one sample ATen implements that as the
exponential race -- `argmax(probs / Exp(1))` -- but first reads two validity
reductions back to the host (`.item()`), i.e. two cudaStreamSynchronize per
decode step. `SAMPLED_DRAW_SYNC_FREE` replaces the call with the same race,
launched without the readbacks, and moves the validity check onto the
device-side bad-row fallback.

What is pinned here:

* **Token identity** -- the draw equals `torch.multinomial` under the same
  seed, both in isolation and through `sample_tokens` (fp32 / bf16, 1..64
  rows, two vocab sizes, plain M1-shape / filtered / mixed-greedy rows, with
  and without NaN and all-masked rows), against the switch-OFF path and, when
  no row is bad, against the reference (device-mask) path.
* **RNG stream position** -- the generator state left behind is the one
  `torch.multinomial` leaves, so the next draw continues identically.
* **No sync** -- the dispatch trace of the switch-ON path contains no syncing
  op and no `multinomial`; the switch-OFF trace is the negative control.
* **Validation is kept, not dropped** -- every row `torch.multinomial` would
  reject (NaN, inf, negative entry, zero sum) is flagged on device, takes the
  deterministic greedy fallback and is counted; the switch-OFF path still
  raises on the same input.
* **Switch integrity** -- default ON, read at call time, OFF selects the old
  call and nothing of the new one.

CUDA: `test_cuda_*` run only where a GPU is present. CPU equality does not
prove CUDA equality (different generator, different offset accounting), so
those tests are the node gate for this change.
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn.functional as F

import batchgen.sampling as sampling
from batchgen.sampling import _draw_one_per_row, _invalid_draw_rows, sample_tokens
from batchgen.worker.sampling_residency import attach_row_index, host_decisions_from_cast
from tests.test_sampling_split_sync_free import _OpTrace

SEED = 20261025


def _set_switch(value):
    previous = sampling.SAMPLED_DRAW_SYNC_FREE
    sampling.SAMPLED_DRAW_SYNC_FREE = value
    return previous


def _decisions(temps, top_ps, top_ks, dtype, device="cpu"):
    """Decisions taken on the cast values, exactly as the residency does."""
    return attach_row_index(
        host_decisions_from_cast(
            temps.to(dtype).tolist(), top_ps.to(dtype).tolist(), top_ks.tolist()
        ),
        device,
    )


def _host_sample(logits, params, seed, sync_free, device="cpu"):
    """One host-decided `sample_tokens` call; returns tokens, counter, RNG state."""
    temps, top_ps, top_ks = params
    decisions = _decisions(temps, top_ps, top_ks, logits.dtype, device)
    counter = torch.zeros((), dtype=torch.int64, device=device)
    previous = _set_switch(sync_free)
    try:
        torch.manual_seed(seed)
        out = sample_tokens(
            logits,
            temperature=temps.to(device),
            top_p=top_ps.to(device),
            top_k=top_ks.to(device),
            host_decisions=decisions,
            bad_row_counter=counter,
        )
        state = (
            torch.cuda.get_rng_state(logits.device)
            if logits.device.type == "cuda"
            else torch.get_rng_state()
        )
    finally:
        _set_switch(previous)
    return out, int(counter), state


# --------------------------------------------------------------------------
# Case generation
# --------------------------------------------------------------------------

#: Sampling-parameter shapes. "m1": every row t=1, top_p=1, top_k=0 (the
#: measured decode workload). "filtered": per-row temperature / top-p / top-k.
#: "mixed": greedy rows interleaved with sampled ones (index split).
MODES = ("m1", "filtered", "mixed")
ROWS = (1, 3, 16, 64)
VOCABS = (48, 1024)
#: Bad-row patterns, applied to sampled rows only: a NaN logit row, and an
#: all -inf logit row (softmax -> NaN, the "filtering removed everything"
#: case), alone and together.
BAD = ("none", "nan", "masked", "both")
SEEDS_PER_CASE = 3


def _params(mode, rows, seed):
    g = torch.Generator().manual_seed(seed)
    if mode == "m1":
        return (torch.ones(rows), torch.ones(rows), torch.zeros(rows, dtype=torch.int64))
    temps = 0.4 + 1.2 * torch.rand(rows, generator=g)
    top_ps = torch.where(torch.rand(rows, generator=g) < 0.5, torch.ones(rows),
                         0.3 + 0.6 * torch.rand(rows, generator=g))
    top_ks = torch.where(torch.rand(rows, generator=g) < 0.5,
                         torch.zeros(rows, dtype=torch.int64),
                         torch.randint(1, 20, (rows,), generator=g))
    if mode == "mixed" and rows > 1:
        temps[::2] = 0.0
    return temps, top_ps, top_ks


def _sampled_rows(temps, dtype):
    return [i for i, t in enumerate(temps.to(dtype).tolist()) if t > 0]


def _logits(rows, vocab, dtype, seed, bad, temps):
    g = torch.Generator().manual_seed(seed)
    logits = (3.0 * torch.randn(rows, vocab, generator=g)).to(dtype)
    sampled = _sampled_rows(temps, dtype)
    if bad in ("nan", "both") and sampled:
        logits[sampled[0]] = float("nan")
    if bad in ("masked", "both") and sampled:
        logits[sampled[-1]] = float("-inf")
    return logits


def _cases(mode):
    for rows, vocab, bad, k in itertools.product(ROWS, VOCABS, BAD, range(SEEDS_PER_CASE)):
        yield rows, vocab, bad, SEED + 1000 * rows + 10 * vocab + 7 * k + BAD.index(bad)


# --------------------------------------------------------------------------
# (a) token identity
# --------------------------------------------------------------------------

def test_the_draw_is_multinomial_in_isolation():
    """500 seeded distributions: smooth, peaked, sparse (exact zeros), one-hot
    and V=1 rows, fp32 / bf16 / fp16. Tokens and the generator state left
    behind must both equal `torch.multinomial`'s."""
    shapes = [(1, 1), (1, 7), (5, 777), (33, 64), (8, 5000)]
    dtypes = [torch.float32, torch.bfloat16, torch.float16]
    kinds = ["smooth", "peaked", "sparse", "onehot"]
    checked = 0
    for s in range(500):
        rows, vocab = shapes[s % len(shapes)]
        dtype = dtypes[s % len(dtypes)]
        kind = kinds[(s // len(shapes)) % len(kinds)]
        g = torch.Generator().manual_seed(s)
        x = torch.randn(rows, vocab, generator=g)
        if kind == "peaked":
            x = x * 25.0
        p = torch.softmax(x, -1)
        if kind == "sparse" and vocab > 1:
            p = p.masked_fill(torch.rand(rows, vocab, generator=g) < 0.7, 0.0)
            p[:, 0] += 1e-3  # keep every row's sum > 0
        if kind == "onehot":
            p = torch.zeros(rows, vocab)
            p[torch.arange(rows), torch.randint(vocab, (rows,), generator=g)] = 1.0
        p = p.to(dtype)

        torch.manual_seed(s)
        expected = torch.multinomial(p, 1)
        expected_state = torch.get_rng_state()
        torch.manual_seed(s)
        got = _draw_one_per_row(p)
        got_state = torch.get_rng_state()

        assert got.dtype == torch.long and got.shape == expected.shape == (rows, 1)
        assert torch.equal(got, expected), f"seed {s} {kind} {dtype} {rows}x{vocab}"
        assert torch.equal(got_state, expected_state), f"seed {s}: RNG state diverged"
        checked += 1
    assert checked == 500


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", MODES)
def test_sample_tokens_is_unchanged_by_the_switch(mode, dtype):
    """Through `sample_tokens`: switch ON == switch OFF (tokens, bad-row
    count, RNG state) for every case, with or without bad rows; and == the
    reference path whenever no row is bad (the reference path compacts good
    rows when one is bad, which was never a contract -- see sample_tokens)."""
    checked = 0
    for rows, vocab, bad, seed in _cases(mode):
        params = _params(mode, rows, seed)
        logits = _logits(rows, vocab, dtype, seed, bad, params[0])
        on, on_bad, on_state = _host_sample(logits, params, seed, sync_free=True)
        off, off_bad, off_state = _host_sample(logits, params, seed, sync_free=False)
        tag = f"{mode} {dtype} rows={rows} V={vocab} bad={bad} seed={seed}"
        assert torch.equal(on, off), f"{tag}: {on.view(-1).tolist()} != {off.view(-1).tolist()}"
        assert on_bad == off_bad, tag
        assert torch.equal(on_state, off_state), f"{tag}: RNG state diverged"

        n_sampled = len(_sampled_rows(params[0], dtype))
        expected_bad = 0
        if n_sampled:
            if bad == "both":
                expected_bad = 1 if n_sampled == 1 else 2
            elif bad != "none":
                expected_bad = 1
        assert on_bad == expected_bad, tag

        if bad == "none":
            temps, top_ps, top_ks = params
            torch.manual_seed(seed)
            reference = sample_tokens(logits, temperature=temps, top_p=top_ps, top_k=top_ks)
            assert torch.equal(on, reference), f"{tag}: differs from the reference path"
        checked += 1
    assert checked == len(ROWS) * len(VOCABS) * len(BAD) * SEEDS_PER_CASE


def test_the_case_matrix_is_at_least_500():
    per_mode = len(ROWS) * len(VOCABS) * len(BAD) * SEEDS_PER_CASE
    assert per_mode * len(MODES) * 2 >= 500  # two dtypes


def test_a_bad_row_falls_back_to_the_unscaled_argmax():
    temps, top_ps, top_ks = _params("m1", 4, SEED)
    logits = _logits(4, 48, torch.float32, SEED, "masked", temps)
    logits[1] = float("nan")
    out, bad, _ = _host_sample(logits, (temps, top_ps, top_ks), SEED, sync_free=True)
    assert bad == 2
    # NaN row: argmax of a NaN row is its first NaN (index 0); all -inf row:
    # every entry ties, first index (0). Both are the reference fallback.
    assert int(out[1]) == int(logits[1].argmax()) and int(out[3]) == int(logits[3].argmax())


# --------------------------------------------------------------------------
# (b) no syncing op on the switch-ON path
# --------------------------------------------------------------------------

def _trace(sync_free, bad="nan"):
    temps, top_ps, top_ks = _params("m1", 6, SEED)
    logits = _logits(6, 48, torch.float32, SEED, bad, temps)
    decisions = _decisions(temps, top_ps, top_ks, logits.dtype)
    counter = torch.zeros((), dtype=torch.int64)
    previous = _set_switch(sync_free)
    trace = _OpTrace()
    try:
        torch.manual_seed(SEED)
        with trace:
            sample_tokens(
                logits, temperature=temps, top_p=top_ps, top_k=top_ks,
                host_decisions=decisions, bad_row_counter=counter,
            )
    finally:
        _set_switch(previous)
    return trace


@pytest.mark.parametrize("bad", ["none", "nan", "both"])
def test_the_sync_free_draw_has_no_syncing_op(bad):
    trace = _trace(sync_free=True, bad=bad)
    assert trace.syncing() == [], trace.syncing()
    assert "aten.multinomial.default" not in trace.ops
    for op in ("aten.exponential_.default", "aten.div.out", "aten.argmax.default",
               "aten.aminmax.default"):
        assert op in trace.ops, (op, trace.ops)


def test_the_detector_sees_multinomial_on_the_switch_off_path():
    """Negative control: OFF is the old call, which syncs internally."""
    trace = _trace(sync_free=False)
    assert trace.syncing() == ["aten.multinomial.default"]
    assert "aten.exponential_.default" not in trace.ops
    assert "aten.aminmax.default" not in trace.ops


# --------------------------------------------------------------------------
# (c) RNG stream position
# --------------------------------------------------------------------------

def test_the_generator_continues_identically_after_the_draw():
    """Two consecutive decode steps under one seed, then one more random op:
    every token and the follow-on draw must match the old path, i.e. the
    stream position after each step is the old one."""
    temps, top_ps, top_ks = _params("filtered", 16, SEED)
    logits_a = _logits(16, 1024, torch.float32, SEED, "none", temps)
    logits_b = _logits(16, 1024, torch.float32, SEED + 1, "nan", temps)
    decisions = _decisions(temps, top_ps, top_ks, logits_a.dtype)

    def run(sync_free):
        previous = _set_switch(sync_free)
        try:
            torch.manual_seed(SEED)
            outs = [
                sample_tokens(l, temperature=temps, top_p=top_ps, top_k=top_ks,
                              host_decisions=decisions)
                for l in (logits_a, logits_b)
            ]
            return outs, torch.rand(8)
        finally:
            _set_switch(previous)

    (on_a, on_b), on_next = run(True)
    (off_a, off_b), off_next = run(False)
    assert torch.equal(on_a, off_a) and torch.equal(on_b, off_b)
    assert torch.equal(on_next, off_next)


# --------------------------------------------------------------------------
# Validation is kept: rows multinomial rejects take the fallback
# --------------------------------------------------------------------------

def _row_rejected_by_multinomial(row):
    try:
        torch.multinomial(row.unsqueeze(0), 1)
    except RuntimeError:
        return True
    return False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_device_check_matches_multinomial_row_by_row(dtype):
    inf, nan = float("inf"), float("nan")
    rows = torch.tensor([
        [0.2, 0.3, 0.5, 0.0],     # valid
        [0.0, 0.0, 0.0, 1.0],     # valid, one-hot
        [1e-30, 1e-30, 0.0, 0.0],  # valid for multinomial (sum > 0)
        [0.2, nan, 0.5, 0.3],     # NaN
        [0.0, 0.0, 0.0, 0.0],     # zero sum
        [0.2, inf, 0.5, 0.3],     # +inf
        [0.2, -inf, 0.5, 0.3],    # -inf
        [0.6, -0.1, 0.5, 0.0],    # negative entry
        [nan, nan, nan, nan],     # all NaN
    ]).to(dtype)
    no_prior = torch.zeros(rows.shape[0], dtype=torch.bool)
    flagged = _invalid_draw_rows(rows, no_prior).tolist()
    expected = [_row_rejected_by_multinomial(r) for r in rows]
    assert flagged == expected
    assert expected == [False, False, False, True, True, True, True, True, True]
    # rows already known bad stay bad
    assert all(_invalid_draw_rows(rows, torch.ones_like(no_prior)).tolist())


def _inject_into_final_softmax(monkeypatch, poison):
    """Make the sampler's final softmax return a distribution multinomial
    rejects but the NaN / zero-sum bad-row test does not catch. Unreachable
    from a real softmax; this pins what happens if that ever changes."""
    real = F.softmax

    def softmax(x, dim=None, **kw):
        out = real(x, dim=dim, **kw)
        poison(out)
        return out

    monkeypatch.setattr(sampling.F, "softmax", softmax)


@pytest.mark.parametrize(
    "poison,row",
    [
        pytest.param(lambda p: p[1].__setitem__(0, float("inf")), 1, id="inf"),
        pytest.param(lambda p: p[2].__setitem__(3, -0.5), 2, id="negative"),
    ],
)
def test_a_row_multinomial_rejects_takes_the_fallback_not_garbage(monkeypatch, poison, row):
    temps, top_ps, top_ks = _params("m1", 4, SEED)
    logits = _logits(4, 48, torch.float32, SEED, "none", temps)
    clean, clean_bad, _ = _host_sample(logits, (temps, top_ps, top_ks), SEED, sync_free=True)
    assert clean_bad == 0
    _inject_into_final_softmax(monkeypatch, poison)

    on, bad, _ = _host_sample(logits, (temps, top_ps, top_ks), SEED, sync_free=True)
    assert bad == 1
    # The poisoned row takes the deterministic greedy fallback (argmax of the
    # unscaled logits) instead of whatever the unvalidated race returned ...
    assert int(on[row]) == int(logits[row].argmax())
    # ... and every other row is the same draw as without the poison: the
    # race consumes the RNG independently of the values.
    for r in range(4):
        if r != row:
            assert int(on[r]) == int(clean[r]), r

    # Switch OFF on the same input: the old call refuses it.
    with pytest.raises(RuntimeError, match="probability tensor contains"):
        _host_sample(logits, (temps, top_ps, top_ks), SEED, sync_free=False)


# --------------------------------------------------------------------------
# (d) switch integrity
# --------------------------------------------------------------------------

def test_the_switch_defaults_on():
    assert sampling.SAMPLED_DRAW_SYNC_FREE is True


def test_the_switch_is_read_at_call_time(monkeypatch):
    monkeypatch.setattr(sampling, "SAMPLED_DRAW_SYNC_FREE", False)
    assert "aten.multinomial.default" in _trace(sync_free=sampling.SAMPLED_DRAW_SYNC_FREE).ops
    monkeypatch.setattr(sampling, "SAMPLED_DRAW_SYNC_FREE", True)
    assert "aten.multinomial.default" not in _trace(sync_free=sampling.SAMPLED_DRAW_SYNC_FREE).ops


def test_the_reference_path_ignores_the_switch():
    """host_decisions=None stays the multinomial oracle with either setting."""
    temps, top_ps, top_ks = _params("m1", 4, SEED)
    logits = _logits(4, 48, torch.float32, SEED, "none", temps)
    for value in (True, False):
        previous = _set_switch(value)
        trace = _OpTrace()
        try:
            with trace:
                sample_tokens(logits, temperature=temps, top_p=top_ps, top_k=top_ks)
        finally:
            _set_switch(previous)
        assert "aten.multinomial.default" in trace.ops
        assert "aten.exponential_.default" not in trace.ops


# --------------------------------------------------------------------------
# CUDA node gate
# --------------------------------------------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_the_draw_is_multinomial(dtype):
    """Production geometry (GLM vocab) on the device generator: tokens and the
    Philox seed/offset state must equal `torch.multinomial`'s, over two
    consecutive draws (offset continuation)."""
    device = torch.device("cuda")
    for s in range(100):
        g = torch.Generator(device=device).manual_seed(s)
        rows = (1, 7, 64, 256)[s % 4]
        p = torch.softmax(
            torch.randn(rows, 151552, generator=g, device=device), -1
        ).to(dtype)
        torch.cuda.manual_seed(s)
        exp = [torch.multinomial(p, 1) for _ in range(2)]
        exp_state = torch.cuda.get_rng_state(device)
        torch.cuda.manual_seed(s)
        got = [_draw_one_per_row(p) for _ in range(2)]
        got_state = torch.cuda.get_rng_state(device)
        assert all(torch.equal(a, b) for a, b in zip(got, exp)), f"seed {s} rows={rows}"
        assert torch.equal(got_state, exp_state), f"seed {s}: Philox state diverged"


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", MODES)
def test_cuda_sample_tokens_is_unchanged_by_the_switch(mode, dtype):
    device = torch.device("cuda")
    for rows, vocab, bad, seed in _cases(mode):
        params = _params(mode, rows, seed)
        logits = _logits(rows, vocab, dtype, seed, bad, params[0]).to(device)
        on, on_bad, on_state = _host_sample(logits, params, seed, True, device)
        off, off_bad, off_state = _host_sample(logits, params, seed, False, device)
        tag = f"{mode} {dtype} rows={rows} V={vocab} bad={bad} seed={seed}"
        assert torch.equal(on, off), tag
        assert on_bad == off_bad, tag
        assert torch.equal(on_state, off_state), tag
