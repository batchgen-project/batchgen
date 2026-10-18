"""Device-resident per-request sampling parameters must behave like the rebuild.

The decode step path no longer rebuilds the three ``[B]`` sampling-parameter
tensors per step; it reads them from
``batchgen.worker.sampling_residency.DecodeSamplingResidency``. The rebuild
(``BatchGenWorker._build_sampling_tensors``) stays in the tree as the
reference, so "same values" is asserted directly rather than re-derived.

What is pinned here:

* **Reference equality** -- for the whole parameter matrix, the resident
  tensors equal the rebuilt tensors element-wise and in the same dtype, and
  ``_select_tokens`` returns bitwise-identical tokens with and without the
  residency under a fixed seed. That covers the two argmax fast paths as well
  as ``sample_tokens``.
* **Invalidation** -- every site that can change the ordered row set
  (membership add/remove, a fresh ``SequenceBatch``, the decode row-epoch bump
  used by the page boundary / async-load finalizer / rebalance remap, and a
  rebound ``batch`` list) makes the residency recompute; and the *negative*
  case, an in-place mutation of a ``sampling_params`` dict with no bump, is
  silent on the fast path and fail-loud under ``BATCHGEN_DECODE_ASSERT``.
* **Steady state** -- with nothing changed, repeated steps do no work at all:
  same plan object, same storage, rebuild count unchanged.
* **The bf16 trap** -- the host-side branch decisions are derived from the
  *cast* values, so they agree with the device predicates they replace even
  where rounding moves a value across a predicate boundary.
* **The aliasing invariant** -- ``sample_tokens`` must never write in place to
  the tensors it is handed, because they are now persistent buffers.

``batchgen.batchgen_worker`` cannot be imported on a CPU-only machine (triton,
UCX, CUDA extensions), so the worker methods under test are lifted out of the
real source file by ``tests/worker_method_source``; nothing is re-implemented.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from batchgen.sampling import sample_tokens  # noqa: E402
from batchgen.sequence import SequenceBatch, SequenceStatus  # noqa: E402
from batchgen.worker.sampling_residency import (  # noqa: E402
    DecodeSamplingResidency,
    SamplingResidencyRowError,
    SamplingResidencyShapeError,
    SamplingResidencyStaleError,
    batch_all_greedy,
    host_decisions_from_cast,
    resolve_sampling_values,
    sampling_params_present,
)
from worker_method_source import load_worker_methods  # noqa: E402

SEED = 20261018
V = 32


# --------------------------------------------------------------------------
# Worker methods under test, lifted from the real source
# --------------------------------------------------------------------------

def _worker_methods(decode_assert: bool = False):
    return load_worker_methods(
        [
            "_build_sampling_tensors",
            "_decode_batch_all_greedy",
            "_select_tokens",
            "_select_tokens_resident",
            "_bump_decode_row_epoch",
        ],
        {
            "torch": torch,
            "logging": logging,
            "Optional": Optional,
            "BATCHGEN_DECODE_ASSERT": decode_assert,
        },
    )


class _Params:
    """Stand-in for the model, for `next(self.model.parameters()).device`."""

    def parameters(self):
        yield torch.zeros(1)


class FakeWorker:
    """The attributes `_select_tokens*` and `_build_sampling_tensors` touch."""

    def __init__(self, methods, global_batch=None):
        self.model = _Params()
        self.rank = 0
        self.global_batch = global_batch if global_batch is not None else SequenceBatch()
        self._decode_row_epoch = 0
        self._methods = methods
        for name, fn in methods.items():
            setattr(self, name, fn.__get__(self, FakeWorker))


class FakeSequence:
    """Enough of `SequenceEntry` for `SequenceBatch` and the sampling path."""

    def __init__(self, uuid, global_idx, sampling_params):
        self.uuid = uuid
        self.global_idx = global_idx
        self.sampling_params = sampling_params
        self.status = SequenceStatus.IN_DECODE
        self.assigned_rank = 0


def _sequences(params_list, prefix="s"):
    return [
        FakeSequence(f"{prefix}-{i}", i, params)
        for i, params in enumerate(params_list)
    ]


# --------------------------------------------------------------------------
# The parameter matrix
# --------------------------------------------------------------------------

#: Every shape of `sampling_params` the admission path can produce, including
#: the falsy-but-present values that pin the `or`-fallback semantics: `.get(k,
#: default) or default` maps a present `0` / `0.0` / `None` to the default,
#: which `.get(k, default)` alone does not.
MATRIX = {
    "all_none": [None, None, None, None],
    "all_empty_dict": [{}, {}, {}, {}],
    "all_greedy_explicit": [{"temperature": 0.0}] * 4,
    "all_sampling": [{"temperature": 1.0, "top_p": 1.0, "top_k": 0}] * 4,
    "mixed_greedy_and_sampling": [
        {"temperature": 0.0},
        {"temperature": 1.0},
        {"temperature": 0.0, "top_k": 4},
        {"temperature": 0.8, "top_p": 0.9},
    ],
    "top_k_rows": [
        {"temperature": 1.0, "top_k": 1},
        {"temperature": 1.0, "top_k": 4},
        {"temperature": 1.0, "top_k": 0},
        {"temperature": 1.0, "top_k": 8},
    ],
    "top_p_rows": [
        {"temperature": 1.0, "top_p": 0.1},
        {"temperature": 1.0, "top_p": 0.5},
        {"temperature": 1.0, "top_p": 0.9},
        {"temperature": 1.0, "top_p": 1.0},
    ],
    "top_k_and_top_p": [
        {"temperature": 0.0, "top_p": 1.0, "top_k": 0},
        {"temperature": 1.0, "top_p": 0.6, "top_k": 5},
        {"temperature": 0.7, "top_p": 0.95, "top_k": 2},
        {"temperature": 1.3, "top_p": 0.3, "top_k": 0},
    ],
    "falsy_but_present": [
        {"temperature": 0, "top_p": 0, "top_k": 0},
        {"temperature": None, "top_p": None, "top_k": None},
        {"temperature": False, "top_p": False, "top_k": False},
        {"temperature": 1.0, "top_p": 0.0, "top_k": 0},
    ],
    "some_none_some_params": [
        None,
        {"temperature": 1.0, "top_p": 0.7},
        None,
        {"temperature": 0.0},
    ],
    "mixed_with_empty_dict": [
        {},
        {"temperature": 1.0},
        {},
        {"temperature": 0.5, "top_k": 3},
    ],
}

#: Row counts the residency has to handle: empty, single, exactly at capacity,
#: and past it (the grow path).
CAPACITY = 4
ROW_COUNTS = (0, 1, 2, CAPACITY, CAPACITY + 3)


def _logits(n, dtype=torch.float32, seed=4242):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(n, V, generator=generator, dtype=torch.float32).to(dtype)


def _resized(params_list, n):
    """`params_list` repeated/truncated to exactly `n` rows."""
    if n == 0:
        return []
    return [params_list[i % len(params_list)] for i in range(n)]


# --------------------------------------------------------------------------
# T1 -- reference equality
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(MATRIX))
def test_resident_tensors_equal_the_rebuilt_tensors(case):
    methods = _worker_methods()
    worker = FakeWorker(methods)
    seqs = _sequences(MATRIX[case])
    rows = list(range(len(seqs)))

    reference = worker._build_sampling_tensors(seqs)
    residency = DecodeSamplingResidency(CAPACITY, "cpu")
    plan = residency.plan(
        seqs,
        logits=_logits(len(seqs)),
        membership_epoch=1,
        row_epoch=0,
        batch_rows=rows,
    )

    if reference[0] is None:
        # The rebuild declines to build when no row carries params. The plan
        # must reach the same branch, by `has_params`.
        assert not plan.has_params
        assert plan.temps is None
        return

    assert plan.has_params
    if plan.all_greedy:
        # No upload happens on the all-greedy fast path, so compare the host
        # values the plan was derived from instead of device tensors.
        temps, top_ps, top_ks = resolve_sampling_values(seqs)
        assert torch.equal(reference[0], torch.tensor(temps, dtype=torch.float32))
        assert torch.equal(reference[1], torch.tensor(top_ps, dtype=torch.float32))
        assert torch.equal(reference[2], torch.tensor(top_ks, dtype=torch.int64))
        assert worker._decode_batch_all_greedy(seqs)
        return

    for name, ref, got in (
        ("temperature", reference[0], plan.temps),
        ("top_p", reference[1], plan.top_ps),
        ("top_k", reference[2], plan.top_ks),
    ):
        assert got is not None, name
        assert got.dtype == ref.dtype, name
        assert got.shape == ref.shape, name
        assert torch.equal(got, ref), f"{name}: {got.tolist()} != {ref.tolist()}"


@pytest.mark.parametrize("n", ROW_COUNTS)
@pytest.mark.parametrize("case", sorted(MATRIX))
def test_select_tokens_is_bitwise_identical_with_and_without_residency(case, n):
    methods = _worker_methods()
    worker = FakeWorker(methods)
    seqs = _sequences(_resized(MATRIX[case], n))
    rows = list(range(n))
    logits = _logits(n)
    residency = DecodeSamplingResidency(CAPACITY, "cpu")

    torch.manual_seed(SEED)
    expected = worker._select_tokens(logits, seqs)
    torch.manual_seed(SEED)
    got = worker._select_tokens(logits, seqs, residency=residency, batch_rows=rows)

    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    assert torch.equal(got, expected), (
        f"{case} n={n}: residency {got.reshape(-1).tolist()} != "
        f"rebuild {expected.reshape(-1).tolist()}"
    )


def test_the_grow_path_fires_only_on_a_row_change_and_stays_correct():
    methods = _worker_methods()
    worker = FakeWorker(methods)
    residency = DecodeSamplingResidency(2, "cpu")
    assert residency.capacity == 2

    for n in (2, 5, 9, 3):
        seqs = _sequences(_resized(MATRIX["top_k_and_top_p"], n), prefix=f"n{n}")
        rows = list(range(n))
        logits = _logits(n)
        torch.manual_seed(SEED)
        expected = worker._select_tokens(logits, seqs)
        torch.manual_seed(SEED)
        got = worker._select_tokens(logits, seqs, residency=residency, batch_rows=rows)
        assert torch.equal(got, expected), n
    # Grew for 5 and for 9, never shrank, and never grew without a rebuild.
    assert residency.capacity == 9
    assert residency.grow_count == 2
    assert residency.grow_count <= residency.rebuild_count


# --------------------------------------------------------------------------
# T2 -- invalidation at every site that can change the ordered row set
# --------------------------------------------------------------------------

def _plan_for(residency, worker, seqs, rows, logits=None, verify=False):
    return residency.plan(
        seqs,
        logits=logits if logits is not None else _logits(len(seqs)),
        membership_epoch=worker.global_batch.composition_epoch,
        row_epoch=worker._decode_row_epoch,
        batch_rows=rows,
        verify=verify,
    )


def test_membership_changes_bump_the_composition_epoch():
    batch = SequenceBatch()
    seen = {batch.composition_epoch}

    a = FakeSequence("a", 0, {"temperature": 1.0})
    batch.add_sequence(a)
    assert batch.composition_epoch not in seen
    seen.add(batch.composition_epoch)

    batch.remove_sequence("a")
    assert batch.composition_epoch not in seen
    seen.add(batch.composition_epoch)

    # A removal that found nothing still stamps: the invariant is "this method
    # may have changed membership", not "it did".
    batch.remove_sequence("absent")
    assert batch.composition_epoch not in seen
    seen.add(batch.composition_epoch)

    # A fresh batch must never collide with the epoch of the batch it replaces.
    assert SequenceBatch().composition_epoch not in seen
    # Views are stamped too, so an epoch always describes its own contents.
    assert batch.create_view([]).composition_epoch not in seen


def test_the_row_epoch_bump_is_monotone():
    worker = FakeWorker(_worker_methods())
    first = worker._bump_decode_row_epoch()
    second = worker._bump_decode_row_epoch()
    assert second > first > 0
    assert worker._decode_row_epoch == second


@pytest.mark.parametrize(
    "site",
    [
        "add_sequence",
        "remove_sequence",
        "new_global_batch",
        "row_epoch_bump",
        "rebound_batch_list",
    ],
)
def test_every_row_set_change_site_recomputes_the_residency(site):
    methods = _worker_methods()
    worker = FakeWorker(methods)
    params = [
        {"temperature": 1.0, "top_p": 0.9},
        {"temperature": 0.5, "top_k": 3},
    ]
    seqs = _sequences(params)
    for seq in seqs:
        worker.global_batch.add_sequence(seq)
    rows = [0, 1]
    residency = DecodeSamplingResidency(CAPACITY, "cpu")

    first = _plan_for(residency, worker, seqs, rows)
    assert residency.rebuild_count == 1
    # Same inputs, no change: no work.
    assert _plan_for(residency, worker, seqs, rows) is first
    assert residency.rebuild_count == 1

    if site == "add_sequence":
        extra = FakeSequence("extra", 7, {"temperature": 1.5, "top_k": 9})
        worker.global_batch.add_sequence(extra)
        seqs = seqs + [extra]
        rows = [0, 1, 2]
    elif site == "remove_sequence":
        worker.global_batch.remove_sequence("s-1")
        seqs = seqs[:1]
        rows = [0]
    elif site == "new_global_batch":
        worker.global_batch = SequenceBatch()
        for seq in seqs:
            worker.global_batch.add_sequence(seq)
    elif site == "row_epoch_bump":
        # Stands in for the page-boundary rebind, the async-load finalizer and
        # the rebalance remap: all three reach the residency only through this.
        worker._bump_decode_row_epoch()
    elif site == "rebound_batch_list":
        # Numerically identical, but a different list object -- which is what a
        # rebind produces. Caught even with both epochs unchanged.
        rows = list(rows)
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(site)

    second = _plan_for(residency, worker, seqs, rows)
    assert residency.rebuild_count == 2, site
    temps, top_ps, top_ks = resolve_sampling_values(seqs)
    if second.has_params and not second.all_greedy:
        assert second.temps.tolist() == pytest.approx(temps)
        assert second.top_ps.tolist() == pytest.approx(top_ps)
        assert second.top_ks.tolist() == top_ks
    assert second.has_params == sampling_params_present(seqs)
    assert second.all_greedy == (second.has_params and batch_all_greedy(temps))


def test_a_reordered_batch_at_the_same_length_is_recomputed():
    methods = _worker_methods()
    worker = FakeWorker(methods)
    seqs = _sequences(
        [{"temperature": 0.25}, {"temperature": 1.75}, {"temperature": 1.0}]
    )
    rows = [0, 1, 2]
    residency = DecodeSamplingResidency(CAPACITY, "cpu")
    first = _plan_for(residency, worker, seqs, rows)
    assert first.temps.tolist() == pytest.approx([0.25, 1.75, 1.0])

    # Right multiset, wrong rows: the mis-ordering failure mode. A remap is
    # stamped by the row epoch; the rebound list also changes identity.
    reordered = [seqs[2], seqs[0], seqs[1]]
    worker._bump_decode_row_epoch()
    second = _plan_for(residency, worker, reordered, [2, 0, 1])
    assert second.temps.tolist() == pytest.approx([1.0, 0.25, 1.75])


def test_an_unbumped_in_place_param_mutation_is_silent_off_and_loud_on():
    methods = _worker_methods()
    worker = FakeWorker(methods)
    params = {"temperature": 1.0, "top_p": 0.9, "top_k": 0}
    seqs = _sequences([params, {"temperature": 0.5}])
    rows = [0, 1]
    residency = DecodeSamplingResidency(CAPACITY, "cpu")
    first = _plan_for(residency, worker, seqs, rows)
    assert first.temps.tolist() == pytest.approx([1.0, 0.5])

    # `sampling_params` is immutable per sequence today -- two writes, both at
    # object birth -- so this cannot happen through any current path. The point
    # is to document the exact protection boundary rather than imply a stronger
    # one: with no epoch bump, the fast path keeps serving the old values.
    params["temperature"] = 0.2

    assert _plan_for(residency, worker, seqs, rows) is first
    assert first.temps.tolist() == pytest.approx([1.0, 0.5])

    with pytest.raises(SamplingResidencyStaleError) as excinfo:
        _plan_for(residency, worker, seqs, rows, verify=True)
    message = str(excinfo.value)
    assert "temperature" in message
    assert "row 0" in message
    assert "s-0" in message


def test_verification_also_catches_params_appearing_on_a_no_params_batch():
    worker = FakeWorker(_worker_methods())
    seqs = _sequences([None, None])
    rows = [0, 1]
    residency = DecodeSamplingResidency(CAPACITY, "cpu")
    plan = _plan_for(residency, worker, seqs, rows)
    assert not plan.has_params

    seqs[1].sampling_params = {"temperature": 1.0}
    with pytest.raises(SamplingResidencyStaleError):
        _plan_for(residency, worker, seqs, rows, verify=True)


def test_decode_assert_drives_verification_from_the_worker():
    """`_select_tokens_resident` must pass the flag through, not ignore it."""
    worker_off = FakeWorker(_worker_methods(decode_assert=False))
    worker_on = FakeWorker(_worker_methods(decode_assert=True))
    params = {"temperature": 1.0, "top_p": 0.9}
    seqs = _sequences([params, {"temperature": 0.5}])
    rows = [0, 1]
    logits = _logits(2)

    for worker in (worker_off, worker_on):
        worker.global_batch = SequenceBatch()
        for seq in seqs:
            worker.global_batch.add_sequence(seq)

    residency_off = DecodeSamplingResidency(CAPACITY, "cpu")
    residency_on = DecodeSamplingResidency(CAPACITY, "cpu")
    torch.manual_seed(SEED)
    worker_off._select_tokens(logits, seqs, residency=residency_off, batch_rows=rows)
    torch.manual_seed(SEED)
    worker_on._select_tokens(logits, seqs, residency=residency_on, batch_rows=rows)

    params["temperature"] = 0.2
    torch.manual_seed(SEED)
    worker_off._select_tokens(logits, seqs, residency=residency_off, batch_rows=rows)
    with pytest.raises(SamplingResidencyStaleError):
        worker_on._select_tokens(
            logits, seqs, residency=residency_on, batch_rows=rows
        )


# --------------------------------------------------------------------------
# Fail-loud checks that are always on
# --------------------------------------------------------------------------

def test_a_row_count_disagreement_is_named_not_guessed():
    worker = FakeWorker(_worker_methods())
    seqs = _sequences([{"temperature": 1.0}] * 3)
    residency = DecodeSamplingResidency(CAPACITY, "cpu")

    with pytest.raises(SamplingResidencyShapeError) as excinfo:
        _plan_for(residency, worker, seqs, [0, 1], logits=_logits(3))
    assert "batch_rows=2" in str(excinfo.value)

    with pytest.raises(SamplingResidencyShapeError) as excinfo:
        _plan_for(residency, worker, seqs, [0, 1, 2], logits=_logits(2))
    assert "logits_rows=2" in str(excinfo.value)


def test_an_unresolved_row_is_an_error_not_a_silent_greedy_row():
    worker = FakeWorker(_worker_methods())
    seqs = _sequences([{"temperature": 1.0}, None, {"temperature": 0.9}])
    seqs[1] = None
    residency = DecodeSamplingResidency(CAPACITY, "cpu")

    with pytest.raises(SamplingResidencyRowError) as excinfo:
        _plan_for(residency, worker, seqs, [4, 9, 11])
    message = str(excinfo.value)
    assert "row 1" in message
    assert "local_idx=9" in message

    # The pre-residency behaviour this replaces: `(None or {})` resolved to
    # temperature 0, so the row silently decoded greedily.
    assert resolve_sampling_values([None])[0] == [0.0]


def test_a_shape_mismatch_in_host_decisions_is_rejected_by_sample_tokens():
    decisions = host_decisions_from_cast([1.0, 1.0], [1.0, 1.0], [0, 0])
    with pytest.raises(SamplingResidencyShapeError):
        sample_tokens(
            _logits(3),
            temperature=torch.ones(3),
            top_p=torch.ones(3),
            top_k=torch.zeros(3, dtype=torch.int64),
            host_decisions=decisions,
        )


def test_residency_requires_the_batch_rows_it_keys_on():
    worker = FakeWorker(_worker_methods())
    residency = DecodeSamplingResidency(CAPACITY, "cpu")
    with pytest.raises(ValueError, match="batch_rows"):
        worker._select_tokens(_logits(1), _sequences([{"temperature": 1.0}]),
                              residency=residency)


# --------------------------------------------------------------------------
# Steady state: no work, no allocation, no copy
# --------------------------------------------------------------------------

def test_the_steady_state_step_does_no_work_at_all():
    worker = FakeWorker(_worker_methods())
    seqs = _sequences(MATRIX["top_k_and_top_p"])
    rows = list(range(len(seqs)))
    residency = DecodeSamplingResidency(CAPACITY, "cpu")

    first = _plan_for(residency, worker, seqs, rows)
    pointers = (
        first.temps.data_ptr(),
        first.top_ps.data_ptr(),
        first.top_ks.data_ptr(),
    )
    for _ in range(64):
        plan = _plan_for(residency, worker, seqs, rows)
        # Identical plan object => no list comprehension, no tensor built, no
        # host->device copy, no new decisions object. Only the key compares ran.
        assert plan is first
    assert residency.rebuild_count == 1
    assert residency.grow_count == 0
    assert (
        first.temps.data_ptr(),
        first.top_ps.data_ptr(),
        first.top_ks.data_ptr(),
    ) == pointers


# --------------------------------------------------------------------------
# T5 -- the two greedy fast paths stay reachable and stay distinct
# --------------------------------------------------------------------------

def _worker_with_exploding_sample_tokens(monkeypatch):
    """A worker whose `sample_tokens` raises, to prove it is not entered.

    Patched through the imported module object rather than the dotted string
    form: the string form walks `batchgen` -> `sampling` as attributes, which
    other test modules in a full-suite run can leave unset, making this test's
    outcome depend on collection order.
    """
    import batchgen.sampling as sampling_module

    def explode(*args, **kwargs):
        raise AssertionError("sample_tokens must not be entered on a greedy batch")

    monkeypatch.setattr(sampling_module, "sample_tokens", explode)
    return FakeWorker(_worker_methods())


def test_an_all_greedy_batch_skips_sample_tokens_and_builds_nothing(monkeypatch):
    worker = _worker_with_exploding_sample_tokens(monkeypatch)
    seqs = _sequences([{"temperature": 0.0}, {"temperature": 0}, {}])
    rows = [0, 1, 2]
    logits = _logits(3)
    residency = DecodeSamplingResidency(CAPACITY, "cpu")

    # Guard against a vacuous test: the patch must actually be reachable, or
    # "sample_tokens was not entered" proves nothing.
    with pytest.raises(AssertionError, match="must not be entered"):
        worker._select_tokens(
            logits,
            _sequences([{"temperature": 1.0}] * 3),
            residency=DecodeSamplingResidency(CAPACITY, "cpu"),
            batch_rows=rows,
        )

    got = worker._select_tokens(logits, seqs, residency=residency, batch_rows=rows)
    assert torch.equal(got, logits.argmax(dim=-1, keepdim=True))

    plan = _plan_for(residency, worker, seqs, rows, logits=logits)
    assert plan.has_params and plan.all_greedy
    # No upload on this path: the tensors are deliberately not populated.
    assert plan.temps is None and plan.top_ps is None and plan.top_ks is None
    assert plan.decisions is None


def test_a_no_params_batch_takes_the_other_argmax_branch(monkeypatch):
    worker = _worker_with_exploding_sample_tokens(monkeypatch)
    seqs = _sequences([None, None])
    rows = [0, 1]
    logits = _logits(2)
    residency = DecodeSamplingResidency(CAPACITY, "cpu")

    got = worker._select_tokens(logits, seqs, residency=residency, batch_rows=rows)
    assert torch.equal(got, torch.argmax(logits, dim=-1, keepdim=True))
    plan = _plan_for(residency, worker, seqs, rows, logits=logits)
    assert not plan.has_params


def test_an_empty_dict_is_params_and_an_absent_dict_is_not():
    """The two argmax branches are selected by exactly this distinction.

    The server sends `{}` for a request with no sampling fields, so getting
    this wrong would flip every default request between the two paths.
    """
    assert sampling_params_present(_sequences([{}])) is True
    assert sampling_params_present(_sequences([None])) is False
    assert sampling_params_present(_sequences([None, {}])) is True


def test_an_empty_batch_plans_like_the_rebuild():
    worker = FakeWorker(_worker_methods())
    residency = DecodeSamplingResidency(CAPACITY, "cpu")
    plan = _plan_for(residency, worker, [], [], logits=_logits(0))
    assert plan.n == 0
    assert not plan.has_params
    # The rebuild returns `(None, None, None)` for an empty list, which is what
    # keeps the `if temps is not None` gate behaving identically.
    assert worker._build_sampling_tensors([]) == (None, None, None)


# --------------------------------------------------------------------------
# The bf16 trap: host decisions must agree with the device predicates
# --------------------------------------------------------------------------

#: Values chosen so the cast moves them across a predicate boundary.
#: bf16 has 8 mantissa bits, so 0.9999 rounds to 1.0 and a float32 value below
#: the smallest bf16 subnormal rounds to 0.
CAST_TRAPS = [
    ([1.0, 1.0], [0.9999, 1.0], [0, 0]),
    ([1e-45, 1.0], [1.0, 1.0], [0, 0]),
    ([0.0, 1.0], [1.0, 0.999], [3, 0]),
    ([2.0, 3.0], [1.0, 1.0], [5, 2]),
    ([0.0, 0.0], [0.5, 0.5], [1, 1]),
    ([1.0, 1e-45, 0.0, 2.0], [0.9999, 0.5, 1.0, 1.0], [0, 7, 2, 4]),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("values", CAST_TRAPS)
def test_host_decisions_match_the_device_predicates_on_cast_values(values, dtype):
    temps_f32, top_ps_f32, top_ks = values
    host_temps = torch.tensor(temps_f32, dtype=torch.float32)
    host_top_ps = torch.tensor(top_ps_f32, dtype=torch.float32)

    cast_temps = host_temps.to(dtype)
    cast_top_ps = host_top_ps.to(dtype)
    decisions = host_decisions_from_cast(
        cast_temps.tolist(), cast_top_ps.tolist(), top_ks
    )

    # Exactly the four predicates `sample_tokens` would otherwise reduce.
    greedy_mask = cast_temps <= 0
    sampling_mask = ~greedy_mask
    assert decisions.any_greedy == bool(greedy_mask.any())
    assert decisions.any_sampling == bool(sampling_mask.any())
    if decisions.any_sampling:
        s_top_ks = torch.tensor(top_ks, dtype=torch.int64)[sampling_mask]
        assert decisions.max_top_k == int(s_top_ks.max().item())
        assert decisions.needs_top_p == bool((cast_top_ps[sampling_mask] < 1.0).any())
    else:
        assert decisions.max_top_k == 0
        assert decisions.needs_top_p is False


def test_the_resident_tensors_stay_fp32_so_the_cast_still_happens_at_use():
    """Storing the tensors pre-cast would change `greedy_mask` and the top_p
    comparisons from "fp32 value, rounded once at use" to "bf16 value stored".
    """
    worker = FakeWorker(_worker_methods())
    seqs = _sequences([{"temperature": 0.7, "top_p": 0.9999, "top_k": 3}])
    residency = DecodeSamplingResidency(CAPACITY, "cpu")
    plan = _plan_for(residency, worker, seqs, [0], logits=_logits(1, torch.bfloat16))
    assert plan.temps.dtype == torch.float32
    assert plan.top_ps.dtype == torch.float32
    assert plan.top_ks.dtype == torch.int64
    assert plan.temps.tolist() == pytest.approx([0.7])
    # ... while the decision it publishes is taken on the cast value, which for
    # bf16 rounds 0.9999 up to 1.0 and therefore disables top_p for that row.
    assert plan.decisions.needs_top_p is False
    assert float(torch.tensor([0.9999]).to(torch.bfloat16)[0]) == 1.0


def test_a_dtype_change_rebuilds_the_decisions():
    worker = FakeWorker(_worker_methods())
    seqs = _sequences([{"temperature": 1.0, "top_p": 0.9999}])
    rows = [0]
    residency = DecodeSamplingResidency(CAPACITY, "cpu")
    fp32 = _plan_for(residency, worker, seqs, rows, logits=_logits(1, torch.float32))
    assert fp32.decisions.needs_top_p is True
    bf16 = _plan_for(residency, worker, seqs, rows, logits=_logits(1, torch.bfloat16))
    assert bf16.decisions.needs_top_p is False
    assert residency.rebuild_count == 2


# --------------------------------------------------------------------------
# T4 -- the aliasing invariant on the now-persistent buffers
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case", sorted(MATRIX))
def test_sample_tokens_never_writes_to_the_tensors_it_is_handed(case):
    """Handing out a persistent buffer makes this load-bearing: an in-place
    write would corrupt every later step of the interval, not just this one.
    """
    worker = FakeWorker(_worker_methods())
    seqs = _sequences(MATRIX[case])
    reference = worker._build_sampling_tensors(seqs)
    if reference[0] is None or worker._decode_batch_all_greedy(seqs):
        pytest.skip("no tensors are handed to sample_tokens in this case")

    temps, top_ps, top_ks = reference
    before = (temps.clone(), top_ps.clone(), top_ks.clone())
    logits = _logits(len(seqs))
    temps_f, top_ps_f, top_ks_f = resolve_sampling_values(seqs)
    decisions = host_decisions_from_cast(
        temps.to(logits.dtype).tolist(),
        top_ps.to(logits.dtype).tolist(),
        top_ks_f,
    )

    for kwargs in ({}, {"host_decisions": decisions}):
        torch.manual_seed(SEED)
        sample_tokens(
            logits, temperature=temps, top_p=top_ps, top_k=top_ks, **kwargs
        )
        assert torch.equal(temps, before[0]), f"{case}: temperature was mutated"
        assert torch.equal(top_ps, before[1]), f"{case}: top_p was mutated"
        assert torch.equal(top_ks, before[2]), f"{case}: top_k was mutated"


# --------------------------------------------------------------------------
# The invalid-probability fallback, which lost its host branch
# --------------------------------------------------------------------------

def _all_bad_logits(n):
    """Logits whose post-filter probabilities are invalid for every row."""
    return torch.full((n, V), float("nan"), dtype=torch.float32)


def test_bad_rows_fall_back_to_the_unscaled_argmax_without_a_readback():
    logits = torch.randn(4, V, generator=torch.Generator().manual_seed(7))
    # Row 2 is poisoned, the rest are ordinary.
    logits[2] = float("nan")
    temps = torch.ones(4)
    top_ps = torch.ones(4)
    top_ks = torch.zeros(4, dtype=torch.int64)
    decisions = host_decisions_from_cast(
        temps.tolist(), top_ps.tolist(), top_ks.tolist()
    )
    counter = torch.zeros((), dtype=torch.int64)

    torch.manual_seed(SEED)
    got = sample_tokens(
        logits,
        temperature=temps,
        top_p=top_ps,
        top_k=top_ks,
        host_decisions=decisions,
        bad_row_counter=counter,
    )
    assert got.shape == (4, 1)
    # The fallback is the argmax over the UNSCALED logits, bit-identical to the
    # branched form's fallback. An all-NaN row's argmax is index 0.
    assert int(got[2, 0]) == int(logits[2].argmax())
    assert int(counter.item()) == 1


def test_the_bad_row_counter_is_drained_as_a_delta():
    residency = DecodeSamplingResidency(CAPACITY, "cpu")
    assert residency.drain_bad_rows() == 0
    residency.bad_row_counter.add_(3)
    assert residency.drain_bad_rows() == 3
    assert residency.drain_bad_rows() == 0
    residency.bad_row_counter.add_(2)
    assert residency.drain_bad_rows() == 2


# --------------------------------------------------------------------------
# The pinned-staging hazard: no rewrite (and no free) under an in-flight copy
# --------------------------------------------------------------------------

class FakeEvent:
    """Records the order of `record()` / `synchronize()` against a shared log."""

    def __init__(self, log):
        self._log = log
        self.recorded = 0
        self.synchronized = 0

    def record(self, stream=None):
        self.recorded += 1
        self._log.append("record")

    def synchronize(self):
        self.synchronized += 1
        self._log.append("synchronize")


def test_a_pinned_rewrite_waits_for_the_previous_copy():
    log = []
    worker = FakeWorker(_worker_methods())
    residency = DecodeSamplingResidency(
        CAPACITY, "cpu", event_factory=lambda: FakeEvent(log)
    )
    params = [{"temperature": 1.0, "top_p": 0.9}, {"temperature": 0.5}]
    seqs = _sequences(params)

    _plan_for(residency, worker, seqs, [0, 1])
    assert log == ["record"], log

    # Second composition change: the staging buffers are about to be rewritten,
    # so the previous copy must be known to have retired first.
    worker._bump_decode_row_epoch()
    _plan_for(residency, worker, seqs, [0, 1])
    assert log == ["record", "synchronize", "record"], log


def test_the_grow_path_waits_before_it_drops_the_old_pinned_buffers():
    """Growing frees the pinned tensors an in-flight copy may still be reading,
    so the wait has to happen before the reallocation, not after it.
    """
    log = []
    worker = FakeWorker(_worker_methods())
    residency = DecodeSamplingResidency(
        2, "cpu", event_factory=lambda: FakeEvent(log)
    )
    _plan_for(residency, worker, _sequences([{"temperature": 1.0}] * 2), [0, 1])
    assert log == ["record"], log

    grown = _sequences([{"temperature": 1.0}] * 5, prefix="g")
    _plan_for(residency, worker, grown, [0, 1, 2, 3, 4])
    assert residency.grow_count == 1
    assert log == ["record", "synchronize", "record"], log


def test_a_clean_batch_is_bitwise_identical_in_both_fallback_formulations():
    """With no bad row, the device-selected form must match the branched form.

    This is the common case, so it is the one that must not move: the filled
    tensor equals `probs` element-for-element and multinomial consumes one
    uniform per row regardless of values.
    """
    logits = torch.randn(6, V, generator=torch.Generator().manual_seed(11))
    temps = torch.tensor([1.0, 0.0, 0.8, 1.2, 0.0, 1.0])
    top_ps = torch.tensor([1.0, 1.0, 0.9, 1.0, 1.0, 0.5])
    top_ks = torch.tensor([0, 0, 4, 0, 0, 3], dtype=torch.int64)
    decisions = host_decisions_from_cast(
        temps.tolist(), top_ps.tolist(), top_ks.tolist()
    )

    torch.manual_seed(SEED)
    branched = sample_tokens(logits, temperature=temps, top_p=top_ps, top_k=top_ks)
    torch.manual_seed(SEED)
    selected = sample_tokens(
        logits,
        temperature=temps,
        top_p=top_ps,
        top_k=top_ks,
        host_decisions=decisions,
    )
    assert torch.equal(branched, selected)
