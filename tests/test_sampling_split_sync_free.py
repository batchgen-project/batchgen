"""The host-decided greedy/sampling split in `sample_tokens` must not sync.

On the decode step path `sample_tokens` is handed `SamplingHostDecisions`, so
it already knows which rows are greedy and which are sampled. Splitting the
batch with a device boolean mask anyway (`logits[sampling_mask]`,
`result[sampling_mask] = ...`) is an implicit `nonzero`, i.e. a
cudaStreamSynchronize: the host stands still until the whole-model graph
replay retires. The split is therefore taken from the host decision:

* a uniform batch (all greedy, or all sampled) is not split at all;
* a mixed batch is split with int64 row-index tensors built once per row-set
  change -- by `DecodeSamplingResidency` through pinned staging -- and gathered
  / scattered with `index_select` / `index_copy_`.

What is pinned here:

* **Token identity** -- for all-greedy, all-sampled and mixed compositions, in
  fp32 and bf16, the host-decided split returns exactly the tokens of the
  reference (device-mask) path under the same seed.
* **No sync op on the split** -- an aten-level dispatch trace of the
  host-decided path contains no `_local_scalar_dense` / `item` /
  `is_nonzero` / `nonzero` / `masked_select`, and no gather/scatter with a
  boolean index. The same trace of the reference path is the negative control
  that proves the detector sees those ops. Syncs that live inside the sampled
  path proper (`torch.multinomial`'s validity check, the boolean index_put_ in
  the top-k / top-p filters) are out of scope here and are enumerated
  explicitly, so a new one cannot slip in unseen.
* **Rebuild correctness** -- across composition changes the residency
  republishes the row partition and its device index tensors, keeps them in
  its resident buffers (no per-step allocation), and the tokens stay equal to
  the reference at every step.
* **Fail-loud** -- a mixed batch without valid row-index tensors is refused
  with `SamplingRowIndexError`; it never falls back to the mask.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from batchgen.sampling import sample_tokens
from batchgen.worker.sampling_residency import (
    DecodeSamplingResidency,
    SamplingHostDecisions,
    SamplingRowIndexError,
    attach_row_index,
    host_decisions_from_cast,
)

SEED = 20261024
V = 48


def _logits(n, dtype=torch.float32, seed=99):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, V, generator=g, dtype=torch.float32).to(dtype)


#: name -> (temperature, top_p, top_k) per row. "m1" is the decode workload
#: shape the sync was measured on: every row sampled, top_p=1, top_k=0.
COMPOSITIONS = {
    "all_greedy": ([0.0] * 5, [1.0] * 5, [0] * 5),
    "m1_all_sampled": ([1.0] * 6, [1.0] * 6, [0] * 6),
    "all_sampled_per_row": (
        [0.5, 1.0, 1.5, 0.7],
        [1.0, 0.9, 1.0, 0.5],
        [0, 4, 0, 7],
    ),
    "mixed": ([0.0, 1.0, 0.0, 0.8, 1.2], [1.0] * 5, [0] * 5),
    "mixed_filters": (
        [1.0, 0.0, 0.7, 0.0, 1.3, 0.9],
        [0.6, 1.0, 0.95, 1.0, 0.3, 1.0],
        [5, 0, 2, 0, 0, 3],
    ),
    "single_greedy_row_in_sampled_batch": ([1.0, 1.0, 0.0, 1.0], [1.0] * 4, [0] * 4),
    "single_sampled_row_in_greedy_batch": ([0.0, 0.0, 0.0, 0.6], [1.0] * 4, [0] * 4),
}


def _tensors(name):
    temps, top_ps, top_ks = COMPOSITIONS[name]
    return (
        torch.tensor(temps, dtype=torch.float32),
        torch.tensor(top_ps, dtype=torch.float32),
        torch.tensor(top_ks, dtype=torch.int64),
    )


def _decisions(temps, top_ps, top_ks, dtype, device="cpu"):
    """Decisions taken on the cast values, exactly as the residency does."""
    return attach_row_index(
        host_decisions_from_cast(
            temps.to(dtype).tolist(), top_ps.to(dtype).tolist(), top_ks.tolist()
        ),
        device,
    )


# --------------------------------------------------------------------------
# (a) token identity against the reference (device-mask) split
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name", sorted(COMPOSITIONS))
def test_the_host_split_returns_the_reference_tokens(name, dtype):
    temps, top_ps, top_ks = _tensors(name)
    logits = _logits(temps.numel(), dtype)
    decisions = _decisions(temps, top_ps, top_ks, dtype)
    for trial in range(8):
        torch.manual_seed(SEED + trial)
        expected = sample_tokens(logits, temperature=temps, top_p=top_ps, top_k=top_ks)
        torch.manual_seed(SEED + trial)
        got = sample_tokens(
            logits,
            temperature=temps,
            top_p=top_ps,
            top_k=top_ks,
            host_decisions=decisions,
        )
        assert got.dtype == expected.dtype == torch.long
        assert got.shape == expected.shape == (temps.numel(), 1)
        assert torch.equal(got, expected), (
            f"{name} {dtype} trial {trial}: host split {got.view(-1).tolist()} "
            f"!= mask split {expected.view(-1).tolist()}"
        )


def test_a_cast_that_makes_every_row_greedy_is_the_reference_argmax():
    """bf16 rounds a tiny positive temperature to 0, so the cast decision is
    all-greedy while the fp32 values are not -- the residency still calls
    `sample_tokens`, and the uniform (unsplit) greedy branch must be the
    reference argmax.
    """
    temps = torch.tensor([1e-45, 1e-45, 1e-45])
    top_ps = torch.ones(3)
    top_ks = torch.zeros(3, dtype=torch.int64)
    logits = _logits(3, torch.bfloat16)
    decisions = _decisions(temps, top_ps, top_ks, torch.bfloat16)
    assert decisions.any_greedy and not decisions.any_sampling
    expected = sample_tokens(logits, temperature=temps, top_p=top_ps, top_k=top_ks)
    got = sample_tokens(
        logits, temperature=temps, top_p=top_ps, top_k=top_ks, host_decisions=decisions
    )
    assert torch.equal(got, expected)
    assert torch.equal(got, logits.float().argmax(dim=-1, keepdim=True))


def test_a_bad_row_in_a_mixed_batch_falls_back_inside_the_sampled_half():
    temps = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0])
    top_ps = torch.ones(5)
    top_ks = torch.zeros(5, dtype=torch.int64)
    logits = _logits(5)
    logits[3] = float("nan")
    decisions = _decisions(temps, top_ps, top_ks, logits.dtype)
    counter = torch.zeros((), dtype=torch.int64)
    torch.manual_seed(SEED)
    got = sample_tokens(
        logits,
        temperature=temps,
        top_p=top_ps,
        top_k=top_ks,
        host_decisions=decisions,
        bad_row_counter=counter,
    ).view(-1)
    argmax = logits.float().argmax(dim=-1)
    assert int(got[0]) == int(argmax[0]) and int(got[2]) == int(argmax[2])
    assert int(got[3]) == int(logits[3].argmax())
    assert int(counter) == 1


def test_the_aliasing_invariant_holds_on_the_unsplit_path():
    """With no split, the sampled half IS the caller's tensors (no gather), so
    nothing downstream may write to them in place."""
    temps, top_ps, top_ks = _tensors("all_sampled_per_row")
    logits = _logits(temps.numel())
    before = [t.clone() for t in (logits, temps, top_ps, top_ks)]
    decisions = _decisions(temps, top_ps, top_ks, logits.dtype)
    torch.manual_seed(SEED)
    sample_tokens(
        logits, temperature=temps, top_p=top_ps, top_k=top_ks, host_decisions=decisions
    )
    for name, now, then in zip(("logits", "temps", "top_ps", "top_ks"),
                               (logits, temps, top_ps, top_ks), before):
        assert torch.equal(now, then), f"{name} was written in place"


# --------------------------------------------------------------------------
# (b) the split performs no syncing op (aten dispatch trace)
# --------------------------------------------------------------------------

#: aten ops that read a device value back to the host (or, for nonzero /
#: masked_select / argwhere, size an output from device data) and therefore
#: block on CUDA.
_SYNC_OPS = (
    "aten._local_scalar_dense",
    "aten.item",
    "aten.is_nonzero",
    "aten.nonzero",
    "aten.nonzero_static",
    "aten.masked_select",
    "aten.argwhere",
)
_INDEX_OPS = ("aten.index.", "aten.index_put", "aten._index_put_impl_")


class _OpTrace(TorchDispatchMode):
    """Records every aten op, tagging gathers/scatters with a boolean index."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if name.startswith(_INDEX_OPS) and len(args) > 1:
            indices = args[1]
            if isinstance(indices, (list, tuple)) and any(
                isinstance(i, torch.Tensor) and i.dtype in (torch.bool, torch.uint8)
                for i in indices
            ):
                name += "[bool-index]"
        self.ops.append(name)
        return func(*args, **(kwargs or {}))

    def syncing(self):
        return [
            op for op in self.ops
            if op.startswith(_SYNC_OPS) or op.endswith("[bool-index]")
        ]


def _trace(name, host):
    temps, top_ps, top_ks = _tensors(name)
    logits = _logits(temps.numel())
    decisions = _decisions(temps, top_ps, top_ks, logits.dtype) if host else None
    trace = _OpTrace()
    torch.manual_seed(SEED)
    with trace:
        sample_tokens(
            logits,
            temperature=temps,
            top_p=top_ps,
            top_k=top_ks,
            host_decisions=decisions,
        )
    return trace


#: Syncs that live inside the sampled path proper and are deliberately out of
#: scope for the split: the boolean index_put_ in the top-k filter
#: (`effective_k[effective_k <= 0] = V`, `sorted_logits[topk_mask] = -inf`)
#: and in the top-p filter (`sorted_mask[no_filter...] = False`).
#: `torch.multinomial` also syncs internally on CUDA (its validity check), but
#: that happens below the dispatcher and is not visible in this trace.
_KNOWN_SAMPLED_PATH_SYNCS = {
    "all_greedy": [],
    "m1_all_sampled": [],
    "mixed": [],
    "single_greedy_row_in_sampled_batch": [],
    "single_sampled_row_in_greedy_batch": [],
    "all_sampled_per_row": ["aten.index_put_.default[bool-index]"] * 3,
    "mixed_filters": ["aten.index_put_.default[bool-index]"] * 3,
}


def test_every_composition_has_a_pinned_sync_inventory():
    assert sorted(_KNOWN_SAMPLED_PATH_SYNCS) == sorted(COMPOSITIONS)


@pytest.mark.parametrize("name", sorted(COMPOSITIONS))
def test_the_host_decided_path_has_no_split_sync(name):
    trace = _trace(name, host=True)
    assert trace.syncing() == _KNOWN_SAMPLED_PATH_SYNCS[name], (
        f"{name}: syncing ops on the host-decided path: {trace.syncing()}"
    )


def test_the_measured_workload_shape_has_no_sync_op_but_multinomial():
    """The decode workload the stall was measured on (all rows sampled,
    top_p=1, top_k=0) must reach `multinomial` with no syncing op before it
    and none after it, and must not split the batch at all."""
    trace = _trace("m1_all_sampled", host=True)
    assert trace.syncing() == []
    assert "aten.index_select.default" not in trace.ops
    assert "aten.index_copy_.default" not in trace.ops
    assert trace.ops.count("aten.multinomial.default") == 1


def test_a_mixed_batch_splits_by_index_not_by_mask():
    trace = _trace("mixed", host=True)
    assert trace.syncing() == []
    # greedy gather + 4 sampled-half gathers (logits, temps, top_ps, top_ks)
    assert trace.ops.count("aten.index_select.default") == 5
    # greedy write-back + sampled write-back
    assert trace.ops.count("aten.index_copy_.default") == 2


@pytest.mark.parametrize("name", ["m1_all_sampled", "mixed"])
def test_the_detector_sees_the_mask_split_on_the_reference_path(name):
    """Negative control: the reference path still splits by mask and reduces
    `.any()` on device, and the trace must see exactly that."""
    syncing = _trace(name, host=False).syncing()
    assert "aten.is_nonzero.default" in syncing
    assert "aten.index.Tensor[bool-index]" in syncing
    assert "aten.index_put_.default[bool-index]" in syncing


# --------------------------------------------------------------------------
# (c) composition changes rebuild the partition and its index tensors
# --------------------------------------------------------------------------

class _Seq:
    def __init__(self, uuid, params):
        self.uuid = uuid
        self.sampling_params = params


def _seqs(temps, tag):
    return [_Seq(f"{tag}-{i}", {"temperature": t}) for i, t in enumerate(temps)]


def _expected_rows(temps, dtype):
    cast = torch.tensor(temps, dtype=torch.float32).to(dtype).tolist()
    greedy = tuple(i for i, t in enumerate(cast) if t <= 0)
    sampled = tuple(i for i, t in enumerate(cast) if t > 0)
    return greedy, sampled


#: Successive row sets one decode round can see: same-length pattern change,
#: mixed -> uniform -> mixed, a grow past capacity, and a bf16 cast that flips
#: rows to greedy.
_SCHEDULE = [
    ([0.0, 1.0, 0.0, 1.0], torch.float32),
    ([1.0, 0.0, 1.0, 0.0], torch.float32),
    ([1.0, 1.0, 1.0, 1.0], torch.float32),
    ([0.0, 0.0, 0.7, 0.0], torch.float32),
    ([0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.5], torch.float32),
    ([1e-45, 1.0, 1e-45, 1.0], torch.bfloat16),
    ([0.0, 0.0, 0.0, 0.0], torch.float32),
]


def test_composition_changes_rebuild_the_split_and_keep_tokens_exact():
    residency = DecodeSamplingResidency(4, "cpu")
    for step, (temps, dtype) in enumerate(_SCHEDULE):
        seqs = _seqs(temps, f"c{step}")
        rows = list(range(len(seqs)))
        logits = _logits(len(seqs), dtype, seed=step)
        plan = residency.plan(
            seqs, logits=logits, membership_epoch=step, row_epoch=0, batch_rows=rows
        )
        assert residency.rebuild_count == step + 1
        if plan.all_greedy:
            # fp32 all-greedy: `_select_tokens` takes argmax, sample_tokens is
            # never entered, and no index is published.
            assert plan.decisions is None
            continue

        decisions = plan.decisions
        greedy, sampled = _expected_rows(temps, dtype)
        assert (decisions.greedy_rows, decisions.sampling_rows) == (greedy, sampled)
        if decisions.mixed:
            assert decisions.greedy_index.tolist() == list(greedy)
            assert decisions.sampling_index.tolist() == list(sampled)
            # Views into the residency's own buffers: rebuilt in place, never
            # allocated on the step path.
            assert decisions.greedy_index.data_ptr() == residency._d_greedy_idx.data_ptr()
            assert decisions.sampling_index.data_ptr() == residency._d_sampling_idx.data_ptr()
        else:
            assert decisions.greedy_index is None and decisions.sampling_index is None

        torch.manual_seed(SEED + step)
        expected = sample_tokens(
            logits, temperature=plan.temps, top_p=plan.top_ps, top_k=plan.top_ks
        )
        torch.manual_seed(SEED + step)
        got = sample_tokens(
            logits,
            temperature=plan.temps,
            top_p=plan.top_ps,
            top_k=plan.top_ks,
            host_decisions=decisions,
            bad_row_counter=residency.bad_row_counter,
        )
        assert torch.equal(got, expected), f"step {step} temps={temps} {dtype}"
    assert residency.grow_count == 1


def test_the_steady_state_reuses_the_same_index_tensors():
    residency = DecodeSamplingResidency(4, "cpu")
    seqs = _seqs([0.0, 1.0, 0.0, 0.9], "s")
    rows = [0, 1, 2, 3]
    logits = _logits(4)
    first = residency.plan(seqs, logits=logits, membership_epoch=1, row_epoch=0, batch_rows=rows)
    for _ in range(32):
        plan = residency.plan(seqs, logits=logits, membership_epoch=1, row_epoch=0, batch_rows=rows)
        assert plan is first
        assert plan.decisions.greedy_index is first.decisions.greedy_index
    assert residency.rebuild_count == 1


def test_one_copy_event_covers_the_index_staging_too():
    """The pinned-rewrite wait at the top of a rebuild must cover the index
    staging buffers, so the index copies are recorded under the same event."""
    log = []

    class Event:
        def record(self, stream=None):
            log.append("record")

        def synchronize(self):
            log.append("synchronize")

    residency = DecodeSamplingResidency(4, "cpu", event_factory=Event)
    seqs = _seqs([0.0, 1.0, 0.0, 1.0], "e")
    residency.plan(seqs, logits=_logits(4), membership_epoch=1, row_epoch=0, batch_rows=[0, 1, 2, 3])
    residency.plan(seqs, logits=_logits(4), membership_epoch=2, row_epoch=0, batch_rows=[0, 1, 2, 3])
    assert log == ["record", "synchronize", "record"], log


# --------------------------------------------------------------------------
# Fail-loud: no mask fallback for a mixed batch
# --------------------------------------------------------------------------

def _mixed_inputs():
    temps = torch.tensor([0.0, 1.0, 0.0, 1.0])
    return _logits(4), temps, torch.ones(4), torch.zeros(4, dtype=torch.int64)


def _call(decisions):
    logits, temps, top_ps, top_ks = _mixed_inputs()
    return sample_tokens(
        logits, temperature=temps, top_p=top_ps, top_k=top_ks, host_decisions=decisions
    )


def _bare_mixed():
    return host_decisions_from_cast([0.0, 1.0, 0.0, 1.0], [1.0] * 4, [0] * 4)


def test_a_mixed_batch_without_row_index_is_refused():
    with pytest.raises(SamplingRowIndexError, match="without its row-index tensors"):
        _call(_bare_mixed())


@pytest.mark.parametrize(
    "greedy_index,sampling_index,match",
    [
        (torch.tensor([0, 2], dtype=torch.int32), torch.tensor([1, 3]), "int64"),
        (torch.tensor([[0, 2]]), torch.tensor([1, 3]), "1-D"),
        (torch.tensor([0]), torch.tensor([1, 3]), "holds 1 rows"),
    ],
)
def test_malformed_row_index_is_refused(greedy_index, sampling_index, match):
    from dataclasses import replace

    decisions = replace(
        _bare_mixed(), greedy_index=greedy_index, sampling_index=sampling_index
    )
    with pytest.raises(SamplingRowIndexError, match=match):
        _call(decisions)


def test_row_lists_that_do_not_partition_the_batch_are_refused():
    decisions = SamplingHostDecisions(
        n=4,
        any_greedy=True,
        any_sampling=True,
        max_top_k=0,
        needs_top_p=False,
        greedy_rows=(0,),
        sampling_rows=(1,),
        greedy_index=torch.tensor([0]),
        sampling_index=torch.tensor([1]),
    )
    with pytest.raises(SamplingRowIndexError, match="partition"):
        _call(decisions)


def test_host_decisions_compare_by_host_values_only():
    a = attach_row_index(_bare_mixed(), "cpu")
    b = attach_row_index(_bare_mixed(), "cpu")
    assert a == b == _bare_mixed()
    assert a.greedy_rows == (0, 2) and a.sampling_rows == (1, 3)
