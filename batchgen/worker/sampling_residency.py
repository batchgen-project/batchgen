"""Resident per-request sampling-parameter tensors for the decode step path.

Why this exists
---------------
``BatchGenWorker._build_sampling_tensors`` rebuilds three ``[B]`` tensors
(``temperature`` / ``top_p`` / ``top_k``) from Python lists on **every** decode
step and pushes them to the device with ``torch.tensor(..., device=cuda)``.
That constructor materializes a **pageable** host tensor and then issues an H2D
copy: a pageable-source copy stages through the driver's own pinned buffer, so
the host blocks until the copy retires, and the copy cannot retire until
everything already queued on that stream has retired -- which, on the decode
step path, is the whole-model CUDA graph replay. The host therefore stands
still inside a parameter upload. The absorbed time is GPU time, not host work
(the work itself is well under 0.15 ms/step), but while it stands there the
step cannot be pipelined at all.

What this module provides
-------------------------
``DecodeSamplingResidency`` keeps the three tensors **resident** on the device
and rewrites them only when the ordered row set of the decode batch changes.
In the steady state the step path performs one integer comparison, one identity
comparison, one ``(device, dtype)`` comparison and three tensor slices: no
allocation, no host->device copy, no device->host readback.

Correctness model
-----------------
``SequenceEntry.sampling_params`` is written exactly twice -- the ``__init__``
default ``None`` and the admission assignment in the worker -- so it is
immutable per sequence from admission to retirement. The value vector is then a
pure function of the *ordered* tuple of the batch's sequences, and that ordering
is governed by two independent things:

* the **parameter source**: membership of ``SequenceBatch``, which changes only
  in ``add_sequence`` / ``remove_sequence`` / ``__init__``. Those bump
  ``SequenceBatch.composition_epoch``.
* the **row set and row order**: the local decode-index list ``batch``, which is
  rebound (never mutated in place) at the page boundary and after an async host
  load completes, and the ``_local_to_uuid_map`` remap during host rebalance.
  Those bump the worker's decode row epoch.

Keying on membership alone would silently mis-order the temperature vector
after a page boundary, so the cache key carries both epochs. It additionally
carries the **identity** of the ``batch`` list object: ``batch`` is rebound to a
fresh list at every composition change and is never mutated in place, so
``cached_rows is batch`` is an exact O(1) statement of "same rows, same order"
that no caller can forget to maintain. The two mechanisms are not redundant: a
rebalance remap can leave ``batch`` numerically identical (and, in principle,
the same object) while changing which sequence each row resolves to, which only
the row epoch catches; and a future bump site someone forgets to add is caught
by the identity check in the common case and by ``verify=True`` always.

Staleness is the one failure mode here that would be *silent* in production --
a wrong temperature still produces plausible tokens -- so the checks are
mandatory rather than best-effort:

* length agreement is always on (``SamplingResidencyShapeError``);
* a ``None`` row, i.e. ``batch`` and the local uuid map disagreeing, is always
  fail-loud (``SamplingResidencyRowError``). The pre-residency code resolved
  such a row to ``{}`` and therefore to temperature 0, silently forcing that
  row greedy;
* under ``BATCHGEN_DECODE_ASSERT`` the full value vectors are recomputed from
  the sequences and compared against the host mirror
  (``SamplingResidencyStaleError``). This is what catches an in-place mutation
  of a ``sampling_params`` dict and any epoch-bump site that was missed.

A device-side re-verification is deliberately *not* offered: reading the device
tensors back to compare them would reintroduce exactly the sync this module
exists to remove. The pinned host mirror is the authority for the values; the
device copy's fidelity is a stream-ordering property, asserted once by the test
matrix rather than per step.

dtype policy
------------
The resident tensors are fp32/fp32/int64 -- the dtypes
``_build_sampling_tensors`` produced -- and ``sample_tokens`` keeps casting them
to ``logits.dtype`` exactly as before. Storing them pre-cast (e.g. bf16) would
change ``greedy_mask`` and the ``top_p`` comparisons from "fp32 value, rounded
once at use" to "bf16 value stored", a different answer for any non-representable
``top_p``, bought for two kernel launches. Caching the cast *result* is a valid
follow-up; it is not done here.

The host-side branch decisions this module publishes
(``SamplingHostDecisions``) must, however, be derived from the **cast** values,
because the device predicates they replace run on the cast tensor: bf16 rounding
can drive a tiny positive temperature to zero, flipping a row from sampling to
greedy, and can round a ``top_p`` just under 1.0 up to 1.0. The residency
therefore keeps a cast host mirror alongside the fp32 one and evaluates the
predicates on it, so predicate and tensor agree by construction.

The greedy/sampling split is part of those decisions. Splitting with the
device boolean mask (``logits[sampling_mask]``) is an implicit ``nonzero`` and
blocks the host behind the graph replay on every step, so the decisions also
carry the row partition: a uniform batch is not split at all, and for a mixed
batch the residency stages the two ascending row lists through pinned memory
into resident int64 device buffers once per rebuild, under the same copy event
as the parameter tensors, and ``sample_tokens`` gathers/scatters with them.

The pure resolution/decision helpers below are torch-free so the semantics can
be tested without a device; ``DecodeSamplingResidency`` itself runs on CPU with
``device="cpu"`` (no pinning, no events), which is how the unit tests drive it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Optional, Sequence, Tuple

import torch


class SamplingResidencyError(RuntimeError):
    """Base class for resident-sampling-tensor failures."""


class SamplingResidencyShapeError(SamplingResidencyError):
    """The decode batch, its sequences and the logits disagree on row count."""


class SamplingResidencyRowError(SamplingResidencyError):
    """A decode row did not resolve to a sequence.

    ``batch`` and ``_local_to_uuid_map`` disagree. Before residency this
    resolved to an empty parameter dict and therefore to temperature 0, which
    silently forced that row to greedy decoding.
    """


class SamplingResidencyStaleError(SamplingResidencyError):
    """A resident value no longer matches the value its sequence carries."""


class SamplingRowIndexError(SamplingResidencyError):
    """A mixed greedy/sampling batch lacks valid per-batch row index tensors.

    The host-decided path in ``sample_tokens`` splits a mixed batch with
    device index tensors built once per row-set change, never with a
    per-step boolean mask (whose ``nonzero`` blocks the host until the device
    catches up). A mixed batch that reaches it without those tensors -- or
    with tensors that do not partition the batch -- has no correct sync-free
    answer, so it is refused rather than silently falling back to the mask.
    """


# --------------------------------------------------------------------------
# Pure helpers (torch-free): the exact semantics of the per-step rebuild.
# --------------------------------------------------------------------------

#: Defaults and the fallback semantics of the rebuild, kept in one place.
#: ``p.get(key, default) or default`` maps a *present but falsy* value (``0``,
#: ``0.0``, ``False``, ``None``) to the default as well, which is NOT what
#: ``p.get(key, default)`` alone does. Reproduced character-for-character.
SAMPLING_DEFAULTS: Tuple[Tuple[str, Any], ...] = (
    ("temperature", 0.0),
    ("top_p", 1.0),
    ("top_k", 0),
)


def sampling_params_present(batch_sequences: Sequence[Any]) -> bool:
    """True iff any row carries a ``sampling_params`` attribute that is not None.

    An **empty dict counts as present** -- it is not ``None`` -- and that is
    precisely what the server sends for a request with no sampling fields. The
    distinction selects between two different greedy code paths in
    ``_select_tokens``, so it must be reproduced exactly.
    """
    return any(
        getattr(seq, "sampling_params", None) is not None
        for seq in batch_sequences
    )


def assert_rows_resolved(
    batch_sequences: Sequence[Any],
    batch_rows: Sequence[int],
) -> None:
    """Fail loud when a decode row did not resolve to a sequence."""
    for i, seq in enumerate(batch_sequences):
        if seq is None:
            local_idx = batch_rows[i] if i < len(batch_rows) else None
            raise SamplingResidencyRowError(
                f"decode row {i} (local_idx={local_idx}) resolved to None: "
                f"batch and _local_to_uuid_map disagree "
                f"(rows={len(batch_rows)}, sequences={len(batch_sequences)})"
            )


def resolve_sampling_values(
    batch_sequences: Sequence[Any],
) -> Tuple[List[float], List[float], List[int]]:
    """Per-row ``(temperatures, top_ps, top_ks)`` in ``batch`` order.

    Identical expressions, identical order and identical fallback semantics to
    ``BatchGenWorker._build_sampling_tensors``.
    """
    params = [getattr(seq, "sampling_params", None) or {} for seq in batch_sequences]
    temps = [p.get("temperature", 0.0) or 0.0 for p in params]
    top_ps = [p.get("top_p", 1.0) or 1.0 for p in params]
    top_ks = [p.get("top_k", 0) or 0 for p in params]
    return temps, top_ps, top_ks


def batch_all_greedy(temps: Sequence[float]) -> bool:
    """True iff no row has a positive effective temperature.

    Evaluated on the **fp32 host values**, which is what
    ``_decode_batch_all_greedy`` does. The ``greedy_mask`` inside
    ``sample_tokens`` is a different predicate on the *cast* values; the two are
    deliberately kept distinct so neither changes behaviour.
    """
    for t in temps:
        if t > 0:
            return False
    return True


@dataclass(frozen=True)
class SamplingHostDecisions:
    """Branch decisions ``sample_tokens`` would otherwise read back from device.

    Every field is derived from the **cast** parameter values, i.e. from the
    same numbers the device tensors hold after ``.to(logits.dtype)``.

    ``greedy_rows`` / ``sampling_rows`` are the ascending row ids of the two
    halves of the split, i.e. exactly the rows ``greedy_mask`` /
    ``sampling_mask`` would select, in the order boolean indexing would
    gather them. They partition ``range(n)``.

    ``greedy_index`` / ``sampling_index`` are the same row ids as int64 device
    tensors. They are only needed -- and only required -- when the batch is
    **mixed** (``any_greedy and any_sampling``): a uniform batch needs no
    split at all. They are built once per row-set change (by
    ``DecodeSamplingResidency`` through pinned staging, or by
    ``attach_row_index`` off the step path), never per step. They are excluded
    from equality so two decisions compare by their host values only.
    """

    n: int
    any_greedy: bool
    any_sampling: bool
    max_top_k: int
    needs_top_p: bool
    greedy_rows: Tuple[int, ...] = ()
    sampling_rows: Tuple[int, ...] = ()
    greedy_index: Optional[torch.Tensor] = field(default=None, compare=False, repr=False)
    sampling_index: Optional[torch.Tensor] = field(default=None, compare=False, repr=False)

    @property
    def mixed(self) -> bool:
        """True iff the batch has both greedy and sampled rows."""
        return self.any_greedy and self.any_sampling


def host_decisions_from_cast(
    cast_temps: Sequence[float],
    cast_top_ps: Sequence[float],
    top_ks: Sequence[int],
) -> SamplingHostDecisions:
    """Compute the host-side branch decisions for one row set.

    Mirrors, in order, ``sampling.py``'s ``greedy_mask.any()``,
    ``sampling_mask.any()``, ``s_top_ks.max().item()`` and
    ``(s_top_ps < 1.0).any()``. The last two are reductions over the *sampling*
    rows only, so they are computed over the same subset the device masks
    select. The greedy/sampling row lists are the rows those masks select.

    Torch-free: the device index tensors a mixed batch needs are attached
    separately (``attach_row_index`` or the residency's pinned path).
    """
    n = len(cast_temps)
    any_greedy = False
    any_sampling = False
    max_top_k = 0
    needs_top_p = False
    greedy_rows: List[int] = []
    sampling_rows: List[int] = []
    for i in range(n):
        if cast_temps[i] <= 0:
            any_greedy = True
            greedy_rows.append(i)
            continue
        any_sampling = True
        sampling_rows.append(i)
        k = top_ks[i]
        if k > max_top_k:
            max_top_k = k
        if cast_top_ps[i] < 1.0:
            needs_top_p = True
    return SamplingHostDecisions(
        n=n,
        any_greedy=any_greedy,
        any_sampling=any_sampling,
        max_top_k=int(max_top_k),
        needs_top_p=needs_top_p,
        greedy_rows=tuple(greedy_rows),
        sampling_rows=tuple(sampling_rows),
    )


def attach_row_index(
    decisions: SamplingHostDecisions, device: Any
) -> SamplingHostDecisions:
    """Return ``decisions`` with device row-index tensors for a mixed batch.

    A uniform batch needs no split, so it is returned unchanged. For a mixed
    batch the tensors are built with ``torch.tensor(..., device=device)``,
    which on CUDA is a pageable-source host->device copy that blocks the host
    behind any queued work: this helper is for tests and for callers off the
    decode step path. The decode step path gets its tensors from
    ``DecodeSamplingResidency``, which stages them through pinned memory once
    per row-set change.
    """
    if not decisions.mixed:
        return decisions
    return replace(
        decisions,
        greedy_index=torch.tensor(decisions.greedy_rows, dtype=torch.int64, device=device),
        sampling_index=torch.tensor(decisions.sampling_rows, dtype=torch.int64, device=device),
    )


# --------------------------------------------------------------------------
# The residency object
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingPlan:
    """What ``_select_tokens`` needs for one step, with no device round trip.

    ``has_params=False`` reproduces the "no row carries sampling params" case,
    which falls through to the plain ``torch.argmax`` branch.
    ``all_greedy=True`` reproduces the all-greedy fast path, which must not
    build any tensor at all. Only when both of those are resolved are
    ``temps`` / ``top_ps`` / ``top_ks`` / ``decisions`` populated.
    """

    n: int
    has_params: bool
    all_greedy: bool
    temps: Optional[torch.Tensor] = None
    top_ps: Optional[torch.Tensor] = None
    top_ks: Optional[torch.Tensor] = None
    decisions: Optional[SamplingHostDecisions] = None


_EMPTY_PLAN = SamplingPlan(n=0, has_params=False, all_greedy=False)


class DecodeSamplingResidency:
    """Device-resident sampling-parameter tensors for one decode round.

    Lifetime is deliberately **one decode round**, owned by a local of
    ``decoding_continuous`` rather than by ``self``: a hot reload rebinds
    methods on the live worker, so state kept on ``self`` could outlive a
    slot-layout change. A new round starts a new residency.
    """

    def __init__(
        self,
        capacity: int,
        device: Any,
        *,
        event_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._device = torch.device(device)
        self._pin = self._device.type == "cuda"
        if event_factory is None and self._device.type == "cuda":
            event_factory = torch.cuda.Event
        self._event_factory = event_factory

        self._capacity = 0
        self._h_temps: Optional[torch.Tensor] = None
        self._h_top_ps: Optional[torch.Tensor] = None
        self._h_top_ks: Optional[torch.Tensor] = None
        self._d_temps: Optional[torch.Tensor] = None
        self._d_top_ps: Optional[torch.Tensor] = None
        self._d_top_ks: Optional[torch.Tensor] = None
        # Row-index tensors for the greedy/sampling split of a mixed batch.
        # Each half holds at most ``capacity`` rows, so each buffer is sized
        # to capacity; only the leading ``len(rows)`` entries are meaningful.
        self._h_greedy_idx: Optional[torch.Tensor] = None
        self._h_sampling_idx: Optional[torch.Tensor] = None
        self._d_greedy_idx: Optional[torch.Tensor] = None
        self._d_sampling_idx: Optional[torch.Tensor] = None
        self._allocate(max(int(capacity), 1))

        self._key: Optional[Tuple[Any, ...]] = None
        self._rows: Optional[Sequence[int]] = None
        self._n = 0
        self._plan = _EMPTY_PLAN
        self._values: Tuple[List[float], List[float], List[int]] = ([], [], [])
        self._uuids: List[Optional[str]] = []
        self._copy_event: Optional[Any] = None

        # Device-resident count of rows that fell back to greedy because
        # filtering left them with invalid probabilities. Incremented on
        # device so the step path never reads it; drained by the caller at
        # heartbeat rate, where the device is already caught up.
        self.bad_row_counter = torch.zeros((), dtype=torch.int64, device=self._device)
        self._bad_rows_reported = 0

        # Telemetry only: how often the buffers had to grow and how often the
        # resident values were rewritten. Both are boundary-rate by
        # construction; a per-step rebuild count would mean the key is wrong.
        self.grow_count = 0
        self.rebuild_count = 0

    # -- buffers ----------------------------------------------------------

    def _allocate(self, capacity: int) -> None:
        self._capacity = capacity
        self._h_temps = torch.empty(capacity, dtype=torch.float32, pin_memory=self._pin)
        self._h_top_ps = torch.empty(capacity, dtype=torch.float32, pin_memory=self._pin)
        self._h_top_ks = torch.empty(capacity, dtype=torch.int64, pin_memory=self._pin)
        self._d_temps = torch.empty(capacity, dtype=torch.float32, device=self._device)
        self._d_top_ps = torch.empty(capacity, dtype=torch.float32, device=self._device)
        self._d_top_ks = torch.empty(capacity, dtype=torch.int64, device=self._device)
        self._h_greedy_idx = torch.empty(capacity, dtype=torch.int64, pin_memory=self._pin)
        self._h_sampling_idx = torch.empty(capacity, dtype=torch.int64, pin_memory=self._pin)
        self._d_greedy_idx = torch.empty(capacity, dtype=torch.int64, device=self._device)
        self._d_sampling_idx = torch.empty(capacity, dtype=torch.int64, device=self._device)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def device(self) -> torch.device:
        return self._device

    # -- the step path ----------------------------------------------------

    def plan(
        self,
        batch_sequences: Sequence[Any],
        *,
        logits: torch.Tensor,
        membership_epoch: int,
        row_epoch: int,
        batch_rows: Sequence[int],
        verify: bool = False,
    ) -> SamplingPlan:
        """Return this step's sampling plan, rebuilding only on a row change.

        ``verify`` recomputes every value from ``batch_sequences`` and compares
        it against the host mirror. It is the ``BATCHGEN_DECODE_ASSERT`` check
        and is the only thing that catches a mutated ``sampling_params`` dict or
        a missing epoch bump; it is O(rows) host work and must stay off in
        production.
        """
        n = len(batch_sequences)
        logits_rows = int(logits.shape[0])
        if n != len(batch_rows) or n != logits_rows:
            raise SamplingResidencyShapeError(
                f"decode row count disagreement: sequences={n}, "
                f"batch_rows={len(batch_rows)}, logits_rows={logits_rows}, "
                f"resident_n={self._n}"
            )

        key = (membership_epoch, row_epoch, str(self._device), logits.dtype, n)
        if self._key != key or self._rows is not batch_rows:
            self._rebuild(batch_sequences, key, batch_rows, logits.dtype)
        elif verify:
            # A fresh rebuild is correct by construction, so only the fast path
            # is worth re-verifying -- that is where staleness could hide.
            self._verify(batch_sequences)
        return self._plan

    # -- rebuild ----------------------------------------------------------

    def _rebuild(
        self,
        batch_sequences: Sequence[Any],
        key: Tuple[Any, ...],
        batch_rows: Sequence[int],
        dtype: torch.dtype,
    ) -> None:
        assert_rows_resolved(batch_sequences, batch_rows)
        n = len(batch_sequences)
        self.rebuild_count += 1
        self._key = key
        self._rows = batch_rows
        self._n = n
        self._uuids = [getattr(seq, "uuid", None) for seq in batch_sequences]

        if n == 0 or not sampling_params_present(batch_sequences):
            self._values = ([], [], [])
            self._plan = SamplingPlan(n=n, has_params=False, all_greedy=False)
            return

        temps, top_ps, top_ks = resolve_sampling_values(batch_sequences)
        self._values = (temps, top_ps, top_ks)

        if batch_all_greedy(temps):
            # The all-greedy fast path returns ``logits.argmax`` without
            # building or reading any parameter tensor, so do not upload.
            self._plan = SamplingPlan(n=n, has_params=True, all_greedy=True)
            return

        if self._copy_event is not None:
            # A pinned staging buffer must not be rewritten -- or dropped -- while
            # a copy out of it is still in flight. Two composition changes in
            # quick succession are the only way to reach that. This runs BEFORE
            # the grow below on purpose: growing rebinds the pinned tensors and
            # frees the old ones, so a copy still reading them would be reading
            # freed host memory, which is strictly worse than a torn value.
            self._copy_event.synchronize()

        if n > self._capacity:
            # Growing is an allocation, so it must never happen mid-interval.
            # It can only happen here, inside a rebuild, i.e. only when the
            # key already changed -- the coupling is structural, not incidental.
            self._allocate(n)
            self.grow_count += 1

        self._h_temps[:n].copy_(torch.tensor(temps, dtype=torch.float32))
        self._h_top_ps[:n].copy_(torch.tensor(top_ps, dtype=torch.float32))
        self._h_top_ks[:n].copy_(torch.tensor(top_ks, dtype=torch.int64))

        # Branch decisions come from the cast mirror, never from the fp32
        # values: the device predicates they replace run on the cast tensor.
        # Taken before the uploads because a mixed batch's row partition is
        # part of what gets uploaded.
        cast_temps = self._h_temps[:n].to(dtype).tolist()
        cast_top_ps = self._h_top_ps[:n].to(dtype).tolist()
        decisions = host_decisions_from_cast(cast_temps, cast_top_ps, top_ks)
        n_greedy = len(decisions.greedy_rows)
        n_sampling = len(decisions.sampling_rows)
        if decisions.mixed:
            self._h_greedy_idx[:n_greedy].copy_(
                torch.tensor(decisions.greedy_rows, dtype=torch.int64)
            )
            self._h_sampling_idx[:n_sampling].copy_(
                torch.tensor(decisions.sampling_rows, dtype=torch.int64)
            )

        # Pinned source => a genuinely asynchronous copy, issued on the stream
        # that will later run the replay and the sampling. Stream order alone
        # makes the values visible to the first consumer; no event wait is
        # needed here. If this update ever moves onto a side stream, an
        # ``event.wait(current_stream)`` becomes mandatory -- omitting it would
        # be a silent wrong-value bug.
        self._d_temps[:n].copy_(self._h_temps[:n], non_blocking=True)
        self._d_top_ps[:n].copy_(self._h_top_ps[:n], non_blocking=True)
        self._d_top_ks[:n].copy_(self._h_top_ks[:n], non_blocking=True)
        if decisions.mixed:
            self._d_greedy_idx[:n_greedy].copy_(
                self._h_greedy_idx[:n_greedy], non_blocking=True
            )
            self._d_sampling_idx[:n_sampling].copy_(
                self._h_sampling_idx[:n_sampling], non_blocking=True
            )
            decisions = replace(
                decisions,
                greedy_index=self._d_greedy_idx[:n_greedy],
                sampling_index=self._d_sampling_idx[:n_sampling],
            )
        # One event covers every copy above, so the pinned-rewrite wait at the
        # top of the next rebuild also covers the index staging buffers.
        if self._event_factory is not None:
            event = self._copy_event
            if event is None:
                event = self._event_factory()
                self._copy_event = event
            event.record()

        self._plan = SamplingPlan(
            n=n,
            has_params=True,
            all_greedy=False,
            temps=self._d_temps[:n],
            top_ps=self._d_top_ps[:n],
            top_ks=self._d_top_ks[:n],
            decisions=decisions,
        )

    # -- checks -----------------------------------------------------------

    def _verify(self, batch_sequences: Sequence[Any]) -> None:
        """Recompute every value and compare against the host mirror."""
        if not self._plan.has_params:
            if sampling_params_present(batch_sequences):
                raise SamplingResidencyStaleError(
                    "resident plan says no row carries sampling params, but "
                    f"{sum(getattr(s, 'sampling_params', None) is not None for s in batch_sequences)} "
                    "row(s) now do"
                )
            return

        temps, top_ps, top_ks = resolve_sampling_values(batch_sequences)
        c_temps, c_top_ps, c_top_ks = self._values
        for name, cached, fresh in (
            ("temperature", c_temps, temps),
            ("top_p", c_top_ps, top_ps),
            ("top_k", c_top_ks, top_ks),
        ):
            if len(cached) != len(fresh):
                raise SamplingResidencyStaleError(
                    f"resident {name} vector has {len(cached)} rows, "
                    f"{len(fresh)} row(s) now resolve"
                )
            for i, (a, b) in enumerate(zip(cached, fresh)):
                if a != b:
                    raise SamplingResidencyStaleError(
                        f"resident {name} row {i} "
                        f"(uuid={self._uuids[i] if i < len(self._uuids) else None}) "
                        f"is {a!r}, sequence now carries {b!r}"
                    )
        if batch_all_greedy(temps) != self._plan.all_greedy:
            raise SamplingResidencyStaleError(
                f"resident all_greedy={self._plan.all_greedy} but the rows now "
                f"resolve to all_greedy={batch_all_greedy(temps)}"
            )

    # -- diagnostics ------------------------------------------------------

    def drain_bad_rows(self) -> int:
        """Rows that fell back to greedy since the last drain.

        Reads the device counter, so the caller must be at a point where the
        device is already caught up (the rank-0 heartbeat is, because the
        sampled-token readback has already synchronized). Never call this from
        the step path.
        """
        total = int(self.bad_row_counter.item())
        delta = total - self._bad_rows_reported
        self._bad_rows_reported = total
        return delta if delta > 0 else 0
