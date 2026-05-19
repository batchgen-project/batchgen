"""Integration tests for the GLM-5 128K single-sequence KV capacity-abort guard.

Issue: batchgen-project/batchgen-internal#1

The capacity-abort path is hard to exercise on a real BatchGenWorker without a
multi-hundred-GB model and a real GPU, so these tests target the seam:

  candidate_info dict
        │
        ▼
  compute_capacity_aborts()  ◄── pure function, easy to drive
        │
        ▼
  BoundaryDecisions(capacity_aborted_uuids=…)
        │
        ▼ (broadcast — simulated by pickle roundtrip)
        ▼
  receiver mirrors _finish_capacity = True on each rank's SequenceEntry
        │
        ▼
  _get_finish_reason(seq) → "capacity"

The three named scenarios — over-cap only, under-cap only, mixed — exercise the
abort decision under realistic candidate shapes. The seam tests catch wiring
regressions (missing default for the new dataclass field, pickle failures,
finish-reason ordering).

The companion env-var override BATCHGEN_DEBUG_SINGLE_SEQ_CAP_PAGES lets a real
running server reproduce the same behavior with a tiny model — see the
docstring in batchgen_worker._init_gpu_kv_with_actual_size.
"""

import math
import pickle
from typing import Dict, List, Optional

import pytest
import torch

from batchgen.sequence import (
    SINGLE_SEQ_PAGE_HEADROOM,
    SequenceEntry,
    SequenceStatus,
)
from batchgen.continuous_batching import (
    BoundaryDecisions,
    compute_capacity_aborts,
)


# ============ Helpers ============


def make_seq(
    uuid: str,
    prompt_length: int = 64,
    max_decode_length: int = 64,
    assigned_rank: int = 0,
    status: SequenceStatus = SequenceStatus.PREFILLED,
) -> SequenceEntry:
    """Build a SequenceEntry suitable for exercising the abort guard.

    Matches the pattern in test_dynamic_host_kv.py — only the fields the
    guard actually reads are populated.
    """
    seq = SequenceEntry(uuid, global_idx=int(uuid[1:]) if uuid[1:].isdigit() else 0,
                        prompt_length=prompt_length, max_decode_length=max_decode_length)
    seq.status = status
    seq.assigned_rank = assigned_rank
    # Lightweight token buffers — the guard never reads them, but downstream
    # validate_metadata calls do on a real worker.
    seq.input_ids = torch.zeros((1, prompt_length + max_decode_length), dtype=torch.long)
    seq.decoded_tokens = torch.zeros((1, max_decode_length), dtype=torch.long)
    return seq


def make_candidate_info(
    seqs: List[SequenceEntry],
    pages_needed: int = 4,
) -> Dict[str, Dict]:
    """Build the candidate_info payload that flows into compute_capacity_aborts.

    `pages_needed` is the per-iteration value the scheduler would set (initial
    load buffer, much smaller than the worst-case lifetime budget). The guard
    deliberately ignores this and reads `kv_token_budget` directly — these
    tests assert that choice is honored.
    """
    return {
        seq.uuid: {
            "pages_needed": pages_needed,
            "assigned_rank": seq.assigned_rank,
            "status": seq.status.name,
            "decoded_length": seq.decoded_length,
        }
        for seq in seqs
    }


class _SeqRegistry:
    """Minimal stand-in for `worker.global_batch.get_sequence`."""

    def __init__(self, seqs: List[SequenceEntry]):
        self._by_uuid = {seq.uuid: seq for seq in seqs}

    def get_sequence(self, uuid: str) -> Optional[SequenceEntry]:
        return self._by_uuid.get(uuid)


class _FakeFinishReasonWorker:
    """Smallest possible stub that satisfies `_get_finish_reason`'s read set.

    Only the attributes touched by the new `_finish_capacity` branch are
    populated. The branch must fire BEFORE the repetition/length/stop branches
    so the stub never reaches them.

    The import of `BatchGenWorker` is lazy and guarded by pytest.skip so the
    seam tests can still run in environments that lack CUDA / NCCL deps.
    """

    rank = 0
    model_context_length = 1 << 20  # huge — guarantees the length branch
                                     # is not the cause of completion
    _ignore_eos = False

    def _get_finish_reason(self, seq):
        try:
            from batchgen.batchgen_worker import BatchGenWorker
        except Exception as exc:
            pytest.skip(f"BatchGenWorker import unavailable in this env: {exc}")
        return BatchGenWorker._get_finish_reason(self, seq)


# ============ T1: over-cap only ============


class TestOverCapOnly:
    """All candidates exceed the per-rank cap; all must be aborted."""

    SAFE_CAP = 32  # pages

    def _build(self) -> List[SequenceEntry]:
        # kv_token_budget = 64 + 8192 = 8256 tokens
        # required pages = ceil(8256/64) + 8 = 137 > 32 cap
        return [
            make_seq("u1", prompt_length=64, max_decode_length=8192),
            make_seq("u2", prompt_length=64, max_decode_length=8192),
            make_seq("u3", prompt_length=64, max_decode_length=8192),
        ]

    def test_all_three_aborted(self):
        seqs = self._build()
        candidate_info = make_candidate_info(seqs)
        registry = _SeqRegistry(seqs)

        aborted = compute_capacity_aborts(
            candidate_info=candidate_info,
            safe_cap=self.SAFE_CAP,
            get_seq_fn=registry.get_sequence,
        )

        assert aborted == ["u1", "u2", "u3"], (
            "All three over-cap candidates must be aborted in iteration order"
        )

    def test_returned_uuids_match_admission_check(self):
        """The threshold is `get_admission_pages_required > safe_cap`, not
        `pages_needed > safe_cap`. With pages_needed=4 (well under cap) but
        admission=137 (over cap), the guard still fires — this is the bug
        the issue tracks."""
        seqs = self._build()
        candidate_info = make_candidate_info(seqs, pages_needed=4)

        aborted = compute_capacity_aborts(
            candidate_info=candidate_info,
            safe_cap=self.SAFE_CAP,
            get_seq_fn=_SeqRegistry(seqs).get_sequence,
        )
        assert len(aborted) == 3
        for seq in seqs:
            assert seq.get_admission_pages_required() > self.SAFE_CAP

    def test_finish_capacity_flag_drives_finish_reason(self):
        """After the receiver mirrors `_finish_capacity = True`, the
        downstream `_get_finish_reason` must return "capacity"."""
        seqs = self._build()
        for s in seqs:
            s._finish_capacity = True
        worker = _FakeFinishReasonWorker()
        for s in seqs:
            assert worker._get_finish_reason(s) == "capacity"


# ============ T2: under-cap only ============


class TestUnderCapOnly:
    """All candidates fit comfortably; abort must be a no-op."""

    SAFE_CAP = 2048  # 128K tokens per seq budget

    def _build(self) -> List[SequenceEntry]:
        # kv_token_budget = 64 + 1024 = 1088 tokens
        # required pages = ceil(1088/64) + 8 = 25 << 2048 cap
        return [
            make_seq("u1", prompt_length=64, max_decode_length=1024),
            make_seq("u2", prompt_length=64, max_decode_length=1024),
            make_seq("u3", prompt_length=64, max_decode_length=1024),
        ]

    def test_no_aborts(self):
        seqs = self._build()
        aborted = compute_capacity_aborts(
            candidate_info=make_candidate_info(seqs),
            safe_cap=self.SAFE_CAP,
            get_seq_fn=_SeqRegistry(seqs).get_sequence,
        )
        assert aborted == []

    def test_finish_capacity_stays_false(self):
        """The guard must not flip _finish_capacity as a side effect — the
        pure helper has no side effects; the worker is responsible for
        setting the flag only on returned uuids."""
        seqs = self._build()
        compute_capacity_aborts(
            candidate_info=make_candidate_info(seqs),
            safe_cap=self.SAFE_CAP,
            get_seq_fn=_SeqRegistry(seqs).get_sequence,
        )
        for s in seqs:
            assert s._finish_capacity is False


# ============ T3: mixed batch ============


class TestMixedBatch:
    """Some over-cap, some under-cap. Critical: under-cap requests must NOT
    be poisoned by the presence of over-cap requests — this is the actual
    regression the 2045/2048 failure was about."""

    SAFE_CAP = 32

    def _build(self) -> List[SequenceEntry]:
        return [
            make_seq("u1_under", prompt_length=64, max_decode_length=512),     # 17 pages
            make_seq("u2_over", prompt_length=64, max_decode_length=8192),    # 137 pages
            make_seq("u3_under", prompt_length=64, max_decode_length=512),     # 17 pages
            make_seq("u4_over", prompt_length=64, max_decode_length=8192),    # 137 pages
            make_seq("u5_under", prompt_length=64, max_decode_length=512),     # 17 pages
        ]

    def test_only_over_cap_aborted(self):
        seqs = self._build()
        aborted = compute_capacity_aborts(
            candidate_info=make_candidate_info(seqs),
            safe_cap=self.SAFE_CAP,
            get_seq_fn=_SeqRegistry(seqs).get_sequence,
        )
        assert aborted == ["u2_over", "u4_over"], (
            "Only the two over-cap UUIDs should be returned, and iteration "
            "order from candidate_info must be preserved so the broadcast "
            "decision is deterministic across ranks"
        )

    def test_under_cap_seqs_not_flagged(self):
        """Simulate the full flow: rank-0 helper returns the abort list,
        worker flips _finish_capacity on those uuids only. Under-cap seqs
        must remain eligible for normal completion."""
        seqs = self._build()
        aborted = compute_capacity_aborts(
            candidate_info=make_candidate_info(seqs),
            safe_cap=self.SAFE_CAP,
            get_seq_fn=_SeqRegistry(seqs).get_sequence,
        )
        registry = _SeqRegistry(seqs)
        for uuid in aborted:
            registry.get_sequence(uuid)._finish_capacity = True

        worker = _FakeFinishReasonWorker()
        # Over-cap seqs report capacity
        assert worker._get_finish_reason(registry.get_sequence("u2_over")) == "capacity"
        assert worker._get_finish_reason(registry.get_sequence("u4_over")) == "capacity"
        # Under-cap seqs remain not-completed: _finish_capacity False, decode
        # not yet at max, no EOS, no repetition. The capacity branch must NOT
        # fire for them.
        for uuid in ("u1_under", "u3_under", "u5_under"):
            seq = registry.get_sequence(uuid)
            assert seq._finish_capacity is False


# ============ Seam tests ============


class TestComputeCapacityAbortsEdgeCases:
    """Boundary conditions for the pure helper."""

    def test_safe_cap_none_is_noop(self):
        """Before _init_gpu_kv_with_actual_size, safe_cap is None and the
        guard must do nothing — even an obviously over-cap request gets
        admitted (the runtime guard fires once the cap is known)."""
        seqs = [make_seq("u1", max_decode_length=131072)]
        aborted = compute_capacity_aborts(
            candidate_info=make_candidate_info(seqs),
            safe_cap=None,
            get_seq_fn=_SeqRegistry(seqs).get_sequence,
        )
        assert aborted == []

    def test_empty_candidate_info_is_noop(self):
        aborted = compute_capacity_aborts(
            candidate_info={},
            safe_cap=32,
            get_seq_fn=lambda u: None,
        )
        assert aborted == []

    def test_missing_sequence_skipped(self):
        """A candidate that was evicted between gather and decide must be
        tolerated, not raise."""
        seqs = [make_seq("u1", max_decode_length=8192)]
        info = make_candidate_info(seqs)
        info["u_missing"] = {"pages_needed": 4, "assigned_rank": 0,
                             "status": "PREFILLED", "decoded_length": 0}
        aborted = compute_capacity_aborts(
            candidate_info=info,
            safe_cap=32,
            get_seq_fn=_SeqRegistry(seqs).get_sequence,
        )
        assert aborted == ["u1"]

    def test_at_threshold_does_not_abort(self):
        """Requirement equal to safe_cap must NOT abort (strict greater-than).

        Tune kv_token_budget so admission_pages == safe_cap exactly.
        """
        # Pick budget such that ceil(budget/64) + 8 == 32, i.e. ceil(budget/64) == 24
        # → budget in (23*64, 24*64] = (1472, 1536]. Use 1536.
        seq = make_seq("u1", prompt_length=0, max_decode_length=1536)
        assert seq.get_admission_pages_required() == 32

        aborted = compute_capacity_aborts(
            candidate_info=make_candidate_info([seq]),
            safe_cap=32,
            get_seq_fn=_SeqRegistry([seq]).get_sequence,
        )
        assert aborted == [], "exactly-at-threshold must not abort"


class TestBoundaryDecisionsBroadcastShape:
    """Catches wiring regressions in the new dataclass field."""

    def test_pickle_roundtrip_preserves_capacity_aborted_uuids(self):
        d = BoundaryDecisions(
            completed_uuids=["a", "b"],
            active_uuids=[],
            host_growth_uuids=[],
            host_growth_pages=[],
            growth_feasible=False,
            host_evicted_uuids=[],
            onhold_uuids=[],
            seqs_needing_extension=[],
            new_load_uuids=[],
            decode_uuids_final=[],
            capacity_aborted_uuids=["a", "b"],
        )
        d2 = pickle.loads(pickle.dumps(d))
        assert d2.capacity_aborted_uuids == ["a", "b"]
        # Aborted uuids must always appear in completed_uuids too — this is
        # the invariant that drives the existing release/report path.
        assert set(d2.capacity_aborted_uuids).issubset(set(d2.completed_uuids))

    def test_backwards_compat_default_is_empty_list(self):
        """A BoundaryDecisions constructed by older code (without the new
        field) must round-trip cleanly with an empty default."""
        d = BoundaryDecisions(
            completed_uuids=[],
            active_uuids=[],
            host_growth_uuids=[],
            host_growth_pages=[],
            growth_feasible=False,
            host_evicted_uuids=[],
            onhold_uuids=[],
            seqs_needing_extension=[],
            new_load_uuids=[],
            decode_uuids_final=[],
        )
        d2 = pickle.loads(pickle.dumps(d))
        assert d2.capacity_aborted_uuids == []
