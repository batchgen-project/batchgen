"""Unit tests for gather_boundary_state + absorb_cross_rank_metadata.

Covers the Phase 2.8.1a port of ``_boundary_gather_state`` (lines
6726-6804) + metadata-absorption slice of ``_boundary_merge_and_decide``
(lines 6836-6888) from ``batchgen/batchgen_worker.py``.

Focus areas:
  - Single ``all_gather_object`` collective for the gather.
  - Rank-owned sequences appear in ``seq_state``; non-owned don't.
  - Load candidates come from rank-owned PREFILLED / ON_HOLD uuids
    that are NOT already in ``decode_uuids``.
  - ``gpu_manager`` free-page reporting handles both live and pre-init.
  - Absorb stamps ``owning_rank`` + copies gathered fields onto shadow
    ``SequenceEntry`` objects for non-owned sequences.
  - Orphan path: uuids absent from every rank's gather → force COMPLETED
    + pruned from ``decode_uuids``.
  - CTX drift on a gathered non-owned sequence is fixed-up silently.
"""

from __future__ import annotations

import types

import pytest
import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.boundary.synchronizer import (
    BoundaryPayload,
    BoundarySynchronizer,
    LoadCandidateState,
    SeqBoundaryState,
)
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator
from tests.unit.worker.fakes import FakeCollectiveBackend, FakeLegacyBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(rank: int = 0, world_size: int = 1) -> WorkerState:
    return WorkerState(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=rank,
        torch_device=torch.device("cpu"),
    )


def _add_seq(
    state: WorkerState,
    uuid: str,
    *,
    assigned_rank: int,
    status: SequenceStatus = SequenceStatus.IN_DECODE,
    prompt_length: int = 10,
    decoded_length: int = 5,
    gpu_pages: int = 2,
    host_pages: int = 4,
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid, global_idx=0, prompt_length=prompt_length, max_decode_length=100, text=""
    )
    seq.original_prompt_length = prompt_length
    seq.decoded_length = decoded_length
    seq.current_context_length = prompt_length + decoded_length
    seq.assigned_rank = assigned_rank
    seq.gpu_pages_allocated = gpu_pages
    seq.host_pages_allocated = host_pages
    seq.host_token_capacity = host_pages * seq.PAGE_SIZE
    # Skip the state-machine transition path by writing status directly;
    # the gather/absorb logic reads the field without triggering
    # transitions, and these helpers only construct fixtures.
    seq.status = status
    state.global_batch.sequences[uuid] = seq
    state.global_batch._status_index[status].add(uuid)
    return seq


def _fake_gpu_manager(*, num_free_pages: int = 0, is_initialized: bool = True) -> object:
    stats = types.SimpleNamespace(num_free_pages=num_free_pages)
    return types.SimpleNamespace(
        is_initialized=is_initialized, get_stats=lambda: stats
    )


# ---------------------------------------------------------------------------
# gather_boundary_state
# ---------------------------------------------------------------------------


class TestGatherBoundaryState:
    def test_issues_single_all_gather_object(self) -> None:
        state = _make_state(rank=0, world_size=1)
        _add_seq(state, "u1", assigned_rank=0)
        legacy = FakeLegacyBackend(rank=0, world_size=1)
        legacy._uuid_to_local = {"u1": 0}
        col = FakeCollectiveBackend(rank=0, world_size=1)
        bsync = BoundarySynchronizer(state, SyncCoordinator(state, col), col)

        gpu = _fake_gpu_manager(num_free_pages=12)
        payloads, chunk_size = bsync.gather_boundary_state(["u1"], gpu, legacy)

        assert col.call_names() == ["all_gather_object"]
        assert chunk_size == legacy.effective_chunk_size()
        assert len(payloads) == 1
        assert payloads[0] is not None
        assert payloads[0].free_pages == 12
        assert set(payloads[0].seq_state.keys()) == {"u1"}

    def test_reports_zero_free_pages_when_gpu_manager_uninitialized(self) -> None:
        state = _make_state(rank=0, world_size=1)
        _add_seq(state, "u1", assigned_rank=0)
        legacy = FakeLegacyBackend(rank=0, world_size=1)
        legacy._uuid_to_local = {"u1": 0}
        col = FakeCollectiveBackend(rank=0, world_size=1)
        bsync = BoundarySynchronizer(state, SyncCoordinator(state, col), col)

        gpu = _fake_gpu_manager(num_free_pages=999, is_initialized=False)
        payloads, _ = bsync.gather_boundary_state(["u1"], gpu, legacy)

        assert payloads[0].free_pages == 0

    def test_skips_uuids_not_owned_by_this_rank(self) -> None:
        """Only sequences whose uuid is in ``uuid_to_local_map`` land in
        ``seq_state`` — that's legacy's exact filter at 6752."""
        state = _make_state(rank=0, world_size=2)
        _add_seq(state, "mine", assigned_rank=0)
        _add_seq(state, "other", assigned_rank=1)
        legacy = FakeLegacyBackend(rank=0, world_size=2)
        legacy._uuid_to_local = {"mine": 0}  # "other" NOT registered on this rank
        col = FakeCollectiveBackend(rank=0, world_size=2)
        bsync = BoundarySynchronizer(state, SyncCoordinator(state, col), col)

        payloads, _ = bsync.gather_boundary_state(["mine", "other"], _fake_gpu_manager(), legacy)
        assert set(payloads[0].seq_state.keys()) == {"mine"}
        assert "other" not in payloads[0].seq_state

    def test_load_candidates_exclude_decode_set_and_completed(self) -> None:
        state = _make_state(rank=0, world_size=1)
        _add_seq(state, "decoding", assigned_rank=0, status=SequenceStatus.IN_DECODE)
        _add_seq(state, "prefilled", assigned_rank=0, status=SequenceStatus.PREFILLED)
        _add_seq(state, "onhold", assigned_rank=0, status=SequenceStatus.ON_HOLD)
        _add_seq(state, "queueing", assigned_rank=0, status=SequenceStatus.QUEUEING)
        _add_seq(state, "done", assigned_rank=0, status=SequenceStatus.COMPLETED)
        legacy = FakeLegacyBackend(rank=0, world_size=1)
        legacy._uuid_to_local = {
            "decoding": 0,
            "prefilled": 1,
            "onhold": 2,
            "queueing": 3,
            "done": 4,
        }
        col = FakeCollectiveBackend(rank=0, world_size=1)
        bsync = BoundarySynchronizer(state, SyncCoordinator(state, col), col)

        payloads, _ = bsync.gather_boundary_state(
            ["decoding"], _fake_gpu_manager(), legacy
        )
        candidates = payloads[0].candidate_state
        assert set(candidates.keys()) == {"prefilled", "onhold"}
        assert isinstance(candidates["prefilled"], LoadCandidateState)
        assert candidates["prefilled"].status == "PREFILLED"


# ---------------------------------------------------------------------------
# absorb_cross_rank_metadata
# ---------------------------------------------------------------------------


class TestAbsorbCrossRankMetadata:
    def test_stamps_owning_rank_from_payload_index(self) -> None:
        state = _make_state(rank=0, world_size=2)
        _add_seq(state, "u1", assigned_rank=1)  # owned by rank 1
        legacy = FakeLegacyBackend(rank=0, world_size=2)
        legacy._uuid_to_local = {}  # rank 0 does not own u1
        col = FakeCollectiveBackend(rank=0, world_size=2)
        bsync = BoundarySynchronizer(state, SyncCoordinator(state, col), col)

        rank1_state = SeqBoundaryState(
            decoded_length=7, current_context_length=17,
            gpu_pages_allocated=3, eos_reached=False, completed=False,
            additional_pages_needed=0, assigned_rank=1,
            needs_host_growth=False, host_growth_pages=0,
            host_pages_allocated=4, host_token_capacity=1024,
            prompt_length=10, total_decoded_before_eviction=0,
        )
        all_payloads: list[BoundaryPayload | None] = [
            BoundaryPayload(free_pages=5, seq_state={}, candidate_state={}),
            BoundaryPayload(free_pages=9, seq_state={"u1": rank1_state}),
        ]

        pruned, gss, gci, per_rank_free = bsync.absorb_cross_rank_metadata(
            ["u1"], all_payloads, legacy
        )
        assert pruned == ["u1"]
        assert gss["u1"].owning_rank == 1
        assert gci == {}
        assert per_rank_free == [5, 9]

    def test_copies_scalars_onto_non_owned_shadow_sequence(self) -> None:
        state = _make_state(rank=0, world_size=2)
        seq = _add_seq(state, "u1", assigned_rank=1, decoded_length=1)
        legacy = FakeLegacyBackend(rank=0, world_size=2)
        legacy._uuid_to_local = {}
        col = FakeCollectiveBackend(rank=0, world_size=2)
        bsync = BoundarySynchronizer(state, SyncCoordinator(state, col), col)

        rank1_state = SeqBoundaryState(
            decoded_length=42, current_context_length=52,
            gpu_pages_allocated=8, eos_reached=True, completed=False,
            additional_pages_needed=0, assigned_rank=1,
            needs_host_growth=False, host_growth_pages=0,
            host_pages_allocated=9, host_token_capacity=1024,
            prompt_length=10, total_decoded_before_eviction=3,
        )
        all_payloads = [
            BoundaryPayload(free_pages=0, seq_state={}),
            BoundaryPayload(free_pages=0, seq_state={"u1": rank1_state}),
        ]
        bsync.absorb_cross_rank_metadata(["u1"], all_payloads, legacy)

        assert seq.decoded_length == 42
        assert seq.current_context_length == 52  # (prompt_length=10 + decoded=42)
        assert seq.gpu_pages_allocated == 8
        assert seq.eos_reached is True
        assert seq.host_pages_allocated == 9
        assert seq.total_decoded_before_eviction == 3

    def test_orphans_uuid_missing_from_every_rank(self) -> None:
        """Legacy's missing_uuids path: sequence in decode_uuids but not
        reported by any rank → force COMPLETED + drop from decode list."""
        state = _make_state(rank=0, world_size=2)
        orphan = _add_seq(
            state, "orphan", assigned_rank=1,
            status=SequenceStatus.IN_DECODE,
            gpu_pages=5, host_pages=7,
        )
        _add_seq(state, "alive", assigned_rank=0)
        legacy = FakeLegacyBackend(rank=0, world_size=2)
        legacy._uuid_to_local = {"alive": 0}
        legacy._sequences_with_gpu_kv = {"orphan", "alive"}
        col = FakeCollectiveBackend(rank=0, world_size=2)
        bsync = BoundarySynchronizer(state, SyncCoordinator(state, col), col)

        alive_state = SeqBoundaryState(
            decoded_length=5, current_context_length=15,
            gpu_pages_allocated=2, eos_reached=False, completed=False,
            additional_pages_needed=0, assigned_rank=0,
            needs_host_growth=False, host_growth_pages=0,
            host_pages_allocated=4, host_token_capacity=1024,
            prompt_length=10, total_decoded_before_eviction=0,
        )
        all_payloads = [
            BoundaryPayload(free_pages=0, seq_state={"alive": alive_state}),
            BoundaryPayload(free_pages=0, seq_state={}),  # rank 1 didn't report orphan
        ]
        pruned, _, _, _ = bsync.absorb_cross_rank_metadata(
            ["orphan", "alive"], all_payloads, legacy
        )

        assert pruned == ["alive"]
        assert orphan.status == SequenceStatus.COMPLETED
        assert orphan.gpu_pages_allocated == 0
        assert orphan.host_pages_allocated == 0
        assert orphan.host_token_capacity == 0
        assert "orphan" not in legacy.sequences_with_gpu_kv()

    def test_orphan_with_invalid_transition_swallows_valueerror(self) -> None:
        """A QUEUEING orphan can't transition to COMPLETED (invalid per
        the state machine); absorption must swallow the ValueError so
        the overall boundary cycle keeps progressing."""
        state = _make_state(rank=0, world_size=1)
        orphan = _add_seq(state, "orphan", assigned_rank=0, status=SequenceStatus.QUEUEING)
        legacy = FakeLegacyBackend(rank=0, world_size=1)
        legacy._uuid_to_local = {}
        col = FakeCollectiveBackend(rank=0, world_size=1)
        bsync = BoundarySynchronizer(state, SyncCoordinator(state, col), col)

        pruned, _, _, _ = bsync.absorb_cross_rank_metadata(
            ["orphan"], [BoundaryPayload(free_pages=0)], legacy
        )
        assert pruned == []
        # Status stayed QUEUEING because the transition is invalid —
        # the orphan's scalar state is still zeroed out above.
        assert orphan.status == SequenceStatus.QUEUEING
        assert orphan.gpu_pages_allocated == 0

    def test_ctx_mismatch_is_repaired_silently(self) -> None:
        """Gathered ctx that doesn't match the formula is rewritten to
        ``original_prompt_length + decoded_length`` without raising."""
        state = _make_state(rank=0, world_size=2)
        seq = _add_seq(state, "u1", assigned_rank=1)
        legacy = FakeLegacyBackend(rank=0, world_size=2)
        legacy._uuid_to_local = {}
        col = FakeCollectiveBackend(rank=0, world_size=2)
        bsync = BoundarySynchronizer(state, SyncCoordinator(state, col), col)

        broken_state = SeqBoundaryState(
            decoded_length=5, current_context_length=999,  # wrong
            gpu_pages_allocated=1, eos_reached=False, completed=False,
            additional_pages_needed=0, assigned_rank=1,
            needs_host_growth=False, host_growth_pages=0,
            host_pages_allocated=1, host_token_capacity=256,
            prompt_length=10, total_decoded_before_eviction=0,
        )
        bsync.absorb_cross_rank_metadata(
            ["u1"],
            [
                BoundaryPayload(free_pages=0),
                BoundaryPayload(free_pages=0, seq_state={"u1": broken_state}),
            ],
            legacy,
        )
        assert seq.current_context_length == seq.original_prompt_length + 5

    def test_owned_sequence_not_overwritten(self) -> None:
        """The absorption skip for rank-owned uuids matches legacy
        line 6871 — this rank's authoritative copy stays untouched."""
        state = _make_state(rank=0, world_size=1)
        seq = _add_seq(state, "mine", assigned_rank=0, decoded_length=5)
        seq.decoded_length = 5
        seq.current_context_length = 15
        legacy = FakeLegacyBackend(rank=0, world_size=1)
        legacy._uuid_to_local = {"mine": 0}
        col = FakeCollectiveBackend(rank=0, world_size=1)
        bsync = BoundarySynchronizer(state, SyncCoordinator(state, col), col)

        peer_state = SeqBoundaryState(
            decoded_length=999, current_context_length=1009,
            gpu_pages_allocated=99, eos_reached=True, completed=True,
            additional_pages_needed=0, assigned_rank=0,
            needs_host_growth=False, host_growth_pages=0,
            host_pages_allocated=99, host_token_capacity=99,
            prompt_length=10, total_decoded_before_eviction=0,
        )
        bsync.absorb_cross_rank_metadata(
            ["mine"],
            [BoundaryPayload(free_pages=0, seq_state={"mine": peer_state})],
            legacy,
        )
        # rank-0 owns "mine"; its authoritative copy must NOT have been
        # stomped by the gathered payload.
        assert seq.decoded_length == 5
        assert seq.current_context_length == 15
