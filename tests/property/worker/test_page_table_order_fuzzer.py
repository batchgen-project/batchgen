"""Page-table-order fuzzer — plan invariant #4.

Invariant: the GPU page table rebuild at every boundary must iterate
sequences in ``sorted(uuids, key=lambda u: (seq.global_idx, seq.uuid))``
order on every rank. Main's decoding loop writes KV into page-table
slots indexed by that order; any deviation places a sequence's KV in
the wrong slot and downstream attention reads the wrong tokens (silent
data corruption — no crash, just garbage output).

In the re-extracted worker, the ordering is established by
:meth:`BoundaryPlanner._plan_per_seq_extension`, which sorts the
IN_DECODE candidate set by ``(global_idx, uuid)`` before iterating.
This fuzzer generates random snapshots with shuffled insertion order
and asserts that every emitted decision's uuid sequence respects the
same ordering — so every rank computes a byte-identical plan and the
executor applies mutations in the same order everywhere.
"""

from __future__ import annotations

from hypothesis import given, settings, strategies as st

from batchgen.sequence import SequenceStatus
from batchgen.worker.boundary.decisions import (
    ExtendPages,
    OnHold,
    ReleasePages,
    SeqMetadata,
)
from batchgen.worker.boundary.planner import BoundaryPlanner, PlannerConfig


PAGE = 64  # SequenceEntry.PAGE_SIZE


def _cfg() -> PlannerConfig:
    return PlannerConfig(
        prefill_watermark_pct=70,
        decision_frequency_pages=2,
        extension_gpu_page_buffer=4,
        host_total_pages=10000,
    )


def _seq_needing_extension(uuid: str, global_idx: int) -> SeqMetadata:
    """SeqMetadata with 2 pages and 1 token of headroom → needs extend."""
    return SeqMetadata(
        uuid=uuid,
        global_idx=global_idx,
        status=int(SequenceStatus.IN_DECODE),
        assigned_rank=0,
        prompt_length=127,
        max_decode_length=10000,
        decoded_length=0,
        current_context_length=127,  # = prompt + decoded
        gpu_pages_allocated=2,
        host_pages_allocated=0,
        had_initial_gpu_reservation=True,
        eos_reached=False,
        rep_detected=False,
    )


def _completed_seq(uuid: str, global_idx: int) -> SeqMetadata:
    return SeqMetadata(
        uuid=uuid,
        global_idx=global_idx,
        status=int(SequenceStatus.COMPLETED),
        assigned_rank=0,
        prompt_length=10,
        max_decode_length=100,
        decoded_length=50,
        current_context_length=60,
        gpu_pages_allocated=2,
        host_pages_allocated=0,
        had_initial_gpu_reservation=True,
        eos_reached=True,
        rep_detected=False,
    )


# ---------------------------------------------------------------------------
# ExtendPages — global_idx ascending
# ---------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    uuid_pool=st.lists(
        st.text(
            alphabet=st.characters(min_codepoint=97, max_codepoint=122),
            min_size=2,
            max_size=6,
        ),
        min_size=2,
        max_size=8,
        unique=True,
    )
)
def test_extend_decisions_are_in_global_idx_order(uuid_pool: list[str]) -> None:
    """Build a snapshot in shuffled insertion order; the planner must
    emit ExtendPages in ``(global_idx, uuid)`` ascending — the same
    order every rank will compute from its own snapshot."""
    # Assign random global_idx values (not necessarily matching insertion
    # order) so the fuzzer can break ties the planner has to resolve.
    snapshot: dict[str, SeqMetadata] = {}
    for i, uuid in enumerate(uuid_pool):
        # Use i as global_idx but in a scrambled order by iterating the
        # pool in reverse. Hypothesis already shuffles uuid_pool values.
        snapshot[uuid] = _seq_needing_extension(uuid, global_idx=len(uuid_pool) - i)

    planner = BoundaryPlanner(_cfg())
    plan = planner.plan(
        snapshot, gpu_free=1000, host_free=10000, has_pending=False
    )

    extends = plan.decisions_of(ExtendPages)
    emitted_uuids = [e.uuid for e in extends]

    expected = sorted(
        snapshot.values(), key=lambda m: (m.global_idx, m.uuid)
    )
    expected_uuids = [m.uuid for m in expected]

    assert emitted_uuids == expected_uuids, (
        f"ExtendPages order deviated from global_idx sort:\n"
        f"  emitted:  {emitted_uuids}\n"
        f"  expected: {expected_uuids}"
    )


# ---------------------------------------------------------------------------
# ReleasePages — alphabetical (cross-rank determinism without global_idx)
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(
    uuid_pool=st.lists(
        st.text(
            alphabet=st.characters(min_codepoint=97, max_codepoint=122),
            min_size=2,
            max_size=6,
        ),
        min_size=1,
        max_size=8,
        unique=True,
    )
)
def test_release_uuids_are_alphabetically_sorted(uuid_pool: list[str]) -> None:
    snapshot: dict[str, SeqMetadata] = {
        uuid: _completed_seq(uuid, global_idx=i) for i, uuid in enumerate(uuid_pool)
    }
    planner = BoundaryPlanner(_cfg())
    plan = planner.plan(
        snapshot, gpu_free=1000, host_free=1000, has_pending=False
    )
    releases = plan.decisions_of(ReleasePages)
    assert len(releases) == 1
    assert list(releases[0].uuids) == sorted(uuid_pool)


# ---------------------------------------------------------------------------
# OnHold (EXTENSION_FAILED) — shortest decoded first, ties by uuid
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    uuid_dl_pairs=st.lists(
        st.tuples(
            st.text(
                alphabet=st.characters(min_codepoint=97, max_codepoint=122),
                min_size=2,
                max_size=6,
            ),
            st.integers(min_value=0, max_value=1000),
        ),
        min_size=2,
        max_size=6,
        unique_by=lambda p: p[0],
    )
)
def test_held_onhold_uuids_sorted_shortest_decoded_first(
    uuid_dl_pairs: list[tuple[str, int]],
) -> None:
    """Every IN_DECODE seq needs extension; gpu_free=0 → all held.
    Held list must be sorted by (decoded_length, uuid) ascending."""
    snapshot: dict[str, SeqMetadata] = {}
    for i, (uuid, dl) in enumerate(uuid_dl_pairs):
        # Each seq needs extension: 2 pages, full ctx
        snapshot[uuid] = SeqMetadata(
            uuid=uuid,
            global_idx=i,
            status=int(SequenceStatus.IN_DECODE),
            assigned_rank=0,
            prompt_length=127 - dl,
            max_decode_length=10000,
            decoded_length=dl,
            current_context_length=127,
            gpu_pages_allocated=2,
            host_pages_allocated=0,
            had_initial_gpu_reservation=True,
            eos_reached=False,
            rep_detected=False,
        )

    planner = BoundaryPlanner(_cfg())
    plan = planner.plan(
        snapshot, gpu_free=0, host_free=10000, has_pending=False
    )
    onholds = plan.decisions_of(OnHold)
    assert len(onholds) == 1
    held = list(onholds[0].uuids)

    expected = [m.uuid for m in sorted(
        snapshot.values(), key=lambda m: (m.decoded_length, m.uuid)
    )]
    assert held == expected, (
        f"OnHold held order not shortest-decoded-first:\n"
        f"  held:     {held}\n"
        f"  expected: {expected}"
    )
