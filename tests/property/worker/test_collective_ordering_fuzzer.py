"""Collective ordering fuzzer — plan invariant #6.

Invariant #6: every ``torch.distributed.*`` call must be issued in the
same order on every rank. A single rank-dependent branch turns into a
deadlock the moment the process group waits for a collective that a
peer didn't send. The same failure can appear as non-determinism —
"works sometimes" because Python dict iteration happened to hash the
uuids the same way on both ranks this run.

This fuzzer pins the ordering through the only end-to-end path that
issues collectives in M5: :meth:`BoundaryHandler.run`. Every random
scenario is replayed multiple times against fresh BoundaryHandler
instances and the resulting :attr:`FakeCollectiveBackend.calls` lists
are compared. A single variation would immediately fail the test.

Scope notes:

  - The fuzzer tests **determinism under replay**, not literal
    multi-rank convergence, because ``FakeCollectiveBackend`` does
    not simulate peer communication. Determinism under replay is a
    strictly weaker property than cross-rank agreement but it is
    necessary: if the single-rank path is non-deterministic, the
    multi-rank path cannot be cross-rank deterministic.
  - :meth:`DecodeScheduler.run_continuous` ends in
    :meth:`BoundaryHandler.run`, so every collective
    DecodeScheduler issues is covered by the same fuzzer indirectly.
"""

from __future__ import annotations

import torch
from hypothesis import given, settings, strategies as st

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.boundary import (
    BoundaryExecutor,
    BoundaryGuards,
    BoundaryHandler,
    BoundaryPlanner,
    BoundarySynchronizer,
    PlannerConfig,
)
from batchgen.worker.host_rebalancer import HostKVRebalancer
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator
from tests.unit.worker.fakes import (
    FakeCollectiveBackend,
    FakeGpuKvBackend,
    FakeHostKvBackend,
)


PAGE = SequenceEntry.PAGE_SIZE


def _make_state() -> WorkerState:
    return WorkerState(
        rank=0,
        local_rank=0,
        world_size=1,
        device=0,
        torch_device=torch.device("cpu"),
    )


def _build_handler() -> tuple[BoundaryHandler, WorkerState, FakeCollectiveBackend]:
    state = _make_state()
    gpu = FakeGpuKvBackend(free_pages=2000)
    host = FakeHostKvBackend(free_pages=2000)
    col = FakeCollectiveBackend(rank=0, world_size=1)
    kv = KVCacheManager(
        state,
        gpu,
        host,
        initial_gpu_page_buffer=32,
        extension_gpu_page_buffer=4,
        host_kv_total_pages=10000,
        prefill_watermark_pct=70,
    )
    sync = SyncCoordinator(state, col)
    rb = HostKVRebalancer(state, kv, sync)
    synchronizer = BoundarySynchronizer(state, sync, col)
    planner = BoundaryPlanner(
        PlannerConfig(
            prefill_watermark_pct=70,
            decision_frequency_pages=2,
            extension_gpu_page_buffer=4,
            host_total_pages=10000,
        )
    )
    executor = BoundaryExecutor(state, kv, rb)
    guards = BoundaryGuards(state)
    return BoundaryHandler(state, synchronizer, planner, executor, guards, kv), state, col


def _install(
    state: WorkerState,
    seqs: list[tuple[str, int, int, int, int, str]],
) -> None:
    """Install a batch of sequences matching the tuple spec.

    Each tuple is (uuid, global_idx, prompt_length, decoded_length,
    gpu_pages_allocated, status_label). Status labels:
      "IN_DECODE", "PREFILLED", "ON_HOLD", "COMPLETED", "QUEUEING".
    """
    for uuid, gidx, pl, dl, gpu_pages, label in seqs:
        seq = SequenceEntry(
            uuid=uuid,
            global_idx=gidx,
            prompt_length=pl,
            max_decode_length=10000,
            text="",
        )
        seq.original_prompt_length = pl
        seq.decoded_length = dl
        seq.current_context_length = pl + dl
        seq.assigned_rank = 0
        seq.gpu_pages_allocated = gpu_pages
        state.global_batch.add_sequence(seq)

        if label == "QUEUEING":
            continue
        path_map = {
            "IN_PREFILL": [SequenceStatus.IN_PREFILL],
            "PREFILLED": [SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED],
            "IN_DECODE": [
                SequenceStatus.IN_PREFILL,
                SequenceStatus.PREFILLED,
                SequenceStatus.IN_DECODE,
            ],
            "ON_HOLD": [
                SequenceStatus.IN_PREFILL,
                SequenceStatus.PREFILLED,
                SequenceStatus.IN_DECODE,
                SequenceStatus.ON_HOLD,
            ],
            "COMPLETED": [
                SequenceStatus.IN_PREFILL,
                SequenceStatus.PREFILLED,
                SequenceStatus.IN_DECODE,
                SequenceStatus.COMPLETED,
            ],
        }
        for s in path_map[label]:
            state.global_batch.update_status(uuid, s)


# ---------------------------------------------------------------------------
# Determinism under replay
# ---------------------------------------------------------------------------


@settings(max_examples=40, deadline=None)
@given(
    scenario=st.lists(
        st.tuples(
            st.text(
                alphabet=st.characters(min_codepoint=97, max_codepoint=122),
                min_size=2,
                max_size=6,
            ),
            st.integers(min_value=0, max_value=10),  # global_idx
            st.integers(min_value=1, max_value=200),  # prompt_length
            st.integers(min_value=0, max_value=500),  # decoded_length
            st.integers(min_value=32, max_value=64),  # gpu_pages_allocated (ample)
            st.sampled_from(["IN_DECODE", "COMPLETED", "QUEUEING"]),
        ),
        min_size=1,
        max_size=5,
        unique_by=lambda t: t[0],  # unique uuids
    ),
)
def test_boundary_run_collective_order_is_deterministic(
    scenario: list[tuple[str, int, int, int, int, str]],
) -> None:
    """Running BoundaryHandler.run three times against an identical
    snapshot must record the same collective call names in the same
    order every time. Any dict-iteration-order or set-order leak would
    immediately differ across the three runs."""
    # Collect uuids that will actually survive to sync_metadata — only
    # sequences that are in the batch when the handler runs.
    uuids = sorted({t[0] for t in scenario})

    call_sequences: list[list[str]] = []
    for _ in range(3):
        handler, state, col = _build_handler()
        _install(state, scenario)
        handler.run(uuids)
        call_sequences.append(col.call_names())

    assert call_sequences[0] == call_sequences[1] == call_sequences[2], (
        "BoundaryHandler.run recorded different collective orders across "
        f"replays of the same scenario:\n  {call_sequences}"
    )


# ---------------------------------------------------------------------------
# Structural floor: every BoundaryHandler.run issues at least the
# sync_metadata + broadcast_plan collective pair.
# ---------------------------------------------------------------------------


@settings(max_examples=30, deadline=None)
@given(
    # Force every scenario to contain at least one IN_DECODE so the
    # plan is not trivially empty. OnHold paths add a trailing
    # all_gather_object from put_on_hold.sync_metadata.
    in_decode_count=st.integers(min_value=1, max_value=4),
)
def test_boundary_run_always_issues_sync_then_broadcast(in_decode_count: int) -> None:
    scenario = [
        (f"u{i}", i, 10, 0, 64, "IN_DECODE")
        for i in range(in_decode_count)
    ]
    uuids = [t[0] for t in scenario]
    handler, state, col = _build_handler()
    _install(state, scenario)

    handler.run(uuids)

    names = col.call_names()
    # First call must be sync_metadata_in → all_gather_object.
    assert names[0] == "all_gather_object"
    # Second must be broadcast_plan → broadcast_object.
    assert names[1] == "broadcast_object"
