"""CTX invariant fuzzer for batchgen.worker.sync.SyncCoordinator.

Two-part property-based test that locks down the plan's single most
load-bearing invariant:

    seq.current_context_length == seq.original_prompt_length + seq.decoded_length

Part A (held-invariant): with every sequence on every rank satisfying the
invariant, `sync_metadata` never raises, every recorded collective is in
lockstep order, and the absorbed state on non-owned sequences matches the
authoritative sender values.

Part B (drift-injected): with drift injected on a random rank/field/value,
`sync_metadata` raises `CtxInvariantViolation` with the correct
(uuid, side, had, expected) fields every time. Drift can land on the
current rank's OWN sequences (→ sender-side raise) or on a received
payload (→ receiver-side raise).

These are the shapes the old scheduler-split bug tail kept silently
violating; the fuzzer is the tripwire.
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings, strategies as st

from batchgen.sequence import SequenceEntry
from batchgen.worker.exceptions import CtxInvariantViolation
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SeqSnapshot, SyncCoordinator
from tests.unit.worker.fakes import FakeCollectiveBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(rank: int, world_size: int) -> WorkerState:
    return WorkerState(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=rank,
        torch_device=torch.device("cpu"),
    )


def _install_seq(
    state: WorkerState,
    uuid: str,
    *,
    owning_rank: int,
    original_prompt_length: int,
    decoded_length: int,
    global_idx: int,
) -> SequenceEntry:
    """Create a SequenceEntry with the CTX invariant satisfied."""
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=original_prompt_length,
        max_decode_length=1024,
        text="",
    )
    seq.original_prompt_length = original_prompt_length
    seq.decoded_length = decoded_length
    seq.current_context_length = original_prompt_length + decoded_length
    seq.assigned_rank = owning_rank
    state.global_batch.add_sequence(seq)
    return seq


def _snapshot(
    uuid: str,
    *,
    original_prompt_length: int,
    decoded_length: int,
    current_context_length: int | None = None,
) -> SeqSnapshot:
    """Build a SeqSnapshot. `current_context_length` defaults to the invariant
    value; pass explicitly to inject drift on the peer side."""
    return SeqSnapshot(
        uuid=uuid,
        prompt_length=original_prompt_length,
        original_prompt_length=original_prompt_length,
        decoded_length=decoded_length,
        current_context_length=(
            current_context_length
            if current_context_length is not None
            else original_prompt_length + decoded_length
        ),
        gpu_pages_allocated=0,
        host_pages_allocated=0,
        eos_reached=False,
    )


# ---------------------------------------------------------------------------
# Part A — held invariant, sync_metadata never raises
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    world_size=st.integers(min_value=1, max_value=4),
    seqs=st.lists(
        st.tuples(
            st.integers(min_value=0, max_value=3),    # owning rank (clamped later)
            st.integers(min_value=1, max_value=200),  # original_prompt_length
            st.integers(min_value=0, max_value=500),  # decoded_length
        ),
        min_size=1,
        max_size=6,
    ),
)
def test_held_invariant_never_raises_and_absorbs_cleanly(
    world_size: int, seqs: list[tuple[int, int, int]]
) -> None:
    state = _make_state(rank=0, world_size=world_size)
    # Clamp owning ranks to [0, world_size) and install sequences
    specs: list[tuple[str, int, int, int]] = []  # (uuid, owner, opl, dl)
    for i, (rank, opl, dl) in enumerate(seqs):
        owner = rank % world_size
        uuid = f"u{i}"
        _install_seq(
            state,
            uuid,
            owning_rank=owner,
            original_prompt_length=opl,
            decoded_length=dl,
            global_idx=i,
        )
        specs.append((uuid, owner, opl, dl))

    # Build authoritative per-rank payloads for the fake all_gather_object
    per_rank: list[dict[str, SeqSnapshot] | None] = [None] * world_size
    for uuid, owner, opl, dl in specs:
        if per_rank[owner] is None:
            per_rank[owner] = {}
        per_rank[owner][uuid] = _snapshot(uuid, original_prompt_length=opl, decoded_length=dl)  # type: ignore[index]
    col = FakeCollectiveBackend(
        rank=0,
        world_size=world_size,
        all_gather_object_responses=[per_rank],
    )

    # Act — must not raise
    SyncCoordinator(state, col).sync_metadata([s[0] for s in specs])

    # Exactly one all_gather_object issued
    assert col.call_names() == ["all_gather_object"]

    # Every shadow copy of a non-owned sequence must match the authoritative value
    for uuid, owner, opl, dl in specs:
        seq = state.global_batch.get_sequence(uuid)
        assert seq is not None
        if owner == 0:
            continue
        assert seq.decoded_length == dl
        assert seq.original_prompt_length == opl
        assert seq.current_context_length == opl + dl


# ---------------------------------------------------------------------------
# Part B — injected drift raises with the right shape
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(
    original_prompt_length=st.integers(min_value=1, max_value=500),
    decoded_length=st.integers(min_value=0, max_value=500),
    drift=st.integers(min_value=-50, max_value=50).filter(lambda x: x != 0),
)
def test_sender_side_drift_always_raises(
    original_prompt_length: int, decoded_length: int, drift: int
) -> None:
    """Rank 0 owns u0 with a drifted current_context_length; must raise
    sender-side before any collective is issued."""
    state = _make_state(rank=0, world_size=2)
    seq = _install_seq(
        state,
        "u0",
        owning_rank=0,
        original_prompt_length=original_prompt_length,
        decoded_length=decoded_length,
        global_idx=0,
    )
    expected = seq.current_context_length  # invariant value
    # Apply drift, guarding against negative (invariant #9 enforces >= 0 in spirit)
    drifted = max(0, expected + drift)
    if drifted == expected:
        pytest.skip("drift cancelled out")
    seq.current_context_length = drifted

    col = FakeCollectiveBackend(rank=0, world_size=2)
    sc = SyncCoordinator(state, col)

    with pytest.raises(CtxInvariantViolation) as exc:
        sc.sync_metadata(["u0"])

    assert exc.value.uuid == "u0"
    assert exc.value.side == "sender"
    assert exc.value.had == drifted
    assert exc.value.expected == expected
    # Zero collectives issued — the guard tripped BEFORE the gather.
    assert col.calls == []


@settings(max_examples=50, deadline=None)
@given(
    opl=st.integers(min_value=1, max_value=500),
    dl=st.integers(min_value=0, max_value=500),
    drift=st.integers(min_value=-50, max_value=50).filter(lambda x: x != 0),
)
def test_receiver_side_drift_always_raises(
    opl: int, dl: int, drift: int
) -> None:
    """Rank 0 is a shadow for u1 (owned by rank 1). Inject a drifted snapshot
    into the gathered payload — must raise receiver-side AFTER the gather."""
    state = _make_state(rank=0, world_size=2)
    # Rank 0 owns u0 (consistent); u1 is rank 1's
    _install_seq(
        state, "u0", owning_rank=0, original_prompt_length=10, decoded_length=3, global_idx=0
    )
    _install_seq(
        state,
        "u1",
        owning_rank=1,
        original_prompt_length=opl,
        decoded_length=dl,
        global_idx=1,
    )
    clean = opl + dl
    drifted = max(0, clean + drift)
    if drifted == clean:
        pytest.skip("drift cancelled out")

    rank1_payload = {
        "u1": _snapshot(
            "u1",
            original_prompt_length=opl,
            decoded_length=dl,
            current_context_length=drifted,
        )
    }
    col = FakeCollectiveBackend(
        rank=0,
        world_size=2,
        all_gather_object_responses=[[None, rank1_payload]],
    )

    with pytest.raises(CtxInvariantViolation) as exc:
        SyncCoordinator(state, col).sync_metadata(["u0"])

    assert exc.value.uuid == "u1"
    assert exc.value.side == "receiver"
    assert exc.value.had == drifted
    assert exc.value.expected == clean
    # Gather DID happen — the guard tripped on the received payload.
    assert col.call_names() == ["all_gather_object"]
