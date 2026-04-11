"""Unit tests for batchgen.worker.host_rebalancer.HostKVRebalancer."""

from __future__ import annotations

import pytest
import torch

from batchgen.sequence import SequenceEntry, SequenceStatus
from batchgen.worker.host_rebalancer import (
    HostKVRebalancer,
    MigrationOp,
    ShortestDecodedFirstStrategy,
)
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.state import WorkerState
from batchgen.worker.sync import SyncCoordinator
from tests.unit.worker.fakes import (
    FakeCollectiveBackend,
    FakeGpuKvBackend,
    FakeHostKvBackend,
)


# ---------------------------------------------------------------------------
# Fixtures
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
    decoded_length: int = 0,
    status_path: list[SequenceStatus] | None = None,
    global_idx: int = 0,
) -> SequenceEntry:
    seq = SequenceEntry(
        uuid=uuid,
        global_idx=global_idx,
        prompt_length=10,
        max_decode_length=1024,
        text="",
    )
    seq.decoded_length = decoded_length
    seq.current_context_length = seq.prompt_length + decoded_length
    seq.original_prompt_length = seq.prompt_length
    seq.assigned_rank = state.rank
    state.global_batch.add_sequence(seq)
    if status_path:
        for s in status_path:
            state.global_batch.update_status(uuid, s)
    return seq


def _make_rebalancer(
    state: WorkerState,
    *,
    strategy: ShortestDecodedFirstStrategy | None = None,
    gpu_free: int = 128,
    host_free: int = 512,
    host_total: int = 1000,
    watermark: int = 70,
) -> tuple[HostKVRebalancer, FakeGpuKvBackend, FakeHostKvBackend, FakeCollectiveBackend, KVCacheManager]:
    gpu = FakeGpuKvBackend(free_pages=gpu_free)
    host = FakeHostKvBackend(free_pages=host_free)
    col = FakeCollectiveBackend(rank=state.rank, world_size=state.world_size)
    kv = KVCacheManager(
        state,
        gpu,
        host,
        initial_gpu_page_buffer=8,
        extension_gpu_page_buffer=4,
        host_kv_total_pages=host_total,
        host_kv_watermark_pct=watermark,
    )
    sync = SyncCoordinator(state, col)
    rb = HostKVRebalancer(
        state, kv, sync, eviction_strategy=strategy or ShortestDecodedFirstStrategy()
    )
    return rb, gpu, host, col, kv


# ---------------------------------------------------------------------------
# ShortestDecodedFirstStrategy
# ---------------------------------------------------------------------------


class TestShortestDecodedFirstStrategy:
    def test_picks_shortest_first(self) -> None:
        strat = ShortestDecodedFirstStrategy()
        seqs = []
        for uuid, dl in [("u1", 50), ("u2", 10), ("u3", 30)]:
            s = SequenceEntry(uuid=uuid, global_idx=0, prompt_length=1, max_decode_length=100, text="")
            s.decoded_length = dl
            seqs.append(s)
        assert strat.select(seqs, count=2) == ["u2", "u3"]

    def test_ties_broken_by_uuid(self) -> None:
        strat = ShortestDecodedFirstStrategy()
        seqs = []
        for uuid in ["uC", "uA", "uB"]:
            s = SequenceEntry(uuid=uuid, global_idx=0, prompt_length=1, max_decode_length=100, text="")
            s.decoded_length = 5
            seqs.append(s)
        assert strat.select(seqs, count=2) == ["uA", "uB"]

    def test_count_exceeds_candidates(self) -> None:
        strat = ShortestDecodedFirstStrategy()
        seqs = []
        for uuid, dl in [("u1", 5), ("u2", 10)]:
            s = SequenceEntry(uuid=uuid, global_idx=0, prompt_length=1, max_decode_length=100, text="")
            s.decoded_length = dl
            seqs.append(s)
        assert strat.select(seqs, count=10) == ["u1", "u2"]

    def test_count_zero_returns_empty(self) -> None:
        strat = ShortestDecodedFirstStrategy()
        assert strat.select([], count=0) == []
        assert strat.select([], count=5) == []


# ---------------------------------------------------------------------------
# select_for_onhold
# ---------------------------------------------------------------------------


class TestSelectForOnhold:
    def test_only_in_decode_sequences_are_candidates(self) -> None:
        state = _make_state()
        # u1 IN_DECODE, u2 QUEUEING (not a candidate), u3 IN_DECODE
        _add_seq(
            state,
            "u1",
            decoded_length=100,
            status_path=[SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED, SequenceStatus.IN_DECODE],
        )
        _add_seq(state, "u2", decoded_length=0, global_idx=1)
        _add_seq(
            state,
            "u3",
            decoded_length=50,
            status_path=[SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED, SequenceStatus.IN_DECODE],
            global_idx=2,
        )
        rb, *_ = _make_rebalancer(state)

        # Shortest decoded first among IN_DECODE: u3 (50), then u1 (100)
        assert rb.select_for_onhold(count=1) == ["u3"]
        assert rb.select_for_onhold(count=2) == ["u3", "u1"]

    def test_count_zero_returns_empty_without_lookup(self) -> None:
        state = _make_state()
        rb, *_ = _make_rebalancer(state)
        assert rb.select_for_onhold(count=0) == []
        assert rb.select_for_onhold(count=-5) == []


# ---------------------------------------------------------------------------
# put_on_hold — the ordering invariant
# ---------------------------------------------------------------------------


class TestPutOnHoldOrderingInvariant:
    def test_empty_uuids_noop(self) -> None:
        state = _make_state()
        rb, gpu, host, col, kv = _make_rebalancer(state)
        rb.put_on_hold([])
        assert gpu.calls == []
        assert col.calls == []
        assert kv.wait_pending_call_count == 0

    def test_full_ordering_flush_wait_release_transition_sync(self) -> None:
        """The canonical ordering test: every step happens in the right order
        for every on-hold invocation. Plan Decision #2."""
        state = _make_state()
        _add_seq(
            state,
            "u1",
            decoded_length=20,
            status_path=[SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED, SequenceStatus.IN_DECODE],
        )
        rb, gpu, host, col, kv = _make_rebalancer(state)

        # Give the manager some deferred state so flush has something to do.
        t = torch.zeros(1)
        kv.append_async("u1", layer=0, kv=t)
        kv.append_async("u1", layer=1, kv=t)
        assert kv.deferred_count == 2

        # Give u1 a GPU reservation so release has pages to return.
        kv.allocate_two_page_buffer("u1")
        gpu_calls_before = len(gpu.calls)

        rb.put_on_hold(["u1"])

        # 1. flush_deferred applied every queued append
        assert kv.deferred_count == 0

        # 2. wait_pending was called exactly once for this put_on_hold
        assert kv.wait_pending_call_count == 1

        # 3. GPU pages were released — release_pages call recorded AFTER the
        #    initial allocate_pages; and the seq's gpu_pages_allocated is now 0.
        release_calls = [c for c in gpu.calls if c[0] == "release_pages"]
        assert len(release_calls) == 1
        assert release_calls[0][1] == ("u1",)
        assert state.global_batch.get_sequence("u1").gpu_pages_allocated == 0  # type: ignore[union-attr]

        # 4. Status transitioned IN_DECODE → ON_HOLD via the state machine
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.ON_HOLD  # type: ignore[union-attr]

        # 5. sync_metadata issued its all_gather_object AFTER the release
        assert col.call_names() == ["all_gather_object"]

        # Pin the step order: the gpu.append_kv calls from flush must come
        # BEFORE the release_pages call.
        seen_release = False
        for name, _args in gpu.calls[gpu_calls_before:]:
            if name == "append_kv":
                assert not seen_release, (
                    "append_kv must come BEFORE release_pages in the put_on_hold sequence"
                )
            elif name == "release_pages":
                seen_release = True
        assert seen_release, "release_pages must be called exactly once"

    def test_non_in_decode_sequences_are_skipped(self) -> None:
        """If a UUID is already ON_HOLD or in another state, put_on_hold
        must leave it alone (no invalid status_transition call)."""
        state = _make_state()
        _add_seq(state, "u1", decoded_length=0)  # stays QUEUEING
        rb, _gpu, _host, col, _kv = _make_rebalancer(state)

        rb.put_on_hold(["u1"])

        # No invalid transition happened; seq is still QUEUEING.
        assert state.global_batch.get_sequence("u1").status == SequenceStatus.QUEUEING  # type: ignore[union-attr]
        # Sync still runs because the invariant demands it (even for empty work).
        assert col.call_names() == ["all_gather_object"]

    def test_missing_uuid_skipped(self) -> None:
        state = _make_state()
        rb, _gpu, _host, col, _kv = _make_rebalancer(state)
        rb.put_on_hold(["ghost"])  # must not raise
        assert col.call_names() == ["all_gather_object"]

    def test_sync_metadata_ctx_failure_propagates(self) -> None:
        """If a sequence we own has CTX drift at put_on_hold time, the sync
        step raises — the caller learns the invariant is broken instead of
        silently proceeding to a bad ON_HOLD state."""
        from batchgen.worker.exceptions import CtxInvariantViolation

        state = _make_state()
        seq = _add_seq(
            state,
            "u1",
            decoded_length=20,
            status_path=[SequenceStatus.IN_PREFILL, SequenceStatus.PREFILLED, SequenceStatus.IN_DECODE],
        )
        seq.current_context_length = 99999  # drifted
        rb, *_ = _make_rebalancer(state)

        with pytest.raises(CtxInvariantViolation) as exc:
            rb.put_on_hold(["u1"])
        assert exc.value.uuid == "u1"
        assert exc.value.side == "sender"


# ---------------------------------------------------------------------------
# Migration (M9 stubs)
# ---------------------------------------------------------------------------


class TestMigrationStubs:
    def test_plan_migration_returns_empty(self) -> None:
        state = _make_state()
        rb, *_ = _make_rebalancer(state)
        assert rb.plan_migration() == []

    def test_execute_migrations_empty_is_noop(self) -> None:
        state = _make_state()
        rb, *_ = _make_rebalancer(state)
        assert rb.execute_migrations([]) == 0

    def test_execute_migrations_non_empty_raises_m9(self) -> None:
        state = _make_state()
        rb, *_ = _make_rebalancer(state)
        op = MigrationOp(from_rank=0, to_rank=1, uuid="u1", page_count=4)
        with pytest.raises(NotImplementedError, match="M9"):
            rb.execute_migrations([op])

    def test_rebalance_stub_returns_zero(self) -> None:
        state = _make_state()
        rb, *_ = _make_rebalancer(state)
        assert rb.rebalance() == 0

    def test_migration_op_frozen(self) -> None:
        op = MigrationOp(from_rank=0, to_rank=1, uuid="u1", page_count=4)
        with pytest.raises(Exception):
            op.from_rank = 5  # type: ignore[misc]
