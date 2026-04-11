"""Unit tests for batchgen.worker.boundary.planner.BoundaryPlanner."""

from __future__ import annotations

import pytest

from batchgen.sequence import SequenceStatus
from batchgen.worker.boundary.decisions import (
    BoundaryPlan,
    ExtendPages,
    OnHold,
    OnHoldReason,
    ReleasePages,
    SeqMetadata,
)
from batchgen.worker.boundary.planner import BoundaryPlanner, PlannerConfig


PAGE = 64  # SequenceEntry.PAGE_SIZE


def _cfg(
    *,
    prefill_watermark_pct: int = 70,
    decision_frequency_pages: int = 2,
    extension_gpu_page_buffer: int = 4,
    host_total_pages: int = 1000,
) -> PlannerConfig:
    return PlannerConfig(
        prefill_watermark_pct=prefill_watermark_pct,
        decision_frequency_pages=decision_frequency_pages,
        extension_gpu_page_buffer=extension_gpu_page_buffer,
        host_total_pages=host_total_pages,
    )


def _meta(
    uuid: str,
    *,
    status: SequenceStatus = SequenceStatus.IN_DECODE,
    global_idx: int = 0,
    decoded_length: int = 0,
    current_context_length: int | None = None,
    gpu_pages_allocated: int = 0,
    prompt_length: int = 10,
    max_decode_length: int = 10000,
) -> SeqMetadata:
    return SeqMetadata(
        uuid=uuid,
        global_idx=global_idx,
        status=int(status),
        assigned_rank=0,
        prompt_length=prompt_length,
        max_decode_length=max_decode_length,
        decoded_length=decoded_length,
        current_context_length=current_context_length
        if current_context_length is not None
        else prompt_length + decoded_length,
        gpu_pages_allocated=gpu_pages_allocated,
        host_pages_allocated=0,
        had_initial_gpu_reservation=True,
        eos_reached=False,
        rep_detected=False,
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_decision_frequency_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="decision_frequency_pages"):
            BoundaryPlanner(_cfg(decision_frequency_pages=0))

    def test_host_total_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="host_total_pages"):
            BoundaryPlanner(_cfg(host_total_pages=0))


# ---------------------------------------------------------------------------
# Empty snapshot
# ---------------------------------------------------------------------------


class TestEmptySnapshot:
    def test_empty_snapshot_returns_empty_plan(self) -> None:
        p = BoundaryPlanner(_cfg()).plan(
            {}, gpu_free=100, host_free=100, has_pending=False
        )
        assert p == BoundaryPlan(decisions=(), metadata_snapshot={})


# ---------------------------------------------------------------------------
# Rule 1: releases
# ---------------------------------------------------------------------------


class TestReleases:
    def test_single_completed_emits_release(self) -> None:
        snap = {"u1": _meta("u1", status=SequenceStatus.COMPLETED)}
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=100, host_free=100, has_pending=False
        )
        releases = plan.decisions_of(ReleasePages)
        assert len(releases) == 1
        assert releases[0].uuids == ("u1",)

    def test_multiple_completed_sorted_alphabetically(self) -> None:
        snap = {
            "uC": _meta("uC", status=SequenceStatus.COMPLETED, global_idx=2),
            "uA": _meta("uA", status=SequenceStatus.COMPLETED, global_idx=0),
            "uB": _meta("uB", status=SequenceStatus.COMPLETED, global_idx=1),
        }
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=100, host_free=100, has_pending=False
        )
        releases = plan.decisions_of(ReleasePages)
        assert releases[0].uuids == ("uA", "uB", "uC")

    def test_no_completed_no_release(self) -> None:
        snap = {
            "u1": _meta("u1", status=SequenceStatus.IN_DECODE, gpu_pages_allocated=10),
        }
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=100, host_free=100, has_pending=False
        )
        assert plan.decisions_of(ReleasePages) == ()


# ---------------------------------------------------------------------------
# Rule 2: prefill watermark trigger
# ---------------------------------------------------------------------------


class TestPrefillWatermarkTrigger:
    def test_above_watermark_with_pending_onholds_all_in_decode(self) -> None:
        snap = {
            "u1": _meta(
                "u1",
                status=SequenceStatus.IN_DECODE,
                gpu_pages_allocated=32,
                decoded_length=50,
            ),
            "u2": _meta(
                "u2",
                status=SequenceStatus.IN_DECODE,
                gpu_pages_allocated=32,
                decoded_length=10,
                global_idx=1,
            ),
        }
        # 800/1000 = 80% > 70 watermark, has_pending=True -> bail to prefill
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=1000, host_free=800, has_pending=True
        )
        onholds = plan.decisions_of(OnHold)
        assert len(onholds) == 1
        assert onholds[0].reason is OnHoldReason.WATERMARK_TRIGGER
        assert set(onholds[0].uuids) == {"u1", "u2"}
        # No extend emitted after the bailout
        assert plan.decisions_of(ExtendPages) == ()

    def test_above_watermark_without_pending_no_trigger(self) -> None:
        snap = {
            "u1": _meta(
                "u1",
                status=SequenceStatus.IN_DECODE,
                gpu_pages_allocated=32,
                decoded_length=50,
            ),
        }
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=1000, host_free=800, has_pending=False
        )
        # No OnHold with reason WATERMARK_TRIGGER
        onholds = plan.decisions_of(OnHold)
        assert all(
            d.reason is not OnHoldReason.WATERMARK_TRIGGER for d in onholds
        )

    def test_at_watermark_not_triggered(self) -> None:
        """700/1000 = 70% == watermark; strict > not >=, so no bailout."""
        snap = {
            "u1": _meta(
                "u1",
                status=SequenceStatus.IN_DECODE,
                gpu_pages_allocated=32,
                decoded_length=50,
            ),
        }
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=1000, host_free=700, has_pending=True
        )
        assert not any(
            d.reason is OnHoldReason.WATERMARK_TRIGGER
            for d in plan.decisions_of(OnHold)
        )

    def test_release_emitted_even_when_watermark_fires(self) -> None:
        """Completed sequences still get released in the bailout path."""
        snap = {
            "u_done": _meta("u_done", status=SequenceStatus.COMPLETED),
            "u_decode": _meta(
                "u_decode",
                status=SequenceStatus.IN_DECODE,
                gpu_pages_allocated=32,
                decoded_length=5,
                global_idx=1,
            ),
        }
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=1000, host_free=800, has_pending=True
        )
        releases = plan.decisions_of(ReleasePages)
        assert len(releases) == 1 and releases[0].uuids == ("u_done",)
        onholds = plan.decisions_of(OnHold)
        assert len(onholds) == 1
        assert onholds[0].reason is OnHoldReason.WATERMARK_TRIGGER
        assert onholds[0].uuids == ("u_decode",)


# ---------------------------------------------------------------------------
# Rule 3: per-seq extension
# ---------------------------------------------------------------------------


def _seq_needing_extension(
    uuid: str, global_idx: int = 0, decoded_length: int = 0
) -> SeqMetadata:
    """Seq with 2 pages allocated but ctx_len within 1 token of overflow.
    headroom = 128 tokens - 127 = 1 < freq*PAGE = 2*64 = 128 → need more."""
    return _meta(
        uuid,
        status=SequenceStatus.IN_DECODE,
        global_idx=global_idx,
        decoded_length=decoded_length,
        prompt_length=10,
        current_context_length=127,  # < 2 pages (128 tokens)
        gpu_pages_allocated=2,
    )


def _seq_with_ample_room(uuid: str, global_idx: int = 0) -> SeqMetadata:
    """Seq with 32 pages and ctx_len small — plenty of headroom."""
    return _meta(
        uuid,
        status=SequenceStatus.IN_DECODE,
        global_idx=global_idx,
        prompt_length=10,
        current_context_length=100,
        gpu_pages_allocated=32,
    )


class TestPerSeqExtension:
    def test_sequence_with_headroom_not_extended(self) -> None:
        snap = {"u1": _seq_with_ample_room("u1")}
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=100, host_free=100, has_pending=False
        )
        assert plan.decisions_of(ExtendPages) == ()

    def test_sequence_needing_extension_with_room_gets_extend(self) -> None:
        snap = {"u1": _seq_needing_extension("u1")}
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=10, host_free=100, has_pending=False
        )
        extends = plan.decisions_of(ExtendPages)
        assert len(extends) == 1
        assert extends[0] == ExtendPages(uuid="u1", additional_pages=4)

    def test_extension_exhausts_gpu_remaining_held(self) -> None:
        """3 seqs need extension, gpu_free=6 → 4 pages per extend, only 1
        fits, the other 2 go on hold."""
        snap = {
            "uA": _seq_needing_extension("uA", global_idx=0, decoded_length=100),
            "uB": _seq_needing_extension("uB", global_idx=1, decoded_length=10),
            "uC": _seq_needing_extension("uC", global_idx=2, decoded_length=50),
        }
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=6, host_free=100, has_pending=False
        )
        extends = plan.decisions_of(ExtendPages)
        held = [d for d in plan.decisions_of(OnHold)
                if d.reason is OnHoldReason.EXTENSION_FAILED]
        # Order: uA (global_idx 0) gets the extend, uB + uC held.
        assert len(extends) == 1
        assert extends[0].uuid == "uA"
        assert len(held) == 1
        # Held list sorted shortest-decoded-first: uB (10) before uC (50).
        assert held[0].uuids == ("uB", "uC")

    def test_zero_gpu_free_holds_everyone(self) -> None:
        snap = {
            "u1": _seq_needing_extension("u1", global_idx=0, decoded_length=5),
            "u2": _seq_needing_extension("u2", global_idx=1, decoded_length=20),
        }
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=0, host_free=100, has_pending=False
        )
        assert plan.decisions_of(ExtendPages) == ()
        held = plan.decisions_of(OnHold)
        assert len(held) == 1
        assert held[0].reason is OnHoldReason.EXTENSION_FAILED
        assert held[0].uuids == ("u1", "u2")  # shortest (5) first

    def test_held_sorted_shortest_decoded_first(self) -> None:
        """Multiple held seqs are ordered by decoded_length ASC, ties by uuid."""
        snap = {
            "longer": _seq_needing_extension("longer", global_idx=2, decoded_length=100),
            "shortest": _seq_needing_extension("shortest", global_idx=0, decoded_length=5),
            "middle": _seq_needing_extension("middle", global_idx=1, decoded_length=50),
        }
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=0, host_free=100, has_pending=False
        )
        held = plan.decisions_of(OnHold)
        assert held[0].uuids == ("shortest", "middle", "longer")

    def test_only_in_decode_considered_for_extension(self) -> None:
        """PREFILLED / ON_HOLD / QUEUEING sequences never show up as
        extension candidates — only IN_DECODE."""
        snap = {
            "u_prefilled": _meta(
                "u_prefilled",
                status=SequenceStatus.PREFILLED,
                gpu_pages_allocated=2,
                current_context_length=127,
            ),
            "u_onhold": _meta(
                "u_onhold",
                status=SequenceStatus.ON_HOLD,
                gpu_pages_allocated=2,
                current_context_length=127,
                global_idx=1,
            ),
            "u_queue": _meta(
                "u_queue", status=SequenceStatus.QUEUEING, gpu_pages_allocated=0, global_idx=2
            ),
        }
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=100, host_free=100, has_pending=False
        )
        assert plan.decisions_of(ExtendPages) == ()
        assert plan.decisions_of(OnHold) == ()

    def test_extension_is_in_global_idx_order(self) -> None:
        """Multiple sequences in need, gpu has room for all — extends are
        emitted in ascending global_idx."""
        snap = {
            "uB": _seq_needing_extension("uB", global_idx=1),
            "uA": _seq_needing_extension("uA", global_idx=0),
            "uC": _seq_needing_extension("uC", global_idx=2),
        }
        plan = BoundaryPlanner(_cfg()).plan(
            snap, gpu_free=100, host_free=100, has_pending=False
        )
        extends = plan.decisions_of(ExtendPages)
        assert [e.uuid for e in extends] == ["uA", "uB", "uC"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_snapshot_produces_same_plan(self) -> None:
        snap = {
            "u1": _meta("u1", status=SequenceStatus.COMPLETED),
            "u2": _seq_needing_extension("u2", global_idx=1, decoded_length=10),
            "u3": _seq_with_ample_room("u3", global_idx=2),
        }
        p = BoundaryPlanner(_cfg())
        plan1 = p.plan(snap, gpu_free=100, host_free=100, has_pending=False)
        plan2 = p.plan(snap, gpu_free=100, host_free=100, has_pending=False)
        assert plan1 == plan2
