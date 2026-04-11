"""Unit tests for batchgen.worker.boundary.decisions."""

from __future__ import annotations

import pytest

from batchgen.worker.boundary.decisions import (
    AsyncLoadHostToGpu,
    BoundaryPlan,
    Evict,
    EvictReason,
    ExtendPages,
    OnHold,
    OnHoldReason,
    PageBoundaryDecision,
    ReleasePages,
    SeqMetadata,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:
    def test_evict_reasons(self) -> None:
        values = {m.name for m in EvictReason}
        assert values == {
            "HOST_KV_WATERMARK",
            "PREEMPT_FOR_QUEUED_PREFILL",
            "CONTEXT_OVERFLOW",
        }

    def test_on_hold_reasons(self) -> None:
        values = {m.name for m in OnHoldReason}
        assert values == {"EXTENSION_FAILED", "WATERMARK_TRIGGER"}


# ---------------------------------------------------------------------------
# SeqMetadata
# ---------------------------------------------------------------------------


def _meta(
    uuid: str = "u1",
    *,
    decoded_length: int = 0,
    max_decode_length: int = 100,
) -> SeqMetadata:
    return SeqMetadata(
        uuid=uuid,
        global_idx=0,
        status=3,  # IN_DECODE
        assigned_rank=0,
        prompt_length=10,
        max_decode_length=max_decode_length,
        decoded_length=decoded_length,
        current_context_length=10 + decoded_length,
        gpu_pages_allocated=8,
        host_pages_allocated=16,
        had_initial_gpu_reservation=True,
        eos_reached=False,
        rep_detected=False,
    )


class TestSeqMetadata:
    def test_is_completed_by_length_true_when_reached(self) -> None:
        m = _meta(decoded_length=100, max_decode_length=100)
        assert m.is_completed_by_length is True

    def test_is_completed_by_length_false_when_below(self) -> None:
        m = _meta(decoded_length=99, max_decode_length=100)
        assert m.is_completed_by_length is False

    def test_frozen(self) -> None:
        m = _meta()
        with pytest.raises(Exception):
            m.uuid = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = _meta("u1")
        b = _meta("u1")
        assert a == b


# ---------------------------------------------------------------------------
# Decision dataclasses
# ---------------------------------------------------------------------------


class TestDecisionDataclasses:
    def test_release_pages_equality_and_frozen(self) -> None:
        a = ReleasePages(uuids=("u1", "u2"))
        b = ReleasePages(uuids=("u1", "u2"))
        assert a == b
        assert a != ReleasePages(uuids=("u1",))
        with pytest.raises(Exception):
            a.uuids = ("u3",)  # type: ignore[misc]

    def test_evict_carries_reason(self) -> None:
        e = Evict(uuids=("u1",), reason=EvictReason.HOST_KV_WATERMARK)
        assert e.reason is EvictReason.HOST_KV_WATERMARK
        assert e != Evict(uuids=("u1",), reason=EvictReason.CONTEXT_OVERFLOW)

    def test_on_hold_carries_reason(self) -> None:
        o = OnHold(uuids=("u1",), reason=OnHoldReason.EXTENSION_FAILED)
        assert o.reason is OnHoldReason.EXTENSION_FAILED

    def test_extend_pages_fields(self) -> None:
        e = ExtendPages(uuid="u1", additional_pages=4)
        assert e.uuid == "u1"
        assert e.additional_pages == 4

    def test_async_load_carries_page_ids(self) -> None:
        a = AsyncLoadHostToGpu(uuid="u1", host_pages=(10, 11, 12))
        assert a.host_pages == (10, 11, 12)

    def test_union_isinstance(self) -> None:
        decisions: list[PageBoundaryDecision] = [
            ReleasePages(uuids=("u1",)),
            Evict(uuids=("u2",), reason=EvictReason.HOST_KV_WATERMARK),
            OnHold(uuids=("u3",), reason=OnHoldReason.WATERMARK_TRIGGER),
            ExtendPages(uuid="u4", additional_pages=2),
            AsyncLoadHostToGpu(uuid="u5", host_pages=(1, 2)),
        ]
        # Each is its own concrete type
        assert isinstance(decisions[0], ReleasePages)
        assert isinstance(decisions[1], Evict)
        assert isinstance(decisions[2], OnHold)
        assert isinstance(decisions[3], ExtendPages)
        assert isinstance(decisions[4], AsyncLoadHostToGpu)


# ---------------------------------------------------------------------------
# BoundaryPlan
# ---------------------------------------------------------------------------


class TestBoundaryPlan:
    def test_default_empty_plan(self) -> None:
        p = BoundaryPlan()
        assert p.decisions == ()
        assert p.metadata_snapshot == {}

    def test_frozen(self) -> None:
        p = BoundaryPlan()
        with pytest.raises(Exception):
            p.decisions = (ReleasePages(uuids=("u1",)),)  # type: ignore[misc]

    def test_decisions_of_filters_by_type(self) -> None:
        p = BoundaryPlan(
            decisions=(
                ReleasePages(uuids=("u1",)),
                Evict(uuids=("u2",), reason=EvictReason.HOST_KV_WATERMARK),
                ExtendPages(uuid="u3", additional_pages=1),
                Evict(uuids=("u4",), reason=EvictReason.CONTEXT_OVERFLOW),
            )
        )
        evicts = p.decisions_of(Evict)
        assert len(evicts) == 2
        assert all(isinstance(d, Evict) for d in evicts)
        assert p.decisions_of(OnHold) == ()
        assert len(p.decisions_of(ReleasePages)) == 1

    def test_metadata_snapshot_roundtrip(self) -> None:
        snap = {"u1": _meta("u1"), "u2": _meta("u2")}
        p = BoundaryPlan(decisions=(), metadata_snapshot=snap)
        assert p.metadata_snapshot["u1"].uuid == "u1"
        assert p.metadata_snapshot["u2"].uuid == "u2"
