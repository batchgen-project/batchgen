"""Unit tests for the Stage 2.8.1b BoundaryPlan schema extensions.

Covers the new decision types (:class:`HostGrow`, :class:`HostEvict`,
:class:`NewLoadAsync`), the new ``watermark_break`` /
``decode_uuids_final`` fields on :class:`BoundaryPlan`, and the new
:class:`BoundaryResult` return dataclass.

Existing `test_decisions.py` still covers the base decision types; the
rules those test stay intact since the schema change was purely
additive (see docs/phase_2.8_stage1_design.md §2 Option A).
"""

from __future__ import annotations

import pytest

from batchgen.worker.boundary.decisions import (
    BoundaryPlan,
    BoundaryResult,
    Evict,
    EvictReason,
    HostEvict,
    HostGrow,
    NewLoadAsync,
    OnHold,
    OnHoldReason,
    PageBoundaryDecision,
    ReleasePages,
)


# ---------------------------------------------------------------------------
# New decision types
# ---------------------------------------------------------------------------


class TestHostGrow:
    def test_fields(self) -> None:
        g = HostGrow(uuids=("u1", "u2"), pages=(3, 5), feasible=True)
        assert g.uuids == ("u1", "u2")
        assert g.pages == (3, 5)
        assert g.feasible is True

    def test_frozen(self) -> None:
        g = HostGrow(uuids=("u1",), pages=(1,), feasible=True)
        with pytest.raises(Exception):
            g.feasible = False  # type: ignore[misc]

    def test_infeasible_growth_is_a_valid_decision(self) -> None:
        """The planner still emits the decision when feasible=False —
        the executor decides to skip the actual grow based on the flag
        (matches legacy batchgen_worker.py:6974)."""
        g = HostGrow(uuids=("u1",), pages=(9999,), feasible=False)
        assert g.feasible is False


class TestHostEvict:
    def test_fields(self) -> None:
        e = HostEvict(uuids=("u1", "u2"))
        assert e.uuids == ("u1", "u2")

    def test_frozen(self) -> None:
        e = HostEvict(uuids=("u1",))
        with pytest.raises(Exception):
            e.uuids = ()  # type: ignore[misc]


class TestNewLoadAsync:
    def test_fields(self) -> None:
        n = NewLoadAsync(uuids=("u1",), rank_pages=((0, 4), (1, 2)))
        assert n.uuids == ("u1",)
        assert n.rank_pages == ((0, 4), (1, 2))

    def test_default_empty_rank_pages(self) -> None:
        n = NewLoadAsync(uuids=("u1",))
        assert n.rank_pages == ()


# ---------------------------------------------------------------------------
# PageBoundaryDecision union widened
# ---------------------------------------------------------------------------


class TestUnionMembership:
    def test_new_types_are_decisions(self) -> None:
        decisions: list[PageBoundaryDecision] = [
            HostGrow(uuids=("u1",), pages=(1,), feasible=True),
            HostEvict(uuids=("u2",)),
            NewLoadAsync(uuids=("u3",)),
        ]
        assert isinstance(decisions[0], HostGrow)
        assert isinstance(decisions[1], HostEvict)
        assert isinstance(decisions[2], NewLoadAsync)


# ---------------------------------------------------------------------------
# BoundaryPlan new fields
# ---------------------------------------------------------------------------


class TestBoundaryPlanExtensions:
    def test_watermark_break_default_false(self) -> None:
        assert BoundaryPlan().watermark_break is False

    def test_watermark_break_roundtrip(self) -> None:
        p = BoundaryPlan(watermark_break=True)
        assert p.watermark_break is True

    def test_decode_uuids_final_default_empty(self) -> None:
        assert BoundaryPlan().decode_uuids_final == ()

    def test_decode_uuids_final_roundtrip(self) -> None:
        p = BoundaryPlan(decode_uuids_final=("u1", "u2"))
        assert p.decode_uuids_final == ("u1", "u2")

    def test_decisions_of_filters_new_types(self) -> None:
        p = BoundaryPlan(
            decisions=(
                ReleasePages(uuids=("u1",)),
                HostGrow(uuids=("u2",), pages=(2,), feasible=True),
                HostEvict(uuids=("u3",)),
                NewLoadAsync(uuids=("u4",)),
                Evict(uuids=("u5",), reason=EvictReason.HOST_KV_WATERMARK),
                OnHold(uuids=("u6",), reason=OnHoldReason.WATERMARK_TRIGGER),
            )
        )
        assert len(p.decisions_of(HostGrow)) == 1
        assert len(p.decisions_of(HostEvict)) == 1
        assert len(p.decisions_of(NewLoadAsync)) == 1
        # The original filters still work.
        assert len(p.decisions_of(ReleasePages)) == 1
        assert len(p.decisions_of(Evict)) == 1
        assert len(p.decisions_of(OnHold)) == 1


# ---------------------------------------------------------------------------
# BoundaryResult
# ---------------------------------------------------------------------------


class TestBoundaryResult:
    def test_defaults_match_empty_run(self) -> None:
        p = BoundaryPlan()
        r = BoundaryResult(plan=p)
        assert r.plan is p
        assert r.decode_uuids == ()
        assert r.batch == ()
        assert r.new_async_task is None
        assert r.new_load_uuids == ()
        assert r.new_load_local == ()
        assert r.new_load_global == ()
        assert r.watermark_triggered is False

    def test_full_populated_result(self) -> None:
        p = BoundaryPlan()
        handle = object()  # opaque async handle
        r = BoundaryResult(
            plan=p,
            decode_uuids=("u1",),
            batch=(0,),
            new_async_task=handle,
            new_load_uuids=("u2",),
            new_load_local=(1,),
            new_load_global=(17,),
            watermark_triggered=True,
        )
        assert r.new_async_task is handle
        assert r.watermark_triggered is True
        assert r.new_load_uuids == ("u2",)

    def test_frozen(self) -> None:
        r = BoundaryResult(plan=BoundaryPlan())
        with pytest.raises(Exception):
            r.watermark_triggered = True  # type: ignore[misc]
