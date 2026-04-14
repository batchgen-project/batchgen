"""Unit tests for BoundaryPlanner.plan_full — the Phase 2.8.1d port.

Covers the six rules ported from legacy ``_compute_boundary_decisions``
(batchgen_worker.py:6443-6629) plus the watermark-trigger bailout and
the IN_DECODE-only OnHold filter (the L4 root-cause fix).

Existing ``test_planner.py`` still exercises the M4 ``plan()`` method
— those tests stay intact; ``plan_full`` is a new method on the same
planner.
"""

from __future__ import annotations

from batchgen.worker.boundary.decisions import (
    ExtendPages,
    HostEvict,
    HostGrow,
    NewLoadAsync,
    OnHold,
    OnHoldReason,
    ReleasePages,
)
from batchgen.worker.boundary.planner import (
    BoundaryPlanner,
    PlannerConfig,
    WorkerViewStats,
)
from batchgen.worker.boundary.synchronizer import (
    LoadCandidateState,
    SeqBoundaryState,
)


PAGE = 64


def _cfg(**overrides) -> PlannerConfig:
    base = dict(
        prefill_watermark_pct=70,
        decision_frequency_pages=2,
        extension_gpu_page_buffer=4,
        host_total_pages=1000,
    )
    base.update(overrides)
    return PlannerConfig(**base)


def _state(
    *,
    decoded_length: int = 5,
    gpu_pages_allocated: int = 2,
    completed: bool = False,
    additional_pages_needed: int = 0,
    assigned_rank: int = 0,
    needs_host_growth: bool = False,
    host_growth_pages: int = 0,
    host_pages_allocated: int = 4,
    host_token_capacity: int = 256,
    owning_rank: int = 0,
    eos_reached: bool = False,
    prompt_length: int = 10,
    total_decoded_before_eviction: int = 0,
) -> SeqBoundaryState:
    return SeqBoundaryState(
        decoded_length=decoded_length,
        current_context_length=prompt_length + decoded_length,
        gpu_pages_allocated=gpu_pages_allocated,
        eos_reached=eos_reached,
        completed=completed,
        additional_pages_needed=additional_pages_needed,
        assigned_rank=assigned_rank,
        needs_host_growth=needs_host_growth,
        host_growth_pages=host_growth_pages,
        host_pages_allocated=host_pages_allocated,
        host_token_capacity=host_token_capacity,
        prompt_length=prompt_length,
        total_decoded_before_eviction=total_decoded_before_eviction,
        owning_rank=owning_rank,
    )


def _plan_full(planner: BoundaryPlanner, **kwargs):
    """Convenience wrapper — fills in sensible defaults."""
    defaults = dict(
        decode_uuids=[],
        global_seq_state={},
        global_candidate_info={},
        per_rank_free=[100],
        chunk_size=PAGE,
        worker_view_stats=WorkerViewStats(num_total_pages=1000, num_free_pages=800),
        has_pending=False,
        world_size=1,
        enable_host_kv_eviction=False,
        host_kv_eviction_watermark=20,
    )
    defaults.update(kwargs)
    return planner.plan_full(**defaults)


# ---------------------------------------------------------------------------
# Rule 1: completed split + Release
# ---------------------------------------------------------------------------


class TestRule1Completed:
    def test_completed_emits_release_and_drops_from_active(self) -> None:
        states = {
            "done": _state(completed=True),
            "live": _state(),
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["done", "live"],
            global_seq_state=states,
        )
        releases = plan.decisions_of(ReleasePages)
        assert len(releases) == 1
        assert releases[0].uuids == ("done",)
        assert plan.decode_uuids_final == ("live",)

    def test_all_completed_leaves_empty_active(self) -> None:
        states = {"a": _state(completed=True), "b": _state(completed=True)}
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["a", "b"],
            global_seq_state=states,
        )
        assert plan.decode_uuids_final == ()


# ---------------------------------------------------------------------------
# Watermark bailout (the L4 fix)
# ---------------------------------------------------------------------------


class TestWatermarkBailout:
    def test_fires_with_pending_and_high_watermark(self) -> None:
        states = {
            "in_decode": _state(gpu_pages_allocated=2),
            "queued": _state(gpu_pages_allocated=0),  # has no GPU pages
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg(prefill_watermark_pct=70)),
            decode_uuids=["in_decode", "queued"],
            global_seq_state=states,
            has_pending=True,
            worker_view_stats=WorkerViewStats(num_total_pages=100, num_free_pages=90),
        )
        onholds = plan.decisions_of(OnHold)
        assert len(onholds) == 1
        assert onholds[0].reason is OnHoldReason.WATERMARK_TRIGGER
        # Only the IN_DECODE uuid lands in OnHold — the queued uuid had
        # gpu_pages_allocated==0 so it's filtered out (L4 fix).
        assert onholds[0].uuids == ("in_decode",)
        assert plan.watermark_break is True
        # No extend / HostGrow / NewLoad decisions emitted after bailout.
        assert plan.decisions_of(ExtendPages) == ()
        assert plan.decisions_of(HostGrow) == ()
        assert plan.decisions_of(NewLoadAsync) == ()

    def test_no_pending_no_bailout(self) -> None:
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["u"],
            global_seq_state={"u": _state()},
            has_pending=False,
            worker_view_stats=WorkerViewStats(num_total_pages=100, num_free_pages=99),
        )
        assert plan.watermark_break is False

    def test_at_watermark_not_triggered(self) -> None:
        """free_pct == watermark: strict > so no bailout."""
        plan = _plan_full(
            BoundaryPlanner(_cfg(prefill_watermark_pct=70)),
            decode_uuids=["u"],
            global_seq_state={"u": _state()},
            has_pending=True,
            worker_view_stats=WorkerViewStats(num_total_pages=100, num_free_pages=70),
        )
        assert plan.watermark_break is False

    def test_in_decode_filter_excludes_zero_gpu_pages(self) -> None:
        """Sequence in decode_uuids but with gpu_pages_allocated=0 is
        QUEUEING / EVICTED in disguise — the planner must not put it
        on hold (that would trip Phase-1 status_transition at runtime)."""
        states = {
            "real_decode": _state(gpu_pages_allocated=3),
            "queued_ghost": _state(gpu_pages_allocated=0),
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["real_decode", "queued_ghost"],
            global_seq_state=states,
            has_pending=True,
            worker_view_stats=WorkerViewStats(num_total_pages=100, num_free_pages=95),
        )
        onholds = plan.decisions_of(OnHold)
        assert onholds[0].uuids == ("real_decode",)


# ---------------------------------------------------------------------------
# Rule 2: HostGrow
# ---------------------------------------------------------------------------


class TestRule2HostGrow:
    def test_emits_host_grow_with_feasibility(self) -> None:
        states = {
            "u": _state(
                needs_host_growth=True,
                host_growth_pages=10,
                host_pages_allocated=20,
            )
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["u"],
            global_seq_state=states,
            worker_view_stats=WorkerViewStats(num_total_pages=1000, num_free_pages=500),
        )
        grows = plan.decisions_of(HostGrow)
        assert len(grows) == 1
        assert grows[0].uuids == ("u",)
        assert grows[0].pages == (10,)
        assert grows[0].feasible is True

    def test_infeasible_growth_still_emitted(self) -> None:
        """When growth > free - 5% safety margin, feasible=False but
        the decision still lands (executor checks the flag)."""
        states = {
            "u": _state(needs_host_growth=True, host_growth_pages=900),
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["u"],
            global_seq_state=states,
            worker_view_stats=WorkerViewStats(num_total_pages=1000, num_free_pages=10),
        )
        grows = plan.decisions_of(HostGrow)
        assert len(grows) == 1
        assert grows[0].feasible is False

    def test_zero_growth_emits_nothing(self) -> None:
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["u"],
            global_seq_state={"u": _state(needs_host_growth=True, host_growth_pages=0)},
        )
        assert plan.decisions_of(HostGrow) == ()


# ---------------------------------------------------------------------------
# Rule 3: HostEvict
# ---------------------------------------------------------------------------


class TestRule3HostEvict:
    def test_below_watermark_evicts_shortest_first(self) -> None:
        states = {
            "longest": _state(
                decoded_length=500, host_pages_allocated=10, assigned_rank=0
            ),
            "shortest": _state(
                decoded_length=10, host_pages_allocated=10, assigned_rank=0
            ),
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["longest", "shortest"],
            global_seq_state=states,
            enable_host_kv_eviction=True,
            host_kv_eviction_watermark=90,
            worker_view_stats=WorkerViewStats(
                num_total_pages=100, num_free_pages=1
            ),
        )
        evicts = plan.decisions_of(HostEvict)
        assert len(evicts) == 1
        # shortest-decoded-first → "shortest" evicted first
        assert evicts[0].uuids == ("shortest",) or "shortest" in evicts[0].uuids

    def test_eviction_disabled(self) -> None:
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["u"],
            global_seq_state={"u": _state(host_pages_allocated=10)},
            enable_host_kv_eviction=False,
            host_kv_eviction_watermark=90,
            worker_view_stats=WorkerViewStats(num_total_pages=100, num_free_pages=1),
        )
        assert plan.decisions_of(HostEvict) == ()

    def test_above_watermark_no_evict(self) -> None:
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["u"],
            global_seq_state={"u": _state()},
            enable_host_kv_eviction=True,
            host_kv_eviction_watermark=10,
            worker_view_stats=WorkerViewStats(num_total_pages=100, num_free_pages=50),
        )
        assert plan.decisions_of(HostEvict) == ()


# ---------------------------------------------------------------------------
# Rule 4: GPU extension + per-rank OnHold(EXTENSION_FAILED)
# ---------------------------------------------------------------------------


class TestRule4GpuExtension:
    def test_fits_gets_extend(self) -> None:
        states = {
            "u": _state(additional_pages_needed=3, assigned_rank=0),
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["u"],
            global_seq_state=states,
            per_rank_free=[10],
        )
        extends = plan.decisions_of(ExtendPages)
        assert len(extends) == 1
        assert extends[0] == ExtendPages(uuid="u", additional_pages=3)

    def test_rank_oversubscribed_holds_shortest_first(self) -> None:
        # Both seqs need 4 pages on rank 0; rank 0 has 6 free (2 pages
        # short), so exactly one must go to ON_HOLD — holding the
        # shortest-progress seq frees its 2 gpu_pages, closing the gap.
        # Shortest-decoded-first → "shortest" picked for OnHold so
        # longer-running "longest" keeps decoding.
        states = {
            "shortest": _state(
                decoded_length=5,
                gpu_pages_allocated=2,
                additional_pages_needed=4,
                assigned_rank=0,
            ),
            "longest": _state(
                decoded_length=200,
                gpu_pages_allocated=2,
                additional_pages_needed=4,
                assigned_rank=0,
            ),
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["shortest", "longest"],
            global_seq_state=states,
            per_rank_free=[6],
        )
        onholds = [
            d for d in plan.decisions_of(OnHold)
            if d.reason is OnHoldReason.EXTENSION_FAILED
        ]
        assert len(onholds) == 1
        assert "shortest" in onholds[0].uuids
        extends = plan.decisions_of(ExtendPages)
        assert [e.uuid for e in extends] == ["longest"]

    def test_multiple_ranks_independent_budgets(self) -> None:
        states = {
            "rank0_u": _state(additional_pages_needed=2, assigned_rank=0),
            "rank1_u": _state(additional_pages_needed=2, assigned_rank=1),
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["rank0_u", "rank1_u"],
            global_seq_state=states,
            per_rank_free=[10, 10],
            world_size=2,
        )
        extends = {e.uuid for e in plan.decisions_of(ExtendPages)}
        assert extends == {"rank0_u", "rank1_u"}

    def test_priority_aware_onhold(self) -> None:
        """HIGH priority (1) gets kept; NORMAL (0) gets held first."""
        states = {
            "normal": _state(
                decoded_length=500,  # longer
                gpu_pages_allocated=2,
                additional_pages_needed=4,
                assigned_rank=0,
            ),
            "high": _state(
                decoded_length=5,  # shorter
                gpu_pages_allocated=2,
                additional_pages_needed=4,
                assigned_rank=0,
            ),
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["normal", "high"],
            global_seq_state=states,
            per_rank_free=[6],
            priority_by_uuid={"normal": 0, "high": 1},
        )
        onholds = [
            d for d in plan.decisions_of(OnHold)
            if d.reason is OnHoldReason.EXTENSION_FAILED
        ]
        # priority 0 (normal) goes to hold first even though it has
        # higher decoded_length — priority sort is primary key.
        assert onholds[0].uuids == ("normal",)


# ---------------------------------------------------------------------------
# Rule 5: NewLoadAsync
# ---------------------------------------------------------------------------


class TestRule5NewLoadAsync:
    def test_candidates_fit_into_adjusted_free(self) -> None:
        states = {
            "active": _state(gpu_pages_allocated=2, assigned_rank=0),
        }
        candidates = {
            "load_a": LoadCandidateState(
                pages_needed=3, assigned_rank=0, status="PREFILLED",
                decoded_length=0,
            ),
            "load_b": LoadCandidateState(
                pages_needed=5, assigned_rank=0, status="ON_HOLD",
                decoded_length=100,
            ),
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["active"],
            global_seq_state=states,
            global_candidate_info=candidates,
            per_rank_free=[10],
        )
        loads = plan.decisions_of(NewLoadAsync)
        assert len(loads) == 1
        # load_b has higher decoded_length → sorted first (legacy
        # preference for longer-running sequences).
        assert loads[0].uuids == ("load_b", "load_a")
        assert set(loads[0].rank_pages) == {(0, 5), (0, 3)}

    def test_budget_drops_candidate_past_limit(self) -> None:
        states = {"active": _state(gpu_pages_allocated=1, assigned_rank=0)}
        candidates = {
            "big": LoadCandidateState(
                pages_needed=10, assigned_rank=0, status="PREFILLED",
                decoded_length=50,
            ),
            "small": LoadCandidateState(
                pages_needed=2, assigned_rank=0, status="PREFILLED",
                decoded_length=100,
            ),
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["active"],
            global_seq_state=states,
            global_candidate_info=candidates,
            per_rank_free=[5],  # fits small but not big
        )
        loads = plan.decisions_of(NewLoadAsync)
        # small (decoded=100) sorts first and fits. big (10 pages) > 3
        # remaining, dropped.
        assert loads[0].uuids == ("small",)

    def test_subtracts_actual_extension(self) -> None:
        """Available budget is per_rank_free minus the extensions the
        planner already scheduled this cycle."""
        states = {
            "extends": _state(
                additional_pages_needed=4,
                gpu_pages_allocated=2,
                assigned_rank=0,
            ),
        }
        candidates = {
            "load": LoadCandidateState(
                pages_needed=7, assigned_rank=0, status="PREFILLED",
                decoded_length=50,
            ),
        }
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["extends"],
            global_seq_state=states,
            global_candidate_info=candidates,
            per_rank_free=[10],  # 10 - 4 (extension) = 6; load needs 7 → drop
        )
        assert plan.decisions_of(NewLoadAsync) == ()


# ---------------------------------------------------------------------------
# Determinism + structural guarantees
# ---------------------------------------------------------------------------


class TestStructural:
    def test_plan_is_deterministic(self) -> None:
        p = BoundaryPlanner(_cfg())
        states = {
            "u1": _state(completed=True),
            "u2": _state(additional_pages_needed=2, gpu_pages_allocated=2),
        }
        args = dict(
            decode_uuids=["u1", "u2"],
            global_seq_state=states,
            per_rank_free=[10],
        )
        plan1 = _plan_full(p, **args)
        plan2 = _plan_full(p, **args)
        assert plan1 == plan2

    def test_watermark_break_surfaces_on_plan(self) -> None:
        plan = _plan_full(
            BoundaryPlanner(_cfg()),
            decode_uuids=["u"],
            global_seq_state={"u": _state(gpu_pages_allocated=2)},
            has_pending=True,
            worker_view_stats=WorkerViewStats(num_total_pages=100, num_free_pages=90),
        )
        assert plan.watermark_break is True
