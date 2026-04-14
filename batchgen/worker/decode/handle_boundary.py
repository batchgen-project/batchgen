"""Phase 2.8.2f — port of ``_decode_handle_boundary`` (7941-8128).

Drives the mid-loop page-boundary transition: invokes
:meth:`BoundaryHandler.run_full`, reads the returned
:class:`BoundaryResult` to decide ``should_break`` / ``should_continue``,
rebuilds the post-boundary ``new_tokens`` tensor.

Two deliberate removals from legacy:

  * **No admission polling.** Legacy polled ``self._admission_queue``
    at every boundary (batchgen_worker.py:8017-8037). That was the
    dual-admission-poller bug POIS fingered as the L4 root cause.
    Admission is the orchestrator's ``AdmissionCoordinator``; the
    decode loop never polls.
  * **No `_put_sequences_on_hold` call.** Legacy called the
    status-blind helper when ``watermark_triggered`` fired
    (batchgen_worker.py:8007). The native path handles watermark via
    the planner's ``OnHold(WATERMARK_TRIGGER)`` decision which the
    executor applies inside ``BoundaryHandler.run_full``; the
    strict native ``put_on_hold`` (Stage 3) is the belt-and-braces
    safety net if we ever re-introduce an out-of-loop eviction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from batchgen.worker.boundary import BoundaryHandler, BoundaryResult
from batchgen.worker.decode.state import DecodeState
from batchgen.worker.protocols import LegacyInfraBackend
from batchgen.worker.state import WorkerState


@dataclass(frozen=True)
class BoundaryOutcome:
    """Result of one ``handle_boundary`` invocation.

    The decode loop reads ``should_break`` / ``should_continue`` to
    decide whether to exit ``run_continuous`` or skip the forward
    step and loop back to the boundary check.
    """

    should_break: bool
    should_continue: bool
    result: BoundaryResult


def handle_boundary(
    state: WorkerState,
    adapter: LegacyInfraBackend,
    boundary: BoundaryHandler,
    decode_state: DecodeState,
    *,
    gpu_manager: Any,
) -> BoundaryOutcome:
    """Drive one native boundary cycle + sync the DecodeState.

    Flow:
      1. ``boundary.run_full`` — runs wait_pending, gather, plan,
         apply, finalize.
      2. Update ``decode_state.decode_uuids`` / ``batch`` from the
         returned result.
      3. Bump cumulative counters.
      4. Watermark break: when either the planner flagged watermark
         (``result.plan.watermark_break``) or finalize reported
         ``result.watermark_triggered``, drain deferred KV appends
         (belt-and-braces so host KV has the full state before the
         sequences go ON_HOLD) and return ``should_break=True``.
      5. Empty decode_uuids: if the handler still has pending async
         loads (the cycle launched a new one), signal
         ``should_continue=True`` so the outer loop re-enters the
         boundary to integrate them; otherwise ``should_break=True``.
      6. Happy path: rebuild ``decode_state.new_tokens`` from the
         post-boundary batch via
         ``adapter.rebuild_input_tokens(batch)`` and mark the page
         table verified (``BoundaryHandler.run_full``'s
         ``check_post_page_table_order`` guard already asserted
         parity).
    """
    result = boundary.run_full(
        decode_uuids=list(decode_state.decode_uuids),
        batch=list(decode_state.batch),
        gpu_manager=gpu_manager,
    )

    decode_state.decode_uuids = list(result.decode_uuids)
    decode_state.batch = list(result.batch)
    decode_state.cumulative_boundaries += 1
    decode_state.page_table_verified = True

    watermark = bool(result.plan.watermark_break) or bool(result.watermark_triggered)
    if watermark:
        adapter.wait_pending_kv_append_tasks()
        return BoundaryOutcome(
            should_break=True, should_continue=False, result=result,
        )

    if not decode_state.decode_uuids:
        # Empty cohort after apply; if a new async load landed on the
        # handler's pending stash, ``should_continue`` asks the outer
        # loop to re-enter the boundary immediately so wait_pending
        # can integrate the load on the next tick.
        has_pending = bool(boundary._pending_load_uuids)
        if has_pending:
            return BoundaryOutcome(
                should_break=False, should_continue=True, result=result,
            )
        return BoundaryOutcome(
            should_break=True, should_continue=False, result=result,
        )

    # Rebuild the input-token tensor for the post-boundary batch. This
    # is pure infrastructure (touches the query-book buffer pool) so
    # stays on the adapter.
    decode_state.new_tokens = adapter.rebuild_input_tokens(
        list(decode_state.batch)
    )
    return BoundaryOutcome(
        should_break=False, should_continue=False, result=result,
    )


__all__ = ["BoundaryOutcome", "handle_boundary"]
