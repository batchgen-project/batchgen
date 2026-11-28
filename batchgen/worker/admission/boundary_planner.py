"""Gather, decision and broadcast of a page boundary.

``BoundaryPlanner.plan`` runs, on one rank, the part of a page boundary that
ends with every rank holding the same decisions: the single batched gather of
per-rank state, the merge, the rank-0 decision (``BoundaryHandler``), and the
broadcast. It reads the rank's state from the decode context. Whether a
sequence has completed is decided by the caller's completion rules and comes
in as a callable.
"""

from __future__ import annotations

# Declared contract, checked against this module's source by
# tests/worker/test_admission_module_contracts.py: what this module reads
# off the objects it is given, what it writes on them directly, what it
# writes on an object it reached THROUGH one of them, what it mutates in
# place, and the orderings it relies on. A change here that the source
# does not match, or a source change this does not cover, fails that test.
MODULE_CONTRACT = {
    "reads": (
        "ctx.core_engine",
        "ctx.facts",
        "ctx.host_view",
        "ctx.sched",
        "gpu_manager.get_stats",
        "gpu_manager.is_initialized",
        "kv._make_boundary_decision_request",
        "seq.uuid",
        "state.get",
    ),
    "writes": (
        "timing.gather_ms",
        "timing.num_completed",
        "timing.num_onhold",
    ),
    "writes_via_local": (
        "seq._rep_detected",
        "seq.eos_reached",
        "seq.kda_record_pages",
        "seq.max_decode_length",
        "seq.original_max_decode_length",
        "seq.prompt_length",
        "seq.reentry_decoded_baseline",
        "seq.total_decoded_before_eviction",
    ),
    "mutates": (),
    "orderings": (
        "the gather precedes the decision, and the decision precedes the broadcast, because every rank must act on the same verdict.",
        "the local mirror of another rank's sequence is updated only after that rank's state has been gathered, and only for a sequence this rank does not own.",
    ),
}

import logging
import time
from typing import Callable, List

import torch.distributed as dist

from batchgen.lifespan import SeqEvent
from batchgen.sequence import SequenceStatus
from batchgen.worker.boundary import (
    BoundaryDecisions,
    BoundaryHandler,
    validate_boundary_payload_alignment,
)
from batchgen.worker.decode import reduce_decode_capacity_snapshot
from batchgen.worker.kv_capacity_engine import KVCapacityEngine


def _trajectory_slot(ctx, seq) -> int:
    if ctx.sched.trajectories.trajectory_book is None:
        raise RuntimeError(f"sequence {seq.uuid} has no active trajectory slot")
    try:
        # Slot numbers are local to a node's allocator, so the same UUID holds
        # different slots on different nodes; never trust the process-global
        # ``_buffer_slot`` field (last set on another node) as the lookup key.
        return ctx.sched.trajectories.slot_for(seq.uuid)
    except KeyError as exc:
        raise RuntimeError(
            f"sequence {seq.uuid} has no active trajectory reservation"
        ) from exc


def _sync_trajectory_metadata(ctx, seq, state) -> None:
    """Refresh a local QueryBook mirror from the owning rank's lengths."""
    if not ctx.sched.trajectories.has_sequence(seq.uuid):
        return
    token_length = state.get("trajectory_token_length")
    if token_length is None:
        return
    ctx.sched.trajectories.synchronize_metadata(
        _trajectory_slot(ctx, seq),
        prompt_length=int(state["prompt_length"]),
        token_length=int(token_length),
        decoded_length=int(
            state.get("trajectory_decoded_length", state["decoded_length"])
        ),
    )


def _compute_decisions(
    ctx, kv, max_rank_bsz, decode_uuids, global_seq_state, global_candidate_info,
    per_rank_free, per_rank_total, chunk_size, per_node_host_stats,
):
    """The rank-0 decision: build the frozen request, run the pure planner."""
    return BoundaryHandler.compute_decisions(
        kv._make_boundary_decision_request(
            ctx, decode_uuids, global_seq_state, global_candidate_info,
            per_rank_free, per_rank_total, chunk_size, per_node_host_stats,
            max_rank_bsz,
        )
    )


class BoundaryPlanner:
    """Stateless: everything it touches comes in through ``plan``."""

    def plan(
        self,
        ctx,
        is_sequence_completed: Callable,
        max_rank_bsz: int,
        decode_uuids: List[str],
        gpu_manager,
        timing,
        t0: float,
    ):
        """Gather, decide on rank 0, broadcast.

        Returns ``(decisions, global_seq_state, worker_view, chunk_size, t0)``:
        the decisions every rank now holds, the merged per-sequence state the
        carry-out reads, the host view the decision was computed against, the
        effective host chunk size, and the timer mark for the next section.
        """
        kv = KVCapacityEngine()
        # ========== PHASE 1: SINGLE BATCHED ALL_GATHER ==========
        t0 = time.perf_counter()

        local_payload_error = None
        try:
            if gpu_manager is not None and gpu_manager.is_initialized:
                gpu_stats = gpu_manager.get_stats()
                local_free_pages = int(gpu_stats.num_free_pages)
                local_total_pages = int(gpu_stats.num_total_pages)
            else:
                local_free_pages = 0
                local_total_pages = 0
        except Exception as exc:
            # Carry allocator stats failures through the payload gather so
            # peers do not block while this rank raises before all-gather.
            local_free_pages = 0
            local_total_pages = 0
            local_payload_error = (
                f"GPU-KV stats: {type(exc).__name__}: {exc}"
            )

        # DEBUG: Log decode_uuids and which ones this rank owns
        my_owned = [u for u in decode_uuids if u in ctx.sched.seq_table.uuid_to_local]
        if ctx.facts.rank == 0:
            logging.debug(
                f"Rank {ctx.facts.rank}: State gathering - decode_uuids_len={len(decode_uuids)}, "
                f"my_owned_count={len(my_owned)}"
            )

        # Build local state for sequences owned by this rank
        try:
            chunk_size = kv._get_effective_chunk_size(ctx)
        except Exception as exc:
            chunk_size = 0
            local_payload_error = (
                f"chunk size: {type(exc).__name__}: {exc}"
            )
        local_seq_state = {}
        for uuid in decode_uuids:
            if uuid in ctx.sched.seq_table.uuid_to_local:
                seq = ctx.sched.pool.get_sequence(uuid)
                try:
                    is_completed = is_sequence_completed(seq)
                    seq.validate_metadata(
                        f"rank {ctx.facts.rank} _page_boundary_fast/decode_state",
                        allow_terminal_capacity_gap=is_completed,
                    )
                except Exception as exc:
                    local_payload_error = (
                        f"decode {uuid}: {type(exc).__name__}: {exc}"
                    )
                    continue
                local_seq_state[uuid] = {
                    'decoded_length': seq.decoded_length,
                    'current_context_length': seq.current_context_length,
                    'gpu_pages_allocated': seq.gpu_pages_allocated,
                    'eos_reached': seq.eos_reached,
                    'rep_detected': getattr(seq, '_rep_detected', False),
                    'completed': is_completed,
                    'additional_pages_needed': seq.get_additional_gpu_pages_needed(),
                    'assigned_rank': seq.assigned_rank,  # Include for consistency
                    # Option 1 (G>1): the serve-group id. The boundary validator keys
                    # group ownership on this (a uuid is reported by exactly its G
                    # contiguous ranks [g*G,(g+1)*G)), not on a single assigned_rank.
                    'decode_dp_group': seq.decode_dp_group,
                    # Host KV growth fields
                    'needs_host_growth': seq.needs_host_kv_growth(chunk_size),
                    'host_growth_pages': seq.get_host_growth_pages(chunk_size),
                    'host_pages_allocated': seq.host_pages_allocated,
                    'host_owned_pages': seq.host_owned_pages,
                    # KDA state record pages in the combined host budget
                    # (表 B): joins the eviction benefit next to
                    # host_owned_pages. 0 when no per-sequence record exists.
                    'kda_record_pages': seq.kda_record_pages,
                    'host_token_capacity': seq.host_token_capacity,
                    # prompt_length: required so Phase 4.C can compute the
                    # re-entry reconstruction length on ALL ranks deterministically,
                    # not just the owner. Without this, non-owning ranks have a
                    # stale prompt_length for re-evicted sequences (where the
                    # owner has already rewritten prompt_length in a prior
                    # eviction). See Phase 4.C.
                    'prompt_length': seq.prompt_length,
                    'reentry_decoded_baseline': seq.reentry_decoded_baseline,
                    'max_decode_length': seq.max_decode_length,
                    'original_max_decode_length': seq.original_max_decode_length,
                    # QueryBook metadata is process-local even though its token
                    # pages are node-shared. Publish the owner's lengths so
                    # non-owning ranks can repair their mirror before re-entry.
                    'trajectory_token_length': (
                        ctx.sched.trajectories.trajectory_book.metadata(_trajectory_slot(ctx, seq)).token_length
                        if ctx.sched.trajectories.trajectory_book is not None
                        and ctx.sched.trajectories.trajectory_book.has_sequence(uuid)
                        else None
                    ),
                    'trajectory_decoded_length': (
                        ctx.sched.trajectories.trajectory_book.metadata(_trajectory_slot(ctx, seq)).decoded_length
                        if ctx.sched.trajectories.trajectory_book is not None
                        and ctx.sched.trajectories.trajectory_book.has_sequence(uuid)
                        else None
                    ),
                    # total_decoded_before_eviction: propagated here so the
                    # next _prepare_prefill_batch's eviction priority sort is
                    # consistent across ranks.
                    'total_decoded_before_eviction': seq.total_decoded_before_eviction,
                }

        # Get candidates for loading - report PREFILLED/ON_HOLD sequences that could be loaded
        # CRITICAL FIX: Only report PREFILLED or ON_HOLD sequences as load candidates.
        # QUEUEING sequences have NOT been registered with host KV yet (registration
        # happens during _config_prefill_for_batch), so trying to load them would fail
        # with "Sequence X is not registered" error from the host KV backend.
        decode_uuids_set = set(decode_uuids)
        local_candidate_state = {}
        valid_load_statuses = {SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD}
        for uuid in ctx.sched.seq_table.uuid_to_local.keys():
            if uuid in decode_uuids_set:
                continue  # Already in decode batch
            seq = ctx.sched.pool.get_sequence(uuid)
            if seq is None:
                continue
            if seq.status == SequenceStatus.COMPLETED:
                continue  # Don't load completed sequences
            if seq.status not in valid_load_statuses:
                continue  # Only load PREFILLED/ON_HOLD (not QUEUEING/IN_PREFILL)
            try:
                seq.validate_metadata(f"rank {ctx.facts.rank} _page_boundary_fast/load_candidate")
            except Exception as exc:
                local_payload_error = (
                    f"load candidate {uuid}: {type(exc).__name__}: {exc}"
                )
                continue
            try:
                pages_needed = seq.get_gpu_pages_for_two_page_buffer()
            except Exception as exc:
                local_payload_error = (
                    f"load candidate {uuid} page count: {type(exc).__name__}: {exc}"
                )
                continue
            # Report this as a potential load candidate
            local_candidate_state[uuid] = {
                'pages_needed': pages_needed,
                'assigned_rank': seq.assigned_rank,
                'decode_dp_group': seq.decode_dp_group,  # Option 1 group-ownership key
                'status': seq.status.name,  # Include status for debugging
                'decoded_length': seq.decoded_length,  # For prioritized loading
            }

        # Pack everything into one dict for single all_gather
        local_payload = {
            'free_pages': local_free_pages,
            'total_pages': local_total_pages,
            'error': local_payload_error,
            'seq_state': local_seq_state,
            'candidate_state': local_candidate_state,
        }

        all_payloads = [None] * ctx.facts.world_size
        dist.all_gather_object(all_payloads, local_payload)
        payload_errors = [
            f"rank {rank_idx}: {payload.get('error')}"
            for rank_idx, payload in enumerate(all_payloads)
            if isinstance(payload, dict) and payload.get('error')
        ]
        if payload_errors:
            raise RuntimeError(
                "[SCHED_INVARIANT] boundary payload validation failed: "
                + " | ".join(payload_errors)
            )
        validate_boundary_payload_alignment(
            decode_uuids, all_payloads, group_size=ctx.facts.attn_tp_size
        )

        timing.gather_ms = (time.perf_counter() - t0) * 1000

        # ========== PHASE 2: MERGE GATHERED DATA + RANK-0 DECISIONS ==========
        t0 = time.perf_counter()

        # Extract per-rank live capacity.  Total pages are immutable after GPU-KV
        # initialization; rank 0 validates that invariant before planning any
        # extension or load.
        per_rank_free = [p['free_pages'] for p in all_payloads]
        per_rank_total = [p['total_pages'] for p in all_payloads]
        # Validate the live allocator snapshot on EVERY rank before rank 0
        # computes or broadcasts decisions.  Keeping this check inside the
        # rank-0 planner would strand peers in broadcast when one total differs.
        reduce_decode_capacity_snapshot(
            tuple(per_rank_total),
            tuple(per_rank_free),
            world_size=ctx.facts.world_size,
            attn_tp_size=ctx.facts.attn_tp_size,
        )

        # Merge sequence state. G==1: each uuid appears exactly once (single owner).
        # G>1 (Option 1): the uuid is reported by all G ranks of its group, so pin a
        # CANONICAL owning_rank = decode_dp_group*G (the group leader) instead of
        # last-writer-wins (which would arbitrarily land on the highest group rank).
        _G_merge = ctx.facts.attn_tp_size
        global_seq_state = {}
        for rank_idx, payload in enumerate(all_payloads):
            if payload and payload['seq_state']:
                for uuid, state in payload['seq_state'].items():
                    global_seq_state[uuid] = state
                    if _G_merge > 1:
                        _g = state.get('decode_dp_group')
                        global_seq_state[uuid]['owning_rank'] = (
                            _g * _G_merge if _g is not None else rank_idx
                        )
                    else:
                        global_seq_state[uuid]['owning_rank'] = rank_idx

        # Merge candidate state
        global_candidate_info = {}
        for payload in all_payloads:
            if payload and payload['candidate_state']:
                global_candidate_info.update(payload['candidate_state'])

        # VALIDATION: Check that all decode_uuids have state reported
        missing_uuids = [u for u in decode_uuids if u not in global_seq_state]
        if missing_uuids:
            missing_details = []
            for missing_uuid in missing_uuids[:10]:
                seq = ctx.sched.pool.get_sequence(missing_uuid)
                expected_rank = seq.assigned_rank if seq else "N/A"
                in_local_map = missing_uuid in ctx.sched.seq_table.uuid_to_local
                seq_status = seq.status.name if seq else "NOT_FOUND"
                rank_reported = [r for r, p in enumerate(all_payloads)
                                if p and p.get('seq_state', {}).get(missing_uuid)]
                missing_details.append(
                    f"{missing_uuid}(assigned_rank={expected_rank}, in_local_map={in_local_map}, "
                    f"status={seq_status}, reported_by_ranks={rank_reported})"
                )
            raise RuntimeError(
                f"Rank {ctx.facts.rank}: [SCHED_INVARIANT] {len(missing_uuids)} active decode "
                f"UUIDs missing from gathered seq_state; decode_uuids_len={len(decode_uuids)}, "
                f"global_seq_state_len={len(global_seq_state)}, details={missing_details}"
            )

        # Update local SequenceEntry with gathered info (for sequences on other ranks)
        post_gather_errors = []
        for uuid, state in global_seq_state.items():
            if uuid not in ctx.sched.seq_table.uuid_to_local:
                seq = ctx.sched.pool.get_sequence(uuid)
                if seq is not None:
                    seq.mirror_decode_state(
                        state['decoded_length'],
                        state['current_context_length'],
                        state['gpu_pages_allocated'],
                    )
                    seq.eos_reached = state['eos_reached']
                    if state.get('rep_detected', False):
                        seq._rep_detected = True
                    # Sync host KV fields to keep all ranks consistent for migration planning
                    seq.mirror_host_counters(
                        state['host_pages_allocated'],
                        state['host_owned_pages'],
                        state['host_token_capacity'],
                    )
                    seq.kda_record_pages = state['kda_record_pages']
                    # Sync prompt_length (may have been rewritten by a prior
                    # eviction on the owner) and total_decoded_before_eviction
                    # so Phase 4 mutations can be computed deterministically on
                    # all ranks, and the next _prepare_prefill_batch selection
                    # priority sort is consistent.
                    if 'prompt_length' in state:
                        seq.prompt_length = state['prompt_length']
                    if 'reentry_decoded_baseline' in state:
                        seq.reentry_decoded_baseline = state['reentry_decoded_baseline']
                    if 'max_decode_length' in state:
                        seq.max_decode_length = state['max_decode_length']
                    if 'original_max_decode_length' in state:
                        seq.original_max_decode_length = state['original_max_decode_length']
                    if 'total_decoded_before_eviction' in state:
                        seq.total_decoded_before_eviction = state['total_decoded_before_eviction']
                    # Validate gathered ctx_len
                    expected_ctx = seq.original_prompt_length + seq.decoded_length
                    if seq.current_context_length != expected_ctx:
                        seq.log_event(SeqEvent.CTX_MISMATCH, ctx.facts.rank,
                            f"gathered_ctx={seq.current_context_length}, expected={expected_ctx}")
                        seq.repair_context_identity()
                    try:
                        seq.validate_metadata(
                            f"rank {ctx.facts.rank} _page_boundary_fast/gathered_state",
                            require_owner_tensors=False,
                            allow_terminal_capacity_gap=bool(state.get('completed', False)),
                        )
                    except Exception as exc:
                        post_gather_errors.append(
                            f"gathered {uuid}: {type(exc).__name__}: {exc}"
                        )
            # The token pages are shared, but the QueryBook metadata is not.
            # Repair every local mirror from the canonical owner state before
            # Phase 4 can call begin_reentry_turn on all ranks.
            seq = ctx.sched.pool.get_sequence(uuid)
            if seq is not None:
                try:
                    _sync_trajectory_metadata(ctx, seq, state)
                except Exception as exc:
                    post_gather_errors.append(
                        f"trajectory {uuid}: {type(exc).__name__}: {exc}"
                    )

        post_gather_reports = [None] * ctx.facts.world_size
        dist.all_gather_object(post_gather_reports, post_gather_errors)
        post_gather_failures = [
            f"rank {rank_idx}: {error}"
            for rank_idx, errors in enumerate(post_gather_reports)
            for error in (errors or [])
        ]
        if post_gather_failures:
            raise RuntimeError(
                "[SCHED_INVARIANT] gathered boundary state validation failed: "
                + " | ".join(post_gather_failures)
            )

        # ========== RANK 0 COMPUTES ALL DECISIONS ==========
        # Only rank 0 makes batching decisions. All other ranks receive via broadcast.
        # This eliminates desync from independent decision-making.
        # The worker owns the mirrored coordinator for DSA.  Using the engine's
        # primary view here would let a partial primary/aux growth split escape the
        # coordinator's transaction guard.
        worker_view = ctx.host_view
        if worker_view is None:
            worker_view = getattr(ctx.core_engine, "host_paged_kv_worker_view", None)
        per_node_host_stats = kv._gather_host_kv_stats_by_node(ctx, worker_view)

        if ctx.facts.rank == 0:
            try:
                decisions = _compute_decisions(ctx, kv, max_rank_bsz,
                    decode_uuids, global_seq_state, global_candidate_info,
                    per_rank_free, per_rank_total, chunk_size, per_node_host_stats,
                )
            except Exception as exc:
                # Rank 0 owns the pure planner, but a malformed gathered state
                # must become a broadcast scheduler error rather than strand peers
                # waiting in the decision broadcast.
                logging.exception("Rank 0 boundary planner failed")
                decisions = BoundaryDecisions(
                    completed_uuids=[], active_uuids=[], host_growth_uuids=[],
                    host_growth_pages=[], growth_feasible=False,
                    host_evicted_uuids=[], onhold_uuids=[],
                    seqs_needing_extension=[], new_load_uuids=[],
                    decode_uuids_final=[],
                    scheduler_error=(
                        "[SCHED_INVARIANT] rank-0 boundary planner failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
        else:
            decisions = None

        # ========== PHASE 3: BROADCAST DECISIONS ==========
        decisions_list = [decisions]
        dist.broadcast_object_list(decisions_list, src=0)
        decisions = decisions_list[0]
        if decisions.scheduler_error:
            raise RuntimeError(f"Rank {ctx.facts.rank}: {decisions.scheduler_error}")

        timing.num_completed = len(decisions.completed_uuids)
        timing.num_onhold = len(decisions.onhold_uuids)

        return decisions, global_seq_state, worker_view, chunk_size, t0
