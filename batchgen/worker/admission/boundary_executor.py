"""Carry-out of the page-boundary decisions.

``BoundaryExecutor.apply`` executes, on one rank, the decisions rank 0
computed and broadcast for a page boundary: releases of completed and evicted
sequences, host growth, the status application, GPU page extension, on-hold
and load. It reads the rank's state from the decode context and reaches the
KV capacity engine with that same context. The two things it cannot do from
the context, reading a completed sequence's text and sending its completion,
arrive as a ``CompletionIO`` from the caller.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.distributed as dist

from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator
from batchgen.lifespan import SeqEvent
from batchgen.sequence import SequenceStatus
from batchgen.worker.admission.boundary_planner import _trajectory_slot
from batchgen.worker.admission.plan import (
    StatusEdge,
    Transition,
    TransitionPlan,
    apply_or_raise,
    update_batch_status,
)
from batchgen.worker.decode_engine import BATCHGEN_CB_DEBUG
from batchgen.worker.indexing import IndexLookupRequest, IndexManager
from batchgen.worker.kv_capacity_engine import KVCapacityEngine


@dataclass(frozen=True)
class CompletionIO:
    """The completion outlet the carry-out calls back into.

    ``gather_completed_tokens(completed_uuids)`` reads the decoded text of the
    completed sequences across ranks; ``report_completion(uuid, text)`` sends
    one completion. Both belong to the caller: they touch the request book and
    the response queue, which the decode context does not carry.
    """

    gather_completed_tokens: Callable
    report_completion: Callable


def _index_request(ctx) -> IndexLookupRequest:
    return IndexLookupRequest(
        rank=ctx.facts.rank,
        local_to_uuid=ctx.sched.seq_table.local_to_uuid,
        uuid_to_local=ctx.sched.seq_table.uuid_to_local,
        global_batch=ctx.sched.pool,
    )


class BoundaryExecutor:
    """Stateless: everything it touches comes in through ``apply``."""

    def apply(
        self,
        ctx,
        completion: CompletionIO,
        decisions,
        decode_uuids: List[str],
        batch: List[int],
        gpu_manager,
        worker_view,
        global_seq_state,
        chunk_size: int,
        timing,
        boundary_start: float,
        t0: float,
    ):
        """Carry out the broadcast boundary decisions on this rank.

        Returns ``(decode_uuids, batch, new_async_task, new_load_uuids,
        new_load_local, new_load_global, no_decode_rows_left,
        async_load_tensors)``. When ``no_decode_rows_left`` is True the
        boundary ends here: no extension, on-hold or load ran.
        ``async_load_tensors`` is the pointer bundle of a load launched here,
        which the caller must keep alive until the load is waited for; None
        when no load was launched.
        """
        kv = KVCapacityEngine()
        async_load_tensors = None
        # ========== PHASE 4: EXECUTE DECISIONS LOCALLY ==========
        # All ranks execute the same decisions, but only operate on locally-owned sequences

        # A. Release completed sequences
        #
        # ORDERING FIX: _release_gpu_kv_pages and _release_host_kv_pages_for_batch
        # must run BEFORE _report_completion. Previously _report_completion ran
        # first, which pops seq.uuid from ctx.sched.seq_table.uuid_to_local on ALL ranks
        # (including the owner). The subsequent
        #     my_completed = [u for u in completed_uuids if u in ctx.sched.seq_table.uuid_to_local]
        # filter then always produced an EMPTY list on the owner — so the host
        # KV worker view never released its pages for completed sequences, and
        # the GPU KV manager never released its pages either. Host KV slowly
        # filled up across the test run, triggering excessive eviction cycles,
        # which amplified the cross-rank state drift that eventually crashed the
        # server at a collective timeout.
        completed_uuids = decisions.completed_uuids
        if completed_uuids:
            # The decode set BEFORE any status write of this boundary. The
            # completion release below rebuilds the GPU page table from it;
            # the boundary's own decode_uuids is NOT a substitute for this
            # live index (rows can be IN_DECODE without being in this call's
            # decode set), so the index is read first, ahead of every status
            # write, and the release derives its remaining rows from this
            # snapshot.
            in_decode_before = ctx.sched.pool.get_sequences_by_status(
                SequenceStatus.IN_DECODE
            )
            update_batch_status(ctx.sched.pool, ctx.facts.rank,
                completed_uuids, SequenceStatus.COMPLETED, StatusEdge.COMPLETION
            )
        # Every PLANNED status change applies here, at one point. The decision
        # partitions the decode set, so the three planned groups are disjoint
        # and each row moves IN_DECODE -> {COMPLETED, EVICTED, ON_HOLD} exactly
        # once (COMPLETED above goes through _update_batch_status because rows
        # the per-step sync already completed must be skipped). The carry-out
        # below (resource releases, scalar updates, event logs, collectives)
        # reads no sequence status, so it is unaffected by the rows' statuses
        # having already changed. On-hold rows are drawn by the planner from
        # the post-eviction decode set, so they cannot exist when the
        # no-decode-rows-left early return below is taken; applying ON_HOLD
        # here is therefore equivalent to the old write point after that
        # return. Only the extension-failure amendment below may change a
        # status after this point, and the IN_DECODE snapshot above is
        # deliberately taken before any of these writes.
        if decisions.host_evicted_uuids:
            apply_or_raise(ctx.sched.pool, TransitionPlan(
                StatusEdge.BOUNDARY,
                tuple(
                    Transition(uuid, SequenceStatus.EVICTED)
                    for uuid in decisions.host_evicted_uuids
                ),
            ))
        if decisions.onhold_uuids:
            apply_or_raise(ctx.sched.pool, TransitionPlan(
                StatusEdge.BOUNDARY,
                tuple(
                    Transition(uuid, SequenceStatus.ON_HOLD)
                    for uuid in decisions.onhold_uuids
                ),
            ))
        if completed_uuids:
            # Gather decoded tokens from owning ranks before reporting
            gathered_texts = completion.gather_completed_tokens(completed_uuids)

            # Release resources on owners BEFORE popping local_map entries via
            # _report_completion (see ordering fix note above).
            my_completed = [u for u in completed_uuids if u in ctx.sched.seq_table.uuid_to_local]
            if my_completed:
                # Only release GPU pages for seqs that were actually GPU-allocated.
                # See note at the matching site (~line 5435) — zero-tok-EOS
                # prefill completions are in _uuid_to_local_map but never
                # registered with the GPU paged manager.
                gpu_allocated = [u for u in my_completed if u in ctx.gpu_resident]
                if gpu_allocated:
                    kv._release_gpu_kv_pages(ctx, IndexManager.get_local_indices_for_uuids(_index_request(ctx), gpu_allocated))
                # The remaining decode rows are the pre-write IN_DECODE
                # snapshot minus the completed rows: exactly what a live
                # status read returned when only the completion update had
                # been applied (that update moves only completed rows out of
                # IN_DECODE). Rows this decision evicts or puts on hold are
                # still in it — their statuses changed at the single point
                # above, but their resources are released later in this
                # boundary.
                completed_set = set(completed_uuids)
                kv._release_host_kv_pages_for_batch(
                    ctx,
                    my_completed,
                    remaining_decode_uuids=[
                        u for u in in_decode_before if u not in completed_set
                    ],
                )

            # All-ranks: zero scalar counters so downstream reads (e.g.
            # migration planning iterating all sequences) never see a stale
            # non-zero page count for completed sequences.
            for uuid in completed_uuids:
                seq = ctx.sched.pool.get_sequence(uuid)
                if seq is not None:
                    seq.finish_and_release()
                    ctx.gpu_resident.drop(uuid)

            # Report completions (this is what pops local_map on the owner).
            # Must run LAST so the _release_*_pages calls above see the
            # correct local_map state.
            for uuid in completed_uuids:
                completion.report_completion(uuid, gathered_text=gathered_texts.get(uuid))
            # Report completions to adaptive chunk sizer
            if ctx.chunk_sizer is not None:
                for uuid in completed_uuids:
                    state = global_seq_state.get(uuid)
                    if state:
                        ctx.chunk_sizer.report_completion(state['decoded_length'])
            # Log completion details for diagnostics
            if ctx.facts.rank == 0 and BATCHGEN_CB_DEBUG:
                for uuid in completed_uuids:
                    seq = ctx.sched.pool.get_sequence(uuid)
                    state = global_seq_state.get(uuid, {})
                    was_evicted = getattr(seq, 'total_decoded_before_eviction', 0) > 0
                    logging.debug(
                        f"[COMPLETION] seq={uuid[:8]} "
                        f"decoded={state.get('decoded_length', 0)} "
                        f"prompt={getattr(seq, 'original_prompt_length', seq.prompt_length)} "
                        f"was_evicted={was_evicted} "
                        f"host_pages={state.get('host_pages_allocated', 0)}"
                    )

        decode_uuids = decisions.active_uuids
        batch = IndexManager.get_local_indices_for_uuids(_index_request(ctx), decode_uuids)

        # B. Host KV eviction
        #
        # SYNC MODEL: Mutations here must keep every rank consistent without
        # requiring a follow-up _sync_sequence_metadata call. The trajectory book
        # is the token source of truth and remains resident across KV eviction;
        # only scalar metadata — prompt_length, current_context_length,
        # total_decoded_before_eviction, host/gpu page counters, status — is
        # updated on ALL ranks deterministically, using values already
        # synchronized in Phase 1/2 of this same boundary call.
        #
        # For re-entry length: new_reentry_len = seq.prompt_length +
        # seq.decoded_length. At Phase 4.C time, both operands are consistent
        # across ranks because Phase 2 synced them from the owner.
        host_evicted_uuids = decisions.host_evicted_uuids
        if host_evicted_uuids:
            # Release on-device/host KV resources. The unified trajectory stays
            # bound to the sequence, so no prompt or decoded-token copy is made.
            my_evicted = [u for u in host_evicted_uuids if u in ctx.sched.seq_table.uuid_to_local]
            if my_evicted:
                # Host-eviction usually targets seqs already in DECODE (so they
                # have GPU pages), but defensively intersect with the source-of-
                # truth set in case an EVICTED seq never reached decode.
                gpu_allocated = [u for u in my_evicted if u in ctx.gpu_resident]
                if gpu_allocated:
                    kv._release_gpu_kv_pages(ctx, IndexManager.get_local_indices_for_uuids(_index_request(ctx), gpu_allocated))
                if BATCHGEN_CB_DEBUG:
                    for uuid in my_evicted:
                        seq = ctx.sched.pool.get_sequence(uuid)
                        logging.debug(
                            f"[HOST_KV_EVICT_DETAIL] seq={uuid[:8]} "
                            f"decoded={seq.decoded_length} "
                            f"host_pages={seq.host_pages_allocated} "
                            f"trajectory_tokens={ctx.sched.trajectories.trajectory_book.metadata(_trajectory_slot(ctx, seq)).token_length}"
                        )
                # Host KV is ONE per-node SHARED shm region keyed by global_idx, so
                # release/unregister must fire EXACTLY once per sequence. Under G>1
                # all G ranks of a group hold the evicted uuid, so releasing on every
                # rank double-frees: the first releaser tombstones the entry and the
                # rest raise "Sequence ID ... not found during release". Filter to the
                # group leader (_owns_host_kv), matching the validated release path in
                # _release_host_kv_pages_for_batch. The all-ranks scalar-metadata loop
                # below iterates host_evicted_uuids and is deliberately NOT filtered.
                evicted_global_ids = [
                    ctx.sched.pool.get_sequence(u).global_idx for u in my_evicted
                    if ctx.facts.rank_ctx.owns_host_kv(ctx.sched.pool.get_sequence(u))
                ]
                # A follower of the group holds the leader's page table MIRRORED in
                # its own view plus a mirrored prefix state, and the re-entry of
                # this row registers the same global_idx again, so those rank-local
                # records must go as well. unregister_sequences edits only the
                # view's own HostKVPageTable — the shared pages are freed exactly
                # once, by the leader above. Always empty at G==1.
                mirrored_global_ids = [
                    ctx.sched.pool.get_sequence(u).global_idx for u in my_evicted
                    if not ctx.facts.rank_ctx.owns_host_kv(ctx.sched.pool.get_sequence(u))
                ]
                if worker_view is not None:
                    # The worker-owned view is a DualHostKVCoordinator for DSA and
                    # mirrors both page tables transactionally.  Calling the auxiliary
                    # view again here would double-release its pages and leave the two
                    # allocators out of sync.
                    worker_view.release_sequence_pages(evicted_global_ids)
                    worker_view.unregister_sequences(evicted_global_ids)
                    kv._release_prefix_cache_attachments(
                        ctx,
                        evicted_global_ids
                    )
                    if mirrored_global_ids:
                        for view in kv._host_views_by_group(ctx).values():
                            view.unregister_sequences(mirrored_global_ids)
                        kv._release_prefix_cache_attachments(ctx, mirrored_global_ids)

            # All-ranks: update scalar metadata deterministically. The retained
            # trajectory length is original_prompt_length + decoded_length; the
            # effective prompt for the next prefill is the previous effective prompt
            # plus only the newly generated suffix from this KV residency cycle.
            for uuid in host_evicted_uuids:
                seq = ctx.sched.pool.get_sequence(uuid)
                baseline = seq.reentry_decoded_baseline
                new_decoded_count = max(0, seq.decoded_length - baseline)
                new_reentry_len = seq.prompt_length + new_decoded_count
                seq.begin_eviction(new_reentry_len)
                if ctx.sched.trajectories.trajectory_book is not None and ctx.sched.trajectories.trajectory_book.has_sequence(uuid):
                    slot = ctx.sched.trajectories.trajectory_book.slot_for(uuid)
                    # Record the replayed prompt boundary at eviction time. Re-entry
                    # calls this idempotently, so migration while ON_HOLD carries a
                    # ledger consistent with the scalar prompt length.
                    ctx.sched.trajectories.trajectory_book.begin_reentry_turn(
                        slot, prompt_length=new_reentry_len
                    )
                saved = new_reentry_len
                seq.log_event(SeqEvent.EVICTED, ctx.facts.rank,
                    f"saved_tokens={saved}, decoded={seq.decoded_length}, "
                    f"new_this_cycle={new_decoded_count}")
                seq.release_all_kv()
                ctx.gpu_resident.drop(uuid)
                # Destroy the head-sharded KDA state the comment below relies
                # on. Host eviction frees the sequence's pages but reached the
                # group drop without releasing its state item, so the slot
                # stayed charged to a sequence that no longer has a group: the
                # pool leaked a slot per eviction, and a re-entry re-grouped
                # onto another group found its stale state on the old one.
                # Released only by the ranks holding the row, exactly like the
                # completion path, because the item is per rank.
                if uuid in ctx.sched.seq_table.uuid_to_local:
                    kv._release_kda_state_slots(ctx, [int(seq.global_idx)])
                # M2b (d): eviction destroys the head-sharded KDA state on the
                # group's ranks, so drop the decode DP-group. A re-prefilled
                # sequence must re-group fresh — otherwise _assign_decode_dp_groups
                # treats the stale non-None group id as a preserved re-entry and
                # skips it, stranding the sequence with no live state. The group
                # is remembered: the decoded suffix lives only in that group's
                # node's trajectory book, so re-grouping must land on it. No-op
                # for G==1 (the field is always None on the validated pure-DP path).
                from batchgen.decode_dp_group import evict_from_decode_dp_group
                evict_from_decode_dp_group(seq)
            # None of the per-sequence steps above reads a status; the rows'
            # EVICTED transition was applied at the single planned point above.

            evicted_set = set(host_evicted_uuids)
            decode_uuids = [u for u in decode_uuids if u not in evicted_set]
            batch = IndexManager.get_local_indices_for_uuids(_index_request(ctx), decode_uuids)

            if ctx.facts.rank == 0:
                logging.info(
                    f"[HOST_KV_EVICT] Evicted {len(host_evicted_uuids)} sequences"
                )

        # The host pool is shared by every rank on a node, but each rank
        # released only its own completed/evicted rows above, while rank 0's
        # plan already counts all of those pages as free. No rank may acquire
        # host pages (growth below, or prefill after an early return) until
        # every rank has finished releasing; otherwise a fast rank sees the
        # pre-release free count and AcquirePages fails. All conditions come
        # from the broadcast decisions, so every rank takes the same branch.
        if ctx.facts.world_size > 1 and (
            decisions.completed_uuids
            or decisions.host_evicted_uuids
            or (decisions.growth_feasible and decisions.host_growth_uuids)
        ):
            dist.barrier()

        # C. Host KV growth. This intentionally runs after completed/evicted
        # host pages have been released so worker_view free pages match the
        # growth-debt-aware plan computed on rank 0.
        if decisions.growth_feasible and decisions.host_growth_uuids:
            host_grow_requests = []
            growth_entries = []
            for uuid, growth_pages in zip(decisions.host_growth_uuids, decisions.host_growth_pages):
                seq = ctx.sched.pool.get_sequence(uuid)
                growth_entries.append((uuid, growth_pages, seq))
                # Host KV is one shared region per node.  G>1 ranks replicate
                # GPU/KDA state, but only the host-KV owner may mutate its page table.
                if ctx.facts.rank_ctx.owns_host_kv(seq):
                    host_grow_requests.append((seq.global_idx, growth_pages))

            # The growth plan counted each node's reclaimable retained prefix
            # pages as supply; release the part free pages do not cover now,
            # after every rank's releases and before the fresh preflight.
            # Gated on the rank-uniform prefix-cache config so every rank
            # reaches the reclaim's collectives.
            if ctx.facts.enable_prefix_cache:
                kv._reclaim_host_prefix_pages_for_growth(
                    ctx,
                    growth_entries, worker_view
                )

            # The rank-0 planner's stats are an admission snapshot.  Recheck the
            # node-shared allocator after all releases have completed and before any
            # owner mutates it.  This mirrors GPU-KV admission and prevents one rank
            # from consuming pages on a stale free-page estimate.
            if not kv._collective_host_kv_growth_capacity_ok(
                ctx,
                host_grow_requests, worker_view
            ):
                raise RuntimeError(
                    "[SCHED_INVARIANT] host-KV boundary growth rejected by fresh "
                    "collective capacity preflight"
                )

            local_growth_success = True
            if host_grow_requests:
                try:
                    if worker_view is None:
                        raise RuntimeError("host-KV worker view is unavailable")
                    worker_view.grow_pages_for_sequences(host_grow_requests)
                except Exception:
                    local_growth_success = False
                    logging.exception(
                        "Rank %s: host-KV boundary growth failed after collective preflight",
                        ctx.facts.rank,
                    )

            if ctx.facts.world_size > 1:
                growth_success = torch.tensor(
                    [1 if local_growth_success else 0],
                    dtype=torch.int32,
                    device=ctx.facts.device,
                )
                dist.all_reduce(growth_success, op=dist.ReduceOp.MIN)
                all_growth_success = bool(int(growth_success.item()))
            else:
                all_growth_success = local_growth_success
            if not all_growth_success:
                raise RuntimeError(
                    "[SCHED_INVARIANT] host-KV boundary growth failed on at least "
                    "one rank after collective preflight; worker state is not safe to continue"
                )

            # Commit scalar metadata only after every owner allocator call has
            # succeeded.  A failed growth therefore cannot leave a sequence claiming
            # pages that were never appended to its host chain.
            for uuid, growth_pages, seq in growth_entries:
                seq.grow_host_kv(growth_pages)

            if host_grow_requests:
                if ctx.facts.rank == 0:
                    logging.debug(
                        f"[HOST_KV_GROWTH] Grew {len(host_grow_requests)} sequences, "
                        f"chunk_size={chunk_size}"
                    )
                if ctx.facts.rank == 0 and BATCHGEN_CB_DEBUG:
                    for uuid, growth_pages in zip(decisions.host_growth_uuids, decisions.host_growth_pages):
                        if uuid in ctx.sched.seq_table.uuid_to_local:
                            seq = ctx.sched.pool.get_sequence(uuid)
                            old_cap = seq.host_token_capacity - growth_pages * seq.PAGE_SIZE
                            runway = seq.host_token_capacity - seq.current_context_length
                            logging.debug(
                                f"[HOST_KV_GROWTH_DETAIL] seq={uuid[:8]} "
                                f"old_cap={old_cap} new_cap={seq.host_token_capacity} "
                                f"runway={runway} pages={growth_pages}"
                            )

            # Every rank reads and writes host KV through its OWN worker-view
            # page table, so a follower whose leader just extended the shared
            # chain still holds the pre-growth table and would append the next
            # token past its last known page. Re-mirror the grown rows (page
            # tables only — the prefix attachment stays the leader's). Reached
            # by every rank under the broadcast growth condition, so the
            # gather inside stays aligned; no-op for G==1.
            kv._mirror_group_host_page_tables(
                ctx,
                [u for u in decisions.host_growth_uuids if u in ctx.sched.seq_table.uuid_to_local]
            )

        timing.process_ms = (time.perf_counter() - t0) * 1000

        # Calculate completed count BEFORE early return to ensure final iteration reports correctly
        timing.total_completed_cumulative = len(ctx.sched.pool.get_sequences_by_status(SequenceStatus.COMPLETED))

        if not decode_uuids:
            timing.total_ms = (time.perf_counter() - boundary_start) * 1000
            return decode_uuids, batch, None, [], [], [], True, None

        # D. GPU page extension / on-hold (using rank-0 decisions)
        t0 = time.perf_counter()
        onhold_uuids = decisions.onhold_uuids
        onhold_set = set(onhold_uuids)

        if onhold_uuids:
            # Owner-only: actually free GPU KV pages for locally-held sequences.
            my_onhold = [u for u in onhold_uuids if u in ctx.sched.seq_table.uuid_to_local]
            if my_onhold:
                local_indices = IndexManager.get_local_indices_for_uuids(_index_request(ctx), my_onhold)
                global_ids = IndexManager.local_indices_to_global_seq_ids(_index_request(ctx), local_indices)
                if global_ids and gpu_manager:
                    gpu_manager.free_pages_for_sequences(global_ids)
                for uuid in my_onhold:
                    ctx.gpu_resident.drop(uuid)

            # All-ranks: scalar metadata must be kept consistent. Non-owners
            # MUST also zero gpu_pages_allocated; otherwise their stale value
            # leaks into subsequent decision-making (e.g. migration planning
            # that iterates over all sequences). This was previously in the
            # my_onhold owner-only branch, creating a cross-rank desync window.
            for uuid in onhold_uuids:
                seq = ctx.sched.pool.get_sequence(uuid)
                seq.release_gpu_pages()
                seq.log_event(SeqEvent.ON_HOLD, ctx.facts.rank, "trigger=boundary")
            # The rows' ON_HOLD transition was applied at the single planned
            # point above; only their resources are released here.

            decode_uuids = [u for u in decode_uuids if u not in onhold_set]
            batch = IndexManager.get_local_indices_for_uuids(_index_request(ctx), decode_uuids)

            if BATCHGEN_CB_DEBUG:
                logging.info(
                    f"Rank {ctx.facts.rank}: After on-hold: batch_size={len(batch)}, "
                    f"num_onhold={len(onhold_uuids)}, my_onhold={len(my_onhold)}"
                )

        # Extend GPU pages for sequences that need it (not on-hold)
        seqs_needing_extension = decisions.seqs_needing_extension
        remaining_needing_ext = [u for u in seqs_needing_extension if u not in onhold_set]
        my_remaining_ext = [u for u in remaining_needing_ext if u in ctx.sched.seq_table.uuid_to_local]
        extension_preflight_ok = True
        if remaining_needing_ext:
            try:
                local_extension_pages = sum(
                    ctx.sched.pool.get_sequence(uuid).get_additional_gpu_pages_needed()
                    for uuid in my_remaining_ext
                )
                local_extension_free = (
                    gpu_manager.get_stats().num_free_pages
                    if gpu_manager is not None and gpu_manager.is_initialized else 0
                )
            except Exception as exc:
                logging.exception("Rank %s: boundary extension snapshot failed", ctx.facts.rank)
                kv._collective_gpu_kv_allocation_preflight(ctx, 1, 0)
                raise RuntimeError(
                    f"[SCHED_INVARIANT] boundary extension snapshot failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            extension_preflight_ok = kv._collective_gpu_kv_allocation_preflight(
                ctx,
                local_extension_pages, local_extension_free
            )

        local_extension_success = extension_preflight_ok
        if extension_preflight_ok and my_remaining_ext:
            try:
                local_extension_success = kv._extend_gpu_kv_allocation(ctx, my_remaining_ext)
            except Exception:
                # The stats preflight is only advisory if the allocator itself
                # rejects a request. Convert that local failure into the second
                # collective so peers can roll back without deadlocking here.
                logging.exception(
                    "Rank %s: GPU-KV extension raised after collective preflight",
                    ctx.facts.rank,
                )
                local_extension_success = False
        if remaining_needing_ext and extension_preflight_ok:
            # An allocator can still reject a request after the stats check. Make
            # that failure global before any rank updates status or proceeds to
            # loading.  Peers that extended successfully release the sequence's
            # full allocation below; this rolls back both the old and new pages
            # before the sequence is moved to ON_HOLD.
            local_extension_success = kv._collective_gpu_kv_allocation_preflight(
                ctx,
                0 if local_extension_success else 1,
                1 if local_extension_success else 0,
            )

        if remaining_needing_ext and not local_extension_success:
            if my_remaining_ext:
                # Extension failed — put failed sequences ON_HOLD to prevent
                # cache_seqlens from exceeding gpu_pages_allocated × PAGE_SIZE,
                # which would cause FlashAttention to read -1 sentinel page
                # indices and trigger CUDA illegal memory access.
                logging.warning(
                    f"Rank {ctx.facts.rank}: GPU page extension FAILED for "
                    f"{len(my_remaining_ext)} sequences at boundary — "
                    f"moving to ON_HOLD to prevent illegal memory access"
                )
                # Owner: release GPU pages
                ext_failed_local = IndexManager.get_local_indices_for_uuids(_index_request(ctx), my_remaining_ext)
                ext_failed_global = IndexManager.local_indices_to_global_seq_ids(_index_request(ctx), ext_failed_local)
                if ext_failed_global and gpu_manager is not None:
                    gpu_manager.free_pages_for_sequences(ext_failed_global)
                for uuid in my_remaining_ext:
                    ctx.gpu_resident.drop(uuid)

            # All ranks: zero scalars and update status for ALL failed seqs
            # (remaining_needing_ext is the globally-consistent list).
            ext_failed_set = set(remaining_needing_ext)
            for uuid in remaining_needing_ext:
                seq = ctx.sched.pool.get_sequence(uuid)
                seq.release_gpu_pages()
                seq.log_event(SeqEvent.ON_HOLD, ctx.facts.rank, "trigger=extension_failed")
            apply_or_raise(ctx.sched.pool, TransitionPlan(
                StatusEdge.BOUNDARY,
                tuple(
                    Transition(uuid, SequenceStatus.ON_HOLD)
                    for uuid in remaining_needing_ext
                ),
            ))

            decode_uuids = [u for u in decode_uuids if u not in ext_failed_set]
            batch = IndexManager.get_local_indices_for_uuids(_index_request(ctx), decode_uuids)

        timing.extension_ms = (time.perf_counter() - t0) * 1000

        # E. Async load (using rank-0 decisions)
        t0 = time.perf_counter()
        new_async_task = None
        new_load_uuids = decisions.new_load_uuids
        new_load_local = []
        new_load_global = []

        if new_load_uuids:
            # Pure DP loads on one assigned rank.  TP decode replicates the GPU
            # KV on every member of the sequence's serve-group, so every group
            # rank must allocate and launch its own local host->GPU load.
            my_new_uuids = [
                u for u in new_load_uuids
                if ctx.facts.rank_ctx.owns_sequence(ctx.sched.pool.get_sequence(u))
            ]
            try:
                new_load_local = IndexManager.get_local_indices_for_uuids(_index_request(ctx), my_new_uuids)
                local_load_pages = sum(
                    ctx.sched.pool.get_sequence(ctx.sched.seq_table.local_to_uuid[local_idx])
                    .get_gpu_pages_for_two_page_buffer()
                    for local_idx in new_load_local
                )
                local_load_free = (
                    gpu_manager.get_stats().num_free_pages
                    if gpu_manager is not None and gpu_manager.is_initialized else 0
                )
            except Exception as exc:
                logging.exception("Rank %s: boundary load snapshot failed", ctx.facts.rank)
                # The failed rank still joins the preflight so peers cannot
                # advance into allocation while this rank raises.
                kv._collective_gpu_kv_allocation_preflight(ctx, 1, 0)
                raise RuntimeError(
                    f"[SCHED_INVARIANT] boundary load snapshot failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            load_preflight_ok = kv._collective_gpu_kv_allocation_preflight(
                ctx,
                local_load_pages, local_load_free
            )
            if not load_preflight_ok:
                raise RuntimeError(
                    f"Rank {ctx.facts.rank}: collective GPU KV load preflight failed for "
                    f"{len(new_load_uuids)} globally selected sequences; "
                    "boundary capacity changed before allocation"
                )

            local_load_success = True
            if new_load_local:
                new_load_global = IndexManager.local_indices_to_global_seq_ids(_index_request(ctx), new_load_local)
                tokens = [
                    ctx.sched.pool.get_sequence(ctx.sched.seq_table.local_to_uuid[local_idx])
                    .get_gpu_pages_for_two_page_buffer() * ctx.facts.page_size
                    for local_idx in new_load_local
                ]

                try:
                    kv._allocate_gpu_pages_with_host_identity(
                        ctx,
                        gpu_manager, new_load_global, tokens
                    )
                except Exception:
                    # Convert allocator failures into the same collective result
                    # before any rank launches an async host-KV copy.
                    logging.exception(
                        "Rank %s: GPU-KV load allocation raised after collective preflight",
                        ctx.facts.rank,
                    )
                    local_load_success = False

            load_success = kv._collective_gpu_kv_allocation_preflight(
                ctx,
                0 if local_load_success else 1,
                1 if local_load_success else 0,
            )
            if not load_success:
                if local_load_success and new_load_global and gpu_manager is not None:
                    gpu_manager.free_pages_for_sequences(new_load_global)
                raise RuntimeError(
                    "[SCHED_INVARIANT] collective GPU-KV load allocation failed "
                    f"for {len(new_load_uuids)} globally selected sequences"
                )

            if new_load_local:
                timing.load_alloc_ms = (time.perf_counter() - t0) * 1000

                t_launch = time.perf_counter()
                if worker_view is not None:
                    existing_global_ids = IndexManager.local_indices_to_global_seq_ids(_index_request(ctx), batch)
                    if isinstance(gpu_manager, DualKVCacheCoordinator):
                        pointers = kv._prepare_dual_kv_load_pointers(
                            ctx,
                            gpu_manager, new_load_global, existing_global_ids
                        )
                        new_async_task = kv._launch_dual_host_kv_load(ctx, pointers)
                        async_load_tensors = pointers
                    else:
                        gpu_manager.rebuild_page_table(new_load_global)
                        k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
                        active_page_counts = gpu_manager.export_active_sequence_page_counts()
                        sequence_tensor = torch.tensor(new_load_global, dtype=torch.int64, device="cpu")
                        new_async_task = worker_view.async_load_layer_paged_kv_to_device(
                            sequence_ids=sequence_tensor,
                            active_page_counts=active_page_counts,
                            k_device_ptrs=k_ptrs,
                            v_device_ptrs=v_ptrs,
                        )
                        if existing_global_ids:
                            gpu_manager.rebuild_page_table(existing_global_ids)
                        async_load_tensors = {
                            'k_ptrs': k_ptrs, 'v_ptrs': v_ptrs,
                            'sequence_tensor': sequence_tensor,
                            'active_page_counts': active_page_counts,
                        }
                    timing.load_launch_ms = (time.perf_counter() - t_launch) * 1000

        timing.num_loaded = len(new_load_uuids)

        return (
            decode_uuids, batch, new_async_task, new_load_uuids,
            new_load_local, new_load_global, False, async_load_tensors,
        )
