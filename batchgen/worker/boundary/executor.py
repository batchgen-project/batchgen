"""BoundaryExecutor — apply a BoundaryPlan in canonical order.

The planner produces a list of :class:`PageBoundaryDecision`; the
executor applies them on every rank in exactly this order (POIS Q2):

    ReleasePages > Evict > OnHold > ExtendPages > AsyncLoadHostToGpu

Why grouped-by-type application rather than list order: the planner
emits decisions in any order, but page-budget math requires freeing
before spending. Release frees GPU pages held by completed sequences.
Evict frees both host and GPU pages when a seq can't expand its host
reservation. OnHold goes through ``HostKVRebalancer.put_on_hold`` (the
load-bearing Decision #2 ordering). Only after all three have run does
the executor spend remaining GPU pages on Extend and AsyncLoad.

Delegations:
  - Release / Extend → ``KVCacheManager`` directly.
  - OnHold → ``HostKVRebalancer.put_on_hold`` (enforces the 5-step
    flush → wait → release → transition → sync sequence).
  - Evict → M4 minimal: status transition + GPU release. Stashing
    ``evicted_token_ids`` + host-chunk release is M5 work alongside
    DecodeScheduler where the host reservation state lives.
  - AsyncLoadHostToGpu → M4 raises ``NotImplementedError``; lands
    alongside DecodeScheduler in M5 (host→GPU async handle machinery).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from batchgen.lifespan import SeqEvent
from batchgen.sequence import SequenceStatus
from batchgen.worker.boundary.decisions import (
    AsyncLoadHostToGpu,
    BoundaryPlan,
    Evict,
    ExtendPages,
    HostEvict,
    HostGrow,
    NewLoadAsync,
    OnHold,
    OnHoldReason,
    ReleasePages,
)
from batchgen.worker.boundary.synchronizer import (
    LoadCandidateState,
    SeqBoundaryState,
)
from batchgen.worker.host_rebalancer import HostKVRebalancer
from batchgen.worker.kv_manager import KVCacheManager
from batchgen.worker.protocols import UUID, LegacyInfraBackend
from batchgen.worker.state import WorkerState

if TYPE_CHECKING:
    import torch


class BoundaryExecutor:
    def __init__(
        self,
        state: WorkerState,
        kv: KVCacheManager,
        rebalancer: HostKVRebalancer,
    ) -> None:
        self._state = state
        self._kv = kv
        self._rebalancer = rebalancer

    def apply(self, plan: BoundaryPlan) -> None:
        """Apply every decision in the canonical order.

        Iterating plan.decisions_of(kind) is deliberate: the planner
        can emit the decisions in any order but the executor always
        frees pages before spending them. An empty decision list for
        any type is a no-op.
        """
        for decision in plan.decisions_of(ReleasePages):
            self._apply_release(decision)  # type: ignore[arg-type]
        for decision in plan.decisions_of(Evict):
            self._apply_evict(decision)  # type: ignore[arg-type]
        for decision in plan.decisions_of(OnHold):
            self._apply_onhold(decision)  # type: ignore[arg-type]
        for decision in plan.decisions_of(ExtendPages):
            self._apply_extend(decision)  # type: ignore[arg-type]
        for decision in plan.decisions_of(AsyncLoadHostToGpu):
            self._apply_async_load(decision)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Per-decision handlers
    # ------------------------------------------------------------------

    def _apply_release(self, decision: ReleasePages) -> None:
        """Free GPU pages for already-COMPLETED sequences.

        CompletionHandler has already transitioned these sequences to
        COMPLETED before the boundary runs, so the executor does not
        touch status here — it only returns GPU pages.
        """
        self._kv.release_pages(list(decision.uuids))

    def _apply_evict(self, decision: Evict) -> None:
        """M4 minimal: transition IN_DECODE/ON_HOLD → EVICTED + release GPU.

        Host-chunk release and evicted_token_ids stashing land in M5
        alongside DecodeScheduler — the chunk reservation state is
        DecodeScheduler's domain and wiring it through BoundaryExecutor
        before that handler exists would create an orphan code path.
        """
        to_release: list[str] = []
        for uuid in decision.uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            if seq.status in (SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD):
                self._state.global_batch.update_status(uuid, SequenceStatus.EVICTED)
                to_release.append(uuid)
        if to_release:
            self._kv.release_pages(to_release)

    def _apply_onhold(self, decision: OnHold) -> None:
        """Route through HostKVRebalancer.put_on_hold to preserve the
        plan Decision #2 5-step ordering (flush → wait → release →
        transition → sync). Never call put_on_hold elsewhere."""
        self._rebalancer.put_on_hold(list(decision.uuids))

    def _apply_extend(self, decision: ExtendPages) -> None:
        """Grow `uuid`'s GPU page allocation by `additional_pages`."""
        self._kv.extend_allocation(decision.uuid, decision.additional_pages)

    def _apply_async_load(self, decision: AsyncLoadHostToGpu) -> None:
        """**M5 stub**: async host→GPU transfer requires the async handle
        machinery DecodeScheduler owns. Raising here is intentional so
        the orchestrator cannot silently wire a planner that emits
        AsyncLoadHostToGpu before M5 lands."""
        raise NotImplementedError(
            "BoundaryExecutor._apply_async_load: AsyncLoadHostToGpu is "
            "deferred to M5 alongside DecodeScheduler.try_load_new"
        )

    # ==================================================================
    # apply_full — Phase 2.8.1e native boundary port
    # ==================================================================

    def apply_full(
        self,
        plan: BoundaryPlan,
        *,
        decode_uuids: list[UUID],
        batch: list[int],
        gpu_manager: Any,
        global_seq_state: dict[UUID, SeqBoundaryState],
        global_candidate_info: dict[UUID, LoadCandidateState],
        chunk_size: int,
        adapter: LegacyInfraBackend,
    ) -> tuple[
        list[UUID], list[int], Any, list[UUID], list[int], list[int]
    ]:
        """Legacy-faithful apply of a Stage 1 ``BoundaryPlan``.

        Ports ``_boundary_execute_decisions`` (batchgen_worker.py:6912-7131)
        and ``_boundary_async_load`` (7132-7223) as a single pass.
        Canonical apply order matches legacy:

            ReleasePages → HostGrow → HostEvict → OnHold → ExtendPages → NewLoadAsync

        Returns:
            ``(decode_uuids, batch, new_async_task, new_load_uuids,
               new_load_local, new_load_global)``. The tuple mirrors
            legacy ``_page_boundary_fast``'s return shape so the
            BoundaryHandler can thread the async handle through the
            next boundary cycle without reshape.
        """
        worker_view = adapter.host_paged_kv_worker_view()
        uuid_to_local = adapter.uuid_to_local_map()
        sequences_with_gpu_kv = adapter.sequences_with_gpu_kv()

        # --- A. ReleasePages: completed sequences ---
        completed_uuids: list[UUID] = []
        for decision in plan.decisions_of(ReleasePages):
            completed_uuids.extend(decision.uuids)  # type: ignore[attr-defined]
        if completed_uuids:
            adapter.update_batch_status(
                completed_uuids, SequenceStatus.COMPLETED
            )
            adapter.submit_completed_to_incremental_writer(completed_uuids)
            gathered_texts = adapter.gather_completed_tokens(completed_uuids)

            my_completed = [u for u in completed_uuids if u in uuid_to_local]
            if my_completed:
                my_completed_local = adapter.get_local_indices_for_uuids(
                    my_completed
                )
                adapter.release_gpu_kv_pages(my_completed_local)
                adapter.release_host_kv_pages_for_batch(my_completed)

            for uuid in completed_uuids:
                seq = self._state.global_batch.get_sequence(uuid)
                if seq is not None:
                    seq.gpu_pages_allocated = 0
                    seq.host_pages_allocated = 0
                    seq.host_token_capacity = 0
                sequences_with_gpu_kv.discard(uuid)

            for uuid in completed_uuids:
                adapter.report_completion(uuid, gathered_texts.get(uuid))

            for uuid in completed_uuids:
                st = global_seq_state.get(uuid)
                if st is not None:
                    adapter.report_chunk_sizer_completion(st.decoded_length)

            completed_set = set(completed_uuids)
            decode_uuids = [u for u in decode_uuids if u not in completed_set]
            batch = adapter.get_local_indices_for_uuids(decode_uuids)

        # --- B. HostGrow: per-uuid host KV growth ---
        for decision in plan.decisions_of(HostGrow):
            grow: HostGrow = decision  # type: ignore[assignment]
            if not grow.feasible or not grow.uuids:
                continue
            host_grow_requests: list[tuple[int, int]] = []
            for uuid, growth_pages in zip(grow.uuids, grow.pages):
                seq = self._state.global_batch.get_sequence(uuid)
                if seq is None:
                    continue
                seq.host_token_capacity += growth_pages * seq.PAGE_SIZE
                seq.host_pages_allocated += growth_pages
                if uuid in uuid_to_local:
                    host_grow_requests.append((seq.global_idx, growth_pages))
            if host_grow_requests and worker_view is not None:
                worker_view.grow_pages_for_sequences(host_grow_requests)

        # --- C. HostEvict: per-uuid host-KV eviction → EVICTED ---
        evicted_drop: set[UUID] = set()
        for decision in plan.decisions_of(HostEvict):
            evict: HostEvict = decision  # type: ignore[assignment]
            if not evict.uuids:
                continue
            my_evicted = [u for u in evict.uuids if u in uuid_to_local]
            if my_evicted:
                my_evicted_local = adapter.get_local_indices_for_uuids(my_evicted)
                adapter.release_gpu_kv_pages(my_evicted_local)
                evicted_global_ids: list[int] = []
                for uuid in my_evicted:
                    seq = self._state.global_batch.get_sequence(uuid)
                    if seq is None:
                        continue
                    # Stash evicted_token_ids = prompt_tokens + new_decoded.
                    # Legacy logic at batchgen_worker.py:7011-7020.
                    self._build_evicted_token_ids(seq)
                    evicted_global_ids.append(seq.global_idx)
                if worker_view is not None and evicted_global_ids:
                    worker_view.release_sequence_pages(evicted_global_ids)
                    worker_view.unregister_sequences(evicted_global_ids)

            # All ranks: update scalar metadata deterministically + status
            for uuid in evict.uuids:
                seq = self._state.global_batch.get_sequence(uuid)
                if seq is None:
                    continue
                baseline = seq.reentry_decoded_baseline
                new_decoded_count = max(0, seq.decoded_length - baseline)
                new_reentry_len = seq.prompt_length + new_decoded_count
                seq.total_decoded_before_eviction = seq.decoded_length
                seq.prompt_length = new_reentry_len
                seq.current_context_length = new_reentry_len
                seq.log_event(
                    SeqEvent.EVICTED,
                    self._state.rank,
                    f"saved_tokens={new_reentry_len}, "
                    f"decoded={seq.decoded_length}, "
                    f"new_this_cycle={new_decoded_count}",
                )
                seq.gpu_pages_allocated = 0
                seq.host_pages_allocated = 0
                seq.host_token_capacity = 0
                sequences_with_gpu_kv.discard(uuid)
                try:
                    self._state.global_batch.update_status(
                        uuid, SequenceStatus.EVICTED
                    )
                except ValueError:
                    # Invalid transition — leave status as-is; the
                    # state-machine invariant caught an upstream bug.
                    # Re-raise so the handler's guards surface it.
                    raise
            evicted_drop.update(evict.uuids)

        if evicted_drop:
            decode_uuids = [u for u in decode_uuids if u not in evicted_drop]
            batch = adapter.get_local_indices_for_uuids(decode_uuids)

        if not decode_uuids:
            return decode_uuids, batch, None, [], [], []

        # --- D. OnHold: watermark trigger + extension failure ---
        onhold_drop: set[UUID] = set()
        for decision in plan.decisions_of(OnHold):
            onhold: OnHold = decision  # type: ignore[assignment]
            if not onhold.uuids:
                continue
            self._apply_onhold_full(
                onhold.uuids,
                reason=onhold.reason,
                gpu_manager=gpu_manager,
                adapter=adapter,
            )
            onhold_drop.update(onhold.uuids)

        if onhold_drop:
            decode_uuids = [u for u in decode_uuids if u not in onhold_drop]
            batch = adapter.get_local_indices_for_uuids(decode_uuids)

        # --- E. ExtendPages: per-rank GPU page growth ---
        ext_decisions = plan.decisions_of(ExtendPages)
        my_remaining_ext: list[UUID] = [
            d.uuid  # type: ignore[attr-defined]
            for d in ext_decisions
            if d.uuid in uuid_to_local  # type: ignore[attr-defined]
        ]
        ext_failed_drop: set[UUID] = set()
        if my_remaining_ext:
            success = adapter.extend_gpu_kv_allocation(my_remaining_ext)
            if not success:
                # Legacy's fallback: free the half-allocated pages and
                # force the sequences we tried to extend to ON_HOLD so
                # decode doesn't touch them next step (batchgen_worker.py:
                # 7106-7126).
                ext_failed_local = adapter.get_local_indices_for_uuids(
                    my_remaining_ext
                )
                ext_failed_global = adapter.local_indices_to_global_seq_ids(
                    ext_failed_local
                )
                if ext_failed_global and gpu_manager is not None:
                    gpu_manager.free_pages_for_sequences(ext_failed_global)
                for uuid in my_remaining_ext:
                    sequences_with_gpu_kv.discard(uuid)
                # All-ranks subset of ext decisions gets forced to ON_HOLD
                all_ext_uuids = [
                    d.uuid for d in ext_decisions  # type: ignore[attr-defined]
                ]
                ext_failed_drop.update(all_ext_uuids)
                for uuid in all_ext_uuids:
                    seq = self._state.global_batch.get_sequence(uuid)
                    if seq is None:
                        continue
                    seq.gpu_pages_allocated = 0
                    seq.log_event(
                        SeqEvent.ON_HOLD, self._state.rank,
                        "trigger=extension_failed",
                    )
                    try:
                        self._state.global_batch.update_status(
                            uuid, SequenceStatus.ON_HOLD
                        )
                    except ValueError:
                        # Not IN_DECODE — status machine rejects.
                        # Surface to guards by re-raising.
                        raise

        if ext_failed_drop:
            decode_uuids = [u for u in decode_uuids if u not in ext_failed_drop]
            batch = adapter.get_local_indices_for_uuids(decode_uuids)

        # --- F. NewLoadAsync: async host→GPU load ---
        new_async_task: Any = None
        new_load_uuids: list[UUID] = []
        new_load_local: list[int] = []
        new_load_global: list[int] = []
        for decision in plan.decisions_of(NewLoadAsync):
            load: NewLoadAsync = decision  # type: ignore[assignment]
            if not load.uuids:
                continue
            new_load_uuids = list(load.uuids)
            if decode_uuids:
                (
                    new_async_task,
                    new_load_local,
                    new_load_global,
                ) = self._apply_new_load_async(
                    new_load_uuids,
                    batch=batch,
                    gpu_manager=gpu_manager,
                    global_candidate_info=global_candidate_info,
                    adapter=adapter,
                    worker_view=worker_view,
                )
            break  # only one NewLoadAsync decision per plan

        return (
            decode_uuids,
            batch,
            new_async_task,
            new_load_uuids,
            new_load_local,
            new_load_global,
        )

    # ------------------------------------------------------------------
    # Sub-helpers for apply_full
    # ------------------------------------------------------------------

    def _build_evicted_token_ids(self, seq: Any) -> None:
        """Port of legacy batchgen_worker.py:7011-7020.

        Stashes ``evicted_token_ids = prompt_tokens + new_decoded`` on
        the sequence so prefill re-entry can rebuild context. Called
        only on the rank that owns the sequence.
        """
        import torch

        if seq.input_ids is None:
            return
        prompt_tokens = seq.input_ids[0, : seq.prompt_length]
        baseline = seq.reentry_decoded_baseline
        if seq.decoded_tokens is not None and seq.decoded_length > baseline:
            new_decoded = seq.decoded_tokens[0, baseline : seq.decoded_length]
            seq.evicted_token_ids = torch.cat([prompt_tokens, new_decoded])
        else:
            seq.evicted_token_ids = prompt_tokens.clone()

    def _apply_onhold_full(
        self,
        uuids: tuple[UUID, ...],
        *,
        reason: OnHoldReason,
        gpu_manager: Any,
        adapter: LegacyInfraBackend,
    ) -> None:
        """Port of the OnHold branch in _boundary_execute_decisions
        (batchgen_worker.py:7074-7091). Applies on every rank; local
        GPU page release only happens for rank-owned uuids.
        """
        uuid_to_local = adapter.uuid_to_local_map()
        sequences_with_gpu_kv = adapter.sequences_with_gpu_kv()
        my_onhold = [u for u in uuids if u in uuid_to_local]
        if my_onhold:
            local_indices = adapter.get_local_indices_for_uuids(my_onhold)
            global_ids = adapter.local_indices_to_global_seq_ids(local_indices)
            if global_ids and gpu_manager is not None:
                gpu_manager.free_pages_for_sequences(global_ids)
            for uuid in my_onhold:
                sequences_with_gpu_kv.discard(uuid)
        trigger = (
            "watermark"
            if reason is OnHoldReason.WATERMARK_TRIGGER
            else "extension_failed"
        )
        for uuid in uuids:
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            seq.gpu_pages_allocated = 0
            seq.log_event(
                SeqEvent.ON_HOLD, self._state.rank, f"trigger={trigger}"
            )
            try:
                self._state.global_batch.update_status(
                    uuid, SequenceStatus.ON_HOLD
                )
            except ValueError:
                # The planner's IN_DECODE-only filter (Stage 1d) should
                # have prevented this; re-raise so the guard surfaces
                # the upstream bug hard.
                raise

    def _apply_new_load_async(
        self,
        new_load_uuids: list[UUID],
        *,
        batch: list[int],
        gpu_manager: Any,
        global_candidate_info: dict[UUID, LoadCandidateState],
        adapter: LegacyInfraBackend,
        worker_view: Any,
    ) -> tuple[Any, list[int], list[int]]:
        """Port of _boundary_async_load (batchgen_worker.py:7132-7222).

        Filters by actual GPU free pages (re-reading the manager after
        the earlier allocations) and kicks off
        ``worker_view.async_load_layer_paged_kv_to_device``. Stashes
        tensors on ``self._async_load_tensors`` so the next cycle's
        ``wait_pending`` can finalize without lifetime concerns.
        """
        import torch

        local_to_uuid = adapter.local_to_uuid_map()
        my_new_uuids = [
            u for u in new_load_uuids
            if (info := global_candidate_info.get(u)) is not None
            and info.assigned_rank == self._state.rank
        ]
        new_load_local = adapter.get_local_indices_for_uuids(my_new_uuids)
        if not new_load_local:
            return None, [], []

        actual_free = 0
        if gpu_manager is not None and getattr(gpu_manager, "is_initialized", False):
            actual_free = int(gpu_manager.get_stats().num_free_pages)

        filtered_local: list[int] = []
        filtered_global: list[int] = []
        filtered_tokens: list[int] = []
        pages_used = 0
        for local_idx in new_load_local:
            uuid = local_to_uuid.get(local_idx)
            if uuid is None:
                continue
            seq = self._state.global_batch.get_sequence(uuid)
            if seq is None:
                continue
            pages_needed = seq.get_gpu_pages_for_two_page_buffer()
            if pages_used + pages_needed <= actual_free:
                filtered_local.append(local_idx)
                filtered_global.append(seq.global_idx)
                filtered_tokens.append(pages_needed * seq.PAGE_SIZE)
                pages_used += pages_needed

        if not filtered_local:
            return None, [], []

        gpu_manager.allocate_pages_for_sequences(
            filtered_global, filtered_tokens
        )

        if worker_view is None:
            # No host view wired: we allocated GPU pages but can't
            # launch the async load. Return the allocations so the
            # caller sees the new local indices, but no async handle.
            return None, filtered_local, filtered_global

        existing_global_ids = adapter.local_indices_to_global_seq_ids(batch)
        gpu_manager.rebuild_page_table(filtered_global)
        k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
        active_page_counts = gpu_manager.export_active_sequence_page_counts()
        sequence_tensor = torch.tensor(
            filtered_global, dtype=torch.int64, device="cpu"
        )

        new_async_task = worker_view.async_load_layer_paged_kv_to_device(
            sequence_ids=sequence_tensor,
            active_page_counts=active_page_counts,
            k_device_ptrs=k_ptrs,
            v_device_ptrs=v_ptrs,
        )

        if existing_global_ids:
            gpu_manager.rebuild_page_table(existing_global_ids)

        # Keep tensors alive until the async load finalizes next
        # boundary cycle. Legacy stashes on self._async_load_tensors.
        self._async_load_tensors = {
            "k_ptrs": k_ptrs,
            "v_ptrs": v_ptrs,
            "sequence_tensor": sequence_tensor,
            "active_page_counts": active_page_counts,
        }

        return new_async_task, filtered_local, filtered_global


__all__ = ["BoundaryExecutor"]
