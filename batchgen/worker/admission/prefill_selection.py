"""Selection of the prefill admission wave's candidate snapshot.

These module-level helpers run, on every rank, the pure parts of prefill
admission that surround the gathered capacity collective in
``PrefillEngine._prepare_prefill_batch``: the candidate digest the admission
gather cross-checks, and the frozen ``PrefillSelectionRequest`` snapshot the
pure scheduler consumes. Each reads the rank's state from the prefill phase
context and never the worker; the worker keeps same-name delegators. The
shared-prefix segments come in as a VALUE the delegator resolved, because
their resolution reads worker state (the trajectory token store, the
prefix-pool gating) and the context never carries a bound worker method.

Still on the worker: ``_reserve_prefill_host_kv`` (its writes interleave with
the capacity collectives and its helpers take the worker),
``_reject_unsatisfiable_prefill_requests`` (its final capacity publication is
gated on a condition its return value does not carry), and the two-line
``_prefill_admission_guard_state`` (a pure read of two worker fields).
"""

from __future__ import annotations

# Declared contract, checked against this module's source by
# tests/worker/test_admission_module_contracts.py: what this module
# reads off the objects it is given, what it writes on them, what it
# mutates through them, and the orderings it relies on. A change here
# that the source does not match, or a source change this does not
# cover, fails that test.
MODULE_CONTRACT = {
    "reads": (
        "ctx.facts",
        "ctx.prefix",
        "ctx.sched",
        "shared_segments.get",
    ),
    "writes": (),
    "mutates": (),
    "orderings": (
        "nothing: the snapshot is a pure function of the candidates it is given.",
    ),
}

import hashlib
from typing import Dict, List, Tuple

from batchgen.sequence import SequenceStatus
import batchgen.sequence as _sequence_module
from batchgen.worker.prefill import (
    PrefillCandidate,
    PrefillSelectionRequest,
)


def _prefill_candidate_digest(ctx, candidate_uuids: List[str]) -> int:
    """Digest the inputs the prefill candidate snapshot is derived from.

    Covers identity and ORDER plus every field
    ``_make_prefill_selection_request`` reads, including the two that decide
    a candidate's node (``assigned_rank``, ``decode_dp_group``). Gathered by
    the admission collective so a divergence is named instead of letting one
    rank reclaim, raise, or enter a collective the others do not.
    """
    hasher = hashlib.blake2b(digest_size=7)
    for uuid in candidate_uuids:
        seq = ctx.sched.pool.get_sequence(uuid)
        hasher.update(
            f"{uuid}:{int(seq.status)}:{seq.global_idx}:{seq.prompt_length}:"
            f"{seq.kv_token_budget}:{seq.PAGE_SIZE}:{seq.assigned_rank}:"
            f"{seq.decode_dp_group}:"
            f"{getattr(seq, 'prefill_order', None)};".encode()
        )
    return int.from_bytes(hasher.digest(), "little")


def _make_prefill_selection_request(
    ctx, all_candidates: List[str],
    shared_segments: Dict[str, Tuple[Tuple[int, int], ...]],
    per_node_host_free: List[int],
    num_nodes: int, chunk_size: int, *,
    per_rank_sequence_free=None, per_node_sequence_free=None,
    per_node_host_reclaimable=None,
    kda_record_pages: int = 0,
) -> PrefillSelectionRequest:
    """Snapshot the candidate metadata `select_prefill_batch` consumes."""
    ctx.prefix.note_wave_credit(bool(shared_segments))
    candidates = []
    for uuid in all_candidates:
        seq = ctx.sched.pool.get_sequence(uuid)
        seq_node = ctx.facts.rank_ctx.node_of(seq.assigned_rank)
        G = ctx.facts.attn_tp_size
        if G > 1:
            if seq.decode_dp_group is None:
                raise RuntimeError(
                    f"sequence {uuid[:8]} has no decode_dp_group before "
                    "TP prefill selection"
                )
            from batchgen.decode_dp_group import host_kv_owner_rank
            seq_node = ctx.facts.rank_ctx.node_of(
                host_kv_owner_rank(seq.decode_dp_group, G)
            )
        candidates.append(PrefillCandidate(
            uuid=uuid,
            assigned_rank=seq.assigned_rank,
            node_id=seq_node,
            is_evicted=(seq.status == SequenceStatus.EVICTED),
            global_idx=seq.global_idx,
            total_decoded_before_eviction=getattr(
                seq, "total_decoded_before_eviction", 0
            ),
            prompt_length=seq.prompt_length,
            kv_token_budget=seq.kv_token_budget,
            page_size=seq.PAGE_SIZE,
            # G>1 no longer replicates the reservation: only the serve-group
            # leader (_owns_host_kv) registers / allocates in the per-node
            # shared host region and the other G-1 ranks mirror its page
            # table (_mirror_group_host_kv), so one admitted sequence costs
            # the node allocator exactly one copy of req_pages.
            host_kv_replication_factor=1,
            prefill_order=getattr(seq, "prefill_order", None),
            shared_prefix_segments=shared_segments.get(uuid, ()),
        ))
    return PrefillSelectionRequest(
        candidates=tuple(candidates),
        per_node_host_free=tuple(per_node_host_free),
        chunk_size=chunk_size,
        num_nodes=num_nodes,
        gpus_per_node=ctx.facts.local_world_size,
        initial_gpu_page_buffer=_sequence_module.INITIAL_GPU_PAGE_BUFFER,
        per_rank_sequence_free=(
            tuple(per_rank_sequence_free)
            if per_rank_sequence_free is not None else None
        ),
        per_node_sequence_free=(
            tuple(per_node_sequence_free)
            if per_node_sequence_free is not None else None
        ),
        per_node_host_reclaimable=(
            tuple(per_node_host_reclaimable)
            if per_node_host_reclaimable is not None else None
        ),
        # 方案甲: each admitted sequence also reserves its KDA record pages
        # in the combined host budget. A VALUE the delegator resolved (the
        # record store is worker state), like the shared segments above.
        kda_record_pages=int(kda_record_pages),
    )
