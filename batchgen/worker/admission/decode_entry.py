"""Selection of the decode entry batch.

``DecodeEntry.select`` runs, on every rank, the decode admission that opens a
decode wave: candidate enumeration from the replicated sequence statuses, the
collective live capacity snapshot, the owner-reported shared-prefix gather,
and the pure ``DecodeScheduler.select_decode_batch`` call. It reads the rank's
state from the decode context and reaches the KV capacity engine with that
same context. Model-specific admission guards and the rank-0 summary log stay
with the caller.
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
        "capacity.free_pages",
        "capacity.total_pages",
        "ctx.facts",
        "ctx.gpu_kv",
        "ctx.sched",
        "shared_segments.get",
    ),
    "writes": (),
    "writes_via_local": (),
    "mutates": (
        "all_candidates.sort()",
    ),
    "orderings": (
        "the candidate list is sorted before it is cut, so the batch is a prefix of one order and not of whichever order arrived.",
    ),
}

from typing import Dict, List, Optional, Tuple

import torch.distributed as dist

from batchgen.sequence import SequenceStatus
from batchgen.worker.decode import (
    DecodeBatchRequest,
    DecodeCandidate,
    DecodeCapacitySnapshot,
    DecodeScheduler,
)
from batchgen.worker.kv_capacity_engine import KVCapacityEngine
from batchgen.worker.shared_prefix_charge import plan_shared_runs


def _decode_shared_prefix_credit_enabled(ctx) -> bool:
    """Whether decode admission may charge a shared GPU prefix once.

    The credit is only honest where the ALLOCATOR shares: GPU pages are
    deduplicated by Host page id inside
    ``allocate_pages_for_sequences_with_page_keys``, which
    ``_allocate_gpu_pages_with_host_identity`` reaches exactly when
    ``_shared_page_identity_sources`` names the pools to key. A DSA dual
    coordinator is now one of those: both of its pools deduplicate, each
    on its own Host region's page ids. Everywhere else each row really
    does load its own copy, so each row must pay for it -- the same
    condition ``_host_identity_page_keys`` tests before returning page
    keys at all.
    """
    if not ctx.facts.shared_prefix_admission:
        return False
    if not ctx.facts.enable_prefix_cache:
        return False
    manager = ctx.gpu_kv
    if manager is None or not manager.is_initialized:
        return False
    return KVCapacityEngine()._shared_page_identity_sources(ctx, manager) is not None


def _local_decode_shared_prefix_runs(
    ctx, all_candidates: List[str]
) -> Dict[str, Tuple[Tuple[int, int], ...]]:
    """Shared leading page runs among the candidates THIS rank owns.

    Grouped per owner because sharing only pays inside one GPU pool: the
    pool belongs to one DP rank (or one replicated TP group whose Host-KV
    leader owns every row), so two rows loading onto different pools each
    need their own physical copy.

    The identity is the HOST page id, which is what the allocator dedups
    on, so a run here means those rows provably reference one physical
    copy -- not merely equal tokens, which two distinct Host copies can
    also have.

    The view comes from ``_shared_page_identity_sources``, so the runs are
    read from exactly the key space the allocator will dedup on. On a DSA
    dual coordinator that is the PRIMARY pool (the MLA latent region): the
    auxiliary pool shares the same rows page for page, and the charge is
    counted in primary-pool pages, so one pool's identity describes the
    credit for both.
    """
    sources = KVCapacityEngine()._shared_page_identity_sources(
        ctx, ctx.gpu_kv
    )
    if not sources:
        return {}
    worker_view = sources[0][2]
    mine = []
    for uuid in all_candidates:
        if uuid not in ctx.sched.seq_table.uuid_to_local:
            continue
        seq = ctx.sched.pool.get_sequence(uuid)
        req_pages = int(seq.get_gpu_pages_for_two_page_buffer())
        if req_pages <= 1:
            continue
        mine.append((uuid, seq, req_pages))
    if len(mine) < 2:
        return {}
    host_tables = worker_view.build_page_table(
        [int(seq.global_idx) for _, seq, _ in mine]
    )
    tables: Dict[int, List[int]] = {}
    for (_uuid, seq, req_pages), host_pages in zip(mine, host_tables):
        # Credit only pages that exist, and never the row's whole
        # reservation: one page stays private so no row is charged zero.
        # A Host table shorter than the reservation is named loudly by
        # _host_identity_page_keys at allocation time; admission must not
        # credit pages it cannot see.
        usable = min(req_pages - 1, len(host_pages))
        if usable <= 0:
            continue
        tables[int(seq.global_idx)] = [
            int(page) for page in host_pages[:usable]
        ]
    runs = plan_shared_runs(tables)
    local: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    for uuid, seq, _req_pages in mine:
        run = runs.get(int(seq.global_idx))
        if run is None:
            continue
        local[uuid] = ((int(run[0]), int(run[1])),)
    return local


def _gather_decode_shared_prefix_segments(
    ctx, all_candidates: List[str]
) -> Dict[str, Tuple[Tuple[int, int], ...]]:
    """Collect every rank's shared runs so selection stays rank-identical.

    ``select_decode_batch`` is pure and runs on every rank; a rank that
    credited a run its peers did not would admit a different batch and the
    next collective would deadlock. Only an owner can read its rows' Host
    page tables, so the ground truth is gathered rather than re-derived.
    ``all_candidates`` is already rank-uniform (it comes from replicated
    sequence status), so every rank reaches this gather together.
    """
    if not _decode_shared_prefix_credit_enabled(ctx):
        return {}
    local = _local_decode_shared_prefix_runs(ctx, all_candidates)
    if ctx.facts.world_size == 1:
        return local
    if not dist.is_initialized():
        raise RuntimeError(
            "distributed process group is required to gather decode "
            "shared-prefix runs"
        )
    gathered: List[Optional[Dict[str, Tuple[Tuple[int, int], ...]]]] = (
        [None] * ctx.facts.world_size
    )
    dist.all_gather_object(gathered, local)
    merged: Dict[str, Tuple[Tuple[int, int], ...]] = {}
    for part in gathered:
        if part:
            merged.update(part)
    return merged


def _make_decode_batch_request(
    ctx, all_candidates: List[str], capacity: DecodeCapacitySnapshot,
    max_rank_bsz: int,
    shared_segments: Optional[Dict[str, Tuple[Tuple[int, int], ...]]] = None,
) -> DecodeBatchRequest:
    """Snapshot the candidate metadata `select_decode_batch` consumes."""
    shared_segments = shared_segments or {}
    attn_tp_size = ctx.facts.attn_tp_size
    num_capacity_groups = ctx.facts.world_size // attn_tp_size
    existing_pages = [0] * num_capacity_groups
    existing_sequence_counts = [0] * num_capacity_groups
    for uuid in ctx.sched.pool.get_sequences_by_status(SequenceStatus.IN_DECODE):
        seq = ctx.sched.pool.get_sequence(uuid)
        if attn_tp_size > 1:
            r = seq.decode_dp_group
            if r is None:
                raise ValueError(
                    f"sequence {uuid[:8]} has no decode_dp_group for "
                    f"attn_tp_size={attn_tp_size}"
                )
        else:
            r = seq.assigned_rank
        if not 0 <= r < num_capacity_groups:
            raise ValueError(
                f"sequence {uuid[:8]} has invalid decode capacity group {r}"
            )
        existing_pages[r] += max(0, seq.gpu_pages_allocated)
        existing_sequence_counts[r] += 1

    candidates = []
    for uuid in all_candidates:
        seq = ctx.sched.pool.get_sequence(uuid)
        candidates.append(DecodeCandidate(
            uuid=uuid,
            assigned_rank=seq.assigned_rank,
            global_idx=seq.global_idx,
            req_pages=seq.get_gpu_pages_for_two_page_buffer(),
            decode_dp_group=seq.decode_dp_group,
            shared_prefix_segments=shared_segments.get(uuid, ()),
        ))
    return DecodeBatchRequest(
        candidates=tuple(candidates),
        total_pages=capacity.total_pages,
        world_size=ctx.facts.world_size,
        max_rank_bsz=max_rank_bsz,
        attn_tp_size=attn_tp_size,
        existing_pages=tuple(existing_pages),
        free_pages=capacity.free_pages,
        existing_sequence_counts=tuple(existing_sequence_counts),
    )


class DecodeEntry:
    """Stateless: everything it touches comes in through ``select``."""

    def select(self, ctx, max_rank_bsz: int) -> List[str]:
        """
        Select sequences for decode phase from PREFILLED sequences.
        Greedily fill the current physical GPU-KV free capacity.
        """
        prefilled_uuids = ctx.sched.pool.get_sequences_by_status(SequenceStatus.PREFILLED)
        onhold_uuids = ctx.sched.pool.get_sequences_by_status(SequenceStatus.ON_HOLD)

        # Combine and sort for deterministic ordering
        all_candidates = prefilled_uuids + onhold_uuids
        all_candidates.sort(key=lambda uuid: ctx.sched.pool.get_sequence(uuid).global_idx)

        # Get a collective live capacity snapshot.  The old implementation used
        # only one rank's total_pages and a metadata-derived resident count, which
        # could admit against stale state or strand a candidate at the 90% guard.
        capacity = KVCapacityEngine()._snapshot_decode_capacity(ctx)
        if not all_candidates:
            return []

        # Candidate enumeration stays here; pure physical-capacity
        # selection is delegated.
        # Owner-reported shared GPU prefixes, gathered before the pure
        # selection so every rank charges the same marginal cost: rows that
        # reference one Host page get ONE physical GPU page, so charging each
        # row its whole context would bound decode by the logical page sum.
        shared_segments = _gather_decode_shared_prefix_segments(
            ctx, all_candidates
        )
        decode_batch = DecodeScheduler.select_decode_batch(
            _make_decode_batch_request(
                ctx, all_candidates, capacity, max_rank_bsz,
                shared_segments=shared_segments,
            )
        )

        return decode_batch
