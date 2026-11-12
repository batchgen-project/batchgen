"""Decode scheduling — GPU-capacity-bounded decode batch selection.

Slice 9 of the worker decouple initiative (issue #175) — the final
slice. Extracts the pure *selection decision* from
``_prepare_decode_batch``:

  - ``DecodeScheduler.select_decode_batch`` — greedily admit PREFILLED /
    ON_HOLD sequences into the decode batch (ordered by ``global_idx``),
    bounded by the current physical free pages of each GPU-KV replica.

Only the *decision* is ported. The candidate enumeration over
``global_batch`` and the ``gpu_paged_kv_cache_manager.get_stats()`` query
(for total pages) stay on the worker; everything downstream of selection
(model forward, KV streaming, metadata binding) is irreducible side
effects that remain on the worker too.

By this slice the decode step's other decisions already live in sibling
handlers: completion (``CompletionHandler``, Slice 2), the rank-0 page
boundary (``BoundaryHandler``, Slice 8), prefill admission
(``PrefillScheduler``, Slice 6), and the watermark trigger
(``KVCacheManager``, Slice 5.5). This slice covers the remaining pure
piece — assembling the decode batch itself.

Design follows the per-slice frozen-snapshot pattern: pure, deterministic
across ranks (candidates sorted by ``global_idx``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from batchgen.worker.shared_prefix_charge import (
    SharedPrefixLedger,
    segments_from_pairs,
)


class DecodeCapacityError(RuntimeError):
    """A candidate cannot fit in the physical GPU-KV capacity."""


@dataclass(frozen=True)
class DecodeCandidate:
    """A PREFILLED / ON_HOLD sequence eligible for the decode batch."""

    uuid: str
    assigned_rank: int
    global_idx: int
    req_pages: int  # GPU pages for its two-page-buffer reservation
    decode_dp_group: Optional[int] = None
    # ``(key, pages)`` leading run this row reads from a GPU page copy SHARED
    # with other candidates. The GPU allocator maps ONE physical page per
    # distinct Host page id (``allocate_pages_for_sequences_with_page_keys``),
    # so rows of one shared prefix cost the pool one copy, not one each.
    # Empty means "charge this row in full".
    shared_prefix_segments: Tuple[Tuple[int, int], ...] = ()
    # Leading pages of this row that the group's pool ALREADY maps, because a
    # sequence already decoding shares them. Joining them costs nothing.
    resident_shared_pages: int = 0


@dataclass(frozen=True)
class DecodeBatchRequest:
    """Frozen snapshot for ``select_decode_batch``.

    ``total_pages`` is the GPU paged-KV manager's per-rank total page count.
    ``free_pages`` is the live free-page snapshot reduced to one value per
    DP/TP capacity group.  It is the admission source of truth; sequence
    metadata is only a diagnostic snapshot because it can lag allocator state.
    Under TP decode, all ranks of a decode group replicate the same sequences,
    so candidates consume one shared capacity bucket per group.
    """

    candidates: Tuple[DecodeCandidate, ...]
    total_pages: int
    world_size: int
    # Per-rank in-decode sequence cap (= the MoE buffer's num_tokens_per_rank = mtp/world_size).
    # Bounds the global decode batch to <= mtp so the pre-reserved padded MoE buffers never
    # overflow (no runtime resize -> no OOM). 0 = unlimited (legacy / non-K2.5).
    max_rank_bsz: int = 0
    attn_tp_size: int = 1
    # GPU pages already held by IN_DECODE rows in each DP/TP capacity group.
    # Kept for diagnostics and backwards-compatible pure callers.  Admission
    # uses ``free_pages`` when supplied, rather than deriving free space from
    # this potentially stale metadata.
    existing_pages: Tuple[int, ...] = ()
    # Current allocator free pages per DP/TP capacity group.  Production
    # callers must provide this collective snapshot.
    free_pages: Tuple[int, ...] = ()
    # Number of IN_DECODE rows already resident in each capacity group.  This
    # count is part of the padded-buffer limit and must include existing rows.
    existing_sequence_counts: Tuple[int, ...] = ()


@dataclass(frozen=True)
class DecodeCapacitySnapshot:
    """Validated live GPU capacity reduced across ranks.

    ``free_pages`` is the minimum free count in each replicated TP group (or
    the rank-local count for pure DP).  All ranks must agree on total pages;
    a mismatch means the allocator configuration is already divergent and is
    rejected before any candidate is selected.
    """

    total_pages: int
    free_pages: Tuple[int, ...]
    rank_total_pages: Tuple[int, ...]
    rank_free_pages: Tuple[int, ...]


def reduce_decode_capacity_snapshot(
    rank_total_pages: Tuple[int, ...],
    rank_free_pages: Tuple[int, ...],
    *,
    world_size: int,
    attn_tp_size: int,
) -> DecodeCapacitySnapshot:
    """Validate and reduce a rank-wide live GPU capacity snapshot.

    Total page count is immutable after GPU-KV initialization, so every rank
    must report the same value.  Free pages are dynamic; for replicated TP
    groups the tightest rank is the only safe capacity for that group.
    """
    if world_size <= 0:
        raise ValueError(f"world_size={world_size} must be positive")
    if len(rank_total_pages) != world_size or len(rank_free_pages) != world_size:
        raise ValueError(
            "capacity snapshot length must equal world_size: "
            f"totals={len(rank_total_pages)}, free={len(rank_free_pages)}, "
            f"world_size={world_size}"
        )
    if attn_tp_size <= 0 or world_size % attn_tp_size != 0:
        raise ValueError(
            f"attn_tp_size={attn_tp_size} must divide world_size={world_size}"
        )
    if any(total < 0 for total in rank_total_pages):
        raise ValueError(f"negative GPU-KV total pages: {rank_total_pages}")
    if any(free < 0 for free in rank_free_pages):
        raise ValueError(f"negative GPU-KV free pages: {rank_free_pages}")
    if any(free > total for total, free in zip(rank_total_pages, rank_free_pages)):
        raise ValueError(
            "GPU-KV free pages exceed total pages: "
            f"totals={rank_total_pages}, free={rank_free_pages}"
        )
    if len(set(rank_total_pages)) != 1:
        raise ValueError(
            "GPU-KV total page count diverged across ranks: "
            f"{rank_total_pages}"
        )

    num_groups = world_size // attn_tp_size
    free_pages = tuple(
        min(rank_free_pages[g * attn_tp_size:(g + 1) * attn_tp_size])
        for g in range(num_groups)
    )
    return DecodeCapacitySnapshot(
        total_pages=rank_total_pages[0],
        free_pages=free_pages,
        rank_total_pages=tuple(rank_total_pages),
        rank_free_pages=tuple(rank_free_pages),
    )


class DecodeScheduler:
    """Decode batch admission decision — pure, deterministic across ranks."""

    @staticmethod
    def marginal_req_pages(
        candidate: DecodeCandidate, ledger: SharedPrefixLedger
    ) -> int:
        """GPU pages this candidate costs GIVEN what the group's pool holds.

        ``req_pages`` is the standalone reservation -- the pages the row needs
        when none of its context is on the device.  It stays the number the
        "larger than a whole replica" check uses, because that asks about the
        row alone.

        Admission asks what ADDING the row costs: the allocator maps one
        physical GPU page per distinct Host page id, so a row whose leading
        pages are already mapped (by a sibling admitted earlier in this walk,
        or by a sequence already decoding) pays only for its private tail.
        Both credits describe a PREFIX of the same page list, so the effective
        credit is their maximum, not their sum.

        Never zero: the ledger floors the charge at one page.
        """
        total = int(candidate.req_pages)
        if total <= 0:
            return total
        credit = min(max(0, int(candidate.resident_shared_pages)), total - 1)
        if candidate.shared_prefix_segments:
            preview = ledger.preview(
                total_pages=total,
                segments=segments_from_pairs(candidate.shared_prefix_segments),
            )
            credit = max(credit, preview.credited_pages)
        return total - credit

    @staticmethod
    def select_decode_batch(req: DecodeBatchRequest) -> List[str]:
        """Greedily fill the decode batch using live physical capacity.

        Candidates (PREFILLED + ON_HOLD) are admitted in ``global_idx``
        order. Pure DP charges ``assigned_rank``; TP decode charges the
        sequence's replicated ``decode_dp_group``. The candidate enumeration
        and GPU ``get_stats()`` query stay on the worker.
        """
        if not req.candidates:
            return []

        candidates = sorted(req.candidates, key=lambda c: c.global_idx)

        group_size = req.attn_tp_size
        if group_size <= 0 or req.world_size % group_size != 0:
            raise ValueError(
                f"attn_tp_size={group_size} must divide world_size={req.world_size}"
            )
        num_capacity_groups = req.world_size // group_size
        if req.existing_pages and len(req.existing_pages) != num_capacity_groups:
            raise ValueError(
                f"existing_pages has {len(req.existing_pages)} groups; "
                f"expected {num_capacity_groups}"
            )
        if req.free_pages:
            if len(req.free_pages) != num_capacity_groups:
                raise ValueError(
                    f"free_pages has {len(req.free_pages)} groups; "
                    f"expected {num_capacity_groups}"
                )
            if any(free < 0 or free > req.total_pages for free in req.free_pages):
                raise ValueError(
                    f"free_pages must be in [0, total_pages={req.total_pages}]: "
                    f"{req.free_pages}"
                )
            capacity_pages_available = list(req.free_pages)
        else:
            # Compatibility for pure callers predating the live snapshot.  The
            # worker always supplies free_pages; derive only when explicitly
            # given a legacy request so tests and older integrations stay safe.
            resident = (
                list(req.existing_pages)
                if req.existing_pages
                else [0] * num_capacity_groups
            )
            capacity_pages_available = [
                max(0, req.total_pages - pages) for pages in resident
            ]
        if req.existing_sequence_counts and len(req.existing_sequence_counts) != num_capacity_groups:
            raise ValueError(
                f"existing_sequence_counts has {len(req.existing_sequence_counts)} groups; "
                f"expected {num_capacity_groups}"
            )

        # A request larger than a whole replica is impossible.  Surface that as
        # an explicit capacity error instead of silently skipping it.  The
        # 90% legacy watermark is deliberately absent here: it is a batching
        # preference, not a physical safety bound.
        if req.total_pages > 0:
            unadmittable = [
                c for c in candidates if c.req_pages > req.total_pages
            ]
            if unadmittable:
                worst = max(unadmittable, key=lambda c: c.req_pages)
                raise DecodeCapacityError(
                    f"{len(unadmittable)} sequence(s) need more GPU KV pages "
                    f"than a replica has: {worst.uuid[:8]} needs "
                    f"{worst.req_pages} pages but the replica has "
                    f"{req.total_pages} total"
                )

        capacity_pages_used = [0] * num_capacity_groups
        # One ledger per capacity group: a GPU pool belongs to one DP rank (or
        # one replicated TP group), so a prefix mapped in one group saves
        # nothing for a row admitted into another. Charging the MARGINAL cost
        # is what makes admission agree with the allocator -- which already
        # shares one physical page between rows that name the same Host page --
        # instead of bounding decode by the logical page sum.
        capacity_prefix_ledgers = [
            SharedPrefixLedger() for _ in range(num_capacity_groups)
        ]
        capacity_seq_count = (
            list(req.existing_sequence_counts)
            if req.existing_sequence_counts
            else [0] * num_capacity_groups
        )
        cap = req.max_rank_bsz  # <= 0 means unlimited
        decode_batch: List[str] = []

        for c in candidates:
            if group_size > 1:
                if c.decode_dp_group is None:
                    raise ValueError(
                        f"candidate {c.uuid} has no decode_dp_group for "
                        f"attn_tp_size={group_size}"
                    )
                r = c.decode_dp_group
            else:
                r = c.assigned_rank
            # Cap per-rank in-decode count so the global batch stays <= mtp (MoE buffer
            # capacity) — prevents overflow of the pre-reserved padded buffers.
            if cap > 0 and capacity_seq_count[r] >= cap:
                continue

            ledger = capacity_prefix_ledgers[r]
            req_pages = DecodeScheduler.marginal_req_pages(c, ledger)

            if capacity_pages_used[r] + req_pages <= capacity_pages_available[r]:
                decode_batch.append(c.uuid)
                capacity_pages_used[r] += req_pages
                capacity_seq_count[r] += 1
                # Charge only on admission: a candidate the greedy skipped
                # loads nothing, so it must leave no credit behind for the
                # next row of its prefix.
                ledger.charge(
                    total_pages=max(1, int(c.req_pages)),
                    segments=segments_from_pairs(c.shared_prefix_segments),
                )

        return decode_batch
