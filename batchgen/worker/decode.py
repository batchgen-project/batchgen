"""Decode scheduling — GPU-capacity-bounded decode batch selection.

Slice 9 of the worker decouple initiative (issue #175) — the final
slice. Extracts the pure *selection decision* from
``_prepare_decode_batch``:

  - ``DecodeScheduler.select_decode_batch`` — greedily admit PREFILLED /
    ON_HOLD sequences into the decode batch (ordered by ``global_idx``),
    bounded per physical GPU-KV replica by a 90% page watermark.

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

# Fraction of total GPU-KV pages a rank may fill when assembling the decode
# batch (the "~90% watermark" from the legacy `_prepare_decode_batch`).
_DECODE_CAPACITY_FRACTION = 0.9


def estimate_max_decode_replica_batch(
    total_candidates: int, world_size: int, attn_tp_size: int
) -> int:
    """Upper-bound rows held by one DP replica (one TP attention group)."""
    if attn_tp_size <= 0 or world_size % attn_tp_size != 0:
        raise ValueError(
            f"attn_tp_size={attn_tp_size} must divide world_size={world_size}"
        )
    num_replicas = world_size // attn_tp_size
    return (total_candidates + num_replicas - 1) // num_replicas


@dataclass(frozen=True)
class DecodeCandidate:
    """A PREFILLED / ON_HOLD sequence eligible for the decode batch."""

    uuid: str
    assigned_rank: int
    global_idx: int
    req_pages: int  # GPU pages for its two-page-buffer reservation
    decode_dp_group: Optional[int] = None


@dataclass(frozen=True)
class DecodeBatchRequest:
    """Frozen snapshot for ``select_decode_batch``.

    ``total_pages`` is the GPU paged-KV manager's per-rank total page count.
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


class DecodeScheduler:
    """Decode batch admission decision — pure, deterministic across ranks."""

    @staticmethod
    def select_decode_batch(req: DecodeBatchRequest) -> List[str]:
        """Greedily fill the decode batch to a per-replica 90% page watermark.

        Candidates (PREFILLED + ON_HOLD) are admitted in ``global_idx``
        order. Pure DP charges ``assigned_rank``; TP decode charges the
        sequence's replicated ``decode_dp_group``. The candidate enumeration
        and GPU ``get_stats()`` query stay on the worker.
        """
        if not req.candidates:
            return []

        capacity_per_rank = int(req.total_pages * _DECODE_CAPACITY_FRACTION)
        candidates = sorted(req.candidates, key=lambda c: c.global_idx)

        group_size = req.attn_tp_size
        if group_size <= 0 or req.world_size % group_size != 0:
            raise ValueError(
                f"attn_tp_size={group_size} must divide world_size={req.world_size}"
            )
        num_capacity_groups = req.world_size // group_size
        capacity_pages_used = [0] * num_capacity_groups
        capacity_seq_count = [0] * num_capacity_groups
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
            if capacity_pages_used[r] + c.req_pages <= capacity_per_rank:
                decode_batch.append(c.uuid)
                capacity_pages_used[r] += c.req_pages
                capacity_seq_count[r] += 1

        return decode_batch
