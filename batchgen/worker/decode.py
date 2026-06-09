"""Decode scheduling — GPU-capacity-bounded decode batch selection.

Slice 9 of the worker decouple initiative (issue #175) — the final
slice. Extracts the pure *selection decision* from
``_prepare_decode_batch``:

  - ``DecodeScheduler.select_decode_batch`` — greedily admit PREFILLED /
    ON_HOLD sequences into the decode batch (ordered by ``global_idx``),
    bounded per rank by a 90% GPU-KV page watermark.

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
from typing import List, Tuple

# Fraction of total GPU-KV pages a rank may fill when assembling the decode
# batch (the "~90% watermark" from the legacy `_prepare_decode_batch`).
_DECODE_CAPACITY_FRACTION = 0.9


@dataclass(frozen=True)
class DecodeCandidate:
    """A PREFILLED / ON_HOLD sequence eligible for the decode batch."""

    uuid: str
    assigned_rank: int
    global_idx: int
    req_pages: int  # GPU pages for its two-page-buffer reservation


@dataclass(frozen=True)
class DecodeBatchRequest:
    """Frozen snapshot for ``select_decode_batch``.

    ``total_pages`` is the GPU paged-KV manager's per-rank total page count
    (the same value for every rank).
    """

    candidates: Tuple[DecodeCandidate, ...]
    total_pages: int
    world_size: int
    # Per-rank in-decode sequence cap (= the MoE buffer's num_tokens_per_rank = mtp/world_size).
    # Bounds the global decode batch to <= mtp so the pre-reserved padded MoE buffers never
    # overflow (no runtime resize -> no OOM). 0 = unlimited (legacy / non-K2.5).
    max_rank_bsz: int = 0


class DecodeScheduler:
    """Decode batch admission decision — pure, deterministic across ranks."""

    @staticmethod
    def select_decode_batch(req: DecodeBatchRequest) -> List[str]:
        """Greedily fill the decode batch to a per-rank 90% page watermark.

        Candidates (PREFILLED + ON_HOLD) are admitted in ``global_idx``
        order; each is taken iff its rank still has room under
        ``int(total_pages * 0.9)``. Pure — the candidate enumeration and the
        GPU ``get_stats()`` query stay on the worker.
        """
        if not req.candidates:
            return []

        capacity_per_rank = int(req.total_pages * _DECODE_CAPACITY_FRACTION)
        candidates = sorted(req.candidates, key=lambda c: c.global_idx)

        rank_pages_used = [0] * req.world_size
        rank_seq_count = [0] * req.world_size
        cap = req.max_rank_bsz  # <= 0 means unlimited
        decode_batch: List[str] = []

        for c in candidates:
            r = c.assigned_rank
            # Cap per-rank in-decode count so the global batch stays <= mtp (MoE buffer
            # capacity) — prevents overflow of the pre-reserved padded buffers.
            if cap > 0 and rank_seq_count[r] >= cap:
                continue
            if rank_pages_used[r] + c.req_pages <= capacity_per_rank:
                decode_batch.append(c.uuid)
                rank_pages_used[r] += c.req_pages
                rank_seq_count[r] += 1

        return decode_batch
