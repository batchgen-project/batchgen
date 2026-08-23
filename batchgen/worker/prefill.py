"""Prefill scheduling — host-KV-capacity-bounded sequence selection.

Slice 6 of the worker decouple initiative (issue #175). Extracts the
pure *selection decision* from ``_prepare_prefill_batch``:

  - ``PrefillScheduler.select_prefill_batch`` — greedily admit candidate
    sequences (evicted first, by recompute priority; then queued, by
    arrival) into a prefill batch, bounded by each node's free host-KV
    pages.

Only the *decision* is ported. The cross-rank ``dist.all_gather`` that
collects per-node host-KV free pages, the candidate enumeration over
``global_batch``, and the rank-0 logging stay on the worker, which
builds the request and uses the returned uuid list.

Host KV is per-node: a sequence assigned to a rank on node N draws from
node N's host-KV capacity, so selection is a per-node bin-packing, not a
global one.

Design follows the per-slice frozen-snapshot pattern (no shared mutable
``WorkerState``): the worker passes exactly the fields this decision
consumes; the handler is pure and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PrefillCandidate:
    """A sequence eligible for prefill admission (EVICTED or QUEUEING).

    ``is_evicted`` selects the priority group: evicted sequences are
    admitted first (ordered by most-decoded-first to minimise wasted
    recompute), then queueing sequences (ordered by arrival ``global_idx``).
    """

    uuid: str
    assigned_rank: int
    is_evicted: bool
    global_idx: int
    total_decoded_before_eviction: int
    prompt_length: int
    kv_token_budget: int
    page_size: int


@dataclass(frozen=True)
class PrefillSelectionRequest:
    """Frozen snapshot for ``select_prefill_batch``.

    ``per_node_host_free`` is indexed by node id (gathered across ranks).
    ``initial_gpu_page_buffer`` is ``INITIAL_GPU_PAGE_BUFFER`` — pages
    reserved on a sequence's first GPU load.
    """

    candidates: Tuple[PrefillCandidate, ...]
    per_node_host_free: Tuple[int, ...]
    chunk_size: int
    num_nodes: int
    gpus_per_node: int
    initial_gpu_page_buffer: int
    # Persistent model state is a separate constraint from the token cap.
    # KDA keeps one state slot for a sequence until completion/eviction,
    # even when prefill is split into smaller token micro-batches.
    max_sequences_per_rank: Optional[int] = None
    max_sequences_per_node: Optional[int] = None


class PrefillScheduler:
    """Prefill admission decision — pure, deterministic across ranks."""

    @staticmethod
    def select_prefill_batch(req: PrefillSelectionRequest) -> List[str]:
        """Select which candidate sequences to prefill, bounded by host KV.

        Priority order: EVICTED sequences first (most decoded → least
        wasted recompute, ``global_idx`` tie-break), then QUEUEING (by
        ``global_idx``). Each candidate is admitted iff its node still has
        room for its initial page reservation.

        Initial reservation (matches the worker's dynamic host-KV sizing):
        ``max(prompt_length + chunk_size, gpu_initial_tokens)`` capped at
        ``kv_token_budget``, rounded up to whole pages — where
        ``gpu_initial_tokens`` covers ``prompt_length + 1`` plus the GPU
        page buffer. No safety margin: selection and allocation use the
        same formula by design.

        Pure: reads only the candidate snapshots + per-node free pages.
        The NCCL gather and the ``global_batch`` enumeration stay on the
        worker. Deterministic across ranks (stable sort keys), so every
        rank produces the identical batch without communication.
        """
        evicted = [c for c in req.candidates if c.is_evicted]
        queueing = [c for c in req.candidates if not c.is_evicted]
        evicted.sort(key=lambda c: (-c.total_decoded_before_eviction, c.global_idx))
        queueing.sort(key=lambda c: c.global_idx)
        all_candidates = evicted + queueing
        if not all_candidates:
            return []

        if (
            req.max_sequences_per_rank is not None
            and req.max_sequences_per_node is not None
        ):
            raise ValueError(
                "prefill sequence capacity must be scoped to rank or node, "
                "not both"
            )
        if req.max_sequences_per_rank is not None and req.max_sequences_per_rank < 0:
            raise ValueError("max_sequences_per_rank must be >= 0")
        if req.max_sequences_per_node is not None and req.max_sequences_per_node < 0:
            raise ValueError("max_sequences_per_node must be >= 0")

        per_node_effective_free = list(req.per_node_host_free)
        node_pages_used = [0] * req.num_nodes
        rank_sequences_used: dict[int, int] = {}
        node_sequences_used = [0] * req.num_nodes
        prefill_batch: List[str] = []

        for c in all_candidates:
            seq_node = c.assigned_rank // req.gpus_per_node
            if (
                req.max_sequences_per_rank is not None
                and rank_sequences_used.get(c.assigned_rank, 0)
                >= req.max_sequences_per_rank
            ):
                continue
            if (
                req.max_sequences_per_node is not None
                and node_sequences_used[seq_node] >= req.max_sequences_per_node
            ):
                continue
            post_prefill_length = c.prompt_length + 1
            gpu_initial_pages = (
                math.ceil(post_prefill_length / c.page_size)
                + req.initial_gpu_page_buffer
            )
            gpu_initial_tokens = gpu_initial_pages * c.page_size
            initial_capacity = max(c.prompt_length + req.chunk_size, gpu_initial_tokens)
            initial_capacity = min(initial_capacity, c.kv_token_budget)
            req_pages = math.ceil(initial_capacity / c.page_size)

            if node_pages_used[seq_node] + req_pages <= per_node_effective_free[seq_node]:
                prefill_batch.append(c.uuid)
                node_pages_used[seq_node] += req_pages
                rank_sequences_used[c.assigned_rank] = (
                    rank_sequences_used.get(c.assigned_rank, 0) + 1
                )
                node_sequences_used[seq_node] += 1

        return prefill_batch
