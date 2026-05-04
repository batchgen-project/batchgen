"""Prefix-aware host KV admission and eviction helpers."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Set, Tuple

from batchgen.sequence import SequenceBatch, SequenceEntry


@dataclass(frozen=True)
class PrefixAdmissionEstimate:
    private_pages: int
    shared_pages: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class PrefixAdmissionEvictionResult:
    target_free_pages: int
    entries_removed: int
    reached_target: bool


def estimate_prefix_allocation_for_admission(
    *,
    seq: SequenceEntry,
    capacity_tokens: int,
    page_size: int,
    prefix_runtime_enabled: bool,
    current_rank: int,
    worker_view: object,
    namespace_hash: int,
    prompt_tokens: Callable[[SequenceEntry], List[int]],
) -> PrefixAdmissionEstimate:
    logical_pages = math.ceil(capacity_tokens / page_size)
    if (
        not prefix_runtime_enabled
        or seq.assigned_rank != current_rank
        or seq.input_ids is None
    ):
        return PrefixAdmissionEstimate(private_pages=logical_pages)

    estimate_fn = getattr(
        worker_view,
        "estimate_pages_for_sequences_with_prefix",
        None,
    )
    if estimate_fn is None:
        return PrefixAdmissionEstimate(private_pages=logical_pages)

    try:
        estimate = estimate_fn(
            [
                (
                    seq.global_idx,
                    prompt_tokens(seq),
                    capacity_tokens,
                    namespace_hash,
                )
            ]
        )
        if not estimate:
            return PrefixAdmissionEstimate(private_pages=logical_pages)
        item = estimate[0]
        private_pages = int(item.get("physical_pages_allocated", logical_pages))
        shared_pages = [
            int(page) for page in item.get("shared_prefix_pages", [])
        ]
        return PrefixAdmissionEstimate(
            private_pages=max(0, private_pages),
            shared_pages=shared_pages,
        )
    except Exception as exc:
        logging.debug(
            "Rank %s prefix admission estimate failed for seq %s: %s",
            current_rank,
            getattr(seq, "global_idx", "unknown"),
            exc,
        )
        return PrefixAdmissionEstimate(private_pages=logical_pages)


def maybe_evict_prefix_cache_for_prefill_admission(
    *,
    all_candidates: Iterable[str],
    global_batch: SequenceBatch,
    current_rank: int,
    get_node_for_rank: Callable[[int], int],
    initial_host_tokens_for_prefill: Callable[[SequenceEntry, int], int],
    estimate_prefix_allocation: Callable[
        [SequenceEntry, int],
        Tuple[int, List[int]],
    ],
    chunk_size: int,
    worker_view: object,
) -> Optional[PrefixAdmissionEvictionResult]:
    """Evict enough unprotected prefix pages to admit at least one candidate."""
    if worker_view is None:
        return None
    try:
        stats = worker_view.get_stats()
        local_free = int(stats.num_free_pages)
    except Exception:
        return None

    my_node = get_node_for_rank(current_rank)
    target_free_pages = 0
    protected_pages: Set[int] = set()
    for uuid in all_candidates:
        seq = global_batch.get_sequence(uuid)
        if seq is None or seq.assigned_rank is None:
            continue
        if seq.assigned_rank != current_rank:
            continue
        if get_node_for_rank(seq.assigned_rank) != my_node:
            continue
        capacity_tokens = initial_host_tokens_for_prefill(seq, chunk_size)
        req_pages, shared_pages = estimate_prefix_allocation(seq, capacity_tokens)
        if req_pages > local_free:
            target_free_pages = (
                req_pages
                if target_free_pages == 0
                else min(target_free_pages, req_pages)
            )
            protected_pages.update(shared_pages)

    if target_free_pages == 0:
        return None

    eviction = worker_view.evict_prefix_cache_until_free(
        target_free_pages,
        protected_pages=list(protected_pages),
    )
    return PrefixAdmissionEvictionResult(
        target_free_pages=target_free_pages,
        entries_removed=int(getattr(eviction, "entries_removed", 0)),
        reached_target=bool(getattr(eviction, "reached_target", False)),
    )
