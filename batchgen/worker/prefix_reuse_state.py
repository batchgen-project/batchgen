"""Owned prefix-reuse state and admission digest helpers.

``PrefixReuseState`` is a non-leaf worker component by design.  It owns the
prefix sequence states, pool pseudo-sid namespace/refcounts, digest cache and
credit latches.  It imports ``PrefixCacheSequenceState`` because that object is
the prefix-cache state representation, so this component is deliberately not
an invariant-5 leaf.  It never resolves worker identities: callers pass plain
global ids, candidate ids, live-id collections and token accessors.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Callable, Collection, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


class PrefixReuseState:
    """State owner for prefix reuse; identity resolution stays with callers."""

    def __init__(self, *, rank: int = 0, world_size: int = 1) -> None:
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.prefix_drain_metric_emitted = False
        self.prefix_sequence_states = {}
        self.active_prefix_materializations = []
        # A7: this seed is unconditional and eager, at the same construction
        # point as the former worker field.
        self.pool_sid_counter = time.time_ns() // 1000
        self.pool_sid_members = {}
        self.pool_member_sids = {}
        self.shared_prefix_credit_suspended = False
        self.shared_prefix_credit_applied = False
        self.prefix_page_digest_cache = {}

    def next_pool_segment_sid(self) -> int:
        self.pool_sid_counter += 1
        return -(self.rank + 1 + self.world_size * self.pool_sid_counter)

    @staticmethod
    def new_sequence_state(**kwargs):
        """Construct the canonical prefix state without making this a leaf."""
        from batchgen.prefix_reuse.prefill import PrefixCacheSequenceState

        return PrefixCacheSequenceState(**kwargs)

    def suspend_shared_prefix_credit(self, reason: str, *, rank: Optional[int] = None) -> bool:
        """Trip the one-round credit latch; return whether it changed state."""
        if not self.shared_prefix_credit_applied:
            return False
        self.shared_prefix_credit_applied = False
        if self.shared_prefix_credit_suspended:
            return False
        self.shared_prefix_credit_suspended = True
        if rank == 0:
            logging.warning(
                "[PREFILL] shared-prefix admission credit suspended for the "
                "next round: a credited wave was %s, so the next wave is "
                "charged per row",
                reason,
            )
        return True

    def shared_prefix_credit_enabled(
        self,
        *,
        shared_prefix_admission: bool,
        enable_prefix_cache: bool,
        pool_enabled: bool,
        prefill_pool_page_budget,
    ) -> bool:
        return bool(
            shared_prefix_admission
            and enable_prefix_cache
            and not self.shared_prefix_credit_suspended
            and pool_enabled
            and prefill_pool_page_budget is not None
        )

    def prefix_page_digests(
        self,
        *,
        global_id: int,
        prompt_length: int,
        page_size: int,
        tokens_for: Callable[[], Iterable[int]],
    ) -> Tuple[int, ...]:
        cached = self.prefix_page_digest_cache.get(global_id)
        if cached is not None:
            return cached
        shareable = (int(prompt_length) - 1) // int(page_size)
        if shareable <= 0:
            self.prefix_page_digest_cache[global_id] = ()
            return ()
        token_array = np.ascontiguousarray(np.asarray(list(tokens_for()), dtype=np.int64))
        digests = []
        running = b"\x00" * 16
        for page in range(shareable):
            block = token_array[page * int(page_size):(page + 1) * int(page_size)]
            running = hashlib.blake2b(running + block.tobytes(), digest_size=16).digest()
            digests.append(int.from_bytes(running[:8], "big", signed=False))
        chain = tuple(digests)
        self.prefix_page_digest_cache[global_id] = chain
        return chain

    def prune_prefix_page_digest_cache(self, live_global_ids: Collection[int]) -> None:
        live = {int(value) for value in live_global_ids}
        for global_id in list(self.prefix_page_digest_cache):
            if global_id not in live:
                del self.prefix_page_digest_cache[global_id]

    def admission_shared_prefix_segments(
        self,
        all_candidates: List[str],
        *,
        sequence_for: Callable[[str], object],
        tokens_for: Callable[[object], Iterable[int]],
        credit_enabled: bool,
    ) -> Dict[str, Tuple[Tuple[int, int], ...]]:
        if not credit_enabled:
            return {}
        ordered = sorted(
            all_candidates,
            key=lambda uuid: (
                self.admission_order_key(sequence_for(uuid)),
                int(sequence_for(uuid).global_idx),
            ),
        )
        digests = [
            self.prefix_page_digests(
                global_id=int(sequence_for(uuid).global_idx),
                prompt_length=int(sequence_for(uuid).prompt_length),
                page_size=int(sequence_for(uuid).PAGE_SIZE),
                tokens_for=lambda uuid=uuid: tokens_for(sequence_for(uuid)),
            )
            for uuid in ordered
        ]

        def shared_pages(left: int, right: int) -> int:
            if left < 0 or right >= len(digests):
                return 0
            first, second = digests[left], digests[right]
            limit = min(len(first), len(second))
            index = 0
            while index < limit and first[index] == second[index]:
                index += 1
            return index

        segments: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        for position, uuid in enumerate(ordered):
            run = max(shared_pages(position - 1, position), shared_pages(position, position + 1))
            if run > 0:
                segments[uuid] = ((digests[position][run - 1], run),)
        return segments

    @staticmethod
    def admission_order_key(seq) -> int:
        order = getattr(seq, "prefill_order", None)
        return int(seq.global_idx if order is None else order)

    def pop_state(self, global_id: int):
        return self.prefix_sequence_states.pop(int(global_id), None)

    def add_pool_member_refs(self, global_id: int, sids: Iterable[int]) -> None:
        """Record one member's pseudo-sid chain and increment each refcount."""
        chain = tuple(int(sid) for sid in sids)
        self.pool_member_sids[int(global_id)] = chain
        for sid in chain:
            self.pool_sid_members[sid] = self.pool_sid_members.get(sid, 0) + 1

    def release_pool_segment_refs(
        self,
        global_id: int,
        *,
        host_views_provider: Callable[[], Mapping[int, object]],
    ) -> None:
        for sid in self.pool_member_sids.pop(int(global_id), ()):
            left = self.pool_sid_members[sid] - 1
            if left:
                self.pool_sid_members[sid] = left
                continue
            del self.pool_sid_members[sid]
            for view in host_views_provider().values():
                view.release_sequence_pages([sid])
                view.unregister_sequences([sid])
