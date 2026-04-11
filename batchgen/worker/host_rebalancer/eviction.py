"""EvictionStrategy Protocol + default implementation.

Plan Decision #1: eviction ordering is pluggable. The default strategy
is "shortest-decoded-first" (preserve longer-running sequences), which
matches main. Future strategies can swap in without touching the
:class:`HostKVRebalancer` class or its tests.
"""

from __future__ import annotations

from typing import Protocol

from batchgen.sequence import SequenceEntry
from batchgen.worker.protocols import UUID


class EvictionStrategy(Protocol):
    """Picks the subset of sequences to put on hold or evict.

    The `sequences` input is the full candidate set in some arbitrary
    deterministic order; the strategy returns `count` UUIDs (or fewer
    if the candidate pool is smaller).
    """

    def select(self, sequences: list[SequenceEntry], count: int) -> list[UUID]: ...


class ShortestDecodedFirstStrategy:
    """Default eviction strategy — evict the sequences that have decoded
    the fewest tokens, preserving the progress of longer-running
    sequences.

    Matches main's behavior at the host-KV watermark path. Ties (equal
    decoded_length) are broken by ``uuid`` so the order is identical
    across ranks without a collective.
    """

    def select(self, sequences: list[SequenceEntry], count: int) -> list[UUID]:
        if count <= 0:
            return []
        ordered = sorted(sequences, key=lambda s: (s.decoded_length, s.uuid))
        return [s.uuid for s in ordered[:count]]


__all__ = ["EvictionStrategy", "ShortestDecodedFirstStrategy"]
