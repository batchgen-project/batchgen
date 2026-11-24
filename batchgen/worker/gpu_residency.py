"""GPU KV residency set of one ``BatchGenWorker`` rank.

``GpuResidencySet`` exclusively owns the set of sequence uuids that currently
hold GPU KV pages. It replaces the bare worker set ``_sequences_with_gpu_kv``;
the worker attribute keeps that name but now holds this owner, and the
decode-phase context field ``gpu_resident`` carries the same object into the
engines, so every module mutates the state only through the named writes:

* ``mark_resident`` — the sequence just received GPU KV pages (allocation,
  decode entry, async host-to-GPU load finalize);
* ``drop`` — the sequence's GPU pages were released (completion, host
  eviction, ON_HOLD, extension failure, migration source);
* ``drop_all`` — the GPU KV pool itself was destroyed.

Writes keep the exact semantics of the set operations they replaced:
``mark_resident`` is idempotent like ``set.add``, ``drop`` never raises like
``set.discard``. Readers get membership, iteration, ``len`` and set
comparisons through the read-only ``collections.abc.Set`` interface, so the
existing ``uuid in ...`` / ``for uuid in ...`` call sites are unchanged while
no inherited operation can mutate the storage.

The owner knows nothing about pages themselves: the GPU paged KV manager
holds the allocations, this set is the worker-side source-of-truth filter
("was this sequence ever GPU-allocated") that release paths intersect with.

Imports: stdlib only.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSetABC
from typing import Iterator, Set


class GpuResidencySet(AbstractSetABC):
    """Single authority for one rank's GPU-resident sequence uuids."""

    __slots__ = ("_gpu_resident_uuids",)

    def __init__(self) -> None:
        self._gpu_resident_uuids: Set[str] = set()

    # ---------------------------------------------------------------- writes

    def mark_resident(self, uuid: str) -> None:
        """Record that ``uuid`` now holds GPU KV pages (idempotent)."""
        self._gpu_resident_uuids.add(uuid)

    def drop(self, uuid: str) -> None:
        """Forget ``uuid`` after its GPU pages were released (no-op if absent)."""
        self._gpu_resident_uuids.discard(uuid)

    def drop_all(self) -> None:
        """Forget every sequence (the GPU KV pool was destroyed)."""
        self._gpu_resident_uuids.clear()

    # ----------------------------------------------------------------- reads

    def __contains__(self, uuid: object) -> bool:
        return uuid in self._gpu_resident_uuids

    def __iter__(self) -> Iterator[str]:
        return iter(self._gpu_resident_uuids)

    def __len__(self) -> int:
        return len(self._gpu_resident_uuids)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({set(self._gpu_resident_uuids)!r})"

    @classmethod
    def _from_iterable(cls, iterable) -> Set[str]:
        # Set-algebra results (``owner - other`` etc.) are plain sets: a new
        # residency authority is never created implicitly.
        return set(iterable)
