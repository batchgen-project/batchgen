"""Local slot index <-> sequence uuid table of one ``BatchGenWorker`` rank.

``SequenceTable`` exclusively owns four pieces of coupled state that used to
be loose worker attributes:

* ``local_idx -> uuid`` (was ``BatchGenWorker._local_to_uuid_map``);
* ``uuid -> local_idx`` (was ``BatchGenWorker._uuid_to_local_map``);
* the free-slot set (was ``_free_local_indices``);
* the next never-used slot (was ``_next_local_idx``).

Every write goes through ``bind`` / ``unbind`` / ``reset``. Readers get
``local_to_uuid`` / ``uuid_to_local``: live read-only ``MappingProxyType``
views of the two dicts. Because ``reset`` clears in place, a view taken once
stays valid for the table's whole lifetime.

The table knows nothing about the resources a bound sequence holds (GPU KV
pages, host-KV pages, query-book entries). See ``unbind`` for the ordering
that callers own.

Imports: stdlib only.
"""

from __future__ import annotations

from collections.abc import Set as AbstractSetABC
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Set,
)

if TYPE_CHECKING:
    from batchgen.worker.rank_context import RankContext


class _FrozenSetView(AbstractSetABC):
    """Live read-only view of a set.

    A snapshot would go stale the moment a slot is bound or freed, so the
    free-slot accessor hands out this instead: it reflects every later write
    and refuses every mutation (no ``add`` / ``discard`` / ``pop``).
    """

    __slots__ = ("_backing",)

    def __init__(self, backing: Set[int]) -> None:
        self._backing = backing

    def __contains__(self, value: object) -> bool:
        return value in self._backing

    def __iter__(self) -> Iterator[int]:
        return iter(self._backing)

    def __len__(self) -> int:
        return len(self._backing)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({set(self._backing)!r})"


class SequenceSlotCollisionError(RuntimeError):
    """An explicit local slot is already bound to a different uuid."""


class SequenceTable:
    """Single authority for one rank's local slot binding."""

    __slots__ = (
        "_local_to_uuid",
        "_uuid_to_local",
        "_free",
        "_next",
        "_local_to_uuid_view",
        "_uuid_to_local_view",
        "_free_view",
    )

    def __init__(self) -> None:
        self._local_to_uuid: Dict[int, str] = {}
        self._uuid_to_local: Dict[str, int] = {}
        self._free: Set[int] = set()
        self._next: int = 0
        self._make_views()

    @classmethod
    def adopting(
        cls,
        *,
        local_to_uuid: Dict[int, str],
        uuid_to_local: Dict[str, int],
        free_local_indices: Set[int],
        next_local_idx: int,
    ) -> "SequenceTable":
        """A table that operates on caller-owned containers, in place.

        Exists only for the pinned legacy helpers
        ``batchgen.query_book.bind_local_sequence_to_query_book`` and
        ``release_local_query_slot``, whose signatures take the raw containers.
        Their mutation logic is this class's ``bind`` / ``unbind``, so the rule
        has one implementation. ``BatchGenWorker`` never uses this.
        """
        table = cls.__new__(cls)
        table._local_to_uuid = local_to_uuid
        table._uuid_to_local = uuid_to_local
        table._free = free_local_indices
        table._next = int(next_local_idx)
        table._make_views()
        return table

    def _make_views(self) -> None:
        self._local_to_uuid_view = MappingProxyType(self._local_to_uuid)
        self._uuid_to_local_view = MappingProxyType(self._uuid_to_local)
        self._free_view = _FrozenSetView(self._free)

    # ---------------------------------------------------------------- writes

    def bind(self, uuid: str, local_idx: Optional[int] = None) -> int:
        """Bind ``uuid`` to a local slot and return that slot.

        With ``local_idx=None`` the slot comes from the free set (an arbitrary
        member, ``set.pop``) or, if the set is empty, from the next never-used
        index.

        Invariant: every slot is handed out at most once. An explicit
        ``local_idx`` therefore leaves both allocation sources: a slot held by
        another uuid raises ``SequenceSlotCollisionError``; the slot is removed
        from the free set; and a slot at or beyond the next never-used index
        advances that index past it, the skipped indices joining the free set.
        (Rebinding a uuid that is already bound elsewhere is not checked here.)
        """
        if local_idx is None:
            if self._free:
                local_idx = self._free.pop()
            else:
                local_idx = self._next
                self._next += 1
        else:
            holder = self._local_to_uuid.get(local_idx)
            if holder is not None and holder != uuid:
                raise SequenceSlotCollisionError(
                    f"local slot {local_idx} is bound to {holder!r}; "
                    f"cannot bind it to {uuid!r}"
                )
            self._free.discard(local_idx)
            if local_idx >= self._next:
                self._free.update(range(self._next, local_idx))
                self._next = local_idx + 1
        self._local_to_uuid[local_idx] = uuid
        self._uuid_to_local[uuid] = local_idx
        return local_idx

    def unbind(self, uuid: str) -> Optional[int]:
        """Unbind ``uuid``, return its freed slot (``None`` if it was unbound).

        ORDERING INVARIANT, owned by the caller: release every resource keyed
        by this sequence's local binding (GPU KV pages, host-KV pages,
        anything filtered by "uuid is bound here") BEFORE calling ``unbind``.
        Once the uuid is unbound, ``uuid in uuid_to_local`` is False, and the
        release filters that select "my" sequences by that membership select
        nothing, so the pages leak. See the ORDERING FIX sites in
        ``BatchGenWorker`` that release and then call ``_report_completion``.
        The table cannot see resource state, so it does not check this.
        """
        local_idx = self._uuid_to_local.pop(uuid, None)
        if local_idx is None:
            return None
        self._local_to_uuid.pop(local_idx, None)
        self._free.add(local_idx)
        return local_idx

    def reset(self) -> None:
        """Drop every binding and restart slot allocation at 0, in place."""
        self._local_to_uuid.clear()
        self._uuid_to_local.clear()
        self._free.clear()
        self._next = 0

    # ----------------------------------------------------------------- reads

    @property
    def local_to_uuid(self) -> Mapping[int, str]:
        """Live read-only view: local slot -> uuid."""
        return self._local_to_uuid_view

    @property
    def uuid_to_local(self) -> Mapping[str, int]:
        """Live read-only view: uuid -> local slot."""
        return self._uuid_to_local_view

    @property
    def free_local_indices(self) -> AbstractSet[int]:
        """Live read-only view of the free-slot set."""
        return self._free_view

    @property
    def next_local_idx(self) -> int:
        """The next never-used slot index."""
        return self._next

    def local_of(self, uuid: str) -> Optional[int]:
        return self._uuid_to_local.get(uuid)

    def uuid_of(self, local_idx: int) -> Optional[str]:
        return self._local_to_uuid.get(local_idx)

    def is_bound(self, uuid: str) -> bool:
        return uuid in self._uuid_to_local

    def __contains__(self, uuid: object) -> bool:
        return uuid in self._uuid_to_local

    def __len__(self) -> int:
        return len(self._uuid_to_local)

    def __iter__(self) -> Iterator[str]:
        """Iterate bound uuids."""
        return iter(self._uuid_to_local)

    # ------------------------------------------------- group-ownership checks

    def check_binding_matches_group(
        self,
        uuids: Iterable[str],
        where: str,
        *,
        rank_ctx: "RankContext",
        get_sequence: Callable[[str], object],
    ) -> None:
        """A local row exists iff this rank's group owns the sequence (G > 1).

        Wave resolution (``_get_local_indices_for_uuids``) reads
        ``uuid_to_local`` alone and silently skips what it does not find, so a
        binding that disagrees with the serve-group map is not refused
        anywhere else: the rank prefills or decodes a row its group does not
        own, or misses one it does. The second form is what raises
        ``KeyError: Sequence <global_idx> has no KDA state item`` from
        ``prepare_decode_step``, by which time the stamping seam is long gone.

        The invariant is an equivalence, so both directions are checked, and
        both are local reads: no collective, O(len(uuids)) per call. Uuids with
        no ``SequenceEntry`` are skipped. G == 1 checks nothing.
        """
        G = rank_ctx.decode_group_size
        if G <= 1:
            return
        offenders: List[str] = []
        for uuid in uuids:
            seq = get_sequence(uuid)
            if seq is None:
                continue
            bound = uuid in self._uuid_to_local
            owns = rank_ctx.owns_sequence(seq)
            if bound != owns:
                offenders.append(
                    f"global_idx={seq.global_idx} group={seq.decode_dp_group} "
                    f"bound={int(bound)} owns={int(owns)}"
                )
        if offenders:
            raise RuntimeError(
                f"Rank {rank_ctx.rank}: [SCHED_INVARIANT] {where}: "
                f"{len(offenders)} sequences whose local row disagrees with the "
                f"serve-group map (G={G}, rank//G={rank_ctx.rank // G}): "
                + "; ".join(offenders[:10])
            )
