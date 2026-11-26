"""Durable sequence-id -> host-pages map of the KDA record store (方案甲).

Why this exists: the per-sequence KDA record store writes each LIVE
sequence's recurrent state into Host KV pages of the node's shared region.
The record BYTES and the Host allocator's page accounting live in that
region and SURVIVE a worker-pool restart (after ``retain`` the pages are
prefix-resident), but the map naming them -- global sequence id -> arena
page ids -- was a Python dict inside each worker process and died with the
pool.  After ``POST /v1/workers/restart`` failed the in-flight batches,
nothing could name the retained pages any more, so every record orphaned
(``record_pages`` Host pages per sequence) until a cold restart; the host
budget is one combined pool, so the orphans shrank usable host KV.

This module is the map's durable home, the same class of treatment as the
host KV page tables: a small fixed-layout table in an anonymous memfd OWNED
BY THE SERVER process -- exactly like the host KV, weights and
prefix-coordinator regions, which "intentionally remain open across" the
cohort reconcile -- that every worker cohort attaches to through
``/proc/<server>/fd/<N>``.  The sequence's OWNER rank registers and updates
its entry; followers never write (their mirrors travel through the existing
host-KV mirror payload), so each entry has exactly one writer at a time and
the flock around every mutation only has to serialize slot claims and the
one-time header write.

Restart contract (the orphan fix):

* Every entry carries the cohort ``generation`` it was written under; the
  header carries the CURRENT generation.
* At pool startup the node's recovery sweep bumps the header generation and
  walks the surviving entries (``kda_lifecycle.release_restart_orphans``):
  a sequence that no longer exists server-side gets its record RELEASED --
  resident pages back to the Host pool for a retained record; entry dropped
  without a release for an unretained one, whose pages were still
  chain-owned by its pseudo row and were already recovered by the server's
  cohort reconcile (``release_all_sequences``) before the respawn.
* A store built afterwards rehydrates the entries whose generation differs
  from the header's (the surviving sequences), owned on the rank that owned
  them before the restart and mirrored elsewhere, so their records stay
  readable.  Entries AT the current generation were written by live ranks
  of this cohort and are skipped: their mirrors arrive through ``adopt``.

A model with no recurrent state never builds a record store, so the region
stays at size zero and the sweep finds nothing -- the structural absence of
the state, not a configuration gate.

This module never imports model-family or worker code.
"""

from __future__ import annotations

import os
import struct
import threading
from dataclasses import dataclass
from typing import Iterable, Mapping

from batchgen.prefix_reuse.trajectory_commit import SnapshotArena

PREFIX = "KDA_RECORD_MAP"

_MAGIC = b"KDARMAP1"
_VERSION = 1

# Header: magic, version, generation, capacity, group_count, pages_per_group,
# unit_count, host_page_bytes_all_layers.  Group ids follow as i32s.
_HEADER = struct.Struct("<8sIIIIIIQ")
# Entry head: state, written, retained, pad, owner_rank, generation, pad,
# sequence_id, pseudo_sid, token_count.  Page ids follow as i32s, group-major
# in header group-id order.
_ENTRY_HEAD = struct.Struct("<BBBBiIIqqq")

_STATE_EMPTY = 0
_STATE_LIVE = 1


class KDARecordMapError(RuntimeError):
    """Base error of the durable KDA record map."""


class KDARecordMapLayoutMismatch(KDARecordMapError):
    """The persisted region's header disagrees with this cohort's geometry.

    The geometry is a set of model/deployment constants, so a mismatch means
    two incompatible deployments met one region -- a bug, never a fallback.
    """


class KDARecordMapFull(KDARecordMapError):
    """No free slot for a new record: the capacity invariant broke.

    The map is sized for every sequence the node can hold at once, so a full
    map is an accounting bug surfacing, exactly like a failed reservation
    after the capacity preflight charged the pages.
    """


class KDARecordMapCorrupt(KDARecordMapError):
    """The map's content contradicts the caller's lifecycle bookkeeping."""


@dataclass(frozen=True)
class KDARecordMapGeometry:
    """The constants the map's binary layout is derived from, exactly."""

    capacity: int
    group_ids: tuple[int, ...]
    pages_per_group: int
    unit_count: int
    host_page_bytes_all_layers: int

    def __post_init__(self) -> None:
        if int(self.capacity) <= 0:
            raise KDARecordMapError(
                f"map capacity must be positive, got {self.capacity}"
            )
        if not self.group_ids:
            raise KDARecordMapError("map needs at least one KV group id")
        if len(set(self.group_ids)) != len(self.group_ids) or tuple(
            sorted(int(g) for g in self.group_ids)
        ) != tuple(int(g) for g in self.group_ids):
            raise KDARecordMapError(
                f"group ids must be sorted and unique, got {self.group_ids}"
            )
        if int(self.pages_per_group) <= 0:
            raise KDARecordMapError(
                f"pages_per_group must be positive, got {self.pages_per_group}"
            )
        if int(self.unit_count) <= 0:
            raise KDARecordMapError(
                f"unit_count must be positive, got {self.unit_count}"
            )
        if int(self.host_page_bytes_all_layers) <= 0:
            raise KDARecordMapError(
                "host_page_bytes_all_layers must be positive, got "
                f"{self.host_page_bytes_all_layers}"
            )


@dataclass(frozen=True)
class KDARecordMapEntry:
    """One persisted record, as read back from the region."""

    sequence_id: int
    pseudo_sid: int
    owner_rank: int
    generation: int
    written: bool
    retained: bool
    token_count: int
    page_ids_by_group: Mapping[int, tuple[int, ...]]
    pages_per_group: int
    unit_count: int
    host_page_bytes_all_layers: int

    def to_arena(self) -> SnapshotArena:
        """The record's arena, byte-identical to the one that was persisted."""

        return SnapshotArena(
            storage_handle=0,
            pseudo_sid=int(self.pseudo_sid),
            page_ids_by_group={
                int(group_id): tuple(int(page) for page in pages)
                for group_id, pages in self.page_ids_by_group.items()
            },
            pages_per_group=int(self.pages_per_group),
            capacity_bytes=int(self.pages_per_group)
            * int(self.host_page_bytes_all_layers),
            payload_bytes=0,
            unit_count=int(self.unit_count),
        )


def _align8(value: int) -> int:
    return (int(value) + 7) // 8 * 8


class KDARecordSharedMap:
    """Fixed-layout sequence-id -> record table over a shared buffer.

    ``buffer`` is any writable byte buffer: the mmap of the server-owned
    memfd in production, a plain ``bytearray`` in tests -- "the same shared
    region" is whatever outlives the store objects built over it.  ``lock``
    is the region's mutual exclusion (an ``flock`` wrapper in production);
    it defaults to a process-local lock, which is exact for a single-process
    region.
    """

    def __init__(
        self,
        buffer,
        *,
        geometry: KDARecordMapGeometry,
        lock=None,
    ) -> None:
        self._view = memoryview(buffer)
        self._geometry = geometry
        self._lock = lock if lock is not None else threading.Lock()
        self._entry_size = _align8(
            _ENTRY_HEAD.size
            + 4 * len(geometry.group_ids) * int(geometry.pages_per_group)
        )
        self._entries_base = _align8(_HEADER.size + 4 * len(geometry.group_ids))
        required = self.required_bytes(geometry)
        if len(self._view) < required:
            raise KDARecordMapLayoutMismatch(
                f"map region holds {len(self._view)} byte(s) but this "
                f"geometry needs {required}"
            )
        with self._lock:
            header = _HEADER.unpack_from(self._view, 0)
            if header[0] == b"\x00" * 8:
                # A fresh region (memfd bytes are zero-filled): write the
                # header once.  Generation starts at 1 so entry generation 0
                # can never collide with a live header generation.
                _HEADER.pack_into(
                    self._view,
                    0,
                    _MAGIC,
                    _VERSION,
                    1,
                    int(geometry.capacity),
                    len(geometry.group_ids),
                    int(geometry.pages_per_group),
                    int(geometry.unit_count),
                    int(geometry.host_page_bytes_all_layers),
                )
                struct.pack_into(
                    f"<{len(geometry.group_ids)}i",
                    self._view,
                    _HEADER.size,
                    *[int(g) for g in geometry.group_ids],
                )
            else:
                self._validate_header(header)
            self._slot_by_sid: dict[int, int] = {}
            self._free_slots: list[int] = []
            self._rescan_locked()

    # ------------------------------------------------------------- attach

    @classmethod
    def from_existing(cls, buffer, *, lock=None) -> "KDARecordSharedMap":
        """Attach to a region some earlier cohort already initialized.

        Self-describing: the geometry is read from the header, so the
        recovery sweep can run before this cohort knows the model's record
        geometry.  A region that was never initialized is refused by name.
        """

        view = memoryview(buffer)
        if len(view) < _HEADER.size:
            raise KDARecordMapCorrupt(
                f"map region of {len(view)} byte(s) cannot hold a header"
            )
        (
            magic,
            version,
            _generation,
            capacity,
            group_count,
            pages_per_group,
            unit_count,
            host_page_bytes,
        ) = _HEADER.unpack_from(view, 0)
        if magic != _MAGIC or int(version) != _VERSION:
            raise KDARecordMapCorrupt(
                f"map region header {magic!r} v{version} is not "
                f"{_MAGIC!r} v{_VERSION}"
            )
        group_ids = struct.unpack_from(
            f"<{int(group_count)}i", view, _HEADER.size
        )
        geometry = KDARecordMapGeometry(
            capacity=int(capacity),
            group_ids=tuple(int(g) for g in group_ids),
            pages_per_group=int(pages_per_group),
            unit_count=int(unit_count),
            host_page_bytes_all_layers=int(host_page_bytes),
        )
        return cls(buffer, geometry=geometry, lock=lock)

    @staticmethod
    def required_bytes(geometry: KDARecordMapGeometry) -> int:
        """The region's exact byte size for ``geometry`` -- no slack."""

        entry_size = _align8(
            _ENTRY_HEAD.size
            + 4 * len(geometry.group_ids) * int(geometry.pages_per_group)
        )
        return (
            _align8(_HEADER.size + 4 * len(geometry.group_ids))
            + int(geometry.capacity) * entry_size
        )

    def _validate_header(self, header) -> None:
        (
            magic,
            version,
            _generation,
            capacity,
            group_count,
            pages_per_group,
            unit_count,
            host_page_bytes,
        ) = header
        geometry = self._geometry
        if magic != _MAGIC or int(version) != _VERSION:
            raise KDARecordMapCorrupt(
                f"map region header {magic!r} v{version} is not "
                f"{_MAGIC!r} v{_VERSION}"
            )
        persisted_groups = tuple(
            int(g)
            for g in struct.unpack_from(
                f"<{int(group_count)}i", self._view, _HEADER.size
            )
        )
        persisted = (
            int(capacity),
            persisted_groups,
            int(pages_per_group),
            int(unit_count),
            int(host_page_bytes),
        )
        expected = (
            int(geometry.capacity),
            tuple(int(g) for g in geometry.group_ids),
            int(geometry.pages_per_group),
            int(geometry.unit_count),
            int(geometry.host_page_bytes_all_layers),
        )
        if persisted != expected:
            raise KDARecordMapLayoutMismatch(
                "persisted KDA record map geometry "
                f"(capacity, group_ids, pages_per_group, unit_count, "
                f"host_page_bytes)={persisted} does not match this cohort's "
                f"{expected}; the region belongs to another deployment"
            )

    # ----------------------------------------------------------- geometry

    @property
    def geometry(self) -> KDARecordMapGeometry:
        return self._geometry

    @property
    def pages_per_group(self) -> int:
        return int(self._geometry.pages_per_group)

    @property
    def unit_count(self) -> int:
        return int(self._geometry.unit_count)

    @property
    def host_page_bytes_all_layers(self) -> int:
        return int(self._geometry.host_page_bytes_all_layers)

    @property
    def generation(self) -> int:
        with self._lock:
            return int(_HEADER.unpack_from(self._view, 0)[2])

    # ---------------------------------------------------------- low level

    def _slot_offset(self, slot: int) -> int:
        return self._entries_base + int(slot) * self._entry_size

    def _read_slot(self, slot: int) -> KDARecordMapEntry | None:
        offset = self._slot_offset(slot)
        (
            state,
            written,
            retained,
            _pad,
            owner_rank,
            generation,
            _pad2,
            sequence_id,
            pseudo_sid,
            token_count,
        ) = _ENTRY_HEAD.unpack_from(self._view, offset)
        if int(state) != _STATE_LIVE:
            return None
        geometry = self._geometry
        pages_flat = struct.unpack_from(
            f"<{len(geometry.group_ids) * geometry.pages_per_group}i",
            self._view,
            offset + _ENTRY_HEAD.size,
        )
        pages_per_group = int(geometry.pages_per_group)
        page_ids_by_group = {
            int(group_id): tuple(
                int(page)
                for page in pages_flat[
                    index * pages_per_group : (index + 1) * pages_per_group
                ]
            )
            for index, group_id in enumerate(geometry.group_ids)
        }
        return KDARecordMapEntry(
            sequence_id=int(sequence_id),
            pseudo_sid=int(pseudo_sid),
            owner_rank=int(owner_rank),
            generation=int(generation),
            written=bool(written),
            retained=bool(retained),
            token_count=int(token_count),
            page_ids_by_group=page_ids_by_group,
            pages_per_group=pages_per_group,
            unit_count=int(geometry.unit_count),
            host_page_bytes_all_layers=int(
                geometry.host_page_bytes_all_layers
            ),
        )

    def _rescan_locked(self) -> None:
        """Rebuild the process-local slot index from the region's bytes.

        The index is a cache: the region stays the single source of truth,
        and every claim re-verifies its slot under the lock, so a stale
        index entry costs a rescan, never a wrong claim.
        """

        self._slot_by_sid.clear()
        self._free_slots.clear()
        for slot in range(int(self._geometry.capacity)):
            entry = self._read_slot(slot)
            if entry is None:
                self._free_slots.append(slot)
            else:
                self._slot_by_sid[int(entry.sequence_id)] = slot
        # Pop from the end; keep low slots first for determinism.
        self._free_slots.reverse()

    def _find_slot_locked(
        self, sequence_id: int, *, rescan_on_miss: bool = True
    ) -> int | None:
        """The live slot holding ``sequence_id``, verified against the bytes.

        ``rescan_on_miss=False`` is the registration fast path: a sequence
        id has exactly ONE writer at a time (its owner rank), and this
        process's index was built from a full scan at construction, so an
        index miss on a sid this process may write is authoritative -- no
        other process can have registered it since.
        """

        sid = int(sequence_id)
        slot = self._slot_by_sid.get(sid)
        if slot is not None:
            entry = self._read_slot(slot)
            if entry is not None and int(entry.sequence_id) == sid:
                return slot
            self._slot_by_sid.pop(sid, None)
        if not rescan_on_miss:
            return None
        # The cache missed (another process mutated the region): rescan once.
        self._rescan_locked()
        return self._slot_by_sid.get(sid)

    def _write_slot_locked(
        self,
        slot: int,
        *,
        sequence_id: int,
        pseudo_sid: int,
        owner_rank: int,
        generation: int,
        written: bool,
        retained: bool,
        token_count: int,
        page_ids_by_group: Mapping[int, Iterable[int]],
    ) -> None:
        geometry = self._geometry
        if tuple(sorted(int(g) for g in page_ids_by_group)) != tuple(
            geometry.group_ids
        ):
            raise KDARecordMapLayoutMismatch(
                f"record of sequence {int(sequence_id)} spans KV groups "
                f"{sorted(int(g) for g in page_ids_by_group)}, the map holds "
                f"{list(geometry.group_ids)}"
            )
        pages_flat: list[int] = []
        for group_id in geometry.group_ids:
            pages = [int(page) for page in page_ids_by_group[group_id]]
            if len(pages) != int(geometry.pages_per_group):
                raise KDARecordMapLayoutMismatch(
                    f"record of sequence {int(sequence_id)} holds "
                    f"{len(pages)} page(s) in group {group_id}, the map "
                    f"holds exactly {geometry.pages_per_group} per group"
                )
            pages_flat.extend(pages)
        offset = self._slot_offset(slot)
        # Pages and head first, the LIVE state byte last: a reader that sees
        # the state sees the whole entry.
        struct.pack_into(
            f"<{len(pages_flat)}i",
            self._view,
            offset + _ENTRY_HEAD.size,
            *pages_flat,
        )
        _ENTRY_HEAD.pack_into(
            self._view,
            offset,
            _STATE_EMPTY,
            1 if written else 0,
            1 if retained else 0,
            0,
            int(owner_rank),
            int(generation),
            0,
            int(sequence_id),
            int(pseudo_sid),
            int(token_count),
        )
        struct.pack_into("<B", self._view, offset, _STATE_LIVE)

    # ------------------------------------------------------------ queries

    def entries(self) -> dict[int, KDARecordMapEntry]:
        """Every live entry, keyed by sequence id (a fresh snapshot)."""

        with self._lock:
            self._rescan_locked()
            return {
                sid: self._read_slot(slot)
                for sid, slot in sorted(self._slot_by_sid.items())
            }

    def get(self, sequence_id: int) -> KDARecordMapEntry | None:
        with self._lock:
            slot = self._find_slot_locked(sequence_id)
            return None if slot is None else self._read_slot(slot)

    def __len__(self) -> int:
        with self._lock:
            self._rescan_locked()
            return len(self._slot_by_sid)

    # ---------------------------------------------------------- mutations

    def put(
        self,
        *,
        sequence_id: int,
        arena: SnapshotArena,
        owner_rank: int,
        written: bool,
        retained: bool,
        token_count: int,
    ) -> None:
        """Register or update the OWNER's entry for ``sequence_id``.

        Followers never call this: their mirrors travel through the host-KV
        mirror payload, so pages are named exactly once in the map.
        """

        if int(arena.pages_per_group) != int(self._geometry.pages_per_group):
            raise KDARecordMapLayoutMismatch(
                f"arena of sequence {int(sequence_id)} holds "
                f"{arena.pages_per_group} page(s) per group, the map holds "
                f"exactly {self._geometry.pages_per_group}"
            )
        if int(arena.unit_count) != int(self._geometry.unit_count):
            raise KDARecordMapLayoutMismatch(
                f"arena of sequence {int(sequence_id)} splits into "
                f"{arena.unit_count} unit(s), the map records "
                f"{self._geometry.unit_count}"
            )
        with self._lock:
            generation = int(_HEADER.unpack_from(self._view, 0)[2])
            slot = self._find_slot_locked(sequence_id, rescan_on_miss=False)
            if slot is None:
                while self._free_slots:
                    candidate = self._free_slots.pop()
                    if self._read_slot(candidate) is None:
                        slot = candidate
                        break
                if slot is None:
                    # Another process may have freed slots since the last
                    # scan; look once more before declaring the bug.
                    self._rescan_locked()
                    while self._free_slots:
                        candidate = self._free_slots.pop()
                        if self._read_slot(candidate) is None:
                            slot = candidate
                            break
                if slot is None:
                    raise KDARecordMapFull(
                        f"KDA record map has no free slot for sequence "
                        f"{int(sequence_id)}: all {self._geometry.capacity} "
                        "slot(s) hold live records, but the map is sized for "
                        "every sequence the node can hold at once"
                    )
            else:
                # An update keeps the entry's registration generation: the
                # current-vs-survivor distinction must not flip because the
                # owner stamped a later lifecycle step.
                generation = int(self._read_slot(slot).generation)
            self._write_slot_locked(
                slot,
                sequence_id=int(sequence_id),
                pseudo_sid=int(arena.pseudo_sid),
                owner_rank=int(owner_rank),
                generation=generation,
                written=bool(written),
                retained=bool(retained),
                token_count=int(token_count),
                page_ids_by_group=arena.page_ids_by_group,
            )
            self._slot_by_sid[int(sequence_id)] = slot

    def remove(self, sequence_id: int) -> None:
        """Drop the entry of a released record.  Absence is a bug by name."""

        with self._lock:
            slot = self._find_slot_locked(sequence_id)
            if slot is None:
                raise KDARecordMapCorrupt(
                    f"sequence {int(sequence_id)} holds no entry in the KDA "
                    "record map, but its record was just accounted for"
                )
            struct.pack_into("<B", self._view, self._slot_offset(slot), _STATE_EMPTY)
            self._slot_by_sid.pop(int(sequence_id), None)
            self._free_slots.append(slot)

    def bump_generation(self) -> int:
        """Open a new cohort generation; returns it.  Recovery-sweep only.

        Every entry written from now on carries the new generation; entries
        still carrying an older one are the previous cohorts' -- dead
        sequences to release, or survivors to rehydrate.
        """

        with self._lock:
            header = list(_HEADER.unpack_from(self._view, 0))
            header[2] = int(header[2]) % 0xFFFFFFFF + 1
            _HEADER.pack_into(self._view, 0, *header)
            return int(header[2])


# ---------------------------------------------------------------------------
# The server-owned memfd region (production home of the map)
# ---------------------------------------------------------------------------


class KDARecordMapRegion:
    """One rank's handle onto the server-owned map memfd.

    Doubles as the map's cross-process lock: entering the context takes an
    exclusive ``flock`` on the region's file (memfd locks contend across
    processes by inode), leaving releases it.  ``close`` drops this rank's
    descriptor and mapping; the server's own open fd keeps the region --
    and everything persisted in it -- alive across pool restarts.
    """

    def __init__(self, fd: int, mapping) -> None:
        self._fd = int(fd)
        self.mapping = mapping

    @property
    def buffer(self):
        return self.mapping

    def __enter__(self) -> "KDARecordMapRegion":
        import fcntl

        fcntl.flock(self._fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        import fcntl

        fcntl.flock(self._fd, fcntl.LOCK_UN)

    def close(self) -> None:
        if self.mapping is not None:
            try:
                self.mapping.close()
            except (BufferError, ValueError):
                # A live memoryview still references the mapping; dropping
                # the fd below is what matters for the descriptor budget.
                pass
            self.mapping = None
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = -1


def attach_kda_record_map_region(
    *,
    creator_pid: int,
    memfd_fd: int,
    required_bytes: int | None = None,
) -> KDARecordMapRegion | None:
    """Open the server's map memfd through ``/proc/<creator>/fd/<N>``.

    With ``required_bytes`` (the store-build path) an empty region is sized
    to exactly that many zero-filled bytes -- idempotent across the node's
    ranks, which all compute the same figure from the same constants -- and
    a non-empty region of any other size is refused by name.  Without it
    (the recovery sweep) an empty region means nothing was ever persisted
    and the attach returns ``None``.
    """

    if int(creator_pid) < 0 or int(memfd_fd) < 0:
        raise KDARecordMapError(
            "the server published no KDA record map memfd "
            f"(pid={creator_pid}, fd={memfd_fd}); the per-sequence KDA "
            "lifecycle cannot run without its durable map"
        )
    import fcntl
    import mmap

    fd = os.open(
        f"/proc/{int(creator_pid)}/fd/{int(memfd_fd)}",
        os.O_RDWR | os.O_CLOEXEC,
    )
    mapping = None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            size = os.fstat(fd).st_size
            if size == 0:
                if required_bytes is None:
                    return None
                os.ftruncate(fd, int(required_bytes))
                size = int(required_bytes)
            elif required_bytes is not None and size != int(required_bytes):
                raise KDARecordMapLayoutMismatch(
                    f"KDA record map region holds {size} byte(s); this "
                    f"cohort's geometry needs exactly {int(required_bytes)}"
                )
            mapping = mmap.mmap(
                fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
            )
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        # Covers the nothing-persisted return and every raise: a region
        # without a mapping has no holder, so the descriptor goes back.
        if mapping is None:
            try:
                os.close(fd)
            except OSError:
                pass
    return KDARecordMapRegion(fd, mapping)
