"""Per-sequence KDA state record store -- the default host lifecycle.

One record per LIVE sequence, keyed by the GLOBAL sequence id.  The record
holds the sequence's whole recurrent (KDA) state in Host KV pages, split into
``unit_count`` equal head units (the model's largest implemented attention
group size), so a record written under one attention group size is readable
under every other implemented one.

This store is NOT the trajectory frontier store.  The frontier store serves
the optional cross-request prefix-reuse channel and is session-keyed by token
content, dedup-installed, publish-boundary aligned, debug-key gated and
capacity-skippable.  This store is the mandatory lifecycle carrier approved as
方案甲 (POIS 2026-11-25): keyed by sequence identity, one record per sequence
with no dedup, written at the sequence's REAL prefill end with no alignment
requirement, and its pages are RESERVED at prefill admission in the same Host
page pool as KV -- so a committed write can never hit out-of-space, and there
is deliberately no skip path.  The two stores coexist and never share records.

Lifecycle of one record:

* ``reserve``      -- prefill admission, host-KV owner rank only.  Takes the
                      arena's pages out of the Host free pool (the planner has
                      already charged them, exactly).
* ``adopt``        -- G>1 followers install the owner's arena page ids from
                      the host-KV mirror payload; they own no pages.
* ``write_rank_units`` -- the sequence's real prefill end, EVERY rank of the
                      owning group: split this rank's live shard into its unit
                      span and copy each unit into its own page window.
* ``retain``       -- post-barrier, owner only: the pages become resident and
                      the pseudo sequence row is dropped.
* ``restore_rank_units`` -- decode admission, every rank of the serving
                      group: read this rank's unit span (under the DECODE
                      layout's group size, which may differ from the writing
                      one) and bind it into a freshly allocated state slot.
* ``release``      -- growth eviction or completion, owner only: the pages go
                      back to the Host pool.  ``drop_mirror`` is the
                      followers' rank-local counterpart.

Exact arithmetic: the per-sequence page count is
``FrontierCharge.arena_host_pages`` at ``page_count=0`` -- U units, each
rounded up to whole Host pages independently -- a model constant at fixed U.
The state's byte size does not depend on the token count, so the count is the
same for every sequence of the model; the token count still stamps the record
so a restore can assert what boundary the state describes.

This module never imports model-family code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Sequence

from batchgen.prefix_reuse.trajectory_commit import (
    SnapshotArena,
    SnapshotArenaCorrupt,
    SnapshotArenaUnavailable,
    SnapshotLayout,
    merge_unit_chunks,
    rank_unit_window,
    split_chunks_into_units,
)
from batchgen.prefix_reuse.trajectory_frontier import (
    FrontierCharge,
    KDASnapshotProfile,
    unit_profile,
)

PREFIX = "KDA_RECORD"


class KDARecordStoreError(RuntimeError):
    """Base error of the per-sequence KDA record store."""


class KDARecordReservationUnavailable(KDARecordStoreError):
    """Admission refusal: the Host pool cannot hold the sequence's record.

    Raised BY NAME at prefill admission, before any state exists on GPU, so
    the sequence stays queueing.  It is the only capacity outcome this store
    has: after a successful reservation every later step is a hard error.
    """


class KDARecordMissing(KDARecordStoreError):
    """A lifecycle step named a sequence this store holds no record for."""


class KDARecordStateInvalid(KDARecordStoreError):
    """A record was used out of lifecycle order (double reserve/write/...)."""


@dataclass
class KDASequenceRecord:
    """One sequence's host state record, as this rank sees it."""

    sequence_id: int
    arena: SnapshotArena
    # True on a rank that adopted the owner's page ids through the mirror:
    # it reads and writes windows of the owner's arena but never mutates
    # allocator state (allocate/retain/release stay with the owner).
    mirrored: bool
    # Pages made prefix-resident (pseudo row dropped); release then goes
    # through release_resident_pages.  Before retain the pages are still
    # chain-owned by the arena's pseudo sequence and leave via rollback.
    retained: bool = False
    # The sequence's TRUE token count at the prefill end that wrote the
    # record; 0 until written.  No alignment is required of it.
    token_count: int = 0
    # Bytes THIS rank wrote into its own unit windows (diagnostic).
    written_bytes: int = 0
    written: bool = False


class KDARecordStore:
    """Per-rank store of per-sequence KDA state records.

    The construction-time ``group_size``/``group_rank`` describe the layout
    the PREFILL side writes under.  The restore side passes its own group
    geometry per call, because the decode layout may shard heads differently;
    the unit record is what makes both reads well-defined.
    """

    def __init__(
        self,
        *,
        rank: int,
        group_size: int,
        group_rank: int,
        unit_count: int,
        rank_profile: KDASnapshotProfile,
        host_page_bytes_all_layers: int,
        arena_allocator,
        arena_reader,
        snapshot_reader,
        state_writer,
        logger=logging,
    ) -> None:
        if int(host_page_bytes_all_layers) <= 0:
            raise KDARecordStateInvalid(
                "host_page_bytes_all_layers must be positive, got "
                f"{host_page_bytes_all_layers}"
            )
        self._rank = int(rank)
        self._group_size = int(group_size)
        self._group_rank = int(group_rank)
        self._unit_count = int(unit_count)
        self._rank_profile = rank_profile
        # Geometry of ONE record head unit; owns every divisibility rule.
        self._record_unit_profile = unit_profile(
            rank_profile,
            group_size=self._group_size,
            unit_count=self._unit_count,
        )
        self._unit_layout = SnapshotLayout(profile=self._record_unit_profile)
        self._host_page_bytes_all_layers = int(host_page_bytes_all_layers)
        self._arena_allocator = arena_allocator
        self._arena_reader = arena_reader
        self._snapshot_reader = snapshot_reader
        self._state_writer = state_writer
        self._logger = logger
        self._records: dict[int, KDASequenceRecord] = {}
        # This rank's unit span under the WRITING layout, fixed at build.
        self._write_unit_indices = rank_unit_window(
            unit_count=self._unit_count,
            group_size=self._group_size,
            group_rank=self._group_rank,
        )

    # ------------------------------------------------------------- geometry

    @property
    def unit_count(self) -> int:
        return self._unit_count

    @property
    def unit_layout(self) -> SnapshotLayout:
        return self._unit_layout

    @property
    def record_pages(self) -> int:
        """Host pages ONE sequence's record costs -- a model constant at U.

        ``FrontierCharge`` stays the one owner of "state bytes -> whole Host
        pages": U units, each rounded up to whole pages independently, with no
        prefix pages in the charge (``page_count=0``).
        """

        return int(
            FrontierCharge(
                page_count=0,
                host_page_bytes_all_layers=self._host_page_bytes_all_layers,
                snapshot_bytes=self._unit_layout.total_bytes,
                unit_count=self._unit_count,
            ).arena_host_pages
        )

    # -------------------------------------------------------------- lookups

    def has_record(self, sequence_id: int) -> bool:
        return int(sequence_id) in self._records

    def is_written(self, sequence_id: int) -> bool:
        record = self._records.get(int(sequence_id))
        return bool(record is not None and record.written)

    def record_for(self, sequence_id: int) -> KDASequenceRecord:
        record = self._records.get(int(sequence_id))
        if record is None:
            raise KDARecordMissing(
                f"sequence {int(sequence_id)} holds no KDA state record on "
                f"rank {self._rank}"
            )
        return record

    def sequence_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._records))

    def state_writer_has_slot(self, sequence_id: int) -> bool:
        """Whether the sequence holds a LIVE GPU state slot on this rank."""

        return bool(self._state_writer.has_slot(int(sequence_id)))

    # ---------------------------------------------------------- reservation

    def reserve(self, sequence_id: int) -> KDASequenceRecord:
        """Owner only: take the record's pages out of the Host free pool.

        Runs at prefill admission.  A refusal is the NAMED admission outcome
        ``KDARecordReservationUnavailable``; any other allocator failure is a
        bug and propagates as itself.
        """

        sid = int(sequence_id)
        if sid in self._records:
            raise KDARecordStateInvalid(
                f"sequence {sid} already holds a KDA state record; a second "
                "reservation would leak the first arena"
            )
        try:
            arena = self._arena_allocator.allocate(
                page_count=self.record_pages,
                unit_count=self._unit_count,
            )
        except SnapshotArenaUnavailable as exc:
            raise KDARecordReservationUnavailable(
                f"Host KV pool cannot reserve {self.record_pages} page(s) for "
                f"the KDA state record of sequence {sid}: {exc}"
            ) from exc
        record = KDASequenceRecord(
            sequence_id=sid, arena=arena, mirrored=False
        )
        self._records[sid] = record
        return record

    def mirror_payload(self, sequence_ids: Sequence[int]) -> dict:
        """The owner's arenas for the host-KV mirror, keyed by sequence id."""

        payload = {}
        for sequence_id in sequence_ids:
            record = self._records.get(int(sequence_id))
            if record is None or record.mirrored:
                continue
            arena = record.arena
            payload[int(sequence_id)] = {
                "pseudo_sid": int(arena.pseudo_sid),
                "unit_count": int(arena.unit_count),
                "pages_per_group": int(arena.pages_per_group),
                "capacity_bytes": int(arena.capacity_bytes),
                "pages": {
                    int(group_id): [int(page) for page in pages]
                    for group_id, pages in arena.page_ids_by_group.items()
                },
                # Lifecycle position travels with the arena: a mirror taken
                # at admission says "not written yet" (the adopter writes its
                # own units at the prefill end); one taken after the write
                # says what boundary the record describes.
                "written": bool(record.written),
                "token_count": int(record.token_count),
            }
        return payload

    def adopt(self, payload: Mapping[int, Mapping]) -> None:
        """Follower side: install the owner's arenas from the mirror payload.

        Idempotent per sequence: re-adoption of an id this rank already holds
        is refused, because a record's identity may not silently change.
        """

        for sequence_id, entry in (payload or {}).items():
            sid = int(sequence_id)
            if sid in self._records:
                raise KDARecordStateInvalid(
                    f"sequence {sid} already holds a KDA state record on rank "
                    f"{self._rank}; the mirror may not replace it"
                )
            arena = SnapshotArena(
                storage_handle=0,
                pseudo_sid=int(entry["pseudo_sid"]),
                page_ids_by_group={
                    int(group_id): tuple(int(page) for page in pages)
                    for group_id, pages in entry["pages"].items()
                },
                pages_per_group=int(entry["pages_per_group"]),
                capacity_bytes=int(entry["capacity_bytes"]),
                payload_bytes=0,
                unit_count=int(entry["unit_count"]),
            )
            self._records[sid] = KDASequenceRecord(
                sequence_id=sid,
                arena=arena,
                mirrored=True,
                written=bool(entry.get("written", False)),
                token_count=int(entry.get("token_count", 0)),
            )

    # -------------------------------------------------------------- writing

    def write_rank_units(
        self, sequence_id: int, *, token_count: int
    ) -> int:
        """Every rank of the owning group: write its own unit windows.

        Runs at the sequence's REAL prefill end.  ``token_count`` is the true
        token count the state describes -- NO alignment requirement, no
        publish-boundary granularity.  The record must have been reserved at
        admission, so there is no capacity outcome here: every failure is a
        hard error.
        """

        sid = int(sequence_id)
        record = self.record_for(sid)
        if record.written:
            raise KDARecordStateInvalid(
                f"sequence {sid} already wrote its KDA state record at token "
                f"count {record.token_count}; a second prefill end for the "
                "same record is a lifecycle bug"
            )
        if int(token_count) <= 0:
            raise KDARecordStateInvalid(
                f"sequence {sid} reports a non-positive token count "
                f"{token_count} at its prefill end"
            )
        chunks = self._snapshot_reader.read_slot(sid)
        unit_chunks = split_chunks_into_units(
            chunks,
            profile=self._rank_profile,
            unit_count=len(self._write_unit_indices),
        )
        written = 0
        for unit_index, per_unit in zip(self._write_unit_indices, unit_chunks):
            written += int(
                self._arena_allocator.write_unit(
                    record.arena, unit_index=unit_index, chunks=per_unit
                )
            )
        record.token_count = int(token_count)
        record.written_bytes = written
        record.written = True
        self._logger.info(
            "[%s] rank=%s seq=%s wrote units=%s/%s bytes=%s tokens=%s into "
            "its per-sequence record (pseudo_sid=%s)",
            PREFIX,
            self._rank,
            sid,
            list(self._write_unit_indices),
            self._unit_count,
            written,
            int(token_count),
            record.arena.pseudo_sid,
        )
        return written

    def retain_written(self) -> tuple[int, ...]:
        """Owner, post-barrier: make every written arena's pages resident.

        Every rank of the group has returned from its own unit writes before
        the caller's barrier, so a retained record binds a COMPLETE U-unit
        state.  Mirrored records are the owner's to retain, never this
        rank's.
        """

        retained: list[int] = []
        for sid in sorted(self._records):
            record = self._records[sid]
            if record.mirrored or record.retained or not record.written:
                continue
            self._arena_allocator.retain(record.arena)
            record.retained = True
            retained.append(sid)
        return tuple(retained)

    # ------------------------------------------------------------ restoring

    def restore_rank_units(
        self,
        sequence_id: int,
        *,
        group_size: int,
        group_rank: int,
    ) -> tuple[int, int]:
        """Read this rank's unit span and bind it into a fresh state slot.

        Runs at decode admission, on every rank of the SERVING group, whose
        ``group_size`` may differ from the writing one -- the unit record
        handles it, and a group size that does not divide the unit count is
        refused by name inside ``rank_unit_window``.  Returns
        ``(slot, bytes_written)``.
        """

        sid = int(sequence_id)
        record = self.record_for(sid)
        if not record.written:
            raise KDARecordStateInvalid(
                f"sequence {sid} reached decode admission with a KDA state "
                "record that was never written at a prefill end"
            )
        if self._state_writer.has_slot(sid):
            raise KDARecordStateInvalid(
                f"sequence {sid} still holds a live KDA state slot at decode "
                "admission; restoring over it would clobber newer state"
            )
        unit_indices = rank_unit_window(
            unit_count=int(record.arena.unit_count),
            group_size=int(group_size),
            group_rank=int(group_rank),
        )
        unit_chunks = [
            self._arena_reader.read(
                record.arena, layout=self._unit_layout, unit_index=unit_index
            )
            for unit_index in unit_indices
        ]
        chunks = merge_unit_chunks(
            unit_chunks, unit_profile=self._record_unit_profile
        )
        slot = int(self._state_writer.allocate_slot(sid))
        try:
            slot, written = self._state_writer.write_slot(sid, chunks)
        except Exception:
            # The slot was taken by THIS restore and holds half-written
            # state; hand it back so no later sequence inherits the bytes.
            try:
                self._state_writer.release_slot(sid)
            except Exception:  # noqa: BLE001 - cleanup is best effort
                self._logger.exception(
                    "[%s] rank=%s seq=%s could not release the KDA slot a "
                    "failed lifecycle restore had taken",
                    PREFIX,
                    self._rank,
                    sid,
                )
            raise
        self._logger.info(
            "[%s] rank=%s seq=%s RESTORED group_rank=%s/%s units=%s slot=%s "
            "bytes=%s tokens=%s",
            PREFIX,
            self._rank,
            sid,
            int(group_rank),
            int(group_size),
            list(unit_indices),
            slot,
            written,
            record.token_count,
        )
        return int(slot), int(written)

    # ------------------------------------------------------------- release

    def release(self, sequence_id: int) -> int:
        """Owner only: give the record's pages back; returns the page count.

        A retained record releases resident pages; one still under its pseudo
        row (reserved, written-but-not-retained) rolls the allocation back.
        Raises ``KDARecordMissing`` for an id this rank holds no record for
        and refuses a mirrored record by name -- the owner path on a follower
        is a bug, not a no-op.
        """

        sid = int(sequence_id)
        record = self.record_for(sid)
        if record.mirrored:
            raise KDARecordStateInvalid(
                f"sequence {sid}'s KDA record on rank {self._rank} is a "
                "mirror of the owner's; followers drop mirrors, never "
                "release pages"
            )
        if record.retained:
            self._arena_allocator.release(record.arena)
        else:
            self._arena_allocator.rollback(record.arena)
        del self._records[sid]
        return int(record.arena.pages_per_group)

    def drop_mirror(self, sequence_id: int) -> bool:
        """Follower side: forget an adopted record.  Pages stay the owner's.

        Tolerant of absence: a follower only holds mirrors for sequences the
        mirror delivered, and the release seams name every row they touch.
        """

        record = self._records.get(int(sequence_id))
        if record is None:
            return False
        if not record.mirrored:
            raise KDARecordStateInvalid(
                f"sequence {int(sequence_id)}'s KDA record on rank "
                f"{self._rank} owns its pages; dropping it as a mirror would "
                "leak them"
            )
        del self._records[int(sequence_id)]
        return True

    def rollback_reserved(self, sequence_ids: Sequence[int]) -> tuple[int, ...]:
        """Owner only: give back reservations of a wave that failed admission.

        Tolerant of ids with no record (the wave may have failed before this
        rank reserved anything); every record it does find must be fresh
        (never written), because this path exists only between reservation
        and the wave's prefill.
        """

        rolled_back: list[int] = []
        for sequence_id in sequence_ids:
            record = self._records.get(int(sequence_id))
            if record is None:
                continue
            if record.mirrored or record.written or record.retained:
                raise KDARecordStateInvalid(
                    f"sequence {int(sequence_id)}'s KDA record is past "
                    "reservation (mirrored/written/retained); the admission "
                    "rollback may not touch it"
                )
            self._arena_allocator.rollback(record.arena)
            del self._records[int(sequence_id)]
            rolled_back.append(int(sequence_id))
        return tuple(rolled_back)
