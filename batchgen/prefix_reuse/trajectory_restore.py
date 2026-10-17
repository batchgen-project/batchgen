"""Look up a committed trajectory frontier, pin it, and restore its KDA slot.

Slice 2 of the trajectory-scoped cross-wave reuse mechanism. Slice 1a owns the
record and the binary hit predicate (``trajectory_frontier``); Slice 1b owns the
commit transaction and where the snapshot bytes physically live
(``trajectory_commit``). This module owns the OTHER direction:

1. **Lookup** -- resolve an incoming prompt against the frontier registry. The
   predicate is Slice 1a's ``is_frontier_hit``: hit iff the committed frontier
   is a prefix of the new prompt. A partial LCP is a FULL miss, never a partial
   hit (note section 3: without KDA state at the LCP the MLA pages below it are
   equally unusable).
2. **Attach** -- take a reference on the record so it cannot be evicted between
   admission and the end of the restore. The MLA pages are pinned by the
   coordinator attachment the prefix lookup already took; this reference pins
   the OTHER half, the snapshot, and the registry entry that binds the two.
3. **Restore** -- read the snapshot arena back out of its Host KV pages and
   write it into the sequence's KDA state slot, before its suffix prefill runs.
4. **Release** -- an attachment is consumed exactly once, on success or on any
   fallback, so the live-attachment counter returns to zero when the batch
   drains.

Fail-safe by construction: every refusal releases the attachment and reports a
named reason, and the sequence then proceeds as a full recompute. Nothing here
blocks a wave, waits for a slot, or raises into the prefill path.

The H2D ordering contract
-------------------------
``KDAStateSnapshotWriter.write_slot`` copies from a PAGEABLE host buffer into
the live state pool tensors with ``Tensor.copy_``, issued on the worker's
CURRENT stream. Two independent facts make the restored bytes visible to the
suffix prefill:

* a copy from pageable host memory is synchronizing -- it has completed on the
  device before ``copy_`` returns, because the driver may not retain the host
  source buffer;
* the copy is issued on the same stream the prefill forward is later launched
  on, so even a staged copy is stream-ordered BEFORE every kernel of that
  forward.

The call site adds the third: the restore runs in the worker's host-side
sequence strictly before the prefill forward is launched for that wave, so no
suffix-prefill kernel can be enqueued ahead of the copy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from batchgen.prefix_reuse.trajectory_commit import (
    PREFIX as COMMIT_LOG_PREFIX,
    SNAPSHOT_CHUNK_NAMES,
    SnapshotArena,
    SnapshotArenaCorrupt,
    SnapshotLayout,
    TrajectorySnapshotShapeMismatch,
    TrajectoryStateUnavailable,
)
from batchgen.prefix_reuse.trajectory_frontier import (
    TrajectoryFrontierRecord,
)


# Slice 2 shares Slice 1b's batch-level flag: ONE flag governs the whole
# trajectory path, so an operator cannot turn commits on and lookups off and
# get a half-wired mechanism. Re-exported so the worker has one import site.
from batchgen.prefix_reuse.trajectory_commit import (  # noqa: E402
    DEBUG_FLAG,
    trajectory_commit_enabled as trajectory_reuse_enabled,
)

PREFIX = "TRAJECTORY_RESTORE"

# The ONE recoverable KDA failure: ``KDAStateGPUManager._ensure_state_item``
# refuses with this message when the free list is empty. Slot exhaustion is a
# capacity outcome, not a bug, and it degrades to recompute without waiting.
_SLOT_PRESSURE_MARKERS = ("Insufficient free KDA state items",)


# ---------------------------------------------------------------------------
# Errors and declared reasons -- every refusal is named.
# ---------------------------------------------------------------------------


class TrajectoryRestoreError(RuntimeError):
    """Base class for every trajectory-restore refusal."""


class KDAStateSlotsExhausted(TrajectoryRestoreError):
    """No free KDA state slot to restore a snapshot into.

    The fail-safe outcome: the attachment is released and the sequence
    recomputes its whole prompt. Never a wait, never a raise into prefill.
    """


class TrajectoryAttachmentInvalid(TrajectoryRestoreError):
    """An attachment was used in a state the mechanism does not allow."""


class _RestoreDeclined(Exception):
    """Internal: a named reason the restore core could not finish.

    Never escapes the module. It exists so the attachment path and the
    mirrored follower path share ONE implementation of read-slot-write and
    still report the same named reasons: two copies of that sequence would be
    two places for the fallback accounting to drift.
    """

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = str(reason)
        self.detail = str(detail)


# Reasons a hit does NOT end in a restore. Strings, not booleans, because the
# fallback log line and the metrics row both have to say WHICH one happened.
#
# ``head_sharded_group`` no longer means "G>1 at all". Slice 1c commits every
# rank's head shard into one arena, so a G>1 record is restorable; what is NOT
# restorable is a record whose SHARD COUNT is not this rank's group size. Its
# windows then describe a different head split, and reading window r of it
# would bind plausible-looking wrong state. That is the refusal this name now
# carries.
DECLINE_HEAD_SHARDED_GROUP = "head_sharded_group"
DECLINE_GROUP_MISMATCH = "group_mismatch"
DECLINE_SUFFIX_OFFSET = "suffix_offset_mismatch"
DECLINE_ARENA_MISSING = "arena_missing"
DECLINE_ARENA_READ_FAILED = "arena_read_failed"
DECLINE_SLOTS_EXHAUSTED = "slots_exhausted"
DECLINE_SLOT_WRITE_FAILED = "slot_write_failed"

DECLINE_REASONS = (
    DECLINE_HEAD_SHARDED_GROUP,
    DECLINE_GROUP_MISMATCH,
    DECLINE_SUFFIX_OFFSET,
    DECLINE_ARENA_MISSING,
    DECLINE_ARENA_READ_FAILED,
    DECLINE_SLOTS_EXHAUSTED,
    DECLINE_SLOT_WRITE_FAILED,
)


# ---------------------------------------------------------------------------
# The attachment
# ---------------------------------------------------------------------------


class TrajectoryAttachment:
    """One live reference to a committed frontier record.

    Created by ``TrajectoryRestorer.lookup`` on a hit and consumed exactly once
    by ``restore`` or ``release``. ``consumed`` is what the post-drain counter
    is built on: an attachment that is neither restored nor released is a leak,
    and the registry would refuse to evict its record forever.
    """

    __slots__ = ("_record", "_sequence_id", "_lcp_tokens", "_consumed")

    def __init__(
        self,
        *,
        record: TrajectoryFrontierRecord,
        sequence_id: int,
        lcp_tokens: int,
    ) -> None:
        self._record = record
        self._sequence_id = int(sequence_id)
        self._lcp_tokens = int(lcp_tokens)
        self._consumed = False

    @property
    def record(self) -> TrajectoryFrontierRecord:
        return self._record

    @property
    def session_key(self) -> str:
        return self._record.session_key

    @property
    def sequence_id(self) -> int:
        return self._sequence_id

    @property
    def lcp_tokens(self) -> int:
        return self._lcp_tokens

    @property
    def boundary_tokens(self) -> int:
        return self._record.frontier_boundary_tokens

    @property
    def owning_group_id(self) -> int:
        return self._record.owning_group_id

    @property
    def storage_handle(self) -> int:
        return int(self._record.snapshot.storage_handle)

    @property
    def consumed(self) -> bool:
        return self._consumed

    def _consume(self) -> None:
        if self._consumed:
            raise TrajectoryAttachmentInvalid(
                f"trajectory attachment for session {self.session_key} "
                f"(sequence {self._sequence_id}) was already consumed; a "
                "second release would drop a reference it does not hold"
            )
        self._consumed = True


@dataclass(frozen=True)
class TrajectoryRestoreOutcome:
    """What one successful restore did."""

    session_key: str
    sequence_id: int
    state_slot: int
    boundary_tokens: int
    owning_group_id: int
    restored_bytes: int
    allocated_slot: bool


# ---------------------------------------------------------------------------
# The arena read boundary (tests monkeypatch this)
# ---------------------------------------------------------------------------


class HostKVSnapshotArenaReader:
    """Read a snapshot blob back out of its Host KV pages.

    The exact reverse of ``HostKVSnapshotArenaAllocator._write_blob``, and it
    reads the arena the same way that method wrote it: in ASCENDING group
    order, each group's whole page-table byte window concatenated, the blob
    starting at offset 0 and allowed to straddle a group boundary. Groups past
    the end of the blob are reserved capacity and are not read, exactly as they
    were not written.

    The pages are prefix-resident -- the pseudo sequence that allocated them is
    long gone -- so a row has to exist to resolve them. A fresh NEGATIVE pseudo
    sid is registered, the arena's pages are attached to it as SHARED prefix
    pages (``attach_shared_prefix_pages`` never transfers ownership, so the
    backend still owns them), the window is read, and the row is unregistered.
    ``unregister_sequences`` drops only the page-table entry: the arena keeps
    its pages. ``release_sequence_pages`` is deliberately NOT used here -- the
    arena's pages are not this row's to free.
    """

    def __init__(
        self,
        *,
        host_views_provider: Callable[[], Mapping[int, object]],
        next_pseudo_sid: Callable[[], int],
        torch_module=None,
    ) -> None:
        self._host_views_provider = host_views_provider
        self._next_pseudo_sid = next_pseudo_sid
        self._torch = torch_module

    def _torch_module(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def read(
        self,
        arena: SnapshotArena,
        *,
        layout: SnapshotLayout,
        shard_index: int = 0,
    ) -> tuple[object, ...]:
        """The blob's chunks in ``SNAPSHOT_CHUNK_NAMES`` order, as uint8.

        ``shard_index`` selects WHICH rank's head shard is read. The window is
        taken from ``SnapshotArena.shard_window``, the same method the commit
        side wrote through, so the read and the write cannot derive it
        differently -- a restore that computed the window itself could read a
        neighbour's shard and bind state that looks entirely plausible.
        ``shard_index=0`` of a one-shard arena is the whole arena, so the
        pure-DP read is unchanged.
        """

        torch = self._torch_module()
        views = dict(self._host_views_provider())
        if not views:
            raise SnapshotArenaCorrupt(
                "no Host KV worker views to read a snapshot arena from"
            )
        needed = int(layout.total_bytes)
        pseudo_sid = int(self._next_pseudo_sid())
        if pseudo_sid >= 0:
            raise SnapshotArenaCorrupt(
                "a snapshot arena must be read under a NEGATIVE pseudo "
                f"sequence id so it cannot collide with a real sequence, got "
                f"{pseudo_sid}"
            )
        window_by_group = arena.shard_window(int(shard_index))
        windows: list[object] = []
        collected = 0
        registered: list[int] = []
        try:
            for group_id, page_ids in sorted(window_by_group.items()):
                view = views.get(int(group_id))
                if view is None:
                    raise SnapshotArenaCorrupt(
                        "missing Host KV worker view for snapshot arena group "
                        f"{group_id}"
                    )
                view.register_sequences([pseudo_sid])
                registered.append(int(group_id))
                view.attach_shared_prefix_pages(
                    pseudo_sid, [int(page) for page in page_ids]
                )
                window = view.read_sequence_kv_to_cpu(pseudo_sid)[0]
                flat = window.reshape(-1).view(torch.uint8)
                windows.append(flat)
                collected += int(flat.numel())
                if collected >= needed:
                    break
        finally:
            self._unregister(views, registered, pseudo_sid)
        if collected < needed:
            raise SnapshotArenaCorrupt(
                f"snapshot arena handed back {collected} B of Host pages but "
                f"the blob is {needed} B; the arena no longer matches its "
                "profile"
            )
        blob = windows[0] if len(windows) == 1 else torch.cat(windows)
        chunks = tuple(
            blob[offset : offset + size]
            for offset, size in zip(layout.chunk_offsets, layout.chunk_bytes)
        )
        if len(chunks) != len(SNAPSHOT_CHUNK_NAMES):
            raise TrajectorySnapshotShapeMismatch(
                f"a KDA snapshot is {len(SNAPSHOT_CHUNK_NAMES)} chunks "
                f"({', '.join(SNAPSHOT_CHUNK_NAMES)}), the layout gave "
                f"{len(chunks)}"
            )
        return chunks

    def _unregister(
        self,
        views: Mapping[int, object],
        registered: Sequence[int],
        pseudo_sid: int,
    ) -> None:
        """Drop the read row from every group it was registered in.

        Page-table only: the arena's pages stay resident. Best effort, because
        this runs from a ``finally`` -- a failure here must not mask the read
        error that sent us into it, and it is reported by name.
        """

        for group_id in sorted({int(value) for value in registered}):
            view = views.get(int(group_id))
            if view is None:
                continue
            try:
                view.unregister_sequences([pseudo_sid])
            except Exception:  # noqa: BLE001 - cleanup is best effort
                logging.warning(
                    "[%s] rank-local unregister of snapshot read row "
                    "pseudo_sid=%s group=%s failed; the arena pages are "
                    "untouched",
                    PREFIX,
                    pseudo_sid,
                    group_id,
                )


# ---------------------------------------------------------------------------
# The H2D / slot boundary (tests monkeypatch this)
# ---------------------------------------------------------------------------


class KDAStateSnapshotWriter:
    """Write a snapshot blob into one sequence's KDA state slot.

    The reverse of ``KDAStateSnapshotReader``: the four chunks go back into
    ``recurrent[:, slot]`` and the three ``conv[:, slot]`` views, in the fixed
    ``SNAPSHOT_CHUNK_NAMES`` order the blob was written in.

    ``allocate_slot`` is the BIND: ``KDAStateGPUManager.allocate_state_item``
    records ``sequence_id -> slot``, which is the same mapping the prefill and
    decode paths resolve through, so no second binding mechanism exists. It is
    idempotent, so ``had_slot`` is captured first: a slot the sequence already
    owned must not be freed by a failed restore.
    """

    def __init__(
        self,
        *,
        state_manager_provider: Callable[[], object],
        profile,
        torch_module=None,
    ) -> None:
        self._state_manager_provider = state_manager_provider
        self._profile = profile
        self._torch = torch_module

    @property
    def profile(self):
        return self._profile

    def _torch_module(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _manager(self):
        manager = self._state_manager_provider()
        if manager is None:
            raise TrajectoryStateUnavailable(
                "this model keeps no KDA state pool, so a trajectory snapshot "
                "has nowhere to be restored into"
            )
        return manager

    def has_slot(self, sequence_id: int) -> bool:
        return bool(self._manager().has_sequence_state_item(int(sequence_id)))

    def allocate_slot(self, sequence_id: int) -> int:
        """Bind a free slot to the sequence, or report exhaustion by name."""

        manager = self._manager()
        try:
            return int(manager.allocate_state_item(int(sequence_id)))
        except Exception as exc:
            if not any(
                marker in str(exc) for marker in _SLOT_PRESSURE_MARKERS
            ):
                # Only slot exhaustion is a capacity outcome. Anything else is
                # a bug and must NOT be downgraded into it.
                raise
            raise KDAStateSlotsExhausted(
                f"no free KDA state slot for sequence {sequence_id}: {exc}"
            ) from exc

    def release_slot(self, sequence_id: int) -> None:
        self._manager().release_sequence_states([int(sequence_id)])

    def write_slot(
        self, sequence_id: int, chunks: Sequence[object]
    ) -> tuple[int, int]:
        """Copy the blob into the sequence's slot; returns (slot, bytes)."""

        manager = self._manager()
        if not manager.has_sequence_state_item(int(sequence_id)):
            raise TrajectoryStateUnavailable(
                f"sequence {sequence_id} holds no KDA state slot to restore "
                "into; allocate_slot must run first"
            )
        slot = int(manager.get_sequence_state_item(int(sequence_id)))
        recurrent = manager.get_recurrent_tensors()[:, slot]
        conv_q, conv_k, conv_v = (
            tensor[:, slot] for tensor in manager.get_conv_tensors()
        )
        targets = (recurrent, conv_q, conv_k, conv_v)
        if len(chunks) != len(targets):
            raise TrajectorySnapshotShapeMismatch(
                f"a KDA snapshot is {len(targets)} chunks "
                f"({', '.join(SNAPSHOT_CHUNK_NAMES)}), got {len(chunks)}"
            )
        written = 0
        for name, target, chunk in zip(SNAPSHOT_CHUNK_NAMES, targets, chunks):
            written += self._write_tensor(name, target, chunk)
        return slot, written

    def _write_tensor(self, name: str, target, chunk) -> int:
        """One chunk into one strided slot view, bytes checked first."""

        expected = int(target.numel()) * int(target.element_size())
        actual = int(chunk.numel())
        if actual != expected:
            raise TrajectorySnapshotShapeMismatch(
                f"KDA snapshot chunk {name!r} carries {actual} B but the slot "
                f"view holds {expected} B"
            )
        # ``[:, slot]`` is strided over the layer dim, so the DESTINATION is
        # not contiguous; ``copy_`` handles that. The SOURCE has to be
        # contiguous before it can be reinterpreted as the slot's dtype.
        source = chunk.contiguous()
        element = int(target.element_size())
        if element > 1 and int(source.storage_offset()) % element:
            # ``Tensor.view(dtype)`` needs the storage offset divisible by the
            # element-size ratio. Chunks after the first start at their byte
            # offset inside the blob, which is only mis-aligned when a later
            # chunk's dtype is WIDER than the bytes before it (a conv dtype
            # wider than the recurrent one). Copying that chunk is the one
            # case where the restore pays an extra host copy, and it is taken
            # only when the layout forces it -- never on every chunk.
            source = source.clone()
        source = source.view(target.dtype).reshape(target.shape)
        target.copy_(source)
        return expected


# ---------------------------------------------------------------------------
# The restorer
# ---------------------------------------------------------------------------


class TrajectoryRestorer:
    """Lookup, attach, restore and release, for one worker rank.

    Reads the committer's registry and its handle->arena table; owns no record
    lifetime of its own. The committer stays the single owner of what a
    frontier IS, so a restore cannot invent a record the commit path would not
    recognise.
    """

    def __init__(
        self,
        *,
        rank: int,
        committer,
        arena_reader,
        state_writer,
        attention_group_size: int = 1,
        shard_index: int = 0,
        logger=None,
    ) -> None:
        self._rank = int(rank)
        self._committer = committer
        self._arena_reader = arena_reader
        self._state_writer = state_writer
        self._attention_group_size = int(attention_group_size)
        if self._attention_group_size <= 0:
            raise TrajectoryRestoreError(
                "attention_group_size must be positive, got "
                f"{attention_group_size}"
            )
        if not 0 <= int(shard_index) < self._attention_group_size:
            raise TrajectoryRestoreError(
                f"shard_index {shard_index} is outside a "
                f"{self._attention_group_size}-rank owning group"
            )
        # WHICH head shard this rank restores: its own position inside the
        # owning group. The commit side wrote window ``shard_index`` from this
        # same rank, so the two are the same index by construction.
        self._shard_index = int(shard_index)
        self._logger = logger if logger is not None else logging
        self._live_attachments = 0
        self._lookups = 0
        self._hits = 0
        self._restores = 0
        self._declines: dict[str, int] = {}

    # -- read ---------------------------------------------------------------

    @property
    def committer(self):
        return self._committer

    @property
    def live_attachments(self) -> int:
        """Attachments taken and not yet consumed. Zero at drain."""

        return self._live_attachments

    def counters(self) -> dict[str, int]:
        """The drain-counter surface: flat ints for the metrics row."""

        fields = {
            "trajectory_lookups": int(self._lookups),
            "trajectory_hits": int(self._hits),
            "trajectory_restores": int(self._restores),
            "trajectory_live_attachments": int(self._live_attachments),
            "trajectory_declines": int(sum(self._declines.values())),
        }
        for reason in DECLINE_REASONS:
            fields[f"trajectory_decline_{reason}"] = int(
                self._declines.get(reason, 0)
            )
        return fields

    # -- lookup + attach ----------------------------------------------------

    def lookup(
        self,
        *,
        namespace_digest: Sequence[int],
        prompt_token_ids: Sequence[int],
        sequence_id: int,
    ) -> TrajectoryAttachment | None:
        """Resolve a prompt against the frontier registry and pin a hit.

        ``None`` is a miss, including a partial LCP: the predicate is binary,
        so the pages below a divergence point are never half-used. A hit takes
        a reference BEFORE returning, so the record cannot be evicted between
        here and the restore -- ``TrajectoryFrontierRegistry.evict`` refuses a
        referenced record by name, which is Slice 1a's semantics and the reason
        the reference is taken at admission rather than at materialization.
        """

        self._lookups += 1
        resolution = self._committer.registry.resolve(
            namespace_digest=namespace_digest,
            prompt_token_ids=prompt_token_ids,
        )
        if not resolution.hit:
            self._logger.info(
                "[%s] rank=%s seq=%s MISS: longest frontier agreement is %s "
                "token(s), which reaches no committed frontier boundary; the "
                "prompt is prefilled in full",
                PREFIX,
                self._rank,
                int(sequence_id),
                int(resolution.lcp_tokens),
            )
            return None
        record = resolution.record
        record.acquire()
        self._live_attachments += 1
        self._hits += 1
        attachment = TrajectoryAttachment(
            record=record,
            sequence_id=int(sequence_id),
            lcp_tokens=int(resolution.lcp_tokens),
        )
        self._logger.info(
            "[%s] rank=%s seq=%s HIT session=%s boundary=%s lcp=%s "
            "group=%s refcount=%s",
            PREFIX,
            self._rank,
            int(sequence_id),
            attachment.session_key,
            attachment.boundary_tokens,
            attachment.lcp_tokens,
            attachment.owning_group_id,
            record.refcount,
        )
        return attachment

    # -- release ------------------------------------------------------------

    def release(self, attachment: TrajectoryAttachment) -> None:
        """Consume an attachment, dropping its reference on the record.

        The registry keeps its OWN reference, so this never evicts a frontier;
        it only lets a later eviction succeed.
        """

        attachment._consume()
        self._live_attachments -= 1
        attachment.record.release()

    def decline(
        self,
        attachment: TrajectoryAttachment,
        *,
        reason: str,
        detail: str = "",
    ) -> None:
        """Release an attachment and say by name why no restore happened."""

        if reason not in DECLINE_REASONS:
            raise TrajectoryAttachmentInvalid(
                f"unknown trajectory decline reason {reason!r}; known reasons "
                "are " + ", ".join(DECLINE_REASONS)
            )
        self._declines[reason] = self._declines.get(reason, 0) + 1
        self._logger.warning(
            "[%s] rank=%s seq=%s session=%s FALLBACK reason=%s boundary=%s "
            "%s; trajectory reuse degrades to full recompute",
            PREFIX,
            self._rank,
            attachment.sequence_id,
            attachment.session_key,
            reason,
            attachment.boundary_tokens,
            detail,
        )
        self.release(attachment)

    # -- restore ------------------------------------------------------------

    def restore(
        self,
        attachment: TrajectoryAttachment,
        *,
        serving_group_id: int,
        compute_cached_tokens: int,
    ) -> TrajectoryRestoreOutcome | None:
        """Write the pinned snapshot into a KDA slot, before suffix prefill.

        ``None`` means the sequence recomputes: every path through it has
        already released the attachment and logged the reason. Nothing here
        raises into the prefill path.

        ``compute_cached_tokens`` is the position the suffix prefill will start
        at, and the restored state is the state after exactly
        ``boundary_tokens`` tokens, so the two must be EQUAL. This is an
        identity, not a tolerance: a suffix that starts one token early would
        re-apply that token on top of a state that already contains it, and a
        suffix that starts late would skip one. The full-hit case
        (``compute_cached_tokens == prompt_length - 1``, the last token
        recomputed so there is a logit to sample) therefore declines here
        rather than restoring a state the forward would double-count.
        """

        if attachment.consumed:
            raise TrajectoryAttachmentInvalid(
                f"trajectory attachment for session {attachment.session_key} "
                "was already consumed; it cannot be restored"
            )
        record_shards = int(attachment.record.shard_count)
        if record_shards != self._attention_group_size:
            # Slice 1c commits EVERY rank's head shard into one arena, so a
            # G>1 record is restorable -- as long as its shard count is this
            # rank's group size. A record taken at a different G describes a
            # different head split: window r of it is not this rank's heads,
            # and binding it would be wrong state no downstream assertion can
            # see. Refused by name rather than reinterpreted.
            self.decline(
                attachment,
                reason=DECLINE_HEAD_SHARDED_GROUP,
                detail=(
                    f"the record binds {record_shards} head shard(s) but this "
                    f"rank serves a {self._attention_group_size}-rank group"
                ),
            )
            return None
        if int(serving_group_id) != attachment.owning_group_id:
            self.decline(
                attachment,
                reason=DECLINE_GROUP_MISMATCH,
                detail=(
                    f"serving_group={int(serving_group_id)} but the snapshot "
                    f"is head-sharded for group {attachment.owning_group_id}"
                ),
            )
            return None
        if int(compute_cached_tokens) != attachment.boundary_tokens:
            self.decline(
                attachment,
                reason=DECLINE_SUFFIX_OFFSET,
                detail=(
                    f"suffix prefill starts at token "
                    f"{int(compute_cached_tokens)} but the snapshot is the "
                    f"state after {attachment.boundary_tokens} tokens"
                ),
            )
            return None

        arena = self._committer.arena_for(attachment.storage_handle)
        if arena is None:
            self.decline(
                attachment,
                reason=DECLINE_ARENA_MISSING,
                detail=(
                    f"storage handle {attachment.storage_handle} has no "
                    "snapshot arena"
                ),
            )
            return None

        sequence_id = attachment.sequence_id
        try:
            slot, written, had_slot = self._read_window_and_bind_slot(
                arena=arena, layout=self._committer.layout, sequence_id=sequence_id
            )
        except _RestoreDeclined as declined:
            self.decline(
                attachment, reason=declined.reason, detail=declined.detail
            )
            return None

        # The restore is complete: the slot holds the committed state and the
        # sequence is bound to it. The attachment has done its job, so it is
        # consumed here -- the frontier advances only at the NEXT commit, so a
        # successful restore advances nothing.
        self.release(attachment)
        self._restores += 1
        self._logger.info(
            "[%s] rank=%s seq=%s session=%s RESTORED boundary=%s slot=%s "
            "bytes=%s group=%s allocated_slot=%s; suffix prefill resumes at "
            "token %s",
            PREFIX,
            self._rank,
            sequence_id,
            attachment.session_key,
            attachment.boundary_tokens,
            slot,
            written,
            attachment.owning_group_id,
            not had_slot,
            int(compute_cached_tokens),
        )
        return TrajectoryRestoreOutcome(
            session_key=attachment.session_key,
            sequence_id=sequence_id,
            state_slot=int(slot),
            boundary_tokens=attachment.boundary_tokens,
            owning_group_id=attachment.owning_group_id,
            restored_bytes=int(written),
            allocated_slot=not had_slot,
        )

    # -- the shared core ----------------------------------------------------

    def _read_window_and_bind_slot(
        self, *, arena, layout, sequence_id: int
    ) -> tuple[int, int, bool]:
        """Read THIS rank's window and write it into the sequence's slot.

        Returns ``(slot, bytes_written, had_slot)``. Every failure leaves the
        pool exactly as it found it and is raised as ``_RestoreDeclined`` with
        the name the caller reports -- the attachment path and the mirrored
        follower path therefore produce identical reasons and identical
        cleanup.
        """

        try:
            chunks = self._arena_reader.read(
                arena, layout=layout, shard_index=self._shard_index
            )
        except Exception as exc:  # noqa: BLE001 - fail-safe by contract
            self._logger.exception(
                "[%s] rank=%s seq=%s reading the snapshot arena failed",
                PREFIX,
                self._rank,
                int(sequence_id),
            )
            raise _RestoreDeclined(
                DECLINE_ARENA_READ_FAILED, f"{type(exc).__name__}: {exc}"
            ) from exc

        # Captured BEFORE the allocation, because allocate_state_item is
        # idempotent: a slot the sequence already owned must survive a failed
        # restore, and only a slot THIS restore took may be handed back.
        try:
            had_slot = self._state_writer.has_slot(int(sequence_id))
        except Exception as exc:  # noqa: BLE001 - fail-safe by contract
            raise _RestoreDeclined(
                DECLINE_SLOT_WRITE_FAILED, f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            self._state_writer.allocate_slot(int(sequence_id))
        except KDAStateSlotsExhausted as exc:
            raise _RestoreDeclined(DECLINE_SLOTS_EXHAUSTED, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - fail-safe by contract
            self._logger.exception(
                "[%s] rank=%s seq=%s binding a KDA slot failed",
                PREFIX,
                self._rank,
                int(sequence_id),
            )
            raise _RestoreDeclined(
                DECLINE_SLOT_WRITE_FAILED, f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            slot, written = self._state_writer.write_slot(
                int(sequence_id), chunks
            )
        except Exception as exc:  # noqa: BLE001 - fail-safe by contract
            self._logger.exception(
                "[%s] rank=%s seq=%s writing the snapshot into its KDA slot "
                "failed",
                PREFIX,
                self._rank,
                int(sequence_id),
            )
            if not had_slot:
                # The slot was taken by THIS restore and holds a half-written
                # state; hand it back so the pool does not leak it and no
                # later sequence inherits the partial bytes.
                try:
                    self._state_writer.release_slot(int(sequence_id))
                except Exception:  # noqa: BLE001 - cleanup is best effort
                    self._logger.exception(
                        "[%s] rank=%s seq=%s could not release the KDA slot a "
                        "failed restore had taken",
                        PREFIX,
                        self._rank,
                        int(sequence_id),
                    )
            raise _RestoreDeclined(
                DECLINE_SLOT_WRITE_FAILED, f"{type(exc).__name__}: {exc}"
            ) from exc
        return int(slot), int(written), bool(had_slot)

    # -- the mirrored (follower) restore ------------------------------------

    def restore_mirrored_shard(
        self,
        *,
        sequence_id: int,
        arena,
        layout,
        session_key: str,
        boundary_tokens: int,
        owning_group_id: int,
        serving_group_id: int,
        compute_cached_tokens: int,
    ) -> TrajectoryRestoreOutcome | None:
        """Restore this rank's shard from the LEADER's mirrored decision.

        Why a follower has no record of its own: a frontier record's lifetime
        is allocator state -- the arena's pages are retained and released by
        exactly one rank. If every rank of the group kept its own record, a
        leader-side eviction would free the arena while G-1 stale records still
        named it, and those ranks would restore freed pages. So the group keeps
        ONE registry, on the leader, and the followers take the leader's
        decision for the pass through the host-KV mirror. Hit/miss and
        commit/skip therefore cannot differ across the group: there is only one
        decision to begin with.

        The leader's own attachment pins the arena for the whole pass, which is
        what makes it safe for this rank to read pages it holds no reference
        to.

        ``None`` means this rank recomputes, with the reason named -- the same
        reasons and the same counters as the attachment path.
        """

        self._lookups += 1
        self._hits += 1
        shards = int(getattr(arena, "shard_count", 1))
        if shards != self._attention_group_size:
            self._decline_mirrored(
                sequence_id=sequence_id,
                session_key=session_key,
                boundary_tokens=boundary_tokens,
                reason=DECLINE_HEAD_SHARDED_GROUP,
                detail=(
                    f"the mirrored arena holds {shards} head shard(s) but this "
                    f"rank serves a {self._attention_group_size}-rank group"
                ),
            )
            return None
        if int(serving_group_id) != int(owning_group_id):
            self._decline_mirrored(
                sequence_id=sequence_id,
                session_key=session_key,
                boundary_tokens=boundary_tokens,
                reason=DECLINE_GROUP_MISMATCH,
                detail=(
                    f"serving_group={int(serving_group_id)} but the snapshot "
                    f"is head-sharded for group {int(owning_group_id)}"
                ),
            )
            return None
        if int(compute_cached_tokens) != int(boundary_tokens):
            self._decline_mirrored(
                sequence_id=sequence_id,
                session_key=session_key,
                boundary_tokens=boundary_tokens,
                reason=DECLINE_SUFFIX_OFFSET,
                detail=(
                    f"suffix prefill starts at token "
                    f"{int(compute_cached_tokens)} but the snapshot is the "
                    f"state after {int(boundary_tokens)} tokens"
                ),
            )
            return None
        try:
            slot, written, had_slot = self._read_window_and_bind_slot(
                arena=arena, layout=layout, sequence_id=int(sequence_id)
            )
        except _RestoreDeclined as declined:
            self._decline_mirrored(
                sequence_id=sequence_id,
                session_key=session_key,
                boundary_tokens=boundary_tokens,
                reason=declined.reason,
                detail=declined.detail,
            )
            return None
        self._restores += 1
        self._logger.info(
            "[%s] rank=%s seq=%s session=%s RESTORED shard=%s/%s boundary=%s "
            "slot=%s bytes=%s group=%s allocated_slot=%s; suffix prefill "
            "resumes at token %s",
            PREFIX,
            self._rank,
            int(sequence_id),
            session_key,
            self._shard_index,
            self._attention_group_size,
            int(boundary_tokens),
            slot,
            written,
            int(owning_group_id),
            not had_slot,
            int(compute_cached_tokens),
        )
        return TrajectoryRestoreOutcome(
            session_key=str(session_key),
            sequence_id=int(sequence_id),
            state_slot=int(slot),
            boundary_tokens=int(boundary_tokens),
            owning_group_id=int(owning_group_id),
            restored_bytes=int(written),
            allocated_slot=not had_slot,
        )

    def _decline_mirrored(
        self,
        *,
        sequence_id: int,
        session_key: str,
        boundary_tokens: int,
        reason: str,
        detail: str,
    ) -> None:
        """Name a mirrored restore's fallback. No attachment to consume."""

        if reason not in DECLINE_REASONS:
            raise TrajectoryAttachmentInvalid(
                f"unknown trajectory decline reason {reason!r}; known reasons "
                "are " + ", ".join(DECLINE_REASONS)
            )
        self._declines[reason] = self._declines.get(reason, 0) + 1
        self._logger.warning(
            "[%s] rank=%s seq=%s session=%s FALLBACK reason=%s boundary=%s "
            "%s; trajectory reuse degrades to full recompute",
            PREFIX,
            self._rank,
            int(sequence_id),
            session_key,
            reason,
            int(boundary_tokens),
            detail,
        )


__all__ = [
    "COMMIT_LOG_PREFIX",
    "DEBUG_FLAG",
    "DECLINE_ARENA_MISSING",
    "DECLINE_ARENA_READ_FAILED",
    "DECLINE_GROUP_MISMATCH",
    "DECLINE_HEAD_SHARDED_GROUP",
    "DECLINE_REASONS",
    "DECLINE_SLOTS_EXHAUSTED",
    "DECLINE_SLOT_WRITE_FAILED",
    "DECLINE_SUFFIX_OFFSET",
    "HostKVSnapshotArenaReader",
    "KDAStateSlotsExhausted",
    "KDAStateSnapshotWriter",
    "PREFIX",
    "TrajectoryAttachment",
    "TrajectoryAttachmentInvalid",
    "TrajectoryRestoreError",
    "TrajectoryRestoreOutcome",
    "TrajectoryRestorer",
    "trajectory_reuse_enabled",
]
