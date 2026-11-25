"""State owner for the worker's node-shared trajectory and frontier state.

``TrajectoryStore`` is a non-leaf worker component by design.  It owns the
trajectory book, frontier lifecycle objects and their small pieces of lazy
state; orchestration, collectives, allocator transactions and identity
resolution deliberately remain on ``BatchGenWorker``.  In particular this
module never resolves worker-owned identities.

The per-sequence recurrent (KDA) state is model-owned, so this module never
imports a model package: the worker injects ``recurrent_state_owner``, a
callable returning the owner the served model declares through its parallel
strategy manager, or ``None`` for a model that keeps no such state.  The owner
exposes ``state_manager`` and ``slot_manager`` (both ``None`` until the pools
exist) and ``free_sequences(ids)``; every call site resolves it lazily, so a
model without the state is a no-op or a raise exactly where it was.  The
trajectory modules are imported inside the builders, matching the old lazy
timing.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional


class TrajectoryStore:
    """Owned trajectory state; deliberately not an invariant-5 leaf.

    The release contract is intentionally split across owners: attachment
    release, arena rollback, pool-segment release, prefix-state pop and
    coordinator release stay separate and are ordered by the worker.
    """

    def __init__(self, recurrent_state_owner: Optional[Callable[[], Any]] = None) -> None:
        self._recurrent_state_owner = recurrent_state_owner
        # The spelling mirrors the old worker fields, minus the leading
        # underscore.  ``None`` is the eager sentinel required by S1.
        self.trajectory_book = None
        self.trajectory_memfd = None
        self.trajectory_pool_capacity_signature = None
        self.trajectory_frontier_committer = None
        self.trajectory_restorer = None
        self.trajectory_attachments = {}
        self.trajectory_arena_payload = {}
        self.trajectory_mirrored_restores = {}
        self.trajectory_session_groups = None
        # Kept as a field rather than a module import so flag-off construction
        # does not load the optional frontier implementation.
        try:
            from batchgen.prefix_reuse.trajectory_commit import PREFIX
        except ImportError:
            PREFIX = "TRAJECTORY_FRONTIER"
        self.FRONTIER_PREFIX = PREFIX

    # ------------------------------ token-book accessors -----------------

    def slot_for(self, uuid: str) -> int:
        if self.trajectory_book is None:
            raise RuntimeError("unified trajectory pool is not initialized")
        return self.trajectory_book.slot_for(uuid)

    def has_sequence(self, uuid: str) -> bool:
        return bool(self.trajectory_book is not None and self.trajectory_book.has_sequence(uuid))

    def metadata(self, slot: int):
        if self.trajectory_book is None:
            raise RuntimeError("unified trajectory pool is not initialized")
        return self.trajectory_book.metadata(slot)

    def append_token(self, slot: int, token_id: int) -> int:
        if self.trajectory_book is None:
            raise RuntimeError("unified trajectory pool is not initialized")
        return self.trajectory_book.append_token(slot, int(token_id))

    def append_tokens(self, slots, token_ids) -> None:
        if self.trajectory_book is None:
            raise RuntimeError("unified trajectory pool is not initialized")
        self.trajectory_book.append_tokens(slots, token_ids)

    def generated_tokens(self, slot):
        if self.trajectory_book is None:
            raise RuntimeError("unified trajectory pool is not initialized")
        return self.trajectory_book.generated_tokens(slot)

    def copy_span_to(self, slot, out, *, start: int, length: int) -> None:
        if self.trajectory_book is None:
            raise RuntimeError("unified trajectory pool is not initialized")
        self.trajectory_book.copy_span_to(slot, out, start=int(start), length=int(length))

    def release(self, slot: int) -> None:
        if self.trajectory_book is not None:
            self.trajectory_book.release(slot)

    def synchronize_metadata(self, slot: int, **fields: int) -> None:
        if self.trajectory_book is not None:
            self.trajectory_book.synchronize_metadata(slot, **fields)

    def reservation_total_capacity(self, model_context_length: int) -> int:
        return self.trajectory_book.reservation_total_capacity(model_context_length)

    def reservation_free_count(self, model_context_length: int) -> int:
        return self.trajectory_book.reservation_free_count(model_context_length)

    def trajectory_frontier_eviction_hook(self):
        """The frontier eviction seam, or None when no frontier is committed."""
        committer = self.trajectory_frontier_committer
        if committer is None:
            return None
        return committer.on_prefix_pages_released

    # ------------------------------- optional KDA state -------------------

    def _state_owner(self):
        """The model's recurrent-state owner, or ``None`` when it keeps none."""
        if self._recurrent_state_owner is None:
            return None
        return self._recurrent_state_owner()

    def _owner_state_manager(self):
        """The owner's state manager as it is now, pools built or not."""
        owner = self._state_owner()
        return None if owner is None else owner.state_manager

    def kda_state_manager(self):
        """Return the optional KDA state manager, or ``None`` when absent."""
        owner = self._state_owner()
        if owner is None or owner.slot_manager is None:
            return None
        return owner.state_manager

    def free_kda_sequences(self, global_sequence_ids: Iterable[int]) -> None:
        """Free KDA slots; a no-op for a model that keeps no such state."""
        ids = list(global_sequence_ids)
        if not ids:
            return
        owner = self._state_owner()
        if owner is not None and owner.slot_manager is not None:
            owner.free_sequences(ids)

    def kda_stats(self) -> tuple[int, int]:
        """Return ``(configured_slots, live_slots)`` for memory diagnostics."""
        kda_slots = kda_live = -1
        try:
            manager = self._owner_state_manager()
            if manager is not None:
                # The pools are deliberately kept across a rebuild, so their
                # slot count must stay FLAT; a growing one would mean the
                # reuse decision stopped matching.
                kda_slots = int(manager.config.num_state_items)
                kda_live = len(manager.sequence_state_items)
        except Exception:
            # Observability must not take down the switch it observes: a
            # half-initialized manager raises AttributeError/TypeError here,
            # and the -1 sentinel is a readable "unknown" in the line. A value
            # read before the failure is kept.
            pass
        return kda_slots, kda_live

    def fold_session_groups(self, gathered, *, rank: int = 0, logger=logging) -> None:
        """Fold the rank-identical frontier summaries, preserving lazy timing."""
        summaries = [
            entry.get("frontier_sessions")
            for entry in gathered
            if isinstance(entry, dict) and entry.get("frontier_sessions")
        ]
        if not summaries and self.trajectory_session_groups is None:
            return
        from batchgen.prefix_reuse.trajectory_frontier import TrajectorySessionGroupIndex

        index = self.trajectory_session_groups
        if index is None:
            index = TrajectorySessionGroupIndex()
            self.trajectory_session_groups = index
        changed = index.fold(summaries)
        if changed:
            logger.info(
                "[TRAJECTORY_FRONTIER] Rank %s folded %s session-group "
                "change(s) from %s leader summary/summaries: %s session(s) "
                "now pinnable",
                rank,
                changed,
                len(summaries),
                len(index),
            )

    # ------------------------------- frontier builders -------------------

    def build_committer(
        self,
        *,
        rank: int,
        runtime,
        group_geometry: tuple,
        unit_count: int,
        host_views_provider: Callable[[], Any],
        next_pseudo_sid: Callable[[], int],
    ):
        """Lazily build the frontier committer, including its optional imports.

        ``group_geometry`` is the live attention group (size, rank-in-group,
        leader) and ``unit_count`` is the RECORD's head-unit count -- a
        property of the model (its largest implemented attention group size),
        owned by ``parallel_declaration.max_supported_attention_group_size``
        and passed in by the worker, never re-derived here.
        """
        if self.trajectory_frontier_committer is not None:
            return self.trajectory_frontier_committer
        from batchgen.prefix_reuse.trajectory_commit import (
            HostKVSnapshotArenaAllocator,
            KDAStateSnapshotReader,
            TrajectoryFrontierCommitter,
            TrajectoryStateUnavailable,
            profile_from_kda_state_config,
        )

        manager = self._owner_state_manager()
        if manager is None:
            raise TrajectoryStateUnavailable(
                "prefix_trajectory_commit needs a live KDA state pool; this "
                "deployment has none"
            )
        page_tokens = int(runtime.group_specs[0].raw_page_tokens)
        page_bytes = int(runtime.host_page_bytes_all_layers)
        group_size, group_rank, is_leader = group_geometry
        self.trajectory_frontier_committer = TrajectoryFrontierCommitter(
            rank=rank,
            page_tokens=page_tokens,
            host_page_bytes_all_layers=page_bytes,
            group_size=group_size,
            group_rank=group_rank,
            unit_count=int(unit_count),
            owns_registry=is_leader,
            arena_allocator=HostKVSnapshotArenaAllocator(
                host_views_provider=host_views_provider,
                page_tokens=page_tokens,
                host_page_bytes_all_layers=page_bytes,
                next_pseudo_sid=next_pseudo_sid,
            ),
            snapshot_reader=KDAStateSnapshotReader(
                state_manager_provider=self._owner_state_manager,
                profile=profile_from_kda_state_config(manager.config),
            ),
        )
        return self.trajectory_frontier_committer

    def build_restorer(
        self,
        *,
        rank: int,
        host_views_provider: Callable[[], Any],
        next_pseudo_sid: Callable[[], int],
    ):
        """Lazily build the restorer after a committer exists."""
        if self.trajectory_frontier_committer is None:
            return None
        if self.trajectory_restorer is not None:
            return self.trajectory_restorer
        from batchgen.prefix_reuse.trajectory_restore import (
            HostKVSnapshotArenaReader,
            KDAStateSnapshotWriter,
            TrajectoryRestorer,
        )

        committer = self.trajectory_frontier_committer
        self.trajectory_restorer = TrajectoryRestorer(
            rank=rank,
            committer=committer,
            arena_reader=HostKVSnapshotArenaReader(
                host_views_provider=host_views_provider,
                next_pseudo_sid=next_pseudo_sid,
            ),
            state_writer=KDAStateSnapshotWriter(
                state_manager_provider=self._owner_state_manager,
                profile=committer.rank_profile,
            ),
            attention_group_size=committer.group_size,
            group_rank=committer.group_rank,
        )
        return self.trajectory_restorer

    def arena_from_payload(self, entry: dict):
        from batchgen.prefix_reuse.trajectory_commit import SnapshotArena

        return SnapshotArena(
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

    def restore_fields(self) -> dict:
        fields = {}
        if self.trajectory_restorer is not None:
            fields.update(self.trajectory_restorer.counters())
        # Placement telemetry is replicated even on ranks without a restorer.
        index = self.trajectory_session_groups
        if index is not None:
            fields["trajectory_group_honored_on_owner"] = int(index.honored_on_owner)
            fields["trajectory_group_placed_offgroup"] = int(index.placed_offgroup)
        return fields

    def commit_requested(self, debug_payload) -> bool:
        """Resolve the batch debug flag without importing it in the worker."""
        from batchgen.prefix_reuse.trajectory_commit import trajectory_commit_enabled

        return trajectory_commit_enabled(debug_payload)

    @staticmethod
    def group_geometry(*, decode_attn_tp_size: int, rank: int) -> tuple:
        G = int(decode_attn_tp_size)
        group_rank = int(rank % G) if G > 1 else 0
        return G, group_rank, group_rank == 0

    @staticmethod
    def commit_tokens(runtime, seq) -> int:
        if runtime is None:
            return 0
        boundary = int(runtime.publish_boundary_tokens)
        prompt_length = int(getattr(seq, "prompt_length", 0) or 0)
        commit_tokens = (prompt_length // boundary) * boundary
        if commit_tokens == 0 or commit_tokens != prompt_length:
            return 0
        return commit_tokens

    def release_attachments(self, global_sequence_ids: Iterable[int], *, rank: int = 0) -> None:
        """Release only trajectory attachments; coordinator ordering stays outside."""
        if not self.trajectory_attachments:
            return
        restorer = self.trajectory_restorer
        for sequence_id in global_sequence_ids:
            attachment = self.trajectory_attachments.pop(int(sequence_id), None)
            if attachment is None or restorer is None:
                continue
            try:
                restorer.release(attachment)
            except Exception:
                logging.exception(
                    "Rank %s: releasing the trajectory attachment of sequence "
                    "%s failed; its frontier stays pinned",
                    rank,
                    int(sequence_id),
                )

    def rollback_reservations(self, global_sequence_ids: Iterable[int], *, rank: int = 0) -> None:
        """Rollback only trajectory arenas; pool/prefix release stays separate."""
        committer = self.trajectory_frontier_committer
        if committer is None:
            return
        for sequence_id in global_sequence_ids:
            sequence_id = int(sequence_id)
            self.trajectory_arena_payload.pop(sequence_id, None)
            self.trajectory_mirrored_restores.pop(sequence_id, None)
            try:
                committer.rollback_reservation(sequence_id=sequence_id)
            except Exception:
                logging.exception(
                    "[TRAJECTORY_FRONTIER] Rank %s: could not give back the "
                    "reserved snapshot arena of sequence %s",
                    rank,
                    sequence_id,
                )

    def restore_payload(
        self, *, group_size: int, is_leader: bool, planned_boundary_for
    ) -> dict:
        if group_size <= 1 or not is_leader or not self.trajectory_attachments:
            return {}
        committer = self.trajectory_frontier_committer
        if committer is None:
            return {}
        payload = {}
        for global_id, attachment in sorted(self.trajectory_attachments.items()):
            # Only attachments the wave plan actually restores ride the mirror:
            # a planned boundary that disagrees with the attachment is stale.
            if planned_boundary_for(int(global_id)) != int(attachment.boundary_tokens):
                continue
            arena = committer.arena_for(attachment.storage_handle)
            if arena is None:
                continue
            payload[int(global_id)] = {
                "session_key": str(attachment.session_key),
                "boundary_tokens": int(attachment.boundary_tokens),
                "lcp_tokens": int(attachment.lcp_tokens),
                "owning_group_id": int(attachment.owning_group_id),
                "unit_count": int(arena.unit_count),
                "pseudo_sid": int(arena.pseudo_sid),
                "pages_per_group": int(arena.pages_per_group),
                "capacity_bytes": int(arena.capacity_bytes),
                "pages": {
                    int(group_id): [int(page) for page in pages]
                    for group_id, pages in arena.page_ids_by_group.items()
                },
            }
        return payload

    def session_summary(self, *, group_size: int, is_leader: bool, rank: int) -> dict:
        if group_size <= 1 or not is_leader:
            return {}
        committer = self.trajectory_frontier_committer
        if committer is None or not committer.owns_registry:
            return {}
        sessions = committer.registry.session_group_summary()
        return {
            "group_id": int(rank // group_size),
            "sessions": {
                str(session_key): [
                    int(owning_group_id),
                    int(boundary_tokens),
                    str(index_key),
                ]
                for session_key, (
                    owning_group_id,
                    boundary_tokens,
                    index_key,
                ) in sessions.items()
            },
        }

    def adopt_mirrored_arenas(self, arenas: dict, *, committer) -> None:
        if not arenas:
            return
        for global_id, entry in arenas.items():
            committer.adopt_arena(
                sequence_id=int(global_id),
                arena=self.arena_from_payload(entry),
            )
