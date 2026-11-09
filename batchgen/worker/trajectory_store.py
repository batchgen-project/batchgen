"""State owner for the worker's node-shared trajectory and frontier state.

``TrajectoryStore`` is a non-leaf worker component by design.  It owns the
trajectory book, frontier lifecycle objects and their small pieces of lazy
state; orchestration, collectives, allocator transactions and identity
resolution deliberately remain on ``BatchGenWorker``.  In particular this
module never resolves worker-owned identities.

The KDA state wrapper is model-owned, so this module never imports a model
package: the worker injects ``kda_wrapper_loader`` and every call site resolves
it lazily inside the same ``ImportError`` handling as before (an absent model
package is still a no-op or a raise exactly where it was).  The trajectory
modules are imported inside the builders, matching the old lazy timing.
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

    def __init__(self, kda_wrapper_loader: Optional[Callable[[], Any]] = None) -> None:
        self._kda_wrapper_loader = kda_wrapper_loader
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

    # ------------------------------- optional KDA imports -----------------

    def _load_kda_wrapper(self):
        """Resolve the injected KDA wrapper; ``ImportError`` when absent."""
        if self._kda_wrapper_loader is None:
            raise ImportError("no KDA wrapper loader was injected")
        return self._kda_wrapper_loader()

    def kda_state_manager(self):
        """Return the optional KDA state manager, or ``None`` when absent."""
        try:
            kda_wrapper = self._load_kda_wrapper()
        except ImportError:
            return None
        if kda_wrapper.slot_manager is None:
            return None
        return kda_wrapper.state_manager

    def free_kda_sequences(self, global_sequence_ids: Iterable[int]) -> None:
        """Free KDA slots while preserving the old ImportError no-op."""
        ids = list(global_sequence_ids)
        if not ids:
            return
        try:
            kda_wrapper = self._load_kda_wrapper()
            if kda_wrapper.slot_manager is not None:
                kda_wrapper.free_sequences(ids)
        except ImportError:
            pass

    def kda_stats(self) -> tuple[int, int]:
        """Return ``(configured_slots, live_slots)`` for memory diagnostics."""
        try:
            kda_wrapper = self._load_kda_wrapper()
            manager = kda_wrapper.state_manager
            if manager is None:
                return -1, -1
            return int(manager.config.num_state_items), len(manager.sequence_state_items)
        except Exception:
            return -1, -1

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
        host_views_provider: Callable[[], Any],
        next_pseudo_sid: Callable[[], int],
    ):
        """Lazily build the frontier committer, including its optional imports."""
        if self.trajectory_frontier_committer is not None:
            return self.trajectory_frontier_committer
        kda_wrapper = self._load_kda_wrapper()
        from batchgen.prefix_reuse.trajectory_commit import (
            HostKVSnapshotArenaAllocator,
            KDAStateSnapshotReader,
            TrajectoryFrontierCommitter,
            TrajectoryStateUnavailable,
            profile_from_kda_state_config,
        )

        manager = kda_wrapper.state_manager
        if manager is None:
            raise TrajectoryStateUnavailable(
                "prefix_trajectory_commit needs a live KDA state pool; this "
                "deployment has none"
            )
        page_tokens = int(runtime.group_specs[0].raw_page_tokens)
        page_bytes = int(runtime.host_page_bytes_all_layers)
        shard_count, shard_index, is_leader = group_geometry
        self.trajectory_frontier_committer = TrajectoryFrontierCommitter(
            rank=rank,
            page_tokens=page_tokens,
            host_page_bytes_all_layers=page_bytes,
            shard_count=shard_count,
            shard_index=shard_index,
            owns_registry=is_leader,
            arena_allocator=HostKVSnapshotArenaAllocator(
                host_views_provider=host_views_provider,
                page_tokens=page_tokens,
                host_page_bytes_all_layers=page_bytes,
                next_pseudo_sid=next_pseudo_sid,
            ),
            snapshot_reader=KDAStateSnapshotReader(
                state_manager_provider=lambda: kda_wrapper.state_manager,
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
        kda_wrapper = self._load_kda_wrapper()
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
                state_manager_provider=lambda: kda_wrapper.state_manager,
                profile=committer.layout.profile,
            ),
            attention_group_size=committer.shard_count,
            shard_index=committer.shard_index,
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
            shard_count=int(entry["shard_count"]),
        )

    def restore_fields(self) -> dict:
        fields = {}
        if self.trajectory_restorer is not None:
            fields.update(self.trajectory_restorer.counters())
        index = self.trajectory_session_groups
        if index is not None:
            fields["trajectory_group_pins"] = int(index.pins)
        return fields

    def commit_requested(self, debug_payload) -> bool:
        """Resolve the batch debug flag without importing it in the worker."""
        from batchgen.prefix_reuse.trajectory_commit import trajectory_commit_enabled

        return trajectory_commit_enabled(debug_payload)

    @staticmethod
    def group_geometry(*, decode_attn_tp_size: int, rank: int) -> tuple:
        G = int(decode_attn_tp_size)
        shard_index = int(rank % G) if G > 1 else 0
        return G, shard_index, shard_index == 0

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

    def restore_payload(self, *, shard_count: int, is_leader: bool) -> dict:
        if shard_count <= 1 or not is_leader or not self.trajectory_attachments:
            return {}
        committer = self.trajectory_frontier_committer
        if committer is None:
            return {}
        payload = {}
        for global_id, attachment in self.trajectory_attachments.items():
            arena = committer.arena_for(attachment.storage_handle)
            if arena is None:
                continue
            payload[int(global_id)] = {
                "session_key": str(attachment.session_key),
                "boundary_tokens": int(attachment.boundary_tokens),
                "owning_group_id": int(attachment.owning_group_id),
                "shard_count": int(arena.shard_count),
                "pseudo_sid": int(arena.pseudo_sid),
                "pages_per_group": int(arena.pages_per_group),
                "capacity_bytes": int(arena.capacity_bytes),
                "pages": {
                    int(group_id): [int(page) for page in pages]
                    for group_id, pages in arena.page_ids_by_group.items()
                },
            }
        return payload

    def session_summary(self, *, shard_count: int, is_leader: bool, rank: int) -> dict:
        if shard_count <= 1 or not is_leader:
            return {}
        committer = self.trajectory_frontier_committer
        if committer is None or not committer.owns_registry:
            return {}
        sessions = committer.registry.session_group_summary()
        return {
            "group_id": int(rank // shard_count),
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
