"""Commit a trajectory frontier: KDA snapshot bytes into Host KV pages.

Slice 1b of the trajectory-scoped cross-wave reuse mechanism. Slice 1a owns the
record, the session key and the charge math (``trajectory_frontier``); this
module owns the *transaction*: where the snapshot bytes physically live, how
they get there, and how a record leaves when its prefix pages are evicted.

Storage contract (R1, adjudicated 2026-10-17)
---------------------------------------------
Snapshot bytes live in Host KV PAGES taken from the SAME allocator the prefix
cache uses, through the existing pool-segment mechanism: a negative pseudo
sequence id, a host-view page allocation, then ``retain_sequence_pages`` so the
pages leave the pseudo sequence's chain and become prefix-resident. Consequences
that are load-bearing, not incidental:

* ``num_free_pages`` drops at allocation, so the boundary planner cannot
  double-promise the bytes. No new budget field, no new invariant.
* ``SequenceEntry.host_owned_pages`` is a RELEASE PROMISE and is NEVER touched
  by a snapshot. The retention deduction at the commit boundary stays exactly as
  it was; nothing is ever credited back to it when a snapshot is freed, because
  nothing was ever promised.
* The pages are a byte ARENA. They carry no KV token semantics -- the pseudo-sid
  namespace is what marks them -- and the blob is laid out in one fixed order
  derived from ``KDASnapshotProfile``.

Everything here is flag-gated by ``batchgen_debug.prefix_trajectory_commit``.
With the flag off nothing in this module runs and no registry write happens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

from batchgen.prefix_reuse.trajectory_frontier import (
    FrontierCharge,
    FrontierReleaseOutcome,
    KDASnapshotDescriptor,
    KDASnapshotProfile,
    TrajectoryFrontierRecord,
    TrajectoryFrontierRegistry,
    dtype_element_bytes,
)


# The batch-level flag and the log prefix every line here carries.
DEBUG_FLAG = "prefix_trajectory_commit"
PREFIX = "TRAJECTORY_FRONTIER"

# The ONE recoverable Host-page failure: the allocator's aggregate free-page
# check (``AcquirePagesForSequences``) refuses the whole request with this
# message before taking a page. Any other failure is a bug and is re-raised.
_POOL_PRESSURE_MARKERS = ("Insufficient free pages",)

# The fixed blob order. ``KDAStateGPUManager`` keeps one recurrent pool and
# three short-conv pools (q, k, v); a snapshot is those four slot slices
# concatenated in this order, and a restore reads them back in it.
SNAPSHOT_CHUNK_NAMES = ("recurrent", "conv_q", "conv_k", "conv_v")


# ---------------------------------------------------------------------------
# Errors -- every refusal is named; nothing degrades silently.
# ---------------------------------------------------------------------------


class TrajectoryCommitError(RuntimeError):
    """Base class for every trajectory-commit refusal."""


class TrajectoryCommitFlagInvalid(ValueError):
    """``prefix_trajectory_commit`` was set to something other than on/off."""


class SnapshotArenaUnavailable(TrajectoryCommitError):
    """Host KV pages for the snapshot arena could not be allocated.

    The ONE recoverable outcome: the commit step is skipped for that sequence
    and trajectory reuse degrades to recompute. It never blocks the retention
    transaction, and it is always logged by name -- a silent skip would read
    exactly like a clean commit.
    """


class SnapshotArenaCorrupt(TrajectoryCommitError):
    """An arena was asked for in a state the mechanism does not allow."""


class TrajectorySnapshotShapeMismatch(TrajectoryCommitError):
    """The copied slot bytes do not match the profile's geometry."""


class TrajectoryStateUnavailable(TrajectoryCommitError):
    """The sequence has no KDA state slot to snapshot."""


# ---------------------------------------------------------------------------
# The batch-level flag
# ---------------------------------------------------------------------------


def parse_trajectory_commit_flag(value) -> bool:
    """``True`` only for ``"on"``; absent means off; anything else is refused.

    Deliberately stricter than the other prefix-reuse debug flags: this one
    gates a transaction that allocates Host pages, so a mistyped value must not
    read as "off". A silent no-op looks exactly like a clean run.
    """

    if value is None:
        return False
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "on":
            return True
        if text == "off":
            return False
    raise TrajectoryCommitFlagInvalid(
        f"{DEBUG_FLAG} must be the string \"on\" or \"off\", got {value!r}"
    )


def trajectory_commit_enabled(debug: Mapping | None) -> bool:
    """THE guard. One place decides whether Slice 1b runs for a batch.

    With it ``False`` nothing in this module is constructed or called, so the
    commit boundary behaves exactly as it did before Slice 1b.
    """

    if not debug:
        return False
    return parse_trajectory_commit_flag(debug.get(DEBUG_FLAG))


def profile_from_kda_state_config(config: object) -> KDASnapshotProfile:
    """Snapshot geometry read off the LIVE ``KDAStateGPUConfig``.

    The slice-1a builder derives the profile from the model config plus the
    attention group size; here the pool already exists, so its own config is
    the more faithful source: the heads and ``conv_dim`` it reports ARE this
    rank's shard, and the two dtypes are the ones the tensors were allocated
    with. A geometry that disagrees with the real tensors would mis-size the
    arena, so this never re-derives anything it can read.
    """

    try:
        num_kda_layers = int(config.num_kda_layers)
        num_heads = int(config.num_heads)
        head_dim = int(config.head_dim)
        conv_dim = int(config.resolved_conv_dim())
        conv_width = int(config.conv_width)
        recurrent_dtype = config.recurrent_dtype
        conv_dtype = config.conv_dtype
    except AttributeError as exc:
        raise TrajectoryStateUnavailable(
            "the KDA state pool config does not expose the geometry a "
            f"trajectory snapshot needs: {exc}"
        ) from exc
    return KDASnapshotProfile(
        num_kda_layers=num_kda_layers,
        num_heads_per_rank=num_heads,
        head_dim=head_dim,
        conv_dim_per_rank=conv_dim,
        conv_width=conv_width,
        recurrent_element_bytes=dtype_element_bytes(str(recurrent_dtype)),
        conv_element_bytes=dtype_element_bytes(str(conv_dtype)),
    )


# ---------------------------------------------------------------------------
# Arena geometry -- pure math, derived from the profile and the page size
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnapshotLayout:
    """Byte layout of ONE snapshot blob inside its arena.

    Offsets come from ``KDASnapshotProfile`` alone, so changing the model or the
    attention group size moves them automatically. The three conv pools are
    equal-sized by construction (same per-layer view shape), which is asserted
    rather than assumed.
    """

    profile: KDASnapshotProfile

    @property
    def chunk_bytes(self) -> tuple[int, ...]:
        conv_total = int(self.profile.conv_bytes)
        pools = int(self.profile.num_conv_pools)
        if pools <= 0 or conv_total % pools:
            raise TrajectorySnapshotShapeMismatch(
                f"{conv_total} conv bytes do not divide into {pools} equal "
                "conv pools; the snapshot blob order would be ambiguous"
            )
        per_pool = conv_total // pools
        return (int(self.profile.recurrent_bytes),) + (per_pool,) * pools

    @property
    def chunk_offsets(self) -> tuple[int, ...]:
        offsets: list[int] = []
        cursor = 0
        for size in self.chunk_bytes:
            offsets.append(cursor)
            cursor += size
        return tuple(offsets)

    @property
    def total_bytes(self) -> int:
        return int(self.profile.total_bytes)

    def arena_page_count(
        self, *, host_page_bytes_all_layers: int, prefix_page_count: int
    ) -> int:
        """Host pages this blob needs, in the #454 charge unit.

        Routed through ``FrontierCharge`` rather than recomputed: the charge is
        the one owner of "snapshot bytes -> whole Host pages", and R1 repurposes
        that page figure as the number of pages to ALLOCATE for the snapshot
        (it is no longer a ``host_owned_pages`` counter delta).
        """

        charge = FrontierCharge(
            page_count=int(prefix_page_count),
            host_page_bytes_all_layers=int(host_page_bytes_all_layers),
            snapshot_bytes=self.total_bytes,
        )
        return int(charge.snapshot_host_pages)


@dataclass(frozen=True)
class SnapshotArena:
    """Host KV pages holding one snapshot blob.

    ``pseudo_sid`` is the negative Host sequence id the pages were allocated
    under. After ``retain``, the pages are prefix-resident and the pseudo
    sequence is unregistered, so the arena is identified by its page ids alone
    and is released with ``release_resident_pages``.
    """

    storage_handle: int
    pseudo_sid: int
    page_ids_by_group: Mapping[int, tuple[int, ...]]
    pages_per_group: int
    capacity_bytes: int
    payload_bytes: int

    def __post_init__(self) -> None:
        if int(self.pseudo_sid) >= 0:
            raise SnapshotArenaCorrupt(
                "a snapshot arena must live under a NEGATIVE pseudo sequence "
                f"id so it cannot collide with a real sequence, got "
                f"{self.pseudo_sid}"
            )
        if not self.page_ids_by_group:
            raise SnapshotArenaCorrupt("a snapshot arena needs pages")
        if int(self.payload_bytes) > int(self.capacity_bytes):
            raise SnapshotArenaCorrupt(
                f"snapshot payload {self.payload_bytes} B exceeds the arena's "
                f"{self.capacity_bytes} B of Host pages"
            )


# ---------------------------------------------------------------------------
# The real Host-KV arena allocator (the allocator boundary tests monkeypatch)
# ---------------------------------------------------------------------------


class HostKVSnapshotArenaAllocator:
    """Allocate, fill and release snapshot arenas in Host KV pages.

    The pool-segment lane, reused verbatim: register a negative pseudo sid in
    every KV group, allocate its pages, write the blob through the worker view's
    CPU path, retain the pages so they survive the pseudo sequence, then
    unregister it. An allocation that fails part way is rolled back here, so a
    caller never sees a half-built arena.
    """

    def __init__(
        self,
        *,
        host_views_provider: Callable[[], Mapping[int, object]],
        page_tokens: int,
        host_page_bytes_all_layers: int,
        next_pseudo_sid: Callable[[], int],
        torch_module=None,
    ) -> None:
        if int(page_tokens) <= 0:
            raise SnapshotArenaCorrupt(
                f"page_tokens must be positive, got {page_tokens}"
            )
        if int(host_page_bytes_all_layers) <= 0:
            raise SnapshotArenaCorrupt(
                "host_page_bytes_all_layers must be positive, got "
                f"{host_page_bytes_all_layers}"
            )
        self._host_views_provider = host_views_provider
        self._page_tokens = int(page_tokens)
        self._host_page_bytes_all_layers = int(host_page_bytes_all_layers)
        self._next_pseudo_sid = next_pseudo_sid
        self._torch = torch_module
        # (group_id, pages) -> (shape, dtype) of a staging buffer for that many
        # Host pages. Learned once from the worker view so the C++ page layout
        # stays the single owner of the geometry, then reused: every snapshot
        # of a given profile allocates the same page count.
        self._staging_spec: dict[tuple[int, int], tuple[tuple, object]] = {}

    def _torch_module(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def store(self, *, page_count: int, chunks: Sequence[object]) -> SnapshotArena:
        """Allocate ``page_count`` Host pages per group and write the blob."""

        pages = int(page_count)
        if pages <= 0:
            raise SnapshotArenaCorrupt(
                f"a snapshot arena needs at least one Host page, got {pages}"
            )
        views = dict(self._host_views_provider())
        if not views:
            raise SnapshotArenaCorrupt(
                "no Host KV worker views to allocate a snapshot arena in"
            )
        pseudo_sid = int(self._next_pseudo_sid())
        tokens = pages * self._page_tokens
        registered: list[int] = []
        page_ids_by_group: dict[int, tuple[int, ...]] = {}
        try:
            for group_id, view in sorted(views.items()):
                view.register_sequences([pseudo_sid])
                registered.append(int(group_id))
                view.allocate_pages_for_sequences([(pseudo_sid, tokens)])
                table = view.build_page_table([pseudo_sid])[0]
                page_ids = tuple(int(page) for page in table)
                if len(page_ids) != pages:
                    raise SnapshotArenaCorrupt(
                        f"snapshot arena got {len(page_ids)} Host pages in "
                        f"group {group_id}, planned {pages}"
                    )
                page_ids_by_group[int(group_id)] = page_ids
        except SnapshotArenaCorrupt:
            self._rollback(views, registered, pseudo_sid)
            raise
        except Exception as exc:
            self._rollback(views, registered, pseudo_sid)
            if not any(
                marker in str(exc) for marker in _POOL_PRESSURE_MARKERS
            ):
                # Only page exhaustion is recoverable. Anything else is a bug
                # and must NOT be downgraded into the fail-safe skip path.
                raise
            # The Host allocator checks the aggregate free count before taking
            # a page, so a refused group allocated nothing.
            raise SnapshotArenaUnavailable(
                f"Host KV pool could not supply {pages} page(s) per group for "
                f"a trajectory snapshot arena: {exc}"
            ) from exc

        capacity = pages * self._host_page_bytes_all_layers
        payload = sum(int(chunk.numel()) for chunk in chunks)
        if payload > capacity:
            self._rollback(views, registered, pseudo_sid)
            raise TrajectorySnapshotShapeMismatch(
                f"snapshot payload {payload} B does not fit the arena's "
                f"{capacity} B ({pages} page(s) x "
                f"{self._host_page_bytes_all_layers} B)"
            )
        try:
            self._write_blob(views, pseudo_sid, chunks)
            for group_id, view in sorted(views.items()):
                retained = [
                    int(page)
                    for page in view.retain_sequence_pages(
                        pseudo_sid, list(page_ids_by_group[int(group_id)])
                    )
                ]
                if retained != list(page_ids_by_group[int(group_id)]):
                    raise SnapshotArenaCorrupt(
                        "Host KV retained different pages for the snapshot "
                        f"arena of group {group_id}"
                    )
        except Exception:
            # Retain may have succeeded for earlier groups; release what is
            # resident and let the chain release handle the rest.
            self._release_pages(views, page_ids_by_group, best_effort=True)
            self._rollback(views, registered, pseudo_sid)
            raise
        # The pages are prefix-resident now; the pseudo sequence has an empty
        # chain and its sequence-table slot can go back.
        for _group_id, view in sorted(views.items()):
            view.unregister_sequences([pseudo_sid])
        return SnapshotArena(
            # The caller stamps the record-facing handle; the arena identity
            # inside the allocator is its pseudo sid.
            storage_handle=0,
            pseudo_sid=pseudo_sid,
            page_ids_by_group=page_ids_by_group,
            pages_per_group=pages,
            capacity_bytes=capacity,
            payload_bytes=payload,
        )

    def release(self, arena: SnapshotArena) -> dict[int, int]:
        """Free an arena's Host pages. Only the arena's own pages, ever."""

        views = dict(self._host_views_provider())
        return self._release_pages(views, arena.page_ids_by_group)

    # -- internals ---------------------------------------------------------

    def _write_blob(
        self,
        views: Mapping[int, object],
        pseudo_sid: int,
        chunks: Sequence[object],
    ) -> None:
        """Copy the blob into the arena's pages, group by group.

        The staging tensor's shape and dtype come from the worker view itself
        (``read_sequence_kv_to_cpu`` on the freshly allocated pseudo sequence),
        so the C++ page layout stays the single owner of that geometry; the
        bytes it returns are discarded.

        ARENA LAYOUT (the restore contract): the arena is the concatenation, in
        ascending group order, of each group's whole page-table byte window.
        The blob is written into it from offset 0 in ``SNAPSHOT_CHUNK_NAMES``
        order and may straddle a group boundary; groups past the end of the
        blob keep their pages as reserved capacity and are not written. That
        matches the arena size, which is ``pages x
        host_page_bytes_all_layers`` -- a figure summed across groups.
        """

        torch = self._torch_module()
        flat_chunks = [chunk.reshape(-1).view(torch.uint8) for chunk in chunks]
        cursor = 0
        remaining = sum(int(chunk.numel()) for chunk in flat_chunks)
        for group_id, view in sorted(views.items()):
            staging = self._staging_buffer(view, int(group_id), pseudo_sid)
            window = staging.reshape(-1).view(torch.uint8)
            window.zero_()
            room = int(window.numel())
            written = 0
            while written < room and cursor < len(flat_chunks):
                chunk = flat_chunks[cursor]
                take = min(int(chunk.numel()), room - written)
                window[written : written + take] = chunk[:take]
                written += take
                if take == int(chunk.numel()):
                    cursor += 1
                else:
                    flat_chunks[cursor] = chunk[take:]
            remaining -= written
            view.write_sequence_kv_from_cpu(pseudo_sid, staging, None)
            if remaining <= 0:
                break
        if remaining > 0:
            raise TrajectorySnapshotShapeMismatch(
                f"{remaining} snapshot byte(s) did not fit the arena"
            )

    def _staging_buffer(self, view: object, group_id: int, pseudo_sid: int):
        """A CPU buffer shaped exactly like this group's arena pages.

        The first call per (group, page count) learns the shape and dtype from
        ``read_sequence_kv_to_cpu`` -- the C++ layout stays the one owner of
        that geometry -- and discards the bytes it read. Later calls allocate
        the buffer directly, so the arena is not read back on every commit.
        """

        torch = self._torch_module()
        pages = len(view.build_page_table([pseudo_sid])[0])
        key = (int(group_id), int(pages))
        spec = self._staging_spec.get(key)
        if spec is None:
            probe = view.read_sequence_kv_to_cpu(pseudo_sid)[0]
            self._staging_spec[key] = (tuple(probe.shape), probe.dtype)
            return probe
        shape, dtype = spec
        return torch.empty(shape, dtype=dtype)

    def _release_pages(
        self,
        views: Mapping[int, object],
        page_ids_by_group: Mapping[int, Sequence[int]],
        *,
        best_effort: bool = False,
    ) -> dict[int, int]:
        released: dict[int, int] = {}
        for group_id, page_ids in sorted(page_ids_by_group.items()):
            view = views.get(int(group_id))
            if view is None:
                raise SnapshotArenaCorrupt(
                    "missing Host KV worker view for snapshot arena group "
                    f"{group_id}"
                )
            ids = [int(page) for page in page_ids]
            if not ids:
                released[int(group_id)] = 0
                continue
            try:
                view.release_resident_pages(ids)
            except Exception:
                if not best_effort:
                    raise
                released[int(group_id)] = 0
                continue
            released[int(group_id)] = len(ids)
        return released

    def _rollback(
        self,
        views: Mapping[int, object],
        registered: Sequence[int],
        pseudo_sid: int,
    ) -> None:
        for group_id in sorted({int(value) for value in registered}):
            view = views.get(int(group_id))
            if view is None:
                continue
            try:
                view.release_sequence_pages([pseudo_sid])
                view.unregister_sequences([pseudo_sid])
            except Exception:  # noqa: BLE001 - rollback is best effort
                logging.warning(
                    "[%s] rank-local rollback of snapshot arena pseudo_sid=%s "
                    "group=%s failed",
                    PREFIX,
                    pseudo_sid,
                    group_id,
                )


# ---------------------------------------------------------------------------
# The D2H copy (the other boundary tests monkeypatch)
# ---------------------------------------------------------------------------


class KDAStateSnapshotReader:
    """Copy one sequence's KDA state slot off the device, in profile order.

    Ordering: the fla kernel writes a sequence's recurrent and conv state in
    place on the worker's CURRENT stream during the prefill forward, and this
    copy is issued on that same stream, so it is stream-ordered after every
    write of the pass. The destination is pageable host memory, so the copy is
    synchronizing and the bytes are complete before they are handed on. The
    commit boundary itself sits after ``retire_pending_prefill_offloads``, so
    the MLA pages the record points at are retired too: both halves of the
    record describe the same boundary.
    """

    def __init__(
        self,
        *,
        state_manager_provider: Callable[[], object],
        profile: KDASnapshotProfile,
        torch_module=None,
    ) -> None:
        self._state_manager_provider = state_manager_provider
        self._profile = profile
        self._torch = torch_module

    @property
    def profile(self) -> KDASnapshotProfile:
        return self._profile

    def _torch_module(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def read_slot(self, sequence_id: int) -> tuple[object, ...]:
        manager = self._state_manager_provider()
        if manager is None:
            raise TrajectoryStateUnavailable(
                "this model keeps no KDA state pool, so a trajectory snapshot "
                "has nothing to copy"
            )
        if not manager.has_sequence_state_item(int(sequence_id)):
            raise TrajectoryStateUnavailable(
                f"sequence {sequence_id} holds no KDA state slot at the commit "
                "boundary"
            )
        slot = int(manager.get_sequence_state_item(int(sequence_id)))
        recurrent = manager.get_recurrent_tensors()[:, slot]
        conv_q, conv_k, conv_v = (
            tensor[:, slot] for tensor in manager.get_conv_tensors()
        )
        return tuple(
            self._to_host_bytes(tensor)
            for tensor in (recurrent, conv_q, conv_k, conv_v)
        )

    def _to_host_bytes(self, tensor):
        torch = self._torch_module()
        # ``[:, slot]`` is strided over the layer dim: make it dense on the
        # device (current stream, after the prefill writes), then copy once.
        return (
            tensor.contiguous()
            .reshape(-1)
            .to("cpu")
            .view(torch.uint8)
        )


# ---------------------------------------------------------------------------
# The commit transaction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryCommitOutcome:
    """What one flag-on commit step did."""

    session_key: str
    record: TrajectoryFrontierRecord
    arena: SnapshotArena
    advanced: bool
    previous_release: FrontierReleaseOutcome | None


class TrajectoryFrontierCommitter:
    """Owns the frontier registry, the snapshot arenas and the reverse index.

    One instance per worker rank. The registry decides record lifetime; this
    class decides where the bytes live and which session a Host page belongs to,
    so an eviction of prefix pages can find and retire the record atomically.
    """

    def __init__(
        self,
        *,
        rank: int,
        page_tokens: int,
        host_page_bytes_all_layers: int,
        arena_allocator,
        snapshot_reader,
        logger=None,
    ) -> None:
        self._rank = int(rank)
        self._host_page_bytes_all_layers = int(host_page_bytes_all_layers)
        self._arena_allocator = arena_allocator
        self._snapshot_reader = snapshot_reader
        self._logger = logger if logger is not None else logging
        self._layout = SnapshotLayout(profile=snapshot_reader.profile)
        self._registry = TrajectoryFrontierRegistry(
            page_tokens=int(page_tokens),
            host_page_bytes_all_layers=int(host_page_bytes_all_layers),
            release_record=self._release_record,
        )
        self._arenas_by_handle: dict[int, SnapshotArena] = {}
        # A resident prefix page can back MORE than one frontier: the
        # coordinator dedups shared prefixes, so two trajectories over the same
        # system prefix legitimately list the same page. Evicting that page
        # retires every frontier standing on it.
        self._page_sessions: dict[tuple[int, int], set[str]] = {}
        self._pages_by_session: dict[str, set[tuple[int, int]]] = {}
        self._next_storage_handle = 0

    # -- read ---------------------------------------------------------------

    @property
    def registry(self) -> TrajectoryFrontierRegistry:
        return self._registry

    @property
    def layout(self) -> SnapshotLayout:
        return self._layout

    def arena_for(self, storage_handle: int) -> SnapshotArena | None:
        return self._arenas_by_handle.get(int(storage_handle))

    def reverse_index(self) -> dict[tuple[int, int], frozenset[str]]:
        """(group_id, page_id) -> session keys, for tests and invariants."""

        return {
            key: frozenset(values)
            for key, values in self._page_sessions.items()
        }

    # -- the commit step ----------------------------------------------------

    def commit(
        self,
        *,
        namespace_digest: Sequence[int],
        frontier_token_ids: Sequence[int],
        page_ids_by_group: Mapping[int, Sequence[int]],
        owning_group_id: int,
        sequence_id: int,
    ) -> TrajectoryCommitOutcome | None:
        """Snapshot the sequence's KDA state and install/advance its frontier.

        Returns ``None`` when the step is deliberately skipped: the frontier is
        already committed at this exact boundary, or the Host pool could not
        supply arena pages. Every other failure is raised by name.
        """

        tokens = tuple(int(token_id) for token_id in frontier_token_ids)
        resolution = self._registry.resolve(
            namespace_digest=namespace_digest, prompt_token_ids=tokens
        )
        previous = resolution.record if resolution.hit else None
        if (
            previous is not None
            and previous.frontier_boundary_tokens == len(tokens)
        ):
            self._logger.info(
                "[%s] rank=%s session=%s already committed at boundary=%s; "
                "nothing to advance",
                PREFIX,
                self._rank,
                previous.session_key,
                len(tokens),
            )
            return None

        prefix_pages = sum(len(pages) for pages in page_ids_by_group.values())
        arena_pages = self._layout.arena_page_count(
            host_page_bytes_all_layers=self._host_page_bytes_all_layers,
            prefix_page_count=prefix_pages,
        )

        # 1. D2H copy of the slot, in the fixed profile order.
        chunks = self._snapshot_reader.read_slot(int(sequence_id))
        self._validate_chunks(chunks)

        # 2. Arena pages from the Host KV allocator. Pool pressure is the one
        #    recoverable outcome: skip the frontier commit, never block the
        #    retention transaction that already completed above us.
        try:
            arena = self._arena_allocator.store(
                page_count=arena_pages, chunks=chunks
            )
        except SnapshotArenaUnavailable as exc:
            self._logger.warning(
                "[%s] rank=%s seq=%s SKIPPED frontier commit at boundary=%s: "
                "no Host KV pages for a %s-page snapshot arena (%s); "
                "trajectory reuse degrades to recompute",
                PREFIX,
                self._rank,
                int(sequence_id),
                len(tokens),
                arena_pages,
                exc,
            )
            return None

        handle = self._allocate_storage_handle()
        arena = replace(arena, storage_handle=handle)
        self._arenas_by_handle[handle] = arena
        snapshot = KDASnapshotDescriptor(
            storage_handle=handle,
            owning_group_id=int(owning_group_id),
            profile=self._layout.profile,
            boundary_tokens=len(tokens),
        )

        # 3. Install. ``advance`` references the new record BEFORE releasing the
        #    old one, so the old snapshot pages are freed only after the swap.
        #    A failure anywhere in here leaves the old record and its pages
        #    intact; the NEW arena is the only thing rolled back.
        try:
            if previous is None:
                record = self._registry.begin(
                    namespace_digest=namespace_digest,
                    frontier_token_ids=tokens,
                    page_ids_by_group=page_ids_by_group,
                    snapshot=snapshot,
                )
                advance = None
            else:
                advance = self._registry.advance(
                    session_key=previous.session_key,
                    namespace_digest=namespace_digest,
                    frontier_token_ids=tokens,
                    page_ids_by_group=page_ids_by_group,
                    snapshot=snapshot,
                )
                record = advance.record
        except Exception:
            self._arenas_by_handle.pop(handle, None)
            self._arena_allocator.release(arena)
            raise

        self._index_pages(record)
        self._logger.info(
            "[%s] rank=%s seq=%s session=%s %s boundary=%s prefix_pages=%s "
            "arena_pages=%s snapshot_bytes=%s",
            PREFIX,
            self._rank,
            int(sequence_id),
            record.session_key,
            "advanced" if advance is not None else "began",
            record.frontier_boundary_tokens,
            record.page_count,
            arena.pages_per_group,
            arena.payload_bytes,
        )
        return TrajectoryCommitOutcome(
            session_key=record.session_key,
            record=record,
            arena=arena,
            advanced=advance is not None,
            previous_release=(
                advance.previous_release if advance is not None else None
            ),
        )

    # -- the eviction hook --------------------------------------------------

    def on_prefix_pages_released(
        self, released_page_ids_by_group: Mapping[int, Sequence[int]]
    ) -> tuple[FrontierReleaseOutcome, ...]:
        """Retire every frontier whose prefix pages just left the pool.

        A frontier is all-or-nothing: ONE evicted page makes the record a miss
        by construction, so the whole record goes. What leaves with it is the
        snapshot arena and the registry entry -- never the prefix pages, which
        the caller has already released and which the prefix node owns.
        """

        sessions: list[str] = []
        seen: set[str] = set()
        for group_id, page_ids in sorted(released_page_ids_by_group.items()):
            for page_id in page_ids:
                owners = self._page_sessions.get((int(group_id), int(page_id)))
                for session_key in sorted(owners or ()):
                    if session_key not in seen:
                        seen.add(session_key)
                        sessions.append(session_key)
        outcomes: list[FrontierReleaseOutcome] = []
        for session_key in sessions:
            if self._registry.get(session_key) is None:
                # Reachable only after a physical release failed and left the
                # record retryable. Named, never skipped: a stale index entry
                # means a snapshot arena is still holding Host pages.
                raise SnapshotArenaCorrupt(
                    f"the page->session index names session {session_key}, "
                    "which the registry no longer holds; a previous frontier "
                    "release failed and was never retried"
                )
            outcome = self._registry.evict(session_key)
            outcomes.append(outcome)
            self._logger.info(
                "[%s] rank=%s session=%s evicted with its prefix pages: "
                "boundary=%s arena_handle=%s",
                PREFIX,
                self._rank,
                session_key,
                outcome.charge.page_count,
                outcome.released_snapshot_handle,
            )
        return tuple(outcomes)

    # -- internals ----------------------------------------------------------

    def _release_record(self, record: TrajectoryFrontierRecord) -> None:
        """Free the record's SNAPSHOT pages and its reverse-index entries.

        Deliberately asymmetric: ``record.page_ids_by_group`` (the retained
        prefix pages) is never passed to the allocator. Those pages belong to
        the prefix node and are released by the prefix-cache eviction path --
        freeing them here would be a double free.
        """

        handle = int(record.snapshot.storage_handle)
        arena = self._arenas_by_handle.pop(handle, None)
        if arena is None:
            raise SnapshotArenaCorrupt(
                f"frontier record for session {record.session_key} references "
                f"snapshot storage handle {handle}, which has no arena"
            )
        self._arena_allocator.release(arena)
        self._drop_indexed_pages(record.session_key)

    def _allocate_storage_handle(self) -> int:
        # Strictly positive: KDASnapshotDescriptor refuses a zero handle, so a
        # record can never carry an unfilled snapshot.
        self._next_storage_handle += 1
        return self._next_storage_handle

    def _validate_chunks(self, chunks: Sequence[object]) -> None:
        expected = self._layout.chunk_bytes
        if len(chunks) != len(expected):
            raise TrajectorySnapshotShapeMismatch(
                f"a KDA snapshot is {len(expected)} chunks "
                f"({', '.join(SNAPSHOT_CHUNK_NAMES)}), got {len(chunks)}"
            )
        for name, chunk, size in zip(SNAPSHOT_CHUNK_NAMES, chunks, expected):
            actual = int(chunk.numel())
            if actual != int(size):
                raise TrajectorySnapshotShapeMismatch(
                    f"KDA snapshot chunk {name!r} copied {actual} B but the "
                    f"profile says {size} B"
                )

    def _index_pages(self, record: TrajectoryFrontierRecord) -> None:
        keys = {
            (int(group_id), int(page_id))
            for group_id, page_ids in record.page_ids_by_group.items()
            for page_id in page_ids
        }
        for key in keys:
            self._page_sessions.setdefault(key, set()).add(record.session_key)
        self._pages_by_session[record.session_key] = keys

    def _drop_indexed_pages(self, session_key: str) -> None:
        key_name = str(session_key)
        for key in self._pages_by_session.pop(key_name, ()):
            owners = self._page_sessions.get(key)
            if owners is None:
                continue
            owners.discard(key_name)
            if not owners:
                del self._page_sessions[key]
