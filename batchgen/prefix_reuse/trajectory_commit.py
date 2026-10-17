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

# Why a commit step did NOT install a frontier. Strings, not booleans, because
# a skip must be countable and nameable in the log: every one of these looks
# exactly like a clean run from the outside.
SKIP_BOUNDARY_MISALIGNED = "boundary_misaligned"
SKIP_ALREADY_COMMITTED = "already_committed"
SKIP_ARENA_UNAVAILABLE = "arena_unavailable"
SKIP_ARENA_MISSING = "arena_missing"

SKIP_REASONS = (
    SKIP_BOUNDARY_MISALIGNED,
    SKIP_ALREADY_COMMITTED,
    SKIP_ARENA_UNAVAILABLE,
    SKIP_ARENA_MISSING,
)


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

    def shard_page_count(
        self, *, host_page_bytes_all_layers: int, prefix_page_count: int
    ) -> int:
        """Host pages ONE rank's shard of this blob needs.

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

    def arena_page_count(
        self,
        *,
        host_page_bytes_all_layers: int,
        prefix_page_count: int,
        shard_count: int = 1,
    ) -> int:
        """Host pages the WHOLE snapshot needs: G x one shard's pages.

        Each shard is rounded up to whole pages independently, so a rank's
        window is a whole number of pages and two ranks can never share one.
        ``shard_count=1`` is the pure-DP arena and is byte-identical to the
        pre-shard figure.
        """

        charge = FrontierCharge(
            page_count=int(prefix_page_count),
            host_page_bytes_all_layers=int(host_page_bytes_all_layers),
            snapshot_bytes=self.total_bytes,
            shard_count=int(shard_count),
        )
        return int(charge.arena_host_pages)


@dataclass(frozen=True)
class SnapshotArena:
    """Host KV pages holding one snapshot blob.

    ``pseudo_sid`` is the negative Host sequence id the pages were allocated
    under. After ``retain``, the pages are prefix-resident and the pseudo
    sequence is unregistered, so the arena is identified by its page ids alone
    and is released with ``release_resident_pages``.

    SHARD LAYOUT (the G>1 contract): the arena holds ``shard_count`` equal
    windows, and rank ``r`` of the owning group owns window ``r``: pages
    ``[r * pages_per_shard : (r + 1) * pages_per_shard]`` of EVERY group's page
    list. The windows are page-disjoint by construction, which is what lets G
    ranks write the same arena concurrently with no coordination beyond knowing
    the page list. ``shard_count == 1`` is the pure-DP arena: one window, the
    whole page list, byte-identical to the pre-shard layout.
    """

    storage_handle: int
    pseudo_sid: int
    page_ids_by_group: Mapping[int, tuple[int, ...]]
    pages_per_group: int
    capacity_bytes: int
    payload_bytes: int
    shard_count: int = 1

    def __post_init__(self) -> None:
        if int(self.pseudo_sid) >= 0:
            raise SnapshotArenaCorrupt(
                "a snapshot arena must live under a NEGATIVE pseudo sequence "
                f"id so it cannot collide with a real sequence, got "
                f"{self.pseudo_sid}"
            )
        if not self.page_ids_by_group:
            raise SnapshotArenaCorrupt("a snapshot arena needs pages")
        if int(self.shard_count) <= 0:
            raise SnapshotArenaCorrupt(
                "a snapshot arena holds at least one shard window, got "
                f"shard_count={self.shard_count}"
            )
        if int(self.pages_per_group) % int(self.shard_count):
            raise SnapshotArenaCorrupt(
                f"{self.pages_per_group} arena page(s) per group do not divide "
                f"into {self.shard_count} equal shard windows; a rank's window "
                "would have to share a Host page with its neighbour"
            )
        for group_id, page_ids in self.page_ids_by_group.items():
            if len(tuple(page_ids)) != int(self.pages_per_group):
                raise SnapshotArenaCorrupt(
                    f"snapshot arena group {group_id} lists "
                    f"{len(tuple(page_ids))} page(s) but the arena is "
                    f"{self.pages_per_group} page(s) per group"
                )
        if int(self.payload_bytes) > int(self.capacity_bytes):
            raise SnapshotArenaCorrupt(
                f"snapshot payload {self.payload_bytes} B exceeds the arena's "
                f"{self.capacity_bytes} B of Host pages"
            )

    @property
    def pages_per_shard(self) -> int:
        """Host pages ONE rank's window holds, in every group."""

        return int(self.pages_per_group) // int(self.shard_count)

    def shard_window(self, shard_index: int) -> dict[int, tuple[int, ...]]:
        """The page ids rank ``shard_index`` owns, per KV group.

        The ONE place the window is derived, so the write side and the read
        side cannot drift: a restore that computed the window differently from
        the commit would read another rank's shard and bind plausible-looking
        wrong state.
        """

        index = int(shard_index)
        if index < 0 or index >= int(self.shard_count):
            raise SnapshotArenaCorrupt(
                f"shard index {index} is outside the arena's "
                f"{self.shard_count} shard window(s)"
            )
        span = self.pages_per_shard
        start = index * span
        return {
            int(group_id): tuple(page_ids)[start : start + span]
            for group_id, page_ids in self.page_ids_by_group.items()
        }


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

    def allocate(
        self, *, page_count: int, shard_count: int = 1
    ) -> SnapshotArena:
        """Take ``page_count`` Host pages per group; write and retain nothing.

        PHASE 1 of the G>1 transaction, and the only phase that mutates
        allocator state, so it runs on the host-KV LEADER alone. The pages leave
        the free pool here -- which is the whole point of R1: the boundary
        planner sees them gone and cannot double-promise the bytes -- while the
        pseudo sequence stays REGISTERED and the pages stay chain-owned, so
        ``retain`` can still find them in a later phase of the same pass.

        The returned arena is not yet usable as a record: it has no bytes in
        it. A caller that abandons it MUST hand it to ``rollback``.
        """

        pages = int(page_count)
        shards = int(shard_count)
        if pages <= 0:
            raise SnapshotArenaCorrupt(
                f"a snapshot arena needs at least one Host page, got {pages}"
            )
        if shards <= 0:
            raise SnapshotArenaCorrupt(
                f"a snapshot arena needs at least one shard, got {shards}"
            )
        if pages % shards:
            raise SnapshotArenaCorrupt(
                f"{pages} arena page(s) do not divide into {shards} equal "
                "shard windows"
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
        return SnapshotArena(
            # The caller stamps the record-facing handle; the arena identity
            # inside the allocator is its pseudo sid.
            storage_handle=0,
            pseudo_sid=pseudo_sid,
            page_ids_by_group=page_ids_by_group,
            pages_per_group=pages,
            capacity_bytes=pages * self._host_page_bytes_all_layers,
            payload_bytes=0,
            shard_count=shards,
        )

    def write_shard(
        self,
        arena: SnapshotArena,
        *,
        shard_index: int,
        chunks: Sequence[object],
    ) -> int:
        """Write ONE rank's shard into its own window. Mutates no allocator.

        PHASE 2, and the phase that runs on EVERY rank of the owning group.
        All local ranks map the same node-shared Host KV region, so a rank only
        needs the arena's page list to reach its own bytes: the local page table
        is narrowed to this shard's window with ``set_sequence_pages`` (a
        rank-local page-table write -- the same call the G>1 host-KV mirror uses
        for the leader's page tables) and the blob is copied into exactly those
        pages. Nothing is registered, allocated, retained or released here, so a
        follower never touches state only the leader may own.

        The full page list is put back into the local row before returning, so
        the leader's later ``retain`` sees the whole arena and a follower's row
        is left describing the arena it was handed rather than a slice of it.
        """

        views = dict(self._host_views_provider())
        if not views:
            raise SnapshotArenaCorrupt(
                "no Host KV worker views to write a snapshot shard into"
            )
        window = arena.shard_window(shard_index)
        missing = sorted(set(window) - set(int(key) for key in views))
        if missing:
            raise SnapshotArenaCorrupt(
                "missing Host KV worker view for snapshot arena group(s) "
                f"{missing}"
            )
        capacity = arena.pages_per_shard * self._host_page_bytes_all_layers
        payload = sum(int(chunk.numel()) for chunk in chunks)
        if payload > capacity:
            raise TrajectorySnapshotShapeMismatch(
                f"snapshot payload {payload} B does not fit shard window "
                f"{int(shard_index)}'s {capacity} B "
                f"({arena.pages_per_shard} page(s) x "
                f"{self._host_page_bytes_all_layers} B)"
            )
        pseudo_sid = int(arena.pseudo_sid)
        try:
            for group_id, page_ids in sorted(window.items()):
                views[int(group_id)].set_sequence_pages(
                    pseudo_sid, [int(page) for page in page_ids]
                )
            self._write_blob(
                {int(key): views[int(key)] for key in window}, pseudo_sid, chunks
            )
        finally:
            # Restore the row to the WHOLE arena, whatever happened: a row left
            # describing one window would make a later retain silently retain a
            # slice.
            for group_id, page_ids in sorted(arena.page_ids_by_group.items()):
                view = views.get(int(group_id))
                if view is None:
                    continue
                try:
                    view.set_sequence_pages(
                        pseudo_sid, [int(page) for page in page_ids]
                    )
                except Exception:  # noqa: BLE001 - restoring the row is best effort
                    logging.warning(
                        "[%s] could not restore the full arena page row for "
                        "pseudo_sid=%s group=%s",
                        PREFIX,
                        pseudo_sid,
                        group_id,
                    )
        return payload

    def retain(self, arena: SnapshotArena) -> SnapshotArena:
        """Make the arena's pages prefix-resident and drop the pseudo row.

        PHASE 3, leader-only. After this the pages survive the pseudo sequence
        and the arena is identified by its page ids alone, exactly as the
        single-shot path leaves it.
        """

        views = dict(self._host_views_provider())
        if not views:
            raise SnapshotArenaCorrupt(
                "no Host KV worker views to retain a snapshot arena in"
            )
        pseudo_sid = int(arena.pseudo_sid)
        try:
            for group_id, page_ids in sorted(arena.page_ids_by_group.items()):
                view = views.get(int(group_id))
                if view is None:
                    raise SnapshotArenaCorrupt(
                        "missing Host KV worker view for snapshot arena group "
                        f"{group_id}"
                    )
                view.set_sequence_pages(
                    pseudo_sid, [int(page) for page in page_ids]
                )
                retained = [
                    int(page)
                    for page in view.retain_sequence_pages(
                        pseudo_sid, [int(page) for page in page_ids]
                    )
                ]
                if retained != [int(page) for page in page_ids]:
                    raise SnapshotArenaCorrupt(
                        "Host KV retained different pages for the snapshot "
                        f"arena of group {group_id}"
                    )
        except Exception:
            # Retain may have succeeded for earlier groups; release what is
            # resident and let the chain release handle the rest.
            self._release_pages(views, arena.page_ids_by_group, best_effort=True)
            self._rollback(views, sorted(views), pseudo_sid)
            raise
        # The pages are prefix-resident now; the pseudo sequence has an empty
        # chain and its sequence-table slot can go back.
        for _group_id, view in sorted(views.items()):
            view.unregister_sequences([pseudo_sid])
        return arena

    def rollback(self, arena: SnapshotArena) -> None:
        """Hand an allocated-but-never-retained arena's pages back.

        Leader-only: it is the inverse of ``allocate``, so it mutates the same
        allocator state. Used for every path that pre-allocated an arena and
        then did not commit a record into it -- a misaligned boundary, a
        coordinator that no longer holds the prefix, an already-committed
        frontier. Without it those pages would stay out of the free pool with
        no record naming them.
        """

        views = dict(self._host_views_provider())
        self._rollback(views, sorted(int(key) for key in views), arena.pseudo_sid)

    def store(self, *, page_count: int, chunks: Sequence[object]) -> SnapshotArena:
        """Allocate ``page_count`` Host pages per group and write the blob.

        The single-shot pure-DP (G==1) transaction: one rank owns the whole
        arena, so all three phases run back to back with no cross-rank ordering
        to respect. Composed from the phases rather than duplicating them, so
        the one-shard and G-shard paths cannot drift.
        """

        arena = self.allocate(page_count=int(page_count), shard_count=1)
        try:
            payload = self.write_shard(arena, shard_index=0, chunks=chunks)
        except Exception:
            self.rollback(arena)
            raise
        # ``retain`` owns its own cleanup, so a failure in it must NOT be
        # rolled back a second time here: the pseudo row is already gone.
        return self.retain(replace(arena, payload_bytes=payload))

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
    """What one flag-on commit step did.

    ``installed`` separates the pure-DP transaction from the G>1 one. At G==1
    one rank owns the whole arena, so the record is installed inside
    ``commit``. At G>1 the record binds G shards written by G different ranks,
    so ``commit`` only writes THIS rank's shard and the record is installed by
    ``finalize`` once every rank has passed the all-ranks barrier.
    """

    session_key: str
    record: TrajectoryFrontierRecord | None
    arena: SnapshotArena
    advanced: bool
    previous_release: FrontierReleaseOutcome | None
    installed: bool = True
    shard_index: int = 0
    shard_bytes: int = 0


@dataclass(frozen=True)
class _PendingFrontier:
    """A shard written, waiting for the barrier before its record installs."""

    sequence_id: int
    namespace_digest: tuple[int, ...]
    frontier_token_ids: tuple[int, ...]
    page_ids_by_group: Mapping[int, tuple[int, ...]]
    snapshot: KDASnapshotDescriptor
    arena: SnapshotArena
    previous_session_key: str | None


@dataclass(frozen=True)
class TrajectoryFinalizeOutcome:
    """What the post-barrier finalize step installed and what it gave back."""

    installed: tuple[TrajectoryCommitOutcome, ...]
    rolled_back_sequence_ids: tuple[int, ...]
    failed_sequence_ids: tuple[int, ...]


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
        shard_count: int = 1,
        shard_index: int = 0,
        owns_registry: bool = True,
        logger=None,
    ) -> None:
        self._rank = int(rank)
        if int(shard_count) <= 0:
            raise SnapshotArenaCorrupt(
                f"shard_count must be positive, got {shard_count}"
            )
        if not 0 <= int(shard_index) < int(shard_count):
            raise SnapshotArenaCorrupt(
                f"shard_index {shard_index} is outside a {shard_count}-shard "
                "owning group"
            )
        if int(shard_count) == 1 and not owns_registry:
            raise SnapshotArenaCorrupt(
                "a pure-DP (shard_count=1) committer is the only rank of its "
                "group, so it must own its registry"
            )
        self._shard_count = int(shard_count)
        self._shard_index = int(shard_index)
        self._owns_registry = bool(owns_registry)
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
        # G>1 only: arenas the LEADER pre-allocated for this pass (sequence id
        # -> arena), and the shards already written into them awaiting the
        # post-barrier install.
        self._reserved: dict[int, SnapshotArena] = {}
        self._pending: dict[int, _PendingFrontier] = {}
        self._skips: dict[str, int] = {}

    # -- read ---------------------------------------------------------------

    @property
    def registry(self) -> TrajectoryFrontierRegistry:
        return self._registry

    @property
    def layout(self) -> SnapshotLayout:
        return self._layout

    @property
    def shard_count(self) -> int:
        return self._shard_count

    @property
    def shard_index(self) -> int:
        return self._shard_index

    @property
    def owns_registry(self) -> bool:
        return self._owns_registry

    def skips(self) -> dict[str, int]:
        """Named skip counters. A skip looks like a clean run; this is how it
        becomes visible."""

        return {reason: int(self._skips.get(reason, 0)) for reason in SKIP_REASONS}

    def reserved_sequence_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._reserved))

    def pending_sequence_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._pending))

    def _skip(self, reason: str, message: str, *args) -> None:
        if reason not in SKIP_REASONS:
            raise SnapshotArenaCorrupt(
                f"unknown trajectory commit skip reason {reason!r}; known "
                "reasons are " + ", ".join(SKIP_REASONS)
            )
        self._skips[reason] = self._skips.get(reason, 0) + 1
        # WARNING, not INFO: every one of these is a frontier the operator
        # asked for and did not get, and it is indistinguishable from a clean
        # run in the rest of the log.
        self._logger.warning(message, *args)

    def arena_for(self, storage_handle: int) -> SnapshotArena | None:
        return self._arenas_by_handle.get(int(storage_handle))

    def reverse_index(self) -> dict[tuple[int, int], frozenset[str]]:
        """(group_id, page_id) -> session keys, for tests and invariants."""

        return {
            key: frozenset(values)
            for key, values in self._page_sessions.items()
        }

    # -- the boundary identity ---------------------------------------------

    def boundary_is_aligned(
        self, *, prompt_token_count: int, commit_tokens: int
    ) -> bool:
        """Does the live KDA slot hold exactly the state the boundary claims?

        THE Slice 1c gate. The D2H reads the LIVE slot, which at the commit
        boundary holds the state after every token the prefill just ran --
        ``prompt_token_count`` of them. The record, meanwhile, is stamped with
        the page-aligned ``commit_tokens``. The two are the same state ONLY
        when they are the same number: a snapshot taken after more tokens than
        its boundary claims is AHEAD of its own restore point, and a restore
        would re-apply the tail tokens onto state that already contains them.
        That is silent wrong state, so a mismatch is refused rather than
        stored. (Capturing mid-prefill at the aligned boundary is the future
        lift; it is not this slice.)
        """

        return int(prompt_token_count) == int(commit_tokens)

    def plan_arena_pages(self, *, prefix_page_count: int) -> int:
        """Host pages one snapshot arena needs: G x one shard's pages."""

        return self._layout.arena_page_count(
            host_page_bytes_all_layers=self._host_page_bytes_all_layers,
            prefix_page_count=int(prefix_page_count),
            shard_count=self._shard_count,
        )

    # -- G>1 arena reservation ---------------------------------------------

    def reserve_arena(
        self, *, sequence_id: int, prefix_page_count: int
    ) -> SnapshotArena | None:
        """Pre-allocate the G-wide arena for a sequence. Leader-only.

        PHASE 1 of the G>1 transaction, run in prefill config so the arena's
        page ids can ride the host-KV mirror that already runs there -- which
        is what lets every rank of the group find its own window with no new
        collective. The pages leave the free pool here, so the R1 invariant
        holds exactly as before: the boundary planner cannot double-promise
        bytes a snapshot is already holding.

        ``None`` means pool pressure, the one recoverable outcome: no arena, so
        no rank writes a shard and the sequence's frontier is simply not
        committed this pass. Every arena handed out here is either consumed by
        a ``commit`` or rolled back by ``finalize`` -- never leaked.
        """

        if not self._owns_registry:
            raise SnapshotArenaCorrupt(
                f"rank {self._rank} does not own its group's frontier "
                "registry, so it must not allocate a snapshot arena; it adopts "
                "the leader's arena instead"
            )
        key = int(sequence_id)
        if key in self._reserved:
            raise SnapshotArenaCorrupt(
                f"sequence {key} already has a reserved snapshot arena; a "
                "second reservation would leak the first one's Host pages"
            )
        arena_pages = self.plan_arena_pages(prefix_page_count=prefix_page_count)
        try:
            arena = self._arena_allocator.allocate(
                page_count=arena_pages, shard_count=self._shard_count
            )
        except SnapshotArenaUnavailable as exc:
            self._skip(
                SKIP_ARENA_UNAVAILABLE,
                "[%s] rank=%s seq=%s SKIPPED frontier commit: no Host KV "
                "pages for a %s-page (%s x %s) snapshot arena (%s); trajectory "
                "reuse degrades to recompute",
                PREFIX,
                self._rank,
                key,
                arena_pages,
                self._shard_count,
                arena_pages // self._shard_count,
                exc,
            )
            return None
        self._reserved[key] = arena
        return arena

    def rollback_reservation(self, *, sequence_id: int) -> bool:
        """Give one reserved-but-unused arena's pages back, out of band.

        ``finalize`` is the normal path. This exists for the teardown seam: a
        pass that dies between the reservation and the barrier never reaches
        finalize, and its arena would otherwise keep Host pages with no record
        naming them AND block the next reservation for that sequence. Returns
        whether there was anything to give back.
        """

        arena = self._reserved.pop(int(sequence_id), None)
        if arena is None:
            return False
        if not self._owns_registry:
            # A follower never owned those pages; dropping the object is all
            # it may do.
            return True
        self._arena_allocator.rollback(arena)
        self._logger.info(
            "[%s] rank=%s seq=%s gave back an unused snapshot arena "
            "(pseudo_sid=%s) outside the finalize step",
            PREFIX,
            self._rank,
            int(sequence_id),
            arena.pseudo_sid,
        )
        return True

    def adopt_arena(self, *, sequence_id: int, arena: SnapshotArena) -> None:
        """Take the leader's arena for a sequence this rank only writes into.

        The follower side of the mirror: the arena object is rebuilt from the
        page ids the leader published, and this rank will write exactly its own
        window into it. It never allocates, retains or releases those pages --
        the leader owns their lifetime.
        """

        if self._owns_registry:
            raise SnapshotArenaCorrupt(
                f"rank {self._rank} owns its group's frontier registry, so it "
                "allocates its arenas rather than adopting one"
            )
        if int(arena.shard_count) != self._shard_count:
            raise SnapshotArenaCorrupt(
                f"the mirrored arena holds {arena.shard_count} shard "
                f"window(s) but this rank serves a {self._shard_count}-rank "
                "group"
            )
        self._reserved[int(sequence_id)] = arena

    # -- the commit step ----------------------------------------------------

    def commit(
        self,
        *,
        namespace_digest: Sequence[int],
        frontier_token_ids: Sequence[int],
        page_ids_by_group: Mapping[int, Sequence[int]],
        owning_group_id: int,
        sequence_id: int,
        prompt_token_count: int,
    ) -> TrajectoryCommitOutcome | None:
        """Snapshot this rank's KDA shard and install/advance the frontier.

        Returns ``None`` when the step is deliberately skipped, by name: the
        prompt does not end on the committed boundary, the frontier is already
        committed at this exact boundary, or the Host pool could not supply
        arena pages. Every other failure is raised by name.

        At G==1 the record is installed here. At G>1 only THIS rank's shard is
        written and the record is left pending for ``finalize``, which runs
        after the all-ranks barrier -- by then every rank of the group has
        written its own window, so the record binds a complete snapshot.
        """

        tokens = tuple(int(token_id) for token_id in frontier_token_ids)
        if not self.boundary_is_aligned(
            prompt_token_count=prompt_token_count, commit_tokens=len(tokens)
        ):
            self._skip(
                SKIP_BOUNDARY_MISALIGNED,
                "[%s] rank=%s seq=%s SKIPPED frontier commit: the prefill ran "
                "%s token(s) but the page-aligned boundary is %s, so the live "
                "KDA slot is AHEAD of the boundary a record would claim; "
                "committing it would let a restore re-apply %s token(s)",
                PREFIX,
                self._rank,
                int(sequence_id),
                int(prompt_token_count),
                len(tokens),
                int(prompt_token_count) - len(tokens),
            )
            return None
        if not self._owns_registry:
            raise SnapshotArenaCorrupt(
                f"rank {self._rank} does not own its group's frontier "
                "registry; it writes its shard with write_shard and installs "
                "nothing"
            )
        resolution = self._registry.resolve(
            namespace_digest=namespace_digest, prompt_token_ids=tokens
        )
        previous = resolution.record if resolution.hit else None
        if (
            previous is not None
            and previous.frontier_boundary_tokens == len(tokens)
        ):
            self._skip(
                SKIP_ALREADY_COMMITTED,
                "[%s] rank=%s session=%s already committed at boundary=%s; "
                "nothing to advance",
                PREFIX,
                self._rank,
                previous.session_key,
                len(tokens),
            )
            return None

        # 1. D2H copy of THIS rank's slot shard, in the fixed profile order.
        chunks = self._snapshot_reader.read_slot(int(sequence_id))
        self._validate_chunks(chunks)

        if self._shard_count > 1:
            return self._commit_shard(
                namespace_digest=namespace_digest,
                tokens=tokens,
                page_ids_by_group=page_ids_by_group,
                owning_group_id=int(owning_group_id),
                sequence_id=int(sequence_id),
                previous=previous,
                chunks=chunks,
            )

        # 2. Pure DP: one rank owns the whole arena, so allocate, write and
        #    retain it in one step. Pool pressure is the one recoverable
        #    outcome: skip the frontier commit, never block the retention
        #    transaction that already completed above us.
        prefix_pages = sum(len(pages) for pages in page_ids_by_group.values())
        arena_pages = self.plan_arena_pages(prefix_page_count=prefix_pages)
        try:
            arena = self._arena_allocator.store(
                page_count=arena_pages, chunks=chunks
            )
        except SnapshotArenaUnavailable as exc:
            self._skip(
                SKIP_ARENA_UNAVAILABLE,
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
            shard_count=self._shard_count,
        )

        # 3. Install. ``advance`` references the new record BEFORE releasing the
        #    old one, so the old snapshot pages are freed only after the swap.
        #    A failure anywhere in here leaves the old record and its pages
        #    intact; the NEW arena is the only thing rolled back.
        try:
            outcome = self._install(
                namespace_digest=namespace_digest,
                tokens=tokens,
                page_ids_by_group=page_ids_by_group,
                snapshot=snapshot,
                arena=arena,
                previous_session_key=(
                    None if previous is None else previous.session_key
                ),
                sequence_id=int(sequence_id),
            )
        except Exception:
            self._arenas_by_handle.pop(handle, None)
            self._arena_allocator.release(arena)
            raise
        return outcome

    def _commit_shard(
        self,
        *,
        namespace_digest: Sequence[int],
        tokens: tuple[int, ...],
        page_ids_by_group: Mapping[int, Sequence[int]],
        owning_group_id: int,
        sequence_id: int,
        previous,
        chunks: Sequence[object],
    ) -> TrajectoryCommitOutcome | None:
        """G>1: write this rank's window and leave the record pending."""

        arena = self._reserved.pop(int(sequence_id), None)
        if arena is None:
            self._skip(
                SKIP_ARENA_MISSING,
                "[%s] rank=%s seq=%s SKIPPED frontier commit at boundary=%s: "
                "no snapshot arena was reserved for it this pass",
                PREFIX,
                self._rank,
                int(sequence_id),
                len(tokens),
            )
            return None
        written = self._arena_allocator.write_shard(
            arena, shard_index=self._shard_index, chunks=chunks
        )
        arena = replace(arena, payload_bytes=written)
        handle = self._allocate_storage_handle()
        arena = replace(arena, storage_handle=handle)
        snapshot = KDASnapshotDescriptor(
            storage_handle=handle,
            owning_group_id=int(owning_group_id),
            profile=self._layout.profile,
            boundary_tokens=len(tokens),
            shard_count=self._shard_count,
        )
        self._pending[int(sequence_id)] = _PendingFrontier(
            sequence_id=int(sequence_id),
            namespace_digest=tuple(int(value) for value in namespace_digest),
            frontier_token_ids=tokens,
            page_ids_by_group={
                int(group_id): tuple(int(page) for page in pages)
                for group_id, pages in page_ids_by_group.items()
            },
            snapshot=snapshot,
            arena=arena,
            previous_session_key=(
                None if previous is None else previous.session_key
            ),
        )
        self._logger.info(
            "[%s] rank=%s seq=%s shard=%s/%s written boundary=%s "
            "arena_pages=%s shard_pages=%s bytes=%s; record pending until the "
            "all-ranks barrier",
            PREFIX,
            self._rank,
            int(sequence_id),
            self._shard_index,
            self._shard_count,
            len(tokens),
            arena.pages_per_group,
            arena.pages_per_shard,
            written,
        )
        return TrajectoryCommitOutcome(
            session_key="",
            record=None,
            arena=arena,
            advanced=previous is not None,
            previous_release=None,
            installed=False,
            shard_index=self._shard_index,
            shard_bytes=int(written),
        )

    def write_shard(
        self,
        *,
        sequence_id: int,
        frontier_token_ids: Sequence[int],
        prompt_token_count: int,
    ) -> int | None:
        """Write this rank's shard into the leader's arena. Follower-only.

        No registry write, no allocator mutation, no record: the leader owns
        what a frontier IS and this rank owns only its own head shard of the
        state. The hit/miss and commit/skip decisions are therefore the
        leader's, carried to this rank as the presence of an adopted arena --
        so the ranks of a group cannot disagree about which sequences have a
        committed frontier.
        """

        if self._owns_registry:
            raise SnapshotArenaCorrupt(
                f"rank {self._rank} owns its group's frontier registry; it "
                "commits records with commit, not write_shard"
            )
        tokens = tuple(int(token_id) for token_id in frontier_token_ids)
        if not self.boundary_is_aligned(
            prompt_token_count=prompt_token_count, commit_tokens=len(tokens)
        ):
            self._skip(
                SKIP_BOUNDARY_MISALIGNED,
                "[%s] rank=%s seq=%s SKIPPED shard write: the prefill ran %s "
                "token(s) but the boundary is %s",
                PREFIX,
                self._rank,
                int(sequence_id),
                int(prompt_token_count),
                len(tokens),
            )
            return None
        arena = self._reserved.pop(int(sequence_id), None)
        if arena is None:
            self._skip(
                SKIP_ARENA_MISSING,
                "[%s] rank=%s seq=%s SKIPPED shard write: its serve-group "
                "leader reserved no snapshot arena this pass",
                PREFIX,
                self._rank,
                int(sequence_id),
            )
            return None
        written = self._arena_allocator.write_shard(
            arena, shard_index=self._shard_index, chunks=self._read_chunks(
                int(sequence_id)
            )
        )
        self._logger.info(
            "[%s] rank=%s seq=%s shard=%s/%s written boundary=%s bytes=%s "
            "into the leader's arena (pseudo_sid=%s)",
            PREFIX,
            self._rank,
            int(sequence_id),
            self._shard_index,
            self._shard_count,
            len(tokens),
            written,
            arena.pseudo_sid,
        )
        return int(written)

    def _read_chunks(self, sequence_id: int) -> tuple[object, ...]:
        chunks = self._snapshot_reader.read_slot(int(sequence_id))
        self._validate_chunks(chunks)
        return chunks

    # -- the post-barrier finalize -----------------------------------------

    def finalize(self) -> TrajectoryFinalizeOutcome:
        """Install every pending record; give back every unused arena.

        Runs after the all-ranks barrier, so every rank of every owning group
        has already written its own shard window: a record installed here binds
        a COMPLETE snapshot. Before that barrier no record exists, so a crash,
        a rejected wave or any skip leaves the previously committed frontier
        exactly as it was.

        Atomic per sequence: ``retain`` makes one arena's pages prefix-resident
        and the record is installed immediately after. A failure in either
        gives that arena's pages back and leaves the old record installed,
        referenced and unreleased; the other sequences are unaffected.
        """

        if not self._owns_registry:
            # A follower installs nothing and owns no arena pages: dropping its
            # adopted arena objects is the whole of its finalize.
            self._reserved.clear()
            self._pending.clear()
            return TrajectoryFinalizeOutcome(
                installed=(), rolled_back_sequence_ids=(), failed_sequence_ids=()
            )
        installed: list[TrajectoryCommitOutcome] = []
        failed: list[int] = []
        for sequence_id in sorted(self._pending):
            pending = self._pending.pop(sequence_id)
            arena = pending.arena
            try:
                self._arena_allocator.retain(arena)
            except Exception:  # noqa: BLE001 - fail-safe by contract
                # ``retain`` hands its own pages back; nothing was installed, so
                # the previously committed frontier is untouched.
                self._logger.exception(
                    "[%s] rank=%s seq=%s retaining the snapshot arena failed; "
                    "its frontier is NOT committed and the previous record "
                    "stands",
                    PREFIX,
                    self._rank,
                    sequence_id,
                )
                failed.append(int(sequence_id))
                continue
            self._arenas_by_handle[int(arena.storage_handle)] = arena
            try:
                installed.append(
                    self._install(
                        namespace_digest=pending.namespace_digest,
                        tokens=pending.frontier_token_ids,
                        page_ids_by_group=pending.page_ids_by_group,
                        snapshot=pending.snapshot,
                        arena=arena,
                        previous_session_key=pending.previous_session_key,
                        sequence_id=int(sequence_id),
                    )
                )
            except Exception:  # noqa: BLE001 - fail-safe by contract
                self._logger.exception(
                    "[%s] rank=%s seq=%s installing the frontier record "
                    "failed; the arena's pages go back and the previous record "
                    "stands",
                    PREFIX,
                    self._rank,
                    sequence_id,
                )
                self._arenas_by_handle.pop(int(arena.storage_handle), None)
                self._arena_allocator.release(arena)
                failed.append(int(sequence_id))
        rolled_back: list[int] = []
        for sequence_id in sorted(self._reserved):
            arena = self._reserved.pop(sequence_id)
            try:
                self._arena_allocator.rollback(arena)
            except Exception:  # noqa: BLE001 - rollback is best effort
                self._logger.exception(
                    "[%s] rank=%s seq=%s could not give back the pages of an "
                    "unused snapshot arena (pseudo_sid=%s)",
                    PREFIX,
                    self._rank,
                    sequence_id,
                    arena.pseudo_sid,
                )
            rolled_back.append(int(sequence_id))
        if rolled_back:
            self._logger.info(
                "[%s] rank=%s gave back %s unused snapshot arena(s) for "
                "sequence(s) %s",
                PREFIX,
                self._rank,
                len(rolled_back),
                rolled_back,
            )
        return TrajectoryFinalizeOutcome(
            installed=tuple(installed),
            rolled_back_sequence_ids=tuple(rolled_back),
            failed_sequence_ids=tuple(failed),
        )

    def _install(
        self,
        *,
        namespace_digest: Sequence[int],
        tokens: tuple[int, ...],
        page_ids_by_group: Mapping[int, Sequence[int]],
        snapshot: KDASnapshotDescriptor,
        arena: SnapshotArena,
        previous_session_key: str | None,
        sequence_id: int,
    ) -> TrajectoryCommitOutcome:
        """Begin or advance the session's record and index its prefix pages."""

        if previous_session_key is None:
            record = self._registry.begin(
                namespace_digest=namespace_digest,
                frontier_token_ids=tokens,
                page_ids_by_group=page_ids_by_group,
                snapshot=snapshot,
            )
            advance = None
        else:
            advance = self._registry.advance(
                session_key=previous_session_key,
                namespace_digest=namespace_digest,
                frontier_token_ids=tokens,
                page_ids_by_group=page_ids_by_group,
                snapshot=snapshot,
            )
            record = advance.record
        self._index_pages(record)
        self._logger.info(
            "[%s] rank=%s seq=%s session=%s %s boundary=%s prefix_pages=%s "
            "arena_pages=%s shards=%s snapshot_bytes=%s",
            PREFIX,
            self._rank,
            int(sequence_id),
            record.session_key,
            "advanced" if advance is not None else "began",
            record.frontier_boundary_tokens,
            record.page_count,
            arena.pages_per_group,
            arena.shard_count,
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
            installed=True,
            shard_index=self._shard_index,
            shard_bytes=int(arena.payload_bytes),
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
