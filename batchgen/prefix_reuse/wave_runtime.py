"""Host side of the in-wave prefix pool for one DP rank.

A pooled segment is computed once, so its KV must land in Host memory once.
Each segment gets its own Host pages under a negative pseudo sequence id, one
copy per prefix-cache KV group (a DSA model holds the MLA latent and the
indexer keys of a segment under the same pseudo id in both Host-KV regions);
every member prompt's Host page table starts with its chain of segment pages
(attached, never owned), followed by its own private pages. At the end of the
wave the segments are committed to the prefix coordinator parents first, so a
member's own commit finds its chain resident and inserts only private pages.

Lifetime: a member holds a coordinator attachment on its chain, and a pseudo
sequence is released only after its last member releases, so Host pages a
member still reads can never be freed underneath it - whether the commit
retained them for the prefix cache or another rank had already published the
same tokens and they stayed with the pseudo sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from batchgen.prefix_reuse.executor import ChunkBatch, PoolChunkExecutor
from batchgen.prefix_reuse.pool import PrefillPrefixPool
from batchgen.prefix_reuse.wave_plan import WavePrefixPlan


@dataclass
class PoolWave:
    plan: WavePrefixPlan
    prompts: list[list[int]]
    uuids: list[str]
    global_ids: list[int]
    page_tokens: int
    total_pages: int
    # Exact transient bound the executor needs beside the planned segment
    # peak; construction may shrink the pool to a re-measured budget as long
    # as peak + scratch still fits.
    scratch_pages: int = 0
    # The GPU pool and its executor are constructed in the prefill phase,
    # after configure_prefill settles the device layout; planning and Host
    # segment allocation run earlier, inside the admission reservation.
    pool: PrefillPrefixPool | None = None
    executor: PoolChunkExecutor | None = None
    chains: list[tuple[int, ...]] = field(default_factory=list)
    segment_sids: dict[int, int] = field(default_factory=dict)
    # The pooled-prefill PROBE keys of this wave's own sequences, read exactly
    # once per wave by the worker and cached here. Four probes (epilogue trace,
    # offload check, kernel dump, pad poison) ask the same question, so without
    # this they would walk the wave's sequences four times to be told "nothing
    # is armed" four times. ``probe_debug is None`` with ``probe_debug_read``
    # set is the production answer: no probe key on any member.
    probe_debug: dict | None = None
    probe_debug_read: bool = False
    # segment id -> KV group id -> that group's Host pages for the segment
    segment_host_pages: dict[int, dict[int, list[int]]] = field(default_factory=dict)
    # Segments this wave LOADS (M3 Slice 3). Their Host pages are the
    # trajectory frontier's, committed and retained by an earlier wave, so
    # this wave allocates none, offloads none and publishes none; it only
    # reads them. Derived from the plan, never set by hand.
    materialized: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        chains: list[list[int]] = [[] for _ in self.prompts]
        for seg in self.plan.segments:  # DFS ids: every parent precedes its child
            for index in seg.members:
                chains[index].append(seg.segment_id)
        self.chains = [tuple(chain) for chain in chains]
        self.materialized = frozenset(self.plan.materialized_segments)

    def computed_segments(self) -> tuple:
        """The plan's segments a row of this wave computes, in plan order."""

        return tuple(
            seg for seg in self.plan.segments if not seg.materialized
        )

    def bind_materialized_host_pages(
        self, segment_id: int, pages_by_group: Mapping[int, Sequence[int]]
    ) -> None:
        """Name the resident Host pages a materialized segment loads from.

        The pages belong to the prefix cache (the frontier record names them),
        so they are recorded here as a read source and never as this wave's
        property: nothing registers them under a pseudo sequence, nothing
        offloads into them, nothing releases them.
        """

        if int(segment_id) not in self.materialized:
            raise RuntimeError(
                f"segment {segment_id} is not materialized in this wave, so "
                "its Host pages are this wave's to allocate"
            )
        expected = self.plan.segments[int(segment_id)].tokens // self.page_tokens
        bound: dict[int, list[int]] = {}
        for group_id, pages in pages_by_group.items():
            page_ids = [int(page) for page in pages]
            if len(page_ids) != expected:
                raise RuntimeError(
                    f"materialized segment {segment_id} was given "
                    f"{len(page_ids)} Host page(s) in group {group_id} for "
                    f"{self.plan.segments[int(segment_id)].tokens} tokens of "
                    f"{self.page_tokens}"
                )
            bound[int(group_id)] = page_ids
        if not bound:
            raise RuntimeError(
                f"materialized segment {segment_id} was given no Host pages"
            )
        self.segment_host_pages[int(segment_id)] = bound

    def chain_tokens(self, index: int) -> int:
        chain = self.chains[index]
        return self.plan.segments[chain[-1]].token_end if chain else 0

    def chain_host_pages(self, index: int) -> dict[int, list[int]]:
        return self._group_pages(self.chains[index])

    def segment_prefix_pages(self, segment_id: int) -> dict[int, list[int]]:
        """Host pages of a segment's ancestors followed by its own, per group."""

        chain = []
        seg = segment_id
        while seg is not None:
            chain.append(seg)
            seg = self.plan.segments[seg].parent_id
        return self._group_pages(reversed(chain))

    def _group_pages(self, segments) -> dict[int, list[int]]:
        """Concatenate the segments' Host pages inside each KV group."""

        pages: dict[int, list[int]] = {}
        for seg in segments:
            for group_id, group_pages in self.segment_host_pages[seg].items():
                pages.setdefault(group_id, []).extend(group_pages)
        return pages

    def row_host_targets(self, batch: ChunkBatch) -> tuple[list[int], list[int]]:
        """Host sequence id and destination start of every row of a chunk."""

        ids, starts = [], []
        for item in batch.items:
            if item.kind == "segment":
                ids.append(self.segment_sids[item.ref])
                starts.append(0)
            else:
                ids.append(self.global_ids[item.ref])
                starts.append(item.token_start)
        return ids, starts


def allocate_segment_host_pages(
    wave: PoolWave,
    worker_views_by_group: Mapping[int, object],
    next_sid: Callable[[], int],
) -> int:
    """Register one pseudo sequence per segment and give it exact pages.

    The same pseudo id and page count is registered in every KV group, so a
    member's chain stays in lockstep across the Host-KV regions.

    Returns the number of Host pages allocated per group.
    """

    page_tokens = wave.page_tokens
    # A materialized segment's pages are already resident and already
    # retained; this wave neither allocates nor owns them, so it registers no
    # pseudo sequence for them either.
    computed = wave.computed_segments()
    unbound = sorted(
        segment_id
        for segment_id in wave.materialized
        if segment_id not in wave.segment_host_pages
    )
    if unbound:
        raise RuntimeError(
            f"materialized segments {unbound} have no Host pages bound; "
            "bind_materialized_host_pages must run before allocation"
        )
    sids = [next_sid() for _ in computed]
    if any(sid >= 0 for sid in sids):
        raise ValueError("pool segment pseudo sequence ids must be negative")
    requests = [(sid, seg.tokens) for sid, seg in zip(sids, computed)]
    # Published before the first group allocates: a group that runs out of pages
    # half way leaves the caller every pseudo id it has to release.
    for sid, seg in zip(sids, computed):
        wave.segment_sids[seg.segment_id] = sid
    tables_by_group: dict[int, list] = {}
    for group_id, worker_view in sorted(worker_views_by_group.items()):
        worker_view.register_sequences(sids)
        worker_view.allocate_pages_for_sequences(requests)
        tables_by_group[int(group_id)] = worker_view.build_page_table(sids)
    total = 0
    for index, seg in enumerate(computed):
        expected = seg.tokens // page_tokens
        group_pages: dict[int, list[int]] = {}
        for group_id, tables in tables_by_group.items():
            pages = [int(page) for page in tables[index]]
            if len(pages) != expected:
                raise RuntimeError(
                    f"segment {seg.segment_id} got {len(pages)} Host pages in "
                    f"group {group_id} for {seg.tokens} tokens of {page_tokens}"
                )
            group_pages[group_id] = pages
        wave.segment_host_pages[seg.segment_id] = group_pages
        total += expected
    return total


def segment_mirror_payload(wave: PoolWave) -> dict:
    """The segments' pseudo sequence ids and Host pages as plain data.

    Host pages belong to the serve group's owner, so only it allocates them -
    but every rank of the group offloads the rows it computed, into the pages
    the owner allocated. This payload is what the other ranks need, picklable
    for one object gather.
    """

    missing = sorted(
        seg.segment_id
        for seg in wave.computed_segments()
        if seg.segment_id not in wave.segment_sids
        or seg.segment_id not in wave.segment_host_pages
    )
    if missing:
        raise RuntimeError(f"segments {missing} have no Host pages to mirror")
    unbound = sorted(
        segment_id
        for segment_id in wave.materialized
        if segment_id not in wave.segment_host_pages
    )
    if unbound:
        raise RuntimeError(
            f"materialized segments {unbound} have no Host pages to mirror"
        )
    return {
        "segment_sids": {
            int(segment_id): int(sid)
            for segment_id, sid in wave.segment_sids.items()
        },
        "segment_host_pages": {
            int(segment_id): _pages_payload(group_pages)
            for segment_id, group_pages in wave.segment_host_pages.items()
            if segment_id not in wave.materialized
        },
        # Read sources, not property: the loaded segments' resident pages, so
        # every rank of the group expands the SAME committed latent into its
        # own head shard of the pool. Mirrored in the wave's own pass, so a
        # page here is one the leader is holding right now.
        "materialized_host_pages": {
            int(segment_id): _pages_payload(
                wave.segment_host_pages[segment_id]
            )
            for segment_id in sorted(wave.materialized)
        },
    }


def _pages_payload(group_pages: Mapping[int, Sequence[int]]) -> dict:
    return {
        int(group_id): [int(page) for page in pages]
        for group_id, pages in group_pages.items()
    }


def install_mirrored_segments(
    wave: PoolWave,
    payload: Mapping,
    worker_views_by_group: Mapping[int, object],
) -> None:
    """Point this rank's segment pseudo sequences at the owner's Host pages.

    Registers the page table the owner already allocated, so this rank's row
    offloads land on the owner's pages. Nothing is allocated here, and so
    nothing may be released here either.

    A materialized segment has no pseudo sequence to point anywhere: no row of
    any rank writes it. Its resident pages are bound as the load source of
    this rank's expansion, which is all a follower needs to fill its own head
    shard of the pool from the same committed latent.
    """

    sids = {
        int(segment_id): int(sid)
        for segment_id, sid in payload["segment_sids"].items()
    }
    pages_by_segment = {
        int(segment_id): {
            int(group_id): [int(page) for page in pages]
            for group_id, pages in group_pages.items()
        }
        for segment_id, group_pages in payload["segment_host_pages"].items()
    }
    # The GPU pool is not built until the prefill phase; the page geometry
    # lives on the wave itself.
    page_tokens = wave.page_tokens
    computed = wave.computed_segments()
    for seg in computed:
        sid = sids.get(seg.segment_id)
        group_pages = pages_by_segment.get(seg.segment_id)
        if sid is None or group_pages is None:
            raise RuntimeError(
                f"segment {seg.segment_id} is missing from the mirror payload"
            )
        if sid >= 0:
            raise ValueError("pool segment pseudo sequence ids must be negative")
        expected = seg.tokens // page_tokens
        for group_id in worker_views_by_group:
            pages = group_pages.get(int(group_id))
            if pages is None or len(pages) != expected:
                raise RuntimeError(
                    f"segment {seg.segment_id} mirrors "
                    f"{0 if pages is None else len(pages)} Host pages in group "
                    f"{group_id} for {seg.tokens} tokens of {page_tokens}"
                )
    for group_id, worker_view in sorted(worker_views_by_group.items()):
        for seg in computed:
            worker_view.set_sequence_pages(
                sids[seg.segment_id],
                pages_by_segment[seg.segment_id][int(group_id)],
            )
    wave.segment_sids = sids
    wave.segment_host_pages = pages_by_segment
    loaded_pages = payload.get("materialized_host_pages") or {}
    for segment_id in sorted(wave.materialized):
        group_pages = loaded_pages.get(segment_id, loaded_pages.get(
            str(segment_id)
        ))
        if group_pages is None:
            raise RuntimeError(
                f"materialized segment {segment_id} is missing from the "
                "mirror payload; this rank cannot load what its leader holds"
            )
        wave.bind_materialized_host_pages(segment_id, group_pages)


def commit_pool_segments(
    wave: PoolWave,
    *,
    coordinator: object,
    worker_views_by_group: Mapping[int, object],
    namespace_digest: Sequence[int],
    publish_boundary_tokens: int,
    max_scan_nodes: int,
) -> list[int]:
    """Publish every segment, parents first; returns handles pinning each chain.

    A commit only protects the nodes it inserted, so right after each commit
    the segment's whole chain is attached: a node another rank published
    first cannot be evicted before a child commit or the member attach. The
    caller releases the returned handles once the members hold their own
    attachments (attach_pool_members).

    A materialized segment is not published here: it IS a published prefix,
    committed and retained by the wave that computed it, and its members
    attach to it like any resident chain. Publishing it again would insert
    nothing and claim pages this wave never allocated.
    """

    from batchgen.prefix_reuse.commit import (
        build_prefix_commit_request,
        retain_inserted_prefix_pages,
    )
    from batchgen.prefix_reuse.eviction import (
        commit_prefix_pages_with_capacity_retry,
    )

    handles: list[int] = []
    try:
        for seg in wave.computed_segments():
            tokens = wave.prompts[seg.members[0]][:seg.token_end]
            request = build_prefix_commit_request(
                namespace_digest=namespace_digest,
                token_ids=tokens,
                publish_boundary_tokens=publish_boundary_tokens,
                pages_by_group=wave.segment_prefix_pages(seg.segment_id),
            )
            if request is None or request.commit_tokens != seg.token_end:
                raise RuntimeError(
                    f"segment {seg.segment_id} ending at {seg.token_end} is not on "
                    f"the publish boundary {publish_boundary_tokens}"
                )
            outcome = commit_prefix_pages_with_capacity_retry(
                request=request,
                coordinator=coordinator,
                worker_views_by_group=worker_views_by_group,
                max_scan_nodes=max_scan_nodes,
            )
            result = outcome.commit_result
            commit_handle = int(result.active_attachment_handle)
            if result.inserted_nodes and not commit_handle:
                raise RuntimeError("segment commit inserted pages without protection")
            try:
                retain_inserted_prefix_pages(
                    commit_result=result,
                    request=request,
                    worker_views_by_group=worker_views_by_group,
                    sequence_id=wave.segment_sids[seg.segment_id],
                )
                handles.append(_attach_chain(coordinator, namespace_digest, tokens))
            finally:
                if commit_handle:
                    coordinator.release_attachment(commit_handle)
    except Exception:
        for handle in handles:
            coordinator.release_attachment(handle)
        raise
    return handles


def _attach_chain(coordinator: object, namespace_digest, tokens) -> int:
    """Attach exactly ``tokens`` of resident prefix; raise if any is missing."""

    result = coordinator.lookup_and_attach([int(v) for v in namespace_digest], list(tokens))
    cached = int(result.common_cached_tokens)
    handle = int(result.attachment_handle)
    if cached != len(tokens) or not handle:
        if handle:
            coordinator.release_attachment(handle)
        raise RuntimeError(
            f"prefix chain of {len(tokens)} tokens has only {cached} resident"
        )
    return handle


def attach_pool_members(
    wave: PoolWave, *, coordinator: object, namespace_digest: Sequence[int]
) -> dict[int, int]:
    """Attach every member to its resident chain; returns wave index -> handle."""

    handles: dict[int, int] = {}
    try:
        for index, chain in enumerate(wave.chains):
            if chain:
                end = wave.chain_tokens(index)
                handles[index] = _attach_chain(
                    coordinator, namespace_digest, wave.prompts[index][:end]
                )
    except Exception:
        for handle in handles.values():
            coordinator.release_attachment(handle)
        raise
    return handles
