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

    def __post_init__(self) -> None:
        chains: list[list[int]] = [[] for _ in self.prompts]
        for seg in self.plan.segments:  # DFS ids: every parent precedes its child
            for index in seg.members:
                chains[index].append(seg.segment_id)
        self.chains = [tuple(chain) for chain in chains]

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
    sids = [next_sid() for _ in wave.plan.segments]
    if any(sid >= 0 for sid in sids):
        raise ValueError("pool segment pseudo sequence ids must be negative")
    requests = [(sid, seg.tokens) for sid, seg in zip(sids, wave.plan.segments)]
    # Published before the first group allocates: a group that runs out of pages
    # half way leaves the caller every pseudo id it has to release.
    for sid, seg in zip(sids, wave.plan.segments):
        wave.segment_sids[seg.segment_id] = sid
    tables_by_group: dict[int, list] = {}
    for group_id, worker_view in sorted(worker_views_by_group.items()):
        worker_view.register_sequences(sids)
        worker_view.allocate_pages_for_sequences(requests)
        tables_by_group[int(group_id)] = worker_view.build_page_table(sids)
    total = 0
    for index, seg in enumerate(wave.plan.segments):
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
        for seg in wave.plan.segments
        if seg.segment_id not in wave.segment_sids
        or seg.segment_id not in wave.segment_host_pages
    )
    if missing:
        raise RuntimeError(f"segments {missing} have no Host pages to mirror")
    return {
        "segment_sids": {
            int(segment_id): int(sid)
            for segment_id, sid in wave.segment_sids.items()
        },
        "segment_host_pages": {
            int(segment_id): {
                int(group_id): [int(page) for page in pages]
                for group_id, pages in group_pages.items()
            }
            for segment_id, group_pages in wave.segment_host_pages.items()
        },
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
    for seg in wave.plan.segments:
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
        for seg in wave.plan.segments:
            worker_view.set_sequence_pages(
                sids[seg.segment_id],
                pages_by_segment[seg.segment_id][int(group_id)],
            )
    wave.segment_sids = sids
    wave.segment_host_pages = pages_by_segment


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
        for seg in wave.plan.segments:
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
