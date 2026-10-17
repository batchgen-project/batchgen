"""Coordinate prefix metadata eviction with physical Host-page release."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from batchgen.prefix_reuse.commit import release_evicted_prefix_pages


_CAPACITY_ERRORS = (
    "Host prefix cache node table is full",
    "Host prefix cache group entry table is full",
    "Host prefix cache page handle arena is full",
)


@dataclass(frozen=True)
class PrefixCommitRetryResult:
    commit_result: object
    eviction_result: object | None = None


def commit_prefix_pages_with_capacity_retry(
    *,
    request: object,
    coordinator: object,
    worker_views_by_group: Mapping[int, object],
    max_scan_nodes: int,
    frontier_hook: Callable[[Mapping[int, list[int]]], object] | None = None,
) -> PrefixCommitRetryResult:
    """Commit once, evict on metadata pressure, then retry exactly once."""

    try:
        return PrefixCommitRetryResult(request.commit(coordinator))
    except RuntimeError as exc:
        if not any(marker in str(exc) for marker in _CAPACITY_ERRORS):
            raise
        nodes, entries, handles = request.capacity_requirements()
        eviction = coordinator.evict_until_free(
            int(nodes),
            int(entries),
            int(handles),
            int(max_scan_nodes),
        )
        release_evicted_prefix_pages(
            eviction_result=eviction,
            worker_views_by_group=worker_views_by_group,
            frontier_hook=frontier_hook,
        )
        return PrefixCommitRetryResult(
            request.commit(coordinator),
            eviction_result=eviction,
        )


def evict_prefix_pages_for_host_allocation(
    *,
    core_engine_module: object,
    coordinator: object,
    worker_views_by_group: Mapping[int, object],
    group_id: int,
    page_deficit: int,
    max_scan_nodes: int,
    frontier_hook: Callable[[Mapping[int, list[int]]], object] | None = None,
) -> int:
    """Release at least ``page_deficit`` unprotected resident Host pages."""

    released_pages = reclaim_prefix_pages_for_host_admission(
        core_engine_module=core_engine_module,
        coordinator=coordinator,
        worker_views_by_group=worker_views_by_group,
        group_id=group_id,
        page_target=page_deficit,
        max_scan_nodes=max_scan_nodes,
        frontier_hook=frontier_hook,
    )
    if released_pages < int(page_deficit):
        raise RuntimeError(
            "prefix cache eviction could not release enough Host KV pages: "
            f"needed={page_deficit}, released={released_pages}"
        )
    return released_pages


def reclaim_prefix_pages_for_host_admission(
    *,
    core_engine_module: object,
    coordinator: object,
    worker_views_by_group: Mapping[int, object],
    group_id: int,
    page_target: int,
    max_scan_nodes: int,
    frontier_hook: Callable[[Mapping[int, list[int]]], object] | None = None,
) -> int:
    """Reclaim up to a target; live attachments may leave a shortfall."""

    target = int(page_target)
    if target <= 0:
        return 0
    requirement = core_engine_module.GroupPageRequirement()
    requirement.group_id = int(group_id)
    requirement.min_pages = target
    eviction = coordinator.evict_until_releasable_pages(
        [requirement], int(max_scan_nodes)
    )
    released = release_evicted_prefix_pages(
        eviction_result=eviction,
        worker_views_by_group=worker_views_by_group,
        frontier_hook=frontier_hook,
    )
    return int(released.get(int(group_id), 0))


class EvictedPageRecorder:
    """Frontier hook that records every reclaimed page, then forwards it.

    Admission reclaim runs on ONE rank per node (the Host-KV region is shared
    per node), but trajectory-frontier records live on whichever rank owns the
    sequence's Host KV -- at G=1 that is any rank of the node. The recorder
    keeps the exact ``{group_id: page_ids}`` map ``release_evicted_prefix_pages``
    hands its hook, so the worker can deliver it to every rank of the node
    (``retire_node_frontiers``). The wrapped hook (``None`` when this rank has
    no committer) still runs here, inside the release, as before.
    """

    def __init__(
        self,
        frontier_hook: Callable[[Mapping[int, list[int]]], object] | None = None,
    ) -> None:
        self._frontier_hook = frontier_hook
        self.pages_by_group: dict[int, list[int]] = {}

    def __call__(self, released_page_ids_by_group: Mapping[int, list[int]]):
        for group_id, page_ids in released_page_ids_by_group.items():
            self.pages_by_group.setdefault(int(group_id), []).extend(
                int(page_id) for page_id in page_ids
            )
        if self._frontier_hook is not None:
            return self._frontier_hook(released_page_ids_by_group)
        return None

    @property
    def page_count(self) -> int:
        return sum(len(pages) for pages in self.pages_by_group.values())


def retire_node_frontiers(
    *,
    gathered: Sequence[tuple[int, Mapping[int, list[int]] | None]],
    node_id: int,
    frontier_hook: Callable[[Mapping[int, list[int]]], object] | None,
) -> int:
    """Run this rank's frontier hook on the pages its node leader reclaimed.

    ``gathered`` is one ``(node_id, pages_by_group or None)`` entry per rank;
    only node leaders report a map. Exactly one leader per node may report: a
    second map for the same node means two ranks reclaimed one shared pool.
    Returns the number of pages handed to the hook (0 without a hook).
    """

    maps = [
        pages for node, pages in gathered
        if int(node) == int(node_id) and pages is not None
    ]
    if len(maps) > 1:
        raise RuntimeError(
            f"node {node_id} reported {len(maps)} admission reclaims; exactly "
            "one rank per node may reclaim its shared Host pool"
        )
    if not maps or frontier_hook is None:
        return 0
    pages_by_group = {
        int(group_id): [int(page) for page in pages]
        for group_id, pages in maps[0].items()
        if pages
    }
    if not pages_by_group:
        return 0
    frontier_hook(pages_by_group)
    return sum(len(pages) for pages in pages_by_group.values())
