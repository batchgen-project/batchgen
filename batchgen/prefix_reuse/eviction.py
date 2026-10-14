"""Coordinate prefix metadata eviction with physical Host-page release."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

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
) -> int:
    """Release at least ``page_deficit`` unprotected resident Host pages."""

    released_pages = reclaim_prefix_pages_for_host_admission(
        core_engine_module=core_engine_module,
        coordinator=coordinator,
        worker_views_by_group=worker_views_by_group,
        group_id=group_id,
        page_target=page_deficit,
        max_scan_nodes=max_scan_nodes,
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
    )
    return int(released.get(int(group_id), 0))
