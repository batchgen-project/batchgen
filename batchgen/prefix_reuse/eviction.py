"""Eviction helpers for Host prefix-cache integration.

The coordinator owns prefix metadata and chooses eviction victims. Host KV
worker views own physical Host pages, so released page handles must be routed
back to the worker view for the matching prefix group.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable, Iterator

from batchgen.prefix_reuse.commit import PrefixCommitRequest


_CAPACITY_ERROR_MARKERS = (
    "Host prefix cache node table is full",
    "Host prefix cache group entry table is full",
    "Host prefix cache page handle arena is full",
)


@dataclass(frozen=True)
class PrefixCommitRetryResult:
    commit_result: object
    eviction_result: object | None = None
    released_pages_by_group: dict[int, int] | None = None


@dataclass(frozen=True)
class PrefixAllocationEvictionResult:
    eviction_result: object | None
    released_pages_by_group: dict[int, int]


def commit_prefix_pages_with_capacity_retry(
    *,
    request: PrefixCommitRequest,
    coordinator: object,
    worker_views_by_group: Mapping[int, object],
    max_scan_nodes: int = 0,
) -> PrefixCommitRetryResult:
    """Commit prefix pages, evicting and retrying once on metadata pressure."""

    try:
        return PrefixCommitRetryResult(commit_result=request.commit(coordinator))
    except RuntimeError as exc:
        if not _is_capacity_error(exc):
            raise
        eviction_result = _evict_for_request_capacity(
            request=request,
            coordinator=coordinator,
            max_scan_nodes=max_scan_nodes,
        )
        released = release_evicted_prefix_pages(
            eviction_result=eviction_result,
            worker_views_by_group=worker_views_by_group,
        )
        return PrefixCommitRetryResult(
            commit_result=request.commit(coordinator),
            eviction_result=eviction_result,
            released_pages_by_group=released,
        )


def evict_prefix_pages_for_host_allocation(
    *,
    core_engine_module: object,
    coordinator: object,
    worker_views_by_group: Mapping[int, object],
    page_deficit_by_group: Mapping[int, int],
    max_scan_nodes: int = 0,
) -> PrefixAllocationEvictionResult:
    """Evict common prefix nodes until enough physical Host pages are released.

    The input is per-group pressure, but the coordinator still evicts whole
    prefix nodes. The returned pages are filtered by the coordinator so only
    pages no longer referenced by any resident prefix node are released.
    """

    requirements = []
    for group_id, deficit in sorted(page_deficit_by_group.items()):
        deficit = int(deficit)
        if deficit <= 0:
            continue
        requirement = core_engine_module.GroupPageRequirement()
        requirement.group_id = int(group_id)
        requirement.min_pages = deficit
        requirements.append(requirement)
    if not requirements:
        return PrefixAllocationEvictionResult(
            eviction_result=None,
            released_pages_by_group={},
        )

    eviction_result = coordinator.evict_until_releasable_pages(
        requirements,
        int(max_scan_nodes),
    )
    released = release_evicted_prefix_pages(
        eviction_result=eviction_result,
        worker_views_by_group=worker_views_by_group,
    )

    missing = {
        int(requirement.group_id): int(requirement.min_pages)
        - int(released.get(int(requirement.group_id), 0))
        for requirement in requirements
        if int(released.get(int(requirement.group_id), 0))
        < int(requirement.min_pages)
    }
    if missing:
        raise RuntimeError(
            "prefix cache eviction could not release enough Host KV pages "
            f"for allocation: missing={missing}, released={released}, "
            f"evicted_nodes={int(eviction_result.evicted_nodes)}, "
            f"protected_nodes={int(eviction_result.protected_nodes)}"
        )

    return PrefixAllocationEvictionResult(
        eviction_result=eviction_result,
        released_pages_by_group=released,
    )


def release_evicted_prefix_pages(
    *,
    eviction_result: object,
    worker_views_by_group: Mapping[int, object],
) -> dict[int, int]:
    """Release coordinator-evicted physical Host pages by KV group."""

    pages_by_group: dict[int, list[int]] = {}
    seen_by_group: dict[int, set[int]] = {}
    for group_pages in eviction_result.evicted_group_pages:
        group_id = int(group_pages.group_id)
        seen = seen_by_group.setdefault(group_id, set())
        pages = pages_by_group.setdefault(group_id, [])
        for page_id in _page_ids(group_pages.pages):
            if page_id in seen:
                continue
            seen.add(page_id)
            pages.append(page_id)

    released: dict[int, int] = {}
    for group_id, page_ids in pages_by_group.items():
        worker_view = worker_views_by_group.get(group_id)
        if worker_view is None:
            raise RuntimeError(
                f"missing Host KV worker view for evicted prefix group {group_id}"
            )
        if not page_ids:
            continue
        worker_view.release_resident_pages(page_ids)
        released[group_id] = len(page_ids)
    return released


def _evict_for_request_capacity(
    *,
    request: PrefixCommitRequest,
    coordinator: object,
    max_scan_nodes: int,
) -> object:
    node_count, group_entry_count, page_handle_count = (
        request.capacity_requirements()
    )
    return coordinator.evict_until_free(
        int(node_count),
        int(group_entry_count),
        int(page_handle_count),
        int(max_scan_nodes),
    )


def _is_capacity_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return any(marker in message for marker in _CAPACITY_ERROR_MARKERS)


def _page_ids(pages: Iterable[object]) -> Iterator[int]:
    for page in pages:
        yield int(getattr(page, "page_id", page))
