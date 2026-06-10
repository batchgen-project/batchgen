"""Admin helpers for managing the Host prefix cache."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from batchgen.prefix_reuse.eviction import release_evicted_prefix_pages

_STATS_FIELDS = (
    "resident_nodes",
    "active_attachments",
    "pending_load_entries",
    "pending_load_refs",
    "used_group_entries",
    "used_page_handles",
    "lookup_hits",
    "lookup_misses",
    "evicted_nodes",
    "eviction_protected_skips",
)

_EVICTION_FIELDS = (
    "evicted_nodes",
    "protected_nodes",
    "freed_group_entries",
    "freed_page_handles",
)


def clear_host_prefix_cache(
    *,
    coordinator: Any,
    host_kv_views_by_group: Mapping[int, Any],
) -> dict[str, Any]:
    """Clear unprotected prefix-cache entries and release their Host KV pages."""

    stats_before = _object_int_fields(coordinator.get_stats(), _STATS_FIELDS)
    eviction_result = coordinator.clear_unprotected()
    released_pages_by_group = release_evicted_prefix_pages(
        eviction_result=eviction_result,
        worker_views_by_group=host_kv_views_by_group,
    )
    stats_after = _object_int_fields(coordinator.get_stats(), _STATS_FIELDS)

    return {
        "status": "success",
        "cleared_all": stats_after["resident_nodes"] == 0,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "eviction": {
            **_object_int_fields(eviction_result, _EVICTION_FIELDS),
            "evicted_pages_by_group": _evicted_page_counts_by_group(
                eviction_result
            ),
            "released_pages_by_group": {
                int(group_id): int(count)
                for group_id, count in sorted(released_pages_by_group.items())
            },
        },
    }


def pin_host_prefix_cache(
    *,
    coordinator: Any,
    namespace_digest: tuple[int, int, int, int],
    token_id_batches: list[list[int]],
) -> dict[str, Any]:
    """Pin existing prefix-cache entries by holding lookup attachments."""

    handles: list[int] = []
    cached_tokens: list[int] = []
    missed_count = 0
    for token_ids in token_id_batches:
        result = coordinator.lookup_and_attach(
            list(namespace_digest),
            [int(token_id) for token_id in token_ids],
        )
        handle = int(result.attachment_handle)
        if handle == 0:
            missed_count += 1
            continue
        handles.append(handle)
        cached_tokens.append(int(result.common_cached_tokens))

    return {
        "status": "success",
        "requested": len(token_id_batches),
        "pinned": len(handles),
        "missed": missed_count,
        "cached_tokens": sum(cached_tokens),
        "cached_tokens_by_request": cached_tokens,
        "attachment_handles": handles,
    }


def unpin_host_prefix_cache(
    *,
    coordinator: Any,
    attachment_handles: list[int],
) -> dict[str, Any]:
    """Release previously pinned prefix-cache lookup attachments."""

    released = 0
    for handle in attachment_handles:
        coordinator.release_attachment(int(handle))
        released += 1
    return {
        "status": "success",
        "released": released,
    }


def host_kv_views_by_prefix_group(
    *,
    primary_host_kv: Any,
    auxiliary_host_kv: Any | None,
) -> dict[int, Any]:
    """Build the prefix-cache group -> Host KV owner map used for page release."""

    views_by_group = getattr(primary_host_kv, "views_by_group", None)
    if views_by_group is not None:
        return {
            int(group_id): view for group_id, view in views_by_group().items()
        }

    result = {0: primary_host_kv}
    if auxiliary_host_kv is not None:
        result[1] = auxiliary_host_kv
    return result


def _object_int_fields(obj: Any, fields: tuple[str, ...]) -> dict[str, int]:
    return {field: int(getattr(obj, field)) for field in fields}


def _evicted_page_counts_by_group(eviction_result: Any) -> dict[int, int]:
    page_counts: dict[int, int] = {}
    for group_pages in eviction_result.evicted_group_pages:
        group_id = int(group_pages.group_id)
        page_counts[group_id] = page_counts.get(group_id, 0) + len(
            group_pages.pages
        )
    return dict(sorted(page_counts.items()))
