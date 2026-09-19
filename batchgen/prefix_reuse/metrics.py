"""Structured, low-volume prefix-cache resource metrics."""

from __future__ import annotations

import json
import logging
from typing import Any


PREFIX_CACHE_METRICS_SCHEMA_VERSION = 1


def emit_prefix_cache_metric(*, phase: str, rank: int, **fields: Any) -> None:
    """Emit one compact record for benchmark evidence collection."""

    payload = {
        "schema_version": PREFIX_CACHE_METRICS_SCHEMA_VERSION,
        "component": "prefix_cache",
        "phase": str(phase),
        "rank": int(rank),
        **fields,
    }
    logging.info(
        "[METRICS] %s",
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
    )


def host_resource_fields(*, coordinator: object, worker_view: object) -> dict[str, int]:
    """Snapshot coordinator metadata and physical Host-KV page usage."""

    prefix = coordinator.get_stats()
    host = worker_view.get_stats()
    return {
        "resident_nodes": int(prefix.resident_nodes),
        "active_attachments": int(prefix.active_attachments),
        "pending_load_entries": int(prefix.pending_load_entries),
        "pending_load_refs": int(prefix.pending_load_refs),
        "used_group_entries": int(prefix.used_group_entries),
        "used_page_handles": int(prefix.used_page_handles),
        "lookup_hits": int(prefix.lookup_hits),
        "lookup_misses": int(prefix.lookup_misses),
        "evicted_nodes_total": int(prefix.evicted_nodes),
        "eviction_protected_skips_total": int(
            prefix.eviction_protected_skips
        ),
        "host_kv_total_pages": int(host.num_total_pages),
        "host_kv_free_pages": int(host.num_free_pages),
        "host_kv_used_pages": int(host.num_used_pages),
    }
