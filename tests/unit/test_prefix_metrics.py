import json
import logging
from types import SimpleNamespace

from batchgen.prefix_reuse.metrics import (
    emit_prefix_cache_metric,
    host_resource_fields,
)


def test_prefix_metric_is_compact_versioned_json(caplog):
    with caplog.at_level(logging.INFO):
        emit_prefix_cache_metric(
            phase="prefix_h2d",
            rank=3,
            host_pages=7,
        )

    record = next(
        item.message for item in caplog.records if item.message.startswith("[METRICS]")
    )
    payload = json.loads(record.removeprefix("[METRICS] "))
    assert payload == {
        "component": "prefix_cache",
        "host_pages": 7,
        "phase": "prefix_h2d",
        "rank": 3,
        "schema_version": 1,
    }


def test_host_resource_fields_are_scalar_snapshots():
    coordinator = SimpleNamespace(
        get_stats=lambda: SimpleNamespace(
            resident_nodes=1,
            active_attachments=2,
            pending_load_entries=3,
            pending_load_refs=4,
            used_group_entries=5,
            used_page_handles=6,
            lookup_hits=7,
            lookup_misses=8,
            evicted_nodes=9,
            eviction_protected_skips=10,
        )
    )
    worker = SimpleNamespace(
        get_stats=lambda: SimpleNamespace(
            num_total_pages=100,
            num_free_pages=60,
            num_used_pages=40,
        )
    )

    fields = host_resource_fields(
        coordinator=coordinator,
        worker_view=worker,
    )

    assert fields["used_page_handles"] == 6
    assert fields["evicted_nodes_total"] == 9
    assert fields["host_kv_used_pages"] == 40
    assert all(isinstance(value, int) for value in fields.values())
