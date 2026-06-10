from types import SimpleNamespace

from batchgen.prefix_reuse.admin import (
    clear_host_prefix_cache,
    host_kv_views_by_prefix_group,
)


class _Coordinator:
    def __init__(self):
        self._stats = [
            _stats(resident_nodes=3, used_group_entries=4),
            _stats(resident_nodes=0, evicted_nodes=3),
        ]
        self.clear_calls = 0

    def get_stats(self):
        return self._stats.pop(0)

    def clear_unprotected(self):
        self.clear_calls += 1
        return SimpleNamespace(
            evicted_nodes=3,
            protected_nodes=0,
            freed_group_entries=4,
            freed_page_handles=5,
            evicted_group_pages=[
                _group_pages(0, [10, 11, 11]),
                _group_pages(1, [20, 21]),
            ],
        )


class _PinCoordinator:
    def __init__(self):
        self.lookup_calls = []
        self.released_handles = []
        self._next_handle = 100

    def lookup_and_attach(self, namespace_digest, token_ids):
        self.lookup_calls.append((list(namespace_digest), list(token_ids)))
        if not token_ids or token_ids[0] < 0:
            return SimpleNamespace(
                attachment_handle=0,
                common_cached_tokens=0,
            )
        self._next_handle += 1
        return SimpleNamespace(
            attachment_handle=self._next_handle,
            common_cached_tokens=len(token_ids),
        )

    def release_attachment(self, handle):
        self.released_handles.append(int(handle))


class _HostKV:
    def __init__(self):
        self.released_pages = []

    def release_resident_pages(self, page_ids):
        self.released_pages.append(list(page_ids))


class _GroupedHostKV:
    def __init__(self, views):
        self._views = views

    def views_by_group(self):
        return self._views


def _stats(**overrides):
    values = {
        "resident_nodes": 0,
        "active_attachments": 0,
        "pending_load_entries": 0,
        "pending_load_refs": 0,
        "used_group_entries": 0,
        "used_page_handles": 0,
        "lookup_hits": 0,
        "lookup_misses": 0,
        "evicted_nodes": 0,
        "eviction_protected_skips": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _group_pages(group_id, pages):
    return SimpleNamespace(
        group_id=group_id,
        pages=[SimpleNamespace(page_id=page_id) for page_id in pages],
    )


def test_clear_host_prefix_cache_releases_evicted_pages_by_group():
    primary = _HostKV()
    auxiliary = _HostKV()
    coordinator = _Coordinator()

    result = clear_host_prefix_cache(
        coordinator=coordinator,
        host_kv_views_by_group={0: primary, 1: auxiliary},
    )

    assert coordinator.clear_calls == 1
    assert primary.released_pages == [[10, 11]]
    assert auxiliary.released_pages == [[20, 21]]
    assert result["cleared_all"] is True
    assert result["stats_before"]["resident_nodes"] == 3
    assert result["stats_after"]["resident_nodes"] == 0
    assert result["eviction"]["evicted_nodes"] == 3
    assert result["eviction"]["evicted_pages_by_group"] == {0: 3, 1: 2}
    assert result["eviction"]["released_pages_by_group"] == {0: 2, 1: 2}


def test_host_kv_views_by_prefix_group_uses_grouped_coordinator_first():
    views = {0: _HostKV(), 3: _HostKV()}
    grouped = _GroupedHostKV(views)

    assert (
        host_kv_views_by_prefix_group(
            primary_host_kv=grouped,
            auxiliary_host_kv=_HostKV(),
        )
        == views
    )


def test_host_kv_views_by_prefix_group_maps_primary_and_auxiliary():
    primary = _HostKV()
    auxiliary = _HostKV()

    assert host_kv_views_by_prefix_group(
        primary_host_kv=primary,
        auxiliary_host_kv=auxiliary,
    ) == {0: primary, 1: auxiliary}


def test_pin_host_prefix_cache_holds_only_lookup_hits():
    from batchgen.prefix_reuse.admin import pin_host_prefix_cache

    coordinator = _PinCoordinator()

    result = pin_host_prefix_cache(
        coordinator=coordinator,
        namespace_digest=(1, 2, 3, 4),
        token_id_batches=[[10, 11, 12], [-1, 2], [20]],
    )

    assert result["requested"] == 3
    assert result["pinned"] == 2
    assert result["missed"] == 1
    assert result["cached_tokens"] == 4
    assert result["cached_tokens_by_request"] == [3, 1]
    assert result["attachment_handles"] == [101, 102]
    assert coordinator.lookup_calls == [
        ([1, 2, 3, 4], [10, 11, 12]),
        ([1, 2, 3, 4], [-1, 2]),
        ([1, 2, 3, 4], [20]),
    ]


def test_unpin_host_prefix_cache_releases_handles():
    from batchgen.prefix_reuse.admin import unpin_host_prefix_cache

    coordinator = _PinCoordinator()

    result = unpin_host_prefix_cache(
        coordinator=coordinator,
        attachment_handles=[101, 102],
    )

    assert result == {"status": "success", "released": 2}
    assert coordinator.released_handles == [101, 102]
