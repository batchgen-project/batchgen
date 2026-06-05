import ctypes
import errno
import random
import string

from batchgen.models.engine_loader import core_engine as bg


_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def _random_shm_name() -> str:
    suffix = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=10)
    )
    return f"/batchgen_prefix_cache_{suffix}"


def _shm_unlink(name: str) -> None:
    result = _LIBC.shm_unlink(name.encode("utf-8"))
    if result != 0:
        err = ctypes.get_errno()
        if err != errno.ENOENT:
            raise OSError(err, f"shm_unlink({name}) failed")


def _group_spec(group_id: int, raw_page_tokens: int):
    spec = bg.HostKVGroupSpec()
    spec.group_id = group_id
    spec.semantic = bg.HostKVGroupSemantic.FULL_KV
    spec.required_for_reuse = True
    spec.raw_page_tokens = raw_page_tokens
    spec.compression_ratio = 1
    return spec


def _page(page_id: int):
    handle = bg.HostPageHandle()
    handle.page_id = page_id
    return handle


def _group_pages(group_id: int, pages):
    group = bg.GroupCommitPages()
    group.group_id = group_id
    group.pages = list(pages)
    return group


def _requirement(group_id: int, min_pages: int):
    requirement = bg.GroupPageRequirement()
    requirement.group_id = group_id
    requirement.min_pages = min_pages
    return requirement


def _config(shm_name: str):
    config = bg.HostPrefixCacheConfig()
    config.shm_name = shm_name
    config.group_specs = [_group_spec(0, 4), _group_spec(1, 8)]
    config.max_nodes = 16
    config.max_group_entries = 32
    config.max_page_handles = 128
    config.max_attachments = 16
    return config


def _small_config(shm_name: str):
    config = _config(shm_name)
    config.max_nodes = 2
    config.max_group_entries = 8
    config.max_page_handles = 32
    config.max_attachments = 4
    return config


def _single_node_config(shm_name: str):
    config = _config(shm_name)
    config.max_nodes = 1
    config.max_group_entries = 4
    config.max_page_handles = 16
    config.max_attachments = 2
    return config


def test_host_prefix_cache_lookup_attach_release():
    shm_name = _random_shm_name()
    namespace = [11, 22, 33, 44]
    token_ids = list(range(16))
    try:
        coordinator = bg.HostPrefixCacheCoordinator(_config(shm_name))
        coordinator.initialize(True)

        assert coordinator.hash_block_tokens == 4
        assert coordinator.commit_boundary_tokens == 8

        commit = coordinator.commit_prefix_pages(
            namespace,
            token_ids,
            16,
            [
                _group_pages(0, [_page(idx) for idx in range(4)]),
                _group_pages(1, [_page(idx) for idx in range(2)]),
            ],
        )
        assert commit.committed_tokens == 16
        assert commit.inserted_nodes == 2
        assert commit.existing_nodes == 0

        estimated = coordinator.estimate_lookup(namespace, token_ids[:12])
        assert estimated.common_cached_tokens == 8
        assert estimated.attachment_handle == 0
        assert [span.group_id for span in estimated.materialization_spans] == [
            0,
            1,
        ]
        assert coordinator.get_stats().active_attachments == 0

        attached = coordinator.lookup_and_attach(namespace, token_ids[:12])
        assert attached.common_cached_tokens == 8
        assert attached.attachment_handle != 0
        assert [span.group_id for span in attached.materialization_spans] == [
            0,
            1,
        ]
        assert [len(span.pages) for span in attached.materialization_spans] == [
            2,
            1,
        ]

        stats = coordinator.get_stats()
        assert stats.resident_nodes == 2
        assert stats.active_attachments == 1
        assert stats.lookup_hits == 1
        assert stats.lookup_misses == 0

        coordinator.release_attachment(attached.attachment_handle)
        assert coordinator.get_stats().active_attachments == 0

        full = coordinator.lookup_and_attach(namespace, token_ids)
        assert full.common_cached_tokens == 16
        assert [
            [page.page_id for page in span.pages]
            for span in full.materialization_spans
        ] == [
            [0, 1, 2, 3],
            [0, 1],
        ]
        coordinator.release_attachment(full.attachment_handle)
    finally:
        _shm_unlink(shm_name)


def test_host_prefix_cache_evicts_lru_and_preserves_active_attachment():
    shm_name = _random_shm_name()
    namespace = [101, 202, 303, 404]
    token_ids = list(range(16))
    try:
        coordinator = bg.HostPrefixCacheCoordinator(_small_config(shm_name))
        coordinator.initialize(True)
        coordinator.commit_prefix_pages(
            namespace,
            token_ids,
            16,
            [
                _group_pages(0, [_page(idx) for idx in range(4)]),
                _group_pages(1, [_page(idx) for idx in range(2)]),
            ],
        )

        active = coordinator.lookup_and_attach(namespace, token_ids)
        assert active.common_cached_tokens == 16

        evicted = coordinator.evict_until_free(2, 0, 0, 2)
        assert evicted.evicted_nodes == 0
        assert evicted.protected_nodes == 2
        assert evicted.freed_group_entries == 0
        assert evicted.freed_page_handles == 0
        assert len(evicted.evicted_group_pages) == 0

        miss = coordinator.estimate_lookup(namespace, token_ids[:8])
        hit = coordinator.estimate_lookup(namespace, token_ids)
        assert miss.common_cached_tokens == 8
        assert hit.common_cached_tokens == 16
        stats = coordinator.get_stats()
        assert stats.resident_nodes == 2
        assert stats.used_group_entries == 4
        assert stats.used_page_handles == 6

        coordinator.release_attachment(active.attachment_handle)
        evicted = coordinator.evict_until_free(2, 0, 0, 2)
        assert evicted.evicted_nodes == 2
        assert [pages.group_id for pages in evicted.evicted_group_pages] == [
            0,
            1,
        ]
        assert [len(pages.pages) for pages in evicted.evicted_group_pages] == [
            4,
            2,
        ]
        assert coordinator.get_stats().resident_nodes == 0
    finally:
        _shm_unlink(shm_name)


def test_host_prefix_cache_evicts_common_nodes_until_pages_releasable():
    shm_name = _random_shm_name()
    namespace = [301, 302, 303, 304]
    token_ids = list(range(16))
    try:
        coordinator = bg.HostPrefixCacheCoordinator(_small_config(shm_name))
        coordinator.initialize(True)
        coordinator.commit_prefix_pages(
            namespace,
            token_ids,
            16,
            [
                _group_pages(0, [_page(idx) for idx in range(4)]),
                _group_pages(1, [_page(idx) for idx in range(2)]),
            ],
        )

        evicted = coordinator.evict_until_releasable_pages(
            [_requirement(0, 1)],
            0,
        )

        # Nodes store only their own block interval, so the first LRU node can
        # release physical pages immediately.
        assert evicted.evicted_nodes == 1
        assert evicted.protected_nodes == 0
        assert [pages.group_id for pages in evicted.evicted_group_pages] == [
            0,
            1,
        ]
        assert [
            [page.page_id for page in pages.pages]
            for pages in evicted.evicted_group_pages
        ] == [
            [0, 1],
            [0],
        ]
        assert coordinator.get_stats().resident_nodes == 1
    finally:
        _shm_unlink(shm_name)


def test_host_prefix_cache_clear_skips_active_entries():
    shm_name = _random_shm_name()
    namespace = [505, 606, 707, 808]
    token_ids = list(range(16))
    try:
        coordinator = bg.HostPrefixCacheCoordinator(_small_config(shm_name))
        coordinator.initialize(True)
        coordinator.commit_prefix_pages(
            namespace,
            token_ids,
            16,
            [
                _group_pages(0, [_page(idx) for idx in range(4)]),
                _group_pages(1, [_page(idx) for idx in range(2)]),
            ],
        )

        active = coordinator.lookup_and_attach(namespace, token_ids)
        clear = coordinator.clear_unprotected()
        assert clear.evicted_nodes == 0
        assert clear.protected_nodes == 2
        assert coordinator.get_stats().resident_nodes == 2
        miss = coordinator.estimate_lookup(namespace, token_ids[:8])
        hit = coordinator.estimate_lookup(namespace, token_ids)
        assert miss.common_cached_tokens == 8
        assert hit.common_cached_tokens == 16

        coordinator.release_attachment(active.attachment_handle)
        clear = coordinator.clear_unprotected()
        assert clear.evicted_nodes == 2
        assert clear.protected_nodes == 0
        assert coordinator.get_stats().resident_nodes == 0
    finally:
        _shm_unlink(shm_name)


def test_host_prefix_cache_pending_load_protects_after_release():
    shm_name = _random_shm_name()
    namespace = [909, 808, 707, 606]
    token_ids = list(range(16))
    try:
        coordinator = bg.HostPrefixCacheCoordinator(_small_config(shm_name))
        coordinator.initialize(True)
        coordinator.commit_prefix_pages(
            namespace,
            token_ids,
            16,
            [
                _group_pages(0, [_page(idx) for idx in range(4)]),
                _group_pages(1, [_page(idx) for idx in range(2)]),
            ],
        )

        active = coordinator.lookup_and_attach(namespace, token_ids)
        coordinator.begin_attachment_load(active.attachment_handle)
        coordinator.release_attachment(active.attachment_handle)
        stats = coordinator.get_stats()
        assert stats.active_attachments == 1
        assert stats.pending_load_entries == 4
        assert stats.pending_load_refs == 4

        evicted = coordinator.evict_until_free(2, 0, 0, 2)
        assert evicted.evicted_nodes == 0
        assert evicted.protected_nodes == 2
        assert coordinator.get_stats().eviction_protected_skips == 2

        coordinator.end_attachment_load(active.attachment_handle)
        stats = coordinator.get_stats()
        assert stats.active_attachments == 0
        assert stats.pending_load_entries == 0
        assert stats.pending_load_refs == 0
        evicted = coordinator.evict_until_free(2, 0, 0, 2)
        assert evicted.evicted_nodes == 2
        assert coordinator.get_stats().evicted_nodes == 2
    finally:
        _shm_unlink(shm_name)


def test_host_prefix_cache_clear_namespace_only_removes_matching_domain():
    shm_name = _random_shm_name()
    namespace_a = [1, 3, 5, 7]
    namespace_b = [2, 4, 6, 8]
    token_ids = list(range(8))
    try:
        coordinator = bg.HostPrefixCacheCoordinator(_config(shm_name))
        coordinator.initialize(True)
        coordinator.commit_prefix_pages(
            namespace_a,
            token_ids,
            8,
            [
                _group_pages(0, [_page(0), _page(1)]),
                _group_pages(1, [_page(0)]),
            ],
        )
        coordinator.commit_prefix_pages(
            namespace_b,
            token_ids,
            8,
            [
                _group_pages(0, [_page(10), _page(11)]),
                _group_pages(1, [_page(10)]),
            ],
        )

        cleared = coordinator.clear_namespace(namespace_a)
        assert cleared.evicted_nodes == 1
        miss = coordinator.estimate_lookup(namespace_a, token_ids)
        hit = coordinator.estimate_lookup(namespace_b, token_ids)
        assert miss.miss_reason_mask
        assert hit.common_cached_tokens == 8
        assert coordinator.get_stats().resident_nodes == 1
    finally:
        _shm_unlink(shm_name)


def test_host_prefix_cache_is_shared_across_process_attachments():
    shm_name = _random_shm_name()
    namespace = [7, 8, 9, 10]
    token_ids = list(range(8))
    try:
        owner = bg.HostPrefixCacheCoordinator(_config(shm_name))
        owner.initialize(True)
        owner.commit_prefix_pages(
            namespace,
            token_ids,
            8,
            [
                _group_pages(0, [_page(0), _page(1)]),
                _group_pages(1, [_page(0)]),
            ],
        )

        worker = bg.HostPrefixCacheCoordinator(_config(shm_name))
        worker.initialize(False)
        attached = worker.lookup_and_attach(namespace, token_ids)

        assert attached.common_cached_tokens == 8
        assert owner.get_stats().active_attachments == 0
        assert worker.get_stats().active_attachments == 1
        evicted = owner.evict_until_free(1, 0, 0, 1)
        assert evicted.evicted_nodes == 0
        assert evicted.protected_nodes == 1

        worker.release_attachment(attached.attachment_handle)
        assert worker.get_stats().active_attachments == 0
        evicted = owner.evict_until_free(1, 0, 0, 1)
        assert evicted.evicted_nodes == 1
    finally:
        _shm_unlink(shm_name)
