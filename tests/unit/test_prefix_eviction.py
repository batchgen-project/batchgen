from types import SimpleNamespace

import pytest

from batchgen.prefix_reuse.eviction import (
    commit_prefix_pages_with_capacity_retry,
    evict_prefix_pages_for_host_allocation,
)


class _View:
    def __init__(self):
        self.released = []

    def release_resident_pages(self, pages):
        self.released.append(list(pages))


def _eviction(pages, *, protected=0):
    return SimpleNamespace(
        protected_nodes=protected,
        evicted_group_pages=[
            SimpleNamespace(group_id=0, pages=list(pages))
        ],
    )


def test_host_allocation_eviction_releases_physical_pages():
    coordinator = SimpleNamespace()
    coordinator.evict_until_releasable_pages = lambda reqs, scan: _eviction(
        [7, 8]
    )
    core = SimpleNamespace()
    core.GroupPageRequirement = type("Requirement", (), {})
    view = _View()

    released = evict_prefix_pages_for_host_allocation(
        core_engine_module=core,
        coordinator=coordinator,
        worker_views_by_group={0: view},
        group_id=0,
        page_deficit=2,
        max_scan_nodes=16,
    )

    assert released == 2
    assert view.released == [[7, 8]]


def test_host_allocation_eviction_fails_when_attachments_protect_pages():
    coordinator = SimpleNamespace()
    coordinator.evict_until_releasable_pages = lambda reqs, scan: _eviction(
        [7], protected=3
    )
    core = SimpleNamespace()
    core.GroupPageRequirement = type("Requirement", (), {})

    with pytest.raises(RuntimeError, match="could not release enough"):
        evict_prefix_pages_for_host_allocation(
            core_engine_module=core,
            coordinator=coordinator,
            worker_views_by_group={0: _View()},
            group_id=0,
            page_deficit=2,
            max_scan_nodes=16,
        )


def test_commit_retries_once_after_capacity_eviction():
    class Request:
        calls = 0

        def commit(self, coordinator):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("Host prefix cache node table is full")
            return "committed"

        def capacity_requirements(self):
            return (2, 2, 4)

    coordinator = SimpleNamespace()
    coordinator.evict_until_free = lambda *args: _eviction([10, 11, 12, 13])
    view = _View()
    outcome = commit_prefix_pages_with_capacity_retry(
        request=Request(),
        coordinator=coordinator,
        worker_views_by_group={0: view},
        max_scan_nodes=32,
    )

    assert outcome.commit_result == "committed"
    assert view.released == [[10, 11, 12, 13]]
