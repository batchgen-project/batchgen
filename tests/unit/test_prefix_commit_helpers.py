import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_MODULE_PATH = (
    Path(__file__).parents[2] / "batchgen" / "prefix_reuse" / "commit.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "test_prefix_cache_commit", _MODULE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

aligned_prefix_tokens = _MODULE.aligned_prefix_tokens
build_prefix_commit_request = _MODULE.build_prefix_commit_request
collect_group_pages_for_commit = _MODULE.collect_group_pages_for_commit
release_evicted_prefix_pages = _MODULE.release_evicted_prefix_pages
retain_inserted_prefix_pages = _MODULE.retain_inserted_prefix_pages


class _WorkerView:
    def __init__(self, pages):
        self.pages = list(pages)
        self.retained = []
        self.released = []

    def build_page_table(self, sequence_ids):
        return [list(self.pages) for _ in sequence_ids]

    def retain_sequence_pages(self, sequence_id, page_ids):
        self.retained.append((int(sequence_id), list(page_ids)))
        return list(page_ids)

    def release_resident_pages(self, page_ids):
        self.released.append(list(page_ids))


def _group(group_id, pages):
    return SimpleNamespace(
        group_id=group_id,
        pages=[SimpleNamespace(page_id=page) for page in pages],
    )


def test_commit_request_aligns_tokens_and_uses_fast_page_id_binding():
    request = build_prefix_commit_request(
        namespace_digest=(1, 2, 3, 4),
        token_ids=list(range(10)),
        publish_boundary_tokens=4,
        pages_by_group={0: [100, 101]},
    )
    coordinator = SimpleNamespace()
    calls = []
    coordinator.commit_prefix_page_ids = lambda *args: calls.append(args) or "ok"

    assert request is not None
    assert request.commit_tokens == 8
    assert request.commit(coordinator) == "ok"
    assert calls == [([1, 2, 3, 4], list(range(10)), 8, [(0, [100, 101])])]


def test_collect_group_pages_requires_complete_logical_prefix():
    worker = _WorkerView([100, 101, 102])

    pages = collect_group_pages_for_commit(
        worker_views_by_group={0: worker},
        sequence_id=42,
        commit_tokens=8,
        raw_page_tokens_by_group={0: 4},
    )

    assert pages == {0: [100, 101]}
    with pytest.raises(RuntimeError, match="expected 4"):
        collect_group_pages_for_commit(
            worker_views_by_group={0: worker},
            sequence_id=42,
            commit_tokens=16,
            raw_page_tokens_by_group={0: 4},
        )


def test_chain_hole_recommit_retains_exact_inserted_pages_only():
    worker = _WorkerView([100, 101, 102, 103, 104, 105])
    request = build_prefix_commit_request(
        namespace_digest=(1, 2, 3, 4),
        token_ids=list(range(24)),
        publish_boundary_tokens=8,
        pages_by_group={0: worker.pages},
    )
    result = SimpleNamespace(
        inserted_nodes=1,
        existing_nodes=2,
        inserted_group_pages=[_group(0, [102, 103])],
    )

    retained = retain_inserted_prefix_pages(
        commit_result=result,
        request=request,
        worker_views_by_group={0: worker},
        sequence_id=7,
    )

    assert retained == {0: [102, 103]}
    assert worker.retained == [(7, [102, 103])]


def test_retain_rejects_unrequested_coordinator_pages():
    worker = _WorkerView([100, 101])
    request = build_prefix_commit_request(
        namespace_digest=(1, 2, 3, 4),
        token_ids=list(range(8)),
        publish_boundary_tokens=4,
        pages_by_group={0: worker.pages},
    )

    with pytest.raises(RuntimeError, match="unrequested pages"):
        retain_inserted_prefix_pages(
            commit_result=SimpleNamespace(
                inserted_group_pages=[_group(0, [999])]
            ),
            request=request,
            worker_views_by_group={0: worker},
            sequence_id=7,
        )
    assert worker.retained == []


def test_eviction_releases_exact_resident_pages():
    worker = _WorkerView([])

    released = release_evicted_prefix_pages(
        eviction_result=SimpleNamespace(
            evicted_group_pages=[_group(0, [102, 103])]
        ),
        worker_views_by_group={0: worker},
    )

    assert released == {0: 2}
    assert worker.released == [[102, 103]]


@pytest.mark.parametrize("tokens, boundary, expected", [(0, 4, 0), (7, 4, 4), (8, 4, 8)])
def test_aligned_prefix_tokens(tokens, boundary, expected):
    assert aligned_prefix_tokens(tokens, boundary) == expected
