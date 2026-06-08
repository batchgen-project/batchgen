from __future__ import annotations

import torch

from batchgen.prefix_reuse.commit import (
    aligned_prefix_tokens,
    build_committable_prefix_token_ids,
    build_prefix_commit_request,
    collect_required_group_pages_for_commit,
)
from batchgen.prefix_reuse.config import (
    PrefixCacheRuntimeConfig,
    PrefixKVGroupSemantic,
    PrefixKVGroupSpec,
)
from batchgen.prefix_reuse.eviction import (
    commit_prefix_pages_with_capacity_retry,
    evict_prefix_pages_for_host_allocation,
    release_evicted_prefix_pages,
)
from batchgen.prefix_reuse.worker_commit import (
    build_sequence_prefix_commit_request,
    retain_newly_committed_prefix_pages,
    sequence_token_ids_for_prefix_commit,
)


class _HostPageHandle:
    def __init__(self):
        self.page_id = 0


class _GroupCommitPages:
    def __init__(self):
        self.group_id = 0
        self.pages = []


class _GroupPageRequirement:
    def __init__(self):
        self.group_id = 0
        self.min_pages = 0


class _Core:
    HostPageHandle = _HostPageHandle
    GroupCommitPages = _GroupCommitPages
    GroupPageRequirement = _GroupPageRequirement


class _Coordinator:
    def __init__(
        self,
        *,
        fail_once_with: RuntimeError | None = None,
        eviction_result=None,
    ):
        self.calls = []
        self.evict_calls = []
        self.fail_once_with = fail_once_with
        self.eviction_result = eviction_result

    def commit_prefix_pages(
        self, namespace_digest, token_ids, commit_tokens, group_pages
    ):
        self.calls.append(
            (namespace_digest, token_ids, commit_tokens, group_pages)
        )
        if self.fail_once_with is not None:
            exc = self.fail_once_with
            self.fail_once_with = None
            raise exc
        return "committed"

    def evict_until_free(
        self,
        min_free_nodes,
        min_free_group_entries,
        min_free_page_handles,
        max_scan_nodes,
    ):
        self.evict_calls.append(
            (
                min_free_nodes,
                min_free_group_entries,
                min_free_page_handles,
                max_scan_nodes,
            )
        )
        return self.eviction_result

    def evict_until_releasable_pages(self, requirements, max_scan_nodes):
        self.evict_calls.append(
            (
                [
                    (int(requirement.group_id), int(requirement.min_pages))
                    for requirement in requirements
                ],
                max_scan_nodes,
            )
        )
        return self.eviction_result


class _WorkerView:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []
        self.retained = []
        self.released = []

    def build_page_table(self, sequence_ids):
        self.calls.append(list(sequence_ids))
        return [list(self.pages) for _ in sequence_ids]

    def retain_sequence_prefix_pages(self, sequence_id, num_pages):
        self.retained.append((int(sequence_id), int(num_pages)))
        return self.pages[: int(num_pages)]

    def retain_sequence_page_range(self, sequence_id, start_page, num_pages):
        self.retained.append(
            (int(sequence_id), int(start_page), int(num_pages))
        )
        start = int(start_page)
        end = start + int(num_pages)
        return self.pages[start:end]

    def retain_sequence_pages(self, sequence_id, page_ids):
        self.retained.append((int(sequence_id), list(page_ids)))
        return list(page_ids)

    def release_resident_pages(self, page_ids):
        self.released.append(list(page_ids))


class _EvictionResult:
    def __init__(self, evicted_group_pages):
        self.evicted_nodes = len(evicted_group_pages)
        self.protected_nodes = 0
        self.evicted_group_pages = evicted_group_pages


def _page(page_id: int) -> _HostPageHandle:
    handle = _HostPageHandle()
    handle.page_id = int(page_id)
    return handle


def _group_pages(group_id: int, pages) -> _GroupCommitPages:
    group = _GroupCommitPages()
    group.group_id = int(group_id)
    group.pages = list(pages)
    return group


class _Seq:
    def __init__(
        self,
        *,
        global_idx=7,
        prompt=None,
        decoded=None,
        decoded_length=0,
        reentry_decoded_baseline=0,
        prefix_shared_tokens=0,
        prefix_committed_tokens=0,
    ):
        prompt = [1, 2, 3, 4] if prompt is None else list(prompt)
        decoded = [] if decoded is None else list(decoded)
        self.global_idx = global_idx
        self.prompt_length = len(prompt)
        self.input_ids = torch.tensor([prompt], dtype=torch.long)
        decoded_capacity = max(len(decoded), decoded_length, 1)
        self.decoded_tokens = torch.zeros(
            (1, decoded_capacity), dtype=torch.long
        )
        if decoded:
            self.decoded_tokens[0, : len(decoded)] = torch.tensor(
                decoded, dtype=torch.long
            )
        self.decoded_length = decoded_length
        self.reentry_decoded_baseline = reentry_decoded_baseline
        self.prefix_shared_tokens = prefix_shared_tokens
        self.prefix_committed_tokens = prefix_committed_tokens


def _runtime_config() -> PrefixCacheRuntimeConfig:
    return PrefixCacheRuntimeConfig(
        shm_name="test",
        namespace_digest=(1, 2, 3, 4),
        group_specs=(
            PrefixKVGroupSpec(
                group_id=0,
                semantic=PrefixKVGroupSemantic.FULL_KV,
                required_for_reuse=True,
                raw_page_tokens=4,
            ),
        ),
        hash_block_tokens=4,
        publish_boundary_tokens=4,
        max_nodes=16,
        max_group_entries=16,
        max_page_handles=32,
        max_attachments=16,
    )


def test_aligned_prefix_tokens_floor_to_publish_boundary():
    assert aligned_prefix_tokens(0, 64) == 0
    assert aligned_prefix_tokens(63, 64) == 0
    assert aligned_prefix_tokens(64, 64) == 64
    assert aligned_prefix_tokens(191, 64) == 128


def test_build_committable_prefix_token_ids_appends_only_new_decode_tokens():
    token_ids = build_committable_prefix_token_ids(
        prompt_token_ids=[1, 2, 3, 4],
        decoded_token_ids=[10, 11, 12],
        decoded_start=2,
        max_tokens=5,
    )

    assert token_ids == [1, 2, 3, 4, 12]


def test_build_committable_prefix_token_ids_clamps_negative_inputs():
    token_ids = build_committable_prefix_token_ids(
        prompt_token_ids=[1, 2],
        decoded_token_ids=[3, 4],
        decoded_start=-8,
        max_tokens=-1,
    )

    assert token_ids == []


def test_sequence_token_ids_for_prefix_commit_skips_reentry_baseline():
    seq = _Seq(
        prompt=[1, 2, 3, 10],
        decoded=[10, 11, 12],
        decoded_length=3,
        reentry_decoded_baseline=1,
    )

    token_ids = sequence_token_ids_for_prefix_commit(
        seq,
        include_new_decode_tokens=True,
        max_tokens=6,
    )

    assert token_ids == [1, 2, 3, 10, 11, 12]


def test_build_sequence_prefix_commit_request_collects_logical_pages():
    seq = _Seq(
        global_idx=42,
        prompt=[1, 2, 3, 4],
        decoded=[5, 6, 7, 8],
        decoded_length=4,
    )

    request_pair = build_sequence_prefix_commit_request(
        core_engine_module=_Core,
        runtime_config=_runtime_config(),
        worker_views_by_group={0: _WorkerView([100, 101])},
        seq=seq,
        include_new_decode_tokens=True,
    )

    assert request_pair is not None
    request, commit_tokens = request_pair
    assert commit_tokens == 8
    assert request.token_ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [page.page_id for page in request.group_pages[0].pages] == [
        100,
        101,
    ]


def test_build_sequence_prefix_commit_request_skips_already_shared_prefix():
    seq = _Seq(
        prompt=[1, 2, 3, 4],
        decoded=[5],
        decoded_length=1,
        prefix_shared_tokens=4,
    )

    request_pair = build_sequence_prefix_commit_request(
        core_engine_module=_Core,
        runtime_config=_runtime_config(),
        worker_views_by_group={0: _WorkerView([100])},
        seq=seq,
        include_new_decode_tokens=False,
    )

    assert request_pair is None


def test_build_sequence_prefix_commit_request_skips_already_committed_prefix():
    seq = _Seq(
        prompt=[1, 2, 3, 4],
        decoded=[5],
        decoded_length=1,
        prefix_committed_tokens=4,
    )

    request_pair = build_sequence_prefix_commit_request(
        core_engine_module=_Core,
        runtime_config=_runtime_config(),
        worker_views_by_group={0: _WorkerView([100])},
        seq=seq,
        include_new_decode_tokens=False,
    )

    assert request_pair is None


def test_build_prefix_commit_request_skips_unaligned_short_prefix():
    request = build_prefix_commit_request(
        core_engine_module=_Core,
        namespace_digest=(1, 2, 3, 4),
        token_ids=[10, 11, 12],
        publish_boundary_tokens=4,
        pages_by_group={0: [7]},
        raw_page_tokens_by_group={0: 4},
    )

    assert request is None


def test_build_prefix_commit_request_uses_existing_group_pages():
    existing = _HostPageHandle()
    existing.page_id = 9

    request = build_prefix_commit_request(
        core_engine_module=_Core,
        namespace_digest=(1, 2, 3, 4),
        token_ids=[10, 11, 12, 13, 14],
        publish_boundary_tokens=4,
        pages_by_group={1: [existing], 0: [5, 6]},
        raw_page_tokens_by_group={0: 4, 1: 4},
    )

    assert request is not None
    assert request.namespace_digest == (1, 2, 3, 4)
    assert request.token_ids == [10, 11, 12, 13, 14]
    assert request.commit_tokens == 4
    assert [group.group_id for group in request.group_pages] == [0, 1]
    assert [page.page_id for page in request.group_pages[0].pages] == [5, 6]
    assert request.group_pages[1].pages == [existing]


def test_prefix_commit_request_invokes_coordinator():
    request = build_prefix_commit_request(
        core_engine_module=_Core,
        namespace_digest=(1, 2, 3, 4),
        token_ids=[10, 11, 12, 13],
        publish_boundary_tokens=4,
        pages_by_group={0: [5]},
        raw_page_tokens_by_group={0: 4},
    )
    coordinator = _Coordinator()

    result = request.commit(coordinator)

    assert result == "committed"
    assert len(coordinator.calls) == 1
    namespace_digest, token_ids, commit_tokens, group_pages = (
        coordinator.calls[0]
    )
    assert namespace_digest == [1, 2, 3, 4]
    assert token_ids == [10, 11, 12, 13]
    assert commit_tokens == 4
    assert [page.page_id for page in group_pages[0].pages] == [5]


def test_prefix_commit_request_capacity_requirements_use_raw_page_rates():
    request = build_prefix_commit_request(
        core_engine_module=_Core,
        namespace_digest=(1, 2, 3, 4),
        token_ids=list(range(16)),
        publish_boundary_tokens=8,
        pages_by_group={0: [0, 1, 2, 3], 1: [10, 11]},
        raw_page_tokens_by_group={0: 4, 1: 8},
    )

    assert request is not None
    assert request.capacity_requirements() == (2, 4, 6)


def test_prefix_commit_request_capacity_requirements_cover_c128_groups():
    request = build_prefix_commit_request(
        core_engine_module=_Core,
        namespace_digest=(1, 2, 3, 4),
        token_ids=list(range(512)),
        publish_boundary_tokens=256,
        pages_by_group={
            0: list(range(8)),
            1: [100, 101],
            2: [200, 201],
        },
        raw_page_tokens_by_group={0: 64, 1: 256, 2: 256},
    )

    assert request is not None
    assert request.capacity_requirements() == (2, 6, 12)


def test_release_evicted_prefix_pages_requires_matching_worker_view():
    evicted = _EvictionResult([_group_pages(9, [_page(100)])])

    try:
        release_evicted_prefix_pages(
            eviction_result=evicted,
            worker_views_by_group={},
        )
    except RuntimeError as exc:
        assert "evicted prefix group 9" in str(exc)
    else:  # pragma: no cover - failure path assertion
        raise AssertionError("missing evicted group worker view should fail")


def test_commit_prefix_pages_retries_after_capacity_eviction():
    evicted = _EvictionResult(
        [
            _group_pages(0, [_page(100), _page(100), _page(101)]),
            _group_pages(1, [_page(200)]),
        ]
    )
    coordinator = _Coordinator(
        fail_once_with=RuntimeError("Host prefix cache node table is full"),
        eviction_result=evicted,
    )
    request = build_prefix_commit_request(
        core_engine_module=_Core,
        namespace_digest=(1, 2, 3, 4),
        token_ids=list(range(16)),
        publish_boundary_tokens=8,
        pages_by_group={0: [0, 1, 2, 3], 1: [10, 11]},
        raw_page_tokens_by_group={0: 4, 1: 8},
    )
    primary = _WorkerView([])
    compressed = _WorkerView([])

    result = commit_prefix_pages_with_capacity_retry(
        request=request,
        coordinator=coordinator,
        worker_views_by_group={0: primary, 1: compressed},
        max_scan_nodes=7,
    )

    assert result.commit_result == "committed"
    assert result.eviction_result is evicted
    assert result.released_pages_by_group == {0: 2, 1: 1}
    assert coordinator.evict_calls == [(2, 4, 6, 7)]
    assert len(coordinator.calls) == 2
    assert primary.released == [[100, 101]]
    assert compressed.released == [[200]]


def test_evict_prefix_pages_for_host_allocation_uses_page_requirements():
    evicted = _EvictionResult(
        [
            _group_pages(0, [_page(100), _page(101)]),
            _group_pages(1, [_page(200), _page(201), _page(201)]),
        ]
    )
    coordinator = _Coordinator(eviction_result=evicted)
    primary = _WorkerView([])
    compressed = _WorkerView([])

    result = evict_prefix_pages_for_host_allocation(
        core_engine_module=_Core,
        coordinator=coordinator,
        worker_views_by_group={0: primary, 1: compressed},
        page_deficit_by_group={0: 2, 1: 1, 2: 0},
        max_scan_nodes=9,
    )

    assert result.eviction_result is evicted
    assert result.released_pages_by_group == {0: 2, 1: 2}
    assert coordinator.evict_calls == [([(0, 2), (1, 1)], 9)]
    assert primary.released == [[100, 101]]
    assert compressed.released == [[200, 201]]


def test_evict_prefix_pages_for_host_allocation_raises_when_short():
    evicted = _EvictionResult([_group_pages(0, [_page(100)])])
    coordinator = _Coordinator(eviction_result=evicted)

    try:
        evict_prefix_pages_for_host_allocation(
            core_engine_module=_Core,
            coordinator=coordinator,
            worker_views_by_group={0: _WorkerView([])},
            page_deficit_by_group={0: 2},
        )
    except RuntimeError as exc:
        assert "could not release enough Host KV pages" in str(exc)
        assert "missing={0: 1}" in str(exc)
    else:  # pragma: no cover - failure path assertion
        raise AssertionError("short eviction result should fail")


def test_commit_prefix_pages_does_not_retry_non_capacity_errors():
    coordinator = _Coordinator(fail_once_with=RuntimeError("other failure"))
    request = build_prefix_commit_request(
        core_engine_module=_Core,
        namespace_digest=(1, 2, 3, 4),
        token_ids=list(range(8)),
        publish_boundary_tokens=4,
        pages_by_group={0: [0, 1]},
        raw_page_tokens_by_group={0: 4},
    )

    try:
        commit_prefix_pages_with_capacity_retry(
            request=request,
            coordinator=coordinator,
            worker_views_by_group={0: _WorkerView([])},
        )
    except RuntimeError as exc:
        assert str(exc) == "other failure"
    else:  # pragma: no cover - failure path assertion
        raise AssertionError("non-capacity RuntimeError should be re-raised")
    assert coordinator.evict_calls == []


def test_retain_newly_committed_prefix_pages_uses_group_raw_page_rates():
    config = PrefixCacheRuntimeConfig(
        shm_name="test",
        namespace_digest=(1, 2, 3, 4),
        group_specs=(
            PrefixKVGroupSpec(
                group_id=0,
                semantic=PrefixKVGroupSemantic.FULL_KV,
                required_for_reuse=True,
                raw_page_tokens=4,
            ),
            PrefixKVGroupSpec(
                group_id=1,
                semantic=PrefixKVGroupSemantic.COMPRESSED_RATIO_KV,
                required_for_reuse=True,
                raw_page_tokens=8,
                compression_ratio=2,
            ),
            PrefixKVGroupSpec(
                group_id=2,
                semantic=PrefixKVGroupSemantic.FULL_KV,
                required_for_reuse=False,
                raw_page_tokens=4,
            ),
        ),
        hash_block_tokens=4,
        publish_boundary_tokens=8,
        max_nodes=16,
        max_group_entries=16,
        max_page_handles=32,
        max_attachments=16,
    )
    primary = _WorkerView([0, 1, 2, 3])
    compressed = _WorkerView([10, 11])

    committed = retain_newly_committed_prefix_pages(
        runtime_config=config,
        worker_views_by_group={0: primary, 1: compressed},
        sequence_id=123,
        previous_committed_tokens=8,
        commit_tokens=16,
    )

    assert committed == 16
    assert primary.retained == [(123, [2, 3])]
    assert compressed.retained == [(123, [11])]


def test_retain_newly_committed_prefix_pages_skips_already_committed_tokens():
    config = _runtime_config()
    primary = _WorkerView([0, 1])

    committed = retain_newly_committed_prefix_pages(
        runtime_config=config,
        worker_views_by_group={0: primary},
        sequence_id=123,
        previous_committed_tokens=8,
        commit_tokens=8,
    )

    assert committed == 8
    assert primary.retained == []


def test_collect_required_group_pages_for_commit_reads_worker_page_tables():
    specs = [
        PrefixKVGroupSpec(
            group_id=0,
            semantic=PrefixKVGroupSemantic.FULL_KV,
            required_for_reuse=True,
            raw_page_tokens=4,
        ),
        PrefixKVGroupSpec(
            group_id=1,
            semantic=PrefixKVGroupSemantic.MLA_COMPRESSED_KV,
            required_for_reuse=True,
            raw_page_tokens=8,
        ),
        PrefixKVGroupSpec(
            group_id=2,
            semantic=PrefixKVGroupSemantic.FULL_KV,
            required_for_reuse=False,
            raw_page_tokens=4,
        ),
    ]
    primary = _WorkerView([10, 11, 12, 13])
    mla = _WorkerView([20, 21])

    pages = collect_required_group_pages_for_commit(
        worker_views_by_group={0: primary, 1: mla},
        sequence_id=100,
        commit_tokens=8,
        group_specs=specs,
    )

    assert pages == {0: [10, 11], 1: [20]}
    assert primary.calls == [[100]]
    assert mla.calls == [[100]]
