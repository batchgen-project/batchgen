from __future__ import annotations

from batchgen.prefix_reuse.commit import (
    aligned_prefix_tokens,
    build_committable_prefix_token_ids,
    build_prefix_commit_request,
    collect_required_group_pages_for_commit,
)
from batchgen.prefix_reuse.config import (
    PrefixKVGroupSemantic,
    PrefixKVGroupSpec,
)


class _HostPageHandle:
    def __init__(self):
        self.page_id = 0


class _GroupCommitPages:
    def __init__(self):
        self.group_id = 0
        self.pages = []


class _Core:
    HostPageHandle = _HostPageHandle
    GroupCommitPages = _GroupCommitPages


class _Coordinator:
    def __init__(self):
        self.calls = []

    def commit_prefix_pages(
        self, namespace_digest, token_ids, commit_tokens, group_pages
    ):
        self.calls.append(
            (namespace_digest, token_ids, commit_tokens, group_pages)
        )
        return "committed"


class _WorkerView:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def build_page_table(self, sequence_ids):
        self.calls.append(list(sequence_ids))
        return [list(self.pages) for _ in sequence_ids]


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


def test_build_prefix_commit_request_skips_unaligned_short_prefix():
    request = build_prefix_commit_request(
        core_engine_module=_Core,
        namespace_digest=(1, 2, 3, 4),
        token_ids=[10, 11, 12],
        publish_boundary_tokens=4,
        pages_by_group={0: [7]},
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
