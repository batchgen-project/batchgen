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
from batchgen.prefix_reuse.worker_commit import (
    build_sequence_prefix_commit_request,
    sequence_token_ids_for_prefix_commit,
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
