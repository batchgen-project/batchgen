"""Publish and retain page-aligned Host-KV prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PrefixCommitRequest:
    namespace_digest: tuple[int, int, int, int]
    token_ids: list[int]
    commit_tokens: int
    publish_boundary_tokens: int
    page_ids_by_group: dict[int, list[int]]

    def commit(self, coordinator: object):
        return coordinator.commit_prefix_page_ids(
            list(self.namespace_digest),
            self.token_ids,
            int(self.commit_tokens),
            [
                (group_id, page_ids)
                for group_id, page_ids in sorted(
                    self.page_ids_by_group.items()
                )
            ],
        )

    def capacity_requirements(self) -> tuple[int, int, int]:
        nodes = self.commit_tokens // self.publish_boundary_tokens
        return (
            nodes,
            nodes * len(self.page_ids_by_group),
            sum(len(pages) for pages in self.page_ids_by_group.values()),
        )


def aligned_prefix_tokens(total_tokens: int, boundary_tokens: int) -> int:
    boundary = int(boundary_tokens)
    if boundary <= 0:
        raise ValueError("boundary_tokens must be positive")
    tokens = max(0, int(total_tokens))
    return (tokens // boundary) * boundary


def build_prefix_commit_request(
    *,
    namespace_digest: Sequence[int],
    token_ids: Sequence[int],
    publish_boundary_tokens: int,
    pages_by_group: Mapping[int, Sequence[int | object]],
) -> PrefixCommitRequest | None:
    commit_tokens = aligned_prefix_tokens(
        len(token_ids), publish_boundary_tokens
    )
    if commit_tokens == 0:
        return None
    page_ids = {
        int(group_id): [_page_id(page) for page in pages]
        for group_id, pages in pages_by_group.items()
    }
    if not page_ids or any(not pages for pages in page_ids.values()):
        raise ValueError("prefix commit requires pages for every KV group")
    return PrefixCommitRequest(
        namespace_digest=tuple(int(value) for value in namespace_digest),
        token_ids=[int(token_id) for token_id in token_ids],
        commit_tokens=commit_tokens,
        publish_boundary_tokens=int(publish_boundary_tokens),
        page_ids_by_group=page_ids,
    )


def collect_group_pages_for_commit(
    *,
    worker_views_by_group: Mapping[int, object],
    sequence_id: int,
    commit_tokens: int,
    raw_page_tokens_by_group: Mapping[int, int],
) -> dict[int, list[int]]:
    pages_by_group: dict[int, list[int]] = {}
    for group_id, raw_page_tokens in sorted(raw_page_tokens_by_group.items()):
        worker_view = worker_views_by_group.get(int(group_id))
        if worker_view is None:
            raise RuntimeError(
                f"missing Host KV worker view for prefix group {group_id}"
            )
        page_tokens = int(raw_page_tokens)
        if page_tokens <= 0 or int(commit_tokens) % page_tokens != 0:
            raise RuntimeError(
                f"prefix group {group_id} cannot represent {commit_tokens} tokens"
            )
        expected_pages = int(commit_tokens) // page_tokens
        logical_pages = list(
            worker_view.build_page_table([int(sequence_id)])[0]
        )
        pages = [int(page) for page in logical_pages[:expected_pages]]
        if len(pages) != expected_pages:
            raise RuntimeError(
                f"prefix group {group_id} has {len(pages)} pages for sequence "
                f"{sequence_id}, expected {expected_pages}"
            )
        pages_by_group[int(group_id)] = pages
    return pages_by_group


def retain_inserted_prefix_pages(
    *,
    commit_result: object,
    request: PrefixCommitRequest,
    worker_views_by_group: Mapping[int, object],
    sequence_id: int,
) -> dict[int, list[int]]:
    """Transfer only coordinator-confirmed new pages to resident ownership."""

    inserted: dict[int, list[int]] = {}
    for group_pages in commit_result.inserted_group_pages:
        group_id = int(group_pages.group_id)
        if group_id in inserted:
            raise RuntimeError(
                f"duplicate inserted prefix group {group_id} in commit result"
            )
        page_ids = [_page_id(page) for page in group_pages.pages]
        requested = set(request.page_ids_by_group.get(group_id, ()))
        if not requested or any(page not in requested for page in page_ids):
            raise RuntimeError(
                f"coordinator returned unrequested pages for prefix group {group_id}"
            )
        if len(page_ids) != len(set(page_ids)):
            raise RuntimeError(
                f"coordinator returned duplicate pages for prefix group {group_id}"
            )
        if page_ids:
            worker_view = worker_views_by_group.get(group_id)
            if worker_view is None:
                raise RuntimeError(
                    f"missing Host KV worker view for prefix group {group_id}"
                )
            retained = [
                int(page)
                for page in worker_view.retain_sequence_pages(
                    int(sequence_id), page_ids
                )
            ]
            if retained != page_ids:
                raise RuntimeError(
                    f"Host KV retained different pages for prefix group {group_id}"
                )
        inserted[group_id] = page_ids
    return inserted


def release_evicted_prefix_pages(
    *,
    eviction_result: object,
    worker_views_by_group: Mapping[int, object],
) -> dict[int, int]:
    released: dict[int, int] = {}
    for group_pages in eviction_result.evicted_group_pages:
        group_id = int(group_pages.group_id)
        if group_id in released:
            raise RuntimeError(
                f"duplicate evicted prefix group {group_id} in eviction result"
            )
        page_ids = [_page_id(page) for page in group_pages.pages]
        if len(page_ids) != len(set(page_ids)):
            raise RuntimeError(
                f"coordinator returned duplicate evicted pages for group {group_id}"
            )
        worker_view = worker_views_by_group.get(group_id)
        if worker_view is None:
            raise RuntimeError(
                f"missing Host KV worker view for evicted prefix group {group_id}"
            )
        if page_ids:
            worker_view.release_resident_pages(page_ids)
        released[group_id] = len(page_ids)
    return released


def _page_id(page: int | object) -> int:
    return int(getattr(page, "page_id", page))
