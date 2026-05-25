"""Helpers for publishing completed Host KV pages to the prefix cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from batchgen.prefix_reuse.config import PrefixKVGroupSpec


@dataclass(frozen=True)
class PrefixCommitRequest:
    namespace_digest: tuple[int, int, int, int]
    token_ids: list[int]
    commit_tokens: int
    group_pages: list[object]

    def commit(self, coordinator: object):
        return coordinator.commit_prefix_pages(
            list(self.namespace_digest),
            self.token_ids,
            int(self.commit_tokens),
            self.group_pages,
        )


def aligned_prefix_tokens(total_tokens: int, publish_boundary_tokens: int) -> int:
    """Return the longest prefix length that can be safely published."""

    boundary = int(publish_boundary_tokens)
    if boundary <= 0:
        raise ValueError("publish_boundary_tokens must be positive")
    token_count = max(0, int(total_tokens))
    return (token_count // boundary) * boundary


def build_prefix_commit_request(
    *,
    core_engine_module: object,
    namespace_digest: Sequence[int],
    token_ids: Sequence[int],
    publish_boundary_tokens: int,
    pages_by_group: Mapping[int, Sequence[int | object]],
) -> PrefixCommitRequest | None:
    """Build a page-aligned prefix cache commit request.

    The coordinator indexes existing Host KV pages. Page allocation, page
    ownership, and eviction-side page release stay with the Host KV managers.
    """

    commit_tokens = aligned_prefix_tokens(
        len(token_ids), publish_boundary_tokens
    )
    if commit_tokens == 0:
        return None

    group_pages = []
    for group_id, page_handles in sorted(pages_by_group.items()):
        group = core_engine_module.GroupCommitPages()
        group.group_id = int(group_id)
        group.pages = [
            _to_host_page_handle(core_engine_module, page)
            for page in page_handles
        ]
        group_pages.append(group)

    return PrefixCommitRequest(
        namespace_digest=tuple(int(value) for value in namespace_digest),
        token_ids=[int(token_id) for token_id in token_ids],
        commit_tokens=commit_tokens,
        group_pages=group_pages,
    )


def collect_required_group_pages_for_commit(
    *,
    worker_views_by_group: Mapping[int, object],
    sequence_id: int,
    commit_tokens: int,
    group_specs: Sequence[PrefixKVGroupSpec],
) -> dict[int, list[int]]:
    """Collect existing Host KV page ids for a page-aligned commit."""

    result: dict[int, list[int]] = {}
    for spec in group_specs:
        if not spec.required_for_reuse:
            continue
        worker_view = worker_views_by_group.get(int(spec.group_id))
        if worker_view is None:
            raise RuntimeError(
                f"missing Host KV worker view for prefix group {spec.group_id}"
            )
        page_count = int(commit_tokens) // int(spec.raw_page_tokens)
        page_table = worker_view.build_page_table([int(sequence_id)])
        pages = list(page_table[0])[:page_count]
        if len(pages) != page_count:
            raise RuntimeError(
                f"prefix group {spec.group_id} has {len(pages)} pages for "
                f"sequence {sequence_id}, expected {page_count}"
            )
        result[int(spec.group_id)] = [int(page_id) for page_id in pages]
    return result


def _to_host_page_handle(core_engine_module: object, page: int | object):
    if hasattr(page, "page_id"):
        return page
    handle = core_engine_module.HostPageHandle()
    handle.page_id = int(page)
    return handle
