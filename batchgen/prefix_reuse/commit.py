"""Helpers for publishing completed Host KV pages to the prefix cache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


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


def _to_host_page_handle(core_engine_module: object, page: int | object):
    if hasattr(page, "page_id"):
        return page
    handle = core_engine_module.HostPageHandle()
    handle.page_id = int(page)
    return handle
