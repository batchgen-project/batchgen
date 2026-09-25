"""Validated prefix-cache lookup helpers for prefill admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PrefixCachePrefillLookup:
    lookup_results: tuple[object, ...]
    attached_tokens: tuple[int, ...]
    compute_cached_tokens: tuple[int, ...]

    @property
    def has_hit(self) -> bool:
        return any(tokens > 0 for tokens in self.compute_cached_tokens)


@dataclass(frozen=True)
class PrefixCacheSequenceState:
    lookup_result: object
    attached_tokens: int
    compute_cached_tokens: int
    commit_attachment_handles: tuple[int, ...] = ()

    @property
    def attachment_handle(self) -> int:
        return int(getattr(self.lookup_result, "attachment_handle", 0))

    @property
    def shared_page_ids(self) -> tuple[int, ...]:
        if self.attached_tokens == 0:
            return ()
        spans = list(getattr(self.lookup_result, "materialization_spans", ()))
        if len(spans) != 1 or int(spans[0].group_id) != 0:
            raise RuntimeError(
                "GPT-OSS prefix state requires exactly one FULL_KV group"
            )
        return tuple(int(page.page_id) for page in spans[0].pages)


def lookup_prefix_cache_for_prefill(
    *,
    coordinator: object,
    namespace_digest: Sequence[int],
    prompt_token_ids: Sequence[Sequence[int]],
    page_size_tokens: int,
) -> PrefixCachePrefillLookup:
    """Attach page-aligned GPT-OSS prefixes in request order.

    A raw full hit remains fully attached in Host memory, but the compute path
    recomputes the last prompt token so it can produce the first output token.
    """

    page_size = int(page_size_tokens)
    if page_size <= 0:
        raise ValueError("page_size_tokens must be positive")

    results: list[object] = []
    attached_tokens: list[int] = []
    compute_cached_tokens: list[int] = []
    try:
        for prompt in prompt_token_ids:
            token_ids = [int(token_id) for token_id in prompt]
            if not token_ids:
                raise ValueError("prefix cache lookup requires a non-empty prompt")
            result = coordinator.lookup_and_attach(
                [int(value) for value in namespace_digest], token_ids
            )
            results.append(result)
            cached = _validate_lookup_result(
                result=result,
                prompt_length=len(token_ids),
                page_size_tokens=page_size,
            )
            attached_tokens.append(cached)
            compute_cached_tokens.append(
                len(token_ids) - 1 if cached == len(token_ids) else cached
            )
    except Exception:
        for result in results:
            handle = int(getattr(result, "attachment_handle", 0))
            if handle:
                coordinator.release_attachment(handle)
        raise

    return PrefixCachePrefillLookup(
        lookup_results=tuple(results),
        attached_tokens=tuple(attached_tokens),
        compute_cached_tokens=tuple(compute_cached_tokens),
    )


def release_prefix_lookup_attachments(
    *, coordinator: object, lookup: PrefixCachePrefillLookup
) -> None:
    for result in lookup.lookup_results:
        handle = int(getattr(result, "attachment_handle", 0))
        if handle:
            coordinator.release_attachment(handle)


def estimate_prefix_cached_pages_for_prefill(
    *,
    coordinator: object,
    namespace_digest: Sequence[int],
    prompt_token_ids: Sequence[int],
    page_size_tokens: int,
) -> int:
    """Estimate reusable Host pages without attaching or evicting them."""
    page_size = int(page_size_tokens)
    if page_size <= 0:
        raise ValueError("page_size_tokens must be positive")
    token_ids = [int(token_id) for token_id in prompt_token_ids]
    if not token_ids:
        raise ValueError("prefix cache estimate requires a non-empty prompt")
    result = coordinator.estimate_lookup(
        [int(value) for value in namespace_digest], token_ids
    )
    cached = int(result.common_cached_tokens)
    if cached < 0 or cached > len(token_ids):
        raise RuntimeError(
            "prefix cache estimate returned an invalid token count: "
            f"cached={cached}, prompt_length={len(token_ids)}"
        )
    if cached % page_size:
        raise RuntimeError(
            "prefix cache estimate returned a non-page-aligned hit: "
            f"cached={cached}, page_size={page_size}"
        )
    return cached // page_size


def _validate_lookup_result(
    *, result: object, prompt_length: int, page_size_tokens: int
) -> int:
    cached = int(result.common_cached_tokens)
    handle = int(result.attachment_handle)
    if cached < 0 or cached > int(prompt_length):
        raise RuntimeError(
            "prefix cache returned an invalid token count: "
            f"cached={cached}, prompt_length={prompt_length}"
        )
    if cached % int(page_size_tokens) != 0:
        raise RuntimeError(
            "prefix cache returned a non-page-aligned hit: "
            f"cached={cached}, page_size={page_size_tokens}"
        )
    if (cached == 0) != (handle == 0):
        raise RuntimeError(
            "prefix cache attachment handle does not match lookup hit state"
        )

    spans = list(getattr(result, "materialization_spans", ()))
    if cached == 0:
        if spans:
            raise RuntimeError("prefix cache miss returned materialization spans")
        return 0
    if len(spans) != 1 or int(spans[0].group_id) != 0:
        raise RuntimeError(
            "GPT-OSS prefix lookup requires exactly one FULL_KV group"
        )
    span = spans[0]
    if int(span.raw_end_token) != cached:
        raise RuntimeError(
            "prefix materialization span does not cover the attached prefix"
        )
    expected_pages = cached // int(page_size_tokens)
    if len(span.pages) != expected_pages:
        raise RuntimeError(
            "prefix materialization span page count mismatch: "
            f"got={len(span.pages)}, expected={expected_pages}"
        )
    return cached
