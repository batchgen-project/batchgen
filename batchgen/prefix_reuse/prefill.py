"""Host prefix-cache lookup helpers for prepacked prefill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from batchgen.prefill.prefix_reuse import (
    PrefixReusePrefillPlan,
    build_prefix_reuse_prefill_plan,
)


@dataclass(frozen=True)
class PrefixCachePrefillLookup:
    lookup_results: tuple[object, ...]
    prefix_shared_tokens: tuple[int, ...]

    @property
    def has_hit(self) -> bool:
        return any(tokens > 0 for tokens in self.prefix_shared_tokens)


@dataclass(frozen=True)
class PrefixCachePrefillEstimate:
    prefix_shared_tokens: tuple[int, ...]

    @property
    def has_hit(self) -> bool:
        return any(tokens > 0 for tokens in self.prefix_shared_tokens)


@dataclass(frozen=True)
class PrefixCachePrefillInputs:
    plan: PrefixReusePrefillPlan
    input_ids_list: list[torch.Tensor]
    attention_mask_list: list[torch.Tensor]


def effective_prefix_shared_tokens(
    *, raw_cached_tokens: int, prompt_length: int
) -> int:
    """Normalize coordinator lookup tokens to the compute-path semantic.

    The coordinator reports raw page-cache hits. The prefill compute path always
    runs at least one query token, so an exact full hit becomes a one-token
    extend with ``prompt_length - 1`` cached tokens. After this boundary,
    callers should propagate only the normalized value.
    """

    prompt_len = int(prompt_length)
    cached = int(raw_cached_tokens)
    if prompt_len <= 0:
        raise ValueError(
            f"prompt_length must be positive for prefix lookup, got {prompt_len}"
        )
    if cached < 0 or cached > prompt_len:
        raise ValueError(
            "raw_cached_tokens must be within prompt length: "
            f"cached={cached}, prompt_length={prompt_len}"
        )
    if cached == prompt_len:
        return max(prompt_len - 1, 0)
    return cached


def lookup_prefix_cache_for_prefill(
    *,
    coordinator: object,
    namespace_digest: Sequence[int],
    prompt_token_ids: Sequence[Sequence[int]],
) -> PrefixCachePrefillLookup:
    """Lookup reusable prompt prefixes for a local prefill batch."""

    lookup_results = []
    prefix_shared_tokens = []
    for token_ids in prompt_token_ids:
        result = coordinator.lookup_and_attach(
            list(namespace_digest),
            [int(token_id) for token_id in token_ids],
        )
        lookup_results.append(result)
        prefix_shared_tokens.append(
            effective_prefix_shared_tokens(
                raw_cached_tokens=int(result.common_cached_tokens),
                prompt_length=len(token_ids),
            )
        )

    return PrefixCachePrefillLookup(
        lookup_results=tuple(lookup_results),
        prefix_shared_tokens=tuple(prefix_shared_tokens),
    )


def estimate_prefix_cache_for_prefill(
    *,
    coordinator: object,
    namespace_digest: Sequence[int],
    prompt_token_ids: Sequence[Sequence[int]],
) -> PrefixCachePrefillEstimate:
    """Estimate reusable prefixes without attaching or pinning cache entries."""

    prefix_shared_tokens = []
    for token_ids in prompt_token_ids:
        result = coordinator.estimate_lookup(
            list(namespace_digest),
            [int(token_id) for token_id in token_ids],
        )
        prefix_shared_tokens.append(
            effective_prefix_shared_tokens(
                raw_cached_tokens=int(result.common_cached_tokens),
                prompt_length=len(token_ids),
            )
        )

    return PrefixCachePrefillEstimate(
        prefix_shared_tokens=tuple(prefix_shared_tokens),
    )


def build_prefix_cache_prefill_inputs(
    *,
    local_indices: Sequence[int],
    sequence_ids: Sequence[int],
    input_ids: Sequence[torch.Tensor],
    prompt_lengths: Sequence[int],
    lookup: PrefixCachePrefillLookup,
) -> PrefixCachePrefillInputs:
    """Build suffix-only prepack inputs from prefix lookup results."""

    plan = build_prefix_reuse_prefill_plan(
        local_indices=local_indices,
        sequence_ids=sequence_ids,
        input_ids=input_ids,
        prompt_lengths=prompt_lengths,
        prefix_shared_tokens=lookup.prefix_shared_tokens,
    )
    suffix_inputs = []
    suffix_masks = []
    for suffix_ids in plan.suffix_input_ids:
        suffix = suffix_ids.view(1, -1)
        suffix_inputs.append(suffix)
        suffix_masks.append(torch.ones_like(suffix, dtype=torch.int64))

    return PrefixCachePrefillInputs(
        plan=plan,
        input_ids_list=suffix_inputs,
        attention_mask_list=suffix_masks,
    )


def release_prefix_cache_lookup_attachments(
    *,
    coordinator: object,
    lookup: PrefixCachePrefillLookup,
) -> None:
    """Release lookup attachments after dependent loads are complete."""

    seen_handles: set[int] = set()
    for result in lookup.lookup_results:
        handle = int(result.attachment_handle)
        if handle == 0 or handle in seen_handles:
            continue
        seen_handles.add(handle)
        coordinator.release_attachment(handle)
