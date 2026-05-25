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
class PrefixCachePrefillInputs:
    plan: PrefixReusePrefillPlan
    input_ids_list: list[torch.Tensor]
    attention_mask_list: list[torch.Tensor]


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
        prefix_shared_tokens.append(int(result.common_cached_tokens))

    return PrefixCachePrefillLookup(
        lookup_results=tuple(lookup_results),
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
        handle = int(getattr(result, "attachment_handle", 0))
        if handle == 0 or handle in seen_handles:
            continue
        seen_handles.add(handle)
        coordinator.release_attachment(handle)
