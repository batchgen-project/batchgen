"""Side-effect-free planning helpers for prefix-reuse prefill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class PrefixReuseSequencePlan:
    local_idx: int
    sequence_id: int
    prompt_length: int
    raw_prefix_shared_tokens: int
    prefix_shared_tokens: int
    suffix_start_pos: int
    suffix_length: int
    full_logical_context_length: int
    is_full_hit: bool
    fallback_reason: Optional[str] = None
    attached_shared_tokens: Optional[int] = None


@dataclass(frozen=True)
class PrefixReusePrefillPlan:
    sequences: list[PrefixReuseSequencePlan]
    suffix_input_ids: list[torch.Tensor]
    suffix_position_ids: list[torch.Tensor]
    cache_seqlens: torch.Tensor
    total_prompt_tokens: int
    total_suffix_tokens: int
    saved_prefill_tokens: int


def _normalize_input_ids(
    input_ids: torch.Tensor, prompt_length: int
) -> torch.Tensor:
    if input_ids.dim() == 2:
        if input_ids.size(0) != 1:
            raise ValueError(
                f"2D input_ids must have batch size 1, got shape={tuple(input_ids.shape)}"
            )
        input_ids = input_ids[0]
    elif input_ids.dim() != 1:
        raise ValueError(
            f"input_ids must be 1D or [1, S], got shape={tuple(input_ids.shape)}"
        )
    if prompt_length < 0:
        raise ValueError(
            f"prompt_length must be non-negative, got {prompt_length}"
        )
    if input_ids.numel() < prompt_length:
        raise ValueError(
            f"input_ids length {input_ids.numel()} is shorter than prompt_length {prompt_length}"
        )
    return input_ids[:prompt_length]


def build_prefix_reuse_prefill_plan(
    *,
    local_indices: Sequence[int],
    sequence_ids: Sequence[int],
    input_ids: Sequence[torch.Tensor],
    prompt_lengths: Sequence[int],
    prefix_shared_tokens: Sequence[int],
    device: Optional[torch.device] = None,
    page_size_tokens: Optional[int] = None,
) -> PrefixReusePrefillPlan:
    """Build suffix-only prefill metadata without mutating runtime state."""

    count = len(local_indices)
    if not (
        len(sequence_ids) == count
        and len(input_ids) == count
        and len(prompt_lengths) == count
        and len(prefix_shared_tokens) == count
    ):
        raise ValueError("All input sequences must have the same length")

    plans: list[PrefixReuseSequencePlan] = []
    suffix_input_ids: list[torch.Tensor] = []
    suffix_position_ids: list[torch.Tensor] = []
    cache_seqlens: list[int] = []
    total_prompt_tokens = 0
    total_suffix_tokens = 0
    page_size = None
    if page_size_tokens is not None:
        page_size = int(page_size_tokens)
        if page_size <= 0:
            raise ValueError(
                f"page_size_tokens must be positive, got {page_size}"
            )

    for idx in range(count):
        prompt_length = int(prompt_lengths[idx])
        shared_tokens = int(prefix_shared_tokens[idx])
        prompt_ids = _normalize_input_ids(input_ids[idx], prompt_length)
        if prompt_length <= 0:
            raise ValueError(
                f"prompt_length must be positive for prefix reuse, got {prompt_length}"
            )
        if shared_tokens < 0:
            raise ValueError(
                f"prefix_shared_tokens must be non-negative, got {shared_tokens}"
            )
        if shared_tokens > prompt_length:
            raise ValueError(
                f"prefix_shared_tokens {shared_tokens} exceeds prompt_length {prompt_length}"
            )

        raw_shared_tokens = shared_tokens
        attached_shared_tokens = raw_shared_tokens
        if page_size is not None:
            attached_shared_tokens = (
                raw_shared_tokens // page_size
            ) * page_size
        attached_shared_tokens = min(attached_shared_tokens, prompt_length)

        is_full_hit = raw_shared_tokens == prompt_length
        if is_full_hit and attached_shared_tokens == prompt_length:
            suffix_start = max(prompt_length - 1, 0)
        else:
            suffix_start = attached_shared_tokens
        suffix_length = prompt_length - suffix_start
        target_device = device if device is not None else prompt_ids.device
        suffix_ids = prompt_ids[suffix_start:prompt_length].to(target_device)
        position_ids = torch.arange(
            suffix_start,
            prompt_length,
            dtype=torch.long,
            device=target_device,
        )

        plans.append(
            PrefixReuseSequencePlan(
                local_idx=int(local_indices[idx]),
                sequence_id=int(sequence_ids[idx]),
                prompt_length=prompt_length,
                raw_prefix_shared_tokens=raw_shared_tokens,
                prefix_shared_tokens=suffix_start,
                suffix_start_pos=suffix_start,
                suffix_length=suffix_length,
                full_logical_context_length=prompt_length,
                is_full_hit=is_full_hit,
                attached_shared_tokens=attached_shared_tokens,
            )
        )
        suffix_input_ids.append(suffix_ids)
        suffix_position_ids.append(position_ids)
        cache_seqlens.append(suffix_start)
        total_prompt_tokens += prompt_length
        total_suffix_tokens += suffix_length

    cache_device = device if device is not None else torch.device("cpu")
    return PrefixReusePrefillPlan(
        sequences=plans,
        suffix_input_ids=suffix_input_ids,
        suffix_position_ids=suffix_position_ids,
        cache_seqlens=torch.tensor(
            cache_seqlens, dtype=torch.int32, device=cache_device
        ),
        total_prompt_tokens=total_prompt_tokens,
        total_suffix_tokens=total_suffix_tokens,
        saved_prefill_tokens=total_prompt_tokens - total_suffix_tokens,
    )


def split_prefix_reuse_plan_for_micro_batch(
    plan: PrefixReusePrefillPlan,
    seq_start: int,
    seq_end: int,
) -> PrefixReusePrefillPlan:
    if seq_start < 0 or seq_end < seq_start or seq_end > len(plan.sequences):
        raise ValueError(
            f"Invalid micro-batch range [{seq_start}, {seq_end}) for "
            f"{len(plan.sequences)} sequences"
        )
    sequences = plan.sequences[seq_start:seq_end]
    suffix_input_ids = plan.suffix_input_ids[seq_start:seq_end]
    suffix_position_ids = plan.suffix_position_ids[seq_start:seq_end]
    cache_seqlens = plan.cache_seqlens[seq_start:seq_end].clone()
    total_prompt_tokens = sum(item.prompt_length for item in sequences)
    total_suffix_tokens = sum(item.suffix_length for item in sequences)
    return PrefixReusePrefillPlan(
        sequences=list(sequences),
        suffix_input_ids=list(suffix_input_ids),
        suffix_position_ids=list(suffix_position_ids),
        cache_seqlens=cache_seqlens,
        total_prompt_tokens=total_prompt_tokens,
        total_suffix_tokens=total_suffix_tokens,
        saved_prefill_tokens=total_prompt_tokens - total_suffix_tokens,
    )
