"""Helpers for OpenAI-compatible token usage reporting."""

from __future__ import annotations

from typing import Any, Dict

from batchgen.server.io_struct import PromptTokensDetails, Usage


def build_usage(
    prompt_tokens: Any,
    completion_tokens: Any,
    cached_tokens: Any = 0,
) -> Usage:
    """Build a usage model with normalized cached prompt token count."""
    prompt_count = _non_negative_int(prompt_tokens)
    completion_count = _non_negative_int(completion_tokens)
    cached_count = min(_non_negative_int(cached_tokens), prompt_count)
    return Usage(
        prompt_tokens=prompt_count,
        completion_tokens=completion_count,
        total_tokens=prompt_count + completion_count,
        prompt_tokens_details=PromptTokensDetails(
            cached_tokens=cached_count
        ),
    )


def build_usage_dict(
    prompt_tokens: Any,
    completion_tokens: Any,
    cached_tokens: Any = 0,
) -> Dict[str, Any]:
    usage = build_usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
    )
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return usage.dict()


def _non_negative_int(value: Any) -> int:
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return 0
