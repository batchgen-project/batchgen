"""Prefill utilities for efficient batch processing."""

from .prepack import (
    PrepackMetadata,
    bin_pack_first_fit_decreasing,
    prepack_sequences,
    unpack_outputs,
    unpack_last_token_logits,
    create_block_diagonal_attention_mask,
    get_prepack_stats,
)
from .prefix_reuse import (
    PrefixReusePrefillPlan,
    PrefixReuseSequencePlan,
    build_prefix_reuse_prefill_plan,
    split_prefix_reuse_plan_for_micro_batch,
    validate_prefix_reuse_plan,
)

__all__ = [
    "PrepackMetadata",
    "bin_pack_first_fit_decreasing",
    "prepack_sequences",
    "unpack_outputs",
    "unpack_last_token_logits",
    "create_block_diagonal_attention_mask",
    "get_prepack_stats",
    "PrefixReusePrefillPlan",
    "PrefixReuseSequencePlan",
    "build_prefix_reuse_prefill_plan",
    "split_prefix_reuse_plan_for_micro_batch",
    "validate_prefix_reuse_plan",
]
