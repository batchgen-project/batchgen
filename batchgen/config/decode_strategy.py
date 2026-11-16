"""Decode strategy normalization and selection for one worker process.

Planning calls ``apply_normalized_attn_mode`` before any consumer reads the
mode. It owns validation; a model family that supports only one decode path
declares that as its planner's forced mode (behind the registration seam),
and this owner applies it after validation. No model names appear here.
Decode entry calls ``strategy_for`` once and passes that strategy to the
legacy loop. Neither function adds per-step config reads.

The mode and model type are process config shared across ranks. These
selection inputs must stay rank-uniform, independent of local batch rows.
Imports: stdlib only (no torch, no batchgen or model imports).
"""

from enum import IntEnum
import logging


_logger = logging.getLogger(__name__)


class DecodeStrategy(IntEnum):
    """Historical numeric decode codes, retained for explicit selection.

    The CPU_ATTN label predates the current GPU model forward in mode 0;
    native attention code also used different meanings for these numbers.
    Modes 0/1/2 are currently non-executable: the legacy loops do not bind
    current wrapper state, and mode 2 is a stub. Restoring them is separate
    work. DECOUPLED_GPU is the deployed ``decoding_continuous`` path.
    """

    CPU_ATTN = 0
    LEGACY_MICROBATCH_GPU = 1
    HYBRID_CPU_GPU = 2
    DECOUPLED_GPU = 3


def normalize_attn_mode(raw_mode: int, forced_mode: int | None = None) -> int:
    """Validate the configured code, then apply the family's forced mode."""
    if raw_mode not in {0, 1, 2, 3}:
        raise ValueError(f"Invalid attn_mode {raw_mode}; valid codes are {{0, 1, 2, 3}}")
    if forced_mode is not None:
        if forced_mode not in {0, 1, 2, 3}:
            raise ValueError(
                f"Invalid forced decode mode {forced_mode}; valid codes are {{0, 1, 2, 3}}"
            )
        return forced_mode
    return raw_mode


def apply_normalized_attn_mode(
    basic_config, model_type: str | None, forced_mode: int | None = None
) -> int:
    """Write and log the effective mode; reapplying writes the same value."""
    raw_mode = basic_config.attn_mode
    effective_mode = normalize_attn_mode(raw_mode, forced_mode)
    basic_config.attn_mode = effective_mode
    _logger.info(
        "decode strategy: raw=%d effective=%d model_type=%s",
        raw_mode, effective_mode, model_type,
    )
    return effective_mode


def strategy_for(mode: int) -> DecodeStrategy:
    """Resolve a normalized code, rejecting unknown values at decode entry."""
    return DecodeStrategy(mode)
