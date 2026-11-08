"""Pure predicates for consecutive prefill phase reuse."""

from __future__ import annotations

from typing import Any, Optional


def is_consecutive_prefill(
    *,
    persistent: bool,
    phase: Optional[str],
    has_model: bool,
    reuse_startup_prefill: bool,
    manager: Any,
    prefill_moe_mode: Any,
) -> bool:
    """Return whether a live prefill instance can be reactivated in place."""
    if persistent or phase != "prefill" or not has_model or reuse_startup_prefill:
        return False
    if not hasattr(manager, "reactivate_prefill_in_place"):
        return False
    if not hasattr(manager, "reactivate_prefill_in_place_applicable"):
        return False
    return bool(
        manager.reactivate_prefill_in_place_applicable(
            prefill_moe_mode=prefill_moe_mode
        )
    )
