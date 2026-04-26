"""Process-local runtime debug flags supplied by BatchGen batch requests."""

from __future__ import annotations

from typing import Any, Dict, Optional


_current_batchgen_debug: Dict[str, Any] = {}


def set_current_batchgen_debug(options: Optional[Dict[str, Any]]) -> None:
    global _current_batchgen_debug
    _current_batchgen_debug = dict(options or {})


def get_current_batchgen_debug() -> Dict[str, Any]:
    return _current_batchgen_debug


def get_glm5_moe_mode() -> str:
    return str(_current_batchgen_debug.get("glm5_moe_mode") or "graph")


def glm5_dispatch_headroom_diag_enabled(default: bool = False) -> bool:
    return bool(_current_batchgen_debug.get("glm5_dispatch_headroom_diag", default))


def glm5_dispatch_headroom_warn_frac(default: float = 0.90) -> float:
    value = _current_batchgen_debug.get("glm5_dispatch_headroom_warn_frac")
    return float(default if value is None else value)
