"""Developer-facing flag parsing for the decode CUDA-graph contract.

Per contract §E:

* Production activation is gated solely by the server-side ``--enable-cuda-graph``
  CLI flag (handled by the worker, not this module).
* Mode selection is automatic via ``adapter.advertised_modes()`` — there is
  no mode env var.
* Developer/maintainer facilities (compare, timing, probe-layer, path-log)
  are env-var gated and observability-only. They MUST NOT influence mode
  selection or sampled tokens.

The 6 GLM-5-prefixed mode env vars are deleted (no back-compat shim);
``warn_on_removed_glm5_env_vars()`` emits a one-shot WARNING when any of them
is still set in the environment so users update their scripts.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

from batchgen.cuda_graph.adapter import DebugOpts

logger = logging.getLogger(__name__)


_REMOVED_GLM5_ENV_VARS = (
    "BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH",
    "BATCHGEN_GLM5_LAYER_CUDA_GRAPH",
    "BATCHGEN_GLM5_DSA_CUDA_GRAPH",
    "BATCHGEN_GLM5_DSA_FULL_CUDA_GRAPH",
    "BATCHGEN_GLM5_MOE_CUDA_GRAPH",
    "BATCHGEN_SEGMENTED_GRAPH",
)


def _read_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip() not in ("", "0", "false", "False", "FALSE")


def _read_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        logger.warning("flags: %s=%r is not a float; using default %r", key, val, default)
        return default


def _read_probe_layers(key: str) -> Tuple[int, ...]:
    val = os.environ.get(key)
    if not val:
        return ()
    val = val.strip()
    if val.lower() == "all":
        # The adapter knows the layer count; "all" is a marker the adapter expands.
        return (-1,)
    out = []
    for tok in val.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            logger.warning("flags: %s probe-layer %r is not an int; skipping", key, tok)
    return tuple(out)


@dataclass(frozen=True)
class DecodeGraphFlags:
    """Parsed developer/maintainer env vars.

    Use ``DecodeGraphFlags.from_env()`` once per decode step (cheap — pure
    os.environ reads) and convert to ``DebugOpts`` via ``.to_debug_opts()``.
    """
    compare: bool = False
    compare_fail: bool = False
    compare_atol: float = 1e-2
    compare_rtol: float = 1e-2
    timing: bool = False
    probe_layers: Tuple[int, ...] = ()
    path_log: bool = False
    max_seqlen: Optional[int] = None
    memory_diag: bool = False

    @classmethod
    def from_env(cls) -> "DecodeGraphFlags":
        max_seqlen_raw = os.environ.get("BATCHGEN_DECODE_GRAPH_MAX_SEQLEN")
        try:
            max_seqlen = int(max_seqlen_raw) if max_seqlen_raw else None
        except ValueError:
            logger.warning(
                "flags: BATCHGEN_DECODE_GRAPH_MAX_SEQLEN=%r is not an int; ignoring",
                max_seqlen_raw,
            )
            max_seqlen = None
        return cls(
            compare=_read_bool("BATCHGEN_DECODE_GRAPH_COMPARE"),
            compare_fail=_read_bool("BATCHGEN_DECODE_GRAPH_COMPARE_FAIL"),
            compare_atol=_read_float("BATCHGEN_DECODE_GRAPH_COMPARE_ATOL", 1e-2),
            compare_rtol=_read_float("BATCHGEN_DECODE_GRAPH_COMPARE_RTOL", 1e-2),
            timing=_read_bool("BATCHGEN_DECODE_GRAPH_TIMING"),
            probe_layers=_read_probe_layers("BATCHGEN_DECODE_GRAPH_PROBE_LAYERS"),
            path_log=_read_bool("BATCHGEN_DECODE_GRAPH_PATH_LOG"),
            max_seqlen=max_seqlen,
            memory_diag=_read_bool("BATCHGEN_DECODE_GRAPH_MEMORY_DIAG"),
        )

    def to_debug_opts(self) -> DebugOpts:
        return DebugOpts(
            compare_against_eager=self.compare,
            fail_on_mismatch=self.compare_fail,
            log_path_breadcrumbs=self.path_log,
            timing=self.timing,
            compare_atol=self.compare_atol,
            compare_rtol=self.compare_rtol,
            probe_layers=self.probe_layers,
        )


_warned_glm5_env = False


def warn_on_removed_glm5_env_vars() -> None:
    """One-shot WARNING if any removed BATCHGEN_GLM5_* mode env var is set.

    Called once at worker boot (or first decode step). After the warning
    fires once per process, subsequent calls are no-ops.
    """
    global _warned_glm5_env
    if _warned_glm5_env:
        return
    set_vars = [name for name in _REMOVED_GLM5_ENV_VARS if os.environ.get(name)]
    if set_vars:
        logger.warning(
            "Removed env var(s) %s are set but no longer recognized; "
            "CUDA-graph is now gated by --enable-cuda-graph (see "
            "batchgen_design/cuda_graph/cuda_graph_contract.md §E). "
            "Please update your scripts.",
            ", ".join(set_vars),
        )
    _warned_glm5_env = True


__all__ = [
    "DecodeGraphFlags",
    "warn_on_removed_glm5_env_vars",
]
