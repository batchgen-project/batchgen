"""Lightweight step-by-step tracing for the GLM-5 prefill hot path.

Gate: BATCHGEN_GLM5_PREFILL_TRACE=1. Off by default. When on, emits one
enter/exit line per instrumented step and one skip line per guard that
short-circuits a hot-path op. Grep-friendly single-line format.

Complements (does NOT replace) the existing BATCHGEN_GLM5_PREFILL_DIAG
probes. DIAG reads tensor values (heavy, holds tensor Python-refs past
sync). TRACE only reads shape/dtype/device/numel + wall-clock (cheap,
safe to leave on).

Design motivation: a latent `if hasattr(self.module, 'indexer')` gate
silently dropped the indexer off the GLM-5 prefill hot path for ~7 days
of investigation. With TRACE enabled, that would have produced a
`step=indexer_path action=skip reason=indexer_attr_missing` line on the
first run. Guard-aware skip logging is the load-bearing feature here.
"""

import logging
import os
import time

_ENV = "BATCHGEN_GLM5_PREFILL_TRACE"
_ON = os.environ.get(_ENV, "0") == "1"
_log = logging.getLogger("batchgen.prefill.trace")


def enabled() -> bool:
    return _ON


def emit(rank, layer, step: str, action: str, **kv) -> None:
    if not _ON:
        return
    parts = [f"rank={rank}", f"layer={layer}", f"step={step}", f"action={action}"]
    for k, v in kv.items():
        if v is None:
            continue
        parts.append(f"{k}={v}")
    _log.warning("[PREFILL-STEP " + " ".join(parts) + "]")


class Span:
    """Context-manager that emits `action=enter` on __enter__ and
    `action=exit ms=<wallclock>` on __exit__. Cheap when trace is off:
    no-ops both sides and takes no timestamp.
    """

    __slots__ = ("rank", "layer", "step", "kw", "_t0")

    def __init__(self, rank, layer, step: str, **kw):
        self.rank = rank
        self.layer = layer
        self.step = step
        self.kw = kw
        self._t0 = 0.0

    def __enter__(self):
        if _ON:
            emit(self.rank, self.layer, self.step, "enter", **self.kw)
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if _ON:
            ms = round((time.perf_counter() - self._t0) * 1000.0, 3)
            emit(self.rank, self.layer, self.step, "exit", ms=ms, **self.kw)
        return False


def skip(rank, layer, step: str, reason: str, **kw) -> None:
    """Log that a guard short-circuited a hot-path op."""
    emit(rank, layer, step, "skip", reason=reason, **kw)
