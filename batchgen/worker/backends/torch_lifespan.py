"""TorchLifespanLogger — wraps ``batchgen/lifespan.py`` SeqEvent logging.

The lifespan module on main adds ``_lifespan_log`` / ``log_event`` to
each :class:`SequenceEntry`. The adapter forwards every
:meth:`log` call through to ``seq.log_event(event, rank, detail)``.

Env-gated: if ``BATCHGEN_SEQ_LIFESPAN`` is not ``"1"`` on main, the
lifespan module's ``log_event`` is a no-op. The adapter does not
re-check the env var — main's gating is sufficient.
"""

from __future__ import annotations

from typing import Any


class TorchLifespanLogger:
    """Production adapter for :class:`LifespanLoggerBackend`."""

    def __init__(self, rank: int) -> None:
        self._rank = rank

    def log(self, seq: Any, event: Any, detail: dict[str, Any]) -> None:
        log_event = getattr(seq, "log_event", None)
        if log_event is None:
            return
        # Main's log_event signature: log_event(event: int, rank: int, detail: str)
        # The detail dict is flattened to a compact str so the existing
        # lifespan dump format stays unchanged.
        log_event(int(event) if not isinstance(event, int) else event, self._rank, _flatten(detail))


def _flatten(detail: dict[str, Any]) -> str:
    if not detail:
        return ""
    parts = []
    for k in sorted(detail):
        parts.append(f"{k}={detail[k]}")
    return " ".join(parts)


__all__ = ["TorchLifespanLogger"]
