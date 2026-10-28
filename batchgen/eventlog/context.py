# ---------------------------------------------------------------------------- #
#  BatchGen                                                                     #
#  Copyright (c) 2025-2026 BatchGen Team                                        #
#  Licensed under the Apache License, Version 2.0                               #
# ---------------------------------------------------------------------------- #
"""Per-thread / per-task log context: ``batch_id``, ``wave`` and ``step``.

Values live in ``contextvars``, so threads and asyncio tasks each see their
own. :class:`ContextFilter` copies them onto the record on the caller's
thread, before the record is queued for the listener thread.
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
from collections.abc import Iterator

from . import schema

_VARS: dict[str, contextvars.ContextVar] = {
    "batch_id": contextvars.ContextVar("bg_batch_id", default=None),
    "wave": contextvars.ContextVar("bg_wave", default=None),
    "step": contextvars.ContextVar("bg_step", default=None),
}
_TYPES = {"batch_id": str, "wave": int, "step": int}
CONTEXT_FIELDS = tuple(_VARS)

_BATCH_ID = _VARS["batch_id"]
_WAVE = _VARS["wave"]
_STEP = _VARS["step"]


def _check(fields: dict[str, object]) -> None:
    for name, value in fields.items():
        if name not in _VARS:
            raise ValueError(
                f"unknown log context field {name!r}; "
                f"expected one of {CONTEXT_FIELDS}"
            )
        if value is not None and not isinstance(value, _TYPES[name]):
            raise TypeError(
                f"log context field {name!r} must be "
                f"{_TYPES[name].__name__} or None, got {type(value).__name__}"
            )


@contextlib.contextmanager
def bind(**fields: object) -> Iterator[None]:
    """Set context fields for the ``with`` body; previous values restored."""
    _check(fields)
    tokens = [(_VARS[name], _VARS[name].set(v)) for name, v in fields.items()]
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


def set_context(**fields: object) -> None:
    _check(fields)
    for name, value in fields.items():
        _VARS[name].set(value)


def clear_context(*names: str) -> None:
    """Unset the named fields, or every field when none are named."""
    _check(dict.fromkeys(names))
    for name in names or CONTEXT_FIELDS:
        _VARS[name].set(None)


def current() -> dict[str, object]:
    """The fields set in the calling thread/task."""
    values = {name: var.get() for name, var in _VARS.items()}
    return {k: v for k, v in values.items() if v is not None}


class ContextFilter(logging.Filter):
    """Stamp the caller's context fields on the record; never drops it."""

    def filter(self, record: logging.LogRecord) -> bool:
        setattr(record, schema.ATTR_BATCH_ID, _BATCH_ID.get())
        setattr(record, schema.ATTR_WAVE, _WAVE.get())
        setattr(record, schema.ATTR_STEP, _STEP.get())
        return True
