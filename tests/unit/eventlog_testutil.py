"""Shared helpers for the ``batchgen.eventlog`` unit tests.

``batchgen/__init__.py`` imports torch-dependent code, so the event-log tests
load ``batchgen.eventlog`` under a stub ``batchgen`` package whose
``__path__`` points at the source tree. The stub is removed again after the
import so it never leaks into other test modules collected later.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
import sys
import types
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parent

RUN_ID = "7f2a9c1e" + "0123456789abcdef" + "01234567"
RUN_STARTED_AT = datetime(2026, 10, 27, 14, 2, 5, tzinfo=timezone.utc)
RUN_DIR = "20261027-140205-7f2a9c1e"


def import_eventlog():
    """Import ``batchgen.eventlog`` without executing ``batchgen/__init__``."""
    cached = sys.modules.get("batchgen.eventlog")
    if cached is not None:
        return cached
    if "batchgen" in sys.modules:
        return importlib.import_module("batchgen.eventlog")
    stub = types.ModuleType("batchgen")
    stub.__path__ = [str(REPO_ROOT / "batchgen")]
    stub.__package__ = "batchgen"
    sys.modules["batchgen"] = stub
    try:
        return importlib.import_module("batchgen.eventlog")
    finally:
        if sys.modules.get("batchgen") is stub:
            del sys.modules["batchgen"]


@contextlib.contextmanager
def isolated_logging(ev) -> Iterator[None]:
    """Restore root logging, context and renderers around one test."""
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    renderers = dict(ev.console._RENDERERS)
    try:
        yield
    finally:
        ev.shutdown()
        ev.clear_context()
        ev.console._RENDERERS.clear()
        ev.console._RENDERERS.update(renderers)
        # Only undo what configure() did: drop leftover queue handlers and
        # restore pre-test handlers, except pytest's per-phase capture
        # handlers, which pytest itself adds and removes.
        for handler in list(root.handlers):
            if isinstance(handler, ev.handlers.EventQueueHandler):
                root.removeHandler(handler)
        for handler in handlers:
            if (
                handler not in root.handlers
                and type(handler).__module__ != "_pytest.logging"
            ):
                root.addHandler(handler)
        root.setLevel(level)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def rank_child_main(log_dir: str, rank: int, count: int) -> None:
    """Target of the ``spawn`` test: one rank process of one run."""
    ev = import_eventlog()
    ev.configure(
        "rank",
        node=0,
        rank=rank,
        local_rank=rank,
        run_id=RUN_ID,
        run_started_at=RUN_STARTED_AT,
        log_dir=log_dir,
        console=False,
    )
    logger = ev.get_logger("batchgen.worker")
    for index in range(count):
        ev.log_event(logger, "worker.test.tick", rank=rank, index=index)
    if not ev.flush(10.0):
        raise RuntimeError("eventlog flush timed out")
    ev.shutdown()
