"""Events that must not happen once a worker has reported ready.

A worker reports ready when every deliberate kernel compile and CUDA-graph
capture has run. A compile or a capture after that point costs a request its
latency and means the warm-up missed a shape, so each one is written to the
log as one line with a fixed prefix and counted per process:

    [POST_READINESS] rank=3 kind=graph_capture count=1 total=1 where=...

``arm`` is called once, as the rank reports ready; ``record`` is a no-op
before that, so the same call sites serve the warm-up itself. The line is the
contract with result recorders: a run whose log holds an event line is not an
official result. Whether an event also raises is the caller's policy; this
module only records.
"""

from __future__ import annotations

import collections
import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MARKER = "[POST_READINESS]"
TRITON_COMPILE = "triton_compile"
DEEPGEMM_COMPILE = "deepgemm_compile"
GRAPH_CAPTURE = "graph_capture"
KINDS = (TRITON_COMPILE, DEEPGEMM_COMPILE, GRAPH_CAPTURE)

_lock = threading.Lock()
_armed = False
_rank: Optional[int] = None
_counts: "collections.Counter[str]" = collections.Counter()


def arm(rank) -> None:
    """The rank reports ready: every later ``record`` is an event."""
    global _armed, _rank
    with _lock:
        already = _armed
        _armed = True
        _rank = rank
    if not already:
        logger.info("%s rank=%s armed: compiles and graph captures from here "
                    "on are recorded as events", MARKER, rank)


def armed() -> bool:
    return _armed


def record(kind: str, where: str, count: int = 1) -> bool:
    """Record ``count`` events of ``kind`` seen at ``where``.

    Returns False, and records nothing, before ``arm``.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown post-readiness event kind {kind!r}; known: {KINDS}")
    if count <= 0:
        return False
    with _lock:
        if not _armed:
            return False
        _counts[kind] += count
        total = _counts[kind]
        rank = _rank
    logger.warning("%s rank=%s kind=%s count=%d total=%d where=%s",
                   MARKER, rank, kind, count, total, " ".join(str(where).split()))
    return True


def counts() -> Dict[str, int]:
    """Events recorded since ``arm``, by kind (every kind present)."""
    with _lock:
        return {kind: int(_counts.get(kind, 0)) for kind in KINDS}


def _reset_for_tests() -> None:
    global _armed, _rank
    with _lock:
        _armed = False
        _rank = None
        _counts.clear()
