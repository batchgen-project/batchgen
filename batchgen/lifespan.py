"""Sequence lifespan monitoring for debugging KV cache corruption.

Enabled via BATCHGEN_SEQ_LIFESPAN=1 environment variable.
When disabled, all operations are no-ops with zero overhead.

Each sequence maintains a ring buffer of state transition events.
On anomaly (CTX_MISMATCH, REPETITION, non-stop completion),
the full lifespan is dumped to log and /tmp/ for post-mortem.
"""

import enum
import logging
import os
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("BATCHGEN_SEQ_LIFESPAN", "0") == "1"
MAX_EVENTS = 256

# Session start time — set once at import, used for relative timestamps
_SESSION_START = time.perf_counter()


class SeqEvent(enum.IntEnum):
    # State transitions
    CREATED       = 0
    PREFILL_START = 1
    PREFILL_DONE  = 2
    DECODE_START  = 3
    ON_HOLD       = 4
    EVICTED       = 5
    REENTRY_START = 6
    COMPLETED     = 7

    # KV operations
    KV_LOAD_START = 10
    KV_LOAD_DONE  = 11
    KV_APPEND     = 12

    # Migration
    MIGRATE_SEND  = 20
    MIGRATE_RECV  = 21

    # Validation
    CTX_REPAIR    = 30
    PAGE_REBUILD  = 31

    # Anomalies
    REPETITION    = 40
    CTX_MISMATCH  = 41


class SeqEventRecord:
    __slots__ = (
        'event', 'timestamp', 'rank', 'decoded_length',
        'current_ctx', 'expected_ctx', 'gpu_pages', 'host_pages',
        'detail',
    )

    def __init__(
        self,
        event: int,
        rank: int,
        decoded_length: int,
        current_ctx: int,
        expected_ctx: int,
        gpu_pages: int,
        host_pages: int,
        detail: str = "",
    ):
        self.event = event
        self.timestamp = time.perf_counter() - _SESSION_START
        self.rank = rank
        self.decoded_length = decoded_length
        self.current_ctx = current_ctx
        self.expected_ctx = expected_ctx
        self.gpu_pages = gpu_pages
        self.host_pages = host_pages
        self.detail = detail


def format_lifespan(
    uuid: str,
    global_idx: int,
    events: List[SeqEventRecord],
    trigger: str,
) -> str:
    """Format a lifespan log into a human-readable string."""
    lines = [
        f"LIFESPAN DUMP [{trigger}] uuid={uuid} gid={global_idx} "
        f"({len(events)} events):"
    ]
    for i, ev in enumerate(events):
        flag = " *** CTX MISMATCH ***" if ev.current_ctx != ev.expected_ctx else ""
        try:
            name = SeqEvent(ev.event).name
        except ValueError:
            name = f"UNKNOWN({ev.event})"
        lines.append(
            f"  [{i:3d}] t={ev.timestamp:8.3f}s {name:16s} "
            f"rank={ev.rank} dec={ev.decoded_length:6d} "
            f"ctx={ev.current_ctx:6d} exp={ev.expected_ctx:6d} "
            f"gpu_pg={ev.gpu_pages:3d} host_pg={ev.host_pages:3d} "
            f"{ev.detail}{flag}"
        )
    return "\n".join(lines)


def dump_lifespan(
    uuid: str,
    global_idx: int,
    events: List[SeqEventRecord],
    trigger: str,
) -> None:
    """Dump lifespan to log and /tmp/ file."""
    if not ENABLED or not events:
        return
    msg = format_lifespan(uuid, global_idx, events, trigger)
    logger.warning(msg)
    try:
        path = f"/tmp/batchgen_lifespan_{uuid}.log"
        with open(path, "w") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def has_ctx_mismatch(events: List[SeqEventRecord]) -> bool:
    """Check if any event in the log is a CTX_MISMATCH."""
    for ev in events:
        if ev.event == SeqEvent.CTX_MISMATCH:
            return True
    return False
