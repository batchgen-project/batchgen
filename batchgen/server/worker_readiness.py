"""Node-local worker readiness signaling."""

import logging
from typing import Any, Optional


def _signal_local_worker_manager_ready(
    ready_event: Optional[Any],
    *,
    local_rank: int,
    global_rank: int,
) -> bool:
    """Signal the node-local launcher after its workers pass the global barrier."""
    if ready_event is None or local_rank != 0:
        return False
    ready_event.set()
    logging.info(
        "Rank %s: Signaled node-local ready event to WorkerManager",
        global_rank,
    )
    return True
