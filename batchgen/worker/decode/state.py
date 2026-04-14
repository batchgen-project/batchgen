"""DecodeState — explicit per-run state for the native decode loop.

Replaces the scatter of ``self._`` fields legacy ``decoding_continuous``
carries on ``BatchGenWorker``. Every helper in the native loop takes a
``DecodeState`` argument, mutates it, and hands it to the next step.
That makes the loop's dependency graph visible at a glance and keeps
unit tests deterministic (a fixture can construct a ``DecodeState``
with exactly the starting shape the test wants).

Fields grow additively as Stage 2 helpers are ported. The current
shape covers Stage 2b-2h; Stage 2i wires it into
``DecodeScheduler.run_continuous``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from batchgen.worker.protocols import UUID


@dataclass
class DecodeState:
    """Mutable state threaded through one ``run_continuous`` call.

    Counters and pending-load slots mirror the legacy scoped locals
    in ``decoding_continuous`` (batchgen_worker.py:8495-8602). Holding
    them in one object lets the helpers stay free functions rather
    than methods on the scheduler.
    """

    # Cohort + batch state (grow/shrink across boundary cycles).
    decode_uuids: list[UUID] = field(default_factory=list)
    batch: list[int] = field(default_factory=list)

    # Iteration counters.
    local_iteration: int = 0
    last_boundary: int = 0
    global_batch_size: int = 0

    # Forward-pass outputs.
    new_tokens: Any = None             # torch.Tensor | None
    page_table_verified: bool = True

    # Cumulative counters — persist across prefill/decode switches.
    # Legacy initialised them on ``BatchGenWorker`` via ``hasattr``
    # guards at batchgen_worker.py:7913-7921. Native path sets them
    # up front on DecodeState and reads/writes in helpers.
    cumulative_iterations: int = 0
    cumulative_boundaries: int = 0
    cumulative_boundary_ms: float = 0.0
    cumulative_forward_ms: float = 0.0


__all__ = ["DecodeState"]
