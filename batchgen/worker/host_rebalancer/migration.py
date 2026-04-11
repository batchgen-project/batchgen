"""MigrationOp dataclass + M9 stubs.

The full migration planner and executor are deferred to a real
hardware session where POIS can root-cause the unresolved CUDA
migration bug from ``ef71616c`` / ``4b3cdef7`` on the old
``tairan/scheduler-split`` branch. This module defines the shape
that production code will fill in; the M4 BoundaryExecutor already
knows how to route migration ops through here.
"""

from __future__ import annotations

from dataclasses import dataclass

from batchgen.worker.protocols import UUID


@dataclass(frozen=True)
class MigrationOp:
    """One host-KV page migration from one rank to another.

    The ``uuid`` is the owning sequence; the ``page_count`` is the
    number of host pages to move. Execution semantics live in
    :meth:`HostKVRebalancer.execute_migrations`, which is still a stub
    (empty plan = no-op; non-empty plan = NotImplementedError with an
    M9 pointer) until the CUDA migration bug is root-caused on real
    hardware.
    """

    from_rank: int
    to_rank: int
    uuid: UUID
    page_count: int


__all__ = ["MigrationOp"]
