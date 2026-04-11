"""Shared state dataclass for the worker handler package.

`WorkerState` is a pure data container — no logic, no lazy init, no collectives.
Every handler in `batchgen.worker.*` receives the same `WorkerState` instance via
constructor and reads/writes through it. The dataclass grows additively as each
handler slice lands (M1 starts with rank identity; M2 adds collective state; M3
adds KV managers, etc.).

Design contract:
  - No method does I/O, NCCL calls, or lazy imports.
  - `__post_init__` validates only field-level invariants that are cheap and
    always-true (non-negativity, rank bounds). Cross-field invariants that
    change over the worker lifetime (e.g. CTX invariant) live in handler
    guards, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from batchgen.sequence import SequenceBatch


@dataclass
class WorkerState:
    """Shared per-rank state for the worker handler package.

    Fields grow additively as each handler slice lands. M1-S1 added the rank
    identity group; M1-S4 (IndexManager) adds the index-map group and a
    default `global_batch`.

    Attributes:
        rank: Global rank (alias for `global_rank` on the legacy worker).
        local_rank: Rank within the local node; indexes into per-node GPUs.
        world_size: Total number of ranks across all nodes.
        device: Physical GPU index on the local node.
        torch_device: `torch.device` handle used by every tensor op on
            this worker. Tests use `torch.device("cpu")`; production uses
            `torch.device(f"cuda:{device}")`.
        global_batch: The authoritative `SequenceBatch` container. All
            handlers read sequences through this object.
        local_to_uuid_map: Local-index → UUID. Owned by `IndexManager`.
        uuid_to_local_map: UUID → local-index. Owned by `IndexManager`.
        free_local_indices: Set of freed local indices available for reuse
            before `next_local_idx` grows. Owned by `IndexManager`; kept in
            lockstep with `QueryBookBufferPool._free_slots` on admission and
            completion (see plan Decision #7).
        next_local_idx: Monotonic counter for the next fresh local index
            when `free_local_indices` is empty. Owned by `IndexManager`.
    """

    rank: int
    local_rank: int
    world_size: int
    device: int
    torch_device: torch.device

    global_batch: SequenceBatch = field(default_factory=SequenceBatch)
    local_to_uuid_map: dict[int, str] = field(default_factory=dict)
    uuid_to_local_map: dict[str, int] = field(default_factory=dict)
    free_local_indices: set[int] = field(default_factory=set)
    next_local_idx: int = 0

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"rank must be in [0, {self.world_size}), got {self.rank}"
            )
        if self.local_rank < 0 or self.local_rank >= self.world_size:
            raise ValueError(
                f"local_rank must be in [0, {self.world_size}), got {self.local_rank}"
            )
        if self.device < 0:
            raise ValueError(f"device must be >= 0, got {self.device}")
        if self.next_local_idx < 0:
            raise ValueError(
                f"next_local_idx must be >= 0, got {self.next_local_idx}"
            )
