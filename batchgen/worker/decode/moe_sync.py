"""Phase 2.8.2c — port of ``_decode_initial_moe_sync`` (7924-7940).

One cross-rank ``all_gather_into_tensor`` of per-rank batch sizes +
two ``parallel_manager`` setters (``set_num_tokens_per_rank`` +
``set_rank_token_counts``). Called once per ``run_continuous``
entry so the MoE EP buffer is sized for the max batch before the
first forward pass — without it, a rank with more tokens than the
initial estimate overflows the buffer.

Collective routed through :class:`CollectiveBackend` so the boundary
fuzzer can lock the order. Setters routed through the adapter
passthroughs added in commit 1c.
"""

from __future__ import annotations

import torch

from batchgen.worker.protocols import CollectiveBackend, LegacyInfraBackend
from batchgen.worker.state import WorkerState


def initial_moe_sync(
    state: WorkerState,
    adapter: LegacyInfraBackend,
    collectives: CollectiveBackend,
    *,
    batch: list[int],
) -> None:
    """Size the MoE EP buffer for the max rank batch before decode.

    Steps (legacy 7930-7939):
      1. Build ``local_batch_size`` tensor of shape [1] containing
         ``len(batch)``.
      2. ``all_gather_into_tensor`` → ``all_rank_counts`` of shape
         ``[world_size]``.
      3. If the gathered max > 0, call
         ``adapter.set_num_tokens_per_rank(max)`` +
         ``adapter.set_rank_token_counts(all_rank_counts)``.

    No-op when ``len(batch) == 0`` across all ranks — legacy 7935
    checked ``max_batch_size > 0`` before calling the setters.
    """
    local_batch_size = torch.tensor(
        [len(batch)], dtype=torch.int64, device=state.torch_device,
    )
    all_rank_counts = torch.zeros(
        state.world_size, dtype=torch.int64, device=state.torch_device,
    )
    collectives.all_gather_into_tensor(all_rank_counts, local_batch_size)
    max_batch_size = int(all_rank_counts.max().item())
    if max_batch_size <= 0:
        return
    adapter.set_num_tokens_per_rank(max_batch_size)
    adapter.set_rank_token_counts(all_rank_counts)


__all__ = ["initial_moe_sync"]
