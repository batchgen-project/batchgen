"""Phase 5 of the boundary cycle — page-table rebuild + MoE sync + watermark check.

Ports legacy ``_boundary_finalize`` (batchgen_worker.py:7224-7333) into
the native boundary sub-package. Each step maps 1:1 to a legacy line
range, with the debug-only logging dropped (native path keeps the
essential work, leaves observability to higher-level metrics).

Collectives: one ``all_gather_into_tensor`` + one ``barrier`` per call,
matching legacy.
"""

from __future__ import annotations

from typing import Any

import torch

from batchgen.worker.protocols import UUID, CollectiveBackend, LegacyInfraBackend
from batchgen.worker.state import WorkerState


def finalize(
    state: WorkerState,
    adapter: LegacyInfraBackend,
    collectives: CollectiveBackend,
    *,
    decode_uuids: list[UUID],
    batch: list[int],
    gpu_manager: Any,
) -> tuple[list[int], bool]:
    """Run the final boundary steps + return the watermark state.

    Steps (legacy order, batchgen_worker.py:7224-7333):
      1. ``adapter.rebuild_page_table_for_batch(batch, gpu_manager)`` —
         materialize the page table for the post-boundary batch.
      2. Cross-rank gather of local batch sizes (one
         ``all_gather_into_tensor`` of shape [world_size]). Used to
         size the MoE EP buffer.
      3. When the gathered max is non-zero and the parallel manager
         has the MoE setters, call
         ``adapter.set_num_tokens_per_rank(max)`` +
         ``adapter.set_rank_token_counts(counts_tensor)``.
      4. ``collectives.barrier()`` — every rank reaches this line
         before the boundary cycle returns.
      5. Batch consistency verify + repair. If ``set(batch)`` differs
         from ``adapter.get_local_indices_for_uuids(decode_uuids)``,
         fix ``batch`` and rebuild the page table again.
      6. Final shape check: the GPU page table's first dim must match
         ``len(batch)``; a mismatch is an assertion failure because
         legacy's Phase 2.5 elevated this to a hard-fail.
      7. ``adapter.check_host_kv_watermark_trigger()`` → returned as
         the ``watermark_triggered`` bool.

    Returns:
        ``(batch, watermark_triggered)``. The batch may have been
        repaired in step 5; the caller must thread it back through
        ``BoundaryResult``.
    """
    # Step 1 — page-table rebuild for the final batch.
    adapter.rebuild_page_table_for_batch(batch, gpu_manager)

    # Step 2 — gather per-rank batch sizes via one all_gather_into_tensor.
    all_rank_counts = torch.zeros(
        state.world_size, dtype=torch.int64, device=state.torch_device
    )
    local_batch_size = torch.tensor(
        [len(batch)], dtype=torch.int64, device=state.torch_device
    )
    collectives.all_gather_into_tensor(all_rank_counts, local_batch_size)
    max_batch_size = int(all_rank_counts.max().item())

    # Step 3 — feed the parallel manager. The adapter bodies (commit 1c)
    # no-op gracefully when the setters are absent on older worker
    # builds, so callers don't need feature detection here.
    if max_batch_size > 0:
        adapter.set_num_tokens_per_rank(max_batch_size)
        adapter.set_rank_token_counts(all_rank_counts)

    # Step 4 — cross-rank barrier before returning.
    collectives.barrier()

    # Step 5 — batch consistency verify + repair.
    expected_local = adapter.get_local_indices_for_uuids(decode_uuids)
    if set(batch) != set(expected_local):
        batch = expected_local
        adapter.rebuild_page_table_for_batch(batch, gpu_manager)

    # Step 6 — final page-table shape check. Raises AssertionError on
    # mismatch (Phase 2.5 discipline: fail loudly, not silently).
    _assert_page_table_shape(gpu_manager, batch)

    # Step 7 — watermark check.
    watermark_triggered = bool(adapter.check_host_kv_watermark_trigger())
    return batch, watermark_triggered


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_page_table_shape(gpu_manager: Any, batch: list[int]) -> None:
    """Raise ``AssertionError`` when the GPU page table's row count does
    not match the batch size. Ported from legacy 7290-7297 but elevated
    from log+continue to a hard-fail per Phase 2.5 (commit c9f2dcd2).
    """
    if gpu_manager is None or not getattr(gpu_manager, "is_initialized", False):
        return
    mgr = getattr(gpu_manager, "_gpu_page_table_manager", None)
    if mgr is None:
        return
    gpu_table = getattr(mgr, "gpu_table", None)
    if gpu_table is None:
        return
    if not batch:
        return
    expected = len(batch)
    actual = gpu_table.shape[0]
    if actual != expected:
        raise AssertionError(
            f"boundary.finalize: page table row count {actual} does not "
            f"match batch size {expected}. The rebuild step failed to "
            f"materialize the expected layout — this is a hard-fail per "
            f"Phase 2.5 (no silent page-table drift)."
        )


__all__ = ["finalize"]
