"""Collective-safe resource cleanup for cancelled pool batches."""

from __future__ import annotations

import torch
import torch.distributed as dist

from batchgen.query_book import release_local_query_slot
from batchgen.sequence import SequenceStatus


def defer_batch_cancellation(worker, message: dict) -> None:
    """Save a collective cancellation until decode-local state is retired."""
    worker._deferred_pool_cancellations.append(message)


def apply_deferred_batch_cancellations(worker) -> bool:
    """Apply every cancellation deferred by the current decode interval."""
    messages = worker._deferred_pool_cancellations
    worker._deferred_pool_cancellations = []
    for message in messages:
        cancel_batch_sequences(worker, message["batch_id"])
    return bool(messages)


def settle_pending_decode_load(
    worker,
    *,
    pending_async_task,
    pending_load_uuids,
    pending_load_local,
    pending_load_global,
    decode_uuids,
    batch,
    gpu_manager,
):
    """Finish a boundary-launched load before leaving for cancellation."""
    if not pending_load_uuids:
        return decode_uuids, batch
    if pending_async_task is not None:
        pending_async_task.wait()
        torch.cuda.synchronize(worker.torch_device)
    dist.barrier()
    return worker._finalize_async_load_minimal(
        pending_async_task,
        pending_load_uuids,
        pending_load_local,
        pending_load_global,
        decode_uuids,
        batch,
        gpu_manager,
    )


def pause_decode_for_cancellation(worker, decode_uuids) -> None:
    """Retire live decode state so surviving batches can re-enter safely."""
    worker._wait_pending_kv_append_tasks(sync_distributed_errors=True)
    worker._put_sequences_on_hold(decode_uuids)


def cancel_batch_sequences(worker, batch_id: str) -> list[str]:
    """Release live sequences for *batch_id* at a worker decision boundary."""
    cancelled = sorted(
        (
            seq.uuid for seq in worker.global_batch
            if seq.batch_id == batch_id and seq.status != SequenceStatus.COMPLETED
        ),
        key=lambda uuid: worker.global_batch.get_sequence(uuid).global_idx,
    )
    local = [uuid for uuid in cancelled if uuid in worker._uuid_to_local_map]
    gpu = [uuid for uuid in local if uuid in worker._sequences_with_gpu_kv]
    host = [
        uuid for uuid in local
        if worker.global_batch.get_sequence(uuid).host_pages_allocated > 0
    ]
    missing_host_view = int(
        bool(host)
        and getattr(worker.core_engine, "host_paged_kv_worker_view", None) is None
    )
    if getattr(worker, "world_size", 1) > 1:
        missing_tensor = torch.tensor(
            [missing_host_view], dtype=torch.int32, device=worker.torch_device
        )
        dist.all_reduce(missing_tensor, op=dist.ReduceOp.MAX)
        missing_host_view = int(missing_tensor.item())
    if missing_host_view:
        raise RuntimeError(
            "Cannot cancel sequences with allocated Host KV: worker view unavailable"
        )
    if gpu:
        worker._release_gpu_kv_pages(worker._get_local_indices_for_uuids(gpu))

    kda_only = [
        uuid for uuid in local if uuid not in gpu
        and worker.global_batch.get_sequence(uuid).status
        in {SequenceStatus.PREFILLED, SequenceStatus.IN_DECODE, SequenceStatus.ON_HOLD}
    ]
    if kda_only:
        worker._release_kda_state_slots([
            worker.global_batch.get_sequence(uuid).global_idx
            for uuid in kda_only
        ])

    if host:
        worker._release_host_kv_pages_for_batch(host)

    for uuid in cancelled:
        seq = worker.global_batch.get_sequence(uuid)
        if seq is None:
            continue
        if getattr(worker, "_buffer_pool", None) is not None and seq._buffer_slot >= 0:
            worker._buffer_pool.free_slot(seq._buffer_slot)
            seq._buffer_slot = -1
        release_local_query_slot(
            uuid,
            uuid_to_local_map=worker._uuid_to_local_map,
            local_to_uuid_map=worker._local_to_uuid_map,
            query_book=worker.query_book,
            free_local_indices=worker._free_local_indices,
        )
        worker._sequences_with_gpu_kv.discard(uuid)
        worker.global_batch.remove_sequence(uuid)

    worker.num_global_queries = len(worker.global_batch)
    worker.num_local_queries = len(
        worker.global_batch.get_sequences_for_rank(worker.rank)
    )

    if worker.rank == 0 and worker._response_queue is not None:
        worker._response_queue.put({
            "type": "batch_cancelled",
            "batch_id": batch_id,
            "request_ids": cancelled,
        })
    return cancelled
