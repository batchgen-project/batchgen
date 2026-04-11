"""TorchCollectiveBackend — wraps a `torch.distributed` process group.

Satisfies :class:`batchgen.worker.protocols.CollectiveBackend` by
forwarding every method to ``torch.distributed.*`` with the configured
process group. Lazy imports so the module loads on CPU-only boxes; the
actual distributed calls only fire when invoked on a node with a
process group initialized.
"""

from __future__ import annotations

from typing import Any


class TorchCollectiveBackend:
    """Production adapter for :class:`CollectiveBackend`.

    Constructed once per worker process by the orchestrator entry point
    (``batchgen/worker_reextract_entry.py``) with the worker's
    ``torch.distributed`` process group, rank, and world_size.
    """

    def __init__(self, process_group: Any, rank: int, world_size: int) -> None:
        self._pg = process_group
        self.rank = rank
        self.world_size = world_size

    def all_reduce_max(self, tensor: Any) -> None:
        import torch.distributed as dist

        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=self._pg)

    def all_reduce_min(self, tensor: Any) -> None:
        import torch.distributed as dist

        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self._pg)

    def all_reduce_sum(self, tensor: Any) -> None:
        import torch.distributed as dist

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self._pg)

    def all_gather_tensor(self, tensor_list: list[Any], tensor: Any) -> None:
        import torch.distributed as dist

        dist.all_gather(tensor_list, tensor, group=self._pg)

    def all_gather_into_tensor(self, out: Any, tensor: Any) -> None:
        import torch.distributed as dist

        dist.all_gather_into_tensor(out, tensor, group=self._pg)

    def all_gather_object(self, obj_list: list[Any], obj: Any) -> None:
        import torch.distributed as dist

        dist.all_gather_object(obj_list, obj, group=self._pg)

    def broadcast_tensor(self, tensor: Any, src: int) -> None:
        import torch.distributed as dist

        dist.broadcast(tensor, src=src, group=self._pg)

    def broadcast_object(self, obj_list: list[Any], src: int) -> None:
        import torch.distributed as dist

        dist.broadcast_object_list(obj_list, src=src, group=self._pg)

    def barrier(self) -> None:
        import torch.distributed as dist

        dist.barrier(group=self._pg)


__all__ = ["TorchCollectiveBackend"]
