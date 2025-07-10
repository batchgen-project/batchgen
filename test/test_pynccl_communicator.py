import os
import time
import traceback
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from moe_gen.distributed.device_communicators.pynccl import PyNcclCommunicator


def setup_dist(rank, world_size, backend="gloo"):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(12355 + int(time.time()) % 1000)
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )


def cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def _test_nccl_worker(rank, world_size, return_dict):
    try:
        setup_dist(rank, world_size, backend="gloo")
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")

        gloo_group = dist.new_group(backend="gloo")
        comm = PyNcclCommunicator(group=gloo_group, device=device)

        with comm.change_state(enable=True):
            # all_reduce
            t = torch.tensor([rank + 1.0], device=device)
            comm.all_reduce(t)
            expected = sum(range(1, world_size + 1))
            comm.stream.synchronize()
            assert torch.allclose(
                t, torch.tensor([expected], dtype=torch.float32, device=device)
            ), f"Rank {rank} all_reduce failed"

            # all_gather
            input_tensor = torch.tensor(
                [rank], dtype=torch.float32, device=device
            )
            output_tensor = torch.empty(
                world_size, dtype=torch.float32, device=device
            )
            comm.all_gather(output_tensor, input_tensor)
            comm.stream.synchronize()
            expected = torch.arange(
                world_size, dtype=torch.float32, device=device
            )
            assert torch.allclose(output_tensor, expected), (
                f"Rank {rank} all_gather failed"
            )

            # reduce_scatter
            input_tensor = torch.tensor([1.0] * world_size, device=device)
            output_tensor = torch.empty(1, device=device)
            comm.reduce_scatter(output_tensor, input_tensor)
            comm.stream.synchronize()
            expected = torch.tensor(
                [world_size], dtype=torch.float32, device=device
            )
            assert torch.allclose(output_tensor, expected), (
                f"Rank {rank} reduce_scatter failed"
            )

            # broadcast
            bcast_tensor = torch.tensor(
                [rank], dtype=torch.float32, device=device
            )
            comm.broadcast(bcast_tensor, src=0)
            comm.stream.synchronize()
            assert bcast_tensor.item() == 0.0, f"Rank {rank} broadcast failed"

            # send/recv (only between rank 0 and 1)
            if world_size >= 2:
                if rank == 0:
                    comm.send(torch.tensor([123.0], device=device), dst=1)
                elif rank == 1:
                    t_recv = torch.empty(1, device=device)
                    comm.recv(t_recv, src=0)
                    comm.stream.synchronize()
                    assert t_recv.item() == 123.0, f"Rank {rank} recv failed"

        return_dict[rank] = "pass"

    except Exception as e:
        return_dict[rank] = f"Exception:\n{traceback.format_exc()}"

    finally:
        cleanup_dist()


@pytest.mark.parametrize("world_size", [2, 4])
def test_nccl_communicator(world_size):
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        _test_nccl_worker,
        args=(world_size, return_dict),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        assert return_dict[rank] == "pass", (
            f"Rank {rank} failed: {return_dict[rank]}"
        )
