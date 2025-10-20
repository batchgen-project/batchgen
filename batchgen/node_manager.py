"""
	Node manager would start a number of python processes each manages a gpu device.
	It would receive jobs from the InferenceRuntime process on Node 0.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from fastapi import FastAPI, HTTPException
from typing import List
import uvicorn
import requests
import asyncio
import logging

# This is a placeholder for a worker process.
# In a real implementation, this would be a more complex class or module.
def worker_process_func(rank: int, world_size: int, master_addr: str, master_port: int, node_manager_port: int):
    """
    The entry point for each worker process.
    It initializes the environment, the torch distributed group, and starts a FastAPI server.
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    # Set the current device
    torch.cuda.set_device(rank)

    app = FastAPI()

    @app.get("/")
    def read_root():
        return {"message": f"Hello from worker {rank}"}

    # Add more endpoints here to receive instructions from InferenceRuntime

    # Each worker runs on a different port
    worker_port = node_manager_port + rank + 1
    print(f"Worker {rank} starting on port {worker_port}")
    uvicorn.run(app, host="0.0.0.0", port=worker_port)


class NodeManager:
    """
    The NodeManager is responsible for creating and managing a set of worker processes,
    each of which is assigned to a single CUDA device. It uses torch.multiprocessing.spawn
    to ensure proper initialization for distributed training.
    """
    def __init__(self, world_size: int, master_addr: str = "127.0.0.1", master_port: int = 29500, node_manager_port: int = 8000):
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.node_manager_port = node_manager_port
        self.processes: List[mp.Process] = []

    def start_workers(self):
        """
        Starts the worker processes using torch.multiprocessing.spawn.
        """
        logging.info(f"Starting {self.world_size} workers...")
        mp.spawn(
            worker_process_func,
            args=(self.world_size, self.master_addr, self.master_port, self.node_manager_port),
            nprocs=self.world_size,
            join=False  # Set to False to manage processes manually
        )
		logging.info("All workers have been started.")

    async def stop_workers(self):
        """
        Stops all worker processes. This is a simple example.
        A more robust implementation would involve sending a shutdown command to each worker's API.
        """
        print("Stopping workers...")
        # In a real scenario, you would gracefully shut down the FastAPI servers
        # For example, by sending a request to a /shutdown endpoint on each worker.
        for i in range(self.world_size):
            worker_port = self.node_manager_port + i + 1
            try:
                # This is a simple way to try and stop the workers.
                # A proper implementation would have a shutdown endpoint on the worker.
                requests.post(f"http://127.0.0.1:{worker_port}/shutdown")
            except requests.exceptions.ConnectionError:
                # This is expected if the server is already down
                pass

        # Terminate any processes that are still alive
        for p in self.processes:
            if p.is_alive():
                p.terminate()
            p.join()
        print("All workers have been stopped.")

if __name__ == '__main__':
    # Example usage:
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        print("No CUDA devices found.")
    else:
        print(f"Found {num_devices} CUDA devices.")
        node_manager = NodeManager(world_size=num_devices)
        node_manager.start_workers()

        # Keep the main process alive to manage workers
        try:
            # In a real application, the InferenceRuntime would be interacting with the workers here.
            # For this example, we just wait.
            loop = asyncio.get_event_loop()
            loop.run_forever()
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            asyncio.run(node_manager.stop_workers())
