import logging
import os
import sys
import traceback
from typing import Any, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from batchgen.batchgen_worker import BatchGenWorker, BatchGenWorkerArgs


def server_worker_main(
	rank_idx: int,
	request_queue: mp.Queue,
	response_queue: mp.Queue,
	args: BatchGenWorkerArgs
):
	try:
		_server_worker_main_impl(rank_idx, request_queue, response_queue, args)
	except Exception as e:
		global_rank = getattr(args, "global_rank", None)
		logging.error(f"[FATAL] Unhandled exception in worker "
					  f"rank_idx={rank_idx}, global_rank={global_rank}: {e}")
		traceback.print_exc()
		try:
			if dist.is_available() and dist.is_initialized():
				dist.destroy_process_group()
		except Exception:
			pass
		os._exit(1)


def _server_worker_main_impl(
	rank_idx: int,
	request_queue: mp.Queue,
	response_queue: mp.Queue,
	args: BatchGenWorkerArgs
):
	"""
	The main loop for the GPU workers.
	Rank 0 acts as the coordinator, reading from the master process queue.
	All ranks receive the full global batch for coordinated scheduling.
	"""
	# Step 0: Hydrate the rank of this process
	num_gpus_per_node = torch.cuda.device_count()
	args.local_rank = rank_idx
	args.global_rank = num_gpus_per_node * args.nnode_rank + rank_idx
	args.device = args.local_rank

	# 1. Initialize Process Group
	logging.info(f"Starting BatchGen Worker on local rank {args.local_rank}, global rank {args.global_rank}")

	try:
		dist.init_process_group(
			backend="nccl",
			init_method="tcp://" + args.dist_init_addr,
			world_size=args.world_size,
			rank=args.global_rank,
			device_id=args.local_rank,
			timeout=torch.distributed.timedelta(seconds=3600),
		)
	except Exception as e:
		logging.error(f"Failed to initialize process group: {e}")
		sys.exit(1)

	torch.cuda.set_device(args.local_rank)
	logging.info(f"Process group initialized for rank {args.global_rank}/{args.world_size}.")

	# 2. Instantiate the BatchGenWorker
	worker = BatchGenWorker(args)

	# 3. Long-lived server loop
	global_rank = args.global_rank
	world_size = args.world_size

	logging.info(f"Entering main server loop on rank {global_rank}.")
	while True:
		# --- STEP 1: Data Acquisition (Rank 0 only) ---
		task_data = None

		if global_rank == 0:
			task_data = request_queue.get()

		# --- STEP 2: Broadcast full task to all ranks ---
		task_container = [task_data]
		dist.broadcast_object_list(task_container, src=0)
		task_data = task_container[0]

		# --- STEP 3: Shutdown Check ---
		if task_data is None:
			logging.info(f"Rank {global_rank} received shutdown signal. Exiting worker.")
			break

		# --- STEP 4: Inference with full global batch ---
		local_results = []
		try:
			# Unpack payload - all ranks now have the full global batch
			global_prompts = task_data.get("prompts", [])
			current_max_input = task_data.get("max_input_len", 1024)
			current_max_output = task_data.get("max_output_len", 128)
			ignore_eos = task_data.get("ignore_eos", False)  # NEW: Extract ignore_eos

			# Clear previous state if supported
			if hasattr(worker, 'reset_runtime_state'):
				worker.reset_runtime_state()

			if len(global_prompts) > 0:
				# Initialize worker with global batch info
				worker.Init(current_max_input, current_max_output, len(global_prompts))
				
				# NEW: Set ignore_eos flag on worker
				worker.set_ignore_eos(ignore_eos)

				# Process the global batch - worker internally handles distribution
				local_results = worker.process_new_batch(global_prompts)
			else:
				local_results = []

		except Exception as e:
			logging.error(f"Error during inference on rank {global_rank}: {e}", exc_info=True)
			local_results = []

		# --- STEP 5: Gather Results back to Rank 0 ---
		gather_list = [None for _ in range(world_size)] if global_rank == 0 else None
		dist.gather_object(local_results, gather_list, dst=0)

		# --- STEP 6: Response (Rank 0 Only) ---
		if global_rank == 0:
			# Flatten results
			final_results = []
			if gather_list:
				for batch_res in gather_list:
					if batch_res:
						final_results.extend(batch_res)
			response_queue.put(final_results)

	# Cleanup
	dist.destroy_process_group()