import logging
import os
import sys
import time
import traceback
from typing import Any, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from batchgen.batchgen_worker import BatchGenWorker, BatchGenWorkerArgs


def _setup_nccl_env():
	"""
	Set up NCCL environment variables for better reliability in multi-node setups.
	These should be set before any NCCL operations.
	"""
	# Increase connection timeout and retry attempts
	# NCCL_SOCKET_TIMEOUT: timeout in milliseconds for socket operations (default: varies)
	if "NCCL_SOCKET_TIMEOUT" not in os.environ:
		os.environ["NCCL_SOCKET_TIMEOUT"] = "300000"  # 5 minutes in ms
	
	# NCCL_NET_RETRY_COUNT: number of retries for network operations
	if "NCCL_NET_RETRY_COUNT" not in os.environ:
		os.environ["NCCL_NET_RETRY_COUNT"] = "100"  # More retries (default is ~10)
	

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
	# Step 0: Set up NCCL environment for reliability
	_setup_nccl_env()
	
	# Step 1: Hydrate the rank of this process
	num_gpus_per_node = torch.cuda.device_count()
	args.local_rank = rank_idx
	args.global_rank = num_gpus_per_node * args.nnode_rank + rank_idx
	args.device = args.local_rank

	# 2. Initialize Process Group
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

	# CRITICAL: Warmup NCCL connections before entering server loop
	# This ensures all inter-node NCCL connections are fully established
	# before any application-level communication happens.
	# Without this, the first collective may fail with "Connection refused"
	# if remote ranks haven't fully initialized their NCCL listeners.
	try:
		logging.info(f"Rank {args.global_rank}: Starting NCCL connection warmup...")
		
		# Step 1: Simple barrier to ensure all ranks have reached this point
		dist.barrier()
		logging.info(f"Rank {args.global_rank}: Barrier 1 passed")
		
		# Step 2: Small all_reduce to establish actual NCCL connections
		# NCCL lazily establishes connections, so we force it here
		warmup_tensor = torch.ones(1, device=f"cuda:{args.local_rank}")
		dist.all_reduce(warmup_tensor, op=dist.ReduceOp.SUM)
		torch.cuda.synchronize()
		
		expected = float(args.world_size)
		if abs(warmup_tensor.item() - expected) > 1e-6:
			raise RuntimeError(f"NCCL warmup failed: expected {expected}, got {warmup_tensor.item()}")
		logging.info(f"Rank {args.global_rank}: all_reduce warmup passed")
		
		# Step 3: Test broadcast_object_list specifically since that's what fails
		test_obj = [args.global_rank] if args.global_rank == 0 else [None]
		dist.broadcast_object_list(test_obj, src=0)
		if test_obj[0] != 0:
			raise RuntimeError(f"broadcast_object_list warmup failed: got {test_obj[0]}")
		logging.info(f"Rank {args.global_rank}: broadcast_object_list warmup passed")
		
		# Final barrier to ensure all warmup is complete
		dist.barrier()
		logging.info(f"Rank {args.global_rank}: NCCL warmup complete, all connections established")
		
	except Exception as e:
		logging.error(f"Rank {args.global_rank}: NCCL warmup failed: {e}")
		logging.error(f"This usually indicates a network issue or startup race condition.")
		logging.error(f"Try restarting the server or check network connectivity between nodes.")
		sys.exit(1)

	# 2. Instantiate the BatchGenWorker
	worker = BatchGenWorker(args)
	
	# CRITICAL: Barrier after worker init to ensure all ranks complete cudaHostRegister
	# The Host KV pinned memory registration can take 200+ seconds and varies per rank.
	# Without this barrier, faster ranks will start the main loop and attempt collective
	# operations while slower ranks are still initializing, causing NCCL errors.
	logging.info(f"Rank {args.global_rank}: Worker initialized, waiting for all ranks at barrier...")
	dist.barrier()
	logging.info(f"Rank {args.global_rank}: All ranks ready, entering main loop.")

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