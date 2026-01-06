import importlib
import logging
import os
import sys
import time
import traceback
from typing import Any, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from batchgen.batchgen_worker import BatchGenWorker, BatchGenWorkerArgs
from batchgen.server.watchdog import Watchdog


def _reload_worker_module():
	"""
	Hot-reload the batchgen_worker module to pick up code changes.
	This reloads the module but doesn't affect already-instantiated objects.
	For method-level changes, we can rebind methods to the existing worker.
	"""
	import batchgen.batchgen_worker as worker_module
	importlib.reload(worker_module)
	return worker_module


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
	args: BatchGenWorkerArgs,
	ready_event: Optional[mp.Event] = None,
):
	try:
		_server_worker_main_impl(rank_idx, request_queue, response_queue, args, ready_event)
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
	args: BatchGenWorkerArgs,
	ready_event: Optional[mp.Event] = None,
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

	# 2.5. Initialize watchdog for stuck process detection
	watchdog_timeout = getattr(args, 'watchdog_timeout', None)
	watchdog_test_stuck_time = getattr(args, 'watchdog_test_stuck_time', 0.0)
	watchdog = Watchdog.create(
		debug_name=f"worker-{args.global_rank}",
		watchdog_timeout=watchdog_timeout,
		soft=False,  # Hard mode: kill parent process on timeout
		test_stuck_time=watchdog_test_stuck_time,
	)
	if watchdog_timeout:
		logging.info(f"Rank {args.global_rank}: Watchdog initialized with timeout={watchdog_timeout}s")

	# CRITICAL: Barrier after worker init to ensure all ranks complete cudaHostRegister
	# The Host KV pinned memory registration can take 200+ seconds and varies per rank.
	# Without this barrier, faster ranks will start the main loop and attempt collective
	# operations while slower ranks are still initializing, causing NCCL errors.
	logging.info(f"Rank {args.global_rank}: Worker initialized, waiting for all ranks at barrier...")
	dist.barrier()
	logging.info(f"Rank {args.global_rank}: All ranks ready, entering main loop.")

	# Signal that workers are ready (only rank 0 sets the event to avoid race conditions)
	if ready_event is not None and args.global_rank == 0:
		ready_event.set()
		logging.info(f"Rank {args.global_rank}: Signaled ready event to WorkerManager")

	# 3. Long-lived server loop
	global_rank = args.global_rank
	world_size = args.world_size
	
	# NCCL timeout prevention strategy:
	# The problem: NCCL watchdog times out if a collective operation doesn't complete within timeout.
	# When Rank 0 is blocking on request_queue.get() while other ranks wait on broadcast, 
	# NCCL will timeout if no work arrives.
	#
	# Solution: Rank 0 uses non-blocking queue.get with timeout, then all ranks periodically 
	# perform a lightweight collective to keep NCCL alive (heartbeat pattern).
	QUEUE_POLL_TIMEOUT = 30.0  # Rank 0 polls queue every 30 seconds

	logging.info(f"Entering main server loop on rank {global_rank}.")
	while True:
		# --- STEP 1: Data Acquisition with heartbeat to prevent NCCL timeout ---
		task_data = None
		work_available = False

		while not work_available:
			if global_rank == 0:
				# Rank 0: Non-blocking poll on queue with timeout
				try:
					task_data = request_queue.get(timeout=QUEUE_POLL_TIMEOUT)
					work_available = True
				except Exception:
					# Queue.get timeout - no work yet, signal others
					work_available = False
			
			# All ranks synchronize on work availability status
			# This is a fast broadcast (single int) that keeps NCCL alive
			status_tensor = torch.tensor([1 if work_available else 0], dtype=torch.int32, device='cuda')
			dist.broadcast(status_tensor, src=0)
			work_available = status_tensor.item() == 1
			
			if not work_available:
				# No work yet - this broadcast acts as a heartbeat to prevent NCCL timeout
				# Also feed watchdog to prevent false stuck detection during idle periods
				watchdog.feed()
				# Loop back and poll again
				continue

		# --- STEP 2: Broadcast full task to all ranks ---
		task_container = [task_data]
		dist.broadcast_object_list(task_container, src=0)
		task_data = task_container[0]

		# --- STEP 3: Shutdown Check ---
		if task_data is None:
			logging.info(f"Rank {global_rank} received shutdown signal. Exiting worker.")
			break
		
		# --- STEP 3.5: Hot Reload Command ---
		if isinstance(task_data, dict) and task_data.get("command") == "reload":
			logging.info(f"Rank {global_rank}: Received reload command, hot-reloading worker module...")
			try:
				new_module = _reload_worker_module()
				# Rebind key methods to the existing worker instance
				# This allows code changes to take effect without recreating the worker
				worker._page_boundary_fast = new_module.BatchGenWorker._page_boundary_fast.__get__(worker, type(worker))
				worker.decoding_continuous_fast = new_module.BatchGenWorker.decoding_continuous_fast.__get__(worker, type(worker))
				worker.generate = new_module.BatchGenWorker.generate.__get__(worker, type(worker))
				worker._rebuild_input_tokens = new_module.BatchGenWorker._rebuild_input_tokens.__get__(worker, type(worker))
				logging.info(f"Rank {global_rank}: Hot reload successful!")
				if global_rank == 0:
					response_queue.put({"status": "reload_success"})
			except Exception as e:
				logging.error(f"Rank {global_rank}: Hot reload failed: {e}", exc_info=True)
				if global_rank == 0:
					response_queue.put({"status": "reload_failed", "error": str(e)})
			continue  # Skip to next iteration

		# --- STEP 4: Inference with full global batch ---
		local_results = []
		inference_error = None
		try:
			# Unpack payload - all ranks now have the full global batch
			global_prompts = task_data.get("prompts", [])
			# max_input_len: If None or not provided, will be determined dynamically
			# from the longest prompt in the batch during tokenization
			current_max_input = task_data.get("max_input_len", None)
			current_max_output = task_data.get("max_output_len", 128)
			ignore_eos = task_data.get("ignore_eos", False)  # NEW: Extract ignore_eos

			# Clear previous state if supported
			if hasattr(worker, 'reset_runtime_state'):
				worker.reset_runtime_state()

			if len(global_prompts) > 0:
				# Initialize worker with global batch info
				# max_input_length can be None - will be determined by longest prompt
				worker.Init(current_max_input, current_max_output, len(global_prompts))
				
				# NEW: Set ignore_eos flag on worker
				worker.set_ignore_eos(ignore_eos)

				# Process the global batch - worker internally handles distribution
				local_results = worker.process_new_batch(global_prompts)
			else:
				local_results = []

		except Exception as e:
			logging.error(f"Error during inference on rank {global_rank}: {e}", exc_info=True)
			inference_error = str(e)
			local_results = []

		# --- STEP 5: Gather Results back to Rank 0 ---
		gather_list = [None for _ in range(world_size)] if global_rank == 0 else None
		dist.gather_object(local_results, gather_list, dst=0)
		
		# Also gather any errors from all ranks
		error_list = [None for _ in range(world_size)] if global_rank == 0 else None
		dist.gather_object(inference_error, error_list, dst=0)

		# --- STEP 6: Response (Rank 0 Only) ---
		if global_rank == 0:
			# Check for errors first
			errors = [e for e in error_list if e is not None] if error_list else []
			if errors:
				logging.error(f"Inference failed with errors from ranks: {errors}")
				# Return error to client
				response_queue.put({"error": errors[0], "all_errors": errors})
				continue
			
			# Flatten results
			final_results = []
			if gather_list:
				for batch_res in gather_list:
					if batch_res:
						final_results.extend(batch_res)
			
			# Safeguard: if results are empty but no errors, something went wrong
			if not final_results and len(task_data.get("prompts", [])) > 0:
				logging.error(f"Results are unexpectedly empty! gather_list={[type(x) for x in gather_list]}")
				response_queue.put({"error": "Results unexpectedly empty after inference"})
				continue
				
			response_queue.put(final_results)

		# Feed watchdog after each successful iteration
		watchdog.feed()

	# Cleanup
	dist.destroy_process_group()