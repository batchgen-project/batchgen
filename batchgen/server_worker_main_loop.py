import importlib
import logging
import os
import signal
import sys
import time
import traceback
from typing import Any, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from batchgen.batchgen_worker import BatchGenWorker, BatchGenWorkerArgs
from batchgen.server.process_utils import install_worker_signal_handlers
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

	# TORCH_NCCL_ENABLE_MONITORING: Disable NCCL HeartbeatMonitor entirely
	# The default 60s timeout can cause false positives during CPU-bound operations
	# (e.g., tokenizing large batches). Setting TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=0
	# does NOT disable it - it sets timeout to 0 which fails immediately.
	# Use TORCH_NCCL_ENABLE_MONITORING=0 to actually disable monitoring.
	if os.environ.get("TORCH_NCCL_ENABLE_MONITORING") is None:
		os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
		logging.info("Disabled NCCL HeartbeatMonitor (TORCH_NCCL_ENABLE_MONITORING=0)")

	# NCCL_BUFFSIZE: buffer size for NCCL operations (default: 4MB)
	# Larger buffer improves throughput for multi-node communication
	if "NCCL_BUFFSIZE" not in os.environ:
		os.environ["NCCL_BUFFSIZE"] = "16777216"  # 16MB buffer size
	

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


def _setup_worker_logging(rank_idx: int, global_rank: int = -1):
	"""
	Configure logging for worker processes.
	Spawned processes don't inherit logging config from the parent process.
	"""
	# Use global_rank if available, otherwise fall back to rank_idx
	rank_label = global_rank if global_rank >= 0 else rank_idx
	logging.basicConfig(
		level=logging.INFO,
		format=f'%(asctime)s - [BatchGenWorker-{rank_label}] - %(levelname)s - %(message)s',
		force=True,  # Override any existing configuration
	)


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
	# Step 0: Configure logging for this worker process first
	_setup_worker_logging(rank_idx)

	# Step 0.1: Install signal handlers so workers respond to Ctrl+C
	# This is critical for multi-node setups where node 1 workers might be
	# blocked in NCCL operations when node 0 is killed.
	def _worker_shutdown_callback():
		"""Cleanup callback when worker receives termination signal."""
		try:
			if dist.is_available() and dist.is_initialized():
				dist.destroy_process_group()
		except Exception:
			pass

	install_worker_signal_handlers(_worker_shutdown_callback)

	# Step 0.5: Set up NCCL environment for reliability
	_setup_nccl_env()
	
	# Step 1: Hydrate the rank of this process
	num_gpus_per_node = torch.cuda.device_count()
	args.local_rank = rank_idx
	args.global_rank = num_gpus_per_node * args.nnode_rank + rank_idx
	args.device = args.local_rank

	# Reconfigure logging with actual global rank for clearer log output
	_setup_worker_logging(rank_idx, args.global_rank)

	# 2. Initialize Process Group
	logging.info(f"Starting BatchGen Worker on local rank {args.local_rank}, global rank {args.global_rank}")

	# Set CUDA device before init_process_group so NCCL uses the correct device context
	torch.cuda.set_device(args.local_rank)

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
	logging.info(f"Process group initialized for rank {args.global_rank}/{args.world_size}.")

	# CRITICAL: Warmup NCCL connections before entering server loop
	# This ensures all inter-node NCCL connections are fully established
	# before any application-level communication happens.
	# Without this, the first collective may fail with "Connection refused"
	# if remote ranks haven't fully initialized their NCCL listeners.
	try:
		logging.debug(f"Rank {args.global_rank}: Starting NCCL connection warmup...")

		# Step 1: Simple barrier to ensure all ranks have reached this point
		dist.barrier()
		logging.debug(f"Rank {args.global_rank}: Barrier 1 passed")

		# Step 2: Small all_reduce to establish actual NCCL connections
		# NCCL lazily establishes connections, so we force it here
		warmup_tensor = torch.ones(1, device=f"cuda:{args.local_rank}")
		dist.all_reduce(warmup_tensor, op=dist.ReduceOp.SUM)
		torch.cuda.synchronize()

		expected = float(args.world_size)
		if abs(warmup_tensor.item() - expected) > 1e-6:
			raise RuntimeError(f"NCCL warmup failed: expected {expected}, got {warmup_tensor.item()}")
		logging.debug(f"Rank {args.global_rank}: all_reduce warmup passed")

		# Step 3: Test broadcast_object_list specifically since that's what fails
		test_obj = [args.global_rank] if args.global_rank == 0 else [None]
		dist.broadcast_object_list(test_obj, src=0)
		if test_obj[0] != 0:
			raise RuntimeError(f"broadcast_object_list warmup failed: got {test_obj[0]}")
		logging.debug(f"Rank {args.global_rank}: broadcast_object_list warmup passed")

		# Final barrier to ensure all warmup is complete
		dist.barrier()
		logging.info(f"Rank {args.global_rank}: NCCL warmup complete")
		
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

	# Pass watchdog to worker for fine-grained feeding during inference
	worker.set_watchdog(watchdog)

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

	# 2.6. Initialize decode watchdog AFTER barrier — only monitors decode steps,
	# not worker init, CUDA graph capture, or NCCL warmup.
	decode_step_timeout = getattr(args, 'decode_step_timeout', None)
	decode_watchdog = Watchdog.create(
		debug_name=f"decode-{args.global_rank}",
		watchdog_timeout=decode_step_timeout,
		soft=False,  # Hard mode: kill parent process on timeout
	)
	if decode_step_timeout:
		logging.info(f"Rank {args.global_rank}: Decode watchdog initialized with timeout={decode_step_timeout}s")
	worker.set_decode_watchdog(decode_watchdog)

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
				worker.decoding_continuous = new_module.BatchGenWorker.decoding_continuous.__get__(worker, type(worker))
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
			ignore_eos = task_data.get("ignore_eos", False)
			# Sampling parameters (None = greedy decoding)
			temperature = task_data.get("temperature", None)
			top_p = task_data.get("top_p", None)
			if global_rank == 0:
				logging.info(f"[PAYLOAD] Extracted from task_data: temperature={temperature}, top_p={top_p}")

			# Stage incremental writer config on rank 0 (writer created after tokenizer init)
			incr_output_dir = task_data.get("incremental_output_dir")
			incr_custom_ids = task_data.get("custom_id_map")
			if global_rank == 0 and incr_output_dir and incr_custom_ids:
				worker._incremental_writer_config = {
					"output_dir": incr_output_dir,
					"batch_id": task_data.get("batch_id", "unknown"),
					"model_name": task_data.get("model_name", "unknown"),
					"custom_id_map": incr_custom_ids,
					"request_urls": task_data.get("request_url_map", {}),
					"prompt_texts": task_data.get("prompt_text_map", {}),
					"parse_thinking": task_data.get("parse_thinking", False),
					"parse_tool_call": task_data.get("parse_tool_call", False),
				}

			# Clear previous state if supported
			if hasattr(worker, 'reset_runtime_state'):
				worker.reset_runtime_state()

			if len(global_prompts) > 0:
				# Initialize worker with global batch info
				# max_input_length can be None - will be determined by longest prompt
				worker.Init(current_max_input, current_max_output, len(global_prompts))

				# Set ignore_eos flag on worker
				worker.set_ignore_eos(ignore_eos)

				# Set sampling parameters (None = greedy decoding)
				worker.set_sampling_params(temperature=temperature, top_p=top_p)

				# Process the global batch - worker internally handles distribution
				local_results = worker.process_new_batch(global_prompts)
			else:
				local_results = []

		except Exception as e:
			logging.error(f"Error during inference on rank {global_rank}: {e}", exc_info=True)
			inference_error = str(e)
			local_results = []
		finally:
			# Close incremental writer regardless of success/failure
			if getattr(worker, '_incremental_writer', None) is not None:
				worker._incremental_writer.close()
				worker._incremental_writer = None
			# Clean up staged config
			if hasattr(worker, '_incremental_writer_config'):
				del worker._incremental_writer_config

		# --- STEP 5: Synchronize and check for errors ---
		# Sync CUDA to catch any async errors
		try:
			torch.cuda.synchronize()
		except RuntimeError as e:
			logging.error(f"[DEBUG] CUDA sync error on rank {global_rank}: {e}")
			inference_error = str(e)

		# Barrier to ensure all ranks complete inference
		dist.barrier()

		# Check for errors across all ranks using all_reduce
		# Each rank contributes 1 if it has an error, 0 otherwise
		error_flag = torch.tensor([1 if inference_error else 0], dtype=torch.int32, device='cuda')
		dist.all_reduce(error_flag, op=dist.ReduceOp.SUM)
		has_any_error = error_flag.item() > 0

		if has_any_error:
			# Gather error messages from all ranks
			error_list = [None for _ in range(world_size)] if global_rank == 0 else None
			dist.gather_object(inference_error, error_list, dst=0)

			if global_rank == 0:
				errors = [e for e in error_list if e is not None]
				logging.error(f"Inference failed with errors from ranks: {errors}")
				response_queue.put({"error": errors[0], "all_errors": errors})
			continue

		# --- STEP 6: Response (Rank 0 Only) ---
		# Results are already on rank 0 (local_results), no gather needed
		# Other ranks have empty results by design
		if global_rank == 0:
			final_results = local_results if local_results else []

			# Safeguard: if results are empty but no errors, something went wrong
			if not final_results and len(task_data.get("prompts", [])) > 0:
				logging.error(f"Results are unexpectedly empty!")
				response_queue.put({"error": "Results unexpectedly empty after inference"})
				continue

			response_queue.put(final_results)

		# NOTE: Watchdog is fed within prefill/decode phases in the worker,
		# not here. This ensures we only monitor actual inference operations.

	# Cleanup
	dist.destroy_process_group()