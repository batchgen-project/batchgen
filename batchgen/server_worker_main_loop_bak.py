import logging
import os
import sys
import traceback
from typing import Any, List

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from batchgen.batchgen_worker import BatchGenWorker, BatchGenWorkerArgs


# Wrapper to print unhandled exceptions and avoid hanging
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
		# Avoid hanging, directly kill this process
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
			backend="nccl",  # Use NCCL for CUDA devices
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
		# --- STEP 1: Data Acquisition & Splitting (Rank 0) ---
		
		# 'scatter_list' will hold payloads for each rank.
		# Payload format: {"chunk": List[str], "max_input_len": int, "max_output_len": int}
		scatter_list = None
		
		if global_rank == 0:
			# Blocking wait for task from Mother Process
			# Expecting task_data to be a Dict or None
			# Structure: { "prompts": [...], "max_input_len": int, "max_output_len": int }
			task_data = request_queue.get()

			# Check Shutdown
			if task_data is None:
				logging.info("Received shutdown signal. Exiting worker.")
				scatter_list = [None for _ in range(world_size)]
			else:
				# Extract Info
				full_prompts = task_data.get("prompts", [])
				req_max_input = task_data.get("max_input_len", 1024)
				req_max_output = task_data.get("max_output_len", 128)
				
				# Split the prompts into chunks for data parallelism
				prompt_chunks = split_list_into_chunks(full_prompts, world_size)
				
				# Package chunks with the config for every rank
				scatter_list = []
				for chunk in prompt_chunks:
					scatter_list.append({
						"prompts": chunk,
						"max_input_len": req_max_input,
						"max_output_len": req_max_output,
						"global_batch_size": len(full_prompts)
					})
		
		# --- STEP 2: Scatter (Distribute tasks to ranks) ---
		
		# Each rank prepares a container to receive its specific payload
		received_payload_container = [None]

		# EXECUTE SCATTER: Rank 0 distributes the list items to all ranks
		dist.scatter_object_list(received_payload_container, scatter_list, src=0)

		# Unwrap the data
		my_payload = received_payload_container[0]

		# --- STEP 3: Shutdown Check ---
		if my_payload is None:
			logging.info(f"Rank {global_rank} received shutdown signal. Exiting worker.")
			break
		
		# --- STEP 4: Inference ---
		local_results = []
		try:
			# Unpack payload
			my_prompts = my_payload["prompts"]
			current_max_input = my_payload["max_input_len"]
			current_max_output = my_payload["max_output_len"]
			current_global_bs = my_payload["global_batch_size"]

			# Clear previous state if supported
			if hasattr(worker, 'reset_runtime_state'):
				worker.reset_runtime_state()

			if len(my_prompts) > 0:
				# Initialize worker with the lengths specific to this request
				# This ensures the KV cache or buffers are sized correctly for this specific run
				worker.Init(current_max_input, current_max_output, len(my_prompts))
				
				# Run Inference
				local_results = worker.process_new_batch(my_prompts, current_global_bs)
			else:
				# Even if this GPU has no prompts (e.g., batch size < num_gpus), 
				# it might need to participate in collective comms inside 'process_new_batch'
				# depending on your engine implementation. 
				# If your engine handles empty batches gracefully, pass empty list.
				# Otherwise, you might need a dummy wait. Assuming graceful handling here:
				local_results = []
				
		except Exception as e:
			logging.error(f"Error during inference on rank {global_rank}: {e}", exc_info=True)
			local_results = []

		# --- STEP 5: Gather Results back to Rank 0 ---
		gather_list = [None for _ in range(world_size)] if global_rank == 0 else None
		dist.gather_object(local_results, gather_list, dst=0)

		# --- STEP 6: Response (Rank 0 Only) ---
		if global_rank == 0:
			# Flatten results: [[res1, res2], [res3, res4]] -> [res1, res2, res3, res4]
			final_results = []
			if gather_list:
				for batch_res in gather_list:
					if batch_res:
						final_results.extend(batch_res)
			response_queue.put(final_results)

	# Cleanup
	dist.destroy_process_group()


# --- Helper Function ---
def split_list_into_chunks(data: List[Any], num_chunks: int) -> List[List[Any]]:
	"""
	Splits a list into 'num_chunks' sub-lists.
	Ensures the output is exactly length 'num_chunks'.
	"""
	avg = len(data) / float(num_chunks)
	chunks = []
	last = 0.0
	
	while last < len(data):
		chunks.append(data[int(last):int(last + avg)])
		last += avg
		
	# Pad with empty lists if we have more ranks than data chunks
	while len(chunks) < num_chunks:
		chunks.append([])
		
	# Handle rare floating point edge cases
	if len(chunks) > num_chunks:
		chunks[num_chunks-1].extend([item for sublist in chunks[num_chunks:] for item in sublist])
		chunks = chunks[:num_chunks]
		
	return chunks