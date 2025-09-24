import concurrent.futures
import copy
import functools
import psutil
import logging
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from .config.config import EngineConfig

from batchgen.parameter_server_client import ParameterServerClient
from .utils import config_torch_module_initializer
from batchgen.batchgen_worker import BatchGenWorker

logging.basicConfig(
	level=logging.INFO,  # Set to the lowest level to capture all messages
	format="%(asctime)s - %(levelname)s - %(message)s",  # Include timestamp
	datefmt="%Y-%m-%d %H:%M:%S",  # Customize timestamp format
)

from .scheduler.scheduler import Scheduler
import signal
import traceback
import sys
import faulthandler

def signal_handler(signum, frame):
	print(f"\n{'='*50}")
	print(f"Process {os.getpid()} (Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 'N/A'}) received signal {signum}")
	print(f"{'='*50}")
	
	# Print current stack trace
	traceback.print_stack(frame)
	
	# For all threads
	import threading
	for thread_id, frame in sys._current_frames().items():
		print(f"\nThread {thread_id} stack:")
		traceback.print_stack(frame)
	
	# Force flush output
	sys.stdout.flush()
	sys.stderr.flush()
	
	# Exit after some delay to allow other processes to print
	import time
	time.sleep(0.5)
	sys.exit(1)


class FastProcessPoolExecutor(concurrent.futures.ProcessPoolExecutor):
	def shutdown(self, wait=True):
		# Get all worker processes
		if hasattr(self, '_processes'):
			for p in self._processes.values():
				if p.is_alive():
					p.terminate()  # Force termination
			# Give a brief moment for termination
			import time
			time.sleep(0.1)
			# Kill any remaining processes
			for p in self._processes.values():
				if p.is_alive():
					p.kill()
		super().shutdown(wait=False)

class BatchGen():
	def __init__(
		self,	
		huggingface_ckpt_name: str,
		queries: List[str],
		max_input_length: int,
		max_decoding_length: int,
		device: List[int],
		engine_config_json_dir: Optional[str] = None,
		hf_cache_dir: Optional[str] = None,
		cache_dir: Optional[str] = None,
		pt_ckpt_dir: Optional[str] = None,
		host_kv_cache_size: Optional[int] = None,  # If not set, use all host memory
		parameter_server_host: str = 'localhost',
		parameter_server_port: int = 10900,
		dist_init_addr: Optional[str] = "localhost:12355",
		nnodes: Optional[int] = 1,
		node_rank: Optional[int] = 0,
		device_per_node: Optional[int] = 8,
		kv_dtype: str = "bfloat16"
	):
		self.huggingface_ckpt_name = huggingface_ckpt_name
		self.queries = queries
		self.max_input_length = max_input_length
		self.max_decoding_length = max_decoding_length
		self.device = device
		self.engine_config_json_dir = engine_config_json_dir
		self.hf_cache_dir = hf_cache_dir
		self.cache_dir = cache_dir
		self.pt_ckpt_dir = pt_ckpt_dir
		self.host_kv_cache_size = host_kv_cache_size
		self.parameter_server_host = parameter_server_host
		self.parameter_server_port = parameter_server_port
		self.dist_init_addr = dist_init_addr
		self.nnodes = nnodes
		self.node_rank = node_rank
		self.device_per_node = device_per_node
		self.kv_dtype = kv_dtype,

		self.system_setup()
		(
			self.shm_name,
			self.tensor_meta_shm_name,
			self.skeleton_state_dict,
			self.parameter_server_size,
			self.pt_ckpt_dir,
			self.host_kv_cache_size,
			self.num_devices,
		) = self.get_metadata()

	def system_setup(self):
		# Set up system-level configurations if needed
		mp.set_start_method("spawn", force=True)
		mp.set_sharing_strategy("file_system")

		config_torch_module_initializer()

		signal.signal(signal.SIGINT, signal_handler)  # For Ctrl+C
		signal.signal(signal.SIGTERM, signal_handler)  # For kill command

		faulthandler.enable()

	def get_metadata(self):
		# Get model info from the parameter server - just retrieve existing info
		logging.info(f"Connecting to parameter server at {self.parameter_server_host}:{self.parameter_server_port}")
		try:
			with ParameterServerClient(host=self.parameter_server_host, port=self.parameter_server_port) as client:
				# First check if the server has a model loaded
				model_info = client.get_model_info()
				
				# If the expected model isn't loaded, we need to request it
				if model_info.get('huggingface_ckpt_name') != self.huggingface_ckpt_name:
					logging.info(f"Parameter server has a different model loaded. Requesting {self.huggingface_ckpt_name}")
					# Terminate directly
					logging.info(f"Terminating process as the model is not loaded in the parameter server.")
					sys.exit(1)
				else:
					logging.info(f"Model {self.huggingface_ckpt_name} already loaded in parameter server")
		except Exception as e:
			raise RuntimeError(f"Failed to connect to parameter server: {e}")
		
		# Extract necessary data from model_info
		shm_name = model_info.get('shm_name')
		tensor_meta_shm_name = model_info.get('tensor_meta_shm_name')
		skeleton_state_dict = model_info.get('skeleton_state_dict')  # This now comes from shared memory
		parameter_server_size = model_info.get('parameter_server_size')
		if self.pt_ckpt_dir == None:
			self.pt_ckpt_dir = model_info.get('pt_ckpt_dir')
		
		if not all([shm_name, tensor_meta_shm_name, skeleton_state_dict, parameter_server_size, self.pt_ckpt_dir]):
			missing = []
			if not shm_name: missing.append('shm_name')
			if not tensor_meta_shm_name: missing.append('tensor_meta_shm_name')
			if not skeleton_state_dict: missing.append('skeleton_state_dict')
			if not parameter_server_size: missing.append('parameter_server_size')
			if not self.pt_ckpt_dir: missing.append('pt_ckpt_dir')
			raise RuntimeError(f"Missing required information from parameter server: {', '.join(missing)}")
		
		# Calculate host KV cache size if not provided
		if self.host_kv_cache_size is None:
			from batchgen.utils import get_physical_memory_info
			mem_info = get_physical_memory_info()
			free_mem = mem_info["actually_free"]
			# We don't need to subtract parameter_server_size since it's in a separate process
			self.host_kv_cache_size = math.floor(free_mem)
			logging.info(f"Host KV Cache Size: {self.host_kv_cache_size}")
		

		if torch.cuda.is_available():
			num_devices = torch.cuda.device_count()
			logging.info(f"Node {self.node_rank} has {num_devices} local devices visible.")

			# TODO: Handle cases where number of devices is not eight.
			assert num_devices == 8, "Current version requires exactly 8 devices per node. Will be fixed in the future version."

		else:
			raise RuntimeError("No CUDA devices available. Please check your setup.")
		
		return (shm_name, tensor_meta_shm_name, skeleton_state_dict, parameter_server_size, self.pt_ckpt_dir, self.host_kv_cache_size, num_devices)
	
	def safe_collect_results(self, futures):
		all_results = []
		for future in futures:
			try:
				result = future.result()
				all_results.extend(result)
			except torch.distributed.DistBackendError as e:
				# Check if it's an out of memory error during cleanup
				if "out of memory" in str(e) and "destroy_process_group" in str(e):
					print("Ignoring CUDA out of memory error during process cleanup")
					# Try to extract results if they were generated before the error
					# This assumes your function might have returned partial results
				else:
					# Re-raise if it's a different type of DistBackendError
					raise
			except Exception as e:
				# Handle other exceptions according to your needs
				print(f"Error in worker process: {e}")
				raise
			
		return all_results

	def run_batchgen(self,args):
		batchgen_instance, numa_node = args
		try:
			current_process = psutil.Process()
			# Get CPUs for this NUMA node
			numa_cpus = psutil.cpu_count(logical=False) // 2  # Assuming 2 NUMA nodes
			if numa_node == 0:
				cpu_list = list(range(0, numa_cpus))
			else:
				cpu_list = list(range(numa_cpus, numa_cpus * 2))

			current_process.cpu_affinity(cpu_list)
			if hasattr(os, 'sched_setaffinity'):
				os.sched_setaffinity(0, cpu_list)
		except Exception as e:
			print(f"Warning: Could not set NUMA affinity: {e}")

		batchgen_instance.Init()
		res = batchgen_instance.generate()
		return res

	def distribute_sequences(self, num_sequences, num_devices):
		"""
		Distributes sequences across devices ensuring each device gets at least one sequence when possible,
		and the distribution is as even as possible.
		
		Args:
			num_sequences: Number of sequences to distribute
			num_devices: Number of available devices
		
		Returns:
			List of (start_idx, end_idx) tuples for each device
		"""
		# If we have fewer sequences than devices, only use as many devices as we have sequences
		active_devices = min(num_sequences, num_devices)
		
		# Calculate base sequences per device and remainder
		base_per_device = num_sequences // active_devices
		remainder = num_sequences % active_devices
		
		distribution = []
		current_idx = 0
		
		for device_idx in range(num_devices):
			if device_idx < active_devices:
				# This device gets work
				# Add one extra sequence for the first 'remainder' devices
				device_sequences = base_per_device + (1 if device_idx < remainder else 0)
				
				start_idx = current_idx
				end_idx = start_idx + device_sequences
				current_idx = end_idx
				
				distribution.append((start_idx, end_idx))
			else:
				# This device gets no work
				distribution.append((0, 0))  # Empty range
		
		return distribution


	def run(self):
		# Distribute queries among devices
		batchgens = []
		num_queries = len(self.queries)
		# num_devices = len(self.device)
		world_size = self.nnodes * self.device_per_node
		logging.info(f"World size: {world_size}")
		logging.info(f"Total number of queries: {num_queries}")

		assert num_queries >= world_size, "Current version requires at least as many queries as devices. Will be fixed in the future version."

		distribution = self.distribute_sequences(num_queries, world_size)
		per_device_host_kv_cache_size = self.host_kv_cache_size // self.num_devices

		# indices: List of tuples (local_rank, global_rank)
		# E.g. node 1: [(0, 8), (1, 9), (2, 10), (3, 11), (4, 12), (5, 13), (6, 14), (7, 15)]
		indices = [(local_rank, local_rank + self.node_rank * self.device_per_node) for local_rank in range(self.device_per_node)]
		logging.info(f"Distributing queries across devices: {indices}")
		
		# For each device, create a batchgen instance
		for local_rank, global_rank in indices:
			# start_query_idx = device_idx * queries_per_device
			# end_query_idx = min((device_idx + 1) * queries_per_device, num_queries)
			start_query_idx, end_query_idx = distribution[global_rank]
			logging.info(f"Global rank {global_rank}: Processing queries from {start_query_idx} to {end_query_idx}")
					
			batchgen_instance = BatchGenWorker(
				huggingface_ckpt_name=self.huggingface_ckpt_name,
				hf_cache_dir=self.hf_cache_dir,
				cache_dir=self.cache_dir,
				queries=self.queries[start_query_idx:end_query_idx],
				max_input_length=self.max_input_length,
				max_decoding_length=self.max_decoding_length,
				device=self.device[local_rank],
				engine_config_json_dir=self.engine_config_json_dir,
				skeleton_state_dict=self.skeleton_state_dict,
				shm_name=self.shm_name,
				tensor_meta_shm_name=self.tensor_meta_shm_name,
				pt_ckpt_dir=self.pt_ckpt_dir,
				host_kv_cache_size=per_device_host_kv_cache_size,
				dist_init_addr = self.dist_init_addr,
				local_rank = local_rank,
				global_rank = global_rank,
				world_size = self.nnodes * self.device_per_node,
				kv_dtype=self.kv_dtype,
			)
			batchgens.append(batchgen_instance)
		
		logging.info(f"Number of batchgen instances: {len(batchgens)}")
		# TODO: check numa setting in the runtime.
		device_to_numa = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1}  
		worker_args = [(batchgen, device_to_numa[i]) for i, batchgen in enumerate(batchgens)]
		start_time = time.perf_counter()
		with FastProcessPoolExecutor() as executor:
			futures = [executor.submit(self.run_batchgen, args) for args in worker_args]
			all_results = self.safe_collect_results(futures)
		end_time = time.perf_counter()
		logging.info(f"Inference complete. Total time: {end_time - start_time:.2f}s")

		all_results = [item for result in all_results for item in result]
		return all_results		

