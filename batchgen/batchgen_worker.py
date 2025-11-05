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
from transformers import AutoConfig, AutoTokenizer

# import nvidia_dlprof_pytorch_nvtx as nvtx
from batchgen.models.Wrapper import Attn_Wrapper, Expert_Wrapper

from .config.config import EngineConfig
from .models.deepseek.deepseek_parameter_server import DeepSeek_Parameter_Server
from .scheduler.host_mem import get_physical_memory_info

from batchgen.parameter_server_client import ParameterServerClient
from .models.deepseek.deepseekv3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from tqdm import trange
import gc
from datetime import timedelta
from dataclasses import dataclass


from .utils import torch_gpu_mem_usage, create_position_ids_from_attention_mask
from .get_initializer import get_initializer
from .get_parallel_strategy_manager import get_parallel_strategy_manager
from batchgen.utils import config_torch_module_initializer
from batchgen.kv_cache.gpu_paged_kv_manager import GPUKVCacheManager


logging.basicConfig(
	level=logging.INFO,  # Set to the lowest level to capture all messages
	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
	datefmt="%Y-%m-%d %H:%M:%S",  # Customize timestamp format
)

from .scheduler.scheduler import Scheduler
# nvtx = False
# if nvtx:
# 	nvidia_dlprof_pytorch_nvtx.init()
import sys

class query:
	def __init__(
		self,
		text: str = None,
		encoded: Dict[str, torch.Tensor] = None,
		decoded_tokens: torch.Tensor = None,
	):
		self.text = text
		self.encoded = encoded
		self.decoded_tokens = decoded_tokens


@dataclass
class InputArguments:
	"""Input arguments as a dataclass with type hints"""
	huggingface_ckpt_name: str
	hf_cache_dir: Optional[str] = None
	cache_dir: Optional[str] = None
	pt_ckpt_dir: Optional[str] = None
	queries: Optional[List[str]] = None
	padding_length: int = 512
	max_decoding_length: int = 128
	device: int = 0
	num_queries: int = 0
	skeleton_state_dict: Optional[Dict] = None
	shm_name: Optional[str] = None
	tensor_meta_shm_name: Optional[str] = None
	engine_config_json_dir: Optional[str] = None
	host_kv_cache_size: Optional[int] = None
	kv_dtype: str = "bfloat16"
	dist_init_addr: Optional[str] = None
	local_rank: int = 0
	rank: int = 0
	global_rank: int = 0
	world_size: int = 1
	gpu_arch: str = "hooper"

	def get(self, key, default=None):
		"""Get attribute value with a default fallback"""
		return getattr(self, key, default)
	
	def to_dict(self) -> Dict:
		"""Convert back to dictionary if needed"""
		return self.__dict__.copy()
	
	def update(self, **kwargs):
		"""Update multiple attributes at once"""
		for key, value in kwargs.items():
			if hasattr(self, key):
				setattr(self, key, value)
			else:
				raise AttributeError(f"InputArguments has no attribute '{key}'")

class BatchGenWorker:
	def __init__(
		self,
		huggingface_ckpt_name: str,
		hf_cache_dir: Optional[str],
		cache_dir: Optional[str],
		pt_ckpt_dir: Optional[str],
		queries: List[str],
		max_input_length: int,
		max_decoding_length: int,
		device: int,
		skeleton_state_dict,
		shm_name,
		tensor_meta_shm_name,
		engine_config_json_dir = None, # Will be deprecated in the future
		host_kv_cache_size: Optional[int] = None,
		kv_dtype: str = "bfloat16",
		dist_init_addr: str = "localhost:12355",
		local_rank: Optional[int] = 0,
		global_rank: Optional[int] = 0,
		world_size: Optional[int] = 1,
		gpu_arch: str = "hooper"
	):
		self.model = None
		# self.hf_cache_dir = hf_cache_dir
		# hf_cache_dir will be deprecated in the future.
		if (hf_cache_dir is None) and (cache_dir is not None):
			self.hf_cache_dir = cache_dir
		self.huggingface_ckpt_name = huggingface_ckpt_name
		self.cache_dir = cache_dir
		self.pt_ckpt_dir = pt_ckpt_dir
		self.global_queries = queries
		# self.num_queries = len(queries)
		self.max_input_length = max_input_length
		self.max_decoding_length = max_decoding_length
		self.skeleton_state_dict = skeleton_state_dict
		# self.rank = rank
		self.dist_init_addr = dist_init_addr
		self.local_rank = local_rank
		self.global_rank = global_rank
		self.rank = global_rank
		self.world_size = world_size
		self.gpu_arch = gpu_arch
		self.engine_config_json_dir = engine_config_json_dir
		self.kv_dtype = kv_dtype


		config_scheduler = Scheduler(max_input_length, max_decoding_length, world_size)
		self.engine_config = config_scheduler.generate_config()
		# self.engine_config = parse_config_from_json(engine_config_json_dir)
		self.engine_config.Basic_Config.device = device
		self.engine_config.Basic_Config.device_torch = torch.device(
			f"cuda:{device}"
		)
		self.engine_config.Basic_Config.max_decoding_length = (
			max_decoding_length
		)
		self.engine_config.Basic_Config.padding_length = max_input_length
		# self.engine_config.Basic_Config.num_queries = self.num_queries
		self.engine_config.Basic_Config.rank = self.global_rank
		self.engine_config.Basic_Config.world_size = world_size

		if(self.rank == 0):
			print(self.engine_config)
		
		self.device = device
		self.torch_device = torch.device(f"cuda:{device}")
		self.host_kv_cache_size = host_kv_cache_size

		self.attn_mode = None
		self.query_book = None
		self.model_batch_book = {}
		# TODO:
		self.token_k_cache_byte_size = 2048  # mixtral
		self.num_k_storage_tokens = math.floor(
			50 * (1024**3) / 32 / 2048
		)  # 50G k cache, 50G v cache. 192G test-bed.

		self.shm_name = shm_name
		self.tensor_meta_shm_name = tensor_meta_shm_name

		# free_memory, total_memory = torch.cuda.mem_get_info()
		# gpu0_memory = free_memory / 1024 / 1024 / 1024
		# total_memory = total_memory / 1024 / 1024 / 1024
		# logging.info(f"GPU 0 free memory moegen instantiate: {gpu0_memory} GB / {total_memory} GB")



	def Init(self):
		logging.info(f"Initializing batchgen with global rank {self.global_rank} and world size {self.world_size} with PID: {os.getpid()}")
		torch.cuda.set_device(self.device)
		COMM_MASTER_ADDR = self.dist_init_addr.split(':')[0]
		os.environ['COMM_MASTER_ADDR'] = COMM_MASTER_ADDR
		self._init_torch_dist()

		torch.cuda.reset_peak_memory_stats()
		logging.info(self.hf_cache_dir)
		self.model_config = AutoConfig.from_pretrained(
			self.hf_cache_dir,
			trust_remote_code=True,
			local_files_only=True,
		)
		config_torch_module_initializer()
		self.tokenizer = AutoTokenizer.from_pretrained(
			# self.huggingface_ckpt_name,
			self.hf_cache_dir,
			# cache_dir=self.hf_cache_dir,
			trust_remote_code=True,
			local_files_only=True,
		)
		# Use flash_attn by default thus right padding.
		self.tokenizer.padding_side = "right"

		# self.queries, self.model_batches = self.vanilla_batching(
		# 	self.global_queries, self.global_rank, self.world_size)
		# self.num_queries = len(self.queries)
		# # TODO: Move to centralized config later.
		# self.engine_config.Basic_Config.num_queries = self.num_queries
		input_arguments = {
			"huggingface_ckpt_name": self.huggingface_ckpt_name,
			"hf_cache_dir": self.hf_cache_dir,
			"cache_dir": self.cache_dir,
			"pt_ckpt_dir": self.pt_ckpt_dir,
			# "queries": self.queries,
			"padding_length": self.max_input_length,
			"max_decoding_length": self.max_decoding_length,
			"device": self.device,
			"skeleton_state_dict": self.skeleton_state_dict,
			"shm_name": self.shm_name,
			"tensor_meta_shm_name": self.tensor_meta_shm_name,
			"engine_config_json_dir": self.engine_config_json_dir,
			"host_kv_cache_size": self.host_kv_cache_size,
			"kv_dtype": self.kv_dtype,
			# "num_queries": len(self.queries),
			"dist_init_addr": self.dist_init_addr,
			"local_rank": self.local_rank,
			"rank": self.global_rank,
			"global_rank": self.global_rank,
			"world_size": self.world_size,
			"gpu_arch": self.gpu_arch
		}
		logging.info(f"kv_dtype: {input_arguments['kv_dtype']}")
		self.input_arguments = InputArguments(**input_arguments)
		self.initializer = get_initializer(self.huggingface_ckpt_name)
		self.initializer = self.initializer(self.input_arguments)
		self.core_engine, self.engine_config, self.model_config, self.hf_model_config = (
			self.initializer.Init()
		)
		self.queries, self.model_batches = self.vanilla_batching(
			self.global_queries, self.global_rank, self.world_size)
		self.num_queries = len(self.queries)
		# TODO: Move to centralized config later.
		self.engine_config.Basic_Config.num_queries = self.num_queries
		
		self.parallel_manager = get_parallel_strategy_manager(self.huggingface_ckpt_name)
		self.parallel_manager = self.parallel_manager(
			self.hf_model_config,
			self.engine_config,
			self.model_config,
			self.core_engine,
			self.skeleton_state_dict,
			self.local_rank,
			self.global_rank,
			self.world_size
		)        
				
		logging.info(f"Engine on device {self.device} initialized.")

	# def distribute_sequences(self, num_sequences, num_devices):
	# 	"""
	# 	Distributes sequences across devices ensuring each device gets at least one sequence when possible,
	# 	and the distribution is as even as possible.
		
	# 	Args:
	# 		num_sequences: Number of sequences to distribute
	# 		num_devices: Number of available devices
		
	# 	Returns:
	# 		List of (start_idx, end_idx) tuples for each device
	# 	"""
	# 	# If we have fewer sequences than devices, only use as many devices as we have sequences
	# 	active_devices = min(num_sequences, num_devices)
		
	# 	# Calculate base sequences per device and remainder
	# 	base_per_device = num_sequences // active_devices
	# 	remainder = num_sequences % active_devices
		
	# 	distribution = []
	# 	current_idx = 0
		
	# 	for device_idx in range(num_devices):
	# 		if device_idx < active_devices:
	# 			# This device gets work
	# 			# Add one extra sequence for the first 'remainder' devices
	# 			device_sequences = base_per_device + (1 if device_idx < remainder else 0)
				
	# 			start_idx = current_idx
	# 			end_idx = start_idx + device_sequences
	# 			current_idx = end_idx
				
	# 			distribution.append((start_idx, end_idx))
	# 		else:
	# 			# This device gets no work
	# 			distribution.append((0, 0))  # Empty range
		
	# 	return distribution

	# def vanilla_batching(self):
	# 	"""
	# 	For the input dataset, batch it to fill the host memory.
	# 	"""
	# 	# Step 0: Create mapping from query idx to query.
	# 	self.query_book = {
	# 		query_idx: query(
	# 			text=text,
	# 			decoded_tokens=torch.zeros(
	# 				1, self.max_decoding_length, dtype=torch.int64
	# 			),
	# 		)
	# 		for query_idx, text in enumerate(self.queries)
	# 	}
	# 	# Step 1: Tokenize full dataset and pad to mad_input_length.
	# 	for query_idx, query_instance in self.query_book.items():
	# 		tokenized_query = self.tokenizer(
	# 			query_instance.text,
	# 			return_tensors="pt",
	# 			max_length=self.max_input_length,
	# 			truncation=True,
	# 			padding="max_length",
	# 		)
	# 		query_instance.encoded = tokenized_query
	# 		extended_size = self.max_input_length + self.max_decoding_length
	# 		input_ids_extended = torch.zeros(
	# 			(1, extended_size), dtype=tokenized_query["input_ids"].dtype
	# 		)
	# 		attention_mask_extended = torch.zeros(
	# 			(1, extended_size),
	# 			dtype=tokenized_query["attention_mask"].dtype,
	# 		)

	# 		seq_len = tokenized_query["input_ids"].size(1)
	# 		input_ids_extended[0, :seq_len] = tokenized_query["input_ids"][0, :]
	# 		attention_mask_extended[0, :seq_len] = tokenized_query[
	# 			"attention_mask"
	# 		][0, :]

	# 		tokenized_query["input_ids"] = input_ids_extended
	# 		tokenized_query["attention_mask"] = attention_mask_extended
	# 		query_instance.encoded = tokenized_query

	# 	# Step 2: Create model batches. Batch size = self.engine_config.KV_Storage_Config.num_host_slots
	# 	self.model_batches = []
	# 	if self.engine_config.Basic_Config.attn_mode != 3:
	# 		model_batch_size = self.engine_config.KV_Storage_Config.num_host_slots
	# 	else:
	# 		model_batch_size = min(self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size, self.engine_config.KV_Storage_Config.num_host_slots)
		
	# 	num_model_batch = math.ceil(
	# 		self.num_queries
	# 		/ model_batch_size
	# 	)
	# 	for model_batch_idx in range(num_model_batch):
	# 		self.model_batches.append(
	# 			list(
	# 				range(
	# 					model_batch_idx
	# 					* model_batch_size,
	# 					min(
	# 						(model_batch_idx + 1)
	# 						* model_batch_size,
	# 						self.num_queries,
	# 					),
	# 				)
	# 			)
	# 		)

	# 	logging.info(
	# 		f"Number of model level batches: {len(self.model_batches)}"
	# 	)
	# 	logging.info(
	# 		f"Model level batch size: {model_batch_size}"
	# 	)

	def distribute_sequences(self, num_sequences, num_devices):
		"""
		Distributes sequences across ALL devices as evenly as possible.
		Some devices may get zero sequences if num_sequences < num_devices.
		
		Args:
			num_sequences: Number of sequences to distribute
			num_devices: Number of available devices (all will be used)
		
		Returns:
			List of (start_idx, end_idx) tuples for each device
		"""
		# Calculate base sequences per device and remainder
		base_per_device = num_sequences // num_devices
		remainder = num_sequences % num_devices
		
		distribution = []
		current_idx = 0
		
		for device_idx in range(num_devices):
			# Each device gets base_per_device sequences
			# The first 'remainder' devices get one extra sequence
			device_sequences = base_per_device + (1 if device_idx < remainder else 0)
			
			start_idx = current_idx
			end_idx = start_idx + device_sequences
			current_idx = end_idx
			
			distribution.append((start_idx, end_idx))
		
		return distribution

	def vanilla_batching(self, global_queries, rank, num_devices):
		"""
		Distributes and batches queries for distributed inference.
		Each rank gets its local slice of queries and creates batches.
		All ranks will have the same number of batches, with empty lists where needed.
		
		Args:
			global_queries: List of all queries across all devices
			rank: Current device rank
			num_devices: Total number of devices
		
		Returns:
			tuple: (local_queries, model_batches)
				- local_queries: List of queries assigned to this rank
				- model_batches: List of batch lists, where each batch contains local query indices
		"""
		# Step 0: Distribute global queries to get local queries for this rank
		num_global_queries = len(global_queries)
		
		# Distribute all queries across ALL ranks (all ranks are active)
		distribution = self.distribute_sequences(num_global_queries, num_devices)
		start_idx, end_idx = distribution[rank]
		
		# Extract local queries for this rank
		local_queries = global_queries[start_idx:end_idx]
		num_local_queries = len(local_queries)
		
		logging.info(f"Rank {rank}: Assigned global query range [{start_idx}, {end_idx}), local query count: {num_local_queries}")
		
		# Step 1: Create mapping from LOCAL query idx to query.
		self.query_book = {
			query_idx: query(
				text=text,
				decoded_tokens=torch.zeros(
					1, self.max_decoding_length, dtype=torch.int64
				),
			)
			for query_idx, text in enumerate(local_queries)
		}
		
		# Step 2: Tokenize local queries and pad to max_input_length.
		for query_idx, query_instance in self.query_book.items():
			tokenized_query = self.tokenizer(
				query_instance.text,
				return_tensors="pt",
				max_length=self.max_input_length,
				truncation=True,
				padding="max_length",
			)
			query_instance.encoded = tokenized_query
			extended_size = self.max_input_length + self.max_decoding_length
			input_ids_extended = torch.zeros(
				(1, extended_size), dtype=tokenized_query["input_ids"].dtype
			)
			attention_mask_extended = torch.zeros(
				(1, extended_size),
				dtype=tokenized_query["attention_mask"].dtype,
			)

			seq_len = tokenized_query["input_ids"].size(1)
			input_ids_extended[0, :seq_len] = tokenized_query["input_ids"][0, :]
			attention_mask_extended[0, :seq_len] = tokenized_query[
				"attention_mask"
			][0, :]

			tokenized_query["input_ids"] = input_ids_extended
			tokenized_query["attention_mask"] = attention_mask_extended
			query_instance.encoded = tokenized_query

		# Step 3: Determine batch size
		if self.engine_config.Basic_Config.attn_mode != 3:
			model_batch_size = self.engine_config.KV_Storage_Config.num_host_slots
		else:
			model_batch_size = min(
				self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size,
				self.engine_config.KV_Storage_Config.num_host_slots
			)
		
		# Step 4: Calculate the maximum number of batches based on the rank with most queries
		# The rank with the most queries determines how many batches all ranks need
		max_local_queries = math.ceil(num_global_queries / num_devices)
		max_num_batches = math.ceil(max_local_queries / model_batch_size)
		
		# Step 5: Create model batches for this rank's local queries
		model_batches = []
		
		for batch_idx in range(max_num_batches):
			# Calculate the local batch range for this rank
			local_batch_start = batch_idx * model_batch_size
			local_batch_end = min((batch_idx + 1) * model_batch_size, num_local_queries)
			
			if local_batch_start < num_local_queries:
				# This rank has sequences in this batch
				local_indices = list(range(local_batch_start, local_batch_end))
				model_batches.append(local_indices)
			else:
				# This rank has no sequences in this batch - append empty list
				model_batches.append([])
		
		# Verification logging
		logging.info(f"=" * 60)
		logging.info(f"Rank {rank} Batching Summary:")
		logging.info(f"  Total global queries: {num_global_queries}")
		logging.info(f"  Assigned global range: [{start_idx}, {end_idx})")
		logging.info(f"  Local query count: {num_local_queries}")
		logging.info(f"  Model batch size: {model_batch_size}")
		logging.info(f"  Max local queries per rank: {max_local_queries}")
		logging.info(f"  Total batches (all ranks): {len(model_batches)}")
		logging.info(f"  Non-empty batches (this rank): {sum(1 for b in model_batches if b)}")
		logging.info(f"  Batch contents (local indices):")
		for idx, batch in enumerate(model_batches):
			if batch:
				logging.info(f"    Batch {idx}: {len(batch)} sequences {batch}")
			else:
				logging.info(f"    Batch {idx}: [] (empty)")
		logging.info(f"=" * 60)
		
		return local_queries, model_batches
			

	def initial_batching(self):
		"""
		For the input dataset, batch it to fill the host memory.
		"""
		# Step 0: Create mapping from query idx to query.
		self.query_book = {
			query_idx: query(
				text=text,
				decoded_tokens=torch.zeros(
					1, self.max_decoding_length, dtype=torch.int64
				),
			)
			for query_idx, text in enumerate(self.queries)
		}

		# Step 1: Tokenize full dataset.
		tokenized_length = []
		for query_idx, query_instance in self.query_book.items():
			tokenized_query = self.tokenizer(
				query_instance.text,
				return_tensors="pt",
				max_length=self.max_input_length,
				truncation=True,
				padding=False,
			)
			query_instance.encoded = tokenized_query
			tokenized_length.append(
				(query_idx, tokenized_query["input_ids"].shape[1])
			)

		# Step 2: Sort the tokenized queries by length
		tokenized_length = sorted(
			tokenized_length, key=lambda x: x[1], reverse=True
		)

		# Step 3: Create batches based on memory constraints
		self.model_batches = []
		current_query_num = 0
		batch_idx = 0
		while True:
			current_batch = []
			current_batch_padding_length = (
				tokenized_length[0][1] + self.max_decoding_length
			)
			num_sequences = math.floor(
				self.num_k_storage_tokens / current_batch_padding_length
			)
			if current_query_num + num_sequences > self.num_queries:
				current_batch = tokenized_length[current_query_num:]
			else:
				current_batch = tokenized_length[
					current_query_num : current_query_num + num_sequences
				]
			self.model_batches.append(current_batch)
			self.model_batch_book[batch_idx] = {
				"input_length": current_batch_padding_length,
				"num_new_tokens": 0,
			}
			current_query_num += num_sequences
			batch_idx += 1
			if current_query_num >= self.num_queries:
				break

		# Step 4: Complete query instances for each sequences by padding to the same length.
		for batch in self.model_batches:
			max_length = batch[0][1]
			for query_idx, _ in batch:
				self.query_book[query_idx].encoded = self.tokenizer.pad(
					self.query_book[query_idx].encoded,
					max_length=max_length,
					padding="max_length",
				)

		# Step 5: clearn model_batches as list of query idx.
		self.model_batches = [
			[query_idx for query_idx, _ in batch]
			for batch in self.model_batches
		]
		logging.debug("Initial batching done.")

	def generate(self):
		from batchgen.distributed.utils import StatelessProcessGroup
		from batchgen.distributed.device_communicators.pynccl import PyNcclCommunicator
		self.rank = dist.get_rank()
		self.world_size = dist.get_world_size()
		device = torch.device("cuda", self.rank % torch.cuda.device_count())
		comm_master_addr = os.getenv("COMM_MASTER_ADDR")
		self.comm = None
		try:
			group = StatelessProcessGroup.create(
				host=comm_master_addr,
				port=20003,
				rank=self.rank,
				world_size=self.world_size,
				data_expiration_seconds=6000,
			)
			self.comm = PyNcclCommunicator(
				group=group,
				device=device
			)		
		except Exception as e:
			logging.error(f"Rank {self.rank}: PyNccl communicator initialization failed - {e}")
			raise RuntimeError(f"Rank {self.rank}: PyNccl communicator initialization failed - {e}")
		generation_start_time = time.perf_counter()
		prefill_time = 0
		decoding_time = 0
		phase_switching_time = 0
		config_prefill_time = 0
		config_decode_time = 0
		for model_batch_idx in tqdm(
			range(len(self.model_batches)), desc="Model Batch"
		):
			dist.barrier()
			# if self.rank == 0:
			# 	logging.info(f"Rank: {self.rank} pre-prefill barrier done.")
			tmp_start = time.perf_counter()
			self._config_prefill()
			config_prefill_time += time.perf_counter() - tmp_start
			prefill_start_time = time.perf_counter()
			if len(self.model_batches[model_batch_idx]) > 0:
				with torch.inference_mode():
					new_token = self.prefill(self.model_batches[model_batch_idx])
			else:
				new_token = torch.empty((0, 1), dtype=torch.int64, device=self.torch_device)
				logging.info(f"Rank {self.rank} model batch {model_batch_idx} is empty, skipping prefill.")
			prefill_time += time.perf_counter() - prefill_start_time
			self._unregister_fp8_weights()
			dist.barrier()


			# Random create new token.
			# new_token = torch.randint(
			#     0,
			#     1000,
			#     # 129280, # self.model_config.vocab_size,
			#     (len(self.model_batches[model_batch_idx]), 1),
			#     device=self.torch_device,
			# )
			# self.update_new_token(new_token, self.model_batches[model_batch_idx], 0)
			# logging.info("Entering kv_storage creation...")
			# self.core_engine.create_fake_kv_storage()
			# self.core_engine.start_h2d_worker()
			# time.sleep(2)
			
			tmp_start = time.perf_counter()
			self._config_decoding(len(new_token), self.comm)
			# self.core_engine.copy_kv_to_worker(self.model_batches[model_batch_idx], self.max_input_length + self.max_decoding_length)
			if self.engine_config.Basic_Config.attn_mode == 3:
				# FULL GPU DECODING MODE.
				# Need to instantiate GPU KV-Cache Here. 
				# gpu_kv_cache = GPUKVCacheManager(self.engine_config).init(self.core_engine)
				# for query_idx in self.model_batches[model_batch_idx]:
				# 	gpu_kv_cache.allocate_pages(query_idx, self.max_input_length + self.max_decoding_length)
				# 	gpu_kv_cache.load_offloaded_context(query_idx, self.max_input_length) # Load offloaded context kv-cache to gpu.

				if self.model_config.model_type == "deepseek_v3":
					past_key_states= self.core_engine.get_past_key_states(self.model_batches[model_batch_idx], self.max_input_length + self.max_decoding_length)
					# Pad the kv cache to be multiple of 64
					bsz, kv_seqlen, _ = past_key_states[0].size()
					if self.engine_config.Basic_Config.kv_dtype == "bfloat16":
						if kv_seqlen % 64 != 0:
							pad_len = 64 - (kv_seqlen % 64)
							for i in range(len(past_key_states)):
								past_key_states[i] = torch.cat([
									past_key_states[i], 
									torch.zeros((bsz, pad_len, past_key_states[i].size(-1)), device=past_key_states[i].device, dtype=past_key_states[i].dtype)
								], dim=1)
					past_value_states = None
					scale_dict = None
					if self.engine_config.Basic_Config.kv_dtype == "float8_e4m3fn":
						if len(self.model_batches[model_batch_idx]) > 0:
							scale_dict = self.core_engine.get_kv_scale(self.model_batches[model_batch_idx], self.max_input_length + self.max_decoding_length)
						
					
			
				else:
					# TODO: we do
					pass

			else:
				past_key_states = None
				past_value_states = None
				scale_dict = None
			config_decode_time += time.perf_counter() - tmp_start
			dist.barrier()
			# torch.cuda.empty_cache()
			decoding_start_time = time.perf_counter()
			with torch.inference_mode():
				logging.info(
					f"decoding batch size: {len(self.model_batches[model_batch_idx])}"
				)
				self.decoding(new_token, self.model_batches[model_batch_idx], past_key_states, past_value_states, scale_dict)
			decoding_time += time.perf_counter() - decoding_start_time
			self.core_engine.clear_kv_storage()
			self._unregister_fp8_weights()
			self.deep_free_model_memory()
			del past_key_states
			del scale_dict
			# gc.collect()
			dist.barrier()
		
		
		# else:
		# 	# For small input batch, some worker might do not have any input.
		# 	# In this case, it only participate in the decoding phase.
		# 	# Todo: 
		# 	self._config_decoding(0)

		# 	# Log used memory before decoding
		# 	if self.rank == 0:
		# 		free_memory, total_memory = torch.cuda.mem_get_info()
		# 		free_memory = free_memory / 1024 / 1024 / 1024
		# 		total_memory = total_memory / 1024 / 1024 / 1024
		# 		logging.info(
		# 			f"Rank: {self.rank} Device torch memory usage before decoding: {torch.cuda.memory_allocated(self.torch_device) / (1024**3)} GB / {total_memory} GB"
		# 		)
		# 		logging.info(
		# 			f"Rank: {self.rank} Device torch free memory before decoding: {free_memory} GB / {total_memory} GB"
		# 		)
		# 	dist.barrier()
		# 	torch.cuda.empty_cache()
		# 	decoding_start_time = time.perf_counter()
		# 	with torch.inference_mode():
		# 		self.decoding(None, None)
		# 	decoding_time += time.perf_counter() - decoding_start_time
		# 	self.core_engine.clear_kv_storage()


		
		# dist.barrier()
		generation_time = time.perf_counter() - generation_start_time

		self.model = None 
		# torch.cuda.empty_cache()
		phase_switching_time = (config_prefill_time + config_decode_time)
		logging.info(
			f"Rank {self.rank} Prefill total time: {prefill_time:.1f} seconds,\n"
			f"Decoding total time: {decoding_time:.1f} seconds,\n"
			f"Generation total time: {generation_time:.1f} seconds."
			f"Phase switching time: {phase_switching_time:.1f} seconds.\n"
			f"Config prefill time: {config_prefill_time:.1f} seconds.\n"
			f"Config decoding time: {config_decode_time:.1f} seconds.\n"
			f"Waiting for process clean up..."
		)

		res = [
			self.query_book[query_idx].decoded_tokens
			for query_idx in range(self.num_queries)
		]

		# Print first 5 sequences
		# for query_idx in range(5):
		#     logging.info(
		#         f"Decoded tokens: {res[query_idx].squeeze().tolist()}"
		#     )

		# Gather results from all rank to rank 0
		# logging.info(f"Rank {self.rank} res: {res}")
		all_results = [None] * self.world_size
		dist.all_gather_object(all_results, res)
		dist.destroy_process_group()
		all_results = [item for sublist in all_results for item in sublist]
		# logging.info(f"Size of all_results: {len(all_results)}")
		# Concat to a single tensor and copy to cpu
		res_tensor = torch.cat(all_results, dim=0).cpu()
		# logging.info(f"res_tensor shape {res_tensor.shape}")
		if self.rank == 0:
			return [res_tensor]
		else:
			return []

	def _parse_state_dict_dp(self):
		model_init_start_time = time.perf_counter()
		self.hf_model_config._attn_implementation = "eager"
		self.model = DeepseekV3ForCausalLM._from_config(
			self.hf_model_config
		).to(self.engine_config.Basic_Config.device_torch)
		self.model.eval()
		logging.info(
			f"torch module init time: {time.perf_counter() - model_init_start_time} s"
		)

		self.weight_copy_task["attn"] = []
		self.weight_copy_task["routed_expert"] = []
		self.weight_copy_task["shared_expert"] = []

		for layer_idx in trange(self.model_config.num_hidden_layers):
			for name, _ in self.model.model.layers[
				layer_idx
			].self_attn.named_parameters():
				tensor_full_name = (
					"model.layers." + str(layer_idx) + ".self_attn." + name
				)
				self.state_dict_name_map[tensor_full_name] = {
					"module_key": "attn_" + str(layer_idx),
					"tensor_key": name,
				}
			self.weight_copy_task["attn"].append("attn_" + str(layer_idx))

			if layer_idx >= self.hf_model_config.first_k_dense_replace:
				for name, _ in self.model.model.layers[
					layer_idx
				].mlp.shared_experts.named_parameters():
					tensor_full_name = (
						"model.layers."
						+ str(layer_idx)
						+ ".mlp.shared_experts."
						+ name
					)
					self.state_dict_name_map[tensor_full_name] = {
						"module_key": "shared_expert_" + str(layer_idx),
						"tensor_key": name,
					}
				self.weight_copy_task["shared_expert"].append(
					"shared_expert_" + str(layer_idx)
				)

				for expert_idx in range(self.model_config.num_local_experts):
					for name, _ in (
						self.model.model.layers[layer_idx]
						.mlp.experts[expert_idx]
						.named_parameters()
					):
						tensor_full_name = (
							"model.layers."
							+ str(layer_idx)
							+ ".mlp.experts."
							+ str(expert_idx)
							+ "."
							+ name
						)
						self.state_dict_name_map[tensor_full_name] = {
							"module_key": "routed_expert_"
							+ str(layer_idx)
							+ "_"
							+ str(expert_idx),
							"tensor_key": name,
						}
					self.weight_copy_task["routed_expert"].append(
						"routed_expert_"
						+ str(layer_idx)
						+ "_"
						+ str(expert_idx)
					)

	# def _config_prefill(self):
	# 	self.model, self.weight_copy_task = self.parallel_manager.configure_prefill()
	# 	self.set_phase("prefill")
	# 	self.core_engine.stop_h2d_worker()
	# 	self.core_engine.clear_weight_copy_queue()
	# 	self.core_engine.reset_prefill_buffer()
	# 	self.core_engine.set_weight_copy_queue(self.weight_copy_task)
	# 	self.core_engine.clear_kv_storage()
	# 	self.core_engine.start_h2d_worker()


	def _config_prefill(self):
		start_time = time.perf_counter()
		logging.info("Starting _config_prefill")
		
		# Step 1: Configure prefill
		step_start = time.perf_counter()
		self.model, self.weight_copy_task = self.parallel_manager.configure_prefill()
		logging.info(f"configure_prefill took {time.perf_counter() - step_start:.4f}s")
		
		# Step 2: Set phase
		step_start = time.perf_counter()
		self.set_phase("prefill")
		logging.info(f"set_phase took {time.perf_counter() - step_start:.4f}s")
		
		# Step 3: Stop H2D worker
		step_start = time.perf_counter()
		self.core_engine.stop_h2d_worker()
		logging.info(f"stop_h2d_worker took {time.perf_counter() - step_start:.4f}s")
		
		# Step 4: Clear weight copy queue
		step_start = time.perf_counter()
		self.core_engine.clear_weight_copy_queue()
		logging.info(f"clear_weight_copy_queue took {time.perf_counter() - step_start:.4f}s")
		
		# Step 5: Reset prefill buffer
		step_start = time.perf_counter()
		self.core_engine.reset_prefill_buffer()
		logging.info(f"reset_prefill_buffer took {time.perf_counter() - step_start:.4f}s")
		
		# Step 6: Set weight copy queue
		step_start = time.perf_counter()
		self.core_engine.set_weight_copy_queue(self.weight_copy_task)
		logging.info(f"set_weight_copy_queue took {time.perf_counter() - step_start:.4f}s")
		
		# Step 7: Clear KV storage
		step_start = time.perf_counter()
		self.core_engine.clear_kv_storage()
		logging.info(f"clear_kv_storage took {time.perf_counter() - step_start:.4f}s")
		
		# Step 8: Start H2D worker
		step_start = time.perf_counter()
		self.core_engine.start_h2d_worker()
		logging.info(f"start_h2d_worker took {time.perf_counter() - step_start:.4f}s")
		
		total_time = time.perf_counter() - start_time
		logging.info(f"_config_prefill completed in {total_time:.4f}s")
	
	def _config_decoding(self, num_seq, comm=None):
		logging.info(f"Start Config Decoding")
		self.deep_free_model_memory()


		# Get number of sequences for each rank 
		num_seq_per_rank = torch.zeros(self.world_size, dtype=torch.int32, device=self.torch_device)
		num_seq_per_rank[self.rank] = num_seq
		dist.all_reduce(num_seq_per_rank, op=dist.ReduceOp.SUM)
		# Get the maximum number of sequences across all ranks
		max_num_seq = int(num_seq_per_rank.max().item())


		# TODO:
		if self.world_size <= 8:
			self.model, self.weight_copy_task = self.parallel_manager.configure_decoding()
			self.set_phase("decode")
			self.core_engine.stop_h2d_worker()
			self.core_engine.clear_kv_copy_queue()
			self.core_engine.clear_kv_buffer()
			self.core_engine.clear_weight_copy_queue()
			self.core_engine.reset_decoding_buffer()
			self.core_engine.set_weight_copy_queue(self.weight_copy_task)
			self.core_engine.start_h2d_worker()
		else:
			self.model, self.weight_copy_task = self.parallel_manager.pure_gpu_decoding(max_num_seq, comm)

			self.set_phase("decode")
			self.core_engine.stop_h2d_worker()
			self.core_engine.clear_kv_copy_queue()
			self.core_engine.clear_kv_buffer()
			self.core_engine.clear_weight_copy_queue()
			self.core_engine.reset_decoding_buffer()

		logging.info(f"{self.rank} End Config Decoding")

	# def _config_decoding(self, num_seq, comm=None):
	# 	start_time = time.perf_counter()
	# 	logging.info(f"Rank {self.rank}: Starting _config_decoding with num_seq={num_seq}")
		
	# 	# Step 1: Deep free model memory
	# 	step_start = time.perf_counter()
	# 	self.deep_free_model_memory()
	# 	logging.info(f"Rank {self.rank}: deep_free_model_memory took {time.perf_counter() - step_start:.4f}s")
		
	# 	# Step 2: Prepare num_seq_per_rank tensor
	# 	step_start = time.perf_counter()
	# 	num_seq_per_rank = torch.zeros(self.world_size, dtype=torch.int32, device=self.torch_device)
	# 	num_seq_per_rank[self.rank] = num_seq
	# 	logging.info(f"Rank {self.rank}: tensor preparation took {time.perf_counter() - step_start:.4f}s")
		
	# 	# Step 3: All-reduce to gather sequence counts
	# 	step_start = time.perf_counter()
	# 	dist.all_reduce(num_seq_per_rank, op=dist.ReduceOp.SUM)
	# 	logging.info(f"Rank {self.rank}: all_reduce took {time.perf_counter() - step_start:.4f}s")
		
	# 	# Step 4: Get max number of sequences
	# 	step_start = time.perf_counter()
	# 	max_num_seq = int(num_seq_per_rank.max().item())
	# 	logging.info(f"Rank {self.rank}: max_num_seq={max_num_seq}, computation took {time.perf_counter() - step_start:.4f}s")
		
	# 	# Branch based on world size
	# 	if self.world_size <= 8:
	# 		logging.info(f"Rank {self.rank}: Taking world_size <= 8 branch")
			
	# 		# Step 5a: Configure decoding
	# 		step_start = time.perf_counter()
	# 		self.model, self.weight_copy_task = self.parallel_manager.configure_decoding()
	# 		logging.info(f"Rank {self.rank}: configure_decoding took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 6a: Set phase
	# 		step_start = time.perf_counter()
	# 		self.set_phase("decode")
	# 		logging.info(f"Rank {self.rank}: set_phase took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 7a: Stop H2D worker
	# 		step_start = time.perf_counter()
	# 		self.core_engine.stop_h2d_worker()
	# 		logging.info(f"Rank {self.rank}: stop_h2d_worker took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 8a: Clear KV copy queue
	# 		step_start = time.perf_counter()
	# 		self.core_engine.clear_kv_copy_queue()
	# 		logging.info(f"Rank {self.rank}: clear_kv_copy_queue took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 9a: Clear KV buffer
	# 		step_start = time.perf_counter()
	# 		self.core_engine.clear_kv_buffer()
	# 		logging.info(f"Rank {self.rank}: clear_kv_buffer took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 10a: Clear weight copy queue
	# 		step_start = time.perf_counter()
	# 		self.core_engine.clear_weight_copy_queue()
	# 		logging.info(f"Rank {self.rank}: clear_weight_copy_queue took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 11a: Reset decoding buffer
	# 		step_start = time.perf_counter()
	# 		self.core_engine.reset_decoding_buffer()
	# 		logging.info(f"Rank {self.rank}: reset_decoding_buffer took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 12a: Set weight copy queue
	# 		step_start = time.perf_counter()
	# 		self.core_engine.set_weight_copy_queue(self.weight_copy_task)
	# 		logging.info(f"Rank {self.rank}: set_weight_copy_queue took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 13a: Start H2D worker
	# 		step_start = time.perf_counter()
	# 		self.core_engine.start_h2d_worker()
	# 		logging.info(f"Rank {self.rank}: start_h2d_worker took {time.perf_counter() - step_start:.4f}s")
			
	# 	else:
	# 		logging.info(f"Rank {self.rank}: Taking world_size > 8 branch (pure GPU decoding)")
			
	# 		# Step 5b: Pure GPU decoding
	# 		step_start = time.perf_counter()
	# 		self.model, self.weight_copy_task = self.parallel_manager.pure_gpu_decoding(max_num_seq, comm)
	# 		logging.info(f"Rank {self.rank}: pure_gpu_decoding took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 6b: Set phase
	# 		step_start = time.perf_counter()
	# 		self.set_phase("decode")
	# 		logging.info(f"Rank {self.rank}: set_phase took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 7b: Stop H2D worker
	# 		step_start = time.perf_counter()
	# 		self.core_engine.stop_h2d_worker()
	# 		logging.info(f"Rank {self.rank}: stop_h2d_worker took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 8b: Clear KV copy queue
	# 		step_start = time.perf_counter()
	# 		self.core_engine.clear_kv_copy_queue()
	# 		logging.info(f"Rank {self.rank}: clear_kv_copy_queue took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 9b: Clear KV buffer
	# 		step_start = time.perf_counter()
	# 		self.core_engine.clear_kv_buffer()
	# 		logging.info(f"Rank {self.rank}: clear_kv_buffer took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 10b: Clear weight copy queue
	# 		step_start = time.perf_counter()
	# 		self.core_engine.clear_weight_copy_queue()
	# 		logging.info(f"Rank {self.rank}: clear_weight_copy_queue took {time.perf_counter() - step_start:.4f}s")
			
	# 		# Step 11b: Reset decoding buffer
	# 		step_start = time.perf_counter()
	# 		self.core_engine.reset_decoding_buffer()
	# 		logging.info(f"Rank {self.rank}: reset_decoding_buffer took {time.perf_counter() - step_start:.4f}s")
		
	# 	total_time = time.perf_counter() - start_time
	# 	logging.info(f"Rank {self.rank}: _config_decoding completed in {total_time:.4f}s")

	def prefill(self, batch: list[int]):
		"""
		Handle the prefill for a full model batch.
		"""

		if "deepseek" in self.model_config.model_type:
			self.model.model._use_flash_attention_2 = False

		input_ids = torch.cat(
			[
				self.query_book[query_idx].encoded["input_ids"][
					:, : self.max_input_length
				]
				for query_idx in batch
			],
			dim=0,
		)
		attention_masks = torch.cat(
			[
				self.query_book[query_idx].encoded["attention_mask"][
					:, : self.max_input_length
				]
				for query_idx in batch
			],
			dim=0,
		)

		num_prefill_micro_batches = math.ceil(
			len(batch)
			/ self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size
		)
		prefill_micro_batch_input_ids = torch.split(
			input_ids,
			self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
		)
		Prefill_micro_batch_attention_masks = torch.split(
			attention_masks,
			self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
		)
		logging.info(
			f"Number of prefill micro batches: {num_prefill_micro_batches}"
		)
		cur_batch_start = 0
		output_tokens = []
		for micro_batch_idx in tqdm(
			range(num_prefill_micro_batches), desc="Prefill Micro Batch"
		):
			with torch.inference_mode():
				Attn_Wrapper.attention_mask = (
					Prefill_micro_batch_attention_masks[micro_batch_idx]
				)
				if "deepseek" in self.model_config.model_type:
					Attn_Wrapper.position_ids = (
						create_position_ids_from_attention_mask(
							Prefill_micro_batch_attention_masks[micro_batch_idx]
						)
					)
				else:
					Attn_Wrapper.position_ids = (
						create_position_ids_from_attention_mask(
							Prefill_micro_batch_attention_masks[micro_batch_idx]
						)
					)
				cur_batch_size = prefill_micro_batch_input_ids[
					micro_batch_idx
				].shape[0]
				cur_batch = batch[
					cur_batch_start : cur_batch_start + cur_batch_size
				]
				Attn_Wrapper.cur_batch = cur_batch
				cur_batch_start += cur_batch_size
				assert len(cur_batch) == cur_batch_size

				outputs = self.model(
					prefill_micro_batch_input_ids[micro_batch_idx].to(
						self.torch_device
					),
					attention_mask=Prefill_micro_batch_attention_masks[
						micro_batch_idx
					].to(self.torch_device),
					# position_ids=micro_batch_position_ids[micro_batch_idx].to(self.torch_device),
					use_cache=False,
				)
				# Greedy
				new_tokens = torch.argmax(
					outputs.logits[:, -1, :], dim=-1
				).view(-1, 1)
				output_tokens.append(new_tokens)

		new_tokens = torch.cat(output_tokens, dim=0)
		self.update_new_token(new_tokens, batch, 0)
		return new_tokens

	def decoding(
		self, 
		new_tokens: torch.Tensor, 
		batch: list[int],
		past_key_states: Optional[torch.Tensor] = None,
		past_value_states: Optional[torch.Tensor] = None,
		scale_dict: Optional[dict] = None,
	):
		"""
		Handle the decoding for a full model batch.
		All the queries reach <EOS> or the max decoding length.

		return
				- answer_set: dict[query_idx, decoded_tokens]
		"""
		if "deepseek" in self.model_config.model_type:
			self.model.model._use_flash_attention_2 = True
		new_token_idx = 1
		# attention_mask = torch.cat([self.query_book[query_idx].encoded["attention_mask"][:,:self.max_max_input_length + new_token_idx] for query_idx in batch], dim=0)
		# if attention_mask.dim() == 2 and (self.model_config.model_type not in ["Qwen2"]):
		#  	attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
		# 	attention_mask = torch.where(attention_mask == 0, torch.finfo(torch.bfloat16).min, torch.tensor(0.0, dtype=torch.bfloat16, device=attention_mask.device))
		# Attn_Wrapper.attention_mask = attention_mask

		RUNTIME_ATTN_MODE = self.engine_config.Basic_Config.attn_mode
		# Log device memory usage
		logging.info(f"{self.rank} Device memory usage: {torch.cuda.memory_allocated(self.torch_device) / (1024**3)} GB")

		if RUNTIME_ATTN_MODE == 3:
			"""
				KV ACCUMULATION IN GPU.
			"""
			Attn_Wrapper.scale = scale_dict
			Attn_Wrapper.past_key_states = past_key_states
			Attn_Wrapper.past_value_states = past_value_states
			while new_token_idx < self.max_decoding_length:
				# Log for every 50 tokens.
				if self.rank == 0 and new_token_idx % 50 == 0:
					logging.info(f"Decoding new token idx: {new_token_idx}")
				
				
				# micro_batch_size = self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
				# num_micro_batches = math.ceil(len(batch) / micro_batch_size)
				# micro_batches = [
				#     batch[
				#         micro_batch_idx * micro_batch_size : (
				#             micro_batch_idx + 1
				#         )
				#         * micro_batch_size
				#     ]
				#     for micro_batch_idx in range(num_micro_batches)
				# ]
				# Attn_Wrapper.cur_batch = micro_batches
				with torch.inference_mode():
					if len(batch) != 0:
						attention_mask = torch.cat(
							[
								self.query_book[query_idx].encoded[
									"attention_mask"
								][:, : self.max_input_length + new_token_idx]
								for query_idx in batch
							],
							dim=0,
						).to(self.torch_device)
					# if "deepseek" in self.model_config.model_type:
					#     position_ids = create_position_ids_from_attention_mask(
					#         attention_mask
					#     )
					# else:
					#     position_ids = create_position_ids_from_attention_mask(
					#         attention_mask
					#     )[:, -1].unsqueeze(-1)

						Attn_Wrapper.attention_mask = attention_mask
						Attn_Wrapper.position_ids = (attention_mask.sum(-1) - 1).unsqueeze(-1)
						Attn_Wrapper.cache_seqlens = attention_mask.sum(dim=1).to(torch.int32)
						Attn_Wrapper.max_seqlen = Attn_Wrapper.cache_seqlens.max().item()
					else:
						attention_mask = torch.zeros((0, self.max_input_length + new_token_idx), dtype=torch.int64, device=self.torch_device)
						Attn_Wrapper.attention_mask = attention_mask
						Attn_Wrapper.position_ids = attention_mask
						Attn_Wrapper.cache_seqlens = torch.zeros((0,), dtype=torch.int32, device=self.torch_device)
						Attn_Wrapper.max_seqlen = 0

					new_tokens = self.model(
						new_tokens.to(self.torch_device),
						attention_mask=attention_mask,
						# position_ids=position_ids.to(self.torch_device),
						use_cache=False,
					)
					new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(
						-1, 1
					)
					self.update_new_token(new_tokens, batch, new_token_idx)
				new_token_idx += 1
			Attn_Wrapper.scale = None
			Attn_Wrapper.past_key_states = None
			Attn_Wrapper.past_value_states = None
		
		
		else:
			while new_token_idx < self.max_decoding_length:
				if self.rank == 0:
					logging.info(f"Decoding new token idx: {new_token_idx}")
				# Step 1: Before each round of decoding, review the attention mode and batching plan.
				# TODO: review attention mode. Current fixing attention mode.
				RUNTIME_ATTN_MODE = self.engine_config.Basic_Config.attn_mode
				# logging.info(f"RUNTIME_ATTN_MODE: {RUNTIME_ATTN_MODE}")

				if RUNTIME_ATTN_MODE == 0:
					"""
						CPU ATTN MODE
							- NO ATTN MICRO BATCH
					"""
					# self.set_attn_mode(0)
					# self.core_engine.set_attn_mode(0)
					with torch.inference_mode():
						Attn_Wrapper.cur_batch = [batch]
						attention_mask = torch.cat(
							[
								self.query_book[query_idx].encoded[
									"attention_mask"
								][:, : self.max_input_length + new_token_idx]
								for query_idx in batch
							],
							dim=0,
						)
						if "deepseek" not in self.model_config.model_type:
							position_ids = create_position_ids_from_attention_mask(
								attention_mask
							)[:, -1].unsqueeze(-1)
						else:
							position_ids = create_position_ids_from_attention_mask(
								attention_mask
							)
						# DeepSeek use flash-attn by default
						
						# if attention_mask.dim() == 2 and (
						#     self.model_config.model_type not in ["Qwen2", "deepseek"]
						# ):
						#     attention_mask = attention_mask.unsqueeze(1).unsqueeze(
						#         2
						#     )
						#     attention_mask = torch.where(
						#         attention_mask == 0,
						#         torch.finfo(torch.bfloat16).min,
						#         torch.tensor(
						#             0.0,
						#             dtype=torch.bfloat16,
						#             device=attention_mask.device,
						#         ),
						#     )

						Attn_Wrapper.attention_mask = attention_mask
						Attn_Wrapper.position_ids = position_ids
						new_tokens = self.model(
							new_tokens.to(self.torch_device),
							attention_mask=attention_mask.to(self.torch_device),
							# position_ids=position_ids.to(self.torch_device),
							use_cache=False,
						)
						new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(
							-1, 1
						)
						# logging.info(f"New tokens: {new_tokens}")
						# start = time.perf_counter()
						self.update_new_token(new_tokens, batch, new_token_idx)
						# logging.info(
						#     f"Update new token time is ms: {(time.perf_counter() - start) * 1000} ms"
						# )

					# TODO: Temporally remove.
					# Check <EOS>, if <EOS>, remove from batch.
					# for idx, query_idx in enumerate(batch):
					# 	if new_tokens[idx] == self.tokenizer.eos_token_id:
					# 		batch.remove(query_idx)
					new_token_idx += 1

				elif RUNTIME_ATTN_MODE == 1:
					"""
						GPU ATTN MODE
							- ATTN MICRO BATCH
					"""
					# Submit KV copy task to the core engine.
					micro_batch_size = self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
					num_micro_batches = math.ceil(len(batch) / micro_batch_size)
					# logging.info(f"num_micro_batches: {num_micro_batches}")
					micro_batches = [
						batch[
							micro_batch_idx * micro_batch_size : (
								micro_batch_idx + 1
							)
							* micro_batch_size
						]
						for micro_batch_idx in range(num_micro_batches)
					]
					Attn_Wrapper.cur_batch = micro_batches
					# TODO: init ModelConfig in the initializer.
					# Resub every 32 new tokens.
					if (new_token_idx - 1) % 32 == 0:
						for idx in range(new_token_idx - 1, new_token_idx + 31):
							# Note: DeepSeek use fp8 kv.
							if "deepseek" in self.model_config.model_type:
								# past_kv_byte_size = (
								#     (self.max_input_length + idx)
								#     * self.model_config.compressed_kv_dim
								#     * 2
								# )
								# past_kv_byte_size = (
								#     (self.max_input_length + idx)
								#     * self.model_config.compressed_kv_dim
								# )

								# Copy one more token to avoid torch::cat in attention forward.
								past_kv_byte_size = (
									(self.max_input_length + idx + 1)
									* self.model_config.compressed_kv_dim
								)

							elif "mixtral" in self.model_config.model_type:
								past_kv_byte_size = (
									(self.max_input_length + idx)
									* self.model_config.num_key_value_heads
									* self.model_config.head_dim
									* 2
								)
							else:
								raise ValueError(
									f"Model architecture {self.model_config.model_type} not supported yet."
								)

							for layer_idx in range(
								self.model_config.num_hidden_layers
							):
								for micro_batch_idx in range(num_micro_batches):
									cur_batch = micro_batches[micro_batch_idx]
									# logging.info(f"token idx: {idx}, layer idx: {layer_idx}, micro_batch_idx: {micro_batch_idx} current batch: {cur_batch}")
									self.core_engine.submit_to_KV_queue(
										cur_batch,
										micro_batch_idx,
										layer_idx,
										past_kv_byte_size,
									)

					with torch.inference_mode():
						attention_mask = torch.cat(
							[
								self.query_book[query_idx].encoded[
									"attention_mask"
								][:, : self.max_input_length + new_token_idx]
								for query_idx in batch
							],
							dim=0,
						).to(self.torch_device)
						if "deepseek" in self.model_config.model_type:
							position_ids = create_position_ids_from_attention_mask(
								attention_mask
							)
						else:
							position_ids = create_position_ids_from_attention_mask(
								attention_mask
							)[:, -1].unsqueeze(-1)

						# if attention_mask.dim() == 2 and (
						#     self.model_config.model_type not in ["Qwen2", "deepseek"]
						# ):
						#     attention_mask = attention_mask.unsqueeze(1).unsqueeze(
						#         2
						#     )
						#     attention_mask = torch.where(
						#         attention_mask == 0,
						#         torch.finfo(torch.bfloat16).min,
						#         torch.tensor(
						#             0.0,
						#             dtype=torch.bfloat16,
						#             device=attention_mask.device,
						#         ),
						#     )

						Attn_Wrapper.attention_mask = attention_mask
						Attn_Wrapper.position_ids = position_ids
						# logging.info(f"rank: {self.rank} attention_mask: {attention_mask}")
						# logging.info(f"rank: {self.rank} position_ids: {position_ids}")
						new_tokens = self.model(
							new_tokens.to(self.torch_device),
							attention_mask=attention_mask.to(self.torch_device),
							# position_ids=position_ids.to(self.torch_device),
							use_cache=False,
						)
						# torch.cuda.synchronize(self.engine_config.Basic_Config.device_torch)
						# torch.cuda.current_stream(self.engine_config.Basic_Config.device_torch).synchronize()
						new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(
							-1, 1
						)
						self.update_new_token(new_tokens, batch, new_token_idx)
						# torch.cuda.synchronize(self.engine_config.Basic_Config.device_torch)
						# torch.cuda.current_stream(self.engine_config.Basic_Config.device_torch).synchronize()
						# logging.info(f"New tokens: {new_tokens}")
					new_token_idx += 1

					# Step 1.1 Config new micro_batch size. Magic Number change every 32 new tokens.
					# seq_len = self.query_book[batch[0]].encoded["input_ids"].shape[1] + self.query_book[batch[0]].num_decoded_tokens
					# ATTN_DECODING_MICRO_BATCH_SIZE = self.engine_config.GPU_Buffer_Config.k_buffer_num_tokens // seq_len
				elif RUNTIME_ATTN_MODE == 2:
					"""
						CPU-GPU Parallel ATTN.
						Deprecated.
					"""
					w = float(os.getenv("SPLIT_RATIO_W", None))
					if w is None:
						logging.info(
							f"CPU compute ratio not set. Default setting applied."
						)
						w = 0.6
					logging.info(f"Split ratio: {w}")
					# TODO: wordload partitioning.
					CPU_batch = batch[: math.ceil(len(batch) * w)]
					GPU_batch = batch[math.ceil(len(batch) * w) :]
					logging.info(
						f"CPU batch size: {len(CPU_batch)}, GPU batch size: {len(GPU_batch)}"
					)

					GPU_micro_batch_size = self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
					num_GPU_micro_batches = math.ceil(
						len(GPU_batch) / GPU_micro_batch_size
					)
					GPU_micro_batches = [
						GPU_batch[
							micro_batch_idx * GPU_micro_batch_size : (
								micro_batch_idx + 1
							)
							* GPU_micro_batch_size
						]
						for micro_batch_idx in range(num_GPU_micro_batches)
					]
					Attn_Wrapper.cur_batch = [CPU_batch] + GPU_micro_batches
					# TODO:
					if (new_token_idx - 1) % 32 == 0:
						for idx in range(new_token_idx - 1, new_token_idx + 31):
							if "deepseek" in self.model_config.model_type:
								past_kv_byte_size = (
									(self.max_input_length + idx)
									* self.model_config.compressed_kv_dim
									* self.engine_config.Basic_Config.torch_dtype.itemsize
								)
							else:
								past_kv_byte_size = (
									(self.max_input_length + idx)
									* self.model_config.num_key_value_heads
									* self.model_config.head_dim
									* self.engine_config.Basic_Config.torch_dtype.itemsize
								)

							if "deepseek" in self.model_config.model_type:
								for layer_idx in range(
									self.model_config.num_hidden_layers
								):
									self.core_engine.submit_to_KV_queue(
										cur_batch, 0, layer_idx, past_kv_byte_size
									)

							else:
								for layer_idx in range(
									self.model_config.num_hidden_layers
								):
									for micro_batch_idx in range(
										num_GPU_micro_batches
									):
										cur_batch = GPU_micro_batches[
											micro_batch_idx
										]
										self.core_engine.submit_to_KV_queue(
											cur_batch,
											micro_batch_idx,
											layer_idx,
											past_kv_byte_size,
										)

					with torch.inference_mode():
						attention_mask = torch.cat(
							[
								self.query_book[query_idx].encoded[
									"attention_mask"
								][:, : self.max_input_length + new_token_idx]
								for query_idx in batch
							],
							dim=0,
						)
						if "deepseek" not in self.model_config.model_type:
							position_ids = create_position_ids_from_attention_mask(
								attention_mask
							)[:, -1].unsqueeze(-1)
						else:
							position_ids = create_position_ids_from_attention_mask(
								attention_mask
							)
						if attention_mask.dim() == 2 and (
							self.model_config.model_type not in ["Qwen2"]
						):
							attention_mask = attention_mask.unsqueeze(1).unsqueeze(
								2
							)
							attention_mask = torch.where(
								attention_mask == 0,
								torch.finfo(torch.bfloat16).min,
								torch.tensor(
									0.0,
									dtype=torch.bfloat16,
									device=attention_mask.device,
								),
							)

						Attn_Wrapper.attention_mask = attention_mask
						Attn_Wrapper.position_ids = position_ids
						new_tokens = self.model(
							new_tokens.to(self.torch_device),
							attention_mask=attention_mask.to(self.torch_device),
							# position_ids=position_ids,
							use_cache=False,
						)
						new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(
							-1, 1
						)
						self.update_new_token(new_tokens, batch, new_token_idx)
						print(f"New tokens: {new_tokens}")
					new_token_idx += 1

		# if RUNTIME_ATTN_MODE == 3:
		#     self.core_engine.clear_kv_gpu_storage()    
	
	def set_phase(self, phase: str):
		"""
		Control different behavior of the engine in different phases.
		"""
		torch.cuda.empty_cache()
		self.core_engine.set_phase(phase)
		Attn_Wrapper.phase = phase
		Expert_Wrapper.phase = phase

	def set_mode(self, mode: str):
		"""
		Control different behavior of the engine in different phases.
		"""
		pass

	# def update_new_token(
	#     self, new_tokens: torch.Tensor, query_idx: List[int], new_token_idx: int
	# ):
	#     new_tokens = new_tokens.to("cpu")
	#     for idx, q_idx in enumerate(query_idx):
	#         self.query_book[q_idx].decoded_tokens[:, new_token_idx] = (
	#             new_tokens[idx]
	#         )
	#         self.query_book[q_idx].encoded["input_ids"][
	#             0, new_token_idx + self.max_input_length
	#         ] = new_tokens[idx]
	#         self.query_book[q_idx].encoded["attention_mask"][
	#             0, new_token_idx + self.max_input_length
	#         ] = torch.tensor(1, dtype=torch.int64)


	def update_new_token(
		self, new_tokens: torch.Tensor, query_idx: List[int], new_token_idx: int
	):
		new_tokens = new_tokens.to("cpu")
		for idx, q_idx in enumerate(query_idx):
			# Update decoded tokens
			self.query_book[q_idx].decoded_tokens[:, new_token_idx] = new_tokens[idx]
			
			# Update encoded input_ids
			# self.query_book[q_idx].encoded["input_ids"][
			#     0, new_token_idx + self.max_input_length
			# ] = new_tokens[idx]
			
			# Get the current attention mask
			attention_mask = self.query_book[q_idx].encoded["attention_mask"][0]
			
			# Find the first 0 in the attention mask
			zeros_positions = (attention_mask == 0).nonzero(as_tuple=True)[0]
			# logging.info(f"zeros_positions: {zeros_positions}")
			if len(zeros_positions) > 0:
				# If a 0 is found, change the first one to 1
				first_zero_pos = zeros_positions[0].item()
				self.query_book[q_idx].encoded["attention_mask"][0, first_zero_pos] = torch.tensor(1, dtype=attention_mask.dtype)
				# self.query_book[q_idx].encoded["input_ids"][0, first_zero_pos] = new_tokens[idx]
			else:
				raise ValueError("No 0 found in the attention mask.")

	def _init_torch_dist(self):
		timeout = timedelta(minutes=5)
		# os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'
		try:
			dist.init_process_group(
				backend="nccl",
				# backend="gloo",
				init_method="tcp://" + self.dist_init_addr,
				world_size=self.world_size,
				rank = self.global_rank,
				device_id=torch.device(f"cuda:{self.local_rank}"),
				timeout=timeout,
			)
		except RuntimeError as e:
			logging.error(f"Failed to initialize torch distributed: {e}")
			raise
	
	
	def _unregister_fp8_weights(self):
		# set all fp8 weights to None
		for layer_idx in range(len(self.model.model.layers)):
			attn_module = self.model.model.layers[layer_idx].self_attn
			attn_module._unregister_fp8_weights()
			if layer_idx >= self.hf_model_config.first_k_dense_replace:
				if hasattr(self.model.model.layers[layer_idx].mlp.shared_experts, '_unregister_fp8_weights'):
					self.model.model.layers[layer_idx].mlp.shared_experts._unregister_fp8_weights()
				for routed_expert_idx in range(self.model_config.num_local_experts):
					if hasattr(self.model.model.layers[layer_idx].mlp.experts[routed_expert_idx], '_unregister_fp8_weights'):
						self.model.model.layers[layer_idx].mlp.experts[routed_expert_idx]._unregister_fp8_weights()
				if hasattr(self.model.model.layers[layer_idx].mlp, "cleanup"):
					self.model.model.layers[layer_idx].mlp.cleanup()

	def deep_free_model_memory(self):
		"""Deep cleanup of model and all its submodules"""
		
		if not hasattr(self, 'model'):
			logging.warning("No model attribute found.")
			return
		
		# Step 1: Set model to eval and disable gradients
		self.model.eval()
		with torch.no_grad():
			# Step 2: Recursively clear all module parameters and buffers
			def clear_module(module):
				# Clear parameters
				for param in module.parameters():
					param.data = torch.empty(0)
					if param.grad is not None:
						param.grad.data = torch.empty(0)
						param.grad = None
				
				# Clear buffers
				for buffer in module.buffers():
					buffer.data = torch.empty(0)
				
				# Clear module hooks
				module._forward_hooks.clear()
				module._forward_pre_hooks.clear()
				module._backward_hooks.clear()
				
				# Recursively clear submodules
				for submodule in module.children():
					clear_module(submodule)
			
			clear_module(self.model)
		
		# Step 3: Move to CPU and delete
		self.model.to('cpu')
		del self.model
		
		# Step 4: Clear optimizer if exists
		# if hasattr(self, 'optimizer'):
		# 	self.optimizer.zero_grad(set_to_none=True)
		# 	del self.optimizer
		
		# # Step 5: Clear any cached computational graphs
		# if torch.cuda.is_available():
		# 	torch.cuda.empty_cache()
		# 	torch.cuda.synchronize()
		
		# # Step 6: Aggressive garbage collection
		# import gc
		# for _ in range(3):  # Multiple passes can help
		# 	gc.collect()
		# 	if torch.cuda.is_available():
		# 		torch.cuda.empty_cache()