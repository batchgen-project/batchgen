import concurrent.futures
import copy
import functools
import psutil
import logging
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

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
import torch.distributed._symmetric_memory as symm_mem


from .utils import torch_gpu_mem_usage, create_position_ids_from_attention_mask
from .get_initializer import get_initializer
from .get_parallel_strategy_manager import get_parallel_strategy_manager
from batchgen.utils import config_torch_module_initializer
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager
from batchgen.models.engine_loader import core_engine

from batchgen.kv_cache.host_kv_mananger_config import (
	build_gpu_kv_config,
	build_host_kv_config,
)
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus

BATCHGEN_ENABLE_ALL_TO_ALL = os.environ.get("BATCHGEN_ENABLE_ALL_TO_ALL")
if BATCHGEN_ENABLE_ALL_TO_ALL == "1":
	try:
		from pplx_kernels import nvshmem_init
	except ImportError as exc:
		logging.warning("Failed to import pplx_kernels.nvshmem_init: %s", exc)
		nvshmem_init = None
else:
	nvshmem_init = None  # Optional dependency for All-To-All features


# logging.basicConfig(
# 	level=logging.INFO,  # Set to the lowest level to capture all messages
# 	format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
# 	datefmt="%Y-%m-%d %H:%M:%S",  # Customize timestamp format
# )

from .scheduler.scheduler import Scheduler
# nvtx = False
# if nvtx:
# 	nvidia_dlprof_pytorch_nvtx.init()
import sys
# import pynvml
# def print_gpu_memory(tag):
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#     info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#     logging.info(f"[{tag}] Global Used Memory: {info.used / 1024**2:.2f} MB")
#     pynvml.nvmlShutdown()

class query:
	def __init__(
		self,
		text: str = None,
		encoded: Dict[str, torch.Tensor] = None,
		decoded_tokens: torch.Tensor = None,
		kv_token_budget: Optional[int] = None,
	):
		self.text = text
		self.encoded = encoded
		self.decoded_tokens = decoded_tokens
		self.kv_token_budget = kv_token_budget



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
	global_host_kv_cache_size_gb: Optional[int] = None
	kv_dtype: str = "bfloat16"
	dist_init_addr: Optional[str] = None
	local_rank: int = 0
	rank: int = 0
	global_rank: int = 0
	world_size: int = 1
	gpu_arch: str = "hopper"

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

@dataclass
class BatchGenWorkerArgs:
	local_rank: int
	global_rank: int
	world_size: int
	nnode_rank: int
	nnodes: int
	dist_init_addr: str

	model_name: str
	hf_cache_dir: Optional[str]
	cache_dir: Optional[str]
	pt_ckpt_dir: Optional[str]
	host_kv_cache_size: int
	global_host_kv_cache_size_gb: int

	shm_name: str
	tensor_meta_shm_name: str
	enable_hugetlbfs: bool
	weight_byte_size: int
	skeleton_state_dict: Optional[Dict]

	device: int
	kv_dtype: str
	gpu_arch: str


class BatchGenWorker:
	"""
	Inference Runtime.
	
	"""
	def __init__(self, args: BatchGenWorkerArgs):
		logging.info(f"Rank {args.global_rank}: Initializing BatchGenWorker.")
		self.args = args
		self.model = None
		# self.hf_cache_dir = hf_cache_dir
		# hf_cache_dir will be deprecated in the future.
		# if (args.hf_cache_dir is None) and (args.cache_dir is not None):
		# 	self.hf_cache_dir = args.cache_dir
		self.hf_cache_dir = args.cache_dir
		self.huggingface_ckpt_name = args.model_name
		self.cache_dir = args.cache_dir
		self.pt_ckpt_dir = args.pt_ckpt_dir
		# self.max_input_length = max_input_length
		# self.max_decoding_length = max_decoding_length
		self.skeleton_state_dict = args.skeleton_state_dict
		# self.rank = rank
		self.dist_init_addr = args.dist_init_addr
		self.local_rank = args.local_rank
		self.global_rank = args.global_rank
		self.rank = args.global_rank
		self.world_size = args.world_size
		self.gpu_arch = args.gpu_arch
		# self.engine_config_json_dir = engine_config_json_dir
		self.kv_dtype = args.kv_dtype

		self.shm_name = args.shm_name
		self.tensor_meta_shm_name = args.tensor_meta_shm_name
		logging.info(f"Rank {self.rank}: Initializing shared memory segments.")
		logging.info(f"Rank {self.rank}: shm_name: {self.shm_name}, tensor_meta_shm_name: {self.tensor_meta_shm_name}, weight_byte_size: {self.args.weight_byte_size}, enable_hugetlbfs: {self.args.enable_hugetlbfs}")
		self.weights_storage = core_engine.Weights_Storage(self.local_rank)

		worker_kv_config = build_host_kv_config(
			model_name=args.model_name,
			host_kv_cache_size=args.global_host_kv_cache_size_gb * (1024**3),
		)
		self.host_paged_kv_worker_view = core_engine.MLAHostPagedKVWorkerView(worker_kv_config)
		self.gpu_paged_kv_cache_manager = None
		
		# self.core_engine.init_weight_storage(self.shm_name, self.tensor_meta_shm_name,
		# 			self.args.weight_byte_size, 
		# 			self.args.enable_hugetlbfs)
		self.weights_storage.Init(self.shm_name, self.args.weight_byte_size, 
					self.tensor_meta_shm_name,
					self.args.enable_hugetlbfs)	
		logging.info(f"Rank {self.rank}: Shared memory segments initialized.")
		
		logging.info(f"Rank {self.rank}: Initializing core engine.")
		self.host_paged_kv_worker_view.initialize(device_index=self.local_rank, create_region=False)
		logging.info(f"Rank {self.rank}: Host KV manager view initialized.")


		# Global batch state
		self.global_batch: Optional[SequenceBatch] = None



	def Init(self, max_input_length, max_decoding_length, num_queries):
		self.max_input_length = max_input_length
		self.max_decoding_length = max_decoding_length
		logging.info(f"Initializing batchgen with global rank {self.args.global_rank} and world size {self.args.world_size} with PID: {os.getpid()}")
		config_torch_module_initializer()
		self.model_config = AutoConfig.from_pretrained(
			self.cache_dir,
			trust_remote_code=True,
			local_files_only=True,
		)
		self.tokenizer = AutoTokenizer.from_pretrained(
			# self.huggingface_ckpt_name,
			self.cache_dir,
			# cache_dir=self.hf_cache_dir,
			trust_remote_code=True,
			local_files_only=True,
		)
		self.tokenizer.padding_side = "right"

		logging.info(f"Rank {self.rank}: Start initializing engine config.")
		config_scheduler = Scheduler(self.max_input_length, self.max_decoding_length, self.args.world_size)
		self.engine_config = config_scheduler.generate_config()
		# self.engine_config = parse_config_from_json(engine_config_json_dir)
		self.engine_config.Basic_Config.device = self.args.device
		self.engine_config.Basic_Config.device_torch = torch.device(
			f"cuda:{self.args.device}"
		)
		self.engine_config.Basic_Config.max_decoding_length = (
			max_decoding_length
		)
		self.engine_config.Basic_Config.padding_length = self.max_input_length
		# self.engine_config.Basic_Config.num_queries = self.num_queries
		self.engine_config.Basic_Config.rank = self.global_rank
		self.engine_config.Basic_Config.world_size = self.world_size

		# if(self.rank == 0):
		# 	print(self.engine_config)
		if not self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens:
			logging.warning(f"kv_buffer_num_tokens is set to {self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens}")
			# exit()
		self.device = self.args.device
		self.torch_device = torch.device(f"cuda:{self.args.device}")
		self.host_kv_cache_size = self.args.host_kv_cache_size
		self.global_host_kv_cache_size_gb = self.args.global_host_kv_cache_size_gb

		self.attn_mode = None
		self.query_book = None
		self.model_batch_book = {}
		# TODO:
		self.token_k_cache_byte_size = 2048  # mixtral
		self.num_k_storage_tokens = math.floor(
			50 * (1024**3) / 32 / 2048
		)  # 50G k cache, 50G v cache. 192G test-bed.

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
			"engine_config_json_dir": None,
			"host_kv_cache_size": self.host_kv_cache_size,
			"global_host_kv_cache_size_gb": self.global_host_kv_cache_size_gb,
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
		# if self.global_rank == 0:
		# 	logging.info(f"Input Arguments for Initializer: {input_arguments}")
			
		self.input_arguments = InputArguments(**input_arguments)
		self.initializer = get_initializer(self.huggingface_ckpt_name)
		self.initializer = self.initializer(self.input_arguments)
		self.core_engine, self.engine_config, self.model_config, self.hf_model_config = (
			self.initializer.Init(self.weights_storage)
		)

		self.core_engine.host_paged_kv_worker_view = self.host_paged_kv_worker_view
		# self.queries, self.model_batches = self.vanilla_batching(
		# 	self.global_queries, self.global_rank, self.world_size)
		# self.num_queries = len(self.queries)
		# TODO: Move to centralized config later.
		self.engine_config.Basic_Config.num_queries = num_queries
		
		self.parallel_manager = get_parallel_strategy_manager(self.huggingface_ckpt_name)
		self.parallel_manager = self.parallel_manager(
			self.hf_model_config,
			self.engine_config,
			self.model_config,
			self.core_engine,
			self.skeleton_state_dict,
			# None,
			self.local_rank,
			self.global_rank,
			self.world_size
		)        
				
		logging.info(f"Engine on device {self.device} initialized.")


	def _tokenization(self, local_batch: List[str]):
		"""
		Handle tokenization of input sequences. 
		Input: 
		Output: 
		
		"""
		# Step 1: Create mapping from LOCAL query idx to query.
		self.query_book = {
			query_idx: query(
				text=text,
				decoded_tokens=torch.zeros(
					1, self.max_decoding_length, dtype=torch.int64
				),
			)
			for query_idx, text in enumerate(local_batch)
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

	def _get_sequence_token_budget(self, sequence_id: int) -> int:
		"""Return cached host allocation tokens for a sequence, computing once."""
		if not hasattr(self, "query_book") or self.query_book is None:
			raise RuntimeError("query_book is not initialized before KV allocation")
		query_entry = self.query_book.get(sequence_id)
		if query_entry is None or query_entry.encoded is None:
			raise KeyError(f"Missing query entry for sequence {sequence_id}")
		if query_entry.kv_token_budget is not None:
			return query_entry.kv_token_budget
		attention_mask = query_entry.encoded.get("attention_mask")
		if attention_mask is None:
			raise KeyError(
				f"No attention_mask available for sequence {sequence_id}"
			)
		mask_row = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
		input_tokens = int(mask_row[: self.max_input_length].sum().item())
		total_tokens = input_tokens + self.max_decoding_length
		query_entry.kv_token_budget = total_tokens
		return total_tokens

	def _compute_host_kv_sequence_tokens(
		self, sequence_ids: List[int]
	) -> List[int]:
		"""Reuse cached token budgets so host/GPU allocations stay consistent."""
		return [self._get_sequence_token_budget(sequence_id) for sequence_id in sequence_ids]


	def _bind_gpu_paged_kv_manager(
		self, manager: GPUPagedKVCacheManager
	) -> None:
		"""Bind GPU KV manager to both worker and core_engine."""
		self.gpu_paged_kv_cache_manager = manager
		if hasattr(self.core_engine, "gpu_paged_kv_manager"):
			# Note: do not set None here because in C++ side it will lead to runtime error.
			self.core_engine.gpu_paged_kv_manager = manager


	def _ensure_gpu_paged_kv_manager(
		self, sequence_tokens: Sequence[int]
	) -> GPUPagedKVCacheManager:
		"""Return a GPU paged KV manager with enough pages for `sequence_tokens`."""
		gpu_config = build_gpu_kv_config(
			model_name=self.huggingface_ckpt_name,
			sequence_tokens=sequence_tokens,
		)

		manager = self.gpu_paged_kv_cache_manager
		required_pages = gpu_config.num_pages
		current_pages = (
			getattr(getattr(manager, "config", None), "num_pages", 0)
			if manager is not None
			else 0
		)

		if manager is not None and current_pages >= required_pages:
			manager.initialize()
			self._bind_gpu_paged_kv_manager(manager)
			return manager

		if manager is not None:
			manager.destroy()
		
		logging.info(
			"Rank %s creating GPUPagedKVCacheManager on %s: "
			"current pages=%d, required pages=%d",
			self.rank, self.local_rank, current_pages, required_pages
		)

		manager = GPUPagedKVCacheManager(
			config=gpu_config,
			device=self.local_rank,
		)
		manager.initialize()
		self._bind_gpu_paged_kv_manager(manager)

		logging.info(
			"Rank %s initialized GPUPagedKVCacheManager on %s with %d pages",
			self.rank,
			manager.device,
			gpu_config.num_pages,
		)
		return manager

	# def _prepare_gpu_paged_kv_cache(self, local_sequence_ids: List[int]) -> None:
	# 	"""Allocate GPU KV pages and load host-resident KV for the batch."""
	# 	if not local_sequence_ids:
	# 		return
	# 	sequence_tokens = self._compute_host_kv_sequence_tokens(local_sequence_ids)
	# 	manager = self._ensure_gpu_paged_kv_manager(sequence_tokens)
	# 	global_sequence_ids = self._build_global_sequence_ids(local_sequence_ids)
	# 	logging.info(
	# 		f"Rank {self.rank} Allocating GPU KV pages (local->global): "
	# 		f"{self._format_sequence_ids_for_log(local_sequence_ids)}"
	# 	)
	# 	manager.allocate_pages_for_sequences(global_sequence_ids, sequence_tokens)
	# 	manager.rebuild_page_table(global_sequence_ids)
	# 	self._load_host_kv_to_gpu(manager, global_sequence_ids)
	def _prepare_gpu_paged_kv_cache(self, local_sequence_ids: List[int]) -> None:
		"""Allocate GPU KV pages and load host-resident KV for the batch."""
		if not local_sequence_ids:
			return
		
		# Convert local indices to global_idx (consistent with host KV registration)
		global_sequence_ids = self._local_indices_to_global_seq_ids(local_sequence_ids)
		
		sequence_tokens = self._compute_host_kv_sequence_tokens(local_sequence_ids)
		manager = self._ensure_gpu_paged_kv_manager(sequence_tokens)
		
		logging.info(
			f"Rank {self.rank} Allocating GPU KV pages for global_idx: {global_sequence_ids}"
		)
		
		manager.allocate_pages_for_sequences(global_sequence_ids, sequence_tokens)
		manager.rebuild_page_table(global_sequence_ids)
		self._load_host_kv_to_gpu(manager, global_sequence_ids)

	def _load_host_kv_to_gpu(
		self,
		manager: GPUPagedKVCacheManager,
		global_sequence_ids: List[int],
	) -> None:
		"""Copy prefetched host KV pages into the GPU cache."""
		if not global_sequence_ids:
			return
		copy_start = time.perf_counter()
		worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		if worker_view is None:
			raise RuntimeError("Host paged KV worker view is not bound to the core engine")
		
		# global_sequence_ids are now already global_idx values, no conversion needed
		sequence_tensor = torch.tensor(global_sequence_ids, dtype=torch.int64, device="cpu")
		k_ptrs, v_ptrs = manager.export_layer_page_pointer_table()
		load_task = worker_view.async_load_layer_kv_to_device(
			sequence_ids=sequence_tensor,
			k_device_ptrs=k_ptrs,
			v_device_ptrs=v_ptrs,
		)
		load_task.wait()
		torch.cuda.synchronize(self.torch_device)
		load_duration = time.perf_counter() - copy_start
		logging.info(
			"Rank %s Loaded host KV for %d sequences into GPU cache in %.3fs",
			self.rank,
			len(global_sequence_ids),
			load_duration,
		)


	def _release_gpu_kv_pages(self, local_sequence_ids: List[int]) -> None:
		"""Return GPU KV pages associated with the provided local sequence ids."""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None or not local_sequence_ids:
			return
		
		# Convert to global_idx (consistent with registration)
		global_sequence_ids = self._local_indices_to_global_seq_ids(local_sequence_ids)
		
		try:
			manager.free_pages_for_sequences(global_sequence_ids)
			logging.info(
				f"Rank {self.rank} Released GPU KV pages for global_idx: {global_sequence_ids}"
			)
		except KeyError as exc:
			logging.warning(
				"Rank %s failed to release GPU KV pages for %s: %s",
				self.rank,
				global_sequence_ids,
				exc,
			)

	def _destroy_gpu_paged_kv_cache(self, *, empty_cuda_cache: bool = False) -> None:
		"""Destroy the GPU paged KV cache manager if it is present."""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			return
		manager.destroy(empty_cuda_cache=empty_cuda_cache)

	def _local_batching(self):
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
		max_local_queries = math.ceil(self.num_global_queries / self.world_size)
		max_num_batches = math.ceil(max_local_queries / model_batch_size)
		
		# Step 5: Create model batches for this rank's local queries
		model_batches = []
		
		for batch_idx in range(max_num_batches):
			# Calculate the local batch range for this rank
			local_batch_start = batch_idx * model_batch_size
			local_batch_end = min((batch_idx + 1) * model_batch_size, self.num_local_queries)
			
			if local_batch_start < self.num_local_queries:
				# This rank has sequences in this batch
				local_indices = list(range(local_batch_start, local_batch_end))
				model_batches.append(local_indices)
			else:
				# This rank has no sequences in this batch - append empty list
				model_batches.append([])
		
		# Verification logging
		logging.info(f"=" * 60)
		logging.info(f"Rank {self.rank} Batching Summary:")
		logging.info(f"  Total global queries: {self.num_global_queries}")
		logging.info(f"  Local query count: {self.num_local_queries}")
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
		return model_batches

	def process_new_batch_bak(self, batch: List[str], num_global_queries: int):
		"""
		Future API.
		"""
		self.num_global_queries = num_global_queries
		self.num_local_queries = len(batch)
		self._tokenization(batch)
		self.model_batches = self._local_batching()
		return self.generate()

	def process_new_batch(self, global_prompts: List[str]) -> List[torch.Tensor]:
		"""
		Process a global batch of prompts.
		All ranks receive the same global_prompts and maintain consistent state.
		"""
		logging.info(
			f"Rank {self.rank}: Processing global batch of {len(global_prompts)} sequences"
		)

		# Step 1: Initialize global batch
		self.global_batch = SequenceBatch()
		for idx, text in enumerate(global_prompts):
			seq = SequenceEntry(
				uuid=f"seq_{idx}",
				global_idx=idx,
				prompt_length=0,  # Will be set after tokenization
				max_decode_length=self.max_decoding_length,
				text=text,
			)
			self.global_batch.add_sequence(seq)

		# Step 2: Tokenize all sequences (all ranks do this identically)
		self._tokenize_global_batch()

		# Step 3: Assign sequences to ranks (round-robin)
		self._assign_sequences_to_ranks()

		# Step 4: Build query_book for backward compatibility
		self._build_local_query_book()

		# Step 5: Set counts for compatibility
		self.num_global_queries = len(global_prompts)
		self.num_local_queries = len(self.global_batch.get_sequences_for_rank(self.rank))

		# Step 6: Run generation with new KV-driven scheduling
		return self.generate()

	# Helper methods for UUID <-> local index conversion
	def _local_to_uuid(self, local_idx: int) -> str:
		return self._local_to_uuid_map.get(local_idx, "")

	def _uuid_to_local(self, uuid: str) -> int:
		return self._uuid_to_local_map.get(uuid, -1)

	def _get_my_sequences_by_status(self, status: SequenceStatus) -> List[str]:
		"""Get UUIDs of sequences assigned to this rank with given status."""
		return self.global_batch.get_sequences_for_rank_with_status(self.rank, status)


	def generate_bak(self):
		self.comm = None
		if os.getenv("BATCHGEN_ENABLE_ALL_TO_ALL","0") == "0":
			from batchgen.distributed.utils import StatelessProcessGroup
			from batchgen.distributed.device_communicators.pynccl import PyNcclCommunicator
			self.rank = dist.get_rank()
			self.world_size = dist.get_world_size()
			device = torch.device("cuda", self.rank % torch.cuda.device_count())
			comm_master_addr = os.getenv("COMM_MASTER_ADDR")
			
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
			self._config_prefill(model_batch_idx)
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
			
			tmp_start = time.perf_counter()
			torch.cuda.empty_cache()
			# Log memory usage before decode phase configuration:
			free_memory, total_memory = torch.cuda.mem_get_info()
			free_memory = free_memory / 1024 / 1024 / 1024
			total_memory = total_memory / 1024 / 1024 / 1024
			logging.info(
				f"Rank: {self.rank} Device torch memory usage before decode phase: {torch.cuda.memory_allocated(self.torch_device) / (1024**3)} GB / {total_memory} GB"
			)
			logging.info(
				f"Rank: {self.rank} Device torch free memory before decode phase: {free_memory} GB / {total_memory} GB"
			)
			self._config_decoding(len(new_token), model_batch_idx, self.comm)
			# self.core_engine.copy_kv_to_worker(self.model_batches[model_batch_idx], self.max_input_length + self.max_decoding_length)
			if self.engine_config.Basic_Config.attn_mode == 3:
				# FULL GPU DECODING MODE.
				# Need to instantiate GPU KV-Cache Here. 
				# gpu_kv_cache = GPUKVCacheManager(self.engine_config).init(self.core_engine)
				# for query_idx in self.model_batches[model_batch_idx]:
				# 	gpu_kv_cache.allocate_pages(query_idx, self.max_input_length + self.max_decoding_length)
				# 	gpu_kv_cache.load_offloaded_context(query_idx, self.max_input_length) # Load offloaded context kv-cache to gpu.

				if self.model_config.model_type == "deepseek_v3":
					# show rank 0 gpu memory usage before getting past key values
					free_memory, total_memory = torch.cuda.mem_get_info()
					free_memory = free_memory / 1024 / 1024 / 1024
					total_memory = total_memory / 1024 / 1024 / 1024
					logging.info(
						f"Rank: {self.rank} Device torch memory usage before getting past key values: {torch.cuda.memory_allocated(self.torch_device) / (1024**3)} GB / {total_memory} GB"
					)
					logging.info(
						f"Rank: {self.rank} Device torch free memory before getting past key values: {free_memory} GB / {total_memory} GB"
					)


					# past_key_states= self.core_engine.get_past_key_states(self.model_batches[model_batch_idx], self.max_input_length + self.max_decoding_length)
					past_key_states = None # Now we donot need this
					# Pad the kv cache to be multiple of 64
					# bsz, kv_seqlen, _ = past_key_states[0].size()
					# if self.engine_config.Basic_Config.kv_dtype == "bfloat16":
					# 	if kv_seqlen % 64 != 0:
					# 		pad_len = 64 - (kv_seqlen % 64)
					# 		for i in range(len(past_key_states)):
					# 			past_key_states[i] = torch.cat([
					# 				past_key_states[i], 
					# 				torch.zeros((bsz, pad_len, past_key_states[i].size(-1)), device=past_key_states[i].device, dtype=past_key_states[i].dtype)
					# 			], dim=1)
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
			# self.core_engine.clear_kv_storage()
			self._release_host_kv_pages(self.model_batches[model_batch_idx])
			self._unregister_fp8_weights()
			self.deep_free_model_memory()
			del past_key_states
			del scale_dict
			# gc.collect()
			dist.barrier()
		
	
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
			# for query_idx in range(self.num_queries)
			for query_idx in range(self.num_local_queries)
		]

		# Print first 5 sequences
		# for query_idx in range(max(0, self.num_local_queries - 5), self.num_local_queries):
		# 	logging.info(
		# 		f"Decoded tokens: {res[query_idx].squeeze().tolist()}"
		# 	)

		# Gather results from all rank to rank 0
		# logging.info(f"Rank {self.rank} res: {res}")
		all_results = [None] * self.world_size
		dist.all_gather_object(all_results, res)
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


	def _config_prefill(self, model_batch_idx: int):
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
		# self.core_engine.clear_kv_storage()
		logging.info(f"clear_kv_storage took {time.perf_counter() - step_start:.4f}s")
		
		# Step 8: Start H2D worker
		step_start = time.perf_counter()
		self.core_engine.start_h2d_worker()
		logging.info(f"start_h2d_worker took {time.perf_counter() - step_start:.4f}s")

		# Step 9: Destroy GPU paged KV cache state
		step_start = time.perf_counter()
		self._destroy_gpu_paged_kv_cache()
		logging.info(f"destroy_gpu_paged_kv_cache took {time.perf_counter() - step_start:.4f}s")

		# Step 10: Allocate Pages for Sequences
		step_start = time.perf_counter()
		sequence_ids = self.model_batches[model_batch_idx]

		if sequence_ids:
			global_sequence_ids = self._build_global_sequence_ids(sequence_ids)
			logging.info(
				f"Rank {self.rank} Allocating host KV pages (local->global): "
				f"{self._format_sequence_ids_for_log(sequence_ids)}"
			)
			self.core_engine.host_paged_kv_worker_view.register_sequences(
				global_sequence_ids
			)

			sequence_tokens = self._compute_host_kv_sequence_tokens(sequence_ids)
			self.core_engine.host_paged_kv_worker_view.allocate_pages_for_sequences(
				list(zip(global_sequence_ids, sequence_tokens))
			)

			kv_stats = self.core_engine.host_paged_kv_worker_view.get_stats()
			logging.info(
				f"Rank {self.rank} Host KV Cache Stats after allocation: {kv_stats}"
			)

		logging.info(f"allocate_pages_for_sequences took {time.perf_counter() - step_start:.4f}s")

		
		total_time = time.perf_counter() - start_time
		logging.info(f"_config_prefill completed in {total_time:.4f}s")



	def init_nvshmem(self):
		if BATCHGEN_ENABLE_ALL_TO_ALL != "1" or nvshmem_init is None:
			logging.info("Skipping NVSHMEM initialization; BATCHGEN_ENABLE_ALL_TO_ALL is disabled or nvshmem_init missing")
			return
		import nvshmem.core as nvshmem
		from cuda.core.experimental import Device	
		# 1. Standard Torch Distributed Init
		rank = dist.get_rank()
		world_size = dist.get_world_size()
		local_rank = rank % torch.cuda.device_count()
		torch.cuda.set_device(local_rank)

		# 2. NVSHMEM Init (Allocates the Symmetric Heap here)
		# Ensure NVSHMEM_SYMMETRIC_SIZE is set in env vars before this runs
		dev = Device(local_rank)
		dev.set_current()
		dist.barrier()
		nvshmem_init(
			global_rank=rank,
			local_rank=local_rank,
			world_size=world_size,
			device=dev
		)
		print(f"Rank {rank}: NVSHMEM initialized and Symmetric Heap allocated.")
	
	def _config_decoding(self, num_seq, model_batch_idx: int, comm=None):
		logging.info(f"Start Config Decoding")
		self.deep_free_model_memory()
		self.init_nvshmem()
		
		# Initialize symmetric memory once during model initialization
		# if not symm_mem.is_nvshmem_available():
		# 	logging.warning("NVSHMEM is not available. Symmetric memory features will be disabled.")
		# 	symm_mem.set_backend("NCCL")
		# else:
		# 	symm_mem.set_backend("NVSHMEM")
		# group_name = dist.group.WORLD.group_name
		# symm_mem.enable_symm_mem_for_group(group_name)


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
			# self.core_engine.clear_kv_buffer()
			self.core_engine.clear_weight_copy_queue()
			self.core_engine.reset_decoding_buffer()
			self.core_engine.set_weight_copy_queue(self.weight_copy_task)
			self.core_engine.start_h2d_worker()
		else:
			sequence_ids = self.model_batches[model_batch_idx]
			self._prepare_gpu_paged_kv_cache(sequence_ids)
			self.model, self.weight_copy_task = self.parallel_manager.pure_gpu_decoding(max_num_seq, comm)

			self.set_phase("decode")
			self.core_engine.stop_h2d_worker()
			self.core_engine.clear_kv_copy_queue()
			# self.core_engine.clear_kv_buffer()
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
			gpu_manager = getattr(self, "gpu_paged_kv_cache_manager", None)
			if gpu_manager is None:
				gpu_manager = getattr(self.core_engine, "gpu_paged_kv_manager", None)
			Attn_Wrapper.gpu_paged_kv_manager = gpu_manager
			Attn_Wrapper.host_paged_kv_worker_view = getattr(
				self.core_engine,
				"host_paged_kv_worker_view",
				None,
			)
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
			Attn_Wrapper.gpu_paged_kv_manager = None
			Attn_Wrapper.host_paged_kv_worker_view = None
		
		
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
		timeout = timedelta(minutes=15)
		# os.environ['GLOO_SOCKET_IFNAME'] = 'bond0'
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
		self.model.to('cpu')
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
		if hasattr(self, 'optimizer'):
			self.optimizer.zero_grad(set_to_none=True)
			del self.optimizer
		
		# Step 5: Clear any cached computational graphs
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.synchronize()
		
		# Step 6: Aggressive garbage collection
		import gc
		for _ in range(3):  # Multiple passes can help
			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	
	def _tokenize_global_batch(self) -> None:
		"""
		Tokenize all sequences in the global batch.
		All ranks execute this identically to maintain consistent state.
		"""
		if self.global_batch is None:
			raise RuntimeError("Global batch not initialized")

		for seq in self.global_batch:
			tokenized = self.tokenizer(
				seq.text,
				return_tensors="pt",
				max_length=self.max_input_length,
				truncation=True,
				padding="max_length",
			)

			# Extend for decoding space
			extended_size = self.max_input_length + self.max_decoding_length
			input_ids_extended = torch.zeros(
				(1, extended_size), dtype=tokenized["input_ids"].dtype
			)
			attention_mask_extended = torch.zeros(
				(1, extended_size), dtype=tokenized["attention_mask"].dtype
			)

			seq_len = tokenized["input_ids"].size(1)
			input_ids_extended[0, :seq_len] = tokenized["input_ids"][0, :]
			attention_mask_extended[0, :seq_len] = tokenized["attention_mask"][0, :]

			seq.input_ids = input_ids_extended
			seq.attention_mask = attention_mask_extended
			seq.decoded_tokens = torch.zeros(
				1, self.max_decoding_length, dtype=torch.int64
			)

			# Update actual prompt length and KV budget
			actual_prompt_len = int(
				tokenized["attention_mask"][0, :self.max_input_length].sum().item()
			)
			seq.prompt_length = actual_prompt_len
			seq.current_context_length = actual_prompt_len
			seq.kv_token_budget = actual_prompt_len + self.max_decoding_length

		logging.info(
			f"Rank {self.rank}: Tokenized {len(self.global_batch)} sequences"
		)

	def _assign_sequences_to_ranks(self) -> None:
		"""
		Assign sequences to ranks using round-robin distribution.
		All ranks execute this identically to maintain consistent assignment.
		"""
		if self.global_batch is None:
			raise RuntimeError("Global batch not initialized")

		# Round-robin assignment based on global_idx
		for seq in self.global_batch:
			assigned_rank = seq.global_idx % self.world_size
			self.global_batch.assign_rank(seq.uuid, assigned_rank)

		# Log assignment summary
		my_seqs = self.global_batch.get_sequences_for_rank(self.rank)
		logging.info(
			f"Rank {self.rank}: Assigned {len(my_seqs)} sequences: {my_seqs}"
		)

	def _build_local_query_book(self) -> None:
		"""
		Build query_book from global_batch for sequences assigned to this rank.
		Maps local indices (0, 1, 2, ...) to sequence data for backward compatibility.
		"""
		my_uuids = sorted(
			self.global_batch.get_sequences_for_rank(self.rank),
			key=lambda uuid: self.global_batch.get_sequence(uuid).global_idx
		)

		self.query_book = {}
		self._local_to_uuid_map: Dict[int, str] = {}
		self._uuid_to_local_map: Dict[str, int] = {}

		for local_idx, uuid in enumerate(my_uuids):
			seq = self.global_batch.get_sequence(uuid)

			# Create query object for backward compatibility
			self.query_book[local_idx] = query(
				text=seq.text,
				encoded={
					"input_ids": seq.input_ids,
					"attention_mask": seq.attention_mask,
				},
				decoded_tokens=seq.decoded_tokens,
				kv_token_budget=seq.kv_token_budget,
			)

			self._local_to_uuid_map[local_idx] = uuid
			self._uuid_to_local_map[uuid] = local_idx
 
		logging.info(
			f"Rank {self.rank}: Built local query_book with {len(self.query_book)} entries"
		)	

	def _get_host_kv_free_pages(self) -> int:
		"""Get current free pages from host KV cache."""
		stats = self.host_paged_kv_worker_view.get_stats()
		return stats.num_free_pages

	def _prepare_prefill_batch(self) -> List[str]:
		"""
		Select sequences for prefill based on host KV cache capacity.
		All ranks execute this identically to get the same global prefill batch.
		
		The host KV cache is SHARED - we compute total capacity and select
		sequences that fit. Each rank will then only allocate for its own subset.
		"""
		queueing_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.QUEUEING)
		queueing_uuids.sort(key=lambda uuid: self.global_batch.get_sequence(uuid).global_idx)
		
		if not queueing_uuids:
			return []
		
		# Total available pages in the SHARED host KV cache
		available_pages = self._get_host_kv_free_pages()
		
		prefill_batch = []
		pages_allocated = 0
		
		for uuid in queueing_uuids:
			seq = self.global_batch.get_sequence(uuid)
			seq_pages = seq.get_pages_required()
			
			if pages_allocated + seq_pages <= available_pages:
				prefill_batch.append(uuid)
				pages_allocated += seq_pages
			else:
				break
		
		logging.info(
			f"Rank {self.rank}: Prepared prefill batch with {len(prefill_batch)} sequences, "
			f"using {pages_allocated}/{available_pages} pages"
		)
		
		return prefill_batch

	def _prepare_decode_batch(self) -> List[str]:
		"""
		Select sequences for decode phase from PREFILLED sequences.
		Limited by decode batch size.
		All ranks execute this identically to maintain consistent global state.
		
		Returns:
			List of UUIDs to decode (global batch, not rank-specific)
		"""
		# Get all PREFILLED sequences sorted by global_idx
		prefilled_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		prefilled_uuids.sort(key=lambda uuid: self.global_batch.get_sequence(uuid).global_idx)
		
		if not prefilled_uuids:
			return []
		
		# Limit by decode batch size
		decode_batch_size = self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size
		decode_batch = prefilled_uuids[:decode_batch_size]
		
		logging.info(
			f"Rank {self.rank}: Prepared decode batch with {len(decode_batch)} sequences "
			f"(max batch size: {decode_batch_size})"
		)
		
		return decode_batch

	def _get_local_indices_for_uuids(self, uuids: List[str]) -> List[int]:
		"""
		Convert global UUIDs to local indices for sequences assigned to this rank.
		Only includes sequences that belong to this rank.
		"""
		local_indices = []
		for uuid in uuids:
			if uuid in self._uuid_to_local_map:
				local_indices.append(self._uuid_to_local_map[uuid])
		return local_indices

	def _update_batch_status(self, uuids: List[str], new_status: SequenceStatus) -> None:
		"""Update status for all sequences in a batch."""
		for uuid in uuids:
			self.global_batch.update_status(uuid, new_status)	

	def generate(self):
		"""
		Main generation loop with KV-cache-driven scheduling.
		
		Flow:
		1. Prefill until host KV cache is full
		2. Decode all prefilled sequences
		3. Release KV pages for completed sequences
		4. Repeat until all sequences completed
		"""
		# Initialize communicator
		self.comm = None
		if os.getenv("BATCHGEN_ENABLE_ALL_TO_ALL", "0") == "0":
			from batchgen.distributed.utils import StatelessProcessGroup
			from batchgen.distributed.device_communicators.pynccl import PyNcclCommunicator
			self.rank = dist.get_rank()
			self.world_size = dist.get_world_size()
			device = torch.device("cuda", self.rank % torch.cuda.device_count())
			comm_master_addr = os.getenv("COMM_MASTER_ADDR")
			
			try:
				group = StatelessProcessGroup.create(
					host=comm_master_addr,
					port=20003,
					rank=self.rank,
					world_size=self.world_size,
					data_expiration_seconds=6000,
				)
				self.comm = PyNcclCommunicator(group=group, device=device)
			except Exception as e:
				logging.error(f"Rank {self.rank}: PyNccl communicator initialization failed - {e}")
				raise RuntimeError(f"Rank {self.rank}: PyNccl communicator initialization failed - {e}")

		# Timing stats
		generation_start_time = time.perf_counter()
		prefill_time = 0
		decoding_time = 0
		config_prefill_time = 0
		config_decode_time = 0
		
		iteration = 0
		
		# Main loop: continue until all sequences are completed
		while not self.global_batch.all_completed():
			iteration += 1
			logging.info(f"{'='*60}")
			logging.info(f"Rank {self.rank}: Starting iteration {iteration}")
			logging.info(
				f"  QUEUEING: {self.global_batch.count_by_status(SequenceStatus.QUEUEING)}, "
				f"PREFILLED: {self.global_batch.count_by_status(SequenceStatus.PREFILLED)}, "
				f"COMPLETED: {self.global_batch.count_by_status(SequenceStatus.COMPLETED)}"
			)
			
			# ============ PREFILL PHASE ============
			# Only prefill if there are queueing sequences and we have capacity
			if self.global_batch.has_queueing():
				dist.barrier()
				
				# Prepare prefill batch (all ranks compute same batch)
				prefill_uuids = self._prepare_prefill_batch()
				
				if prefill_uuids:
					# Update status: QUEUEING -> IN_PREFILL
					self._update_batch_status(prefill_uuids, SequenceStatus.IN_PREFILL)
					
					# Get local indices for this rank's sequences
					local_prefill_indices = self._get_local_indices_for_uuids(prefill_uuids)
					
					# Config prefill
					tmp_start = time.perf_counter()
					self._config_prefill_for_batch(prefill_uuids)
					config_prefill_time += time.perf_counter() - tmp_start
					
					# Execute prefill
					prefill_start_time = time.perf_counter()
					if local_prefill_indices:
						with torch.inference_mode():
							new_token = self.prefill(local_prefill_indices)
					else:
						new_token = torch.empty((0, 1), dtype=torch.int64, device=self.torch_device)
						logging.info(f"Rank {self.rank}: No local sequences to prefill")
					prefill_time += time.perf_counter() - prefill_start_time
					
					self._unregister_fp8_weights()
					
					# Update status: IN_PREFILL -> PREFILLED
					self._update_batch_status(prefill_uuids, SequenceStatus.PREFILLED)
					
					dist.barrier()
			
			# ============ DECODE PHASE ============
			# Decode all prefilled sequences in batches
			while self.global_batch.has_prefilled():
				dist.barrier()
				
				# Prepare decode batch (all ranks compute same batch)
				decode_uuids = self._prepare_decode_batch()
				
				if not decode_uuids:
					break
				
				# Update status: PREFILLED -> IN_DECODE
				self._update_batch_status(decode_uuids, SequenceStatus.IN_DECODE)
				
				# Get local indices for this rank's sequences
				local_decode_indices = self._get_local_indices_for_uuids(decode_uuids)
				
				# Config decoding
				tmp_start = time.perf_counter()
				torch.cuda.empty_cache()
				self._config_decoding_for_batch(decode_uuids, local_decode_indices, self.comm)
				
				# Prepare KV states
				past_key_states = None
				past_value_states = None
				scale_dict = None
				
				if self.engine_config.Basic_Config.attn_mode == 3:
					if self.model_config.model_type == "deepseek_v3":
						if self.engine_config.Basic_Config.kv_dtype == "float8_e4m3fn":
							if local_decode_indices:
								scale_dict = self.core_engine.get_kv_scale(
									local_decode_indices,
									self.max_input_length + self.max_decoding_length
								)
				
				config_decode_time += time.perf_counter() - tmp_start
				
				dist.barrier()
				
				# Execute decoding
				decoding_start_time = time.perf_counter()
				with torch.inference_mode():
					logging.info(f"Rank {self.rank}: Decoding batch size: {len(local_decode_indices)}")
					# Get the new tokens from prefill for this batch
					if local_decode_indices:
						# Collect the first decoded token for each sequence
						new_tokens = torch.cat([
							self.query_book[idx].decoded_tokens[:, 0:1]
							for idx in local_decode_indices
						], dim=0).to(self.torch_device)
					else:
						new_tokens = torch.empty((0, 1), dtype=torch.int64, device=self.torch_device)
					
					self.decoding(new_tokens, local_decode_indices, past_key_states, past_value_states, scale_dict)
				decoding_time += time.perf_counter() - decoding_start_time
				
				# Update status: IN_DECODE -> COMPLETED
				self._update_batch_status(decode_uuids, SequenceStatus.COMPLETED)
				
				# Release KV pages for completed sequences
				self._release_host_kv_pages_for_batch(decode_uuids)
				
				self._unregister_fp8_weights()
				self.deep_free_model_memory()
				
				del past_key_states
				del scale_dict
				
				dist.barrier()
		
		# Log timing stats
		generation_time = time.perf_counter() - generation_start_time
		phase_switching_time = config_prefill_time + config_decode_time
		
		logging.info(
			f"Rank {self.rank} Generation completed:\n"
			f"  Prefill total time: {prefill_time:.1f}s\n"
			f"  Decoding total time: {decoding_time:.1f}s\n"
			f"  Generation total time: {generation_time:.1f}s\n"
			f"  Phase switching time: {phase_switching_time:.1f}s\n"
			f"  Config prefill time: {config_prefill_time:.1f}s\n"
			f"  Config decoding time: {config_decode_time:.1f}s"
		)
		
		# Gather results
		res = [
			self.query_book[query_idx].decoded_tokens
			for query_idx in range(self.num_local_queries)
		]
		
		all_results = [None] * self.world_size
		dist.all_gather_object(all_results, res)
		all_results = [item for sublist in all_results for item in sublist]
		res_tensor = torch.cat(all_results, dim=0).cpu()
		
		if self.rank == 0:
			return [res_tensor]
		else:
			return []

	def _config_prefill_for_batch(self, prefill_uuids: List[str]) -> None:
		"""Configure prefill phase for a batch of sequences."""
		start_time = time.perf_counter()
		logging.info(f"Rank {self.rank}: Starting _config_prefill_for_batch")
		
		# Configure model for prefill
		self.model, self.weight_copy_task = self.parallel_manager.configure_prefill()
		self.set_phase("prefill")
		
		# Reset engine state
		self.core_engine.stop_h2d_worker()
		self.core_engine.clear_weight_copy_queue()
		self.core_engine.reset_prefill_buffer()
		self.core_engine.set_weight_copy_queue(self.weight_copy_task)
		self.core_engine.start_h2d_worker()
		
		# Destroy GPU paged KV cache
		self._destroy_gpu_paged_kv_cache()
		
		# Only allocate host KV pages for THIS RANK's sequences in the prefill batch
		my_prefill_uuids = [uuid for uuid in prefill_uuids if uuid in self._uuid_to_local_map]
		
		if my_prefill_uuids:
			global_sequence_ids = []
			sequence_tokens = []
			
			for uuid in my_prefill_uuids:
				seq = self.global_batch.get_sequence(uuid)
				global_sequence_ids.append(seq.global_idx)
				sequence_tokens.append(seq.kv_token_budget)
			
			logging.info(
				f"Rank {self.rank}: Registering {len(global_sequence_ids)} sequences for host KV: {global_sequence_ids}"
			)
			
			self.core_engine.host_paged_kv_worker_view.register_sequences(global_sequence_ids)
			self.core_engine.host_paged_kv_worker_view.allocate_pages_for_sequences(
				list(zip(global_sequence_ids, sequence_tokens))
			)
			
			kv_stats = self.core_engine.host_paged_kv_worker_view.get_stats()
			logging.info(f"Rank {self.rank}: Host KV Stats after allocation: {kv_stats}")
		
		logging.info(f"Rank {self.rank}: _config_prefill_for_batch completed in {time.perf_counter() - start_time:.4f}s")


	def _config_decoding_for_batch(
		self, 
		decode_uuids: List[str], 
		local_decode_indices: List[int],
		comm=None
	) -> None:
		"""Configure decoding phase for a batch of sequences."""
		logging.info(f"Rank {self.rank}: Starting _config_decoding_for_batch")
		
		self.deep_free_model_memory()
		self.init_nvshmem()
		
		# Get number of sequences for each rank
		num_local_seq = len(local_decode_indices)
		num_seq_per_rank = torch.zeros(self.world_size, dtype=torch.int32, device=self.torch_device)
		num_seq_per_rank[self.rank] = num_local_seq
		dist.all_reduce(num_seq_per_rank, op=dist.ReduceOp.SUM)
		max_num_seq = int(num_seq_per_rank.max().item())
		
		if self.world_size <= 8:
			self.model, self.weight_copy_task = self.parallel_manager.configure_decoding()
			self.set_phase("decode")
			self.core_engine.stop_h2d_worker()
			self.core_engine.clear_kv_copy_queue()
			self.core_engine.clear_weight_copy_queue()
			self.core_engine.reset_decoding_buffer()
			self.core_engine.set_weight_copy_queue(self.weight_copy_task)
			self.core_engine.start_h2d_worker()
		else:
			# Prepare GPU paged KV cache for local sequences
			self._prepare_gpu_paged_kv_cache(local_decode_indices)
			self.model, self.weight_copy_task = self.parallel_manager.pure_gpu_decoding(max_num_seq, comm)
			
			self.set_phase("decode")
			self.core_engine.stop_h2d_worker()
			self.core_engine.clear_kv_copy_queue()
			self.core_engine.clear_weight_copy_queue()
			self.core_engine.reset_decoding_buffer()
		
		logging.info(f"Rank {self.rank}: _config_decoding_for_batch completed")

	def _release_host_kv_pages_for_batch(self, uuids: List[str]) -> None:
		"""Release host KV pages for completed sequences owned by this rank."""
		if not uuids:
			return
		
		worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		if worker_view is None:
			logging.warning("Host paged KV worker view is unavailable")
			return
		
		# Only release for THIS RANK's sequences
		my_uuids = [uuid for uuid in uuids if uuid in self._uuid_to_local_map]
		
		if my_uuids:
			global_sequence_ids = [
				self.global_batch.get_sequence(uuid).global_idx
				for uuid in my_uuids
			]
			
			logging.info(f"Rank {self.rank}: Releasing host KV pages for global_idx: {global_sequence_ids}")
			
			worker_view.release_sequence_pages(global_sequence_ids)
			worker_view.unregister_sequences(global_sequence_ids)
			
			# Also release GPU KV pages
			local_indices = self._get_local_indices_for_uuids(my_uuids)
			if local_indices:
				self._release_gpu_kv_pages(local_indices)

	def _local_indices_to_global_seq_ids(self, local_indices: List[int]) -> List[int]:
		"""Convert local indices to global sequence IDs (global_idx from SequenceEntry)."""
		global_seq_ids = []
		for local_idx in local_indices:
			uuid = self._local_to_uuid_map.get(local_idx)
			if uuid:
				seq = self.global_batch.get_sequence(uuid)
				global_seq_ids.append(seq.global_idx)
		return global_seq_ids