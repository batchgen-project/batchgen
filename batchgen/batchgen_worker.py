import concurrent.futures
import copy
import functools
import psutil
import logging
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

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
from batchgen.distributed.utils import StatelessProcessGroup
from batchgen.distributed.device_communicators.pynccl import PyNcclCommunicator


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
	nvshmem_init = None


from .scheduler.scheduler import Scheduler


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
		return getattr(self, key, default)
	
	def to_dict(self) -> Dict:
		return self.__dict__.copy()
	
	def update(self, **kwargs):
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
	Inference Runtime with Host-KV-First scheduling and Continuous Batching.
	"""
	PAGE_SIZE = 64  # Alignment for page boundary checks

	def __init__(self, args: BatchGenWorkerArgs):
		logging.info(f"Rank {args.global_rank}: Initializing BatchGenWorker.")
		
		# 1. Store Arguments & Rank Information
		self.args = args
		self.local_rank = args.local_rank
		self.global_rank = args.global_rank
		self.rank = args.global_rank # Alias for compatibility
		self.world_size = args.world_size
		self.gpu_arch = args.gpu_arch
		self.kv_dtype = args.kv_dtype
		self.device = args.device
		self.torch_device = torch.device(f"cuda:{args.device}")

		# 2. Set Device immediately
		torch.cuda.set_device(self.local_rank)

		# 3. Path & Model Configurations
		self.model_name = args.model_name
		self.huggingface_ckpt_name = args.model_name
		self.hf_cache_dir = args.hf_cache_dir
		self.cache_dir = args.cache_dir
		self.pt_ckpt_dir = args.pt_ckpt_dir
		self.skeleton_state_dict = args.skeleton_state_dict
		
		# 4. Initialize Shared Memory for Weights (Crucial for multiprocess)
		self.shm_name = args.shm_name
		self.tensor_meta_shm_name = args.tensor_meta_shm_name
		self.weight_byte_size = args.weight_byte_size
		self.enable_hugetlbfs = args.enable_hugetlbfs
		
		logging.info(f"Rank {self.rank}: Initializing shared memory segments.")
		logging.info(
			f"Rank {self.rank}: shm_name: {self.shm_name}, "
			f"tensor_meta_shm_name: {self.tensor_meta_shm_name}, "
			f"weight_byte_size: {self.weight_byte_size}, "
			f"enable_hugetlbfs: {self.enable_hugetlbfs}"
		)

		self.weights_storage = core_engine.Weights_Storage(self.local_rank)
		self.weights_storage.Init(
			self.shm_name, 
			self.weight_byte_size, 
			self.tensor_meta_shm_name,
			self.enable_hugetlbfs
		)
		logging.info(f"Rank {self.rank}: Shared memory segments initialized.")

		# 5. Initialize Host KV Cache Manager View
		# This allows the worker to map to the host memory allocated by the main process
		self.host_kv_cache_size = args.host_kv_cache_size
		self.global_host_kv_cache_size_gb = args.global_host_kv_cache_size_gb
		
		worker_kv_config = build_host_kv_config(
			model_name=args.model_name,
			host_kv_cache_size=args.global_host_kv_cache_size_gb * (1024**3),
		)
		
		self.host_paged_kv_worker_view = core_engine.MLAHostPagedKVWorkerView(worker_kv_config)
		logging.info(f"Rank {self.rank}: Initializing core engine Host KV view.")
		# create_region=False because the main process created it; we just attach
		self.host_paged_kv_worker_view.initialize(device_index=self.local_rank, create_region=False)
		logging.info(f"Rank {self.rank}: Host KV manager view initialized.")

		# 6. Initialize Placeholders for Core Components
		# These are populated later in Init() / _initialize_core_components
		self.gpu_paged_kv_cache_manager = None
		self.model = None
		self.model_config = None
		self.hf_model_config = None
		self.engine_config = None
		self.core_engine = None
		self.tokenizer = None
		self.initializer = None
		self.parallel_manager = None
		
		# 7. Batch State Placeholders
		self.global_batch: Optional[SequenceBatch] = None
		self.query_book: Optional[Dict] = None
		self.model_batch_book: Dict = {}
		self._local_to_uuid_map: Dict[int, str] = {}
		self._uuid_to_local_map: Dict[str, int] = {}
		
		# 8. Runtime State
		self.eos_token_id: Optional[int] = None
		self.max_input_length = 0
		self.max_decoding_length = 0
		self.num_global_queries = 0
		self.num_local_queries = 0
		
		# 9. Initialization Flags
		self._core_initialized = False
		self._batch_completed = False
		self._nvshmem_initialized_this_run = False
		
		# 10. Distributed Communication Info
		self.dist_init_addr = args.dist_init_addr
		self.comm = None # Initialized lazily or in Init()

		COMM_MASTER_ADDR = self.dist_init_addr.split(':')[0]
		os.environ['COMM_MASTER_ADDR'] = COMM_MASTER_ADDR

		logging.info(f"Rank {self.rank}: BatchGenWorker __init__ completed.")

	def Init(self, max_input_length, max_decoding_length, num_queries):
		"""
		Initialize/reconfigure for a new batch.
		- First call: performs full initialization of core_engine, parallel_manager, etc.
		- Subsequent calls: only updates batch parameters and resets state.
		"""
		# Check if we need to reset state from previous batch
		if self._core_initialized and self.global_batch is not None:
			self._reset_for_new_batch()
		
		# Update batch-specific parameters
		self.max_input_length = max_input_length
		self.max_decoding_length = max_decoding_length
		
		logging.info(f"Initializing batchgen with global rank {self.args.global_rank} and world size {self.args.world_size} with PID: {os.getpid()}")
		
		# One-time initialization (only on first call)
		if not self._core_initialized:
			self._initialize_core_components(num_queries)
			self._core_initialized = True
		else:
			# Just update the num_queries and batch-related config
			self._update_batch_config(num_queries)
		
		logging.info(f"Engine on device {self.device} initialized/reconfigured.")

	def _initialize_core_components(self, num_queries: int) -> None:
		"""
		One-time initialization of heavy components.
		Called only on the first Init() call.
		"""
		logging.info(f"Rank {self.rank}: Performing one-time core initialization")
		
		config_torch_module_initializer()
		
		self.model_config = AutoConfig.from_pretrained(
			self.cache_dir,
			trust_remote_code=True,
			local_files_only=True,
		)
		self.tokenizer = AutoTokenizer.from_pretrained(
			self.cache_dir,
			trust_remote_code=True,
			local_files_only=True,
		)
		self.tokenizer.padding_side = "right"
		
		# Set EOS token ID from tokenizer
		self.eos_token_id = self.tokenizer.eos_token_id
		logging.info(f"Rank {self.rank}: EOS token ID set to {self.eos_token_id}")

		logging.info(f"Rank {self.rank}: Start initializing engine config.")
		config_scheduler = Scheduler(self.max_input_length, self.max_decoding_length, self.args.world_size)
		self.engine_config = config_scheduler.generate_config()
		self.engine_config.Basic_Config.device = self.args.device
		self.engine_config.Basic_Config.device_torch = torch.device(
			f"cuda:{self.args.device}"
		)
		self.engine_config.Basic_Config.max_decoding_length = self.max_decoding_length
		self.engine_config.Basic_Config.padding_length = self.max_input_length
		self.engine_config.Basic_Config.rank = self.global_rank
		self.engine_config.Basic_Config.world_size = self.world_size

		if not self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens:
			logging.warning(f"kv_buffer_num_tokens is set to {self.engine_config.GPU_Buffer_Config.kv_buffer_num_tokens}")

		self.device = self.args.device
		self.torch_device = torch.device(f"cuda:{self.args.device}")
		self.host_kv_cache_size = self.args.host_kv_cache_size
		self.global_host_kv_cache_size_gb = self.args.global_host_kv_cache_size_gb

		self.attn_mode = None
		self.query_book = None
		self.model_batch_book = {}
		self.token_k_cache_byte_size = 2048
		self.num_k_storage_tokens = math.floor(50 * (1024**3) / 32 / 2048)

		input_arguments = {
			"huggingface_ckpt_name": self.huggingface_ckpt_name,
			"hf_cache_dir": self.hf_cache_dir,
			"cache_dir": self.cache_dir,
			"pt_ckpt_dir": self.pt_ckpt_dir,
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
			self.initializer.Init(self.weights_storage)
		)

		self.core_engine.host_paged_kv_worker_view = self.host_paged_kv_worker_view
		self.engine_config.Basic_Config.num_queries = num_queries
		
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
		
		logging.info(f"Rank {self.rank}: One-time core initialization completed")

	def _update_batch_config(self, num_queries: int) -> None:
		"""
		Update configuration for a new batch without reinitializing heavy components.
		Called on subsequent Init() calls after the first.
		"""
		logging.info(f"Rank {self.rank}: Updating batch config for new batch")
		
		# Update engine config with new batch parameters
		self.engine_config.Basic_Config.max_decoding_length = self.max_decoding_length
		self.engine_config.Basic_Config.padding_length = self.max_input_length
		self.engine_config.Basic_Config.num_queries = num_queries
		
		# Update input_arguments for any components that might reference them
		if hasattr(self, 'input_arguments'):
			self.input_arguments.padding_length = self.max_input_length
			self.input_arguments.max_decoding_length = self.max_decoding_length
			self.input_arguments.num_queries = num_queries
		
		# Reset per-batch state
		self.query_book = None
		self.model_batch_book = {}
		
		logging.info(f"Rank {self.rank}: Batch config updated (max_input={self.max_input_length}, max_decode={self.max_decoding_length}, num_queries={num_queries})")

	# ============ KV Cache Helper Methods ============

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
			raise KeyError(f"No attention_mask available for sequence {sequence_id}")
		mask_row = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
		input_tokens = int(mask_row[: self.max_input_length].sum().item())
		total_tokens = input_tokens + self.max_decoding_length
		query_entry.kv_token_budget = total_tokens
		return total_tokens

	def _compute_host_kv_sequence_tokens(self, sequence_ids: List[int]) -> List[int]:
		"""Reuse cached token budgets so host/GPU allocations stay consistent."""
		return [self._get_sequence_token_budget(sequence_id) for sequence_id in sequence_ids]

	def _bind_gpu_paged_kv_manager(self, manager: GPUPagedKVCacheManager) -> None:
		"""Bind GPU KV manager to both worker and core_engine."""
		self.gpu_paged_kv_cache_manager = manager
		if hasattr(self.core_engine, "gpu_paged_kv_manager"):
			self.core_engine.gpu_paged_kv_manager = manager

	def _ensure_gpu_paged_kv_manager(self, sequence_tokens: Sequence[int]) -> GPUPagedKVCacheManager:
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
			self.rank, manager.device, gpu_config.num_pages,
		)
		return manager

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
		
		# allocate_pages_for_sequences implicitly registers the sequences
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
			self.rank, len(global_sequence_ids), load_duration,
		)

	def _release_gpu_kv_pages(self, local_sequence_ids: List[int]) -> None:	
		"""Return GPU KV pages associated with the provided local sequence ids."""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None or not local_sequence_ids:
			return
		
		global_sequence_ids = self._local_indices_to_global_seq_ids(local_sequence_ids)
		
		if not global_sequence_ids:
			return
		
		try:
			manager.free_pages_for_sequences(global_sequence_ids)
			logging.info(
				f"Rank {self.rank} Released GPU KV pages for global_idx: {global_sequence_ids}"
			)
		except KeyError as exc:
			logging.warning(
				"Rank %s failed to release GPU KV pages for %s: %s",
				self.rank, global_sequence_ids, exc,
			)

	def _destroy_gpu_paged_kv_cache(self, *, empty_cuda_cache: bool = False) -> None:
		"""Destroy the GPU paged KV cache manager if it is present."""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			return
		manager.destroy(empty_cuda_cache=empty_cuda_cache)

	def _get_host_kv_free_pages(self) -> int:
		"""Get current free pages from host KV cache."""
		stats = self.host_paged_kv_worker_view.get_stats()
		return stats.num_free_pages

	def _get_gpu_kv_free_pages(self) -> int:
		"""Get current free pages from GPU KV cache."""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			return 0
		return manager.get_stats().num_free_pages

	# ============ Main Entry Point ============

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
				prompt_length=0,
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

		# Step 6: Run generation with KV-driven scheduling
		return self.generate()

	# ============ UUID/Index Conversion Helpers ============

	def _local_to_uuid(self, local_idx: int) -> str:
		return self._local_to_uuid_map.get(local_idx, "")

	def _uuid_to_local(self, uuid: str) -> int:
		return self._uuid_to_local_map.get(uuid, -1)

	def _local_indices_to_global_seq_ids(self, local_indices: List[int]) -> List[int]:
		"""Convert local indices to global sequence IDs (global_idx from SequenceEntry)."""
		global_seq_ids = []
		for local_idx in local_indices:
			uuid = self._local_to_uuid_map.get(local_idx)
			if uuid:
				seq = self.global_batch.get_sequence(uuid)
				global_seq_ids.append(seq.global_idx)
		return global_seq_ids

	def _get_my_sequences_by_status(self, status: SequenceStatus) -> List[str]:
		"""Get UUIDs of sequences assigned to this rank with given status."""
		return self.global_batch.get_sequences_for_rank_with_status(self.rank, status)

	def _get_local_indices_for_uuids(self, uuids: List[str]) -> List[int]:
		"""Convert global UUIDs to local indices for sequences assigned to this rank."""
		local_indices = []
		for uuid in uuids:
			if uuid in self._uuid_to_local_map:
				local_indices.append(self._uuid_to_local_map[uuid])
		return local_indices

	# def _update_batch_status(self, uuids: List[str], new_status: SequenceStatus) -> None:
	# 	"""Update status for all sequences in a batch."""
	# 	for uuid in uuids:
	# 		self.global_batch.update_status(uuid, new_status)
	def _update_batch_status(self, uuids: List[str], new_status: SequenceStatus):
		"""Update status for sequences, skipping if already in target status."""
		if isinstance(uuids, str):
			uuids = [uuids]
		for uuid in uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				logging.warning(f"Rank {self.rank}: Sequence {uuid} not found in global_batch")
				continue
			if seq.status == new_status:
				continue  # Skip redundant transition
			if seq.status == SequenceStatus.COMPLETED:
				continue  # Don't change completed sequences
			self.global_batch.update_status(uuid, new_status)

	# ============ Tokenization and Assignment ============

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

			actual_prompt_len = int(
				tokenized["attention_mask"][0, :self.max_input_length].sum().item()
			)
			seq.prompt_length = actual_prompt_len
			seq.current_context_length = actual_prompt_len
			seq.kv_token_budget = actual_prompt_len + self.max_decoding_length

		logging.info(f"Rank {self.rank}: Tokenized {len(self.global_batch)} sequences")

	def _assign_sequences_to_ranks(self) -> None:
		"""
		Assign sequences to ranks using round-robin distribution.
		All ranks execute this identically to maintain consistent assignment.
		"""
		if self.global_batch is None:
			raise RuntimeError("Global batch not initialized")

		for seq in self.global_batch:
			assigned_rank = seq.global_idx % self.world_size
			self.global_batch.assign_rank(seq.uuid, assigned_rank)

		my_seqs = self.global_batch.get_sequences_for_rank(self.rank)
		logging.info(f"Rank {self.rank}: Assigned {len(my_seqs)} sequences: {my_seqs}")

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
 
		logging.info(f"Rank {self.rank}: Built local query_book with {len(self.query_book)} entries")

	# ============ KV-Driven Batch Preparation ============

	def _get_node_for_rank(self, rank: int) -> int:
		"""Get node ID for a rank. Assumes uniform GPUs per node."""
		gpus_per_node = torch.cuda.device_count()
		return rank // gpus_per_node

	def _get_num_nodes(self) -> int:
		"""Get total number of nodes."""
		gpus_per_node = torch.cuda.device_count()
		return self.world_size // gpus_per_node

	def _prepare_prefill_batch(self) -> List[str]:
		"""
		Select sequences for prefill based on HOST KV cache capacity.
		
		Key constraint: Host KV cache is PER NODE.
		- Each node has its own host KV capacity
		- Sequences assigned to ranks on node N use node N's host KV
		- Must check per-node capacity, not global
		"""
		queueing_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.QUEUEING)
		queueing_uuids.sort(key=lambda uuid: self.global_batch.get_sequence(uuid).global_idx)
		
		if not queueing_uuids:
			return []
		
		gpus_per_node = torch.cuda.device_count()
		num_nodes = self._get_num_nodes()
		my_node = self._get_node_for_rank(self.rank)
		
		# Step 1: Get this node's host KV free pages
		local_host_free = self._get_host_kv_free_pages()
		
		# Step 2: Gather host KV free pages from first rank on each node
		# Only rank 0, 8, 16, ... (first on each node) reports actual value
		if self.rank % gpus_per_node == 0:
			report_free = local_host_free
		else:
			report_free = 0  # Non-first ranks report 0
		
		free_tensor = torch.tensor([report_free], dtype=torch.int64, device=self.torch_device)
		gathered = [torch.zeros_like(free_tensor) for _ in range(self.world_size)]
		dist.all_gather(gathered, free_tensor)
		
		# Extract per-node host KV free pages
		per_node_host_free = []
		for node in range(num_nodes):
			first_rank = node * gpus_per_node
			per_node_host_free.append(int(gathered[first_rank].item()))
		
		if self.rank == 0:
			logging.info(f"Per-node host KV free pages: {per_node_host_free}")
		
		# Step 3: Select sequences considering per-node host KV capacity
		node_pages_used = [0] * num_nodes
		prefill_batch = []
		
		for uuid in queueing_uuids:
			seq = self.global_batch.get_sequence(uuid)
			assigned_rank = seq.assigned_rank
			seq_node = self._get_node_for_rank(assigned_rank)
			
			req_pages = seq.get_pages_required()
			
			if node_pages_used[seq_node] + req_pages <= per_node_host_free[seq_node]:
				prefill_batch.append(uuid)
				node_pages_used[seq_node] += req_pages
		
		logging.info(
			f"Rank {self.rank}: Prefill batch: {len(prefill_batch)} sequences, "
			f"per-node pages used: {node_pages_used}"
		)
		
		return prefill_batch


	def _prepare_decode_batch(self) -> List[str]:
		"""
		Select sequences for decode phase from PREFILLED sequences.
		
		For INITIAL decode batch (before GPU KV manager exists):
		- Use per-rank MoE limit as the constraint
		- GPU KV manager will be sized appropriately in _config_decoding_for_batch
		
		For CONTINUOUS batching (GPU KV manager exists):
		- Use get_stats().num_free_pages
		"""
		prefilled_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		prefilled_uuids.sort(key=lambda uuid: self.global_batch.get_sequence(uuid).global_idx)
		
		if not prefilled_uuids:
			return []
		
		# Per-rank sequence limit from MoE config
		max_seqs_per_rank = self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size
		
		# Check if GPU KV manager exists (continuous batching case)
		gpu_free_pages = 0
		if self.gpu_paged_kv_cache_manager is not None and self.gpu_paged_kv_cache_manager.is_initialized:
			gpu_free_pages = self.gpu_paged_kv_cache_manager.get_stats().num_free_pages
			logging.info(f"Rank {self.rank}: GPU KV has {gpu_free_pages} free pages")
		else:
			# Initial batch - GPU manager doesn't exist yet
			# Use a large number; actual limit will be per-rank MoE limit
			gpu_free_pages = float('inf')
		
		# Count sequences per rank and enforce limits
		rank_counts = [0] * self.world_size
		decode_batch = []
		total_pages_needed = 0
		
		for uuid in prefilled_uuids:
			seq = self.global_batch.get_sequence(uuid)
			assigned_rank = seq.assigned_rank
			
			# Check per-rank limit
			if rank_counts[assigned_rank] >= max_seqs_per_rank:
				continue
			
			# Check GPU KV page capacity (for continuous batching)
			seq_pages = seq.get_pages_required()
			if total_pages_needed + seq_pages > gpu_free_pages:
				logging.info(
					f"Rank {self.rank}: GPU KV capacity reached at {len(decode_batch)} sequences "
					f"({total_pages_needed}/{gpu_free_pages} pages)"
				)
				break
			
			decode_batch.append(uuid)
			rank_counts[assigned_rank] += 1
			total_pages_needed += seq_pages
		
		remaining = len(prefilled_uuids) - len(decode_batch)
		logging.info(
			f"Rank {self.rank}: Prepared decode batch: {len(decode_batch)} sequences, "
			f"{total_pages_needed} pages. {remaining} remain PREFILLED."
		)
		
		return decode_batch

	def _check_and_handle_completions(
		self, 
		decode_uuids: List[str], 
		local_decode_indices: List[int],
		new_token_idx: int
	) -> Tuple[List[str], List[int], List[str]]:
		"""
		Check for completed sequences at page boundaries.
		Returns:
			- updated decode_uuids (active sequences)
			- updated local_decode_indices (active local indices)
			- completed_uuids (sequences that completed)
		"""
		completed_uuids = []
		active_uuids = []
		active_local_indices = []
		
		for uuid in decode_uuids:
			seq = self.global_batch.get_sequence(uuid)
			
			# Check if sequence should be completed
			if seq.check_completion(self.eos_token_id):
				completed_uuids.append(uuid)
				logging.info(
					f"Rank {self.rank}: Sequence {uuid} completed at token {new_token_idx} "
					f"(decoded_length={seq.decoded_length}, eos_reached={seq.eos_reached})"
				)
			else:
				active_uuids.append(uuid)
				if uuid in self._uuid_to_local_map:
					active_local_indices.append(self._uuid_to_local_map[uuid])
		
		return active_uuids, active_local_indices, completed_uuids

	def _try_load_new_sequences(
		self, 
		current_decode_uuids: List[str],
		current_local_indices: List[int]
	) -> Tuple[List[str], List[int]]:
		"""
		Load PREFILLED sequences from Host KV to GPU KV if space available.
		Maintains deterministic ordering across all ranks.
		"""
		gpu_free_pages = self._get_gpu_kv_free_pages()
		candidates = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		
		# Sort for deterministic ordering across all ranks
		candidates.sort(key=lambda u: self.global_batch.get_sequence(u).global_idx)
		
		new_uuids = []
		pages_needed = 0
		
		for uuid in candidates:
			seq = self.global_batch.get_sequence(uuid)
			req = seq.get_pages_required()
			if pages_needed + req <= gpu_free_pages:
				new_uuids.append(uuid)
				pages_needed += req
			else:
				break
		
		if not new_uuids:
			return current_decode_uuids, current_local_indices

		# Get local indices for sequences belonging to THIS rank
		new_local_indices = self._get_local_indices_for_uuids(new_uuids)
		
		if new_local_indices:
			# Allocate and load (without final rebuild)
			self._allocate_and_load_gpu_kv_for_new_sequences(new_local_indices)

		# Update status AFTER load completes
		self._update_batch_status(new_uuids, SequenceStatus.IN_DECODE)
		
		# Build updated lists
		updated_uuids = current_decode_uuids + new_uuids
		updated_batch = current_local_indices + new_local_indices
		
		# Final page table rebuild with ALL active sequences
		if self.gpu_paged_kv_cache_manager is not None and updated_batch:
			all_global_ids = self._local_indices_to_global_seq_ids(updated_batch)
			self.gpu_paged_kv_cache_manager.rebuild_page_table(all_global_ids)
		
		logging.info(
			f"Rank {self.rank}: Loaded {len(new_uuids)} new sequences, "
			f"total decode batch now {len(updated_uuids)}"
		)
		
		return updated_uuids, updated_batch

	# def _allocate_and_load_gpu_kv_for_new_sequences(self, local_sequence_ids: List[int]) -> None:
	# 	"""
	# 	Allocate GPU KV pages and load host KV for newly added sequences during continuous batching.
	# 	This is different from _prepare_gpu_paged_kv_cache which may recreate the manager.
	# 	"""
	# 	if not local_sequence_ids:
	# 		return
		
	# 	manager = self.gpu_paged_kv_cache_manager
	# 	if manager is None:
	# 		logging.warning("GPU KV manager not initialized, cannot load new sequences")
	# 		return
		
	# 	# Convert local indices to global sequence IDs
	# 	global_sequence_ids = self._local_indices_to_global_seq_ids(local_sequence_ids)
	# 	sequence_tokens = self._compute_host_kv_sequence_tokens(local_sequence_ids)
		
	# 	logging.info(
	# 		f"Rank {self.rank}: Allocating GPU KV pages for new sequences: {global_sequence_ids}"
	# 	)
		
	# 	# Allocate pages for the new sequences
	# 	manager.allocate_pages_for_sequences(global_sequence_ids, sequence_tokens)
		
	# 	# Rebuild page table to include new sequences
	# 	# Get all currently active sequences (existing + new)
	# 	all_active_global_ids = []
	# 	for uuid in self.global_batch.get_sequences_by_status(SequenceStatus.IN_DECODE):
	# 		if uuid in self._uuid_to_local_map:
	# 			seq = self.global_batch.get_sequence(uuid)
	# 			all_active_global_ids.append(seq.global_idx)
		
	# 	# Sort for deterministic ordering
	# 	all_active_global_ids.sort()
		
	# 	if all_active_global_ids:
	# 		manager.rebuild_page_table(all_active_global_ids)
		
	# 	# Load host KV to GPU for the new sequences
	# 	self._load_host_kv_to_gpu(manager, global_sequence_ids)
	def _allocate_and_load_gpu_kv_for_new_sequences(self, local_sequence_ids: List[int]) -> None:
		"""
		Allocates GPU pages and triggers blocking load from Host.
		"""
		if not local_sequence_ids: return
		
		manager = self.gpu_paged_kv_cache_manager
		global_ids = self._local_indices_to_global_seq_ids(local_sequence_ids)
		tokens = self._compute_host_kv_sequence_tokens(local_sequence_ids)

		# 1. Allocate GPU Pages
		manager.allocate_pages_for_sequences(global_ids, tokens)

		# 2. Rebuild Page Table (Critical: Ensure kernel sees new pointers)
		# We rebuild specifically for the sequences we are about to load
		manager.rebuild_page_table(global_ids)

		# 3. Load Host -> GPU (BLOCKING)
		# "The load api is non-blocked, but we can use .wait() to let it be blocking for now."
		self._load_host_kv_to_gpu(manager, global_ids) 

		# 4. Rebuild Page Table for ALL active sequences (for next Attention forward)
		active_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.IN_DECODE)
		all_active_ids = [self.global_batch.get_sequence(u).global_idx for u in active_uuids if u in self._uuid_to_local_map]
		# Union with new ids
		final_ids = sorted(list(set(all_active_ids + global_ids)))
		manager.rebuild_page_table(final_ids)

	# ============ Main Generation Loop ============

	def generate(self):
		"""
		Main Loop: Config Prefill -> Prefill -> Config Decode -> Decode (Continuous).
		"""
		# Initialize timing trackers
		generation_start_time = time.perf_counter()
		prefill_time = 0.0
		decoding_time = 0.0
		config_prefill_time = 0.0
		config_decode_time = 0.0
		
		# Ensure communicator is ready
		if os.getenv("BATCHGEN_ENABLE_ALL_TO_ALL", "0") == "0":
			# Verify rank consistency
			if dist.is_initialized():
				assert self.rank == dist.get_rank(), \
					f"Rank mismatch: self.rank={self.rank}, dist.get_rank()={dist.get_rank()}"
			
			device = torch.device("cuda", self.local_rank)
			comm_master_addr = os.getenv("COMM_MASTER_ADDR")
			self.comm = None
			
			if comm_master_addr is None:
				logging.warning(f"Rank {self.rank}: COMM_MASTER_ADDR not set, skipping PyNccl init")
			elif StatelessProcessGroup is not None and PyNcclCommunicator is not None:
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

		iteration = 0
		
		# Continues until ALL sequences in the global batch are COMPLETED
		while not self.global_batch.all_completed():
			iteration += 1
			logging.info(f"--- Iteration {iteration} ---")
			
			# =================================================================
			# 1. PREFILL PHASE: Fill Host KV Cache
			# =================================================================
			if self.global_batch.has_queueing():
				dist.barrier()
				
				prefill_uuids = self._prepare_prefill_batch()
				
				if prefill_uuids:
					logging.info(f"Rank {self.rank}: Entering PREFILL for {len(prefill_uuids)} sequences")
					self._update_batch_status(prefill_uuids, SequenceStatus.IN_PREFILL)
					local_prefill_indices = self._get_local_indices_for_uuids(prefill_uuids)

					# A. Config Prefill
					config_start = time.perf_counter()
					self._config_prefill_for_batch(prefill_uuids)
					config_prefill_time += time.perf_counter() - config_start

					# B. Execute Prefill
					if local_prefill_indices:
						prefill_start = time.perf_counter()
						with torch.inference_mode():
							self.prefill(local_prefill_indices)
						prefill_time += time.perf_counter() - prefill_start
					
					# Cleanup & Status Update
					self._unregister_fp8_weights()
					self._update_batch_status(prefill_uuids, SequenceStatus.PREFILLED)
					dist.barrier()

			# =================================================================
			# 2. DECODE PHASE: Continuous Batching (Host -> GPU Streaming)
			# =================================================================
			while self.global_batch.has_prefilled() or self.global_batch.has_in_decode():
				dist.barrier()
				
				# A. Prepare Initial Decode Batch (from PREFILLED)
				decode_uuids = self._prepare_decode_batch()
				
				# Include currently running sequences - PRESERVE ORDER
				current_decoding = self.global_batch.get_sequences_by_status(SequenceStatus.IN_DECODE)
				seen = set(decode_uuids)
				for uuid in current_decoding:
					if uuid not in seen:
						decode_uuids.append(uuid)
						seen.add(uuid)
				
				# Sort for deterministic cross-rank ordering
				decode_uuids.sort(key=lambda u: self.global_batch.get_sequence(u).global_idx)
				
				if not decode_uuids:
					break
				
				self._update_batch_status(decode_uuids, SequenceStatus.IN_DECODE)
				local_decode_indices = self._get_local_indices_for_uuids(decode_uuids)

				# B. Config Decode
				config_start = time.perf_counter()
				self._config_decoding_for_batch(decode_uuids, local_decode_indices, self.comm)
				config_decode_time += time.perf_counter() - config_start

				# C. Execute Continuous Decode
				decode_start = time.perf_counter()
				with torch.inference_mode():
					if local_decode_indices:
						new_tokens = self._rebuild_input_tokens(local_decode_indices)
					else:
						new_tokens = torch.empty((0, 1), dtype=torch.int64, device=self.torch_device)

					self.decoding_continuous(new_tokens, decode_uuids, local_decode_indices)
				decoding_time += time.perf_counter() - decode_start

				# D. Cleanup
				self._unregister_fp8_weights()
				self.deep_free_model_memory()
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
		
		# ============ Gather Results in Original Order ============
		res_with_idx = []
		for local_idx in range(self.num_local_queries):
			uuid = self._local_to_uuid_map[local_idx]
			global_idx = self.global_batch.get_sequence(uuid).global_idx
			decoded_tokens = self.query_book[local_idx].decoded_tokens
			res_with_idx.append((global_idx, decoded_tokens))
		
		all_results = [None] * self.world_size
		dist.all_gather_object(all_results, res_with_idx)
		
		all_results = [item for sublist in all_results for item in sublist]
		all_results.sort(key=lambda x: x[0])
		
		sorted_tokens = [item[1] for item in all_results]
		res_tensor = torch.cat(sorted_tokens, dim=0).cpu()
		
		dist.barrier()
		self._batch_completed = True
		
		if self.rank == 0:
			return [res_tensor]
		else:
			return []

	# ============ Phase Configuration ============

	def _config_prefill_for_batch(self, prefill_uuids: List[str]) -> None:
		"""Configure prefill phase for a batch of sequences."""
		start_time = time.perf_counter()
		logging.info(f"Rank {self.rank}: Starting _config_prefill_for_batch")
		
		self.model, self.weight_copy_task = self.parallel_manager.configure_prefill()
		self.set_phase("prefill")
		
		self.core_engine.stop_h2d_worker()
		self.core_engine.clear_weight_copy_queue()
		self.core_engine.reset_prefill_buffer()
		self.core_engine.set_weight_copy_queue(self.weight_copy_task)
		self.core_engine.start_h2d_worker()
		
		self._destroy_gpu_paged_kv_cache()
		
		# Only allocate host KV pages for THIS RANK's sequences
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
		
		my_uuids = [uuid for uuid in uuids if uuid in self._uuid_to_local_map]
		
		if my_uuids:
			global_sequence_ids = [
				self.global_batch.get_sequence(uuid).global_idx
				for uuid in my_uuids
			]
			
			logging.info(f"Rank {self.rank}: Releasing host KV pages for global_idx: {global_sequence_ids}")
			
			# Release GPU KV pages first
			local_indices = self._get_local_indices_for_uuids(my_uuids)
			if local_indices:
				self._release_gpu_kv_pages(local_indices)
			
			# Then release host KV pages
			worker_view.release_sequence_pages(global_sequence_ids)
			worker_view.unregister_sequences(global_sequence_ids)
			
			# Rebuild GPU page table with remaining active sequences
			manager = self.gpu_paged_kv_cache_manager
			if manager is not None and manager.is_initialized:
				remaining_in_decode = self.global_batch.get_sequences_by_status(SequenceStatus.IN_DECODE)
				remaining_global_ids = []
				for uuid in remaining_in_decode:
					if uuid in self._uuid_to_local_map and uuid not in my_uuids:
						seq = self.global_batch.get_sequence(uuid)
						remaining_global_ids.append(seq.global_idx)
				
				if remaining_global_ids:
					remaining_global_ids.sort()
					manager.rebuild_page_table(remaining_global_ids)

	# ============ Prefill and Decode ============

	def prefill(self, batch: list[int]):
		"""
		Handle the prefill for a batch.
		batch: list of local indices
		"""
		if "deepseek" in self.model_config.model_type:
			self.model.model._use_flash_attention_2 = False

		input_ids = torch.cat(
			[
				self.query_book[query_idx].encoded["input_ids"][:, : self.max_input_length]
				for query_idx in batch
			],
			dim=0,
		)
		attention_masks = torch.cat(
			[
				self.query_book[query_idx].encoded["attention_mask"][:, : self.max_input_length]
				for query_idx in batch
			],
			dim=0,
		)

		num_prefill_micro_batches = math.ceil(
			len(batch) / self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size
		)
		prefill_micro_batch_input_ids = torch.split(
			input_ids,
			self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
		)
		prefill_micro_batch_attention_masks = torch.split(
			attention_masks,
			self.engine_config.Module_Batching_Config.MoE_prefill_micro_batch_size,
		)
		logging.info(f"Number of prefill micro batches: {num_prefill_micro_batches}")
		
		cur_batch_start = 0
		output_tokens = []
		
		for micro_batch_idx in tqdm(range(num_prefill_micro_batches), desc="Prefill Micro Batch"):
			with torch.inference_mode():
				Attn_Wrapper.attention_mask = prefill_micro_batch_attention_masks[micro_batch_idx]
				Attn_Wrapper.position_ids = create_position_ids_from_attention_mask(
					prefill_micro_batch_attention_masks[micro_batch_idx]
				)
				
				cur_batch_size = prefill_micro_batch_input_ids[micro_batch_idx].shape[0]
				cur_batch_local = batch[cur_batch_start : cur_batch_start + cur_batch_size]
				
				# Pass local indices - the C++ layer handles rank offset internally
				Attn_Wrapper.cur_batch = self._local_indices_to_global_seq_ids(cur_batch_local)
				
				cur_batch_start += cur_batch_size
				assert len(cur_batch_local) == cur_batch_size

				outputs = self.model(
					prefill_micro_batch_input_ids[micro_batch_idx].to(self.torch_device),
					attention_mask=prefill_micro_batch_attention_masks[micro_batch_idx].to(self.torch_device),
					use_cache=False,
				)
				new_tokens = torch.argmax(outputs.logits[:, -1, :], dim=-1).view(-1, 1)
				output_tokens.append(new_tokens)

		new_tokens = torch.cat(output_tokens, dim=0)
		self.update_new_token(new_tokens, batch, 0)
		
		# Update sequence state after prefill
		for i, local_idx in enumerate(batch):
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			seq.decoded_length = 1
			seq.current_context_length = seq.prompt_length + 1
			
			# Check for EOS in first token
			if new_tokens[i].item() == self.eos_token_id:
				seq.eos_reached = True
		
		return new_tokens

	def decoding_continuous(
		self, 
		new_tokens: torch.Tensor, 
		decode_uuids: List[str],
		batch: List[int],
		past_key_states: Optional[torch.Tensor] = None,
		past_value_states: Optional[torch.Tensor] = None,
		scale_dict: Optional[dict] = None,
	) -> Tuple[List[str], List[int]]:
		"""
		Continuous decoding with page-boundary synchronization.
		
		Design:
		- Forward pass happens every iteration (all ranks participate)
		- Completion check is LOCAL every iteration (no sync)
		- Eviction/Load/Sync happens every PAGE_SIZE (64) iterations
		"""
		if "deepseek" in self.model_config.model_type:
			self.model.model._use_flash_attention_2 = True
		
		RUNTIME_ATTN_MODE = self.engine_config.Basic_Config.attn_mode
		logging.info(
			f"Rank {self.rank}: Starting continuous decode - "
			f"{len(decode_uuids)} global, {len(batch)} local sequences"
		)

		if RUNTIME_ATTN_MODE != 3:
			self._decoding_legacy_modes(new_tokens, decode_uuids, batch, 1)
			return decode_uuids, batch

		# Setup GPU paged KV manager
		gpu_manager = self.gpu_paged_kv_cache_manager
		if gpu_manager is None:
			gpu_manager = getattr(self.core_engine, "gpu_paged_kv_manager", None)
		
		# Bind wrapper state
		Attn_Wrapper.gpu_paged_kv_manager = gpu_manager
		Attn_Wrapper.host_paged_kv_worker_view = getattr(
			self.core_engine, "host_paged_kv_worker_view", None
		)
		Attn_Wrapper.scale = scale_dict
		Attn_Wrapper.past_key_states = past_key_states
		Attn_Wrapper.past_value_states = past_value_states
		Attn_Wrapper.cur_batch = self._local_indices_to_global_seq_ids(batch) if batch else []
		
		iteration = 0
		last_page_boundary_check = 0
		
		while decode_uuids:
			iteration += 1
			
			if self.rank == 0 and iteration % 100 == 0:
				logging.info(f"Decode iteration {iteration}, {len(decode_uuids)} active")
			
			# =============================================================
			# PAGE BOUNDARY (every 64 iterations): Sync, Evict, Load
			# =============================================================
			if iteration - last_page_boundary_check >= self.PAGE_SIZE:
				last_page_boundary_check = iteration
				
				# All ranks sync here
				dist.barrier()
				
				# 1. Sync completion status across all ranks
				decode_uuids, completed_uuids = self._sync_completion_status_at_boundary(decode_uuids)
				
				# 2. Evict completed sequences (release KV pages)
				if completed_uuids:
					self._update_batch_status(completed_uuids, SequenceStatus.COMPLETED)
					my_completed = [u for u in completed_uuids if u in self._uuid_to_local_map]
					if my_completed:
						self._release_host_kv_pages_for_batch(my_completed)
					logging.info(
						f"Rank {self.rank}: Evicted {len(completed_uuids)} completed sequences"
					)
				
				# 3. Update local batch after eviction
				batch = self._get_local_indices_for_uuids(decode_uuids)
				
				# 4. Load new sequences if GPU pages available
				if decode_uuids or self.global_batch.has_prefilled():
					prev_global = len(decode_uuids)
					prev_local = len(batch)
					decode_uuids, batch = self._try_load_new_sequences_at_boundary(decode_uuids, batch)
					
					if len(decode_uuids) > prev_global:
						logging.info(
							f"Rank {self.rank}: Loaded {len(decode_uuids) - prev_global} new sequences "
							f"(global: {prev_global}->{len(decode_uuids)}, local: {prev_local}->{len(batch)})"
						)
				
				# 5. Rebuild GPU page table with current batch
				if gpu_manager is not None and gpu_manager.is_initialized:
					if batch:
						global_ids = self._local_indices_to_global_seq_ids(batch)
						global_ids.sort()
						gpu_manager.rebuild_page_table(global_ids)
						Attn_Wrapper.cur_batch = global_ids
					else:
						Attn_Wrapper.cur_batch = []
				
				# 6. Rebuild input tokens for current batch
				new_tokens = self._rebuild_input_tokens(batch)
				
				# All ranks sync after page boundary operations
				dist.barrier()
				
				if not decode_uuids:
					logging.info(f"Rank {self.rank}: All sequences completed")
					break
			
			# =============================================================
			# FORWARD PASS (all ranks must participate)
			# =============================================================
			with torch.inference_mode():
				if batch:
					# Build attention metadata
					attention_masks = []
					cache_seqlens = []
					
					for local_idx in batch:
						uuid = self._local_to_uuid_map[local_idx]
						seq = self.global_batch.get_sequence(uuid)
						ctx_len = seq.current_context_length
						
						mask = self.query_book[local_idx].encoded["attention_mask"][:, :ctx_len]
						attention_masks.append(mask)
						cache_seqlens.append(ctx_len)
					
					# Pad masks
					max_ctx = max(cache_seqlens)
					padded = []
					for mask in attention_masks:
						if mask.shape[1] < max_ctx:
							pad = torch.zeros((1, max_ctx - mask.shape[1]), dtype=mask.dtype, device=mask.device)
							mask = torch.cat([mask, pad], dim=1)
						padded.append(mask)
					
					attention_mask = torch.cat(padded, dim=0).to(self.torch_device)
					
					Attn_Wrapper.attention_mask = attention_mask
					Attn_Wrapper.position_ids = (attention_mask.sum(-1) - 1).unsqueeze(-1)
					Attn_Wrapper.cache_seqlens = torch.tensor(cache_seqlens, dtype=torch.int32, device=self.torch_device)
					Attn_Wrapper.max_seqlen = max_ctx
					
					# Ensure tokens match batch
					if new_tokens.shape[0] != len(batch):
						new_tokens = self._rebuild_input_tokens(batch)
				else:
					# Empty local batch
					Attn_Wrapper.attention_mask = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
					Attn_Wrapper.position_ids = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
					Attn_Wrapper.cache_seqlens = torch.zeros((0,), dtype=torch.int32, device=self.torch_device)
					Attn_Wrapper.max_seqlen = 0
					Attn_Wrapper.cur_batch = []
					new_tokens = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
				
				# Forward pass - ALL RANKS MUST CALL
				outputs = self.model(
					new_tokens,
					attention_mask=Attn_Wrapper.attention_mask,
					use_cache=False,
				)
				
				# Update local sequences only
				if batch:
					new_tokens = torch.argmax(outputs.logits, dim=-1).view(-1, 1)
					
					for i, local_idx in enumerate(batch):
						uuid = self._local_to_uuid_map[local_idx]
						seq = self.global_batch.get_sequence(uuid)
						
						# Skip if already complete
						if seq.eos_reached:
							continue
						
						decode_pos = seq.decoded_length
						
						# Bounds check
						if decode_pos >= self.max_decoding_length:
							seq.eos_reached = True
							continue
						
						# Store token
						self.query_book[local_idx].decoded_tokens[:, decode_pos] = new_tokens[i].cpu()
						
						# Update attention mask
						attn_mask = self.query_book[local_idx].encoded["attention_mask"][0]
						next_pos = seq.current_context_length
						if next_pos < attn_mask.shape[0]:
							attn_mask[next_pos] = 1
						
						# Increment counters
						seq.decoded_length += 1
						seq.current_context_length += 1
						
						# Check EOS (local only - synced at page boundary)
						if new_tokens[i].item() == self.eos_token_id:
							seq.eos_reached = True
							logging.info(f"Rank {self.rank}: Seq {uuid} hit EOS at {seq.decoded_length}")
						
						# Check max length
						if seq.decoded_length >= self.max_decoding_length:
							seq.eos_reached = True
				else:
					new_tokens = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
		
		# Final cleanup
		if decode_uuids:
			self._update_batch_status(decode_uuids, SequenceStatus.COMPLETED)
			my_remaining = [u for u in decode_uuids if u in self._uuid_to_local_map]
			if my_remaining:
				self._release_host_kv_pages_for_batch(my_remaining)
		
		# Clear wrapper
		Attn_Wrapper.scale = None
		Attn_Wrapper.past_key_states = None
		Attn_Wrapper.past_value_states = None
		Attn_Wrapper.gpu_paged_kv_manager = None
		Attn_Wrapper.host_paged_kv_worker_view = None
		Attn_Wrapper.cur_batch = None
		
		return decode_uuids, batch


	def _sync_completion_status_at_boundary(
		self, 
		decode_uuids: List[str]
	) -> Tuple[List[str], List[str]]:
		"""
		Efficient completion sync at page boundaries using all_reduce.
		
		Each sequence is owned by exactly one rank. Only the owner knows
		if it hit EOS. We use all_reduce(MAX) to broadcast completion status.
		
		Returns: (active_uuids, completed_uuids)
		"""
		if not decode_uuids:
			return [], []
		
		n = len(decode_uuids)
		
		# Build completion tensor - each rank marks its owned sequences
		completion = torch.zeros(n, dtype=torch.int32, device=self.torch_device)
		
		for i, uuid in enumerate(decode_uuids):
			# Only mark if this rank owns the sequence
			if uuid in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				if seq.eos_reached or seq.decoded_length >= self.max_decoding_length:
					completion[i] = 1
		
		# All-reduce MAX: completed on ANY rank means globally completed
		dist.all_reduce(completion, op=dist.ReduceOp.MAX)
		
		# Split based on result
		active = []
		completed = []
		
		for i, uuid in enumerate(decode_uuids):
			if completion[i].item() == 1:
				completed.append(uuid)
				# Ensure local state is consistent
				seq = self.global_batch.get_sequence(uuid)
				seq.eos_reached = True
			else:
				active.append(uuid)
		
		return active, completed

	def _try_load_new_sequences_at_boundary(
		self, 
		current_decode_uuids: List[str],
		current_batch: List[int]
	) -> Tuple[List[str], List[int]]:
		"""
		Load PREFILLED sequences to GPU at page boundaries.
		
		Architecture:
		- Host KV cache is PER NODE
		- GPU KV cache is PER RANK
		- A sequence prefilled by rank R has host KV on node (R // gpus_per_node)
		- Only ranks on THAT node can load this sequence to their GPU
		
		Sync strategy:
		1. All-gather free GPU pages from all ranks
		2. All ranks compute IDENTICAL loading decision
		3. Each rank only loads sequences assigned to it
		4. All ranks update decode_uuids identically
		"""
		gpus_per_node = torch.cuda.device_count()
		my_node = self._get_node_for_rank(self.rank)
		
		# Step 1: All-gather free GPU pages from ALL ranks
		manager = self.gpu_paged_kv_cache_manager
		local_free = manager.get_stats().num_free_pages if manager and manager.is_initialized else 0
		
		free_tensor = torch.tensor([local_free], dtype=torch.int64, device=self.torch_device)
		gathered = [torch.zeros_like(free_tensor) for _ in range(self.world_size)]
		dist.all_gather(gathered, free_tensor)
		per_rank_free = [int(t.item()) for t in gathered]
		
		if self.rank == 0:
			logging.info(f"Per-rank GPU free pages: {per_rank_free}")
		
		# Step 2: Get PREFILLED candidates (all ranks see identical list)
		candidates = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		candidates.sort(key=lambda u: self.global_batch.get_sequence(u).global_idx)
		
		if not candidates:
			return current_decode_uuids, current_batch
		
		# Step 3: Current per-rank state
		max_per_rank = self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size
		rank_seq_counts = [0] * self.world_size
		for uuid in current_decode_uuids:
			seq = self.global_batch.get_sequence(uuid)
			rank_seq_counts[seq.assigned_rank] += 1
		
		rank_pages_used = [0] * self.world_size
		
		# Step 4: Select sequences (IDENTICAL computation on all ranks)
		new_uuids = []
		
		for uuid in candidates:
			seq = self.global_batch.get_sequence(uuid)
			assigned_rank = seq.assigned_rank
			seq_node = self._get_node_for_rank(assigned_rank)
			
			# Host KV constraint: sequence's host KV is on seq_node
			# Only assigned_rank (which is on seq_node) will load it
			# This is implicitly enforced by using assigned_rank
			
			# Check per-rank sequence limit
			if rank_seq_counts[assigned_rank] >= max_per_rank:
				continue
			
			# Check GPU page capacity on assigned rank
			req_pages = seq.get_pages_required()
			if rank_pages_used[assigned_rank] + req_pages > per_rank_free[assigned_rank]:
				continue
			
			# Accept this sequence
			new_uuids.append(uuid)
			rank_pages_used[assigned_rank] += req_pages
			rank_seq_counts[assigned_rank] += 1
		
		if not new_uuids:
			return current_decode_uuids, current_batch
		
		# Step 5: Load GPU KV for THIS RANK's new sequences only
		my_new_uuids = [u for u in new_uuids 
					if self.global_batch.get_sequence(u).assigned_rank == self.rank]
		new_local_indices = self._get_local_indices_for_uuids(my_new_uuids)
		
		if new_local_indices:
			self._allocate_and_load_gpu_kv_for_new_sequences(new_local_indices)
			logging.info(
				f"Rank {self.rank} (node {my_node}): Loaded {len(my_new_uuids)} sequences, "
				f"{rank_pages_used[self.rank]}/{per_rank_free[self.rank]} GPU pages"
			)
		
		# Step 6: Update status globally (all ranks do this identically)
		self._update_batch_status(new_uuids, SequenceStatus.IN_DECODE)
		
		# Step 7: Return updated lists
		updated_decode_uuids = current_decode_uuids + new_uuids
		updated_batch = current_batch + new_local_indices
		
		logging.info(
			f"Rank {self.rank}: Loaded {len(new_uuids)} sequences globally "
			f"(decode: {len(current_decode_uuids)}->{len(updated_decode_uuids)}, "
			f"local: {len(current_batch)}->{len(updated_batch)})"
		)
		
		return updated_decode_uuids, updated_batch


	def _rebuild_input_tokens(self, batch: List[int]) -> torch.Tensor:
		"""Build input tokens from each sequence's last decoded position."""
		if not batch:
			return torch.empty((0, 1), dtype=torch.int64, device=self.torch_device)
		
		tokens = []
		for local_idx in batch:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			pos = max(0, seq.decoded_length - 1)
			token = self.query_book[local_idx].decoded_tokens[:, pos:pos+1]
			tokens.append(token)
		
		return torch.cat(tokens, dim=0).to(self.torch_device)


	def _sync_completion_status(
		self, 
		decode_uuids: List[str]
	) -> Tuple[List[str], List[str]]:
		"""
		Synchronize completion status across all ranks using all-reduce.
		Returns (active_uuids, completed_uuids).
		"""
		if not decode_uuids:
			return [], []
		
		# Build completion mask
		completion_mask = torch.zeros(len(decode_uuids), dtype=torch.int32, device=self.torch_device)
		
		for i, uuid in enumerate(decode_uuids):
			if uuid in self._uuid_to_local_map:
				# This rank owns this sequence - check actual state
				seq = self.global_batch.get_sequence(uuid)
				if seq.eos_reached or seq.decoded_length >= self.max_decoding_length:
					completion_mask[i] = 1
		
		# All-reduce MAX: if ANY rank says complete, it's complete
		dist.all_reduce(completion_mask, op=dist.ReduceOp.MAX)
		
		# Split into active and completed
		active_uuids = []
		completed_uuids = []
		
		for i, uuid in enumerate(decode_uuids):
			seq = self.global_batch.get_sequence(uuid)
			if completion_mask[i].item() == 1:
				completed_uuids.append(uuid)
				# Update local state for consistency
				seq.eos_reached = True
			else:
				active_uuids.append(uuid)
		
		return active_uuids, completed_uuids


	def _decoding_legacy_modes(
		self,
		new_tokens: torch.Tensor,
		decode_uuids: List[str],
		batch: List[int],
		start_token_idx: int
	) -> None:
		"""Legacy decoding for modes 0, 1, 2 with continuous batching support."""
		new_token_idx = start_token_idx
		
		while new_token_idx < self.max_decoding_length and (decode_uuids or batch):
			if self.rank == 0:
				logging.info(f"Decoding new token idx: {new_token_idx}")
			
			# Page boundary check
			if new_token_idx > 0 and new_token_idx % self.PAGE_SIZE == 0:
				dist.barrier()
				decode_uuids, batch, completed_uuids = self._check_and_handle_completions(
					decode_uuids, batch, new_token_idx
				)
				if completed_uuids:
					self._update_batch_status(completed_uuids, SequenceStatus.COMPLETED)
					self._release_host_kv_pages_for_batch(completed_uuids)
				
				if decode_uuids:
					decode_uuids, batch = self._try_load_new_sequences(decode_uuids, batch)
				
				dist.barrier()
				
				if not decode_uuids:
					break
			
			RUNTIME_ATTN_MODE = self.engine_config.Basic_Config.attn_mode

			if RUNTIME_ATTN_MODE == 0:
				"""CPU ATTN MODE - NO ATTN MICRO BATCH"""
				with torch.inference_mode():
					Attn_Wrapper.cur_batch = [batch]
					attention_mask = torch.cat(
						[
							self.query_book[query_idx].encoded["attention_mask"][
								:, : self.max_input_length + new_token_idx
							]
							for query_idx in batch
						],
						dim=0,
					)
					if "deepseek" not in self.model_config.model_type:
						position_ids = create_position_ids_from_attention_mask(
							attention_mask
						)[:, -1].unsqueeze(-1)
					else:
						position_ids = create_position_ids_from_attention_mask(attention_mask)

					Attn_Wrapper.attention_mask = attention_mask
					Attn_Wrapper.position_ids = position_ids
					new_tokens = self.model(
						new_tokens.to(self.torch_device),
						attention_mask=attention_mask.to(self.torch_device),
						use_cache=False,
					)
					new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(-1, 1)
					self.update_new_token(new_tokens, batch, new_token_idx)
					
					# Update sequence state
					for i, local_idx in enumerate(batch):
						uuid = self._local_to_uuid_map[local_idx]
						seq = self.global_batch.get_sequence(uuid)
						seq.decoded_length = new_token_idx + 1
						seq.current_context_length = seq.prompt_length + new_token_idx + 1
						if new_tokens[i].item() == self.eos_token_id:
							seq.eos_reached = True
				new_token_idx += 1

			elif RUNTIME_ATTN_MODE == 1:
				"""GPU ATTN MODE - ATTN MICRO BATCH"""
				micro_batch_size = self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
				num_micro_batches = math.ceil(len(batch) / micro_batch_size)
				micro_batches = [
					batch[micro_batch_idx * micro_batch_size : (micro_batch_idx + 1) * micro_batch_size]
					for micro_batch_idx in range(num_micro_batches)
				]
				Attn_Wrapper.cur_batch = micro_batches
				
				if (new_token_idx - 1) % 32 == 0:
					for idx in range(new_token_idx - 1, new_token_idx + 31):
						if "deepseek" in self.model_config.model_type:
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
							raise ValueError(f"Model architecture {self.model_config.model_type} not supported yet.")

						for layer_idx in range(self.model_config.num_hidden_layers):
							for micro_batch_idx in range(num_micro_batches):
								cur_batch = micro_batches[micro_batch_idx]
								self.core_engine.submit_to_KV_queue(
									cur_batch, micro_batch_idx, layer_idx, past_kv_byte_size,
								)

				with torch.inference_mode():
					attention_mask = torch.cat(
						[
							self.query_book[query_idx].encoded["attention_mask"][
								:, : self.max_input_length + new_token_idx
							]
							for query_idx in batch
						],
						dim=0,
					).to(self.torch_device)
					if "deepseek" in self.model_config.model_type:
						position_ids = create_position_ids_from_attention_mask(attention_mask)
					else:
						position_ids = create_position_ids_from_attention_mask(attention_mask)[:, -1].unsqueeze(-1)

					Attn_Wrapper.attention_mask = attention_mask
					Attn_Wrapper.position_ids = position_ids
					new_tokens = self.model(
						new_tokens.to(self.torch_device),
						attention_mask=attention_mask.to(self.torch_device),
						use_cache=False,
					)
					new_tokens = torch.argmax(new_tokens.logits, dim=-1).view(-1, 1)
					self.update_new_token(new_tokens, batch, new_token_idx)
					
					# Update sequence state
					for i, local_idx in enumerate(batch):
						uuid = self._local_to_uuid_map[local_idx]
						seq = self.global_batch.get_sequence(uuid)
						seq.decoded_length = new_token_idx + 1
						seq.current_context_length = seq.prompt_length + new_token_idx + 1
						if new_tokens[i].item() == self.eos_token_id:
							seq.eos_reached = True
				new_token_idx += 1

			elif RUNTIME_ATTN_MODE == 2:
				"""CPU-GPU Parallel ATTN - Deprecated"""
				logging.warning("RUNTIME_ATTN_MODE 2 is deprecated")
				new_token_idx += 1

	# ============ Utility Methods ============

	def set_phase(self, phase: str):
		"""Control different behavior of the engine in different phases."""
		torch.cuda.empty_cache()
		self.core_engine.set_phase(phase)
		Attn_Wrapper.phase = phase
		Expert_Wrapper.phase = phase

	def update_new_token(
		self, new_tokens: torch.Tensor, query_idx: List[int], new_token_idx: int
	):
		new_tokens = new_tokens.to("cpu")
		for idx, q_idx in enumerate(query_idx):
			self.query_book[q_idx].decoded_tokens[:, new_token_idx] = new_tokens[idx]
			
			attention_mask = self.query_book[q_idx].encoded["attention_mask"][0]
			zeros_positions = (attention_mask == 0).nonzero(as_tuple=True)[0]
			if len(zeros_positions) > 0:
				first_zero_pos = zeros_positions[0].item()
				self.query_book[q_idx].encoded["attention_mask"][0, first_zero_pos] = torch.tensor(1, dtype=attention_mask.dtype)
			else:
				raise ValueError("No 0 found in the attention mask.")

	def init_nvshmem(self):
		"""Initialize NVSHMEM only once per batch, not per decode iteration."""
		if BATCHGEN_ENABLE_ALL_TO_ALL != "1" or nvshmem_init is None:
			logging.info("Skipping NVSHMEM initialization; BATCHGEN_ENABLE_ALL_TO_ALL is disabled or nvshmem_init missing")
			return
		
		# Check if already initialized this run
		if getattr(self, '_nvshmem_initialized_this_run', False):
			logging.debug(f"Rank {self.rank}: NVSHMEM already initialized this run, skipping")
			return
			
		import nvshmem.core as nvshmem
		from cuda.core.experimental import Device    
		rank = dist.get_rank()
		world_size = dist.get_world_size()
		local_rank = rank % torch.cuda.device_count()
		torch.cuda.set_device(local_rank)

		dev = Device(local_rank)
		dev.set_current()
		dist.barrier()
		nvshmem_init(
			global_rank=rank,
			local_rank=local_rank,
			world_size=world_size,
			device=dev
		)
		self._nvshmem_initialized_this_run = True
		print(f"Rank {rank}: NVSHMEM initialized and Symmetric Heap allocated.")

	def _finalize_nvshmem(self) -> None:
		"""Finalize NVSHMEM if it was initialized."""
		if not getattr(self, '_nvshmem_initialized_this_run', False):
			return
			
		if BATCHGEN_ENABLE_ALL_TO_ALL != "1":
			return
			
		try:
			import nvshmem.core as nvshmem
			# Check if nvshmem has a finalize method
			if hasattr(nvshmem, 'finalize'):
				nvshmem.finalize()
				logging.info(f"Rank {self.rank}: NVSHMEM finalized")
		except Exception as e:
			logging.warning(f"Rank {self.rank}: Failed to finalize NVSHMEM: {e}")
		
		self._nvshmem_initialized_this_run = False

	def _init_torch_dist(self):
		timeout = timedelta(minutes=15)
		try:
			dist.init_process_group(
				backend="nccl",
				init_method="tcp://" + self.dist_init_addr,
				world_size=self.world_size,
				rank=self.global_rank,
				device_id=torch.device(f"cuda:{self.local_rank}"),
				timeout=timeout,
			)
		except RuntimeError as e:
			logging.error(f"Failed to initialize torch distributed: {e}")
			raise

	def _unregister_fp8_weights(self):
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
		if not hasattr(self, 'model') or self.model is None:
			return
		
		self.model.eval()
		self.model.to('cpu')
		with torch.no_grad():
			def clear_module(module):
				for param in module.parameters():
					param.data = torch.empty(0)
					if param.grad is not None:
						param.grad.data = torch.empty(0)
						param.grad = None
				for buffer in module.buffers():
					buffer.data = torch.empty(0)
				module._forward_hooks.clear()
				module._forward_pre_hooks.clear()
				module._backward_hooks.clear()
				for submodule in module.children():
					clear_module(submodule)
			
			clear_module(self.model)
		
		self.model.to('cpu')
		del self.model
		self.model = None
		
		if hasattr(self, 'optimizer'):
			self.optimizer.zero_grad(set_to_none=True)
			del self.optimizer
		
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.synchronize()
		
		for _ in range(3):
			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	def _reset_for_new_batch(self) -> None:
		"""
		Reset batch-specific state to prepare for a new batch.
		Does NOT reinitialize core_engine, parallel_manager, or other heavy components.
		"""
		logging.info(f"Rank {self.rank}: Resetting state for new batch")
		
		# Synchronize all ranks before cleanup
		dist.barrier()
		
		# 1. Cleanup communicator from previous batch
		if hasattr(self, 'comm') and self.comm is not None:
			try:
				del self.comm
				self.comm = None
			except Exception as e:
				logging.warning(f"Rank {self.rank}: Failed to cleanup comm: {e}")
		
		# 2. Release any remaining host KV pages
		if hasattr(self, 'global_batch') and self.global_batch is not None:
			try:
				worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
				if worker_view is not None and hasattr(self, '_uuid_to_local_map'):
					for uuid in self._uuid_to_local_map.keys():
						seq = self.global_batch.get_sequence(uuid)
						if seq is not None:
							try:
								worker_view.release_sequence_pages([seq.global_idx])
								worker_view.unregister_sequences([seq.global_idx])
							except Exception:
								pass  # Ignore errors for already-released pages
			except Exception as e:
				logging.warning(f"Rank {self.rank}: Failed to cleanup host KV: {e}")
		
		# 3. Destroy GPU KV cache (but keep the manager reference for reuse)
		self._destroy_gpu_paged_kv_cache(empty_cuda_cache=True)
		self.gpu_paged_kv_cache_manager = None
		
		# 4. Reset global batch state
		self.global_batch = None
		
		# 5. Reset query book and mappings
		self.query_book = None
		self._local_to_uuid_map = {}
		self._uuid_to_local_map = {}
		
		# 6. Reset counters
		self.num_global_queries = 0
		self.num_local_queries = 0
		
		# 7. Clean up model weights (but NOT core_engine or parallel_manager)
		if hasattr(self, 'model') and self.model is not None:
			try:
				self.deep_free_model_memory()
			except Exception as e:
				logging.warning(f"Rank {self.rank}: Failed to cleanup model: {e}")
		self.model = None
		
		# 8. Clear CUDA cache
		torch.cuda.empty_cache()
		torch.cuda.synchronize()
		
		# 9. Force garbage collection
		gc.collect()
		
		# Synchronize all ranks after cleanup
		dist.barrier()
		
		logging.info(f"Rank {self.rank}: State reset completed")