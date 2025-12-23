import concurrent.futures
import copy
import functools
import psutil
import logging
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Set

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

@dataclass
class BoundaryTimingStats:
	"""Timing statistics for page boundary operations."""
	# Phase 1: Sync
	wait_kv_append_ms: float = 0.0
	finalize_async_load_ms: float = 0.0
	rebuild_after_integration_ms: float = 0.0
	barrier_after_sync_ms: float = 0.0
	
	# Phase 2: Eviction
	sync_completion_ms: float = 0.0
	release_completed_ms: float = 0.0
	rebuild_after_eviction_ms: float = 0.0
	
	# Phase 3: Extension
	extend_page_buffer_ms: float = 0.0
	rebuild_after_extension_ms: float = 0.0
	
	# Phase 4: Async Load Launch
	allgather_free_pages_ms: float = 0.0
	select_candidates_ms: float = 0.0
	allocate_pages_ms: float = 0.0
	launch_async_load_ms: float = 0.0
	restore_page_table_ms: float = 0.0
	
	# Overall
	total_boundary_ms: float = 0.0
	barrier_final_ms: float = 0.0
	
	# Counts
	num_completed: int = 0
	num_onhold: int = 0
	num_loaded: int = 0
	
	def __str__(self) -> str:
		return (
			f"Boundary Timing (total={self.total_boundary_ms:.2f}ms):\n"
			f"  Phase 1 SYNC:\n"
			f"    wait_kv_append={self.wait_kv_append_ms:.2f}ms\n"
			f"    finalize_load={self.finalize_async_load_ms:.2f}ms\n"
			f"    rebuild_integration={self.rebuild_after_integration_ms:.2f}ms\n"
			f"    barrier={self.barrier_after_sync_ms:.2f}ms\n"
			f"  Phase 2 EVICTION ({self.num_completed} completed):\n"
			f"    sync_completion={self.sync_completion_ms:.2f}ms\n"
			f"    release={self.release_completed_ms:.2f}ms\n"
			f"    rebuild={self.rebuild_after_eviction_ms:.2f}ms\n"
			f"  Phase 3 EXTENSION ({self.num_onhold} on_hold):\n"
			f"    extend={self.extend_page_buffer_ms:.2f}ms\n"
			f"    rebuild={self.rebuild_after_extension_ms:.2f}ms\n"
			f"  Phase 4 ASYNC LOAD ({self.num_loaded} new):\n"
			f"    allgather={self.allgather_free_pages_ms:.2f}ms\n"
			f"    select={self.select_candidates_ms:.2f}ms\n"
			f"    allocate={self.allocate_pages_ms:.2f}ms\n"
			f"    launch={self.launch_async_load_ms:.2f}ms\n"
			f"    restore_pt={self.restore_page_table_ms:.2f}ms\n"
			f"  Final barrier={self.barrier_final_ms:.2f}ms"
		)


@dataclass
class FastBoundaryTimingStats:
	"""Detailed timing for optimized page boundary."""
	total_ms: float = 0.0
	# Phase 0: Async wait
	wait_kv_append_ms: float = 0.0
	num_kv_append_tasks: int = 0  # Track number of tasks waited
	wait_async_load_ms: float = 0.0
	finalize_load_ms: float = 0.0
	# Phase 1: Gather
	gather_ms: float = 0.0
	# Phase 2: Process
	process_ms: float = 0.0
	extension_ms: float = 0.0
	# Phase 3: Async load launch
	load_select_ms: float = 0.0
	load_alloc_ms: float = 0.0
	load_launch_ms: float = 0.0
	# Phase 4: Rebuild + MoE buffer + barrier
	rebuild_ms: float = 0.0
	moe_buffer_update_ms: float = 0.0  # Time to sync and update MoE buffer size
	barrier_ms: float = 0.0
	# Counts
	num_completed: int = 0
	num_onhold: int = 0
	num_loaded: int = 0
	# Status counts
	total_active: int = 0
	total_prefilled: int = 0
	total_completed_cumulative: int = 0


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
	# GPU_KV_CACHE_SIZE_GB = 20.0  # Default GPU KV cache size
	GPU_KV_CACHE_SIZE_GB = float(os.environ.get("BATCHGEN_GPU_KV_CACHE_SIZE_GB", "20.0"))


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
		self._ignore_eos: bool = False
		
		# 9. Initialization Flags
		self._core_initialized = False
		self._batch_completed = False
		self._nvshmem_initialized_this_run = False
		
		# 10. Distributed Communication Info
		self.dist_init_addr = args.dist_init_addr
		self.comm = None # Initialized lazily or in Init()

		COMM_MASTER_ADDR = self.dist_init_addr.split(':')[0]
		os.environ['COMM_MASTER_ADDR'] = COMM_MASTER_ADDR

		# Add GPU KV cache size configuration
		self.gpu_kv_cache_size_gb = getattr(args, 'gpu_kv_cache_size_gb', self.GPU_KV_CACHE_SIZE_GB)
		
		# Track sequences currently with GPU KV allocated
		self._sequences_with_gpu_kv: Set[str] = set()

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

	def _initialize_gpu_kv_manager_fixed_size(self) -> GPUPagedKVCacheManager:
		"""
		Initialize GPU KV manager with pre-determined fixed size.
		Called once at the start of decoding.
		"""
		from batchgen.kv_cache.host_kv_mananger_config import build_gpu_kv_config_fixed_size
		
		config = build_gpu_kv_config_fixed_size(
			model_name=self.huggingface_ckpt_name,
			gpu_kv_cache_size_gb=self.gpu_kv_cache_size_gb,
		)
		
		manager = GPUPagedKVCacheManager(
			config=config,
			device=self.local_rank,
		)
		manager.initialize()
		self._bind_gpu_paged_kv_manager(manager)
		
		logging.info(
			f"Rank {self.rank}: Initialized fixed-size GPU KV manager: "
			f"{self.gpu_kv_cache_size_gb}GB, {config.num_pages} pages"
		)
		
		return manager
		
	def set_ignore_eos(self, ignore_eos: bool) -> None:
		"""
		Set whether to ignore EOS tokens during decoding.
		
		When True, sequences will decode to max_decoding_length regardless of EOS.
		Useful for benchmarking to ensure consistent workload across all sequences.
		
		Args:
			ignore_eos: If True, ignore EOS tokens
		"""
		self._ignore_eos = ignore_eos
		logging.info(f"Rank {self.rank}: ignore_eos set to {ignore_eos}")

	def _should_stop_at_eos(self, token_id: int) -> bool:
		"""
		Check if we should stop at this token.
		
		Returns True if token is EOS AND we're not ignoring EOS.
		"""
		if self._ignore_eos:
			return False
		return token_id == self.eos_token_id

	def _is_sequence_completed(self, seq) -> bool:
		"""
		Unified completion check that respects ignore_eos.
		
		A sequence is completed if:
		1. It reached max_decoding_length (always checked), OR
		2. It hit EOS AND ignore_eos is False
		"""
		# Always complete at max length
		if seq.decoded_length >= self.max_decoding_length:
			return True
		
		# Only complete at EOS if not ignoring EOS
		if seq.eos_reached and not self._ignore_eos:
			return True
		
		return False

	def _compute_two_page_buffer_allocation(
		self, 
		uuids: List[str]
	) -> Dict[str, int]:
		"""
		Compute GPU page allocation for two-page buffer design.
		
		Returns:
			Dict mapping uuid -> pages_to_allocate
		"""
		allocations = {}
		for uuid in uuids:
			seq = self.global_batch.get_sequence(uuid)
			pages_needed = seq.get_gpu_pages_for_two_page_buffer()
			allocations[uuid] = pages_needed
		return allocations

	def _compute_two_page_buffer_tokens(self, local_indices: List[int]) -> List[int]:
		"""Compute tokens for two-page buffer GPU allocation (NOT full context)."""
		tokens = []
		for local_idx in local_indices:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			pages = seq.get_gpu_pages_for_two_page_buffer()
			tokens.append(pages * self.PAGE_SIZE)
		return tokens

	def _allocate_gpu_kv_two_page_buffer(
		self, 
		local_sequence_ids: List[int],
		load_from_host: bool = True
	) -> bool:
		"""
		Allocate GPU KV pages using two-page buffer strategy.
		
		Returns:
			True if allocation succeeded, False otherwise.
		"""
		if not local_sequence_ids:
			return True
		
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			return False
		
		global_ids = self._local_indices_to_global_seq_ids(local_sequence_ids)
		
		pages_per_seq = []
		total_pages = 0
		
		for local_idx in local_sequence_ids:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			pages = seq.get_gpu_pages_for_two_page_buffer()
			pages_per_seq.append(pages * self.PAGE_SIZE)  # tokens for API
			total_pages += pages

		free_pages = manager.get_stats().num_free_pages
		if total_pages > free_pages:
			logging.error(
				f"Rank {self.rank}: Cannot allocate GPU KV - need {total_pages} pages, "
				f"only {free_pages} free"
			)
			# Don't set gpu_pages_allocated since we're failing
			return False
		
		# Now safe to update tracking (allocation will succeed)
		for local_idx in local_sequence_ids:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			pages = seq.get_gpu_pages_for_two_page_buffer()
			seq.gpu_pages_allocated = pages
		
		manager.allocate_pages_for_sequences(global_ids, pages_per_seq)
		manager.rebuild_page_table(global_ids)
		
		if load_from_host:
			self._load_host_kv_to_gpu(manager, global_ids)
		
		# Track in set
		for local_idx in local_sequence_ids:
			uuid = self._local_to_uuid_map[local_idx]
			self._sequences_with_gpu_kv.add(uuid)
		
		# Rebuild page table with ALL active sequences
		all_active_global_ids = []
		for uuid in self._sequences_with_gpu_kv:
			if uuid in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				all_active_global_ids.append(seq.global_idx)
		all_active_global_ids.sort()
		
		if all_active_global_ids:
			manager.rebuild_page_table(all_active_global_ids)
		
		logging.info(
			f"Rank {self.rank}: Allocated two-page buffer GPU KV for {len(global_ids)} sequences"
		)
		return True

	def _extend_gpu_kv_allocation(self, uuids: List[str]) -> bool:
		"""
		Extend GPU KV allocation for sequences that need more pages.
		
		Returns:
			True if all extensions succeeded, False if insufficient pages
		"""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			return False
		
		free_pages = manager.get_stats().num_free_pages
		
		extensions_needed = []
		total_additional = 0
		
		for uuid in uuids:
			if uuid not in self._uuid_to_local_map:
				continue
			seq = self.global_batch.get_sequence(uuid)
			additional = seq.get_additional_gpu_pages_needed()
			if additional > 0:
				extensions_needed.append((uuid, additional))
				total_additional += additional
		
		if total_additional > free_pages:
			logging.warning(
				f"Rank {self.rank}: Insufficient GPU pages for extension: "
				f"need {total_additional}, have {free_pages}"
			)
			return False
		
		# Perform extensions
		for uuid, additional in extensions_needed:
			seq = self.global_batch.get_sequence(uuid)
			local_idx = self._uuid_to_local_map[uuid]
			global_id = seq.global_idx
			
			# Extend allocation
			new_total_pages = seq.gpu_pages_allocated + additional
			new_total_tokens = new_total_pages * self.PAGE_SIZE
			
			manager.extend_pages_for_sequence(global_id, new_total_tokens)
			seq.gpu_pages_allocated = new_total_pages
		
		return True

	def _select_sequences_for_onhold(
		self, 
		active_uuids: List[str],
		required_free_pages: int
	) -> List[str]:
		"""
		Select sequences to put ON_HOLD to free up GPU pages.
		
		Strategy: Select sequences with most progress (closest to completion)
		as they have more KV in host already.
		
		Returns:
			List of uuids to put ON_HOLD
		"""
		manager = self.gpu_paged_kv_cache_manager
		current_free = manager.get_stats().num_free_pages if manager else 0
		pages_to_free = required_free_pages - current_free
		
		if pages_to_free <= 0:
			return []
		
		# Sort by decoded_length descending (most progress first)
		candidates = []
		for uuid in active_uuids:
			if uuid not in self._uuid_to_local_map:
				continue
			seq = self.global_batch.get_sequence(uuid)
			candidates.append((uuid, seq.decoded_length, seq.gpu_pages_allocated))
		
		candidates.sort(key=lambda x: x[1], reverse=True)
		
		onhold_uuids = []
		freed = 0
		
		for uuid, _, pages in candidates:
			if freed >= pages_to_free:
				break
			onhold_uuids.append(uuid)
			freed += pages
		
		return onhold_uuids

	def _put_sequences_onhold(self, uuids: List[str]) -> None:
		"""Put sequences ON_HOLD: release GPU KV pages, keep host KV."""
		if not uuids:
			return
		
		my_uuids = [u for u in uuids if u in self._uuid_to_local_map]
		
		if my_uuids:
			local_indices = self._get_local_indices_for_uuids(my_uuids)
			global_ids = self._local_indices_to_global_seq_ids(local_indices)
			
			manager = self.gpu_paged_kv_cache_manager
			if manager is not None:
				manager.free_pages_for_sequences(global_ids)
			
			for uuid in my_uuids:
				seq = self.global_batch.get_sequence(uuid)
				seq.gpu_pages_allocated = 0
				self._sequences_with_gpu_kv.discard(uuid)
		
		# self._update_batch_status(uuids, SequenceStatus.ON_HOLD)
		
		# FIX: Rebuild page table with remaining active sequences
		manager = self.gpu_paged_kv_cache_manager
		if manager is not None and manager.is_initialized:
			remaining_uuids = [u for u in self._sequences_with_gpu_kv if u not in set(uuids)]
			if remaining_uuids:
				remaining_local = self._get_local_indices_for_uuids(remaining_uuids)
				remaining_global = self._local_indices_to_global_seq_ids(remaining_local)
				manager.rebuild_page_table(remaining_global)

	def _append_decode_kv_to_host_async(
		self,
		layer_idx: int,
		batch: List[int],
		k_tensor: torch.Tensor,
	) -> None:
		"""
		Fire-and-forget KV append to host.
		
		Adds task to pending list, does NOT wait.
		Tasks are waited at page boundary via _wait_pending_kv_append_tasks().
		
		Safety: Host writes don't race with GPU reads (different memory spaces).
		
		CRITICAL: Must keep tensor references alive until async operation completes!
		PyTorch's CUDA caching allocator can reuse memory if tensor is dereferenced
		while async operation is still reading from it.
		"""
		if not batch:
			return
		
		worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		if worker_view is None:
			return
		
		# Build sequence info
		sequence_ids = []
		sequence_lengths = []
		
		for local_idx in batch:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			sequence_ids.append(seq.global_idx)
			# Write position is current position (0-indexed)
			sequence_lengths.append(seq.current_context_length - 1)
		
		# Reshape for MLA if needed
		if k_tensor.dim() == 3:
			k_tensor = k_tensor.unsqueeze(2)  # [B, 1, D] -> [B, 1, 1, D]
		
		# Launch async append
		task = worker_view.async_append_decode_kv_to_host(
			layer_idx=layer_idx,
			sequence_ids=sequence_ids,
			k_tensor=k_tensor,
			v_tensor=None,  # MLA doesn't have separate V
			sequence_lengths=sequence_lengths,
		)
		
		# CRITICAL FIX: Store tensor reference alongside task to prevent GC
		if not hasattr(self, '_pending_kv_append_tensors'):
			self._pending_kv_append_tensors = []
		self._pending_kv_append_tensors.append(k_tensor)
		
		# Add to pending list - will be waited at page boundary
		if task is not None:
			self._pending_kv_append_tasks.append(task)
		# NO wait here!

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
		k_ptrs, v_ptrs = manager.get_padded_3d_page_pointers()
		active_sequence_page_counts = manager.export_active_sequence_page_counts()
		
		logging.debug(
			f"Rank {self.rank}: _load_host_kv_to_gpu launching async load for "
			f"{len(global_sequence_ids)} sequences..."
		)
		
		load_task = worker_view.async_load_layer_paged_kv_to_device(
			sequence_ids=sequence_tensor,
			active_page_counts=active_sequence_page_counts,
			k_device_ptrs=k_ptrs,
			v_device_ptrs=v_ptrs,
		)
		
		# Wait for load to complete (this is synchronous load path used during prefill)
		load_task.wait()
		
		# NOTE: No cuda sync needed - load_task.wait() ensures data is ready
		load_duration = time.perf_counter() - copy_start
		logging.debug(
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
			# NOTE: No sync needed - page deallocation is synchronous to the allocator
			logging.debug(
				f"Rank {self.rank} Released GPU KV pages for global_idx: {global_sequence_ids}"
			)
			
			# FIX Bug 2: Remove from tracking set and reset gpu_pages_allocated
			for local_idx in local_sequence_ids:
				uuid = self._local_to_uuid_map.get(local_idx)
				if uuid:
					self._sequences_with_gpu_kv.discard(uuid)
					seq = self.global_batch.get_sequence(uuid)
					if seq is not None:
						seq.gpu_pages_allocated = 0
					
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
		
		# FIX Bug 2: Clear tracking set when GPU KV is destroyed
		self._sequences_with_gpu_kv.clear()

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

		# VALIDATION: All ranks must have same global batch size
		local_batch_size = torch.tensor([len(self.global_batch)], dtype=torch.int64, device=self.torch_device)
		all_sizes = [torch.zeros_like(local_batch_size) for _ in range(self.world_size)]
		dist.all_gather(all_sizes, local_batch_size)
		all_sizes_list = [int(t.item()) for t in all_sizes]
		if len(set(all_sizes_list)) > 1:
			logging.error(
				f"Rank {self.rank}: CRITICAL - global_batch sizes DIFFER across ranks! "
				f"Sizes: {all_sizes_list}"
			)
			raise RuntimeError(f"Global batch size mismatch: {all_sizes_list}")
		
		logging.info(f"Rank {self.rank}: All ranks have {all_sizes_list[0]} sequences in global_batch")

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
			try:
				self.global_batch.update_status(uuid, new_status)
			except ValueError as e:
				logging.warning(f"Rank {self.rank}: Invalid status transition for {uuid}: {e}")

	def _sync_sequence_metadata(self, decode_uuids: List[str]) -> None:
		"""
		Synchronize sequence metadata (decoded_length, current_context_length, 
		gpu_pages_allocated) across all ranks.
		
		Each rank reports its local sequences' state, and all ranks update their 
		local SequenceEntry objects with the gathered info.
		
		CRITICAL: Must be called at page boundaries to maintain consistent view.
		"""
		if not decode_uuids:
			return
		
		# Step 1: Each rank reports state for sequences it owns
		local_state = {}
		for uuid in decode_uuids:
			if uuid in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				local_state[uuid] = {
					'decoded_length': seq.decoded_length,
					'current_context_length': seq.current_context_length,
					'gpu_pages_allocated': seq.gpu_pages_allocated,
					'eos_reached': seq.eos_reached,
				}
		
		# Step 2: All-gather state from all ranks
		all_states = [None] * self.world_size
		dist.all_gather_object(all_states, local_state)
		
		# Step 3: Merge and update local SequenceEntry objects
		for rank_state in all_states:
			if rank_state:
				for uuid, state in rank_state.items():
					if uuid not in self._uuid_to_local_map:
						# This sequence belongs to another rank - update our local copy
						seq = self.global_batch.get_sequence(uuid)
						if seq is not None:
							seq.decoded_length = state['decoded_length']
							seq.current_context_length = state['current_context_length']
							seq.gpu_pages_allocated = state['gpu_pages_allocated']
							seq.eos_reached = state['eos_reached']

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
		
		# Validation: Check that we have all sequences assigned to this rank
		expected_count = 0
		for seq in self.global_batch:
			if seq.global_idx % self.world_size == self.rank:
				expected_count += 1
		
		if len(my_uuids) != expected_count:
			logging.error(
				f"Rank {self.rank}: CRITICAL MISMATCH - expected {expected_count} sequences "
				f"but got {len(my_uuids)} from get_sequences_for_rank!"
			)
		
		logging.info(
			f"Rank {self.rank}: Built local query_book with {len(self.query_book)} entries "
			f"(expected {expected_count}, global_batch has {len(self.global_batch)} sequences)"
		)

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
		Greedily fill GPU KV cache to ~90% capacity.
		"""
		prefilled_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		onhold_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
		
		# Combine and sort for deterministic ordering
		all_candidates = prefilled_uuids + onhold_uuids
		all_candidates.sort(key=lambda uuid: self.global_batch.get_sequence(uuid).global_idx)
		
  
		if not all_candidates:
			return []
		
		# Get GPU page capacity
		if self.gpu_paged_kv_cache_manager is not None and self.gpu_paged_kv_cache_manager.is_initialized:
			total_pages = self.gpu_paged_kv_cache_manager.get_stats().num_total_pages
		else:
			# Initial batch: estimate from config
			from batchgen.kv_cache.host_kv_mananger_config import build_gpu_kv_config_fixed_size
			config = build_gpu_kv_config_fixed_size(
				model_name=self.huggingface_ckpt_name,
				gpu_kv_cache_size_gb=self.gpu_kv_cache_size_gb,
			)
			total_pages = config.num_pages
		
		# 90% watermark
		capacity_per_rank = int(total_pages * 0.9)
		
		# Greedily fill
		rank_pages_used = [0] * self.world_size
		decode_batch = []
		
		for uuid in all_candidates:
			seq = self.global_batch.get_sequence(uuid)
			assigned_rank = seq.assigned_rank
			req_pages = seq.get_gpu_pages_for_two_page_buffer()
			
			if rank_pages_used[assigned_rank] + req_pages <= capacity_per_rank:
				decode_batch.append(uuid)
				rank_pages_used[assigned_rank] += req_pages
		
		logging.info(
			f"Rank {self.rank}: Prepared decode batch: {len(decode_batch)} sequences, "
			f"pages per rank: {rank_pages_used}"
		)
		
		return decode_batch

	def _check_and_extend_page_buffer(
		self,
		decode_uuids: List[str],
		batch: List[int]
	) -> Tuple[List[str], List[int], List[str]]:
		"""
		Ensure all active sequences maintain two-page buffer invariant.
		
		CRITICAL: All ranks MUST participate in ALL collective operations.
		No early returns before the final collective sync.
		"""
		if not decode_uuids:
			return [], [], []
		
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			return decode_uuids, batch, []
		
		# VALIDATION: Check that all decode_uuids exist in global_batch with valid assigned_rank
		invalid_uuids = []
		for uuid in decode_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				invalid_uuids.append((uuid, "NOT_IN_GLOBAL_BATCH", None))
			elif seq.assigned_rank is None:
				invalid_uuids.append((uuid, "NO_ASSIGNED_RANK", seq.global_idx))
		
		if invalid_uuids:
			logging.error(
				f"Rank {self.rank}: VALIDATION FAILED - {len(invalid_uuids)} invalid sequences in decode_uuids! "
				f"First 10: {invalid_uuids[:10]}"
			)
		
		logging.info(
			f"Rank {self.rank}: _check_and_extend ENTER: "
			f"decode_uuids={len(decode_uuids)}, batch={len(batch)}"
		)
		
		# ============ Step 1: Each rank reports extension needs ============
		local_ext_info = {}
		# DEBUG: Track which sequences SHOULD be mine but aren't in map
		should_be_mine_but_missing = []
		for uuid in decode_uuids:
			seq = self.global_batch.get_sequence(uuid)
			expected_owner = seq.global_idx % self.world_size
			if expected_owner == self.rank and uuid not in self._uuid_to_local_map:
				should_be_mine_but_missing.append((uuid, seq.global_idx, seq.assigned_rank))
			
			if uuid in self._uuid_to_local_map:
				local_ext_info[uuid] = {
					'global_idx': seq.global_idx,
					'decoded_length': seq.decoded_length,
					'gpu_pages_allocated': seq.gpu_pages_allocated,
					'additional_needed': seq.get_additional_gpu_pages_needed(),
					'current_context_length': seq.current_context_length,
				}
		
		if should_be_mine_but_missing:
			logging.error(
				f"Rank {self.rank}: OWNERSHIP BUG - {len(should_be_mine_but_missing)} sequences "
				f"should be mine but not in _uuid_to_local_map! First 5: {should_be_mine_but_missing[:5]}"
			)
			logging.error(f"Rank {self.rank}: _uuid_to_local_map has {len(self._uuid_to_local_map)} entries")
		
		# ============ Step 2: ALL-GATHER extension info (COLLECTIVE #1) ============
		all_ext_info = [None] * self.world_size
		dist.all_gather_object(all_ext_info, local_ext_info)
		
		# DEBUG: Log what each rank reported
		per_rank_reported = [len(r) if r else 0 for r in all_ext_info]
		logging.info(f"Rank {self.rank}: Per-rank reported sequences: {per_rank_reported}, total decode_uuids={len(decode_uuids)}")
		
		global_seq_info = {}
		for rank_idx, rank_info in enumerate(all_ext_info):
			if rank_info:
				for uuid, info in rank_info.items():
					global_seq_info[uuid] = info
					global_seq_info[uuid]['owning_rank'] = rank_idx
		
		# DEBUG: Check for missing sequences
		missing_uuids = [u for u in decode_uuids if u not in global_seq_info]
		if missing_uuids:
			logging.error(
				f"Rank {self.rank}: After gather, {len(missing_uuids)} sequences MISSING from global_seq_info. "
				f"First 10: {missing_uuids[:10]}"
			)
			# Check which rank SHOULD own them
			missing_by_expected_owner = {}
			for uuid in missing_uuids:
				seq = self.global_batch.get_sequence(uuid)
				expected_owner = seq.global_idx % self.world_size
				actual_assigned = seq.assigned_rank
				if expected_owner not in missing_by_expected_owner:
					missing_by_expected_owner[expected_owner] = []
				missing_by_expected_owner[expected_owner].append((uuid, seq.global_idx, actual_assigned))
			logging.error(f"Rank {self.rank}: Missing sequences by expected owner: {[(k, len(v)) for k, v in missing_by_expected_owner.items()]}")
		
		# ============ FIX Bug 5-6: Update local SequenceEntry with gathered info ============
		# This ensures all ranks have consistent view of sequence state
		for uuid, info in global_seq_info.items():
			if uuid not in self._uuid_to_local_map:
				# This sequence belongs to another rank - update our local copy
				seq = self.global_batch.get_sequence(uuid)
				if seq is not None:
					seq.decoded_length = info['decoded_length']
					seq.current_context_length = info['current_context_length']
					seq.gpu_pages_allocated = info['gpu_pages_allocated']
		
		# ============ Step 3: All-gather free pages per rank (COLLECTIVE #2) ============
		local_free = manager.get_stats().num_free_pages
		free_tensor = torch.tensor([local_free], dtype=torch.int64, device=self.torch_device)
		gathered_free = [torch.zeros_like(free_tensor) for _ in range(self.world_size)]
		dist.all_gather(gathered_free, free_tensor)
		per_rank_free = {r: int(gathered_free[r].item()) for r in range(self.world_size)}
		
		# ============ Step 4: Group sequences by assigned rank ============
		seqs_by_rank = {r: [] for r in range(self.world_size)}
		missing_from_global_info = []  # Track sequences with no metadata
		for uuid in decode_uuids:
			if uuid not in global_seq_info:
				logging.error(f"Rank {self.rank}: MISSING uuid={uuid} from global_seq_info")
				missing_from_global_info.append(uuid)
				continue
			seq = self.global_batch.get_sequence(uuid)
			info = global_seq_info[uuid]
			seqs_by_rank[seq.assigned_rank].append({
				'uuid': uuid,
				**info
			})
		
		# CRITICAL: Sequences with no metadata are unsafe to process
		# Add them to onhold_set to exclude from active batch
		missing_set = set(missing_from_global_info)
		
		# Check if all ranks can extend (MUST BE COMPUTED IDENTICALLY ON ALL RANKS)
		all_can_extend = True
		for r in range(self.world_size):
			rank_additional = sum(s['additional_needed'] for s in seqs_by_rank[r])
			if rank_additional > per_rank_free[r]:
				all_can_extend = False
				break
		
		# ============ Initialize eviction state ============
		global_onhold = []
		onhold_set = set(missing_from_global_info)  # Include missing sequences in onhold
		local_extension_failed = []
		
		# ============ Step 5-8: Extension or Eviction (conditional logic) ============
		if all_can_extend:
			# No eviction needed - just extend locally
			my_uuids_needing_extension = [
				uuid for uuid in decode_uuids
				if uuid in self._uuid_to_local_map 
				and global_seq_info.get(uuid, {}).get('additional_needed', 0) > 0
			]
			
			if my_uuids_needing_extension:
				success = self._extend_gpu_kv_allocation(my_uuids_needing_extension)
				if not success:
					logging.error(f"Rank {self.rank}: Extension FAILED unexpectedly in no-eviction path")
					local_extension_failed = my_uuids_needing_extension
			
			logging.info(
				f"Rank {self.rank}: _check_and_extend (no eviction path): "
				f"{len(decode_uuids)} uuids, {len(batch)} batch"
			)
			# DO NOT RETURN - must participate in collective #3 below
			
		else:
			# ============ Step 6: Need eviction ============
			logging.info(f"Rank {self.rank}: EVICTION REQUIRED")
			
			# Sort by decoded_length descending
			for r in seqs_by_rank:
				seqs_by_rank[r].sort(key=lambda x: x['decoded_length'], reverse=True)
			
			# Compute eviction list (GLOBALLY CONSISTENT)
			for r in range(self.world_size):
				rank_seqs = seqs_by_rank[r]
				rank_free = per_rank_free[r]
				rank_additional = sum(s['additional_needed'] for s in rank_seqs)
				
				if rank_additional <= rank_free:
					continue
				
				pages_to_free = rank_additional - rank_free
				pages_freed = 0
				
				for s in rank_seqs:
					if pages_freed >= pages_to_free:
						break
					global_onhold.append(s['uuid'])
					pages_freed += s['gpu_pages_allocated']
			
			logging.info(f"Rank {self.rank}: global_onhold={len(global_onhold)} sequences")
			
			# ============ Step 7: Execute eviction ============
			onhold_set = set(global_onhold)
			my_onhold = [u for u in global_onhold if u in self._uuid_to_local_map]
			
			if my_onhold:
				local_indices = self._get_local_indices_for_uuids(my_onhold)
				global_ids = self._local_indices_to_global_seq_ids(local_indices)
				
				if global_ids:
					manager.free_pages_for_sequences(global_ids)
					# NOTE: No sync needed - page operations are synchronous
				
				for uuid in my_onhold:
					seq = self.global_batch.get_sequence(uuid)
					seq.gpu_pages_allocated = 0
					self._sequences_with_gpu_kv.discard(uuid)
				
				logging.info(f"Rank {self.rank}: Evicted {len(my_onhold)} local sequences")
			
			# Update status globally
			for uuid in global_onhold:
				self.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)
			
			# ============ Step 8: Extend remaining sequences ============
			my_remaining_needing_extension = [
				uuid for uuid in decode_uuids
				if uuid in self._uuid_to_local_map 
				and uuid not in onhold_set
				and global_seq_info.get(uuid, {}).get('additional_needed', 0) > 0
			]

			if my_remaining_needing_extension:
				success = self._extend_gpu_kv_allocation(my_remaining_needing_extension)
				if not success:
					logging.error(f"Rank {self.rank}: Extension FAILED - putting sequences ON_HOLD")
					local_extension_failed = my_remaining_needing_extension
					
					# Release their GPU allocation
					for uuid in local_extension_failed:
						seq = self.global_batch.get_sequence(uuid)
						if seq.gpu_pages_allocated > 0:
							global_id = seq.global_idx
							manager.free_pages_for_sequences([global_id])
						seq.gpu_pages_allocated = 0
						self._sequences_with_gpu_kv.discard(uuid)

		# ============ ALL-GATHER extension failures (COLLECTIVE #3 - ALL RANKS MUST CALL) ============
		all_failed = [None] * self.world_size
		dist.all_gather_object(all_failed, local_extension_failed)

		for rank_failed in all_failed:
			if rank_failed:
				for uuid in rank_failed:
					onhold_set.add(uuid)
					if uuid not in global_onhold:
						global_onhold.append(uuid)
					self.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)

		# Also mark missing sequences as ON_HOLD (they had no metadata reported)
		for uuid in missing_from_global_info:
			if uuid not in global_onhold:
				global_onhold.append(uuid)
			self.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)
		
		if missing_from_global_info:
			logging.warning(
				f"Rank {self.rank}: Put {len(missing_from_global_info)} sequences ON_HOLD "
				f"because no rank reported metadata for them"
			)

		# ============ Step 9: Build GLOBALLY CONSISTENT active lists ============
		active_uuids = [u for u in decode_uuids if u not in onhold_set]
		active_batch = self._get_local_indices_for_uuids(active_uuids)

		# ============ CRITICAL VALIDATION WITH REMOVAL ============
		valid_active_batch = []
		local_invalid_uuids = []

		for local_idx in active_batch:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			
			is_valid = True
			
			if seq.gpu_pages_allocated == 0:
				logging.error(f"Rank {self.rank}: REMOVING uuid={uuid} - gpu_pages_allocated=0")
				is_valid = False
			
			if uuid not in self._sequences_with_gpu_kv:
				logging.error(f"Rank {self.rank}: REMOVING uuid={uuid} - not in _sequences_with_gpu_kv")
				is_valid = False
			
			if is_valid:
				valid_active_batch.append(local_idx)
			else:
				local_invalid_uuids.append(uuid)

		# ============ SYNCHRONIZE INVALID SEQUENCES ACROSS RANKS (COLLECTIVE) ============
		# CRITICAL FIX: Each rank only validates its LOCAL sequences, so we must sync
		# invalid sequences globally to ensure all ranks have consistent active_uuids
		all_invalid = [None] * self.world_size
		dist.all_gather_object(all_invalid, local_invalid_uuids)
		
		global_invalid_set = set()
		for rank_invalid in all_invalid:
			if rank_invalid:
				for uuid in rank_invalid:
					global_invalid_set.add(uuid)
					onhold_set.add(uuid)
					if uuid not in global_onhold:
						global_onhold.append(uuid)
					# Update status on all ranks
					self.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)

		active_batch = valid_active_batch
		active_uuids = [u for u in active_uuids if u not in global_invalid_set]
		
		# ============ ALL-REDUCE VALIDATION (COLLECTIVE #4) ============
		local_active_count = torch.tensor([len(active_uuids)], dtype=torch.int64, device=self.torch_device)
		all_active_counts = [torch.zeros_like(local_active_count) for _ in range(self.world_size)]
		dist.all_gather(all_active_counts, local_active_count)

		counts = [int(t.item()) for t in all_active_counts]
		if len(set(counts)) > 1:
			logging.error(f"Rank {self.rank}: DIVERGENCE! active_uuids counts differ across ranks: {counts}")

		logging.info(
			f"Rank {self.rank}: _check_and_extend EXIT: "
			f"active_uuids={len(active_uuids)}, active_batch={len(active_batch)}, "
			f"onhold={len(global_onhold)}, all_can_extend={all_can_extend}"
		)

		return active_uuids, active_batch, global_onhold

	def _check_and_handle_completions(
		self, 
		decode_uuids: List[str], 
		local_decode_indices: List[int],
		new_token_idx: int
	) -> Tuple[List[str], List[int], List[str]]:
		"""
		Check for completed sequences at page boundaries.
		FIXED: Respects ignore_eos flag.
		"""
		completed_uuids = []
		active_uuids = []
		active_local_indices = []
		
		for uuid in decode_uuids:
			seq = self.global_batch.get_sequence(uuid)
			
			# FIXED: Use unified completion check
			if self._is_sequence_completed(seq):
				completed_uuids.append(uuid)
				logging.info(
					f"Rank {self.rank}: Sequence {uuid} completed at token {new_token_idx} "
					f"(decoded_length={seq.decoded_length}, eos_reached={seq.eos_reached}, "
					f"ignore_eos={self._ignore_eos})"
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
	# 	Allocates GPU pages and triggers blocking load from Host.
	# 	"""
	# 	if not local_sequence_ids: return
		
	# 	manager = self.gpu_paged_kv_cache_manager
	# 	global_ids = self._local_indices_to_global_seq_ids(local_sequence_ids)
	# 	tokens = self._compute_host_kv_sequence_tokens(local_sequence_ids)

	# 	# 1. Allocate GPU Pages
	# 	manager.allocate_pages_for_sequences(global_ids, tokens)

	# 	# 2. Rebuild Page Table (Critical: Ensure kernel sees new pointers)
	# 	# We rebuild specifically for the sequences we are about to load
	# 	manager.rebuild_page_table(global_ids)

	# 	# 3. Load Host -> GPU (BLOCKING)
	# 	# "The load api is non-blocked, but we can use .wait() to let it be blocking for now."
	# 	self._load_host_kv_to_gpu(manager, global_ids) 

	# 	# 4. Rebuild Page Table for ALL active sequences (for next Attention forward)
	# 	# active_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.IN_DECODE)
	# 	# all_active_ids = [self.global_batch.get_sequence(u).global_idx for u in active_uuids if u in self._uuid_to_local_map]
	# 	# # Union with new ids
	# 	# final_ids = sorted(list(set(all_active_ids + global_ids)))
	# 	# manager.rebuild_page_table(final_ids) t


	def _allocate_and_load_gpu_kv_for_new_sequences(self, local_sequence_ids: List[int]) -> None:
		"""
		Allocates GPU pages using TWO-PAGE BUFFER strategy and triggers blocking load from Host.
		"""
		if not local_sequence_ids:
			return
		
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			logging.warning("GPU KV manager not initialized, cannot load new sequences")
			return
		
		global_ids = self._local_indices_to_global_seq_ids(local_sequence_ids)
		tokens = self._compute_two_page_buffer_tokens(local_sequence_ids)
		
		# Guard before allocation
		total_pages_needed = sum(t // self.PAGE_SIZE for t in tokens)
		free_pages = manager.get_stats().num_free_pages
		if total_pages_needed > free_pages:
			logging.error(
				f"Rank {self.rank}: Cannot allocate GPU KV - need {total_pages_needed} pages, "
				f"only {free_pages} free. Skipping load for {len(global_ids)} sequences."
			)
			return
		
		# 1. Allocate GPU Pages
		manager.allocate_pages_for_sequences(global_ids, tokens)

		# 2. Rebuild Page Table
		manager.rebuild_page_table(global_ids)

		# 3. Load Host -> GPU (BLOCKING)
		self._load_host_kv_to_gpu(manager, global_ids)
		
		# ← FIX: Update tracking state AFTER successful load
		for local_idx in local_sequence_ids:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			seq.gpu_pages_allocated = seq.get_gpu_pages_for_two_page_buffer()
			self._sequences_with_gpu_kv.add(uuid)

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
			while (self.global_batch.has_prefilled() or 
			   self.global_batch.has_in_decode() or 
			   self.global_batch.has_on_hold()):
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
				
				# CRITICAL FIX: Synchronize decode_uuids AND completion status across all ranks
				# Each rank reports its local view including completion status
				local_seq_status = {}
				for uuid in decode_uuids:
					seq = self.global_batch.get_sequence(uuid)
					local_seq_status[uuid] = {
						'completed': seq.status == SequenceStatus.COMPLETED or seq.eos_reached,
						'global_idx': seq.global_idx,
					}
				
				all_seq_status = [None] * self.world_size
				dist.all_gather_object(all_seq_status, local_seq_status)
				
				# Build global view: a sequence is COMPLETED if ANY rank marks it so
				global_completed = set()
				global_decode_candidates = set()
				
				for rank_status in all_seq_status:
					if rank_status:
						for uuid, info in rank_status.items():
							global_decode_candidates.add(uuid)
							if info['completed']:
								global_completed.add(uuid)
				
				# Also check local COMPLETED/eos_reached in case sequence wasn't in decode_uuids
				for seq in self.global_batch:
					if seq.status == SequenceStatus.COMPLETED or seq.eos_reached:
						global_completed.add(seq.uuid)
				
				# Sync global_completed across ranks to ensure consistency
				all_completed = [None] * self.world_size
				dist.all_gather_object(all_completed, list(global_completed))
				for rank_completed in all_completed:
					if rank_completed:
						global_completed.update(rank_completed)
				
				# Update local completion status to match global view
				for uuid in global_completed:
					seq = self.global_batch.get_sequence(uuid)
					if seq is not None:
						seq.eos_reached = True
						if seq.status != SequenceStatus.COMPLETED:
							try:
								self.global_batch.update_status(uuid, SequenceStatus.COMPLETED)
							except ValueError:
								pass  # Already in incompatible state
				
				# Build final decode_uuids excluding completed sequences
				decode_uuids = []
				for uuid in sorted(global_decode_candidates):
					if uuid not in global_completed:
						seq = self.global_batch.get_sequence(uuid)
						if seq is not None:
							decode_uuids.append(uuid)
				
				# Re-sort for deterministic ordering
				decode_uuids.sort(key=lambda u: self.global_batch.get_sequence(u).global_idx)
				
				# Verify all ranks have same decode_uuids count
				local_count = torch.tensor([len(decode_uuids)], dtype=torch.int64, device=self.torch_device)
				all_counts = [torch.zeros_like(local_count) for _ in range(self.world_size)]
				dist.all_gather(all_counts, local_count)
				counts = [int(t.item()) for t in all_counts]
				if len(set(counts)) > 1:
					logging.error(f"Rank {self.rank}: decode_uuids STILL divergent after sync! counts={counts}")
				
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

					# Use optimized decoding by default, set BATCHGEN_LEGACY_DECODE=1 for old version
					use_legacy = os.environ.get("BATCHGEN_LEGACY_DECODE", "0") == "1"
					if use_legacy:
						self.decoding_continuous(new_tokens, decode_uuids, local_decode_indices)
					else:
						self.decoding_continuous_fast(new_tokens, decode_uuids, local_decode_indices)
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
		"""Configure decoding phase with pre-sized GPU KV manager."""
		logging.info(f"Rank {self.rank}: Starting _config_decoding_for_batch")
		
		self.deep_free_model_memory()
		self.init_nvshmem()
		
		num_local_seq = len(local_decode_indices)
		num_seq_per_rank = torch.zeros(self.world_size, dtype=torch.int32, device=self.torch_device)
		num_seq_per_rank[self.rank] = num_local_seq
		dist.all_reduce(num_seq_per_rank, op=dist.ReduceOp.SUM)
		max_num_seq = int(num_seq_per_rank.max().item())
		
		# Initialize GPU KV manager with fixed size if not already done
		if self.gpu_paged_kv_cache_manager is None:
			self._initialize_gpu_kv_manager_fixed_size()
		
		if self.world_size <= 8:
			self.model, self.weight_copy_task = self.parallel_manager.configure_decoding()
			self.set_phase("decode")
			self.core_engine.stop_h2d_worker()
			self.core_engine.clear_kv_copy_queue()
			self.core_engine.clear_weight_copy_queue()
			self.core_engine.reset_decoding_buffer()
			self.core_engine.set_weight_copy_queue(self.weight_copy_task)
			self.core_engine.start_h2d_worker()
			
			# FIX Bug 2: Track GPU KV allocation for world_size <= 8 path as well
			if local_decode_indices:
				self._allocate_gpu_kv_two_page_buffer(local_decode_indices, load_from_host=True)
				for local_idx in local_decode_indices:
					uuid = self._local_to_uuid_map[local_idx]
					seq = self.global_batch.get_sequence(uuid)
					seq.gpu_pages_allocated = seq.get_gpu_pages_for_two_page_buffer()
					self._sequences_with_gpu_kv.add(uuid)
		else:
			# Use two-page buffer allocation, NOT _prepare_gpu_paged_kv_cache
			self._allocate_gpu_kv_two_page_buffer(local_decode_indices, load_from_host=True)
			
			# Track correctly
			for local_idx in local_decode_indices:
				uuid = self._local_to_uuid_map[local_idx]
				seq = self.global_batch.get_sequence(uuid)
				seq.gpu_pages_allocated = seq.get_gpu_pages_for_two_page_buffer()
				self._sequences_with_gpu_kv.add(uuid)
			
			self.model, self.weight_copy_task = self.parallel_manager.pure_gpu_decoding(max_num_seq, comm)
			
			self.set_phase("decode")
			self.core_engine.stop_h2d_worker()
			self.core_engine.clear_kv_copy_queue()
			self.core_engine.clear_weight_copy_queue()
			self.core_engine.reset_decoding_buffer()
		
		logging.info(f"Rank {self.rank}: _config_decoding_for_batch completed")

	def _prepare_decode_batch_two_page_buffer(self) -> List[str]:
		"""
		Select sequences for decode using two-page buffer strategy.
		Considers both PREFILLED and ON_HOLD sequences.
		"""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			return []
		
		free_pages = manager.get_stats().num_free_pages
		max_seqs_per_rank = self.engine_config.Module_Batching_Config.MoE_decoding_micro_batch_size
		
		# Get candidates: PREFILLED and ON_HOLD
		prefilled = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		onhold = self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
		
		candidates = prefilled + onhold
		candidates.sort(key=lambda u: self.global_batch.get_sequence(u).global_idx)
		
		if not candidates:
			return []
		
		# Select based on two-page buffer requirements
		rank_counts = [0] * self.world_size
		decode_batch = []
		total_pages_needed = 0
		
		for uuid in candidates:
			seq = self.global_batch.get_sequence(uuid)
			assigned_rank = seq.assigned_rank
			
			if rank_counts[assigned_rank] >= max_seqs_per_rank:
				continue
			
			# Calculate two-page buffer pages needed
			pages = seq.get_gpu_pages_for_two_page_buffer()
			
			if total_pages_needed + pages > free_pages:
				break
			
			decode_batch.append(uuid)
			rank_counts[assigned_rank] += 1
			total_pages_needed += pages
		
		logging.info(
			f"Rank {self.rank}: Prepared decode batch (two-page buffer): "
			f"{len(decode_batch)} sequences, {total_pages_needed} pages"
		)
		
		return decode_batch

	def _try_load_new_sequences_at_boundary_v2(
		self, 
		current_decode_uuids: List[str],
		current_batch: List[int]
	) -> Tuple[List[str], List[int]]:
		"""
		Load sequences at page boundary. Greedily fill available GPU pages.
		"""
		# Step 1: All-gather free GPU pages
		manager = self.gpu_paged_kv_cache_manager
		local_free = manager.get_stats().num_free_pages if manager and manager.is_initialized else 0
		
		free_tensor = torch.tensor([local_free], dtype=torch.int64, device=self.torch_device)
		gathered = [torch.zeros_like(free_tensor) for _ in range(self.world_size)]
		dist.all_gather(gathered, free_tensor)
		per_rank_free = [int(t.item()) for t in gathered]
		
		# Step 2: Get candidates
		prefilled = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		onhold = self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
		candidates = prefilled + onhold
		candidates.sort(key=lambda u: self.global_batch.get_sequence(u).global_idx)
		
		if not candidates:
			return current_decode_uuids, current_batch
		
		# Step 3: Greedily select based on available pages
		rank_pages_used = [0] * self.world_size
		new_uuids = []
		
		for uuid in candidates:
			seq = self.global_batch.get_sequence(uuid)
			assigned_rank = seq.assigned_rank
			req_pages = seq.get_gpu_pages_for_two_page_buffer()
			
			if rank_pages_used[assigned_rank] + req_pages <= per_rank_free[assigned_rank]:
				new_uuids.append(uuid)
				rank_pages_used[assigned_rank] += req_pages
		
		if not new_uuids:
			return current_decode_uuids, current_batch
		
		# Step 4: Load for THIS RANK
		my_new_uuids = [u for u in new_uuids 
					if self.global_batch.get_sequence(u).assigned_rank == self.rank]
		new_local_indices = self._get_local_indices_for_uuids(my_new_uuids)
		
		if new_local_indices:
			self._allocate_gpu_kv_two_page_buffer(new_local_indices, load_from_host=True)
		
		# Step 5: Update status
		self._update_batch_status(new_uuids, SequenceStatus.IN_DECODE)
		
		updated_decode_uuids = current_decode_uuids + new_uuids
		updated_batch = current_batch + new_local_indices
		
		logging.info(
			f"Rank {self.rank}: Loaded {len(new_uuids)} new sequences"
		)
		
		return updated_decode_uuids, updated_batch


	def _release_host_kv_pages_for_batch(self, uuids: List[str]) -> None:
		"""Release host KV pages for completed sequences owned by this rank.
		
		NOTE: This function only releases HOST KV pages. GPU KV pages should be
		released separately by calling _release_gpu_kv_pages() BEFORE this function.
		"""
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
			
			# NOTE: GPU KV pages should already be released by caller
			# Do NOT call _release_gpu_kv_pages here to avoid double-free
			
			# Release host KV pages
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
			
			# MODIFIED: Check for EOS respecting ignore_eos flag
			if self._should_stop_at_eos(new_tokens[i].item()):
				seq.eos_reached = True
		
		return new_tokens

	# ============ OPTIMIZED PAGE BOUNDARY (Consolidated Collectives) ============

	def _page_boundary_fast(
		self,
		decode_uuids: List[str],
		batch: List[int],
		gpu_manager: GPUPagedKVCacheManager,
		pending_async_load_task: Optional[object],
		pending_load_uuids: List[str],
		pending_load_local_indices: List[int],
		pending_load_global_ids: List[int],
		cumulative_completed: int = 0,  # Track total completed so far
	) -> Tuple[List[str], List[int], Optional[object], List[str], List[int], List[int], FastBoundaryTimingStats]:
		"""
		OPTIMIZED page boundary with consolidated collective operations.
		
		Reduces 10+ collectives to 2-3 by batching:
		1. Single all_gather_object for: sequence metadata + completion status + extension info + free pages
		2. One final barrier
		
		CRITICAL INVARIANTS FOR RANK ALIGNMENT:
		- All ranks must compute IDENTICAL decode_uuids, completed_uuids, onhold_uuids, new_load_uuids
		- Local operations (GPU page allocation, KV release) are rank-specific but globally coordinated
		- All decisions are based on gathered global state, not local state
		
		Returns:
			(decode_uuids, batch, new_async_task, new_load_uuids, new_load_local, new_load_global, timing)
		"""
		timing = FastBoundaryTimingStats()
		boundary_start = time.perf_counter()
		
		# ========== PHASE 0: Wait for pending async operations ==========
		t0 = time.perf_counter()
		timing.num_kv_append_tasks = self._wait_pending_kv_append_tasks()
		timing.wait_kv_append_ms = (time.perf_counter() - t0) * 1000
		
		# Integrate previous async load if any
		if pending_load_uuids:  # ALL ranks have identical pending_load_uuids
			t0 = time.perf_counter()
			if pending_async_load_task is not None:
				pending_async_load_task.wait()
				torch.cuda.synchronize(self.torch_device)
			timing.wait_async_load_ms = (time.perf_counter() - t0) * 1000
			
			# CRITICAL: barrier ensures all ranks finish async load before continuing
			dist.barrier()
			
			t0 = time.perf_counter()
			decode_uuids, batch = self._finalize_async_load_minimal(
				pending_async_load_task,
				pending_load_uuids,
				pending_load_local_indices,
				pending_load_global_ids,
				decode_uuids,
				batch,
				gpu_manager
			)
			timing.finalize_load_ms = (time.perf_counter() - t0) * 1000
		
		if not decode_uuids:
			timing.total_ms = (time.perf_counter() - boundary_start) * 1000
			return decode_uuids, batch, None, [], [], [], timing
		
		# ========== PHASE 1: SINGLE BATCHED ALL_GATHER ==========
		t0 = time.perf_counter()
		
		local_free_pages = gpu_manager.get_stats().num_free_pages if gpu_manager and gpu_manager.is_initialized else 0
		
		# Build local state for sequences owned by this rank
		local_seq_state = {}
		for uuid in decode_uuids:
			if uuid in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				local_seq_state[uuid] = {
					'decoded_length': seq.decoded_length,
					'current_context_length': seq.current_context_length,
					'gpu_pages_allocated': seq.gpu_pages_allocated,
					'eos_reached': seq.eos_reached,
					'completed': self._is_sequence_completed(seq),
					'additional_pages_needed': seq.get_additional_gpu_pages_needed(),
					'assigned_rank': seq.assigned_rank,  # Include for consistency
				}
		
		# Get candidates for loading (PREFILLED + ON_HOLD)
		prefilled = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		onhold_seqs = self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
		load_candidates = prefilled + onhold_seqs
		
		local_candidate_state = {}
		for uuid in load_candidates:
			if uuid in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				local_candidate_state[uuid] = {
					'pages_needed': seq.get_gpu_pages_for_two_page_buffer(),
					'assigned_rank': seq.assigned_rank,
				}
		
		# Pack everything into one dict for single all_gather
		local_payload = {
			'free_pages': local_free_pages,
			'seq_state': local_seq_state,
			'candidate_state': local_candidate_state,
		}
		
		all_payloads = [None] * self.world_size
		dist.all_gather_object(all_payloads, local_payload)
		
		timing.gather_ms = (time.perf_counter() - t0) * 1000
		
		# ========== PHASE 2: PROCESS GATHERED DATA (LOCAL COMPUTATION) ==========
		# CRITICAL: All computations below must be DETERMINISTIC across all ranks
		t0 = time.perf_counter()
		
		# Extract per-rank free pages
		per_rank_free = [p['free_pages'] for p in all_payloads]
		
		# Merge sequence state - each uuid appears exactly once (owned by one rank)
		global_seq_state = {}
		for rank_idx, payload in enumerate(all_payloads):
			if payload and payload['seq_state']:
				for uuid, state in payload['seq_state'].items():
					global_seq_state[uuid] = state
					global_seq_state[uuid]['owning_rank'] = rank_idx
		
		# Merge candidate state
		global_candidate_info = {}
		for payload in all_payloads:
			if payload and payload['candidate_state']:
				global_candidate_info.update(payload['candidate_state'])
		
		# VALIDATION: Check that all decode_uuids have state reported
		missing_uuids = [u for u in decode_uuids if u not in global_seq_state]
		if missing_uuids:
			logging.error(
				f"Rank {self.rank}: CRITICAL - {len(missing_uuids)} sequences missing from gathered state! "
				f"This indicates a bug. Missing: {missing_uuids[:5]}"
			)
			# Safe fallback: remove missing sequences from decode_uuids
			decode_uuids = [u for u in decode_uuids if u in global_seq_state]
		
		# Update local SequenceEntry with gathered info (for sequences on other ranks)
		for uuid, state in global_seq_state.items():
			if uuid not in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				if seq is not None:
					seq.decoded_length = state['decoded_length']
					seq.current_context_length = state['current_context_length']
					seq.gpu_pages_allocated = state['gpu_pages_allocated']
					seq.eos_reached = state['eos_reached']
		
		# ========== IDENTIFY COMPLETED SEQUENCES (DETERMINISTIC) ==========
		completed_uuids = []
		active_uuids = []
		for uuid in decode_uuids:
			state = global_seq_state.get(uuid)
			# CRITICAL: Use gathered 'completed' flag, not local computation
			if state and state['completed']:
				completed_uuids.append(uuid)
			else:
				active_uuids.append(uuid)
		
		timing.num_completed = len(completed_uuids)
		
		# ========== RELEASE COMPLETED SEQUENCES ==========
		if completed_uuids:
			self._update_batch_status(completed_uuids, SequenceStatus.COMPLETED)
			my_completed = [u for u in completed_uuids if u in self._uuid_to_local_map]
			if my_completed:
				my_completed_local = self._get_local_indices_for_uuids(my_completed)
				self._release_gpu_kv_pages(my_completed_local)
				self._release_host_kv_pages_for_batch(my_completed)
		
		decode_uuids = active_uuids
		batch = self._get_local_indices_for_uuids(decode_uuids)
		
		timing.process_ms = (time.perf_counter() - t0) * 1000
		
		if not decode_uuids:
			timing.total_ms = (time.perf_counter() - boundary_start) * 1000
			return decode_uuids, batch, None, [], [], [], timing
		
		# ========== CHECK/EXTEND PAGE BUFFERS (DETERMINISTIC) ==========
		t0 = time.perf_counter()
		# CRITICAL: Use gathered 'assigned_rank' to ensure all ranks agree
		seqs_needing_extension = []
		total_additional_by_rank = [0] * self.world_size
		
		for uuid in decode_uuids:
			state = global_seq_state.get(uuid)
			if state and state['additional_pages_needed'] > 0:
				# CRITICAL: Use assigned_rank from gathered state, not local
				assigned_rank = state['assigned_rank']
				total_additional_by_rank[assigned_rank] += state['additional_pages_needed']
				if uuid in self._uuid_to_local_map:
					seqs_needing_extension.append(uuid)
		
		# Check if all ranks can extend (DETERMINISTIC computation)
		all_can_extend = all(
			total_additional_by_rank[r] <= per_rank_free[r] 
			for r in range(self.world_size)
		)
		
		onhold_uuids = []
		if all_can_extend and seqs_needing_extension:
			# Simple extension - no eviction needed
			self._extend_gpu_kv_allocation(seqs_needing_extension)
		elif not all_can_extend:
			# Need eviction - put longest-decoded sequences on hold
			# CRITICAL: Sort DETERMINISTICALLY by (decoded_length DESC, global_idx ASC)
			for r in range(self.world_size):
				if total_additional_by_rank[r] > per_rank_free[r]:
					# Use gathered state for filtering AND sorting
					rank_seqs = [
						(uuid, global_seq_state[uuid]) 
						for uuid in decode_uuids 
						if uuid in global_seq_state and global_seq_state[uuid]['assigned_rank'] == r
					]
					# CRITICAL: Stable sort with tie-breaker for determinism
					rank_seqs.sort(
						key=lambda x: (-x[1]['decoded_length'], 
									self.global_batch.get_sequence(x[0]).global_idx)
					)
					
					pages_to_free = total_additional_by_rank[r] - per_rank_free[r]
					freed = 0
					for uuid, state in rank_seqs:
						if freed >= pages_to_free:
							break
						onhold_uuids.append(uuid)
						freed += state['gpu_pages_allocated']
			
			# Evict locally owned sequences
			my_onhold = [u for u in onhold_uuids if u in self._uuid_to_local_map]
			if my_onhold:
				local_indices = self._get_local_indices_for_uuids(my_onhold)
				global_ids = self._local_indices_to_global_seq_ids(local_indices)
				if global_ids and gpu_manager:
					gpu_manager.free_pages_for_sequences(global_ids)
				for uuid in my_onhold:
					seq = self.global_batch.get_sequence(uuid)
					seq.gpu_pages_allocated = 0
					self._sequences_with_gpu_kv.discard(uuid)
			
			for uuid in onhold_uuids:
				self.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)
			
			# Update active lists
			onhold_set = set(onhold_uuids)
			decode_uuids = [u for u in decode_uuids if u not in onhold_set]
			batch = self._get_local_indices_for_uuids(decode_uuids)
			
			# Extend remaining sequences
			remaining_needing_ext = [u for u in seqs_needing_extension if u not in onhold_set]
			if remaining_needing_ext:
				self._extend_gpu_kv_allocation(remaining_needing_ext)
		
		timing.num_onhold = len(onhold_uuids)
		timing.extension_ms = (time.perf_counter() - t0) * 1000
		
		# ========== PHASE 3: SELECT AND LAUNCH ASYNC LOAD (DETERMINISTIC) ==========
		t0 = time.perf_counter()
		new_async_task = None
		new_load_uuids = []
		new_load_local = []
		new_load_global = []
		
		if load_candidates and decode_uuids:
			# Sort candidates DETERMINISTICALLY by global_idx
			load_candidates.sort(key=lambda u: self.global_batch.get_sequence(u).global_idx)
			
			# CRITICAL: Recalculate available pages per rank
			# We need fresh counts after extension/eviction
			# Use: original_free - pages_used_for_extension + pages_freed_from_eviction
			# Simpler: just get fresh count from local manager
			local_free_now = gpu_manager.get_stats().num_free_pages if gpu_manager and gpu_manager.is_initialized else 0
			
			# For other ranks, estimate: original - additional_used
			adjusted_per_rank_free = list(per_rank_free)  # Copy
			for r in range(self.world_size):
				if r == self.rank:
					adjusted_per_rank_free[r] = local_free_now
				else:
					# Estimate: freed from completed + freed from onhold - used for extension
					freed_from_completed = sum(
						global_seq_state.get(u, {}).get('gpu_pages_allocated', 0)
						for u in completed_uuids
						if global_seq_state.get(u, {}).get('assigned_rank') == r
					)
					freed_from_onhold = sum(
						global_seq_state.get(u, {}).get('gpu_pages_allocated', 0)
						for u in onhold_uuids
						if global_seq_state.get(u, {}).get('assigned_rank') == r
					)
					used_for_extension = total_additional_by_rank[r] if all_can_extend else 0
					adjusted_per_rank_free[r] = per_rank_free[r] + freed_from_completed + freed_from_onhold - used_for_extension
			
			timing.load_select_ms = (time.perf_counter() - t0) * 1000
			
			# Select candidates that fit in ADJUSTED available GPU pages
			rank_pages_used = [0] * self.world_size
			for uuid in load_candidates:
				info = global_candidate_info.get(uuid)
				if info is None:
					continue
				
				req_pages = info['pages_needed']
				assigned_rank = info['assigned_rank']
				
				if req_pages == 0:
					continue
				
				if rank_pages_used[assigned_rank] + req_pages <= adjusted_per_rank_free[assigned_rank]:
					new_load_uuids.append(uuid)
					rank_pages_used[assigned_rank] += req_pages
			
			t_alloc = time.perf_counter()
			if new_load_uuids:
				# Get this rank's sequences to load
				my_new_uuids = [u for u in new_load_uuids 
							if global_candidate_info.get(u, {}).get('assigned_rank') == self.rank]
				new_load_local = self._get_local_indices_for_uuids(my_new_uuids)
				
				if new_load_local:
					new_load_global = self._local_indices_to_global_seq_ids(new_load_local)
					tokens = self._compute_two_page_buffer_tokens(new_load_local)
					
					# Allocate pages
					gpu_manager.allocate_pages_for_sequences(new_load_global, tokens)
					timing.load_alloc_ms = (time.perf_counter() - t_alloc) * 1000
					
					# Get pointers and launch async load
					t_launch = time.perf_counter()
					worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
					if worker_view is not None:
						existing_global_ids = self._local_indices_to_global_seq_ids(batch)
						gpu_manager.rebuild_page_table(new_load_global)
						k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
						active_page_counts = gpu_manager.export_active_sequence_page_counts()
						sequence_tensor = torch.tensor(new_load_global, dtype=torch.int64, device="cpu")
						
						new_async_task = worker_view.async_load_layer_paged_kv_to_device(
							sequence_ids=sequence_tensor,
							active_page_counts=active_page_counts,
							k_device_ptrs=k_ptrs,
							v_device_ptrs=v_ptrs,
						)
						
						# Restore page table to current batch
						if existing_global_ids:
							gpu_manager.rebuild_page_table(existing_global_ids)
						
						# Store tensor refs
						self._async_load_tensors = {
							'k_ptrs': k_ptrs, 'v_ptrs': v_ptrs,
							'sequence_tensor': sequence_tensor,
							'active_page_counts': active_page_counts,
						}
					timing.load_launch_ms = (time.perf_counter() - t_launch) * 1000
		
		timing.num_loaded = len(new_load_uuids)
		
		# ========== FINAL PAGE TABLE REBUILD ==========
		t0 = time.perf_counter()
		self._rebuild_page_table_for_batch(batch, gpu_manager)
		timing.rebuild_ms = (time.perf_counter() - t0) * 1000
		
		# ========== UPDATE MOE BUFFER SIZE ==========
		# Find max batch size across all ranks to minimize all-gather/all-reduce communication
		t0 = time.perf_counter()
		local_batch_size = torch.tensor([len(batch)], dtype=torch.int64, device=self.torch_device)
		dist.all_reduce(local_batch_size, op=dist.ReduceOp.MAX)
		max_batch_size = local_batch_size.item()
		
		# Update MoE layers with the actual max batch size for this page
		if max_batch_size > 0 and hasattr(self, 'parallel_manager') and self.parallel_manager is not None:
			if hasattr(self.parallel_manager, 'set_num_tokens_per_rank'):
				self.parallel_manager.set_num_tokens_per_rank(max_batch_size)
		timing.moe_buffer_update_ms = (time.perf_counter() - t0) * 1000
		
		# ========== SINGLE FINAL BARRIER ==========
		t0 = time.perf_counter()
		dist.barrier()
		timing.barrier_ms = (time.perf_counter() - t0) * 1000
		
		# ========== COLLECT STATUS COUNTS ==========
		timing.total_active = len(decode_uuids)
		timing.total_prefilled = len(self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED))
		timing.total_completed_cumulative = len(self.global_batch.get_sequences_by_status(SequenceStatus.COMPLETED))
		
		# ========== VERIFY BATCH CONSISTENCY ==========
		# CRITICAL: Ensure batch matches decode_uuids for THIS rank
		expected_local = self._get_local_indices_for_uuids(decode_uuids)
		if set(batch) != set(expected_local):
			logging.error(
				f"Rank {self.rank}: BATCH MISMATCH after boundary! "
				f"batch={sorted(batch)}, expected={sorted(expected_local)}"
			)
			batch = expected_local  # Fix it
		
		timing.total_ms = (time.perf_counter() - boundary_start) * 1000
		
		return decode_uuids, batch, new_async_task, new_load_uuids, new_load_local, new_load_global, timing

	def _finalize_async_load_minimal(
		self,
		async_task: object,
		pending_uuids: List[str],
		pending_local_indices: List[int],
		pending_global_ids: List[int],
		current_decode_uuids: List[str],
		current_batch: List[int],
		gpu_manager: GPUPagedKVCacheManager
	) -> Tuple[List[str], List[int]]:
		"""Minimal finalize without extra rebuilds - rebuild done once at end."""
		Attn_Wrapper.async_kv_load_active = False
		Attn_Wrapper.async_kv_load_task = None
		
		if hasattr(self, '_async_load_tensors'):
			self._async_load_tensors = None
		
		self._update_batch_status(pending_uuids, SequenceStatus.IN_DECODE)
		
		for local_idx in pending_local_indices:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			seq.gpu_pages_allocated = seq.get_gpu_pages_for_two_page_buffer()
			self._sequences_with_gpu_kv.add(uuid)
		
		updated_uuids = current_decode_uuids + pending_uuids
		updated_uuids.sort(key=lambda u: self.global_batch.get_sequence(u).global_idx)
		
		uuid_to_local = {}
		for idx in current_batch:
			uuid = self._local_to_uuid_map.get(idx)
			if uuid:
				uuid_to_local[uuid] = idx
		for idx in pending_local_indices:
			uuid = self._local_to_uuid_map.get(idx)
			if uuid:
				uuid_to_local[uuid] = idx
		
		updated_batch = [uuid_to_local[u] for u in updated_uuids if u in uuid_to_local]
		
		return updated_uuids, updated_batch

	def decoding_continuous_fast(
		self, 
		new_tokens: torch.Tensor, 
		decode_uuids: List[str],
		batch: List[int],
		past_key_states: Optional[torch.Tensor] = None,
		past_value_states: Optional[torch.Tensor] = None,
		scale_dict: Optional[dict] = None,
	) -> Tuple[List[str], List[int]]:
		"""
		OPTIMIZED continuous decoding with minimal collective overhead.
		
		Key optimizations:
		1. Single batched all_gather per page boundary (vs 10+ in original)
		2. Single page table rebuild per boundary (vs 4 in original)
		3. Reduced logging overhead
		4. No timing object allocation in hot path
		"""
		if "deepseek" in self.model_config.model_type:
			self.model.model._use_flash_attention_2 = True
		
		RUNTIME_ATTN_MODE = self.engine_config.Basic_Config.attn_mode
		if RUNTIME_ATTN_MODE != 3:
			self._decoding_legacy_modes(new_tokens, decode_uuids, batch, 1)
			return decode_uuids, batch
		
		# Setup
		gpu_manager = self.gpu_paged_kv_cache_manager
		if gpu_manager is None:
			gpu_manager = getattr(self.core_engine, "gpu_paged_kv_manager", None)
		
		worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		
		Attn_Wrapper.gpu_paged_kv_manager = gpu_manager
		Attn_Wrapper.host_paged_kv_worker_view = worker_view
		Attn_Wrapper.scale = scale_dict
		Attn_Wrapper.past_key_states = past_key_states
		Attn_Wrapper.past_value_states = past_value_states
		Attn_Wrapper.cur_batch = self._local_indices_to_global_seq_ids(batch) if batch else []
		
		# Async state
		self._pending_kv_append_tasks = []
		self._pending_kv_append_tensors = []
		
		pending_async_task = None
		pending_load_uuids = []
		pending_load_local = []
		pending_load_global = []
		
		# Validation
		for local_idx in batch:
			uuid = self._local_to_uuid_map.get(local_idx)
			if uuid and uuid not in self._sequences_with_gpu_kv:
				self._sequences_with_gpu_kv.add(uuid)
		
		iteration = 0
		last_boundary = 0
		total_boundary_ms = 0.0
		total_forward_ms = 0.0
		num_boundaries = 0
		initial_batch_size = len(decode_uuids)
		
		# Main decode loop
		while decode_uuids:
			iteration += 1
			
			# Page boundary check
			if iteration - last_boundary >= self.PAGE_SIZE:
				last_boundary = iteration
				
				(decode_uuids, batch, 
				 pending_async_task, pending_load_uuids, 
				 pending_load_local, pending_load_global, 
				 timing) = self._page_boundary_fast(
					decode_uuids, batch, gpu_manager,
					pending_async_task, pending_load_uuids,
					pending_load_local, pending_load_global
				)
				
				total_boundary_ms += timing.total_ms
				num_boundaries += 1
				
				# Detailed logging at every boundary (only rank 0)
				if self.rank == 0:
					# Get status counts
					num_onhold = len(self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD))
					num_prefilled = timing.total_prefilled
					num_completed_total = timing.total_completed_cumulative
					num_in_decode = timing.total_active
					
					logging.info(
						f"[Boundary {num_boundaries}] "
						f"iter={iteration}, "
						f"total={timing.total_ms:.1f}ms | "
						f"wait_kv={timing.wait_kv_append_ms:.1f}({timing.num_kv_append_tasks}), "
						f"wait_async={timing.wait_async_load_ms:.1f}, "
						f"finalize={timing.finalize_load_ms:.1f}, "
						f"gather={timing.gather_ms:.1f}, "
						f"proc={timing.process_ms:.1f}, "
						f"ext={timing.extension_ms:.1f}, "
						f"load_sel={timing.load_select_ms:.1f}, "
						f"load_alloc={timing.load_alloc_ms:.1f}, "
						f"load_launch={timing.load_launch_ms:.1f}, "
						f"rebuild={timing.rebuild_ms:.1f}, "
						f"moe_buf={timing.moe_buffer_update_ms:.1f}, "
						f"barrier={timing.barrier_ms:.1f}ms | "
						f"STATUS: active={num_in_decode}, completed={num_completed_total}/{initial_batch_size}, "
						f"onhold={num_onhold}, prefilled={num_prefilled}, "
						f"this_boundary: +completed={timing.num_completed}, +loaded={timing.num_loaded}, +onhold={timing.num_onhold}"
					)
				
				if not decode_uuids:
					# Check for pending loads
					if pending_load_uuids:
						if pending_async_task is not None:
							pending_async_task.wait()
							torch.cuda.synchronize(self.torch_device)
						dist.barrier()
						
						decode_uuids, batch = self._finalize_async_load_minimal(
							pending_async_task, pending_load_uuids,
							pending_load_local, pending_load_global,
							decode_uuids, batch, gpu_manager
						)
						self._rebuild_page_table_for_batch(batch, gpu_manager)
						
						if batch:
							new_tokens = self._rebuild_input_tokens(batch)
						
						pending_async_task = None
						pending_load_uuids = []
						pending_load_local = []
						pending_load_global = []
						
						if decode_uuids:
							continue
					break
				
				new_tokens = self._rebuild_input_tokens(batch)
			
			# Forward pass
			forward_start = time.perf_counter()
			
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
					
					max_ctx = max(cache_seqlens)
					padded_masks = []
					for mask in attention_masks:
						if mask.shape[1] < max_ctx:
							pad = torch.zeros((1, max_ctx - mask.shape[1]), dtype=mask.dtype, device=mask.device)
							mask = torch.cat([mask, pad], dim=1)
						padded_masks.append(mask)
					
					attention_mask = torch.cat(padded_masks, dim=0).to(self.torch_device)
					
					Attn_Wrapper.attention_mask = attention_mask
					Attn_Wrapper.position_ids = (attention_mask.sum(-1) - 1).unsqueeze(-1)
					Attn_Wrapper.cache_seqlens = torch.tensor(cache_seqlens, dtype=torch.int32, device=self.torch_device)
					Attn_Wrapper.max_seqlen = max_ctx
					
					if new_tokens.shape[0] != len(batch):
						new_tokens = self._rebuild_input_tokens(batch)
				else:
					Attn_Wrapper.attention_mask = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
					Attn_Wrapper.position_ids = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
					Attn_Wrapper.cache_seqlens = torch.zeros((0,), dtype=torch.int32, device=self.torch_device)
					Attn_Wrapper.max_seqlen = 0
					Attn_Wrapper.cur_batch = []
					new_tokens = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
				
				if batch:
					Attn_Wrapper.cur_batch = self._local_indices_to_global_seq_ids(batch)
				
				# KV append callback
				current_batch = list(batch)
				def kv_append_callback(layer_idx: int, k_tensor: torch.Tensor):
					self._append_decode_kv_to_host_async(layer_idx, current_batch, k_tensor)
				Attn_Wrapper.kv_append_callback = kv_append_callback
				
				# Forward
				outputs = self.model(new_tokens, attention_mask=Attn_Wrapper.attention_mask, use_cache=False)
				new_tokens = torch.argmax(outputs.logits, dim=-1).view(-1, 1)
				
				# Update sequences
				for i, local_idx in enumerate(batch):
					uuid = self._local_to_uuid_map[local_idx]
					seq = self.global_batch.get_sequence(uuid)
					
					if self._is_sequence_completed(seq):
						continue
					
					decode_pos = seq.decoded_length
					self.query_book[local_idx].decoded_tokens[:, decode_pos] = new_tokens[i].cpu()
					
					attn_mask = self.query_book[local_idx].encoded["attention_mask"][0]
					next_pos = seq.current_context_length
					if next_pos < attn_mask.shape[0]:
						attn_mask[next_pos] = 1
					
					seq.decoded_length += 1
					seq.current_context_length += 1
					
					if self._should_stop_at_eos(new_tokens[i].item()):
						seq.eos_reached = True
					
					if seq.decoded_length >= self.max_decoding_length:
						seq.eos_reached = True
			
			total_forward_ms += (time.perf_counter() - forward_start) * 1000
		
		# Cleanup
		self._wait_pending_kv_append_tasks()
		if pending_async_task is not None:
			pending_async_task.wait()
			torch.cuda.synchronize(self.torch_device)
		
		Attn_Wrapper.kv_append_callback = None
		Attn_Wrapper.scale = None
		Attn_Wrapper.past_key_states = None
		Attn_Wrapper.past_value_states = None
		Attn_Wrapper.gpu_paged_kv_manager = None
		Attn_Wrapper.host_paged_kv_worker_view = None
		Attn_Wrapper.cur_batch = None
		
		# Summary
		if self.rank == 0 and num_boundaries > 0:
			avg_forward = total_forward_ms / iteration if iteration > 0 else 0
			avg_boundary = total_boundary_ms / num_boundaries
			logging.info(
				f"\n{'='*50}\n"
				f"FAST DECODE SUMMARY (Rank 0)\n"
				f"{'='*50}\n"
				f"Iterations: {iteration}, Boundaries: {num_boundaries}\n"
				f"Avg forward: {avg_forward:.2f}ms\n"
				f"Avg boundary: {avg_boundary:.2f}ms\n"
				f"Boundary overhead/token: {avg_boundary / self.PAGE_SIZE:.3f}ms\n"
				f"{'='*50}"
			)
		
		return decode_uuids, batch

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
		Continuous decoding with async optimizations and detailed timing.
		
		Key Optimizations:
		1. KV append: fire-and-forget, wait only at page boundaries
		2. Load new sequences: async, overlap with next page's decoding
		3. Consolidated page table rebuilds
		4. Reduced barriers
		
		Timing instrumentation added for performance analysis.
		"""
		if "deepseek" in self.model_config.model_type:
			self.model.model._use_flash_attention_2 = True
		
		RUNTIME_ATTN_MODE = self.engine_config.Basic_Config.attn_mode
		
		if RUNTIME_ATTN_MODE != 3:
			self._decoding_legacy_modes(new_tokens, decode_uuids, batch, 1)
			return decode_uuids, batch
		
		# =========================================
		# SETUP
		# =========================================
		gpu_manager = self.gpu_paged_kv_cache_manager
		if gpu_manager is None:
			gpu_manager = getattr(self.core_engine, "gpu_paged_kv_manager", None)
		
		worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		
		Attn_Wrapper.gpu_paged_kv_manager = gpu_manager
		Attn_Wrapper.host_paged_kv_worker_view = worker_view
		Attn_Wrapper.scale = scale_dict
		Attn_Wrapper.past_key_states = past_key_states
		Attn_Wrapper.past_value_states = past_value_states
		Attn_Wrapper.cur_batch = self._local_indices_to_global_seq_ids(batch) if batch else []
		
		# =========================================
		# ASYNC STATE TRACKING
		# =========================================
		self._pending_kv_append_tasks: List = []
		self._pending_kv_append_tensors: List = []  # Keep tensor refs alive during async ops
		
		pending_async_load_task = None
		pending_load_uuids: List[str] = []
		pending_load_local_indices: List[int] = []
		pending_load_global_ids: List[int] = []
		
		# =========================================
		# VALIDATION: Ensure tracking consistency
		# =========================================
		for local_idx in batch:
			uuid = self._local_to_uuid_map.get(local_idx)
			if uuid and uuid not in self._sequences_with_gpu_kv:
				# This can happen if config_decoding didn't properly track
				logging.warning(
					f"Rank {self.rank}: Sequence {uuid} in batch but not in "
					f"_sequences_with_gpu_kv. Adding to tracking."
				)
				self._sequences_with_gpu_kv.add(uuid)
		
		# Timing accumulator
		boundary_timings: List[BoundaryTimingStats] = []
		forward_times_ms: List[float] = []
		
		iteration = 0
		last_page_boundary_check = 0
		
		# =========================================
		# MAIN DECODE LOOP
		# =========================================
		while decode_uuids:
			iteration += 1
			
			if self.rank == 0 and iteration % 100 == 0:
				logging.info(
					f"Decode iteration {iteration}, {len(decode_uuids)} active, "
					f"{len(batch)} local"
				)
			
			# =============================================================
			# PAGE BOUNDARY LOGIC (with detailed timing)
			# =============================================================
			if iteration - last_page_boundary_check >= self.PAGE_SIZE:
				last_page_boundary_check = iteration
				
				timing = BoundaryTimingStats()
				boundary_start = time.perf_counter()
				
				# --------------------------------------------------------
				# PHASE 1: SYNC ALL PENDING OPERATIONS
				# --------------------------------------------------------
				
				# 1a. Wait for KV append tasks (local operation)
				t0 = time.perf_counter()
				self._wait_pending_kv_append_tasks()
				timing.wait_kv_append_ms = (time.perf_counter() - t0) * 1000
				
				# 1a-bis. FIX Bug 5-6: Sync sequence metadata across all ranks
				# This ensures all ranks have consistent view of decoded_length, 
				# current_context_length, etc. before making any decisions
				self._sync_sequence_metadata(decode_uuids)
				
				# 1b. Integrate PREVIOUS async load
				# CRITICAL: Check pending_load_uuids (globally consistent) not pending_async_load_task
				t0 = time.perf_counter()
				if pending_load_uuids:  # ALL ranks have identical value
					logging.debug(
						f"Rank {self.rank}: Phase 1b - Integrating {len(pending_load_uuids)} pending sequences"
					)
					
					# Wait for async task if THIS rank has one
					if pending_async_load_task is not None:
						pending_async_load_task.wait()
						# Sync GPU to ensure async DMA completes before using the data
						torch.cuda.synchronize(self.torch_device)
					
					# COLLECTIVE: All ranks synchronize here
					dist.barrier()
					
					# Finalize integration (updates tracking, merges batches)
					decode_uuids, batch = self._finalize_async_load(
						pending_async_load_task,  # Can be None for ranks with no local sequences
						pending_load_uuids,
						pending_load_local_indices,
						pending_load_global_ids,
						decode_uuids,
						batch,
						gpu_manager
					)
					
					# Clear pending state
					pending_async_load_task = None
					pending_load_uuids = []
					pending_load_local_indices = []
					pending_load_global_ids = []
				
				timing.finalize_async_load_ms = (time.perf_counter() - t0) * 1000
				
				# 1c. Rebuild page table ONCE after integration
				t0 = time.perf_counter()
				self._rebuild_page_table_for_batch(batch, gpu_manager)
				timing.rebuild_after_integration_ms = (time.perf_counter() - t0) * 1000
				
				# 1d. COLLECTIVE: Ensure all ranks synchronized
				t0 = time.perf_counter()
				dist.barrier()
				timing.barrier_after_sync_ms = (time.perf_counter() - t0) * 1000
				
				# --------------------------------------------------------
				# PHASE 2: EVICTION (Completed Sequences)
				# --------------------------------------------------------
				
				# 2a. Sync completion status
				t0 = time.perf_counter()
				decode_uuids, completed_uuids = self._sync_completion_status_at_boundary(decode_uuids)
				timing.sync_completion_ms = (time.perf_counter() - t0) * 1000
				timing.num_completed = len(completed_uuids)
				
				# 2b. Evict completed sequences
				t0 = time.perf_counter()
				need_rebuild_after_eviction = False
				if completed_uuids:
					self._update_batch_status(completed_uuids, SequenceStatus.COMPLETED)
					my_completed = [u for u in completed_uuids if u in self._uuid_to_local_map]
					if my_completed:
						# FIX FLAW 2: Release BOTH GPU and Host KV pages for completed sequences
						my_completed_local_indices = self._get_local_indices_for_uuids(my_completed)
						self._release_gpu_kv_pages(my_completed_local_indices)
						self._release_host_kv_pages_for_batch(my_completed)
						need_rebuild_after_eviction = True
					batch = self._get_local_indices_for_uuids(decode_uuids)
				else:
					batch = self._get_local_indices_for_uuids(decode_uuids)
				timing.release_completed_ms = (time.perf_counter() - t0) * 1000
				
				# 2c. Rebuild only if we evicted
				t0 = time.perf_counter()
				if need_rebuild_after_eviction:
					self._rebuild_page_table_for_batch(batch, gpu_manager)
				timing.rebuild_after_eviction_ms = (time.perf_counter() - t0) * 1000
				
				# --------------------------------------------------------
				# PHASE 3: EXTENSION (Grow Page Buffers)
				# --------------------------------------------------------
				t0 = time.perf_counter()
				decode_uuids, batch, onhold_uuids = self._check_and_extend_page_buffer(
					decode_uuids, batch
				)
				timing.extend_page_buffer_ms = (time.perf_counter() - t0) * 1000
				timing.num_onhold = len(onhold_uuids)
				
				# Rebuild only if we put sequences ON_HOLD
				t0 = time.perf_counter()
				# if onhold_uuids:
				# 	self._rebuild_page_table_for_batch(batch, gpu_manager)
				self._rebuild_page_table_for_batch(batch, gpu_manager)
				timing.rebuild_after_extension_ms = (time.perf_counter() - t0) * 1000
				
				# --------------------------------------------------------
				# PHASE 4: LAUNCH ASYNC LOAD FOR NEW SEQUENCES
				# --------------------------------------------------------
				t0 = time.perf_counter()
				(pending_async_load_task, 
				pending_load_uuids, 
				pending_load_local_indices,
				pending_load_global_ids,
				load_timing) = self._launch_async_load_new_sequences_timed(
					decode_uuids, batch, gpu_manager
				)
				launch_total_ms = (time.perf_counter() - t0) * 1000
				
				# Unpack sub-timings
				timing.allgather_free_pages_ms = load_timing.get('allgather_ms', 0)
				timing.select_candidates_ms = load_timing.get('select_ms', 0)
				timing.allocate_pages_ms = load_timing.get('allocate_ms', 0)
				timing.launch_async_load_ms = load_timing.get('launch_ms', 0)
				timing.restore_page_table_ms = load_timing.get('restore_ms', 0)
				timing.num_loaded = len(pending_load_uuids)
				
				# --------------------------------------------------------
				# PREPARE FOR NEXT PAGE
				# --------------------------------------------------------
				new_tokens = self._rebuild_input_tokens(batch)
				
				# NOTE: No cuda sync needed here - barrier provides synchronization
				# and page table rebuilds are immediate GPU operations
				
				t0 = time.perf_counter()
				dist.barrier()
				timing.barrier_final_ms = (time.perf_counter() - t0) * 1000
				
				timing.total_boundary_ms = (time.perf_counter() - boundary_start) * 1000
				boundary_timings.append(timing)
				
				# Log timing for this boundary
				if self.rank == 0:
					logging.info(f"Page boundary {len(boundary_timings)}:\n{timing}")
								
				if not decode_uuids:
					if pending_load_uuids:  # Global check - all ranks enter
						if pending_async_load_task is not None:
							pending_async_load_task.wait()
							# NOTE: Sync only when there was actual async work
							torch.cuda.synchronize(self.torch_device)
						dist.barrier()
						
						# MOVE OUTSIDE the task check:
						decode_uuids, batch = self._finalize_async_load(
							pending_async_load_task,  # Can be None - handled correctly inside
							pending_load_uuids,
							pending_load_local_indices,
							pending_load_global_ids,
							decode_uuids,
							batch,
							gpu_manager
						)
						
						pending_async_load_task = None
						pending_load_uuids = []
						pending_load_local_indices = []
						pending_load_global_ids = []
						
						self._rebuild_page_table_for_batch(batch, gpu_manager)
						# NOTE: No sync needed - page table rebuild is immediate
						
						if batch:
							new_tokens = self._rebuild_input_tokens(batch)
						
						if decode_uuids:
							logging.debug(f"Rank {self.rank}: Resuming decode with {len(decode_uuids)} sequences")
							continue
					
					break
			
			# =============================================================
			# FORWARD PASS (with timing)
			# =============================================================
			forward_start = time.perf_counter()
			
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
					
					max_ctx = max(cache_seqlens)
					padded_masks = []
					for mask in attention_masks:
						if mask.shape[1] < max_ctx:
							pad = torch.zeros(
								(1, max_ctx - mask.shape[1]), 
								dtype=mask.dtype, 
								device=mask.device
							)
							mask = torch.cat([mask, pad], dim=1)
						padded_masks.append(mask)
					
					attention_mask = torch.cat(padded_masks, dim=0).to(self.torch_device)
					
					Attn_Wrapper.attention_mask = attention_mask
					Attn_Wrapper.position_ids = (attention_mask.sum(-1) - 1).unsqueeze(-1)
					Attn_Wrapper.cache_seqlens = torch.tensor(
						cache_seqlens, dtype=torch.int32, device=self.torch_device
					)
					Attn_Wrapper.max_seqlen = max_ctx
					
					if new_tokens.shape[0] != len(batch):
						new_tokens = self._rebuild_input_tokens(batch)
				else:
					Attn_Wrapper.attention_mask = torch.zeros(
						(0, 1), dtype=torch.int64, device=self.torch_device
					)
					Attn_Wrapper.position_ids = torch.zeros(
						(0, 1), dtype=torch.int64, device=self.torch_device
					)
					Attn_Wrapper.cache_seqlens = torch.zeros(
						(0,), dtype=torch.int32, device=self.torch_device
					)
					Attn_Wrapper.max_seqlen = 0
					Attn_Wrapper.cur_batch = []
					new_tokens = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
				
				# FIX Bug 3 & 4: Update Attn_Wrapper.cur_batch and capture batch for callback
				# AFTER all batch validation is complete
				if batch:
					Attn_Wrapper.cur_batch = self._local_indices_to_global_seq_ids(batch)
				
				# KV append callback - FIRE AND FORGET
				# Capture batch by value AFTER all modifications are done
				current_batch = list(batch)
				def kv_append_callback(layer_idx: int, k_tensor: torch.Tensor):
					self._append_decode_kv_to_host_async(layer_idx, current_batch, k_tensor)
					# Returns None - NO BLOCKING WAIT
				
				Attn_Wrapper.kv_append_callback = kv_append_callback
				
				# ============ PRE-FORWARD VALIDATION ============
				if len(batch) > 0:
					# Validate tensor shapes match batch size
					expected_bsz = len(batch)
					actual_token_bsz = new_tokens.shape[0]
					actual_mask_bsz = Attn_Wrapper.attention_mask.shape[0]
					actual_pos_bsz = Attn_Wrapper.position_ids.shape[0]
					actual_seqlen_bsz = Attn_Wrapper.cache_seqlens.shape[0]
					
					if actual_token_bsz != expected_bsz:
						logging.error(
							f"Rank {self.rank}: BATCH SIZE MISMATCH! "
							f"expected={expected_bsz}, new_tokens.shape[0]={actual_token_bsz}"
						)
					if actual_mask_bsz != expected_bsz:
						logging.error(
							f"Rank {self.rank}: ATTENTION MASK MISMATCH! "
							f"expected={expected_bsz}, attention_mask.shape[0]={actual_mask_bsz}"
						)
					if actual_pos_bsz != expected_bsz:
						logging.error(
							f"Rank {self.rank}: POSITION_IDS MISMATCH! "
							f"expected={expected_bsz}, position_ids.shape[0]={actual_pos_bsz}"
						)
					if actual_seqlen_bsz != expected_bsz:
						logging.error(
							f"Rank {self.rank}: CACHE_SEQLENS MISMATCH! "
							f"expected={expected_bsz}, cache_seqlens.shape[0]={actual_seqlen_bsz}"
						)
					
					# Log summary for debugging
					logging.debug(
						f"Rank {self.rank}: Forward pass: batch={expected_bsz}, "
						f"new_tokens={new_tokens.shape}, "
						f"attention_mask={Attn_Wrapper.attention_mask.shape}, "
						f"position_ids={Attn_Wrapper.position_ids.shape}, "
						f"cache_seqlens={Attn_Wrapper.cache_seqlens.shape}, "
						f"max_seqlen={Attn_Wrapper.max_seqlen}"
					)
				
				# Forward pass
				outputs = self.model(
					new_tokens,
					attention_mask=Attn_Wrapper.attention_mask,
					use_cache=False,
				)
				
				# Update sequences
				new_tokens = torch.argmax(outputs.logits, dim=-1).view(-1, 1)
				
				for i, local_idx in enumerate(batch):
					uuid = self._local_to_uuid_map[local_idx]
					seq = self.global_batch.get_sequence(uuid)
					
					# FIXED: Check if sequence is already completed
					# Use _is_sequence_completed to properly respect ignore_eos
					if self._is_sequence_completed(seq):
						continue
					
					decode_pos = seq.decoded_length
					
					# Store token
					self.query_book[local_idx].decoded_tokens[:, decode_pos] = new_tokens[i].cpu()
					
					# Update attention mask
					attn_mask = self.query_book[local_idx].encoded["attention_mask"][0]
					next_pos = seq.current_context_length
					if next_pos < attn_mask.shape[0]:
						attn_mask[next_pos] = 1
					
					# Update sequence state
					seq.decoded_length += 1
					seq.current_context_length += 1
					
					# FIXED: Check EOS respecting ignore_eos flag
					# Only set eos_reached if we should stop at this EOS
					if self._should_stop_at_eos(new_tokens[i].item()):
						seq.eos_reached = True
						logging.debug(
							f"Rank {self.rank}: Sequence {uuid} hit EOS at position {decode_pos}"
						)
					
					# Check max length (always enforced)
					if seq.decoded_length >= self.max_decoding_length:
						seq.eos_reached = True  # Mark as done
						logging.debug(
							f"Rank {self.rank}: Sequence {uuid} reached max_decoding_length"
						)
			
			forward_times_ms.append((time.perf_counter() - forward_start) * 1000)
		
		# =========================================
		# FINAL CLEANUP
		# =========================================
		self._wait_pending_kv_append_tasks()
		
		if pending_async_load_task is not None:
			pending_async_load_task.wait()
			torch.cuda.synchronize(self.torch_device)
		
		Attn_Wrapper.kv_append_callback = None
		Attn_Wrapper.scale = None
		Attn_Wrapper.past_key_states = None
		Attn_Wrapper.past_value_states = None
		Attn_Wrapper.gpu_paged_kv_manager = None
		Attn_Wrapper.host_paged_kv_worker_view = None
		Attn_Wrapper.cur_batch = None
		
		# =========================================
		# TIMING SUMMARY
		# =========================================
		if self.rank == 0 and boundary_timings:
			avg_forward = sum(forward_times_ms) / len(forward_times_ms) if forward_times_ms else 0
			avg_boundary = sum(t.total_boundary_ms for t in boundary_timings) / len(boundary_timings)
			
			# Aggregate sub-timings
			avg_wait_kv = sum(t.wait_kv_append_ms for t in boundary_timings) / len(boundary_timings)
			avg_finalize = sum(t.finalize_async_load_ms for t in boundary_timings) / len(boundary_timings)
			avg_sync_completion = sum(t.sync_completion_ms for t in boundary_timings) / len(boundary_timings)
			avg_allgather = sum(t.allgather_free_pages_ms for t in boundary_timings) / len(boundary_timings)
			avg_barriers = sum(t.barrier_after_sync_ms + t.barrier_final_ms for t in boundary_timings) / len(boundary_timings)
			avg_rebuilds = sum(
				t.rebuild_after_integration_ms + t.rebuild_after_eviction_ms + 
				t.rebuild_after_extension_ms + t.restore_page_table_ms 
				for t in boundary_timings
			) / len(boundary_timings)
			
			logging.info(
				f"\n{'='*60}\n"
				f"TIMING SUMMARY (Rank 0)\n"
				f"{'='*60}\n"
				f"Total iterations: {iteration}\n"
				f"Total boundaries: {len(boundary_timings)}\n"
				f"Avg forward pass: {avg_forward:.2f}ms\n"
				f"Avg boundary total: {avg_boundary:.2f}ms\n"
				f"  - wait_kv_append: {avg_wait_kv:.2f}ms\n"
				f"  - finalize_load: {avg_finalize:.2f}ms\n"
				f"  - sync_completion: {avg_sync_completion:.2f}ms\n"
				f"  - allgather: {avg_allgather:.2f}ms\n"
				f"  - barriers: {avg_barriers:.2f}ms\n"
				f"  - page_table_rebuilds: {avg_rebuilds:.2f}ms\n"
				f"Boundary overhead per token: {avg_boundary / self.PAGE_SIZE:.2f}ms\n"
				f"{'='*60}"
			)
		
		logging.info(f"Rank {self.rank}: decoding_continuous completed after {iteration} iterations")
		
		return decode_uuids, batch

	def _wait_pending_kv_append_tasks(self) -> int:
		"""
		Wait for all pending KV append tasks at page boundary.
		Returns the number of tasks that were waited for.
		"""
		if not hasattr(self, '_pending_kv_append_tasks'):
			return 0
		
		num_tasks = len(self._pending_kv_append_tasks)
		for task in self._pending_kv_append_tasks:
			if task is not None:
				task.wait()
		
		self._pending_kv_append_tasks.clear()
		
		# CRITICAL: Clear tensor references AFTER tasks complete
		# Tensors can now be safely garbage collected / memory reused
		if hasattr(self, '_pending_kv_append_tensors'):
			self._pending_kv_append_tensors.clear()
		
		return num_tasks

	def _rebuild_page_table_for_batch(
		self,
		batch: List[int],
		gpu_manager: GPUPagedKVCacheManager
	) -> None:
		"""Consolidated page table rebuild - single place to rebuild."""
		if gpu_manager is None or not gpu_manager.is_initialized:
			Attn_Wrapper.cur_batch = []
			return
		
		if not batch:
			Attn_Wrapper.cur_batch = []
			return
		
		global_ids = self._local_indices_to_global_seq_ids(batch)
		gpu_manager.rebuild_page_table(global_ids)
		Attn_Wrapper.cur_batch = global_ids

	def _append_decode_kv_to_host_async(
		self,
		layer_idx: int,
		batch: List[int],
		k_tensor: torch.Tensor,
	) -> None:  # Returns None, not the task
		"""
		Async append - adds task to pending list, does NOT wait.
		
		CRITICAL: Must keep tensor references alive until async operation completes!
		"""
		if not batch:
			return
		
		worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		if worker_view is None:
			return
		
		sequence_ids = []
		sequence_lengths = []
		
		for local_idx in batch:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			sequence_ids.append(seq.global_idx)
			sequence_lengths.append(seq.current_context_length - 1)
		
		if k_tensor.dim() == 3:
			k_tensor = k_tensor.unsqueeze(2)
		
		task = worker_view.async_append_decode_kv_to_host(
			layer_idx=layer_idx,
			sequence_ids=sequence_ids,
			k_tensor=k_tensor,
			v_tensor=None,
			sequence_lengths=sequence_lengths,
		)
		
		# CRITICAL FIX: Store tensor reference alongside task to prevent GC
		# PyTorch's CUDA caching allocator can reuse memory if tensor is dereferenced
		# while async operation is still reading from it!
		if not hasattr(self, '_pending_kv_append_tensors'):
			self._pending_kv_append_tensors = []
		self._pending_kv_append_tensors.append(k_tensor)
		
		# Add to pending list - will be waited at page boundary
		self._pending_kv_append_tasks.append(task)
		# NO return, NO wait

	def _launch_async_load_new_sequences(
		self,
		current_decode_uuids: List[str],
		current_batch: List[int],
		gpu_manager: GPUPagedKVCacheManager
	) -> Tuple[Optional[object], List[str], List[int], List[int]]:
		"""
		Launch async load for new sequences using TWO-PAGE BUFFER strategy.
		
		FIXED: Uses two-page buffer tokens, not full context.
		FIXED: Adds pre-allocation guard.
		"""
		if gpu_manager is None or not gpu_manager.is_initialized:
			return None, [], [], []
		
		# Step 1: All-gather free GPU pages
		local_free = gpu_manager.get_stats().num_free_pages
		free_tensor = torch.tensor([local_free], dtype=torch.int64, device=self.torch_device)
		gathered = [torch.zeros_like(free_tensor) for _ in range(self.world_size)]
		dist.all_gather(gathered, free_tensor)
		per_rank_free = [int(t.item()) for t in gathered]
		
		# Step 2: Get candidates
		prefilled = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		onhold = self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
		candidates = prefilled + onhold
		candidates.sort(key=lambda u: self.global_batch.get_sequence(u).global_idx)
		
		if not candidates:
			return None, [], [], []
		
		# Step 3: Greedy selection using TWO-PAGE BUFFER pages
		rank_pages_used = [0] * self.world_size
		new_uuids = []
		
		for uuid in candidates:
			seq = self.global_batch.get_sequence(uuid)
			assigned_rank = seq.assigned_rank
			# FIXED: Use two-page buffer calculation
			req_pages = seq.get_gpu_pages_for_two_page_buffer()
			
			if rank_pages_used[assigned_rank] + req_pages <= per_rank_free[assigned_rank]:
				new_uuids.append(uuid)
				rank_pages_used[assigned_rank] += req_pages
		
		if not new_uuids:
			return None, [], [], []
		
		# Step 4: Get THIS RANK's sequences
		my_new_uuids = [u for u in new_uuids 
					if self.global_batch.get_sequence(u).assigned_rank == self.rank]
		new_local_indices = self._get_local_indices_for_uuids(my_new_uuids)
		
		if not new_local_indices:
			return None, new_uuids, [], []
		
		new_global_ids = self._local_indices_to_global_seq_ids(new_local_indices)
		
		# FIXED: Use two-page buffer tokens, NOT full context
		tokens = self._compute_two_page_buffer_tokens(new_local_indices)
		
		# FIXED: Guard before allocation
		total_pages_needed = sum(t // self.PAGE_SIZE for t in tokens)
		current_free = gpu_manager.get_stats().num_free_pages
		if total_pages_needed > current_free:
			logging.warning(
				f"Rank {self.rank}: Skipping async load - need {total_pages_needed} pages, "
				f"only {current_free} free"
			)
			return None, new_uuids, [], []
		
		# Step 5: Allocate GPU pages
		gpu_manager.allocate_pages_for_sequences(new_global_ids, tokens)
		
		# Step 6: Temp rebuild for pointers
		gpu_manager.rebuild_page_table(new_global_ids)
		k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
		active_page_counts = gpu_manager.export_active_sequence_page_counts()
		
		# Step 7: Launch async load
		worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		if worker_view is None:
			existing_global_ids = self._local_indices_to_global_seq_ids(current_batch)
			if existing_global_ids:
				gpu_manager.rebuild_page_table(existing_global_ids)
			return None, new_uuids, new_local_indices, new_global_ids
		
		sequence_tensor = torch.tensor(new_global_ids, dtype=torch.int64, device="cpu")
		async_task = worker_view.async_load_layer_paged_kv_to_device(
			sequence_ids=sequence_tensor,
			active_page_counts=active_page_counts,
			k_device_ptrs=k_ptrs,
			v_device_ptrs=v_ptrs,
		)
		
		# Step 8: Restore page table
		existing_global_ids = self._local_indices_to_global_seq_ids(current_batch)
		if existing_global_ids:
			gpu_manager.rebuild_page_table(existing_global_ids)
		
		return async_task, new_uuids, new_local_indices, new_global_ids

	def _launch_async_load_new_sequences_timed(
		self,
		current_decode_uuids: List[str],
		current_batch: List[int],
		gpu_manager: GPUPagedKVCacheManager
	) -> Tuple[Optional[object], List[str], List[int], List[int], Dict[str, float]]:
		"""
		Launch async load with detailed timing.
		
		CRITICAL FIX: All-gather sequence state before selection to ensure
		all ranks compute identical new_uuids.
		"""
		timing = {}
		
		if gpu_manager is None or not gpu_manager.is_initialized:
			return None, [], [], [], timing
		
		worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		
		# ============ PHASE 1: Gather global state (COLLECTIVE) ============
		t0 = time.perf_counter()
		local_free = gpu_manager.get_stats().num_free_pages
		free_tensor = torch.tensor([local_free], dtype=torch.int64, device=self.torch_device)
		gathered = [torch.zeros_like(free_tensor) for _ in range(self.world_size)]
		dist.all_gather(gathered, free_tensor)
		per_rank_free = [int(t.item()) for t in gathered]
		timing['allgather_ms'] = (time.perf_counter() - t0) * 1000
		
		# ============ PHASE 2: Get candidates and gather their state ============
		t0 = time.perf_counter()
		prefilled = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		onhold = self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
		candidates = prefilled + onhold
		candidates.sort(key=lambda u: self.global_batch.get_sequence(u).global_idx)
		
		if not candidates:
			timing['select_ms'] = (time.perf_counter() - t0) * 1000
			return None, [], [], [], timing
		
		# ============ PHASE 2b: ALL-GATHER SEQUENCE STATE (CRITICAL FIX) ============
		# Each rank reports state for sequences it owns
		local_seq_state = {}
		for uuid in candidates:
			if uuid in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				local_seq_state[uuid] = seq.get_gpu_pages_for_two_page_buffer()
		
		all_seq_state = [None] * self.world_size
		dist.all_gather_object(all_seq_state, local_seq_state)
		
		# Merge: each uuid appears exactly once (owned by one rank)
		global_pages_needed = {}
		for rank_state in all_seq_state:
			if rank_state:
				global_pages_needed.update(rank_state)
		
		# ============ PHASE 3: Deterministic selection using GATHERED state ============
		rank_pages_used = [0] * self.world_size
		new_uuids = []
		
		for uuid in candidates:
			seq = self.global_batch.get_sequence(uuid)
			assigned_rank = seq.assigned_rank
			
			# CRITICAL: Use gathered page count, not local (potentially stale) value
			req_pages = global_pages_needed.get(uuid)
			if req_pages is None:
				logging.warning(f"Rank {self.rank}: No page count for {uuid}, skipping")
				continue
			
			if rank_pages_used[assigned_rank] + req_pages <= per_rank_free[assigned_rank]:
				new_uuids.append(uuid)
				rank_pages_used[assigned_rank] += req_pages
		
		timing['select_ms'] = (time.perf_counter() - t0) * 1000

		if not new_uuids:
			return None, [], [], [], timing

		# ============ PHASE 3b: Get THIS RANK's sequences ============
		t0 = time.perf_counter()

		my_new_uuids = [u for u in new_uuids 
					if self.global_batch.get_sequence(u).assigned_rank == self.rank]
		new_local_indices = self._get_local_indices_for_uuids(my_new_uuids)

		if new_local_indices:
			new_global_ids = self._local_indices_to_global_seq_ids(new_local_indices)
			tokens = self._compute_two_page_buffer_tokens(new_local_indices)
			total_pages_needed = sum(t // self.PAGE_SIZE for t in tokens)
			current_free = gpu_manager.get_stats().num_free_pages
			local_can_allocate = 1 if total_pages_needed <= current_free else 0
		else:
			new_global_ids = []
			tokens = []
			total_pages_needed = 0
			current_free = 0
			local_can_allocate = 1  # No allocation needed = success
			
		# ============ PHASE 4: Global consensus on allocation (COLLECTIVE) ============
		# CRITICAL: ALL ranks must participate BEFORE any early return
		can_allocate_tensor = torch.tensor([local_can_allocate], dtype=torch.int32, device=self.torch_device)
		dist.all_reduce(can_allocate_tensor, op=dist.ReduceOp.MIN)
		
		if can_allocate_tensor.item() == 0:
			# At least one rank failed - ALL ranks abort with empty lists
			logging.warning(
				f"Rank {self.rank}: Global allocation consensus failed "
				f"(local: need {total_pages_needed}, have {current_free}). "
				f"All ranks skipping async load to maintain consistency."
			)
			timing['allocate_ms'] = (time.perf_counter() - t0) * 1000
			# CRITICAL: Return empty new_uuids so ALL ranks have consistent state
			return None, [], [], [], timing
		
		# ============ PHASE 5: Handle ranks with no local sequences ============
		# Consensus passed - safe to return early for ranks with no work
		if not new_local_indices:
			timing['allocate_ms'] = (time.perf_counter() - t0) * 1000
			# Return new_uuids (non-empty) for status update consistency
			# This rank will enter `if pending_load_uuids:` block in caller
			return None, new_uuids, [], [], timing
		
		# ============ PHASE 6: Allocate GPU pages ============
		gpu_manager.allocate_pages_for_sequences(new_global_ids, tokens)
		timing['allocate_ms'] = (time.perf_counter() - t0) * 1000
		
		# ============ PHASE 7: Prepare for async load ============
		t0 = time.perf_counter()
		
		# Capture existing batch for later restoration
		existing_global_ids = self._local_indices_to_global_seq_ids(current_batch)
		
		# Rebuild page table for NEW sequences to get their pointers
		gpu_manager.rebuild_page_table(new_global_ids)
		k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
		active_page_counts = gpu_manager.export_active_sequence_page_counts()
		sequence_tensor = torch.tensor(new_global_ids, dtype=torch.int64, device="cpu")
		
		# FIX: Restore page table to ONLY existing sequences (not all)
		# New sequences are being loaded async and NOT part of current forward pass
		if existing_global_ids:
			gpu_manager.rebuild_page_table(existing_global_ids)
		# If no existing sequences, page table will be rebuilt when batch becomes non-empty
		
		timing['prepare_ms'] = (time.perf_counter() - t0) * 1000
		
		# ============ PHASE 8: Launch async load ============
		t0 = time.perf_counter()
		
		if worker_view is None:
			logging.warning(f"Rank {self.rank}: worker_view is None, cannot launch async load")
			timing['launch_ms'] = (time.perf_counter() - t0) * 1000
			return None, new_uuids, new_local_indices, new_global_ids, timing
		
		async_task = worker_view.async_load_layer_paged_kv_to_device(
			sequence_ids=sequence_tensor,
			active_page_counts=active_page_counts,
			k_device_ptrs=k_ptrs,
			v_device_ptrs=v_ptrs,
		)
		
		# ASYNC MODE: Return task without waiting - wait happens at page boundary
		# The async load overlaps with the next page's decoding iterations
		Attn_Wrapper.async_kv_load_active = True
		Attn_Wrapper.async_kv_load_task = async_task
		
		timing['launch_ms'] = (time.perf_counter() - t0) * 1000
		
		# Store tensor references to prevent GC during async operation
		self._async_load_tensors = {
			'k_ptrs': k_ptrs,
			'v_ptrs': v_ptrs,
			'sequence_tensor': sequence_tensor,
			'active_page_counts': active_page_counts,
		}
		
		return async_task, new_uuids, new_local_indices, new_global_ids, timing

	def _finalize_async_load(
		self,
		async_task: object,
		pending_uuids: List[str],
		pending_local_indices: List[int],
		pending_global_ids: List[int],
		current_decode_uuids: List[str],
		current_batch: List[int],
		gpu_manager: GPUPagedKVCacheManager
	) -> Tuple[List[str], List[int]]:
		"""
		Integrate new sequences after async load completes.
		
		NOTE: Caller is responsible for waiting on async_task before calling this.
		NOTE: Does NOT rebuild page table - caller must rebuild after.
		"""
		# Clear async load flag and task reference - load is complete
		Attn_Wrapper.async_kv_load_active = False
		Attn_Wrapper.async_kv_load_task = None
		
		# Clear tensor references (task is complete)
		if hasattr(self, '_async_load_tensors'):
			self._async_load_tensors = None
		
		# Log completion
		if pending_global_ids:
			logging.info(
				f"Rank {self.rank}: Async load completed for {len(pending_global_ids)} sequences"
			)
		
		# Update status for ALL new sequences (globally consistent)
		self._update_batch_status(pending_uuids, SequenceStatus.IN_DECODE)
		
		# Update tracking for THIS RANK's sequences
		for local_idx in pending_local_indices:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			seq.gpu_pages_allocated = seq.get_gpu_pages_for_two_page_buffer()
			self._sequences_with_gpu_kv.add(uuid)
		
		# Merge into decode batch with deterministic ordering
		updated_uuids = current_decode_uuids + pending_uuids
		updated_uuids.sort(key=lambda u: self.global_batch.get_sequence(u).global_idx)
		
		# Derive updated local batch
		uuid_to_local = {}
		for idx in current_batch:
			uuid = self._local_to_uuid_map.get(idx)
			if uuid:
				uuid_to_local[uuid] = idx
		for idx in pending_local_indices:
			uuid = self._local_to_uuid_map.get(idx)
			if uuid:
				uuid_to_local[uuid] = idx
		
		updated_batch = [uuid_to_local[u] for u in updated_uuids if u in uuid_to_local]
		
		logging.info(
			f"Rank {self.rank}: Integrated {len(pending_uuids)} loaded sequences, "
			f"decode batch: {len(current_decode_uuids)} -> {len(updated_uuids)}, "
			f"local batch: {len(current_batch)} -> {len(updated_batch)}"
		)
		
		return updated_uuids, updated_batch

	def _sync_completion_status_at_boundary(
		self, 
		decode_uuids: List[str]
	) -> Tuple[List[str], List[str]]:
		"""
		Efficient completion sync at page boundaries using all_reduce.
		FIXED: Correctly respects ignore_eos.
		"""
		if not decode_uuids:
			return [], []
		
		n = len(decode_uuids)
		completion = torch.zeros(n, dtype=torch.int32, device=self.torch_device)
		
		for i, uuid in enumerate(decode_uuids):
			if uuid in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				# FIXED: Use unified completion check
				if self._is_sequence_completed(seq):
					completion[i] = 1
		
		dist.all_reduce(completion, op=dist.ReduceOp.MAX)
		
		active = []
		completed = []
		
		for i, uuid in enumerate(decode_uuids):
			if completion[i].item() == 1:
				completed.append(uuid)
				seq = self.global_batch.get_sequence(uuid)
				# Mark as completed (for consistency)
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
		FIXED: Respects ignore_eos flag.
		"""
		if not decode_uuids:
			return [], []
		
		completion_mask = torch.zeros(len(decode_uuids), dtype=torch.int32, device=self.torch_device)
		
		for i, uuid in enumerate(decode_uuids):
			if uuid in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				# FIXED: Use unified completion check
				if self._is_sequence_completed(seq):
					completion_mask[i] = 1
		
		dist.all_reduce(completion_mask, op=dist.ReduceOp.MAX)
		
		active_uuids = []
		completed_uuids = []
		
		for i, uuid in enumerate(decode_uuids):
			seq = self.global_batch.get_sequence(uuid)
			if completion_mask[i].item() == 1:
				completed_uuids.append(uuid)
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
				
				# FIXED: Use updated _check_and_handle_completions
				decode_uuids, batch, completed_uuids = self._check_and_handle_completions(
					decode_uuids, batch, new_token_idx
				)
				
				if completed_uuids:
					self._update_batch_status(completed_uuids, SequenceStatus.COMPLETED)
					# FIX: Must release GPU KV pages BEFORE releasing host KV pages
					my_completed = [u for u in completed_uuids if u in self._uuid_to_local_map]
					if my_completed:
						my_completed_local_indices = self._get_local_indices_for_uuids(my_completed)
						self._release_gpu_kv_pages(my_completed_local_indices)
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
						
						# Only mark eos_reached if we should stop at EOS
						if self._should_stop_at_eos(new_tokens[i].item()):
							seq.eos_reached = True
						
						# Always check max length
						if seq.decoded_length >= self.max_decoding_length:
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
						
						# Only mark eos_reached if we should stop at EOS
						if self._should_stop_at_eos(new_tokens[i].item()):
							seq.eos_reached = True
						
						# Always check max length
						if seq.decoded_length >= self.max_decoding_length:
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
		self._ignore_eos = False

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

