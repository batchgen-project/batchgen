import concurrent.futures
import copy
import functools
import psutil
import logging
import math
import os
import socket
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Set

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from batchgen.config.model_registry import load_config
from batchgen.config.tokenizer_registry import load_tokenizer

# Use new wrapper system - Attn_Wrapper/Expert_Wrapper are aliases for backward compatibility
from batchgen.models.wrappers import BaseModuleWrapper, AttnWrapperBase, ExpertWrapperBase
# Aliases for backward compatibility with existing code
Attn_Wrapper = AttnWrapperBase
Expert_Wrapper = ExpertWrapperBase

from .config.config import EngineConfig
from .models.deepseek.deepseek_parameter_server import DeepSeek_Parameter_Server
from .scheduler.host_mem import get_physical_memory_info

from batchgen.parameter_server_client import ParameterServerClient
from .models.deepseek.deepseekv3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from tqdm import trange
import gc
import numpy as np
from datetime import timedelta
from contextlib import contextmanager
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
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus, INITIAL_GPU_PAGE_BUFFER, EXTENSION_GPU_PAGE_BUFFER, DECISION_FREQUENCY_PAGES, configure_page_buffers
from batchgen.prefill.prepack import prepack_sequences, unpack_last_token_logits, get_prepack_stats, PrepackMetadata

# Import modularized components
# FastBoundaryTimingStats: Timing dataclass for page boundary operations
from batchgen.continuous_batching import FastBoundaryTimingStats
# sample_tokens: Token sampling with temperature/top_p support
from batchgen.sampling import sample_tokens
# Migration data structures for KV cache migration between nodes
from batchgen.migration import MigrationOp, HostKVStats

BATCHGEN_ENABLE_ALL_TO_ALL = os.environ.get("BATCHGEN_ENABLE_ALL_TO_ALL")
if BATCHGEN_ENABLE_ALL_TO_ALL == "1":
	try:
		from pplx_kernels import nvshmem_init
	except ImportError as exc:
		logging.warning("Failed to import pplx_kernels.nvshmem_init: %s", exc)
		nvshmem_init = None
else:
	nvshmem_init = None

# Debug logging level for continuous batching page boundary
BATCHGEN_CB_DEBUG = os.environ.get("BATCHGEN_CB_LOG", "").upper() == "DEBUG"

# Prepack mode for efficient prefill batching (DEPRECATED: use --enable-prepack CLI arg)
# Default: enabled (recommended always on). Use --no-prepack to disable.
BATCHGEN_ENABLE_PREPACK = os.environ.get("BATCHGEN_ENABLE_PREPACK", "1") == "1"

# Optional runtime checks for NaN/Inf in KV tensors (disabled by default)
BATCHGEN_ENABLE_NAN_CHECK = os.environ.get('BATCHGEN_ENABLE_NAN_CHECK', '0') == '1'

# Optional gate for expensive/critical diagnostics (default off in production)
BATCHGEN_ENABLE_CRITICAL_DIAGS = os.environ.get('BATCHGEN_ENABLE_CRITICAL_DIAGS', '0') == '1'

# Decode preemption configuration (DEPRECATED: use CLI args instead)
# --host-kv-watermark: Default 70% (when free slots exceed this threshold, prefill is prioritized)
# --enable-decode-preemption / --no-decode-preemption: Default enabled
HOST_KV_WATERMARK_PERCENT = int(os.environ.get('BATCHGEN_HOST_KV_WATERMARK', '70'))
ENABLE_DECODE_PREEMPTION = os.environ.get('BATCHGEN_ENABLE_DECODE_PREEMPTION', '1') == '1'  # Default ON

# GPU KV cache size override (DEPRECATED: use --gpu-memory-frac CLI arg instead)
# If set, overrides the automatic calculation. Otherwise, size is computed as:
# gpu_kv_cache = GPU_mem * gpu_memory_frac - model_instance_size
_GPU_KV_CACHE_SIZE_OVERRIDE = os.environ.get("BATCHGEN_GPU_KV_CACHE_SIZE_GB")
NUM_GPUS_PER_NODE = int(os.environ.get('NUM_GPUS_PER_NODE', '8'))


# Note: Generic Scheduler removed - config is now created by model-specific Planner in initializer


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
	converted_ckpt_dir: Optional[str] = None
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
	# EP with offloading settings
	enable_ep_with_offloading: bool = False
	ep_offloading_ratio: float = 0.0

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


def _is_port_available(host: str, port: int) -> bool:
	"""Check if a port is available for binding on the given host."""
	try:
		with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
			s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
			s.bind((host, port))
			return True
	except (OSError, socket.error):
		return False


def _find_available_port(host: str, start_port: int, max_attempts: int = 100) -> int:
	"""Find an available port starting from start_port.

	Args:
		host: Host address to bind to
		start_port: Starting port number to check
		max_attempts: Maximum number of ports to try

	Returns:
		An available port number

	Raises:
		RuntimeError: If no available port is found within max_attempts
	"""
	for offset in range(max_attempts):
		port = start_port + offset
		if _is_port_available(host, port):
			return port
	raise RuntimeError(f"No available port found in range [{start_port}, {start_port + max_attempts})")


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
	converted_ckpt_dir: Optional[str]
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

	# Watchdog configuration
	watchdog_timeout: Optional[float] = 600.0  # Seconds before declaring process stuck (10 min for long inference)
	watchdog_test_stuck_time: float = 0.0  # Deliberate delay for testing
	watchdog_heartbeat_interval: Optional[float] = None  # Heartbeat interval

	# Prepack optimization (default: enabled, recommended always on)
	enable_prepack: bool = True
	# Host KV watermark percentage (default: 70% free = underutilized threshold)
	host_kv_watermark: int = 70
	# Decode preemption: interrupt decode for prefill when host KV is underutilized (default: enabled)
	enable_decode_preemption: bool = True
	# GPU memory fraction for KV cache calculation (default: 0.9)
	gpu_memory_frac: float = 0.9
	# GPU page buffer settings for decode scheduling
	initial_gpu_page_buffer: int = 32  # Pages to reserve on first GPU load
	extension_gpu_page_buffer: int = 4  # Pages to add at boundaries
	decision_frequency_pages: int = 2  # How often to make scheduling decisions (in pages)

	# EP with offloading settings
	enable_ep_with_offloading: bool = False  # Enable Expert Parallelism with offloading
	ep_offloading_ratio: float = 0.0  # Ratio of experts per layer to offload (0.0-1.0)


class BatchGenWorker:
	"""
	Inference Runtime with Host-KV-First scheduling and Continuous Batching.
	"""
	PAGE_SIZE = 64  # Tokens per page (fixed)
	# Decision frequency: check boundaries every N pages (configurable via DECISION_FREQUENCY_PAGES)
	DECISION_INTERVAL = DECISION_FREQUENCY_PAGES * 64  # Tokens between boundary checks


	def __init__(self, args: BatchGenWorkerArgs):
		logging.info(f"Rank {args.global_rank}: Initializing BatchGenWorker.")

		# Configure page buffer settings from args (must be done before using the globals)
		configure_page_buffers(
			initial_gpu_page_buffer=args.initial_gpu_page_buffer,
			extension_gpu_page_buffer=args.extension_gpu_page_buffer,
			decision_frequency_pages=args.decision_frequency_pages,
		)
		# Update class attribute after configuration
		BatchGenWorker.DECISION_INTERVAL = args.decision_frequency_pages * 64

		# Watchdog for stuck detection (can be set via set_watchdog())
		self._watchdog = None

		# Log page buffer configuration (only on rank 0 to avoid spam)
		if args.global_rank == 0:
			logging.info(
				f"GPU Page Buffer Configuration: "
				f"initial_gpu_page_buffer={args.initial_gpu_page_buffer} pages ({args.initial_gpu_page_buffer * 64} tokens), "
				f"extension_gpu_page_buffer={args.extension_gpu_page_buffer} pages ({args.extension_gpu_page_buffer * 64} tokens), "
				f"decision_frequency_pages={args.decision_frequency_pages} pages ({args.decision_frequency_pages * 64} tokens)"
			)
		
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
		self.converted_ckpt_dir = args.converted_ckpt_dir
		self.skeleton_state_dict = args.skeleton_state_dict
		
		# 4. Initialize Shared Memory for Weights (Crucial for multiprocess)
		self.shm_name = args.shm_name
		self.tensor_meta_shm_name = args.tensor_meta_shm_name
		self.weight_byte_size = args.weight_byte_size
		self.enable_hugetlbfs = args.enable_hugetlbfs

		# Prepack and decode preemption configuration from args
		self.enable_prepack = args.enable_prepack
		self.host_kv_watermark = args.host_kv_watermark
		self.enable_decode_preemption = args.enable_decode_preemption

		# 4. Initialize Weights Storage (cudaHostRegister for weights)
		logging.info(f"Rank {self.rank}: Initializing shared memory segments (local_rank={self.local_rank}).")
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

		# 5. Initialize Host KV Cache Manager View (cudaHostRegister for Host KV)
		self.host_kv_cache_size = args.host_kv_cache_size
		self.global_host_kv_cache_size_gb = args.global_host_kv_cache_size_gb
		
		worker_kv_config = build_host_kv_config(
			model_name=args.model_name,
			host_kv_cache_size=args.global_host_kv_cache_size_gb * (1024**3),
		)

		# Select worker view based on model's KV cache configuration
		# MLA models (num_v_heads=0) don't have V cache, GQA/MHA models (num_v_heads>0) do
		if worker_kv_config.num_v_heads == 0:
			self.host_paged_kv_worker_view = core_engine.MLAHostPagedKVWorkerView(worker_kv_config)
		else:
			self.host_paged_kv_worker_view = core_engine.DefaultHostPagedKVWorkerView(worker_kv_config)

		# Initialize Host KV view (parallel cudaHostRegister for all local ranks)
		logging.info(f"Rank {self.rank}: Initializing Host KV view with parallel cudaHostRegister (local_rank={self.local_rank})")
		self.host_paged_kv_worker_view.initialize(device_index=self.local_rank, create_region=False)
		logging.info(f"Rank {self.rank}: Host KV cudaHostRegister completed (local_rank={self.local_rank})")

		# 6. Initialize Placeholders for Core Components
		# These are populated later in Init() / _initialize_core_components
		self.gpu_paged_kv_cache_manager = None
		self.model = None
		self.model_config = None
		self.loaded_model_config = None
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
		self._free_local_indices: Set[int] = set()  # Track freed indices for O(1) allocation
		self._next_local_idx: int = 0  # Next index if free list is empty
		
		# 8. Runtime State
		self.eos_token_id: Optional[int] = None
		self.max_input_length = 0
		self.max_decoding_length = 0
		self.model_context_length = 131072  # Default 128K, updated from model config
		self.num_global_queries = 0
		self.num_local_queries = 0
		self._ignore_eos: bool = False
		self._temperature: Optional[float] = None  # Sampling temperature (None = greedy)
		self._top_p: Optional[float] = None  # Nucleus sampling threshold (None = disabled)
		self._logged_greedy: bool = False  # Track if we've logged greedy mode this batch
		self._logged_sampling: bool = False  # Track if we've logged sampling mode this batch

		# 9. Initialization Flags
		self._core_initialized = False
		self._batch_completed = False
		self._nvshmem_initialized_this_run = False
		
		# 10. Distributed Communication Info
		self.dist_init_addr = args.dist_init_addr
		self.comm = None  # Initialized lazily or in Init()
		self._nccl_group = None  # StatelessProcessGroup for PyNccl (stores TCPStore)

		COMM_MASTER_ADDR = self.dist_init_addr.split(':')[0]
		os.environ['COMM_MASTER_ADDR'] = COMM_MASTER_ADDR

		# GPU KV cache configuration
		# Store gpu_memory_frac, actual size calculated later right before GPU KV manager init
		self.gpu_memory_frac = args.gpu_memory_frac
		self.gpu_kv_cache_size_gb: Optional[float] = None  # Calculated in _calculate_gpu_kv_cache_size()
		
		# Track sequences currently with GPU KV allocated
		self._sequences_with_gpu_kv: Set[str] = set()

		logging.info(f"Rank {self.rank}: BatchGenWorker __init__ completed.")

	def Init(self, max_input_length, max_decoding_length, num_queries):
		"""
		Initialize/reconfigure for a new batch.
		- First call: performs full initialization of core_engine, parallel_manager, etc.
		- Subsequent calls: only updates batch parameters and resets state.
		
		Args:
			max_input_length: Maximum input length hint. If None, will be determined dynamically
			                  during tokenization as the longest prompt in the batch.
			                  For first initialization, a default of 8192 is used if None.
			max_decoding_length: Maximum number of tokens to decode.
			num_queries: Number of queries in the global batch.
		"""
		# Check if we need to reset state from previous batch
		if self._core_initialized and self.global_batch is not None:
			self._reset_for_new_batch()
		
		# Update batch-specific parameters
		# max_input_length can be None - will be set during tokenization
		# For first initialization, use a reasonable default if None (needed for scheduler)
		if max_input_length is None or max_input_length <= 0:
			# Default hint for scheduler; actual value determined during tokenization
			self.max_input_length = 8192 if not self._core_initialized else 0
		else:
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

	def _calculate_gpu_kv_cache_size(self) -> float:
		"""
		Calculate GPU KV cache size based on actual GPU memory usage.

		Uses torch.cuda.mem_get_info() to get real memory usage after model is loaded.
		Formula: gpu_kv_cache = total_gpu_mem * gpu_memory_frac - used_mem

		This reserves (1-gpu_memory_frac) of total GPU memory for activations and overhead.

		IMPORTANT: Must be called in _initialize_core_components() right after model loading,
		BEFORE any inference (prefill/decode). If called during/after prefill, activation
		memory will be included in 'used_mem', resulting in incorrect (possibly negative) size.

		Rank 0 calculates and broadcasts to all ranks to ensure consistency.
		"""
		# Check for environment variable override first
		if _GPU_KV_CACHE_SIZE_OVERRIDE is not None:
			gpu_kv_cache_gb = float(_GPU_KV_CACHE_SIZE_OVERRIDE)
			if self.rank == 0:
				logging.info(
					f"[GPU-KV] Size from env override: {gpu_kv_cache_gb:.2f} GB "
					f"(BATCHGEN_GPU_KV_CACHE_SIZE_GB)"
				)
			return gpu_kv_cache_gb

		# Rank 0 calculates, then broadcasts to all ranks
		if self.rank == 0:
			# Get actual GPU memory usage (after model is loaded)
			free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(self.local_rank)
			free_mem_gb = free_mem_bytes / (1024 ** 3)
			total_mem_gb = total_mem_bytes / (1024 ** 3)
			used_mem_gb = total_mem_gb - free_mem_gb

			# Formula: gpu_kv_cache = total * frac - used
			# This reserves (1-frac) of GPU memory for activations and overhead
			gpu_kv_cache_gb = total_mem_gb * self.gpu_memory_frac - used_mem_gb

			# Ensure positive value
			if gpu_kv_cache_gb <= 0:
				logging.warning(
					f"[GPU-KV] Calculated size is non-positive ({gpu_kv_cache_gb:.2f} GB). "
					f"Total: {total_mem_gb:.2f} GB × frac: {self.gpu_memory_frac} - used: {used_mem_gb:.2f} GB. "
					f"Using minimum 1 GB."
				)
				gpu_kv_cache_gb = 1.0

			logging.info(
				f"[GPU-KV] Size calculated: {gpu_kv_cache_gb:.2f} GB "
				f"(total: {total_mem_gb:.2f} GB × frac: {self.gpu_memory_frac} - used: {used_mem_gb:.2f} GB)"
			)
		else:
			gpu_kv_cache_gb = 0.0

		# Broadcast from rank 0 to all ranks
		size_tensor = torch.tensor([gpu_kv_cache_gb], dtype=torch.float32, device=self.torch_device)
		dist.broadcast(size_tensor, src=0)
		gpu_kv_cache_gb = float(size_tensor.item())

		return gpu_kv_cache_gb

	def _initialize_gpu_kv_manager_fixed_size(self) -> GPUPagedKVCacheManager:
		"""
		Initialize GPU KV manager with pre-determined fixed size.
		Called once at the start of decoding.
		"""
		from batchgen.kv_cache.host_kv_mananger_config import build_gpu_kv_config_fixed_size

		# Calculate GPU KV cache size if not already done
		if self.gpu_kv_cache_size_gb is None:
			self.gpu_kv_cache_size_gb = self._calculate_gpu_kv_cache_size()

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

		if self.rank == 0:
			logging.info(
				f"[GPU-KV] Initialized: {self.gpu_kv_cache_size_gb:.2f} GB, {config.num_pages} pages"
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

	def set_sampling_params(self, temperature: Optional[float] = None, top_p: Optional[float] = None) -> None:
		"""
		Set sampling parameters for token generation.

		Args:
			temperature: Sampling temperature. None or 0 = greedy decoding (deterministic).
			            Higher values (e.g., 0.7-1.0) increase randomness.
			top_p: Nucleus sampling threshold. None or 1.0 = disabled.
			       Lower values (e.g., 0.9) restrict sampling to top tokens.
		"""
		self._temperature = temperature
		self._top_p = top_p
		# Always log on rank 0 - use WARNING to ensure visibility
		if self.rank == 0:
			if temperature is not None or top_p is not None:
				logging.warning(f"[SAMPLING] temperature={temperature}, top_p={top_p} - will use sampling")
			else:
				logging.info(f"[SAMPLING] temperature=None, top_p=None - will use greedy decoding")

	def _select_tokens(self, logits: torch.Tensor) -> torch.Tensor:
		"""
		Select next tokens from logits using greedy or sampling strategy.

		Args:
			logits: [batch_size, vocab_size] logits from model

		Returns:
			[batch_size, 1] selected token indices
		"""
		# Fast path: greedy decoding (default)
		if self._temperature is None or self._temperature <= 0:
			# Log once per batch (only rank 0, first decode step)
			if not getattr(self, '_logged_greedy', False) and self.rank == 0:
				logging.debug(f"Using GREEDY decoding (temperature={self._temperature})")
				self._logged_greedy = True
			return torch.argmax(logits, dim=-1, keepdim=True)

		# Sampling with temperature/top_p
		# Log once per batch (only rank 0, first decode step)
		if not getattr(self, '_logged_sampling', False) and self.rank == 0:
			logging.info(f"Using SAMPLING: temperature={self._temperature}, top_p={self._top_p}")
			self._logged_sampling = True
		from batchgen.sampling import sample_tokens
		return sample_tokens(logits, temperature=self._temperature, top_p=self._top_p)

	def _log_prefill_timing(self):
		"""Log prefill timing stats if available (GPT-OSS specific)."""
		try:
			from batchgen.models.openai.gpt_oss_120b.wrappers import PrefillTimingStats
			if PrefillTimingStats.enabled:
				PrefillTimingStats.log_summary()
				PrefillTimingStats.reset()  # Reset for next prefill batch
		except ImportError:
			pass  # Not GPT-OSS or module not available

	def _log_decode_timing(self):
		"""Log decode timing stats if available (GPT-OSS specific)."""
		try:
			from batchgen.models.openai.gpt_oss_120b.wrappers import DecodeTimingStats
			if DecodeTimingStats.enabled:
				DecodeTimingStats.log_summary()
				DecodeTimingStats.reset()  # Reset for next decode batch
		except ImportError:
			pass  # Not GPT-OSS or module not available

	def set_watchdog(self, watchdog) -> None:
		"""
		Set the watchdog for stuck detection during inference.

		The watchdog will be fed periodically during generation to prevent
		false timeout detection on long-running inference.

		Args:
			watchdog: Watchdog instance with a feed() method, or None to disable
		"""
		self._watchdog = watchdog

	def feed_watchdog(self) -> None:
		"""Feed the watchdog to prevent timeout during long operations."""
		if self._watchdog is not None:
			self._watchdog.feed()

	@contextmanager
	def disable_watchdog(self):
		"""Context manager to temporarily disable watchdog during non-critical phases.

		Use this during tokenization, setup, and other phases where we don't want
		the watchdog to trigger. The watchdog should only monitor prefill and decode.
		"""
		if self._watchdog is not None:
			with self._watchdog.disable():
				yield
		else:
			yield

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
		2. It hit EOS AND ignore_eos is False, OR
		3. current_context_length >= model_context_length (context limit reached)
		"""
		# Always complete at max decoding length
		if seq.decoded_length >= self.max_decoding_length:
			return True
		
		# Complete if context length limit reached (prompt + decoded >= model max)
		if seq.current_context_length >= self.model_context_length:
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
		
		# DIAGNOSTIC: Log allocation details for KV corruption investigation (debug-only / opt-in)
		alloc_details = []
		for local_idx in local_sequence_ids:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			pages = seq.get_gpu_pages_for_two_page_buffer()
			pages_per_seq.append(pages * self.PAGE_SIZE)  # tokens for API
			total_pages += pages
			
			# Track details for resuming sequences (decoded_length > 0)
			if seq.decoded_length > 0:
				alloc_details.append({
					'uuid': uuid[:8],
					'global_idx': seq.global_idx,
					'decoded_length': seq.decoded_length,
					'current_context_length': seq.current_context_length,
					'pages_allocating': pages,
					'had_initial_gpu_reservation': seq.had_initial_gpu_reservation,
				})
		
		if alloc_details and BATCHGEN_CB_DEBUG and BATCHGEN_ENABLE_CRITICAL_DIAGS:
			logging.debug(
				f"Rank {self.rank}: _allocate_gpu_kv_two_page_buffer: Allocating GPU KV for {len(alloc_details)} RESUMING sequences. First 5: {alloc_details[:5]}"
			)

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
			# Mark that this sequence has received its initial GPU reservation
			seq.mark_initial_gpu_reservation_done()
		
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

		# DIAGNOSTIC: Log page table order after allocation for debugging order mismatch (debug-only / opt-in)
		if manager and manager._gpu_page_table_manager and BATCHGEN_CB_DEBUG and BATCHGEN_ENABLE_CRITICAL_DIAGS:
			final_slot_order = list(manager._gpu_page_table_manager.slot_to_seq_id) if manager._gpu_page_table_manager.slot_to_seq_id else []
			logging.debug(
				f"Rank {self.rank}: _allocate_gpu_kv_two_page_buffer finished. "
				f"input_global_ids={global_ids[:5]}{'...' if len(global_ids) > 5 else ''} (len={len(global_ids)}), "
				f"all_active_sorted={all_active_global_ids[:5]}{'...' if len(all_active_global_ids) > 5 else ''} (len={len(all_active_global_ids)}), "
				f"final_slot_to_seq_id={final_slot_order[:5]}{'...' if len(final_slot_order) > 5 else ''} (len={len(final_slot_order)})"
			)

		logging.debug(
			f"Rank {self.rank}: Allocated GPU KV for {len(global_ids)} sequences"
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

		Strategy: Evict SHORTEST decoded sequences first (least progress).
		Rationale: Keep longer-decoded sequences in GPU because:
		  1. They are closer to completion (may finish soon)
		  2. We want to prioritize finishing sequences over starting new ones

		Returns:
			List of uuids to put ON_HOLD
		"""
		manager = self.gpu_paged_kv_cache_manager
		current_free = manager.get_stats().num_free_pages if manager else 0
		pages_to_free = required_free_pages - current_free

		if pages_to_free <= 0:
			return []

		# Sort by decoded_length ASCENDING (least progress first - evict these)
		candidates = []
		for uuid in active_uuids:
			if uuid not in self._uuid_to_local_map:
				continue
			seq = self.global_batch.get_sequence(uuid)
			candidates.append((uuid, seq.decoded_length, seq.gpu_pages_allocated))

		candidates.sort(key=lambda x: (x[1], x[0]))  # ascending by decoded_length, then uuid for determinism

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
		v_tensor: torch.Tensor = None,
	) -> None:
		"""
		Fire-and-forget KV append to host.

		Adds task to pending list, does NOT wait.
		Tasks are waited at page boundary via _wait_pending_kv_append_tasks().

		Safety: Host writes don't race with GPU reads (different memory spaces).

		CRITICAL: Must keep tensor references alive until async operation completes!
		PyTorch's CUDA caching allocator can reuse memory if tensor is dereferenced
		while async operation is still reading from it.

		Args:
			layer_idx: Layer index
			batch: List of local indices in the batch
			k_tensor: Key tensor to append
			v_tensor: Value tensor to append (optional, for GQA models like GPT-OSS)
		"""
		if not batch:
			return
		
		worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		if worker_view is None:
			return
		
		# Build sequence info
		sequence_ids = []
		sequence_lengths = []
		
		# DIAGNOSTIC: Track host KV append positions for debugging
		append_diag = []
		
		for local_idx in batch:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			sequence_ids.append(seq.global_idx)
			# Write position is current position (0-indexed)
			write_pos = seq.current_context_length - 1
			sequence_lengths.append(write_pos)
			
			# Track for debugging (only first few sequences)
			if len(append_diag) < 3 and seq.decoded_length > 1:
				append_diag.append({
					'gid': seq.global_idx,
					'ctx_len': seq.current_context_length,
					'decoded_len': seq.decoded_length,
					'write_pos': write_pos,
				})
		
		# Log append positions for resumed sequences (layer 0 only to reduce spam)
		if layer_idx == 0 and append_diag and BATCHGEN_CB_DEBUG:
			logging.debug(
				f"Rank {self.rank}: layer=0 append positions: first_3_resumed_seqs={append_diag}"
			)
		
		# Reshape for MLA if needed (MLA has 3D tensors, GQA has 4D)
		if k_tensor.dim() == 3:
			k_tensor = k_tensor.unsqueeze(2)  # [B, 1, D] -> [B, 1, 1, D]
		if v_tensor is not None and v_tensor.dim() == 3:
			v_tensor = v_tensor.unsqueeze(2)  # [B, 1, D] -> [B, 1, 1, D]
		
		# Optional NaN/Inf detection (disabled by default to avoid redundant checks/logs)
		if BATCHGEN_ENABLE_NAN_CHECK and layer_idx == 0 and torch.isnan(k_tensor).any():
			nan_mask = torch.isnan(k_tensor).any(dim=-1).any(dim=-1).any(dim=-1)  # [batch]
			nan_indices = torch.where(nan_mask)[0].tolist()
			nan_seq_info = []
			for idx in nan_indices:
				if idx < len(batch):
					local_idx = batch[idx]
					uuid = self._local_to_uuid_map.get(local_idx, "unknown")
					seq = self.global_batch.get_sequence(uuid) if uuid != "unknown" else None
					nan_seq_info.append({
						'batch_idx': idx,
						'local_idx': local_idx,
						'uuid': uuid[:8] if uuid != "unknown" else "unknown",
						'global_idx': seq.global_idx if seq else -1,
						'ctx_len': seq.current_context_length if seq else -1,
					})
			logging.error(
				f"Rank {self.rank}: NaN detected in k_tensor BEFORE host append (layer={layer_idx}) - affected_seqs={nan_seq_info}"
			)
		
		# CRITICAL: Sync default stream before launching async D2H copy.
		# The k_tensor was computed on the default stream, but the async D2H copy
		# uses a separate copy stream. Without this sync, the copy stream might
		# start reading k_tensor before the default stream has finished writing it.
		# This is the root cause of KV corruption after decode interruption/resume.
		torch.cuda.current_stream(self.torch_device).synchronize()
		
		# Launch async append
		task = worker_view.async_append_decode_kv_to_host(
			layer_idx=layer_idx,
			sequence_ids=sequence_ids,
			k_tensor=k_tensor,
			v_tensor=v_tensor,  # GQA models (GPT-OSS) have separate V; MLA models pass None
			sequence_lengths=sequence_lengths,
		)

		# CRITICAL FIX: Store tensor references alongside task to prevent GC
		# Must store BOTH k and v tensors to prevent memory reuse during async D2H copy
		if not hasattr(self, '_pending_kv_append_tensors'):
			self._pending_kv_append_tensors = []
		self._pending_kv_append_tensors.append(k_tensor)
		if v_tensor is not None:
			self._pending_kv_append_tensors.append(v_tensor)
		
		# Add to pending list - will be waited at page boundary
		if task is not None:
			self._pending_kv_append_tasks.append(task)

		# THROTTLING FIX: Prevent "Resource temporarily unavailable" (EAGAIN) error
		# std::async creates a new thread for each task. With 61 layers and 64 tokens
		# per boundary, we can hit ~3900 concurrent threads per boundary interval.
		# Wait and clear when threshold is reached to avoid exhausting system thread limits.
		MAX_PENDING_KV_TASKS = 256
		if len(self._pending_kv_append_tasks) >= MAX_PENDING_KV_TASKS:
			self._wait_pending_kv_append_tasks()

	def _initialize_core_components(self, num_queries: int) -> None:
		"""
		One-time initialization of heavy components.
		Called only on the first Init() call.
		"""
		logging.info(f"Rank {self.rank}: Performing one-time core initialization")
		
		config_torch_module_initializer()
		
		self.model_config = load_config(self.huggingface_ckpt_name)
		
		# Extract model's maximum context length from config
		# This is used for completion criteria: prompt_length + decoded_length < context_length
		self.model_context_length = getattr(self.model_config, 'max_position_embeddings', 131072)  # Default 128K
		logging.info(f"Rank {self.rank}: Model context length set to {self.model_context_length}")
		
		# Load tokenizer using BatchGen's tokenizer abstraction
		# This removes the dependency on transformers.AutoTokenizer
		# Pass model identifier for pattern matching; tokenizer loads from package dir
		self.tokenizer = load_tokenizer(self.huggingface_ckpt_name)

		# Set EOS token ID from tokenizer
		self.eos_token_id = self.tokenizer.eos_token_id
		logging.info(f"Rank {self.rank}: EOS token ID set to {self.eos_token_id}")

		logging.info(f"Rank {self.rank}: Start initializing engine config.")
		# Note: EngineConfig is created by the model-specific initializer which uses a Planner
		# to compute all config values. The initializer is the single source of truth.
		# No need to create a separate scheduler here - it would be thrown away anyway.

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
			"converted_ckpt_dir": self.converted_ckpt_dir,
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
			"gpu_arch": self.gpu_arch,
			# EP with offloading settings
			"enable_ep_with_offloading": self.args.enable_ep_with_offloading,
			"ep_offloading_ratio": self.args.ep_offloading_ratio,
		}
		logging.info(f"kv_dtype: {input_arguments['kv_dtype']}")
			
		self.input_arguments = InputArguments(**input_arguments)
		self.initializer = get_initializer(self.huggingface_ckpt_name)
		self.initializer = self.initializer(self.input_arguments)
		self.core_engine, self.engine_config, self.model_config, self.loaded_model_config = (
			self.initializer.Init(self.weights_storage)
		)

		self.core_engine.host_paged_kv_worker_view = self.host_paged_kv_worker_view
		self.engine_config.Basic_Config.num_queries = num_queries

		# Set EP offloading config from command-line args
		self.engine_config.EP_Config.enable_offloading = self.args.enable_ep_with_offloading
		self.engine_config.EP_Config.offloading_ratio = self.args.ep_offloading_ratio
		if self.engine_config.EP_Config.enable_offloading:
			logging.info(
				f"Rank {self.rank}: EP with offloading enabled, "
				f"offloading_ratio={self.engine_config.EP_Config.offloading_ratio}"
			)

		self.parallel_manager = get_parallel_strategy_manager(self.huggingface_ckpt_name)
		self.parallel_manager = self.parallel_manager(
			self.loaded_model_config,
			self.engine_config,
			self.model_config,
			self.core_engine,
			self.skeleton_state_dict,
			self.local_rank,
			self.global_rank,
			self.world_size
		)

		# NOTE: GPU KV cache size is calculated in generate() via _init_gpu_kv_with_actual_size()
		# after _load_decode_model() loads model weights to GPU. At this point (init),
		# only the model skeleton exists and weights haven't been loaded yet.

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

	def _update_config_after_tokenization(self) -> None:
		"""
		Update engine config after tokenization determines the actual max_input_length.
		This is called after _tokenize_global_batch() which sets self.max_input_length
		to the longest prompt in the batch.
		"""
		if self.engine_config is None:
			return
			
		old_padding_length = self.engine_config.Basic_Config.padding_length
		if old_padding_length != self.max_input_length:
			logging.info(
				f"Rank {self.rank}: Updating padding_length from {old_padding_length} to {self.max_input_length} "
				f"(based on actual longest prompt)"
			)
			self.engine_config.Basic_Config.padding_length = self.max_input_length
			
			if hasattr(self, 'input_arguments') and self.input_arguments is not None:
				self.input_arguments.padding_length = self.max_input_length

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
		
		# DIAGNOSTIC: Check if these are resuming sequences (have decoded tokens)
		resuming_seq_info = []
		for global_idx in global_sequence_ids:
			# Find the sequence by global_idx
			for uuid, local_idx in self._uuid_to_local_map.items():
				seq = self.global_batch.get_sequence(uuid)
				if seq and seq.global_idx == global_idx and seq.decoded_length > 0:
					resuming_seq_info.append({
						'global_idx': global_idx,
						'decoded_length': seq.decoded_length,
						'current_context_length': seq.current_context_length,
					})
					break
		
		if resuming_seq_info and BATCHGEN_CB_DEBUG:
			logging.debug(
				f"Rank {self.rank}: _load_host_kv_to_gpu loading KV for {len(resuming_seq_info)} RESUMING sequences. First 5: {resuming_seq_info[:5]}"
			)
		
		sequence_tensor = torch.tensor(global_sequence_ids, dtype=torch.int64, device="cpu")
		k_ptrs, v_ptrs = manager.get_padded_3d_page_pointers()
		active_sequence_page_counts = manager.export_active_sequence_page_counts()

		# DEBUG: Check what's being passed to the host→GPU load
		if os.environ.get("BATCHGEN_DEBUG_KV_LOAD", "0") == "1":
			print(f"\n[HOST->GPU DEBUG] === Before async_load_layer_paged_kv_to_device ===")
			print(f"[HOST->GPU DEBUG] global_sequence_ids = {global_sequence_ids[:10]}... (total={len(global_sequence_ids)})")
			print(f"[HOST->GPU DEBUG] sequence_tensor = {sequence_tensor[:10].tolist()}...")
			print(f"[HOST->GPU DEBUG] active_sequence_page_counts = {active_sequence_page_counts[:10].tolist()}...")
			print(f"[HOST->GPU DEBUG] k_ptrs.shape = {k_ptrs.shape}")
			# Check if global_sequence_ids are all the same (BUG)
			unique_ids = set(global_sequence_ids)
			if len(unique_ids) < len(global_sequence_ids):
				print(f"[HOST->GPU DEBUG] *** WARNING: DUPLICATE sequence IDs! Unique: {len(unique_ids)}, Total: {len(global_sequence_ids)} ***")
			else:
				print(f"[HOST->GPU DEBUG] OK: All {len(global_sequence_ids)} sequence IDs are unique")

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
		# CRITICAL: Sync CUDA after async task completes to ensure H2D DMA is done
		torch.cuda.synchronize(self.torch_device)

		load_duration = time.perf_counter() - copy_start
		logging.debug(
			"Rank %s Loaded host KV for %d sequences into GPU cache in %.3fs",
			self.rank, len(global_sequence_ids), load_duration,
		)

		# DEBUG: Verify KV content is different across sequences after host→GPU load
		if os.environ.get("BATCHGEN_DEBUG_KV_LOAD", "0") == "1" and len(global_sequence_ids) >= 2:
			k_cache, v_cache = manager.get_kv_tensors()
			# k_cache shape: [num_layers, num_pages, page_size, num_kv_heads, head_dim]
			print(f"\n[KV LOAD DEBUG] === After Host→GPU Load ===")
			print(f"[KV LOAD DEBUG] Loaded {len(global_sequence_ids)} sequences: {global_sequence_ids[:5]}...")
			print(f"[KV LOAD DEBUG] k_cache.shape={k_cache.shape}")

			# Get page table to find physical pages for each sequence
			page_table = manager._gpu_page_table_manager.gpu_table
			print(f"[KV LOAD DEBUG] page_table.shape={page_table.shape}")

			# Compare KV content at position 0 (first token) for first 3 sequences
			# Layer 0, position 0 should contain different values if prefill worked correctly
			num_to_check = min(3, len(global_sequence_ids))
			kv_samples = []
			for i in range(num_to_check):
				# Get the GPU page for this sequence's position 0
				slot_idx = i  # slot_indices are 0, 1, 2, ...
				page_idx = 0  # position 0
				gpu_page = int(page_table[slot_idx, page_idx].item())
				offset = 0  # position 0 within page

				# Read K at layer 0, this page, position 0, head 0, first 4 dims
				k_sample = k_cache[0, gpu_page, offset, 0, :4].cpu().tolist()
				v_sample = None
				if v_cache is not None:
					v_sample = v_cache[0, gpu_page, offset, 0, :4].cpu().tolist()

				kv_samples.append({
					'seq': i,
					'global_idx': global_sequence_ids[i],
					'slot': slot_idx,
					'gpu_page': gpu_page,
					'k_sample': k_sample,
					'v_sample': v_sample,
				})
				print(f"[KV LOAD DEBUG] seq{i} (global={global_sequence_ids[i]}): slot={slot_idx}, gpu_page={gpu_page}, K[0,:4]={k_sample}")

			# Check if all K samples are identical (BAD)
			all_k_same = all(s['k_sample'] == kv_samples[0]['k_sample'] for s in kv_samples)
			if all_k_same:
				print(f"[KV LOAD DEBUG] *** WARNING: ALL {num_to_check} SEQUENCES HAVE IDENTICAL K VALUES! ***")
				print(f"[KV LOAD DEBUG] This indicates prefill wrote identical KV or host→GPU load is corrupted!")
			else:
				print(f"[KV LOAD DEBUG] OK: K values differ across sequences (expected for different prompts)")

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

		# DIAGNOSTIC: Log state before destruction for KV corruption investigation
		if self.global_batch is not None:
			seqs_with_gpu_alloc = []
			for seq in self.global_batch:
				if seq.gpu_pages_allocated > 0 or seq.had_initial_gpu_reservation:
					seqs_with_gpu_alloc.append({
						'uuid': seq.uuid[:8],
						'global_idx': seq.global_idx,
						'status': seq.status.name,
						'gpu_pages_allocated': seq.gpu_pages_allocated,
						'had_initial_gpu_reservation': seq.had_initial_gpu_reservation,
						'current_context_length': seq.current_context_length,
						'decoded_length': seq.decoded_length,
					})
			if seqs_with_gpu_alloc and BATCHGEN_CB_DEBUG:
				logging.debug(
					f"Rank {self.rank}: _destroy_gpu_paged_kv_cache called with "
					f"{len(seqs_with_gpu_alloc)} sequences having GPU allocation state. "
					f"First 5: {seqs_with_gpu_alloc[:5]}"
				)

		manager.destroy(empty_cuda_cache=empty_cuda_cache)
		
		# FIX Bug 2: Clear tracking set when GPU KV is destroyed
		self._sequences_with_gpu_kv.clear()
		
		# CRITICAL FIX: Reset GPU allocation state for ALL non-completed sequences
		# Without this, sequences retain stale had_initial_gpu_reservation=True,
		# causing them to get insufficient GPU buffer on resume after prefill interruption
		if self.global_batch is not None:
			reset_count = 0
			for seq in self.global_batch:
				if seq.status != SequenceStatus.COMPLETED:
					if seq.gpu_pages_allocated > 0 or seq.had_initial_gpu_reservation:
						if BATCHGEN_CB_DEBUG:
							logging.debug(
								f"Rank {self.rank}: Resetting GPU state for {seq.uuid[:8]} "
								f"(status={seq.status.name}, gpu_pages={seq.gpu_pages_allocated}, "
								f"had_initial={seq.had_initial_gpu_reservation})"
							)
						seq.reset_gpu_allocation()
						reset_count += 1
			if reset_count > 0:
				logging.info(
					f"Rank {self.rank}: Reset GPU allocation state for {reset_count} sequences"
				)

	def _get_host_kv_free_pages(self) -> int:
		"""Get current free pages from host KV cache."""
		stats = self.host_paged_kv_worker_view.get_stats()
		return stats.num_free_pages

	def _get_or_create_gloo_group(self):
		"""Get or create a Gloo process group for CPU tensor migrations.

		Gloo backend supports CPU tensors and can use RDMA if available.
		This is more memory efficient than NCCL (which requires GPU staging).

		Returns:
			The Gloo process group for CPU tensor operations.
		"""
		if not hasattr(self, '_gloo_migration_group') or self._gloo_migration_group is None:
			logging.debug(f"Rank {self.rank}: Creating Gloo process group for CPU migrations")
			# Create a new group with Gloo backend including all ranks
			self._gloo_migration_group = dist.new_group(
				ranks=list(range(self.world_size)),
				backend="gloo"
			)
			logging.debug(f"Rank {self.rank}: Gloo process group created")
		return self._gloo_migration_group

	def _destroy_gloo_group(self):
		"""Destroy the Gloo process group after migrations are done."""
		if hasattr(self, '_gloo_migration_group') and self._gloo_migration_group is not None:
			logging.debug(f"Rank {self.rank}: Destroying Gloo process group")
			dist.destroy_process_group(self._gloo_migration_group)
			self._gloo_migration_group = None

	def _get_host_kv_utilization(self) -> Dict[str, int]:
		"""Get host KV stats counting sequences with KV in host memory.

		Valid sequences = PREFILLED, ON_HOLD, and IN_DECODE (all have KV in host).
		- PREFILLED: KV stored in host after prefill
		- ON_HOLD: KV retained in host when evicted from GPU
		- IN_DECODE: KV streams to host after each attention layer

		Free pages = Total - used by valid sequences.

		IMPORTANT: Host KV is shared per-node, so we count sequences from ALL ranks
		on this node, not just this rank.

		Returns:
			Dict with: rank, node_id, num_free_pages, num_total_pages, num_used_pages, free_percent
		"""
		stats = self.host_paged_kv_worker_view.get_stats()

		# Count pages used by sequences with KV in host on THIS NODE (all ranks on node)
		# Host KV is shared across all GPUs on a node
		node_id = self.rank // NUM_GPUS_PER_NODE
		node_rank_start = node_id * NUM_GPUS_PER_NODE
		node_rank_end = min(node_rank_start + NUM_GPUS_PER_NODE, self.world_size)

		# CRITICAL FIX: IN_DECODE sequences also have KV in host (streams after each layer)
		valid_statuses = {SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD, SequenceStatus.IN_DECODE}
		
		# Count sequences per status for detailed logging
		status_counts = {status: [] for status in valid_statuses}
		for rank_on_node in range(node_rank_start, node_rank_end):
			for status in valid_statuses:
				seqs = self.global_batch.get_sequences_for_rank_with_status(rank_on_node, status)
				status_counts[status].extend(seqs)
		
		valid_sequences = []
		for seqs in status_counts.values():
			valid_sequences.extend(seqs)

		# Calculate pages used by valid sequences
		used_pages = 0
		for uuid in valid_sequences:
			seq = self.global_batch.get_sequence(uuid)
			pages_needed = math.ceil(seq.kv_token_budget / self.PAGE_SIZE)
			used_pages += pages_needed

		# Free pages = total - used by valid sequences
		free_pages = stats.num_total_pages - used_pages
		free_percent = int((free_pages / stats.num_total_pages) * 100) if stats.num_total_pages > 0 else 100

		return {
			'rank': self.rank,
			'node_id': self.rank // NUM_GPUS_PER_NODE,
			'num_free_pages': free_pages,
			'num_total_pages': stats.num_total_pages,
			'num_used_pages': used_pages,
			'free_percent': free_percent,
			# Include sequence counts for global aggregation
			'num_in_decode': len(status_counts[SequenceStatus.IN_DECODE]),
			'num_onhold': len(status_counts[SequenceStatus.ON_HOLD]),
			'num_prefilled': len(status_counts[SequenceStatus.PREFILLED]),
			'num_valid_sequences': len(valid_sequences),
		}

	def _check_host_kv_watermark_trigger(self) -> bool:
		"""Check if any node exceeds host KV free page watermark.

		Watermark = 70% FREE (underutilized).
		Only checks if this rank is local_rank 0 (one check per node).

		Returns:
			True if should interrupt decode and switch to prefill
		"""
		if not self.enable_decode_preemption:
			return False

		# Only local_rank 0 reports (one per node)
		if self.local_rank == 0:
			local_stats = self._get_host_kv_utilization()
		else:
			local_stats = None

		# Gather stats from all local_rank 0 representatives
		all_stats = [None] * self.world_size
		dist.all_gather_object(all_stats, local_stats)

		# Filter to only node representatives
		node_stats = [s for s in all_stats if s is not None]

		if not node_stats:
			return False

		# Check if any node above watermark (too much free space)
		max_free_percent = max(s['free_percent'] for s in node_stats)
		above_watermark = max_free_percent > self.host_kv_watermark

		# Check if queued sequences available
		has_queued = self.global_batch.has_queueing()

		should_trigger = above_watermark and has_queued

		# Log global host KV cache stats (rank 0 only, aggregated across all nodes)
		if self.rank == 0:
			# Aggregate stats across all nodes
			total_used_pages = sum(s['num_used_pages'] for s in node_stats)
			total_pages = sum(s['num_total_pages'] for s in node_stats)
			total_free_pages = sum(s['num_free_pages'] for s in node_stats)
			global_used_percent = int((total_used_pages / total_pages) * 100) if total_pages > 0 else 0
			global_free_percent = 100 - global_used_percent

			# Store page stats for use in decode step logging
			self._host_kv_page_stats = {
				'used': total_used_pages,
				'total': total_pages,
				'free_percent': global_free_percent,
				'num_nodes': len(node_stats),
			}
			
			if should_trigger:
				logging.info(
					f"[Host KV Cache] PREFILL TRIGGER: max_node_free={max_free_percent}% > {self.host_kv_watermark}%, "
					f"queued_sequences={len(self.global_batch.get_sequences_by_status(SequenceStatus.QUEUEING))}"
				)
				for s in node_stats:
					logging.info(
						f"[Host KV Cache]   Node {s['node_id']}: {s['num_used_pages']}/{s['num_total_pages']} "
						f"pages ({100-s['free_percent']}% used, {s['free_percent']}% free)"
					)
		else:
			# Log summary even when not triggering (every 10th check to avoid spam)
			if not hasattr(self, '_watermark_check_counter'):
				self._watermark_check_counter = 0
			self._watermark_check_counter += 1
			if self._watermark_check_counter % 10 == 0:
				logging.debug(
					f"[Host KV Cache] Check #{self._watermark_check_counter}: max_free={max_free_percent}%, "
					f"threshold={self.host_kv_watermark}%, has_queued={has_queued}, trigger={should_trigger}"
				)

		return should_trigger

	def _plan_kv_migration(self) -> List[MigrationOp]:
		"""Plan sequence migrations to rebalance host KV across nodes.

		Returns:
			List of MigrationOp objects describing planned migrations.
		"""
		# Gather host KV stats from all local_rank 0
		if self.local_rank == 0:
			local_stats = self._get_host_kv_utilization()
		else:
			local_stats = None

		all_stats = [None] * self.world_size
		dist.all_gather_object(all_stats, local_stats)
		node_stats = {s['node_id']: s for s in all_stats if s is not None}

		if len(node_stats) <= 1:
			# Only one node, no migration needed
			if self.rank == 0:
				logging.info("MIGRATION: Single node detected, skipping rebalancing")
			return []

		# Calculate target pages per node
		total_used = sum(s['num_used_pages'] for s in node_stats.values())
		num_nodes = len(node_stats)
		target_per_node = total_used // num_nodes

		if self.rank == 0:
			logging.info(
				f"MIGRATION: Planning rebalance: {total_used} total pages across {num_nodes} nodes, "
				f"target {target_per_node} pages/node"
			)
			for nid, s in sorted(node_stats.items()):
				imbalance = s['num_used_pages'] - target_per_node
				logging.info(
					f"MIGRATION:   Node {nid}: {s['num_used_pages']} pages "
					f"({'+' if imbalance > 0 else ''}{imbalance} vs target)"
				)

		# Identify overloaded and underutilized nodes
		overloaded = [(nid, s) for nid, s in node_stats.items() if s['num_used_pages'] > target_per_node]
		underutilized = [(nid, s) for nid, s in node_stats.items() if s['num_used_pages'] < target_per_node]

		if not overloaded or not underutilized:
			# Already balanced
			if self.rank == 0:
				logging.info("MIGRATION: Already balanced, no migrations needed")
			return []

		overloaded.sort(key=lambda x: x[1]['num_used_pages'], reverse=True)
		underutilized.sort(key=lambda x: x[1]['num_used_pages'])

		# Greedy migration planning
		migrations = []
		used_by_node = {nid: s['num_used_pages'] for nid, s in node_stats.items()}
		# Track sequences already selected for migration to avoid duplicates
		migrated_uuids = set()

		# CRITICAL: Reset dest_rank_counter at start of each planning round
		# to ensure deterministic behavior across all ranks
		self._dest_rank_counter = {}

		for src_node_id, _ in overloaded:
			while used_by_node[src_node_id] > target_per_node and underutilized:
				# Find sequences to migrate from src_node (excluding already selected)
				src_rank_base = src_node_id * NUM_GPUS_PER_NODE
				candidate_sequences = []
				for gpu_offset in range(NUM_GPUS_PER_NODE):
					src_rank = src_rank_base + gpu_offset
					if src_rank >= self.world_size:
						break
					for status in [SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD]:
						for uuid in self.global_batch.get_sequences_for_rank_with_status(src_rank, status):
							if uuid not in migrated_uuids:
								candidate_sequences.append(uuid)

				if not candidate_sequences:
					if self.rank == 0:
						if BATCHGEN_CB_DEBUG:
							logging.debug(f"MIGRATION: No more candidates on node {src_node_id}, stopping")
					break

				# CRITICAL: Sort candidates deterministically before selection
				# Set operations (get_sequences_for_rank_with_status) don't preserve order,
				# so we must sort to ensure all ranks pick the same sequence
				candidate_sequences.sort(key=lambda u: self.global_batch.get_sequence(u).global_idx)

				# Pick smallest sequence (better packing), with global_idx as tie-breaker
				# This ensures deterministic selection across all ranks
				uuid = min(candidate_sequences, key=lambda u: (
					self.global_batch.get_sequence(u).kv_token_budget,
					self.global_batch.get_sequence(u).global_idx  # Tie-breaker
				))
				seq = self.global_batch.get_sequence(uuid)
				pages_needed = math.ceil(seq.kv_token_budget / self.PAGE_SIZE)

				if self.rank == 0:
					if BATCHGEN_CB_DEBUG:
						logging.debug(
							f"MIGRATION: Selected seq {uuid[:8]}... from {len(candidate_sequences)} candidates "
							f"(global_idx={seq.global_idx}, from_rank={seq.assigned_rank}, pages={pages_needed})"
						)

				# Find dest node with most free space (lowest used pages)
				# Use node_id as tie-breaker for determinism
				dest_node_id = min(underutilized, key=lambda x: (used_by_node[x[0]], x[0]))[0]

				# Distribute across ranks on dest node for load balancing
				# Use round-robin based on migration count to this node
				# (counter is reset at start of each planning round)
				if dest_node_id not in self._dest_rank_counter:
					self._dest_rank_counter[dest_node_id] = 0

				dest_rank_offset = self._dest_rank_counter[dest_node_id] % NUM_GPUS_PER_NODE
				dest_rank = dest_node_id * NUM_GPUS_PER_NODE + dest_rank_offset
				if dest_rank >= self.world_size:
					dest_rank = dest_node_id * NUM_GPUS_PER_NODE  # Fallback to rank 0
				self._dest_rank_counter[dest_node_id] += 1

				# Record migration using MigrationOp dataclass
				migrations.append(MigrationOp(
					uuid=uuid,
					from_rank=seq.assigned_rank,
					to_rank=dest_rank,
					pages=pages_needed
				))

				# Mark as migrated to avoid selecting again
				migrated_uuids.add(uuid)

				# Update bookkeeping
				used_by_node[src_node_id] -= pages_needed
				used_by_node[dest_node_id] += pages_needed

				# Check if dest node is now balanced
				if used_by_node[dest_node_id] >= target_per_node:
					underutilized = [(nid, s) for nid, s in underutilized if nid != dest_node_id]

		# Sanity check: ensure no duplicate UUIDs in migrations
		migration_uuids = [m.uuid for m in migrations]
		if len(migration_uuids) != len(set(migration_uuids)):
			duplicate_uuids = [u for u in migration_uuids if migration_uuids.count(u) > 1]
			logging.error(
				f"[MIGRATION] BUG DETECTED: Duplicate sequences in migration plan! "
				f"Duplicates: {[u[:8] for u in set(duplicate_uuids)]}"
			)
			# Remove duplicates, keep only first occurrence
			seen = set()
			unique_migrations = []
			for mig in migrations:
				if mig.uuid not in seen:
					seen.add(mig.uuid)
					unique_migrations.append(mig)
			migrations = unique_migrations
			logging.warning(f"MIGRATION: Removed duplicates, {len(migrations)} unique migrations remain")

		if self.rank == 0:
			if migrations:
				logging.info(f"MIGRATION: Planned {len(migrations)} sequence migrations")
				for i, mig in enumerate(migrations[:5]):  # Log first 5
					logging.info(
						f"MIGRATION:   #{i+1}: seq {mig.uuid[:8]}... "
						f"rank {mig.from_rank} -> {mig.to_rank} ({mig.pages} pages)"
					)
				if len(migrations) > 5:
					logging.info(f"MIGRATION:   ... and {len(migrations)-5} more")
			else:
				logging.info("MIGRATION: No migrations needed after planning")

		return migrations

	def _execute_kv_migrations_parallel(self, migrations: List[MigrationOp]) -> None:
		"""Execute multiple KV migrations in parallel to utilize all network cards.

		Groups migrations by independent rank pairs and executes them concurrently.
		All ranks participate - those not involved in a particular migration round
		call barrier to stay synchronized.

		Args:
			migrations: List of MigrationOp objects describing migrations to execute.
		"""
		if not migrations:
			return

		# CRITICAL: Create Gloo group BEFORE migrations start.
		# dist.new_group() is a COLLECTIVE operation - ALL ranks must call it together.
		# We create it here so all ranks participate, not just sender/receiver.
		self._get_or_create_gloo_group()
		dist.barrier()  # Ensure all ranks have created the group

		# Group migrations into parallel rounds
		# Each round contains migrations that can execute concurrently (no shared ranks)
		rounds = self._group_migrations_for_parallel_execution(migrations)

		if self.rank == 0:
			logging.info(f"MIGRATION: Executing {len(migrations)} migrations in {len(rounds)} parallel rounds")

		for round_idx, round_migrations in enumerate(rounds):
			if self.rank == 0:
				logging.info(f"MIGRATION: Round {round_idx+1}/{len(rounds)}: {len(round_migrations)} parallel migrations")

			# Find if this rank participates in this round
			my_migration = None
			for mig in round_migrations:
				if self.rank == mig.from_rank or self.rank == mig.to_rank:
					my_migration = mig
					break

			# Execute migration if participating, otherwise just sync tensor shape info
			if my_migration is not None:
				# Verify sequence exists and is in expected state before migration
				seq = self.global_batch.get_sequence(my_migration.uuid)
				if seq is None:
					logging.error(f"MIGRATION: Rank {self.rank}: SKIP migration - seq {my_migration.uuid[:8]}... not found!")
				else:
					if BATCHGEN_CB_DEBUG:
						logging.debug(
							f"MIGRATION: Rank {self.rank}: Executing migration for {my_migration.uuid[:8]}... "
							f"(global_idx={seq.global_idx}, status={seq.status}, assigned_rank={seq.assigned_rank})"
						)
					self._execute_single_kv_migration(
						uuid=my_migration.uuid,
						from_rank=my_migration.from_rank,
						to_rank=my_migration.to_rank
					)

			# Barrier after each round to ensure all transfers in this round complete
			dist.barrier()

		if self.rank == 0:
			logging.info(f"MIGRATION: All {len(rounds)} parallel rounds completed")

	def _group_migrations_for_parallel_execution(self, migrations: List[MigrationOp]) -> List[List[MigrationOp]]:
		"""Group migrations into rounds that can execute in parallel.

		Migrations in the same round must not share any source or destination ranks.
		This ensures no rank is involved in multiple send/recv operations simultaneously.

		Args:
			migrations: List of MigrationOp objects

		Returns:
			List of rounds, where each round is a list of migrations that can run in parallel
		"""
		rounds = []
		remaining = list(migrations)

		while remaining:
			round_migrations = []
			used_ranks = set()

			for mig in remaining[:]:  # Iterate over copy
				from_rank = mig.from_rank
				to_rank = mig.to_rank

				# Check if either rank is already used in this round
				if from_rank not in used_ranks and to_rank not in used_ranks:
					round_migrations.append(mig)
					used_ranks.add(from_rank)
					used_ranks.add(to_rank)
					remaining.remove(mig)

			rounds.append(round_migrations)

		return rounds

	def _execute_single_kv_migration(self, uuid: str, from_rank: int, to_rank: int) -> None:
		"""Migrate KV cache for one sequence from source to dest rank.

		Migration path: Direct host-to-host copy via network (no GPU staging)
		Uses PyTorch distributed send/recv on CPU tensors for efficient inter-node transfer.

		Args:
			uuid: Sequence UUID to migrate
			from_rank: Source rank (current owner)
			to_rank: Destination rank (new owner)
		"""
		seq = self.global_batch.get_sequence(uuid)
		if seq is None:
			logging.error(f"Rank {self.rank}: Cannot migrate {uuid[:8]}... - sequence not found")
			return

		# CRITICAL: Use GPU KV manager's config for tensor shape - this matches what
		# copy_kv_to_tensor() returns and what copy_tensor_to_kv() expects.
		# For MLA: num_k_heads=1 (latent attention), k_head_dim=576 (compressed KV)
		# Do NOT use model_config.num_key_value_heads or loaded_model_config.qk_rope_head_dim
		# as those have different values!
		gpu_kv_config = self.gpu_paged_kv_cache_manager.config
		num_layers = self.model_config.num_hidden_layers
		num_k_heads = gpu_kv_config.num_k_heads  # For MLA: 1
		k_head_dim = gpu_kv_config.k_head_dim    # For MLA: 576 (compressed KV)
		kv_dtype = gpu_kv_config.kv_dtype
		page_size = gpu_kv_config.page_size_tokens  # Should be 64

		global_idx = seq.global_idx
		pages_needed = math.ceil(seq.kv_token_budget / page_size)

		# Tensor shape matches GPU KV manager: [num_layers, pages, page_size, num_k_heads, k_head_dim]
		k_shape = (num_layers, pages_needed, page_size, num_k_heads, k_head_dim)
		if self.rank == from_rank:
			# ===== SOURCE RANK: Read from host KV, send directly over network =====
			t0 = time.perf_counter()
			logging.debug(
				f"[MIGRATION] Rank {self.rank}: Send {uuid[:8]}... → rank {to_rank} "
				f"({pages_needed} pages)"
			)
			# Allocate CPU buffer for KV data
			# We'll load host KV → GPU → CPU buffer, then send
			# (Temporary workaround - ideally would read directly from host memory)
			manager = self.gpu_paged_kv_cache_manager
			worker_view = self.host_paged_kv_worker_view
			if manager is None:
				logging.error(f"Rank {self.rank}: GPU KV manager not initialized")
				return
			# Ensure GPU KV manager is initialized (may be destroyed between decode/prefill phases)
			if not manager.is_initialized:
				logging.debug(f"[MIGRATION] Rank {self.rank}: Re-initializing GPU KV manager for migration")
				manager.initialize()
			tokens_needed = pages_needed * page_size
			# Allocate temporary GPU pages
			manager.allocate_pages_for_sequences([global_idx], [tokens_needed])
			# CRITICAL: Must rebuild page table after allocation before using get_padded_3d_page_pointers
			# The GPU KV manager requires this to set up active slot mappings
			manager.rebuild_page_table([global_idx])
			# Load host KV → GPU
			sequence_tensor = torch.tensor([global_idx], dtype=torch.int64, device="cpu")
			k_ptrs, v_ptrs = manager.get_padded_3d_page_pointers()
			active_page_counts = manager.export_active_sequence_page_counts()
			
			# PRE-LOAD DIAGNOSTIC: Log host KV state before loading
			if BATCHGEN_CB_DEBUG:
				host_stats = worker_view.get_stats()
				logging.debug(
					f"MIGRATION: Rank {self.rank}: Loading host KV for {uuid[:8]}... "
					f"global_idx={global_idx}, tokens_needed={tokens_needed}, "
					f"active_page_counts={active_page_counts.tolist()}, "
					f"host_stats=(used={host_stats.num_used_pages}, total={host_stats.num_total_pages})"
				)
			
			load_task = worker_view.async_load_layer_paged_kv_to_device(
				sequence_ids=sequence_tensor,
				active_page_counts=active_page_counts,
				k_device_ptrs=k_ptrs,
				v_device_ptrs=v_ptrs,
			)
			load_task.wait()
			# CRITICAL: Sync CUDA after async task completes to ensure H2D DMA is done
			torch.cuda.synchronize(self.torch_device)
			t_load = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.rank}: Host→GPU load: {(t_load-t0)*1000:.1f}ms")
			# Extract to contiguous tensor on GPU
			k_gpu = manager.copy_kv_to_tensor(global_idx)
			t_extract = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.rank}: GPU tensor extraction: {(t_extract-t_load)*1000:.1f}ms")

			# MIGRATION SEND VALIDATION: expensive validation only when explicitly enabled
			if BATCHGEN_ENABLE_CRITICAL_DIAGS:
				# NOTE: Only validate the VALID portion of KV (up to current_context_length)
				# The last page may have uninitialized slots beyond the actual token count
				valid_tokens = seq.current_context_length
				total_slots = pages_needed * page_size
			
				# Reshape layer 0 to [total_tokens, num_k_heads, k_head_dim] to slice valid portion
				flat_k = k_gpu[0].reshape(total_slots, num_k_heads, k_head_dim)
				valid_k = flat_k[:valid_tokens]
			
				send_k_mean = valid_k.float().mean().item()
				send_k_std = valid_k.float().std().item()
				send_has_nan = torch.isnan(valid_k).any().item()
				send_is_zero = (valid_k == 0).all().item()
			
				# Check if NaN only in padding (this is OK)
				full_has_nan = torch.isnan(k_gpu[0]).any().item()
				padding_info = ""
				if full_has_nan and not send_has_nan:
					padding_info = f" [NaN in padding only - {total_slots - valid_tokens} unused slots]"
			
				logging.info(
					f"MIGRATION SEND: Rank {self.rank}: Validating KV for {uuid[:8]}... (global_idx={global_idx}): "
					f"k_gpu_shape={list(k_gpu.shape)}, valid_tokens={valid_tokens}/{total_slots}, "
					f"layer0_mean={send_k_mean:.4f}, std={send_k_std:.4f}, "
					f"has_nan={send_has_nan}, is_zero={send_is_zero}{padding_info}, "
					f"first_values={valid_k[0, 0, :4].tolist() if valid_k.numel() > 0 else 'N/A'}"
				)
				if send_is_zero:
					logging.error(
						f"MIGRATION SEND: Rank {self.rank}: CRITICAL - KV to send is ALL ZEROS for {uuid[:8]}! "
						f"Host KV may be corrupted or load failed."
					)
				if send_has_nan:
					logging.error(f"MIGRATION SEND: Rank {self.rank}: CRITICAL - KV to send has NaN for {uuid[:8]}!")
					# DEEP LAYER-BY-LAYER NaN ANALYSIS (debug-only): Find exactly which layers have NaN
					if BATCHGEN_CB_DEBUG:
						nan_layers = []
						for layer_idx in range(k_gpu.shape[0]):
							layer_k = k_gpu[layer_idx].reshape(total_slots, num_k_heads, k_head_dim)
							layer_valid_k = layer_k[:valid_tokens]
							if torch.isnan(layer_valid_k).any():
								# Find which tokens have NaN in this layer
								nan_token_mask = torch.isnan(layer_valid_k).any(dim=-1).any(dim=-1)  # [tokens]
								nan_token_indices = torch.where(nan_token_mask)[0][:5].tolist()  # First 5
								nan_layers.append({
									'layer': layer_idx,
									'nan_token_count': nan_token_mask.sum().item(),
									'first_nan_tokens': nan_token_indices,
								})
						logging.error(
							f"MIGRATION SEND: Rank {self.rank}: NaN layer analysis for {uuid[:8]}: "
							f"total_nan_layers={len(nan_layers)}/{k_gpu.shape[0]}, "
							f"details={nan_layers[:5]}"  # First 5 layers with NaN
						)

			# Move GPU → CPU for Gloo transfer (Gloo supports CPU tensors, more memory efficient)
			k_cpu = k_gpu.cpu().contiguous()
			t_cpu = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.rank}: GPU→CPU copy: {(t_cpu-t_extract)*1000:.1f}ms")
			# Send via Gloo backend (supports CPU tensors and RDMA if available)
			gloo_group = self._get_or_create_gloo_group()
			dist.send(tensor=k_cpu, dst=to_rank, group=gloo_group)
			t_send = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.rank}: Gloo send: {(t_send-t_cpu)*1000:.1f}ms")
			# Free GPU pages
			manager.free_pages_for_sequences([global_idx])
			# Free host KV pages
			worker_view.release_sequence_pages([global_idx])
			# Also send query_book data (input_ids, attention_mask, decoded_tokens)
			local_idx = self._uuid_to_local_map.get(uuid)
			if local_idx is not None and local_idx in self.query_book:
				qb = self.query_book[local_idx]
				# Send tensors via Gloo
				dist.send(tensor=qb.encoded["input_ids"].cpu().contiguous(), dst=to_rank, group=gloo_group)
				dist.send(tensor=qb.encoded["attention_mask"].cpu().contiguous(), dst=to_rank, group=gloo_group)
				dist.send(tensor=qb.decoded_tokens.cpu().contiguous(), dst=to_rank, group=gloo_group)
				if BATCHGEN_CB_DEBUG:
					logging.debug(f"MIGRATION: Rank {self.rank}: Sent query_book for {uuid[:8]}...")
			else:
				logging.warning(f"MIGRATION: Rank {self.rank}: No query_book entry for {uuid[:8]}... (local_idx={local_idx})")

			t_total = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(
					f"MIGRATION: Rank {self.rank}: Sent {uuid[:8]}... "
					f"in {(t_total-t0)*1000:.1f}ms"
				)
		elif self.rank == to_rank:
			# ===== DEST RANK: Receive over network via Gloo, write to host KV =====
			t0 = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(
					f"MIGRATION: Rank {self.rank}: Recv {uuid[:8]}... ← rank {from_rank} "
					f"({pages_needed} pages)"
				)
			# Allocate CPU buffer for receiving (Gloo supports CPU tensors)
			k_cpu = torch.empty(k_shape, dtype=kv_dtype, device="cpu", pin_memory=True)
			# Receive via Gloo backend
			gloo_group = self._get_or_create_gloo_group()
			dist.recv(tensor=k_cpu, src=from_rank, group=gloo_group)
			t_recv = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.rank}: Gloo recv: {(t_recv-t0)*1000:.1f}ms")
			# Register and allocate host KV pages
			worker_view = self.host_paged_kv_worker_view
			tokens_needed = pages_needed * page_size
			worker_view.register_sequences([global_idx])
			worker_view.allocate_pages_for_sequences([(global_idx, tokens_needed)])
			t_alloc = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.rank}: Host allocation: {(t_alloc-t_recv)*1000:.1f}ms")
			# Move CPU → GPU for offload to host KV
			k_gpu = k_cpu.to(self.device, non_blocking=True)
			torch.cuda.synchronize(self.torch_device)
			t_gpu = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.rank}: CPU→GPU copy: {(t_gpu-t_alloc)*1000:.1f}ms")

			# Offload layer-by-layer to host using async_offload_layer_kv_to_host
			# API expects: k_tensor [batch=1, seq_len, num_heads, head_dim]
			# Our k_gpu is [num_layers, num_pages, page_size, num_k_heads, k_head_dim]
			# Reshape: num_pages * page_size = total tokens
			seq_len = pages_needed * page_size
			# API expects sequence_ids as Python list, not tensor
			sequence_ids_list = [global_idx]
			sequence_lengths = [seq_len]

			# MIGRATION RECV VALIDATION: expensive validation only when explicitly enabled
			if BATCHGEN_ENABLE_CRITICAL_DIAGS:
				# NOTE: Only validate the VALID portion of KV (up to current_context_length)
				# The last page may have uninitialized slots beyond the actual token count
				first_layer_k = k_gpu[0]  # [num_pages, page_size, num_k_heads, k_head_dim]
				valid_tokens = seq.current_context_length
				total_slots = pages_needed * page_size
				
				# Reshape to [total_tokens, num_k_heads, k_head_dim] to easily slice valid portion
				flat_k = first_layer_k.reshape(total_slots, num_k_heads, k_head_dim)
				valid_k = flat_k[:valid_tokens]  # Only validate actual tokens
				
				migration_k_mean = valid_k.float().mean().item()
				migration_k_std = valid_k.float().std().item()
				migration_has_nan = torch.isnan(valid_k).any().item()
				migration_is_zero = (valid_k == 0).all().item()
				
				# Also check if the ENTIRE buffer has NaN (for debugging padding issues)
				full_has_nan = torch.isnan(first_layer_k).any().item()
				padding_info = ""
				if full_has_nan and not migration_has_nan:
					# NaN only in padding region - this is expected and OK
					padding_info = f" [NaN in padding only - {total_slots - valid_tokens} unused slots]"
				
				if BATCHGEN_CB_DEBUG:
					logging.info(
						f"MIGRATION: Rank {self.rank}: Validating received KV for {uuid[:8]}... (global_idx={global_idx}): "
						f"k_gpu_shape={list(k_gpu.shape)}, valid_tokens={valid_tokens}/{total_slots}, "
						f"layer0_mean={migration_k_mean:.4f}, std={migration_k_std:.4f}, "
						f"has_nan={migration_has_nan}, is_zero={migration_is_zero}{padding_info}, "
						f"first_values={valid_k[0, 0, :4].tolist() if valid_k.numel() > 0 else 'N/A'}"
					)
				if migration_is_zero:
					logging.error(
						f"MIGRATION RECV: Rank {self.rank}: CRITICAL - Received KV is ALL ZEROS for {uuid[:8]}! "
						f"This means network transfer failed or source had invalid data."
					)
				if migration_has_nan:
					logging.error(
						f"MIGRATION RECV: Rank {self.rank}: CRITICAL - Received KV has NaN for {uuid[:8]}!"
					)
					# DEEP LAYER-BY-LAYER NaN ANALYSIS (debug-only): Find exactly which layers have NaN
					if BATCHGEN_CB_DEBUG:
						nan_layers = []
						for layer_idx in range(k_gpu.shape[0]):
							layer_k = k_gpu[layer_idx].reshape(total_slots, num_k_heads, k_head_dim)
							layer_valid_k = layer_k[:valid_tokens]
							if torch.isnan(layer_valid_k).any():
								# Find which tokens have NaN in this layer
								nan_token_mask = torch.isnan(layer_valid_k).any(dim=-1).any(dim=-1)  # [tokens]
								nan_token_indices = torch.where(nan_token_mask)[0][:5].tolist()  # First 5
								nan_layers.append({
									'layer': layer_idx,
									'nan_token_count': nan_token_mask.sum().item(),
									'first_nan_tokens': nan_token_indices,
								})
						logging.error(
							f"MIGRATION RECV: Rank {self.rank}: NaN layer analysis for {uuid[:8]}: "
							f"total_nan_layers={len(nan_layers)}/{k_gpu.shape[0]}, "
							f"details={nan_layers[:5]}"  # First 5 layers with NaN
						)

			for layer_idx in range(num_layers):
				# Extract layer [num_pages, page_size, num_k_heads, k_head_dim]
				layer_k = k_gpu[layer_idx]  # [num_pages, page_size, num_k_heads, k_head_dim]
				# Reshape to [seq_len, num_k_heads, k_head_dim] then add batch dim
				layer_k_flat = layer_k.reshape(seq_len, num_k_heads, k_head_dim)
				layer_k_batch = layer_k_flat.unsqueeze(0)  # [1, seq_len, num_k_heads, k_head_dim]

				# CRITICAL: Keep a reference to the per-layer tensor until the
				# async offload completes. The offload runs on a separate copy
				# stream and uses the tensor's device memory; if Python GC
				# frees/reuses that memory before the copy finishes we get
				# corrupted data. We clear these refs after synchronizing below.
				if not hasattr(self, '_pending_migration_offload_tensors'):
					self._pending_migration_offload_tensors = []
				self._pending_migration_offload_tensors.append(layer_k_batch)

				worker_view.async_offload_layer_kv_to_host(
					layer_idx=layer_idx,
					sequence_ids=sequence_ids_list,
					k_tensor=layer_k_batch,
					v_tensor=None,  # MLA has no V
					sequence_lengths=sequence_lengths,
				)
				# Note: async_offload_layer_kv_to_host is fire-and-forget for each layer

			# Sync to ensure all offloads complete
			torch.cuda.synchronize(self.torch_device)
			# Clear held references for migration offload tensors so memory
			# can be reclaimed now that copies are guaranteed complete.
			if hasattr(self, '_pending_migration_offload_tensors'):
				self._pending_migration_offload_tensors.clear()
			t_store = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.rank}: GPU→Host offload all layers: {(t_store-t_gpu)*1000:.1f}ms")
			# Note: GPU tensor k_gpu was only a staging buffer, not allocated in GPU paged KV manager
			# It will be freed automatically when it goes out of scope

			# Receive query_book data (input_ids, attention_mask, decoded_tokens)
			# Get tensor shapes from the Sequence object (all ranks have seq metadata)
			input_ids_shape = seq.input_ids.shape
			attention_mask_shape = seq.attention_mask.shape
			decoded_tokens_shape = seq.decoded_tokens.shape

			input_ids_recv = torch.empty(input_ids_shape, dtype=seq.input_ids.dtype, device="cpu")
			attention_mask_recv = torch.empty(attention_mask_shape, dtype=seq.attention_mask.dtype, device="cpu")
			decoded_tokens_recv = torch.empty(decoded_tokens_shape, dtype=seq.decoded_tokens.dtype, device="cpu")

			dist.recv(tensor=input_ids_recv, src=from_rank, group=gloo_group)
			dist.recv(tensor=attention_mask_recv, src=from_rank, group=gloo_group)
			dist.recv(tensor=decoded_tokens_recv, src=from_rank, group=gloo_group)
			
			# Verify received attention_mask has correct number of 1s
			recv_attn_ones = attention_mask_recv.sum().item()
			expected_ones = seq.current_context_length
			if recv_attn_ones != expected_ones:
				logging.error(
					f"MIGRATION RECV: Rank {self.rank}: ATTENTION MASK MISMATCH for {uuid[:8]}! "
					f"received_ones={int(recv_attn_ones)}, expected={expected_ones} (ctx_len), "
					f"prompt_len={seq.prompt_length}, decoded_len={seq.decoded_length}"
				)
			else:
				if BATCHGEN_CB_DEBUG:
					logging.info(
						f"MIGRATION: Rank {self.rank}: Attention mask OK for {uuid[:8]}: "
						f"ones={int(recv_attn_ones)} == ctx_len={expected_ones}"
					)

			# Store in pending dict for later query_book creation
			if not hasattr(self, '_pending_migrated_query_book'):
				self._pending_migrated_query_book = {}
			# Track migrated sequences for corruption correlation
			if not hasattr(self, '_migrated_sequences'):
				self._migrated_sequences = set()
			self._migrated_sequences.add(uuid)
			self._pending_migrated_query_book[uuid] = {
				'text': seq.text,
				'input_ids': input_ids_recv,
				'attention_mask': attention_mask_recv,
				'decoded_tokens': decoded_tokens_recv,
				'kv_token_budget': seq.kv_token_budget,
			}
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.rank}: Recvd query_book for {uuid[:8]}...")

			t_total = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(
					f"MIGRATION: Rank {self.rank}: Recvd {uuid[:8]}... "
					f"in {(t_total-t0)*1000:.1f}ms"
				)
		# No barrier here - will be done in _rebalance_host_kv after all migrations

	def _rebalance_host_kv(self) -> None:
		"""Rebalance host KV cache by migrating sequences between nodes.

		Called during _config_prefill_for_batch() before assigning new sequences.
		This orchestrates the full rebalancing process:
		1. Plan migrations (deterministic across all ranks)
		2. Execute all migrations (NCCL transfers)
		3. Barrier to ensure all transfers complete
		4. Update sequence ownership metadata
		5. Barrier to ensure metadata consistency
		"""
		if not self.enable_decode_preemption:
			return

		rebalance_start = time.perf_counter()
		if self.rank == 0:
			logging.info("REBALANCE: Starting host KV rebalancing")

		# Plan migrations (all ranks compute same plan deterministically)
		migrations = self._plan_kv_migration()

		if not migrations:
			if self.rank == 0:
				logging.info("REBALANCE: No migrations needed, host KV already balanced")
			return

		# Log migration summary
		if self.rank == 0:
			total_pages = sum(m.pages for m in migrations)
			logging.info(
				f"REBALANCE: Executing {len(migrations)} migrations "
				f"({total_pages} total pages, ~{total_pages * 64} tokens)"
			)

		# STEP 1: Execute all migrations in parallel (host-to-host transfers)
		# Parallel execution utilizes all network cards by having multiple rank pairs
		# communicate simultaneously
		migration_start = time.perf_counter()
		self._execute_kv_migrations_parallel(migrations)
		migration_end = time.perf_counter()
		if self.rank == 0:
			logging.info(
				f"REBALANCE: All migrations completed in {(migration_end-migration_start)*1000:.1f}ms "
				f"({(migration_end-migration_start)*1000/len(migrations):.1f}ms per migration avg)"
			)

		# STEP 2: Update sequence ownership metadata and local mappings
		# CRITICAL: All ranks must update global_batch consistently
		# MUST use assign_rank() to update both seq.assigned_rank AND _rank_index
		for mig in migrations:
			uuid = mig.uuid
			new_rank = mig.to_rank

			# CRITICAL FIX: Use assign_rank() instead of direct assignment!
			# Direct assignment (seq.assigned_rank = x) only updates the attribute.
			# assign_rank() also updates the _rank_index which is used by
			# get_sequences_for_rank_with_status() - without this, the index
			# becomes inconsistent and causes cross-rank state divergence.
			try:
				self.global_batch.assign_rank(uuid, new_rank)
			except KeyError:
				logging.error(f"Rank {self.rank}: Cannot update ownership for {uuid[:8]}... - sequence not found")
				continue

			# IMPORTANT: Don't change sequence status - it remains PREFILLED or ON_HOLD
			# The sequence is still valid, just owned by a different rank now

		# Barrier to ensure all ranks have updated global_batch
		dist.barrier()

		# CRITICAL FIX: Sync sequence metadata BEFORE updating local mappings!
		# At this point:
		# - SEND side still has migrated sequences in _uuid_to_local_map (will report correct state)
		# - RECV side does NOT have them in _uuid_to_local_map yet (will receive and update)
		# If we sync AFTER updating local mappings, RECV side would skip updating because
		# uuid would be in its _uuid_to_local_map, but its state is stale!
		migrated_uuids = [m.uuid for m in migrations]
		if migrated_uuids:
			self._sync_sequence_metadata(migrated_uuids)
			logging.info(
				f"Rank {self.rank}: REBALANCE: Synced metadata for {len(migrated_uuids)} sequences "
				f"BEFORE local mapping update (SEND side still owns them)"
			)

		# Barrier to ensure all ranks have synced metadata
		dist.barrier()

		# STEP 3: Update local mappings (rank-specific, after global metadata is consistent)
		for mig in migrations:
			old_rank = mig.from_rank
			new_rank = mig.to_rank
			uuid = mig.uuid

			# Update local mappings on source rank (remove)
			if self.rank == old_rank:
				local_idx = self._uuid_to_local_map.pop(uuid, None)
				if local_idx is not None:
					self._local_to_uuid_map.pop(local_idx, None)
					self._sequences_with_gpu_kv.discard(uuid)
					# Remove query_book entry
					self.query_book.pop(local_idx, None)
					# Add freed index to free list for O(1) reuse
					self._free_local_indices.add(local_idx)
					logging.debug(f"Rank {self.rank}: Removed {uuid[:8]}... from local mappings (freed local_idx={local_idx})")

			# Update local mappings on dest rank (add)
			if self.rank == new_rank:
				# O(1) allocation: prefer reusing freed indices, otherwise use next available
				if self._free_local_indices:
					new_local_idx = self._free_local_indices.pop()
				else:
					new_local_idx = self._next_local_idx
					self._next_local_idx += 1

				self._uuid_to_local_map[uuid] = new_local_idx
				self._local_to_uuid_map[new_local_idx] = uuid
				# Note: Don't add to _sequences_with_gpu_kv - KV is in host, not GPU

				# Create query_book entry from pending migrated data
				if hasattr(self, '_pending_migrated_query_book') and uuid in self._pending_migrated_query_book:
					pending = self._pending_migrated_query_book.pop(uuid)
					self.query_book[new_local_idx] = query(
						text=pending['text'],
						encoded={
							"input_ids": pending['input_ids'],
							"attention_mask": pending['attention_mask'],
						},
						decoded_tokens=pending['decoded_tokens'],
						kv_token_budget=pending['kv_token_budget'],
					)
					logging.debug(f"Rank {self.rank}: Created query_book[{new_local_idx}] for migrated {uuid[:8]}...")
				else:
					logging.error(f"Rank {self.rank}: No pending query_book data for migrated {uuid[:8]}...")

				logging.debug(f"Rank {self.rank}: Added {uuid[:8]}... to local mappings (new local_idx={new_local_idx})")

		# BARRIER 2: Ensure all local mapping updates are complete across all ranks
		dist.barrier()

		# NOTE: Metadata sync was already done BEFORE local mapping updates (above)
		# At this point, all ranks have consistent metadata for migrated sequences.

		rebalance_end = time.perf_counter()
		if self.rank == 0:
			logging.info(
				f"[REBALANCE] Completed: {len(migrations)} sequences migrated "
				f"in {(rebalance_end-rebalance_start)*1000:.1f}ms total"
			)

			# Log final distribution
			if self.local_rank == 0:
				final_stats = self._get_host_kv_utilization()
				logging.info(
					f"  Node {final_stats['node_id']} final state: "
					f"{final_stats['num_used_pages']}/{final_stats['num_total_pages']} pages "
					f"({100-final_stats['free_percent']}% utilized)"
				)

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

		# Disable watchdog during setup phase - only monitor prefill/decode
		with self.disable_watchdog():
			# Step 2: Tokenize all sequences (all ranks do this identically)
			# This determines the actual max_input_length dynamically
			self._tokenize_global_batch()

			# Step 2.5: Update engine config with actual max_input_length after tokenization
			self._update_config_after_tokenization()

			# Step 3: Assign sequences to ranks (round-robin)
			self._assign_sequences_to_ranks()

			# Step 4: Build query_book for backward compatibility
			self._build_local_query_book()

			# Step 5: Set counts for compatibility
			self.num_global_queries = len(global_prompts)
			self.num_local_queries = len(self.global_batch.get_sequences_for_rank(self.rank))

		# Step 6: Run generation with KV-driven scheduling
		# Watchdog is now active - monitors prefill and decode phases
		return self.generate()

	# ============ UUID/Index Conversion Helpers ============

	def _local_to_uuid(self, local_idx: int) -> str:
		return self._local_to_uuid_map.get(local_idx, "")

	def _uuid_to_local(self, uuid: str) -> int:
		return self._uuid_to_local_map.get(uuid, -1)

	def _local_indices_to_global_seq_ids(self, local_indices: List[int]) -> List[int]:
		"""Convert local indices to global sequence IDs (global_idx from SequenceEntry)."""
		global_seq_ids = []
		missing_indices = []
		for local_idx in local_indices:
			uuid = self._local_to_uuid_map.get(local_idx)
			if uuid:
				seq = self.global_batch.get_sequence(uuid)
				global_seq_ids.append(seq.global_idx)
			else:
				missing_indices.append(local_idx)

		# CRITICAL: Log if any local indices are missing - this causes length mismatch
		# which leads to KV corruption (wrong sequence KV read for wrong batch position)
		if missing_indices:
			logging.error(
				f"Rank {self.rank}: MISSING LOCAL INDICES in _local_indices_to_global_seq_ids! "
				f"input_len={len(local_indices)}, output_len={len(global_seq_ids)}, "
				f"missing={missing_indices[:10]}..."
			)
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
		# CRITICAL FIX: Also compute and send prompt_length so receivers can validate ctx_len
		local_state = {}
		for uuid in decode_uuids:
			if uuid in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				# CRITICAL: Ensure current_context_length is consistent before sending
				# The invariant is: current_context_length = prompt_length + decoded_length
				expected_ctx = seq.prompt_length + seq.decoded_length
				if seq.current_context_length != expected_ctx:
					logging.warning(
						f"Rank {self.rank}: Correcting ctx_len for {uuid[:8]} before sync: "
						f"{seq.current_context_length} → {expected_ctx}"
					)
					seq.current_context_length = expected_ctx
				
				local_state[uuid] = {
					'decoded_length': seq.decoded_length,
					'current_context_length': seq.current_context_length,
					'gpu_pages_allocated': seq.gpu_pages_allocated,
					'eos_reached': seq.eos_reached,
					'prompt_length': seq.prompt_length,  # Include for validation
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
							
							# VALIDATION: Ensure received ctx_len is consistent
							expected_ctx = seq.prompt_length + seq.decoded_length
							if seq.current_context_length != expected_ctx:
								logging.error(
									f"Rank {self.rank}: [SYNC-VALIDATE] Received inconsistent ctx_len for {uuid[:8]}: "
									f"received={seq.current_context_length}, expected={expected_ctx} "
									f"(prompt={seq.prompt_length}, decoded={seq.decoded_length})"
								)
								seq.current_context_length = expected_ctx

	def _sync_completion_status_tensor(
		self,
		decode_uuids: List[str],
	) -> Tuple[Set[str], List[str]]:
		"""
		Synchronize completion status across all ranks using tensor operations.

		OPTIMIZATION: Replaces expensive all_gather_object with tensor-based all_reduce.
		- all_gather_object requires Python serialization (pickle) - ~1-5ms per call
		- all_reduce on tensors is pure NCCL - ~0.1ms per call

		Returns:
			(global_completed_uuids, active_decode_uuids) - both sorted by global_idx
		"""
		if not decode_uuids:
			return set(), []

		# Build global_idx to uuid mapping for decode candidates
		idx_to_uuid = {}
		uuid_to_idx = {}
		for uuid in decode_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is not None:
				idx_to_uuid[seq.global_idx] = uuid
				uuid_to_idx[uuid] = seq.global_idx

		if not idx_to_uuid:
			return set(), []

		# Get max global_idx to size the tensor
		max_idx = max(idx_to_uuid.keys())

		# Create completion tensor: 1 = completed, 0 = not completed
		# Each rank marks its LOCAL sequences' completion status
		completion_tensor = torch.zeros(max_idx + 1, dtype=torch.int32, device=self.torch_device)

		for uuid in decode_uuids:
			if uuid in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				if seq is not None and uuid in uuid_to_idx:
					is_completed = (seq.status == SequenceStatus.COMPLETED or seq.eos_reached)
					if is_completed:
						completion_tensor[uuid_to_idx[uuid]] = 1

		# all_reduce with MAX: if ANY rank marks a sequence complete, result is 1
		dist.all_reduce(completion_tensor, op=dist.ReduceOp.MAX)

		# Decode back to UUIDs
		global_completed = set()
		active_uuids = []

		# Sort by global_idx for deterministic ordering
		for global_idx in sorted(idx_to_uuid.keys()):
			uuid = idx_to_uuid[global_idx]
			if completion_tensor[global_idx].item() == 1:
				global_completed.add(uuid)
				# Update local sequence status
				seq = self.global_batch.get_sequence(uuid)
				if seq is not None:
					seq.eos_reached = True
					if seq.status != SequenceStatus.COMPLETED:
						try:
							self.global_batch.update_status(uuid, SequenceStatus.COMPLETED)
						except ValueError:
							pass
			else:
				active_uuids.append(uuid)

		return global_completed, active_uuids

	def _sync_decode_uuids_tensor(
		self,
		decode_uuids: List[str],
	) -> List[str]:
		"""
		Synchronize decode_uuids across all ranks using tensor operations.

		Uses global_idx as the common identifier and all_reduce to find intersection.
		Returns sorted list of UUIDs that ALL ranks agree on.
		"""
		if not decode_uuids:
			return []

		# Build global_idx to uuid mapping
		idx_to_uuid = {}
		uuid_to_idx = {}
		for seq in self.global_batch:
			idx_to_uuid[seq.global_idx] = seq.uuid
			uuid_to_idx[seq.uuid] = seq.global_idx

		max_idx = max(idx_to_uuid.keys()) if idx_to_uuid else 0

		# Create presence tensor: 1 = in decode_uuids, 0 = not
		presence_tensor = torch.zeros(max_idx + 1, dtype=torch.int32, device=self.torch_device)
		for uuid in decode_uuids:
			if uuid in uuid_to_idx:
				presence_tensor[uuid_to_idx[uuid]] = 1

		# all_reduce with MIN: only sequences present on ALL ranks will have value world_size
		# First broadcast local counts, then sum
		dist.all_reduce(presence_tensor, op=dist.ReduceOp.MIN)

		# Extract UUIDs where all ranks agree (value == 1 after MIN means all had 1)
		synced_uuids = []
		for global_idx in sorted(idx_to_uuid.keys()):
			if presence_tensor[global_idx].item() == 1:
				synced_uuids.append(idx_to_uuid[global_idx])

		return synced_uuids

	# ============ Tokenization and Assignment ============

	def _tokenize_global_batch(self) -> None:
		"""
		Tokenize all sequences in the global batch without truncation.
		The max_prompt_length is determined dynamically as the longest prompt.

		PARALLEL TOKENIZATION: Each rank tokenizes a subset of sequences, then
		results are gathered across all ranks. This reduces tokenization time
		by ~world_size and keeps NCCL alive during the process (prevents
		NCCL HeartbeatMonitor timeout for large batches).

		After tokenization, completion criteria uses:
		- EOS token reached, OR
		- decoded_length >= max_decoding_length, OR
		- prompt_length + decoded_length >= model_context_length
		"""
		if self.global_batch is None:
			raise RuntimeError("Global batch not initialized")

		# Phase 1: PARALLEL batch tokenization across ranks
		# Each rank tokenizes sequences[rank::world_size] to divide the work
		all_texts = [seq.text for seq in self.global_batch]
		num_sequences = len(all_texts)

		# Determine this rank's subset of sequences to tokenize
		my_indices = list(range(self.rank, num_sequences, self.world_size))
		my_texts = [all_texts[i] for i in my_indices]

		if self.rank == 0:
			logging.info(
				f"Parallel tokenizing {num_sequences} sequences across {self.world_size} ranks "
				f"(~{len(my_indices)} per rank)..."
			)

		tokenize_start = time.perf_counter()

		# Each rank tokenizes its subset
		if my_texts:
			my_batch_tokenized = self.tokenizer(
				my_texts,
				return_tensors="pt",
				truncation=False,  # No truncation - keep full input
				padding=True,      # Pad to longest in this subset
				return_attention_mask=True,
			)
			# Extract individual sequences from the batch result
			my_tokenized = []
			for i in range(len(my_texts)):
				actual_len = int(my_batch_tokenized["attention_mask"][i].sum().item())
				my_tokenized.append({
					"global_idx": my_indices[i],
					"input_ids": my_batch_tokenized["input_ids"][i, :actual_len].tolist(),
					"length": actual_len,
				})
		else:
			my_tokenized = []

		local_tokenize_time = time.perf_counter() - tokenize_start
		logging.debug(f"Rank {self.rank}: Local tokenization of {len(my_texts)} sequences in {local_tokenize_time:.2f}s")

		# DEBUG: Print tokenized prompts
		if os.environ.get("BATCHGEN_DEBUG_TOKENIZE", "0") == "1" and self.rank == 0 and my_tokenized:
			print(f"\n[TOKENIZE DEBUG] === First 3 tokenized prompts ===")
			for i in range(min(3, len(my_tokenized))):
				item = my_tokenized[i]
				token_ids = item["input_ids"]
				print(f"\n[TOKENIZE DEBUG] Sequence {item['global_idx']} (length={item['length']})")
				# Show first 50 tokens
				print(f"[TOKENIZE DEBUG] First 50 tokens: {token_ids[:50]}")
				# Show last 50 tokens (includes question end)
				print(f"[TOKENIZE DEBUG] Last 50 tokens: {token_ids[-50:]}")
				# Decode first 200 chars of prompt
				try:
					decoded_start = self.tokenizer.decode(token_ids[:100])
					decoded_end = self.tokenizer.decode(token_ids[-100:])
					print(f"[TOKENIZE DEBUG] Start of prompt (decoded): {repr(decoded_start[:300])}")
					print(f"[TOKENIZE DEBUG] End of prompt (decoded): {repr(decoded_end[-300:])}")
				except Exception as e:
					print(f"[TOKENIZE DEBUG] Decode error: {e}")
				# Check for special tokens
				special_token_ids = [199998, 199999, 200000, 200001, 200002, 200003, 200004, 200005, 200006, 200007, 200008, 200012]
				found_special = [tid for tid in token_ids if tid in special_token_ids]
				if found_special:
					print(f"[TOKENIZE DEBUG] Special tokens found: {found_special}")

		# Phase 1.5: Gather all tokenized results to all ranks
		# This keeps NCCL alive and shares results efficiently
		gather_start = time.perf_counter()
		all_tokenized_lists = [None] * self.world_size
		dist.all_gather_object(all_tokenized_lists, my_tokenized)
		gather_time = time.perf_counter() - gather_start

		# Merge results from all ranks, indexed by global_idx
		# Store only lightweight data (lists), not tensors, to minimize memory
		tokenized_by_idx = {}
		for rank_results in all_tokenized_lists:
			if rank_results:
				for item in rank_results:
					tokenized_by_idx[item["global_idx"]] = item

		# Free the gathered lists immediately
		del all_tokenized_lists

		total_tokenize_time = time.perf_counter() - tokenize_start
		if self.rank == 0:
			logging.info(
				f"Parallel tokenization complete in {total_tokenize_time:.2f}s "
				f"(local: {local_tokenize_time:.2f}s, gather: {gather_time:.2f}s)"
			)

		# Phase 2: Find the longest prompt length to use as max_prompt_length
		# Use lightweight length field instead of creating tensors
		prompt_lengths = [tokenized_by_idx[i]["length"] for i in range(num_sequences)]
		max_prompt_length = max(prompt_lengths)

		# Warn if any prompt exceeds model context length
		if max_prompt_length >= self.model_context_length:
			logging.warning(
				f"Rank {self.rank}: Longest prompt ({max_prompt_length} tokens) exceeds or equals "
				f"model context length ({self.model_context_length}). Some sequences may not decode."
			)

		# Update self.max_input_length to the actual longest prompt
		# This is used for attention mask shape: [bsz, max_prompt_length + max_decoding_length]
		self.max_input_length = max_prompt_length
		logging.info(
			f"Rank {self.rank}: Dynamic max_prompt_length set to {max_prompt_length} "
			f"(prompt lengths: min={min(prompt_lengths)}, max={max(prompt_lengths)})"
		)

		# Phase 3: Create per-sequence tensors sized to their actual prompt length
		# MEMORY OPTIMIZATION: Create tensors one at a time directly from gathered lists,
		# avoiding intermediate tensor storage. Each sequence only needs space for its
		# own prompt + decoding, critical for long-tailed distributions.
		for seq in self.global_batch:
			# CRITICAL: Use seq.global_idx to lookup, NOT enumeration index
			# tokenized_by_idx is keyed by global_idx from parallel tokenization
			item = tokenized_by_idx[seq.global_idx]
			input_ids_list = item["input_ids"]
			actual_prompt_len = item["length"]

			# Validation: ensure token list length matches stored length
			if len(input_ids_list) != actual_prompt_len:
				logging.error(
					f"Rank {self.rank}: Token length mismatch for seq {seq.global_idx}: "
					f"list_len={len(input_ids_list)}, stored_len={actual_prompt_len}"
				)
				actual_prompt_len = len(input_ids_list)  # Use actual list length

			# Each sequence gets its own sized tensor: actual_prompt_len + max_decoding_length
			# Capped by model context length to avoid wasting memory on impossible decoding
			seq_extended_size = min(
				actual_prompt_len + self.max_decoding_length,
				self.model_context_length
			)

			input_ids_extended = torch.zeros((1, seq_extended_size), dtype=torch.long)
			attention_mask_extended = torch.zeros((1, seq_extended_size), dtype=torch.int64)

			# Copy the actual tokens directly from list (left-aligned, no truncation)
			input_ids_extended[0, :actual_prompt_len] = torch.tensor(input_ids_list, dtype=torch.long)
			# CRITICAL: Set attention mask to exactly match input_ids length
			# This ensures attention_mask.sum() == prompt_length == current_context_length
			attention_mask_extended[0, :actual_prompt_len] = 1

			seq.input_ids = input_ids_extended
			seq.attention_mask = attention_mask_extended
			seq.decoded_tokens = torch.zeros(1, self.max_decoding_length, dtype=torch.int64)

			# Free the tokenized data for this sequence immediately
			del tokenized_by_idx[seq.global_idx]

			seq.prompt_length = actual_prompt_len
			seq.current_context_length = actual_prompt_len
			# kv_token_budget matches the tensor size
			seq.kv_token_budget = seq_extended_size

		logging.info(f"Rank {self.rank}: Tokenized {len(self.global_batch)} sequences")

	def _assign_sequences_to_ranks(self) -> None:
		"""
		Assign sequences to ranks balancing predicted attention tile workload.
		All ranks execute this identically to maintain consistent assignment.

		Uses greedy bin-packing: sort sequences by predicted tiles (descending),
		then assign each to the rank with fewest total tiles. This balances
		attention compute across ranks, reducing synchronization wait time.
		"""
		if self.global_batch is None:
			raise RuntimeError("Global batch not initialized")

		# Sort sequences by predicted total context (descending) for better bin-packing
		# Larger sequences first ensures better balance
		sequences = list(self.global_batch)
		sequences.sort(
			key=lambda s: s.prompt_length + s.max_decode_length,
			reverse=True
		)

		# Track total tiles per rank (attention tile = 128 tokens)
		TILE_SIZE = 128
		rank_tiles = [0] * self.world_size

		for seq in sequences:
			# Predict total context length at decode completion
			predicted_context = seq.prompt_length + seq.max_decode_length
			predicted_tiles = (predicted_context + TILE_SIZE - 1) // TILE_SIZE  # ceil_div

			# Assign to rank with fewest tiles (greedy)
			target_rank = rank_tiles.index(min(rank_tiles))
			self.global_batch.assign_rank(seq.uuid, target_rank)
			rank_tiles[target_rank] += predicted_tiles

		# Log balance quality
		my_seqs = self.global_batch.get_sequences_for_rank(self.rank)
		if self.rank == 0:
			imbalance = (max(rank_tiles) - min(rank_tiles)) / max(rank_tiles) * 100 if max(rank_tiles) > 0 else 0
			logging.info(
				f"Workload distribution (tiles per rank): {rank_tiles}, "
				f"imbalance: {imbalance:.1f}%"
			)
		logging.info(
			f"Rank {self.rank}: Assigned {len(my_seqs)} sequences, "
			f"tiles={rank_tiles[self.rank]}"
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
		self._free_local_indices: Set[int] = set()  # Reset free list
		self._next_local_idx = len(my_uuids)  # Next available index after initial assignment

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
		expected_count = sum(
			1 for seq in self.global_batch if seq.assigned_rank == self.rank
		)

		if len(my_uuids) != expected_count:
			logging.error(
				f"Rank {self.rank}: CRITICAL MISMATCH - expected {expected_count} sequences "
				f"but got {len(my_uuids)} from get_sequences_for_rank!"
			)

		logging.info(
			f"Rank {self.rank}: Built local query_book with {len(self.query_book)} entries "
			f"(global_batch has {len(self.global_batch)} sequences)"
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
		
		if self.rank == 0:
			logging.info(
				f"[PREFILL] Selected {len(prefill_batch)} sequences, "
				f"per-node pages: {node_pages_used}"
			)

		return prefill_batch

	def _put_sequences_on_hold(self, uuids: List[str]) -> None:
		"""Move IN_DECODE sequences to ON_HOLD, freeing GPU KV but keeping host KV."""
		if not uuids:
			return

		if self.rank == 0:
			logging.info(
				f"[WATERMARK] Putting {len(uuids)} sequences ON_HOLD"
			)

		# DIAGNOSTIC: Verify attention_mask consistency BEFORE going ON_HOLD
		# This helps identify if the mask-context mismatch is introduced before or after ON_HOLD
		onhold_mask_diag = []
		for uuid in uuids:
			if uuid in self._uuid_to_local_map:
				local_idx = self._uuid_to_local_map[uuid]
				seq = self.global_batch.get_sequence(uuid)
				ctx_len = seq.current_context_length
				full_mask = self.query_book[local_idx].encoded["attention_mask"]
				mask_ones = full_mask[:, :ctx_len].sum().item()
				if mask_ones != ctx_len:
					onhold_mask_diag.append({
						'uuid': uuid[:8],
						'ctx_len': ctx_len,
						'mask_ones': int(mask_ones),
						'decoded_len': seq.decoded_length,
						'diff': int(mask_ones - ctx_len),
					})
		if onhold_mask_diag:
			logging.error(
				f"Rank {self.rank}: {len(onhold_mask_diag)} sequences have attention_mask mismatch BEFORE going ON_HOLD. First 5: {onhold_mask_diag[:5]}"
			)

		# CRITICAL FIX: Sync sequence metadata BEFORE putting on hold
		# This ensures all ranks have consistent current_context_length values
		# which is essential for correct KV migration validation later
		self._sync_sequence_metadata(uuids)

		# Free GPU pages for these sequences
		# CRITICAL FIX: GPU KV manager uses global_idx (not local_idx) as sequence ID
		if hasattr(self, 'gpu_paged_kv_cache_manager') and self.gpu_paged_kv_cache_manager:
			global_seq_ids = []
			for uuid in uuids:
				seq = self.global_batch.get_sequence(uuid)
				if seq.assigned_rank == self.rank:
					# Verify sequence is in local map (should be for IN_DECODE sequences)
					if uuid in self._uuid_to_local_map:
						global_seq_ids.append(seq.global_idx)  # Use global_idx, not local_idx!

			if global_seq_ids:
				self.gpu_paged_kv_cache_manager.free_pages_for_sequences(global_seq_ids)
				# Also remove from tracking set
				for uuid in uuids:
					seq = self.global_batch.get_sequence(uuid)
					if seq.assigned_rank == self.rank:
						self._sequences_with_gpu_kv.discard(uuid)

		# Update sequence status and reset GPU allocation
		for uuid in uuids:
			seq = self.global_batch.get_sequence(uuid)
			seq.reset_gpu_allocation()  # Reset gpu_pages_allocated = 0
			self.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)

		# Synchronize state across all ranks
		dist.barrier()

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
		
		# Get GPU page capacity - GPU KV manager must be initialized before batch selection
		# (model loading and GPU KV init happen in generate() BEFORE this call)
		if self.gpu_paged_kv_cache_manager is None or not self.gpu_paged_kv_cache_manager.is_initialized:
			raise RuntimeError(
				"GPU KV manager must be initialized before _prepare_decode_batch(). "
				"Ensure _load_decode_model() and _init_gpu_kv_with_actual_size() are called first."
			)
		total_pages = self.gpu_paged_kv_cache_manager.get_stats().num_total_pages
		
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

		if self.rank == 0:
			logging.info(
				f"[DECODE] Prepared batch: {len(decode_batch)} sequences"
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
			
			# Update status globally AND reset GPU allocation state
			# CRITICAL FIX: Must call reset_gpu_allocation() so sequences get proper initial buffer on resume
			for uuid in global_onhold:
				seq = self.global_batch.get_sequence(uuid)
				if seq.gpu_pages_allocated > 0 or seq.had_initial_gpu_reservation:
					if BATCHGEN_CB_DEBUG:
						logging.debug(
							f"Rank {self.rank}: Resetting GPU state for ON_HOLD seq {uuid[:8]}"
						)
					seq.reset_gpu_allocation()
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
		
		# DIAGNOSTIC: Log details for resuming sequences (decoded_length > 0)
		resuming_diag = []
		for local_idx in local_sequence_ids:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			if seq.decoded_length > 0:
				qb = self.query_book.get(local_idx)
				attn_mask_sum = qb.encoded["attention_mask"].sum().item() if qb else "N/A"
				resuming_diag.append({
					'uuid': uuid[:8],
					'decoded_len': seq.decoded_length,
					'ctx_len': seq.current_context_length,
					'prompt_len': seq.prompt_length,
					'attn_mask_sum': attn_mask_sum,
				})
		if resuming_diag and BATCHGEN_CB_DEBUG:
			logging.debug(
				f"Rank {self.rank}: Loading GPU KV for {len(resuming_diag)} resuming sequences. First 3: {resuming_diag[:3]}"
			)
		
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
		
		# DIAGNOSTIC: After load, verify loaded data matches expected context length
		post_load_diag = []
		for local_idx in local_sequence_ids:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			if seq.decoded_length > 0:  # Resuming sequence
				allocated_pages = seq.get_gpu_pages_for_two_page_buffer()
				allocated_tokens = allocated_pages * self.PAGE_SIZE
				expected_kv_tokens = seq.current_context_length
				post_load_diag.append({
					'uuid': uuid[:8],
					'decoded_len': seq.decoded_length,
					'ctx_len': expected_kv_tokens,
					'alloc_pages': allocated_pages,
					'alloc_tokens': allocated_tokens,
					'excess': allocated_tokens - expected_kv_tokens,
				})
		if post_load_diag and BATCHGEN_CB_DEBUG:
			logging.debug(
				f"Rank {self.rank}: Loaded {len(post_load_diag)} resuming sequences. First 3: {post_load_diag[:3]}"
			)
		
		# ← FIX: Update tracking state AFTER successful load
		for local_idx in local_sequence_ids:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			seq.gpu_pages_allocated = seq.get_gpu_pages_for_two_page_buffer()
			# Mark that this sequence has received its initial GPU reservation
			seq.mark_initial_gpu_reservation_done()
			self._sequences_with_gpu_kv.add(uuid)

	# ============ Batch Statistics ============

	def _log_batch_statistics(self) -> None:
		"""
		Log statistics about the completed batch including:
		- Global batch size
		- Prompt lengths: min, max, mean, median, P95, P99
		- Decoded token lengths: min, max, mean, median, P95, P99

		Only called from rank 0.
		"""
		if self.global_batch is None:
			return

		# Gather all sequences
		prompt_lengths = []
		decoded_lengths = []
		for seq in self.global_batch:
			prompt_lengths.append(seq.prompt_length)
			decoded_lengths.append(seq.decoded_length)

		if not prompt_lengths:
			logging.info("[BATCH STATS] No sequences in batch.")
			return

		# Convert to numpy for statistics
		prompt_arr = np.array(prompt_lengths)
		decoded_arr = np.array(decoded_lengths)

		# Compute statistics
		def compute_stats(arr: np.ndarray) -> dict:
			return {
				'min': int(np.min(arr)),
				'max': int(np.max(arr)),
				'mean': float(np.mean(arr)),
				'median': float(np.median(arr)),
				'p95': float(np.percentile(arr, 95)),
				'p99': float(np.percentile(arr, 99)),
			}

		prompt_stats = compute_stats(prompt_arr)
		decoded_stats = compute_stats(decoded_arr)
		batch_size = len(prompt_lengths)

		# Log formatted output
		logging.info(
			f"\n{'='*60}\n"
			f"BATCH STATISTICS\n"
			f"{'='*60}\n"
			f"  Global Batch Size: {batch_size}\n"
			f"\n"
			f"  Prompt Lengths:\n"
			f"    Min: {prompt_stats['min']:,}  Max: {prompt_stats['max']:,}\n"
			f"    Mean: {prompt_stats['mean']:,.1f}  Median: {prompt_stats['median']:,.1f}\n"
			f"    P95: {prompt_stats['p95']:,.1f}  P99: {prompt_stats['p99']:,.1f}\n"
			f"\n"
			f"  Decoded Token Lengths:\n"
			f"    Min: {decoded_stats['min']:,}  Max: {decoded_stats['max']:,}\n"
			f"    Mean: {decoded_stats['mean']:,.1f}  Median: {decoded_stats['median']:,.1f}\n"
			f"    P95: {decoded_stats['p95']:,.1f}  P99: {decoded_stats['p99']:,.1f}\n"
			f"{'='*60}"
		)

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

		# Initialize cumulative decode counters (persist across prefill/decode switches)
		self._cumulative_decode_iterations = 0
		self._cumulative_decode_boundaries = 0
		self._cumulative_boundary_ms = 0.0
		self._cumulative_forward_ms = 0.0
		
		# NOTE: torch.distributed health was already verified in _reset_for_new_batch() via
		# _ensure_dist_healthy(). This is just a sanity check - should never fail here.
		logging.info(f"Rank {self.rank}: Verifying distributed connections...")
		if not dist.is_initialized():
			raise RuntimeError(f"Rank {self.rank}: torch.distributed not initialized (should have been verified in _reset_for_new_batch)")
		if not self._check_and_reinit_pynccl():
			raise RuntimeError(f"Rank {self.rank}: Failed to ensure healthy PyNccl communicator")
		logging.info(f"Rank {self.rank}: Distributed connections verified")
		
		# Ensure communicator is ready
		if os.getenv("BATCHGEN_ENABLE_ALL_TO_ALL", "0") == "0":
			# Verify rank consistency
			if dist.is_initialized():
				assert self.rank == dist.get_rank(), \
					f"Rank mismatch: self.rank={self.rank}, dist.get_rank()={dist.get_rank()}"

			# Skip PyNccl initialization for single GPU (no inter-GPU communication needed)
			if self.world_size == 1:
				logging.debug("Single GPU mode: skipping PyNccl communicator initialization")
			else:
				comm_master_addr = os.getenv("COMM_MASTER_ADDR")

				# Coordinate PyNccl initialization across all ranks
				# Use all_reduce to check if ANY rank needs to (re)init the communicator
				need_init = 1 if self.comm is None else 0
				need_init_tensor = torch.tensor([need_init], dtype=torch.int32, device=self.torch_device)
				dist.all_reduce(need_init_tensor, op=dist.ReduceOp.MAX)
				any_rank_needs_init = need_init_tensor.item() > 0

				if any_rank_needs_init:
					# All ranks must participate in init - destroy any existing comm first
					if self.comm is not None:
						logging.info(f"Rank {self.rank}: Destroying existing comm for coordinated reinit")
						try:
							self.comm.destroy()
						except Exception:
							pass
						self.comm = None
						if hasattr(self, '_nccl_group') and self._nccl_group is not None:
							del self._nccl_group
							self._nccl_group = None

					device = torch.device("cuda", self.local_rank)

					if comm_master_addr is None:
						logging.warning(f"Rank {self.rank}: COMM_MASTER_ADDR not set, skipping PyNccl init")
					elif StatelessProcessGroup is not None and PyNcclCommunicator is not None:
						# Track port - incremented in _check_and_reinit_pynccl on failures
						if not hasattr(self, '_nccl_port'):
							self._nccl_port = 20003

						# Rank 0 finds an available port, then broadcasts to all ranks
						if self.rank == 0:
							try:
								self._nccl_port = _find_available_port(comm_master_addr, self._nccl_port)
								logging.debug(f"Rank 0: Found available port {self._nccl_port} for PyNccl")
							except RuntimeError as e:
								logging.error(f"Rank 0: Failed to find available port: {e}")
								raise

						# Broadcast the chosen port from rank 0 to all ranks
						port_tensor = torch.tensor([self._nccl_port], dtype=torch.int32, device=self.torch_device)
						dist.broadcast(port_tensor, src=0)
						self._nccl_port = port_tensor.item()

						# CRITICAL: Barrier before TCPStore creation to ensure rank 0 (the server)
						# is ready before other ranks try to connect. Different ranks may reach
						# this point at very different times due to tokenization workload.
						logging.debug(f"Rank {self.rank}: Waiting for all ranks before PyNccl init...")
						dist.barrier()

						try:
							logging.debug(f"Rank {self.rank}: Creating PyNccl communicator on port {self._nccl_port}")

							# Store group separately so we can properly destroy it on reinit
							self._nccl_group = StatelessProcessGroup.create(
								host=comm_master_addr,
								port=self._nccl_port,
								rank=self.rank,
								world_size=self.world_size,
								data_expiration_seconds=36000,  # 10 hours
							)
							self.comm = PyNcclCommunicator(
								group=self._nccl_group,
								device=device
							)
							# Only rank 0 logs at INFO level to reduce verbosity
							if self.rank == 0:
								logging.info(f"PyNccl communicator initialized on port {self._nccl_port}")
							else:
								logging.debug(f"Rank {self.rank}: PyNccl communicator initialized on port {self._nccl_port}")
						except Exception as e:
							logging.error(f"Rank {self.rank}: PyNccl communicator initialization failed - {e}")
							raise RuntimeError(f"Rank {self.rank}: PyNccl communicator initialization failed - {e}")

		iteration = 0

		# Continues until ALL sequences in the global batch are COMPLETED
		while not self.global_batch.all_completed():
			iteration += 1
			if self.rank == 0:
				logging.info(f"--- Iteration {iteration} ---")

			# NOTE: Watchdog is fed within prefill and decode loops, not here.
			# This ensures we only monitor the actual inference phases.

			# =================================================================
			# 1. PREFILL PHASE: Fill Host KV Cache
			# =================================================================
			if self.global_batch.has_queueing():
				dist.barrier()

				# CRITICAL FIX: Sync sequence metadata BEFORE rebalancing
				# After decode interruption or prefill completion, each rank has divergent
				# metadata for sequences it doesn't own locally. This sync ensures all ranks
				# have consistent current_context_length values before migration. PREFILLED
				# sequences must be synced because their attention mask has been updated
				# (prompt_len + 1) after prefill, and migration includes PREFILLED status.
				prefilled_uuids = [seq.uuid for seq in self.global_batch if seq.status == SequenceStatus.PREFILLED]
				on_hold_uuids = [seq.uuid for seq in self.global_batch if seq.status == SequenceStatus.ON_HOLD]
				in_decode_uuids = [seq.uuid for seq in self.global_batch if seq.status == SequenceStatus.IN_DECODE]
				all_active_uuids = prefilled_uuids + on_hold_uuids + in_decode_uuids
				if all_active_uuids:
					self._sync_sequence_metadata(all_active_uuids)
					logging.debug(f"Rank {self.rank}: Synced metadata for {len(all_active_uuids)} sequences before rebalance")

				# STEP 0: Rebalance host KV BEFORE batch selection
				# This ensures batch selection uses accurate post-migration capacities
				if self.enable_decode_preemption:
					rebalance_start = time.perf_counter()
					self._rebalance_host_kv()
					if self.rank == 0:
						logging.info(
							f"[PREFILL] Host KV rebalancing: {(time.perf_counter() - rebalance_start)*1000:.1f}ms"
						)

				prefill_uuids = self._prepare_prefill_batch()
				
				if prefill_uuids:
					if self.rank == 0:
						logging.info(f"[PREFILL] Starting for {len(prefill_uuids)} sequences")
					self._update_batch_status(prefill_uuids, SequenceStatus.IN_PREFILL)

					# A. Config Prefill (this adds new sequences to _uuid_to_local_map)
					config_start = time.perf_counter()
					self._config_prefill_for_batch(prefill_uuids)
					config_prefill_time += time.perf_counter() - config_start

					# Get local indices AFTER config (new sequences now in map)
					local_prefill_indices = self._get_local_indices_for_uuids(prefill_uuids)

					# B. Execute Prefill
					if local_prefill_indices:
						prefill_start = time.perf_counter()
						with torch.inference_mode():
							if self.enable_prepack:
								self.prefill_prepacked(local_prefill_indices)
							else:
								self.prefill(local_prefill_indices)
						prefill_time += time.perf_counter() - prefill_start

						# CRITICAL: Wait for all async KV offloads to complete before decode
						# The async_offload_layer_kv_to_host() calls during prefill are fire-and-forget.
						# Decode reads KV from host, so offloads MUST complete first.
						torch.cuda.synchronize(self.torch_device)

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
				# NOTE: Barrier removed - tensor sync operations below provide synchronization

				# ============ STEP A: Load model FIRST (needed for accurate GPU KV size) ============
				# Estimate max sequences per rank for buffer allocation
				# Use PREFILLED + ON_HOLD + IN_DECODE as upper bound
				prefilled_count = len(self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED))
				onhold_count = len(self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD))
				in_decode_count = len(self.global_batch.get_sequences_by_status(SequenceStatus.IN_DECODE))
				total_candidates = prefilled_count + onhold_count + in_decode_count
				# Estimate max per rank (ceiling division)
				max_num_seq_estimate = (total_candidates + self.world_size - 1) // self.world_size
				# Ensure at least some minimum
				max_num_seq_estimate = max(max_num_seq_estimate, 16)

				self._load_decode_model(max_num_seq_estimate, self.comm)

				# ============ STEP B: Init GPU KV with ACTUAL size ============
				# Only initializes if not already done; subsequent iterations skip
				self._init_gpu_kv_with_actual_size()

				# ============ STEP C: Prepare decode batch (uses real GPU KV capacity) ============
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

				# OPTIMIZATION: Use tensor-based sync instead of expensive all_gather_object
				# This reduces completion sync from ~5-10ms to ~0.2ms per decode iteration

				# Step 1: Sync decode_uuids across ranks using tensor operations
				# This ensures all ranks have the same decode candidates
				decode_uuids = self._sync_decode_uuids_tensor(decode_uuids)

				# Step 2: Sync completion status using tensor-based all_reduce
				# Returns (completed_set, active_list) - active_list is already sorted by global_idx
				global_completed, decode_uuids = self._sync_completion_status_tensor(decode_uuids)

				# Verification: Check count consistency (fast tensor operation)
				local_count = torch.tensor([len(decode_uuids)], dtype=torch.int64, device=self.torch_device)
				all_counts = [torch.zeros_like(local_count) for _ in range(self.world_size)]
				dist.all_gather(all_counts, local_count)
				counts = [int(t.item()) for t in all_counts]

				if len(set(counts)) > 1:
					# Rare case: still divergent after tensor sync, use tensor intersection
					logging.warning(
						f"Rank {self.rank}: decode_uuids DIVERGENT after tensor sync! counts={counts}. "
						f"Re-syncing with tensor intersection..."
					)
					decode_uuids = self._sync_decode_uuids_tensor(decode_uuids)
				
				if not decode_uuids:
					break
				
				self._update_batch_status(decode_uuids, SequenceStatus.IN_DECODE)
				local_decode_indices = self._get_local_indices_for_uuids(decode_uuids)

				# B. Config Decode
				config_start = time.perf_counter()
				self._config_decoding_for_batch(decode_uuids, local_decode_indices)
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

				# CRITICAL FIX: After decode returns (possibly due to watermark trigger),
				# check if there are queued sequences waiting for prefill.
				# If so, break out of inner decode loop to allow outer loop to enter prefill.
				if self.global_batch.has_queueing():
					if self.rank == 0:
						num_queued = len(self.global_batch.get_sequences_by_status(SequenceStatus.QUEUEING))
						logging.info(f"[DECODE] Breaking for prefill - {num_queued} sequences queued")
					break  # Exit inner decode while loop, outer loop will check has_queueing()
		
		# Log timing stats
		generation_time = time.perf_counter() - generation_start_time
		phase_switching_time = config_prefill_time + config_decode_time

		# Compute throughput metrics from all sequences
		total_prompt_tokens = 0
		total_decoded_tokens = 0
		num_sequences = 0
		if self.global_batch is not None:
			for seq in self.global_batch:
				total_prompt_tokens += seq.prompt_length
				total_decoded_tokens += seq.decoded_length
				num_sequences += 1

		# Calculate throughput (tokens/second)
		prefill_throughput = total_prompt_tokens / prefill_time if prefill_time > 0 else 0
		decode_throughput = total_decoded_tokens / decoding_time if decoding_time > 0 else 0
		total_tokens = total_prompt_tokens + total_decoded_tokens
		overall_throughput = total_tokens / generation_time if generation_time > 0 else 0

		if self.rank == 0:
			logging.info(
				f"Generation completed:\n"
				f"  Prefill total time: {prefill_time:.1f}s\n"
				f"  Decoding total time: {decoding_time:.1f}s\n"
				f"  Generation total time: {generation_time:.1f}s\n"
				f"  Phase switching time: {phase_switching_time:.1f}s\n"
				f"  Config prefill time: {config_prefill_time:.1f}s\n"
				f"  Config decoding time: {config_decode_time:.1f}s\n"
				f"  ---\n"
				f"  Total sequences: {num_sequences}\n"
				f"  Total prompt tokens: {total_prompt_tokens:,}\n"
				f"  Total decoded tokens: {total_decoded_tokens:,}\n"
				f"  Prefill throughput: {prefill_throughput:,.1f} tokens/s\n"
				f"  Decode throughput: {decode_throughput:,.1f} tokens/s\n"
				f"  Overall throughput: {overall_throughput:,.1f} tokens/s"
			)

			# Compute and log batch statistics
			self._log_batch_statistics()

		# ============ Gather Results in Original Order ============
		# NOTE: After migrations, sequences may have moved between ranks.
		# Iterate over actual entries in _local_to_uuid_map (not sequential range)
		# to handle cases where local indices were freed or added during migration.
		res_with_idx = []
		for local_idx, uuid in self._local_to_uuid_map.items():
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				logging.warning(f"Rank {self.rank}: Sequence {uuid} not found in global_batch during result gathering")
				continue
			global_idx = seq.global_idx
			if local_idx not in self.query_book:
				logging.warning(f"Rank {self.rank}: query_book missing for local_idx={local_idx}, uuid={uuid[:8]}...")
				continue
			decoded_tokens = self.query_book[local_idx].decoded_tokens
			res_with_idx.append((global_idx, decoded_tokens))

		all_results = [None] * self.world_size
		dist.all_gather_object(all_results, res_with_idx)

		all_results = [item for sublist in all_results for item in sublist]
		all_results.sort(key=lambda x: x[0])

		# Decode tokens to strings (only on rank 0 since only rank 0 returns results)
		decoded_strings = []
		if self.rank == 0:
			decode_start = time.perf_counter()
			for global_idx, tokens in all_results:
				decoded_str = self._decode_tokens_to_string(tokens)
				decoded_strings.append(decoded_str)
			decode_time = time.perf_counter() - decode_start
			logging.info(f"Detokenization complete: {len(decoded_strings)} sequences in {decode_time:.2f}s")

			# Log decode timing stats (GPT-OSS specific)
			self._log_decode_timing()

		dist.barrier()
		self._batch_completed = True

		if self.rank == 0:
			return decoded_strings
		else:
			return []

	def _decode_tokens_to_string(self, tokens: torch.Tensor, min_tokens: int = 1) -> str:
		"""Decode token IDs to string, stopping at first EOS token.

		Args:
			tokens: Tensor of token IDs, shape [1, seq_len] or [seq_len]
			min_tokens: Minimum tokens before considering EOS (to avoid empty outputs)

		Returns:
			Decoded string, truncated at first valid EOS position
		"""
		# Flatten to 1D if needed
		if tokens.dim() > 1:
			tokens = tokens.squeeze(0)

		tokens_list = tokens.tolist()

		# Find first EOS token position (after min_tokens)
		eos_positions = [i for i, t in enumerate(tokens_list) if t == self.eos_token_id and i >= min_tokens]

		if eos_positions:
			end_pos = eos_positions[0]
		else:
			# No EOS found, use all non-zero tokens
			# Find last non-zero token
			non_zero = [i for i, t in enumerate(tokens_list) if t != 0]
			end_pos = non_zero[-1] + 1 if non_zero else len(tokens_list)

		# Decode tokens up to end position
		return self.tokenizer.decode(tokens_list[:end_pos], skip_special_tokens=False)

	# ============ Phase Configuration ============

	def _config_prefill_for_batch(self, prefill_uuids: List[str]) -> None:
		"""Configure prefill phase for a batch of sequences."""
		start_time = time.perf_counter()
		if self.rank == 0:
			logging.info(
				f"[PREFILL] Configuring prefill phase for {len(prefill_uuids)} sequences"
			)

		# DIAGNOSTIC: Log state of IN_DECODE/ON_HOLD sequences before prefill config
		# This helps track KV corruption issues during decode→prefill→decode transitions
		in_decode = self.global_batch.get_sequences_by_status(SequenceStatus.IN_DECODE)
		on_hold = self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
		prefilling = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		if (in_decode or on_hold) and BATCHGEN_CB_DEBUG:
			logging.debug(
				f"Rank {self.rank}: _config_prefill_for_batch called while "
				f"{len(in_decode)} IN_DECODE, {len(on_hold)} ON_HOLD, {len(prefilling)} PREFILLED sequences exist. "
				f"This is a decode→prefill transition."
			)
			# Log details of sequences that will be affected
			for uuid in (in_decode + on_hold)[:5]:
				seq = self.global_batch.get_sequence(uuid)
				logging.debug(
					f"Rank {self.rank}: Affected seq {seq.uuid[:8]}: "
					f"status={seq.status.name}, decoded_len={seq.decoded_length}, "
					f"ctx_len={seq.current_context_length}, gpu_pages={seq.gpu_pages_allocated}, "
					f"had_initial={seq.had_initial_gpu_reservation}"
				)

		# CRITICAL FIX: Flush pending KV append tasks before destroying GPU cache
		# Without this, async KV writes may be in-flight when GPU cache is destroyed
		if hasattr(self, '_pending_kv_append_tasks') and self._pending_kv_append_tasks:
			logging.info(
				f"Rank {self.rank}: Flushing {len(self._pending_kv_append_tasks)} pending KV append tasks before prefill config"
			)
			self._wait_pending_kv_append_tasks()
			torch.cuda.synchronize(self.torch_device)

		# NOTE: Rebalancing is now done BEFORE _prepare_prefill_batch() in the main loop
		# to ensure batch selection uses accurate post-migration capacities.

		# CRITICAL: Deep free decode model memory BEFORE configuring prefill (Bug Fix 7)
		# This mirrors the cleanup done in _load_decode_model() for prefill→decode transitions
		# Without this, decode model (~92 GB) stays in memory when prefill model loads → OOM
		logging.info("Deep freeing model memory before prefill config...")
		self.deep_free_model_memory()

		# CRITICAL: Destroy GPU KV cache BEFORE configure_prefill (Bug Fix 7.2)
		# The GPU KV cache holds ~20-30GB that must be freed before loading prefill model
		# Previously this was called AFTER configure_prefill() which caused OOM
		self._destroy_gpu_paged_kv_cache()

		# STEP 1: Configure model for prefill
		self.model, self.weight_copy_task = self.parallel_manager.configure_prefill()
		self.set_phase("prefill")

		self.core_engine.stop_h2d_worker()
		self.core_engine.clear_weight_copy_queue()
		self.core_engine.reset_prefill_buffer()
		self.core_engine.set_weight_copy_queue(self.weight_copy_task)
		self.core_engine.start_h2d_worker()

		# NOTE: _destroy_gpu_paged_kv_cache() moved before configure_prefill() (Bug Fix 7.2)

		# STEP 3: Allocate host KV pages for new sequences (only THIS RANK's sequences)
		# Check by assigned_rank, NOT by _uuid_to_local_map (which may not have new sequences yet)
		my_prefill_uuids = []
		for uuid in prefill_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq.assigned_rank == self.rank:
				my_prefill_uuids.append(uuid)
				# Add to local maps if not already present (for new sequences)
				if uuid not in self._uuid_to_local_map:
					# O(1) allocation: prefer reusing freed indices, otherwise use next available
					if self._free_local_indices:
						new_local_idx = self._free_local_indices.pop()
					else:
						new_local_idx = self._next_local_idx
						self._next_local_idx += 1
					self._uuid_to_local_map[uuid] = new_local_idx
					self._local_to_uuid_map[new_local_idx] = uuid
					logging.debug(
						f"Rank {self.rank}: Added new sequence {uuid[:8]}... to local maps "
						f"(local_idx={new_local_idx})"
					)

		if my_prefill_uuids:
			global_sequence_ids = []
			sequence_tokens = []

			for uuid in my_prefill_uuids:
				seq = self.global_batch.get_sequence(uuid)
				global_sequence_ids.append(seq.global_idx)
				sequence_tokens.append(seq.kv_token_budget)

			logging.debug(
				f"Rank {self.rank}: Registering {len(global_sequence_ids)} sequences for host KV"
			)

			self.core_engine.host_paged_kv_worker_view.register_sequences(global_sequence_ids)
			self.core_engine.host_paged_kv_worker_view.allocate_pages_for_sequences(
				list(zip(global_sequence_ids, sequence_tokens))
			)

			kv_stats = self.core_engine.host_paged_kv_worker_view.get_stats()
			if self.rank == 0:
				logging.info(f"[PREFILL] Host KV allocated: {kv_stats.num_used_pages}/{kv_stats.num_total_pages} pages")

		if self.rank == 0:
			logging.info(f"[PREFILL] Config completed: {(time.perf_counter() - start_time)*1000:.1f}ms")

	def _load_decode_model(self, max_num_seq: int, comm=None) -> None:
		"""
		Load model for decoding phase. Must be called ONCE at the start of decode phase,
		BEFORE batch selection, so we know actual GPU KV capacity.

		Uses unified configure_decoding() which handles all scenarios:
		- Multi-node (world_size > 8): all experts persistent
		- Single-node with EP offloading: partial persistence based on offloading_ratio
		- Single-node without offloading: all experts persistent

		Args:
			max_num_seq: Maximum number of sequences per rank for buffer allocation.
			comm: NCCL communicator for distributed MoE forward.
		"""
		self.deep_free_model_memory()
		self.init_nvshmem()

		# Unified method handles all deployment scenarios
		self.model, self.weight_copy_task = self.parallel_manager.configure_decoding(
			padding_bsz=max_num_seq, comm=comm
		)
		self.set_phase("decode")
		self.core_engine.stop_h2d_worker()
		self.core_engine.clear_kv_copy_queue()
		self.core_engine.clear_weight_copy_queue()
		self.core_engine.reset_decoding_buffer()

		# Only start H2D worker if there are experts to offload
		if self.weight_copy_task.get("routed_expert"):
			self.core_engine.set_weight_copy_queue(self.weight_copy_task)
			self.core_engine.start_h2d_worker()

		if self.rank == 0:
			logging.info(f"[DECODE] Model loaded for decoding phase")

	def _init_gpu_kv_with_actual_size(self) -> None:
		"""
		Calculate actual GPU KV size AFTER model loading and initialize the manager.
		This replaces the theoretical estimation - must be called after _load_decode_model().

		Only runs the full calculation and initialization on the first call;
		subsequent calls skip if the manager is already initialized.
		"""
		# Skip if GPU KV manager is already initialized (subsequent decode iterations)
		if self.gpu_paged_kv_cache_manager is not None and self.gpu_paged_kv_cache_manager.is_initialized:
			return

		# First time: Calculate actual GPU KV size
		torch.cuda.synchronize(self.torch_device)
		torch.cuda.empty_cache()

		free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(self.local_rank)
		free_mem_gb = free_mem_bytes / (1024 ** 3)
		total_mem_gb = total_mem_bytes / (1024 ** 3)
		used_mem_gb = total_mem_gb - free_mem_gb

		# Formula: gpu_kv_cache = total * frac - used
		new_gpu_kv_cache_size = total_mem_gb * self.gpu_memory_frac - used_mem_gb
		if new_gpu_kv_cache_size > 0:
			self.gpu_kv_cache_size_gb = new_gpu_kv_cache_size
		else:
			# Fallback to minimum
			self.gpu_kv_cache_size_gb = 1.0
			if self.rank == 0:
				logging.warning(
					f"[GPU-KV] Calculated size non-positive ({new_gpu_kv_cache_size:.2f} GB). "
					f"Using minimum 1 GB."
				)

		if self.rank == 0:
			logging.info(
				f"[GPU-KV] Actual size after model loading: {self.gpu_kv_cache_size_gb:.2f} GB "
				f"(total: {total_mem_gb:.2f} GB × frac: {self.gpu_memory_frac} - used: {used_mem_gb:.2f} GB)"
			)

		# Broadcast to ensure all ranks use same value
		size_tensor = torch.tensor([self.gpu_kv_cache_size_gb], dtype=torch.float32, device=self.torch_device)
		dist.broadcast(size_tensor, src=0)
		self.gpu_kv_cache_size_gb = float(size_tensor.item())

		# Initialize GPU KV manager with actual size
		self._initialize_gpu_kv_manager_fixed_size()

		if self.rank == 0:
			stats = self.gpu_paged_kv_cache_manager.get_stats()
			logging.info(f"[GPU-KV] Initialized: {self.gpu_kv_cache_size_gb:.2f} GB, {stats.num_total_pages} pages")

	def _config_decoding_for_batch(
		self,
		decode_uuids: List[str],
		local_decode_indices: List[int]
	) -> None:
		"""
		Configure decoding for a specific batch - allocates GPU KV pages.

		NOTE: This method is SIMPLIFIED - model loading and GPU KV manager init
		now happen earlier in generate() via _load_decode_model() and
		_init_gpu_kv_with_actual_size(). This method only handles:
		1. Context length repair
		2. Validation/diagnostics
		3. GPU KV page allocation
		"""
		start_time = time.perf_counter()
		
		# ============ CRITICAL FIX: Repair current_context_length for ALL sequences FIRST ============
		# This must happen BEFORE any validation or diagnostics that read current_context_length.
		# The root cause of ctx_len=0 bug is that current_context_length can become stale during
		# decode→prefill→decode transitions, especially after migrations.
		# The fix: current_context_length = prompt_length + decoded_length is ALWAYS the correct value
		# for sequences that have started decoding (decoded_length > 0 or have been prefilled).
		ctx_len_repaired_count = 0
		for uuid in decode_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				continue
			
			# Compute the correct context length
			# For sequences with decoded tokens: ctx_len = prompt_length + decoded_length
			# For freshly prefilled sequences: ctx_len should equal prompt_length (decoded_length=0)
			expected_ctx = seq.prompt_length + seq.decoded_length
			
			# Repair if mismatched
			if seq.current_context_length != expected_ctx:
				old_ctx = seq.current_context_length
				seq.current_context_length = expected_ctx
				ctx_len_repaired_count += 1
				if old_ctx == 0 or abs(old_ctx - expected_ctx) > 100:
					# Only log significant mismatches to avoid log spam
					logging.warning(
						f"Rank {self.rank}: Repaired {uuid[:8]} gid={seq.global_idx}: "
						f"ctx_len {old_ctx} → {expected_ctx} (prompt={seq.prompt_length}, decoded={seq.decoded_length})"
					)
		
		if ctx_len_repaired_count > 0:
			logging.info(
				f"Rank {self.rank}: Repaired current_context_length for {ctx_len_repaired_count}/{len(decode_uuids)} sequences"
			)
		
		# ============ END CRITICAL FIX ============
		
		# VALIDATION: Verify decode_uuids consistency across all ranks
		local_uuid_count = torch.tensor([len(decode_uuids)], dtype=torch.int64, device=self.torch_device)
		all_uuid_counts = [torch.zeros_like(local_uuid_count) for _ in range(self.world_size)]
		dist.all_gather(all_uuid_counts, local_uuid_count)
		uuid_counts = [int(t.item()) for t in all_uuid_counts]
		
		if len(set(uuid_counts)) > 1:
			logging.error(
				f"Rank {self.rank}: CRITICAL - decode_uuids count mismatch at _config_decoding_for_batch entry! Counts: {uuid_counts}."
			)
		
		# DIAGNOSTIC: Log sequence states at decode config entry
		# This helps identify KV corruption issues during prefill→decode transitions
		resuming_seqs = []
		fresh_seqs = []
		for uuid in decode_uuids:
			seq = self.global_batch.get_sequence(uuid)
			
			seq_info = {
				'uuid': seq.uuid[:8],
				'global_idx': seq.global_idx,
				'status': seq.status.name,
				'decoded_length': seq.decoded_length,
				'current_context_length': seq.current_context_length,
				'gpu_pages_allocated': seq.gpu_pages_allocated,
				'had_initial_gpu_reservation': seq.had_initial_gpu_reservation,
			}
			if seq.decoded_length > 0:
				resuming_seqs.append(seq_info)
			else:
				fresh_seqs.append(seq_info)
		
		if resuming_seqs and BATCHGEN_CB_DEBUG:
			logging.debug(
				f"Rank {self.rank}: _config_decoding_for_batch: "
				f"{len(resuming_seqs)} RESUMING sequences (decoded_length > 0). "
				f"First 5: {resuming_seqs[:5]}"
			)
			# Check for potential issues: sequences with decoded tokens but no GPU reservation flag reset
			problematic = [s for s in resuming_seqs 
						   if s['gpu_pages_allocated'] == 0 and s['had_initial_gpu_reservation']]
			if problematic:
				logging.error(
					f"Rank {self.rank}: POTENTIAL BUG: {len(problematic)} sequences have "
					f"decoded_length>0, gpu_pages_allocated=0, but had_initial_gpu_reservation=True! "
					f"First 5: {problematic[:5]}"
				)
		
		if fresh_seqs and self.rank == 0 and BATCHGEN_CB_DEBUG:
			logging.debug(
				f"_config_decoding_for_batch: {len(fresh_seqs)} FRESH sequences (decoded_length=0)"
			)

		# ============ SIMPLIFIED: Model and GPU KV manager already initialized ============
		# Model loading and GPU KV manager init now happen in generate() BEFORE batch selection
		# via _load_decode_model() and _init_gpu_kv_with_actual_size()
		assert self.model is not None, (
			"Model must be loaded before _config_decoding_for_batch(). "
			"Ensure _load_decode_model() was called first."
		)
		assert self.gpu_paged_kv_cache_manager is not None and self.gpu_paged_kv_cache_manager.is_initialized, (
			"GPU KV manager must be initialized before _config_decoding_for_batch(). "
			"Ensure _init_gpu_kv_with_actual_size() was called first."
		)

		# Allocate GPU KV for sequences
		if local_decode_indices:
			self._allocate_gpu_kv_two_page_buffer(local_decode_indices, load_from_host=True)
			for local_idx in local_decode_indices:
				uuid = self._local_to_uuid_map[local_idx]
				seq = self.global_batch.get_sequence(uuid)
				seq.gpu_pages_allocated = seq.get_gpu_pages_for_two_page_buffer()
				# Mark initial reservation done
				seq.mark_initial_gpu_reservation_done()
				self._sequences_with_gpu_kv.add(uuid)
		
		if self.rank == 0:
			logging.info(f"[DECODE] Config completed: {(time.perf_counter() - start_time)*1000:.1f}ms, {len(decode_uuids)} sequences")

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

		if self.rank == 0:
			logging.info(
				f"[DECODE] Prepared batch (two-page): {len(decode_batch)} sequences, "
				f"{total_pages_needed} pages"
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

			logging.debug(f"Rank {self.rank}: Releasing host KV pages for global_idx: {global_sequence_ids}")
			
			# NOTE: GPU KV pages should already be released by caller
			# Do NOT call _release_gpu_kv_pages here to avoid double-free
			
			# Release host KV pages
			# NOTE: release_sequence_pages already calls unregister_sequences internally,
			# so we don't need to call unregister_sequences separately
			worker_view.release_sequence_pages(global_sequence_ids)
			
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

		# Dynamic padding: find max length within THIS batch, not global max
		# This is critical for long-tailed distributions
		batch_seq_lengths = [
			self.query_book[query_idx].encoded["input_ids"].shape[1]
			for query_idx in batch
		]
		batch_max_len = max(batch_seq_lengths)

		# Pad each sequence to batch_max_len
		padded_input_ids = []
		padded_attention_masks = []
		for query_idx in batch:
			seq_input_ids = self.query_book[query_idx].encoded["input_ids"]
			seq_attention_mask = self.query_book[query_idx].encoded["attention_mask"]
			seq_len = seq_input_ids.shape[1]

			if seq_len < batch_max_len:
				# Pad with zeros (left-aligned tokens, right-padded)
				pad_len = batch_max_len - seq_len
				seq_input_ids = torch.cat([
					seq_input_ids,
					torch.zeros((1, pad_len), dtype=seq_input_ids.dtype)
				], dim=1)
				seq_attention_mask = torch.cat([
					seq_attention_mask,
					torch.zeros((1, pad_len), dtype=seq_attention_mask.dtype)
				], dim=1)

			padded_input_ids.append(seq_input_ids)
			padded_attention_masks.append(seq_attention_mask)

		input_ids = torch.cat(padded_input_ids, dim=0)
		attention_masks = torch.cat(padded_attention_masks, dim=0)

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
		if self.rank == 0:
			logging.info(f"Number of prefill micro batches: {num_prefill_micro_batches}")

		cur_batch_start = 0
		output_tokens = []
		
		for micro_batch_idx in tqdm(range(num_prefill_micro_batches), desc="Prefill Micro Batch"):
			# Feed watchdog during long prefill operations
			self.feed_watchdog()

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
				new_tokens = self._select_tokens(outputs.logits[:, -1, :])
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

	def prefill_prepacked(self, batch: list[int]):
		"""
		Handle prefill for a batch using prepack optimization.

		Prepack combines multiple shorter sequences into rows to minimize padding waste,
		which is especially beneficial for MLP/MoE layers.

		Args:
			batch: list of local indices
		"""
		if "deepseek" in self.model_config.model_type:
			self.model.model._use_flash_attention_2 = False

		# Collect input_ids and attention_masks as lists for prepacking
		input_ids_list = []
		attention_mask_list = []
		seq_lengths = []

		for query_idx in batch:
			input_ids = self.query_book[query_idx].encoded["input_ids"][:, :self.max_input_length]
			attention_mask = self.query_book[query_idx].encoded["attention_mask"][:, :self.max_input_length]

			# Get actual sequence length
			actual_len = int(attention_mask.sum().item())
			seq_lengths.append(actual_len)

			input_ids_list.append(input_ids)
			attention_mask_list.append(attention_mask)

		# Prepack sequences
		# Row capacity is set by planner in config (None = no limit, use max sequence length)
		row_capacity = self.engine_config.Module_Batching_Config.prepack_row_capacity
		prepack_meta = prepack_sequences(
			input_ids_list,
			attention_mask_list,
			row_capacity=row_capacity,
			device=self.torch_device,
		)

		# Log prepack statistics
		if self.rank == 0:
			stats = get_prepack_stats(prepack_meta)
			logging.info(
				f"Prepack stats: {stats['num_sequences']} seqs -> {stats['num_packed_rows']} rows, "
				f"padding saved: {stats['padding_saved']} tokens, "
				f"efficiency: {stats['packing_efficiency']:.2%}"
			)

		# Create flattened tensors for prepacked forward
		# Flatten packed_input_ids to [total_tokens]
		total_tokens = sum(prepack_meta.original_seq_lengths)

		# Extract only valid tokens (non-padding) in order
		packed_input_ids_flat = []
		packed_position_ids_flat = []

		for seq_idx in range(prepack_meta.num_original_sequences):
			row_idx, start_pos = prepack_meta.pack_assignment[seq_idx]
			seq_len = prepack_meta.original_seq_lengths[seq_idx]

			# Extract tokens for this sequence
			seq_input_ids = prepack_meta.packed_input_ids[row_idx, start_pos:start_pos + seq_len]
			packed_input_ids_flat.append(seq_input_ids)

			# Position IDs are 0, 1, 2, ... for each sequence
			packed_position_ids_flat.append(torch.arange(seq_len, device=self.torch_device))

		packed_input_ids_flat = torch.cat(packed_input_ids_flat, dim=0)  # [total_tokens]
		packed_position_ids_flat = torch.cat(packed_position_ids_flat, dim=0)  # [total_tokens]

		# Split sequences into micro-batches based on TOKEN count (not sequence count)
		# This prevents OOM when sequences have varying lengths
		# Token cap is set by planner in config, worker reads from config (no hardcoded values)
		MAX_TOKENS_PER_MICRO_BATCH = self.engine_config.Module_Batching_Config.prefill_micro_batch_token_cap
		num_sequences = prepack_meta.num_original_sequences
		seq_lengths_list = prepack_meta.original_seq_lengths

		# Create micro-batches bounded by token count
		micro_batches = []
		current_batch_start = 0
		current_batch_tokens = 0

		for seq_idx in range(num_sequences):
			seq_len = seq_lengths_list[seq_idx]

			# If adding this sequence would exceed limit, finalize current batch
			if current_batch_tokens + seq_len > MAX_TOKENS_PER_MICRO_BATCH and current_batch_tokens > 0:
				micro_batches.append((current_batch_start, seq_idx))
				current_batch_start = seq_idx
				current_batch_tokens = 0

			current_batch_tokens += seq_len

		# Don't forget the last batch
		if current_batch_start < num_sequences:
			micro_batches.append((current_batch_start, num_sequences))

		if self.rank == 0:
			total_tokens = sum(seq_lengths_list)
			logging.info(
				f"Prepacked prefill: {len(micro_batches)} micro batches, "
				f"{total_tokens:,} total tokens, max {MAX_TOKENS_PER_MICRO_BATCH:,} tokens/batch"
			)

		output_tokens = []

		with torch.inference_mode():
			for batch_idx, (seq_start, seq_end) in tqdm(
				enumerate(micro_batches),
				total=len(micro_batches),
				desc="Prepacked Prefill",
				disable=(self.rank != 0)  # Only show progress on rank 0
			):
				# Feed watchdog during long prefill operations
				self.feed_watchdog()

				# Get sequences for this micro-batch
				batch_seq_lengths = seq_lengths_list[seq_start:seq_end]
				batch_num_seqs = seq_end - seq_start

				# Extract tokens for this micro-batch
				batch_input_ids = []
				batch_position_ids = []
				token_offset = sum(seq_lengths_list[:seq_start])  # Offset into flat tensors

				for seq_idx in range(seq_start, seq_end):
					seq_len = seq_lengths_list[seq_idx]
					# Calculate where this sequence's tokens are in the flat tensor
					seq_token_start = sum(seq_lengths_list[:seq_idx])
					seq_token_end = seq_token_start + seq_len

					batch_input_ids.append(packed_input_ids_flat[seq_token_start:seq_token_end])
					batch_position_ids.append(packed_position_ids_flat[seq_token_start:seq_token_end])

				batch_input_ids_flat = torch.cat(batch_input_ids, dim=0)
				batch_position_ids_flat = torch.cat(batch_position_ids, dim=0)

				# Create cu_seqlens for this micro-batch
				batch_cu_seqlens = torch.zeros(batch_num_seqs + 1, dtype=torch.int32, device=self.torch_device)
				for i, seq_len in enumerate(batch_seq_lengths):
					batch_cu_seqlens[i + 1] = batch_cu_seqlens[i] + seq_len

				batch_max_seqlen = max(batch_seq_lengths)

				# Set up Attn_Wrapper for this micro-batch
				Attn_Wrapper.prepack_mode = True
				Attn_Wrapper.prepack_cu_seqlens = batch_cu_seqlens
				Attn_Wrapper.prepack_max_seqlen = batch_max_seqlen
				Attn_Wrapper.prepack_num_sequences = batch_num_seqs
				Attn_Wrapper.prepack_seq_lengths = batch_seq_lengths
				Attn_Wrapper.position_ids = batch_position_ids_flat
				# Map local batch indices to global seq ids for this micro-batch
				batch_local_indices = batch[seq_start:seq_end]
				Attn_Wrapper.cur_batch = self._local_indices_to_global_seq_ids(batch_local_indices)

				# CRITICAL: Also bind to AttnWrapperBase for models using new wrapper system (GPT-OSS)
				# Without this, GPT-OSS uses _forward_prefill instead of _forward_prefill_prepacked,
				# which does NOT offload KV to host, causing decode to read garbage.
				AttnWrapperBase.prepack_mode = True
				AttnWrapperBase.prepack_cu_seqlens = batch_cu_seqlens
				AttnWrapperBase.prepack_max_seqlen = batch_max_seqlen
				AttnWrapperBase.prepack_num_sequences = batch_num_seqs
				AttnWrapperBase.prepack_seq_lengths = batch_seq_lengths
				AttnWrapperBase.position_ids = batch_position_ids_flat
				AttnWrapperBase.cur_batch = Attn_Wrapper.cur_batch

				# Embed tokens
				inputs_embeds = self.model.model.embed_tokens(batch_input_ids_flat.to(self.torch_device))

				# Reshape to 3D: [1, batch_total_tokens, hidden_dim]
				hidden_states = inputs_embeds.unsqueeze(0)

				# Forward through model layers
				for layer_idx, decoder_layer in enumerate(self.model.model.layers):
					layer_outputs = decoder_layer(
						hidden_states,
						attention_mask=None,
						position_ids=None,
						past_key_value=None,
						output_attentions=False,
						use_cache=False,
					)
					hidden_states = layer_outputs[0]

				# Final norm
				hidden_states = self.model.model.norm(hidden_states)

				# Extract last token hidden states for each sequence
				last_token_indices = batch_cu_seqlens[1:] - 1
				last_token_hidden = hidden_states[0, last_token_indices, :]

				# DEBUG: Verify per-sequence hidden states after prefill
				import os
				if os.environ.get("BATCHGEN_DEBUG_PREFILL_OUTPUT", "0") == "1":
					print(f"\n[PREFILL OUTPUT DEBUG] === Micro-batch {batch_idx} ===")
					print(f"[PREFILL OUTPUT DEBUG] hidden_states.shape = {hidden_states.shape}")
					print(f"[PREFILL OUTPUT DEBUG] batch_cu_seqlens = {batch_cu_seqlens.tolist()}")
					print(f"[PREFILL OUTPUT DEBUG] last_token_indices = {last_token_indices.tolist()}")
					print(f"[PREFILL OUTPUT DEBUG] last_token_hidden.shape = {last_token_hidden.shape}")
					# Check if hidden states differ across sequences
					for i in range(min(3, batch_num_seqs)):
						h = last_token_hidden[i, :8].tolist()
						print(f"[PREFILL OUTPUT DEBUG] seq{i} (pos={last_token_indices[i].item()}): hidden[:8] = {[f'{v:.4f}' for v in h]}")
					if batch_num_seqs >= 2:
						diff = (last_token_hidden[0] - last_token_hidden[1]).abs().max().item()
						print(f"[PREFILL OUTPUT DEBUG] max_diff seq0-seq1: {diff:.6f}")
						if diff < 1e-4:
							print(f"[PREFILL OUTPUT DEBUG] *** CRITICAL: seq0 and seq1 have IDENTICAL hidden states! ***")

				# Call lm_head directly using F.linear to bypass the hook
				logits = torch.nn.functional.linear(
					last_token_hidden,
					self.model.lm_head.weight,
					self.model.lm_head.bias if hasattr(self.model.lm_head, 'bias') and self.model.lm_head.bias is not None else None
				)

				batch_new_tokens = self._select_tokens(logits)
				output_tokens.append(batch_new_tokens)

				# DEBUG: Show logits and sampled tokens
				if os.environ.get("BATCHGEN_DEBUG_PREFILL_OUTPUT", "0") == "1":
					print(f"[PREFILL OUTPUT DEBUG] logits.shape = {logits.shape}")
					for i in range(min(3, batch_num_seqs)):
						top_vals, top_ids = torch.topk(logits[i], k=5)
						print(f"[PREFILL OUTPUT DEBUG] seq{i} top5_ids={top_ids.tolist()}, top5_vals={[f'{v:.2f}' for v in top_vals.tolist()]}")
					print(f"[PREFILL OUTPUT DEBUG] sampled_tokens[:5] = {batch_new_tokens[:5].flatten().tolist()}")
					if batch_num_seqs >= 2:
						if batch_new_tokens[0].item() == batch_new_tokens[1].item():
							print(f"[PREFILL OUTPUT DEBUG] *** WARNING: seq0 and seq1 sampled SAME token! ***")

		# Reset prepack mode
		Attn_Wrapper.prepack_mode = False
		Attn_Wrapper.prepack_cu_seqlens = None
		Attn_Wrapper.prepack_max_seqlen = None
		Attn_Wrapper.prepack_num_sequences = None
		Attn_Wrapper.prepack_seq_lengths = None

		# Also reset AttnWrapperBase for models using new wrapper system (GPT-OSS)
		AttnWrapperBase.prepack_mode = False
		AttnWrapperBase.prepack_cu_seqlens = None
		AttnWrapperBase.prepack_max_seqlen = None
		AttnWrapperBase.prepack_num_sequences = None
		AttnWrapperBase.prepack_seq_lengths = None

		# Log timing summary for GPT-OSS if timing was enabled
		self._log_prefill_timing()

		new_tokens = torch.cat(output_tokens, dim=0)
		self.update_new_token(new_tokens, batch, 0)

		# Update sequence state after prefill
		for i, local_idx in enumerate(batch):
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			seq.decoded_length = 1
			seq.current_context_length = seq.prompt_length + 1

			# Check for EOS respecting ignore_eos flag
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
	) -> Tuple[List[str], List[int], Optional[object], List[str], List[int], List[int], FastBoundaryTimingStats, bool]:
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
			(decode_uuids, batch, new_async_task, new_load_uuids, new_load_local, new_load_global, timing, watermark_triggered)
		"""
		timing = FastBoundaryTimingStats()
		boundary_start = time.perf_counter()
		
		# ========== PHASE 0: Wait for pending async operations ==========
		t0 = time.perf_counter()
		timing.num_kv_append_tasks = self._wait_pending_kv_append_tasks()
		timing.wait_kv_append_ms = (time.perf_counter() - t0) * 1000
		
		# ========== CRITICAL: SYNC decode_uuids BEFORE finalize_async_load ==========
		# decode_uuids may have drifted between boundaries. We MUST sync BEFORE
		# _finalize_async_load_minimal because it concatenates valid_pending_uuids 
		# (synced) with current_decode_uuids (potentially desync'd).
		# If we don't sync first, the output will be desync'd.
		t_sync = time.perf_counter()
		local_decode_set = set(decode_uuids)
		all_decode_sets = [None] * self.world_size
		dist.all_gather_object(all_decode_sets, local_decode_set)
		
		# Check for desync
		all_sets_equal = all(s == local_decode_set for s in all_decode_sets if s is not None)
		if not all_sets_equal:
			# Log detailed desync info
			for r, s in enumerate(all_decode_sets):
				if s != local_decode_set:
					diff_in_r = s - local_decode_set if s else set()
					diff_in_local = local_decode_set - s if s else local_decode_set
					logging.error(
						f"Rank {self.rank}: decode_uuids DESYNC detected at boundary start! "
						f"Rank {r} has {len(diff_in_r)} extra: {list(diff_in_r)[:5]}, "
						f"Rank {self.rank} has {len(diff_in_local)} extra: {list(diff_in_local)[:5]}"
					)
			# Use UNION to ensure all sequences get their state gathered
			global_decode_set = set()
			for s in all_decode_sets:
				if s:
					global_decode_set.update(s)
			decode_uuids = sorted(
				global_decode_set,
				key=lambda u: self.global_batch.get_sequence(u).global_idx if self.global_batch.get_sequence(u) else float('inf')
			)
			batch = self._get_local_indices_for_uuids(decode_uuids)
			logging.warning(f"Rank {self.rank}: Using union to sync at boundary start, decode_uuids now {len(decode_uuids)}")
		timing.sync_decode_uuids_ms = (time.perf_counter() - t_sync) * 1000
		
		# Integrate previous async load if any
		if pending_load_uuids:  # ALL ranks have identical pending_load_uuids
			t0 = time.perf_counter()

			if BATCHGEN_CB_DEBUG:
				logging.debug(
					f"Rank {self.rank}: Integrating {len(pending_load_uuids)} async-loaded sequences"
				)

			if pending_async_load_task is not None:
				pending_async_load_task.wait()
				torch.cuda.synchronize(self.torch_device)

			timing.wait_async_load_ms = (time.perf_counter() - t0) * 1000

			# barrier ensures all ranks finish async load before continuing
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
			
			# Rebuild page table to include newly loaded sequences
			if batch and gpu_manager is not None and gpu_manager.is_initialized:
				self._rebuild_page_table_for_batch(batch, gpu_manager)
				# Verify page table matches batch, fix if needed
				if gpu_manager._gpu_page_table_manager:
					post_finalize_slot_order = list(gpu_manager._gpu_page_table_manager.slot_to_seq_id) if gpu_manager._gpu_page_table_manager.slot_to_seq_id else []
					post_finalize_batch_global_ids = self._local_indices_to_global_seq_ids(batch)
					if post_finalize_slot_order != post_finalize_batch_global_ids:
						gpu_manager.rebuild_page_table(post_finalize_batch_global_ids)

		if not decode_uuids:
			timing.total_ms = (time.perf_counter() - boundary_start) * 1000
			return decode_uuids, batch, None, [], [], [], timing, False
		
		# ========== PHASE 1: SINGLE BATCHED ALL_GATHER ==========
		t0 = time.perf_counter()
		
		local_free_pages = gpu_manager.get_stats().num_free_pages if gpu_manager and gpu_manager.is_initialized else 0
		
		# DEBUG: Log decode_uuids and which ones this rank owns
		my_owned = [u for u in decode_uuids if u in self._uuid_to_local_map]
		if self.rank == 0:
			logging.debug(
				f"Rank {self.rank}: State gathering - decode_uuids_len={len(decode_uuids)}, "
				f"my_owned_count={len(my_owned)}"
			)
		
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
		
		# Get candidates for loading - report PREFILLED/ON_HOLD sequences that could be loaded
		# CRITICAL FIX: Only report PREFILLED or ON_HOLD sequences as load candidates.
		# QUEUEING sequences have NOT been registered with host KV yet (registration
		# happens during _config_prefill_for_batch), so trying to load them would fail
		# with "Sequence X is not registered" error from the host KV backend.
		decode_uuids_set = set(decode_uuids)
		local_candidate_state = {}
		valid_load_statuses = {SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD}
		for uuid in self._uuid_to_local_map.keys():
			if uuid in decode_uuids_set:
				continue  # Already in decode batch
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				continue
			if seq.status == SequenceStatus.COMPLETED:
				continue  # Don't load completed sequences
			if seq.status not in valid_load_statuses:
				continue  # Only load PREFILLED/ON_HOLD (not QUEUEING/IN_PREFILL)
			# Report this as a potential load candidate
			local_candidate_state[uuid] = {
				'pages_needed': seq.get_gpu_pages_for_two_page_buffer(),
				'assigned_rank': seq.assigned_rank,
				'status': seq.status.name,  # Include status for debugging
				'decoded_length': seq.decoded_length,  # For prioritized loading
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
			# Enhanced diagnostics for debugging
			for missing_uuid in missing_uuids[:10]:
				seq = self.global_batch.get_sequence(missing_uuid)
				expected_rank = seq.assigned_rank if seq else "N/A"
				in_local_map = missing_uuid in self._uuid_to_local_map
				seq_status = seq.status.name if seq else "NOT_FOUND"
				# Check what state each rank reported for this uuid
				rank_reported = [r for r, p in enumerate(all_payloads) 
								if p and p.get('seq_state', {}).get(missing_uuid)]
				logging.error(
					f"Rank {self.rank}: Missing UUID={missing_uuid}, assigned_rank={expected_rank}, "
					f"in_local_map={in_local_map}, status={seq_status}, reported_by_ranks={rank_reported}"
				)
			logging.error(
				f"Rank {self.rank}: CRITICAL - {len(missing_uuids)} sequences missing from gathered state! "
				f"decode_uuids_len={len(decode_uuids)}, global_seq_state_len={len(global_seq_state)}, "
				f"Missing first 5: {missing_uuids[:5]}"
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

		# Calculate completed count BEFORE early return to ensure final iteration reports correctly
		timing.total_completed_cumulative = len(self.global_batch.get_sequences_by_status(SequenceStatus.COMPLETED))

		if not decode_uuids:
			timing.total_ms = (time.perf_counter() - boundary_start) * 1000
			return decode_uuids, batch, None, [], [], [], timing, False

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
		onhold_set = set()  # Initialize early for later use
		# Track actual pages used for extension (needed for load selection)
		actual_extension_by_rank = [0] * self.world_size
		
		if all_can_extend and seqs_needing_extension:
			# Simple extension - no eviction needed
			self._extend_gpu_kv_allocation(seqs_needing_extension)
			# All extension pages are used
			actual_extension_by_rank = list(total_additional_by_rank)
		elif not all_can_extend:
			# Need eviction - put SHORTEST-decoded sequences on hold first
			# Rationale: Keep longer-decoded sequences in GPU because:
			#   1. They are closer to completion (may finish soon)
			#   2. We want to prioritize finishing sequences over starting new ones
			# CRITICAL: Sort DETERMINISTICALLY by (decoded_length ASC, global_idx ASC)
			for r in range(self.world_size):
				if total_additional_by_rank[r] > per_rank_free[r]:
					# Use gathered state for filtering AND sorting
					rank_seqs = [
						(uuid, global_seq_state[uuid])
						for uuid in decode_uuids
						if uuid in global_seq_state and global_seq_state[uuid]['assigned_rank'] == r
					]
					# CRITICAL: Stable sort with tie-breaker for determinism
					# Shortest decoded_length first (ascending), then by global_idx
					rank_seqs.sort(
						key=lambda x: (x[1]['decoded_length'],
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
			
			# DEBUG: Log batch size after on-hold
			if BATCHGEN_CB_DEBUG:
				logging.info(
					f"Rank {self.rank}: After on-hold: batch_size={len(batch)}, "
					f"num_onhold={len(onhold_uuids)}, my_onhold={len(my_onhold)}"
				)
			
			# Extend remaining sequences and track actual extension pages used
			remaining_needing_ext = [u for u in seqs_needing_extension if u not in onhold_set]
			if remaining_needing_ext:
				self._extend_gpu_kv_allocation(remaining_needing_ext)
				# Calculate actual extension pages used per rank
				for uuid in remaining_needing_ext:
					state = global_seq_state.get(uuid, {})
					r = state.get('assigned_rank')
					if r is not None:
						actual_extension_by_rank[r] += state.get('additional_pages_needed', 0)
		
		timing.num_onhold = len(onhold_uuids)
		timing.extension_ms = (time.perf_counter() - t0) * 1000
		
		# ========== PHASE 3: SELECT AND LAUNCH ASYNC LOAD (DETERMINISTIC) ==========
		t0 = time.perf_counter()
		new_async_task = None
		new_load_uuids = []
		new_load_local = []
		new_load_global = []
		
		# CRITICAL: Use global_candidate_info keys as the authoritative load_candidates list
		# This ensures all ranks have the same view of candidates, since global_candidate_info
		# is built from gathered state from all ranks.
		# Local status queries (PREFILLED/ON_HOLD) can be desynchronized across ranks.
		# ALSO: Filter out sequences that were marked COMPLETED in this boundary
		# (their host KV pages have been released, so we can't load them)
		# ALSO: Filter out sequences put ON_HOLD in THIS boundary (to avoid loading just-evicted sequences)
		completed_set = set(completed_uuids)
		# LOADING STRATEGY: Prioritize LONGEST decoded sequences first
		# Rationale: Longer-decoded sequences are closer to completion, so loading them
		# helps finish sequences faster and reduces long-tail latency.
		# Sort by decoded_length DESCENDING, global_idx as tie-breaker for determinism.
		load_candidates_synced = sorted(
			[u for u in global_candidate_info.keys() if u not in completed_set and u not in onhold_set],
			key=lambda u: (
				-global_candidate_info[u].get('decoded_length', 0),  # Descending (longest first)
				self.global_batch.get_sequence(u).global_idx if self.global_batch.get_sequence(u) else float('inf')
			)
		)
		
		if load_candidates_synced and decode_uuids:
			
			# CRITICAL: Gather ACTUAL free pages from all ranks AFTER extension/eviction
			# This ensures accurate selection instead of relying on estimates
			local_free_after = gpu_manager.get_stats().num_free_pages if gpu_manager and gpu_manager.is_initialized else 0
			all_free_after = [0] * self.world_size
			dist.all_gather_object(all_free_after, local_free_after)
			
			# Use gathered actual free pages for selection
			adjusted_per_rank_free = all_free_after
			
			timing.load_select_ms = (time.perf_counter() - t0) * 1000
			
			# Select candidates that fit in ADJUSTED available GPU pages
			rank_pages_used = [0] * self.world_size
			for uuid in load_candidates_synced:
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
				
				# Track which UUIDs this rank actually loaded (for sync)
				my_actually_loaded = set()
				
				if new_load_local:
					# SAFETY CHECK: Verify actual pages needed doesn't exceed actual free pages
					# The estimate may be off due to state drift, so we filter here
					actual_free = gpu_manager.get_stats().num_free_pages if gpu_manager and gpu_manager.is_initialized else 0
					
					# Filter sequences that fit in actual free pages
					filtered_local = []
					filtered_global = []
					filtered_tokens = []
					pages_used = 0
					
					for local_idx in new_load_local:
						uuid = self._local_to_uuid_map[local_idx]
						seq = self.global_batch.get_sequence(uuid)
						pages_needed = seq.get_gpu_pages_for_two_page_buffer()
						
						if pages_used + pages_needed <= actual_free:
							filtered_local.append(local_idx)
							filtered_global.append(seq.global_idx)
							filtered_tokens.append(pages_needed * self.PAGE_SIZE)
							pages_used += pages_needed
							my_actually_loaded.add(uuid)
						else:
							# Log that we're dropping this sequence due to insufficient pages
							gathered_pages = global_candidate_info.get(uuid, {}).get('pages_needed', 'N/A')
							logging.warning(
								f"Rank {self.rank}: Dropping {uuid} from load - "
								f"need={pages_needed}, gathered={gathered_pages}, "
								f"pages_used={pages_used}, actual_free={actual_free}"
							)
					
					if filtered_local:
						new_load_local = filtered_local
						new_load_global = filtered_global
						tokens = filtered_tokens
						
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
					else:
						# All sequences were dropped due to insufficient pages
						new_load_local = []
						new_load_global = []
						logging.warning(
							f"Rank {self.rank}: All load candidates dropped due to insufficient pages, "
							f"actual_free={actual_free}"
						)
				
				# CRITICAL: Sync which sequences were actually loaded across ALL ranks
				# This must be called by ALL ranks (even those with no sequences to load)
				# If any rank dropped sequences, all ranks must update new_load_uuids
				all_actually_loaded = [None] * self.world_size
				dist.all_gather_object(all_actually_loaded, my_actually_loaded)
				
				# new_load_uuids should only include sequences that their owning rank actually loaded
				actually_loaded_global = set()
				for loaded_set in all_actually_loaded:
					if loaded_set:
						actually_loaded_global.update(loaded_set)
				
				# Update new_load_uuids to match what was actually loaded
				original_count = len(new_load_uuids)
				new_load_uuids = [u for u in new_load_uuids if u in actually_loaded_global]
				if len(new_load_uuids) != original_count:
					logging.warning(
						f"Rank {self.rank}: new_load_uuids reduced from {original_count} to {len(new_load_uuids)} "
						f"due to safety filter"
					)
		
		timing.num_loaded = len(new_load_uuids)
		
		# ========== FINAL PAGE TABLE REBUILD ==========
		t0 = time.perf_counter()
		# DEBUG: Log batch size before final rebuild
		if BATCHGEN_CB_DEBUG:
			global_ids_for_rebuild = self._local_indices_to_global_seq_ids(batch) if batch else []
			logging.info(
				f"Rank {self.rank}: FINAL REBUILD: batch_size={len(batch)}, "
				f"global_ids_count={len(global_ids_for_rebuild)}"
			)
		self._rebuild_page_table_for_batch(batch, gpu_manager)
		
		# DEBUG: Verify page table size after rebuild
		if BATCHGEN_CB_DEBUG and gpu_manager and gpu_manager.is_initialized:
			mgr = gpu_manager._gpu_page_table_manager
			if mgr and mgr.gpu_table is not None:
				logging.info(
					f"Rank {self.rank}: After rebuild: gpu_table.shape={mgr.gpu_table.shape}, "
					f"slot_to_seq_id_len={len(mgr.slot_to_seq_id)}"
				)
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
			batch = expected_local  # Fix the batch
			# CRITICAL: Rebuild page table to match the corrected batch
			self._rebuild_page_table_for_batch(batch, gpu_manager)
			logging.info(f"Rank {self.rank}: Page table rebuilt after batch correction")
		
		# FINAL VERIFICATION: Ensure page table matches batch before returning
		if batch and gpu_manager and gpu_manager.is_initialized:
			mgr = gpu_manager._gpu_page_table_manager
			if mgr and mgr.gpu_table is not None:
				if mgr.gpu_table.shape[0] != len(batch):
					logging.error(
						f"Rank {self.rank}: CRITICAL - Page table STILL mismatched at function return! "
						f"gpu_table.shape[0]={mgr.gpu_table.shape[0]}, batch_size={len(batch)}"
					)
		
		timing.total_ms = (time.perf_counter() - boundary_start) * 1000

		# Check watermark trigger for dynamic prefill switching
		watermark_triggered = self._check_host_kv_watermark_trigger()

		return decode_uuids, batch, new_async_task, new_load_uuids, new_load_local, new_load_global, timing, watermark_triggered

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
		
		# VALIDATION: Verify all pending_uuids exist and have assigned ranks
		valid_pending_uuids = []
		for uuid in pending_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				logging.error(f"Rank {self.rank}: pending_uuid {uuid} not found in global_batch!")
				continue
			if seq.assigned_rank is None:
				logging.error(f"Rank {self.rank}: pending_uuid {uuid} has no assigned_rank!")
				continue
			valid_pending_uuids.append(uuid)
		
		if len(valid_pending_uuids) != len(pending_uuids):
			logging.warning(
				f"Rank {self.rank}: Filtered {len(pending_uuids) - len(valid_pending_uuids)} invalid "
				f"pending_uuids out of {len(pending_uuids)}"
			)
		
		self._update_batch_status(valid_pending_uuids, SequenceStatus.IN_DECODE)
		
		for local_idx in pending_local_indices:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			seq.gpu_pages_allocated = seq.get_gpu_pages_for_two_page_buffer()
			# Mark that this sequence has received its initial GPU reservation
			seq.mark_initial_gpu_reservation_done()
			self._sequences_with_gpu_kv.add(uuid)
		
		updated_uuids = current_decode_uuids + valid_pending_uuids
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
		Continuous decoding with optimized collective operations.

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

		# Also bind to AttnWrapperBase for models using new wrapper system (e.g., GPT-OSS)
		AttnWrapperBase.gpu_paged_kv_manager = gpu_manager
		AttnWrapperBase.host_paged_kv_worker_view = worker_view
		AttnWrapperBase.cur_batch = Attn_Wrapper.cur_batch

		# CRITICAL FIX: Ensure page table matches cur_batch at entry
		# This fixes order mismatch that can occur during decode→prefill→decode transitions
		if gpu_manager and gpu_manager._gpu_page_table_manager:
			entry_slot_order = list(gpu_manager._gpu_page_table_manager.slot_to_seq_id) if gpu_manager._gpu_page_table_manager.slot_to_seq_id else []
			entry_cur_batch = list(Attn_Wrapper.cur_batch) if Attn_Wrapper.cur_batch else []
			if entry_slot_order != entry_cur_batch:
				logging.error(
					f"Rank {self.rank}: ORDER MISMATCH at decoding_continuous entry: "
					f"slot_to_seq_id={entry_slot_order[:5]}{'...' if len(entry_slot_order) > 5 else ''} (len={len(entry_slot_order)}), "
					f"cur_batch={entry_cur_batch[:5]}{'...' if len(entry_cur_batch) > 5 else ''} (len={len(entry_cur_batch)}). Rebuilding page table..."
				)
				# Rebuild page table to match cur_batch order
				if entry_cur_batch:
					gpu_manager.rebuild_page_table(entry_cur_batch)
					logging.info(f"Rank {self.rank}: Page table rebuilt to match cur_batch order")
			else:
				if BATCHGEN_CB_DEBUG:
					logging.debug(
						f"Rank {self.rank}: decoding_continuous entry OK. "
						f"batch_size={len(batch)}, cur_batch={entry_cur_batch[:5]}{'...' if len(entry_cur_batch) > 5 else ''}"
					)

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
		
		# Use cumulative counters that persist across prefill/decode switches
		# Initialize instance vars if not present (shouldn't happen, but safety)
		if not hasattr(self, '_cumulative_decode_iterations'):
			self._cumulative_decode_iterations = 0
		if not hasattr(self, '_cumulative_decode_boundaries'):
			self._cumulative_decode_boundaries = 0
		if not hasattr(self, '_cumulative_boundary_ms'):
			self._cumulative_boundary_ms = 0.0
		if not hasattr(self, '_cumulative_forward_ms'):
			self._cumulative_forward_ms = 0.0

		# Local iteration counter (for boundary interval tracking within this decode round)
		local_iteration = 0
		last_boundary = 0
		global_batch_size = len(self.global_batch)

		# ========== INITIAL MOE BUFFER SYNC ==========
		# Sync buffer size BEFORE first forward pass to prevent overflow.
		# The boundary sync (in _page_boundary_fast) only happens after DECISION_INTERVAL
		# iterations, but the first forward pass runs immediately. Without this sync,
		# if one rank has more tokens than the initial estimate (ceil(total/world_size)),
		# we get buffer overflow.
		local_batch_size = torch.tensor([len(batch)], dtype=torch.int64, device=self.torch_device)
		dist.all_reduce(local_batch_size, op=dist.ReduceOp.MAX)
		max_batch_size = local_batch_size.item()

		if max_batch_size > 0 and hasattr(self, 'parallel_manager') and self.parallel_manager is not None:
			if hasattr(self.parallel_manager, 'set_num_tokens_per_rank'):
				self.parallel_manager.set_num_tokens_per_rank(max_batch_size)

		# OPTIMIZATION: Track if page table was verified since last batch change
		# Avoids redundant page table checks between boundaries
		_page_table_verified_this_batch = True  # Start True after entry check
		
		# Main decode loop
		while decode_uuids:
			local_iteration += 1
			self._cumulative_decode_iterations += 1

			# Feed watchdog to prevent timeout during long decoding
			self.feed_watchdog()

			# Page boundary check - use DECISION_INTERVAL (configurable via BATCHGEN_DECISION_FREQUENCY_PAGES)
			if local_iteration - last_boundary >= self.DECISION_INTERVAL:
				last_boundary = local_iteration

				(decode_uuids, batch,
				 pending_async_task, pending_load_uuids,
				 pending_load_local, pending_load_global,
				 timing, watermark_triggered) = self._page_boundary_fast(
					decode_uuids, batch, gpu_manager,
					pending_async_task, pending_load_uuids,
					pending_load_local, pending_load_global
				)

				self._cumulative_boundary_ms += timing.total_ms
				self._cumulative_decode_boundaries += 1

				# Batch may have changed - need to verify page table
				_page_table_verified_this_batch = False

				# Post-boundary: verify page table matches batch and fix if needed
				if batch and gpu_manager and gpu_manager.is_initialized and gpu_manager._gpu_page_table_manager:
					post_boundary_slot_order = list(gpu_manager._gpu_page_table_manager.slot_to_seq_id) if gpu_manager._gpu_page_table_manager.slot_to_seq_id else []
					post_boundary_batch_global_ids = self._local_indices_to_global_seq_ids(batch)

					if post_boundary_slot_order != post_boundary_batch_global_ids:
						# Fix: Rebuild page table to match batch
						gpu_manager.rebuild_page_table(post_boundary_batch_global_ids)

				# Page table is now verified for this batch
				_page_table_verified_this_batch = True

				# Check if watermark triggered - interrupt decode for prefill
				if watermark_triggered:
					# CRITICAL FIX: Wait for pending KV append tasks BEFORE going ON_HOLD!
					# Without this, KV data may not be fully written to host when sequences
					# are later resumed, causing KV corruption and gibberish output.
					num_waited = self._wait_pending_kv_append_tasks()
					if num_waited > 0:
						logging.info(
							f"[WATERMARK-KV-SYNC] Rank {self.rank}: Waited for {num_waited} pending KV append tasks "
							f"before putting sequences ON_HOLD"
						)
					
					logging.info(
						f"[WATERMARK] Rank {self.rank}: Decode interrupted - putting {len(decode_uuids)} "
						f"sequences ON_HOLD, will trigger prefill"
					)
					# Put all remaining sequences ON_HOLD
					self._put_sequences_on_hold(decode_uuids)
					# Exit decode loop - will return to generate() which will trigger prefill
					break
				
				# Detailed logging at every boundary (only rank 0)
				if self.rank == 0:
					# Get status counts
					# - in_decode: sequences currently in decode batch (IN_DECODE status)
					# - onhold: sequences paused with host KV (ON_HOLD status)  
					# - prefilled: sequences prefilled but not yet decoding (PREFILLED status)
					# - host_kv_total: total sequences with host KV = prefilled + onhold + in_decode
					num_in_decode = timing.total_active
					num_onhold = len(self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD))
					num_prefilled = timing.total_prefilled
					num_completed_total = timing.total_completed_cumulative
					num_host_kv_total = num_prefilled + num_onhold + num_in_decode
					
					# Get page stats if available
					page_info = ""
					if hasattr(self, '_host_kv_page_stats') and self._host_kv_page_stats:
						ps = self._host_kv_page_stats
						page_info = f" | Host KV: {ps['used']}/{ps['total']} pages ({ps['free_percent']}% free)"

					if BATCHGEN_CB_DEBUG:
						# Detailed timing log when debug is enabled
						logging.info(
							f"[Decode Interval {self._cumulative_decode_boundaries}] "
							f"iter={self._cumulative_decode_iterations}, "
							f"total={timing.total_ms:.1f}ms | "
							f"wait_kv={timing.wait_kv_append_ms:.1f}({timing.num_kv_append_tasks}), "
							f"wait_async={timing.wait_async_load_ms:.1f}, "
							f"finalize={timing.finalize_load_ms:.1f}, "
							f"sync_uuids={timing.sync_decode_uuids_ms:.1f}, "
							f"gather={timing.gather_ms:.1f}, "
							f"proc={timing.process_ms:.1f}, "
							f"ext={timing.extension_ms:.1f}, "
							f"load_sel={timing.load_select_ms:.1f}, "
							f"load_alloc={timing.load_alloc_ms:.1f}, "
							f"load_launch={timing.load_launch_ms:.1f}, "
							f"rebuild={timing.rebuild_ms:.1f}, "
							f"moe_buf={timing.moe_buffer_update_ms:.1f}, "
							f"barrier={timing.barrier_ms:.1f}ms | "
							f"STATUS: in_decode={num_in_decode}, onhold={num_onhold}, prefilled={num_prefilled}, "
							f"host_kv_total={num_host_kv_total}, completed={num_completed_total}/{global_batch_size}, "
							f"Δ completed={timing.num_completed}, loaded={timing.num_loaded}, onhold={timing.num_onhold}"
							f"{page_info}"
						)
					else:
						# Minimal log without timing details
						logging.info(
							f"[Decode {self._cumulative_decode_boundaries}] iter={self._cumulative_decode_iterations} | "
							f"STATUS: in_decode={num_in_decode}, onhold={num_onhold}, prefilled={num_prefilled}, "
							f"host_kv_total={num_host_kv_total}, completed={num_completed_total}/{global_batch_size}, "
							f"Δ completed={timing.num_completed}, loaded={timing.num_loaded}, onhold={timing.num_onhold}"
							f"{page_info}"
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
				# DEBUG: Log tokens rebuild after boundary
				if new_tokens.shape[0] != len(batch):
					logging.error(
						f"Rank {self.rank}: POST-BOUNDARY new_tokens mismatch! "
						f"batch_size={len(batch)}, new_tokens.shape={new_tokens.shape}"
					)
			
			# Forward pass
			forward_start = time.perf_counter()

			# Pre-compute batch_sequences for use in both forward setup and update loop
			batch_sequences = [self.global_batch.get_sequence(self._local_to_uuid_map[idx]) for idx in batch] if batch else []

			with torch.inference_mode():
				if batch:
					# Collect context lengths, handling rare edge case of ctx_len == 0
					cache_seqlens = []
					for seq in batch_sequences:
						ctx_len = seq.current_context_length
						if ctx_len == 0:  # Rare edge case - trust prompt_length + decoded_length
							ctx_len = seq.prompt_length + seq.decoded_length
							if ctx_len > 0:
								seq.current_context_length = ctx_len
						cache_seqlens.append(ctx_len)

					max_ctx = max(cache_seqlens)
					# Build attention metadata directly on GPU (avoids CPU→GPU copy of list)
					positions = torch.arange(max_ctx, device=self.torch_device)
					seqlens_tensor = torch.tensor(cache_seqlens, dtype=torch.int64, device=self.torch_device)
					attention_mask = (positions.unsqueeze(0) < seqlens_tensor.unsqueeze(1)).to(torch.int64)

					Attn_Wrapper.attention_mask = attention_mask
					# Optimization: Compute position_ids from cache_seqlens directly (O(batch))
					# instead of attention_mask.sum(-1) which is O(batch × max_ctx)
					Attn_Wrapper.cache_seqlens = seqlens_tensor.to(torch.int32)
					Attn_Wrapper.position_ids = (Attn_Wrapper.cache_seqlens - 1).unsqueeze(-1).to(torch.int64)
					Attn_Wrapper.max_seqlen = max_ctx

					# CRITICAL: Also bind to AttnWrapperBase for models using new wrapper system (GPT-OSS)
					# Without this, GPT-OSS attention uses stale cache_seqlens (always None),
					# causing same KV positions to be read/written every decode step.
					AttnWrapperBase.attention_mask = attention_mask
					AttnWrapperBase.cache_seqlens = Attn_Wrapper.cache_seqlens
					AttnWrapperBase.position_ids = Attn_Wrapper.position_ids
					AttnWrapperBase.max_seqlen = max_ctx

					# DEBUG: Print cache_seqlens and input tokens
					if os.environ.get("BATCHGEN_DEBUG_DECODE", "0") == "1" and local_iteration <= 5:
						print(f"\n[DECODE DEBUG] Iteration {local_iteration}")
						print(f"[DECODE DEBUG] cache_seqlens[:5]: {Attn_Wrapper.cache_seqlens[:5].tolist()}")
						print(f"[DECODE DEBUG] position_ids[:5]: {Attn_Wrapper.position_ids[:5].flatten().tolist()}")
						print(f"[DECODE DEBUG] new_tokens[:5]: {new_tokens[:5].flatten().tolist()}")

					if new_tokens.shape[0] != len(batch):
						new_tokens = self._rebuild_input_tokens(batch)
				else:
					Attn_Wrapper.attention_mask = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
					Attn_Wrapper.position_ids = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
					Attn_Wrapper.cache_seqlens = torch.zeros((0,), dtype=torch.int32, device=self.torch_device)
					Attn_Wrapper.max_seqlen = 0
					Attn_Wrapper.cur_batch = []
					new_tokens = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
					# Also bind empty state to AttnWrapperBase for GPT-OSS
					AttnWrapperBase.attention_mask = Attn_Wrapper.attention_mask
					AttnWrapperBase.position_ids = Attn_Wrapper.position_ids
					AttnWrapperBase.cache_seqlens = Attn_Wrapper.cache_seqlens
					AttnWrapperBase.max_seqlen = 0
					AttnWrapperBase.cur_batch = []
				
				if batch:
					Attn_Wrapper.cur_batch = self._local_indices_to_global_seq_ids(batch)
					AttnWrapperBase.cur_batch = Attn_Wrapper.cur_batch

					# OPTIMIZATION: Only check page table if not already verified this batch
					# Between boundaries, batch doesn't change so page table stays valid
					if not _page_table_verified_this_batch:
						# CRITICAL FIX: Ensure page table order matches batch order BEFORE forward pass
						# This is the root cause of KV corruption after resume - if they don't match,
						# cache_seqlens[i] will correspond to wrong page_table[i], causing gibberish output
						if gpu_manager and gpu_manager._gpu_page_table_manager:
							slot_order = list(gpu_manager._gpu_page_table_manager.slot_to_seq_id) if gpu_manager._gpu_page_table_manager.slot_to_seq_id else []
							batch_global_order = Attn_Wrapper.cur_batch
							if slot_order != batch_global_order:
								# Fix: Rebuild page table to match batch order
								gpu_manager.rebuild_page_table(batch_global_order)
						_page_table_verified_this_batch = True
				
				# NOTE: Do NOT skip forward pass even with empty batch!
				# MoE models have all-to-all collective operations that ALL ranks must participate in.
				# Skipping would cause deadlock as other ranks wait for this rank.
				
				# KV append callback
				# NOTE: v_tensor is optional for backward compatibility with MLA models (DeepSeek)
				# GPT-OSS uses GQA with separate K and V, so it passes both tensors
				current_batch = list(batch)
				def kv_append_callback(layer_idx: int, k_tensor: torch.Tensor, v_tensor: torch.Tensor = None):
					self._append_decode_kv_to_host_async(layer_idx, current_batch, k_tensor, v_tensor)
				Attn_Wrapper.kv_append_callback = kv_append_callback
				# Also bind to AttnWrapperBase for models using new wrapper system (e.g., GPT-OSS)
				AttnWrapperBase.kv_append_callback = kv_append_callback

				# Forward
				# CRITICAL: Pass position_ids to model to ensure correct RoPE positioning during decode.
				# Without this, the model generates position_ids = [[0]] for all decode steps,
				# causing RoPE to be applied at position 0 instead of the actual token position.
				outputs = self.model(
					new_tokens,
					attention_mask=Attn_Wrapper.attention_mask,
					position_ids=Attn_Wrapper.position_ids,
					use_cache=False
				)
				new_tokens_out = self._select_tokens(outputs.logits[:, -1, :])

			new_tokens = new_tokens_out

			# DEBUG: Print sampled tokens and logits comparison
			if os.environ.get("BATCHGEN_DEBUG_DECODE", "0") == "1" and local_iteration <= 5:
				print(f"\n[DECODE DEBUG] === Iteration {local_iteration} ===")
				token_ids = new_tokens[:5].flatten().tolist()
				print(f"[DECODE DEBUG] sampled_tokens[:5]: {token_ids}")
				# Decode tokens to show actual text
				try:
					decoded_tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
					print(f"[DECODE DEBUG] decoded_text[:5]: {decoded_tokens}")
				except Exception as e:
					print(f"[DECODE DEBUG] decode error: {e}")
				# Check if all tokens are the same
				if len(new_tokens) >= 2:
					all_same = all(new_tokens[i].item() == new_tokens[0].item() for i in range(min(5, len(new_tokens))))
					if all_same:
						print(f"[DECODE DEBUG] *** WARNING: All sequences sampled SAME token! ***")
				# Show logits for first 3 sequences
				if os.environ.get("BATCHGEN_DEBUG_DECODE_LOGITS", "0") == "1":
					logits = outputs.logits[:, -1, :]
					for i in range(min(3, len(logits))):
						top_vals, top_ids = torch.topk(logits[i], k=5)
						print(f"[DECODE DEBUG] seq{i} top5_ids={top_ids.tolist()}, top5_vals={[f'{v:.2f}' for v in top_vals.tolist()]}")

			# Optimization: Single GPU→CPU transfer for all tokens (vs N transfers in loop)
			# This avoids N GPU synchronizations which cause heavy CPU overhead
			new_tokens_cpu = new_tokens.cpu()

			# Update sequences (reuse batch_sequences from forward pass setup)
			for i, (local_idx, seq) in enumerate(zip(batch, batch_sequences)):
				if self._is_sequence_completed(seq):
					continue

				decode_pos = seq.decoded_length
				self.query_book[local_idx].decoded_tokens[:, decode_pos] = new_tokens_cpu[i]

				attn_mask = self.query_book[local_idx].encoded["attention_mask"][0]
				next_pos = seq.current_context_length
				if next_pos < attn_mask.shape[0]:
					attn_mask[next_pos] = 1

				seq.decoded_length += 1
				seq.current_context_length += 1

				# Use CPU tensor to avoid GPU sync
				if self._should_stop_at_eos(new_tokens_cpu[i].item()):
					seq.eos_reached = True

				if seq.decoded_length >= self.max_decoding_length:
					seq.eos_reached = True
			
			self._cumulative_forward_ms += (time.perf_counter() - forward_start) * 1000

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

		# Also cleanup AttnWrapperBase for models using new wrapper system (e.g., GPT-OSS)
		AttnWrapperBase.gpu_paged_kv_manager = None
		AttnWrapperBase.host_paged_kv_worker_view = None
		AttnWrapperBase.cache_seqlens = None
		AttnWrapperBase.attention_mask = None
		AttnWrapperBase.position_ids = None
		AttnWrapperBase.max_seqlen = None
		AttnWrapperBase.cur_batch = None
		AttnWrapperBase.kv_append_callback = None
		
		# Summary (uses cumulative counters for accurate cross-round totals)
		# Only show when BATCHGEN_CB_LOG=DEBUG
		if self.rank == 0 and self._cumulative_decode_boundaries > 0 and BATCHGEN_CB_DEBUG:
			avg_forward = self._cumulative_forward_ms / self._cumulative_decode_iterations if self._cumulative_decode_iterations > 0 else 0
			avg_round = self._cumulative_boundary_ms / self._cumulative_decode_boundaries
			logging.debug(
				f"\n{'='*50}\n"
				f"DECODE SUMMARY (Rank 0)\n"
				f"{'='*50}\n"
				f"Total Iterations: {self._cumulative_decode_iterations}, Total Rounds: {self._cumulative_decode_boundaries}\n"
				f"Avg forward: {avg_forward:.2f}ms\n"
				f"Avg round overhead: {avg_round:.2f}ms\n"
				f"Round overhead/token: {avg_round / self.DECISION_INTERVAL:.3f}ms\n"
				f"{'='*50}"
			)

		return decode_uuids, batch

	def _wait_pending_kv_append_tasks(self) -> int:
		"""
		Wait for all pending KV append tasks at page boundary.
		Returns the number of tasks that were waited for.
		
		CRITICAL: Also syncs CUDA to ensure all D2H DMA operations complete.
		Without this, KV data may not be fully written to host memory when
		sequences are later resumed, causing KV corruption.
		"""
		if not hasattr(self, '_pending_kv_append_tasks'):
			return 0
		
		num_tasks = len(self._pending_kv_append_tasks)
		for task in self._pending_kv_append_tasks:
			if task is not None:
				task.wait()
		
		# CRITICAL FIX: Sync CUDA after waiting for tasks
		# The async tasks use a separate CUDA stream for D2H copies.
		# Even though each task internally syncs its stream via cudaEventSynchronize,
		# we need a full device sync to ensure ALL pending operations complete
		# before we allow GPU pages to be freed/reused.
		if num_tasks > 0:
			torch.cuda.synchronize(self.torch_device)
		
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
			# Clear the page table to empty state when batch is empty
			Attn_Wrapper.cur_batch = []
			gpu_manager.clear_page_table()
			return
		
		global_ids = self._local_indices_to_global_seq_ids(batch)
		gpu_manager.rebuild_page_table(global_ids)
		Attn_Wrapper.cur_batch = global_ids

	def _append_decode_kv_to_host_async(
		self,
		layer_idx: int,
		batch: List[int],
		k_tensor: torch.Tensor,
		v_tensor: torch.Tensor = None,
	) -> None:  # Returns None, not the task
		"""
		Async append - adds task to pending list, does NOT wait.

		CRITICAL: Must keep tensor references alive until async operation completes!
		GPT-OSS uses GQA with separate K and V caches, so v_tensor must be passed.
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
		if v_tensor is not None and v_tensor.dim() == 3:
			v_tensor = v_tensor.unsqueeze(2)

		# NaN DETECTION: Check for NaN in KV tensor BEFORE appending to host
		# This catches attention computation issues that would propagate to host KV
		if layer_idx == 0 and torch.isnan(k_tensor).any():
			nan_mask = torch.isnan(k_tensor).any(dim=-1).any(dim=-1).any(dim=-1)  # [batch]
			nan_indices = torch.where(nan_mask)[0].tolist()
			nan_seq_info = []
			for idx in nan_indices:
				if idx < len(batch):
					local_idx = batch[idx]
					uuid = self._local_to_uuid_map.get(local_idx, "unknown")
					seq = self.global_batch.get_sequence(uuid) if uuid != "unknown" else None
					nan_seq_info.append({
						'batch_idx': idx,
						'local_idx': local_idx,
						'uuid': uuid[:8] if uuid != "unknown" else "unknown",
						'global_idx': seq.global_idx if seq else -1,
						'ctx_len': seq.current_context_length if seq else -1,
					})
			logging.error(
				f"[KV-NaN-DETECT] Rank {self.rank}: NaN detected in k_tensor BEFORE host append! "
				f"layer={layer_idx}, k_tensor_shape={list(k_tensor.shape)}, "
				f"affected_seqs={nan_seq_info}"
			)
		
		# CRITICAL: Sync default stream before launching async D2H copy.
		# The k_tensor was computed on the default stream, but the async D2H copy
		# uses a separate copy stream. Without this sync, the copy stream might
		# start reading k_tensor before the default stream has finished writing it.
		# This is the root cause of KV corruption after decode interruption/resume.
		torch.cuda.current_stream(self.torch_device).synchronize()
		
		task = worker_view.async_append_decode_kv_to_host(
			layer_idx=layer_idx,
			sequence_ids=sequence_ids,
			k_tensor=k_tensor,
			v_tensor=v_tensor,  # GQA models (GPT-OSS) have separate V; MLA models pass None
			sequence_lengths=sequence_lengths,
		)

		# CRITICAL FIX: Store tensor references alongside task to prevent GC
		# PyTorch's CUDA caching allocator can reuse memory if tensor is dereferenced
		# while async operation is still reading from it!
		# Must store BOTH k and v tensors for GQA models
		if not hasattr(self, '_pending_kv_append_tensors'):
			self._pending_kv_append_tensors = []
		self._pending_kv_append_tensors.append(k_tensor)
		if v_tensor is not None:
			self._pending_kv_append_tensors.append(v_tensor)
		
		# Add to pending list - will be waited at page boundary
		self._pending_kv_append_tasks.append(task)

		# THROTTLING FIX: Prevent "Resource temporarily unavailable" (EAGAIN) error
		# std::async creates a new thread for each task. With 61 layers and 64 tokens
		# per boundary, we can hit ~3900 concurrent threads per boundary interval.
		# Wait and clear when threshold is reached to avoid exhausting system thread limits.
		# Threshold: 256 tasks (conservative to leave room for other threads)
		MAX_PENDING_KV_TASKS = 256
		if len(self._pending_kv_append_tasks) >= MAX_PENDING_KV_TASKS:
			self._wait_pending_kv_append_tasks()

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
			# Mark that this sequence has received its initial GPU reservation
			seq.mark_initial_gpu_reservation_done()
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
			uuid = self._local_to_uuid_map.get(local_idx)
			if uuid is None:
				continue
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				continue
			pos = max(0, seq.decoded_length - 1)
			query_entry = self.query_book.get(local_idx)
			if query_entry is None:
				continue
			token = query_entry.decoded_tokens[:, pos:pos+1]
			tokens.append(token)
		
		result = torch.cat(tokens, dim=0).to(self.torch_device) if tokens else torch.empty((0, 1), dtype=torch.int64, device=self.torch_device)
		
		if result.shape[0] != len(batch):
			logging.error(
				f"Rank {self.rank}: _rebuild_input_tokens MISMATCH: "
				f"batch_size={len(batch)}, result_size={result.shape[0]}, "
				f"tokens_collected={len(tokens)}"
			)
		
		return result


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
			
			# Page boundary check - use DECISION_INTERVAL
			if new_token_idx > 0 and new_token_idx % self.DECISION_INTERVAL == 0:
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
					new_tokens = self._select_tokens(new_tokens.logits[:, -1, :])
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
					new_tokens = self._select_tokens(new_tokens.logits[:, -1, :])
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
		BaseModuleWrapper.phase = phase

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
			if self.rank == 0:
				logging.debug("Skipping NVSHMEM initialization; BATCHGEN_ENABLE_ALL_TO_ALL is disabled")
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
		# Use maximum timeout (about 24 days) to handle long server idle periods
		# timedelta max is about 999999999 days, but NCCL has internal limits
		# 35791 minutes ≈ 24.8 days, which is close to the max NCCL supports
		timeout = timedelta(days=24)
		try:
			dist.init_process_group(
				backend="nccl",
				init_method="tcp://" + self.dist_init_addr,
				world_size=self.world_size,
				rank=self.global_rank,
				device_id=torch.device(f"cuda:{self.local_rank}"),
				timeout=timeout,
			)
			logging.info(f"Rank {self.rank}: torch.distributed initialized with timeout={timeout}")
		except RuntimeError as e:
			logging.error(f"Failed to initialize torch distributed: {e}")
			raise

	def _ensure_dist_healthy(self) -> bool:
		"""
		Ensure torch.distributed is healthy before starting a new batch.

		This performs a lightweight health check first. Only if the check fails
		does it attempt to reinitialize with coordinated retries.

		The key insight is: DON'T destroy a working connection. Only reinit if broken.

		Returns True if healthy, False if reinit failed after all retries.
		"""
		MAX_REINIT_RETRIES = 5
		INITIAL_RETRY_DELAY = 2.0  # seconds

		# Step 1: Check if dist is even initialized
		if not dist.is_initialized():
			logging.warning(f"Rank {self.rank}: torch.distributed not initialized, attempting init...")
			return self._coordinated_dist_reinit(MAX_REINIT_RETRIES, INITIAL_RETRY_DELAY)

		# Step 2: Quick health check - use async op with short timeout
		try:
			health_tensor = torch.ones(1, device=self.torch_device)
			work = dist.all_reduce(health_tensor, op=dist.ReduceOp.SUM, async_op=True)

			# Wait with short timeout (10 seconds should be enough for healthy connection)
			success = work.wait(timeout=timedelta(seconds=10))
			if not success:
				raise RuntimeError("Health check timed out")

			expected = float(self.world_size)
			if abs(health_tensor.item() - expected) > 1e-6:
				raise RuntimeError(f"Health check mismatch: got {health_tensor.item()}, expected {expected}")

			logging.debug(f"Rank {self.rank}: torch.distributed health check passed")
			return True

		except Exception as e:
			logging.warning(f"Rank {self.rank}: torch.distributed health check failed: {e}")
			logging.info(f"Rank {self.rank}: Attempting coordinated reinit...")
			return self._coordinated_dist_reinit(MAX_REINIT_RETRIES, INITIAL_RETRY_DELAY)

	def _coordinated_dist_reinit(self, max_retries: int, initial_delay: float) -> bool:
		"""
		Perform coordinated torch.distributed reinitialization with retries.

		The challenge: when NCCL is broken, we can't use NCCL to coordinate.
		Solution: Use exponential backoff retries. Rank 0 (which hosts TCPStore)
		will eventually be ready when other ranks retry.

		Args:
			max_retries: Maximum number of reinit attempts
			initial_delay: Initial delay between retries (doubles each attempt)

		Returns:
			True if reinit succeeded, False otherwise
		"""
		delay = initial_delay

		for attempt in range(max_retries):
			logging.info(f"Rank {self.rank}: Reinit attempt {attempt + 1}/{max_retries}")

			# Step 1: Clean up existing process group
			if dist.is_initialized():
				try:
					dist.destroy_process_group()
					logging.debug(f"Rank {self.rank}: Destroyed existing process group")
				except Exception as e:
					logging.warning(f"Rank {self.rank}: Error destroying process group: {e}")

			# Step 2: Clean up PyNccl communicator (must be done after destroying dist)
			if hasattr(self, 'comm') and self.comm is not None:
				try:
					self.comm.destroy()
					logging.debug(f"Rank {self.rank}: Destroyed PyNccl communicator")
				except Exception as e:
					logging.warning(f"Rank {self.rank}: Error destroying PyNccl communicator: {e}")
				self.comm = None

			# Step 3: Wait before retry (exponential backoff)
			# Rank 0 waits less so it sets up TCPStore first
			rank_delay = delay * (0.5 if self.rank == 0 else 1.0)
			logging.debug(f"Rank {self.rank}: Waiting {rank_delay:.1f}s before reinit...")
			time.sleep(rank_delay)

			# Step 4: Try to reinitialize
			try:
				self._init_torch_dist()
				logging.info(f"Rank {self.rank}: torch.distributed reinitialized successfully on attempt {attempt + 1}")
				return True
			except Exception as e:
				logging.warning(f"Rank {self.rank}: Reinit attempt {attempt + 1} failed: {e}")
				delay *= 2  # Exponential backoff

		logging.error(f"Rank {self.rank}: Failed to reinitialize torch.distributed after {max_retries} attempts")
		return False

	def _check_and_reinit_distributed(self) -> bool:
		"""
		Check if torch.distributed is healthy. If not, attempt to reinitialize.
		Returns True if distributed is healthy (or was successfully reinitialized).
		Returns False if reinitialization failed.
		"""
		if not dist.is_initialized():
			logging.warning(f"Rank {self.rank}: torch.distributed not initialized, attempting to initialize...")
			try:
				self._init_torch_dist()
				return True
			except Exception as e:
				logging.error(f"Rank {self.rank}: Failed to initialize torch.distributed: {e}")
				return False
		
		# Perform a quick health check with a short timeout
		try:
			# Use a simple all_reduce as a health check
			health_tensor = torch.ones(1, device=self.torch_device)
			
			# Create a new process group with short timeout for health check
			# This avoids blocking forever if the connection is stale
			work = dist.all_reduce(health_tensor, op=dist.ReduceOp.SUM, async_op=True)
			
			# Wait with a short timeout (30 seconds)
			success = work.wait(timeout=timedelta(seconds=30))
			
			if not success:
				raise RuntimeError("Health check timed out")
			
			# Verify the result
			expected = float(self.world_size)
			if abs(health_tensor.item() - expected) > 1e-6:
				raise RuntimeError(f"Health check result mismatch: got {health_tensor.item()}, expected {expected}")
			
			logging.debug(f"Rank {self.rank}: Distributed health check passed")
			return True
			
		except Exception as e:
			logging.warning(f"Rank {self.rank}: Distributed health check failed: {e}")
			logging.info(f"Rank {self.rank}: Attempting to reinitialize torch.distributed...")
			
			# Destroy and reinitialize
			try:
				dist.destroy_process_group()
			except Exception as destroy_e:
				logging.warning(f"Rank {self.rank}: Error destroying process group: {destroy_e}")
			
			try:
				self._init_torch_dist()
				logging.info(f"Rank {self.rank}: Successfully reinitialized torch.distributed")
				return True
			except Exception as reinit_e:
				logging.error(f"Rank {self.rank}: Failed to reinitialize torch.distributed: {reinit_e}")
				return False

	def _proactive_dist_reinit(self) -> None:
		"""
		[DEPRECATED] Proactively destroy and reinitialize torch.distributed.

		WARNING: This function is NO LONGER USED in production and should NOT be called
		between batches. Destroying/reinitializing torch.distributed unconditionally in
		multi-node setups causes NCCL connection failures because ranks destroy/reinit
		at different times.

		USE INSTEAD: _ensure_dist_healthy()
		- Performs a lightweight health check first
		- Only reinitializes if the connection is actually broken
		- Uses coordinated retries with exponential backoff

		This function is kept only for emergency debugging scenarios.
		"""
		logging.info(f"Rank {self.rank}: Proactively reinitializing torch.distributed for new batch")
		
		# Step 1: Destroy existing PyNccl communicator
		# This must be done BEFORE destroying torch.distributed, and will be recreated
		# lazily in generate() after torch.distributed is reinitialized.
		if hasattr(self, 'comm') and self.comm is not None:
			try:
				self.comm.destroy()
				logging.debug(f"Rank {self.rank}: Destroyed PyNccl communicator")
			except Exception as e:
				logging.warning(f"Rank {self.rank}: Error destroying PyNccl communicator: {e}")
			self.comm = None
		
		if hasattr(self, '_nccl_group') and self._nccl_group is not None:
			try:
				del self._nccl_group
				self._nccl_group = None
				gc.collect()
				logging.debug(f"Rank {self.rank}: Destroyed PyNccl group")
			except Exception as e:
				logging.warning(f"Rank {self.rank}: Error destroying PyNccl group: {e}")
			self._nccl_group = None
		
		# Increment port for PyNccl to avoid "Address already in use" on recreate
		if hasattr(self, '_nccl_port'):
			self._nccl_port += 1
			logging.debug(f"Rank {self.rank}: Incremented PyNccl port to {self._nccl_port}")
		
		# Step 2: Destroy existing process group if it exists
		if dist.is_initialized():
			try:
				dist.destroy_process_group()
				logging.debug(f"Rank {self.rank}: Destroyed existing process group")
			except Exception as e:
				logging.warning(f"Rank {self.rank}: Error destroying process group: {e}")
		
		# Step 3: Small sleep to allow socket cleanup
		# This helps prevent "Address already in use" errors
		time.sleep(0.5)
		
		# Step 4: Reinitialize torch.distributed
		try:
			self._init_torch_dist()
			logging.info(f"Rank {self.rank}: torch.distributed reinitialized successfully")
		except Exception as e:
			logging.error(f"Rank {self.rank}: Failed to reinitialize torch.distributed: {e}")
			raise RuntimeError(f"Rank {self.rank}: Failed to reinitialize torch.distributed: {e}")
	
	def _check_and_reinit_pynccl(self) -> bool:
		"""
		Check if PyNccl communicator is healthy. If not, attempt to reinitialize.
		Returns True if communicator is healthy (or was successfully reinitialized).
		"""
		if self.comm is None:
			# Will be lazily initialized in generate()
			return True

		# Skip health check if communicator is not available (e.g., single GPU)
		if not self.comm.available:
			logging.debug(f"Rank {self.rank}: PyNccl communicator not available, skipping health check")
			return True

		try:
			# Quick health check using PyNccl all_reduce
			# CRITICAL: Must enable the communicator first - it's disabled by default after init
			health_tensor = torch.ones(1, device=self.torch_device, dtype=torch.float32)
			with self.comm.change_state(enable=True):
				self.comm.all_reduce(health_tensor, op=dist.ReduceOp.SUM, stream=torch.cuda.current_stream())
			torch.cuda.synchronize(self.torch_device)

			expected = float(self.world_size)
			if abs(health_tensor.item() - expected) > 1e-6:
				raise RuntimeError(f"PyNccl health check mismatch: got {health_tensor.item()}, expected {expected}")

			logging.debug(f"Rank {self.rank}: PyNccl health check passed")
			return True
			
		except Exception as e:
			logging.warning(f"Rank {self.rank}: PyNccl health check failed: {e}")
			logging.info(f"Rank {self.rank}: Attempting to reinitialize PyNccl communicator...")

			# Destroy old communicator
			try:
				if self.comm is not None:
					self.comm.destroy()
					self.comm = None
					logging.info(f"Rank {self.rank}: NCCL communicator destroyed successfully")
			except Exception as destroy_e:
				logging.warning(f"Rank {self.rank}: Error destroying PyNccl communicator: {destroy_e}")
				self.comm = None

			# Destroy old group (releases TCPStore and port)
			try:
				if self._nccl_group is not None:
					# The group's store should be garbage collected when group is deleted
					del self._nccl_group
					self._nccl_group = None
					# Force garbage collection to release TCPStore socket
					gc.collect()
					logging.info(f"Rank {self.rank}: NCCL group destroyed successfully")
			except Exception as group_e:
				logging.warning(f"Rank {self.rank}: Error destroying NCCL group: {group_e}")
				self._nccl_group = None

			# Synchronize all ranks before any tries to recreate (uses torch.distributed)
			# This ensures all ranks have released their connections before rank 0
			# tries to create a new TCPStore server
			try:
				if dist.is_initialized():
					dist.barrier()
					logging.debug(f"Rank {self.rank}: Barrier after NCCL cleanup passed")
			except Exception as barrier_e:
				logging.warning(f"Rank {self.rank}: Barrier after NCCL cleanup failed: {barrier_e}")

			# Find next available port for reinitialization
			# Rank 0 finds the port, then broadcasts to all ranks
			if not hasattr(self, '_nccl_port'):
				self._nccl_port = 20003
			comm_master_addr = os.getenv("COMM_MASTER_ADDR", "127.0.0.1")
			if self.rank == 0:
				try:
					self._nccl_port = _find_available_port(comm_master_addr, self._nccl_port + 1)
					logging.debug(f"Rank 0: Found available port {self._nccl_port} for PyNccl reinit")
				except RuntimeError as e:
					logging.error(f"Rank 0: Failed to find available port: {e}")
					return False
			# Broadcast port to all ranks
			port_tensor = torch.tensor([self._nccl_port], dtype=torch.int32, device=self.torch_device)
			dist.broadcast(port_tensor, src=0)
			self._nccl_port = port_tensor.item()
			logging.debug(f"Rank {self.rank}: Next PyNccl port will be {self._nccl_port}")

			# Delay to allow OS to fully release resources
			# Rank 0 needs extra time since it's the TCPStore server
			if self.rank == 0:
				time.sleep(1.0)
			else:
				time.sleep(0.5)

			# Will be reinitialized lazily in generate()
			return True

	def _unregister_fp8_weights(self):
		# Skip FP8 unregistration for models that don't use FP8 (e.g., GPT-OSS uses MXFP4)
		if not hasattr(self.loaded_model_config, 'first_k_dense_replace'):
			return

		for layer_idx in range(len(self.model.model.layers)):
			attn_module = self.model.model.layers[layer_idx].self_attn
			if hasattr(attn_module, '_unregister_fp8_weights'):
				attn_module._unregister_fp8_weights()
			if layer_idx >= self.loaded_model_config.first_k_dense_replace:
				if hasattr(self.model.model.layers[layer_idx].mlp.shared_experts, '_unregister_fp8_weights'):
					self.model.model.layers[layer_idx].mlp.shared_experts._unregister_fp8_weights()
				for routed_expert_idx in range(self.model_config.num_local_experts):
					if hasattr(self.model.model.layers[layer_idx].mlp.experts[routed_expert_idx], '_unregister_fp8_weights'):
						self.model.model.layers[layer_idx].mlp.experts[routed_expert_idx]._unregister_fp8_weights()
				if hasattr(self.model.model.layers[layer_idx].mlp, "cleanup"):
					self.model.model.layers[layer_idx].mlp.cleanup()

	def deep_free_model_memory(self):
		"""Release model memory without CPU transfer overhead.

		Previous implementation moved model to CPU before deletion, causing
		unnecessary PCIe traffic for large models. This minimal approach:
		1. Synchronizes CUDA to ensure pending ops complete
		2. Deletes model reference directly
		3. Releases memory back to CUDA allocator
		"""
		if not hasattr(self, 'model') or self.model is None:
			return

		# Ensure all GPU operations complete before deletion
		if torch.cuda.is_available():
			torch.cuda.synchronize(self.torch_device)

		# Delete model directly without CPU transfer
		del self.model
		self.model = None

		# Release memory
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()

	def _reset_for_new_batch(self) -> None:
		"""
		Reset batch-specific state to prepare for a new batch.
		Does NOT reinitialize core_engine, parallel_manager, or other heavy components.
		NOTE: We keep self.comm (PyNcclCommunicator) alive across batches to avoid re-initialization overhead.
		NOTE: torch.distributed is initialized at server startup. If NCCL connection is stale after
		      long idle periods, we attempt coordinated reinit with retries.
		"""
		logging.info(f"Rank {self.rank}: Resetting state for new batch")

		# Check if torch.distributed needs reinitialization
		# This only reinits if the connection is actually broken, not unconditionally
		if not self._ensure_dist_healthy():
			raise RuntimeError(f"Rank {self.rank}: Failed to ensure healthy torch.distributed connection")

		# Synchronize all ranks before cleanup
		dist.barrier()
		self._ignore_eos = False
		# Reset logging flags for new batch (to log sampling mode once per batch)
		self._logged_greedy = False
		self._logged_sampling = False

		# NOTE: We intentionally do NOT destroy self.comm here.
		# PyNccl communicator is reused across batches to avoid:
		# 1. NCCL re-initialization overhead
		# 2. TCPStore port binding issues
		# The communicator is only destroyed when the worker is shut down.
		
		# 1. Release any remaining host KV pages for THIS RANK's sequences
		# NOTE: Many sequences may already be released during normal decode completion.
		# We only need to cleanup sequences that might still be registered.
		if hasattr(self, 'global_batch') and self.global_batch is not None:
			try:
				worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
				if worker_view is not None and hasattr(self, '_uuid_to_local_map') and self._uuid_to_local_map:
					# Collect all global_idx values for this rank's sequences
					global_ids_to_release = []
					for uuid in self._uuid_to_local_map.keys():
						seq = self.global_batch.get_sequence(uuid)
						if seq is not None:
							global_ids_to_release.append(seq.global_idx)
					
					if global_ids_to_release:
						logging.info(
							f"Rank {self.rank}: Attempting to release host KV for {len(global_ids_to_release)} sequences"
						)
						# Try to release each sequence individually to handle already-released ones
						released_count = 0
						for seq_id in global_ids_to_release:
							try:
								worker_view.release_sequence_pages([seq_id])
								released_count += 1
							except Exception:
								# Sequence was already released during decode - this is normal
								pass
						logging.info(f"Rank {self.rank}: Released {released_count}/{len(global_ids_to_release)} sequences (others already released)")
			except Exception as e:
				logging.warning(f"Rank {self.rank}: Failed to cleanup host KV: {e}")
		
		# 2. Reset batch completion flag
		self._batch_completed = False
		
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
		
		# 7. Reset GPU KV tracking
		self._sequences_with_gpu_kv = set()
		
		# 8. Clean up model weights (but NOT core_engine or parallel_manager)
		if hasattr(self, 'model') and self.model is not None:
			try:
				self.deep_free_model_memory()
			except Exception as e:
				logging.warning(f"Rank {self.rank}: Failed to cleanup model: {e}")
		self.model = None
		
		# 9. Clear CUDA cache
		torch.cuda.empty_cache()
		torch.cuda.synchronize(self.torch_device)
		
		# 10. Force garbage collection
		gc.collect()
		
		# Synchronize all ranks after cleanup
		dist.barrier()
		
		logging.info(f"Rank {self.rank}: State reset completed")

