import concurrent.futures
import copy
import functools
import json
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
from batchgen.deprecation import LegacyInferenceDeprecated

# Use new wrapper system - Attn_Wrapper/Expert_Wrapper are aliases for backward compatibility
from batchgen.models.wrappers import BaseModuleWrapper, AttnWrapperBase, ExpertWrapperBase
# Phase C: GLM-5-specific ClassVars (dispatch_trace_*, _dsa_short_count,
# glm5_decode_*_slot_indices, glm5_dsa_*) live on GLM5AttnWrapper, not
# AttnWrapperBase, per audit §A finding #8.
from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper
# Aliases for backward compatibility with existing code
Attn_Wrapper = AttnWrapperBase
Expert_Wrapper = ExpertWrapperBase

from .config.config import EngineConfig
from .scheduler.host_mem import get_physical_memory_info

from batchgen.parameter_server_client import ParameterServerClient
from batchgen import lifespan


from batchgen.lifespan import SeqEvent

REP_DETECTION = os.environ.get("BATCHGEN_REP_DETECTION", "1") == "1"

def _check_repeating_pattern(token_ids: torch.Tensor, decoded_length: int,
                              min_pattern: int = 2, max_pattern: int = 100,
                              min_count: int = 32) -> bool:
	"""Check if the tail of token_ids has a repeating N-gram pattern.

	Scans pattern lengths from min_pattern to max_pattern. Returns True if
	the last (pattern_len * min_count) tokens consist of the same pattern
	repeated min_count times.
	"""
	if decoded_length < min_pattern * min_count:
		return False
	max_check = min(max_pattern + 1, decoded_length // min_count + 1)
	for pattern_len in range(min_pattern, max_check):
		is_repeat = True
		for offset in range(pattern_len):
			target = token_ids[decoded_length - 1 - offset].item()
			for rep in range(1, min_count):
				if token_ids[decoded_length - 1 - offset - pattern_len * rep].item() != target:
					is_repeat = False
					break
			if not is_repeat:
				break
		if is_repeat:
			return True
	return False
from tqdm import trange
import gc
import numpy as np
from datetime import timedelta
from contextlib import contextmanager
from dataclasses import dataclass, replace
import torch.distributed._symmetric_memory as symm_mem
from batchgen.distributed.utils import StatelessProcessGroup
from batchgen.distributed.device_communicators.pynccl import PyNcclCommunicator


from .utils import torch_gpu_mem_usage, create_position_ids_from_attention_mask
from .get_initializer import get_initializer
from .get_parallel_strategy_manager import get_parallel_strategy_manager
from batchgen.batch_order import (
	batch_matches_expected_uuid_order,
	build_prefill_sequence_spans,
	local_indices_to_uuid_order,
	prefill_sequence_spans_to_cu_seqlens,
	prefill_sequence_spans_to_global_seq_ids,
)
from batchgen.query_book import (
	QueryBookEntry as query,
	bind_local_sequence_to_query_book,
	make_query_book_entry,
	release_local_query_slot,
)
from batchgen.utils import config_torch_module_initializer
from batchgen.config.model_name_utils import is_kimi_k25_backend_model
from batchgen.models.glm.glm5.cuda_graph_policy import (
	glm5_any_cuda_graph_requested_for_model,
	glm5_dsa_cuda_graph_requested_for_model,
	glm5_dsa_full_cuda_graph_requested,
	glm5_moe_cuda_graph_requested_for_model,
	glm5_segmented_cuda_graph_requested_for_model,
	glm5_whole_model_cuda_graph_requested_for_model,
)
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager
from batchgen.models.engine_loader import core_engine
from batchgen.worker.indexing import IndexLookupRequest, IndexManager
from batchgen.worker.completion import CompletionContext, CompletionHandler
from batchgen.worker.sync import (
	SyncContext,
	SyncCoordinator,
	TorchDistCollectiveBackend,
)
from batchgen.worker.batch_formation import BatchFormation, BatchFormationContext
from batchgen.worker.prefill import (
	PrefillCandidate,
	PrefillScheduler,
	PrefillSelectionRequest,
)
from batchgen.worker.host_rebalancer import HostKVRebalancer
from batchgen.worker.boundary import (
	BoundaryDecisionRequest,
	BoundaryHandler,
	BoundarySeqMeta,
)
from batchgen.worker.decode import (
	DecodeBatchRequest,
	DecodeCandidate,
	DecodeScheduler,
	estimate_max_decode_replica_batch,
)
from batchgen.worker.kv_manager import (
	GpuKvManagerPlan,
	GpuKvManagerRequest,
	KVCacheManager,
	KVStats,
	KVUtilizationRequest,
	MigrationCandidate,
	MigrationPlanRequest,
	PageTableCapacityRequest,
	TokenBudgetRequest,
	WatermarkTriggerRequest,
)
import dataclasses as _dataclasses

from batchgen.kv_cache.host_kv_mananger_config import (
	build_host_kv_config,
	build_host_kv_worker_view,
	is_dsa_model,
)
from batchgen.kv_cache.dual_kv_cache_coordinator import DualKVCacheCoordinator
from batchgen.kv_cache.dual_host_kv_coordinator import DualAsyncKVTask, DualHostKVCoordinator
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus, INITIAL_GPU_PAGE_BUFFER, EXTENSION_GPU_PAGE_BUFFER, DECISION_FREQUENCY_PAGES, configure_page_buffers
from batchgen.prefill.prepack import (
	prepack_sequences,
	unpack_last_token_logits,
	get_prepack_stats,
	PrepackMetadata,
	build_prefill_micro_batches,
)

# Import modularized components
# FastBoundaryTimingStats: Timing dataclass for page boundary operations
from batchgen.continuous_batching import (
	AdaptiveChunkSizer,
	BoundaryDecisions,
	FastBoundaryTimingStats,
	plan_host_kv_growth_evictions,
	EvictionStrategy,
	LoadingStrategy,
	select_sequences_for_loading,
	validate_boundary_payload_alignment,
)
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

# Optional decode-time invariant check: assert cache_seqlens <= gpu_pages_allocated * PAGE_SIZE
BATCHGEN_DECODE_ASSERT = os.environ.get("BATCHGEN_DECODE_ASSERT", "0") == "1"

# Multi-batch diagnostic logging for investigating metadata corruption
BATCHGEN_MULTI_BATCH_DIAG = os.environ.get("BATCHGEN_MULTI_BATCH_DIAG", "0") == "1"

# Force synchronous KV offload (disable deferred flush) for debugging
BATCHGEN_SYNC_KV = os.environ.get("BATCHGEN_SYNC_KV", "0") == "1"

# Prepack mode for efficient prefill batching (DEPRECATED: use --enable-prepack CLI arg)
# Default: enabled (recommended always on). Use --no-prepack to disable.
BATCHGEN_ENABLE_PREPACK = os.environ.get("BATCHGEN_ENABLE_PREPACK", "1") == "1"

# Optional runtime checks for NaN/Inf in KV tensors (disabled by default)
BATCHGEN_ENABLE_NAN_CHECK = os.environ.get('BATCHGEN_ENABLE_NAN_CHECK', '0') == '1'

# Optional gate for expensive/critical diagnostics (default off in production)
BATCHGEN_ENABLE_CRITICAL_DIAGS = os.environ.get('BATCHGEN_ENABLE_CRITICAL_DIAGS', '0') == '1'

# Max per-rank in-decode batch. Bounds the K2.5 MoE padded buffer (mtp = round_up(world_size *
# this)) so init does not OOM on a large candidate pool, and is the per-rank admission cap so
# the pre-reserved buffer never overflows at runtime. Raise to fill more GPU KV (memory permitting).
_MAX_DECODE_RANK_BSZ = int(os.environ.get("BATCHGEN_MAX_DECODE_RANK_BSZ", "128"))

# The one served model whose workload-independent startup work runs before the
# server reports ready (prepare_kimi_k3_startup). Matched EXACTLY: the sibling
# Kimi-Linear/K2.5 checkpoints share this architecture but not this lifecycle.
KIMI_K3_MODEL_ID = "moonshotai/Kimi-K3"

# Decode budget the startup pass uses. max_decoding_length only sizes scheduler
# hints at this point; the real per-batch value arrives with the pool "init"
# message, which re-enters Init() before any sequence is admitted.
_K3_STARTUP_MAX_DECODING_LENGTH = 4096

# Optional Nsight Systems capture window for decode-forward profiling.
# Start nsys with: --capture-range=cudaProfilerApi --capture-range-end=stop.
BATCHGEN_NSYS_DECODE_PROFILE = os.environ.get("BATCHGEN_NSYS_DECODE_PROFILE", "0") == "1"
BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT = int(
	os.environ.get("BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT", "3")
)
if BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT <= 0:
	raise ValueError("BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT must be positive")
BATCHGEN_NSYS_DECODE_PROFILE_EXIT = os.environ.get("BATCHGEN_NSYS_DECODE_PROFILE_EXIT", "1") == "1"

def _parse_nsys_controller_ranks() -> Set[int]:
	raw = os.environ.get("BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS", "0")
	return {int(part.strip()) for part in raw.split(",") if part.strip()}

BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS = _parse_nsys_controller_ranks()
if BATCHGEN_NSYS_DECODE_PROFILE and not BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS:
	raise ValueError("BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS must not be empty")

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


class _DualAsyncLoadTask:
	"""Forwards .wait() to both primary and aux async KV load tasks.

	Either side may be None (non-DSA model or aux-disabled diagnostic mode);
	.wait() skips None. Used by mid-decode reload paths where we must ensure
	both caches are on-GPU before the next decode step reads them.
	"""
	__slots__ = ("_primary", "_aux")

	def __init__(self, primary_task, aux_task):
		self._primary = primary_task
		self._aux = aux_task

	def wait(self):
		if self._primary is not None:
			self._primary.wait()
		if self._aux is not None:
			self._aux.wait()


@dataclass
class _DualKVLoadPointers:
	sequence_tensor: torch.Tensor
	primary_k_ptrs: torch.Tensor
	primary_v_ptrs: Optional[torch.Tensor]
	primary_page_counts: torch.Tensor
	aux_k_ptrs: torch.Tensor
	aux_v_ptrs: Optional[torch.Tensor]
	aux_page_counts: torch.Tensor


class QueryBookPoolCapacityError(RuntimeError):
	"""A QueryBook pool request exceeded the rows/width actually allocated."""


def allocate_node_shared_int64(
	name: str,
	rows: int,
	width: int,
	is_creator: bool,
	barrier,
) -> Tuple[torch.Tensor, object]:
	"""Map ONE int64 ``[rows, width]`` CPU tensor per node into every worker.

	The tokenized global batch is identical on every rank (``_tokenize_global_batch``
	all-gathers the results to all of them), so each worker used to hold its own
	private copy of the same input-ids table — ``world_size`` duplicates of the
	same bytes, which is what OOM-killed the node.

	``is_creator`` must be true on exactly one rank per node. The creator makes
	the segment (POSIX guarantees it is zero-filled, matching the ``torch.zeros``
	it replaces), everyone waits on ``barrier``, then the rest attach. ``barrier``
	is ``dist.barrier`` in the worker and an ``mp.Barrier`` in tests.

	Returns ``(tensor, shm)``. The caller MUST keep ``shm`` alive for as long as
	the tensor is reachable: the tensor points straight into the mapping.

	Two operational notes: the segment lands in /dev/shm, so the container's
	shm budget has to cover it; and CPython < 3.13 registers a segment with the
	resource_tracker on attach as well as on create, so every non-creator rank
	prints one "leaked shared_memory objects" warning at shutdown. That warning
	is cosmetic — unlink only drops the name, never a live mapping.
	"""
	from multiprocessing import shared_memory

	nbytes = rows * width * 8
	if is_creator:
		try:
			# A crashed predecessor can leave the name behind; reusing its
			# (possibly smaller) segment would silently truncate.
			shared_memory.SharedMemory(name=name).unlink()
		except FileNotFoundError:
			pass
		shm = shared_memory.SharedMemory(name=name, create=True, size=nbytes)
	barrier()
	if not is_creator:
		shm = shared_memory.SharedMemory(name=name)
	if shm.size < nbytes:
		raise QueryBookPoolCapacityError(
			f"shared input_ids segment '{name}' is {shm.size} bytes, "
			f"need {nbytes} ({rows} rows x {width} tokens x 8B)"
		)
	buf = torch.frombuffer(shm.buf, dtype=torch.int64, count=rows * width).view(rows, width)
	# Nobody may unlink/close until every rank has mapped it.
	barrier()
	return buf, shm


class QueryBookBufferPool:
	"""Pre-allocated contiguous buffers for query book tensors.

	Eliminates per-sequence tensor allocation in Phase 3 of _tokenize_global_batch().
	With 16 ranks each creating 12K tensors, allocator contention causes ~19 min init.
	This replaces 24K allocations per rank with 2 large allocations + views.

	``input_ids_buffer`` may be passed in as a node-shared tensor (see
	``allocate_node_shared_int64``) — its contents are identical on every rank,
	so one copy per node is enough. ``decoded_tokens_buffer`` stays PRIVATE: only
	the owning rank writes a sequence's decoded tokens, so the ranks' copies
	legitimately differ.

	``input_ids_width`` is the widest ``seq_extended_size`` the pool can serve.
	It is sized from the batch that is actually being admitted, NOT from the
	model context length: at K3's 1,048,576-token context a 10240-slot pool
	would be 80 GiB of zeros per worker.
	"""

	def __init__(
		self,
		num_sequences: int,
		input_ids_width: int,
		max_decoding_length: int,
		pad_token_id: int = 0,
		input_ids_buffer: Optional[torch.Tensor] = None,
		input_ids_shm: object = None,
	):
		if input_ids_buffer is None:
			input_ids_buffer = torch.zeros((num_sequences, input_ids_width), dtype=torch.long)
		elif tuple(input_ids_buffer.shape) != (num_sequences, input_ids_width):
			raise QueryBookPoolCapacityError(
				f"shared input_ids buffer has shape {tuple(input_ids_buffer.shape)}, "
				f"pool needs ({num_sequences}, {input_ids_width})"
			)
		self.input_ids_buffer = input_ids_buffer
		self.input_ids_shm = input_ids_shm
		self.decoded_tokens_buffer = torch.full((num_sequences, max_decoding_length), pad_token_id, dtype=torch.int64)
		self.pad_token_id = pad_token_id
		self.num_sequences = num_sequences
		self.input_ids_width = input_ids_width
		self.max_decoding_length = max_decoding_length
		self._free_slots: set = set()
		self._next_slot: int = 0

	def adopt(self, old: "QueryBookBufferPool") -> None:
		"""Carry contents and slot bookkeeping over from a superseded pool."""
		rows = min(self.num_sequences, old.num_sequences)
		cols = min(self.input_ids_width, old.input_ids_width)
		self.input_ids_buffer[:rows, :cols] = old.input_ids_buffer[:rows, :cols]
		dec = min(self.max_decoding_length, old.max_decoding_length)
		self.decoded_tokens_buffer[:rows, :dec] = old.decoded_tokens_buffer[:rows, :dec]
		self._free_slots = set(old._free_slots)
		self._next_slot = old._next_slot

	def reset(self) -> None:
		"""Return the pool to its just-allocated state (legacy per-batch reuse)."""
		self._free_slots = set()
		self._next_slot = 0
		self.input_ids_buffer.zero_()
		self.decoded_tokens_buffer.fill_(self.pad_token_id)

	def allocate_slot(self) -> int:
		if self._free_slots:
			slot = self._free_slots.pop()
			# Clear stale data from previous occupant to prevent EOS/token contamination
			self.decoded_tokens_buffer[slot, :] = self.pad_token_id
			self.input_ids_buffer[slot, :] = 0
			return slot
		slot = self._next_slot
		if slot >= self.num_sequences:
			raise QueryBookPoolCapacityError(
				f"QueryBookBufferPool exhausted: {self.num_sequences} slots used "
				f"(raise --max-pool-size)"
			)
		self._next_slot += 1
		return slot

	def free_slot(self, slot: int):
		self._free_slots.add(slot)

	def get_input_ids_view(self, slot: int, seq_extended_size: int) -> torch.Tensor:
		if seq_extended_size > self.input_ids_width:
			# Slicing would silently hand back a SHORT view and truncate the
			# prompt. The pool must be grown instead (_ensure_buffer_pool).
			raise QueryBookPoolCapacityError(
				f"input_ids view of {seq_extended_size} tokens requested from a pool "
				f"allocated {self.input_ids_width} tokens wide (slot={slot})"
			)
		return self.input_ids_buffer[slot:slot+1, :seq_extended_size]

	def get_decoded_tokens_view(self, slot: int) -> torch.Tensor:
		return self.decoded_tokens_buffer[slot:slot+1, :]


@dataclass
class InputArguments:
	"""Input arguments as a dataclass with type hints"""
	huggingface_ckpt_name: str
	hf_cache_dir: Optional[str] = None
	cache_dir: Optional[str] = None
	converted_ckpt_dir: Optional[str] = None
	queries: Optional[List[str]] = None
	max_prompt_length: Optional[int] = None
	padding_length: Optional[int] = None  # Deprecated alias for older initializers.
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
	pre_dequantize_weights: bool = False
	distributed_weight_config: Optional[str] = None

	def __post_init__(self):
		if self.max_prompt_length is None and self.padding_length is not None:
			self.max_prompt_length = self.padding_length
		elif self.max_prompt_length is not None:
			self.padding_length = self.max_prompt_length

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
	skeleton_state_dict_file: Optional[str]

	device: int
	kv_dtype: str
	gpu_arch: str

	# Watchdog configuration
	watchdog_timeout: Optional[float] = 600.0  # Seconds before declaring process stuck (10 min for long inference)
	watchdog_test_stuck_time: float = 0.0  # Deliberate delay for testing
	watchdog_heartbeat_interval: Optional[float] = None  # Heartbeat interval
	decode_step_timeout: Optional[float] = None  # Max seconds per decode step

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
	pre_dequantize_weights: bool = False  # Pre-dequantize MoE routed expert MXFP4 weights to BF16
	enable_cuda_graph: bool = False  # Explicitly enable CUDA graph capture for supported models
	disable_cuda_graphs: bool = True  # Disable CUDA graph capture for decode attention (default: off due to 128K+ crash)
	cuda_graph_max_bucket_size: int = 128  # Max batch size per rank for CUDA graph capture
	cuda_graph_num_buckets: int = 16  # Number of CUDA graph bucket sizes
	detokenization_include_special_tokens: bool = False  # When True, include special tokens in detokenized output
	# Dynamic host KV reservation
	host_kv_chunk_size: int = 8192  # Initial host KV chunk size in tokens
	host_kv_eviction_watermark: int = 10  # Trigger eviction when free < this %
	enable_host_kv_eviction: bool = False  # Enable host KV eviction + recompute
	adaptive_chunk: bool = True  # EMA-based adaptive chunk sizing
	adaptive_chunk_min: int = 1024
	adaptive_chunk_max: int = 65536
	adaptive_chunk_ema_alpha: float = 0.1
	adaptive_chunk_multiplier: float = 1.5
	# --fast-init (memfd_create + THP)
	fast_init: bool = False
	kv_memfd_pid: int = -1
	kv_memfd_fd: int = -1
	kv_aux_memfd_fd: int = -1  # Separate memfd fd for auxiliary (indexer) KV cache
	weights_memfd_pid: int = -1
	weights_memfd_fd: int = -1
	distributed_weight_config: Optional[str] = None
	# Request pool: max QueryBook capacity (pre-allocated, metadata only)
	max_pool_size: int = 10240  # Default enables pool mode. 0 = legacy batch-FIFO.


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

		# Dynamic host KV reservation
		self.host_kv_chunk_size = args.host_kv_chunk_size
		self.host_kv_eviction_watermark = args.host_kv_eviction_watermark
		# Eviction is always enabled — it's a correctness requirement for chunked host KV
		self.enable_host_kv_eviction = True
		if args.adaptive_chunk:
			self.adaptive_chunk_sizer = AdaptiveChunkSizer(
				initial_chunk=args.host_kv_chunk_size,
				min_chunk=args.adaptive_chunk_min,
				max_chunk=args.adaptive_chunk_max,
				ema_alpha=args.adaptive_chunk_ema_alpha,
				multiplier=args.adaptive_chunk_multiplier,
			)
		else:
			self.adaptive_chunk_sizer = None

		if args.global_rank == 0:
			logging.info(
				f"Dynamic Host KV Config: chunk_size={args.host_kv_chunk_size}, "
				f"eviction_watermark={args.host_kv_eviction_watermark}%, "
				f"eviction_enabled={args.enable_host_kv_eviction}, "
				f"adaptive_chunk={args.adaptive_chunk}"
			)

		# Page boundary counter for periodic diagnostic logging
		self._boundary_count = 0

		# Watchdog for stuck detection (can be set via set_watchdog())
		self._watchdog = None
		# Decode watchdog: per-decode-step timeout (separate from general watchdog)
		self._decode_watchdog = None

		# Incremental writer for crash-resilient result saving
		# Config is staged by server_worker_main_loop; writer created after tokenizer init
		self._incremental_writer = None
		self._incremental_writer_config = None

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

		# CUDA graph state
		self._cuda_graph_manager = None
		# Phase C: cuda-graph adapter (cached from initializer.get_cuda_graph_adapter())
		# becomes the default and only decode-graph path. When an adapter is
		# present it owns eligibility / replay-inputs / KV staging; the legacy
		# `_glm5_*` whole-model code path is unreachable for models that ship
		# an adapter. Phase B's BATCHGEN_DECODE_GRAPH_ADAPTER_DUAL env gate is
		# retired — adapters are now the production path, not a migration toggle.
		self._cuda_graph_adapter = None
		self._cuda_graph_adapter_dual = True
		# SyncCoordinator instantiated lazily on first use so we don't
		# touch torch.distributed before it's initialized.
		self._sync_coordinator: Optional[SyncCoordinator] = None
		# Phase 5.1b of worker decouple (issue #175): dual-path gate for the
		# KVCacheManager stats tier. NATIVE=1 routes the 3 read-only stat
		# helpers through `batchgen.worker.kv_manager.KVCacheManager`.
		# COMPARE=1 runs both paths and asserts equal results.
		self._kv_stats_native = os.environ.get("BATCHGEN_WORKER_KV_STATS_NATIVE", "0") == "1"
		self._kv_stats_compare = os.environ.get("BATCHGEN_WORKER_KV_STATS_COMPARE", "0") == "1"
		self._kv_cache_manager: Optional[KVCacheManager] = None
		self._glm5_moe_cuda_graph_manager = None
		self._glm5_layer_cuda_graph_manager = None
		self._glm5_layer_graph_failed_buckets = set()
		self._glm5_layer_graph_capture_attempted_for_batch = False
		self._glm5_layer_graph_signature = None
		self._glm5_layer_graph_max_seqlen = None
		self._glm5_moe_graph_failed_buckets = set()
		self._glm5_dsa_graph_capture_attempted_for_batch = False
		self._glm5_moe_graph_capture_attempted_for_batch = False
		self._glm5_dsa_graph_page_table_change_after_capture_logged = False
		self._whole_model_graph = False
		self._glm5_whole_model_graph = False
		self._glm5_whole_model_graph_failed_buckets = set()
		self._glm5_whole_model_graph_capture_attempted_for_batch = False
		self._glm5_whole_model_graph_state_change_after_capture_logged = False
		self._glm5_whole_model_graph_signature = None
		self._glm5_whole_model_graph_unavailable_reason = None
		self._nsys_decode_profile_forward_count = 0
		self._nsys_decode_profile_started = False
		self._nsys_decode_profile_stopped = False
		self._decode_local_count_tensor = None
		self._decode_all_rank_counts = None
		self._decode_cache_seqlens_i32 = None
		self._decode_position_ids_i64 = None
		self._decode_cache_seqlens_cpu_staging = None
		self._decode_metadata_batch_key = None
		self._decode_metadata_cpu_seqlens = None
		# Kimi-K3 streamed-SP8 prefill pipeline identity. Host-RDMA preserves its
		# remote-daemon cursor after the first install; hierarchical GDR uses the
		# same fingerprint but reseeds its local queue/ring on each admission.
		self._streamed_sp8_h2d_installed = False
		self._streamed_sp8_weight_copy_fingerprint = None
		# Kimi-K3 pre-readiness startup (prepare_kimi_k3_startup). The first
		# flag makes that pass idempotent; the second hands the prepared prefill
		# phase to the first real admission so it is not torn down and rebuilt.
		self._k3_startup_completed = False
		self._k3_startup_prefill_ready = False
		self._k3_startup_prefill_mode = None

		# 2. Set Device immediately
		torch.cuda.set_device(self.local_rank)

		# 3. Path & Model Configurations
		self.model_name = args.model_name
		self.huggingface_ckpt_name = args.model_name
		self.hf_cache_dir = args.hf_cache_dir
		self.cache_dir = args.cache_dir
		self.converted_ckpt_dir = args.converted_ckpt_dir

		# Load skeleton_state_dict from temp file (avoids passing tensors through mp.spawn)
		if args.skeleton_state_dict_file:
			logging.info(f"Rank {args.global_rank}: Loading skeleton state dict from {args.skeleton_state_dict_file}")
			self.skeleton_state_dict = torch.load(args.skeleton_state_dict_file)
			logging.info(f"Rank {args.global_rank}: Loaded skeleton state dict with {len(self.skeleton_state_dict)} keys")
		else:
			self.skeleton_state_dict = None

		# 4. Initialize Shared Memory for Weights (Crucial for multiprocess)
		self.shm_name = args.shm_name
		self.tensor_meta_shm_name = args.tensor_meta_shm_name
		self.weight_byte_size = args.weight_byte_size
		self.enable_hugetlbfs = args.enable_hugetlbfs

		# Prepack and decode preemption configuration from args
		self.enable_prepack = args.enable_prepack
		self.host_kv_watermark = args.host_kv_watermark
		self.enable_decode_preemption = args.enable_decode_preemption
		self.detokenization_include_special_tokens = getattr(args, 'detokenization_include_special_tokens', False)

		# 4. Initialize Weights Storage (cudaHostRegister for weights)
		logging.info(f"Rank {self.rank}: Initializing shared memory segments (local_rank={self.local_rank}).")
		logging.info(
			f"Rank {self.rank}: shm_name: {self.shm_name}, "
			f"tensor_meta_shm_name: {self.tensor_meta_shm_name}, "
			f"weight_byte_size: {self.weight_byte_size}, "
			f"enable_hugetlbfs: {self.enable_hugetlbfs}, "
			f"fast_init: {args.fast_init}"
		)
		import time as _time
		_t0 = _time.monotonic()
		self.weights_storage = core_engine.Weights_Storage(self.local_rank)
		if args.distributed_weight_config:
			self.weights_storage.InitDistributed(
				args.distributed_weight_config
			)
			self.skeleton_state_dict = self.weights_storage.get_tensor(
				"__skeleton__"
			)
			logging.info(
				f"Rank {self.rank}: Loaded compact skeleton state dict "
				f"with {len(self.skeleton_state_dict)} keys"
			)
		else:
			self.weights_storage.Init(
				self.shm_name,
				self.weight_byte_size,
				self.tensor_meta_shm_name,
				self.enable_hugetlbfs,
				args.fast_init,
				args.weights_memfd_pid,
				args.weights_memfd_fd,
			)
		logging.info(f"Rank {self.rank}: [startup] Weights storage init: {_time.monotonic() - _t0:.2f}s")

		# 5. Initialize Host KV Cache Manager View (cudaHostRegister for Host KV)
		self.host_kv_cache_size = args.host_kv_cache_size
		self.global_host_kv_cache_size_gb = args.global_host_kv_cache_size_gb

		# DSA models: create DualHostKVCoordinator with proportional budget split.
		# Non-DSA models get a single-view worker below.
		host_budget_bytes = int(args.global_host_kv_cache_size_gb * (1024**3))
		dual_host = DualHostKVCoordinator.from_budget(
			model_name=args.model_name,
			host_kv_cache_size=host_budget_bytes,
			core_engine_module=core_engine,
			enable_memfd=args.fast_init,
			memfd_creator_pid=args.kv_memfd_pid if args.fast_init else -1,
			memfd_fd=args.kv_memfd_fd if args.fast_init else -1,
			aux_memfd_fd=args.kv_aux_memfd_fd if args.fast_init else -1,
		)
		if dual_host is not None:
			self.host_paged_kv_worker_view = dual_host
			logging.info(f"Rank {self.rank}: Initializing DualHostKVCoordinator with parallel cudaHostRegister (local_rank={self.local_rank})")
			dual_host.initialize(device_index=self.local_rank, create_region=False)
			logging.info(f"Rank {self.rank}: DualHostKVCoordinator cudaHostRegister completed (local_rank={self.local_rank})")
		else:
			worker_kv_config = build_host_kv_config(
				model_name=args.model_name,
				host_kv_cache_size=host_budget_bytes,
			)
			if args.fast_init:
				worker_kv_config.enable_memfd = True
				worker_kv_config.memfd_creator_pid = args.kv_memfd_pid
				worker_kv_config.memfd_fd = args.kv_memfd_fd

			# K3 keeps 93 logical engine-layer ids over 24 dense physical MLA
			# rows, so the worker view must honor the profile's layer map.
			self.host_paged_kv_worker_view = build_host_kv_worker_view(
				core_engine, worker_kv_config
			)

			# Initialize Host KV view (parallel cudaHostRegister for all local ranks)
			_t0 = _time.monotonic()
			logging.info(f"Rank {self.rank}: Initializing Host KV view with cudaHostRegister (local_rank={self.local_rank}, fast_init={args.fast_init})")
			self.host_paged_kv_worker_view.initialize(device_index=self.local_rank, create_region=False)
			logging.info(f"Rank {self.rank}: [startup] Host KV init (cudaHostRegister): {_time.monotonic() - _t0:.2f}s")

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
		self._stop_token_ids: set = set()
		self.max_input_length = 0
		self.max_decoding_length = 0
		self.max_context_length = None  # Set per-batch from client; None = use model max
		self.model_context_length = None  # Updated from model config during init
		self.num_global_queries = 0
		self.num_local_queries = 0
		self._ignore_eos: bool = False
		self._temperature: Optional[float] = None  # Sampling temperature (None = greedy)
		self._top_p: Optional[float] = None  # Nucleus sampling threshold (None = disabled)
		self._logged_greedy: bool = False  # Track if we've logged greedy mode this batch
		self._logged_sampling: bool = False  # Track if we've logged sampling mode this batch
		# Per-request sampling parameters (list of dicts, one per prompt in batch order)
		self._per_sequence_sampling_params: Optional[list] = None
		self._batchgen_debug: Optional[dict] = None

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

		# Request pool: admission queue and response queue for persistent loop
		self._admission_queue = None  # mp.Queue, set via set_admission_queue()
		self._response_queue = None   # mp.Queue, set via set_response_queue()
		# global_idx -> decoded text for sequences completed during PREFILL
		# (C4). Captured before _report_completion pops the local maps; the
		# legacy end-of-generate() gather merges it. Legacy mode only.
		self._prefill_completed_results: Dict[int, str] = {}
		self._shutdown_requested = False
		self._max_pool_size = args.max_pool_size  # 0 = legacy mode

		# QueryBook buffer pool. Allocated lazily by _ensure_buffer_pool() once
		# the first batch's tokenized lengths are known — its input_ids buffer
		# is ONE shared-memory segment per node, so it cannot be sized from
		# static config.
		self._buffer_pool: Optional[QueryBookBufferPool] = None
		self._buffer_pool_generation = 0
		self._shared_buffer_tag: Optional[str] = None
		# Superseded pools stay mapped for the process lifetime (see
		# _retire_buffer_pool).
		self._retired_buffer_pools: List[QueryBookBufferPool] = []

		logging.info(f"Rank {self.rank}: BatchGenWorker __init__ completed.")

	def Init(self, max_input_length, max_decoding_length, num_queries, max_context_length=None):
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
			max_context_length: Maximum total context length (prompt + decode). None = use model max.
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
		self.max_context_length = max_context_length

		# Cap adaptive chunk sizer's max_chunk by max_decoding_length
		if self.adaptive_chunk_sizer is not None and max_decoding_length > 0:
			capped_max = min(self.adaptive_chunk_sizer.max_chunk, max_decoding_length)
			capped_max = math.ceil(capped_max / SequenceEntry.PAGE_SIZE) * SequenceEntry.PAGE_SIZE
			self.adaptive_chunk_sizer.max_chunk = capped_max

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

		For DSA models, splits the memory budget between primary (MLA) and
		auxiliary (indexer) caches, wrapping both in a DualKVCacheCoordinator.
		"""
		from batchgen.kv_cache.host_kv_mananger_config import (
			build_gpu_kv_config_fixed_size,
			is_dsa_model,
			_resolve_indexer_profile,
			_torch_dtype_from_string,
		)
		from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVConfig

		# Calculate GPU KV cache size if not already done
		if self.gpu_kv_cache_size_gb is None:
			self.gpu_kv_cache_size_gb = self._calculate_gpu_kv_cache_size()

		if is_dsa_model(self.huggingface_ckpt_name):
			# Split memory budget between primary MLA cache and auxiliary indexer cache.
			# Compute the ratio of bytes-per-page for primary vs auxiliary so both
			# get the same number of pages (they share the same page table).
			from batchgen.kv_cache.host_kv_mananger_config import _resolve_profile
			primary_profile = _resolve_profile(self.huggingface_ckpt_name)
			aux_profile = _resolve_indexer_profile(self.huggingface_ckpt_name)

			primary_bytes_per_page = primary_profile.bytes_per_page() * primary_profile.num_layers
			aux_bytes_per_page = aux_profile.bytes_per_page() * aux_profile.num_layers
			combined_bytes_per_page = primary_bytes_per_page + aux_bytes_per_page

			total_bytes = int(self.gpu_kv_cache_size_gb * (1024 ** 3))
			num_pages = total_bytes // combined_bytes_per_page

			primary_config = GPUPagedKVConfig(
				num_layers=primary_profile.num_layers,
				num_pages=num_pages,
				page_size_tokens=primary_profile.page_size,
				num_k_heads=primary_profile.num_k_heads,
				k_head_dim=primary_profile.k_head_dim,
				num_v_heads=primary_profile.num_v_heads,
				v_head_dim=primary_profile.v_head_dim,
				kv_dtype=_torch_dtype_from_string(primary_profile.kv_dtype),
			)
			primary_config = self._with_cuda_graph_page_table_capacity(primary_config)
			aux_config = GPUPagedKVConfig(
				num_layers=aux_profile.num_layers,
				num_pages=num_pages,
				page_size_tokens=aux_profile.page_size,
				num_k_heads=aux_profile.num_k_heads,
				k_head_dim=aux_profile.k_head_dim,
				num_v_heads=aux_profile.num_v_heads,
				v_head_dim=aux_profile.v_head_dim,
				kv_dtype=_torch_dtype_from_string(aux_profile.kv_dtype),
			)
			aux_config = self._with_cuda_graph_page_table_capacity(aux_config)

			primary = GPUPagedKVCacheManager(config=primary_config, device=self.local_rank)
			primary.initialize()
			auxiliary = GPUPagedKVCacheManager(config=aux_config, device=self.local_rank)
			auxiliary.initialize()
			manager = DualKVCacheCoordinator(primary, auxiliary)
			self._bind_gpu_paged_kv_manager(manager)

			if self.rank == 0:
				primary_gb = (primary_bytes_per_page * num_pages) / (1024 ** 3)
				aux_gb = (aux_bytes_per_page * num_pages) / (1024 ** 3)
				logging.info(
					f"[GPU-KV] DualKVCacheCoordinator initialized: "
					f"{num_pages} pages, primary={primary_gb:.2f} GB (dim={primary_profile.k_head_dim}), "
					f"auxiliary={aux_gb:.2f} GB (dim={aux_profile.k_head_dim})"
				)
			return manager
		else:
			config = build_gpu_kv_config_fixed_size(
				model_name=self.huggingface_ckpt_name,
				gpu_kv_cache_size_gb=self.gpu_kv_cache_size_gb,
			)
			config = self._with_cuda_graph_page_table_capacity(config)

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
		Set global sampling parameters for token generation (legacy /v1/inference path).

		Args:
			temperature: Sampling temperature. None or 0 = greedy decoding (deterministic).
			            Higher values (e.g., 0.7-1.0) increase randomness.
			top_p: Nucleus sampling threshold. None or 1.0 = disabled.
			       Lower values (e.g., 0.9) restrict sampling to top tokens.
		"""
		self._temperature = temperature
		self._top_p = top_p
		self._per_sequence_sampling_params = None  # Clear per-sequence params
		# Always log on rank 0 - use WARNING to ensure visibility
		if self.rank == 0:
			if temperature is not None or top_p is not None:
				logging.warning(f"[SAMPLING] temperature={temperature}, top_p={top_p} - will use sampling")
			else:
				logging.info(f"[SAMPLING] temperature=None, top_p=None - will use greedy decoding")

	def set_per_sequence_sampling_params(self, params: list) -> None:
		"""
		Set per-request sampling parameters from batch API.

		Args:
			params: List of dicts, one per prompt. Each dict has keys:
			        temperature (float|None), top_p (float|None), top_k (int|None).
		"""
		self._per_sequence_sampling_params = params
		self._temperature = None  # Clear global params
		self._top_p = None
		if self.rank == 0:
			# Summarize the params
			n_greedy = sum(1 for p in params if p.get('temperature') is None or p.get('temperature', 1.0) <= 0)
			n_sampling = len(params) - n_greedy
			logging.warning(
				f"[SAMPLING] Per-request params for {len(params)} prompts: "
				f"{n_greedy} greedy, {n_sampling} sampling"
			)

	def set_batchgen_debug(self, debug: Optional[dict]) -> None:
		self._batchgen_debug = debug if isinstance(debug, dict) and debug else None
		if self.rank == 0 and self._batchgen_debug:
			logging.warning(f"[BATCHGEN_DEBUG] enabled flags: {sorted(self._batchgen_debug.keys())}")

	def _active_batchgen_debug_for_sequences(self, batch_sequences) -> Optional[dict]:
		if self._batchgen_debug:
			return self._batchgen_debug
		merged = {}
		for seq in batch_sequences or []:
			seq_debug = getattr(seq, "batchgen_debug", None)
			if isinstance(seq_debug, dict):
				for key, value in seq_debug.items():
					if value is not None and key not in merged:
						merged[key] = value
		return merged or None

	def _glm5_dispatch_trace_enabled(self, debug: Optional[dict]) -> bool:
		if isinstance(debug, dict) and self._debug_flag_enabled(debug.get("glm5_dispatch_trace")):
			return True
		return os.environ.get("BATCHGEN_GLM5_DISPATCH_TRACE", "0") == "1"

	def _flush_glm5_dispatch_trace_summary(self, reason: str) -> None:
		if not getattr(GLM5AttnWrapper, "glm5_dispatch_trace_enabled", False):
			return
		counts = dict(getattr(GLM5AttnWrapper, "glm5_dispatch_counts", {}) or {})
		if not counts:
			return
		context = getattr(GLM5AttnWrapper, "glm5_dispatch_trace_context", None) or {}
		counts_text = ",".join(f"{key}={counts[key]}" for key in sorted(counts))
		logging.warning(
			"[GLM5_DISPATCH_TRACE] rank=%s summary reason=%s trace=%s "
			"batch_ids=%s global_ids=%s bsz=%s debug_dsa=%s debug_moe=%s counts=%s",
			context.get("rank", self.rank),
			reason,
			getattr(GLM5AttnWrapper, "glm5_dispatch_trace_id", None) or "unknown",
			context.get("batch_ids", "-"),
			context.get("global_ids", "-"),
			context.get("bsz", "-"),
			context.get("glm5_dsa_mode", "-"),
			context.get("glm5_moe_mode", "-"),
			counts_text,
		)

	def _configure_glm5_dispatch_trace(self, batch_sequences) -> None:
		debug = getattr(AttnWrapperBase, "batchgen_debug", None) or {}
		if not isinstance(debug, dict):
			debug = {}
		enabled = self._glm5_dispatch_trace_enabled(debug)
		if not enabled:
			if getattr(GLM5AttnWrapper, "glm5_dispatch_trace_enabled", False):
				self._flush_glm5_dispatch_trace_summary("disabled")
			GLM5AttnWrapper.glm5_dispatch_trace_enabled = False
			GLM5AttnWrapper.glm5_dispatch_trace_id = None
			GLM5AttnWrapper.glm5_dispatch_trace_context = None
			GLM5AttnWrapper.glm5_dispatch_counts = {}
			GLM5AttnWrapper.glm5_dispatch_seen = set()
			return

		seqs = list(batch_sequences or [])
		batch_ids = sorted({
			str(getattr(seq, "batch_id", None) or "-") for seq in seqs
		})
		global_ids = [str(getattr(seq, "global_idx", "-")) for seq in sorted(
			seqs,
			key=lambda seq: getattr(seq, "global_idx", -1),
		)]
		context = {
			"rank": self.rank,
			"batch_ids": ",".join(batch_ids) if batch_ids else "-",
			"global_ids": ",".join(global_ids) if global_ids else "-",
			"bsz": len(seqs),
			"glm5_dsa_mode": debug.get("glm5_dsa_mode", "-"),
			"glm5_moe_mode": debug.get("glm5_moe_mode", "-"),
			"glm5_moe_router_mode": debug.get("glm5_moe_router_mode", "-"),
		}
		trace_id = (
			f"batches={context['batch_ids']}|global_ids={context['global_ids']}|"
			f"dsa={context['glm5_dsa_mode']}|moe={context['glm5_moe_mode']}|"
			f"router={context['glm5_moe_router_mode']}"
		)
		if (
			not getattr(GLM5AttnWrapper, "glm5_dispatch_trace_enabled", False)
			or getattr(GLM5AttnWrapper, "glm5_dispatch_trace_id", None) != trace_id
		):
			self._flush_glm5_dispatch_trace_summary("switch")
			GLM5AttnWrapper.glm5_dispatch_trace_enabled = True
			GLM5AttnWrapper.glm5_dispatch_trace_id = trace_id
			GLM5AttnWrapper.glm5_dispatch_trace_context = context
			GLM5AttnWrapper.glm5_dispatch_counts = {}
			GLM5AttnWrapper.glm5_dispatch_seen = set()
			logging.warning(
				"[GLM5_DISPATCH_TRACE] rank=%s begin trace=%s batch_ids=%s "
				"global_ids=%s bsz=%s debug_dsa=%s debug_moe=%s "
				"debug_moe_router=%s",
				self.rank,
				trace_id,
				context["batch_ids"],
				context["global_ids"],
				context["bsz"],
				context["glm5_dsa_mode"],
				context["glm5_moe_mode"],
				context["glm5_moe_router_mode"],
			)
		else:
			GLM5AttnWrapper.glm5_dispatch_trace_context = context

	def _debug_sequences_for_decode_uuids(self, decode_uuids) -> list:
		if self.global_batch is None:
			return []
		sequences = []
		for uuid in decode_uuids or []:
			seq = self.global_batch.get_sequence(uuid)
			if seq is not None:
				sequences.append(seq)
		return sequences

	# ============ Request Pool: Admission Queue ============

	def set_admission_queue(self, queue) -> None:
		"""Set the mp.Queue used to receive new admission messages during generate()."""
		self._admission_queue = queue

	def set_response_queue(self, queue) -> None:
		"""Set the mp.Queue used to send per-request completion results."""
		self._response_queue = queue

	def _poll_admissions(self) -> bool:
		"""Poll for new admission messages. Called at top of generate() outer loop.

		Only rank 0 polls the queue; result is broadcast to all ranks.
		New sequences are tokenized, assigned ranks, and added to global_batch as QUEUEING.

		Returns:
			True if new sequences were admitted.
		"""
		import queue as queue_mod

		has_new = False
		msg_data = None

		if self.rank == 0 and self._admission_queue is not None:
			try:
				msg = self._admission_queue.get_nowait()
				if msg is None:
					self._shutdown_requested = True
				elif isinstance(msg, dict) and msg.get("type") == "admit":
					msg_data = msg
					has_new = True
				elif isinstance(msg, dict) and "prompts" in msg:
					# A legacy /v1/inference payload (worker_manager.infer builds
					# exactly this shape). It matches no branch above, so it used
					# to be dropped right here while the caller sat on
					# response_queue.get() and took the next batch's completion.
					# The HTTP route now returns 410, so reaching this line means
					# some other producer is putting legacy payloads on the queue:
					# fail loudly rather than park. Deliberately NOT a catch-all
					# for unknown messages -- {"command": "reload"} also lands
					# here and must keep its current handling.
					raise LegacyInferenceDeprecated()
			except queue_mod.Empty:
				pass

		# Broadcast status to all ranks
		status = torch.tensor(
			[1 if has_new else 0, 1 if self._shutdown_requested else 0],
			dtype=torch.int32, device=self.torch_device,
		)
		dist.broadcast(status, src=0)
		has_new = status[0].item() == 1
		self._shutdown_requested = status[1].item() == 1

		if has_new:
			container = [msg_data]
			dist.broadcast_object_list(container, src=0)
			msg_data = container[0]
			self._admit_sequences_from_message(msg_data)

		return has_new

	def _admit_sequences_from_message(self, msg: dict) -> None:
		"""Admit new sequences from an admission message into the live global_batch.

		This is a lightweight version of process_new_batch steps 1-4, designed
		to add sequences to an already-running generate() loop without resetting state.

		Args:
			msg: Dict with keys:
				- "entries": List of dicts, each with "request_id", "text", "max_tokens",
				  "batch_id", "priority", and optionally "sampling_params"
		"""
		entries = msg.get("entries", [])
		if not entries:
			return

		# Determine starting global_idx (continue from existing batch)
		existing_max_idx = max(
			(seq.global_idx for seq in self.global_batch), default=-1
		)
		start_idx = existing_max_idx + 1

		# Step 1: Create SequenceEntry objects
		new_uuids = []
		for i, entry in enumerate(entries):
			global_idx = start_idx + i
			max_dec = entry.get("max_tokens", self.max_decoding_length)
			seq = SequenceEntry(
				uuid=entry["request_id"],
				global_idx=global_idx,
				prompt_length=0,  # Set during tokenization
				max_decode_length=max_dec,
				text=entry.get("text", ""),
			)
			seq.batch_id = entry.get("batch_id")
			seq.batchgen_debug = entry.get("batchgen_debug")
			seq.priority = entry.get("priority", 0)
			seq.sampling_params = entry.get("sampling_params")
			self.global_batch.add_sequence(seq)
			new_uuids.append(seq.uuid)

		# Step 2: Tokenize new sequences (all ranks, parallel)
		self._tokenize_admitted_sequences(new_uuids)

		# Step 2.5: Update max_input_length from admitted sequences
		# This is critical — engine config uses max_input_length for attention mask shape
		max_prompt = max(
			(self.global_batch.get_sequence(u).prompt_length
			 for u in new_uuids if self.global_batch.get_sequence(u) is not None),
			default=0,
		)
		if max_prompt > self.max_input_length:
			self.max_input_length = max_prompt
			if self.rank == 0:
				logging.info(f"[ADMIT] Updated max_input_length to {self.max_input_length}")
			self._update_config_after_tokenization()

		# Step 3: Assign ranks (round-robin, continuing from existing)
		self._assign_admitted_sequences_to_ranks(new_uuids)

		# Step 3b (Option 1, CORE): assign the serve-group at ADMISSION. Under
		# unified resident TP (G>1) a sequence binds to ALL G ranks of its
		# decode_dp_group from PREFILL onward (head-sharded KDA state + o_proj
		# all_reduce need the group replicated at prefill, not reshuffled at the
		# decode transition). No-op for G==1 (the validated pure-DP path never
		# carries a group id). _config_prefill_for_batch re-runs this idempotently
		# so evicted re-entries (whose group was cleared) re-group before binding.
		self._assign_decode_dp_groups(new_uuids)

		# Step 4: Build local query book entries for new sequences
		self._build_local_query_book_for_admitted(new_uuids)

		if self.rank == 0:
			logging.info(
				f"[ADMIT] Admitted {len(entries)} sequences "
				f"(global_idx {start_idx}-{start_idx + len(entries) - 1}), "
				f"global_batch now has {len(self.global_batch)} sequences"
			)

	def _tokenize_admitted_sequences(self, uuids: List[str]) -> None:
		"""Tokenize newly admitted sequences and assign buffer pool slots.

		Reuses the same parallel tokenization + buffer pool fill pattern as
		_tokenize_global_batch Phase 1 + Phase 3. Key differences:
		- Allocates the buffer pool on the first admission and grows it when a
		  later admission is wider (the pool cannot be pre-sized: its widths
		  come from the requests, not from static config)
		- Only processes the new sequences, not the full global_batch

		Optimization: uses padding=False to avoid creating a large padded 2D
		tensor on CPU. The tokenizer returns List[List[int]] directly, which
		is lighter than a [N, max_len] padded tensor + attention_mask.
		"""
		sequences = [self.global_batch.get_sequence(u) for u in uuids]
		all_texts = [seq.text for seq in sequences]
		num_new = len(all_texts)

		# Phase 1: Parallel tokenization across ranks (same as _tokenize_global_batch)
		my_indices = list(range(self.rank, num_new, self.world_size))
		my_texts = [all_texts[i] for i in my_indices]

		if my_texts:
			# padding=False + return_tensors=None: returns List[List[int]]
			# directly — no padded 2D tensor, no attention_mask overhead.
			# Must pass return_tensors=None explicitly because model-specific
			# tokenizers (e.g., Kimi K2.5) default to "pt" which crashes on
			# ragged lists.
			my_batch_tokenized = self.tokenizer(
				my_texts,
				return_tensors=None,
				truncation=False,
				padding=False,
				return_attention_mask=False,
			)
			my_tokenized = [
				{
					"idx": my_indices[i],
					"input_ids": my_batch_tokenized["input_ids"][i],
					"length": len(my_batch_tokenized["input_ids"][i]),
				}
				for i in range(len(my_texts))
			]
		else:
			my_tokenized = []

		# Phase 1.5: Gather across ranks
		all_tokenized_lists = [None] * self.world_size
		dist.all_gather_object(all_tokenized_lists, my_tokenized)

		tokenized_by_idx = {}
		for rank_results in all_tokenized_lists:
			if rank_results:
				for item in rank_results:
					tokenized_by_idx[item["idx"]] = item
		del all_tokenized_lists

		# Phase 2.5: Reject sequences exceeding context length
		rejected_uuids = []
		for i, seq in enumerate(sequences):
			item = tokenized_by_idx.get(i)
			if item is None:
				rejected_uuids.append(seq.uuid)
				continue
			if item["length"] >= self.model_context_length:
				rejected_uuids.append(seq.uuid)
				if self.rank == 0:
					logging.warning(
						f"[ADMIT] Rejecting {seq.uuid}: prompt length {item['length']} >= "
						f"model context {self.model_context_length}"
					)

		for uuid in rejected_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if self._response_queue is not None and self.rank == 0 and seq is not None:
				self._response_queue.put({
					"type": "completion",
					"request_id": uuid,
					"batch_id": getattr(seq, 'batch_id', None),
					"error": {
						"code": "context_length_exceeded",
						"message": (
							f"Prompt length {getattr(seq, 'prompt_length', '?')} exceeds "
							f"model context {self.model_context_length}"
						),
					},
					"text": "",
				})
			self.global_batch.remove_sequence(uuid)

		# Phase 2.75: size the pool for what this admission actually needs.
		# COLLECTIVE — every rank runs it with the same numbers: the admission
		# message was broadcast and the tokenized lengths were all-gathered above.
		required_input_width = 0
		required_decode_width = 0
		for i, seq in enumerate(sequences):
			if seq.uuid in rejected_uuids:
				continue
			prompt_len = tokenized_by_idx[i]["length"]
			required_input_width = max(
				required_input_width,
				min(prompt_len + seq.max_decode_length, self.model_context_length),
			)
			required_decode_width = max(
				required_decode_width,
				min(seq.max_decode_length, self.model_context_length),
			)
		if required_input_width > 0:
			# Rows keep their --max-pool-size meaning: the pool is NOT widened to
			# fit an over-subscribed batch, allocate_slot() still hard-fails.
			self._ensure_buffer_pool(
				required_rows=(
					self._max_pool_size if self._max_pool_size > 0 else len(sequences)
				),
				required_input_width=required_input_width,
				required_decode_width=required_decode_width,
				reason=f"admission of {len(sequences)} sequences",
			)

		# Phase 3: Assign buffer pool slots and fill token data
		# Same pattern as _tokenize_global_batch Phase 3 — allocate slot from
		# existing buffer pool, write tokens directly into the view.
		for i, seq in enumerate(sequences):
			if seq.uuid in rejected_uuids:
				continue
			item = tokenized_by_idx[i]
			input_ids_list = item["input_ids"]
			actual_prompt_len = item["length"]

			seq_extended_size = min(
				actual_prompt_len + seq.max_decode_length,
				self.model_context_length,
			)

			slot = self._buffer_pool.allocate_slot()
			try:
				input_ids_view = self._buffer_pool.get_input_ids_view(slot, seq_extended_size)
				input_ids_view[0, :actual_prompt_len] = torch.tensor(input_ids_list, dtype=torch.long)
				seq.input_ids = input_ids_view
				seq.decoded_tokens = self._buffer_pool.get_decoded_tokens_view(slot)
			except Exception:
				self._buffer_pool.free_slot(slot)
				raise
			seq._buffer_slot = slot

			seq.prompt_length = actual_prompt_len
			seq.original_prompt_length = actual_prompt_len
			seq.current_context_length = actual_prompt_len
			seq.kv_token_budget = seq_extended_size

	def _assign_admitted_sequences_to_ranks(self, uuids: List[str]) -> None:
		"""Assign newly admitted sequences to ranks.

		Default (BATCHGEN_L2_BALANCE=1, default): least-sum(L²) argmin with
		FFD ordering (longest first). Attention is O(L²) so balancing on L²
		minimizes the wall-clock spread between fastest and slowest rank
		during prefill — without this, all LongBench long-context seqs land
		on rank 14-15 under round-robin and stall the per-iteration barrier.

		Fallback (BATCHGEN_L2_BALANCE=0): least-count argmin (legacy).
		"""
		import os as _os
		use_l2 = _os.environ.get("BATCHGEN_L2_BALANCE", "1") == "1"

		if use_l2:
			# Per-rank load = sum of (prompt_length ** 2) over already-assigned seqs.
			rank_load = [0.0] * self.world_size
			for seq in self.global_batch:
				if seq.uuid in uuids or seq.assigned_rank is None:
					continue
				L = getattr(seq, "prompt_length", 0) or 0
				rank_load[seq.assigned_rank] += float(L) * float(L)

			# Resolve uuids → seqs and sort by length DESC (FFD).
			pending = []
			for uuid in uuids:
				seq = self.global_batch.get_sequence(uuid)
				if seq is None:
					continue
				L = getattr(seq, "prompt_length", 0) or 0
				pending.append((L, uuid))
			pending.sort(key=lambda t: -t[0])

			for L, uuid in pending:
				min_rank = min(range(self.world_size), key=lambda r: rank_load[r])
				self.global_batch.assign_rank(uuid, min_rank)
				rank_load[min_rank] += float(L) * float(L)

			if self.rank == 0 and rank_load:
				lo = min(rank_load); hi = max(rank_load)
				ratio = (hi / lo) if lo > 0 else float("inf")
				logging.info(
					f"[L2_BALANCE] per-rank sum(L^2): min={lo:.3e} max={hi:.3e} "
					f"ratio={ratio:.2f} ranks={[f'{x:.2e}' for x in rank_load]}"
				)
			return

		# Legacy: round-robin / least-count
		rank_counts = [0] * self.world_size
		for seq in self.global_batch:
			if seq.uuid not in uuids and seq.assigned_rank is not None:
				rank_counts[seq.assigned_rank] += 1

		for uuid in uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				continue
			min_rank = rank_counts.index(min(rank_counts))
			self.global_batch.assign_rank(uuid, min_rank)
			rank_counts[min_rank] += 1

	def _decode_attn_tp_size(self) -> int:
		"""G (attn_tp_size) for decode; 1 (pure-DP) unless the PSM head-shards."""
		pm = getattr(self, "parallel_manager", None)
		return int(getattr(pm, "attn_tp_size", 1)) if pm is not None else 1

	def _owns_local_sequence(self, seq) -> bool:
		"""Which ranks bind a sequence into the LOCAL maps (query_book +
		_uuid_to_local_map) and drive it through prefill/decode.

		Pure DP (G==1, the validated path): the single ``assigned_rank`` owns it.
		Option 1 unified resident TP (G>1): ALL G ranks of the sequence's
		``decode_dp_group`` hold it — the group runs prefill+decode in TP-G
		lockstep on identical sequences (replicated attention, head-sharded KDA
		state), so the o_proj all_reduce couples matching tokens and no
		prefill->decode reshard is ever needed. Requires ``decode_dp_group`` to
		be stamped (done at admission / prefill config before binding)."""
		G = self._decode_attn_tp_size()
		if G <= 1:
			return seq.assigned_rank == self.rank
		from batchgen.decode_dp_group import rank_in_decode_group
		return rank_in_decode_group(seq.decode_dp_group, self.rank, G)

	def _owns_host_kv(self, seq) -> bool:
		"""Which SINGLE rank drives the per-node SHARED host-KV region for a seq.

		The host paged KV cache is ONE shm region per node
		(``batchgen_host_kv_cache``) keyed by ``global_idx``, so register /
		allocate / grow / release must fire EXACTLY once per sequence. This is
		NARROWER than ``_owns_local_sequence``: G>1 replicates a seq onto all G
		ranks of its group (they each hold GPU KV + head-sharded KDA state), but
		only the group LEADER ``decode_dp_group*G`` may touch the shared host
		region — otherwise the G ranks double-register (G x host reservation) and
		double-release (2nd releaser hits ``IndexError: Sequence ID ... not found
		during release``). G==1 is the validated single-owner path: the
		``assigned_rank`` owner, identical to ``uuid in _uuid_to_local_map``."""
		G = self._decode_attn_tp_size()
		if G <= 1:
			return seq.assigned_rank == self.rank
		from batchgen.decode_dp_group import host_kv_owner_rank
		return (
			seq.decode_dp_group is not None
			and host_kv_owner_rank(seq.decode_dp_group, G) == self.rank
		)

	def _assign_decode_dp_groups(self, uuids: List[str]) -> None:
		"""Stamp ``seq.decode_dp_group`` (Option 1: at admission / prefill config,
		BEFORE the group-predicate binding — not at the decode transition).

		ADDITIVE — ``assigned_rank`` is untouched. For G==1 the group equals the
		rank and nothing keys on this field, so we skip entirely: the validated
		pure-DP path never carries a group id. For G>1 the G ranks of a group own
		the SAME sequences from prefill onward, so the assignment is over
		``num_dp = world_size // G`` groups, L^2-balanced against the sequences
		already grouped (mirrors ``_assign_admitted_sequences_to_ranks``).
		Idempotent (skips already-grouped seqs) and deterministic across ranks:
		identical inputs -> identical map.
		"""
		G = self._decode_attn_tp_size()
		if G <= 1 or not uuids:
			return
		from batchgen.decode_dp_group import (
			assign_decode_dp_groups,
			num_decode_dp_groups,
		)
		num_dp = num_decode_dp_groups(self.world_size, G)
		prior = [0.0] * num_dp
		to_assign = set(uuids)
		for seq in self.global_batch:
			if seq.uuid in to_assign or seq.decode_dp_group is None:
				continue
			L = getattr(seq, "prompt_length", 0) or 0
			prior[seq.decode_dp_group] += float(L) * float(L)
		lengths, seqs = [], []
		for uuid in uuids:
			seq = self.global_batch.get_sequence(uuid)
			# Idempotent: only FIRST entry (PREFILLED->IN_DECODE). A re-entry
			# (ON_HOLD->IN_DECODE) keeps its group — its head-sharded KDA state
			# already lives on that group's ranks, so re-grouping would strand it.
			if seq is None or seq.decode_dp_group is not None:
				continue
			lengths.append(getattr(seq, "prompt_length", 0) or 0)
			seqs.append(seq)
		if not seqs:
			return
		groups = assign_decode_dp_groups(lengths, num_dp, prior_load=prior)
		for seq, g in zip(seqs, groups):
			seq.decode_dp_group = g

	def _bind_local_sequence_to_query_book(
		self,
		uuid: str,
		local_idx: Optional[int] = None,
	) -> int:
		"""Bind a sequence UUID to a local slot and refresh its query_book entry."""
		seq = self.global_batch.get_sequence(uuid)
		if self.query_book is None:
			self.query_book = {}
		local_idx, self._next_local_idx = bind_local_sequence_to_query_book(
			uuid,
			seq,
			query_book=self.query_book,
			local_to_uuid_map=self._local_to_uuid_map,
			uuid_to_local_map=self._uuid_to_local_map,
			free_local_indices=self._free_local_indices,
			next_local_idx=self._next_local_idx,
			local_idx=local_idx,
		)
		return local_idx

	def _build_local_query_book_for_admitted(self, uuids: List[str]) -> None:
		"""Build local query book entries for newly admitted sequences on this rank.

		Option 1: under G>1 every rank of a sequence's serve-group binds it (see
		``_owns_local_sequence``), so the group holds the sequence replicated from
		admission. G==1 keeps the single-owner (assigned_rank) binding unchanged.
		"""
		for uuid in uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None or not self._owns_local_sequence(seq):
				continue
			self._bind_local_sequence_to_query_book(uuid)

	def _report_completion(self, uuid: str, gathered_text: str = None) -> None:
		"""Report a single sequence completion to the response queue.

		Also frees the QueryBook buffer slot so it can be reused by new admissions.

		Args:
			uuid: Sequence UUID.
			gathered_text: Pre-gathered decoded text from _gather_completed_tokens.
				If provided, uses this instead of reading from local decoded_tokens
				(which may be empty on rank 0 for sequences owned by other ranks).
		"""
		seq = self.global_batch.get_sequence(uuid)
		if seq is None:
			return

		# Free buffer slot (all ranks do this to keep state consistent)
		if hasattr(self, '_buffer_pool') and self._buffer_pool is not None:
			if seq._buffer_slot >= 0:
				self._buffer_pool.free_slot(seq._buffer_slot)
				# Guard against double-free / stale reuse: a re-entered report
				# for this seq must not free a slot now owned by another seq.
				seq._buffer_slot = -1

		# Free local index mapping.
		# DIAGNOSTIC: log the pop on the owning rank so we can correlate
		# stray pops with downstream "Missing UUID" errors.
		local_idx = release_local_query_slot(
			uuid,
			uuid_to_local_map=self._uuid_to_local_map,
			local_to_uuid_map=self._local_to_uuid_map,
			query_book=self.query_book,
			free_local_indices=self._free_local_indices,
		)
		if local_idx is not None:
			if seq.assigned_rank == self.rank:
				logging.debug(
					f"Rank {self.rank}: [LOCALMAP-POP] _report_completion popped "
					f"{uuid[:8]} (local_idx={local_idx}, status={seq.status.name})"
				)

		# Only rank 0 sends to response queue
		if self.rank != 0 or self._response_queue is None:
			return

		# Use gathered text if provided, otherwise read from local buffer
		text = gathered_text if gathered_text is not None else ""
		if text == "" and seq.decoded_tokens is not None and seq.decoded_length > 0:
			token_ids = seq.decoded_tokens[0, :seq.decoded_length].tolist()
			try:
				text = self.tokenizer.decode(token_ids)
			except Exception:
				text = ""
		self._response_queue.put({
			"type": "completion",
			"request_id": uuid,
			"batch_id": getattr(seq, "batch_id", None),
			"global_idx": seq.global_idx,
			"text": text,
			"prompt_length": seq.prompt_length,
			"decoded_length": seq.decoded_length,
			"finish_reason": self._get_finish_reason(seq),
		})

	def _gather_completed_tokens(self, completed_uuids: List[str]) -> dict:
		"""Gather decoded tokens from owning ranks for completed sequences.

		Each rank writes decoded tokens only for sequences it owns. This method
		uses all_gather_object to collect tokens from all ranks so rank 0 can
		report them correctly.

		Returns:
			Dict mapping uuid -> decoded text string.
		"""
		if not completed_uuids:
			return {}

		# Each rank provides tokens for its locally-owned completed sequences
		my_tokens = {}
		for uuid in completed_uuids:
			if uuid in self._uuid_to_local_map:
				local_idx = self._uuid_to_local_map[uuid]
				seq = self.global_batch.get_sequence(uuid)
				if seq is not None and local_idx in self.query_book:
					token_ids = self.query_book[local_idx].decoded_tokens[0, :seq.decoded_length].tolist()
					try:
						text = self.tokenizer.decode(token_ids)
					except Exception:
						text = ""
					my_tokens[uuid] = text

		# All ranks participate in gather
		all_tokens = [None] * self.world_size
		dist.all_gather_object(all_tokens, my_tokens)

		# Merge: each uuid is owned by exactly one rank
		merged = {}
		for rank_tokens in all_tokens:
			if rank_tokens:
				merged.update(rank_tokens)
		return merged

	# ============ End Request Pool Methods ============

	def _build_sampling_tensors(self, batch_sequences: list) -> tuple:
		"""Build [B] sampling param tensors for the active decode batch.

		Returns:
			(temps, top_ps, top_ks) tensors on the model's device, or (None, None, None)
			if using global scalar params.
		"""
		if not batch_sequences:
			return None, None, None

		has_sequence_params = any(
			getattr(seq, "sampling_params", None) is not None
			for seq in batch_sequences
		)
		if self._per_sequence_sampling_params is None and not has_sequence_params:
			return None, None, None

		device = next(self.model.parameters()).device
		params = []
		for seq in batch_sequences:
			seq_params = getattr(seq, "sampling_params", None)
			if seq_params is None and self._per_sequence_sampling_params is not None:
				global_idx = getattr(seq, "global_idx", -1)
				if 0 <= global_idx < len(self._per_sequence_sampling_params):
					seq_params = self._per_sequence_sampling_params[global_idx]
			params.append(seq_params or {})

		temps = torch.tensor(
			[p.get('temperature', 0.0) or 0.0 for p in params],
			dtype=torch.float32, device=device
		)
		top_ps = torch.tensor(
			[p.get('top_p', 1.0) or 1.0 for p in params],
			dtype=torch.float32, device=device
		)
		top_ks = torch.tensor(
			[p.get('top_k', 0) or 0 for p in params],
			dtype=torch.int64, device=device
		)
		return temps, top_ps, top_ks

	def _decode_batch_all_greedy(self, batch_sequences: list) -> bool:
		"""True iff every decode-batch sequence is greedy (effective temperature
		<= 0), resolving params exactly as _build_sampling_tensors. Pure host-side
		check (no device sync), so an all-greedy batch can skip sample_tokens and
		its per-token .any() syncs / H2D param builds.
		"""
		for seq in batch_sequences:
			seq_params = getattr(seq, "sampling_params", None)
			if seq_params is None and self._per_sequence_sampling_params is not None:
				global_idx = getattr(seq, "global_idx", -1)
				if 0 <= global_idx < len(self._per_sequence_sampling_params):
					seq_params = self._per_sequence_sampling_params[global_idx]
			if ((seq_params or {}).get('temperature', 0.0) or 0.0) > 0:
				return False
		return True

	def _select_tokens(self, logits: torch.Tensor, batch_sequences: Optional[list] = None) -> torch.Tensor:
		"""
		Select next tokens from logits using greedy or sampling strategy.
		Supports both global params and per-sequence params.

		Args:
			logits: [batch_size, vocab_size] logits from model

		Returns:
			[batch_size, 1] selected token indices
		"""
		from batchgen.sampling import sample_tokens

		# Per-sequence sampling path. In pool mode, sampling params are attached
		# to SequenceEntry objects; in legacy mode, fall back to global_idx lookup
		# in the original per-prompt list.
		if (
			self._per_sequence_sampling_params is not None
			or (
				batch_sequences is not None
				and any(getattr(seq, "sampling_params", None) is not None for seq in batch_sequences)
			)
		):
			active_sequences = batch_sequences or []
			# All-greedy fast path: if every sequence's effective temperature is
			# <= 0, argmax directly (fp32, bit-identical to sample_tokens' greedy
			# branch) instead of entering sample_tokens. This skips its two per-
			# token .any() device->host syncs (sampling.py:84,88) and the three
			# per-token H2D sampling-tensor builds — pure pipeline stalls for a
			# temp-0 batch. Host-side check (no device sync), recomputed each step
			# so no membership-invalidation hazard.
			if self._decode_batch_all_greedy(active_sequences):
				return logits.float().argmax(dim=-1, keepdim=True)
			temps, top_ps, top_ks = self._build_sampling_tensors(active_sequences)
			if not getattr(self, '_logged_sampling', False) and self.rank == 0:
				logging.info(f"Using PER-SEQUENCE sampling for {logits.shape[0]} sequences")
				self._logged_sampling = True
			if temps is not None:
				return sample_tokens(logits, temperature=temps, top_p=top_ps, top_k=top_ks)

		# Global sampling path (legacy)
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

	def set_decode_watchdog(self, watchdog) -> None:
		"""Set a per-decode-step watchdog. Starts disabled; enabled only during decode."""
		self._decode_watchdog = watchdog
		# Start disabled — only enable around actual decode iterations
		if hasattr(watchdog, '_active'):
			watchdog._active = False

	def feed_watchdog(self) -> None:
		"""Feed the watchdog to prevent timeout during long operations."""
		if self._watchdog is not None:
			self._watchdog.feed()

	def feed_decode_watchdog(self) -> None:
		"""Feed the decode watchdog at the start of each decode step."""
		if self._decode_watchdog is not None:
			self._decode_watchdog.feed()

	def enable_decode_watchdog(self) -> None:
		"""Enable decode watchdog monitoring (call before decode loop)."""
		if self._decode_watchdog is not None and hasattr(self._decode_watchdog, '_active'):
			self._decode_watchdog._active = True
			self._decode_watchdog.feed()  # Reset timer

	def disable_decode_watchdog(self) -> None:
		"""Disable decode watchdog monitoring (call after decode loop)."""
		if self._decode_watchdog is not None and hasattr(self._decode_watchdog, '_active'):
			self._decode_watchdog._active = False

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

	# Thin delegations to `batchgen.worker.completion.CompletionHandler`.
	# The worker owns the canonical config (`self._ignore_eos`,
	# `self.eos_token_ids`, `self.model_context_length`, `self.rank`);
	# `_make_completion_context` snapshots them into a frozen
	# `CompletionContext` per call.

	def _make_completion_context(self) -> CompletionContext:
		return CompletionContext(
			ignore_eos=self._ignore_eos,
			eos_token_ids=frozenset(self.eos_token_ids),
			model_context_length=self.model_context_length,
			rank=self.rank,
		)

	def _should_stop_at_eos(self, token_id: int) -> bool:
		return CompletionHandler.should_stop_at_eos(self._make_completion_context(), token_id)

	def _is_sequence_completed(self, seq) -> bool:
		return CompletionHandler.is_sequence_completed(self._make_completion_context(), seq)

	def _get_finish_reason(self, seq) -> str:
		return CompletionHandler.get_finish_reason(self._make_completion_context(), seq)

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

	def _flush_deferred_kv_to_host(self) -> None:
		"""Flush all deferred KV host offload entries accumulated during forward.

		Batch-launch D2H copies for all layers (primary MLA KV + the DSA
		auxiliary indexer KV if present). The C++ async append APIs order their
		dedicated D2H stream after the producer stream with a CUDA event, so this
		function must not synchronize the CPU with the producer stream first.
		"""
		entries = getattr(self, '_deferred_kv_entries', [])
		entries_aux = getattr(self, '_deferred_kv_entries_aux', [])
		if not entries and not entries_aux:
			return

		worker_view = getattr(self, '_deferred_kv_worker_view', None)
		batch_info = getattr(self, '_deferred_kv_batch', None)
		aux_view = getattr(self, '_deferred_kv_worker_view_aux', None)
		if entries_aux and aux_view is None:
			raise RuntimeError(
				"DSA auxiliary host KV worker view is required for deferred aux KV offload"
			)
		if (worker_view is None or batch_info is None) and not aux_view:
			self._deferred_kv_entries = []
			self._deferred_kv_entries_aux = []
			return

		sequence_ids, sequence_lengths = batch_info if batch_info is not None else (None, None)
		if sequence_ids is not None and sequence_lengths is not None:
			self._ensure_host_kv_append_capacity(sequence_ids, sequence_lengths)

		def _assert_deferred_kv_rows(cache_name: str, layer_idx: int, tensor: torch.Tensor) -> None:
			if sequence_ids is None:
				return
			if tensor.shape[0] != len(sequence_ids):
				raise RuntimeError(
					f"{cache_name} deferred KV row mismatch at layer {layer_idx}: "
					f"tensor_rows={tensor.shape[0]}, sequence_ids={len(sequence_ids)}, "
					f"gids={sequence_ids[:8] if sequence_ids is not None else []}, "
					f"write_pos={sequence_lengths[:8] if sequence_lengths is not None else []}"
				)

		# Fire all D2H copies
		if not hasattr(self, '_pending_kv_append_tensors'):
			self._pending_kv_append_tensors = []

		_use_uva_kernel = os.environ.get(
			"BATCHGEN_KV_OFFLOAD_UVA_KERNEL", "1") == "1"

		if _use_uva_kernel and hasattr(worker_view, "async_append_decode_kv_to_host_batched_kernel"):
			if entries and worker_view is not None and sequence_ids is not None:
				_prepared_entries = []
				for layer_idx, k_tensor, v_tensor in entries:
					if k_tensor.dim() == 3:
						k_tensor = k_tensor.unsqueeze(2)
					if v_tensor is not None and v_tensor.dim() == 3:
						v_tensor = v_tensor.unsqueeze(2)
					_assert_deferred_kv_rows("primary", layer_idx, k_tensor)
					if v_tensor is not None:
						_assert_deferred_kv_rows("primary_v", layer_idx, v_tensor)
					_prepared_entries.append((layer_idx, k_tensor, v_tensor))
					self._pending_kv_append_tensors.append(k_tensor)
					if v_tensor is not None:
						self._pending_kv_append_tensors.append(v_tensor)
				task = worker_view.async_append_decode_kv_to_host_batched_kernel(
					entries=_prepared_entries,
					sequence_ids=sequence_ids,
					sequence_lengths=sequence_lengths,
				)
				if task is not None:
					self._pending_kv_append_tasks.append(task)

			if entries_aux and aux_view is not None and sequence_ids is not None:
				_prepared_aux = []
				for layer_idx, k_tensor, v_tensor in entries_aux:
					if k_tensor.dim() == 3:
						k_tensor = k_tensor.unsqueeze(2)
					_assert_deferred_kv_rows("aux", layer_idx, k_tensor)
					_prepared_aux.append((layer_idx, k_tensor, None))
					self._pending_kv_append_tensors.append(k_tensor)
				task = aux_view.async_append_decode_kv_to_host_batched_kernel(
					entries=_prepared_aux,
					sequence_ids=sequence_ids,
					sequence_lengths=sequence_lengths,
				)
				if task is not None:
					self._pending_kv_append_tasks.append(task)
		else:
			if entries and worker_view is not None and sequence_ids is not None:
				for layer_idx, k_tensor, v_tensor in entries:
					if k_tensor.dim() == 3:
						k_tensor = k_tensor.unsqueeze(2)
					if v_tensor is not None and v_tensor.dim() == 3:
						v_tensor = v_tensor.unsqueeze(2)
					_assert_deferred_kv_rows("primary", layer_idx, k_tensor)
					if v_tensor is not None:
						_assert_deferred_kv_rows("primary_v", layer_idx, v_tensor)

					task = worker_view.async_append_decode_kv_to_host(
						layer_idx=layer_idx,
						sequence_ids=sequence_ids,
						k_tensor=k_tensor,
						v_tensor=v_tensor,
						sequence_lengths=sequence_lengths,
					)

					self._pending_kv_append_tensors.append(k_tensor)
					if v_tensor is not None:
						self._pending_kv_append_tensors.append(v_tensor)
					if task is not None:
						self._pending_kv_append_tasks.append(task)

			if entries_aux and aux_view is not None and sequence_ids is not None:
				for layer_idx, k_tensor, v_tensor in entries_aux:
					if k_tensor.dim() == 3:
						k_tensor = k_tensor.unsqueeze(2)
					_assert_deferred_kv_rows("aux", layer_idx, k_tensor)

					task = aux_view.async_append_decode_kv_to_host(
						layer_idx=layer_idx,
						sequence_ids=sequence_ids,
						k_tensor=k_tensor,
						v_tensor=None,
						sequence_lengths=sequence_lengths,
					)

					self._pending_kv_append_tensors.append(k_tensor)
					if task is not None:
						self._pending_kv_append_tasks.append(task)

		# Throttle: prevent thread exhaustion from std::async
		if len(self._pending_kv_append_tasks) >= 256:
			self._wait_pending_kv_append_tasks(defer_errors=True)

		self._deferred_kv_entries = []
		self._deferred_kv_entries_aux = []
		self._deferred_kv_batch = None
		self._deferred_kv_worker_view = None
		self._deferred_kv_worker_view_aux = None

	def _ensure_host_kv_append_capacity(
		self,
		sequence_ids: List[int],
		sequence_lengths: List[int],
	) -> None:
		if len(sequence_ids) != len(sequence_lengths):
			raise RuntimeError(
				f"host KV append metadata mismatch: ids={len(sequence_ids)} lengths={len(sequence_lengths)}"
			)
		if self.global_batch is None:
			return
		by_gid = {seq.global_idx: seq for seq in self.global_batch}
		grow_requests = []
		grow_metadata = []
		for global_idx, write_pos in zip(sequence_ids, sequence_lengths):
			seq = by_gid.get(int(global_idx))
			if seq is None:
				raise RuntimeError(f"host KV append for unknown global_idx={global_idx}")
			if int(write_pos) < 0:
				raise RuntimeError(
					f"host KV append negative write position for gid={global_idx}: {write_pos}"
				)
			required_tokens = int(write_pos) + 1
			if int(seq.host_token_capacity) <= 0:
				raise RuntimeError(
					f"host KV append for unallocated gid={global_idx}: "
					f"write_pos={write_pos}, host_token_capacity={seq.host_token_capacity}, "
					f"ctx={seq.current_context_length}, decoded={seq.decoded_length}, "
					f"status={seq.status.name}"
				)
			if required_tokens > int(seq.kv_token_budget):
				raise RuntimeError(
					f"host KV append would exceed token budget for gid={global_idx}: "
					f"required_tokens={required_tokens}, kv_token_budget={seq.kv_token_budget}, "
					f"ctx={seq.current_context_length}, decoded={seq.decoded_length}, "
					f"status={seq.status.name}"
				)
			if required_tokens > int(seq.host_token_capacity):
				growth_pages = math.ceil(
					(required_tokens - int(seq.host_token_capacity)) / seq.PAGE_SIZE
				)
				grow_requests.append((int(global_idx), growth_pages))
				grow_metadata.append((seq, growth_pages, int(seq.host_token_capacity), required_tokens))

		if not grow_requests:
			return

		worker_view = getattr(self, "host_paged_kv_worker_view", None)
		if worker_view is None:
			worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		if worker_view is None:
			raise RuntimeError(
				f"host KV append needs growth but no host KV worker is available: "
				f"requests={grow_requests[:8]}"
			)

		waited = self._wait_pending_kv_append_tasks(defer_errors=False)
		worker_view.grow_pages_for_sequences(grow_requests)
		for seq, growth_pages, old_capacity, required_tokens in grow_metadata:
			seq.host_token_capacity += growth_pages * seq.PAGE_SIZE
			seq.host_pages_allocated += growth_pages
			logging.warning(
				f"Rank {self.rank}: [HOST_KV_APPEND_GROW] grew gid={seq.global_idx} "
				f"old_cap={old_capacity} new_cap={seq.host_token_capacity} "
				f"required={required_tokens} pages={growth_pages} waited_tasks={waited} "
				f"ctx={seq.current_context_length} decoded={seq.decoded_length} "
				f"status={seq.status.name}"
			)

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
		
		# Launch async D2H append — no CPU-side sync needed here.
		# The C++ side runs on a background thread with its own D2H stream.
		# Tensor references are kept alive in _pending_kv_append_tensors to
		# prevent GC/memory reuse. All tasks are waited at decision boundary
		# via _wait_pending_kv_append_tasks().
		task = worker_view.async_append_decode_kv_to_host(
			layer_idx=layer_idx,
			sequence_ids=sequence_ids,
			k_tensor=k_tensor,
			v_tensor=v_tensor,  # GQA models (GPT-OSS) have separate V; MLA models pass None
			sequence_lengths=sequence_lengths,
		)

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
			self._wait_pending_kv_append_tasks(sync_distributed_errors=True)

	def _append_decode_kv_to_host_aux_async(
		self,
		layer_idx: int,
		batch: List[int],
		k_tensor: torch.Tensor,
		v_tensor: torch.Tensor = None,
	) -> None:
		"""Fire-and-forget auxiliary (indexer) KV append to host.

		Mirrors _append_decode_kv_to_host_async but uses the auxiliary host
		worker view. Shares the same pending task list for unified flushing.
		"""
		aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)
		if aux_view is None or not batch:
			return

		sequence_ids = []
		sequence_lengths = []
		for local_idx in batch:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			sequence_ids.append(seq.global_idx)
			write_pos = seq.current_context_length - 1
			sequence_lengths.append(write_pos)

		if k_tensor.dim() == 3:
			k_tensor = k_tensor.unsqueeze(2)

		# Launch async D2H — no CPU-side sync needed (same as primary path).
		task = aux_view.async_append_decode_kv_to_host(
			layer_idx=layer_idx,
			sequence_ids=sequence_ids,
			k_tensor=k_tensor,
			v_tensor=None,
			sequence_lengths=sequence_lengths,
		)

		if not hasattr(self, '_pending_kv_append_tensors'):
			self._pending_kv_append_tensors = []
		self._pending_kv_append_tensors.append(k_tensor)

		if task is not None:
			self._pending_kv_append_tasks.append(task)

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
		model_max = getattr(self.model_config, 'max_position_embeddings', 131072)
		client_max = getattr(self, 'max_context_length', None)
		# Model's native context window is the only hard cap.
		# Batch-level max_context_length should NOT override per-request max_tokens.
		# Per-request values are ground truth (docs/input-format.md).
		self.model_context_length = model_max
		if self.rank == 0:
			logging.info(
				f"Model context length set to {self.model_context_length} "
				f"(model_config={model_max}, client_max_context_length={client_max})"
			)
		
		# Load tokenizer using BatchGen's tokenizer abstraction
		# This removes the dependency on transformers.AutoTokenizer
		# Pass model identifier for pattern matching; tokenizer loads from package dir
		self.tokenizer = load_tokenizer(self.huggingface_ckpt_name)

		# Set EOS token IDs from tokenizer (support multiple stop tokens)
		self.eos_token_id = self.tokenizer.eos_token_id
		self.eos_token_ids = getattr(self.tokenizer, 'eos_token_ids', {self.eos_token_id})
		self.pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
		logging.info(f"Rank {self.rank}: EOS token IDs set to {self.eos_token_ids}, pad_token_id={self.pad_token_id}")

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
			"max_prompt_length": self.max_input_length,
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
			"pre_dequantize_weights": self.args.pre_dequantize_weights,
			"distributed_weight_config": self.args.distributed_weight_config,
		}
		logging.info(f"kv_dtype: {input_arguments['kv_dtype']}")
			
		self.input_arguments = InputArguments(**input_arguments)
		self.initializer = get_initializer(self.huggingface_ckpt_name)
		self.initializer = self.initializer(self.input_arguments)
		self.core_engine, self.engine_config, self.model_config, self.loaded_model_config = (
			self.initializer.Init(self.weights_storage)
		)
		# Phase B: pull the cuda-graph adapter from the initializer (None if not implemented).
		# Worker uses `_cuda_graph_adapter` only when `BATCHGEN_DECODE_GRAPH_ADAPTER_DUAL=1`
		# (dual-path migration gate). Legacy `_glm5_*` path is the default until Phase C.
		self._cuda_graph_adapter = getattr(
			self.initializer, "get_cuda_graph_adapter", lambda: None
		)()
		if self._cuda_graph_adapter is not None:
			logging.info(
				"Cuda-graph adapter discovered: %s (advertises %s)",
				type(self._cuda_graph_adapter).__name__,
				[m.value for m in self._cuda_graph_adapter.advertised_modes()],
			)

		if isinstance(self.host_paged_kv_worker_view, DualHostKVCoordinator):
			self.core_engine.host_paged_kv_worker_view = self.host_paged_kv_worker_view.primary
			self.host_paged_kv_worker_view_aux = self.host_paged_kv_worker_view.auxiliary
		else:
			self.core_engine.host_paged_kv_worker_view = self.host_paged_kv_worker_view
		self.engine_config.Basic_Config.num_queries = num_queries

		# Set CUDA graph config from command-line args
		if self.args.disable_cuda_graphs:
			self.engine_config.Basic_Config.enable_cuda_graphs = False
		elif (
			"glm" in (getattr(self, "model_name", "") or "").lower()
			and (
				glm5_any_cuda_graph_requested_for_model(
					getattr(self, "model_name", None),
					enable_cuda_graph=getattr(self.args, "enable_cuda_graph", False),
				)
				or os.environ.get("BATCHGEN_GLM5_MOE_GRAPH_COMPARE", "0") == "1"
			)
		):
			self.engine_config.Basic_Config.enable_cuda_graphs = True
		elif (
			is_kimi_k25_backend_model(getattr(self, "model_name", "") or "")
			and getattr(self.args, "enable_cuda_graph", False)
		):
			# K2.5 whole-model decode graph (cuda-graph adapter). The config flag
			# defaults off; --enable-cuda-graph flips it on (mirrors GLM-5's
			# force-enable) so _warmup_cuda_graphs() reaches _setup_cuda_graphs().
			self.engine_config.Basic_Config.enable_cuda_graphs = True

		# Set EP offloading config from command-line args
		self.engine_config.EP_Config.enable_offloading = self.args.enable_ep_with_offloading
		self.engine_config.EP_Config.offloading_ratio = self.args.ep_offloading_ratio

		# Set pre-dequantize flag on model config (affects MoE routed expert weights only)
		if hasattr(self.model_config, 'pre_dequantize_weights'):
			self.model_config.pre_dequantize_weights = self.args.pre_dequantize_weights
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
		self.engine_config.Basic_Config.set_max_prompt_length(self.max_input_length)
		self.engine_config.Basic_Config.num_queries = num_queries
		
		# Update input_arguments for any components that might reference them
		if hasattr(self, 'input_arguments'):
			self.input_arguments.max_prompt_length = self.max_input_length
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
			
		old_max_prompt_length = self.engine_config.Basic_Config.get_max_prompt_length()
		if old_max_prompt_length != self.max_input_length:
			logging.info(
				f"Rank {self.rank}: Updating max_prompt_length from {old_max_prompt_length} to {self.max_input_length} "
				f"(based on actual longest prompt)"
			)
			self.engine_config.Basic_Config.set_max_prompt_length(self.max_input_length)
			
			if hasattr(self, 'input_arguments') and self.input_arguments is not None:
				self.input_arguments.max_prompt_length = self.max_input_length
				self.input_arguments.padding_length = self.max_input_length

	# ============ KV Cache Helper Methods ============

	# Phase 5.3.2 (issue #175): dual-path gate for the KV token-budget cache.
	# `_make_token_budget_request` constructs the frozen snapshot the handler
	# consumes; cache state lives on `query_book[sequence_id].kv_token_budget`
	# (passed by reference through the snapshot) so the handler can memoize
	# without touching worker attributes.
	def _make_token_budget_request(self) -> TokenBudgetRequest:
		query_book = self.query_book if getattr(self, "query_book", None) is not None else {}
		return TokenBudgetRequest(
			query_book=query_book,
			local_to_uuid=self._local_to_uuid_map,
			global_batch=self.global_batch,
			max_decoding_length=self.max_decoding_length,
		)

	def _get_sequence_token_budget(self, sequence_id: int) -> int:
		return KVCacheManager.get_sequence_token_budget(
			self._make_token_budget_request(), sequence_id
		)

	def _compute_host_kv_sequence_tokens(self, sequence_ids: List[int]) -> List[int]:
		return KVCacheManager.compute_host_kv_sequence_tokens(
			self._make_token_budget_request(), sequence_ids
		)

	def _bind_gpu_paged_kv_manager(self, manager) -> None:
		"""Bind GPU KV manager to both worker and core_engine.

		If manager is a DualKVCacheCoordinator, the primary manager is bound
		to existing gpu_paged_kv_manager slots and the auxiliary (indexer) is
		bound to gpu_paged_kv_manager_aux slots.
		"""
		self.gpu_paged_kv_cache_manager = manager
		if isinstance(manager, DualKVCacheCoordinator):
			if hasattr(self.core_engine, "gpu_paged_kv_manager"):
				self.core_engine.gpu_paged_kv_manager = manager.primary
			if hasattr(self.core_engine, "gpu_paged_kv_manager_aux"):
				self.core_engine.gpu_paged_kv_manager_aux = manager.auxiliary
		else:
			if hasattr(self.core_engine, "gpu_paged_kv_manager"):
				self.core_engine.gpu_paged_kv_manager = manager

	def _get_cuda_graph_gpu_manager(self):
		"""Return the GPU KV manager object to use for CUDA graph setup."""
		manager = self.gpu_paged_kv_cache_manager
		if isinstance(manager, DualKVCacheCoordinator):
			return manager
		if manager is not None:
			return manager
		return getattr(self.core_engine, "gpu_paged_kv_manager", None)

	# Page-table capacity helpers — thin delegations to `KVCacheManager`.
	# `_make_page_table_capacity_request` normalizes the worker's optional
	# `engine_config` / `model_config` / `args` attributes into the typed
	# `PageTableCapacityRequest` frozen snapshot.

	def _make_page_table_capacity_request(
		self, sequence_tokens: Optional[Sequence[int]] = None,
	) -> PageTableCapacityRequest:
		st = tuple(int(t) for t in (sequence_tokens or ()))
		max_input_length = int(getattr(self, "max_input_length", 0) or 0)
		max_decoding_length = int(getattr(self, "max_decoding_length", 0) or 0)
		engine_config = getattr(self, "engine_config", None)
		engine_max_prompt: Optional[int] = None
		engine_max_decode: Optional[int] = None
		engine_module_global_batch_size: Optional[int] = None
		engine_module_attn_decoding_micro_batch_size: Optional[int] = None
		engine_basic_num_queries: Optional[int] = None
		if engine_config is not None:
			basic = engine_config.Basic_Config
			module_batching = engine_config.Module_Batching_Config
			engine_max_prompt = basic.get_max_prompt_length()
			engine_max_decode = getattr(basic, "max_decoding_length", None)
			engine_module_global_batch_size = module_batching.global_batch_size
			engine_module_attn_decoding_micro_batch_size = module_batching.attn_decoding_micro_batch_size
			engine_basic_num_queries = basic.num_queries
		model_config = getattr(self, "model_config", None)
		model_max_position_embeddings = getattr(model_config, "max_position_embeddings", None)
		args = getattr(self, "args", None)
		args_cuda_graph_max_bucket_size = getattr(args, "cuda_graph_max_bucket_size", None) if args is not None else None
		return PageTableCapacityRequest(
			sequence_tokens=st,
			max_input_length=max_input_length,
			max_decoding_length=max_decoding_length,
			engine_max_prompt=engine_max_prompt,
			engine_max_decode=engine_max_decode,
			engine_module_global_batch_size=engine_module_global_batch_size,
			engine_module_attn_decoding_micro_batch_size=engine_module_attn_decoding_micro_batch_size,
			engine_basic_num_queries=engine_basic_num_queries,
			model_max_position_embeddings=model_max_position_embeddings,
			args_cuda_graph_max_bucket_size=args_cuda_graph_max_bucket_size,
		)

	def _cuda_graph_page_table_token_capacity(
		self,
		sequence_tokens: Optional[Sequence[int]] = None,
	) -> int:
		return KVCacheManager.page_table_token_capacity(
			self._make_page_table_capacity_request(sequence_tokens)
		)

	def _cuda_graph_page_table_slot_capacity(self) -> int:
		return KVCacheManager.page_table_slot_capacity(
			self._make_page_table_capacity_request()
		)

	def _with_cuda_graph_page_table_capacity(
		self,
		config,
		sequence_tokens: Optional[Sequence[int]] = None,
	):
		return KVCacheManager.apply_page_table_capacity(
			self._make_page_table_capacity_request(sequence_tokens), config
		)

	def _make_gpu_kv_manager_request(
		self, sequence_tokens: Sequence[int]
	) -> GpuKvManagerRequest:
		"""Snapshot the worker state `plan_gpu_kv_manager` consumes."""
		manager = self.gpu_paged_kv_cache_manager
		current_pages = (
			getattr(getattr(manager, "config", None), "num_pages", 0)
			if manager is not None
			else 0
		)
		return GpuKvManagerRequest(
			model_name=self.huggingface_ckpt_name,
			sequence_tokens=tuple(int(t) for t in sequence_tokens),
			has_manager=manager is not None,
			current_num_pages=int(current_pages),
			capacity=self._make_page_table_capacity_request(sequence_tokens),
		)

	def _apply_gpu_kv_manager_plan(
		self, plan: GpuKvManagerPlan
	) -> GPUPagedKVCacheManager:
		"""Apply a `GpuKvManagerPlan`: the GPU side effects live here only.

		The worker is the sole mutator — create / destroy / initialize / bind.
		"""
		manager = self.gpu_paged_kv_cache_manager

		if plan.reuse:
			manager.initialize()
			self._bind_gpu_paged_kv_manager(manager)
			return manager

		current_pages = (
			getattr(getattr(manager, "config", None), "num_pages", 0)
			if manager is not None
			else 0
		)
		if plan.destroy_existing and manager is not None:
			manager.destroy()

		logging.info(
			"Rank %s creating GPUPagedKVCacheManager on %s: "
			"current pages=%d, required pages=%d",
			self.rank, self.local_rank, current_pages, plan.primary_config.num_pages
		)

		primary = GPUPagedKVCacheManager(
			config=plan.primary_config,
			device=self.local_rank,
		)

		# For DSA models, wrap primary + auxiliary (indexer) in a coordinator
		if plan.aux_config is not None:
			auxiliary = GPUPagedKVCacheManager(
				config=plan.aux_config,
				device=self.local_rank,
			)
			manager = DualKVCacheCoordinator(primary, auxiliary)
			manager.initialize()
			self._bind_gpu_paged_kv_manager(manager)

			logging.info(
				"Rank %s initialized DualKVCacheCoordinator on %s: "
				"primary=%d pages (dim=%d), auxiliary=%d pages (dim=%d)",
				self.rank, self.local_rank,
				plan.primary_config.num_pages, plan.primary_config.k_head_dim,
				plan.aux_config.num_pages, plan.aux_config.k_head_dim,
			)
		else:
			manager = primary
			manager.initialize()
			self._bind_gpu_paged_kv_manager(manager)

			logging.info(
				"Rank %s initialized GPUPagedKVCacheManager on %s with %d pages",
				self.rank, self.local_rank, plan.primary_config.num_pages,
			)
		return manager

	def _ensure_gpu_paged_kv_manager(self, sequence_tokens: Sequence[int]) -> GPUPagedKVCacheManager:
		"""Return a GPU paged KV manager with enough pages for `sequence_tokens`.

		For DSA models, returns a DualKVCacheCoordinator wrapping both primary
		(MLA) and auxiliary (indexer) managers.

		The reuse/recreate decision is delegated to
		``KVCacheManager.plan_gpu_kv_manager``; ``_apply_gpu_kv_manager_plan``
		performs the GPU side effects.
		"""
		plan = KVCacheManager.plan_gpu_kv_manager(
			self._make_gpu_kv_manager_request(sequence_tokens)
		)
		return self._apply_gpu_kv_manager_plan(plan)

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

	def _launch_aux_host_kv_load(self, sequence_tensor: torch.Tensor):
		"""Launch async aux (DSA indexer) host->GPU load. Returns task or None.

		Why: mid-decode reload paths at lines 7688, 9201, 9370 load primary KV
		only. For DSA models, aux pages are allocated (coordinator mirrors
		allocate/grow/free) but never filled on reload, so the indexer reads
		stale or zeroed K vectors and produces garbage top-K. This helper
		mirrors the primary load under the same rebuilt-page-table state.

		Safe to call when aux is not configured: returns None without side
		effects. The returned task must be .wait()'d before the first decode
		step that consumes the aux cache.
		"""
		aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)
		if aux_view is None:
			return None
		if not isinstance(self.gpu_paged_kv_cache_manager, DualKVCacheCoordinator):
			return None
		aux_mgr = self.gpu_paged_kv_cache_manager.auxiliary
		k_ptrs_aux, v_ptrs_aux = aux_mgr.get_padded_3d_page_pointers()
		page_counts_aux = aux_mgr.export_active_sequence_page_counts()
		return aux_view.async_load_layer_paged_kv_to_device(
			sequence_ids=sequence_tensor,
			active_page_counts=page_counts_aux,
			k_device_ptrs=k_ptrs_aux,
			v_device_ptrs=v_ptrs_aux,
		)

	def _prepare_dual_kv_load_pointers(
		self,
		gpu_manager: DualKVCacheCoordinator,
		new_global_ids: List[int],
		existing_global_ids: Optional[List[int]] = None,
	) -> _DualKVLoadPointers:
		if not isinstance(gpu_manager, DualKVCacheCoordinator):
			raise RuntimeError("DSA dual KV load requires DualKVCacheCoordinator")
		if not new_global_ids:
			raise ValueError("_prepare_dual_kv_load_pointers requires non-empty sequence ids")

		sequence_tensor = torch.tensor(new_global_ids, dtype=torch.int64, device="cpu")
		try:
			gpu_manager.rebuild_page_table(new_global_ids)

			primary_order = list(gpu_manager.primary._gpu_page_table_manager.slot_to_seq_id)
			aux_order = list(gpu_manager.auxiliary._gpu_page_table_manager.slot_to_seq_id)
			if primary_order != new_global_ids or aux_order != new_global_ids:
				raise RuntimeError(
					f"DSA dual load page-table order mismatch: requested={new_global_ids[:10]} "
					f"primary={primary_order[:10]} aux={aux_order[:10]}"
				)

			primary_k, primary_v = gpu_manager.primary.get_padded_3d_page_pointers()
			primary_counts = gpu_manager.primary.export_active_sequence_page_counts()
			aux_k, aux_v = gpu_manager.auxiliary.get_padded_3d_page_pointers()
			aux_counts = gpu_manager.auxiliary.export_active_sequence_page_counts()
			if primary_counts.tolist() != aux_counts.tolist():
				raise RuntimeError(
					f"DSA dual load page-count mismatch: "
					f"primary={primary_counts.tolist()} aux={aux_counts.tolist()}"
				)

			return _DualKVLoadPointers(
				sequence_tensor=sequence_tensor,
				primary_k_ptrs=primary_k,
				primary_v_ptrs=primary_v,
				primary_page_counts=primary_counts,
				aux_k_ptrs=aux_k,
				aux_v_ptrs=aux_v,
				aux_page_counts=aux_counts,
			)
		finally:
			if existing_global_ids:
				gpu_manager.rebuild_page_table(existing_global_ids)
			else:
				gpu_manager.clear_page_table()

	def _launch_dual_host_kv_load(self, pointers: _DualKVLoadPointers) -> DualAsyncKVTask:
		host_view = self.host_paged_kv_worker_view
		if not isinstance(host_view, DualHostKVCoordinator):
			raise RuntimeError("DSA dual KV load requires DualHostKVCoordinator")
		return host_view.async_load_layer_paged_kv_to_device_dual(
			sequence_ids=pointers.sequence_tensor,
			primary_active_page_counts=pointers.primary_page_counts,
			primary_k_device_ptrs=pointers.primary_k_ptrs,
			primary_v_device_ptrs=pointers.primary_v_ptrs,
			aux_active_page_counts=pointers.aux_page_counts,
			aux_k_device_ptrs=pointers.aux_k_ptrs,
			aux_v_device_ptrs=pointers.aux_v_ptrs,
			tensors=pointers,
		)

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
		
		logging.debug(
			f"Rank {self.rank}: _load_host_kv_to_gpu launching async load for "
			f"{len(global_sequence_ids)} sequences..."
		)

		if isinstance(manager, DualKVCacheCoordinator):
			pointers = self._prepare_dual_kv_load_pointers(manager, global_sequence_ids)
			load_task = self._launch_dual_host_kv_load(pointers)
		else:
			sequence_tensor = torch.tensor(global_sequence_ids, dtype=torch.int64, device="cpu")
			k_ptrs, v_ptrs = manager.get_padded_3d_page_pointers()
			active_sequence_page_counts = manager.export_active_sequence_page_counts()
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

		# Option 1 (unified resident TP): NO prefill->decode KDA reshard. Under
		# G>1 the sequence's serve-group ran prefill in TP-G lockstep, so each of
		# its G ranks already holds its own head-shard of the KDA recurrent/conv
		# state in the GPU state pool (slot keyed by global seq id, persists
		# prefill->decode). This host->GPU load handles only the MLA paged KV,
		# which is REPLICATED across the group — nothing to reshard.

	def _release_gpu_kv_pages(self, local_sequence_ids: List[int]) -> None:
		"""Return GPU KV pages associated with the provided local sequence ids."""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None or not local_sequence_ids:
			return
		
		global_sequence_ids = self._local_indices_to_global_seq_ids(local_sequence_ids)
		
		if not global_sequence_ids:
			return
		
		# All call sites now intersect `my_completed` with `_sequences_with_gpu_kv`
		# before reaching here, so a KeyError from the manager indicates a real
		# bookkeeping bug (the source-of-truth set drifted from the manager's
		# state). Surface it loudly instead of swallowing.
		manager.free_pages_for_sequences(global_sequence_ids)
		# NOTE: No sync needed - page deallocation is synchronous to the allocator
		logging.debug(
			f"Rank {self.rank} Released GPU KV pages for global_idx: {global_sequence_ids}"
		)

		self._release_kda_state_slots(global_sequence_ids)

		# FIX Bug 2: Remove from tracking set and reset gpu_pages_allocated
		for local_idx in local_sequence_ids:
			uuid = self._local_to_uuid_map.get(local_idx)
			if uuid:
				self._sequences_with_gpu_kv.discard(uuid)
				seq = self.global_batch.get_sequence(uuid)
				if seq is not None:
					seq.gpu_pages_allocated = 0

	def _release_kda_state_slots(self, global_sequence_ids: List[int]) -> None:
		"""Release Kimi-Linear KDA state independently of paged GPU KV."""
		if not global_sequence_ids:
			return
		try:
			from batchgen.models.moonshotai.kimi_linear.wrappers import KimiLinearKDAWrapper
			if KimiLinearKDAWrapper.slot_manager is not None:
				KimiLinearKDAWrapper.free_sequences(global_sequence_ids)
		except ImportError:
			pass

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

	# Phase 5.1b of worker decouple (issue #175): the 3 read-only KV stat
	# helpers below route through `KVCacheManager` when
	# BATCHGEN_WORKER_KV_STATS_NATIVE=1; COMPARE=1 runs both paths and
	# asserts equal results. Phase 5.1c deletes the legacy bodies after
	# parity validation.

	def _make_kv_cache_manager(self) -> KVCacheManager:
		"""Lazy-construct: KV managers may not be bound at __init__ time."""
		if self._kv_cache_manager is None:
			worker = self
			class _TorchKVStatsBackend:
				def get_host_stats(self) -> KVStats:
					s = worker.host_paged_kv_worker_view.get_stats()
					return KVStats(
						num_free_pages=s.num_free_pages,
						num_used_pages=s.num_used_pages,
						num_total_pages=s.num_total_pages,
					)
				def get_gpu_stats(self) -> "Optional[KVStats]":
					m = worker.gpu_paged_kv_cache_manager
					if m is None:
						return None
					s = m.get_stats()
					return KVStats(
						num_free_pages=s.num_free_pages,
						num_used_pages=s.num_used_pages,
						num_total_pages=s.num_total_pages,
					)
			self._kv_cache_manager = KVCacheManager(backend=_TorchKVStatsBackend())
		return self._kv_cache_manager

	def _make_kv_utilization_request(self) -> KVUtilizationRequest:
		return KVUtilizationRequest(
			rank=self.rank,
			world_size=self.world_size,
			local_rank=self.local_rank,
			num_gpus_per_node=NUM_GPUS_PER_NODE,
			global_batch=self.global_batch,
		)

	def _get_host_kv_free_pages(self) -> int:
		if self._kv_stats_compare:
			legacy = self._legacy_get_host_kv_free_pages()
			native = self._make_kv_cache_manager().get_host_free_pages()
			assert legacy == native, f"kv_stats compare mismatch: get_host_kv_free_pages legacy={legacy} native={native}"
			return native if self._kv_stats_native else legacy
		if self._kv_stats_native:
			return self._make_kv_cache_manager().get_host_free_pages()
		return self._legacy_get_host_kv_free_pages()

	def _legacy_get_host_kv_free_pages(self) -> int:
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

	def _get_host_kv_utilization(self) -> Dict[str, int]:
		if self._kv_stats_compare:
			legacy = self._legacy_get_host_kv_utilization()
			native_obj = self._make_kv_cache_manager().get_host_utilization(self._make_kv_utilization_request())
			native = _dataclasses.asdict(native_obj)
			assert legacy == native, f"kv_stats compare mismatch: get_host_kv_utilization legacy={legacy} native={native}"
			return native if self._kv_stats_native else legacy
		if self._kv_stats_native:
			native_obj = self._make_kv_cache_manager().get_host_utilization(self._make_kv_utilization_request())
			return _dataclasses.asdict(native_obj)
		return self._legacy_get_host_kv_utilization()

	def _legacy_get_host_kv_utilization(self) -> Dict[str, int]:
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

		# Use C++ ground truth for page counts — shared memory atomic counters
		# are accurate per-node, unlike per-sequence host_pages_allocated which
		# is stale on non-owner ranks between metadata syncs.
		used_pages = stats.num_used_pages
		free_pages = stats.num_free_pages
		free_percent = int((free_pages / stats.num_total_pages) * 100) if stats.num_total_pages > 0 else 100

		if self.local_rank == 0:
			logging.debug(
				f"[HOST_KV_UTIL] C++ stats: used={used_pages}, free={free_pages}, "
				f"total={stats.num_total_pages}, {len(valid_sequences)} valid seqs"
			)

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

	def _gather_host_kv_stats_by_node(self, worker_view: Optional[object]) -> List[Dict[str, int]]:
		"""Gather one host-KV pool stat record per node.

		Host KV is shared by ranks on the same node, not globally. Rank 0 uses
		these per-node stats to plan dynamic host growth against the same pool
		that each owner rank will later allocate from.
		"""
		gpus_per_node = NUM_GPUS_PER_NODE
		num_nodes = max(1, math.ceil(self.world_size / gpus_per_node))
		report_free = 0
		report_total = 0
		report_node = -1
		if worker_view is not None and self.local_rank == 0:
			stats = worker_view.get_stats()
			report_node = self.rank // gpus_per_node
			report_free = int(stats.num_free_pages)
			report_total = int(stats.num_total_pages)

		stats_tensor = torch.tensor(
			[report_node, report_free, report_total],
			dtype=torch.int64,
			device=self.torch_device,
		)
		gathered = [torch.zeros_like(stats_tensor) for _ in range(self.world_size)]
		dist.all_gather(gathered, stats_tensor)

		per_node_stats = []
		reports_by_node = {}
		for item in gathered:
			node_id = int(item[0].item())
			if node_id >= 0:
				reports_by_node[node_id] = {
					'node_id': node_id,
					'num_free_pages': int(item[1].item()),
					'num_total_pages': int(item[2].item()),
				}

		for node in range(num_nodes):
			per_node_stats.append(reports_by_node.get(node, {
				'node_id': node,
				'num_free_pages': 0,
				'num_total_pages': 0,
			}))

		return per_node_stats

	def _make_watermark_trigger_request(
		self, node_stats: List[dict]
	) -> WatermarkTriggerRequest:
		"""Snapshot the worker state `plan_watermark_trigger` consumes."""
		return WatermarkTriggerRequest(
			node_stats=tuple(node_stats),
			host_kv_watermark=self.host_kv_watermark,
			has_queued=self.global_batch.has_queueing(),
			has_evicted=self.enable_host_kv_eviction and self.global_batch.has_evicted(),
		)

	def _check_host_kv_watermark_trigger(self) -> bool:
		"""Check if any node exceeds host KV free page watermark.

		Watermark = 70% FREE (underutilized).
		Only checks if this rank is local_rank 0 (one check per node).

		The aggregate+threshold decision is delegated to
		``KVCacheManager.plan_watermark_trigger``; the NCCL gather and the
		apply (store stats + logs) stay here.

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

		# Gather stats from all local_rank 0 representatives (NCCL side effect)
		all_stats = [None] * self.world_size
		dist.all_gather_object(all_stats, local_stats)

		# Filter to only node representatives
		node_stats = [s for s in all_stats if s is not None]

		if not node_stats:
			return False

		# Decide: aggregate + threshold. NCCL gather + apply stay here.
		req = self._make_watermark_trigger_request(node_stats)
		plan = KVCacheManager.plan_watermark_trigger(req)

		should_trigger = plan.should_trigger
		max_free_percent = plan.max_free_percent

		# Apply the plan: store aggregated stats + log (rank 0 only).
		if self.rank == 0:
			gs = plan.global_stats
			self._host_kv_page_stats = {
				'used': gs.used,
				'total': gs.total,
				'free_percent': gs.free_percent,
				'num_nodes': gs.num_nodes,
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
					f"threshold={self.host_kv_watermark}%, has_queued={req.has_queued}, trigger={should_trigger}"
				)

		return should_trigger

	def _make_migration_plan_request(self, node_stats: dict) -> MigrationPlanRequest:
		"""Enumerate PREFILLED / ON_HOLD candidates the planner consumes.

		Mirrors the legacy candidate pool: every PREFILLED / ON_HOLD sequence
		across all ranks, snapshotted with the metadata the greedy planner
		reads. Status is stable during one planning round, so this static
		pre-gather reproduces the legacy lazy enumeration (both sort by
		``global_idx``).
		"""
		candidates = []
		for rank in range(self.world_size):
			for status in (SequenceStatus.PREFILLED, SequenceStatus.ON_HOLD):
				for uuid in self.global_batch.get_sequences_for_rank_with_status(rank, status):
					seq = self.global_batch.get_sequence(uuid)
					if seq is None:
						continue
					candidates.append(MigrationCandidate(
						uuid=uuid,
						assigned_rank=seq.assigned_rank,
						global_idx=seq.global_idx,
						kv_token_budget=seq.kv_token_budget,
						host_pages_allocated=seq.host_pages_allocated,
					))
		return MigrationPlanRequest(
			node_stats=node_stats,
			candidates=tuple(candidates),
			num_gpus_per_node=NUM_GPUS_PER_NODE,
			world_size=self.world_size,
		)

	def _plan_kv_migration(self) -> List[MigrationOp]:
		"""Plan sequence migrations to rebalance host KV across nodes.

		Gathers per-node host-KV stats (NCCL) then delegates the greedy
		rebalance decision; the NCCL migration execution is applied by the
		caller (``_rebalance_host_kv``).

		Returns:
			List of MigrationOp objects describing planned migrations.
		"""
		# Unified resident TP (G>1): a sequence is physically owned by ALL G
		# ranks of its decode_dp_group, so the legacy planner/executor -- which
		# keys from_rank/to_rank and the local-map update on the SINGLE
		# ``assigned_rank`` -- targets a rank that may hold no HostKVPageTable
		# registration at all (IndexError), and a correct move would have to
		# carry the replicated query-book / KDA / local-map state across all G
		# ranks, which this executor does not implement. Decode-group admission
		# already balances host KV across groups, so fail safe: no migrations,
		# BEFORE any Gloo group creation or NCCL execution. G==1 is unchanged.
		G = self._decode_attn_tp_size()
		if G > 1:
			if not getattr(self, '_logged_migration_skip_tp', False) and self.rank == 0:
				logging.info(
					f"MIGRATION: attn_tp_size={G} (>1): decode-group admission owns "
					f"host-KV balancing; skipping legacy single-rank migration"
				)
				self._logged_migration_skip_tp = True
			return []

		# Gather host KV stats from all local_rank 0 (NCCL side effect)
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

		migrations = KVCacheManager.plan_kv_migration(
			self._make_migration_plan_request(node_stats)
		)
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

			# Execute migration if participating, otherwise just sync tensor shape info
			my_migration = None
			for mig in round_migrations:
				if self.rank == mig.from_rank or self.rank == mig.to_rank:
					my_migration = mig
					break

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
		"""Group migrations into conflict-free parallel rounds.

		Delegates to ``HostKVRebalancer``; the NCCL execution of the rounds
		stays in ``_execute_kv_migrations_parallel``.
		"""
		return HostKVRebalancer.group_migrations_for_parallel_execution(
			migrations, NUM_GPUS_PER_NODE
		)

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

		global_idx = seq.global_idx
		pages_needed = seq.host_pages_allocated
		if pages_needed <= 0:
			logging.error(f"Rank {self.rank}: Cannot migrate {uuid[:8]}... - no host pages allocated")
			return

		# Use the unwrapped primary view for migration. Aux (DSA indexer) KV is
		# mirrored explicitly below — the coordinator does not implement
		# read/write_sequence_kv_to_cpu, so go direct on primary and aux.
		worker_view = self.core_engine.host_paged_kv_worker_view
		aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)

		if self.rank == from_rank:
			# ===== SOURCE RANK: Read host KV directly to CPU, send via Gloo =====
			# No GPU staging needed — uses C++ ReadSequenceKVToCPU (memcpy from shared memory)
			t0 = time.perf_counter()
			logging.info(
				f"[MIGRATION] Rank {self.rank}: Send {uuid[:8]}... → rank {to_rank} "
				f"({pages_needed} pages, direct host→CPU)"
			)

			k_cpu, v_cpu = worker_view.read_sequence_kv_to_cpu(global_idx)
			k_cpu_aux = aux_view.read_sequence_kv_to_cpu(global_idx)[0] if aux_view is not None else None
			t_read = time.perf_counter()
			logging.debug(
				f"MIGRATION: Rank {self.rank}: Host→CPU read: {(t_read-t0)*1000:.1f}ms, "
				f"k_shape={list(k_cpu.shape)}"
			)

			# Send via Gloo backend
			gloo_group = self._get_or_create_gloo_group()
			dist.send(tensor=k_cpu.contiguous(), dst=to_rank, group=gloo_group)
			if v_cpu.numel() > 0:
				dist.send(tensor=v_cpu.contiguous(), dst=to_rank, group=gloo_group)
			if k_cpu_aux is not None:
				dist.send(tensor=k_cpu_aux.contiguous(), dst=to_rank, group=gloo_group)
			t_send = time.perf_counter()
			if BATCHGEN_CB_DEBUG:
				logging.debug(f"MIGRATION: Rank {self.rank}: Gloo send: {(t_send-t_read)*1000:.1f}ms")
			# Free host KV pages on source (mirror aux for DSA)
			worker_view.release_sequence_pages([global_idx])
			if aux_view is not None:
				aux_view.release_sequence_pages([global_idx])
			# Also send query_book data (input_ids, decoded_tokens)
			local_idx = self._uuid_to_local_map.get(uuid)
			if local_idx is not None and local_idx in self.query_book:
				qb = self.query_book[local_idx]
				# Send tensors via Gloo — must use .clone() because buffer pool views
				# are already contiguous (.contiguous() returns same tensor, not a copy)
				dist.send(tensor=qb.encoded["input_ids"].clone(), dst=to_rank, group=gloo_group)
				dist.send(tensor=qb.decoded_tokens.clone(), dst=to_rank, group=gloo_group)
				# The buffer slot is NOT freed here. slot -> row is GLOBAL
				# state: every rank allocates the same slot for the same
				# sequence at tokenization and frees it in _report_completion,
				# and the destination below reuses this very slot index.
				# Freeing it only on the source made this rank's pool disagree
				# with every other rank's -- it could hand row S to a new
				# admission while everyone else still reads S as this sequence,
				# and it left _buffer_slot = -1, so an eviction re-entry wrote
				# into row -1 (the LAST row). Now that input_ids is one
				# node-shared segment that divergence is data corruption.
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
			# ===== DEST RANK: Receive via Gloo, write directly to host KV =====
			t0 = time.perf_counter()
			logging.info(
				f"[MIGRATION] Rank {self.rank}: Recv {uuid[:8]}... ← rank {from_rank} "
				f"({pages_needed} pages, direct CPU→host)"
			)
			gloo_group = self._get_or_create_gloo_group()

			# Allocate host KV pages for the incoming sequence (mirror aux for DSA)
			tokens_needed = pages_needed * SequenceEntry.PAGE_SIZE
			worker_view.register_sequences([global_idx])
			worker_view.allocate_pages_for_sequences([(global_idx, tokens_needed)])
			if aux_view is not None:
				aux_view.register_sequences([global_idx])
				aux_view.allocate_pages_for_sequences([(global_idx, tokens_needed)])

			# Read empty pages to get a tensor with correct shape/dtype for recv buffer.
			# Both nodes have identical host KV config, so shape matches source's output.
			k_recv, v_recv = worker_view.read_sequence_kv_to_cpu(global_idx)
			dist.recv(tensor=k_recv, src=from_rank, group=gloo_group)
			if v_recv.numel() > 0:
				dist.recv(tensor=v_recv, src=from_rank, group=gloo_group)

			# Write received data to host pages
			worker_view.write_sequence_kv_from_cpu(
				global_idx, k_recv, v_recv if v_recv.numel() > 0 else None
			)

			# Mirror aux KV: recv aux K and write into aux host pages.
			if aux_view is not None:
				k_recv_aux = aux_view.read_sequence_kv_to_cpu(global_idx)[0]
				dist.recv(tensor=k_recv_aux, src=from_rank, group=gloo_group)
				aux_view.write_sequence_kv_from_cpu(global_idx, k_recv_aux, None)
			logging.info(
				f"MIGRATION: Rank {self.rank}: Recv+write {uuid[:8]}... "
				f"in {(time.perf_counter()-t0)*1000:.1f}ms"
			)

			# Receive query_book data (input_ids, decoded_tokens)
			input_ids_shape = seq.input_ids.shape
			decoded_tokens_shape = seq.decoded_tokens.shape
			input_ids_recv = torch.empty(input_ids_shape, dtype=seq.input_ids.dtype, device="cpu")
			decoded_tokens_recv = torch.empty(decoded_tokens_shape, dtype=seq.decoded_tokens.dtype, device="cpu")
			dist.recv(tensor=input_ids_recv, src=from_rank, group=gloo_group)
			dist.recv(tensor=decoded_tokens_recv, src=from_rank, group=gloo_group)

			if not hasattr(self, '_pending_migrated_query_book'):
				self._pending_migrated_query_book = {}
			if not hasattr(self, '_migrated_sequences'):
				self._migrated_sequences = set()
			self._migrated_sequences.add(uuid)
			self._pending_migrated_query_book[uuid] = {
				'text': seq.text,
				'input_ids': input_ids_recv,
				'decoded_tokens': decoded_tokens_recv,
				'kv_token_budget': seq.kv_token_budget,
			}

			t_total = time.perf_counter()
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
				seq_for_log = self.global_batch.get_sequence(uuid)
				if seq_for_log:
					old_rank = seq_for_log.assigned_rank
					if old_rank == self.rank:
						seq_for_log.log_event(SeqEvent.MIGRATE_SEND, self.rank,
							f"to_rank={new_rank}")
					elif new_rank == self.rank:
						seq_for_log.log_event(SeqEvent.MIGRATE_RECV, self.rank,
							f"from_rank={old_rank}")
				self.global_batch.assign_rank(uuid, new_rank)
			except KeyError:
				logging.error(f"Rank {self.rank}: Cannot update ownership for {uuid[:8]}... - sequence not found")
				continue

			# IMPORTANT: Don't change sequence status - it remains PREFILLED or ON_HOLD
			# The sequence is still valid, just owned by a different rank now

			# Update host KV tracking to match actual allocation on dest.
			# All ranks execute this (migration list is deterministic), keeping fields consistent.
			seq = self.global_batch.get_sequence(uuid)
			if seq is not None:
				seq.host_pages_allocated = mig.host_pages
				seq.host_token_capacity = mig.host_pages * self.PAGE_SIZE

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
					logging.debug(
						f"Rank {self.rank}: [LOCALMAP-POP] migration popped "
						f"{uuid[:8]} (local_idx={local_idx}, from_rank={old_rank}, "
						f"to_rank={new_rank})"
					)

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

				# Create query_book entry from pending migrated data, copying into buffer pool
				if hasattr(self, '_pending_migrated_query_book') and uuid in self._pending_migrated_query_book:
					pending = self._pending_migrated_query_book.pop(uuid)
					budget = pending['kv_token_budget']
					# Reuse existing buffer slot — Phase 3 already allocated a slot for every
					# sequence in global_batch, so seq._buffer_slot is valid
					seq = self.global_batch.get_sequence(uuid)
					existing_slot = seq._buffer_slot
					logging.info(
						f"Rank {self.rank}: Migration receive {uuid[:8]}: "
						f"reusing existing_slot={existing_slot}, budget={budget}"
					)
					if existing_slot < 0:
						# Was: allocate a fresh slot here. That is a rank-LOCAL
						# allocation of a globally-agreed index, so this rank would
						# then write the sequence into a row every other rank reads
						# as somebody else's -- silent corruption of the shared
						# input_ids segment. The slot is allocated on every rank at
						# tokenization and released on every rank in
						# _report_completion, so reaching here means that invariant
						# is already broken.
						raise QueryBookPoolCapacityError(
							f"Rank {self.rank}: migration receive of {uuid[:8]} found no "
							f"buffer slot (_buffer_slot={existing_slot}); slot assignment "
							f"has diverged from the other ranks"
						)
					self._buffer_pool.input_ids_buffer[existing_slot, :budget] = pending['input_ids'][0, :budget]
					self._buffer_pool.decoded_tokens_buffer[existing_slot, :] = pending['decoded_tokens'][0, :]
					input_ids_view = self._buffer_pool.get_input_ids_view(existing_slot, budget)
					decoded_view = self._buffer_pool.get_decoded_tokens_view(existing_slot)
					seq.input_ids = input_ids_view
					seq.decoded_tokens = decoded_view
					self.query_book[new_local_idx] = query(
						text=pending['text'],
						encoded={"input_ids": input_ids_view},
						decoded_tokens=decoded_view,
						kv_token_budget=budget,
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
		if self._kv_stats_compare:
			legacy = self._legacy_get_gpu_kv_free_pages()
			native = self._make_kv_cache_manager().get_gpu_free_pages()
			assert legacy == native, f"kv_stats compare mismatch: get_gpu_kv_free_pages legacy={legacy} native={native}"
			return native if self._kv_stats_native else legacy
		if self._kv_stats_native:
			return self._make_kv_cache_manager().get_gpu_free_pages()
		return self._legacy_get_gpu_kv_free_pages()

	def _legacy_get_gpu_kv_free_pages(self) -> int:
		"""Get current free pages from GPU KV cache."""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			return 0
		return manager.get_stats().num_free_pages

	# ============ Main Entry Point ============

	# _reject_overlimit_sequences logic is now inside _tokenize_global_batch()
	# between Phase 2 (prompt length computation) and Phase 3 (buffer allocation).

	def _init_incremental_writer(self) -> None:
		"""Create IncrementalWriter from staged config (rank 0 only).

		Called after _tokenize_global_batch() so tokenizer and eos_token_ids
		are available. Config is staged by server_worker_main_loop.
		"""
		cfg = getattr(self, '_incremental_writer_config', None)
		if cfg is None or self.rank != 0:
			return
		from batchgen.server.incremental_writer import IncrementalWriter
		self._incremental_writer = IncrementalWriter(
			output_dir=cfg["output_dir"],
			batch_id=cfg["batch_id"],
			model_name=cfg["model_name"],
			custom_id_map=cfg["custom_id_map"],
			request_urls=cfg["request_urls"],
			prompt_texts=cfg["prompt_texts"],
			tokenizer=self.tokenizer,
			eos_token_ids=self.eos_token_ids,
			pad_token_id=self.pad_token_id,
			parse_thinking=cfg.get("parse_thinking", False),
			parse_tool_call=cfg.get("parse_tool_call", False),
		)

	def process_new_batch(
		self,
		global_prompts: List[str],
		per_sequence_max_tokens: Optional[List[int]] = None,
	) -> List[torch.Tensor]:
		"""
		Process a global batch of prompts.
		All ranks receive the same global_prompts and maintain consistent state.

		Args:
			global_prompts: List of prompt strings.
			per_sequence_max_tokens: Optional per-sequence max output token limits.
				Falls back to self.max_decoding_length if None or if individual entry is None.
		"""
		logging.info(
			f"Rank {self.rank}: Processing global batch of {len(global_prompts)} sequences"
		)

		# Step 1: Initialize global batch
		self.global_batch = SequenceBatch()
		self._prefill_completed_results = {}
		for idx, text in enumerate(global_prompts):
			max_dec = self.max_decoding_length
			if per_sequence_max_tokens is not None and idx < len(per_sequence_max_tokens):
				max_dec = per_sequence_max_tokens[idx] if per_sequence_max_tokens[idx] is not None else self.max_decoding_length
			seq = SequenceEntry(
				uuid=f"seq_{idx}",
				global_idx=idx,
				prompt_length=0,
				max_decode_length=max_dec,
				text=text,
			)
			seq.batchgen_debug = self._batchgen_debug
			if self._per_sequence_sampling_params is not None and idx < len(self._per_sequence_sampling_params):
				seq.sampling_params = self._per_sequence_sampling_params[idx]
			seq.log_event(SeqEvent.CREATED, self.rank, f"max_dec={max_dec}")
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
			t_step = time.perf_counter()
			self._tokenize_global_batch()
			logging.info(f"Rank {self.rank}: [INIT TIMING] Step 2 _tokenize_global_batch: {time.perf_counter()-t_step:.2f}s")

			# Rejection of over-limit sequences now happens inside _tokenize_global_batch()
			# (between Phase 2 and Phase 3). self._rejected_sequences is set there.

			# If all sequences rejected, skip inference entirely
			if len(self.global_batch) == 0:
				logging.info(f"Rank {self.rank}: All sequences rejected. Skipping inference.")
				self._init_incremental_writer()
				if self.rank == 0 and self._incremental_writer:
					for global_idx, prompt_length in self._rejected_sequences:
						self._incremental_writer.submit_error(
							global_idx, "context_length_exceeded",
							f"This model's maximum context length is {self.model_context_length} tokens. "
							f"However, your messages resulted in {prompt_length} tokens. "
							f"Please reduce the length of the messages.",
						)
				return {}

			# Step 2.1: Create incremental writer now that tokenizer/eos_token_ids are available
			t_step = time.perf_counter()
			self._init_incremental_writer()
			logging.info(f"Rank {self.rank}: [INIT TIMING] Step 2.1 _init_incremental_writer: {time.perf_counter()-t_step:.2f}s")

			# Step 2.15: Write rejection errors via incremental writer
			if self.rank == 0 and self._incremental_writer and self._rejected_sequences:
				for global_idx, prompt_length in self._rejected_sequences:
					self._incremental_writer.submit_error(
						global_idx, "context_length_exceeded",
						f"This model's maximum context length is {self.model_context_length} tokens. "
						f"However, your messages resulted in {prompt_length} tokens. "
						f"Please reduce the length of the messages.",
					)
				logging.info(f"Rank 0: Wrote {len(self._rejected_sequences)} rejection errors to incremental output")

			# Step 2.5: Update engine config with actual max_input_length after tokenization
			t_step = time.perf_counter()
			self._update_config_after_tokenization()
			logging.info(f"Rank {self.rank}: [INIT TIMING] Step 2.5 _update_config_after_tokenization: {time.perf_counter()-t_step:.2f}s")

			# Step 3: Assign sequences to ranks (round-robin)
			t_step = time.perf_counter()
			self._assign_sequences_to_ranks()
			logging.info(f"Rank {self.rank}: [INIT TIMING] Step 3 _assign_sequences_to_ranks: {time.perf_counter()-t_step:.2f}s")

			# Step 4: Build query_book for backward compatibility
			t_step = time.perf_counter()
			self._build_local_query_book()
			logging.info(f"Rank {self.rank}: [INIT TIMING] Step 4 _build_local_query_book: {time.perf_counter()-t_step:.2f}s")

			# Step 5: Set counts for compatibility
			self.num_global_queries = len(global_prompts)
			self.num_local_queries = len(self.global_batch.get_sequences_for_rank(self.rank))

		# Step 6: Run generation with KV-driven scheduling
		# Watchdog is now active - monitors prefill and decode phases
		return self.generate()

	# ============ UUID/Index Conversion Helpers ============
	#
	# Thin delegations to `batchgen.worker.indexing.IndexManager`. The worker
	# owns the canonical maps (`self._local_to_uuid_map`, `self._uuid_to_local_map`,
	# `self.global_batch`); `_make_index_lookup_req` snapshots them into a
	# frozen `IndexLookupRequest` per call.

	def _make_index_lookup_req(self) -> IndexLookupRequest:
		return IndexLookupRequest(
			rank=self.rank,
			local_to_uuid=self._local_to_uuid_map,
			uuid_to_local=self._uuid_to_local_map,
			global_batch=self.global_batch,
		)

	def _local_to_uuid(self, local_idx: int) -> str:
		return IndexManager.local_to_uuid(self._make_index_lookup_req(), local_idx)

	def _uuid_to_local(self, uuid: str) -> int:
		return IndexManager.uuid_to_local(self._make_index_lookup_req(), uuid)

	def _local_indices_to_global_seq_ids(self, local_indices: List[int]) -> List[int]:
		return IndexManager.local_indices_to_global_seq_ids(self._make_index_lookup_req(), local_indices)

	def _get_my_sequences_by_status(self, status: SequenceStatus) -> List[str]:
		return IndexManager.get_my_sequences_by_status(self._make_index_lookup_req(), status)

	def _get_local_indices_for_uuids(self, uuids: List[str]) -> List[int]:
		return IndexManager.get_local_indices_for_uuids(self._make_index_lookup_req(), uuids)

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
				expected_ctx = seq.original_prompt_length + seq.decoded_length
				if seq.current_context_length != expected_ctx:
					logging.warning(
						f"Rank {self.rank}: Correcting ctx_len for {uuid[:8]} before sync: "
						f"{seq.current_context_length} → {expected_ctx}"
					)
					seq.log_event(SeqEvent.CTX_REPAIR, self.rank,
						f"old={seq.current_context_length}, new={expected_ctx}")
					seq.current_context_length = expected_ctx
				seq.validate_metadata(f"rank {self.rank} _sync_sequence_metadata/send")

				local_state[uuid] = {
					'decoded_length': seq.decoded_length,
					'current_context_length': seq.current_context_length,
					'gpu_pages_allocated': seq.gpu_pages_allocated,
					'eos_reached': seq.eos_reached,
					'rep_detected': getattr(seq, '_rep_detected', False),
					'prompt_length': seq.prompt_length,  # Include for validation
					'reentry_decoded_baseline': seq.reentry_decoded_baseline,
					'max_decode_length': seq.max_decode_length,
					'original_max_decode_length': seq.original_max_decode_length,
					'host_pages_allocated': seq.host_pages_allocated,
					'host_token_capacity': seq.host_token_capacity,
					# total_decoded_before_eviction: needed so non-owning ranks
					# sort eviction candidates consistently in _prepare_prefill_batch.
					'total_decoded_before_eviction': seq.total_decoded_before_eviction,
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
							if state.get('rep_detected', False):
								seq._rep_detected = True
							# Sync prompt_length too. For EVICTED sequences the
							# owner rewrites prompt_length at eviction time to
							# the reconstructed re-entry length; non-owners must
							# pick that up or prefill selection under-counts.
							if 'prompt_length' in state:
								seq.prompt_length = state['prompt_length']
							if 'reentry_decoded_baseline' in state:
								seq.reentry_decoded_baseline = state['reentry_decoded_baseline']
							if 'max_decode_length' in state:
								seq.max_decode_length = state['max_decode_length']
							if 'original_max_decode_length' in state:
								seq.original_max_decode_length = state['original_max_decode_length']
							# Sync host KV fields for consistent migration planning
							if 'host_pages_allocated' in state:
								seq.host_pages_allocated = state['host_pages_allocated']
							if 'host_token_capacity' in state:
								seq.host_token_capacity = state['host_token_capacity']
							# Eviction-related fields
							if 'total_decoded_before_eviction' in state:
								seq.total_decoded_before_eviction = state['total_decoded_before_eviction']
							
							# VALIDATION: Ensure received ctx_len is consistent
							expected_ctx = seq.original_prompt_length + seq.decoded_length
							if seq.current_context_length != expected_ctx:
								logging.error(
									f"Rank {self.rank}: [SYNC-VALIDATE] Received inconsistent ctx_len for {uuid[:8]}: "
									f"received={seq.current_context_length}, expected={expected_ctx} "
									f"(prompt={seq.prompt_length}, decoded={seq.decoded_length})"
								)
								seq.log_event(SeqEvent.CTX_REPAIR, self.rank,
									f"sync_recv old={seq.current_context_length}, new={expected_ctx}")
								seq.current_context_length = expected_ctx
							seq.validate_metadata(
								f"rank {self.rank} _sync_sequence_metadata/recv",
								require_owner_tensors=False,
							)

	# Thin delegations to `batchgen.worker.sync.SyncCoordinator`. The worker
	# owns the canonical state; `_make_sync_context` snapshots it into a
	# frozen `SyncContext` per call. `SyncCoordinator` itself is constructed
	# lazily on first use so we don't touch `torch.distributed` before the
	# process group is initialized.

	def _make_sync_coordinator(self) -> SyncCoordinator:
		if self._sync_coordinator is None:
			self._sync_coordinator = SyncCoordinator(backend=TorchDistCollectiveBackend())
		return self._sync_coordinator

	def _make_sync_context(self) -> SyncContext:
		return SyncContext(
			rank=self.rank,
			uuid_to_local=self._uuid_to_local_map,
			global_batch=self.global_batch,
			torch_device=self.torch_device,
		)

	def _sync_completion_status_tensor(
		self,
		decode_uuids: List[str],
	) -> Tuple[Set[str], List[str]]:
		return self._make_sync_coordinator().sync_completion_status_tensor(
			self._make_sync_context(), decode_uuids
		)

	def _sync_decode_uuids_tensor(
		self,
		decode_uuids: List[str],
	) -> List[str]:
		return self._make_sync_coordinator().sync_decode_uuids_tensor(
			self._make_sync_context(), decode_uuids
		)

	# ============ QueryBook Buffer Pool ============

	def _node_shared_tag(self) -> str:
		"""Run-unique tag shared by every rank, for shared-memory segment names."""
		if self._shared_buffer_tag is None:
			tag = [os.urandom(6).hex() if self.rank == 0 else None]
			dist.broadcast_object_list(tag, src=0)
			self._shared_buffer_tag = tag[0]
		return self._shared_buffer_tag

	def _ensure_buffer_pool(
		self,
		required_rows: int,
		required_input_width: int,
		required_decode_width: int,
		reason: str,
	) -> None:
		"""Allocate — or grow — the QueryBook buffer pool.

		COLLECTIVE: every rank must call this with identical arguments. They do,
		because both call sites derive the requirement from the tokenized batch,
		which is all-gathered to every rank before this runs.

		``input_ids_buffer`` is ONE shared-memory segment per node. Sizing is by
		actual need: ``required_input_width`` is the widest ``seq_extended_size``
		(prompt + that request's decode budget) the batch will ask for, capped at
		the model context length — never the context length itself, and never the
		``--max-pool-size`` flag, which keeps its row-count meaning only.

		A later admission that needs more never silently truncates: it grows the
		pool with a WARNING naming both sizes, copies the live rows over and
		rebinds every view. ``get_input_ids_view`` hard-fails
		(``QueryBookPoolCapacityError``) if a request ever slips past this.
		"""
		old = self._buffer_pool
		if old is not None and (
			required_rows <= old.num_sequences
			and required_input_width <= old.input_ids_width
			and required_decode_width <= old.max_decoding_length
		):
			return

		rows = max(required_rows, old.num_sequences if old is not None else 0)
		in_w = max(required_input_width, old.input_ids_width if old is not None else 0)
		dec_w = max(required_decode_width, old.max_decoding_length if old is not None else 0)

		self._buffer_pool_generation += 1
		node_id = self.rank // NUM_GPUS_PER_NODE
		name = (
			f"batchgen_input_ids_{self._node_shared_tag()}"
			f"_n{node_id}_g{self._buffer_pool_generation}"
		)
		is_creator = (self.rank % NUM_GPUS_PER_NODE) == 0
		shared_input_ids, shm = allocate_node_shared_int64(
			name, rows, in_w, is_creator, dist.barrier
		)
		new_pool = QueryBookBufferPool(
			num_sequences=rows,
			input_ids_width=in_w,
			max_decoding_length=dec_w,
			pad_token_id=self.pad_token_id,
			input_ids_buffer=shared_input_ids,
			input_ids_shm=shm,
		)
		shared_gib = rows * in_w * 8 / 2**30
		private_gib = rows * dec_w * 8 / 2**30
		if old is None:
			logging.info(
				f"Rank {self.rank}: QueryBook pool allocated ({reason}): rows={rows}, "
				f"input_ids_width={in_w}, decoded_width={dec_w} -> input_ids "
				f"{shared_gib:.3f} GiB SHARED per node ('{name}'), decoded_tokens "
				f"{private_gib:.3f} GiB per rank"
			)
		else:
			logging.warning(
				f"Rank {self.rank}: QueryBook pool GROWN ({reason}): rows "
				f"{old.num_sequences}->{rows}, input_ids_width "
				f"{old.input_ids_width}->{in_w}, decoded_width "
				f"{old.max_decoding_length}->{dec_w}; new input_ids segment "
				f"{shared_gib:.3f} GiB SHARED per node ('{name}')"
			)
			new_pool.adopt(old)
		self._buffer_pool = new_pool
		if old is not None:
			self._rebind_buffer_pool_views()
			self._retire_buffer_pool(old, is_creator)

	def _retire_buffer_pool(self, old: QueryBookBufferPool, is_creator: bool) -> None:
		"""Drop a superseded pool's NAME but keep its mapping alive.

		Views handed out before the grow may still be referenced somewhere this
		rebind does not reach; unmapping under them would segfault. Unlinking on
		the node's creator keeps /dev/shm from accumulating one entry per grow —
		POSIX frees the pages once the last mapping goes, i.e. at process exit.
		"""
		self._retired_buffer_pools.append(old)
		if is_creator and old.input_ids_shm is not None:
			try:
				old.input_ids_shm.unlink()
			except FileNotFoundError:
				pass

	def _rebind_buffer_pool_views(self) -> None:
		"""Repoint every live sequence and query-book entry at the current pool."""
		pool = self._buffer_pool
		rebound = 0
		for seq in self.global_batch:
			slot = getattr(seq, '_buffer_slot', -1)
			if slot < 0:
				continue
			input_ids_view = pool.get_input_ids_view(slot, seq.kv_token_budget)
			decoded_view = pool.get_decoded_tokens_view(slot)
			seq.input_ids = input_ids_view
			seq.decoded_tokens = decoded_view
			local_idx = self._uuid_to_local_map.get(seq.uuid)
			if local_idx is not None and self.query_book and local_idx in self.query_book:
				entry = self.query_book[local_idx]
				entry.encoded["input_ids"] = input_ids_view
				entry.decoded_tokens = decoded_view
			rebound += 1
		logging.warning(f"Rank {self.rank}: rebound {rebound} sequences onto the grown QueryBook pool")

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

		# Each rank tokenizes its subset.
		# padding=False + return_tensors=None avoids the padded 2D tensor.
		if my_texts:
			my_batch_tokenized = self.tokenizer(
				my_texts,
				return_tensors=None,
				truncation=False,
				padding=False,
				return_attention_mask=False,
			)
			my_tokenized = [
				{
					"global_idx": my_indices[i],
					"input_ids": my_batch_tokenized["input_ids"][i],
					"length": len(my_batch_tokenized["input_ids"][i]),
				}
				for i in range(len(my_texts))
			]
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

		# Phase 2.5: Reject sequences exceeding context length BEFORE buffer allocation.
		# Must happen here because Phase 3 would crash trying to copy oversized tokens
		# into model_context_length-sized buffers.
		self._rejected_sequences = []
		uuids_to_remove = []
		for seq in self.global_batch:
			pl = tokenized_by_idx[seq.global_idx]["length"]
			if pl >= self.model_context_length:
				self._rejected_sequences.append((seq.global_idx, pl))
				uuids_to_remove.append(seq.uuid)
				# Free tokenized data for rejected sequence
				del tokenized_by_idx[seq.global_idx]

		for uuid in uuids_to_remove:
			self.global_batch.remove_sequence(uuid)

		if self._rejected_sequences:
			logging.info(
				f"Rank {self.rank}: Rejected {len(self._rejected_sequences)}/"
				f"{len(self._rejected_sequences) + len(self.global_batch)} "
				f"sequences exceeding context length {self.model_context_length}"
			)

		# Recalculate max_prompt_length after rejection (remaining sequences only)
		num_sequences = len(self.global_batch)
		if num_sequences > 0:
			remaining_lengths = [tokenized_by_idx[seq.global_idx]["length"] for seq in self.global_batch]
			max_prompt_length = max(remaining_lengths)
		else:
			max_prompt_length = 0

		# Update self.max_input_length to the actual longest prompt
		# This is used for attention mask shape: [bsz, max_prompt_length + max_decoding_length]
		self.max_input_length = max_prompt_length
		if num_sequences > 0:
			logging.info(
				f"Rank {self.rank}: Dynamic max_prompt_length set to {max_prompt_length} "
				f"(prompt lengths: min={min(remaining_lengths)}, max={max(remaining_lengths)}, "
				f"count={num_sequences})"
			)

		# Phase 3: Create per-sequence tensor views from pre-allocated buffer pool.
		# Pre-allocating 2 large contiguous buffers eliminates allocator contention
		# when 16 ranks run Phase 3 simultaneously (was 192K allocations → now 32).
		# Skip if all sequences were rejected in Phase 2.5.
		if num_sequences == 0:
			logging.info(f"Rank {self.rank}: All sequences rejected, skipping Phase 3 buffer allocation")
			return

		phase3_start = time.perf_counter()
		num_seqs = len(self.global_batch)

		# Use max_pool_size for pre-allocation if in pool mode (allows future admissions)
		pool_capacity = max(num_seqs, self._max_pool_size) if self._max_pool_size > 0 else num_seqs
		# Width by actual need: the widest seq_extended_size the loop below will
		# ask get_input_ids_view() for. Sizing it at model_context_length instead
		# costs 8 bytes x pool_capacity x context — 80 GiB per worker at K3's 1M
		# context — for a buffer whose rows are only ever read up to their own
		# prompt length.
		required_width = min(
			max_prompt_length + self.max_decoding_length,
			self.model_context_length,
		)
		self._ensure_buffer_pool(
			required_rows=pool_capacity,
			required_input_width=required_width,
			required_decode_width=self.max_decoding_length,
			reason="legacy batch tokenization",
		)
		# Legacy mode tokenizes a whole new global batch per call, so the slot
		# bookkeeping (and buffer contents) must start clean even when the
		# existing allocation is reused.
		self._buffer_pool.reset()
		t_alloc = time.perf_counter() - phase3_start
		logging.info(
			f"Rank {self.rank}: Phase 3 buffer pool ready in {t_alloc:.2f}s "
			f"(input_ids: [{pool_capacity}, {self._buffer_pool.input_ids_width}] shared per node, "
			f"decoded_tokens: [{pool_capacity}, {self._buffer_pool.max_decoding_length}] per rank)"
		)

		for seq_i, seq in enumerate(self.global_batch):
			item = tokenized_by_idx[seq.global_idx]
			input_ids_list = item["input_ids"]
			actual_prompt_len = item["length"]

			if len(input_ids_list) != actual_prompt_len:
				logging.error(
					f"Rank {self.rank}: Token length mismatch for seq {seq.global_idx}: "
					f"list_len={len(input_ids_list)}, stored_len={actual_prompt_len}"
				)
				actual_prompt_len = len(input_ids_list)

			seq_extended_size = min(
				actual_prompt_len + self.max_decoding_length,
				self.model_context_length
			)

			slot = self._buffer_pool.allocate_slot()
			seq._buffer_slot = slot

			input_ids_view = self._buffer_pool.get_input_ids_view(slot, seq_extended_size)
			input_ids_view[0, :actual_prompt_len] = torch.tensor(input_ids_list, dtype=torch.long)
			seq.input_ids = input_ids_view
			seq.decoded_tokens = self._buffer_pool.get_decoded_tokens_view(slot)

			# Free the tokenized data for this sequence immediately
			del tokenized_by_idx[seq.global_idx]

			seq.prompt_length = actual_prompt_len
			seq.original_prompt_length = actual_prompt_len  # Must match prompt_length at tokenization time
			seq.current_context_length = actual_prompt_len
			seq.kv_token_budget = seq_extended_size

			if (seq_i + 1) % 3000 == 0:
				elapsed = time.perf_counter() - phase3_start
				logging.info(
					f"Rank {self.rank}: Phase 3 progress: {seq_i+1}/{num_seqs} sequences "
					f"({elapsed:.1f}s elapsed)"
				)

		phase3_total = time.perf_counter() - phase3_start
		logging.info(
			f"Rank {self.rank}: Phase 3 complete: {num_seqs} sequences in {phase3_total:.2f}s "
			f"(buffer alloc: {t_alloc:.2f}s, fill: {phase3_total-t_alloc:.2f}s)"
		)

		logging.info(f"Rank {self.rank}: Tokenized {len(self.global_batch)} sequences")

	def _assign_sequences_to_ranks(self) -> None:
		"""Assign sequences to ranks via `BatchFormation.plan_rank_assignment`.

		Greedy bin-packing planner returns the assignment; worker is the sole
		mutator of `global_batch.assign_rank`. All ranks running the same
		algorithm on the same `global_batch` produce identical plans without
		explicit cross-rank sync.
		"""
		if self.global_batch is None:
			raise RuntimeError("Global batch not initialized")
		ctx = BatchFormationContext(
			world_size=self.world_size, rank=self.rank, global_batch=self.global_batch,
		)
		plan = BatchFormation.plan_rank_assignment(ctx)
		for uuid, target in plan.assignments.items():
			self.global_batch.assign_rank(uuid, target)

		rank_tiles = plan.tiles_per_rank
		my_seqs = self.global_batch.get_sequences_for_rank(self.rank)
		if self.rank == 0:
			imbalance = (max(rank_tiles) - min(rank_tiles)) / max(rank_tiles) * 100 if max(rank_tiles) > 0 else 0
			logging.info(
				f"Workload distribution (tiles per rank): {list(rank_tiles)}, "
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

			self.query_book[local_idx] = make_query_book_entry(seq)

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
		"""Get physical host-KV node ID for a rank."""
		if self.world_size <= NUM_GPUS_PER_NODE:
			return 0
		return rank // NUM_GPUS_PER_NODE

	def _get_num_nodes(self) -> int:
		"""Get total number of physical host-KV nodes."""
		return max(1, math.ceil(self.world_size / NUM_GPUS_PER_NODE))

	def _get_effective_chunk_size(self) -> int:
		"""Return the current host KV chunk size, considering adaptive sizing.

		The chunk size is capped by max_decoding_length since allocating more
		than the maximum possible decode tokens is wasteful. The result is
		always rounded up to a page boundary (multiple of PAGE_SIZE=64).
		"""
		if self.adaptive_chunk_sizer is not None:
			chunk = self.adaptive_chunk_sizer.get_chunk_size()
		else:
			chunk = self.host_kv_chunk_size
		# Cap by max_decoding_length — no point reserving more than max decode
		if self.max_decoding_length > 0:
			chunk = min(chunk, self.max_decoding_length)
		# Round up to page boundary
		chunk = math.ceil(chunk / SequenceEntry.PAGE_SIZE) * SequenceEntry.PAGE_SIZE
		return chunk

	def _prepare_prefill_batch(self) -> List[str]:
		"""
		Select sequences for prefill based on HOST KV cache capacity.

		Key constraint: Host KV cache is PER NODE.
		- Each node has its own host KV capacity
		- Sequences assigned to ranks on node N use node N's host KV
		- Must check per-node capacity, not global

		With dynamic host KV reservation, sequences only need prompt + chunk_size
		pages initially (not the full kv_token_budget). This allows more sequences
		to be prefilled concurrently.

		EVICTED sequences get weighted priority (more decoded = higher priority)
		and re-enter through the prefill path.
		"""
		# Collect candidates: evicted sequences first (weighted priority), then new
		evicted_uuids = []
		if self.enable_host_kv_eviction:
			evicted_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.EVICTED)
			# Weighted priority: more decoded tokens = higher priority (less wasted work)
			evicted_uuids.sort(key=lambda u: (
				-self.global_batch.get_sequence(u).total_decoded_before_eviction,
				self.global_batch.get_sequence(u).global_idx
			))

		queueing_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.QUEUEING)
		queueing_uuids.sort(key=lambda uuid: self.global_batch.get_sequence(uuid).global_idx)

		all_candidates = evicted_uuids + queueing_uuids
		if not all_candidates:
			return []

		gpus_per_node = NUM_GPUS_PER_NODE
		num_nodes = self._get_num_nodes()
		chunk_size = self._get_effective_chunk_size()
		sequence_limits = self._prefill_sequence_limits()
		node_sequence_free = sequence_limits.get("max_sequences_per_node")
		rank_sequence_free = sequence_limits.get("max_sequences_per_rank")
		if node_sequence_free is not None and rank_sequence_free is not None:
			raise ValueError(
				"prefill sequence capacity must be scoped to rank or node, not both"
			)
		if node_sequence_free is not None:
			limit_scope = 2
			local_sequence_free = int(node_sequence_free)
		elif rank_sequence_free is not None:
			limit_scope = 1
			local_sequence_free = int(rank_sequence_free)
		else:
			limit_scope = 0
			local_sequence_free = -1

		# Step 1: Get this node's host KV free pages
		local_host_free = self._get_host_kv_free_pages()

		# Step 2: Gather host KV plus persistent sequence capacity. Host KV is
		# one shared region per node, so only the node leader reports it. KDA
		# state is per GPU; every rank reports its free count and the scheduler
		# uses the node minimum. This keeps every rank's selection identical
		# even if a prior lifecycle bug left one TP group asymmetric.
		report_node = self.rank // gpus_per_node
		report_host_free = local_host_free if self.local_rank == 0 else -1
		free_tensor = torch.tensor(
			[report_node, report_host_free, limit_scope, local_sequence_free],
			dtype=torch.int64,
			device=self.torch_device,
		)
		gathered = [torch.zeros_like(free_tensor) for _ in range(self.world_size)]
		dist.all_gather(gathered, free_tensor)

		# Extract per-node host KV free pages
		reports_by_node = {}
		for item in gathered:
			node_id = int(item[0].item())
			host_free = int(item[1].item())
			if node_id >= 0 and host_free >= 0:
				reports_by_node[node_id] = host_free
		per_node_host_free = []
		for node in range(num_nodes):
			per_node_host_free.append(reports_by_node.get(node, 0))

		gathered_scopes = {int(item[2].item()) for item in gathered}
		if len(gathered_scopes) != 1:
			raise RuntimeError(
				f"prefill sequence-capacity scope diverged across ranks: "
				f"{sorted(gathered_scopes)}"
			)
		per_rank_sequence_free = None
		per_node_sequence_free = None
		gathered_scope = next(iter(gathered_scopes))
		if gathered_scope == 1:
			per_rank_sequence_free = [int(item[3].item()) for item in gathered]
		elif gathered_scope == 2:
			per_node_sequence_free = []
			for node in range(num_nodes):
				values = [
					int(item[3].item())
					for item in gathered
					if int(item[0].item()) == node
				]
				if not values:
					raise RuntimeError(
						f"missing persistent sequence-capacity report for node {node}"
					)
				per_node_sequence_free.append(min(values))
		elif gathered_scope != 0:
			raise RuntimeError(
				f"unknown prefill sequence-capacity scope {gathered_scope}"
			)

		if self.rank == 0:
			logging.info(f"Per-node host KV free pages: {per_node_host_free} (chunk_size={chunk_size})")
			if per_node_sequence_free is not None:
				logging.info(
					f"[PREFILL] Persistent sequence slots free by node: "
					f"{per_node_sequence_free}"
				)

		# Step 3: Select sequences considering per-node host KV capacity.
		# The NCCL gather above and the logging below stay here; the greedy
		# per-node admission is delegated to PrefillScheduler. Kimi-Linear
		# additionally supplies a persistent KDA-state limit: unlike the token
		# cap used later by prepack, a KDA slot remains occupied until the
		# sequence completes or is evicted.
		prefill_batch = PrefillScheduler.select_prefill_batch(
			self._make_prefill_selection_request(
				all_candidates,
				per_node_host_free,
				num_nodes,
				chunk_size,
				per_rank_sequence_free=per_rank_sequence_free,
				per_node_sequence_free=per_node_sequence_free,
			)
		)

		if self.rank == 0:
			n_evicted = sum(
				1 for u in prefill_batch
				if self.global_batch.get_sequence(u).status == SequenceStatus.EVICTED
			)
			logging.info(
				f"[PREFILL] Selected {len(prefill_batch)} sequences "
				f"({n_evicted} recompute from eviction)"
			)

		return prefill_batch

	def _prefill_sequence_limits(self) -> dict:
		"""Return persistent model-state limits for prefill admission.

		Kimi-K3's TP8 attention group replicates each sequence's KDA state on
		all eight ranks of one node. Its limit is therefore per node, not per
		rank. Other models/PSMs return no limits and retain the host-KV-only
		selection contract.
		"""
		manager = getattr(self, "parallel_manager", None)
		method = getattr(manager, "prefill_sequence_limits", None)
		if method is None:
			return {}
		limits = method()
		if limits is None:
			return {}
		if not isinstance(limits, dict):
			raise TypeError(
				"parallel_manager.prefill_sequence_limits() must return a dict"
			)
		return limits

	def _plan_prefill_micro_batches(
		self, seq_lengths: List[int]
	) -> Tuple[List[Tuple[int, int]], int, bool]:
		"""Build the one authoritative prepacked-forward plan for local rows."""
		token_cap = (
			self.engine_config.Module_Batching_Config
			.prefill_micro_batch_token_cap
		)
		use_l2 = os.environ.get("BATCHGEN_L2_BALANCE", "1") == "1"
		single_sequence_only = (
			token_cap > 0
			and max(seq_lengths) > token_cap
			and hasattr(self.parallel_manager, "prefill_uses_streamed_sp8")
			and self.parallel_manager.prefill_uses_streamed_sp8()
		)
		micro_batches, l2_cap = build_prefill_micro_batches(
			seq_lengths,
			token_cap,
			l2_balance=use_l2,
			single_sequence_only=single_sequence_only,
		)
		return micro_batches, l2_cap, single_sequence_only

	def _prefill_model_pass_count(self, batch: List[int]) -> int:
		"""Return the number of complete model forwards local prefill will run."""
		if not batch:
			return 0
		if not self.enable_prepack:
			micro_batch_size = (
				self.engine_config.Module_Batching_Config
				.MoE_prefill_micro_batch_size
			)
			return math.ceil(len(batch) / micro_batch_size)

		seq_lengths = [
			self.global_batch.get_sequence(
				self._local_to_uuid_map[local_idx]
			).prompt_length
			for local_idx in batch
		]
		micro_batches, _, _ = self._plan_prefill_micro_batches(seq_lengths)
		return len(micro_batches)

	def _streamed_sp8_prefill_pass_alignment(
		self, local_prefill_indices: List[int]
	) -> Tuple[int, int]:
		"""Return local/global pass counts for hierarchical streamed-SP8."""
		method = getattr(
			self.parallel_manager,
			"streamed_sp8_requires_global_pass_alignment",
			None,
		)
		if method is None or not method():
			return 0, 0

		local_passes = self._prefill_model_pass_count(local_prefill_indices)
		global_passes = torch.tensor(
			[local_passes],
			dtype=torch.int64,
			device=self.torch_device,
		)
		dist.all_reduce(global_passes, op=dist.ReduceOp.MAX)
		global_passes = int(global_passes.item())
		if global_passes < local_passes:
			raise RuntimeError(
				f"Rank {self.rank}: global streamed-SP8 prefill pass count "
				f"{global_passes} is below local count {local_passes}"
			)
		return local_passes, global_passes

	def _make_prefill_selection_request(
		self, all_candidates: List[str], per_node_host_free: List[int],
		num_nodes: int, chunk_size: int, *,
		per_rank_sequence_free=None, per_node_sequence_free=None,
	) -> PrefillSelectionRequest:
		"""Snapshot the candidate metadata `select_prefill_batch` consumes."""
		from batchgen.sequence import INITIAL_GPU_PAGE_BUFFER
		candidates = []
		for uuid in all_candidates:
			seq = self.global_batch.get_sequence(uuid)
			seq_node = seq.assigned_rank // NUM_GPUS_PER_NODE
			G = self._decode_attn_tp_size()
			if G > 1:
				if seq.decode_dp_group is None:
					raise RuntimeError(
						f"sequence {uuid[:8]} has no decode_dp_group before "
						"TP prefill selection"
					)
				from batchgen.decode_dp_group import host_kv_owner_rank
				seq_node = (
					host_kv_owner_rank(seq.decode_dp_group, G)
					// NUM_GPUS_PER_NODE
				)
			candidates.append(PrefillCandidate(
				uuid=uuid,
				assigned_rank=seq.assigned_rank,
				node_id=seq_node,
				is_evicted=(seq.status == SequenceStatus.EVICTED),
				global_idx=seq.global_idx,
				total_decoded_before_eviction=getattr(
					seq, "total_decoded_before_eviction", 0
				),
				prompt_length=seq.prompt_length,
				kv_token_budget=seq.kv_token_budget,
				page_size=seq.PAGE_SIZE,
				host_kv_replication_factor=G,
			))
		return PrefillSelectionRequest(
			candidates=tuple(candidates),
			per_node_host_free=tuple(per_node_host_free),
			chunk_size=chunk_size,
			num_nodes=num_nodes,
			gpus_per_node=NUM_GPUS_PER_NODE,
			initial_gpu_page_buffer=INITIAL_GPU_PAGE_BUFFER,
			per_rank_sequence_free=(
				tuple(per_rank_sequence_free)
				if per_rank_sequence_free is not None else None
			),
			per_node_sequence_free=(
				tuple(per_node_sequence_free)
				if per_node_sequence_free is not None else None
			),
		)

	def _put_sequences_on_hold(self, uuids: List[str]) -> None:
		"""Move IN_DECODE sequences to ON_HOLD, freeing GPU KV but keeping host KV."""
		if not uuids:
			return

		if self.rank == 0:
			logging.info(
				f"[WATERMARK] Putting {len(uuids)} sequences ON_HOLD"
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
				if uuid in self._uuid_to_local_map:
					global_seq_ids.append(seq.global_idx)  # Use global_idx, not local_idx!

			if global_seq_ids:
				# Filter to only sequences the GPU manager actually tracks
				mgr = self.gpu_paged_kv_cache_manager
				known_ids = [gid for gid in global_seq_ids if gid in mgr._sequences]
				if known_ids:
					mgr.free_pages_for_sequences(known_ids)
				if len(known_ids) < len(global_seq_ids):
					unknown = len(global_seq_ids) - len(known_ids)
					logging.debug(
						f"Rank {self.rank}: Skipped freeing {unknown} sequences not in GPU KV manager"
					)
				# Also remove from tracking set
				for uuid in uuids:
					if uuid in self._uuid_to_local_map:
						self._sequences_with_gpu_kv.discard(uuid)

		# Update sequence status and reset GPU allocation
		# NOTE: Only reset gpu_pages_allocated, NOT had_initial_gpu_reservation.
		# ON_HOLD sequences are continuing decode when reloaded, so they should
		# get EXTENSION_GPU_PAGE_BUFFER (smaller), not INITIAL_GPU_PAGE_BUFFER.
		for uuid in uuids:
			seq = self.global_batch.get_sequence(uuid)
			seq.gpu_pages_allocated = 0
			seq.log_event(SeqEvent.ON_HOLD, self.rank, "trigger=watermark")
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

		# Greedy per-rank 90%-watermark fill. The candidate enumeration above
		# and the logging below stay here; the selection is delegated.
		decode_batch = DecodeScheduler.select_decode_batch(
			self._make_decode_batch_request(all_candidates, total_pages)
		)

		if self.rank == 0:
			logging.info(
				f"[DECODE] Prepared batch: {len(decode_batch)} sequences"
			)

		return decode_batch

	def _make_decode_batch_request(
		self, all_candidates: List[str], total_pages: int,
	) -> DecodeBatchRequest:
		"""Snapshot the candidate metadata `select_decode_batch` consumes."""
		candidates = []
		for uuid in all_candidates:
			seq = self.global_batch.get_sequence(uuid)
			candidates.append(DecodeCandidate(
				uuid=uuid,
				assigned_rank=seq.assigned_rank,
				global_idx=seq.global_idx,
				req_pages=seq.get_gpu_pages_for_two_page_buffer(),
				decode_dp_group=seq.decode_dp_group,
			))
		return DecodeBatchRequest(
			candidates=tuple(candidates),
			total_pages=total_pages,
			world_size=self.world_size,
			max_rank_bsz=getattr(self, "_decode_padding_bsz", 0) or 0,
			attn_tp_size=self._decode_attn_tp_size(),
		)

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
		n = len(decode_uuids)
		if n == 0:
			return [], [], []

		# Vectorized completion check: build tensors once, compare in batch
		decoded_lens = torch.empty(n, dtype=torch.int64)
		max_lens = torch.empty(n, dtype=torch.int64)
		ctx_lens = torch.empty(n, dtype=torch.int64)
		eos_flags = torch.empty(n, dtype=torch.bool)
		ignore_eos = self._ignore_eos

		seqs = []
		for i, uuid in enumerate(decode_uuids):
			seq = self.global_batch.get_sequence(uuid)
			seqs.append(seq)
			decoded_lens[i] = seq.decoded_length
			max_lens[i] = seq.max_decode_length
			ctx_lens[i] = seq.current_context_length
			eos_flags[i] = seq.eos_reached and not ignore_eos

		# Variable-length N-gram repetition detection at decision boundary
		# Catches repeating patterns of length 2-100 tokens (32 repetitions required)
		if REP_DETECTION:
			for i in range(n):
				seq = seqs[i]
				if seq.decoded_length >= 64 and not seq._rep_detected:
					uuid = decode_uuids[i]
					local_idx = self._uuid_to_local_map.get(uuid)
					if local_idx is not None and local_idx in self.query_book:
						dl = seq.decoded_length
						tokens = self.query_book[local_idx].decoded_tokens[0]
						if _check_repeating_pattern(tokens, dl):
							seq._rep_detected = True
							seq.eos_reached = True
							logging.warning(
								f"Rank {self.rank}: REPETITION (ngram) {seq.uuid} "
								f"gid={seq.global_idx} at decoded_len={dl}"
							)

		rep_flags = torch.tensor([seqs[i]._rep_detected for i in range(n)], dtype=torch.bool)
		completed_mask = (
			(decoded_lens >= max_lens)
			| (ctx_lens >= self.model_context_length)
			| eos_flags
			| rep_flags
		)

		completed_uuids = []
		active_uuids = []
		active_local_indices = []
		for i in range(n):
			uuid = decode_uuids[i]
			if completed_mask[i]:
				completed_uuids.append(uuid)
				seq = seqs[i]
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

	def _submit_completed_to_incremental_writer(
		self,
		completed_uuids: List[str],
	) -> None:
		"""Gather completed sequence tokens from all ranks and submit to writer.

		Sequences are distributed across ranks (each rank owns a subset).
		Uses all_gather_object to collect decoded tokens from the owning
		rank to rank 0 where the writer lives. All ranks must participate
		in the collective.
		"""
		if not completed_uuids:
			return

		# Quick check: does rank 0 have a writer? Broadcast to all ranks.
		writer = getattr(self, '_incremental_writer', None)
		has_writer = torch.tensor(
			[1 if writer is not None else 0],
			dtype=torch.int32, device=self.torch_device
		)
		dist.all_reduce(has_writer, op=dist.ReduceOp.MAX)
		if has_writer.item() == 0:
			return

		# Each rank collects tokens + finish_reason for its locally-owned completed sequences
		my_completed_tokens = []
		for uuid in completed_uuids:
			if uuid in self._uuid_to_local_map:
				local_idx = self._uuid_to_local_map[uuid]
				seq = self.global_batch.get_sequence(uuid)
				if seq is not None and local_idx in self.query_book:
					finish_reason = self._get_finish_reason(seq)
					my_completed_tokens.append(
						(seq.global_idx, self.query_book[local_idx].decoded_tokens[:, :seq.decoded_length].clone(), finish_reason)
					)

		# All ranks participate in gather (NCCL collective requirement)
		all_completed_tokens = [None] * self.world_size
		dist.all_gather_object(all_completed_tokens, my_completed_tokens)

		# Rank 0 submits to writer
		# Each global_idx is owned by exactly one rank, so no duplicates possible
		if writer is not None:
			for rank_tokens in all_completed_tokens:
				if rank_tokens:
					for global_idx, tokens, finish_reason in rank_tokens:
						writer.submit(global_idx, tokens, finish_reason=finish_reason)

	def _finish_prefill_completed_sequences(self, prefill_uuids: List[str]) -> List[str]:
		"""Complete the sequences whose budget is satisfied by the prefill token.

		Prefill already samples the first token and appends it through the
		normal decode write path (``query_book[..].decoded_tokens`` at
		``seq.decoded_length``, then ``decoded_length += 1``; see the writeback
		loop at the end of ``prefill``/``prefill_prepacked``). For a
		``max_tokens=1`` request that token IS the whole completion, so the
		sequence is finished before decode starts. Previously every prefilled
		sequence was handed to the decode phase unconditionally, which
		 - loaded the decode model and configured decoding for nothing, and
		 - on a prefill-only model (Kimi-K3 ``stream_all_modules``, M-PR-6)
		   replaced the answer with decode's ``NotImplementedError`` text.

		Only the length budget is checked here: that is exactly the first test
		decode's own boundary check makes (``CompletionHandler`` /
		``_check_and_handle_completions``: ``decoded_length >=
		max_decode_length``), and it is also the test that wins in
		``get_finish_reason``, so these sequences report the same
		length-capped ``finish_reason`` they report today. Sequences that
		stop for any other reason (EOS, context limit) are left PREFILLED and
		reach decode exactly as before — ``max_tokens > 1`` behaviour is
		unchanged.

		Rank alignment: ``decoded_length`` is advanced only on the owning
		rank, so the set is derived AFTER ``_sync_sequence_metadata``
		replicates it to every rank. Every rank then computes the identical
		set from identical batch-global state — required both for the
		collectives below and because the resulting PREFILLED -> COMPLETED
		transition is what makes the decode ``while`` loop's
		``has_prefilled()`` false. A rank-divergent set would deadlock the
		next collective.

		Returns the list of completed uuids (identical on every rank).
		"""
		if not prefill_uuids:
			return []

		# Replicate owner-side decoded_length / current_context_length to all
		# ranks. Also required by _report_completion, which reads those fields
		# on rank 0 for sequences owned elsewhere.
		self._sync_sequence_metadata(prefill_uuids)

		completed_uuids = []
		for uuid in prefill_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None or seq.status != SequenceStatus.PREFILLED:
				continue
			if seq.decoded_length >= seq.max_decode_length:
				completed_uuids.append(uuid)

		if not completed_uuids:
			return []

		if self.rank == 0:
			logging.info(
				f"[PREFILL] {len(completed_uuids)}/{len(prefill_uuids)} sequences "
				f"completed at prefill (decode budget satisfied by the first "
				f"sampled token); they skip the decode phase"
			)

		# Same order as the decode-phase completion handling in generate():
		# writer -> gather text -> release KV -> scalar cleanup -> status ->
		# report. _submit_completed_to_incremental_writer and
		# _gather_completed_tokens are collectives; every rank calls them with
		# the identical uuid list.
		self._submit_completed_to_incremental_writer(completed_uuids)
		gathered_texts = self._gather_completed_tokens(completed_uuids)

		my_completed = [u for u in completed_uuids if u in self._uuid_to_local_map]
		if my_completed:
			# prefill_prepacked writes KV straight to host, so most of these
			# never registered with the GPU paged manager.
			gpu_allocated = [u for u in my_completed if u in self._sequences_with_gpu_kv]
			kda_only = [u for u in my_completed if u not in self._sequences_with_gpu_kv]
			if gpu_allocated:
				self._release_gpu_kv_pages(self._get_local_indices_for_uuids(gpu_allocated))
			if kda_only:
				self._release_kda_state_slots([
					self.global_batch.get_sequence(uuid).global_idx
					for uuid in kda_only
				])
			self._release_host_kv_pages_for_batch(my_completed)
		for uuid in completed_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is not None:
				seq.gpu_pages_allocated = 0
				seq.host_pages_allocated = 0
				seq.host_token_capacity = 0
				self._sequences_with_gpu_kv.discard(uuid)

		self._update_batch_status(completed_uuids, SequenceStatus.COMPLETED)

		# Legacy /v1/inference gathers results at the END of generate() by
		# iterating _local_to_uuid_map. _report_completion below pops that map
		# (release_local_query_slot also drops the query_book entry), so a
		# C4-completed sequence would be invisible to that gather and the
		# request would return "Results unexpectedly empty after inference".
		# Capture the text while the slot still exists.
		# Gated on the absence of a response queue: pool/batch mode is fed by
		# _report_completion and _submit_completed_to_incremental_writer, so it
		# must NOT also accumulate here -- that store is never drained in a
		# persistent server and would grow without bound.
		if self._response_queue is None:
			# getattr, not a plain attribute read: hot reload rebinds methods on a
			# LIVE worker and never re-runs __init__, so an attribute introduced in
			# __init__ is absent on a reloaded process. _validate_reload
			# (server_worker_main_loop.py:68) warns about exactly this and does not
			# fix it -- "These will cause AttributeError if accessed."
			store = getattr(self, '_prefill_completed_results', None)
			if store is None:
				store = self._prefill_completed_results = {}
			for uuid in completed_uuids:
				local_idx = self._uuid_to_local_map.get(uuid)
				seq = self.global_batch.get_sequence(uuid)
				if local_idx is None or seq is None or local_idx not in self.query_book:
					continue
				_decoded = self.query_book[local_idx].decoded_tokens[:, :seq.decoded_length]
				store[seq.global_idx] = self._decode_tokens_to_string(_decoded)

		# Runs LAST: _report_completion pops the local-index map and frees the
		# buffer-pool slot.
		for uuid in completed_uuids:
			self._report_completion(uuid, gathered_text=gathered_texts.get(uuid))

		return completed_uuids

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
		
		# M2b: assign the decode DP-group before the transition (no-op for G==1).
		self._assign_decode_dp_groups(new_uuids)

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
				resuming_diag.append({
					'uuid': uuid[:8],
					'decoded_len': seq.decoded_length,
					'ctx_len': seq.current_context_length,
					'prompt_len': seq.prompt_length,
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

	def generate_persistent(self):
		"""Pool mode entry point: init core, empty batch, persistent generate() loop.

		Called from server_worker_main_loop when pool mode is active.
		Uses Init() to set up model/tokenizer/KV, then enters generate()
		with an empty global_batch that accepts sequences via admission messages.
		"""
		logging.info(f"Rank {self.rank}: Entering persistent generate() mode")

		# Use Init() to set up core components (model, tokenizer, KV config)
		# num_queries=0 means no sequences yet — they'll come via admission
		if not self._core_initialized:
			self.Init(None, self.max_decoding_length, 0,
				max_context_length=self.max_context_length)

		# Initialize empty global batch (Init may have created one via _reset)
		self.global_batch = SequenceBatch()

		# The buffer pool is NOT pre-allocated here. Both of its widths depend on
		# the requests: input_ids needs prompt + that request's decode budget,
		# decoded_tokens needs that request's max_completion_tokens — and neither
		# is known until the first admission is tokenized. Sizing them at
		# model_context_length "just in case" is what allocated 2 x 80 GiB per
		# worker at K3's 1,048,576-token context. _tokenize_admitted_sequences
		# allocates on first admission and grows if a later one needs more.
		self._buffer_pool = None
		logging.info(
			f"Rank {self.rank}: Buffer pool deferred to first admission "
			f"(rows={self._max_pool_size}, widths sized per batch)"
		)

		# Initialize index maps
		self._local_to_uuid_map = {}
		self._uuid_to_local_map = {}
		self._free_local_indices = set()
		self._next_local_idx = 0
		self.num_global_queries = 0
		self.num_local_queries = 0
		self._rejected_sequences = []

		# Reset max_input_length from Init's 8192 default to 0.
		# In legacy mode, _tokenize_global_batch sets max_input_length to the
		# actual longest prompt, then _update_config_after_tokenization propagates
		# it to max_prompt_length in engine config BEFORE prefill/decode.
		# In pool mode, Init(None,...) defaults max_input_length to 8192 for the
		# initializer, but once core components are ready we must reset it so the
		# first admission batch correctly sets it from actual prompt lengths.
		# Without this, max_prompt_length stays at 8192 which causes wrong
		# KV_Storage_Config.reserved_length and GPU buffer sizing.
		self.max_input_length = 0

		# Enter the persistent generate loop
		return self.generate()

	def _nsys_decode_profile_begin_forward(
		self,
		*,
		local_iteration: int,
		local_bsz: int,
		max_rank_bsz: int,
	) -> Optional[int]:
		"""Start an env-gated nsys decode-forward capture window."""
		if not BATCHGEN_NSYS_DECODE_PROFILE:
			return None
		self._nsys_decode_profile_forward_count += 1
		forward_idx = self._nsys_decode_profile_forward_count
		if forward_idx > BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT:
			return None

		if (
			self.rank in BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS
			and not self._nsys_decode_profile_started
		):
			torch.cuda.synchronize(self.torch_device)
			logging.info(
				"[NSYS_DECODE_PROFILE] rank=%s starting cuda profiler capture "
				"limit=%s controller_ranks=%s",
				self.rank,
				BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT,
				sorted(BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS),
			)
			torch.cuda.cudart().cudaProfilerStart()
			self._nsys_decode_profile_started = True

		range_name = (
			f"BatchGen_decode_forward_{forward_idx}"
			f"_rank_{self.rank}_local_bsz_{local_bsz}_max_rank_bsz_{max_rank_bsz}"
			f"_iter_{local_iteration}"
		)
		torch.cuda.nvtx.range_push(range_name)
		return forward_idx

	def _nsys_decode_profile_end_forward(self, forward_idx: Optional[int]) -> None:
		"""End one env-gated nsys decode-forward range and optionally exit."""
		if forward_idx is None:
			return
		torch.cuda.nvtx.range_pop()
		if forward_idx < BATCHGEN_NSYS_DECODE_PROFILE_FORWARD_LIMIT:
			return

		torch.cuda.synchronize(self.torch_device)
		if dist.is_available() and dist.is_initialized():
			dist.barrier()
		if (
			self.rank in BATCHGEN_NSYS_DECODE_PROFILE_CONTROLLER_RANKS
			and not self._nsys_decode_profile_stopped
		):
			logging.info(
				"[NSYS_DECODE_PROFILE] rank=%s stopping cuda profiler capture "
				"after %s decode forwards",
				self.rank,
				forward_idx,
			)
			torch.cuda.cudart().cudaProfilerStop()
			self._nsys_decode_profile_stopped = True
		if dist.is_available() and dist.is_initialized():
			dist.barrier()

		if BATCHGEN_NSYS_DECODE_PROFILE_EXIT:
			logging.info(
				"[NSYS_DECODE_PROFILE] rank=%s exiting after %s profiled decode forwards",
				self.rank,
				forward_idx,
			)
			sys.stdout.flush()
			sys.stderr.flush()
			os._exit(0)

	def _ensure_pynccl_communicator(self) -> None:
		"""Create the PyNccl communicator unless every rank already holds one.

		Collective and idempotent: the need-init vote, the port broadcast and
		the pre-TCPStore barrier are all torch.distributed collectives, so every
		rank must call this together. Once all ranks have a communicator the
		vote fails and nothing is created.
		"""
		if os.getenv("BATCHGEN_ENABLE_ALL_TO_ALL", "0") != "0":
			return

		# Verify rank consistency
		if dist.is_initialized():
			assert self.rank == dist.get_rank(), \
				f"Rank mismatch: self.rank={self.rank}, dist.get_rank()={dist.get_rank()}"

		# Skip PyNccl initialization for single GPU (no inter-GPU communication needed)
		if self.world_size == 1:
			logging.debug("Single GPU mode: skipping PyNccl communicator initialization")
			return

		comm_master_addr = os.getenv("COMM_MASTER_ADDR")

		# Coordinate PyNccl initialization across all ranks
		# Use all_reduce to check if ANY rank needs to (re)init the communicator
		need_init = 1 if self.comm is None else 0
		need_init_tensor = torch.tensor([need_init], dtype=torch.int32, device=self.torch_device)
		dist.all_reduce(need_init_tensor, op=dist.ReduceOp.MAX)
		any_rank_needs_init = need_init_tensor.item() > 0

		if not any_rank_needs_init:
			return

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

	def prepare_kimi_k3_startup(self) -> None:
		"""Run Kimi-K3's workload-independent one-time work before readiness.

		Everything here is fixed by the checkpoint and the topology, not by the
		request: core components, required CUDA-extension loading, the PyNccl
		communicator, the model build, the resident-EP decode shards, and — on
		the distributed weight store — the streamed-SP8 buffers plus the single
		H2D weight schedule. Doing it on the first admission instead made a
		healthy server take minutes to answer its first request.

		What stays at admission time cannot be sized here: the QueryBook buffer
		pool needs the tokenized prompt/decode widths, and the resident prefill
		output needs the micro-batch token count of the admitted sequences.

		Collective (Init, _ensure_pynccl_communicator and configure_decoding all
		run torch.distributed collectives) and idempotent.
		"""
		if self._k3_startup_completed:
			return

		logging.info(
			f"Rank {self.rank}: [K3_STARTUP] Building model and weight pipeline "
			f"before readiness"
		)
		self.Init(None, _K3_STARTUP_MAX_DECODING_LENGTH, 0)
		self._preload_kimi_k3_runtime_extensions()
		self._ensure_pynccl_communicator()
		# The resident-EP decode MoE runs its own all_gather/all_reduce, so the
		# manager needs the communicator before configure_decoding, not at the
		# first decode phase.
		self.parallel_manager.set_comm(self.comm)

		# Decode first: configure_decoding materializes the stacked MXFP4 EP
		# shard that both phases keep resident. _MAX_DECODE_RANK_BSZ is the cap
		# the runtime decode path applies to its own estimate, so the padded MoE
		# buffers are already sized for the largest batch that can be admitted.
		self._load_decode_model(_MAX_DECODE_RANK_BSZ, comm=self.comm)

		# Then hand the built model to the prefill mode this weight topology
		# will actually use, so the first admission inherits this phase directly
		# instead of rebuilding the streamed-SP8 buffers itself.
		startup_prefill_mode = self.parallel_manager.default_prefill_moe_mode()
		self.parallel_manager.set_prefill_moe_mode(startup_prefill_mode)
		self.model, self.weight_copy_task = self.parallel_manager.configure_prefill()
		self.set_phase("prefill")
		self._install_prefill_weight_copy_pipeline(k3_prefill_profile=False)

		self._k3_startup_completed = True
		self._k3_startup_prefill_ready = True
		self._k3_startup_prefill_mode = startup_prefill_mode
		logging.info(f"Rank {self.rank}: [K3_STARTUP] Prefill phase ready")

	def _preload_kimi_k3_runtime_extensions(self) -> None:
		"""Load K3's first-forward CUDA extensions before HTTP readiness."""
		from batchgen.moe.dispatch_scatter_3d import _load_dispatch_reduce_module
		from batchgen_kernels.conv1d import _get_ext as _get_causal_conv1d_ext
		try:
			from batchgen_kernels.attention.kda_fused_decode import (
				_get_ext as _get_kda_fused_decode_ext,
			)
		except ImportError as exc:
			raise RuntimeError(
				f"Rank {self.rank}: required Kimi-K3 extension "
				f"kda_fused_decode failed to load - {exc}"
			) from exc

		required_extensions = (
			("causal_conv1d", _get_causal_conv1d_ext),
			("dispatch_scatter_3d", _load_dispatch_reduce_module),
			("kda_fused_decode", _get_kda_fused_decode_ext),
		)
		for name, loader in required_extensions:
			if loader() is None:
				raise RuntimeError(
					f"Rank {self.rank}: required Kimi-K3 extension {name} failed to load"
				)
			logging.info(
				f"Rank {self.rank}: [K3_STARTUP] Loaded runtime extension {name}"
			)

		# K3's pure-BF16 NoPE decode consumes these two symbols directly.  Import,
		# validate, and kernel-warm them here so its function-local cached import
		# cannot trigger extension setup after the server has reported ready.
		try:
			import flash_mla
		except ImportError as e:
			raise RuntimeError(
				f"Rank {self.rank}: required Kimi-K3 extension flash_mla failed to load - {e}"
			)
		for symbol in ("flash_mla_with_kvcache", "get_mla_metadata"):
			if not callable(getattr(flash_mla, symbol, None)):
				raise RuntimeError(
					f"Rank {self.rank}: required Kimi-K3 extension flash_mla is "
					f"missing callable {symbol}"
				)
		self._warmup_kimi_k3_flash_mla(flash_mla)
		logging.info(
			f"Rank {self.rank}: [K3_STARTUP] Loaded and warmed runtime extension flash_mla"
		)

	def _warmup_kimi_k3_flash_mla(self, flash_mla) -> None:
		"""Launch the exact K3 decode kernel once before HTTP readiness."""
		device = torch.device("cuda", torch.cuda.current_device())
		cache_seqlens = torch.ones((1,), dtype=torch.int32, device=device)
		query = torch.zeros(
			(1, 1, 12, 576), dtype=torch.bfloat16, device=device
		)
		blocked_k = torch.zeros(
			(1, 64, 1, 576), dtype=torch.bfloat16, device=device
		)
		block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
		tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
			cache_seqlens, 12, 1
		)
		flash_mla.flash_mla_with_kvcache(
			query,
			blocked_k,
			block_table,
			cache_seqlens,
			512,
			tile_scheduler_metadata,
			num_splits,
			causal=True,
		)
		torch.cuda.synchronize(device)

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
		self._timing_logged = False  # Print timing once per batch group
		self._decode_group_idx = 0  # Track decode groups for diagnostic logging
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
		self._ensure_pynccl_communicator()

		iteration = 0

		# Persistent loop: continues until all completed AND no more admissions expected
		while True:
			# --- ADMISSION CHECK: Poll for new sequences from IntakePool ---
			if self._admission_queue is not None:
				admitted = self._poll_admissions()
				if admitted and self.rank == 0:
					logging.info(f"[POOL] Admitted new sequences, total in batch: {len(self.global_batch)}")
					self._timing_logged = False  # Reset for new batch group

			# --- TERMINATION CHECK ---
			if self.global_batch.all_completed():
				# Print timing summary when all current work is done
				if not self._timing_logged and self.rank == 0:
					gen_time = time.perf_counter() - generation_start_time
					total_prompt = sum(s.prompt_length for s in self.global_batch)
					total_decoded = sum(s.decoded_length for s in self.global_batch)
					num_seq = len(self.global_batch)
					pf_tp = total_prompt / prefill_time if prefill_time > 0 else 0
					dc_tp = total_decoded / decoding_time if decoding_time > 0 else 0
					ov_tp = (total_prompt + total_decoded) / gen_time if gen_time > 0 else 0
					logging.info(
						f"Pool batch group completed:\n"
						f"  Sequences: {num_seq}\n"
						f"  Prefill: {prefill_time:.1f}s ({pf_tp:,.0f} tok/s)\n"
						f"  Decode: {decoding_time:.1f}s ({dc_tp:,.0f} tok/s)\n"
						f"  Total: {gen_time:.1f}s ({ov_tp:,.0f} tok/s)\n"
						f"  Prompt tokens: {total_prompt:,}, Decoded tokens: {total_decoded:,}"
					)
					self._timing_logged = True
				if self._admission_queue is None:
					break  # Legacy mode: no pool, just finish
				if self._shutdown_requested:
					break  # Pool mode: shutdown requested and all done
				# Pool mode: wait briefly for more work before exiting
				# status tensor encoding: [has_new_work, shutdown, reload]
				import queue as queue_mod
				if self.rank == 0:
					try:
						msg = self._admission_queue.get(timeout=1.0)
						if msg is None:
							self._shutdown_requested = True
							status = torch.tensor([0, 1, 0], dtype=torch.int32, device=self.torch_device)
							dist.broadcast(status, src=0)
						elif isinstance(msg, dict) and msg.get("type") == "admit":
							# Broadcast that we got new work
							status = torch.tensor([1, 0, 0], dtype=torch.int32, device=self.torch_device)
							dist.broadcast(status, src=0)
							container = [msg]
							dist.broadcast_object_list(container, src=0)
							self._admit_sequences_from_message(msg)
							# Reset per-batch-group timing so each admission cycle
							# emits its own "Pool batch group completed" summary.
							prefill_time = 0.0
							decoding_time = 0.0
							generation_start_time = time.perf_counter()
							self._timing_logged = False
							# Continue loop — new sequences will be picked up
						elif isinstance(msg, dict) and msg.get("command") == "reload":
							# Hot-reload command — broadcast to all ranks then handle.
							# Result is written to /tmp/batchgen_reload_status/rank_<N>.json
							# inside _handle_hot_reload (via _write_reload_status), NOT
							# put on response_queue. Putting on response_queue would
							# deadlock the FastAPI event loop because the sync HTTP
							# handler can't drain mp.Queue while blocking.
							status = torch.tensor([0, 0, 1], dtype=torch.int32, device=self.torch_device)
							dist.broadcast(status, src=0)
							container = [msg]
							dist.broadcast_object_list(container, src=0)
							self._handle_hot_reload(msg)
						elif isinstance(msg, dict) and "prompts" in msg:
							# Legacy /v1/inference payload -- see the matching
							# guard in _poll_admissions. The `else` below would
							# swallow it and the caller would then steal the next
							# completion off the shared response queue.
							raise LegacyInferenceDeprecated()
						else:
							status = torch.tensor([0, 0, 0], dtype=torch.int32, device=self.torch_device)
							dist.broadcast(status, src=0)
					except queue_mod.Empty:
						# No work arrived, broadcast no-work to other ranks
						status = torch.tensor([0, 0, 0], dtype=torch.int32, device=self.torch_device)
						dist.broadcast(status, src=0)
						continue  # Try again
				else:
					# Non-rank-0: wait for rank 0's broadcast
					status = torch.tensor([0, 0, 0], dtype=torch.int32, device=self.torch_device)
					dist.broadcast(status, src=0)
					has_new = status[0].item() == 1
					is_shutdown = status[1].item() == 1
					is_reload = status[2].item() == 1
					if has_new:
						container = [None]
						dist.broadcast_object_list(container, src=0)
						self._admit_sequences_from_message(container[0])
						# Reset per-batch-group timing (matches rank-0 branch).
						prefill_time = 0.0
						decoding_time = 0.0
						generation_start_time = time.perf_counter()
						self._timing_logged = False
					elif is_reload:
						container = [None]
						dist.broadcast_object_list(container, src=0)
						self._handle_hot_reload(container[0])
					elif is_shutdown:
						self._shutdown_requested = True
					# Continue loop regardless
				if self.global_batch.all_completed() and self._shutdown_requested:
					break
				if self.global_batch.all_completed():
					continue  # Keep waiting

			iteration += 1
			if self.rank == 0:
				logging.info(f"--- Iteration {iteration} ---")

			# HBM diagnostic: track memory across iterations to detect leaks
			if torch.cuda.is_available():
				free_mem, total_mem = torch.cuda.mem_get_info(self.local_rank)
				allocated = torch.cuda.memory_allocated(self.local_rank) / 1e9
				reserved = torch.cuda.memory_reserved(self.local_rank) / 1e9
				logging.info(
					f"[HBM] Rank {self.rank} iter {iteration} START: "
					f"free={free_mem/1e9:.2f}GB alloc={allocated:.2f}GB rsv={reserved:.2f}GB"
				)

			# NOTE: Watchdog is fed within prefill and decode loops, not here.
			# This ensures we only monitor the actual inference phases.

			# =================================================================
			# 1. PREFILL PHASE: Fill Host KV Cache
			# =================================================================
			if self.global_batch.has_queueing() or (self.enable_host_kv_eviction and self.global_batch.has_evicted()):
				dist.barrier()

				# CRITICAL FIX: Sync sequence metadata BEFORE rebalancing
				# After decode interruption or prefill completion, each rank has divergent
				# metadata for sequences it doesn't own locally. This sync ensures all ranks
				# have consistent current_context_length values before migration. PREFILLED
				# sequences must be synced because their attention mask has been updated
				# (prompt_len + 1) after prefill, and migration includes PREFILLED status.
				# EVICTED sequences MUST be synced too: the owner rewrites their
				# prompt_length at eviction time (in _page_boundary_fast) to the
				# reconstructed re-entry length, and _prepare_prefill_batch (called
				# a few lines below) reads prompt_length on all ranks to size the
				# host KV reservation. Without this sync, non-owners read the stale
				# original prompt length, under-count host KV pages, over-admit, and
				# crash at allocate_pages_for_sequences.
				prefilled_uuids = [seq.uuid for seq in self.global_batch if seq.status == SequenceStatus.PREFILLED]
				on_hold_uuids = [seq.uuid for seq in self.global_batch if seq.status == SequenceStatus.ON_HOLD]
				in_decode_uuids = [seq.uuid for seq in self.global_batch if seq.status == SequenceStatus.IN_DECODE]
				evicted_uuids_for_sync = (
					[seq.uuid for seq in self.global_batch if seq.status == SequenceStatus.EVICTED]
					if self.enable_host_kv_eviction else []
				)
				all_active_uuids = prefilled_uuids + on_hold_uuids + in_decode_uuids + evicted_uuids_for_sync
				if all_active_uuids:
					self._sync_sequence_metadata(all_active_uuids)
					logging.debug(
						f"Rank {self.rank}: Synced metadata for {len(all_active_uuids)} sequences before rebalance "
						f"(prefilled={len(prefilled_uuids)}, on_hold={len(on_hold_uuids)}, "
						f"in_decode={len(in_decode_uuids)}, evicted={len(evicted_uuids_for_sync)})"
					)

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
					for uuid in prefill_uuids:
						seq = self.global_batch.get_sequence(uuid)
						is_reentry = seq.evicted_token_ids is not None
						seq.log_event(SeqEvent.PREFILL_START, self.rank,
							f"evicted_reentry={is_reentry}")
					self._update_batch_status(prefill_uuids, SequenceStatus.IN_PREFILL)

					# A. Config Prefill (this adds new sequences to _uuid_to_local_map)
					config_start = time.perf_counter()
					self._config_prefill_for_batch(prefill_uuids)
					config_prefill_time += time.perf_counter() - config_start

					# Get local indices AFTER config (new sequences now in map)
					local_prefill_indices = self._get_local_indices_for_uuids(prefill_uuids)
					local_prefill_passes, global_prefill_passes = (
						self._streamed_sp8_prefill_pass_alignment(
							local_prefill_indices
						)
					)

					# B. Execute Prefill
					if local_prefill_indices:
						if torch.cuda.is_available():
							free_mem, total_mem = torch.cuda.mem_get_info(self.local_rank)
							allocated = torch.cuda.memory_allocated(self.local_rank) / 1e9
							logging.info(
								f"[HBM] Rank {self.rank} BEFORE prefill ({len(local_prefill_indices)} seqs): "
								f"free={free_mem/1e9:.2f}GB alloc={allocated:.2f}GB"
							)
						# DEBUG instrumentation (env-gated, off by default):
						# per-allocation attribution of the prefill HBM peak.
						# BATCHGEN_MEM_PROFILE=1 turns on torch's allocator history
						# recorder around the prefill forward and dumps a snapshot
						# (also on OOM, via the finally) to BATCHGEN_MEM_PROFILE_DIR.
						_memprof = os.environ.get("BATCHGEN_MEM_PROFILE", "0") == "1"
						if _memprof:
							_memprof_base = torch.cuda.memory_allocated()
							# context="alloc"/stacks="python": frames on allocation
							# events only. K3 prefill does ~1e6 allocations (the
							# 896-expert moe_infer loop), so recording free-event
							# and C++ stacks too would double cost for no signal.
							torch.cuda.memory._record_memory_history(
								context="alloc",
								stacks=os.environ.get("BATCHGEN_MEM_PROFILE_STACKS", "python"),
								max_entries=int(os.environ.get("BATCHGEN_MEM_PROFILE_ENTRIES", "3000000")),
							)
							torch.cuda.reset_peak_memory_stats()
							logging.info(
								f"[MEMPROF] Rank {self.rank}: recording ON, "
								f"baseline_alloc={_memprof_base/2**30:.3f}GiB"
							)
						prefill_start = time.perf_counter()
						try:
							with torch.inference_mode():
								if self.enable_prepack:
									self.prefill_prepacked(local_prefill_indices)
								else:
									self.prefill(local_prefill_indices)
						finally:
							if _memprof:
								logging.info(
									f"[MEMPROF] Rank {self.rank}: "
									f"peak_alloc={torch.cuda.max_memory_allocated()/2**30:.3f}GiB "
									f"cur_alloc={torch.cuda.memory_allocated()/2**30:.3f}GiB "
									f"reserved={torch.cuda.memory_reserved()/2**30:.3f}GiB "
									f"peak_reserved={torch.cuda.max_memory_reserved()/2**30:.3f}GiB "
									f"baseline_alloc={_memprof_base/2**30:.3f}GiB"
								)
								_memprof_path = os.path.join(
									os.environ.get("BATCHGEN_MEM_PROFILE_DIR", "/tmp"),
									f"memprof_rank{self.rank}_{int(time.time())}.pickle",
								)
								try:
									torch.cuda.memory._dump_snapshot(_memprof_path)
									logging.info(f"[MEMPROF] Rank {self.rank}: snapshot -> {_memprof_path}")
								except Exception as _memprof_err:
									logging.error(f"[MEMPROF] Rank {self.rank}: dump failed: {_memprof_err}")
								torch.cuda.memory._record_memory_history(enabled=None)
						prefill_time += time.perf_counter() - prefill_start

						# CRITICAL: Wait for all async KV offloads to complete before decode.
						# async_offload_layer_kv_to_host returns a future backed by a
						# std::async CPU thread that issues cudaMemcpyAsync on a d2h
						# stream. Discarding the future (fire-and-forget) is unsafe —
						# the CPU thread may not have run yet, so torch.cuda.synchronize
						# would have nothing to wait for. Wait on every captured future
						# first, then sync the device to flush the d2h stream.
						from batchgen.models.wrappers.attention import AttnWrapperBase as _AWB
						num_retired = _AWB.retire_pending_prefill_offloads(
							device=self.torch_device,
							reason="end of prefill",
						)
						if num_retired and self.rank == 0:
							logging.info(
								f"[PREFILL_SYNC] waited on {num_retired} async KV offload tasks"
							)

					transport_only_passes = (
						global_prefill_passes - local_prefill_passes
					)
					if transport_only_passes:
						logging.info(
							f"Rank {self.rank}: joining {transport_only_passes} "
							"streamed-SP8 prefill weight schedules with no local rows"
						)
						transport_start = time.perf_counter()
						for _ in range(transport_only_passes):
							self.feed_watchdog()
							with torch.inference_mode():
								self.parallel_manager.run_streamed_sp8_transport_only_prefill(
									1
								)
						prefill_time += time.perf_counter() - transport_start

					# Cleanup & Status Update
					self._unregister_fp8_weights()
					for uuid in prefill_uuids:
						seq = self.global_batch.get_sequence(uuid)
						seq.log_event(SeqEvent.PREFILL_DONE, self.rank,
							f"decoded_len={seq.decoded_length}")
					self._update_batch_status(prefill_uuids, SequenceStatus.PREFILLED)
					dist.barrier()

					# C4: a request whose whole budget is the prefill-sampled
					# token is DONE here. Completing it now (instead of sending
					# it into decode to be completed on the first boundary
					# check) keeps the decode phase — and its decode-model load
					# — off the critical path for max_tokens=1, and is the only
					# way a prefill-only model can answer at all.
					self._finish_prefill_completed_sequences(prefill_uuids)

				# After prefill completes, poll for newly arrived sequences.
				# If more QUEUEING sequences exist and host KV has capacity,
				# loop back to prefill instead of entering decode.
				if self._admission_queue is not None:
					self._poll_admissions()
				if self.global_batch.has_queueing():
					next_prefill = self._prepare_prefill_batch()
					if next_prefill:
						if self.rank == 0:
							logging.info(
								f"[PREFILL] Back-to-back prefill: {len(next_prefill)} new sequences ready"
							)
						continue  # loop back to prefill phase

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
				# TP attention replicates one group's sequences across all G ranks,
				# so each rank needs the per-group batch, not total/world_size.
				max_num_seq_estimate = estimate_max_decode_replica_batch(
					total_candidates,
					self.world_size,
					self._decode_attn_tp_size(),
				)
				# Ensure at least some minimum
				max_num_seq_estimate = max(max_num_seq_estimate, 16)
				# Cap per-rank decode batch so the MoE buffer's mtp (= round_up(world_size *
				# this)) stays bounded — a large candidate pool must NOT inflate the padded
				# buffers (that re-OOMs init). The page-boundary admission also caps in-decode
				# at this value (decode.py max_rank_bsz), so the buffer never overflows.
				max_num_seq_estimate = min(max_num_seq_estimate, _MAX_DECODE_RANK_BSZ)

				self._load_decode_model(max_num_seq_estimate, self.comm)

				if torch.cuda.is_available():
					free_mem, total_mem = torch.cuda.mem_get_info(self.local_rank)
					allocated = torch.cuda.memory_allocated(self.local_rank) / 1e9
					logging.info(
						f"[HBM] Rank {self.rank} AFTER decode model: "
						f"free={free_mem/1e9:.2f}GB alloc={allocated:.2f}GB"
					)

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

				# Incremental write: submit sequences completed between decode rounds
				if global_completed:
					# Refresh rank-0's sequence replicas first: _report_completion
					# reads prompt_length/decoded_length from the local entry,
					# which is stale here for sequences owned by other ranks
					# (only the completion BIT was all-reduced above).
					self._sync_sequence_metadata(list(global_completed))
					self._submit_completed_to_incremental_writer(list(global_completed))
					# Gather decoded tokens from owning ranks before reporting
					# (each rank only writes decoded tokens for its own sequences)
					gathered_texts = self._gather_completed_tokens(list(global_completed))
					# ORDERING FIX: release resources BEFORE _report_completion
					# pops local_map entries. See matching fix in _page_boundary_fast
					# Phase 4.A and in the legacy decode path.
					completed_list = list(global_completed)
					my_completed = [u for u in completed_list if u in self._uuid_to_local_map]
					if my_completed:
						# Only release GPU pages for seqs that were actually GPU-allocated.
						# prefill_prepacked writes KV directly to host (never registers
						# with the GPU paged manager), so zero-tok-EOS prefill completions
						# are in _uuid_to_local_map but never in manager._sequences.
						# _sequences_with_gpu_kv is the source-of-truth tracking set
						# (added at :1619/:4904/:6191, discarded on release/eviction).
						gpu_allocated = [u for u in my_completed if u in self._sequences_with_gpu_kv]
						if gpu_allocated:
							self._release_gpu_kv_pages(self._get_local_indices_for_uuids(gpu_allocated))
						self._release_host_kv_pages_for_batch(my_completed)
					# All-ranks scalar cleanup
					for uuid in completed_list:
						seq = self.global_batch.get_sequence(uuid)
						if seq is not None:
							seq.gpu_pages_allocated = 0
							seq.host_pages_allocated = 0
							seq.host_token_capacity = 0
							self._sequences_with_gpu_kv.discard(uuid)
					# Report completions (pops local_map; runs LAST).
					# Guard: only report if status actually reached COMPLETED.
					# _sync_completion_status_tensor may detect eos_reached=True
					# for a PREFILLED sequence (stale from pre-eviction), but
					# PREFILLED→COMPLETED is an invalid transition. Without this
					# guard, _report_completion pops local_map for a sequence
					# whose status never changed, creating an orphan.
					for uuid in completed_list:
						seq = self.global_batch.get_sequence(uuid)
						if seq is not None and seq.status == SequenceStatus.COMPLETED:
							self._report_completion(uuid, gathered_text=gathered_texts.get(uuid))
						elif seq is not None:
							logging.warning(
								f"Rank {self.rank}: Skipping _report_completion for {uuid[:8]} "
								f"(status={seq.status.name}, expected COMPLETED). "
								f"Likely stale eos_reached from pre-eviction cycle."
							)

				if not decode_uuids:
					break
				
				for uuid in decode_uuids:
					seq = self.global_batch.get_sequence(uuid)
					prev_status = "ON_HOLD" if seq.had_initial_gpu_reservation else "PREFILLED"
					seq.log_event(SeqEvent.DECODE_START, self.rank,
						f"from={prev_status}")

				# ============ CRITICAL: Sync metadata before decode config ============
				# After decode→prefill→decode transitions, sequence metadata
				# (decoded_length, current_context_length, host_pages_allocated) may be
				# stale on non-owning ranks. The last sync was at the previous decode
				# group's final boundary. Sequences decoded additional tokens after that
				# boundary without cross-rank sync. Without this sync,
				# _allocate_gpu_kv_two_page_buffer may allocate too few GPU pages
				# (capped by stale host_pages_allocated), causing KV corruption at the
				# DECISION_INTERVAL boundary (~134-token truncation bug).
				if decode_uuids:
					self._sync_sequence_metadata(decode_uuids)

				# M2b: stamp the decode DP-group before local-index resolution so
				# the G ranks of a group resolve the SAME sequences (no-op, G==1).
				self._assign_decode_dp_groups(decode_uuids)
				local_decode_indices = self._get_local_indices_for_uuids(decode_uuids)
				global_decode_sequences = self._debug_sequences_for_decode_uuids(decode_uuids)
				AttnWrapperBase.batchgen_debug = self._active_batchgen_debug_for_sequences(
					global_decode_sequences
				)
				self._configure_glm5_dispatch_trace(global_decode_sequences)

				# B. Config Decode
				config_start = time.perf_counter()
				self._config_decoding_for_batch(decode_uuids, local_decode_indices)
				self._sync_decode_moe_rank_counts(
					local_decode_indices,
					reason="pre_decode_warmup",
				)
				self._bind_decode_attention_metadata_for_graph_config(local_decode_indices)
				config_decode_time += time.perf_counter() - config_start
				self._update_batch_status(decode_uuids, SequenceStatus.IN_DECODE)
				self._sync_sequence_metadata(decode_uuids)

				# Kimi-K3's segmented attention + resident-MXFP4 MoE graphs
				# must be captured after the real GPU KV manager/page table and
				# synchronized decode row counts exist, but before the first
				# measured decode forward.  The PSM keeps the model-specific
				# capture logic; this hook only supplies the worker-owned KV
				# manager.  Other models have no such method and retain their
				# existing warmup path below.
				prewarm_kimi_graphs = getattr(
					self.parallel_manager, "prewarm_decode_graphs", None
				)
				if prewarm_kimi_graphs is not None:
					prewarm_kimi_graphs(self._get_cuda_graph_gpu_manager())

				# CUDA Graph Warmup (configure-time, one-time for whole-model GLM).
				# Whole-model graph captures every configured bucket before decode;
				# segmented GLM-5 paths retain their existing partial-capture policy.
				from batchgen.models.glm.glm5.cuda_graph_policy import (
					should_warmup_cuda_graphs_before_decode,
				)

				has_queueing = self.global_batch.has_queueing()
				glm5_whole_graph_requested = (
					self._glm5_whole_model_graph_requested_for_current_batch()
					and "glm" in (getattr(self, "model_name", "") or "").lower()
				)
				# Phase C: layer / DSA / MoE / segmented graph modes are retired;
				# only the whole-model graph remains for GLM-5. Warmup triggers
				# when (a) the generic should_warmup helper says so, or (b) the
				# whole-model graph is requested but its bucket is missing.
				generic_cuda_graph_warmup_needed = should_warmup_cuda_graphs_before_decode(
					graph_manager_is_initialized=self._cuda_graph_manager is not None,
					global_batch_has_queueing=has_queueing,
					model_name=getattr(self, "model_name", None),
					enable_cuda_graph=getattr(self.args, "enable_cuda_graph", False),
				)
				if generic_cuda_graph_warmup_needed or (
					glm5_whole_graph_requested and self._cuda_graph_manager is None
				) or self._glm5_whole_model_graph_current_bucket_missing():
					if has_queueing:
						logging.info(
							f"Rank {self.rank}: warming GLM-5 CUDA graph with queued "
							"prefill work still pending when the requested graph path supports it"
						)
					self._warmup_cuda_graphs()

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

				# Poll for new admissions after each decode interval.
				# This ensures newly submitted batches are admitted to global_batch
				# so has_queueing() can detect them and trigger prefill.
				if self._admission_queue is not None:
					admitted = self._poll_admissions()
					if admitted and self.rank == 0:
						logging.info(f"[DECODE] Mid-cycle admission, total in batch: {len(self.global_batch)}")

				# Check if there are queued sequences waiting for prefill AND
				# host KV has enough free capacity to make prefill worthwhile.
				# Without the watermark check, decode oscillates: breaks every
				# DECISION_INTERVAL, puts all seqs ON_HOLD (~12s reload), prefills
				# only a handful of sequences, then resumes — destroying throughput.
				has_pending = self.global_batch.has_queueing() or (
					self.enable_host_kv_eviction and self.global_batch.has_evicted()
				)
				needs_prefill = has_pending and self._check_host_kv_watermark_trigger()
				if needs_prefill:
					if self.rank == 0:
						num_queued = len(self.global_batch.get_sequences_by_status(SequenceStatus.QUEUEING))
						num_evicted = len(self.global_batch.get_sequences_by_status(SequenceStatus.EVICTED)) if self.enable_host_kv_eviction else 0
						logging.info(f"[DECODE] Breaking for prefill (watermark) - {num_queued} queued, {num_evicted} evicted")
					in_decode_uuids = [
						u for u in decode_uuids
						if self.global_batch.get_sequence(u).status == SequenceStatus.IN_DECODE
					]
					# DIAG: Log ON_HOLD transition details
					if BATCHGEN_MULTI_BATCH_DIAG and self.rank == 0 and in_decode_uuids:
						sample = in_decode_uuids[:5]
						for u in sample:
							s = self.global_batch.get_sequence(u)
							logging.info(
								f"[MULTI_DIAG] ON_HOLD transition: {u[:8]} gid={s.global_idx} "
								f"decoded={s.decoded_length} ctx={s.current_context_length} "
								f"prompt={s.prompt_length} gpu_pages={s.gpu_pages_allocated}"
							)
						logging.info(f"[MULTI_DIAG] Putting {len(in_decode_uuids)} seqs ON_HOLD (decode_group={self._decode_group_idx})")
					if in_decode_uuids:
						self._put_sequences_on_hold(in_decode_uuids)
					self._decode_group_idx += 1
					break
		
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
		# Detokenize locally on each rank to avoid gathering large token tensors.
		# With 12K sequences × 1MB tensors = 12GB, all_gather_object OOMs.
		# Gathering strings (~KB each) instead reduces memory by ~100x.
		local_results = []
		# Sequences completed during prefill (C4) were reported and popped from
		# the local maps back then; their text was captured at that point.
		local_results.extend(getattr(self, '_prefill_completed_results', {}).items())
		for local_idx, uuid in self._local_to_uuid_map.items():
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				logging.warning(f"Rank {self.rank}: Sequence {uuid} not found in global_batch during result gathering")
				continue
			global_idx = seq.global_idx
			if local_idx not in self.query_book:
				logging.warning(f"Rank {self.rank}: query_book missing for local_idx={local_idx}, uuid={uuid[:8]}...")
				continue
			decoded_tokens = self.query_book[local_idx].decoded_tokens[:, :seq.decoded_length]
			decoded_str = self._decode_tokens_to_string(decoded_tokens)
			local_results.append((global_idx, decoded_str))

		all_results = [None] * self.world_size
		dist.all_gather_object(all_results, local_results)
		all_results = [item for sublist in all_results for item in sublist]
		result_dict = {global_idx: decoded_str for global_idx, decoded_str in all_results}

		if self.rank == 0:
			logging.info(f"Detokenization complete: {len(result_dict)} sequences (distributed across {self.world_size} ranks)")
			self._log_decode_timing()

		dist.barrier()
		self._batch_completed = True

		if self.rank == 0:
			return result_dict
		else:
			return {}

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
		eos_positions = [i for i, t in enumerate(tokens_list) if t in self.eos_token_ids and i >= min_tokens]

		if eos_positions:
			end_pos = eos_positions[0]
			if self.detokenization_include_special_tokens:
				end_pos += 1  # Include the stop token itself
		else:
			# No EOS found, use all non-padding tokens
			non_pad = [i for i, t in enumerate(tokens_list) if t != self.pad_token_id]
			end_pos = non_pad[-1] + 1 if non_pad else len(tokens_list)

		# Decode tokens up to end position
		return self.tokenizer.decode(tokens_list[:end_pos], skip_special_tokens=(not self.detokenization_include_special_tokens))

	# ============ Phase Configuration ============

	@staticmethod
	def _weight_copy_task_fingerprint(weight_copy_task) -> tuple:
		"""Immutable identity of an H2D weight-copy schedule.

		Both the module types and the per-type ORDER matter: the copy engine
		drains each type's list front to front and the consumer blocks on the
		head, so two tasks are interchangeable only if their lists are equal.
		"""
		return tuple(
			(str(module_type), tuple(str(name) for name in names))
			for module_type, names in sorted((weight_copy_task or {}).items())
		)

	def _install_prefill_weight_copy_pipeline(self, k3_prefill_profile: bool) -> None:
		"""Point the H2D copy engine at the prefill weight-copy schedule.

		Requires ``self.weight_copy_task`` from the configure_prefill that just
		ran. Shared by the pre-readiness Kimi-K3 startup and by every per-batch
		prefill config, so first-install and transport-specific reentry rules live
		in one place no matter which caller runs first.
		"""
		# Host-RDMA drives one free-running remote-daemon schedule and erases each
		# generation it releases, so that transport must preserve its queue cursor
		# and prefetched GPU leases. Hierarchical GDR is different: source ranks
		# read their local compact store directly and non-sources have an empty H2D
		# task. Start that transport at the schedule boundary on every admission;
		# preserving an arbitrary partially filled ring across resident decode can
		# leave a full-batch consumer holding the published leases while it waits
		# for a slot that can no longer be produced.
		streamed_sp8 = (
			hasattr(self.parallel_manager, "prefill_uses_streamed_sp8")
			and self.parallel_manager.prefill_uses_streamed_sp8()
		)
		sp8_reentry = streamed_sp8 and self._streamed_sp8_h2d_installed
		reseed_reentry = (
			sp8_reentry
			and hasattr(
				self.parallel_manager,
				"streamed_sp8_reseeds_h2d_on_reentry",
			)
			and self.parallel_manager.streamed_sp8_reseeds_h2d_on_reentry()
		)
		self.core_engine.stop_h2d_worker()
		fingerprint = self._weight_copy_task_fingerprint(self.weight_copy_task)
		if sp8_reentry:
			# configure_prefill rebuilt the task from scratch. Both a preserved
			# cursor and a fresh schedule boundary require the same immutable task.
			if fingerprint != self._streamed_sp8_weight_copy_fingerprint:
					raise RuntimeError(
					f"Rank {self.rank}: streamed-SP8 weight-copy schedule "
					"changed after the H2D pipeline was installed; the "
					"installed pipeline no longer matches the rebuilt task"
				)
		if not sp8_reentry or reseed_reentry:
			self.core_engine.clear_weight_copy_queue()
			self.core_engine.reset_prefill_buffer()
		self.core_engine.reset_weight_stream_profile(k3_prefill_profile)
		if streamed_sp8 or k3_prefill_profile:
			from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert import (
				KimiK3MXFP4ExpertWrapper,
			)
			KimiK3MXFP4ExpertWrapper.reset_prefill_profile(k3_prefill_profile)
			from batchgen.moe.streamed_sp8_mxfp4 import (
				StreamedSP8MXFP4MoELayer,
			)
			StreamedSP8MXFP4MoELayer.reset_prefill_profile(k3_prefill_profile)
		if not sp8_reentry or reseed_reentry:
			self.core_engine.set_weight_copy_queue(self.weight_copy_task)
			if any(self.weight_copy_task.values()):
				self.core_engine.start_h2d_worker()
		elif any(self.weight_copy_task.values()):
			self.core_engine.start_h2d_worker()
		self._streamed_sp8_h2d_installed = streamed_sp8
		self._streamed_sp8_weight_copy_fingerprint = (
			fingerprint if streamed_sp8 else None
		)

	def _config_prefill_for_batch(self, prefill_uuids: List[str]) -> None:
		"""Configure prefill phase for a batch of sequences."""
		start_time = time.perf_counter()
		prefill_sequences = []
		for uuid in prefill_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is not None:
				prefill_sequences.append(seq)
		prefill_debug = (
			self._active_batchgen_debug_for_sequences(prefill_sequences) or {}
		)
		k3_prefill_profile = self._debug_flag_enabled(
			prefill_debug.get("k3_prefill_profile")
		)
		AttnWrapperBase.batchgen_debug = prefill_debug or None
		_default_k3_prefill_mode = "streamed"
		if hasattr(self.parallel_manager, "default_prefill_moe_mode"):
			_default_k3_prefill_mode = (
				self.parallel_manager.default_prefill_moe_mode()
			)
		k3_prefill_moe_mode = prefill_debug.get(
			"k3_prefill_moe_mode", _default_k3_prefill_mode
		)
		reuse_startup_prefill = (
			self._k3_startup_prefill_ready
			and k3_prefill_moe_mode == self._k3_startup_prefill_mode
			and not k3_prefill_profile
		)
		if hasattr(self.parallel_manager, "set_prefill_moe_mode"):
			self.parallel_manager.set_prefill_moe_mode(k3_prefill_moe_mode)
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

		# Resident TP groups must be known before sizing the reusable global
		# FP32 MoE output. Fresh admissions are already grouped; this also
		# restores groups for evicted re-entries before the normal idempotent
		# assignment later in this method.
		self._assign_decode_dp_groups(prefill_uuids)

		# CRITICAL: Deep free decode model memory BEFORE configuring prefill (Bug Fix 7)
		# This mirrors the cleanup done in _load_decode_model() for prefill→decode transitions
		# Without this, decode model (~92 GB) stays in memory when prefill model loads → OOM
		if reuse_startup_prefill:
			# EXCEPT on the first admission after the Kimi-K3 startup pass: no
			# decode model has run yet, and this prefill phase — model,
			# streamed-SP8 buffers, installed H2D schedule — is the one startup
			# built. deep_free_model_memory() would release those buffers and
			# strand the weight daemon's preserved cursor. Consumed here, so every
			# later prefill (which does follow a decode phase) frees normally.
			self._k3_startup_prefill_ready = False
			logging.info("Reusing Kimi-K3 startup-prepared prefill model...")
		else:
			# Consume a startup handoff that the request cannot reuse (for example,
			# a diagnostic mode/profile override) before rebuilding that phase.
			self._k3_startup_prefill_ready = False
			logging.info("Deep freeing model memory before prefill config...")
			self.deep_free_model_memory()

		# CRITICAL: Destroy GPU KV cache BEFORE configure_prefill (Bug Fix 7.2)
		# The GPU KV cache holds ~20-30GB that must be freed before loading prefill model
		# Previously this was called AFTER configure_prefill() which caused OOM
		self._destroy_gpu_paged_kv_cache()
		if k3_prefill_profile and torch.cuda.is_available():
			torch.cuda.reset_peak_memory_stats(self.local_rank)

		if (
			hasattr(self.parallel_manager, "prefill_uses_resident_ep")
			and self.parallel_manager.prefill_uses_resident_ep()
		):
			local_lengths = [
				int(seq.prompt_length)
				for seq in prefill_sequences
				if self._owns_local_sequence(seq)
			]
			token_cap = (
				self.engine_config.Module_Batching_Config
				.prefill_micro_batch_token_cap
			)
			use_l2 = os.environ.get("BATCHGEN_L2_BALANCE", "1") == "1"
			predicted_batches, _ = build_prefill_micro_batches(
				local_lengths,
				token_cap,
				l2_balance=use_l2,
			)
			local_max_tokens = max(
				(
					sum(local_lengths[start:end])
					for start, end in predicted_batches
				),
				default=0,
			)
			moe_ntp = self._sync_prefill_moe_rank_counts(
				local_max_tokens,
				reason="prefill_output_preallocate",
			)
			self.parallel_manager.prepare_resident_ep_prefill_output(
				self.world_size * moe_ntp
			)

		if torch.cuda.is_available():
			free_mem, total_mem = torch.cuda.mem_get_info(self.local_rank)
			allocated = torch.cuda.memory_allocated(self.local_rank) / 1e9
			reserved = torch.cuda.memory_reserved(self.local_rank) / 1e9
			logging.info(
				f"[HBM] Rank {self.rank} BEFORE configure_prefill: "
				f"free={free_mem/1e9:.2f}GB alloc={allocated:.2f}GB rsv={reserved:.2f}GB"
			)

		# STEP 1: Configure model for prefill. The normal first K3 admission
		# inherits the exact phase installed before readiness; do not re-enter
		# configure_prefill or stop/restart its H2D pipeline lazily here.
		if not reuse_startup_prefill:
			# Hand the NCCL communicator to managers that need it during prefill
			# (e.g. Kimi-Linear MoE EP all-reduce); harmless no-op for others.
			if hasattr(self.parallel_manager, "set_comm"):
				self.parallel_manager.set_comm(self.comm)
			self.model, self.weight_copy_task = self.parallel_manager.configure_prefill()
			self.set_phase("prefill")

		if torch.cuda.is_available():
			torch.cuda.synchronize(self.torch_device)
			free_mem, total_mem = torch.cuda.mem_get_info(self.local_rank)
			allocated = torch.cuda.memory_allocated(self.local_rank) / 1e9
			logging.info(
				f"[HBM] Rank {self.rank} AFTER configure_prefill: "
				f"free={free_mem/1e9:.2f}GB alloc={allocated:.2f}GB"
			)

		if not reuse_startup_prefill:
			self._install_prefill_weight_copy_pipeline(k3_prefill_profile)

		# NOTE: _destroy_gpu_paged_kv_cache() moved before configure_prefill() (Bug Fix 7.2)

		# STEP 3: Prepare evicted sequences for re-entry (before host KV allocation)
		#
		# Split into two loops:
		#   (a) All-ranks scalar metadata update (runs on every rank using
		#       fields already synchronized via Phase 4.C of the eviction
		#       boundary and via _sync_sequence_metadata).
		#   (b) Owner-only tensor buffer setup (only the owning rank has
		#       the QueryBookBufferPool slot for this sequence).
		#
		# The previous single-loop version ran both steps gated on
		# evicted_token_ids — which is an owner-only tensor — so non-owning
		# ranks silently skipped the scalar updates and held stale values
		# for decoded_length / reentry_decoded_baseline / max_decode_length
		# until the next _sync_sequence_metadata call.

		# (a) All-ranks scalar metadata update for re-entering sequences.
		for uuid in prefill_uuids:
			seq = self.global_batch.get_sequence(uuid)
			# total_decoded_before_eviction > 0 identifies sequences that have
			# been evicted at least once and are now re-entering. This field is
			# synced across ranks, unlike evicted_token_ids which is owner-only.
			if seq.total_decoded_before_eviction == 0:
				continue

			# seq.prompt_length and seq.current_context_length were already
			# updated to the new reconstructed length by Phase 4.C of the
			# eviction boundary and synced to all ranks.

			# Baseline = accumulated historical output length carried forward
			# into decoded_tokens. With the Phase 4.C cascade fix, this is
			# exactly (prompt_length - original_prompt_length) = sum of new
			# decoded counts across all past cycles.
			baseline_candidate = seq.prompt_length - seq.original_prompt_length
			n_old = min(baseline_candidate, self.max_decoding_length)
			if n_old < 0:
				n_old = 0
			seq.decoded_length = n_old
			seq.reentry_decoded_baseline = n_old

			# decoded_length is cumulative across eviction/re-entry cycles, so
			# max_decode_length must remain the absolute per-request completion
			# cap. Compute remaining budget as
			# original_max_decode_length - decoded_length at call sites instead
			# of storing a relative value here.
			seq.max_decode_length = seq.original_max_decode_length

			# Reset completion flags — the sequence may have hit EOS in its
			# previous decode cycle before being evicted. Without this reset,
			# _sync_completion_status_tensor falsely detects the re-entering
			# sequence as completed (stale eos_reached=True), calls
			# _report_completion (popping local_map), but the PREFILLED→COMPLETED
			# status transition fails (invalid), leaving a zombie: PREFILLED
			# status with no local_map entry, invisible to the boundary load
			# mechanism (which iterates local_map), stuck for the entire decode
			# cycle until _prepare_decode_batch picks it up → CRITICAL error.
			seq.eos_reached = False
			if hasattr(seq, '_rep_detected'):
				seq._rep_detected = False

		# (b) Owner-only tensor buffer setup. Also clears seq.evicted_token_ids.
		for uuid in prefill_uuids:
			seq = self.global_batch.get_sequence(uuid)
			# Gate on evicted_token_ids (owner-only tensor); non-owners fall
			# through here because their copy is always None.
			if seq.evicted_token_ids is None:
				continue

			evicted_ids = seq.evicted_token_ids  # 1D tensor
			new_prompt_len = len(evicted_ids)
			prev_decoded = seq.total_decoded_before_eviction
			seq.log_event(SeqEvent.REENTRY_START, self.rank,
				f"new_prompt_len={new_prompt_len}, prev_decoded={prev_decoded}")

			# Sanity: owner-side new_prompt_len must match scalar math done
			# in loop (a). Mismatches indicate a drift between the tensor
			# built by Phase 4.C and the scalar accounting.
			if new_prompt_len != seq.prompt_length:
				logging.error(
					f"Rank {self.rank}: re-entry prep length mismatch for "
					f"{uuid[:8]}: tensor={new_prompt_len}, scalar="
					f"{seq.prompt_length}. Trusting tensor."
				)
				seq.prompt_length = new_prompt_len
				seq.current_context_length = new_prompt_len

			# Rebuild input_ids with new prompt — reuse buffer pool slot
			seq_extended_size = seq.kv_token_budget
			slot = seq._buffer_slot
			if slot < 0:
				# A negative index silently rewrites the LAST row of the pool,
				# which is another sequence's prompt (and, now that input_ids is
				# node-shared, every rank's copy of it).
				raise QueryBookPoolCapacityError(
					f"Rank {self.rank}: re-entry of {uuid[:8]} has no buffer slot "
					f"(_buffer_slot={slot}); slot assignment has diverged"
				)
			self._buffer_pool.input_ids_buffer[slot, :] = 0
			self._buffer_pool.input_ids_buffer[slot, :new_prompt_len] = evicted_ids
			seq.input_ids = self._buffer_pool.get_input_ids_view(slot, seq_extended_size)

			# Pre-fill decoded_tokens with previously decoded tokens (Q1/Q2)
			# so the final decoded_tokens contains the COMPLETE response.
			self._buffer_pool.decoded_tokens_buffer[slot, :] = self._buffer_pool.pad_token_id
			seq.decoded_tokens = self._buffer_pool.get_decoded_tokens_view(slot)
			if prev_decoded > 0:
				old_decoded = evicted_ids[seq.original_prompt_length:]
				n_old = min(len(old_decoded), self.max_decoding_length)
				seq.decoded_tokens[0, :n_old] = old_decoded[:n_old]
				# decoded_length and reentry_decoded_baseline are already set
				# by loop (a); setting them here is redundant but harmless and
				# acts as a local invariant check.
				if seq.decoded_length != n_old:
					logging.error(
						f"Rank {self.rank}: re-entry decoded_length mismatch for "
						f"{uuid[:8]}: tensor_n_old={n_old}, scalar="
						f"{seq.decoded_length}. Trusting tensor."
					)
					seq.decoded_length = n_old
					seq.reentry_decoded_baseline = n_old

			# Clear eviction state
			seq.evicted_token_ids = None

			# Recreate query_book entry for this rank's evicted sequences (Q4)
			if seq.assigned_rank == self.rank and uuid in self._uuid_to_local_map:
				local_idx = self._uuid_to_local_map[uuid]
				self.query_book[local_idx] = make_query_book_entry(seq)

			logging.info(
				f"Rank {self.rank}: Prepared EVICTED seq {uuid[:8]} for re-entry: "
				f"new_prompt={new_prompt_len}, prev_decoded={prev_decoded}, "
				f"remaining_decode={seq.max_decode_length}, kv_budget={seq.kv_token_budget}"
			)

		# STEP 3.5 (Option 1, CORE): (re)assign the serve-group before binding.
		# Idempotent for fresh admits (already grouped at admission); RE-groups
		# evicted re-entries (whose decode_dp_group was cleared on eviction) so
		# the group predicate below binds them on all G ranks. No-op for G==1.
		self._assign_decode_dp_groups(prefill_uuids)

		# STEP 4: Allocate host KV pages for sequences this rank serves.
		# Ownership: G==1 -> the single assigned_rank; G>1 (Option 1) -> ALL G
		# ranks of the sequence's serve-group, so the group holds the sequence's
		# replicated MLA KV and head-sharded KDA state from prefill onward.
		# Check by _owns_local_sequence, NOT _uuid_to_local_map (which may not
		# have new sequences yet).
		my_prefill_uuids = []
		for uuid in prefill_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if self._owns_local_sequence(seq):
				my_prefill_uuids.append(uuid)
				# Add to local maps if not already present (for new sequences)
				if uuid not in self._uuid_to_local_map:
					new_local_idx = self._bind_local_sequence_to_query_book(uuid)
					logging.debug(
						f"Rank {self.rank}: Added new sequence {uuid[:8]}... to local maps "
						f"(local_idx={new_local_idx})"
					)

		if my_prefill_uuids:
			global_sequence_ids = []
			sequence_tokens = []
			chunk_size = self._get_effective_chunk_size()

			for uuid in my_prefill_uuids:
				seq = self.global_batch.get_sequence(uuid)
				global_sequence_ids.append(seq.global_idx)
				# Dynamic reservation: allocate prompt + chunk_size, not full budget.
				# Must also cover the GPU initial load which needs
				# ceil((prompt+1)/PAGE_SIZE) + INITIAL_GPU_PAGE_BUFFER pages.
				# The +1 accounts for the first decoded token produced during prefill
				# (current_context_length = prompt_length + 1 after prefill).
				from batchgen.sequence import INITIAL_GPU_PAGE_BUFFER
				post_prefill_length = seq.prompt_length + 1  # prefill produces 1 decode token
				gpu_initial_pages = math.ceil(post_prefill_length / seq.PAGE_SIZE) + INITIAL_GPU_PAGE_BUFFER
				gpu_initial_tokens = gpu_initial_pages * seq.PAGE_SIZE
				initial_capacity = max(seq.prompt_length + chunk_size, gpu_initial_tokens)
				initial_capacity = min(initial_capacity, seq.kv_token_budget)
				seq.host_pages_allocated = math.ceil(initial_capacity / seq.PAGE_SIZE)
				seq.host_token_capacity = seq.host_pages_allocated * seq.PAGE_SIZE
				sequence_tokens.append(seq.host_token_capacity)

			# Safety assertion: log if selection over-admitted. This should not
			# happen after the EVICTED-length fix in _prepare_prefill_batch —
			# if it fires, there's another selection bug to investigate.
			kv_stats = self.core_engine.host_paged_kv_worker_view.get_stats()
			total_pages_needed = sum(math.ceil(t / seq.PAGE_SIZE) for t in sequence_tokens)
			if total_pages_needed > kv_stats.num_free_pages:
				# Log per-sequence breakdown to help diagnose the selection bug.
				seq_details = []
				for gid, tokens in list(zip(global_sequence_ids, sequence_tokens))[:10]:
					s = self.global_batch.get_sequence(
						next(u for u in my_prefill_uuids if self.global_batch.get_sequence(u).global_idx == gid)
					)
					seq_details.append(
						f"gid={gid} prompt_len={s.prompt_length} "
						f"was_evicted={s.total_decoded_before_eviction > 0} "
						f"tokens={tokens}"
					)
				logging.error(
					f"Rank {self.rank}: Host KV OVER-ADMISSION: need {total_pages_needed} pages, "
					f"have {kv_stats.num_free_pages}. Selection should have prevented this. "
					f"First 10 seqs: {seq_details}"
				)

			logging.debug(
				f"Rank {self.rank}: Registering {len(global_sequence_ids)} sequences for host KV "
				f"(chunk_size={chunk_size})"
			)

			self.core_engine.host_paged_kv_worker_view.register_sequences(global_sequence_ids)
			self.core_engine.host_paged_kv_worker_view.allocate_pages_for_sequences(
				list(zip(global_sequence_ids, sequence_tokens))
			)
			# DSA: mirror registration on auxiliary host KV
			aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)
			if aux_view is not None:
				aux_view.register_sequences(global_sequence_ids)
				aux_view.allocate_pages_for_sequences(
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
		# Remember the per-rank batch the MoE buffer was sized for; the decode admission caps
		# in-decode at this value so the padded buffer (mtp = round_up(world_size*max_num_seq))
		# never overflows.
		self._decode_padding_bsz = max_num_seq
		self.set_phase("decode")
		self.core_engine.stop_h2d_worker()
		self.core_engine.clear_kv_copy_queue()
		# Resident decode does not stream routed experts. Keep the installed
		# streamed-SP8 state through this transition: host-RDMA needs its daemon
		# cursor, while hierarchical GDR resets its local queue/ring at the next
		# prefill boundary. reset_decoding_buffer() must not resize those slots to
		# the decode layout in either case.
		if not self._streamed_sp8_h2d_installed:
			self.core_engine.clear_weight_copy_queue()
			self.core_engine.reset_decoding_buffer()

		# Only start H2D worker if there are experts to offload
		if self.weight_copy_task.get("routed_expert"):
			if self._streamed_sp8_h2d_installed:
				raise RuntimeError(
					f"Rank {self.rank}: streamed decode cannot replace an "
					"installed streamed-SP8 prefill pipeline; use resident-EP "
					"decode so the distributed weight schedule stays aligned"
				)
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
			expected_ctx = seq.original_prompt_length + seq.decoded_length
			
			# Repair if mismatched
			if seq.current_context_length != expected_ctx:
				old_ctx = seq.current_context_length
				seq.log_event(SeqEvent.CTX_REPAIR, self.rank,
					f"config_decode old={old_ctx}, new={expected_ctx}")
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

		for uuid in decode_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				raise RuntimeError(
					f"Rank {self.rank}: decode uuid {uuid[:8]} missing at _config_decoding_for_batch entry"
				)
			seq.validate_metadata(f"rank {self.rank} _config_decoding_for_batch/entry")
		
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
			alloc_ok = self._allocate_gpu_kv_two_page_buffer(local_decode_indices, load_from_host=True)
			if alloc_ok:
				# _allocate_gpu_kv_two_page_buffer already sets gpu_pages_allocated,
				# mark_initial_gpu_reservation_done, and _sequences_with_gpu_kv.
				# Keep these for safety / idempotence.
				for local_idx in local_decode_indices:
					uuid = self._local_to_uuid_map[local_idx]
					seq = self.global_batch.get_sequence(uuid)
					seq.gpu_pages_allocated = seq.get_gpu_pages_for_two_page_buffer()
					# Mark initial reservation done
					seq.mark_initial_gpu_reservation_done()
					self._sequences_with_gpu_kv.add(uuid)
			else:
				raise RuntimeError(
					f"Rank {self.rank}: GPU KV allocation failed for "
					f"{len(local_decode_indices)} locally owned sequences; "
					"decode admission exceeded the available replica capacity"
				)
		
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

		# Host KV is ONE per-node SHARED shm region (batchgen_host_kv_cache) keyed
		# by global_idx. Under Option 1 (G>1) all G ranks of a group hold the uuid
		# in _uuid_to_local_map, so releasing on every rank double-frees the single
		# shared entry -- the first releaser tombstones it and the rest raise
		# "Sequence ID ... not found during release". Release the shared entry on
		# EXACTLY the group leader (_owns_host_kv). G==1: host_release_uuids ==
		# my_uuids, so the validated single-owner path is byte-identical.
		host_release_uuids = [
			uuid for uuid in my_uuids
			if self._owns_host_kv(self.global_batch.get_sequence(uuid))
		]

		if host_release_uuids:
			global_sequence_ids = [
				self.global_batch.get_sequence(uuid).global_idx
				for uuid in host_release_uuids
			]

			logging.debug(f"Rank {self.rank}: Releasing host KV pages for global_idx: {global_sequence_ids}")

			# NOTE: GPU KV pages should already be released by caller
			# Do NOT call _release_gpu_kv_pages here to avoid double-free

			# Release host KV pages
			# NOTE: release_sequence_pages already calls unregister_sequences internally,
			# so we don't need to call unregister_sequences separately
			worker_view.release_sequence_pages(global_sequence_ids)
			# DSA: release auxiliary host KV pages too
			aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)
			if aux_view is not None:
				aux_view.release_sequence_pages(global_sequence_ids)

		# GPU page table is PER-RANK (GPU KV is replicated across the group's G
		# ranks under Option 1), so EVERY rank that held these sequences rebuilds
		# its own remaining page table -- keyed on my_uuids, not the leader subset.
		if my_uuids:
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
		# Bind AttnWrapperBase.host_paged_kv_worker_view_aux BEFORE the decoder
		# loop. Without this binding, GLM-5's prefill indexer-K offload at
		# wrappers.py:_offload_prepacked_indexer_kv silently early-returns
		# (host_paged_kv_worker_view_aux is None), so the aux cache is never
		# populated for prompt tokens and any later decode past 2048 tokens
		# reads unwritten aux pages.
		# Prefill offloads KV directly to host via host_paged_kv_worker_view_aux;
		# it does NOT use the GPU paged KV manager. Binding host_*_aux here
		# ensures `_offload_prepacked_indexer_kv` actually pushes indexer K to
		# the host aux cache instead of early-returning on a None view.
		AttnWrapperBase.host_paged_kv_worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		AttnWrapperBase.host_paged_kv_worker_view_aux = getattr(self, "host_paged_kv_worker_view_aux", None)

		if "deepseek" in self.model_config.model_type:
			self.model.model._use_flash_attention_2 = False

		# Dynamic padding: find max length within THIS batch, not global max
		# This is critical for long-tailed distributions
		batch_seq_lengths = [
			self.query_book[query_idx].encoded["input_ids"].shape[1]
			for query_idx in batch
		]
		batch_max_len = max(batch_seq_lengths)

		# Pad each sequence to batch_max_len and construct attention masks on-the-fly
		padded_input_ids = []
		padded_attention_masks = []
		for query_idx in batch:
			seq_input_ids = self.query_book[query_idx].encoded["input_ids"]
			uuid = self._local_to_uuid_map[query_idx]
			seq = self.global_batch.get_sequence(uuid)
			prompt_len = seq.prompt_length
			seq_len = seq_input_ids.shape[1]

			# Construct attention mask from prompt_length (1s for valid tokens, 0s for padding)
			seq_attention_mask = torch.zeros((1, seq_len), dtype=torch.int64)
			seq_attention_mask[0, :prompt_len] = 1

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
				cur_batch_sequences = [
					self.global_batch.get_sequence(self._local_to_uuid_map[local_idx])
					for local_idx in cur_batch_local
				]
				new_tokens = self._select_tokens(outputs.logits[:, -1, :], cur_batch_sequences)
				output_tokens.append(new_tokens)

		new_tokens = torch.cat(output_tokens, dim=0)

		# Update sequence state after prefill
		# For evicted re-entry: first new token goes at decoded_length offset (not 0)
		# For fresh sequences: decoded_length is 0, so offset is 0 (same as before)
		new_tokens_cpu = new_tokens.cpu()
		for i, local_idx in enumerate(batch):
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			# Write token at correct offset (handles both fresh and re-entered sequences)
			token_pos = seq.decoded_length  # 0 for fresh, prev_decoded for re-entry
			self.query_book[local_idx].decoded_tokens[:, token_pos] = new_tokens_cpu[i]
			seq.decoded_length = token_pos + 1
			seq.current_context_length = seq.original_prompt_length + seq.decoded_length

			# MODIFIED: Check for EOS respecting ignore_eos flag
			if self._should_stop_at_eos(new_tokens_cpu[i].item()):
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
		# Bind AttnWrapperBase.host_paged_kv_worker_view_aux BEFORE the decoder
		# loop. Without this binding, GLM-5's prefill indexer-K offload at
		# wrappers.py:_offload_prepacked_indexer_kv silently early-returns
		# (host_paged_kv_worker_view_aux is None), so the aux cache is never
		# populated for prompt tokens and any later decode past 2048 tokens
		# reads unwritten aux pages.
		# Prefill offloads KV directly to host via host_paged_kv_worker_view_aux;
		# it does NOT use the GPU paged KV manager. Binding host_*_aux here
		# ensures `_offload_prepacked_indexer_kv` actually pushes indexer K to
		# the host aux cache instead of early-returning on a None view.
		AttnWrapperBase.host_paged_kv_worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		AttnWrapperBase.host_paged_kv_worker_view_aux = getattr(self, "host_paged_kv_worker_view_aux", None)

		if "deepseek" in self.model_config.model_type:
			self.model.model._use_flash_attention_2 = False

		# Collect input_ids and attention_masks as lists for prepacking
		input_ids_list = []
		attention_mask_list = []
		seq_lengths = []

		for query_idx in batch:
			uuid = self._local_to_uuid_map[query_idx]
			seq = self.global_batch.get_sequence(uuid)
			query_entry = self.query_book[query_idx]
			encoded = query_entry.encoded["input_ids"]
			if encoded.data_ptr() != seq.input_ids.data_ptr():
				raise RuntimeError(
					f"Rank {self.rank}: stale query_book input_ids binding for "
					f"local_idx={query_idx} uuid={uuid[:8]} "
					f"(query_book_ptr={encoded.data_ptr():#x}, seq_ptr={seq.input_ids.data_ptr():#x})"
				)
			if query_entry.decoded_tokens.data_ptr() != seq.decoded_tokens.data_ptr():
				raise RuntimeError(
					f"Rank {self.rank}: stale query_book decoded_tokens binding for "
					f"local_idx={query_idx} uuid={uuid[:8]} "
					f"(query_book_ptr={query_entry.decoded_tokens.data_ptr():#x}, "
					f"seq_ptr={seq.decoded_tokens.data_ptr():#x})"
				)
			# NO truncation: every prompt is tokenized to its OWN length.
			# An earlier `[:, :self.max_input_length]` slice silently dropped
			# the tail of long LongBench prompts when max_input_length was
			# carried over from a smaller earlier admit batch, causing the
			# model to "continue" mid-sentence instead of answering. Bind
			# everything to seq.prompt_length directly.
			L = seq.prompt_length
			assert encoded.size(-1) >= L, (
				f"encoded prompt length {encoded.size(-1)} < seq.prompt_length {L} "
				f"for query_idx={query_idx} uuid={uuid[:8]}"
			)
			input_ids = encoded[:, :L]
			seq_lengths.append(L)

			# Per-seq mask marks the L valid positions for the prepacker.
			# Causal attention is enforced by FA varlen + cu_seqlens.
			attention_mask = torch.zeros_like(input_ids, dtype=torch.int64)
			attention_mask[0, :L] = 1

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

		# Create micro-batches bounded by token count, optionally also by sum(L^2)
		# so the per-microbatch attention work (which is O(L^2)) doesn't pile up
		# on one micro-batch when a single very long sequence is present.
		# The planner's token cap is a bound between sequence boundaries.  A
		# single K3 prompt may itself be much larger than that cap, so the
		# ordinary greedy planner cannot split it and would still co-reside two
		# 262K-token sequences in one decoder pass.  That pass allocates the
		# block-attention-residual scratch for *all* rows and exceeded H20 HBM
		# before the first streamed-SP8 expert layer.  K3's KDA/MLA state is
		# persistent per sequence, so keeping each long sequence in its own
		# prepack pass preserves the state contract; it is not a token-axis
		# split pretending to be a resumable model forward.
		micro_batches, l2_cap, _use_single_sequence_mb = (
			self._plan_prefill_micro_batches(seq_lengths_list)
		)
		total_tokens_all = sum(seq_lengths_list)
		if (
			hasattr(self.parallel_manager, "prefill_uses_resident_ep")
			and self.parallel_manager.prefill_uses_resident_ep()
		):
			local_micro_batches = torch.tensor(
				[len(micro_batches)],
				dtype=torch.int64,
				device=self.torch_device,
			)
			all_micro_batches = torch.empty(
				(self.world_size,),
				dtype=torch.int64,
				device=self.torch_device,
			)
			dist.all_gather_into_tensor(
				all_micro_batches, local_micro_batches
			)
			micro_batch_counts = [
				int(value) for value in all_micro_batches.cpu().tolist()
			]
			if len(set(micro_batch_counts)) != 1:
				raise RuntimeError(
					"resident-EP prefill requires the same microbatch count "
					f"on all ranks; got {micro_batch_counts}"
				)

		if self.rank == 0:
			logging.info(
				f"Prepacked prefill: {len(micro_batches)} micro batches, "
				f"{total_tokens_all:,} total tokens, max {MAX_TOKENS_PER_MICRO_BATCH:,} tokens/batch"
				+ (", single-sequence long-prompt guard" if _use_single_sequence_mb else "")
				+ (f", l2_cap={l2_cap:,}" if l2_cap > 0 else "")
			)

		output_tokens = []
		prefill_sequences = [
			self.global_batch.get_sequence(self._local_to_uuid_map[local_idx])
			for local_idx in batch
		]
		prefill_debug = (
			self._active_batchgen_debug_for_sequences(prefill_sequences) or {}
		)
		_k3_profile_enabled = self._debug_flag_enabled(
			prefill_debug.get("k3_prefill_profile")
		)
		_k3_profile_logits = []

		# Pure forward wall time: started here so configure_prefill (already
		# reported separately as `Config completed`) is NEVER folded in.
		_prefill_forward_t0 = time.perf_counter()
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
				if (
					hasattr(self.parallel_manager, "prefill_uses_resident_ep")
					and self.parallel_manager.prefill_uses_resident_ep()
				):
					self._sync_prefill_moe_rank_counts(
						int(batch_input_ids_flat.numel()),
						reason=f"prefill_microbatch_{batch_idx}",
					)

				batch_local_indices = batch[seq_start:seq_end]
				local_to_global_seq_id_map = {}
				for local_idx in batch_local_indices:
					uuid = self._local_to_uuid_map.get(local_idx)
					if uuid is None:
						raise RuntimeError(
							f"Rank {self.rank}: missing UUID for prefill local_idx={local_idx}"
						)
					seq = self.global_batch.get_sequence(uuid)
					if seq is None:
						raise RuntimeError(
							f"Rank {self.rank}: missing SequenceEntry for prefill uuid={uuid[:8]}"
						)
					local_to_global_seq_id_map[local_idx] = seq.global_idx

				batch_spans = build_prefill_sequence_spans(
					batch_local_indices,
					batch_seq_lengths,
					self._local_to_uuid_map,
					local_to_global_seq_id_map,
				)
				batch_cu_seqlens = torch.tensor(
					prefill_sequence_spans_to_cu_seqlens(batch_spans),
					dtype=torch.int32,
					device=self.torch_device,
				)
				batch_max_seqlen = max(batch_seq_lengths)

				# Set up Attn_Wrapper for this micro-batch.
				# These class attrs are the per-step worker->model contract read
				# by attention wrappers; see semantics in
				#   batchgen-context/architecture/PSM_WORKER_CONTRACT.md (§2)
				Attn_Wrapper.prepack_mode = True
				Attn_Wrapper.prepack_cu_seqlens = batch_cu_seqlens
				Attn_Wrapper.prepack_max_seqlen = batch_max_seqlen
				Attn_Wrapper.prepack_num_sequences = batch_num_seqs
				Attn_Wrapper.prepack_seq_lengths = batch_seq_lengths
				Attn_Wrapper.position_ids = batch_position_ids_flat
				Attn_Wrapper.cur_batch = prefill_sequence_spans_to_global_seq_ids(batch_spans)

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
				# unsqueeze is a VIEW, so `hidden_states` already keeps the
				# embedding storage alive for exactly as long as layer 0 needs
				# it. Keeping `inputs_embeds` bound as well pins that storage
				# for the WHOLE stack instead -- 1.75 GiB dead across layers
				# 1-92 at S=131,072 / H=7168 / bf16
				# (batchgen_design/model_support/kimi_k3/
				#  PREFILL_MEMORY_AUDIT.md section 7, fix 4).
				del inputs_embeds

				# Block Attention Residuals (Kimi-K3): the depth-mix REPLACES the
				# classic residual body, so the per-layer state has to be carried
				# here. Without it every layer sees a zero-width residual and the
				# model runs happily while computing something that is not K3 --
				# wrong text, no error. Mirrors KimiLinearModel.forward
				# (kimi_linear/model.py:880-910), which is the eager reference.
				use_attn_res = getattr(self.model.model, "use_attn_residuals", False)
				block_residual = None
				if use_attn_res:
					# Zero-column view of a buffer preallocated for ALL the
					# stack's block boundaries, so the per-boundary `cat` never
					# holds the (S,nb,H) and (S,nb+1,H) tensors at once (12.25
					# GiB at K3's last boundary; PREFILL_MEMORY_AUDIT.md fix 3).
					# `block_residual = None` above is load-bearing: it drops
					# the previous micro-batch's view before the next buffer is
					# allocated.
					block_residual = self.model.model._new_block_residual(hidden_states)

				# DEBUG (env-gated): per-layer HBM watermarks. Cheap (allocator
				# bookkeeping is CPU-side, no sync) and independent of the
				# allocator-history trace, so it still localises the peak layer
				# if the trace ring buffer wraps.
				_memprof_layers = os.environ.get("BATCHGEN_MEM_PROFILE", "0") == "1"
				# The allocator-history ring buffer only holds the LAST max_entries
				# events, and one K3 prefill emits ~2e7 of them (896-expert
				# moe_infer loop x 92 layers), so a whole-forward trace only ever
				# retains the tail. The HBM peak is set in the FIRST layer, so
				# BATCHGEN_MEM_PROFILE_STOP_LAYER=N dumps and stops recording right
				# after layer N: a short, wrap-free trace of the window that
				# actually contains the peak.
				_memprof_stop_layer = int(os.environ.get("BATCHGEN_MEM_PROFILE_STOP_LAYER", "-1"))

				for layer_idx, decoder_layer in enumerate(self.model.model.layers):
					if use_attn_res:
						hidden_states, block_residual = decoder_layer(
							hidden_states,
							attention_mask=None,
							position_ids=None,
							past_key_value=None,
							output_attentions=False,
							use_cache=False,
							block_residual=block_residual,
						)
					else:
						layer_outputs = decoder_layer(
							hidden_states,
							attention_mask=None,
							position_ids=None,
							past_key_value=None,
							output_attentions=False,
							use_cache=False,
						)
						hidden_states = layer_outputs[0]
					if _memprof_layers:
						logging.info(
							f"[MEMPROF-L] rank={self.rank} layer={layer_idx} "
							f"attn={type(getattr(decoder_layer, 'self_attn', decoder_layer)).__name__} "
							f"alloc={torch.cuda.memory_allocated()/2**30:.3f}GiB "
							f"cum_peak={torch.cuda.max_memory_allocated()/2**30:.3f}GiB "
							f"reserved={torch.cuda.memory_reserved()/2**30:.3f}GiB "
							f"bres={tuple(block_residual.shape) if block_residual is not None else None}"
						)
						if layer_idx == _memprof_stop_layer:
							_p = os.path.join(
								os.environ.get("BATCHGEN_MEM_PROFILE_DIR", "/tmp"),
								f"memprof_rank{self.rank}_thru_layer{layer_idx}_{int(time.time())}.pickle",
							)
							torch.cuda.memory._dump_snapshot(_p)
							torch.cuda.memory._record_memory_history(enabled=None)
							logging.info(
								f"[MEMPROF] Rank {self.rank}: stop-layer snapshot -> {_p} "
								f"(peak_alloc so far={torch.cuda.max_memory_allocated()/2**30:.3f}GiB)"
							)

				# Output depth-mix, then norm -- that ORDER is load-bearing
				# (kimi_linear/model.py:904-913).
				if use_attn_res:
					hidden_dim = hidden_states.shape[2]
					batch_sz, seq_sz = hidden_states.shape[:2]
					hidden_states = self.model.model._apply_output_attn_res(
						hidden_states.view(-1, hidden_dim), block_residual
					).view(batch_sz, seq_sz, hidden_dim)

				# Final norm
				hidden_states = self.model.model.norm(hidden_states)

				# Extract last token hidden states for each sequence
				last_token_indices = batch_cu_seqlens[1:] - 1
				gather_last_token_hidden = getattr(
					self.parallel_manager,
					"gather_prefill_last_token_hidden",
					None,
				)
				if gather_last_token_hidden is None:
					last_token_hidden = hidden_states[
						0, last_token_indices, :
					]
				else:
					last_token_hidden = gather_last_token_hidden(
						hidden_states,
						last_token_indices,
						batch_input_ids_flat.numel(),
					)

				# lm_head matmul: BF16 by default (matches HF / SGLang / vLLM).
				# Opt into FP32-cast via BATCHGEN_GLM5_LMHEAD_FP32=1 for debugging.
				if os.environ.get("BATCHGEN_GLM5_LMHEAD_FP32", "0") == "1":
					logits = torch.nn.functional.linear(
						last_token_hidden.float(),
						self.model.lm_head.weight.float(),
						self.model.lm_head.bias.float() if hasattr(self.model.lm_head, 'bias') and self.model.lm_head.bias is not None else None
					)
				else:
					logits = torch.nn.functional.linear(
						last_token_hidden,
						self.model.lm_head.weight,
						self.model.lm_head.bias if hasattr(self.model.lm_head, 'bias') and self.model.lm_head.bias is not None else None
					).float()
				if _k3_profile_enabled:
					_k3_profile_logits.append(logits.detach())

				batch_sequences = [
					self.global_batch.get_sequence(self._local_to_uuid_map[local_idx])
					for local_idx in batch_local_indices
				]
				batch_new_tokens = self._select_tokens(logits, batch_sequences)
				if batch_new_tokens.shape[0] != batch_num_seqs:
					raise RuntimeError(
						f"Rank {self.rank}: prefill token selection shape mismatch, "
						f"got {batch_new_tokens.shape[0]} rows for {batch_num_seqs} sequences"
					)
				output_tokens.append(batch_new_tokens)

				# The FIRST generated token, straight out of prefill. A
				# max_tokens=1 request is now completed right after prefill
				# (PREFILL_PLAN C4, _finish_prefill_completed_sequences) and
				# the token reaches the client through the normal response
				# path, so this log is a cross-check of that path rather than
				# the only way to see the token. Rank 0 only; ids only (the
				# worker has no tokenizer -- decode them client-side).
				if self.rank == 0:
					logging.info(
						"[PREFILL] first sampled token ids: %s",
						batch_new_tokens.reshape(-1).tolist()[:16])

		_prefill_forward_s = time.perf_counter() - _prefill_forward_t0
		if _k3_profile_enabled:
			# Freeze the cyclic producer before any Python import or JSON work.
			# Otherwise it can refill newly released slots after the final
			# expert and make one completed prefill look like a partial second
			# pass.
			self.core_engine.stop_h2d_worker()
			_k3_profile_topk = []
			for profile_logits in _k3_profile_logits:
				top_values, top_indices = torch.topk(
					profile_logits,
					k=min(8, profile_logits.shape[-1]),
					dim=-1,
				)
				_k3_profile_topk.append({
					"ids": top_indices.cpu().tolist(),
					"values": top_values.cpu().tolist(),
				})
			from batchgen.models.moonshotai.kimi_linear.k3.mxfp4_expert import (
				KimiK3MXFP4ExpertWrapper,
			)
			from batchgen.moe.streamed_sp8_mxfp4 import (
				StreamedSP8MXFP4MoELayer,
			)
			free_hbm, total_hbm = torch.cuda.mem_get_info(self.local_rank)
			# The top-k ``.cpu()`` transfers above have drained the compute stream.
			# Pending ingress events precede the ready events that stream waited on,
			# so every timing pair is complete before ``elapsed_time`` is queried.
			logging.info("[K3_PREFILL_PROFILE] %s", json.dumps({
				"rank": self.rank,
				"hbm": {
					"current_allocated_bytes": torch.cuda.memory_allocated(
						self.local_rank
					),
					"current_reserved_bytes": torch.cuda.memory_reserved(
						self.local_rank
					),
					"peak_allocated_bytes": torch.cuda.max_memory_allocated(
						self.local_rank
					),
					"peak_reserved_bytes": torch.cuda.max_memory_reserved(
						self.local_rank
					),
					"free_bytes": free_hbm,
					"total_bytes": total_hbm,
				},
				"weight_stream": self.core_engine.get_weight_stream_profile(),
				"expert_consumer": (
					KimiK3MXFP4ExpertWrapper.prefill_profile_snapshot()
				),
				"streamed_sp8": (
					StreamedSP8MXFP4MoELayer.prefill_profile_snapshot()
				),
				"logit_topk": _k3_profile_topk,
			}, separators=(",", ":")))

		# Structured prefill record, one JSON line per rank that actually ran a
		# prefill (batchgen-benchmark docs/prefill_metrics_proposal.md). The
		# report tool prefers this over scraping the tqdm bar, which is
		# presentation, not an API.
		#
		# Deliberately NOT gated on rank 0, departing from the proposal's
		# "rank 0 only". prefill_prepacked runs only under
		# `if local_prefill_indices:`, and the tqdm bar above is
		# disable=(self.rank != 0) -- so when rank 0 owns none of the batch
		# there is neither a bar nor a rank-0 line anywhere in the log, and the
		# run reports `Prefill: 0.0s` with nothing to scrape. That is exactly
		# what the 131,069-token run produced when rank 2 owned the sequence.
		# Every participating rank emits its own tagged line; the wall time for
		# the batch is the MAX of `prefill_s` over the emitting ranks.
		logging.info("[METRICS] %s", json.dumps({
			"phase": "prefill",
			"prefill_s": _prefill_forward_s,
			"sequences": num_sequences,
			"tokens_total": total_tokens_all,
			"seq_len_min": min(seq_lengths_list) if seq_lengths_list else 0,
			"seq_len_max": max(seq_lengths_list) if seq_lengths_list else 0,
			"micro_batches": len(micro_batches),
			"max_tokens_per_micro_batch": MAX_TOKENS_PER_MICRO_BATCH,
			"world_size": self.world_size,
			"rank": self.rank,
			# Guarded: the proposal's unguarded output_tokens[0] is an IndexError
			# on a rank that ran zero micro-batches.
			"first_sampled_token_ids": (
				output_tokens[0].reshape(-1).tolist()[:16] if output_tokens else []
			),
		}, separators=(",", ":")))

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
		if new_tokens.shape[0] != len(batch):
			raise RuntimeError(
				f"Rank {self.rank}: prefill writeback shape mismatch, "
				f"got {new_tokens.shape[0]} rows for {len(batch)} local sequences"
			)

		# Update sequence state after prefill
		# For evicted re-entry: first new token goes at decoded_length offset (not 0)
		new_tokens_cpu = new_tokens.cpu()
		for i, local_idx in enumerate(batch):
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			token_pos = seq.decoded_length  # 0 for fresh, prev_decoded for re-entry
			self.query_book[local_idx].decoded_tokens[:, token_pos] = new_tokens_cpu[i]
			seq.decoded_length = token_pos + 1
			seq.current_context_length = seq.original_prompt_length + seq.decoded_length

			# Check for EOS respecting ignore_eos flag
			if self._should_stop_at_eos(new_tokens_cpu[i].item()):
				seq.eos_reached = True

		return new_tokens

	# ============ RANK-0 BOUNDARY DECISION COMPUTATION ============

	def _make_boundary_decision_request(
		self,
		decode_uuids: List[str],
		global_seq_state: Dict[str, Dict],
		global_candidate_info: Dict[str, Dict],
		per_rank_free: List[int],
		chunk_size: int,
		per_node_host_stats: Optional[List[Dict[str, int]]],
	) -> BoundaryDecisionRequest:
		"""Snapshot the boundary-decision inputs (rank 0 only).

		``seq_meta`` is sourced from ``global_batch`` (rank 0 holds every
		sequence) for the union of decode + loadable-candidate uuids,
		replacing the handler's direct ``global_batch`` reads. Entries are
		omitted for uuids with no sequence — the handler treats a missing
		entry as ``global_idx = inf`` / ``priority = 0``, matching the
		legacy ``if seq is not None else ...`` fallbacks.
		"""
		seq_meta: Dict[str, BoundarySeqMeta] = {}
		for uuid in set(decode_uuids) | set(global_candidate_info.keys()):
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				continue
			seq_meta[uuid] = BoundarySeqMeta(
				global_idx=seq.global_idx,
				priority=getattr(seq, 'priority', 0),
				current_context_length=getattr(seq, 'current_context_length', 0),
				host_token_capacity=getattr(seq, 'host_token_capacity', 0),
				host_pages_allocated=getattr(seq, 'host_pages_allocated', 0),
			)
		return BoundaryDecisionRequest(
			decode_uuids=tuple(decode_uuids),
			global_seq_state=global_seq_state,
			global_candidate_info=global_candidate_info,
			per_rank_free=tuple(per_rank_free),
			chunk_size=chunk_size,
			per_node_host_stats=tuple(per_node_host_stats) if per_node_host_stats else None,
			seq_meta=seq_meta,
			world_size=self.world_size,
			num_gpus_per_node=NUM_GPUS_PER_NODE,
			enable_host_kv_eviction=self.enable_host_kv_eviction,
			host_kv_eviction_watermark=self.host_kv_eviction_watermark,
			attn_tp_size=self._decode_attn_tp_size(),
		)

	def _compute_boundary_decisions(
		self,
		decode_uuids: List[str],
		global_seq_state: Dict[str, Dict],
		global_candidate_info: Dict[str, Dict],
		per_rank_free: List[int],
		chunk_size: int,
		per_node_host_stats: Optional[List[Dict[str, int]]],
	) -> 'BoundaryDecisions':
		"""Compute the rank-0 page-boundary decision.

		Delegates to ``BoundaryHandler``; the NCCL gather that builds the
		input dicts and the broadcast/execution of the result stay on the
		worker (in ``_page_boundary_fast``).
		"""
		return BoundaryHandler.compute_decisions(
			self._make_boundary_decision_request(
				decode_uuids, global_seq_state, global_candidate_info,
				per_rank_free, chunk_size, per_node_host_stats,
			)
		)

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
		timing.num_kv_append_tasks = self._wait_pending_kv_append_tasks(sync_distributed_errors=True)
		timing.wait_kv_append_ms = (time.perf_counter() - t0) * 1000
		
		# decode_uuids sync: only run in debug mode for desync detection.
		# In production, rank 0 makes all decisions so sync is unnecessary.
		t_sync = time.perf_counter()
		if BATCHGEN_CB_DEBUG:
			local_decode_set = set(decode_uuids)
			all_decode_sets = [None] * self.world_size
			dist.all_gather_object(all_decode_sets, local_decode_set)
			all_sets_equal = all(s == local_decode_set for s in all_decode_sets if s is not None)
			if not all_sets_equal:
				for r, s in enumerate(all_decode_sets):
					if s != local_decode_set:
						diff_in_r = s - local_decode_set if s else set()
						diff_in_local = local_decode_set - s if s else local_decode_set
						logging.error(
							f"Rank {self.rank}: decode_uuids DESYNC detected at boundary start! "
							f"Rank {r} has {len(diff_in_r)} extra: {list(diff_in_r)[:5]}, "
							f"Rank {self.rank} has {len(diff_in_local)} extra: {list(diff_in_local)[:5]}"
						)
				# Use RANK 0 as authoritative source
				rank0_set = all_decode_sets[0] if all_decode_sets[0] is not None else set()
				decode_uuids = sorted(
					rank0_set,
					key=lambda u: self.global_batch.get_sequence(u).global_idx if self.global_batch.get_sequence(u) else float('inf')
				)
				batch = self._get_local_indices_for_uuids(decode_uuids)
				logging.warning(f"Rank {self.rank}: Using rank-0 authoritative set at boundary start, decode_uuids now {len(decode_uuids)}")
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
		chunk_size = self._get_effective_chunk_size()
		local_seq_state = {}
		for uuid in decode_uuids:
			if uuid in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				seq.validate_metadata(f"rank {self.rank} _page_boundary_fast/decode_state")
				is_completed = self._is_sequence_completed(seq)
				local_seq_state[uuid] = {
					'decoded_length': seq.decoded_length,
					'current_context_length': seq.current_context_length,
					'gpu_pages_allocated': seq.gpu_pages_allocated,
					'eos_reached': seq.eos_reached,
					'rep_detected': getattr(seq, '_rep_detected', False),
					'completed': is_completed,
					'additional_pages_needed': seq.get_additional_gpu_pages_needed(),
					'assigned_rank': seq.assigned_rank,  # Include for consistency
					# Option 1 (G>1): the serve-group id. The boundary validator keys
					# group ownership on this (a uuid is reported by exactly its G
					# contiguous ranks [g*G,(g+1)*G)), not on a single assigned_rank.
					'decode_dp_group': seq.decode_dp_group,
					# Host KV growth fields
					'needs_host_growth': seq.needs_host_kv_growth(chunk_size),
					'host_growth_pages': seq.get_host_growth_pages(chunk_size),
					'host_pages_allocated': seq.host_pages_allocated,
					'host_token_capacity': seq.host_token_capacity,
					# prompt_length: required so Phase 4.C can compute the
					# re-entry reconstruction length on ALL ranks deterministically,
					# not just the owner. Without this, non-owning ranks have a
					# stale prompt_length for re-evicted sequences (where the
					# owner has already rewritten prompt_length in a prior
					# eviction). See Phase 4.C.
					'prompt_length': seq.prompt_length,
					'reentry_decoded_baseline': seq.reentry_decoded_baseline,
					'max_decode_length': seq.max_decode_length,
					'original_max_decode_length': seq.original_max_decode_length,
					# total_decoded_before_eviction: propagated here so the
					# next _prepare_prefill_batch's eviction priority sort is
					# consistent across ranks.
					'total_decoded_before_eviction': seq.total_decoded_before_eviction,
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
			seq.validate_metadata(f"rank {self.rank} _page_boundary_fast/load_candidate")
			# Report this as a potential load candidate
			local_candidate_state[uuid] = {
				'pages_needed': seq.get_gpu_pages_for_two_page_buffer(),
				'assigned_rank': seq.assigned_rank,
				'decode_dp_group': seq.decode_dp_group,  # Option 1 group-ownership key
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
		validate_boundary_payload_alignment(
			decode_uuids, all_payloads, group_size=self._decode_attn_tp_size()
		)
		
		timing.gather_ms = (time.perf_counter() - t0) * 1000
		
		# ========== PHASE 2: MERGE GATHERED DATA + RANK-0 DECISIONS ==========
		t0 = time.perf_counter()

		# Extract per-rank free pages
		per_rank_free = [p['free_pages'] for p in all_payloads]

		# Merge sequence state. G==1: each uuid appears exactly once (single owner).
		# G>1 (Option 1): the uuid is reported by all G ranks of its group, so pin a
		# CANONICAL owning_rank = decode_dp_group*G (the group leader) instead of
		# last-writer-wins (which would arbitrarily land on the highest group rank).
		_G_merge = self._decode_attn_tp_size()
		global_seq_state = {}
		for rank_idx, payload in enumerate(all_payloads):
			if payload and payload['seq_state']:
				for uuid, state in payload['seq_state'].items():
					global_seq_state[uuid] = state
					if _G_merge > 1:
						_g = state.get('decode_dp_group')
						global_seq_state[uuid]['owning_rank'] = (
							_g * _G_merge if _g is not None else rank_idx
						)
					else:
						global_seq_state[uuid]['owning_rank'] = rank_idx

		# Merge candidate state
		global_candidate_info = {}
		for payload in all_payloads:
			if payload and payload['candidate_state']:
				global_candidate_info.update(payload['candidate_state'])

		# VALIDATION: Check that all decode_uuids have state reported
		missing_uuids = [u for u in decode_uuids if u not in global_seq_state]
		if missing_uuids:
			missing_details = []
			for missing_uuid in missing_uuids[:10]:
				seq = self.global_batch.get_sequence(missing_uuid)
				expected_rank = seq.assigned_rank if seq else "N/A"
				in_local_map = missing_uuid in self._uuid_to_local_map
				seq_status = seq.status.name if seq else "NOT_FOUND"
				rank_reported = [r for r, p in enumerate(all_payloads)
								if p and p.get('seq_state', {}).get(missing_uuid)]
				missing_details.append(
					f"{missing_uuid}(assigned_rank={expected_rank}, in_local_map={in_local_map}, "
					f"status={seq_status}, reported_by_ranks={rank_reported})"
				)
			raise RuntimeError(
				f"Rank {self.rank}: [SCHED_INVARIANT] {len(missing_uuids)} active decode "
				f"UUIDs missing from gathered seq_state; decode_uuids_len={len(decode_uuids)}, "
				f"global_seq_state_len={len(global_seq_state)}, details={missing_details}"
			)

		# Update local SequenceEntry with gathered info (for sequences on other ranks)
		for uuid, state in global_seq_state.items():
			if uuid not in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				if seq is not None:
					seq.decoded_length = state['decoded_length']
					seq.current_context_length = state['current_context_length']
					seq.gpu_pages_allocated = state['gpu_pages_allocated']
					seq.eos_reached = state['eos_reached']
					if state.get('rep_detected', False):
						seq._rep_detected = True
					# Sync host KV fields to keep all ranks consistent for migration planning
					seq.host_pages_allocated = state['host_pages_allocated']
					seq.host_token_capacity = state['host_token_capacity']
					# Sync prompt_length (may have been rewritten by a prior
					# eviction on the owner) and total_decoded_before_eviction
					# so Phase 4 mutations can be computed deterministically on
					# all ranks, and the next _prepare_prefill_batch selection
					# priority sort is consistent.
					if 'prompt_length' in state:
						seq.prompt_length = state['prompt_length']
					if 'reentry_decoded_baseline' in state:
						seq.reentry_decoded_baseline = state['reentry_decoded_baseline']
					if 'max_decode_length' in state:
						seq.max_decode_length = state['max_decode_length']
					if 'original_max_decode_length' in state:
						seq.original_max_decode_length = state['original_max_decode_length']
					if 'total_decoded_before_eviction' in state:
						seq.total_decoded_before_eviction = state['total_decoded_before_eviction']
					# Validate gathered ctx_len
					expected_ctx = seq.original_prompt_length + seq.decoded_length
					if seq.current_context_length != expected_ctx:
						seq.log_event(SeqEvent.CTX_MISMATCH, self.rank,
							f"gathered_ctx={seq.current_context_length}, expected={expected_ctx}")
						seq.current_context_length = expected_ctx
					seq.validate_metadata(
						f"rank {self.rank} _page_boundary_fast/gathered_state",
						require_owner_tensors=False,
					)

		# ========== RANK 0 COMPUTES ALL DECISIONS ==========
		# Only rank 0 makes batching decisions. All other ranks receive via broadcast.
		# This eliminates desync from independent decision-making.
		worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		per_node_host_stats = self._gather_host_kv_stats_by_node(worker_view)

		if self.rank == 0:
			decisions = self._compute_boundary_decisions(
				decode_uuids, global_seq_state, global_candidate_info,
				per_rank_free, chunk_size, per_node_host_stats,
			)
		else:
			decisions = None

		# ========== PHASE 3: BROADCAST DECISIONS ==========
		decisions_list = [decisions]
		dist.broadcast_object_list(decisions_list, src=0)
		decisions = decisions_list[0]
		if decisions.scheduler_error:
			raise RuntimeError(f"Rank {self.rank}: {decisions.scheduler_error}")

		timing.num_completed = len(decisions.completed_uuids)
		timing.num_onhold = len(decisions.onhold_uuids)

		# ========== PHASE 4: EXECUTE DECISIONS LOCALLY ==========
		# All ranks execute the same decisions, but only operate on locally-owned sequences

		# A. Release completed sequences
		#
		# ORDERING FIX: _release_gpu_kv_pages and _release_host_kv_pages_for_batch
		# must run BEFORE _report_completion. Previously _report_completion ran
		# first, which pops seq.uuid from self._uuid_to_local_map on ALL ranks
		# (including the owner). The subsequent
		#     my_completed = [u for u in completed_uuids if u in self._uuid_to_local_map]
		# filter then always produced an EMPTY list on the owner — so the host
		# KV worker view never released its pages for completed sequences, and
		# the GPU KV manager never released its pages either. Host KV slowly
		# filled up across the test run, triggering excessive eviction cycles,
		# which amplified the cross-rank state drift that eventually crashed the
		# server at a collective timeout.
		completed_uuids = decisions.completed_uuids
		if completed_uuids:
			self._update_batch_status(completed_uuids, SequenceStatus.COMPLETED)
			# Incremental write: gather completed tokens to rank 0
			self._submit_completed_to_incremental_writer(completed_uuids)
			# Gather decoded tokens from owning ranks before reporting
			gathered_texts = self._gather_completed_tokens(completed_uuids)

			# Release resources on owners BEFORE popping local_map entries via
			# _report_completion (see ordering fix note above).
			my_completed = [u for u in completed_uuids if u in self._uuid_to_local_map]
			if my_completed:
				# Only release GPU pages for seqs that were actually GPU-allocated.
				# See note at the matching site (~line 5435) — zero-tok-EOS
				# prefill completions are in _uuid_to_local_map but never
				# registered with the GPU paged manager.
				gpu_allocated = [u for u in my_completed if u in self._sequences_with_gpu_kv]
				if gpu_allocated:
					self._release_gpu_kv_pages(self._get_local_indices_for_uuids(gpu_allocated))
				self._release_host_kv_pages_for_batch(my_completed)

			# All-ranks: zero scalar counters so downstream reads (e.g.
			# migration planning iterating all sequences) never see a stale
			# non-zero page count for completed sequences.
			for uuid in completed_uuids:
				seq = self.global_batch.get_sequence(uuid)
				if seq is not None:
					seq.gpu_pages_allocated = 0
					seq.host_pages_allocated = 0
					seq.host_token_capacity = 0
					self._sequences_with_gpu_kv.discard(uuid)

			# Report completions (this is what pops local_map on the owner).
			# Must run LAST so the _release_*_pages calls above see the
			# correct local_map state.
			for uuid in completed_uuids:
				self._report_completion(uuid, gathered_text=gathered_texts.get(uuid))
			# Report completions to adaptive chunk sizer
			if self.adaptive_chunk_sizer is not None:
				for uuid in completed_uuids:
					state = global_seq_state.get(uuid)
					if state:
						self.adaptive_chunk_sizer.report_completion(state['decoded_length'])
			# Log completion details for diagnostics
			if self.rank == 0 and BATCHGEN_CB_DEBUG:
				for uuid in completed_uuids:
					seq = self.global_batch.get_sequence(uuid)
					state = global_seq_state.get(uuid, {})
					was_evicted = getattr(seq, 'total_decoded_before_eviction', 0) > 0
					logging.debug(
						f"[COMPLETION] seq={uuid[:8]} "
						f"decoded={state.get('decoded_length', 0)} "
						f"prompt={getattr(seq, 'original_prompt_length', seq.prompt_length)} "
						f"was_evicted={was_evicted} "
						f"host_pages={state.get('host_pages_allocated', 0)}"
					)

		decode_uuids = decisions.active_uuids
		batch = self._get_local_indices_for_uuids(decode_uuids)

		# B. Host KV eviction
		#
		# SYNC MODEL: Mutations here must keep every rank consistent without
		# requiring a follow-up _sync_sequence_metadata call. The only pieces
		# that can only live on the owning rank are the actual token tensors
		# (evicted_token_ids, input_ids view, decoded_tokens buffer). All
		# scalar metadata — prompt_length, current_context_length,
		# total_decoded_before_eviction, host/gpu page counters, status — is
		# updated on ALL ranks deterministically, using values already
		# synchronized in Phase 1/2 of this same boundary call.
		#
		# For re-entry length: new_reentry_len = seq.prompt_length +
		# seq.decoded_length. At Phase 4.C time, both operands are consistent
		# across ranks because Phase 2 synced them from the owner.
		host_evicted_uuids = decisions.host_evicted_uuids
		if host_evicted_uuids:
			# Owner-only: build and stash the evicted_token_ids tensor and
			# release on-device resources (GPU KV pages, host KV worker view).
			#
			# CASCADING RE-ENTRY FIX: only append decoded tokens BEYOND the
			# re-entry baseline. For a fresh sequence the baseline is 0 (all
			# decoded tokens are genuinely new). For a sequence that has
			# already been re-entered, decoded_tokens[0:reentry_decoded_baseline]
			# contains the historical output copied in at the last re-entry
			# prep — those tokens ALSO live inside the current reconstructed
			# prompt (input_ids[original_prompt_length:prompt_length]), so
			# re-appending them here would double-count them and the next
			# re-entry cycle would receive a prompt that grew by prev_decoded
			# instead of by new_decoded_count, producing the geometric
			# doubling seen in multi-eviction runs.
			my_evicted = [u for u in host_evicted_uuids if u in self._uuid_to_local_map]
			if my_evicted:
				# Host-eviction usually targets seqs already in DECODE (so they
				# have GPU pages), but defensively intersect with the source-of-
				# truth set in case an EVICTED seq never reached decode.
				gpu_allocated = [u for u in my_evicted if u in self._sequences_with_gpu_kv]
				if gpu_allocated:
					self._release_gpu_kv_pages(self._get_local_indices_for_uuids(gpu_allocated))
				for uuid in my_evicted:
					seq = self.global_batch.get_sequence(uuid)
					prompt_tokens = seq.input_ids[0, :seq.prompt_length]
					baseline = seq.reentry_decoded_baseline
					if (
						seq.decoded_tokens is not None
						and seq.decoded_length > baseline
					):
						new_decoded = seq.decoded_tokens[0, baseline:seq.decoded_length]
						seq.evicted_token_ids = torch.cat([prompt_tokens, new_decoded])
					else:
						seq.evicted_token_ids = prompt_tokens.clone()
					if BATCHGEN_CB_DEBUG:
						logging.debug(
							f"[HOST_KV_EVICT_DETAIL] seq={uuid[:8]} "
							f"decoded={seq.decoded_length} "
							f"host_pages={seq.host_pages_allocated} "
							f"tokens_saved={len(seq.evicted_token_ids)}"
						)
				evicted_global_ids = [
					self.global_batch.get_sequence(u).global_idx for u in my_evicted
				]
				if worker_view is not None:
					worker_view.release_sequence_pages(evicted_global_ids)
					worker_view.unregister_sequences(evicted_global_ids)
					# DSA: mirror release + unregister on auxiliary host KV
					aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)
					if aux_view is not None:
						aux_view.release_sequence_pages(evicted_global_ids)
						aux_view.unregister_sequences(evicted_global_ids)

			# All-ranks: update scalar metadata deterministically. Compute
			# new_reentry_len from already-synced prompt_length, decoded_length,
			# and reentry_decoded_baseline so every rank arrives at the same
			# value without needing the owner's evicted_token_ids tensor.
			#
			# Matches the owner's tensor computation exactly:
			#   len(evicted_token_ids)
			#   = len(prompt_tokens[:prompt_length]) + len(decoded_tokens[baseline:decoded_length])
			#   = prompt_length + max(0, decoded_length - baseline)
			for uuid in host_evicted_uuids:
				seq = self.global_batch.get_sequence(uuid)
				baseline = seq.reentry_decoded_baseline
				new_decoded_count = max(0, seq.decoded_length - baseline)
				new_reentry_len = seq.prompt_length + new_decoded_count
				# total_decoded_before_eviction tracks cumulative output length,
				# which at this point equals seq.decoded_length (the full output
				# buffer including historical tokens carried forward across
				# re-entry cycles). Used downstream for eviction priority
				# sorting and for computing remaining_decode_budget at the
				# next re-entry.
				seq.total_decoded_before_eviction = seq.decoded_length
				seq.prompt_length = new_reentry_len
				seq.current_context_length = new_reentry_len
				saved = new_reentry_len
				seq.log_event(SeqEvent.EVICTED, self.rank,
					f"saved_tokens={saved}, decoded={seq.decoded_length}, "
					f"new_this_cycle={new_decoded_count}")
				seq.gpu_pages_allocated = 0
				seq.host_pages_allocated = 0
				seq.host_token_capacity = 0
				self._sequences_with_gpu_kv.discard(uuid)
				# M2b (d): eviction destroys the head-sharded KDA state on the
				# group's ranks, so drop the decode DP-group. A re-prefilled
				# sequence must re-group fresh — otherwise _assign_decode_dp_groups
				# treats the stale non-None group id as a preserved re-entry and
				# skips it, stranding the sequence with no live state. No-op for
				# G==1 (the field is always None on the validated pure-DP path).
				seq.decode_dp_group = None
				self.global_batch.update_status(uuid, SequenceStatus.EVICTED)

			evicted_set = set(host_evicted_uuids)
			decode_uuids = [u for u in decode_uuids if u not in evicted_set]
			batch = self._get_local_indices_for_uuids(decode_uuids)

			if self.rank == 0:
				logging.info(
					f"[HOST_KV_EVICT] Evicted {len(host_evicted_uuids)} sequences"
				)

		# C. Host KV growth. This intentionally runs after completed/evicted
		# host pages have been released so worker_view free pages match the
		# growth-debt-aware plan computed on rank 0.
		if decisions.growth_feasible and decisions.host_growth_uuids:
			host_grow_requests = []
			for uuid, growth_pages in zip(decisions.host_growth_uuids, decisions.host_growth_pages):
				# Update metadata on ALL ranks (decisions are broadcast from rank 0).
				# This keeps host_pages_allocated consistent across ranks, which is
				# critical for deterministic migration planning in _plan_kv_migration().
				seq = self.global_batch.get_sequence(uuid)
				seq.host_token_capacity += growth_pages * seq.PAGE_SIZE
				seq.host_pages_allocated += growth_pages
				# Only do actual host page allocation on owner rank
				if uuid in self._uuid_to_local_map:
					host_grow_requests.append((seq.global_idx, growth_pages))

			if host_grow_requests and worker_view is not None:
				worker_view.grow_pages_for_sequences(host_grow_requests)
				# DSA: mirror growth on auxiliary host KV
				aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)
				if aux_view is not None:
					aux_view.grow_pages_for_sequences(host_grow_requests)
				if self.rank == 0:
					logging.debug(
						f"[HOST_KV_GROWTH] Grew {len(host_grow_requests)} sequences, "
						f"chunk_size={chunk_size}"
					)
				if self.rank == 0 and BATCHGEN_CB_DEBUG:
					for uuid, growth_pages in zip(decisions.host_growth_uuids, decisions.host_growth_pages):
						if uuid in self._uuid_to_local_map:
							seq = self.global_batch.get_sequence(uuid)
							old_cap = seq.host_token_capacity - growth_pages * seq.PAGE_SIZE
							runway = seq.host_token_capacity - seq.current_context_length
							logging.debug(
								f"[HOST_KV_GROWTH_DETAIL] seq={uuid[:8]} "
								f"old_cap={old_cap} new_cap={seq.host_token_capacity} "
								f"runway={runway} pages={growth_pages}"
							)

		timing.process_ms = (time.perf_counter() - t0) * 1000

		# Calculate completed count BEFORE early return to ensure final iteration reports correctly
		timing.total_completed_cumulative = len(self.global_batch.get_sequences_by_status(SequenceStatus.COMPLETED))

		if not decode_uuids:
			timing.total_ms = (time.perf_counter() - boundary_start) * 1000
			return decode_uuids, batch, None, [], [], [], timing, False

		# D. GPU page extension / on-hold (using rank-0 decisions)
		t0 = time.perf_counter()
		onhold_uuids = decisions.onhold_uuids
		onhold_set = set(onhold_uuids)

		if onhold_uuids:
			# Owner-only: actually free GPU KV pages for locally-held sequences.
			my_onhold = [u for u in onhold_uuids if u in self._uuid_to_local_map]
			if my_onhold:
				local_indices = self._get_local_indices_for_uuids(my_onhold)
				global_ids = self._local_indices_to_global_seq_ids(local_indices)
				if global_ids and gpu_manager:
					gpu_manager.free_pages_for_sequences(global_ids)
				for uuid in my_onhold:
					self._sequences_with_gpu_kv.discard(uuid)

			# All-ranks: scalar metadata must be kept consistent. Non-owners
			# MUST also zero gpu_pages_allocated; otherwise their stale value
			# leaks into subsequent decision-making (e.g. migration planning
			# that iterates over all sequences). This was previously in the
			# my_onhold owner-only branch, creating a cross-rank desync window.
			for uuid in onhold_uuids:
				seq = self.global_batch.get_sequence(uuid)
				seq.gpu_pages_allocated = 0
				seq.log_event(SeqEvent.ON_HOLD, self.rank, "trigger=boundary")
				self.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)

			decode_uuids = [u for u in decode_uuids if u not in onhold_set]
			batch = self._get_local_indices_for_uuids(decode_uuids)

			if BATCHGEN_CB_DEBUG:
				logging.info(
					f"Rank {self.rank}: After on-hold: batch_size={len(batch)}, "
					f"num_onhold={len(onhold_uuids)}, my_onhold={len(my_onhold)}"
				)

		# Extend GPU pages for sequences that need it (not on-hold)
		seqs_needing_extension = decisions.seqs_needing_extension
		remaining_needing_ext = [u for u in seqs_needing_extension if u not in onhold_set]
		my_remaining_ext = [u for u in remaining_needing_ext if u in self._uuid_to_local_map]
		if my_remaining_ext:
			success = self._extend_gpu_kv_allocation(my_remaining_ext)
			if not success:
				# Extension failed — put failed sequences ON_HOLD to prevent
				# cache_seqlens from exceeding gpu_pages_allocated × PAGE_SIZE,
				# which would cause FlashAttention to read -1 sentinel page
				# indices and trigger CUDA illegal memory access.
				logging.warning(
					f"Rank {self.rank}: GPU page extension FAILED for "
					f"{len(my_remaining_ext)} sequences at boundary — "
					f"moving to ON_HOLD to prevent illegal memory access"
				)
				# Owner: release GPU pages
				ext_failed_local = self._get_local_indices_for_uuids(my_remaining_ext)
				ext_failed_global = self._local_indices_to_global_seq_ids(ext_failed_local)
				if ext_failed_global:
					gpu_manager.free_pages_for_sequences(ext_failed_global)
				for uuid in my_remaining_ext:
					self._sequences_with_gpu_kv.discard(uuid)

				# All ranks: zero scalars and update status for ALL failed seqs
				# (remaining_needing_ext is the globally-consistent list)
				ext_failed_set = set(remaining_needing_ext)
				for uuid in remaining_needing_ext:
					seq = self.global_batch.get_sequence(uuid)
					seq.gpu_pages_allocated = 0
					seq.log_event(SeqEvent.ON_HOLD, self.rank, "trigger=extension_failed")
					self.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)

				decode_uuids = [u for u in decode_uuids if u not in ext_failed_set]
				batch = self._get_local_indices_for_uuids(decode_uuids)

		timing.extension_ms = (time.perf_counter() - t0) * 1000

		# E. Async load (using rank-0 decisions)
		t0 = time.perf_counter()
		new_async_task = None
		new_load_uuids = decisions.new_load_uuids
		new_load_local = []
		new_load_global = []

		if new_load_uuids:
			# Pure DP loads on one assigned rank.  TP decode replicates the GPU
			# KV on every member of the sequence's serve-group, so every group
			# rank must allocate and launch its own local host->GPU load.
			my_new_uuids = [
				u for u in new_load_uuids
				if self._owns_local_sequence(self.global_batch.get_sequence(u))
			]
			new_load_local = self._get_local_indices_for_uuids(my_new_uuids)

			if new_load_local:
				actual_free = gpu_manager.get_stats().num_free_pages if gpu_manager and gpu_manager.is_initialized else 0

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
					else:
						logging.warning(
							f"Rank {self.rank}: Dropping {uuid[:8]} from load - "
							f"need={pages_needed}, pages_used={pages_used}, actual_free={actual_free}"
						)

				if filtered_local:
					new_load_local = filtered_local
					new_load_global = filtered_global
					tokens = filtered_tokens

					gpu_manager.allocate_pages_for_sequences(new_load_global, tokens)
					timing.load_alloc_ms = (time.perf_counter() - t0) * 1000

					t_launch = time.perf_counter()
					if worker_view is not None:
						existing_global_ids = self._local_indices_to_global_seq_ids(batch)
						if isinstance(gpu_manager, DualKVCacheCoordinator):
							pointers = self._prepare_dual_kv_load_pointers(
								gpu_manager, new_load_global, existing_global_ids
							)
							new_async_task = self._launch_dual_host_kv_load(pointers)
							self._async_load_tensors = pointers
						else:
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
							if existing_global_ids:
								gpu_manager.rebuild_page_table(existing_global_ids)
							self._async_load_tensors = {
								'k_ptrs': k_ptrs, 'v_ptrs': v_ptrs,
								'sequence_tensor': sequence_tensor,
								'active_page_counts': active_page_counts,
							}
					timing.load_launch_ms = (time.perf_counter() - t_launch) * 1000
				else:
					new_load_local = []
					new_load_global = []
					logging.warning(
						f"Rank {self.rank}: All load candidates dropped due to insufficient pages, "
						f"actual_free={actual_free}"
					)
		
		timing.num_loaded = len(new_load_uuids)
		
		# ========== FINAL PAGE TABLE REBUILD ==========
		t0 = time.perf_counter()
		if BATCHGEN_CB_DEBUG:
			global_ids_for_rebuild = self._local_indices_to_global_seq_ids(batch) if batch else []
			logging.debug(
				f"Rank {self.rank}: FINAL REBUILD: batch_size={len(batch)}, "
				f"global_ids_count={len(global_ids_for_rebuild)}"
			)
		self._rebuild_page_table_for_batch(batch, gpu_manager)
		if BATCHGEN_CB_DEBUG and gpu_manager and gpu_manager.is_initialized:
			mgr = gpu_manager._gpu_page_table_manager
			if mgr and mgr.gpu_table is not None:
				logging.debug(
					f"Rank {self.rank}: After rebuild: gpu_table.shape={mgr.gpu_table.shape}, "
					f"slot_to_seq_id_len={len(mgr.slot_to_seq_id)}"
				)
		timing.rebuild_ms = (time.perf_counter() - t0) * 1000
		
		# ========== UPDATE MOE BUFFER SIZE ==========
		# Find max batch size across all ranks to minimize all-gather/all-reduce communication
		t0 = time.perf_counter()
		self._sync_decode_moe_rank_counts(batch, reason="page_boundary")
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
		# Compare the LOCAL batch against the LOCAL subset of decode_uuids.
		# (Earlier versions passed the full cross-rank decode_uuids to
		# batch_matches_expected_uuid_order, which returns False whenever
		# len(batch) != len(decode_uuids) — i.e., always on world_size > 1.
		# The self-assignment `batch = expected_local` that followed was a
		# no-op; the error line was spurious noise.)
		expected_local = self._get_local_indices_for_uuids(decode_uuids)
		if list(batch) != expected_local:
			actual_uuids = local_indices_to_uuid_order(batch, self._local_to_uuid_map)
			expected_uuids_local = [
				self._local_to_uuid_map.get(idx) for idx in expected_local
			]
			logging.error(
				f"Rank {self.rank}: BATCH MISMATCH after boundary! "
				f"batch={batch} expected_local={expected_local} "
				f"actual_uuids={actual_uuids} expected_uuids={expected_uuids_local}"
			)
			batch = expected_local
			self._rebuild_page_table_for_batch(batch, gpu_manager)
			logging.info(f"Rank {self.rank}: Page table rebuilt after batch correction")
		
		# FINAL VERIFICATION: Ensure page table matches batch before returning
		if batch and gpu_manager and gpu_manager.is_initialized:
			mgr = gpu_manager._gpu_page_table_manager
			if mgr and mgr.gpu_table is not None:
				if len(mgr.slot_to_seq_id) != len(batch):
					logging.error(
						f"Rank {self.rank}: CRITICAL - Page table STILL mismatched at function return! "
						f"active_slots={len(mgr.slot_to_seq_id)}, batch_size={len(batch)}, "
						f"gpu_table.shape={tuple(mgr.gpu_table.shape)}"
					)
		
		timing.total_ms = (time.perf_counter() - boundary_start) * 1000

		# Periodic host KV diagnostic summary
		self._boundary_count += 1
		if self.rank == 0 and BATCHGEN_CB_DEBUG and self._boundary_count % 10 == 0:
			worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
			if worker_view is not None:
				hs = worker_view.get_stats()
				used = hs.num_total_pages - hs.num_free_pages
				pct = (used / hs.num_total_pages * 100) if hs.num_total_pages > 0 else 0
				# Gather status counts
				status_counts = {}
				for s in SequenceStatus:
					cnt = len(self.global_batch.get_sequences_by_status(s))
					if cnt > 0:
						status_counts[s.name] = cnt
				# Per-sequence host page stats
				host_pages_list = []
				for uuid in decode_uuids:
					seq = self.global_batch.get_sequence(uuid)
					if seq is not None:
						host_pages_list.append(seq.host_pages_allocated)
				chunk_val = self._get_effective_chunk_size()
				hp_min = min(host_pages_list) if host_pages_list else 0
				hp_max = max(host_pages_list) if host_pages_list else 0
				hp_avg = sum(host_pages_list) / len(host_pages_list) if host_pages_list else 0
				logging.info(
					f"[HOST_KV_SUMMARY][Iter {self._boundary_count}] "
					f"host_pages: total={hs.num_total_pages} free={hs.num_free_pages} "
					f"used={used} ({pct:.1f}%) "
					f"chunk_size={chunk_val} | {status_counts} | "
					f"per_seq_host_pages: min={hp_min} max={hp_max} avg={hp_avg:.0f}"
				)

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

		if pending_local_indices and isinstance(gpu_manager, DualKVCacheCoordinator):
			if not isinstance(async_task, DualAsyncKVTask):
				raise RuntimeError(
					"DSA async load finalize requires a completed DualAsyncKVTask"
				)

		pending_local_uuid_set = {
			self._local_to_uuid_map[idx]
			for idx in pending_local_indices
			if idx in self._local_to_uuid_map
		}

		# VALIDATION: Verify all pending_uuids exist and every local rank that
		# must hold a replica confirmed its load.  Under TP decode, the old
		# assigned-rank-only check admitted a UUID after only one of the G ranks
		# had GPU pages, causing the other ranks to enter decode with zero pages.
		group_size = self._decode_attn_tp_size()
		valid_pending_uuids = []
		invalid_pending = []
		for uuid in pending_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				invalid_pending.append(f"{uuid[:8]} missing from global_batch")
				continue
			if seq.assigned_rank is None:
				invalid_pending.append(f"{uuid[:8]} gid={seq.global_idx} has no assigned_rank")
				continue
			if group_size > 1:
				from batchgen.decode_dp_group import rank_in_decode_group
				must_confirm_local = rank_in_decode_group(
					seq.decode_dp_group, self.rank, group_size
				)
			else:
				must_confirm_local = seq.assigned_rank == self.rank
			if must_confirm_local and uuid not in pending_local_uuid_set:
				invalid_pending.append(
					f"{uuid[:8]} gid={seq.global_idx} rank {self.rank} "
					"did not confirm local load"
				)
				continue
			valid_pending_uuids.append(uuid)

		if invalid_pending:
			raise RuntimeError(
				f"Rank {self.rank}: invalid async-load pending UUIDs; "
				f"pending_count={len(pending_uuids)}, invalid={invalid_pending[:10]}"
			)

		self._update_batch_status(valid_pending_uuids, SequenceStatus.IN_DECODE)

		for local_idx in pending_local_indices:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			seq.gpu_pages_allocated = seq.get_gpu_pages_for_two_page_buffer()
			# Mark that this sequence has received its initial GPU reservation
			seq.mark_initial_gpu_reservation_done()
			self._sequences_with_gpu_kv.add(uuid)
			seq.validate_metadata(f"rank {self.rank} _finalize_async_load_minimal")
			seq.log_event(SeqEvent.KV_LOAD_DONE, self.rank,
				f"gpu_pages={seq.gpu_pages_allocated}")
			logging.debug(
				f"[LOAD_CONFIRM] Rank {self.rank}: finalized uuid={uuid[:8]} "
				f"gid={seq.global_idx} status=IN_DECODE gpu_pages={seq.gpu_pages_allocated} "
				f"ctx={seq.current_context_length} host_pages={seq.host_pages_allocated}"
			)
			# Refresh query_book entry for resumed ON_HOLD sequences to prevent stale references
			if local_idx in self.query_book:
				self.query_book[local_idx] = make_query_book_entry(seq)

		if hasattr(self, '_async_load_tensors'):
			self._async_load_tensors = None

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

	def _ensure_decode_metadata_capacity(self, batch_size: int) -> None:
		"""Allocate reusable decode metadata buffers sized for the local batch."""
		current = self._decode_cache_seqlens_i32
		if current is not None and current.shape[0] >= batch_size:
			return

		capacity = max(1, batch_size)
		# Allocate OUTSIDE inference_mode so these persistent, reused buffers are
		# normal tensors. Otherwise, when this lazily (re)allocates inside the
		# decode loop's inference_mode, they become "inference tensors" and the
		# later configure-time bind (_bind_decode_attention_metadata_for_graph_config,
		# called from generate() outside inference_mode on each new prefill wave)
		# fails with "Inplace update to inference tensor outside InferenceMode".
		with torch.inference_mode(False):
			self._decode_cache_seqlens_i32 = torch.empty(
				(capacity,), dtype=torch.int32, device=self.torch_device
			)
			self._decode_position_ids_i64 = torch.empty(
				(capacity, 1), dtype=torch.int64, device=self.torch_device
			)
			self._decode_cache_seqlens_cpu_staging = torch.empty(
				(capacity,), dtype=torch.int32, pin_memory=True
			)
		self._decode_metadata_batch_key = None
		self._decode_metadata_cpu_seqlens = None

	def _bind_decode_attention_metadata(
		self,
		batch_sequences: List[SequenceEntry],
		cache_seqlens: List[int],
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Bind cache_seqlens/position_ids while avoiding steady-state HtoD copies.

		The first decode step for a batch seeds the reusable device buffers from
		CPU sequence metadata. While the same local batch remains active, every
		sequence advances by exactly one token per step, so metadata advances with
		device-side increments instead of rebuilding CUDA tensors from Python lists.
		"""
		batch_size = len(cache_seqlens)
		self._ensure_decode_metadata_capacity(batch_size)

		batch_key = tuple(seq.global_idx for seq in batch_sequences)
		cache_seqlens_key = tuple(int(v) for v in cache_seqlens)
		prev_key = self._decode_metadata_batch_key
		prev_seqlens = self._decode_metadata_cpu_seqlens
		can_advance_on_device = (
			prev_key == batch_key
			and prev_seqlens is not None
			and len(prev_seqlens) == batch_size
			and all(cur == prev + 1 for cur, prev in zip(cache_seqlens_key, prev_seqlens))
		)

		cache_view = self._decode_cache_seqlens_i32[:batch_size]
		position_view = self._decode_position_ids_i64[:batch_size]

		if can_advance_on_device:
			cache_view.add_(1)
			position_view.add_(1)
		else:
			staging = self._decode_cache_seqlens_cpu_staging
			for i, value in enumerate(cache_seqlens_key):
				staging[i] = value
			cache_view.copy_(staging[:batch_size], non_blocking=True)
			position_view[:, 0].copy_(cache_view.to(dtype=torch.int64))
			position_view.sub_(1)

		self._decode_metadata_batch_key = batch_key
		self._decode_metadata_cpu_seqlens = cache_seqlens_key
		return cache_view, position_view

	def _bind_decode_attention_metadata_for_graph_config(
		self,
		local_decode_indices: List[int],
	) -> None:
		"""Bind decode metadata before configure-time CUDA graph capture."""
		if local_decode_indices:
			batch_sequences = [
				self.global_batch.get_sequence(self._local_to_uuid_map[idx])
				for idx in local_decode_indices
			]
			cache_seqlens = []
			for seq in batch_sequences:
				expected = seq.original_prompt_length + seq.decoded_length
				ctx_len = seq.current_context_length
				if ctx_len != expected:
					seq.current_context_length = expected
					ctx_len = expected
				cache_seqlens.append(ctx_len)
			cache_view, position_view = self._bind_decode_attention_metadata(
				batch_sequences,
				cache_seqlens,
			)
			max_ctx = max(cache_seqlens)
			cur_batch = self._local_indices_to_global_seq_ids(local_decode_indices)
		else:
			cache_view = torch.zeros((0,), dtype=torch.int32, device=self.torch_device)
			position_view = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
			max_ctx = 0
			cur_batch = []

		Attn_Wrapper.attention_mask = None
		Attn_Wrapper.cache_seqlens = cache_view
		Attn_Wrapper.position_ids = position_view
		Attn_Wrapper.max_seqlen = max_ctx
		Attn_Wrapper.cur_batch = cur_batch
		AttnWrapperBase.attention_mask = None
		AttnWrapperBase.cache_seqlens = cache_view
		AttnWrapperBase.position_ids = position_view
		AttnWrapperBase.max_seqlen = max_ctx
		AttnWrapperBase.cur_batch = cur_batch
		index_topk = getattr(self.model_config, "index_topk", None)
		if index_topk is not None and cache_view.numel() > 0:
			GLM5AttnWrapper._dsa_short_count = int((cache_view <= int(index_topk)).sum().item())
		else:
			GLM5AttnWrapper._dsa_short_count = 0 if cache_view.numel() == 0 else None

		# GLM-5.2 DSA indexer reuse: clear prev top-k once per decode step (before layer 0)
		# so shared layers never reuse a stale value carried over from the previous step.
		GLM5AttnWrapper._dsa_prev_topk_indices = None
		gpu_manager = self._get_cuda_graph_gpu_manager()
		if gpu_manager is not None and cur_batch:
			manager = getattr(gpu_manager, "primary", gpu_manager)
			page_manager = getattr(manager, "_gpu_page_table_manager", None)
			if page_manager is not None:
				slot_order = list(page_manager.slot_to_seq_id) if page_manager.slot_to_seq_id else []
				if slot_order != cur_batch:
					manager.rebuild_page_table(cur_batch)

	def _sync_decode_moe_rank_counts(self, batch: List[int], *, reason: str) -> int:
		"""Synchronize per-rank decode row counts for 3D MoE padding masks."""
		local_count = int(len(batch))
		if self._decode_local_count_tensor is None:
			self._decode_local_count_tensor = torch.empty(
				(1,), dtype=torch.int64, device=self.torch_device
			)
		if (
			self._decode_all_rank_counts is None
			or self._decode_all_rank_counts.shape[0] != self.world_size
		):
			self._decode_all_rank_counts = torch.empty(
				(self.world_size,), dtype=torch.int64, device=self.torch_device
			)
		local_count_tensor = self._decode_local_count_tensor
		all_rank_counts = self._decode_all_rank_counts
		local_count_tensor.fill_(local_count)
		dist.all_gather_into_tensor(all_rank_counts, local_count_tensor)
		max_batch_size = int(all_rank_counts.max().item())

		self._current_decode_local_batch_size = local_count
		self._current_decode_max_rank_batch_size = max_batch_size
		self._current_decode_rank_token_counts = all_rank_counts

		if max_batch_size > 0 and hasattr(self, 'parallel_manager') and self.parallel_manager is not None:
			# M2b: under DP-(world/G) x TP-G decode the 32-way max above is the
			# per-GROUP batch B_grp (the G ranks of a group hold identical
			# sequences). Decode scatters those rows across the group's G ranks
			# before the DP-32 resident MoE, so the padded all_gather/all_reduce
			# layout is sized by the POST-scatter share ceil(B_grp/G), not B_grp.
			# G==1 -> ceil(max/1)==max, byte-identical to the pure-DP path.
			from batchgen.decode_dp_group import moe_ntp_from_group_max
			_G = getattr(self.parallel_manager, "attn_tp_size", 1)
			moe_ntp = moe_ntp_from_group_max(max_batch_size, _G)
			if hasattr(self.parallel_manager, 'set_num_tokens_per_rank'):
				self.parallel_manager.set_num_tokens_per_rank(moe_ntp)
			if hasattr(self.parallel_manager, 'set_rank_token_counts'):
				self.parallel_manager.set_rank_token_counts(all_rank_counts)

		if BATCHGEN_MULTI_BATCH_DIAG:
			try:
				counts_list = all_rank_counts.detach().cpu().tolist()
			except RuntimeError:
				counts_list = ["<unavailable>"]
			logging.info(
				f"[GLM5_MOE_COUNTS] Rank {self.rank}: reason={reason} "
				f"local={local_count} max={max_batch_size} counts={counts_list}"
			)
		return max_batch_size

	def _sync_prefill_moe_rank_counts(
		self, local_token_count: int, *, reason: str
	) -> int:
		"""Size resident-EP prefill collectives after TP-group row scattering."""
		local = torch.tensor(
			[int(local_token_count)],
			dtype=torch.int64,
			device=self.torch_device,
		)
		counts = torch.empty(
			(self.world_size,), dtype=torch.int64, device=self.torch_device
		)
		dist.all_gather_into_tensor(counts, local)
		G = self._decode_attn_tp_size()
		counts_list = [int(value) for value in counts.cpu().tolist()]
		for start in range(0, self.world_size, G):
			group = counts_list[start:start + G]
			if len(set(group)) != 1:
				raise RuntimeError(
					"resident-EP prefill requires identical token rows within "
					f"each TP group; ranks {start}:{start + G} reported {group}"
				)
		max_group_tokens = max(counts_list)
		from batchgen.decode_dp_group import moe_ntp_from_group_max
		moe_ntp = moe_ntp_from_group_max(max_group_tokens, G)
		self.parallel_manager.set_num_tokens_per_rank(moe_ntp)
		if self.rank == 0:
			logging.info(
				"[K3_PREFILL_MOE] reason=%s group_token_counts=%s "
				"post_scatter_ntp=%s",
				reason,
				[counts_list[start] for start in range(0, self.world_size, G)],
				moe_ntp,
			)
		return moe_ntp

	def _warmup_cuda_graphs(self):
		"""One-time CUDA graph warmup phase with model guard.

		Called from generate() after model and GPU KV manager are ready.
		Only captures graphs for supported models (currently GPT-OSS-120B).
		"""
		# Model guard: only capture for supported models
		# Phase C: layer / MoE / DSA / DSA-full segmented graph modes are retired
		# (predicate functions return False); only the whole-model graph remains
		# on GLM-5, plus K2.5's per-layer attn segment and GPT-OSS-120B's
		# whole-model graph.
		model_name = getattr(self, 'model_name', '') or ''
		model_name_l = model_name.lower()
		glm5_whole_graph_enabled = (
			self._glm5_whole_model_graph_requested_for_current_batch()
			and "glm" in model_name_l
		)
		if not self.engine_config.Basic_Config.enable_cuda_graphs:
			if glm5_whole_graph_enabled:
				self.engine_config.Basic_Config.enable_cuda_graphs = True
			else:
				return
		if (
			"gpt-oss-120b" not in model_name_l
			and not is_kimi_k25_backend_model(model_name)
			and not glm5_whole_graph_enabled
		):
			logging.info(f"Rank {self.rank}: CUDA graphs not supported for '{model_name}', skipping")
			return

		gpu_manager = self._get_cuda_graph_gpu_manager()
		if gpu_manager is None:
			logging.warning(f"Rank {self.rank}: No GPU KV manager, skipping CUDA graph warmup")
			return

		self._setup_cuda_graphs(gpu_manager)

	@staticmethod
	def _glm5_dsa_graph_score_capacity_tokens(
		primary_page_table,
		primary_page_size: int,
		aux_page_table,
		aux_page_size: int,
		*,
		model_max_position_embeddings: int | None = None,
	) -> int:
		primary_capacity = int(primary_page_table.shape[1]) * int(primary_page_size)
		aux_capacity = int(aux_page_table.shape[1]) * int(aux_page_size)
		capacities = [primary_capacity, aux_capacity]
		if model_max_position_embeddings is not None and int(model_max_position_embeddings) > 0:
			capacities.append(int(model_max_position_embeddings))
		capacity = min(capacities)
		if capacity <= 0:
			raise RuntimeError(
				"GLM-5 DSA CUDA graph requires positive primary/aux page-table capacity"
			)
		return capacity

	def _debug_flag_enabled(self, value) -> bool:
		if isinstance(value, bool):
			return value
		if isinstance(value, (int, float)):
			return value != 0
		if isinstance(value, str):
			return value.strip().lower() in {"1", "true", "yes", "on"}
		return False

	def _glm5_graph_path_log_requested_for_current_batch(self) -> bool:
		if os.environ.get("BATCHGEN_GLM5_GRAPH_PATH_LOG", "0") == "1":
			return True
		debug = self._batchgen_debug or getattr(AttnWrapperBase, "batchgen_debug", None) or {}
		if not isinstance(debug, dict):
			return False
		return self._debug_flag_enabled(debug.get("glm5_graph_path_log"))

	def _glm5_whole_model_graph_requested_for_current_batch(self) -> bool:
		# Phase C: whole-model graph is the only supported GLM-5 graph mode.
		# Activation gated solely by --enable-cuda-graph CLI flag (translated
		# into engine_config.Basic_Config.enable_cuda_graphs upstream).
		# Old per-mode env vars (BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH etc.)
		# are retired — `BATCHGEN_DECODE_GRAPH_COMPARE=1` (developer-only)
		# is the supported way to enable the compare facility.
		if getattr(getattr(self, "args", None), "disable_cuda_graphs", False):
			return False
		return glm5_whole_model_cuda_graph_requested_for_model(
			getattr(self, "model_name", None),
			enable_cuda_graph=getattr(
				getattr(self, "args", None),
				"enable_cuda_graph",
				False,
			),
		)

	def _glm5_whole_model_graph_compare_requested_for_current_batch(self) -> bool:
		if os.environ.get("BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE", "0") == "1":
			return True
		debug = self._batchgen_debug or getattr(AttnWrapperBase, "batchgen_debug", None) or {}
		if not isinstance(debug, dict):
			return False
		return self._debug_flag_enabled(debug.get("glm5_whole_model_graph_compare"))

	def _glm5_whole_model_graph_timing_requested_for_current_batch(self) -> bool:
		if os.environ.get("BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_TIMING", "0") == "1":
			return True
		debug = self._batchgen_debug or getattr(AttnWrapperBase, "batchgen_debug", None) or {}
		if isinstance(debug, dict) and self._debug_flag_enabled(debug.get("glm5_whole_model_graph_timing")):
			return True
		return self._glm5_whole_model_graph_compare_requested_for_current_batch()

	def _glm5_whole_model_graph_compare_fail_on_mismatch(self) -> bool:
		if os.environ.get("BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE_FAIL", "0") == "1":
			return True
		debug = self._batchgen_debug or getattr(AttnWrapperBase, "batchgen_debug", None) or {}
		if not isinstance(debug, dict):
			return False
		return self._debug_flag_enabled(debug.get("glm5_whole_model_graph_compare_fail"))

	@contextmanager
	def _glm5_force_segmented_graph_eager(self):
		old_debug = getattr(AttnWrapperBase, "batchgen_debug", None)
		debug = dict(old_debug) if isinstance(old_debug, dict) else {}
		debug["glm5_dsa_mode"] = "eager"
		debug["glm5_moe_mode"] = "eager"
		AttnWrapperBase.batchgen_debug = debug
		try:
			yield
		finally:
			AttnWrapperBase.batchgen_debug = old_debug

	def _glm5_decode_model_forward(self, new_tokens: torch.Tensor):
		def _forward():
			return self.model(
				new_tokens,
				attention_mask=Attn_Wrapper.attention_mask,
				position_ids=Attn_Wrapper.position_ids,
				use_cache=False,
			)

		# Phase C: layer/MoE/DSA segmented graph modes are retired; the
		# whole-model graph composes per-layer captures internally so the
		# eager fallback always runs through `_forward()` directly. The
		# legacy `_glm5_force_segmented_graph_eager` wrap stays for the
		# compare-mode eager re-run path (used in the legacy whole-model
		# compare branch that still wraps `self.model.model(...)`).
		model_name_l = (getattr(self, "model_name", "") or "").lower()
		if "glm" in model_name_l and self._glm5_whole_model_graph_requested_for_current_batch():
			with self._glm5_force_segmented_graph_eager():
				return _forward()
		return _forward()

	def _glm5_whole_model_graph_capture_signature(self, bucket_size: Optional[int] = None):
		gpu_manager = self._get_cuda_graph_gpu_manager()
		if gpu_manager is None:
			return None
		primary_manager = getattr(gpu_manager, "primary", gpu_manager)
		aux_manager = getattr(
			gpu_manager,
			"auxiliary",
			getattr(self.core_engine, "gpu_paged_kv_manager_aux", None),
		)
		if aux_manager is None:
			return None

		def _table_sig(manager):
			get_storage = getattr(manager, "get_cuda_graph_page_table_storage", None)
			get_graph_table = getattr(manager, "get_cuda_graph_page_table", None)
			try:
				if get_storage is not None:
					table = get_storage()
				else:
					table = get_graph_table() if get_graph_table is not None else None
			except RuntimeError:
				return None
			if table is None:
				return None
			return (
				int(table.data_ptr()),
				tuple(int(dim) for dim in table.shape),
				str(table.dtype),
				str(table.device),
			)

		return (
			_table_sig(primary_manager),
			_table_sig(aux_manager),
		)

	def _glm5_whole_model_graph_current_bucket_missing(self) -> bool:
		model_name_l = (getattr(self, 'model_name', '') or '').lower()
		if (
			not self._glm5_whole_model_graph_requested_for_current_batch()
			or "glm" not in model_name_l
		):
			return False
		if getattr(self, "_glm5_whole_model_graph_unavailable_reason", None):
			return False
		max_bsz = int(getattr(self, "_current_decode_max_rank_batch_size", 0) or 0)
		if max_bsz <= 0:
			return False
		capture_attempted = bool(
			getattr(self, "_glm5_whole_model_graph_capture_attempted_for_batch", False)
		)
		if self._cuda_graph_manager is None or not getattr(self, "_glm5_whole_model_graph", False):
			return not capture_attempted
		try:
			bucket = self._cuda_graph_manager.bucketing.get_padded_size(max_bsz)
		except ValueError:
			return False
		if bucket in getattr(self, "_glm5_whole_model_graph_failed_buckets", set()):
			return False
		if not self._cuda_graph_manager.has_bucket_for_all_segments(max_bsz):
			if capture_attempted:
				return False
			return True
		current_max_seqlen = int(getattr(AttnWrapperBase, "max_seqlen", 0) or 0)
		captured_max_seqlen = int(getattr(getattr(self, "_whole_model_segment", None), "max_seqlen", 0) or 0)
		if (
			current_max_seqlen > 0
			and captured_max_seqlen > 0
			and current_max_seqlen > captured_max_seqlen
		):
			if (
				capture_attempted
				and not getattr(self, "_glm5_whole_model_graph_state_change_after_capture_logged", False)
			):
				logging.info(
					f"Rank {self.rank}: GLM-5 whole-model CUDA graph max_seqlen "
					f"{captured_max_seqlen} is below current decode max_seqlen "
					f"{current_max_seqlen}; using eager decode instead of recapturing"
				)
				self._glm5_whole_model_graph_state_change_after_capture_logged = True
			return not capture_attempted
		signature = self._glm5_whole_model_graph_capture_signature(bucket)
		if signature != getattr(self, "_glm5_whole_model_graph_signature", None):
			if (
				capture_attempted
				and not getattr(self, "_glm5_whole_model_graph_state_change_after_capture_logged", False)
			):
				logging.info(
					f"Rank {self.rank}: GLM-5 whole-model CUDA graph page-table storage "
					"changed after configured capture; using eager decode instead of recapturing"
				)
				self._glm5_whole_model_graph_state_change_after_capture_logged = True
			return not capture_attempted
		return False

	def _release_glm5_whole_model_graph_state(self, *, empty_cuda_cache: bool = False) -> None:
		manager = getattr(self, "_cuda_graph_manager", None)
		segment = getattr(self, "_whole_model_segment", None)
		device = getattr(self, "torch_device", None)
		if torch.cuda.is_available() and getattr(device, "type", None) == "cuda":
			torch.cuda.synchronize(device)
		drop_bucket = getattr(manager, "drop_bucket", None)
		bucketing = getattr(manager, "bucketing", None)
		if drop_bucket is not None and bucketing is not None:
			for bucket in list(getattr(bucketing, "bucket_sizes", []) or []):
				drop_bucket(int(bucket))
		elif segment is not None:
			release = getattr(segment, "release_static_buffers", None)
			if release is not None:
				bucket_sizes = list(getattr(getattr(self, "_whole_model_bucketing", None), "bucket_sizes", []) or [])
				for bucket in bucket_sizes:
					release(int(bucket))
		self._cuda_graph_manager = None
		self._whole_model_segment = None
		self._whole_model_bucketing = None
		self._whole_model_graph = False
		self._glm5_whole_model_graph = False
		self._glm5_whole_model_graph_signature = None
		# Phase B: also drop the adapter's reference to the captured segment so
		# the next /v1/reload + recapture starts from a clean adapter context.
		if self._cuda_graph_adapter is not None:
			try:
				self._cuda_graph_adapter.release_all(manager=None)
			except Exception as _exc:
				logging.warning(
					"Phase B: failed to release adapter state: %s", _exc,
				)
		manager = None
		segment = None
		gc.collect()
		if empty_cuda_cache and torch.cuda.is_available():
			torch.cuda.empty_cache()

	def _glm5_whole_graph_path_state(self, max_rank_bsz: int):
		model_name_l = (getattr(self, "model_name", "") or "").lower()
		if (
			not self._glm5_whole_model_graph_requested_for_current_batch()
			or "glm" not in model_name_l
		):
			return "disabled", None, "not_requested"
		if max_rank_bsz <= 0:
			return "eager", None, "empty_global_batch"
		if getattr(self, "_glm5_whole_model_graph_unavailable_reason", None):
			return "eager", None, "unavailable"
		manager = getattr(self, "_cuda_graph_manager", None)
		if manager is None or not getattr(self, "_glm5_whole_model_graph", False):
			return "eager", None, "no_manager"
		try:
			bucket = manager.bucketing.get_padded_size(max_rank_bsz)
		except ValueError:
			return "eager", None, "over_bucket"
		if bucket in getattr(self, "_glm5_whole_model_graph_failed_buckets", set()):
			return "eager", bucket, "failed_bucket"
		if not manager.has_bucket_for_all_segments(max_rank_bsz):
			return "eager", bucket, "bucket_not_captured"
		current_max_seqlen = int(getattr(AttnWrapperBase, "max_seqlen", 0) or 0)
		captured_max_seqlen = int(getattr(getattr(self, "_whole_model_segment", None), "max_seqlen", 0) or 0)
		if (
			current_max_seqlen > 0
			and captured_max_seqlen > 0
			and current_max_seqlen > captured_max_seqlen
		):
			return "eager", bucket, "max_seqlen_exceeds_capture"
		signature = self._glm5_whole_model_graph_capture_signature(bucket)
		if signature != getattr(self, "_glm5_whole_model_graph_signature", None):
			return "eager", bucket, "page_table_storage_changed"
		return "graph", bucket, "captured"

	def _prepare_glm5_layer_graph_inputs(
		self,
		*,
		local_bsz: int,
		bucket: int,
		gpu_manager,
		graph_max_seqlen_override: int | None = None,
	):
		primary_manager = getattr(gpu_manager, "primary", gpu_manager)
		aux_manager = getattr(
			gpu_manager,
			"auxiliary",
			getattr(self.core_engine, "gpu_paged_kv_manager_aux", None),
		)
		if aux_manager is None:
			raise RuntimeError("GLM-5 layer graph replay requires auxiliary GPU KV manager")

		active_sequence_ids = list(getattr(AttnWrapperBase, "cur_batch", None) or [])

		def _graph_slots(manager):
			ensure_graph_table = getattr(manager, "ensure_cuda_graph_page_table", None)
			if ensure_graph_table is not None:
				ensure_graph_table(active_sequence_ids)
			slot_indices = manager._gpu_page_table_manager._slot_index_tensor
			if slot_indices is None:
				slot_indices = torch.arange(
					local_bsz,
					dtype=torch.int32,
					device=self.torch_device,
				)
			return slot_indices[:local_bsz].to(dtype=torch.int32, device=self.torch_device)

		cache_seqlens = getattr(AttnWrapperBase, "cache_seqlens", None)
		position_ids = getattr(AttnWrapperBase, "position_ids", None)
		if cache_seqlens is None or position_ids is None:
			raise RuntimeError("GLM-5 layer graph replay requires decode metadata")
		if local_bsz > 0:
			cache_seqlens_i32 = cache_seqlens[:local_bsz].to(dtype=torch.int32, device=self.torch_device)
			position_ids_i64 = position_ids[:local_bsz].to(dtype=torch.int64, device=self.torch_device)
		else:
			cache_seqlens_i32 = torch.empty((0,), dtype=torch.int32, device=self.torch_device)
			position_ids_i64 = torch.empty((0, 1), dtype=torch.int64, device=self.torch_device)

		layers = getattr(getattr(self.model, "model", None), "layers", None)
		if not layers:
			raise RuntimeError("GLM-5 layer graph replay requires decoder layers")
		wrapper = layers[0].self_attn
		index_topk = int(getattr(getattr(wrapper.module, "indexer", None), "index_topk", 2048))
		num_heads = int(getattr(wrapper.module, "num_heads", 64))
		graph_max_seqlen = int(
			graph_max_seqlen_override
			or getattr(self, "_glm5_layer_graph_max_seqlen", None)
			or getattr(AttnWrapperBase, "max_seqlen", 0)
			or index_topk
		)
		selected_lengths = torch.empty(
			(bucket,),
			dtype=torch.int32,
			device=self.torch_device,
		)
		if local_bsz > 0:
			selected_lengths[:local_bsz].copy_(
				torch.clamp(cache_seqlens_i32, max=index_topk),
				non_blocking=True,
			)
		if local_bsz < bucket:
			selected_lengths[local_bsz:].fill_(min(graph_max_seqlen, index_topk))
		from batchgen.attention.dsa.sparse_decode_mla import (
			prepare_sparse_flash_mla_decode_tensor_metadata,
		)
		tile_scheduler_metadata, num_splits = prepare_sparse_flash_mla_decode_tensor_metadata(
			selected_lengths,
			num_heads,
		)
		num_valid_tokens = torch.empty((1,), dtype=torch.int32, device=self.torch_device)
		num_valid_tokens.fill_(local_bsz)
		return {
			"cache_seqlens": cache_seqlens_i32,
			"position_ids": position_ids_i64,
			"primary_slot_indices": _graph_slots(primary_manager),
			"aux_slot_indices": _graph_slots(aux_manager),
			"num_valid_tokens": num_valid_tokens,
			"flashmla_tile_scheduler_metadata": tile_scheduler_metadata,
			"flashmla_num_splits": num_splits,
		}

	def _log_glm5_graph_path_for_forward(
		self,
		*,
		local_bsz: int,
		max_rank_bsz: int,
		rank_counts,
		gpu_manager,
		decode_iter: int,
	) -> None:
		# Phase C: layer / DSA / MoE / segmented graph modes are retired;
		# the log shows only the whole-model graph path.
		model_name_l = (getattr(self, "model_name", "") or "").lower()
		if "glm" not in model_name_l or not self._glm5_graph_path_log_requested_for_current_batch():
			return
		whole_path, whole_bucket, whole_reason = self._glm5_whole_graph_path_state(max_rank_bsz)
		counts_repr = None if rank_counts is None else f"device_tensor(shape={tuple(rank_counts.shape)})"
		logging.info(
			"[GLM5_GRAPH_PATH] rank=%s decode_iter=%s local_bsz=%s "
			"max_rank_bsz=%s rank_counts=%s "
			"whole=%s whole_bucket=%s whole_reason=%s",
			self.rank,
			decode_iter,
			local_bsz,
			max_rank_bsz,
			counts_repr,
			whole_path,
			whole_bucket,
			whole_reason,
		)

	@staticmethod
	def _generate_bucket_sizes(max_bucket: int, num_buckets: int) -> list:
		"""Generate exactly num_buckets bucket sizes from 1 to max_bucket.

		Uses geometric spacing for initial placement with magnitude-aware
		rounding (small values exact, large values rounded to clean multiples).
		Fills any gaps from rounding collisions by splitting the largest gaps.
		Caps at max_bucket if num_buckets > max_bucket.

		Examples:
		  max=256, num=9  → [1,2,4,8,16,32,64,128,256]
		  max=256, num=16 → [1,2,3,4,6,10,14,20,28,40,56,80,128,160,192,256]
		  max=16,  num=16 → [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
		"""
		import math
		num_buckets = min(num_buckets, max_bucket)
		if num_buckets <= 1:
			return [max_bucket]

		def _round_nice(x):
			"""Round to nearest clean multiple that scales with magnitude."""
			if x <= 8:
				return int(round(x))
			log2 = int(math.log2(x))
			step = max(1 << (log2 - 2), 1)
			return max(1, round(x / step) * step)

		# Geometric spacing with nice rounding
		ratio = max_bucket ** (1.0 / (num_buckets - 1))
		sizes = set()
		for i in range(num_buckets):
			sizes.add(max(1, _round_nice(ratio ** i)))
		sizes.add(1)
		sizes.add(max_bucket)
		sizes = sorted(sizes)

		# Fill gaps from rounding collisions
		while len(sizes) < num_buckets:
			best_gap, best_idx = 0, -1
			for i in range(len(sizes) - 1):
				gap = sizes[i + 1] - sizes[i]
				if gap > best_gap:
					best_gap = gap
					best_idx = i
			if best_gap < 2:
				break
			mid = _round_nice((sizes[best_idx] + sizes[best_idx + 1]) / 2)
			if mid <= sizes[best_idx] or mid >= sizes[best_idx + 1]:
				mid = (sizes[best_idx] + sizes[best_idx + 1]) // 2
			if mid in sizes or mid <= sizes[best_idx] or mid >= sizes[best_idx + 1]:
				break
			sizes.insert(best_idx + 1, mid)

		return sizes

	def _make_glm5_whole_model_capture_inputs(
		self,
		*,
		bucket: int,
		num_heads: int,
	):
		from batchgen.attention.dsa.sparse_decode_mla import (
			prepare_sparse_flash_mla_decode_tensor_metadata,
		)

		bucket = int(bucket)
		selected_lengths = torch.ones(
			(bucket,),
			dtype=torch.int32,
			device=self.torch_device,
		)
		tile_scheduler_metadata, num_splits = prepare_sparse_flash_mla_decode_tensor_metadata(
			selected_lengths,
			int(num_heads),
		)
		return {
			"input_ids": torch.empty((0, 1), dtype=torch.int64, device=self.torch_device),
			"cache_seqlens": torch.empty((0,), dtype=torch.int32, device=self.torch_device),
			"position_ids": torch.empty((0, 1), dtype=torch.int64, device=self.torch_device),
			"primary_slot_indices": torch.empty((0,), dtype=torch.int32, device=self.torch_device),
			"aux_slot_indices": torch.empty((0,), dtype=torch.int32, device=self.torch_device),
			"rank_token_counts": torch.zeros((self.world_size,), dtype=torch.int64, device=self.torch_device),
			"num_valid_tokens": torch.zeros((1,), dtype=torch.int32, device=self.torch_device),
			"flashmla_tile_scheduler_metadata": tile_scheduler_metadata,
			"flashmla_num_splits": num_splits,
		}

	def _setup_cuda_graphs(self, gpu_manager):
		"""Capture CUDA graphs for decode: full attention block per layer.

		Each graph captures the entire attention block in one shot:
		  RMSNorm → QKV proj → split → reshape → RoPE → KV write → FA → O_proj
		  → residual add + post-attn RMSNorm

		Dynamic metadata (cache_seqlens) is passed as a static-address input buffer.
		KV cache, page table, and cos/sin tables are at fixed GPU addresses.
		"""
		from batchgen.cuda_graph import BatchSizeBucketing, CUDAGraphManager
		from batchgen.models.openai.gpt_oss_120b.cuda_graph_segments import (
			FullAttnSegment, MoESegment, MoEComputeSegment, SharedMoEBufferPool,
			WholeModelSegment,
		)
		from batchgen.models.wrappers.attention import AttnWrapperBase

		# Detect K2.5 model for specialized graph segment
		_is_k25 = is_kimi_k25_backend_model(self.model_name)

		max_bucket = self.args.cuda_graph_max_bucket_size
		num_buckets = self.args.cuda_graph_num_buckets
		# Generate exactly num_buckets geometrically-spaced bucket sizes.
		# e.g. max=256, num=9  → [1,2,4,8,16,32,64,128,256]
		# e.g. max=256, num=16 → [1,2,3,4,6,10,14,20,28,40,56,80,128,160,192,256]
		# e.g. max=16,  num=16 → [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
		bucket_sizes = self._generate_bucket_sizes(max_bucket, num_buckets)
		logging.info(f"CUDA graph bucket sizes: {bucket_sizes} (max={max_bucket}, num_buckets={num_buckets})")
		bucketing = BatchSizeBucketing(bucket_sizes)
		manager = CUDAGraphManager(bucketing, device=self.torch_device)

		# Use model's max_position_embeddings (not max_context_length) so the
		# RoPE cos/sin cache captured in the graph covers ALL possible positions.
		max_rope_len = getattr(self.model_config, 'max_position_embeddings', 131072)
		model_name_l = (getattr(self, 'model_name', '') or '').lower()
		# Phase C: only whole-model graph remains for GLM-5 (other modes retired).
		# The dsa/moe/dsa_full/layer enables are pinned False so the now-dead
		# downstream branches stay typeable until a subsequent commit deletes
		# them outright. The whole-model graph is the only live path here.
		glm5_dsa_graph_enabled = False
		glm5_dsa_full_graph_enabled = False
		glm5_moe_graph_enabled = False
		glm5_layer_graph_enabled = False
		glm5_whole_graph_enabled = (
			self._glm5_whole_model_graph_requested_for_current_batch()
			and "glm" in model_name_l
		)
		if glm5_whole_graph_enabled:
			whole_graph_required = self._glm5_whole_model_graph_requested_for_current_batch()
			local_bsz = int(getattr(self, "_current_decode_local_batch_size", 0) or 0)
			max_bsz = int(getattr(self, "_current_decode_max_rank_batch_size", 0) or 0)
			capture_attempted = bool(
				getattr(self, "_glm5_whole_model_graph_capture_attempted_for_batch", False)
			)
			if max_bsz <= 0:
				logging.info(
					f"Rank {self.rank}: no global GLM-5 decode rows; skipping whole-model graph capture"
				)
				return
			failed_buckets = getattr(self, "_glm5_whole_model_graph_failed_buckets", set())
			capture_buckets = [
				int(bucket)
				for bucket in bucketing.bucket_sizes
				if int(bucket) not in failed_buckets
			]
			if not capture_buckets:
				self._glm5_whole_model_graph_capture_attempted_for_batch = True
				if whole_graph_required:
					raise RuntimeError("GLM-5 whole-model CUDA graph required but all configured buckets failed")
				return
			cur_batch = getattr(AttnWrapperBase, "cur_batch", None) or []
			if len(cur_batch) != local_bsz:
				logging.info(
					f"Rank {self.rank}: GLM-5 whole-model graph capture deferred until "
					"decode wrapper state is bound"
				)
				return
			existing_manager = getattr(self, "_cuda_graph_manager", None)
			existing_signature = self._glm5_whole_model_graph_capture_signature()
			if existing_manager is not None and getattr(self, "_glm5_whole_model_graph", False):
				missing_configured = any(
					not existing_manager.has_bucket_for_all_segments(bucket)
					for bucket in capture_buckets
				)
				if (
					not missing_configured
					and existing_signature == getattr(self, "_glm5_whole_model_graph_signature", None)
				):
					self._glm5_whole_model_graph_capture_attempted_for_batch = True
					return
				if capture_attempted:
					logging.info(
						f"Rank {self.rank}: GLM-5 whole-model CUDA graph state changed after "
						"configure-time capture; using eager decode instead of recapturing"
					)
					return

			from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
				Glm5WholeModelSegment,
				make_glm5_whole_model_graph_segment_name,
			)
			from batchgen.models.glm.glm5.cuda_graph_segments import Glm5FullDsaAttnSegment
			from batchgen.models.glm.glm5.layer_cuda_graph_segments import (
				Glm5DecoderLayerGraphSegment,
			)
			from batchgen.models.glm.glm5.model import Glm5MoE, _GLM5_3D_MTP
			from batchgen.models.glm.glm5.moe_cuda_graph_segments import (
				Glm5MoEGraphBufferPool,
				Glm5MoEGraphSegment,
			)

			primary_manager = getattr(gpu_manager, "primary", gpu_manager)
			aux_manager = getattr(
				gpu_manager,
				"auxiliary",
				getattr(self.core_engine, "gpu_paged_kv_manager_aux", None),
			)
			if aux_manager is None:
				raise RuntimeError("GLM-5 whole-model CUDA graph requested but auxiliary GPU KV manager is missing")
			active_sequence_ids = list(cur_batch)
			primary_manager.ensure_cuda_graph_page_table(active_sequence_ids)
			aux_manager.ensure_cuda_graph_page_table(active_sequence_ids)
			primary_page_table = primary_manager.get_cuda_graph_page_table_storage()
			aux_page_table = aux_manager.get_cuda_graph_page_table_storage()
			if primary_page_table is None or aux_page_table is None:
				raise RuntimeError(
					"GLM-5 whole-model CUDA graph requested but GPU page-table storage is not initialized"
				)
			primary_page_size = int(primary_manager.config.page_size_tokens)
			aux_page_size = int(aux_manager.config.page_size_tokens)
			capacity_seqlen = self._glm5_dsa_graph_score_capacity_tokens(
				primary_page_table,
				primary_page_size,
				aux_page_table,
				aux_page_size,
				model_max_position_embeddings=getattr(self.model_config, "max_position_embeddings", None),
			)
			env_graph_max_seqlen = os.environ.get("BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH_MAX_SEQLEN")
			graph_max_seqlen = int(env_graph_max_seqlen) if env_graph_max_seqlen else int(capacity_seqlen)
			if graph_max_seqlen <= 0:
				raise RuntimeError("BATCHGEN_GLM5_WHOLE_MODEL_CUDA_GRAPH_MAX_SEQLEN must be positive")
			if int(getattr(AttnWrapperBase, "max_seqlen", 0) or 0) > graph_max_seqlen:
				raise RuntimeError(
					f"GLM-5 whole-model CUDA graph max_seqlen={AttnWrapperBase.max_seqlen} "
					f"exceeds cap {graph_max_seqlen}"
				)
			if graph_max_seqlen > int(capacity_seqlen):
				raise RuntimeError(
					f"GLM-5 whole-model CUDA graph max_seqlen={graph_max_seqlen} "
					f"exceeds page-table capacity {capacity_seqlen}"
				)

			AttnWrapperBase.gpu_paged_kv_manager = primary_manager
			AttnWrapperBase.gpu_paged_kv_manager_aux = aux_manager
			primary_k_cache, _ = primary_manager.get_kv_tensors()
			aux_k_cache, _ = aux_manager.get_kv_tensors()
			rank_counts = getattr(self, "_current_decode_rank_token_counts", None)
			if rank_counts is None:
				rank_counts = torch.full(
					(self.world_size,),
					local_bsz,
					dtype=torch.int64,
					device=self.torch_device,
				)
			for layer_idx, decoder_layer in enumerate(self.model.model.layers):
				wrapper = decoder_layer.self_attn
				indexer = getattr(wrapper.module, "indexer", None)
				if indexer is None:
					raise RuntimeError(f"Layer {layer_idx}: GLM-5 whole-model graph requires DSA indexer")
				if getattr(wrapper, "_fp8_absorb_weights", None) is None:
					wrapper.initialize_decode_absorb()
				if getattr(wrapper, "_fused_wqb_weights", None) is None or getattr(wrapper, "_indexer_cuda_module", None) is None:
					wrapper.initialize_fused_kernels()
				if getattr(wrapper, "_fp8_absorb_weights", None) is None:
					raise RuntimeError(f"Layer {layer_idx}: GLM-5 whole-model graph requires FP8 absorb weights")
				if getattr(wrapper, "_fused_wqb_weights", None) is None:
					raise RuntimeError(f"Layer {layer_idx}: GLM-5 whole-model graph requires fused WQB weights")
				if getattr(wrapper, "_indexer_cuda_module", None) is None:
					raise RuntimeError(f"Layer {layer_idx}: GLM-5 whole-model graph requires fused indexer CUDA module")
			moe_not_ready = []
			for layer_idx, decoder_layer in enumerate(self.model.model.layers):
				mlp = getattr(decoder_layer, "mlp", None)
				if hasattr(mlp, "experts_per_rank") and not getattr(mlp, "_fp8_blockwise_ready", False):
					moe_not_ready.append(layer_idx)
			if moe_not_ready:
				reason = (
					"GLM-5 whole-model CUDA graph requires all local MoE experts "
					"to be persistent and stacked for the 3D FP8 path; unavailable "
					f"for layers {moe_not_ready[:5]}{'...' if len(moe_not_ready) > 5 else ''}. "
					"Single-node partial-persistent expert configs can run eager/mixed "
					"decode, but cannot validate the real whole-model graph."
				)
				self._glm5_whole_model_graph_unavailable_reason = reason
				if whole_graph_required:
					raise RuntimeError(reason)
				logging.warning("%s Using eager decode without whole-model graph compare.", reason)
				return

			if getattr(self, "_glm5_whole_model_graph", False) or getattr(self, "_whole_model_segment", None) is not None:
				logging.info(
					f"Rank {self.rank}: releasing stale GLM-5 whole-model CUDA graph "
					"before configure-time capture"
				)
				self._release_glm5_whole_model_graph_state(empty_cuda_cache=True)

			manager = CUDAGraphManager(bucketing, device=self.torch_device)
			moe_layers = [
				layer.mlp for layer in self.model.model.layers
				if isinstance(getattr(layer, "mlp", None), Glm5MoE)
			]
			moe_pool = None
			if moe_layers:
				first_moe = moe_layers[0]
				moe_pool = Glm5MoEGraphBufferPool(
					world_size=self.world_size,
					hidden_size=first_moe.hidden_size,
					num_experts_per_tok=first_moe.num_experts_per_tok,
					num_local_experts=first_moe.experts_per_rank,
					intermediate_size=first_moe.config.moe_intermediate_size,
					device=self.torch_device,
					bucket_sizes=bucket_sizes,
					base_mtp=_GLM5_3D_MTP,
				)
			shared_dsa_buffers = {}
			layer_segments = []
			for layer_idx, decoder_layer in enumerate(self.model.model.layers):
				wrapper = decoder_layer.self_attn
				indexer = wrapper.module.indexer
				primary_blocked_k = primary_k_cache[layer_idx]
				aux_blocked_k = aux_k_cache[layer_idx]
				dummy = torch.empty(
					1,
					1,
					indexer.rope_head_dim,
					device=primary_blocked_k.device,
					dtype=torch.bfloat16,
				)
				cos_table, sin_table = indexer.rotary_emb(dummy, seq_len=graph_max_seqlen)
				dsa_segment = Glm5FullDsaAttnSegment(
					wrapper=wrapper,
					primary_blocked_k=primary_blocked_k,
					aux_blocked_k=aux_blocked_k,
					primary_page_table=primary_page_table,
					aux_page_table=aux_page_table,
					wq_b_weights=wrapper._fused_wqb_weights,
					absorb_weights=wrapper._fp8_absorb_weights,
					cuda_module=wrapper._indexer_cuda_module,
					cos_table=cos_table,
					sin_table=sin_table,
					max_seqlen=graph_max_seqlen,
					index_topk=indexer.index_topk,
					page_size=primary_page_size,
					aux_page_size=aux_page_size,
					shared_buffers=shared_dsa_buffers,
				)
				moe_segment = None
				moe = getattr(decoder_layer, "mlp", None)
				if isinstance(moe, Glm5MoE):
					moe_segment = Glm5MoEGraphSegment(
						moe,
						moe_pool,
						moe.comm,
						world_size=self.world_size,
						rank=self.rank,
						device=self.torch_device,
					)
				layer_segments.append(
					Glm5DecoderLayerGraphSegment(
						layer=decoder_layer,
						dsa_segment=dsa_segment,
						moe_segment=moe_segment,
						device=self.torch_device,
						world_size=self.world_size,
						capture_local_bsz=local_bsz,
						capture_rank_token_counts=rank_counts,
					)
				)
			vocab_size = getattr(self.model, 'vocab_size', None) or self.model.config.vocab_size
			hidden_size = self.model.config.hidden_size
			probe_layers_env = os.environ.get("BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_PROBE_LAYERS", "")
			if probe_layers_env.strip().lower() == "all":
				compare_probe_layers = tuple(range(len(self.model.model.layers)))
			elif probe_layers_env.strip():
				compare_probe_layers = tuple(
					int(part.strip())
					for part in probe_layers_env.split(",")
					if part.strip()
				)
			else:
				compare_probe_layers = ()
			whole_seg = Glm5WholeModelSegment(
				model=self.model,
				device=self.torch_device,
				world_size=self.world_size,
				max_pages_per_seq=primary_page_table.shape[1],
				max_aux_pages_per_seq=aux_page_table.shape[1],
				vocab_size=vocab_size,
				hidden_size=hidden_size,
				max_bucket_size=bucketing._max_bucket,
				max_seqlen=graph_max_seqlen,
				include_embedding=True,
				include_lm_head=True,
				compare_probe_layers=compare_probe_layers,
				layer_segments=layer_segments,
			)
			segment_name = make_glm5_whole_model_graph_segment_name()
			manager.register_segment(segment_name, whole_seg)
			first_wrapper = self.model.model.layers[0].self_attn.module
			num_heads = int(getattr(first_wrapper, "num_heads", 64))
			logging.info(
				f"Rank {self.rank}: capturing GLM-5 whole-model CUDA graph "
				f"segment={segment_name} buckets={capture_buckets}, "
				f"max_seqlen_cap={graph_max_seqlen}"
			)
			torch.cuda.synchronize(self.torch_device)
			dist.barrier()
			self._cuda_graph_manager = manager
			self._whole_model_graph = True
			self._glm5_whole_model_graph = True
			self._whole_model_bucketing = bucketing
			self._whole_model_segment = whole_seg
			self._glm5_whole_model_graph_capture_attempted_for_batch = True
			try:
				for capture_bucket in capture_buckets:
					torch.cuda.synchronize(self.torch_device)
					dist.barrier()
					whole_seg.set_capture_inputs(
						**self._make_glm5_whole_model_capture_inputs(
							bucket=capture_bucket,
							num_heads=num_heads,
						)
					)
					manager.warmup_and_capture_buckets([capture_bucket])
					torch.cuda.synchronize(self.torch_device)
					dist.barrier()
			except torch.OutOfMemoryError as exc:
				manager.drop_bucket(capture_bucket)
				self._glm5_whole_model_graph_failed_buckets.add(capture_bucket)
				self._cuda_graph_manager = None
				self._whole_model_segment = None
				self._whole_model_bucketing = None
				self._glm5_whole_model_capture_input_ids = None
				self._whole_model_graph = False
				self._glm5_whole_model_graph = False
				torch.cuda.empty_cache()
				if whole_graph_required:
					raise
				logging.error(
					f"Rank {self.rank}: GLM-5 whole-model CUDA graph configure-time "
					f"capture for bucket BS={capture_bucket} ran out of memory; using eager decode: {exc}"
				)
				return
			self._glm5_whole_model_graph_signature = self._glm5_whole_model_graph_capture_signature()
			# Phase B: hand the just-captured segment to the adapter so its
			# eligibility() / prepare_replay_inputs() / stage_post_graph_kv()
			# can run against the same captured graph the legacy path uses.
			# Without this attach, adapter._ctx stays None and eligibility()
			# returns EAGER/adapter_not_built every step (the worker silently
			# falls back to the legacy path).
			if self._cuda_graph_adapter is not None:
				try:
					_adapter_max_seqlen = int(getattr(whole_seg, "max_seqlen", 0) or 0)
					_adapter_gpu_mgr = gpu_manager
					_attach = getattr(self._cuda_graph_adapter, "attach_existing_segment", None)
					if _attach is not None:
						_attach(
							model=self.model,
							whole_model_segment=whole_seg,
							bucketing=bucketing,
							gpu_kv_manager=_adapter_gpu_mgr,
							device=self.torch_device,
							max_seqlen_cap=_adapter_max_seqlen,
						)
					for _bucket in capture_buckets:
						_sig = self._cuda_graph_adapter.capture_signature(
							bucket=int(_bucket),
							gpu_kv_manager=_adapter_gpu_mgr,
							max_seqlen=_adapter_max_seqlen,
						)
						self._cuda_graph_adapter.record_capture(
							segment_name=segment_name,
							bucket=int(_bucket),
							signature=_sig,
						)
				except Exception as _exc:
					logging.warning(
						"Phase B: failed to attach/record_capture on adapter: %s", _exc,
					)
			stats = manager.get_capture_stats()
			logging.info(
				f"Rank {self.rank}: GLM-5 whole-model CUDA graph ready in "
				f"{stats['total_capture_time_ms']:.0f}ms for buckets {capture_buckets}"
			)
			if self._glm5_whole_model_graph_timing_requested_for_current_batch():
				logging.info(
					"[GLM5_WHOLE_GRAPH_TIMING] rank=%s buckets=%s capture_ms=%.3f",
					self.rank,
					capture_buckets,
					stats["total_capture_time_ms"],
				)
			return

		# GPT-OSS-specific pre-warm and per-layer segment registration
		# K2.5 uses MLA (not GQA) and has its own segment class, skip per-layer setup
		has_moe_graph = False
		moe_pool = None
		if not _is_k25:
			# Pre-warm: initialize sinks and RoPE cache before capture
			for layer_idx, decoder_layer in enumerate(self.model.model.layers):
				wrapper = decoder_layer.self_attn
				# Initialize sinks for persistent mode
				if wrapper.sinks is None and wrapper.persistent and hasattr(wrapper.module, 'sinks'):
					wrapper.sinks = wrapper.module.sinks.data.to(self.torch_device)
				elif wrapper.sinks is not None:
					wrapper.sinks = wrapper.sinks.to(self.torch_device)
				# Pre-warm RoPE cos/sin cache to max position embeddings
				dummy = torch.zeros(1, 1, wrapper.num_kv_heads, wrapper.head_dim, device=self.torch_device)
				wrapper.module.rotary_emb(dummy, seq_len=max_rope_len)

			# Register full attention segments
			# Use max possible pages based on max sequence length, not current state.
			# The page_table static buffer column width is baked into the graph —
			# if sequences grow beyond this during decode, FlashAttention reads
			# past the buffer causing illegal memory access.
			page_size_tokens = gpu_manager.config.page_size_tokens
			max_seq_len = self.model.config.max_position_embeddings
			max_pages = (max_seq_len + page_size_tokens - 1) // page_size_tokens
			for layer_idx, decoder_layer in enumerate(self.model.model.layers):
				attn_wrapper = decoder_layer.self_attn
				seg = FullAttnSegment(decoder_layer, attn_wrapper, layer_idx, max_rope_len,
									  max_pages, page_size_tokens)
				manager.register_segment(f"layer_{layer_idx}_full_attn", seg)

			# Register MoE segments with shared buffer pool (EP mode)
			for layer_idx, decoder_layer in enumerate(self.model.model.layers):
				moe_decode = decoder_layer.mlp
				if (hasattr(moe_decode, 'persistent_expert_indices')
						and len(moe_decode.persistent_expert_indices) > 0
						and hasattr(moe_decode, 'comm') and moe_decode.comm is not None):
					# Create shared pool once from first MoE layer's params
					if moe_pool is None:
						moe_pool = SharedMoEBufferPool(
							world_size=self.world_size,
							hidden_size=moe_decode.hidden_size,
							total_experts=moe_decode.total_experts,
							num_experts_per_tok=moe_decode.num_experts_per_tok,
							num_local_experts=len(moe_decode.persistent_expert_indices),
							N_intermediate=moe_decode.gate_weight_ref.shape[0],
							device=self.torch_device,
						)
						moe_pool.setup(bucketing.bucket_sizes)
					moe_seg = MoESegment(
						moe_decode, moe_pool, moe_decode.comm,
						self.world_size, self.rank, self.torch_device,
					)
					decoder_layer._moe_segment = moe_seg
					decoder_layer._moe_bucketing = bucketing
					# Register compute segment for graph capture.
					# All_gather is graph-captured; all_reduce remains eager.
					if not os.environ.get("BATCHGEN_MOE_EAGER"):
						moe_compute_seg = MoEComputeSegment(
							moe_decode, moe_pool, moe_decode.comm,
							self.world_size, self.rank, self.torch_device,
						)
						manager.register_segment(f"layer_{layer_idx}_moe", moe_compute_seg)
						has_moe_graph = True
		else:
			# K2.5: Pre-warm RoPE cache (shared instance)
			rotary_emb = self.model.model._shared_rotary_emb
			dummy = torch.zeros(1, 1, 1, rotary_emb.dim, device=self.torch_device)
			rotary_emb(dummy, seq_len=max_rope_len)
			# Compute max_pages for K2.5
			page_size_tokens = gpu_manager.config.page_size_tokens
			max_seq_len = self.model.config.max_position_embeddings
			max_pages = (max_seq_len + page_size_tokens - 1) // page_size_tokens

		# Set gpu_paged_kv_manager so segments can access it during capture
		AttnWrapperBase.gpu_paged_kv_manager = gpu_manager

		# K2.5: whole-model graph via the cuda-graph adapter (minimal parallel
		# path). POIS decision: serialize the shared expert inline to reclaim
		# the eager-MoE launch overhead. Self-contained; leaves the GLM-5 /
		# GPT-OSS whole-model paths byte-for-byte untouched. Falls back to the
		# legacy per-layer K2.5 path below if no adapter is present.
		if _is_k25 and self._cuda_graph_adapter is not None:
			from batchgen.models.moonshotai.kimi_k25.cuda_graph_adapter import (
				SEGMENT_NAME_WHOLE_MODEL as _K25_WM_NAME,
			)
			bundle = self._cuda_graph_adapter.build_segments(
				model=self.model,
				bucketing=bucketing,
				gpu_kv_manager=gpu_manager,
				world_size=self.world_size,
				rank=self.rank,
				device=self.torch_device,
				max_seqlen_cap=max_rope_len,
			)
			whole_seg = bundle.whole_model
			manager.register_segment(_K25_WM_NAME, whole_seg)
			if self.rank == 0:
				logging.info(
					f"CUDA graph capture (K2.5 whole-model): segment={_K25_WM_NAME} "
					f"× {len(bucketing.bucket_sizes)} buckets {bucketing.bucket_sizes}"
				)
			# NCCL collectives are baked into the graph — all ranks must capture
			# the same buckets simultaneously.
			torch.cuda.synchronize(self.torch_device)
			dist.barrier()
			manager.warmup_and_capture_all()
			for _bucket in bucketing.bucket_sizes:
				_sig = self._cuda_graph_adapter.capture_signature(
					bucket=_bucket, gpu_kv_manager=gpu_manager, max_seqlen=max_rope_len,
				)
				self._cuda_graph_adapter.record_capture(
					segment_name=_K25_WM_NAME, bucket=_bucket, signature=_sig,
				)
			self._cuda_graph_manager = manager
			self._whole_model_graph = True
			self._k25_whole_model_graph = True
			self._whole_model_segment_name = _K25_WM_NAME
			self._whole_model_bucketing = bucketing
			self._whole_model_segment = whole_seg
			if self.rank == 0:
				stats = manager.get_capture_stats()
				logging.info(
					f"CUDA graphs ready (K2.5 whole-model): {stats['total_capture_time_ms']:.0f}ms"
				)
			return

		# Whole-model graph is the default for GPT-OSS.
		# K2.5 without an adapter falls back to per-layer (segmented) mode.
		if _is_k25:
			use_whole_model = False
		else:
			use_whole_model = True

		if use_whole_model:
			# Whole-model mode: single graph for entire decode pass.
			# Discard per-layer segments, register one WholeModelSegment instead.
			manager = CUDAGraphManager(bucketing, device=self.torch_device)

			vocab_size = getattr(self.model, 'vocab_size', None) or self.model.config.vocab_size
			hidden_size = self.model.config.hidden_size

			if _is_k25:
				# K2.5 uses MLA attention + 3D strided MoE — different segment class
				from batchgen.models.moonshotai.kimi_k25.cuda_graph_segments import K25WholeModelSegment
				whole_seg = K25WholeModelSegment(
					model=self.model,
					device=self.torch_device,
					max_pages_per_seq=max_pages,
					vocab_size=vocab_size,
					hidden_size=hidden_size,
					max_bucket_size=bucketing._max_bucket,
				)
			else:
				# GPT-OSS: GQA attention + SharedMoEBufferPool
				# Build MoE segments dict (layer_idx → MoESegment) for WholeModelSegment.
				# Use MoESegment (not MoEComputeSegment) because it includes all_reduce
				# inside the graph — required for single-graph whole-model capture.
				moe_segments = {}
				for layer_idx, decoder_layer in enumerate(self.model.model.layers):
					moe_decode = decoder_layer.mlp
					if (hasattr(moe_decode, 'persistent_expert_indices')
							and len(moe_decode.persistent_expert_indices) > 0
							and hasattr(moe_decode, 'comm') and moe_decode.comm is not None):
						moe_segments[layer_idx] = MoESegment(
							moe_decode, moe_pool, moe_decode.comm,
							self.world_size, self.rank, self.torch_device,
						)

				whole_seg = WholeModelSegment(
					model=self.model,
					moe_pool=moe_pool,
					moe_segments=moe_segments,
					device=self.torch_device,
					max_pages_per_seq=max_pages,
					vocab_size=vocab_size,
					hidden_size=hidden_size,
					max_bucket_size=bucketing._max_bucket,
				)

			manager.register_segment("whole_model", whole_seg)

			if self.rank == 0:
				logging.info(
					f"CUDA graph capture: whole-model × "
					f"{len(bucketing.bucket_sizes)} buckets {bucketing.bucket_sizes}"
				)

			# Sync all ranks — NCCL collectives require simultaneous participation
			torch.cuda.synchronize(self.torch_device)
			dist.barrier()

			manager.warmup_and_capture_all()

			# Reset capture mode flags
			for layer in self.model.model.layers:
				layer._graph_capture_mode = False

			self._cuda_graph_manager = manager
			self._whole_model_graph = True
			self._whole_model_bucketing = bucketing
			self._whole_model_segment = whole_seg
			if self.rank == 0:
				stats = manager.get_capture_stats()
				logging.info(
					f"CUDA graphs ready (whole-model): {stats['total_capture_time_ms']:.0f}ms"
				)
		else:
			# Per-layer mode: capture attention graph per layer, MoE stays eager.
			self._whole_model_graph = False

			if _is_k25:
				# K2.5: Register K25AttnSegment per layer (MLA attention only, no MoE graph).
				# MoE stays eager to preserve async shared expert overlap.
				# Each rank uses local batch_size for bucket selection (DP-attention, no NCCL).
				from batchgen.models.moonshotai.kimi_k25.cuda_graph_segments import K25AttnSegment

				for layer_idx, decoder_layer in enumerate(self.model.model.layers):
					attn_wrapper = decoder_layer.self_attn
					seg = K25AttnSegment(
						decoder_layer, attn_wrapper, layer_idx,
						max_seq_len=max_rope_len,
						max_pages_per_seq=max_pages,
						page_size_tokens=page_size_tokens,
					)
					seg_name = f"layer_{layer_idx}_attn"
					manager.register_segment(seg_name, seg)

				if self.rank == 0:
					logging.info(
						f"CUDA graph capture (K2.5 MLA): {len(self.model.model.layers)} layers (attn only) × "
						f"{len(bucketing.bucket_sizes)} buckets {bucketing.bucket_sizes}"
					)

				manager.warmup_and_capture_all()

				# Enable graph mode on each decoder layer
				for layer_idx, decoder_layer in enumerate(self.model.model.layers):
					decoder_layer.enable_cuda_graph(
						manager,
						attn_name=f"layer_{layer_idx}_attn",
						max_pages_per_seq=max_pages,
					)
			else:
				# GPT-OSS: per-layer mode (existing behavior)
				if self.rank == 0:
					num_segs = "attn+moe" if has_moe_graph else "attn"
					logging.info(
						f"CUDA graph capture: {len(self.model.model.layers)} layers ({num_segs}) × "
						f"{len(bucketing.bucket_sizes)} buckets {bucketing.bucket_sizes}"
					)

				# Sync all ranks before warmup — MoE segments use NCCL collectives
				# which require all ranks to participate simultaneously.
				if has_moe_graph:
					torch.cuda.synchronize(self.torch_device)
					dist.barrier()

				manager.warmup_and_capture_all()

				# Enable graph mode on each decoder layer
				for layer_idx, decoder_layer in enumerate(self.model.model.layers):
					moe_name = f"layer_{layer_idx}_moe" if has_moe_graph else None
					decoder_layer.enable_cuda_graph(
						manager,
						full_attn_name=f"layer_{layer_idx}_full_attn",
						moe_name=moe_name,
					)

			self._cuda_graph_manager = manager
			if self.rank == 0:
				stats = manager.get_capture_stats()
				logging.info(
					f"CUDA graphs ready: {stats['total_capture_time_ms']:.0f}ms"
				)

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
		
		from batchgen.models.glm.glm5.cuda_graph_policy import (
			glm5_effective_decode_attn_mode,
		)

		RUNTIME_ATTN_MODE = glm5_effective_decode_attn_mode(
			getattr(self.model_config, "model_type", None),
			self.engine_config.Basic_Config.attn_mode,
		)
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
		if isinstance(gpu_manager, DualKVCacheCoordinator):
			AttnWrapperBase.gpu_paged_kv_manager = gpu_manager.primary
			AttnWrapperBase.gpu_paged_kv_manager_aux = gpu_manager.auxiliary
		else:
			AttnWrapperBase.gpu_paged_kv_manager = gpu_manager
			AttnWrapperBase.gpu_paged_kv_manager_aux = None
		AttnWrapperBase.host_paged_kv_worker_view = worker_view
		AttnWrapperBase.host_paged_kv_worker_view_aux = getattr(self, "host_paged_kv_worker_view_aux", None)
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
		max_batch_size = self._sync_decode_moe_rank_counts(batch, reason="decode_entry")

		# OPTIMIZATION: Track if page table was verified since last batch change
		# Avoids redundant page table checks between boundaries
		_page_table_verified_this_batch = True  # Start True after entry check

		# P0: Pre-allocate pinned memory and one reusable completion event for the
		# only mandatory steady-state GPU→CPU dependency: the sampled token IDs.
		_new_tokens_pinned = torch.empty(max(max_batch_size, 1), 1, dtype=torch.long, pin_memory=True)
		_new_tokens_ready = torch.cuda.Event()

		# Heartbeat state for the rate-limited [DECODE] progress line below
		_hb_last_time = time.perf_counter()
		_hb_tokens = 0

		# Main decode loop — enable decode watchdog for monitoring
		self.enable_decode_watchdog()
		while decode_uuids:
			local_iteration += 1
			self._cumulative_decode_iterations += 1

			# Feed watchdogs to prevent timeout during long decoding
			self.feed_watchdog()
			self.feed_decode_watchdog()

			# Rate-limited decode heartbeat (rank 0, ~every 30 s) so the log
			# monitor sees liveness during long decode phases
			_hb_tokens += len(decode_uuids)
			if self.rank == 0 and time.perf_counter() - _hb_last_time >= 30.0:
				_hb_elapsed = time.perf_counter() - _hb_last_time
				_hb_finished = len(self.global_batch.get_sequences_by_status(SequenceStatus.COMPLETED))
				logging.info(
					f"[DECODE] step={self._cumulative_decode_iterations} "
					f"active={len(decode_uuids)} finished={_hb_finished} "
					f"tok/s={_hb_tokens / _hb_elapsed:.2f}"
				)
				_hb_last_time = time.perf_counter()
				_hb_tokens = 0

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
					num_waited = self._wait_pending_kv_append_tasks(sync_distributed_errors=True)
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

				# Poll for new admissions at each page boundary.
				# New batches may have been submitted during decode — drain them
				# and break for prefill if QUEUEING sequences arrive.
				if self._admission_queue is not None:
					admitted = self._poll_admissions()
					if admitted and self.rank == 0:
						logging.info(
							f"[DECODE] Mid-decode admission at iter {self._cumulative_decode_iterations}, "
							f"total in batch: {len(self.global_batch)}"
						)
					has_q = self.global_batch.has_queueing()
					if BATCHGEN_MULTI_BATCH_DIAG and self.rank == 0 and has_q:
						num_q = len(self.global_batch.get_sequences_by_status(SequenceStatus.QUEUEING))
						logging.info(
							f"[MULTI_DIAG] has_queueing={has_q} num_q={num_q} "
							f"watermark={watermark_triggered} admitted={admitted}"
						)
					if has_q and watermark_triggered:
						if self.rank == 0:
							logging.info(f"[DECODE] Breaking for new batch prefill (watermark triggered)")
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
						self._sync_decode_moe_rank_counts(
							batch,
							reason="post_pending_load_finalize",
						)
						
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
			global_decode_sequences = self._debug_sequences_for_decode_uuids(decode_uuids)
			AttnWrapperBase.batchgen_debug = self._active_batchgen_debug_for_sequences(
				global_decode_sequences
			)
			self._configure_glm5_dispatch_trace(global_decode_sequences)

			# Phase C: MoE-only graph mode retired; no MoE-specific warmup needed.

			# Invariant check: cache_seqlens must not exceed allocated pages.
			# Violations cause FlashAttention to read -1 sentinel → CUDA illegal access.
			if BATCHGEN_DECODE_ASSERT and batch:
				for seq in batch_sequences:
					max_tokens = seq.gpu_pages_allocated * SequenceEntry.PAGE_SIZE
					if seq.current_context_length > max_tokens:
						logging.error(
							f"DECODE_ASSERT FAIL rank={self.rank}: {seq.uuid[:8]} gid={seq.global_idx} "
							f"ctx={seq.current_context_length} > max_tokens={max_tokens} "
							f"(pages={seq.gpu_pages_allocated}, PAGE_SIZE={SequenceEntry.PAGE_SIZE}, "
							f"prompt={seq.prompt_length}, orig_prompt={seq.original_prompt_length}, "
							f"decoded={seq.decoded_length}, baseline={seq.reentry_decoded_baseline}, "
							f"status={seq.status})"
						)
						raise RuntimeError(
							f"cache_seqlens overrun: ctx={seq.current_context_length} > "
							f"pages={seq.gpu_pages_allocated}×{SequenceEntry.PAGE_SIZE}="
							f"{max_tokens} for {seq.uuid[:8]}"
						)

			with torch.inference_mode():
				if batch:
					# Collect context lengths with invariant validation
					# ALWAYS: current_context_length == original_prompt_length + decoded_length
					cache_seqlens = []
					for seq in batch_sequences:
						ctx_len = seq.current_context_length
						expected = seq.original_prompt_length + seq.decoded_length
						if ctx_len != expected:
							logging.error(
								f"Rank {self.rank}: CTX MISMATCH {seq.uuid[:8]} gid={seq.global_idx}: "
								f"ctx={ctx_len} expected={expected} (orig_prompt={seq.original_prompt_length}, "
								f"prompt={seq.prompt_length}, decoded={seq.decoded_length})"
							)
							seq.log_event(SeqEvent.CTX_MISMATCH, self.rank,
								f"ctx={ctx_len}, expected={expected}, prompt={seq.prompt_length}")
							lifespan.dump_lifespan(seq.uuid, seq.global_idx, seq._lifespan_log, "CTX_MISMATCH")
							seq.current_context_length = expected
							ctx_len = expected
						cache_seqlens.append(ctx_len)

					max_ctx = max(cache_seqlens)

					# DIAG: Log cache_seqlens at first iteration of each decode group
					if BATCHGEN_MULTI_BATCH_DIAG and self.rank == 0 and local_iteration <= 1:
						fresh = [(s.uuid[:8], s.decoded_length, ctx) for s, ctx in zip(batch_sequences, cache_seqlens) if s.decoded_length <= 1]
						resumed = [(s.uuid[:8], s.decoded_length, ctx, s.gpu_pages_allocated) for s, ctx in zip(batch_sequences, cache_seqlens) if s.decoded_length > 1]
						logging.info(
							f"[MULTI_DIAG] decode_group={self._decode_group_idx} iter={local_iteration}: "
							f"batch={len(batch)}, fresh={len(fresh)}, resumed={len(resumed)}, "
							f"max_ctx={max_ctx}"
						)
						for uid, dl, ctx in fresh[:5]:
							logging.info(f"[MULTI_DIAG]   FRESH: {uid} decoded={dl} cache_seqlen={ctx}")
						for uid, dl, ctx, pg in resumed[:5]:
							logging.info(f"[MULTI_DIAG]   RESUMED: {uid} decoded={dl} cache_seqlen={ctx} gpu_pages={pg}")

					Attn_Wrapper.attention_mask = None  # Removed: no longer used in decode
					(
						Attn_Wrapper.cache_seqlens,
						Attn_Wrapper.position_ids,
					) = self._bind_decode_attention_metadata(batch_sequences, cache_seqlens)
					Attn_Wrapper.max_seqlen = max_ctx

					# CRITICAL: Also bind to AttnWrapperBase for models using new wrapper system (GPT-OSS)
					AttnWrapperBase.attention_mask = None  # Removed: no longer used in decode
					AttnWrapperBase.cache_seqlens = Attn_Wrapper.cache_seqlens
					AttnWrapperBase.position_ids = Attn_Wrapper.position_ids
					AttnWrapperBase.max_seqlen = max_ctx

					# Per-step DSA dispatch hint: count sequences whose cache is
					# short enough to take the dense short-circuit instead of
					# indexer scoring. Computing once here instead of inside
					# every layer's _forward_decode_dsa drops 77 of 78 D2H syncs
					# per decode step on DSA models (GLM-5).
					_dsa_index_topk = getattr(self.model_config, "index_topk", None)
					if _dsa_index_topk is not None:
						GLM5AttnWrapper._dsa_short_count = int(
							(Attn_Wrapper.cache_seqlens <= _dsa_index_topk).sum().item()
						)
					else:
						GLM5AttnWrapper._dsa_short_count = None

						# GLM-5.2 DSA indexer reuse: clear prev top-k once per decode
						# step (before layer 0) so shared layers never reuse a stale
						# value from the previous step. (Second decode path; the
						# graph-config path resets it separately.)
						GLM5AttnWrapper._dsa_prev_topk_indices = None

					if new_tokens.shape[0] != len(batch):
						new_tokens = self._rebuild_input_tokens(batch)
				else:
					Attn_Wrapper.attention_mask = None
					Attn_Wrapper.position_ids = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
					Attn_Wrapper.cache_seqlens = torch.zeros((0,), dtype=torch.int32, device=self.torch_device)
					Attn_Wrapper.max_seqlen = 0
					Attn_Wrapper.cur_batch = []
					new_tokens = torch.zeros((0, 1), dtype=torch.int64, device=self.torch_device)
					# Also bind empty state to AttnWrapperBase for GPT-OSS
					AttnWrapperBase.attention_mask = None
					AttnWrapperBase.position_ids = Attn_Wrapper.position_ids
					AttnWrapperBase.cache_seqlens = Attn_Wrapper.cache_seqlens
					AttnWrapperBase.max_seqlen = 0
					AttnWrapperBase.cur_batch = []
					GLM5AttnWrapper._dsa_short_count = 0
					GLM5AttnWrapper.glm5_dsa_graph_forward_state = None
					GLM5AttnWrapper.glm5_dsa_flashmla_graph_metadata = None
					self._decode_metadata_batch_key = None
					self._decode_metadata_cpu_seqlens = None
				
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
								# Log page rebuild for affected sequences
								for seq in batch_sequences:
									seq.log_event(SeqEvent.PAGE_REBUILD, self.rank,
										f"batch_size={len(batch)}")
						_page_table_verified_this_batch = True
				
				# NOTE: Do NOT skip forward pass even with empty batch!
				# MoE models have all-to-all collective operations that ALL ranks must participate in.
				# Skipping would cause deadlock as other ranks wait for this rank.

				# MoE buffer sync: only needed at decision boundaries (batch size changes).
				# Between boundaries, batch size is constant — skip the all_reduce + .item()
				# CPU-GPU sync that drains the GPU pipeline every step.
				# The sync is done in _page_boundary_fast and at initial setup (line ~7099).
				# Phase C: layer-graph mode retired; only whole-model needs the
				# globally-synced rank-count reuse.
				if (
					getattr(self, '_whole_model_graph', False)
					or self._glm5_whole_model_graph_requested_for_current_batch()
				):
					# Whole-model graph needs globally synced counts for NCCL bucket
					# matching, but the count vector only changes at decode-entry,
					# page-boundary, and async-load-finalize sync points. Reusing it
					# avoids a per-token NCCL all_gather + D2H .item() sync.
					_all_rank_counts = getattr(self, "_current_decode_rank_token_counts", None)
					_cached_local_bsz = int(getattr(self, "_current_decode_local_batch_size", -1))
					_max_bs = int(getattr(self, "_current_decode_max_rank_batch_size", 0) or 0)
					if _all_rank_counts is None or _max_bs <= 0 or _cached_local_bsz != len(batch):
						_max_bs = self._sync_decode_moe_rank_counts(
							batch,
							reason="decode_step_batch_change",
						)
						_all_rank_counts = getattr(self, "_current_decode_rank_token_counts", None)
					_max_bs = max(int(_max_bs), 1)
				else:
					# Per-layer graph or eager: no NCCL in graph, use local batch size
					_max_bs = max(len(batch), 1)
					_all_rank_counts = None

				# KV append callback — deferred: accumulate during forward, single sync after
				current_batch = list(batch)
				_kv_worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)

				if _kv_worker_view is not None:
					_kv_seq_ids = []
					_kv_seq_lengths = []
					for local_idx in current_batch:
						uuid = self._local_to_uuid_map[local_idx]
						seq = self.global_batch.get_sequence(uuid)
						_kv_seq_ids.append(seq.global_idx)
						_kv_seq_lengths.append(seq.current_context_length - 1)
					self._deferred_kv_batch = (_kv_seq_ids, _kv_seq_lengths)
					self._deferred_kv_entries = []
					self._deferred_kv_entries_aux = []
					self._deferred_kv_worker_view = _kv_worker_view
					self._deferred_kv_worker_view_aux = getattr(self, "host_paged_kv_worker_view_aux", None)

				if BATCHGEN_SYNC_KV and _kv_worker_view is not None:
					# SYNC MODE: Immediately write each layer's KV to host (no deferral)
					_sync_kv_seq_ids = _kv_seq_ids
					_sync_kv_seq_lengths = _kv_seq_lengths
					_sync_kv_worker_view = _kv_worker_view
					def kv_append_callback(layer_idx: int, k_tensor: torch.Tensor, v_tensor: torch.Tensor = None):
						if k_tensor.dim() == 3:
							k_tensor = k_tensor.unsqueeze(2)
						if v_tensor is not None and v_tensor.dim() == 3:
							v_tensor = v_tensor.unsqueeze(2)
						torch.cuda.synchronize(self.torch_device)
						task = _sync_kv_worker_view.async_append_decode_kv_to_host(
							layer_idx=layer_idx,
							sequence_ids=_sync_kv_seq_ids,
							k_tensor=k_tensor,
							v_tensor=v_tensor,
							sequence_lengths=_sync_kv_seq_lengths,
						)
						if task is not None:
							task.wait()
				else:
					def kv_append_callback(layer_idx: int, k_tensor: torch.Tensor, v_tensor: torch.Tensor = None):
						self._deferred_kv_entries.append((layer_idx, k_tensor, v_tensor))

				Attn_Wrapper.kv_append_callback = kv_append_callback
				# Also bind to AttnWrapperBase for models using new wrapper system (e.g., GPT-OSS)
				AttnWrapperBase.kv_append_callback = kv_append_callback

				# DSA: auxiliary KV append callback for indexer host cache.
				# In deferred mode (BATCHGEN_SYNC_KV=0, the default) layers push
				# to _deferred_kv_entries_aux; a single event.synchronize in
				# _flush_deferred_kv_to_host covers both primary and aux caches.
				aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)
				if aux_view is not None:
					if BATCHGEN_SYNC_KV:
						def kv_append_callback_aux(layer_idx: int, k_tensor: torch.Tensor, v_tensor: torch.Tensor = None):
							self._append_decode_kv_to_host_aux_async(layer_idx, current_batch, k_tensor, v_tensor)
					else:
						def kv_append_callback_aux(layer_idx: int, k_tensor: torch.Tensor, v_tensor: torch.Tensor = None):
							self._deferred_kv_entries_aux.append((layer_idx, k_tensor, v_tensor))
					AttnWrapperBase.kv_append_callback_aux = kv_append_callback_aux
				else:
					AttnWrapperBase.kv_append_callback_aux = None

				# Phase C: layer-graph mode retired; only the whole-model graph
				# may need re-warmup here.
				if self._glm5_whole_model_graph_current_bucket_missing():
					logging.info(
						f"Rank {self.rank}: GLM-5 whole-model CUDA graph was not captured "
						"during decode configuration; using eager decode instead of "
						"capturing in the decode loop"
					)
					self._glm5_whole_model_graph_capture_attempted_for_batch = True

				self._log_glm5_graph_path_for_forward(
					local_bsz=len(batch),
					max_rank_bsz=int(getattr(self, "_current_decode_max_rank_batch_size", 0) or 0),
					rank_counts=getattr(self, "_current_decode_rank_token_counts", None),
					gpu_manager=gpu_manager,
					decode_iter=self._cumulative_decode_iterations,
				)
				# Phase C: DSA-only graph metadata prep retired (DSA graph mode
				# is no longer reachable). The whole-model graph builds its
				# FlashMLA metadata in-line during prepare_replay_inputs.

				_nsys_forward_idx = self._nsys_decode_profile_begin_forward(
					local_iteration=local_iteration,
					local_bsz=len(batch),
					max_rank_bsz=int(getattr(self, "_current_decode_max_rank_batch_size", 0) or 0),
				)

				# Forward
				# Phase C: layer-graph mode retired. The whole-model graph
				# composes per-layer captures internally; no separate
				# layer-graph dispatch is needed.

				_glm5_whole_graph_active = bool(
					getattr(self, "_glm5_whole_model_graph", False)
					and self._cuda_graph_manager is not None
				)
				_glm5_whole_graph_over_bucket = False
				if _glm5_whole_graph_active:
					try:
						_glm5_whole_bucket = self._whole_model_bucketing.get_padded_size(_max_bs)
					except ValueError:
						_glm5_whole_graph_over_bucket = True
						_glm5_whole_graph_active = False
					else:
						_glm5_whole_graph_active = (
							_glm5_whole_bucket not in getattr(self, "_glm5_whole_model_graph_failed_buckets", set())
							and self._cuda_graph_manager.has_bucket_for_all_segments(_max_bs)
							and int(getattr(AttnWrapperBase, "max_seqlen", 0) or 0) <= int(getattr(self._whole_model_segment, "max_seqlen", 0))
							and self._glm5_whole_model_graph_capture_signature(_glm5_whole_bucket) == getattr(self, "_glm5_whole_model_graph_signature", None)
						)
				elif self._glm5_whole_model_graph_requested_for_current_batch():
					try:
						configured_max_bucket = int(self.args.cuda_graph_max_bucket_size)
					except AttributeError:
						pass
					else:
						_glm5_whole_graph_over_bucket = _max_bs > configured_max_bucket
				if (
					self._glm5_whole_model_graph_requested_for_current_batch()
					and self._glm5_whole_model_graph_requested_for_current_batch()
					and not _glm5_whole_graph_active
					and not _glm5_whole_graph_over_bucket
				):
					_, _required_bucket, _required_reason = self._glm5_whole_graph_path_state(_max_bs)
					raise RuntimeError(
						"GLM-5 whole-model CUDA graph was required but no replayable "
						"whole-model graph is available for this decode step "
						f"(bucket={_required_bucket}, reason={_required_reason})"
					)
				_use_graph = (
					getattr(self, '_whole_model_graph', False)
					and self._cuda_graph_manager is not None
					and _max_bs <= self._whole_model_bucketing._max_bucket
					and (
						not getattr(self, "_glm5_whole_model_graph", False)
						or _glm5_whole_graph_active
					)
				)
				if _use_graph:
					_glm5_whole_compare = bool(
						getattr(self, "_glm5_whole_model_graph", False)
						and self._glm5_whole_model_graph_compare_requested_for_current_batch()
					)
					_glm5_whole_timing = bool(
						getattr(self, "_glm5_whole_model_graph", False)
						and self._glm5_whole_model_graph_timing_requested_for_current_batch()
					)
					_glm5_whole_timing_items = {}
					_glm5_skip_graph_kv_offload = False
					# Whole-model CUDA graph replay.
					# CRITICAL: Use _max_bs (globally-synced max batch size) for bucket
					# computation, NOT local len(batch). The graph has NCCL all_reduce
					# baked inside — all ranks MUST replay the same bucket's graph,
					# otherwise mismatched NCCL ops cause deadlock.
					batch_size = len(batch)
					bucket = self._whole_model_bucketing.get_padded_size(_max_bs)
					# Phase B: dual-path gate. When BATCHGEN_DECODE_GRAPH_ADAPTER_DUAL=1
					# and an adapter is present, route replay through it. Legacy path
					# (default) preserves today's behavior exactly.
					# K2.5 always routes whole-model replay through the adapter
					# (no legacy non-adapter K2.5 whole-model path exists).
					_k25_whole_active = getattr(self, "_k25_whole_model_graph", False)
					_adapter_dual_active = (
						self._cuda_graph_adapter is not None
						and (
							_k25_whole_active
							or (
								self._cuda_graph_adapter_dual
								and getattr(self, "_glm5_whole_model_graph", False)
							)
						)
					)
					_wm_seg_name = getattr(
						self, "_whole_model_segment_name", "glm5_whole_model"
					)
					_adapter_decision = None
					_adapter_batch_state = None
					if _adapter_dual_active:
						from batchgen.cuda_graph.adapter import BatchState as _BatchState
						_adapter_batch_state = _BatchState(
							local_bsz=batch_size,
							max_rank_bsz=_max_bs,
							rank_token_counts=_all_rank_counts,
							cache_seqlens=AttnWrapperBase.cache_seqlens,
							position_ids=AttnWrapperBase.position_ids,
							max_seqlen=int(getattr(AttnWrapperBase, "max_seqlen", 0) or 0),
							cur_batch_sequence_ids=tuple(getattr(AttnWrapperBase, "cur_batch", None) or ()),
							gpu_kv_manager=gpu_manager,
							decode_iter=0,
							input_ids=new_tokens,
							device=self.torch_device,
						)
						_adapter_decision = self._cuda_graph_adapter.eligibility(_adapter_batch_state)
						if _adapter_decision.mode.value != "whole_model":
							logging.info(
								"Phase B: adapter eligibility=%s/%s; using legacy path for parity",
								_adapter_decision.mode.value, _adapter_decision.reason,
							)
							_adapter_dual_active = False
					if _adapter_dual_active:
						# ADAPTER PATH (Phase B dual gate)
						replay_inputs = self._cuda_graph_adapter.prepare_replay_inputs(
							decision=_adapter_decision,
							batch_state=_adapter_batch_state,
							segment_name=_wm_seg_name,
						)
						if _glm5_whole_timing:
							torch.cuda.synchronize(self.torch_device)
							_glm5_replay_start = time.perf_counter()
						graph_out = self._cuda_graph_manager.replay(
							_wm_seg_name, _max_bs, **replay_inputs,
						)
						if _glm5_whole_timing:
							torch.cuda.synchronize(self.torch_device)
							_glm5_whole_timing_items["replay_ms"] = (
								time.perf_counter() - _glm5_replay_start
							) * 1000.0
						# K2.5 whole-model eager-vs-graph compare via the standard
						# model-agnostic facility (observability-only; does NOT change
						# token selection). GLM-5 keeps its own bespoke compare block
						# below; this path serves any adapter that has no such block.
						if _k25_whole_active:
							_dbg = self._cuda_graph_adapter.debug_options(_adapter_batch_state)
							if _dbg.compare_against_eager:
								from batchgen.cuda_graph.compare import compare_decode_outputs
								# graph_out has padded bucket rows; the eager reference
								# produces local_bsz (=batch_size) rows. Slice to align.
								# Graph probe keys are `probe_layer_<NNN>_hidden`; the
								# eager reference uses `hidden_states_layer_<i>` — remap
								# so the diff aligns by key.
								_graph_cmp = {}
								for _k, _v in graph_out.items():
									_vs = _v[:batch_size]
									if _k.startswith("probe_layer_") and _k.endswith("_hidden"):
										_idx = int(_k[len("probe_layer_"):-len("_hidden")])
										_graph_cmp[f"hidden_states_layer_{_idx}"] = _vs
									else:
										_graph_cmp[_k] = _vs
								_report = compare_decode_outputs(
									adapter=self._cuda_graph_adapter,
									decision=_adapter_decision,
									batch_state=_adapter_batch_state,
									segment_name=_wm_seg_name,
									captured_inputs=replay_inputs,
									graph_outputs=_graph_cmp,
									probe_layers=_dbg.probe_layers,
									atol=_dbg.compare_atol,
									rtol=_dbg.compare_rtol,
									fail_on_mismatch=_dbg.fail_on_mismatch,
								)
								_log = logging.info if _report.passed else logging.error
								_log(
									"[K25_WHOLE_GRAPH_COMPARE] rank=%s bucket=%s batch=%s "
									"status=%s max_abs=%.6g max_rel=%.6g mismatched=%s "
									"probes=%s",
									self.rank, _adapter_decision.bucket, batch_size,
									"OK" if _report.passed else "MISMATCH",
									_report.max_abs, _report.max_rel,
									_report.mismatched_keys, _report.probe_results,
								)
					elif getattr(self, "_glm5_whole_model_graph", False):
						primary_manager = getattr(gpu_manager, "primary", gpu_manager)
						aux_manager = getattr(
							gpu_manager,
							"auxiliary",
							getattr(self.core_engine, "gpu_paged_kv_manager_aux", None),
						)
						if aux_manager is None:
							raise RuntimeError("GLM-5 whole-model graph replay requires auxiliary GPU KV manager")
						graph_inputs = self._prepare_glm5_layer_graph_inputs(
							local_bsz=batch_size,
							bucket=bucket,
							gpu_manager=gpu_manager,
							graph_max_seqlen_override=int(getattr(self._whole_model_segment, "max_seqlen", 0) or 0),
						)
						if _glm5_whole_timing:
							torch.cuda.synchronize(self.torch_device)
							_glm5_replay_start = time.perf_counter()
						graph_out = self._cuda_graph_manager.replay(
							"glm5_whole_model", _max_bs,
							input_ids=new_tokens[:batch_size],
							cache_seqlens=graph_inputs["cache_seqlens"],
							position_ids=graph_inputs["position_ids"],
							primary_slot_indices=graph_inputs["primary_slot_indices"],
							aux_slot_indices=graph_inputs["aux_slot_indices"],
							rank_token_counts=_all_rank_counts,
							num_valid_tokens=graph_inputs["num_valid_tokens"],
							flashmla_tile_scheduler_metadata=graph_inputs["flashmla_tile_scheduler_metadata"],
							flashmla_num_splits=graph_inputs["flashmla_num_splits"],
						)
						if _glm5_whole_timing:
							torch.cuda.synchronize(self.torch_device)
							_glm5_whole_timing_items["replay_ms"] = (
								time.perf_counter() - _glm5_replay_start
							) * 1000.0
					else:
						page_table_tensor = gpu_manager._gpu_page_table_manager.gpu_table
						slot_indices_tensor = gpu_manager._gpu_page_table_manager._slot_index_tensor
						if slot_indices_tensor is None:
							# Rebuild may have cleared it; reconstruct as simple arange
							slot_indices_tensor = torch.arange(
								page_table_tensor.shape[0], dtype=torch.int32,
								device=self.torch_device,
							)
						# Page table may have fewer columns than the static buffer
						# (gpu_table gets rebuilt with varying max_pages_per_sequence).
						# Pad to match the captured spec width.
						wm_max_pages = self._whole_model_segment.max_pages_per_seq
						pt_slice = page_table_tensor[:batch_size]
						if pt_slice.shape[1] < wm_max_pages:
							pt_slice = torch.nn.functional.pad(
								pt_slice, (0, wm_max_pages - pt_slice.shape[1]), value=0
							)
						elif pt_slice.shape[1] > wm_max_pages:
							pt_slice = pt_slice[:, :wm_max_pages]
						graph_out = self._cuda_graph_manager.replay(
							"whole_model", bucket,
							input_ids=new_tokens,
							cache_seqlens=AttnWrapperBase.cache_seqlens[:batch_size],
							page_table=pt_slice,
							slot_indices=slot_indices_tensor[:batch_size],
						)

					logits = graph_out["logits"][:batch_size]
					graph_hidden_states = graph_out.get("hidden_states")
					if graph_hidden_states is not None:
						graph_hidden_states = graph_hidden_states[:batch_size]
					if _glm5_whole_compare:
						graph_probe_hidden_states = {
							key: value[:batch_size]
							for key, value in graph_out.items()
							if key.startswith("probe_layer_")
						}
						graph_tokens_for_compare = torch.argmax(logits, dim=-1, keepdim=True)
						if _glm5_whole_timing:
							torch.cuda.synchronize(self.torch_device)
							_glm5_eager_start = time.perf_counter()
						with self._glm5_force_segmented_graph_eager():
							if getattr(self._whole_model_segment, "compare_probe_layers", ()):
								eager_probe_outputs = self._whole_model_segment.run_model_with_probes(
									input_ids=new_tokens,
									attention_mask=Attn_Wrapper.attention_mask,
									position_ids=Attn_Wrapper.position_ids,
									use_layer_segments=False,
								)
								eager_hidden_states = eager_probe_outputs["hidden_states"]
								eager_logits = eager_probe_outputs["logits"]
								eager_probe_hidden_states = {
									key: value
									for key, value in eager_probe_outputs.items()
									if key.startswith("probe_layer_")
								}
							else:
								eager_model_outputs = self.model.model(
									input_ids=new_tokens,
									attention_mask=Attn_Wrapper.attention_mask,
									position_ids=Attn_Wrapper.position_ids,
									use_cache=False,
								)
								eager_hidden_states = eager_model_outputs[0][:, -1, :]
								eager_logits = self.model.lm_head(eager_model_outputs[0])[:, -1, :]
								eager_probe_hidden_states = {}
						if _glm5_whole_timing:
							torch.cuda.synchronize(self.torch_device)
							_glm5_whole_timing_items["eager_ms"] = (
								time.perf_counter() - _glm5_eager_start
							) * 1000.0
						eager_tokens_for_compare = torch.argmax(eager_logits, dim=-1, keepdim=True)
						new_tokens_out = self._select_tokens(eager_logits, batch_sequences)
						from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
							compare_glm5_whole_model_graph_logits,
						)
						compare = compare_glm5_whole_model_graph_logits(
							eager_logits=eager_logits,
							graph_logits=logits,
							eager_hidden_states=eager_hidden_states,
							graph_hidden_states=graph_hidden_states,
							eager_probe_hidden_states=eager_probe_hidden_states,
							graph_probe_hidden_states=graph_probe_hidden_states,
							eager_tokens=eager_tokens_for_compare,
							graph_tokens=graph_tokens_for_compare,
							atol=float(os.environ.get("BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE_ATOL", "1e-2")),
							rtol=float(os.environ.get("BATCHGEN_GLM5_WHOLE_MODEL_GRAPH_COMPARE_RTOL", "1e-2")),
						)
						_log = logging.info if compare["ok"] else logging.error
						_log(
							"[GLM5_WHOLE_GRAPH_COMPARE] rank=%s bucket=%s batch=%s status=%s "
							"max_abs=%.6g mean_abs=%.6g hidden_max_abs=%.6g "
							"hidden_mean_abs=%.6g probe_first_mismatch=%s "
							"probe_max_abs=%.6g probe_mean_abs=%.6g "
							"argmax_mismatch=%s token_mismatch=%s",
							self.rank,
							bucket,
							batch_size,
							"OK" if compare["ok"] else "MISMATCH",
							compare["max_abs"],
							compare["mean_abs"],
							compare["hidden_max_abs"],
							compare["hidden_mean_abs"],
							compare["probe_first_mismatch"],
							compare["probe_max_abs"],
							compare["probe_mean_abs"],
							compare["argmax_mismatch"],
							compare["token_mismatch"],
						)
						if not compare["ok"] and self._glm5_whole_model_graph_compare_fail_on_mismatch():
							raise RuntimeError(f"GLM-5 whole-model CUDA graph compare mismatch: {compare}")
						_glm5_skip_graph_kv_offload = True
					else:
						new_tokens_out = self._select_tokens(logits, batch_sequences)

					if not _glm5_skip_graph_kv_offload and _adapter_dual_active:
						# Phase B: adapter owns post-graph KV staging (audit §A finding #6:
						# contiguous-only clone path; no per-layer fallback branch).
						if _glm5_whole_timing:
							_glm5_offload_start = time.perf_counter()
						self._cuda_graph_adapter.stage_post_graph_kv(
							decision=_adapter_decision,
							batch_state=_adapter_batch_state,
							graph_outputs=graph_out,
						)
						if _glm5_whole_timing:
							_glm5_whole_timing_items["offload_callback_ms"] = (
								time.perf_counter() - _glm5_offload_start
							) * 1000.0
					elif not _glm5_skip_graph_kv_offload:
						if _glm5_whole_timing:
							_glm5_offload_start = time.perf_counter()
						# Fire KV host offload callbacks for all layers.
						# KV buffers are static-address tensors written inside the graph.
						# Stage primary and aux as two contiguous clones before async
						# D2H; cloning per layer adds 156 small GPU copies per decode
						# token on GLM-5 and dominates the whole-graph replay overhead.
						kv_cb = getattr(AttnWrapperBase, 'kv_append_callback', None)
						wm_seg = getattr(self, '_whole_model_segment', None)
						if (
							batch_size > 0
							and kv_cb is not None
							and wm_seg is not None
							and wm_seg._kv_buffers is not None
						):
							primary_stage = None
							primary_key_buffer = getattr(wm_seg, "_kv_key_buffer", None)
							if primary_key_buffer is not None:
								primary_stage = primary_key_buffer[:, :batch_size].clone()
							for layer_idx in range(wm_seg.num_layers):
								kv_buf = wm_seg._kv_buffers[layer_idx]
								# K2.5 MLA has no separate V cache — pass None for v_tensor
								v_buf = kv_buf.get("value")
								v_clone = v_buf[:batch_size].clone() if v_buf is not None and v_buf.numel() > 0 and not getattr(wm_seg, '_no_v_cache', False) else None
								k_tensor = (
									primary_stage[layer_idx]
									if primary_stage is not None
									else kv_buf["key"][:batch_size].clone()
								)
								kv_cb(
									layer_idx,
									k_tensor,
									v_clone,
								)
						aux_cb = getattr(AttnWrapperBase, 'kv_append_callback_aux', None)
						aux_buffers = getattr(wm_seg, "_aux_kv_buffers", None) if wm_seg is not None else None
						if batch_size > 0 and aux_cb is not None and aux_buffers is not None:
							aux_stage = None
							aux_key_buffer = getattr(wm_seg, "_aux_kv_key_buffer", None)
							if aux_key_buffer is not None:
								aux_stage = aux_key_buffer[:, :batch_size].clone()
							for layer_idx in range(wm_seg.num_layers):
								aux_cb(
									layer_idx,
									aux_stage[layer_idx]
									if aux_stage is not None
									else aux_buffers[layer_idx]["key"][:batch_size].clone(),
									None,
								)
						if _glm5_whole_timing:
							_glm5_whole_timing_items["offload_callback_ms"] = (
								time.perf_counter() - _glm5_offload_start
							) * 1000.0
					if _glm5_whole_timing:
						logging.info(
							"[GLM5_WHOLE_GRAPH_TIMING] rank=%s bucket=%s batch=%s replay_ms=%.3f "
							"eager_ms=%.3f offload_callback_ms=%.3f compare=%s",
							self.rank,
							bucket,
							batch_size,
							_glm5_whole_timing_items.get("replay_ms", -1.0),
							_glm5_whole_timing_items.get("eager_ms", -1.0),
							_glm5_whole_timing_items.get("offload_callback_ms", -1.0),
							_glm5_whole_compare,
						)
				else:
					# Per-layer graph or eager forward
					# CRITICAL: Pass position_ids to model to ensure correct RoPE positioning during decode.
					# Without this, the model generates position_ids = [[0]] for all decode steps,
					# causing RoPE to be applied at position 0 instead of the actual token position.
					outputs = self._glm5_decode_model_forward(new_tokens)
					new_tokens_out = self._select_tokens(outputs.logits[:, -1, :], batch_sequences)
				self._nsys_decode_profile_end_forward(_nsys_forward_idx)

			new_tokens = new_tokens_out

			# P1: Non-blocking GPU→CPU token transfer via pinned memory. Record the
			# exact token-readback boundary before launching host-KV copies on their
			# independent D2H stream.
			bs = new_tokens.shape[0]
			if bs > _new_tokens_pinned.shape[0]:
				_new_tokens_pinned = torch.empty(bs, 1, dtype=torch.long, pin_memory=True)
			_new_tokens_pinned[:bs].copy_(new_tokens[:bs], non_blocking=True)
			_new_tokens_ready.record(torch.cuda.current_stream(self.torch_device))

			# Host-KV offload orders its own stream with a device-side event. Wait
			# only for the sampled-token readback required by exact EOS and output
			# bookkeeping; do not drain host-KV copies or the whole CUDA device.
			self._flush_deferred_kv_to_host()
			_new_tokens_ready.synchronize()
			new_tokens_cpu = _new_tokens_pinned[:bs]

			# Update sequences (reuse batch_sequences from forward pass setup)
			for i, (local_idx, seq) in enumerate(zip(batch, batch_sequences)):
				if self._is_sequence_completed(seq):
					continue

				decode_pos = seq.decoded_length
				if BATCHGEN_CB_DEBUG:
					qb_ptr = self.query_book[local_idx].decoded_tokens.data_ptr()
					seq_ptr = seq.decoded_tokens.data_ptr()
					if qb_ptr != seq_ptr:
						logging.error(
							f"Rank {self.rank}: query_book/seq decoded_tokens MISMATCH for "
							f"local_idx={local_idx}, uuid={seq.uuid[:8]}, "
							f"qb_ptr={qb_ptr:#x}, seq_ptr={seq_ptr:#x}"
						)
				self.query_book[local_idx].decoded_tokens[:, decode_pos] = new_tokens_cpu[i]

				seq.decoded_length += 1
				seq.current_context_length += 1

				# Use CPU tensor to avoid GPU sync
				token_id = new_tokens_cpu[i].item()

				# DIAG: Log first 3 tokens for first 10 seqs in each decode group
				if BATCHGEN_MULTI_BATCH_DIAG and self.rank == 0 and local_iteration <= 3 and i < 10:
					logging.info(
						f"[MULTI_DIAG] iter={local_iteration} seq={seq.uuid[:8]} "
						f"decoded_len={seq.decoded_length} token={token_id}"
					)
				if self._should_stop_at_eos(token_id):
					seq.eos_reached = True

				if seq.decoded_length >= seq.max_decode_length:
					seq.eos_reached = True

				# Repetition detection: consecutive same-token check (BATCHGEN_REP_DETECTION=1)
				if REP_DETECTION and not seq._rep_detected:
					if token_id == seq._rep_last_token:
						seq._rep_count += 1
						if seq._rep_count >= 32:
							seq._rep_detected = True
							seq.eos_reached = True
							seq.log_event(SeqEvent.REPETITION, self.rank,
								f"token={token_id}, count={seq._rep_count}")
							lifespan.dump_lifespan(seq.uuid, seq.global_idx,
								seq._lifespan_log, "REPETITION")
							logging.warning(
								f"Rank {self.rank}: REPETITION {seq.uuid} gid={seq.global_idx} "
								f"token={token_id} x{seq._rep_count} at decoded_len={seq.decoded_length}"
							)
					else:
						seq._rep_last_token = token_id
						seq._rep_count = 1
					# Variable-length N-gram pattern check (every 64 tokens)
					if not seq._rep_detected and seq.decoded_length >= 6 and seq.decoded_length % 64 == 0:
						_dl = seq.decoded_length
						_tokens = self.query_book[local_idx].decoded_tokens[0]
						if _check_repeating_pattern(_tokens, _dl):
							seq._rep_detected = True
							seq.eos_reached = True
							logging.warning(
								f"Rank {self.rank}: REPETITION (ngram) {seq.uuid} "
								f"gid={seq.global_idx} at decoded_len={_dl}"
							)

			self._cumulative_forward_ms += (time.perf_counter() - forward_start) * 1000

			# Decode timing ablation (BATCHGEN_DECODE_TIMING=1)
			from batchgen.timing import get_decode_timer
			_dt = get_decode_timer()
			if _dt and _dt.enabled:
				_dt.log_summary()
				_dt.reset()

		# Cleanup
		self._wait_pending_kv_append_tasks(sync_distributed_errors=True)
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
		AttnWrapperBase.gpu_paged_kv_manager_aux = None
		AttnWrapperBase.host_paged_kv_worker_view = None
		AttnWrapperBase.host_paged_kv_worker_view_aux = None
		AttnWrapperBase.cache_seqlens = None
		AttnWrapperBase.attention_mask = None
		AttnWrapperBase.position_ids = None
		AttnWrapperBase.max_seqlen = None
		AttnWrapperBase.cur_batch = None
		self._flush_glm5_dispatch_trace_summary("decode_end")
		AttnWrapperBase.batchgen_debug = None
		GLM5AttnWrapper.glm5_dispatch_trace_enabled = False
		GLM5AttnWrapper.glm5_dispatch_trace_id = None
		GLM5AttnWrapper.glm5_dispatch_trace_context = None
		GLM5AttnWrapper.glm5_dispatch_counts = {}
		GLM5AttnWrapper.glm5_dispatch_seen = set()
		AttnWrapperBase.kv_append_callback = None
		AttnWrapperBase.kv_append_callback_aux = None
		GLM5AttnWrapper.glm5_decode_primary_slot_indices = None
		GLM5AttnWrapper.glm5_decode_aux_slot_indices = None
		GLM5AttnWrapper.glm5_dsa_graph_forward_state = None
		GLM5AttnWrapper.glm5_dsa_flashmla_graph_metadata = None

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

		self.disable_decode_watchdog()
		return decode_uuids, batch

	def _wait_pending_kv_append_tasks(
		self,
		*,
		sync_distributed_errors: bool = False,
		defer_errors: bool = False,
	) -> int:
		"""
		Wait for all pending KV append tasks at page boundary.
		Returns the number of tasks that were waited for.
		
		CRITICAL: Also syncs CUDA to ensure all D2H DMA operations complete.
		Without this, KV data may not be fully written to host memory when
		sequences are later resumed, causing KV corruption.
		"""
		deferred_errors = getattr(self, "_deferred_kv_append_wait_errors", [])
		if not hasattr(self, '_pending_kv_append_tasks'):
			if sync_distributed_errors:
				error_payload = {
					"rank": self.rank,
					"errors": list(deferred_errors),
				} if deferred_errors else None
				all_errors = [None] * self.world_size
				dist.all_gather_object(all_errors, error_payload)
				if hasattr(self, "_deferred_kv_append_wait_errors"):
					self._deferred_kv_append_wait_errors.clear()
				flat_errors = [e for e in all_errors if e is not None]
				if flat_errors:
					raise RuntimeError(
						f"KV append/offload failed on at least one rank: {flat_errors[:8]}"
					)
			elif deferred_errors and not defer_errors:
				raise RuntimeError(
					f"Rank {self.rank}: KV append/offload failed: {deferred_errors[:4]}"
				)
			return 0
		
		num_tasks = len(self._pending_kv_append_tasks)
		wait_errors = list(deferred_errors)
		if deferred_errors and hasattr(self, "_deferred_kv_append_wait_errors"):
			self._deferred_kv_append_wait_errors.clear()
		for task in self._pending_kv_append_tasks:
			if task is not None:
				try:
					task.wait()
				except Exception as e:
					wait_errors.append(f"{type(e).__name__}: {e}")
		
		# CRITICAL FIX: Sync CUDA after waiting for tasks
		# The async tasks use a separate CUDA stream for D2H copies.
		# Even though each task internally syncs its stream via cudaEventSynchronize,
		# we need a full device sync to ensure ALL pending operations complete
		# before we allow GPU pages to be freed/reused.
		if num_tasks > 0 and not wait_errors:
			try:
				torch.cuda.synchronize(self.torch_device)
			except Exception as e:
				wait_errors.append(f"{type(e).__name__}: {e}")
		
		self._pending_kv_append_tasks.clear()
		
		# CRITICAL: Clear tensor references AFTER tasks complete
		# Tensors can now be safely garbage collected / memory reused
		if hasattr(self, '_pending_kv_append_tensors'):
			self._pending_kv_append_tensors.clear()

		if sync_distributed_errors:
			error_payload = {
				"rank": self.rank,
				"errors": wait_errors,
			} if wait_errors else None
			all_errors = [None] * self.world_size
			dist.all_gather_object(all_errors, error_payload)
			flat_errors = [e for e in all_errors if e is not None]
			if flat_errors:
				raise RuntimeError(
					f"KV append/offload failed on at least one rank: {flat_errors[:8]}"
				)
		elif wait_errors and defer_errors:
			if not hasattr(self, "_deferred_kv_append_wait_errors"):
				self._deferred_kv_append_wait_errors = []
			self._deferred_kv_append_wait_errors.extend(wait_errors)
		elif wait_errors:
			raise RuntimeError(
				f"Rank {self.rank}: KV append/offload failed: {wait_errors[:4]}"
			)
		
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
		# DEFENSIVE FIX: Filter out sequences not registered in the GPU manager.
		# During decode→prefill→decode transitions with mid-decode admission, the
		# batch can contain sequences whose GPU KV allocation failed or was not
		# yet registered. Passing such IDs to rebuild_page_table crashes with
		# KeyError. Filter them here and log.
		manager_sequences = getattr(gpu_manager, '_sequences', None)
		if manager_sequences is not None:
			allocated_ids = [gid for gid in global_ids if gid in manager_sequences]
			if len(allocated_ids) < len(global_ids):
				missing = [gid for gid in global_ids if gid not in manager_sequences]
				logging.error(
					f"Rank {self.rank}: _rebuild_page_table_for_batch: filtering "
					f"{len(missing)} unallocated sequences out of {len(global_ids)}: "
					f"first_missing={missing[:10]}"
				)
				global_ids = allocated_ids
		if not global_ids:
			Attn_Wrapper.cur_batch = []
			gpu_manager.clear_page_table()
			return
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
		
		# Launch async D2H append — no CPU-side sync needed.
		# Tensor references kept alive in _pending_kv_append_tensors.
		# All tasks waited at decision boundary via _wait_pending_kv_append_tasks().
		task = worker_view.async_append_decode_kv_to_host(
			layer_idx=layer_idx,
			sequence_ids=sequence_ids,
			k_tensor=k_tensor,
			v_tensor=v_tensor,  # GQA models (GPT-OSS) have separate V; MLA models pass None
			sequence_lengths=sequence_lengths,
		)

		# Store tensor references alongside task to prevent GC/memory reuse
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
		
		existing_global_ids = self._local_indices_to_global_seq_ids(current_batch)

		# Step 7: Launch async load
		worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		if worker_view is None:
			if existing_global_ids:
				gpu_manager.rebuild_page_table(existing_global_ids)
			return None, new_uuids, new_local_indices, new_global_ids

		if isinstance(gpu_manager, DualKVCacheCoordinator):
			pointers = self._prepare_dual_kv_load_pointers(
				gpu_manager, new_global_ids, existing_global_ids
			)
			async_task = self._launch_dual_host_kv_load(pointers)
			self._async_load_tensors = pointers
		else:
			gpu_manager.rebuild_page_table(new_global_ids)
			k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
			active_page_counts = gpu_manager.export_active_sequence_page_counts()
			sequence_tensor = torch.tensor(new_global_ids, dtype=torch.int64, device="cpu")
			async_task = worker_view.async_load_layer_paged_kv_to_device(
				sequence_ids=sequence_tensor,
				active_page_counts=active_page_counts,
				k_device_ptrs=k_ptrs,
				v_device_ptrs=v_ptrs,
			)
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

		if isinstance(gpu_manager, DualKVCacheCoordinator):
			pointers = self._prepare_dual_kv_load_pointers(
				gpu_manager, new_global_ids, existing_global_ids
			)
			sequence_tensor = pointers.sequence_tensor
			k_ptrs = pointers.primary_k_ptrs
			v_ptrs = pointers.primary_v_ptrs
			active_page_counts = pointers.primary_page_counts
		else:
			gpu_manager.rebuild_page_table(new_global_ids)
			k_ptrs, v_ptrs = gpu_manager.get_padded_3d_page_pointers()
			active_page_counts = gpu_manager.export_active_sequence_page_counts()
			sequence_tensor = torch.tensor(new_global_ids, dtype=torch.int64, device="cpu")
			if existing_global_ids:
				gpu_manager.rebuild_page_table(existing_global_ids)

		timing['prepare_ms'] = (time.perf_counter() - t0) * 1000

		# ============ PHASE 8: Launch async load ============
		t0 = time.perf_counter()

		if worker_view is None:
			logging.warning(f"Rank {self.rank}: worker_view is None, cannot launch async load")
			timing['launch_ms'] = (time.perf_counter() - t0) * 1000
			return None, new_uuids, new_local_indices, new_global_ids, timing

		if isinstance(gpu_manager, DualKVCacheCoordinator):
			async_task = self._launch_dual_host_kv_load(pointers)
		else:
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
		self._async_load_tensors = pointers if isinstance(gpu_manager, DualKVCacheCoordinator) else {
			'k_ptrs': k_ptrs,
			'v_ptrs': v_ptrs,
			'sequence_tensor': sequence_tensor,
			'active_page_counts': active_page_counts,
		}
		
		return async_task, new_uuids, new_local_indices, new_global_ids, timing

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
		- A sequence prefilled by rank R has host KV on node (R // NUM_GPUS_PER_NODE)
		- Only ranks on THAT node can load this sequence to their GPU
		
		Sync strategy:
		1. All-gather free GPU pages from all ranks
		2. All ranks compute IDENTICAL loading decision
		3. Each rank only loads sequences assigned to it
		4. All ranks update decode_uuids identically
		"""
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
					# Incremental write: gather completed tokens to rank 0
					self._submit_completed_to_incremental_writer(completed_uuids)
					# Gather decoded tokens from owning ranks before reporting
					gathered_texts = self._gather_completed_tokens(completed_uuids)
					# ORDERING FIX: release GPU/host KV BEFORE _report_completion
					# pops local_map entries. Previously the filter below
					# captured an empty list because _report_completion ran
					# first and popped every local_map entry on the owner.
					my_completed = [u for u in completed_uuids if u in self._uuid_to_local_map]
					if my_completed:
						# Intersect with source-of-truth GPU-allocated set (see
						# note at the matching site ~line 5435).
						gpu_allocated = [u for u in my_completed if u in self._sequences_with_gpu_kv]
						if gpu_allocated:
							self._release_gpu_kv_pages(self._get_local_indices_for_uuids(gpu_allocated))
					self._release_host_kv_pages_for_batch(completed_uuids)
					# Report completions (this pops local_map; must run LAST).
					for uuid in completed_uuids:
						self._report_completion(uuid, gathered_text=gathered_texts.get(uuid))
				
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
					# Build attention mask on-the-fly from sequence metadata
					max_len = self.max_input_length + new_token_idx
					cache_seqlens = []
					for query_idx in batch:
						uuid = self._local_to_uuid_map[query_idx]
						seq = self.global_batch.get_sequence(uuid)
						cache_seqlens.append(seq.current_context_length)
					seqlens_tensor = torch.tensor(cache_seqlens, dtype=torch.int64)
					positions = torch.arange(max_len)
					attention_mask = (positions.unsqueeze(0) < seqlens_tensor.unsqueeze(1)).to(torch.int64)
					if "deepseek" not in self.model_config.model_type:
						position_ids = (seqlens_tensor - 1).unsqueeze(-1)
					else:
						position_ids = create_position_ids_from_attention_mask(attention_mask)

					Attn_Wrapper.attention_mask = attention_mask
					Attn_Wrapper.position_ids = position_ids
					new_tokens = self.model(
						new_tokens.to(self.torch_device),
						attention_mask=attention_mask.to(self.torch_device),
						use_cache=False,
					)
					batch_sequences = [
						self.global_batch.get_sequence(self._local_to_uuid_map[local_idx])
						for local_idx in batch
					]
					new_tokens = self._select_tokens(new_tokens.logits[:, -1, :], batch_sequences)
					self.update_new_token(new_tokens, batch, new_token_idx)

					# Update sequence state
					for i, local_idx in enumerate(batch):
						uuid = self._local_to_uuid_map[local_idx]
						seq = self.global_batch.get_sequence(uuid)
						seq.decoded_length = new_token_idx + 1
						seq.current_context_length = seq.prompt_length + new_token_idx + 1

						# Only mark eos_reached if we should stop at EOS
						token_id = new_tokens[i].item()
						if self._should_stop_at_eos(token_id):
							seq.eos_reached = True

						# Always check max length
						if seq.decoded_length >= seq.max_decode_length:
							seq.eos_reached = True

						# Repetition detection (BATCHGEN_REP_DETECTION=1)
						if REP_DETECTION and not seq._rep_detected:
							if token_id == seq._rep_last_token:
								seq._rep_count += 1
								if seq._rep_count >= 32:
									seq._rep_detected = True
									seq.eos_reached = True
									seq.log_event(SeqEvent.REPETITION, self.rank,
										f"token={token_id}, count={seq._rep_count}")
									lifespan.dump_lifespan(seq.uuid, seq.global_idx,
										seq._lifespan_log, "REPETITION")
							else:
								seq._rep_last_token = token_id
								seq._rep_count = 1

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
					# Build attention mask on-the-fly from sequence metadata
					max_len = self.max_input_length + new_token_idx
					cache_seqlens = []
					for query_idx in batch:
						uuid = self._local_to_uuid_map[query_idx]
						seq = self.global_batch.get_sequence(uuid)
						cache_seqlens.append(seq.current_context_length)
					seqlens_tensor = torch.tensor(cache_seqlens, dtype=torch.int64, device=self.torch_device)
					positions = torch.arange(max_len, device=self.torch_device)
					attention_mask = (positions.unsqueeze(0) < seqlens_tensor.unsqueeze(1)).to(torch.int64)
					if "deepseek" in self.model_config.model_type:
						position_ids = create_position_ids_from_attention_mask(attention_mask)
					else:
						position_ids = (seqlens_tensor - 1).unsqueeze(-1)

					Attn_Wrapper.attention_mask = attention_mask
					Attn_Wrapper.position_ids = position_ids
					new_tokens = self.model(
						new_tokens.to(self.torch_device),
						attention_mask=attention_mask.to(self.torch_device),
						use_cache=False,
					)
					batch_sequences = [
						self.global_batch.get_sequence(self._local_to_uuid_map[local_idx])
						for local_idx in batch
					]
					new_tokens = self._select_tokens(new_tokens.logits[:, -1, :], batch_sequences)
					self.update_new_token(new_tokens, batch, new_token_idx)

					# Update sequence state
					for i, local_idx in enumerate(batch):
						uuid = self._local_to_uuid_map[local_idx]
						seq = self.global_batch.get_sequence(uuid)
						seq.decoded_length = new_token_idx + 1
						seq.current_context_length = seq.prompt_length + new_token_idx + 1

						# Only mark eos_reached if we should stop at EOS
						token_id = new_tokens[i].item()
						if self._should_stop_at_eos(token_id):
							seq.eos_reached = True

						# Always check max length
						if seq.decoded_length >= seq.max_decode_length:
							seq.eos_reached = True

						# Repetition detection (BATCHGEN_REP_DETECTION=1)
						if REP_DETECTION and not seq._rep_detected:
							if token_id == seq._rep_last_token:
								seq._rep_count += 1
								if seq._rep_count >= 32:
									seq._rep_detected = True
									seq.eos_reached = True
									seq.log_event(SeqEvent.REPETITION, self.rank,
										f"token={token_id}, count={seq._rep_count}")
									lifespan.dump_lifespan(seq.uuid, seq.global_idx,
										seq._lifespan_log, "REPETITION")
							else:
								seq._rep_last_token = token_id
								seq._rep_count = 1

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

		# Models whose MoE layers don't expose DeepSeek-style `.mlp` (e.g.
		# Kimi-Linear uses `.block_sparse_moe` with BF16 experts) have no FP8
		# weights to unregister.
		_fkd = self.loaded_model_config.first_k_dense_replace
		if _fkd < len(self.model.model.layers) and not hasattr(
			self.model.model.layers[_fkd], 'mlp'
		):
			return

		for layer_idx in range(len(self.model.model.layers)):
			attn_module = self.model.model.layers[layer_idx].self_attn
			if hasattr(attn_module, '_unregister_fp8_weights'):
				attn_module._unregister_fp8_weights()
			if layer_idx >= self.loaded_model_config.first_k_dense_replace:
				shared_experts = getattr(self.model.model.layers[layer_idx].mlp, 'shared_experts', None)
				if shared_experts is not None and hasattr(shared_experts, '_unregister_fp8_weights'):
					shared_experts._unregister_fp8_weights()
				for routed_expert_idx in range(self.model_config.num_local_experts):
					if hasattr(self.model.model.layers[layer_idx].mlp.experts[routed_expert_idx], '_unregister_fp8_weights'):
						self.model.model.layers[layer_idx].mlp.experts[routed_expert_idx]._unregister_fp8_weights()
				if hasattr(self.model.model.layers[layer_idx].mlp, "cleanup"):
					self.model.model.layers[layer_idx].mlp.cleanup()

	def _handle_hot_reload(self, msg: dict) -> dict:
		"""Hot-reload batchgen_worker module and rebind methods on this instance.

		Called from inside generate_persistent() admission loop. Both rank 0
		and other ranks must call this so all ranks reload in lockstep.

		Returns: dict with status, rebound count, skipped count, missing attrs.
		"""
		import importlib
		import inspect
		import re
		import sys
		import logging as _log
		try:
			reload_deps = msg.get("reload_deps", True) if isinstance(msg, dict) else True

			# Reload commonly-changed dependent modules first
			if reload_deps:
				dep_modules = [
					"batchgen.server.batch_scheduler",
					"batchgen.server.intake_pool",
					"batchgen.server.scheduling_pool",
					"batchgen.kv_cache.gpu_paged_kv_manager",
					"batchgen.attention.dsa.glm5_decode_selector",
				]
				for mod_name in dep_modules:
					if mod_name in sys.modules:
						importlib.reload(sys.modules[mod_name])
						_log.info(f"Rank {self.rank}: Reloaded dependency {mod_name}")

			# Reload the worker module itself
			import batchgen.batchgen_worker as worker_module
			importlib.reload(worker_module)
			NewClass = worker_module.BatchGenWorker

			# Validate: warn if new __init__ adds attrs missing on this instance
			missing = []
			try:
				new_init_src = inspect.getsource(NewClass.__init__)
				old_init_src = inspect.getsource(type(self).__init__)
				if new_init_src != old_init_src:
					new_attrs = set(re.findall(r"self\.(\w+)\s*=", new_init_src))
					missing = sorted([a for a in new_attrs if not hasattr(self, a)])
					if missing:
						_log.warning(
							f"Rank {self.rank}: RELOAD WARNING — new __init__ has "
							f"{len(missing)} attrs missing on existing worker: {missing[:10]}"
						)
			except (OSError, TypeError):
				pass

			# Rebind methods (skip __init__ and dunders). Preserve descriptor
			# semantics so hot reload does not turn staticmethods into bound
			# instance methods.
			rebound = 0
			skipped = 0
			for name, descriptor in NewClass.__dict__.items():
				if name == "__init__":
					skipped += 1
					continue
				if name.startswith("__") and name.endswith("__"):
					continue
				try:
					if isinstance(descriptor, staticmethod):
						setattr(self, name, descriptor.__func__)
					elif isinstance(descriptor, classmethod):
						setattr(self, name, descriptor.__func__.__get__(type(self), type(self)))
					elif inspect.isfunction(descriptor):
						setattr(self, name, descriptor.__get__(self, type(self)))
					else:
						continue
					rebound += 1
				except Exception:
					skipped += 1

			_log.info(
				f"Rank {self.rank}: Hot reload SUCCESS — "
				f"rebound {rebound} methods, skipped {skipped}"
				+ (f", {len(missing)} missing attrs" if missing else "")
			)
			result = {
				"status": "reload_success",
				"rank": self.rank,
				"rebound": rebound,
				"skipped": skipped,
				"missing_attrs": missing,
			}
			self._write_reload_status(result)
			return result
		except Exception as e:
			_log.error(f"Rank {self.rank}: Hot reload FAILED: {e}", exc_info=True)
			result = {"status": "reload_failed", "rank": self.rank, "error": str(e)}
			self._write_reload_status(result)
			return result

	def _write_reload_status(self, result: dict) -> None:
		"""Write reload status atomically to /tmp/batchgen_reload_status/rank_<N>.json.

		The HTTP server polls these files instead of waiting on a queue,
		which avoids deadlocks when the FastAPI event loop is blocked.
		"""
		import json
		import os
		import tempfile
		import time as _time
		try:
			result_with_time = dict(result)
			result_with_time["timestamp"] = _time.time()
			status_dir = "/tmp/batchgen_reload_status"
			os.makedirs(status_dir, exist_ok=True)
			# Write to temp then atomic rename
			fd, tmp_path = tempfile.mkstemp(dir=status_dir, suffix=".json")
			with os.fdopen(fd, "w") as f:
				json.dump(result_with_time, f)
			final_path = os.path.join(status_dir, f"rank_{self.rank}.json")
			os.rename(tmp_path, final_path)
		except Exception as e:
			import logging as _log
			_log.warning(f"Rank {self.rank}: Failed to write reload status: {e}")

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

		# Free WGMMA shared buffers if they exist (class-level, survives model deletion)
		try:
			from batchgen.models.glm.glm5.model import Glm5MoE
			if getattr(Glm5MoE, '_wgmma_shared_bufs', None) is not None:
				Glm5MoE._wgmma_shared_bufs.free_buffers()
				Glm5MoE._wgmma_shared_bufs = None
				Glm5MoE._wgmma_next_layer_id = 0
		except ImportError:
			pass

		# Delete model directly without CPU transfer
		del self.model
		self.model = None
		self._cuda_graph_manager = None
		self._glm5_moe_cuda_graph_manager = None
		self._glm5_layer_cuda_graph_manager = None
		self._glm5_layer_graph_failed_buckets = set()
		self._glm5_layer_graph_capture_attempted_for_batch = False
		self._glm5_layer_graph_signature = None
		self._glm5_layer_graph_max_seqlen = None
		self._glm5_dsa_graph_capture_attempted_for_batch = False
		self._glm5_moe_graph_capture_attempted_for_batch = False
		self._glm5_dsa_graph_page_table_change_after_capture_logged = False
		self._whole_model_segment = None
		self._whole_model_bucketing = None
		self._glm5_whole_model_capture_input_ids = None
		self._glm5_moe_graph_failed_buckets = set()
		self._whole_model_graph = False
		self._glm5_whole_model_graph = False
		self._glm5_whole_model_graph_failed_buckets = set()
		self._glm5_whole_model_graph_capture_attempted_for_batch = False
		self._glm5_whole_model_graph_state_change_after_capture_logged = False
		self._glm5_whole_model_graph_signature = None
		self._glm5_whole_model_graph_unavailable_reason = None

		# Defense-in-depth: free PSM-owned GPU buffers that survive model deletion
		# (INT4 contiguous weight buffers, MoE class-level buffers)
		if hasattr(self, 'parallel_manager') and self.parallel_manager is not None:
			pm = self.parallel_manager
			if hasattr(pm, "_release_streamed_sp8_prefill"):
				pm._release_streamed_sp8_prefill()
			for attr in ('_int4_packed_gpu_buf', '_int4_scale_gpu_buf'):
				if hasattr(pm, attr):
					delattr(pm, attr)

		# Adapter holds Python refs to model/segment/KV manager via _ctx;
		# without release_context() the captured segment's static KV buffers
		# survive empty_cache() and prefill OOMs on the next batch.
		if getattr(self, '_cuda_graph_adapter', None) is not None:
			self._cuda_graph_adapter.release_context()

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
						aux_view_shutdown = getattr(self, "host_paged_kv_worker_view_aux", None)
						for seq_id in global_ids_to_release:
							try:
								worker_view.release_sequence_pages([seq_id])
								if aux_view_shutdown is not None:
									aux_view_shutdown.release_sequence_pages([seq_id])
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
		self._cuda_graph_manager = None
		self._glm5_moe_cuda_graph_manager = None
		self._glm5_layer_cuda_graph_manager = None
		self._whole_model_segment = None
		self._whole_model_bucketing = None
		self._glm5_whole_model_capture_input_ids = None
		self._glm5_moe_graph_failed_buckets = set()
		self._glm5_layer_graph_failed_buckets = set()
		self._glm5_dsa_graph_capture_attempted_for_batch = False
		self._glm5_moe_graph_capture_attempted_for_batch = False
		self._glm5_layer_graph_capture_attempted_for_batch = False
		self._glm5_dsa_graph_page_table_change_after_capture_logged = False
		self._whole_model_graph = False
		self._glm5_whole_model_graph = False
		self._glm5_whole_model_graph_failed_buckets = set()
		self._glm5_whole_model_graph_capture_attempted_for_batch = False
		self._glm5_whole_model_graph_state_change_after_capture_logged = False
		self._glm5_whole_model_graph_signature = None
		self._glm5_layer_graph_signature = None
		self._glm5_layer_graph_max_seqlen = None

		# 9. Clear CUDA cache
		torch.cuda.empty_cache()
		torch.cuda.synchronize(self.torch_device)
		
		# 10. Force garbage collection
		gc.collect()
		
		# Synchronize all ranks after cleanup
		dist.barrier()
		
		logging.info(f"Rank {self.rank}: State reset completed")
