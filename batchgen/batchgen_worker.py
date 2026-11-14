import concurrent.futures
import copy
import functools
import hashlib
import json
import psutil
import logging
import math
import os
import socket
import sys
import time
from typing import (
	AbstractSet,
	Callable,
	Dict,
	List,
	Mapping,
	Optional,
	Sequence,
	Set,
	Tuple,
)

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from tqdm import tqdm
from batchgen.config.model_registry import load_config
from batchgen.config.tokenizer_registry import load_tokenizer
from batchgen.deprecation import LegacyInferenceDeprecated
from batchgen.config.config import detect_gpu_arch
from batchgen.gemm.deepgemm_pdl_policy import apply_deepgemm_pdl

# Use new wrapper system - Attn_Wrapper/Expert_Wrapper are aliases for backward compatibility
from batchgen.models.wrappers import BaseModuleWrapper, AttnWrapperBase, ExpertWrapperBase
# Phase C: GLM-5-specific ClassVars (dispatch_trace_*, _dsa_short_count,
# glm5_decode_*_slot_indices, glm5_dsa_*) live on GLM5AttnWrapper, not
# AttnWrapperBase, per audit §A finding #8.
from batchgen.models.glm.glm5.wrappers import GLM5AttnWrapper
# Captured-span policy, shared with the Triton prewarm so the kernels warmed
# before capture are the ones the capture compiles.
from batchgen.config.graph_span_policy import (
	GRAPH_CONTEXT_STATICS_BUDGET_BYTES,
	graph_context_span_cap_tokens,
)
# Aliases for backward compatibility with existing code
Attn_Wrapper = AttnWrapperBase
Expert_Wrapper = ExpertWrapperBase

from .config.config import EngineConfig
from .scheduler.host_mem import get_physical_memory_info

from batchgen.parameter_server_client import ParameterServerClient
from batchgen import lifespan


from batchgen.lifespan import SeqEvent

REP_DETECTION = os.environ.get("BATCHGEN_REP_DETECTION", "1") == "1"

# --- Bounded decode-step token wait (see _wait_decode_tokens) --------------
# A rank whose peer died mid-decode blocks in the per-step token-event sync far
# below Python, where no flag check can retire it, and aborting the dead
# communicator does not reliably release it: the survivors then burn the
# supervisor's whole grace window.  The wait is therefore a poll.  It costs
# the same class of CPU as the blocking call it replaces: torch creates its CUDA
# events with ``blocking=False`` by default (torch/cuda/streams.py Event.__new__)
# and only a blocking event carries ``cudaEventBlockingSync``
# (torch/csrc/jit/cuda/cuda.h), so cudaEventSynchronize already busy-spins in
# the driver rather than parking the thread.
#
# The spin budget keeps the healthy fast path flat -- a decode step's readback
# lands well inside it, so the common case is a handful of cudaEventQuery calls
# and no sleep, no probe read at all.
_DECODE_EVENT_SPIN_BUDGET_S = 0.002
_DECODE_EVENT_POLL_INTERVAL_S = 1e-4
# ``_wait_decode_tokens``' two outcomes (_DECODE_WAIT_COMPLETED /
# _DECODE_WAIT_ABANDONED) have one owner, decode_engine, imported below.
# Returned by _finish_deferred_decode_step when the cohort teardown flag was
# seen after the step's host-KV drain (the deferred twin of the post-drain
# flag check in the undeferred step).
_DECODE_STEP_SHUTDOWN = "shutdown"

# --- Deferred sampled-token readback (depth-1 decode pipeline) -------------
# True: the decode loop enqueues step N+1 before it waits on step N's sampled
# tokens, then does step N's token-consuming host work (trajectory append,
# stop decisions) while step N+1 runs on the GPU. The next step's input ids
# are already a device tensor and its per-row metadata advances by one per
# live row regardless of the token value, so only the host bookkeeping is
# deferred, and every consumer of host tokens (page boundary, input-id
# rebuild, loop exit) drains the pending step first. Stop decisions lag by at
# most the one step that is already computed for every finished row anyway
# (finished rows stay in the batch until the next boundary and are skipped by
# the completed-row mask), so emitted tokens are unchanged.
# False: the undeferred step exactly as before (byte-identical behavior).
# Read at call time, so it can be flipped on the module.
# Total budget for the LOCAL quiesce on a decode teardown exit.  An abandoned
# step leaves its host-KV appends and its async KV load parked behind device
# work the dead collective will never let finish, so the quiesce bounds them
# itself (see _quiesce_decode_for_cohort_shutdown) and self-exits rather than
# unwind while they still hold page-table pointers.
#
# The bound is established by NON-BLOCKING probes only -- never by running a
# blocking wait under a timeout, on this thread or any other.  MEASURED: the
# append task's wait is bound as `.def("wait", &kv::KVAsyncTask::wait)`
# (core/batchgen_Binding.cpp:968) with no `py::call_guard<gil_scoped_release>`,
# so a task parked behind the dead collective blocks in
# `std::shared_future::get` HOLDING THE GIL.  No other Python can then run --
# not the deadline, not the abort watchdog, not the supervisor's self-exit net
# -- so the rank burns the whole grace window.  `KVAsyncTask::done()`
# (core/KV_Storage/host_paged_kv_worker_view.h:93-99) is a `wait_for(0s)` status
# test, and a CUDA event query is the same class of probe for device work; both
# are safe to call under the GIL because neither can wait.
_COHORT_QUIESCE_DEADLINE_S = 5.0
_COHORT_QUIESCE_POLL_INTERVAL_S = 1e-3
_COHORT_QUIESCE_DEADLINE_REASON = "quiesce-deadline"

# Seconds between two "step exceeds the captured span" warnings. The condition
# holds for every step of a long sequence, so it is logged at a rate a human
# reads rather than once per step.
_GRAPH_SPAN_WARN_INTERVAL_S = 60.0


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

def _repetition_check_enabled(seq) -> bool:
	"""Per-request gate for the repetition breaker.

	ignore_eos declares the content irrelevant and the length contract binding
	(fixed-shape benchmark semantics): the breaker's premise — reclaiming
	capacity from degenerate tokens the caller did not want — is void, so the
	check is disabled for that sequence. Requests without ignore_eos keep the
	breaker unchanged.
	"""
	return not seq.ignore_eos


def _admitted_prompt_ids(entry: dict) -> torch.Tensor:
	"""The prompt token ids one admission entry carries, as a CPU int32 tensor.

	The server builds every prompt's ids (``batch_scheduler.prompt_ids_array``)
	and sends them as a 1-D int32 numpy array; the returned tensor shares its
	memory. Anything else is a producer defect and raises.
	"""
	token_ids = entry.get("input_ids")
	if not (
		isinstance(token_ids, np.ndarray)
		and token_ids.ndim == 1
		and token_ids.dtype == np.int32
	):
		raise TypeError(
			f"admission entry {entry.get('request_id')!r}: input_ids must be a "
			f"1-D int32 numpy array, got {type(token_ids).__name__}"
			+ (
				f" of shape {token_ids.shape} and dtype {token_ids.dtype}"
				if isinstance(token_ids, np.ndarray) else ""
			)
		)
	return torch.from_numpy(token_ids)

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
from batchgen.memfd import create_memfd
from batchgen.query_book import (
	QueryBook,
	bind_query_book_slot,
	make_query_book_entry,
	release_query_book_slot,
)
from batchgen.config.decode_strategy import DecodeStrategy, strategy_for
from batchgen.config.legacy_kv_accounting import legacy_past_kv_byte_size
from batchgen.worker.graph_runtime import GraphRuntime
from batchgen.worker.trajectory_store import TrajectoryStore
from batchgen.worker.prefix_reuse_state import PrefixReuseState
from batchgen.worker import decode_engine as _decode_engine
from batchgen.worker.decode_engine import (
	BATCHGEN_CB_DEBUG,
	BATCHGEN_DECODE_ASSERT,
	BATCHGEN_MULTI_BATCH_DIAG,
	_DECODE_WAIT_ABANDONED,
	_DECODE_WAIT_COMPLETED,
	DecoupledGpuDecode,
)
from batchgen.worker.prefill_engine import PrefillEngine


def _load_kda_wrapper():
	# The only model-package import behind TrajectoryStore: batchgen/worker/
	# must not depend on batchgen.models, so the store receives this loader.
	# ImportError propagates so each store call site keeps its old handling.
	from batchgen.models.moonshotai.kimi_linear.wrappers import KimiLinearKDAWrapper
	return KimiLinearKDAWrapper
from batchgen.worker.rank_context import RankContext, node_of_rank
from batchgen.worker.sequence_table import SequenceTable
from batchgen.utils import config_torch_module_initializer
from batchgen.config.model_name_utils import (
	is_kimi_k25_backend_model,
	# Re-exported for the worker main loop: which served ids take the
	# pre-readiness startup is a model-name fact (exact ids, not a family).
	runs_startup_before_readiness,
)
from batchgen.config.cuda_graph_flag_policy import (
	cli_enables_whole_model_graph,
	glm5_any_cuda_graph_requested,
	glm5_whole_model_cuda_graph_requested,
	require_cuda_graph_flag_supported,
)
from batchgen.kv_cache.gpu_paged_kv_manager import GPUPagedKVCacheManager
from batchgen.native_loader import core_engine
from batchgen.worker.indexing import IndexLookupRequest, IndexManager
from batchgen.worker.completion import CompletionContext, CompletionHandler
from batchgen.worker.sync import (
	SyncContext,
	SyncCoordinator,
	TorchDistCollectiveBackend,
)
from batchgen.worker.prefill import (
	PrefillAdmissionStalled,
	PrefillAdmissionStateDivergence,
	PrefillCandidate,
	PrefillProgressGuard,
	PrefillScheduler,
	PrefillSelectionRequest,
	compute_prefill_host_reservation,
	describe_prefill_blockers,
	host_kv_eviction_enabled,
	host_owned_pages_after_sharing,
	prefill_host_reserve_page_deficit,
	prefill_reclaim_target_pages,
)
from batchgen.worker.host_rebalancer import HostKVRebalancer
from batchgen.worker.startup_phases import NULL_STARTUP_PHASE_REPORTER
from batchgen.worker import startup_transient as _startup_transient
from batchgen.worker.boundary import (
	BoundaryDecisionRequest,
	BoundaryHandler,
	BoundarySeqMeta,
)
from batchgen.worker.decode import (
	DecodeBatchRequest,
	DecodeCandidate,
	DecodeScheduler,
	DecodeCapacitySnapshot,
	reduce_decode_capacity_snapshot,
)
from batchgen.cuda_graph.buckets import (
	DecodeWidthExceedsGraphBucketError,
	check_decode_width_within_buckets,
	decode_graph_bucket_sizes,
)
from batchgen.worker.shared_prefix_charge import (
	SharedPrefixLedger,
	plan_shared_runs,
)
from batchgen.config.max_model_len import (
	SequenceLengthCapacity, resolve_max_model_len, verify_max_model_len,
)
from batchgen.worker.kv_manager import (
	GPU_KV_POOL_HEADROOM_GB,
	KVCacheManager,
	KVStats,
	KVUtilizationRequest,
	MigrationCandidate,
	MigrationPlanRequest,
	PageTableCapacityRequest,
	WatermarkTriggerRequest,
)
from batchgen.worker.kv_plan import (
	NodeCapacityReport,
	credit_reclaimable_pages,
	effective_chunk_size,
	host_capacity_due,
	prefill_owner_demand_pages,
	provisional_pool_gb,
	verdict_node_capacity,
)
from batchgen.worker.sampling_residency import DecodeSamplingResidency
from batchgen.eventlog.counters import STEP_COUNTERS_INTERVAL_S, StepCounters
from batchgen.worker.step_timing import (
	FORWARD_EVENT_RING_SLOTS,
	ForwardEventRing,
	accumulate_decode_step_split,
	format_decode_step_split,
	new_decode_step_split,
)
from batchgen.phase_lifecycle import (
	AdmitContext,
	ExitMode,
	PhaseEnterContext,
	PhaseKey,
	verify_phase_contract,
)
from batchgen.worker.deferred_readback import (
	DeferredTokenReadbackError,
	PendingDecodeStep,
	TokenReadbackRing,
	deferred_decode_step_marks,
)
import dataclasses as _dataclasses

from batchgen.kv_cache.host_kv_mananger_config import (
	build_host_kv_config,
	build_host_kv_worker_view,
	is_dsa_model,
)
from batchgen.kv_cache.dual_kv_cache_coordinator import (
	DualKVCacheCoordinator,
	DualPoolPageKeys,
)
from batchgen.kv_cache.dual_host_kv_coordinator import DualAsyncKVTask, DualHostKVCoordinator
from batchgen.sequence import SequenceBatch, SequenceEntry, SequenceStatus, configure_page_buffers
# The page-buffer sizes are module globals that configure_page_buffers()
# rebinds at worker init; read them through the module at use time, never
# import them by value (that binds the import-time defaults).
import batchgen.sequence as _sequence_module
from batchgen.kv_cache.host_kv_health import format_host_kv_health
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
	host_kv_owner_reports,
	validate_boundary_payload_alignment,
)
# sample_tokens: Token sampling with temperature/top_p support
from batchgen.sampling import sample_tokens
# Migration data structures for KV cache migration between nodes
from batchgen.migration import MigrationOp

BATCHGEN_ENABLE_ALL_TO_ALL = os.environ.get("BATCHGEN_ENABLE_ALL_TO_ALL")
if BATCHGEN_ENABLE_ALL_TO_ALL == "1":
	try:
		from pplx_kernels import nvshmem_init
	except ImportError as exc:
		logging.warning("Failed to import pplx_kernels.nvshmem_init: %s", exc)
		nvshmem_init = None
else:
	nvshmem_init = None

# BATCHGEN_CB_DEBUG / BATCHGEN_DECODE_ASSERT / BATCHGEN_MULTI_BATCH_DIAG are
# evaluated once, by their single owner decode_engine (imported above).

# Force synchronous KV offload (disable deferred flush) for debugging
BATCHGEN_SYNC_KV = os.environ.get("BATCHGEN_SYNC_KV", "0") == "1"

# Optional gate for expensive/critical diagnostics (default off in production)
BATCHGEN_ENABLE_CRITICAL_DIAGS = os.environ.get('BATCHGEN_ENABLE_CRITICAL_DIAGS', '0') == '1'

# Max per-rank in-decode batch. Bounds the K2.5 MoE padded buffer (mtp = round_up(world_size *
# this)) so init does not OOM on a large candidate pool, and is the per-rank admission cap so
# the pre-reserved buffer never overflows at runtime. Raise to fill more GPU KV (memory permitting).
_MAX_DECODE_RANK_BSZ = int(os.environ.get("BATCHGEN_MAX_DECODE_RANK_BSZ", "128"))

def _reentry_history_length(
	prompt_length: int, original_prompt_length: int, original_max_decode_length: int,
) -> int:
	"""Tokens a re-entering sequence already decoded, capped by its own budget.

	The reconstructed prompt carries the whole history, so the history length is
	prompt_length - original_prompt_length. Cap it by the request's absolute
	decode budget, which also bounds its decoded_tokens row (admission grows the
	pool to every request's max_decode_length). The worker-wide
	max_decoding_length must not be used: in pool mode it is the budget of the
	first batch the server saw and can be smaller than a later request's history.
	"""
	return max(0, min(prompt_length - original_prompt_length, original_max_decode_length))


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
ENABLE_DECODE_PREEMPTION = os.environ.get('BATCHGEN_ENABLE_DECODE_PREEMPTION', '1') == '1'  # Default ON

# GPU KV cache size override (DEPRECATED: use --gpu-memory-frac CLI arg instead)
# If set, overrides the automatic calculation. Otherwise, size is computed as:
# gpu_kv_cache = GPU_mem * gpu_memory_frac - model_instance_size
_GPU_KV_CACHE_SIZE_OVERRIDE = os.environ.get("BATCHGEN_GPU_KV_CACHE_SIZE_GB")
# Note: Generic Scheduler removed - config is now created by model-specific Planner in initializer


def _normalize_aux_kv_entry(entry) -> tuple:
	"""(layer_idx, k_tensor, v_tensor, scale_tensor) of a deferred aux append.

	A page-split FP8 aux region needs the fp32 per-token scales alongside the
	quantized keys, so a producer may append a 4-tuple; a bf16 AoS producer
	keeps appending a 3-tuple and gets ``scale_tensor=None``.
	"""

	if len(entry) == 4:
		return tuple(entry)
	if len(entry) == 3:
		layer_idx, k_tensor, v_tensor = entry
		return layer_idx, k_tensor, v_tensor, None
	raise RuntimeError(
		"deferred aux KV entry must be (layer_idx, k_tensor, v_tensor) or "
		f"(layer_idx, k_tensor, v_tensor, scale_tensor); got {len(entry)} fields"
	)


class _DualAsyncLoadTask:
	"""Forwards .wait() to both primary and aux async KV load tasks.

	Either side may be None (non-DSA model or aux-disabled diagnostic mode);
	.wait() skips None. Used by mid-decode paths where we must ensure
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


class _WeightPasses:
	"""Runs every model forward that can consume streamed weights inside one
	weight pass of the core engine.

	A weight pass is one traversal of every installed weight-copy list, i.e.
	one forward; ``get_weights`` fails outside one. Inside ``scope()``,
	``begin()`` arms the pass of the forward about to run and ``end()`` closes
	it once that forward has consumed its modules. ``begin(carry=True)`` also
	arms the pass of the forward that follows, so the producer can continue
	into its first layers while this one finishes; the next ``begin()`` takes
	that pass instead of arming one. Leaving the scope normally cancels and
	ends a carried pass whose forward will not run. Leaving it by an
	exception aborts every open pass instead: ending one waits for the
	producer to drain it, which a sibling rank stuck in a collective can
	stall until the engine's stall timeout. Either way no pass crosses a phase
	boundary. Pass 0 means nothing is installed and costs no engine call.
	"""

	_COUNTED = ("copied", "skipped", "released_on_landing")

	def __init__(self, core_engine, rank):
		self._core_engine = core_engine
		self._rank = rank
		self._phase = None
		self._current = None
		self._carried = None
		# Batch-level debug skew (batchgen_debug.weight_pass_end_delay): this
		# rank sleeps before closing each prefill pass, so its siblings run
		# ahead exactly where a per-rank stop used to strand the node daemon.
		self.end_delay_s = 0.0
		# Summed end_weight_pass reports of the current (or last) scope.
		self.totals = self._empty_totals()

	@classmethod
	def _empty_totals(cls):
		totals = {name: {} for name in cls._COUNTED}
		totals["passes"] = 0
		totals["carried"] = 0
		return totals

	@classmethod
	def _describe(cls, record):
		def per_type(counts):
			return ",".join(
				f"{name}:{int(counts[name])}" for name in sorted(counts)
			) or "-"

		return " ".join(
			[f"{name}={per_type(record[name])}" for name in cls._COUNTED]
			+ [f"carried={int(record['carried'])}"]
		)

	@contextmanager
	def scope(self, phase):
		"""One forward loop; every pass it armed is closed on the way out."""
		if self._phase is not None:
			raise RuntimeError(
				f"Rank {self._rank}: a {phase} weight-pass scope was opened "
				f"inside the {self._phase} one"
			)
		self._phase = phase
		self.totals = self._empty_totals()
		try:
			yield self
		except BaseException as error:
			try:
				self._abort()
			except BaseException:
				# The forward's failure stays the error raised; the cleanup
				# failure becomes its context.
				raise error
			raise
		else:
			self.close()
		finally:
			self._phase = None

	def begin(self, carry=False):
		"""Open the pass of the forward about to run."""
		if self._phase is None:
			raise RuntimeError(
				f"Rank {self._rank}: a weight pass was begun outside a scope"
			)
		if self._current is not None:
			raise RuntimeError(
				f"Rank {self._rank}: weight pass {self._current} was not ended "
				"before the next forward began"
			)
		if self._carried is not None:
			self._current, self._carried = self._carried, None
		else:
			self._current = self._core_engine.begin_weight_pass()
		if carry and self._current != 0:
			self._carried = self._core_engine.begin_weight_pass()

	def end(self):
		"""Close the pass of the forward that just ran."""
		if self._current is None:
			raise RuntimeError(f"Rank {self._rank}: no weight pass to end")
		if self._current != 0:
			if self._phase == "prefill" and self.end_delay_s > 0:
				time.sleep(self.end_delay_s)
			self._record(self._core_engine.end_weight_pass(self._current))
		self._current = None

	def close(self):
		"""Cancel and end every pass still open: a carried pass whose forward
		will not run."""
		if self._current is not None:
			self._abandon(self._current)
			self._current = None
		if self._carried is not None:
			self._abandon(self._carried)
			self._carried = None
		if self._phase == "decode" and self.totals["passes"]:
			logging.info(
				"[WEIGHT_PASS] rank %d decode passes=%d %s",
				self._rank,
				self.totals["passes"],
				self._describe(self.totals),
			)

	def _abandon(self, pass_id):
		if pass_id != 0:
			self._core_engine.cancel_weight_pass(pass_id)
			self._record(self._core_engine.end_weight_pass(pass_id))

	def _abort(self):
		"""Drop the passes of a forward that raised without draining them.

		Draining would make the producer retire every remaining module, and a
		node-daemon module retires only once the daemon hands it over, which
		needs the sibling ranks this one has stopped matching. The producer is
		stopped instead (a stop is bounded), and the engine frees every slot
		and drops every open pass. A producer failure raised by the stop is
		chained, never lost. The producer restarts with the next phase
		install; a distributed schedule cannot be resumed and refuses it.
		"""
		open_passes = [p for p in (self._current, self._carried) if p]
		self._current = None
		self._carried = None
		if not open_passes:
			return
		try:
			self._core_engine.stop_h2d_worker()
		finally:
			report = self._core_engine.abort_weight_passes()
			logging.warning(
				"[WEIGHT_PASS] rank %d %s passes %s aborted: %s",
				self._rank,
				self._phase,
				open_passes,
				report,
			)

	def _record(self, report):
		self.totals["passes"] += 1
		for name in self._COUNTED:
			counts = self.totals[name]
			for module_type, count in report[name].items():
				counts[module_type] = counts.get(module_type, 0) + int(count)
		self.totals["carried"] += int(report["carried"])
		# Every rank logs its own passes: node validation compares ranks.
		if self._phase == "prefill":
			logging.info(
				"[WEIGHT_PASS] rank %d pass %d %s",
				self._rank,
				int(report["pass"]),
				self._describe(report),
			)


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


class NodeSharedMemfd:
	"""One node's anonymous memfd for the QueryBook ``input_ids`` table.

	``fd`` is this rank's own descriptor onto the region: the creator's memfd
	itself, or the handle an attacher got from ``/proc/<creator>/fd/<N>``. The
	region carries no name anywhere in the filesystem, so the kernel reclaims
	it with the last mapping however the processes died.

	``release_fd`` drops only the descriptor. ``mapping`` deliberately stays
	open: views handed out before a pool grow may still be live, and unmapping
	under them would segfault. The mapping itself keeps the memfd alive, so a
	released fd never shortens the region's life.
	"""

	def __init__(self, fd: int, mapping):
		self.fd = fd
		self.mapping = mapping

	def release_fd(self) -> None:
		if self.fd < 0:
			return
		try:
			os.close(self.fd)
		except OSError:
			pass
		self.fd = -1


def allocate_node_shared_int64(
	label: str,
	rows: int,
	width: int,
	is_creator: bool,
	exchange,
	barrier,
) -> Tuple[torch.Tensor, NodeSharedMemfd]:
	"""Map ONE int64 ``[rows, width]`` CPU tensor per node into every worker.

	The prompt ids are identical on every rank (every rank receives them in the
	admission message, ``_bind_admitted_prompts``), so each worker used to hold its own
	private copy of the same input-ids table — ``world_size`` duplicates of the
	same bytes, which is what OOM-killed the node.

	``is_creator`` must be true on exactly one rank per node. The creator makes
	an UNNAMED memfd (``ftruncate`` guarantees it is zero-filled, matching the
	``torch.zeros`` it replaces) and the rest open it through
	``/proc/<creator_pid>/fd/<N>``. ``label`` never names an object: it is only
	the ``memfd:`` tag that shows up in ``/proc/<pid>/fd`` and ``/proc/<pid>/maps``.

	``exchange(pid, fd) -> (creator_pid, creator_fd)`` publishes this rank's
	identity and returns the node creator's. It must synchronize — the creator's
	memfd only exists once it has entered — which is why it REPLACES the barrier
	that used to sit here rather than adding to it. ``barrier`` still guards the
	other end: no fd may be released until every rank has mapped the region.

	Returns ``(tensor, memfd)``. The caller MUST keep ``memfd`` alive for as long
	as the tensor is reachable: the tensor points straight into the mapping.
	"""
	import mmap

	nbytes = rows * width * 8
	fd = -1
	if is_creator:
		fd = create_memfd(label)
		os.ftruncate(fd, nbytes)
	creator_pid, creator_fd = exchange(os.getpid(), fd)
	if not is_creator:
		if creator_pid < 0 or creator_fd < 0:
			raise RuntimeError(
				f"no rank published a memfd for the '{label}' input_ids region "
				f"(got pid={creator_pid}, fd={creator_fd})"
			)
		fd = os.open(
			f"/proc/{creator_pid}/fd/{creator_fd}", os.O_RDWR | os.O_CLOEXEC
		)
	try:
		size = os.fstat(fd).st_size
		if size < nbytes:
			raise QueryBookPoolCapacityError(
				f"shared input_ids memfd '{label}' is {size} bytes, "
				f"need {nbytes} ({rows} rows x {width} tokens x 8B)"
			)
		mapping = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
	except BaseException:
		os.close(fd)
		raise
	# Admissions can later mutate this persistent mapping outside InferenceMode.
	with torch.inference_mode(False):
		buf = torch.frombuffer(mapping, dtype=torch.int64, count=rows * width).view(rows, width)
	# Nobody may release an fd until every rank has mapped it.
	barrier()
	return buf, NodeSharedMemfd(fd, mapping)


def allocate_node_shared_tensor(
	label: str,
	rows: int,
	width: int,
	dtype: torch.dtype,
	is_creator: bool,
	exchange,
	barrier,
) -> Tuple[torch.Tensor, NodeSharedMemfd]:
	"""Map one anonymous node-shared CPU tensor into every local worker.

	The creator publishes its memfd identity through ``exchange``. All ranks
	then map the same fixed-size region and wait at ``barrier`` before any
	caller can release its descriptor. This is the int32 counterpart of the
	legacy int64 input table helper and is used by the unified trajectory book.
	"""
	import mmap
	if dtype not in (torch.int32, torch.int64):
		raise ValueError(f"unsupported shared tensor dtype: {dtype}")
	if rows <= 0 or width <= 0:
		raise ValueError(f"shared tensor shape must be positive, got ({rows}, {width})")
	element_bytes = torch.empty((), dtype=dtype).element_size()
	nbytes = rows * width * element_bytes
	fd = -1
	if is_creator:
		fd = create_memfd(label)
		os.ftruncate(fd, nbytes)
	creator_pid, creator_fd = exchange(os.getpid(), fd)
	if not is_creator:
		if creator_pid < 0 or creator_fd < 0:
			raise RuntimeError(
				f"no rank published a memfd for '{label}' "
				f"(got pid={creator_pid}, fd={creator_fd})"
			)
		fd = os.open(f"/proc/{creator_pid}/fd/{creator_fd}", os.O_RDWR | os.O_CLOEXEC)
	try:
		size = os.fstat(fd).st_size
		if size < nbytes:
			raise QueryBookPoolCapacityError(
				f"shared memfd '{label}' is {size} bytes, need {nbytes}"
			)
		mapping = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
		with torch.inference_mode(False):
			buf = torch.frombuffer(mapping, dtype=dtype, count=rows * width).view(rows, width)
	except BaseException:
		os.close(fd)
		raise
	barrier()
	return buf, NodeSharedMemfd(fd, mapping)


class QueryBookBufferPool:
	"""Pre-allocated contiguous buffers for query book tensors.

	Eliminates per-sequence tensor allocation in _bind_admitted_prompts().
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
		input_ids_memfd: Optional["NodeSharedMemfd"] = None,
	):
		if input_ids_buffer is None and input_ids_width > 0:
			with torch.inference_mode(False):
				input_ids_buffer = torch.zeros((num_sequences, input_ids_width), dtype=torch.long)
		elif input_ids_buffer is not None and tuple(input_ids_buffer.shape) != (num_sequences, input_ids_width):
			raise QueryBookPoolCapacityError(
				f"shared input_ids buffer has shape {tuple(input_ids_buffer.shape)}, "
				f"pool needs ({num_sequences}, {input_ids_width})"
			)
		self.input_ids_buffer = input_ids_buffer
		self.input_ids_memfd = input_ids_memfd
		with torch.inference_mode(False):
			self.decoded_tokens_buffer = torch.full((num_sequences, max_decoding_length), pad_token_id, dtype=torch.int64)
		self.pad_token_id = pad_token_id
		self.num_sequences = num_sequences
		self.input_ids_width = input_ids_width
		self.max_decoding_length = max_decoding_length

	def adopt(self, old: "QueryBookBufferPool") -> None:
		"""Carry contents over from a superseded pool."""
		rows = min(self.num_sequences, old.num_sequences)
		cols = min(self.input_ids_width, old.input_ids_width)
		if self.input_ids_buffer is not None and old.input_ids_buffer is not None:
			self.input_ids_buffer[:rows, :cols] = old.input_ids_buffer[:rows, :cols]
		dec = min(self.max_decoding_length, old.max_decoding_length)
		self.decoded_tokens_buffer[:rows, :dec] = old.decoded_tokens_buffer[:rows, :dec]

	def get_input_ids_view(self, slot: int, seq_extended_size: int) -> torch.Tensor:
		if self.input_ids_buffer is None:
			raise QueryBookPoolCapacityError("input_ids are stored in the trajectory book")
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
	gpu_arch: Optional[str] = None
	# EP with offloading settings
	enable_ep_with_offloading: bool = False
	ep_offloading_ratio: float = 0.0
	pre_dequantize_weights: bool = False
	distributed_weight_config: Optional[str] = None
	enable_deepep: bool = False  # Enable DeepEP low-latency EP exchange for decode (default: NCCL)

	def __post_init__(self):
		if self.max_prompt_length is None and self.padding_length is not None:
			self.max_prompt_length = self.padding_length
		elif self.max_prompt_length is not None:
			self.padding_length = self.max_prompt_length
		if self.gpu_arch is None:
			self.gpu_arch = detect_gpu_arch()

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
	local_world_size: int
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
	enable_prefix_cache: bool = False
	prefix_cache_debug_stats: bool = False
	# Charge a prefix several sequences read from one physical copy once per
	# resource instead of once per row. Default on; the opt-out restores the
	# per-row logical charge for comparison runs.
	shared_prefix_admission: bool = True

	# Watchdog configuration
	watchdog_timeout: Optional[float] = 600.0  # Seconds before declaring process stuck (10 min for long inference)
	watchdog_test_stuck_time: float = 0.0  # Deliberate delay for testing
	watchdog_heartbeat_interval: Optional[float] = None  # Heartbeat interval
	decode_step_timeout: Optional[float] = None  # Max seconds per decode step

	# Prepack optimization (default: enabled, recommended always on)
	enable_prepack: bool = True
	# DIAGNOSTIC (glm52 memcpy32_post A/B): capture the GLM-5 whole-model
	# decode graph without the per-layer host-KV staging copies. Decode tokens
	# are then never appended to host KV, so this must stay off in production.
	glm5_disable_decode_host_kv_staging: bool = False
	# Host KV watermark percentage (default: 70% free = underutilized threshold)
	host_kv_watermark: int = 70
	# Decode preemption: interrupt decode for prefill when host KV is underutilized (default: enabled)
	enable_decode_preemption: bool = True
	# GPU memory fraction for KV cache calculation (default: 0.9)
	gpu_memory_frac: float = 0.9
	max_model_len: Optional[int] = None
	# GPU page buffer settings for decode scheduling
	initial_gpu_page_buffer: int = 32  # Pages to reserve on first GPU load
	extension_gpu_page_buffer: int = 4  # Pages to add at boundaries
	decision_frequency_pages: int = 2  # How often to make scheduling decisions (in pages)

	# EP with offloading settings
	enable_ep_with_offloading: bool = False  # Enable Expert Parallelism with offloading
	ep_offloading_ratio: float = 0.0  # Ratio of experts per layer to offload (0.0-1.0)
	pre_dequantize_weights: bool = False  # Pre-dequantize MoE routed expert MXFP4 weights to BF16
	enable_deepep: bool = False  # Enable DeepEP low-latency EP exchange for decode (default: NCCL)
	enable_cuda_graph: bool = False  # Explicitly enable CUDA graph capture for supported models
	disable_cuda_graphs: bool = True  # Disable CUDA graph capture for decode attention (default: off due to 128K+ crash)
	cuda_graph_max_bucket_size: int = 128  # Max batch size per rank for CUDA graph capture
	cuda_graph_num_buckets: int = 16  # Number of CUDA graph bucket sizes
	persistent_phase_instances: bool = False  # Keep prefill/decode instances across phase switches
	detokenization_include_special_tokens: bool = False  # When True, include special tokens in detokenized output
	# Dynamic host KV reservation
	host_kv_chunk_size: int = 8192  # Initial host KV chunk size in tokens
	enable_host_kv_eviction: bool = False  # Enable host KV eviction + recompute
	adaptive_chunk: bool = True  # EMA-based adaptive chunk sizing
	adaptive_chunk_min: int = 1024
	adaptive_chunk_max: int = 65536
	adaptive_chunk_ema_alpha: float = 0.1
	adaptive_chunk_multiplier: float = 1.5
	# --fast-init (THP + prefault). Host KV, model weights and tensor metadata
	# always use memfd_create, so the creator pids and fds below are populated
	# in every mode. weights_memfd_pid owns both weights and tensor metadata;
	# --enable-hugetlbfs only changes the weights memfd's page size.
	fast_init: bool = False
	# Prefix-cache coordinator region (anonymous memfd owned by the server).
	prefix_memfd_pid: int = -1
	prefix_memfd_fd: int = -1
	kv_memfd_pid: int = -1
	kv_memfd_fd: int = -1
	kv_aux_memfd_fd: int = -1  # Separate memfd fd for auxiliary (indexer) KV cache
	weights_memfd_pid: int = -1
	weights_memfd_fd: int = -1
	tensor_meta_memfd_fd: int = -1
	distributed_weight_config: Optional[str] = None
	# Worker cohort epoch the distributed-weight daemon admits for this spawn.
	distributed_weight_cohort: int = 0
	# Request pool: max QueryBook capacity (pre-allocated, metadata only)
	max_pool_size: int = 10240  # Deprecated compatibility field.
	input_ids_pool_size_gb: float = 50.0  # Fixed node-shared int32 token budget.
	host_kv_shm_name: str = "batchgen_host_kv_cache"
	host_kv_aux_shm_name: str = "batchgen_host_kv_cache_aux"
	prefix_cache_shm_name: str = "batchgen_prefix_cache"
	query_book_shm_prefix: str = "batchgen_input_ids"
	runtime_dir: str = "/tmp"
	pynccl_port_base: int = 20003
	pynccl_port_span: int = 100
	# Structured event log of this rank: the server's log root and level, and
	# the run directory every process of the run writes under (see
	# batchgen.eventlog.process). None log_dir means the eventlog default.
	log_dir: Optional[str] = None
	log_level: str = "INFO"
	run_id: Optional[str] = None
	run_started_at: Optional[float] = None
	# Restart generation of this worker cohort (0 at the first launch), stamped
	# as ``cohort`` on every record so successive cohorts stay distinguishable
	# in the appended per-rank files.
	log_cohort: int = 0


def _reset_glm5_dsa_step_state(wrapper_cls, cache_seqlens, index_topk) -> None:
	"""Per-decode-step DSA state reset for the eager (second) decode path.

	Computes the dense short-circuit hint once per step (77 of 78 D2H syncs
	saved on DSA models), and ALWAYS clears the previous step's reused top-k
	indices before layer 0 runs. The clear must be unconditional: when the
	model config carries ``index_topk`` (GLM-5.2 does), a clear nested under
	the hint's else-branch never executes and shared reuse-topk layers can
	consume a stale top-k from the previous step. The graph-config path
	resets it separately.
	"""
	# NOTE: for GLM-5.2 this branch is currently unreachable, and that is a
	# defect being tracked elsewhere, not a property to rely on: the engine
	# ModelConfig projection drops ``index_topk``, so the caller's
	# ``getattr(self.model_config, "index_topk", None)`` is always None and
	# ``_dsa_short_count`` stays None. See sub-project
	# ``batchgen-glm52-index-topk-config-projection-loss``, which owns the
	# fix (project the field / fail loud instead of defaulting to None). Once
	# that lands, this ``.sum().item()`` becomes a live per-step device sync
	# on the eager selector path and has to be gated on the non-graph path —
	# so the line stays here rather than being deleted as dead code.
	if index_topk is not None:
		wrapper_cls._dsa_short_count = int(
			(cache_seqlens <= int(index_topk)).sum().item()
		)
	else:
		wrapper_cls._dsa_short_count = None
	wrapper_cls._dsa_prev_topk_indices = None


class BatchGenWorker:
	"""
	Inference Runtime with Host-KV-First scheduling and Continuous Batching.
	"""
	PAGE_SIZE = 64  # Tokens per page (fixed)
	# Decision frequency: check boundaries every N pages. Import-time default
	# only; __init__ rebinds it after configure_page_buffers().
	DECISION_INTERVAL = _sequence_module.DECISION_FREQUENCY_PAGES * 64  # Tokens between boundary checks


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
		# The deprecated CLI switch cannot disable re-entry. Any positive chunk
		# creates bounded host-KV reservations, so eviction is part of that mode's
		# correctness contract.
		self.enable_host_kv_eviction = host_kv_eviction_enabled(self.host_kv_chunk_size)
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
				f"eviction_enabled={self.enable_host_kv_eviction}, "
				f"adaptive_chunk={args.adaptive_chunk}"
			)

		# Page boundary counter for periodic diagnostic logging
		self._boundary_count = 0

		# Watchdog for stuck detection (can be set via set_watchdog())
		self._watchdog = None
		# Decode watchdog: per-decode-step timeout (separate from general watchdog)
		self._decode_watchdog = None

		# Cohort teardown probe, set by the worker supervisor before the main
		# loop (see set_cohort_shutdown_probe).  None until then, and on every
		# path that never serves a pool, so every call site must tolerate it.
		self._cohort_shutdown_probe = None
		# Latched by the decode-step boundary check so the supervisor can name
		# the exit path this rank actually took.  Never cleared: a cohort
		# teardown is terminal for this process.
		self._cohort_shutdown_observed = False

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
		self.local_world_size = args.local_world_size
		# S3.5 state owners.  They are created eagerly so flag-off paths are
		# plain field reads and the pool pseudo-sid seed is unconditional.
		self._traj = TrajectoryStore(kda_wrapper_loader=_load_kda_wrapper)
		self._prefix = PrefixReuseState(rank=self.rank, world_size=self.world_size)
		self.pynccl_port_base = args.pynccl_port_base
		self.pynccl_port_span = args.pynccl_port_span
		self._nccl_port = self.pynccl_port_base
		self.gpu_arch = args.gpu_arch
		self.kv_dtype = args.kv_dtype
		self.device = args.device
		self.torch_device = torch.device(f"cuda:{args.device}")

		# Monotone stamp for the ORDERED decode row set: the local-index list
		# `batch` and the `_local_to_uuid_map` it is resolved through. Decode
		# keeps per-request sampling-parameter tensors resident on device and
		# may only reuse them while row set AND row order are unchanged.
		# `SequenceBatch.composition_epoch` covers the parameter source
		# (membership); this covers the rows. Keying on membership alone would
		# silently mis-order the temperature vector after a page boundary.
		self._decode_row_epoch = 0

		# CUDA graph state: every field of the decode-graph state machine is
		# owned by `batchgen.worker.graph_runtime.GraphRuntime`, which also
		# owns the cuda-graph-layer imports this module used to make.
		self._graph = GraphRuntime()
		self._prefill_engine = PrefillEngine()
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
		# Decode criticality probe. The model's graph adapter parses the request
		# from batchgen_debug and builds the probe; the worker only holds the
		# round's request (set at decode configure), the probe installed in the
		# resident decode graph, and the one-shot recapture a changed request
		# arms. All None/False without a request.
		self._criticality_probe = None
		self._criticality_probe_request = None
		self._criticality_probe_recapture = False
		self._nsys_decode_profile_forward_count = 0
		self._nsys_decode_profile_started = False
		self._nsys_decode_profile_stopped = False
		self._decode_local_count_tensor = None
		self._decode_all_rank_counts = None
		self._decode_cache_seqlens_i32 = None
		self._decode_position_ids_i64 = None
		self._decode_cache_seqlens_cpu_staging = None
		# Completion event of the last H2D copy out of the pinned staging
		# above; only recorded under DECODE_DEFERRED_TOKEN_READBACK.
		self._decode_metadata_staging_event = None
		# True while a decode step's sampled tokens are launched but unread
		# (deferred readback); host-token consumers fail loud on it.
		self._decode_tokens_unread = False
		self._decode_metadata_batch_key = None
		self._decode_metadata_cpu_seqlens = None
		# Kimi-K3 streamed-SP8 prefill ring. Resident decode keeps its prefill
		# layout, so only the first install allocates it; the fingerprint pins
		# the schedule that install handed the distributed weight transport.
		self._streamed_sp8_h2d_installed = False
		self._streamed_sp8_weight_copy_fingerprint = None
		# Kimi-K3 pre-readiness startup (prepare_kimi_k3_startup). The first
		# flag makes that pass idempotent; the second hands the prepared prefill
		# phase to the first real admission so it is not torn down and rebuilt.
		self._k3_startup_completed = False

		# 2. Set Device immediately
		torch.cuda.set_device(self.local_rank)

		# Process-global DeepGEMM PDL toggle; must precede every deep_gemm
		# call (CUDA graph capture records the launch attribute).
		if apply_deepgemm_pdl() and self.rank == 0:
			logging.info("DeepGEMM PDL enabled for this worker process")

		# 3. Path & Model Configurations
		self.model_name = args.model_name
		self.huggingface_ckpt_name = args.model_name
		self.hf_cache_dir = args.hf_cache_dir
		self.cache_dir = args.cache_dir
		self.converted_ckpt_dir = args.converted_ckpt_dir

		# Load skeleton_state_dict (avoids passing tensors through mp.spawn). The
		# server holds it in an anonymous memfd, so this path is its
		# /proc/<server pid>/fd/<N> — nothing named to leave behind on a crash.
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
		self.glm5_disable_decode_host_kv_staging = args.glm5_disable_decode_host_kv_staging
		if self.glm5_disable_decode_host_kv_staging:
			logging.warning(
				"GLM-5 decode host-KV staging DISABLED (diagnostic A/B): "
				"decode tokens will NOT be appended to host KV"
			)
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
		self.weights_storage = core_engine.Weights_Storage(
			self.local_rank, args.log_level
		)
		if args.distributed_weight_config:
			self.weights_storage.InitDistributed(
				args.distributed_weight_config,
				args.distributed_weight_cohort,
			)
			self.skeleton_state_dict = self.weights_storage.get_tensor(
				"__skeleton__"
			)
			logging.info(
				f"Rank {self.rank}: Loaded compact skeleton state dict "
				f"with {len(self.skeleton_state_dict)} keys"
			)
		else:
			# The server's weights and tensor metadata are anonymous memfds in
			# every mode, so this attach always goes through
			# /proc/<creator>/fd/<N>; --fast-init only adds THP.
			self.weights_storage.Init(
				self.shm_name,
				self.weight_byte_size,
				self.tensor_meta_shm_name,
				self.enable_hugetlbfs,
				enable_thp=args.fast_init,
				memfd_creator_pid=args.weights_memfd_pid,
				memfd_fd=args.weights_memfd_fd,
				tensor_meta_memfd_fd=args.tensor_meta_memfd_fd,
			)
		logging.info(f"Rank {self.rank}: [startup] Weights storage init: {_time.monotonic() - _t0:.2f}s")

		# 5. Initialize Host KV Cache Manager View (cudaHostRegister for Host KV)
		self.host_kv_cache_size = args.host_kv_cache_size
		self.global_host_kv_cache_size_gb = args.global_host_kv_cache_size_gb
		self.enable_prefix_cache = bool(args.enable_prefix_cache)
		self.prefix_cache_debug_stats = bool(args.prefix_cache_debug_stats)
		self.shared_prefix_admission = bool(
			getattr(args, "shared_prefix_admission", True)
		)
		self.prefix_cache_runtime_config = None
		self.prefix_cache_coordinator = None
		# Edge trigger for the prefix_drain metric: set when the row is
		# emitted, cleared when admission adds new sequences.
		self._prefix.prefix_drain_metric_emitted = False
		self._prefix.prefix_sequence_states = {}
		self._prefix.active_prefix_materializations = []
		self._prefill_pool_wave = None
		# Prefill-phase pool page budget: measured after configure_prefill,
		# None until the first prefill of this deployment.
		self._prefill_pool_page_budget = None
		# Device bytes a prefill forward may hold as prefix KV beside the
		# planner's forward workspace, measured at the same point.
		self._prefill_prefix_kv_budget_bytes = None
		# (allocated, reserved, device use outside the allocator) once the pool is up
		self._prefill_pool_base = (0, 0, 0)
		# Pseudo Host ids must not collide with entries a previous worker of
		# this node left in the shared Host KV region (hot reload).
		# ``PrefixReuseState`` seeded and initialized these fields eagerly.
		# Scheduler-loop hard guard against identical empty prefill waves with
		# nothing live; _prepare_prefill_batch publishes the rank-identical
		# admission state (and its selection snapshot) the guard compares.
		self._prefill_progress_guard = PrefillProgressGuard()
		self._prefill_admission_state = None
		self._prefill_admission_selection = None
		self._prefill_admission_totals = None
		# Reduced across ranks by the admission gather, so the guard's verdict
		# cannot differ by rank.
		self._prefill_admission_live = False
		# None / "committed" / "rejected_by_capacity_snapshot" /
		# "rolled_back_after_allocator_failure": the wave's Host reservation
		# outcome, folded into the guard state so a rejected transaction is
		# not mistaken for a wave that selected nothing.
		self._prefill_reservation_outcome = None
		# Shared-prefix admission credit: charge an in-wave shared prefix once
		# per node instead of once per row. Suspended for the NEXT admission
		# round whenever a credited wave is rejected by the reservation
		# transaction -- the wave planner may decline to pool (a budget too
		# small for the chain, the threshold ladder), and then each member
		# really does compute and allocate its own prefix. Re-admitting the
		# same credited wave would then repeat the same rejection forever, so
		# the retry falls back to the full charge, makes progress, and the
		# latch clears on the next committed reservation. Rank-uniform: both
		# the rejection and the commit are decided by a collective.
		self._prefix.shared_prefix_credit_suspended = False
		self._prefix.shared_prefix_credit_applied = False
		# global_idx -> cumulative per-page prompt digests, one entry per
		# SHAREABLE page (a prompt's last page is private by construction, so
		# every row keeps a tail). Cached because a digest is a property of the
		# prompt, not of the round: recomputing it on every scheduler iteration
		# would re-hash the whole candidate set each time.
		self._prefix.prefix_page_digest_cache = {}
		# M3 trajectory frontier: built on first flag-on commit, None otherwise.
		self._traj.trajectory_frontier_committer = None
		# M3 Slice 2: built on the first flag-on lookup that finds a committer,
		# None otherwise. global_idx -> the live TrajectoryAttachment a hit
		# took; empty whenever the flag is off, so the restore seam is free.
		self._traj.trajectory_restorer = None
		self._traj.trajectory_attachments = {}
		# M3 Slice 1c, G>1 only. Both are per-PASS and both are published by
		# the serve-group leader on the host-KV mirror that already runs in
		# prefill config, so no new collective exists:
		#   _trajectory_arena_payload: global_id -> the leader's G-wide arena,
		#     so every rank can address its own shard window at the commit
		#     boundary;
		#   _trajectory_mirrored_restores: global_id -> the leader's frontier
		#     HIT for this pass, so the group restores on one decision instead
		#     of G independent ones.
		self._traj.trajectory_arena_payload = {}
		self._traj.trajectory_mirrored_restores = {}
		# M3 Slice 3a: the WORLD's session -> owning-group map, folded from
		# every leader's summary on that same mirror. Built on the first fold,
		# None whenever the trajectory flag never turned on, so the group
		# assignment of a flag-off run reads one attribute and moves on.
		self._traj.trajectory_session_groups = None
		if self.enable_prefix_cache:
			from batchgen.prefix_reuse.config import (
				require_prefix_cache_model_support,
			)

			require_prefix_cache_model_support(args.model_name)

		# DSA models: create DualHostKVCoordinator with proportional budget split.
		# Non-DSA models get a single-view worker below.
		host_budget_bytes = int(args.global_host_kv_cache_size_gb * (1024**3))
		dual_host = DualHostKVCoordinator.from_budget(
			model_name=args.model_name,
			host_kv_cache_size=host_budget_bytes,
			core_engine_module=core_engine,
			enable_thp=args.fast_init,
			memfd_creator_pid=args.kv_memfd_pid,
			memfd_fd=args.kv_memfd_fd,
			aux_memfd_fd=args.kv_aux_memfd_fd,
			primary_shm_name=args.host_kv_shm_name,
			aux_shm_name=args.host_kv_aux_shm_name,
		)
		if dual_host is not None:
			self.host_paged_kv_worker_view = dual_host
			logging.info(f"Rank {self.rank}: Initializing DualHostKVCoordinator with parallel cudaHostRegister (local_rank={self.local_rank})")
			dual_host.initialize(device_index=self.local_rank, create_region=False)
			logging.info(f"Rank {self.rank}: DualHostKVCoordinator cudaHostRegister completed (local_rank={self.local_rank})")
			# The same split the views were built from (MLA latent + indexer keys).
			prefix_host_config, prefix_aux_config = DualHostKVCoordinator.build_host_configs(
				args.model_name, host_budget_bytes,
				args.host_kv_shm_name, args.host_kv_aux_shm_name,
			)
		else:
			worker_kv_config = build_host_kv_config(
				model_name=args.model_name,
				host_kv_cache_size=host_budget_bytes,
				shm_name=args.host_kv_shm_name,
			)
			# The server's host KV region is an anonymous memfd in every mode;
			# this view always attaches through /proc/<creator>/fd/<N>.
			worker_kv_config.enable_thp = args.fast_init
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
			prefix_host_config, prefix_aux_config = worker_kv_config, None

		if self.enable_prefix_cache:
			from batchgen.prefix_reuse.config import (
				build_prefix_cache_runtime_config,
				create_host_prefix_cache_coordinator,
			)

			self.prefix_cache_runtime_config = (
				build_prefix_cache_runtime_config(
					model_name=args.model_name,
					kv_dtype=args.kv_dtype,
					host_kv_config=prefix_host_config,
					aux_host_kv_config=prefix_aux_config,
					shm_name=args.prefix_cache_shm_name,
					debug_stats=self.prefix_cache_debug_stats,
					memfd_creator_pid=args.prefix_memfd_pid,
					memfd_fd=args.prefix_memfd_fd,
				)
			)
			self.prefix_cache_coordinator = (
				create_host_prefix_cache_coordinator(
					core_engine_module=core_engine,
					runtime_config=self.prefix_cache_runtime_config,
					create_region=False,
				)
			)

		# 6. Initialize Placeholders for Core Components
		# These are populated later in Init() / _initialize_core_components
		self.gpu_paged_kv_cache_manager = None
		# Set once by _bind_gpu_paged_kv_manager and never cleared: whether
		# this process has ever created a GPU KV pool.
		self._gpu_kv_pool_created = False
		# Compatibility key for the currently entered manager phase.
		self._phase = None
		self._live_phase_key = None
		# Prefill MoE mode of the most recent prefill entry; see
		# _desired_phase_key("decode").
		self._prefill_moe_mode_in_effect = None
		self.model = None
		self.model_config = None
		self.loaded_model_config = None
		self.engine_config = None
		self.core_engine = None
		# Weight passes of core_engine's model forwards, bound with it in
		# _initialize_core_components().
		self._weight_passes = None
		self.tokenizer = None
		self.initializer = None
		self.parallel_manager = None
		
		# 7. Batch State Placeholders
		self.global_batch: Optional[SequenceBatch] = None
		self.query_book: Optional[Dict] = None
		self.model_batch_book: Dict = {}
		# Local slot binding (both maps) and its free-slot allocator live in
		# ONE owner; `_local_to_uuid_map` / `_uuid_to_local_map` below are
		# read-only views of it. Every write goes through `self.seq_table`.
		self.seq_table = SequenceTable()
		# Frozen rank identity. Built in `Init` once the parallel strategy
		# manager exists, because the decode attention TP size G comes from it.
		self.rank_ctx: Optional[RankContext] = None
		
		# 8. Runtime State
		self.eos_token_id: Optional[int] = None
		self._stop_token_ids: set = set()
		self.max_input_length = 0
		self.max_decoding_length = 0
		self.model_context_length = None  # Updated from model config during init
		self.num_global_queries = 0
		self.num_local_queries = 0
		self._logged_sampling: bool = False  # Track if we've logged sampling mode this batch
		self._batchgen_debug: Optional[dict] = None
		# Sticky: set once an admitted sequence carries per-request debug
		# flags. Only gates the per-row scan in
		# _active_batchgen_debug_for_sequences, whose result is None while no
		# sequence carries any, so a sticky True merely restores the scan.
		self._any_sequence_batchgen_debug: bool = False

		# 9. Initialization Flags
		self._core_initialized = False
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
		# Measure-once-cache sizing (semantic contract, POIS 2026-10-05): the
		# pool size is measured once after the full graph warmup/capture and
		# cached; every later pool (re)creation reuses the cached value.
		self._gpu_kv_cached_pool_gb: Optional[float] = None
		self._gpu_kv_sizing_provisional: bool = False
		# Deployment-wide startup decode transient, one dict per flavor, set
		# by _probe_startup_transient before readiness and published with the
		# deployment capacity. Report-only: nothing sizes from it.
		self.startup_transient: Optional[List[dict]] = None
		
		# Track sequences currently with GPU KV allocated
		self._sequences_with_gpu_kv: Set[str] = set()

		# Request pool: admission queue and response queue for persistent loop
		self._admission_queue = None  # mp.Queue, set via set_admission_queue()
		self._response_queue = None   # mp.Queue, set via set_response_queue()
		self._shutdown_requested = False
		self._max_pool_size = args.max_pool_size  # Deprecated; no capacity semantics.
		self._input_ids_pool_size_gb = float(args.input_ids_pool_size_gb)

		# Legacy pool fields remain only for checkpoint/object compatibility; the
		# production path uses the fixed trajectory book initialized with the model.
		self._buffer_pool: Optional[QueryBookBufferPool] = None
		self._traj.trajectory_book = None
		self._traj.trajectory_memfd: Optional[NodeSharedMemfd] = None
		self._prompt_tensors: Dict[str, torch.Tensor] = {}
		self._buffer_pool_generation = 0
		self._query_book_shm_prefix = args.query_book_shm_prefix
		# Superseded pools stay mapped for the process lifetime (see
		# _retire_buffer_pool).
		self._retired_buffer_pools: List[QueryBookBufferPool] = []

		logging.info(f"Rank {self.rank}: BatchGenWorker __init__ completed.")

	# ============ Local slot binding (owned by SequenceTable) ============
	#
	# Read-only views. `SequenceTable` is the single writer: bind / unbind /
	# reset. Mutating one of these raises, which is the point — a stray
	# `pop`/`[...]=` here used to be how the two maps drifted apart.

	@property
	def _local_to_uuid_map(self) -> Mapping[int, str]:
		return self.seq_table.local_to_uuid

	@property
	def _uuid_to_local_map(self) -> Mapping[str, int]:
		return self.seq_table.uuid_to_local

	@property
	def _free_local_indices(self) -> AbstractSet[int]:
		return self.seq_table.free_local_indices

	@property
	def _next_local_idx(self) -> int:
		return self.seq_table.next_local_idx

	def Init(self, max_input_length, max_decoding_length, num_queries):
		"""
		Initialize/reconfigure before the persistent generate() loop starts.
		- First call: performs full initialization of core_engine, parallel_manager, etc.
		- Subsequent calls: only updates batch parameters.

		Every caller runs before generate_persistent(), i.e. while global_batch
		is still None; the per-batch state reset the legacy path needed between
		two global batches went away with that path.

		Args:
			max_input_length: Maximum input length hint. If None, will be determined dynamically
			                  during tokenization as the longest prompt in the batch.
			                  For first initialization, a default of 8192 is used if None.
			max_decoding_length: Decode-budget hint for init-time sizing. In
			                     pool mode it is the budget of the first batch the
			                     server saw, so nothing may cap a sequence with it:
			                     each admission entry carries that sequence's own
			                     budget (``kv_token_budget``).
			num_queries: Number of queries in the global batch.
		"""
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

		# Freeze this rank's identity. G (`_decode_attn_tp_size`) reads the
		# parallel strategy manager, which `_initialize_core_components`
		# constructs, so this is the earliest point it is knowable. Every
		# field is stable for the process lifetime, so a re-Init rebuilds an
		# equal object.
		self.rank_ctx = RankContext(
			rank=self.rank,
			world_size=self.world_size,
			local_rank=self.local_rank,
			local_world_size=self.local_world_size,
			decode_group_size=self._decode_attn_tp_size(),
		)

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

			# Formula: gpu_kv_cache = total * frac - used (single definition:
			# kv_plan.provisional_pool_gb).
			# This reserves (1-frac) of GPU memory for activations and overhead
			calculated_gb = provisional_pool_gb(
				total_mem_gb, self.gpu_memory_frac, used_mem_gb, floor_gb=None
			)
			# Ensure positive value: a non-positive remainder warns here and is
			# replaced by the 1 GB floor kv_plan substitutes.
			gpu_kv_cache_gb = provisional_pool_gb(
				total_mem_gb, self.gpu_memory_frac, used_mem_gb, floor_gb=1.0
			)
			if calculated_gb <= 0:
				logging.warning(
					f"[GPU-KV] Calculated size is non-positive ({calculated_gb:.2f} GB). "
					f"Total: {total_mem_gb:.2f} GB × frac: {self.gpu_memory_frac} - used: {used_mem_gb:.2f} GB. "
					f"Using minimum 1 GB."
				)

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

	def _kv_fixed_address_pool_available(self) -> bool:
		"""Whether the KV pools can be carved from a reserved virtual range.

		CUDA VMM is the only implementation, and it needs an indexed device.
		Every serving path has one; CPU-only unit tests that build a manager
		directly keep the allocator-backed default.
		"""
		return bool(torch.cuda.is_available() and isinstance(self.local_rank, int))

	def _initialize_gpu_kv_manager_fixed_size(self) -> GPUPagedKVCacheManager:
		"""
		Initialize GPU KV manager with pre-determined fixed size.
		Called once at the start of decoding.

		For DSA models, splits the memory budget between primary (MLA) and
		auxiliary (indexer) caches, wrapping both in a DualKVCacheCoordinator.

		The pools are carved from one reserved virtual range per pool
		(``fixed_address_pool``), so the phase-switch lifecycle can hand their
		physical pages back to the driver for the prefill phase and map them
		again afterwards WITHOUT the pool base, the page-pointer tables or the
		page-table storage moving. A captured decode graph bakes those
		pointers, so releasing an allocator-backed pool under it would make the
		next replay read freed memory.
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
				# Engine-layer -> cache-slot map, exactly as the dense builder
				# passes it. A fully-DSA model (GLM) has None here (engine ==
				# physical); an interleaved model (Kimi-K3.1: 24 MLA rows among
				# 93 engine layers) addresses BOTH regions by engine layer id,
				# and without the map the primary pool bounds-checks engine ids
				# against its 24 physical slots and refuses layer 27 at the
				# first decode-graph capture.
				logical_to_physical_layer=(
					list(primary_profile.logical_to_physical_layer)
					if getattr(primary_profile, "logical_to_physical_layer", None)
					is not None
					else None
				),
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
				logical_to_physical_layer=(
					list(aux_profile.logical_to_physical_layer)
					if getattr(aux_profile, "logical_to_physical_layer", None) is not None
					else None
				),
			)
			aux_config = self._with_cuda_graph_page_table_capacity(aux_config)

			fixed_addresses = self._kv_fixed_address_pool_available()
			primary = GPUPagedKVCacheManager(
				config=primary_config,
				device=self.local_rank,
				fixed_address_pool=fixed_addresses,
			)
			primary.initialize()
			auxiliary = GPUPagedKVCacheManager(
				config=aux_config,
				device=self.local_rank,
				fixed_address_pool=fixed_addresses,
			)
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
				fixed_address_pool=self._kv_fixed_address_pool_available(),
			)
			manager.initialize()
			self._bind_gpu_paged_kv_manager(manager)

			if self.rank == 0:
				logging.info(
					f"[GPU-KV] Initialized: {self.gpu_kv_cache_size_gb:.2f} GB, {config.num_pages} pages"
				)

			return manager
		
	def _active_batchgen_debug_for_sequences(self, batch_sequences) -> Optional[dict]:
		if self._batchgen_debug:
			return self._batchgen_debug
		if not self._any_sequence_batchgen_debug:
			# No sequence admitted on this worker ever carried per-request
			# debug flags, so the merge below can only return None. Decode
			# calls this every step, once per row.
			return None
		merged = {}
		for seq in batch_sequences or []:
			seq_debug = getattr(seq, "batchgen_debug", None)
			if isinstance(seq_debug, dict):
				for key, value in seq_debug.items():
					if value is not None and key not in merged:
						merged[key] = value
		return merged or None

	def _glm5_dispatch_trace_enabled(self, debug: Optional[dict]) -> bool:
		return isinstance(debug, dict) and self._debug_flag_enabled(debug.get("glm5_dispatch_trace"))

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
					# A legacy /v1/inference payload. It matches no branch above,
					# so it used to be dropped right here while the caller sat on
					# response_queue.get() and took the next batch's completion.
					# Nothing in the tree still builds one (the HTTP route returns
					# 410 and WorkerManager.infer is gone), so reaching this line
					# means some other producer is putting legacy payloads on the
					# queue: fail loudly rather than park. Deliberately NOT a
					# Unknown messages must fail loudly rather than being swallowed.
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

		Adds sequences to an already-running generate() loop without resetting
		state: create entries, bind their prompt ids, assign ranks, build the
		query book.

		Args:
			msg: Dict with keys:
				- "entries": List of dicts, each with "request_id", "input_ids"
				  (the prompt token ids, a 1-D int32 numpy array), "max_tokens",
				  "batch_id", "priority", and optionally "sampling_params"
		"""
		entries = msg.get("entries", [])
		if not entries:
			return
		# Each entry's own budget is the only decode cap. The worker-wide
		# max_decoding_length is the first pool batch's budget, so defaulting
		# to it would silently truncate a later request. Checked before any
		# entry is created so a malformed message admits nothing.
		missing_budget = [
			entry.get("request_id") for entry in entries
			if entry.get("max_tokens") is None
		]
		if missing_budget:
			raise ValueError(
				f"admission entries {missing_budget!r} carry no max_tokens; every "
				"admitted sequence must carry its own decode budget"
			)
		# New live work: the next drain is a new edge for the prefix_drain row.
		self._prefix.prefix_drain_metric_emitted = False

		# Determine starting global_idx (continue from existing batch)
		existing_max_idx = max(
			(seq.global_idx for seq in self.global_batch), default=-1
		)
		start_idx = existing_max_idx + 1

		# Step 1: Create SequenceEntry objects
		admitted_at = time.time()
		new_uuids = []
		prompt_ids = []
		for i, entry in enumerate(entries):
			global_idx = start_idx + i
			max_dec = int(entry["max_tokens"])
			prompt_ids.append(_admitted_prompt_ids(entry))
			seq = SequenceEntry(
				uuid=entry["request_id"],
				global_idx=global_idx,
				prompt_length=0,  # Set when the prompt is bound
				max_decode_length=max_dec,
			)
			seq.batch_id = entry.get("batch_id")
			seq.batchgen_debug = entry.get("batchgen_debug")
			if isinstance(seq.batchgen_debug, dict) and seq.batchgen_debug:
				self._any_sequence_batchgen_debug = True
			seq.priority = entry.get("priority", 0)
			seq.sampling_params = entry.get("sampling_params")
			# Per-request ignore_eos: _should_stop_at_eos(token, seq) never marks
			# this sequence done on an EOS or stop token.
			seq.ignore_eos = bool(entry.get("ignore_eos", False))
			seq.traj_id = entry.get("traj_id")
			seq.session_id = entry.get("session_id")
			seq.parent_session_id = entry.get("parent_session_id")
			seq.admitted_at = admitted_at
			self.global_batch.add_sequence(seq)
			new_uuids.append(seq.uuid)
		self._group_uuids.update(new_uuids)

		# Step 2: Bind the prompt ids (identical on every rank) into the book
		self._bind_admitted_prompts(new_uuids, prompt_ids)

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

		# Step 3: Build local query book entries for new sequences. Ownership and
		# decode groups were assigned during tokenization, before arena views were
		# materialized.
		self._build_local_query_book_for_admitted(new_uuids)

		for uuid in new_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is not None and seq.session_id is not None and seq.assigned_rank == self.rank:
				logging.info(
					"[SESSION] admit request=%s traj=%s session=%s parent=%s "
					"prompt_tokens=%d rank=%d",
					seq.uuid, seq.traj_id, seq.session_id, seq.parent_session_id,
					seq.prompt_length, self.rank,
				)

		if self.rank == 0:
			logging.info(
				f"[ADMIT] Admitted {len(entries)} sequences "
				f"(global_idx {start_idx}-{start_idx + len(entries) - 1}), "
				f"global_batch now has {len(self.global_batch)} sequences"
			)

	def _bind_admitted_prompts(
		self, uuids: List[str], prompt_ids: List[torch.Tensor]
	) -> None:
		"""Bind newly admitted prompts into trajectory reservations.

		``prompt_ids[i]`` is the prompt of ``uuids[i]`` exactly as the server
		built it: every rank receives the same ids in the admission message, so
		every rank runs this deterministically without a collective. Prompts
		that do not fit the model context are rejected; the rest are written
		into the fixed unified token pool. Only the new sequences are
		processed; the pool reservation is never resized or copied during a
		later admission.
		"""
		sequences = [self.global_batch.get_sequence(u) for u in uuids]
		lengths = [int(token_ids.numel()) for token_ids in prompt_ids]

		# Reject sequences exceeding context length
		rejected_uuids = []
		for seq, length in zip(sequences, lengths):
			if length < self.model_context_length:
				continue
			rejected_uuids.append(seq.uuid)
			if self.rank == 0:
				logging.warning(
					f"[ADMIT] Rejecting {seq.uuid}: prompt length {length} >= "
					f"model context {self.model_context_length}"
				)
			if self._response_queue is not None and self.rank == 0:
				self._response_queue.put({
					"type": "completion",
					"request_id": seq.uuid,
					"batch_id": getattr(seq, 'batch_id', None),
					"error": {
						"code": "context_length_exceeded",
						"message": (
							f"Prompt length {length} exceeds "
							f"model context {self.model_context_length}"
						),
					},
					"text": "",
				})
			self.global_batch.remove_sequence(seq.uuid)

		# Token lengths are now known. Ownership is assigned AFTER the
		# trajectory-book write below: sharing-aware assignment reads prompts
		# through the book, which is the only token store after admission.
		valid_uuids = []
		for i, seq in enumerate(sequences):
			if seq.uuid in rejected_uuids:
				continue
			seq.prompt_length = lengths[i]
			valid_uuids.append(seq.uuid)

		# Reserve the complete fixed model-context page chain and
		# copy the prompt into the same store that will receive decode tokens.
		# Every rank executes this deterministically so local metadata and the
		# node-shared storage have identical slot assignments.
		if self._traj.trajectory_book is None:
			raise RuntimeError("unified trajectory pool is not initialized")
		valid = [
			(seq, prompt_ids[i])
			for i, seq in enumerate(sequences)
			if seq.uuid not in rejected_uuids
		]
		sequence_ids = [seq.uuid for seq, _ in valid]
		reserved_tokens = [self.model_context_length] * len(valid)
		reserved_slots = self._traj.trajectory_book.bind_batch(sequence_ids, reserved_tokens)
		try:
			for (seq, token_ids), slot in zip(valid, reserved_slots):
				actual_prompt_len = int(token_ids.numel())
				self._traj.trajectory_book.write_prompt(slot, token_ids)
				seq._buffer_slot = slot
				# Token storage is now accessed through QueryBook; these fields are
				# intentionally empty so no stale dense view can escape the book.
				seq.input_ids = None
				seq.decoded_tokens = None
				seq.prompt_length = actual_prompt_len
				seq.original_prompt_length = actual_prompt_len
				seq.current_context_length = actual_prompt_len
				# Keep the API generation limit and all SequenceEntry invariants in
				# sync when prompt + requested completion exceeds model context. The
				# physical QueryBook reservation remains a full model-context chain.
				seq.clamp_decode_to_context(actual_prompt_len, self.model_context_length)
		except Exception:
			for slot in reserved_slots:
				if slot in self._traj.trajectory_book._records:
					self._traj.trajectory_book.release(slot)
			raise
		self._publish_trajectory_pool_capacity()

		# packing_v2: order the admission by prompt prefix before ownership,
		# so a prefix family is not split across two prefill waves.
		if self._prefill_prefix_pool_enabled():
			self._stamp_prefix_prefill_order(valid_uuids)

		# Ownership assignment runs identically on every rank; the book write
		# above already materialized each prompt node-shared, so the
		# sharing-aware assignment can read tokens through the book.
		if hasattr(self, "_assign_admitted_sequences_to_ranks"):
			self._assign_admitted_sequences_to_ranks(valid_uuids)
		if hasattr(self, "_assign_decode_dp_groups"):
			self._assign_decode_dp_groups(valid_uuids)

	def _assign_admitted_sequences_to_ranks(self, uuids: List[str]) -> None:
		"""Assign newly admitted sequences to ranks.

		Default (BATCHGEN_L2_BALANCE=1, default): least-sum(L²) argmin with
		FFD ordering (longest first). Attention is O(L²) so balancing on L²
		minimizes the wall-clock spread between fastest and slowest rank
		during prefill — without this, all LongBench long-context seqs land
		on rank 14-15 under round-robin and stall the per-iteration barrier.

		Fallback (BATCHGEN_L2_BALANCE=0): least-count argmin (legacy).
		"""
		if self._prefill_prefix_pool_enabled():
			self._assign_ranks_for_prefix_sharing(uuids)
			return
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
		"""Delegate: which ranks bind a sequence into the LOCAL table.

		Rule and rationale live on `RankContext.owns_sequence`.
		"""
		return self.rank_ctx.owns_sequence(seq)

	def _owns_host_kv(self, seq) -> bool:
		"""Delegate: which SINGLE rank drives the shared host-KV region.

		Rule and rationale live on `RankContext.owns_host_kv`.
		"""
		return self.rank_ctx.owns_host_kv(seq)

	def _mirror_group_host_kv(
		self,
		uuids: List[str],
		prefix_states: dict,
		*,
		trajectory: bool = False,
	) -> None:
		"""Copy the serve-group leader's host-KV bookkeeping onto its followers.

		G>1 replicates a sequence onto all G ranks of its group, but only the
		leader (``_owns_host_kv``) registers / allocates it in the per-node
		SHARED host region, so the other G-1 ranks come out of prefill config
		with an empty worker-view page table and no prefix state entry
		entry for sequences they must still read. Both are rank-local views of
		page ids the leader has already committed, so one object all-gather
		reproduces them exactly: every follower installs ITS OWN leader's entry.

		The payload is plain tuples and dicts — the coordinator attachment and
		its lookup result belong to the leader alone — so a mirrored state
		carries ``lookup_result=None`` plus the leader's page ids per KV group,
		which is how ``shared_page_ids_for_group`` resolves a planned chain. A
		prefill prefix-pool wave adds its segments' pseudo sequence ids and Host
		pages, so every rank's segment row offloads land on the leader's pages.

		No-op for G==1, where the host-KV owner is the only rank holding the
		sequence. For G>1 every rank reaches the collective: the admission is
		identical on all ranks while ``uuids`` is not, and the gather sits
		outside the leader branch.
		"""
		G = self._decode_attn_tp_size()
		if G <= 1:
			return
		# The mirrored page ids index the leader's per-node host region, so a
		# serve group must not straddle a node boundary.
		if self.local_world_size % G:
			raise RuntimeError(
				f"attn_tp_size={G} must divide local_world_size="
				f"{self.local_world_size}: a serve group spanning nodes would "
				f"read another node's host-KV page ids"
			)
		from batchgen.decode_dp_group import host_kv_owner_rank
		from batchgen.prefix_reuse.prefill import PrefixCacheSequenceState

		host_views = self._host_views_by_group()
		wave = getattr(self, "_prefill_pool_wave", None)
		if trajectory:
			# Pendings created during a pass's DECODE phase (a commit whose
			# D2H retired after the post-prefill finalize, e.g. the last
			# overflow wave's sequence) would otherwise install only at the
			# NEXT pass's post-prefill finalize -- one publish window too
			# late for this gather. Ordering is the pass boundary itself: a
			# rank reaches this config only after returning from the previous
			# pass, so every rank's shard writes for those pendings are done,
			# and no pending from THIS pass can exist yet (its prefill has
			# not run).
			self._finalize_trajectory_frontiers()
		payload = None
		if uuids and all(
			self._owns_host_kv(self.global_batch.get_sequence(uuid))
			for uuid in uuids
		):
			segments = None
			if wave is not None:
				from batchgen.prefix_reuse.wave_runtime import segment_mirror_payload

				segments = segment_mirror_payload(wave)
			global_ids = [
				int(self.global_batch.get_sequence(uuid).global_idx)
				for uuid in uuids
			]
			prefix = {}
			for global_id in global_ids:
				state = prefix_states.get(global_id)
				if state is None:
					continue
				prefix[global_id] = (
					int(state.attached_tokens),
					int(state.compute_cached_tokens),
					{
						group_id: tuple(
							int(page)
							for page in state.shared_page_ids_for_group(group_id)
						)
						for group_id in host_views
					},
					# The plan's restore decision travels with the start it
					# set, so a follower's state carries the leader's.
					int(state.trajectory_restore_boundary),
				)
			payload = {
				"global_ids": global_ids,
				"page_tables": {
					group_id: [
						[int(page) for page in pages]
						for pages in view.build_page_table(global_ids)
					]
					for group_id, view in host_views.items()
				},
				"prefix": prefix,
				# A pooled segment's Host pages are the leader's too, but every
				# rank of the group offloads the segment rows it computed, so the
				# followers need the pages those rows must land on.
				"segments": segments,
				# M3 Slice 1c rides THIS gather rather than adding one: the
				# G-wide snapshot arena's page ids (so every rank can address
				# its own shard window at the commit boundary) and the leader's
				# frontier hits (so the group restores on one decision).
				"frontier_arenas": (
					self._reserve_trajectory_arenas(uuids)
					if trajectory
					else {}
				),
				"frontier_restores": (
					self._trajectory_restore_payload() if trajectory else {}
				),
				# M3 Slice 3a: this group's committed sessions. The gather
				# below is over the DEFAULT process group -- the whole world,
				# not the serve group -- so one leader's summary reaches every
				# rank of every group, which is what makes the admission pin
				# world-consistent.
				"frontier_sessions": (
					self._trajectory_session_summary() if trajectory else {}
				),
			}
		if payload is None and trajectory:
			# A leader with no prefill sequence this pass still speaks for its
			# group's committed sessions: the summary is registry state, not
			# pass state, and every rank's admission pin folds it from this
			# same gather. Gating it on having host KV to mirror starves the
			# world map whenever a flag-on pass lands entirely on the other
			# group(s) -- the other groups' sessions then surface only one
			# pass later, after that pass's pins and pool plan already ran.
			summary = self._trajectory_session_summary()
			if summary:
				payload = {"frontier_sessions": summary}

		gathered = [None] * self.world_size
		dist.all_gather_object(gathered, payload)

		if trajectory:
			# EVERY rank folds EVERY leader's summary -- leaders included, so
			# this runs before the leader's early return below. Group
			# assignment is replicated across ranks and must agree, so its
			# input may not be a rank-local subset.
			self._fold_trajectory_session_groups(gathered)

		leader = host_kv_owner_rank(self.rank // G, G)
		if leader == self.rank:
			return
		entry = gathered[leader]
		if entry is None or "global_ids" not in entry:
			# None, or a summary-only payload from a leader that served no
			# prefill this pass: either way the leader mirrored no host KV
			# (the fold above already consumed its session summary).
			if uuids:
				raise RuntimeError(
					f"Rank {self.rank}: serve-group leader {leader} mirrored no "
					f"host KV for the {len(uuids)} sequences this rank serves"
				)
			return
		for group_id, view in host_views.items():
			for global_id, pages in zip(
				entry["global_ids"], entry["page_tables"][group_id]
			):
				view.set_sequence_pages(global_id, pages)
		segments = entry.get("segments")
		if (segments is None) != (wave is None):
			raise RuntimeError(
				f"Rank {self.rank}: serve-group leader {leader} planned "
				f"{'a' if segments is not None else 'no'} prefix pool wave while "
				f"this rank planned {'one' if wave is not None else 'none'}"
			)
		if segments is not None:
			from batchgen.prefix_reuse.wave_runtime import install_mirrored_segments

			install_mirrored_segments(wave, segments, host_views)
		for global_id, (attached, cached, shared, restore) in entry["prefix"].items():
			state = PrefixCacheSequenceState(
				lookup_result=None,
				attached_tokens=attached,
				compute_cached_tokens=cached,
				planned_shared_page_ids={
					int(group_id): tuple(pages)
					for group_id, pages in shared.items()
				},
				trajectory_restore_boundary=int(restore),
			)
			prefix_states[global_id] = state
			self._prefix.prefix_sequence_states[global_id] = state
		if trajectory:
			# M3 Slice 1c, follower side. Both halves come from the leader: the
			# arena this rank writes its own shard into, and the hit it must
			# restore. Taking the leader's decision is what keeps the group's
			# prefill in lockstep -- G independent lookups could disagree, and
			# a rank that declined while its peers restored would run the same
			# suffix against a zeroed head shard.
			self._traj.trajectory_arena_payload = dict(
				entry.get("frontier_arenas") or {}
			)
			self._traj.trajectory_mirrored_restores = dict(
				entry.get("frontier_restores") or {}
			)
			self._adopt_mirrored_trajectory_arenas(
				self._traj.trajectory_arena_payload
			)
			if self._traj.trajectory_arena_payload or self._traj.trajectory_mirrored_restores:
				logging.info(
					"[TRAJECTORY_FRONTIER] Rank %s shard=%s adopted %s arena(s) "
					"and %s mirrored restore(s) from serve-group leader %s",
					self.rank,
					self._trajectory_group_geometry()[1],
					len(self._traj.trajectory_arena_payload),
					len(self._traj.trajectory_mirrored_restores),
					leader,
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

		``packing_v2`` batches served through the in-wave prefix pool with more
		than one group instead pack by predicted prefill seconds, so a prefix
		family shares one group's pool (``_pack_decode_dp_groups_for_sharing``).

		A committed trajectory contributes resident leading tokens on its owning
		group to that same objective, even without ``packing_v2``. The packer
		may place a continuation elsewhere when recomputing the prefix lowers
		the busiest group's cost. Evicted re-entries remain hard-pinned.
		"""
		G = self._decode_attn_tp_size()
		if G <= 1 or not uuids:
			return
		from batchgen.decode_dp_group import (
			assign_decode_dp_groups,
			num_decode_dp_groups,
		)
		num_dp = num_decode_dp_groups(self.world_size, G)
		# Free and byte-identical whenever no session map exists, which is
		# every run that never turned the trajectory flag on.
		residency = self._trajectory_session_residency(uuids, num_groups=num_dp)
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
		if (
			num_dp > 1
			and (
				residency
				or (self._prefill_prefix_pool_enabled() and self._prefix_rank_packing_mode(seqs))
			)
		):
			if residency:
				seqs.sort(key=lambda seq: seq.uuid)
			groups = self._pack_decode_dp_groups_for_sharing(seqs, num_dp, residency)
		else:
			# An evicted re-entry returns to the group it decoded on: its
			# decoded suffix lives only in that group's node's trajectory book.
			# It loads that group like any grouped sequence; the rest balance.
			groups = [seq.evicted_from_dp_group for seq in seqs]
			fresh = [i for i, g in enumerate(groups) if g is None]
			for i, g in enumerate(groups):
				if g is not None:
					prior[g] += float(lengths[i]) * float(lengths[i])
			placed = assign_decode_dp_groups(
				[lengths[i] for i in fresh], num_dp, prior_load=prior
			)
			for i, g in zip(fresh, placed):
				groups[i] = g
		for seq, g in zip(seqs, groups):
			seq.decode_dp_group = g
			# Grouped again: the eviction record has served its purpose, and
			# a later eviction records the then-current group afresh.
			seq.evicted_from_dp_group = None
		for uuid in sorted(residency):
			owner, boundary = residency[uuid]
			seq = self.global_batch.get_sequence(uuid)
			honored = seq.decode_dp_group == owner
			self._traj.trajectory_session_groups.record_pin(placed_on_owner=honored)
			logging.info(
				"[TRAJECTORY_FRONTIER] Rank %s sequence %s owner_group=%s "
				"placed_group=%s boundary=%s honored_on_owner=%s placed_offgroup=%s",
				self.rank, seq.uuid, owner, seq.decode_dp_group, boundary,
				honored, not honored,
			)

	def _pack_decode_dp_groups_for_sharing(self, seqs, num_dp: int, residency=None) -> List[int]:
		"""Place ungrouped sequences on serve groups by predicted prefill seconds.

		A group prefills its members through its own in-wave prefix pool, so a
		prefix family split across groups recomputes the shared prefix once per
		group. ``pack_groups_for_sharing`` keeps a family on one group unless
		splitting it lowers the busiest group's predicted seconds.

		Load already committed to a group is pinned there: every grouped
		sequence that still owes a prefill, scored by its length alone (see
		``prompt_score_from_length``). Sequences already decoding owe no prefill
		seconds and are not counted, unlike the L^2 prior, which charges a group
		for its whole standing decode batch.

		Prompts are read from the node-shared trajectory book, which admission
		writes on every rank before grouping, and every input here is replicated,
		so every rank computes the same map. A re-entry is not packed and its
		prompt is never read: its decoded suffix lives only in the trajectory
		book of the node it decoded on, so it returns to the group recorded at
		eviction (``evicted_from_dp_group``) and pins its length-only score
		there. A re-entry without that record cannot be placed and is refused.
		"""
		from batchgen.prefix_reuse.rank_packing import (
			pack_groups_for_sharing,
			prompt_score_from_length,
		)

		profile = self._cached_prefix_rank_cost_profile(
			self._prefix_rank_cost_profile_disabled(seqs)
		)
		page_tokens = int(self.prefix_cache_runtime_config.group_specs[0].raw_page_tokens)
		chunk_tokens = int(
			self.engine_config.Module_Batching_Config.prefill_micro_batch_token_cap
		)

		def pinned_load(seq):
			# Length-only: a re-entry's tokens are readable on one node only.
			return prompt_score_from_length(
				int(seq.prompt_length),
				block_tokens=page_tokens,
				chunk_tokens=chunk_tokens,
				profile=profile,
			)

		groups = []
		queued = []
		fixed = []
		for seq in seqs:
			group = seq.evicted_from_dp_group
			if group is None and (
				seq.status == SequenceStatus.EVICTED
				or getattr(seq, "evicted_token_ids", None) is not None
			):
				raise RuntimeError(
					f"re-entering sequence {seq.uuid} has no recorded serve group "
					"to return to; its decoded tokens live only on the node it "
					"decoded on, so it cannot be placed identically on every rank"
				)
			if group is not None:
				fixed.append((int(group), pinned_load(seq)))
			else:
				queued.append(seq)
			groups.append(group)
		owes_prefill = (
			SequenceStatus.QUEUEING, SequenceStatus.IN_PREFILL, SequenceStatus.EVICTED
		)
		for seq in self.global_batch:
			if seq.decode_dp_group is None or seq.status not in owes_prefill:
				continue
			fixed.append((int(seq.decode_dp_group), pinned_load(seq)))
		prompts = [
			self._trajectory_tokens(seq, length=seq.prompt_length).tolist()
			for seq in queued
		]
		packed = iter(pack_groups_for_sharing(
			prompts,
			num_groups=num_dp,
			block_tokens=page_tokens,
			chunk_tokens=chunk_tokens,
			profile=profile,
			fixed=fixed,
			resident=[
				{residency[seq.uuid][0]: residency[seq.uuid][1]}
				if seq.uuid in residency else {}
				for seq in queued
			] if residency else (),
		))
		return [next(packed) if group is None else group for group in groups]

	def _check_decode_dp_group_plan_identical(self, uuids: List[str]) -> None:
		"""All ranks must have stamped the SAME serve-group map. Collective.

		``rank_in_decode_group`` resolves ownership from each rank's own copy of
		the map, so the map is not a hint: it decides which ranks bind a
		sequence, run its prefill and allocate its head-sharded KDA state. A
		rank that derived a different group for one sequence binds a row whose
		state another group created, and the mismatch first appears far away —
		``KeyError: Sequence <global_idx> has no KDA state item`` at the decode
		entry, on a rank that never prefilled it.

		Both stampers are deterministic functions of replicated inputs: the
		admission list (broadcast from rank 0), prompt lengths, and the prompts
		themselves read through the node-shared trajectory book, which every
		rank wrote before grouping. So agreement is derivable, not hoped for,
		and a divergence means one of those inputs was not in fact replicated —
		which is a bug to name here, at the stamp, rather than to debug three
		phases later. The rank packer already carries this check
		(``_assign_ranks_for_prefix_sharing``); the group packer did not.

		Cheap: one int64 all_gather of two values per wave, plus a second one
		only on the failing path to name the first diverging sequence.
		"""
		if self._decode_attn_tp_size() <= 1 or self.world_size <= 1:
			return
		from batchgen.decode_dp_group import (
			decode_group_plan_digest,
			first_group_disagreement,
		)

		pairs = []
		for uuid in uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				continue
			pairs.append((int(seq.global_idx), seq.decode_dp_group))
		pairs.sort(key=lambda item: item[0])
		check = torch.tensor(
			[decode_group_plan_digest(pairs), len(pairs)],
			dtype=torch.int64, device=self.torch_device,
		)
		gathered = [torch.zeros_like(check) for _ in range(self.world_size)]
		dist.all_gather(gathered, check)
		# This rank's own digest joins the comparison explicitly rather than
		# through ``gathered[self.rank]``: the check must hold on what this
		# rank actually stamped, not on the slot the collective happened to
		# fill for it.
		digests = {int(item[0].item()) for item in gathered}
		digests.add(int(check[0].item()))
		if len(digests) == 1:
			return
		counts = {int(item[1].item()) for item in gathered}
		counts.add(len(pairs))
		named = None
		if len(counts) == 1:
			# Same sequences everywhere, so the group vectors are comparable
			# column by column. Every rank reaches this second collective: they
			# all saw the same differing digests.
			groups = torch.tensor(
				[-1 if group is None else int(group) for _, group in pairs],
				dtype=torch.int64, device=self.torch_device,
			)
			views = [torch.zeros_like(groups) for _ in range(self.world_size)]
			dist.all_gather(views, groups)
			named = first_group_disagreement(
				[global_idx for global_idx, _ in pairs],
				# This rank's own vector is compared explicitly, for the same
				# reason its digest is.
				[groups.tolist()] + [view.tolist() for view in views],
			)
		raise RuntimeError(
			"decode serve-group map diverged across ranks: rank "
			f"{self.rank} digest {int(check[0].item())} over {len(pairs)} "
			f"sequences; per-rank sequence counts {sorted(counts)}"
			+ (
				f"; first disagreement global_idx={named[0]} "
				f"groups_by_rank={named[1]}" if named is not None else ""
			)
		)

	def _check_local_binding_matches_group(
		self, uuids: List[str], where: str
	) -> None:
		"""Delegate: a local row exists iff this rank's group owns the sequence.

		Rule and rationale live on `SequenceTable.check_binding_matches_group`.
		"""
		self.seq_table.check_binding_matches_group(
			uuids,
			where,
			rank_ctx=self.rank_ctx,
			get_sequence=self.global_batch.get_sequence,
		)

	def _bind_local_sequence_to_query_book(
		self,
		uuid: str,
		local_idx: Optional[int] = None,
	) -> int:
		"""Bind a sequence UUID to a local slot and refresh its query_book entry."""
		seq = self.global_batch.get_sequence(uuid)
		if self.query_book is None:
			self.query_book = {}
		return bind_query_book_slot(
			uuid,
			seq,
			query_book=self.query_book,
			table=self.seq_table,
			local_idx=local_idx,
		)

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

	def _trajectory_slot(self, seq: SequenceEntry) -> int:
		if self._traj.trajectory_book is None:
			raise RuntimeError(f"sequence {seq.uuid} has no active trajectory slot")
		try:
			# Slot numbers are local to a node's allocator, so the same UUID holds
			# different slots on different nodes; never trust the process-global
			# ``_buffer_slot`` field (last set on another node) as the lookup key.
			return self._traj.slot_for(seq.uuid)
		except KeyError as exc:
			raise RuntimeError(
				f"sequence {seq.uuid} has no active trajectory reservation"
			) from exc

	def _trajectory_tokens(
		self,
		seq: SequenceEntry,
		*,
		start: int = 0,
		length: Optional[int] = None,
		dtype: torch.dtype = torch.int32,
	) -> torch.Tensor:
		"""Materialize a trajectory span; no persistent storage view escapes."""
		if self._traj.trajectory_book is None:
			raise RuntimeError("unified trajectory pool is not initialized")
		if dtype not in (torch.int32, torch.int64):
			raise ValueError(f"unsupported trajectory dtype: {dtype}")
		valid = self._traj.metadata(self._trajectory_slot(seq)).token_length
		if length is None:
			length = valid - start
		out = torch.empty(int(length), dtype=torch.int32)
		self._traj.copy_span_to(
			self._trajectory_slot(seq), out, start=int(start), length=int(length)
		)
		return out if dtype == torch.int32 else out.to(dtype=torch.int64)

	def _trajectory_decoded_tokens(self, seq: SequenceEntry) -> torch.Tensor:
		"""Materialize generated tokens for output and repetition checks."""
		return self._traj.generated_tokens(self._trajectory_slot(seq))

	def _append_trajectory_token(self, seq: SequenceEntry, token_id: int) -> int:
		if self._traj.trajectory_book is None:
			raise RuntimeError("unified trajectory pool is not initialized")
		return self._traj.append_token(self._trajectory_slot(seq), int(token_id))

	def _append_trajectory_tokens(
		self, sequences: Sequence[SequenceEntry], token_ids: Sequence[int]
	) -> None:
		"""Append one host-side token per sequence through a single call.

		The caller passes exactly the rows that must receive a token: the
		decode loop drops retired rows first (see [COMPLETED_ROW_SKIP]), so
		this helper never decides which rows are live.
		"""
		if self._traj.trajectory_book is None:
			raise RuntimeError("unified trajectory pool is not initialized")
		if len(sequences) != len(token_ids):
			raise ValueError(
				f"trajectory append batch mismatch: {len(sequences)} != {len(token_ids)}"
			)
		if not sequences:
			return
		slots = [self._trajectory_slot(seq) for seq in sequences]
		self._traj.append_tokens(slots, token_ids)

	def _append_trajectory_batch(
		self, batch: Sequence[int], token_ids: torch.Tensor
	) -> None:
		if self._traj.trajectory_book is None:
			raise RuntimeError("unified trajectory pool is not initialized")
		if token_ids.numel() != len(batch):
			raise ValueError(
				f"trajectory append batch mismatch: {len(batch)} != {token_ids.numel()}"
			)
		sequences = []
		for local_idx in batch:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				raise KeyError(f"missing sequence for local_idx={local_idx}")
			sequences.append(seq)
		self._append_trajectory_tokens(
			sequences, token_ids.reshape(-1).to(device="cpu").tolist()
		)

	def _sync_trajectory_metadata(
		self, seq: SequenceEntry, state: Dict[str, object]
	) -> None:
		"""Refresh a local QueryBook mirror from the owning rank's lengths."""
		if not self._traj.has_sequence(seq.uuid):
			return
		token_length = state.get("trajectory_token_length")
		if token_length is None:
			return
		self._traj.synchronize_metadata(
			self._trajectory_slot(seq),
			prompt_length=int(state["prompt_length"]),
			token_length=int(token_length),
			decoded_length=int(
				state.get("trajectory_decoded_length", state["decoded_length"])
			),
		)

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
		fallback_text = ""
		trajectory_slot = None
		if self._traj.trajectory_book is not None and self._traj.trajectory_book.has_sequence(uuid):
			trajectory_slot = self._traj.trajectory_book.slot_for(uuid)
		if (
			self.rank == 0
			and gathered_text is None
			and seq.decoded_length > 0
			and trajectory_slot is not None
		):
			try:
				fallback_text = self._decode_tokens_to_string(
					self._trajectory_decoded_tokens(seq)
				)
			except Exception:
				fallback_text = ""

		# Free the complete trajectory reservation on every rank. This is the
		# only token-storage release point; multi-turn re-entry keeps the slot.
		if trajectory_slot is not None:
			self._traj.trajectory_book.release(trajectory_slot)
		self._publish_trajectory_pool_capacity()
		seq._buffer_slot = -1

		# Free local index mapping.
		# DIAGNOSTIC: log the pop on the owning rank so we can correlate
		# stray pops with downstream "Missing UUID" errors.
		local_idx = release_query_book_slot(
			uuid,
			query_book=self.query_book,
			table=self.seq_table,
		)
		if local_idx is not None:
			if seq.assigned_rank == self.rank:
				logging.debug(
					f"Rank {self.rank}: [LOCALMAP-POP] _report_completion popped "
					f"{uuid[:8]} (local_idx={local_idx}, status={seq.status.name})"
				)

		self._maybe_emit_prefix_drain_metric()

		# Cached/computed refer to the sequence's last prefill: computed is the
		# prompt tokens that prefill ran, cached the ones it took from cache.
		cached_tokens = int(seq.prefix_cached_tokens)
		computed_tokens = max(int(seq.prompt_length) - cached_tokens, 0)
		if seq.session_id is not None and seq.assigned_rank == self.rank:
			logging.info(
				"[SESSION] complete request=%s traj=%s session=%s parent=%s "
				"prompt_tokens=%d cached_tokens=%d computed_tokens=%d "
				"decoded_tokens=%d rank=%d",
				seq.uuid, seq.traj_id, seq.session_id, seq.parent_session_id,
				seq.original_prompt_length, cached_tokens, computed_tokens,
				seq.decoded_length, self.rank,
			)

		# Only rank 0 sends to response queue
		if self.rank != 0 or self._response_queue is None:
			return

		# Use gathered text if provided, otherwise read from local buffer
		text = gathered_text if gathered_text is not None else fallback_text
		self._response_queue.put({
			"type": "completion",
			"request_id": uuid,
			"batch_id": getattr(seq, "batch_id", None),
			"global_idx": seq.global_idx,
			"text": text,
			# Re-entry turns prior completion tokens into the internal prompt so
			# the KV state can be recomputed.  OpenAI usage must still report the
			# caller's original prompt, not that internal reconstructed length.
			"prompt_length": seq.original_prompt_length,
			"decoded_length": seq.decoded_length,
			"finish_reason": self._get_finish_reason(seq),
			"cached_tokens": cached_tokens,
			"computed_tokens": computed_tokens,
			"admitted_at": seq.admitted_at,
			"completed_at": time.time(),
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

		# Each rank provides tokens for its locally-owned completed sequences.
		# The prompt tokens their last prefill took from the prefix cache come
		# only from the rank that owns the sequence's host KV
		# (_owns_host_kv): under G>1 every rank of the serve group holds the
		# sequence, but only the leader drives its prefix state, so a
		# follower's count is not authoritative. Others send None.
		my_tokens = {}
		for uuid in completed_uuids:
			if uuid in self._uuid_to_local_map:
				local_idx = self._uuid_to_local_map[uuid]
				seq = self.global_batch.get_sequence(uuid)
				if seq is not None and local_idx in self.query_book:
					decoded_tokens = self._trajectory_decoded_tokens(seq)
					try:
						text = self._decode_tokens_to_string(decoded_tokens)
					except Exception:
						text = ""
					cached = None
					if self._owns_host_kv(seq):
						state = self._prefix.prefix_sequence_states.get(seq.global_idx)
						cached = 0 if state is None else int(state.compute_cached_tokens)
					my_tokens[uuid] = (text, cached)

		# All ranks participate in gather
		all_tokens = [None] * self.world_size
		dist.all_gather_object(all_tokens, my_tokens)

		# Merge. Under G>1 a uuid arrives from every rank of its serve group;
		# exactly one of them (the host-KV owner) carries its cached count.
		# Every rank merges the same gathered data, so a violation raises on
		# all ranks alike rather than desynchronising them.
		merged = {}
		owner_cached = {}
		for src_rank, rank_tokens in enumerate(all_tokens):
			if rank_tokens:
				for uuid, (text, cached) in rank_tokens.items():
					merged[uuid] = text
					if cached is None:
						continue
					if uuid in owner_cached and owner_cached[uuid][1] != cached:
						raise RuntimeError(
							f"cached_tokens for {uuid} disagree between host-KV "
							f"owners: rank {owner_cached[uuid][0]} reports "
							f"{owner_cached[uuid][1]}, rank {src_rank} reports {cached}"
						)
					owner_cached[uuid] = (src_rank, cached)
		missing = sorted(set(merged) - set(owner_cached))
		if missing:
			raise RuntimeError(
				f"no host-KV owner reported cached_tokens for completed "
				f"sequence(s) {missing[:8]} ({len(missing)} total)"
			)
		for uuid, (_, cached) in owner_cached.items():
			seq = self.global_batch.get_sequence(uuid)
			if seq is not None:
				seq.prefix_cached_tokens = cached
		return merged

	# ============ End Request Pool Methods ============

	def _bump_decode_row_epoch(self) -> int:
		"""Stamp a change to the ordered decode row set.

		Call sites are the two places `batch` is rebound (the page boundary
		planner and the async-load finalizer) and the host-rebalance remap of
		`_local_to_uuid_map`, which can change which sequence a row resolves to
		without changing `batch` at all. Anything that holds state keyed to the
		row order -- today, the resident sampling-parameter tensors -- is
		invalidated by this.
		"""
		self._decode_row_epoch += 1
		return self._decode_row_epoch

	def _build_sampling_tensors(self, batch_sequences: list) -> tuple:
		"""Build [B] sampling param tensors for the active decode batch.

		Reference implementation. The steady-state decode step path does NOT
		call this -- it reads the resident tensors
		(`batchgen.worker.sampling_residency`) instead, because this function's
		`torch.tensor(..., device=cuda)` calls are pageable-source H2D copies
		that block the host until the whole-model graph replay already queued on
		the stream retires. This stays as the reference the residency is tested
		against, and as the path every non-decode caller keeps using.

		Returns:
			(temps, top_ps, top_ks) tensors on the model's device, or (None, None, None)
			if no sequence in the batch carries sampling params.
		"""
		if not batch_sequences:
			return None, None, None

		has_sequence_params = any(
			getattr(seq, "sampling_params", None) is not None
			for seq in batch_sequences
		)
		if not has_sequence_params:
			return None, None, None

		device = next(self.model.parameters()).device
		params = [getattr(seq, "sampling_params", None) or {} for seq in batch_sequences]

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
			if ((seq_params or {}).get('temperature', 0.0) or 0.0) > 0:
				return False
		return True

	def _select_tokens(
		self,
		logits: torch.Tensor,
		batch_sequences: Optional[list] = None,
		residency=None,
		batch_rows: Optional[list] = None,
	) -> torch.Tensor:
		"""
		Select next tokens from logits using greedy or sampling strategy.
		Sampling parameters are per-sequence; a batch without any falls
		through to greedy argmax.

		Args:
			logits: [batch_size, vocab_size] logits from model
			batch_sequences: active decode-batch sequences, or None
			residency: optional `DecodeSamplingResidency` for the steady-state
				decode step path. When given, the per-request sampling tensors
				are read from device-resident buffers instead of being rebuilt
				and re-uploaded every step, and `sample_tokens` is handed the
				host-side branch decisions so it stops reducing values the host
				itself wrote. Every other call site passes nothing and keeps the
				rebuild path unchanged.
			batch_rows: the local decode-index list the rows came from, required
				with `residency` -- it is both the row-order identity the cache
				keys on and what the fail-loud shape/row checks name.

		Returns:
			[batch_size, 1] selected token indices
		"""
		from batchgen.sampling import sample_tokens

		if residency is not None:
			if batch_rows is None:
				raise ValueError(
					"_select_tokens: batch_rows is required with residency "
					"(it is the row-order identity the resident tensors key on)"
				)
			return self._select_tokens_resident(
				logits, batch_sequences or [], residency, batch_rows, sample_tokens
			)

		# Per-sequence sampling path: sampling params ride on the SequenceEntry
		# objects of the active decode batch.
		if (
			batch_sequences is not None
			and any(getattr(seq, "sampling_params", None) is not None for seq in batch_sequences)
		):
			active_sequences = batch_sequences
			# All-greedy fast path: if every sequence's effective temperature is
			# <= 0, argmax directly (fp32, bit-identical to sample_tokens' greedy
			# branch) instead of entering sample_tokens. This skips its two per-
			# token .any() device->host syncs (sampling.py:84,88) and the three
			# per-token H2D sampling-tensor builds — pure pipeline stalls for a
			# temp-0 batch. Host-side check (no device sync), recomputed each step
			# so no membership-invalidation hazard.
			if self._decode_batch_all_greedy(active_sequences):
				# argmax directly on the lm_head dtype: an fp32 upcast preserves
				# bf16 ordering exactly, and the .float() copy was a fresh
				# [bs, vocab] fp32 allocation on every decode step — large
				# enough to destabilize graph-enabled boots at high
				# gpu-memory-frac.
				return logits.argmax(dim=-1, keepdim=True)
			temps, top_ps, top_ks = self._build_sampling_tensors(active_sequences)
			if not getattr(self, '_logged_sampling', False) and self.rank == 0:
				logging.info(f"Using PER-SEQUENCE sampling for {logits.shape[0]} sequences")
				self._logged_sampling = True
			if temps is not None:
				return sample_tokens(logits, temperature=temps, top_p=top_ps, top_k=top_ks)

		# No sequence in the batch carries sampling params — greedy decoding.
		return torch.argmax(logits, dim=-1, keepdim=True)

	def _select_tokens_resident(
		self,
		logits: torch.Tensor,
		batch_sequences: list,
		residency,
		batch_rows: list,
		sample_tokens,
	) -> torch.Tensor:
		"""`_select_tokens` over device-resident sampling-parameter tensors.

		Reproduces the three branches of the rebuild path exactly:

		* no row carries `sampling_params` (`None`, not `{}`) -> the plain
		  `torch.argmax` at the end of `_select_tokens`;
		* every row is effectively greedy -> `logits.argmax` on the lm_head
		  dtype, with no parameter tensor built and `sample_tokens` not entered;
		* otherwise -> `sample_tokens` on the resident tensors.

		The two argmax branches are deliberately distinct calls, as before: they
		are reached under different conditions and must both stay reachable.
		"""
		plan = residency.plan(
			batch_sequences,
			logits=logits,
			membership_epoch=self.global_batch.composition_epoch,
			row_epoch=self._decode_row_epoch,
			batch_rows=batch_rows,
			verify=BATCHGEN_DECODE_ASSERT,
		)
		if not plan.has_params:
			return torch.argmax(logits, dim=-1, keepdim=True)
		if plan.all_greedy:
			# argmax directly on the lm_head dtype: an fp32 upcast preserves
			# bf16 ordering exactly, and the .float() copy was a fresh
			# [bs, vocab] fp32 allocation on every decode step.
			return logits.argmax(dim=-1, keepdim=True)
		if not getattr(self, '_logged_sampling', False) and self.rank == 0:
			logging.info(f"Using PER-SEQUENCE sampling for {logits.shape[0]} sequences")
			self._logged_sampling = True
		return sample_tokens(
			logits,
			temperature=plan.temps,
			top_p=plan.top_ps,
			top_k=plan.top_ks,
			host_decisions=plan.decisions,
			bad_row_counter=residency.bad_row_counter,
		)

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

	# This worker's slot in the server's shared startup-phase record, installed
	# by the supervisor.  The null default keeps every phase-reporting step
	# below working in a worker no server is watching (tests, the legacy
	# launcher) while still rejecting an undeclared phase name.
	_startup_phase_reporter = NULL_STARTUP_PHASE_REPORTER

	def set_startup_phase_reporter(self, reporter) -> None:
		"""Install the handle this worker reports its startup phases through.

		Like the cohort-shutdown probe below, the shape behind it -- a
		lock-free shared record the server polls -- stays in the supervisor;
		this class only ever says which phase it has entered.
		"""
		self._startup_phase_reporter = reporter

	def _startup_phase(self, name: str):
		"""Report one named startup phase for the duration of the body.

		``name`` must be one of ``batchgen.worker.startup_phases.STARTUP_PHASES``;
		the reporter refuses anything else.
		"""
		return self._startup_phase_reporter.phase(name)

	def set_cohort_shutdown_probe(self, probe) -> None:
		"""Install the zero-argument cohort-teardown probe.

		``probe()`` answers one question -- has this cohort's teardown begun --
		and the supervisor owns what backs it (a plain load from a lock-free
		shared byte the manager stores to).  Keeping it a callable is what keeps
		the flag's shape, and the no-handshake contract that goes with it, out
		of this class.
		"""
		self._cohort_shutdown_probe = probe

	def _cohort_shutdown_requested(self) -> bool:
		"""Has this cohort's teardown begun?

		Called once per decode step on the hot path, so it stays a single
		attribute load and, when armed, one shared-byte load behind the probe:
		no lock, no allocation, no logging.  Unarmed (no supervisor, or a code
		path that never serves a pool) it is a ``None`` test and nothing else.
		"""
		probe = self._cohort_shutdown_probe
		return probe is not None and probe()

	def _wait_decode_tokens(self, event) -> str:
		"""Wait for a decode step's sampled-token readback, pollably.

		This replaces the per-step ``event.synchronize()`` that is the measured
		wedge point.  A rank whose peer died mid-decode blocks there forever: it
		runs no Python, so the decode-step teardown boundary a few lines below
		it is unreachable, and aborting the dead communicator does not reliably
		release it.  Polling is the only way the main thread can see the
		teardown flag at all.

		Unarmed -- ``_cohort_shutdown_probe is None``, i.e. every path that
		never serves a worker pool -- this IS the legacy call and nothing else.
		There is no poll, no clock read and no behavior to regress.

		Armed, the wait has two phases.  First a bare query spin bounded by
		``_DECODE_EVENT_SPIN_BUDGET_S``: a healthy decode step's readback lands
		inside it, so the hot path stays a few cudaEventQuery calls with no
		sleep and no probe read.  Then the poll proper, which reads the probe
		BEFORE the event every iteration.  That order is deliberate: the event
		may never fire, so a teardown that is already requested must not be made
		to wait for it.  Returning ABANDONED is how a wedged rank gets to leave.

		Returns ``_DECODE_WAIT_COMPLETED`` -- identical semantics to
		``synchronize()`` returning -- or ``_DECODE_WAIT_ABANDONED``.  The caller
		must read none of the step's outputs on ABANDONED.

		No logging and no allocation happen in either loop; the only names
		touched are module constants and the two bound callables.
		"""
		probe = self._cohort_shutdown_probe
		if probe is None:
			event.synchronize()
			return _DECODE_WAIT_COMPLETED
		spin_until = time.perf_counter() + _DECODE_EVENT_SPIN_BUDGET_S
		while True:
			if event.query():
				return _DECODE_WAIT_COMPLETED
			if time.perf_counter() >= spin_until:
				break
		while True:
			if probe():
				return _DECODE_WAIT_ABANDONED
			if event.query():
				return _DECODE_WAIT_COMPLETED
			time.sleep(_DECODE_EVENT_POLL_INTERVAL_S)

	def _apply_decode_row_token(self, seq, token_id: int, row: int, local_iteration: int) -> None:
		"""Token-dependent per-row decode bookkeeping for one live row.

		The caller has already advanced ``decoded_length`` /
		``current_context_length`` for this step and appended ``token_id`` to
		the trajectory book. This is the single implementation of the
		per-step stop semantics (EOS, length limit, repetition breaker) for
		both the undeferred and the deferred decode step.
		"""
		# DIAG: Log first 3 tokens for first 10 seqs in each decode group
		if BATCHGEN_MULTI_BATCH_DIAG and self.rank == 0 and local_iteration <= 3 and row < 10:
			logging.info(
				f"[MULTI_DIAG] iter={local_iteration} seq={seq.uuid[:8]} "
				f"decoded_len={seq.decoded_length} token={token_id}"
			)
		if self._should_stop_at_eos(token_id, seq):
			seq.eos_reached = True

		if seq.decoded_length >= seq.max_decode_length:
			seq.eos_reached = True

		# Repetition detection: consecutive same-token check (BATCHGEN_REP_DETECTION=1)
		if REP_DETECTION and not seq._rep_detected and _repetition_check_enabled(seq):
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
				_tokens = self._trajectory_decoded_tokens(seq)
				if _check_repeating_pattern(_tokens, _dl):
					seq._rep_detected = True
					seq.eos_reached = True
					logging.warning(
						f"Rank {self.rank}: REPETITION (ngram) {seq.uuid} "
						f"gid={seq.global_idx} at decoded_len={_dl}"
					)

	def _require_decode_tokens_read(self, where: str) -> None:
		"""Fail loud if a host-token consumer runs ahead of the deferred readback.

		Under DECODE_DEFERRED_TOKEN_READBACK the last launched decode step's
		tokens are not in the trajectory book yet. Anything that reads host
		tokens or makes a decision from them (input-id rebuild, the page
		boundary planner) must drain that step first; reaching here with one
		pending is a loop-ordering bug, never something to paper over.
		"""
		if getattr(self, "_decode_tokens_unread", False):
			raise DeferredTokenReadbackError(
				f"Rank {getattr(self, 'rank', '?')}: {where} needs this rank's host "
				"tokens, but the last decode step's sampled-token readback is still "
				"deferred; the decode loop must drain the pending step first"
			)

	def _begin_deferred_decode_step(
		self,
		batch: List[int],
		batch_sequences: List[SequenceEntry],
		slot: int,
		bs: int,
		local_iteration: int,
	) -> PendingDecodeStep:
		"""Token-independent half of a launched step's bookkeeping.

		Decides the live rows exactly where the undeferred step does (after the
		previous step's tokens were consumed, before this step's) and advances
		them by one, so the next step's setup reads the same context lengths
		the undeferred loop would give it. The token value is not needed for
		any of this.
		"""
		row_active = [not self._is_sequence_completed(seq) for seq in batch_sequences]
		for seq, live in zip(batch_sequences, row_active):
			if not live:
				continue
			seq.decoded_length += 1
			seq.current_context_length = (
				seq.original_prompt_length + seq.decoded_length
			)
		self._decode_tokens_unread = True
		return PendingDecodeStep(
			slot=slot,
			bs=bs,
			batch=list(batch),
			batch_sequences=list(batch_sequences),
			row_active=row_active,
			local_iteration=local_iteration,
		)

	def _consume_deferred_decode_tokens(
		self, step: PendingDecodeStep, token_ids: List[int]
	) -> None:
		"""Token-dependent half: trajectory append and stop decisions."""
		if len(token_ids) < len(step.batch_sequences):
			raise RuntimeError(
				f"decode sampled {len(token_ids)} tokens for "
				f"{len(step.batch_sequences)} rows"
			)
		# [COMPLETED_ROW_SKIP] with the mask decided at launch: completion
		# must not be re-evaluated here, the advance already moved the rows.
		self._append_trajectory_tokens(
			[seq for seq, live in zip(step.batch_sequences, step.row_active) if live],
			[tok for tok, live in zip(token_ids, step.row_active) if live],
		)
		for i, seq in enumerate(step.batch_sequences):
			if not step.row_active[i]:
				continue
			self._apply_decode_row_token(seq, token_ids[i], i, step.local_iteration)

	def _finish_deferred_decode_step(
		self,
		step: PendingDecodeStep,
		ring: TokenReadbackRing,
		*,
		overlapped: bool,
		marks: Optional[List[float]] = None,
	) -> str:
		"""Wait for a deferred step's tokens, drain its host-KV appends, consume.

		The same three phases, in the same order and with the same teardown
		exits, as the undeferred step's tail: pollable token wait (ABANDONED
		leaves before anything else), per-step [KV_APPEND_DRAIN], post-drain
		cohort flag test, bookkeeping.

		``overlapped``: the NEXT step is already enqueued on the stream, so the
		drain must not end in its device-wide sync -- that would wait for the
		whole next forward. Each append task's ``wait()`` already ends in its
		own copy-stream event sync (``SynchronizeWithEvent`` in the native
		task), so the step's host writes are complete and its staging tensors
		have no reader left; the device-wide sync only matters before GPU pages
		are freed, which happens at the page boundary, where this is called
		with ``overlapped=False``.

		Returns ``_DECODE_WAIT_COMPLETED``, ``_DECODE_WAIT_ABANDONED`` or
		``_DECODE_STEP_SHUTDOWN``; on the latter two the caller takes the
		ordered teardown exit and none of the step's tokens are read.
		``marks`` (optional) receives perf_counter readings after the wait,
		after the drain and after the bookkeeping.
		"""
		status = self._wait_decode_tokens(ring.event(step.slot))
		if marks is not None:
			marks.append(time.perf_counter())
		if status == _DECODE_WAIT_ABANDONED:
			ring.abandon(step.slot)
			return _DECODE_WAIT_ABANDONED
		if self._pending_kv_append_tasks:
			self._wait_pending_kv_append_tasks(
				defer_errors=True, device_sync=not overlapped
			)
		if marks is not None:
			marks.append(time.perf_counter())
		if self._cohort_shutdown_requested():
			ring.abandon(step.slot)
			return _DECODE_STEP_SHUTDOWN
		token_ids = ring.read(step.slot, step.bs)
		self._consume_deferred_decode_tokens(step, token_ids)
		self._decode_tokens_unread = False
		if marks is not None:
			marks.append(time.perf_counter())
		return _DECODE_WAIT_COMPLETED

	def _warn_sampling_bad_rows(self, residency) -> None:
		"""Drain the device-side sampling fallback counter and warn on it.

		Reads a device counter, so callers must be where the device has caught
		up with the sampling that wrote it (see drain_bad_rows).
		"""
		_bad_rows = residency.drain_bad_rows()
		if _bad_rows:
			logging.warning(
				f"[DECODE] invalid probabilities after filtering for "
				f"{_bad_rows} row(s) since the last heartbeat; those rows "
				"fell back to greedy"
			)

	def _poll_until(self, ready, deadline) -> bool:
		"""Poll one NON-BLOCKING readiness probe until the shared deadline.

		``ready()`` must answer "is this work finished?" without waiting:
		``KVAsyncTask.done()`` (a ``wait_for(0s)`` status test) and
		``cudaEventQuery`` behind ``torch.cuda.Event.query()`` are the two
		shapes used here.  That is the whole point of this helper.  The waits
		themselves may NOT be bounded this way, nor on a helper thread: the
		append task's ``wait`` holds the GIL while it blocks, so one such call
		anywhere in the process freezes the deadline that is supposed to bound
		it, the abort watchdog and the supervisor's self-exit net along with it.

		The probe is read once before the clock, so work that is already
		finished proves itself even if the budget is gone -- readiness is
		readiness, and self-exiting a rank whose readers have all returned would
		throw away a perfectly ordered unwind.

		Returns True only if the probe answered True.  Anything else, including
		a probe that raises (a task type with no non-blocking ``done()``), is
		reported as not-quiesced: this runs during teardown, and the caller's
		safe answer to "I cannot prove the local readers are done" is to end the
		process rather than unwind.  It never raises, for the same reason the
		quiesce itself never raises.
		"""
		try:
			while True:
				if ready():
					return True
				if time.perf_counter() >= deadline:
					return False
				time.sleep(_COHORT_QUIESCE_POLL_INTERVAL_S)
		except BaseException:
			logging.warning(
				"Rank %s: probing a decode teardown quiesce wait failed; "
				"treating it as not quiesced",
				self.rank,
				exc_info=True,
			)
			return False

	def _pending_kv_appends_done(self) -> bool:
		"""Has every pending host-KV append task finished?  Never waits.

		``done()`` is ``future_.wait_for(0s) == ready`` in the native task
		(``core/KV_Storage/host_paged_kv_worker_view.h:93-99``, bound at
		``core/batchgen_Binding.cpp:969``), so this is a status read per task and
		nothing else.  ``None`` entries are skipped exactly as the real drain
		skips them, and a worker that never created the list (the drain's own
		``hasattr`` guard admits that case) has nothing in flight.
		"""
		for task in getattr(self, "_pending_kv_append_tasks", ()):
			if task is not None and not task.done():
				return False
		return True

	def _record_quiesce_device_event(self):
		"""A non-blocking probe for the device work this rank already queued.

		This replaces the ``torch.cuda.synchronize(self.torch_device)`` the load
		drain used to end in: on an abandoned step that sync is unbounded, and
		it is the last thing standing between the teardown and the unwakeable
		state one level down.  An event recorded on the worker's current stream
		(the stream the decode step records its token event on) is ready exactly
		once the work queued ahead of it has run, and ``query()`` reads that
		with ``cudaEventQuery`` -- no wait, no GIL hazard.

		Returns the event, or ``None`` if it could not even be recorded, which
		the caller treats as "the device state cannot be proven" and self-exits.
		"""
		try:
			event = torch.cuda.Event()
			event.record(torch.cuda.current_stream(self.torch_device))
			return event
		except BaseException:
			logging.warning(
				"Rank %s: recording the decode teardown quiesce device probe "
				"failed; treating it as not quiesced",
				self.rank,
				exc_info=True,
			)
			return None

	def _self_exit_on_quiesce_deadline(self) -> None:
		"""End this process instead of unwinding past a quiesce that did not land.

		The ordered teardown releases the very mappings the parked waits hold
		raw pointers into -- the host-KV appends are std::async threads reading
		the worker view's page table unlocked -- so an unwind that cannot prove
		they finished is a use-after-free during teardown.  That is the one
		hazard this whole exit path exists to avoid, so when the bound expires
		the rank takes the supervisor's own bounded outcome directly.

		The marker keeps the ``path=self-exit`` grep key the supervisor's safety
		net emits, so marker accounting still sums, and adds a reason so the two
		causes stay distinguishable in the logs.  ``os._exit`` runs no atexit
		hook and no stream flush, so the marker is flushed first or it is lost.

		The status, the path marker and the pre-exit flush are the worker exit
		protocol the supervisor uses too; both take them from
		``batchgen.eventlog.process``, so there is one definition of each.
		"""
		from batchgen.eventlog.process import (
			EXIT_PATH_SELF_EXIT,
			WORKER_SELF_EXIT_CODE,
			flush_logs_before_exit,
			record_exit,
		)

		logging.error(
			"[shutdown] worker exit path=%s rank=%s pid=%s reason=%s",
			EXIT_PATH_SELF_EXIT,
			self.rank,
			os.getpid(),
			_COHORT_QUIESCE_DEADLINE_REASON,
		)
		flush_logs_before_exit()
		record_exit(WORKER_SELF_EXIT_CODE, _COHORT_QUIESCE_DEADLINE_REASON)
		os._exit(WORKER_SELF_EXIT_CODE)

	def _quiesce_decode_for_cohort_shutdown(self, pending_async_task=None) -> None:
		"""Let LOCAL in-flight decode work finish before the ordered teardown.

		Called on the decode loop's teardown exits.  Everything waited on here
		is work this rank already launched and that holds raw pointers into
		mappings the ordered teardown is about to release -- the host-KV append
		tasks (std::async threads reading the worker view's page table with no
		lock) and an async host->GPU KV load.  Returning out of decode with
		either still in flight is a use-after-free during teardown, which is
		why the exit quiesces instead of just leaving.

		Every wait here is deliberately LOCAL.  ``defer_errors=True`` keeps the
		append drain off the cross-rank error sync, which is the one thing this
		exit must never do: the peer whose death started this teardown will
		never join a collective again, so a guarded one would hang exactly
		where the unfixed code hung.

		Nothing raises out of here.  This runs on a teardown path whose whole
		purpose is to unwind normally -- an exception would escape as an error
		and lose the ordered exit -- so a failed quiesce is logged and the
		teardown continues; the process is about to drop these mappings anyway.

		The quiesce is BOUNDED, by one shared ``_COHORT_QUIESCE_DEADLINE_S``
		budget, and the bound is established by NON-BLOCKING PROBES ONLY.  On
		the exit taken when the step's token event was ABANDONED the step's
		device work is parked behind the dead collective, so every wait here is
		parked with it -- and MEASURED, the append task's ``wait`` blocks while
		HOLDING THE GIL (no ``gil_scoped_release`` in its binding), which
		freezes the deadline meant to bound it, the abort watchdog and the
		supervisor's self-exit net.  A blocking wait therefore may not be
		called, on this thread or any other, until something non-blocking has
		proven it cannot block:

		  1. every pending host-KV append task reports ``done()``;
		  2. the async KV load task reports ``done()`` (the dual coordinator's
		     composite task answers for both of its halves);
		  3. the device-work probe recorded at entry reports ``query()``, which
		     is what the load drain's unbounded device sync became.

		All three inside the budget: the real drains then run with every future
		ready and the ordered unwind proceeds exactly as before.  Any of them
		expiring is the one case that may not unwind at all (the readers are
		still live, holding page-table pointers), so it self-exits with the
		reason recorded, inside ~5 s instead of the supervisor's grace window.
		On the healthy clean-restart exits all three probes answer immediately.
		"""
		deadline = time.perf_counter() + _COHORT_QUIESCE_DEADLINE_S
		device_event = self._record_quiesce_device_event()

		quiesced = self._poll_until(self._pending_kv_appends_done, deadline)
		if quiesced and pending_async_task is not None:
			quiesced = self._poll_until(lambda: pending_async_task.done(), deadline)
		if quiesced:
			quiesced = device_event is not None and self._poll_until(
				device_event.query, deadline
			)

		if quiesced:
			try:
				if self._pending_kv_append_tasks:
					self._wait_pending_kv_append_tasks(defer_errors=True)
			except BaseException:
				logging.warning(
					"Rank %s: draining host-KV appends at the decode teardown "
					"boundary failed; continuing the ordered teardown",
					self.rank,
					exc_info=True,
				)
			try:
				if pending_async_task is not None:
					pending_async_task.wait()
			except BaseException:
				logging.warning(
					"Rank %s: draining the pending async KV load at the decode "
					"teardown boundary failed; continuing the ordered teardown",
					self.rank,
					exc_info=True,
				)

		self.disable_decode_watchdog()
		if not quiesced:
			self._self_exit_on_quiesce_deadline()

	@property
	def cohort_shutdown_observed(self) -> bool:
		"""True once the decode-step boundary check retired this rank.

		Read by the worker supervisor to name the exit path.  It is a latch, not
		a re-read of the probe: the supervisor must be able to tell a rank that
		left THROUGH the decode boundary from one that merely happened to have
		the flag set by the time it finished something else.
		"""
		return self._cohort_shutdown_observed

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
	# The worker owns the canonical config (`self.eos_token_ids`,
	# `self.model_context_length`, `self.rank`);
	# `_make_completion_context` snapshots them into a frozen
	# `CompletionContext` per call.

	def _make_completion_context(self) -> CompletionContext:
		return CompletionContext(
			eos_token_ids=frozenset(self.eos_token_ids),
			model_context_length=self.model_context_length,
			rank=self.rank,
		)

	def _should_stop_at_eos(self, token_id: int, seq) -> bool:
		# A per-request ignore_eos keeps the EOS token from marking the sequence
		# done at the source, so every reader of eos_reached agrees.
		return CompletionHandler.should_stop_at_eos(
			self._make_completion_context(), token_id, getattr(seq, "ignore_eos", False)
		)

	def _is_sequence_completed(self, seq) -> bool:
		return CompletionHandler.is_sequence_completed(self._make_completion_context(), seq)

	def _get_finish_reason(self, seq) -> str:
		return CompletionHandler.get_finish_reason(self._make_completion_context(), seq)

	def _collective_gpu_kv_allocation_preflight(
		self, required_pages: int, free_pages: int
	) -> bool:
		"""Require every rank to approve a GPU-KV allocation before mutation."""
		local_can_allocate = int(required_pages >= 0 and required_pages <= free_pages)
		if self.world_size == 1:
			return bool(local_can_allocate)
		if not dist.is_initialized():
			raise RuntimeError(
				"distributed process group is required for GPU-KV allocation preflight"
			)
		can_allocate = torch.tensor(
			[local_can_allocate], dtype=torch.int32, device=self.torch_device
		)
		dist.all_reduce(can_allocate, op=dist.ReduceOp.MIN)
		return bool(int(can_allocate.item()))

	def _allocate_gpu_kv_two_page_buffer(
		self, 
		local_sequence_ids: List[int],
		load_from_host: bool = True,
		collective_preflight: bool = False,
	) -> bool:
		"""
		Allocate GPU KV pages using two-page buffer strategy.
		
		Returns:
			True if allocation succeeded, False otherwise.
		"""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			if collective_preflight:
				return self._collective_gpu_kv_allocation_preflight(0, 0)
			return False
		if not manager.is_initialized:
			if collective_preflight:
				return self._collective_gpu_kv_allocation_preflight(0, 0)
			return False
		if not local_sequence_ids:
			if collective_preflight:
				return self._collective_gpu_kv_allocation_preflight(
					0, manager.get_stats().num_free_pages
				)
			return True
		
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
		# With prefix sharing the physical demand is lower than the logical
		# page sum: a Host page already mapped (or repeated in this batch)
		# costs nothing. The preview runs the allocator's own validation, so
		# a demand it accepts cannot fail validation at allocation time.
		try:
			page_keys = self._host_identity_page_keys(
				manager, global_ids, pages_per_seq
			)
			demand_pages = (
				total_pages
				if page_keys is None
				else manager.count_new_shared_pages(
					global_ids, pages_per_seq, page_keys
				)
			)
		except Exception:
			if collective_preflight:
				# The failed rank still joins the collective so peers cannot
				# advance into allocation while this rank raises.
				self._collective_gpu_kv_allocation_preflight(1, 0)
			raise
		if collective_preflight:
			if not self._collective_gpu_kv_allocation_preflight(
				demand_pages, free_pages
			):
				logging.warning(
					f"Rank {self.rank}: collective GPU KV allocation preflight failed "
					f"(need {demand_pages} of {total_pages} logical, "
					f"local free {free_pages})"
				)
				return False
		elif demand_pages > free_pages:
			logging.error(
				f"Rank {self.rank}: Cannot allocate GPU KV - need {total_pages} pages, "
				f"only {free_pages} free"
			)
			# Don't set gpu_pages_allocated since we're failing
			return False
		
		try:
			self._allocate_gpu_pages_with_host_identity(
				manager, global_ids, pages_per_seq
			)
		except RuntimeError as exc:
			if "Insufficient free pages" not in str(exc):
				raise
			logging.error(
				"Rank %s cannot allocate shared GPU KV pages: %s",
				self.rank,
				exc,
			)
			return False

		# Now safe to update tracking.
		for local_idx in local_sequence_ids:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			pages = seq.get_gpu_pages_for_two_page_buffer()
			seq.gpu_pages_allocated = pages
			# Mark that this sequence has received its initial GPU reservation
			seq.mark_initial_gpu_reservation_done()
		
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
				for layer_idx, k_tensor, v_tensor, scale_tensor in (
					_normalize_aux_kv_entry(entry) for entry in entries_aux
				):
					if k_tensor.dim() == 3:
						k_tensor = k_tensor.unsqueeze(2)
					_assert_deferred_kv_rows("aux", layer_idx, k_tensor)
					_prepared_aux.append((layer_idx, k_tensor, None, scale_tensor))
					self._pending_kv_append_tensors.append(k_tensor)
					if scale_tensor is not None:
						self._pending_kv_append_tensors.append(scale_tensor)
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
				for layer_idx, k_tensor, v_tensor, scale_tensor in (
					_normalize_aux_kv_entry(entry) for entry in entries_aux
				):
					if k_tensor.dim() == 3:
						k_tensor = k_tensor.unsqueeze(2)
					_assert_deferred_kv_rows("aux", layer_idx, k_tensor)

					task = aux_view.async_append_decode_kv_to_host(
						layer_idx=layer_idx,
						sequence_ids=sequence_ids,
						k_tensor=k_tensor,
						v_tensor=None,
						sequence_lengths=sequence_lengths,
						scale_tensor=scale_tensor,
					)

					self._pending_kv_append_tensors.append(k_tensor)
					if scale_tensor is not None:
						self._pending_kv_append_tensors.append(scale_tensor)
					if task is not None:
						self._pending_kv_append_tasks.append(task)

		# This function only LAUNCHES the D2H append tasks. The decode loop
		# drains them every step right after the sampled-token readback, in
		# a step-split slot of its own; see [KV_APPEND_DRAIN] in
		# ``decoding_continuous`` for why that drain must stay per step.
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
		# Resolve through SequenceBatch's maintained gid index. This runs on
		# every decode step, so rebuilding a gid -> entry dict here cost one
		# dict comprehension over the whole batch per step.
		grow_requests = []
		for global_idx, write_pos in zip(sequence_ids, sequence_lengths):
			seq = self.global_batch.get_sequence_by_global_idx(int(global_idx))
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
				# The guards above bound required_tokens to the token budget,
				# so the owner's budget cap never engages here.
				growth_pages = seq.get_host_growth_pages(
					required_tokens - int(seq.host_token_capacity)
				)
				grow_requests.append((int(global_idx), growth_pages))

		if not grow_requests:
			return

		# Host pages are allocated only by the rank-0 boundary planner.  A
		# deferred append is already inside a decode interval whose reservation
		# was validated before forward; reaching this branch means the scheduler
		# violated that invariant.  Growing here would bypass completion release,
		# eviction, mirrored DSA allocation, and G>1 host ownership.
		raise RuntimeError(
			"host KV append reached without a boundary reservation: "
			f"requests={grow_requests[:8]}; decode must enter the page-boundary "
			"planner before issuing the forward"
		)

	def _aux_layers_with_slots(self):
		"""Logical layer ids that have a slot in the aux (indexer) KV pools.

		None = no layer mapping (every engine layer has an aux slot). GLM-5.2
		maps only its 21 indexer layers; skip layers' index-K is never read,
		so decode appends for them are dropped at the queueing choke points —
		the mapped host view hard-fails on unmapped layers by design.
		"""
		cached = getattr(self, "_aux_layers_with_slots_cache", False)
		if cached is not False:
			return cached
		coord = getattr(self, "gpu_paged_kv_cache_manager", None)
		aux_mgr = getattr(coord, "auxiliary", None)
		if aux_mgr is None:
			return None  # aux not initialized yet; do not cache
		aux_map = getattr(aux_mgr.config, "logical_to_physical_layer", None)
		result = None
		if aux_map is not None:
			result = frozenset(
				l for l, p in enumerate(aux_map) if p is not None and int(p) >= 0
			)
		self._aux_layers_with_slots_cache = result
		return result

	def _append_decode_kv_to_host_aux_async(
		self,
		layer_idx: int,
		batch: List[int],
		k_tensor: torch.Tensor,
		v_tensor: torch.Tensor = None,
		scale_tensor: torch.Tensor = None,
	) -> None:
		"""Fire-and-forget auxiliary (indexer) KV append to host.

		Uses the auxiliary host worker view and the shared pending task list,
		so one flush drains primary and auxiliary appends together.

		``scale_tensor`` carries the fp32 per-token scales that a page-split
		FP8 aux region requires alongside the quantized keys; an AoS region
		rejects it. The host view validates the pairing, so a producer that
		forgets the scales fails loud instead of leaving a page's scale region
		at its previous contents.
		"""
		aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)
		if aux_view is None or not batch:
			return
		valid_layers = self._aux_layers_with_slots()
		if valid_layers is not None and layer_idx not in valid_layers:
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
			scale_tensor=scale_tensor,
		)

		if not hasattr(self, '_pending_kv_append_tensors'):
			self._pending_kv_append_tensors = []
		self._pending_kv_append_tensors.append(k_tensor)
		if scale_tensor is not None:
			self._pending_kv_append_tensors.append(scale_tensor)

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
		self.model_context_length = resolve_max_model_len(
			getattr(self.args, "max_model_len", None), model_max
		)
		if self.rank == 0:
			logging.info("Deployment max_model_len requested: %s", self.model_context_length)

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
			"enable_deepep": getattr(self.args, "enable_deepep", False),
		}
		logging.info(f"kv_dtype: {input_arguments['kv_dtype']}")
			
		self.input_arguments = InputArguments(**input_arguments)
		self.initializer = get_initializer(self.huggingface_ckpt_name)
		self.initializer = self.initializer(self.input_arguments)
		self.core_engine, self.engine_config, self.model_config, self.loaded_model_config = (
			self.initializer.Init(self.weights_storage)
		)
		self.engine_config.Basic_Config.max_model_len = self.model_context_length
		self._weight_passes = _WeightPasses(self.core_engine, self.rank)
		# Pull the cuda-graph adapter from the initializer (None if the model
		# does not implement one). GraphRuntime owns the handle, the one
		# startup type backstop against the ModelCudaGraphAdapter ABC, and the
		# import of that ABC.
		self._graph.adopt_adapter(
			self.initializer, model_name=getattr(self, 'model_name', None)
		)

		if isinstance(self.host_paged_kv_worker_view, DualHostKVCoordinator):
			self.core_engine.host_paged_kv_worker_view = self.host_paged_kv_worker_view.primary
			self.host_paged_kv_worker_view_aux = self.host_paged_kv_worker_view.auxiliary
		else:
			self.core_engine.host_paged_kv_worker_view = self.host_paged_kv_worker_view
		self.engine_config.Basic_Config.num_queries = num_queries

		# Set CUDA graph config from command-line args. --enable-cuda-graph on a
		# model with no flag-driven graph path raises (validate_server_args
		# rejects it first, before any worker starts).
		require_cuda_graph_flag_supported(
			getattr(self, "model_name", None),
			enable_cuda_graph=getattr(self.args, "enable_cuda_graph", False),
		)
		if self.args.disable_cuda_graphs:
			self.engine_config.Basic_Config.enable_cuda_graphs = False
		elif (
			"glm" in (getattr(self, "model_name", "") or "").lower()
			and glm5_any_cuda_graph_requested(
				getattr(self, "model_name", None),
				enable_cuda_graph=getattr(self.args, "enable_cuda_graph", False),
			)
		):
			self.engine_config.Basic_Config.enable_cuda_graphs = True
		elif cli_enables_whole_model_graph(
			getattr(self, "model_name", None),
			enable_cuda_graph=getattr(self.args, "enable_cuda_graph", False),
		):
			# K2.5 (cuda-graph adapter) and GPT-OSS-120B whole-model decode
			# graphs. The config flag defaults off; --enable-cuda-graph flips it
			# on so _warmup_cuda_graphs() reaches _setup_cuda_graphs().
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
		self._verify_manager_phase_contract()

		# The token pool is fixed for the lifetime of the worker. Its byte budget
		# is explicit, while the number of admitted trajectories is derived from
		# the model context and the canonical 64-token page size.
		self._initialize_trajectory_book()

		# NOTE: GPU KV cache size is calculated in generate() via _init_gpu_kv_with_actual_size()
		# after _load_decode_model() loads model weights to GPU. At this point (init),
		# only the model skeleton exists and weights haven't been loaded yet.

		logging.info(f"Rank {self.rank}: One-time core initialization completed")

	def _initialize_trajectory_book(self) -> None:
		"""Create or attach the node-shared fixed input-id trajectory pool."""
		if self._traj.trajectory_book is not None:
			return
		if self.model_context_length is None:
			raise RuntimeError("model context must be known before creating QueryBook")
		page_tokens = SequenceEntry.PAGE_SIZE
		capacity_bytes = int(self._input_ids_pool_size_gb * (1024 ** 3))
		page_bytes = page_tokens * QueryBook.TOKEN_BYTES
		page_count = capacity_bytes // page_bytes
		if page_count <= 0:
			raise ValueError(
				f"input_ids pool budget {capacity_bytes} bytes cannot hold one "
				f"{page_tokens}-token page"
			)
		storage, memfd = allocate_node_shared_tensor(
			label=f"{self._query_book_shm_prefix}trajectory_n{node_of_rank(self.rank, self.local_world_size)}",
			rows=page_count,
			width=page_tokens,
			dtype=torch.int32,
			is_creator=self.local_rank == 0,
			exchange=self._exchange_node_memfd_identity,
			barrier=dist.barrier,
		)
		self._traj.trajectory_memfd = memfd
		self._traj.trajectory_book = QueryBook(
			capacity_bytes=page_count * page_bytes,
			page_tokens=page_tokens,
			storage=storage,
		)
		logging.info(
			"Rank %s: unified trajectory pool ready: budget=%.3f GiB "
			"pages=%s page_tokens=%s max_context=%s sequence_capacity=%s",
			self.rank,
			self._traj.trajectory_book.memory_bytes / (1024 ** 3),
			self._traj.trajectory_book.page_count,
			page_tokens,
			self.model_context_length,
			self._traj.trajectory_book.reservation_total_capacity(self.model_context_length),
		)

	def _publish_trajectory_pool_capacity(self) -> None:
		"""Publish derived scheduling capacity once rank zero has a response queue."""
		if self.rank != 0 or self._response_queue is None or self._traj.trajectory_book is None:
			return
		signature = (
			self._traj.trajectory_book.reservation_total_capacity(self.model_context_length),
			self._traj.trajectory_book.reservation_free_count(self.model_context_length),
			self._traj.trajectory_book.free_page_count,
			self._traj.trajectory_book.active_count,
			self._traj.trajectory_book.largest_free_extent_pages,
		)
		if self._traj.trajectory_pool_capacity_signature == signature:
			return
		self._traj.trajectory_pool_capacity_signature = signature
		self._response_queue.put({
			"type": "trajectory_pool_capacity",
			# Reject pre-protocol snapshots: the legacy ``capacity`` field
			# represented free reservations and cannot size the fixed scheduler.
			"capacity_semantics_version": 1,
			# ``total_capacity`` is immutable for the worker lifetime and sizes
			# SchedulingPool.  ``capacity`` remains a compatibility spelling for
			# clients that only display the total.
			"total_capacity": signature[0],
			"capacity": signature[0],
			"free_reservations": signature[1],
			"free_pages": signature[2],
			"active_count": signature[3],
			"largest_free_extent_pages": signature[4],
			"page_tokens": self._traj.trajectory_book.page_tokens,
			"model_context_length": self.model_context_length,
			"max_model_len": self.model_context_length,
			"capacity_bytes": self._traj.trajectory_book.memory_bytes,
		})

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
		This is called after _bind_admitted_prompts() has raised
		self.max_input_length to the longest admitted prompt.
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

	def _shared_page_identity_sources(self, manager):
		"""The pools whose GPU pages can be deduplicated by Host page id.

		Returns one ``(pool name, GPU manager, Host worker view)`` triple per
		pool, or ``None`` when shared-page identity does not apply at all.
		``None`` is the single condition behind BOTH the allocator and the
		decode admission credit: plain allocation gives every row its own
		physical copy, so a credit there would charge for a copy that was
		really made.

		A DSA model (GLM-5.x, DeepSeek-V3.2) runs TWO pools -- primary MLA
		latent KV and auxiliary indexer KV -- backed by two separate Host KV
		regions. Each pool therefore gets its own Host worker view, which is
		its own key space; group 0 is the MLA region and group 1 the indexer
		region (``DualHostKVCoordinator.views_by_group``), matching
		primary/auxiliary on the GPU side.
		"""
		if manager is None or not self.enable_prefix_cache:
			return None
		if isinstance(manager, DualKVCacheCoordinator):
			host_view = getattr(self, "host_paged_kv_worker_view", None)
			if not isinstance(host_view, DualHostKVCoordinator):
				return None
			views = host_view.views_by_group()
			primary_view = views.get(0)
			auxiliary_view = views.get(1)
			if primary_view is None or auxiliary_view is None:
				return None
			return (
				("primary", manager.primary, primary_view),
				("auxiliary", manager.auxiliary, auxiliary_view),
			)
		worker_view = getattr(
			self.core_engine, "host_paged_kv_worker_view", None
		)
		if worker_view is None:
			return None
		return (("primary", manager, worker_view),)

	def _host_identity_page_keys(
		self,
		manager,
		global_sequence_ids: Sequence[int],
		sequence_tokens: Sequence[int],
	):
		"""The page-key argument for ``manager``'s keyed allocation, or ``None``.

		One manager: a list of Host page id rows, one row per sequence, in
		logical page order -- equal ids mean one physical GPU page.

		A DSA dual coordinator: ``DualPoolPageKeys``, which carries that same
		row list ONCE PER POOL. The two pools are separate Host KV regions, so
		a Host page id identifies content only inside its own region and the
		two key spaces must never be crossed; each pool deduplicates on its
		own ids.

		``None`` means shared-page identity does not apply (prefix cache off,
		or the Host views that define the identity are not bound) and plain
		allocation is used.
		"""
		sources = self._shared_page_identity_sources(manager)
		if sources is None:
			return None
		rows_by_pool: Dict[str, List[List[int]]] = {}
		for pool_name, pool_manager, worker_view in sources:
			rows_by_pool[pool_name] = self._pool_host_identity_page_keys(
				pool_name,
				pool_manager,
				worker_view,
				global_sequence_ids,
				sequence_tokens,
			)
		if isinstance(manager, DualKVCacheCoordinator):
			return DualPoolPageKeys.from_rows(
				rows_by_pool["primary"], rows_by_pool["auxiliary"]
			)
		return rows_by_pool["primary"]

	def _pool_host_identity_page_keys(
		self,
		pool_name: str,
		pool_manager,
		worker_view,
		global_sequence_ids: Sequence[int],
		sequence_tokens: Sequence[int],
	) -> List[List[int]]:
		"""One pool's Host page ids behind each sequence's GPU reservation."""
		host_tables = worker_view.build_page_table(global_sequence_ids)
		page_size = int(pool_manager.config.page_size_tokens)
		page_keys: List[List[int]] = []
		for sequence_id, tokens, host_pages in zip(
			global_sequence_ids, sequence_tokens, host_tables
		):
			required = math.ceil(int(tokens) / page_size)
			if len(host_pages) < required:
				raise RuntimeError(
					f"Host page table ({pool_name} pool) is shorter than GPU "
					f"reservation: sequence={sequence_id}, "
					f"host_pages={len(host_pages)}, required={required}"
				)
			page_keys.append([int(page) for page in host_pages[:required]])
		self._assert_private_writable_page_keys(
			pool_name, pool_manager, global_sequence_ids, page_keys, page_size
		)
		return page_keys

	def _assert_private_writable_page_keys(
		self,
		pool_name: str,
		pool_manager,
		global_sequence_ids: Sequence[int],
		page_keys: Sequence[Sequence[int]],
		page_size: int,
	) -> None:
		"""Fail loud if a page the row will WRITE carries a shared key.

		Sharing a physical page is safe only for a committed prefix, which is
		immutable and page aligned. Everything from the row's write head on is
		its private tail: decode writes the new token at position
		``current_context_length`` (``update_layer_decode_new_token``), i.e.
		into logical page ``current_context_length // page_size``, and every
		later page of the reservation is future tail. Those pages come from
		the row's OWN Host reservation, so their Host page ids -- the keys --
		are private by construction; this states that invariant as a check
		instead of trusting it, because a shared key here would let two rows
		write one physical page.
		"""
		write_heads: List[int] = []
		for sequence_id in global_sequence_ids:
			sequence = self.global_batch.get_sequence_by_global_idx(
				int(sequence_id)
			)
			if sequence is None:
				raise RuntimeError(
					"shared GPU page identity needs the row's context length: "
					f"global sequence {sequence_id} is not in the batch"
				)
			write_heads.append(
				int(sequence.current_context_length) // int(page_size)
			)
		rows_per_key: Dict[int, int] = {}
		for row in page_keys:
			for key in row:
				rows_per_key[int(key)] = rows_per_key.get(int(key), 0) + 1
		for sequence_id, row, write_head in zip(
			global_sequence_ids, page_keys, write_heads
		):
			for index in range(max(0, write_head), len(row)):
				key = int(row[index])
				if (
					rows_per_key[key] > 1
					or pool_manager.is_shared_page_key_mapped(key)
				):
					raise RuntimeError(
						f"{pool_name} GPU pool: sequence {sequence_id} would "
						f"share writable logical page {index} (write head "
						f"{write_head}); only a committed prefix page may be "
						"shared"
					)

	def _allocate_gpu_pages_with_host_identity(
		self,
		manager,
		global_sequence_ids: Sequence[int],
		sequence_tokens: Sequence[int],
	):
		"""Share GPU pages when sequences reference the same Host page ids."""

		page_keys = self._host_identity_page_keys(
			manager, global_sequence_ids, sequence_tokens
		)
		if page_keys is None:
			return manager.allocate_pages_for_sequences(
				global_sequence_ids, sequence_tokens
			)
		allocations = manager.allocate_pages_for_sequences_with_page_keys(
			global_sequence_ids,
			sequence_tokens,
			page_keys,
		)
		if self.prefix_cache_debug_stats:
			sharing = manager.get_prefix_gpu_share_stats()
			new_physical_pages = {
				int(page)
				for row in allocations.values()
				for page in row
			}
			# Both pools request the same logical page count per row, so the
			# primary rows describe the request for either shape.
			primary_rows = (
				page_keys.primary
				if isinstance(page_keys, DualPoolPageKeys)
				else page_keys
			)
			self._emit_prefix_cache_metric(
				"prefix_gpu_share",
				sequences=len(global_sequence_ids),
				requested_logical_pages=sum(len(row) for row in primary_rows),
				new_physical_pages=len(new_physical_pages),
				logical_page_references=sharing.logical_page_references,
				physical_pages=sharing.physical_pages,
				keyed_physical_pages=sharing.keyed_physical_pages,
				multi_referenced_pages=sharing.multi_referenced_pages,
				reused_logical_references=sharing.reused_logical_references,
			)
		return allocations

	def _bind_gpu_paged_kv_manager(self, manager) -> None:
		"""Bind GPU KV manager to both worker and core_engine.

		If manager is a DualKVCacheCoordinator, the primary manager is bound
		to existing gpu_paged_kv_manager slots and the auxiliary (indexer) is
		bound to gpu_paged_kv_manager_aux slots.
		"""
		self.gpu_paged_kv_cache_manager = manager
		self._gpu_kv_pool_created = True
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

	def _deployment_context_bound_tokens(self) -> int:
		"""Longest context any sequence can reach for the life of this server.

		Admission clamps every sequence to `model_context_length`
		(`clamp_decode_to_context`, and the `ctx_lens >= model_context_length`
		completion test), so no decode step's `max_seqlen` can ever exceed it.
		That makes it the one DEPLOYMENT bound available to capture-time
		sizing: `max_input_length` / `max_decoding_length` are per-batch — set
		by `Init` and grown at admission — so sizing from them describes
		whichever batch happens to arrive first, not the deployment.

		`_initialize_core_components` sets it from the model config (131072
		default) before any KV pool or graph exists, so it is always available
		to the pool and capture paths. Returns 0 only before core init; the
		caller decides whether that is an error.
		"""
		return int(getattr(self, "model_context_length", 0) or 0)

	def _make_page_table_capacity_request(self) -> PageTableCapacityRequest:
		engine_config = getattr(self, "engine_config", None)
		engine_module_global_batch_size: Optional[int] = None
		engine_module_attn_decoding_micro_batch_size: Optional[int] = None
		engine_basic_num_queries: Optional[int] = None
		engine_basic_decode_graph_max_bucket: Optional[int] = None
		if engine_config is not None:
			basic = engine_config.Basic_Config
			module_batching = engine_config.Module_Batching_Config
			engine_module_global_batch_size = module_batching.global_batch_size
			engine_module_attn_decoding_micro_batch_size = module_batching.attn_decoding_micro_batch_size
			engine_basic_num_queries = basic.num_queries
			# A whole-model decode graph addresses one page-table row per
			# sequence of its bucket; the planner's top bucket (K3 H200: the
			# KDA slot count, 160 > the 128 arg default) must fit.
			graph_buckets = getattr(basic, "decode_graph_buckets", None) or ()
			engine_basic_decode_graph_max_bucket = max(graph_buckets) if graph_buckets else None
		model_config = getattr(self, "model_config", None)
		# Deployment bound first, model config second. `model_context_length`
		# is what admission clamps to, and it exists even when the config class
		# does not declare `max_position_embeddings` — a config that lacks the
		# field used to drop the bound from the candidate set entirely, leaving
		# the page table's token span at the first batch's
		# `max_input_length + max_decoding_length`. A graph captured against
		# that span then rejected every later, longer decode step.
		model_max_position_embeddings = self._deployment_context_bound_tokens() or getattr(
			model_config, "max_position_embeddings", None
		)
		args = getattr(self, "args", None)
		args_cuda_graph_max_bucket_size = getattr(args, "cuda_graph_max_bucket_size", None) if args is not None else None
		# The derived ladder can top out above the flag (at the per-rank
		# decode width); the page table must address every row of that bucket.
		decode_graph_top_bucket = (
			self._decode_graph_top_bucket() if engine_config is not None else None
		)
		return PageTableCapacityRequest(
			engine_module_global_batch_size=engine_module_global_batch_size,
			engine_module_attn_decoding_micro_batch_size=engine_module_attn_decoding_micro_batch_size,
			engine_basic_num_queries=engine_basic_num_queries,
			model_max_position_embeddings=model_max_position_embeddings,
			args_cuda_graph_max_bucket_size=args_cuda_graph_max_bucket_size,
			engine_basic_decode_graph_max_bucket=engine_basic_decode_graph_max_bucket,
			decode_graph_top_bucket=decode_graph_top_bucket,
		)

	def _with_cuda_graph_page_table_capacity(self, config):
		return KVCacheManager.apply_page_table_capacity(
			self._make_page_table_capacity_request(), config
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
		self._traj.free_kda_sequences(global_sequence_ids)

	def _destroy_gpu_paged_kv_cache(
		self, *, empty_cuda_cache: bool = False, keep_buffers: bool = False
	) -> None:
		"""Destroy the GPU paged KV cache manager if it is present.

		With ``keep_buffers`` the manager only drops every sequence; its KV
		pools and CUDA-graph page-table storage stay at their addresses.
		"""
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

		if keep_buffers:
			manager.reset_allocations()
		else:
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
			num_gpus_per_node=self.local_world_size,
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
		node_id = self.rank_ctx.node_id
		node_rank_start = node_id * self.local_world_size
		node_rank_end = min(node_rank_start + self.local_world_size, self.world_size)

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
			'node_id': self.rank_ctx.node_id,
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

	def _log_host_kv_health(self) -> None:
		"""Log the audited health of this node's host-KV region(s).

		One line per region: the used-page identity, the sequence-table
		occupancy, and the region's lifetime counters (see
		``batchgen.kv_cache.host_kv_health``). ``get_audited_stats()`` is
		O(table + pages) under both region locks; call this only from the
		periodic heartbeat.
		"""
		view = getattr(self, "host_paged_kv_worker_view", None)
		if view is None:
			return
		if isinstance(view, DualHostKVCoordinator):
			regions = (
				("primary", view.primary),
				("aux", view.require_auxiliary("host-KV health")),
			)
		else:
			regions = (("host", view),)
		node_id = self.rank_ctx.node_id
		for name, region_view in regions:
			logging.info(
				f"[HOST_KV] node={node_id} "
				f"{format_host_kv_health(name, region_view.get_audited_stats())}"
			)

	def _gather_host_kv_stats_by_node(self, worker_view: Optional[object]) -> List[Dict[str, int]]:
		"""Gather one host-KV pool stat record per node.

		Host KV is shared by ranks on the same node, not globally. Rank 0 uses
		these per-node stats to plan dynamic host growth against the same pool
		that each owner rank will later allocate from.

		Each record also carries the retained prefix pages the node can reclaim
		right now (resident, no live attachment, no pending load): growth may
		consume them exactly as prefill admission does, and the boundary's
		growth phase releases them on demand before the fresh capacity
		preflight. Zero when the prefix cache is off.
		"""
		num_nodes = self.rank_ctx.num_nodes
		report_free = 0
		report_total = 0
		report_reclaimable = -1
		report_node = -1
		if worker_view is not None and self.local_rank == 0:
			try:
				stats = worker_view.get_stats()
				report_node = self.rank_ctx.node_of(self.rank)
				report_free = int(stats.num_free_pages)
				report_total = int(stats.num_total_pages)
				report_reclaimable = (
					max(0, int(self.prefix_cache_coordinator.count_releasable_pages(0)))
					if (
						self.enable_prefix_cache
						and self.prefix_cache_coordinator is not None
					)
					else 0
				)
			except Exception:
				# Carry a rank-local stats failure through the collective so
				# peers do not block while this rank raises before all-gather.
				logging.exception("Rank %s: host-KV stats snapshot failed", self.rank)
				report_node = -2
				report_free = -1
				report_total = -1
				report_reclaimable = -1

		stats_tensor = torch.tensor(
			[report_node, report_free, report_total, report_reclaimable],
			dtype=torch.int64,
			device=self.torch_device,
		)
		gathered = [torch.zeros_like(stats_tensor) for _ in range(self.world_size)]
		dist.all_gather(gathered, stats_tensor)
		if any(int(item[0].item()) == -2 for item in gathered):
			raise RuntimeError(
				"[SCHED_INVARIANT] host-KV stats snapshot failed on one or more ranks"
			)

		per_node_stats = []
		reports_by_node = {}
		for item in gathered:
			node_id = int(item[0].item())
			if node_id >= 0:
				reports_by_node[node_id] = {
					'node_id': node_id,
					'num_free_pages': int(item[1].item()),
					'num_total_pages': int(item[2].item()),
					'num_reclaimable_pages': int(item[3].item()),
				}

		for node in range(num_nodes):
			per_node_stats.append(reports_by_node.get(node, {
				'node_id': node,
				'num_free_pages': 0,
				'num_total_pages': 0,
				'num_reclaimable_pages': 0,
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

	def _credit_reclaimable_prefix_pages(
		self, stats: Dict[str, int]
	) -> Dict[str, int]:
		"""Report reclaimable retained prefix pages as free, not as pressure.

		The Host allocator's counters are already PHYSICAL: a prefix several
		sequences attach is one allocation and is charged to ``used`` exactly
		once, so this node's utilisation never double-counts a shared page.
		What it cannot distinguish is a page holding live KV from a page
		holding a RETAINED prefix that nobody is reading -- a copy the prefix
		cache keeps for a future hit and that admission reclaims on demand,
		which is why ``per_node_host_reclaimable`` already counts it as part of
		the admission budget.

		Left as ``used``, those pages hold free% under the watermark forever,
		so a pool filled with retained prefixes never preempts decode for
		prefill -- the one situation the trigger exists for. Crediting them
		here (and only here, so the legacy/native utilisation parity check
		keeps comparing the raw counters) makes the trigger see the capacity
		admission would actually get.

		Read only when work is waiting: the trigger cannot fire otherwise, and
		the count is an O(max_nodes) scan under the coordinator's lock.
		"""
		if not self.enable_prefix_cache:
			return stats
		coordinator = getattr(self, "prefix_cache_coordinator", None)
		if coordinator is None:
			return stats
		waiting = self.global_batch.has_queueing() or (
			self.enable_host_kv_eviction and self.global_batch.has_evicted()
		)
		if not waiting:
			return stats
		return credit_reclaimable_pages(
			stats, int(coordinator.count_releasable_pages(0))
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
			local_stats = self._credit_reclaimable_prefix_pages(
				self._get_host_kv_utilization()
			)
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
			num_gpus_per_node=self.local_world_size,
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
			migrations, self.local_world_size
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

		# Migration reads/writes each cache directly because the coordinator only
		# wraps lifecycle operations.  Registration, allocation, and release still
		# go through that coordinator so DSA cannot split its two page tables.
		lifecycle_view = getattr(self, "host_paged_kv_worker_view", None)
		if lifecycle_view is None:
			lifecycle_view = self.core_engine.host_paged_kv_worker_view
		if isinstance(lifecycle_view, DualHostKVCoordinator):
			worker_view = lifecycle_view.primary
			aux_view = lifecycle_view.auxiliary
		else:
			worker_view = lifecycle_view
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

			# Snapshot the trajectory before releasing any source-side resources.
			# The destination must acknowledge restoring this exact payload before
			# the source allocator gives its pages back.
			if self._traj.trajectory_book is None or not self._traj.trajectory_book.has_sequence(uuid):
				raise QueryBookPoolCapacityError(
					f"Rank {self.rank}: migration source has no trajectory slot for {uuid[:8]}"
				)
			trajectory = self._trajectory_tokens(seq, dtype=torch.int32).contiguous()
			trajectory_slot = self._trajectory_slot(seq)
			turn_payload = self._traj.trajectory_book.turns_payload(trajectory_slot).contiguous()
			header = torch.tensor(
				[1, trajectory.numel(), seq.prompt_length, seq.decoded_length, turn_payload.shape[0]],
				dtype=torch.int64,
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
			dist.send(tensor=header, dst=to_rank, group=gloo_group)
			dist.send(tensor=trajectory, dst=to_rank, group=gloo_group)
			dist.send(tensor=turn_payload, dst=to_rank, group=gloo_group)
			# Free host KV pages only after the complete source payload is sent.
			# The destination restores the trajectory before the migration barrier
			# permits source-slot release.
			lifecycle_view.release_sequence_pages([global_idx])
			self._release_prefix_cache_attachments([global_idx])
			if BATCHGEN_CB_DEBUG:
				logging.debug(
					f"MIGRATION: Rank {self.rank}: Sent trajectory for {uuid[:8]} "
					f"({trajectory.numel()} tokens)"
				)

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
			lifecycle_view.register_sequences([global_idx])
			lifecycle_view.allocate_pages_for_sequences([(global_idx, tokens_needed)])

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

			# Receive and restore the complete trajectory into the reservation
			# admission made for this UUID on every rank of every node. Migration
			# moves token content only; it never binds or releases a reservation,
			# so the allocators of all ranks sharing this node's store stay equal.
			header = torch.empty(5, dtype=torch.int64)
			dist.recv(tensor=header, src=from_rank, group=gloo_group)
			version, trajectory_length, prompt_length, decoded_length, turn_count = [
				int(value) for value in header.tolist()
			]
			if version != 1 or trajectory_length < 0 or turn_count < 0:
				raise RuntimeError(
					f"Rank {self.rank}: invalid trajectory migration header for {uuid[:8]}: "
					f"version={version} length={trajectory_length} turns={turn_count}"
				)
			trajectory_recv = torch.empty(trajectory_length, dtype=torch.int32)
			dist.recv(tensor=trajectory_recv, src=from_rank, group=gloo_group)
			turn_payload = torch.empty((turn_count, 6), dtype=torch.int64)
			dist.recv(tensor=turn_payload, src=from_rank, group=gloo_group)
			turns = QueryBook.turns_from_payload(turn_payload)
			if self._traj.trajectory_book is None:
				raise QueryBookPoolCapacityError(
					f"Rank {self.rank}: trajectory book is unavailable during migration"
				)
			if not self._traj.trajectory_book.has_sequence(uuid):
				raise QueryBookPoolCapacityError(
					f"Rank {self.rank}: migration destination has no admission "
					f"reservation for {uuid[:8]}"
				)
			destination_slot = self._traj.trajectory_book.slot_for(uuid)
			self._traj.trajectory_book.restore(
				destination_slot,
				trajectory_recv,
				prompt_length=prompt_length,
				decoded_length=decoded_length,
				turns=turns,
			)
			seq._buffer_slot = destination_slot

			if not hasattr(self, '_pending_migrated_query_book'):
				self._pending_migrated_query_book = {}
			if not hasattr(self, '_migrated_sequences'):
				self._migrated_sequences = set()
			self._migrated_sequences.add(uuid)
			self._pending_migrated_query_book[uuid] = {
				'trajectory': trajectory_recv,
				'prompt_length': prompt_length,
				'decoded_length': decoded_length,
				'turns': turns,
				'restored': True,
				'slot': destination_slot,
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
		# Trajectory reservations are not touched here: every rank of every node
		# keeps the reservation admission made until completion or rejection
		# releases it everywhere, so re-entry finds it on every rank and the
		# allocators sharing one node store never diverge.
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
				# Dest allocates all host_pages into the sequence's own chain.
				seq.host_owned_pages = mig.host_pages
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
				# unbind drops both directions and returns the slot to the
				# free list for O(1) reuse.
				local_idx = self.seq_table.unbind(uuid)
				if local_idx is not None:
					# A remap can leave `batch` byte-identical while changing
					# which sequence a row resolves to, so row-keyed state must
					# be invalidated here and not only where `batch` is rebound.
					self._bump_decode_row_epoch()
					self._sequences_with_gpu_kv.discard(uuid)
					# Remove query_book entry
					self.query_book.pop(local_idx, None)
					logging.debug(
						f"Rank {self.rank}: [LOCALMAP-POP] migration popped "
						f"{uuid[:8]} (local_idx={local_idx}, from_rank={old_rank}, "
						f"to_rank={new_rank})"
					)

			# Update local mappings on dest rank (add)
			if self.rank == new_rank:
				# O(1) allocation: prefer reusing freed indices, otherwise use
				# next available.
				new_local_idx = self.seq_table.bind(uuid)
				self._bump_decode_row_epoch()
				# Note: Don't add to _sequences_with_gpu_kv - KV is in host, not GPU

				# Trajectory bytes go into the reservation admission made for this
				# UUID; slot numbers are node-local, so look it up by UUID.
				if hasattr(self, '_pending_migrated_query_book') and uuid in self._pending_migrated_query_book:
					pending = self._pending_migrated_query_book.pop(uuid)
					seq = self.global_batch.get_sequence(uuid)
					if self._traj.trajectory_book is None:
						raise QueryBookPoolCapacityError(
							f"Rank {self.rank}: trajectory book unavailable for migration receive {uuid[:8]}"
						)
					if not self._traj.trajectory_book.has_sequence(uuid):
						raise QueryBookPoolCapacityError(
							f"Rank {self.rank}: migration destination has no admission "
							f"reservation for {uuid[:8]}"
						)
					destination_slot = self._traj.trajectory_book.slot_for(uuid)
					logging.info(
						f"Rank {self.rank}: Migration receive {uuid[:8]}: "
						f"restoring slot={destination_slot}, "
						f"tokens={pending['trajectory'].numel()}"
					)
					if not pending.get('restored', False):
						self._traj.trajectory_book.restore(
							destination_slot,
							pending['trajectory'],
							prompt_length=pending['prompt_length'],
							decoded_length=pending['decoded_length'],
							turns=pending.get('turns'),
						)
					seq._buffer_slot = destination_slot
					seq.prompt_length = pending['prompt_length']
					seq.decoded_length = pending['decoded_length']
					seq.current_context_length = (
						seq.original_prompt_length + seq.decoded_length
					)
					self.query_book[new_local_idx] = make_query_book_entry(seq)
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
					'host_owned_pages': seq.host_owned_pages,
					'host_token_capacity': seq.host_token_capacity,
					# G>1: names the serve group, whose leader's Host counters
					# are the canonical ones (host_kv_owner_reports).
					'decode_dp_group': seq.decode_dp_group,
					# QueryBook metadata is process-local even though its token
					# pages are node-shared.  Publish the owner's lengths so
					# non-owning ranks can repair their mirror before re-entry.
					'trajectory_token_length': (
						self._traj.trajectory_book.metadata(self._trajectory_slot(seq)).token_length
						if self._traj.trajectory_book is not None
						and self._traj.trajectory_book.has_sequence(uuid)
						else None
					),
					'trajectory_decoded_length': (
						self._traj.trajectory_book.metadata(self._trajectory_slot(seq)).decoded_length
						if self._traj.trajectory_book is not None
						and self._traj.trajectory_book.has_sequence(uuid)
						else None
					),
					# total_decoded_before_eviction: needed so non-owning ranks
					# sort eviction candidates consistently in _prepare_prefill_batch.
					'total_decoded_before_eviction': seq.total_decoded_before_eviction,
				}
		
		# Step 2: All-gather state from all ranks
		all_states = [None] * self.world_size
		dist.all_gather_object(all_states, local_state)
		
		# Host chain counters have ONE writer per sequence, its host-KV owner.
		# At G>1 every rank of the serve group holds and reports the row, but
		# only the leader deducts attached and retained prefix pages from
		# host_owned_pages, so every other rank -- followers included -- takes
		# the counters from the leader's report, never from whichever group
		# rank merged last. At G==1 the single reporter is the owner.
		host_kv_reports = host_kv_owner_reports(
			all_states, group_size=self._decode_attn_tp_size()
		)

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
							host_state = host_kv_reports[uuid]
							seq.host_pages_allocated = host_state['host_pages_allocated']
							seq.host_owned_pages = host_state['host_owned_pages']
							seq.host_token_capacity = host_state['host_token_capacity']
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
					# Keep the local QueryBook mirror aligned with the owner even
					# when this rank also holds a replicated decode entry.
					seq = self.global_batch.get_sequence(uuid)
					if seq is not None:
						self._sync_trajectory_metadata(seq, state)

		# A G>1 follower holds the row locally, so the merge above skipped it;
		# it still never writes the Host chain, so it adopts the leader's
		# counters here. This is what carries the leader's prefix attach and
		# commit-retain deductions to the followers before the decode boundary
		# requires the group to agree. A no-op on the owner and at G==1.
		for uuid, host_state in host_kv_reports.items():
			if uuid not in self._uuid_to_local_map:
				continue
			seq = self.global_batch.get_sequence(uuid)
			if seq is None or self._owns_host_kv(seq):
				continue
			seq.host_pages_allocated = host_state['host_pages_allocated']
			seq.host_owned_pages = host_state['host_owned_pages']
			seq.host_token_capacity = host_state['host_token_capacity']
			seq.validate_metadata(
				f"rank {self.rank} _sync_sequence_metadata/host_kv_owner"
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

	def _exchange_node_memfd_identity(self, pid: int, fd: int) -> Tuple[int, int]:
		"""Publish this rank's ``input_ids`` memfd identity, take its node's.

		COLLECTIVE, one per pool generation, issued identically by every rank on
		every path. It REPLACES the post-create barrier ``allocate_node_shared_int64``
		used to run: an all_gather cannot deliver on any rank before every rank
		has entered, so the creator's memfd is guaranteed to exist — and to be
		sized — by the time an attacher reads its pid and fd. Non-creators
		contribute ``fd=-1``, which is never selected.
		"""
		# Pool allocation can run inside InferenceMode (see the buffer creation
		# in allocate_node_shared_int64); the collective writes into `gathered`.
		with torch.inference_mode(False):
			identity = torch.tensor(
				[pid, fd], dtype=torch.int64, device=self.torch_device
			)
			gathered = [torch.zeros_like(identity) for _ in range(self.world_size)]
			dist.all_gather(gathered, identity)
			# Pre-`rank_ctx`: this collective runs inside core initialization.
			creator_rank = (
				node_of_rank(self.rank, self.local_world_size) * self.local_world_size
			)
			creator = gathered[creator_rank]
			return int(creator[0].item()), int(creator[1].item())

	# ============ KV-Driven Batch Preparation ============

	def _get_node_for_rank(self, rank: int) -> int:
		"""Delegate: physical host-KV node id of a rank.

		The single definition is `RankContext.node_of`. The old
		`world_size <= local_world_size -> 0` special case is subsumed by it:
		every valid rank is < local_world_size in that configuration, so the
		floor division already yields 0.
		"""
		return self.rank_ctx.node_of(rank)

	def _get_num_nodes(self) -> int:
		"""Delegate: total number of physical host-KV nodes.

		The single definition is `RankContext.num_nodes`.
		"""
		return self.rank_ctx.num_nodes

	def _get_effective_chunk_size(self) -> int:
		"""Return the current host KV chunk size, considering adaptive sizing.

		The value is worker-wide; every consumer bounds it by the sequence's
		own ``kv_token_budget`` (``compute_prefill_host_reservation`` for the
		initial reservation, ``SequenceEntry.get_host_growth_pages`` for
		growth), so a short sequence never reserves past its budget. Capping it
		by the worker-wide ``max_decoding_length`` instead would size every
		later pool-mode batch by the first batch's budget. The result is always
		rounded up to a page boundary (multiple of PAGE_SIZE=64).
		"""
		if self.adaptive_chunk_sizer is not None:
			chunk = self.adaptive_chunk_sizer.get_chunk_size()
		else:
			chunk = self.host_kv_chunk_size
		# Clamp + page rounding: single definition in kv_plan.
		return effective_chunk_size(
			chunk,
			decision_interval=self.DECISION_INTERVAL,
			page_size=SequenceEntry.PAGE_SIZE,
		)

	def _host_capacity_due_local(self, batch: List[int]) -> bool:
		"""Whether a local decode row needs planning before its next forward."""
		for local_idx in batch:
			uuid = self._local_to_uuid_map.get(local_idx)
			if uuid is None:
				continue
			seq = self.global_batch.get_sequence(uuid)
			if seq is None or self._is_sequence_completed(seq):
				continue
			if host_capacity_due(
				int(seq.host_token_capacity), int(seq.current_context_length)
			):
				return True
		return False

	def _host_capacity_due_all_ranks(self, batch: List[int]) -> bool:
		"""Collectively detect a re-entry row with no next-token runway.

		Metadata is owner-local during decode, so a local check cannot decide
		whether all ranks must enter the collective boundary planner.  This check
		runs only at decode entry and immediately after a boundary, never once per
		decode token.
		"""
		local_due = self._host_capacity_due_local(batch)
		if self.world_size <= 1:
			return local_due
		flag = torch.tensor(
			[1 if local_due else 0], dtype=torch.int32, device=self.torch_device
		)
		dist.all_reduce(flag, op=dist.ReduceOp.MAX)
		return bool(int(flag.item()))

	def _prefill_host_reservation_plan(
		self, prefill_uuids: List[str]
	) -> Tuple[Dict[str, Tuple[int, int]], List[Tuple[str, int]]]:
		"""Build the immutable host reservation plan for a prefill wave.

		The selector and allocator share this exact page formula.  The returned
		owner list contains one entry per physical host-KV allocation: the host
		pool is node-shared, so TP replicas must not allocate the same sequence
		more than once.
		"""
		chunk_size = self._get_effective_chunk_size()
		reservations: Dict[str, Tuple[int, int]] = {}
		owner_requests: List[Tuple[str, int]] = []
		for uuid in prefill_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				raise RuntimeError(
					f"Rank {self.rank}: prefill reservation references unknown {uuid}"
				)
			pages, tokens = compute_prefill_host_reservation(
				prompt_length=seq.prompt_length,
				kv_token_budget=seq.kv_token_budget,
				page_size=seq.PAGE_SIZE,
				chunk_size=chunk_size,
				initial_gpu_page_buffer=_sequence_module.INITIAL_GPU_PAGE_BUFFER,
			)
			reservations[uuid] = (pages, tokens)
			if self._owns_host_kv(seq):
				private_tokens = tokens
				state = (
					self._prefix.prefix_sequence_states.get(seq.global_idx)
					if self.enable_prefix_cache
					else None
				)
				if state is not None and state.attached_tokens:
					# Attached prefix pages already sit in the Host pool; the
					# owner allocates only the private remainder. Metadata
					# still commits the TOTAL capacity below. Raising here
					# (before the capacity collective) rejects the wave on
					# every rank.
					private_pages = host_owned_pages_after_sharing(
						chain_pages=pages,
						shared_pages=state.attached_tokens // seq.PAGE_SIZE,
						sequence_id=seq.global_idx,
						where="prefill reservation plan",
					)
					private_tokens = private_pages * seq.PAGE_SIZE
				owner_requests.append((uuid, private_tokens))
		return reservations, owner_requests

	def _collective_prefill_host_capacity_ok(
		self,
		owner_requests: List[Tuple[str, int]],
		local_error: bool = False,
	) -> Tuple[bool, Optional[Dict[int, Tuple[int, int]]]]:
		"""Check a fresh per-node host-KV snapshot before any mutation.

		Every rank reports its owner demand.  Only the node leader reports the
		shared allocator's free/total pages; summing demands across ranks and
		comparing against that one live snapshot prevents one rank from making a
		decision from stale local state.  The final MIN reduction makes the
		decision explicit even when a rank's stats call fails.

		Returns ``(ok, rows)``, where ``rows`` maps node id to the gathered
		``(need, free)`` pair the reserve-time reclaim computes its deficit
		from. ``rows`` is ``None`` when the snapshot itself is unusable (a
		rank's stats call failed, a node reported no capacity at all, or two
		ranks of one node disagreed): a deficit derived from it would be
		meaningless, so the caller must not retry on it. Both verdicts are
		reduced, hence rank-uniform -- the retry branch is collective.
		"""
		# Members' demands exclude their shared chains, but the pooled
		# segments themselves allocate one Host copy (mirrored per group, so
		# one count). Leaving them out over-admits the wave and the
		# allocation fails right after the preflight passed. Only the wave's
		# host-KV LEADER allocates segments, so only a rank with owner demand
		# reports them — at G>1 the followers reporting the same segments
		# would multiply the node demand by G and falsely reject the wave.
		local_need = prefill_owner_demand_pages(
			[int(tokens) for _, tokens in owner_requests],
			page_size=self.PAGE_SIZE,
			pool_segment_pages=(
				self._prefill_pool_segment_host_pages() if owner_requests else 0
			),
		)
		node_id = self.rank_ctx.node_id
		local_free = -1
		local_total = -1
		snapshot_error = 1 if local_error else 0
		host_view = getattr(self, "host_paged_kv_worker_view", None)
		if host_view is None:
			host_view = getattr(
				getattr(self, "core_engine", None),
				"host_paged_kv_worker_view",
				None,
			)
		if self.local_rank == 0:
			try:
				stats = host_view.get_stats()
				local_free = int(stats.num_free_pages)
				local_total = int(stats.num_total_pages)
			except Exception:
				snapshot_error = 1
				logging.exception(
					"Rank %s: host-KV prefill capacity snapshot failed", self.rank
				)

		report = torch.tensor(
			[node_id, local_need, local_free, local_total, snapshot_error],
			dtype=torch.int64,
			device=self.torch_device,
		)
		if self.world_size == 1:
			if snapshot_error:
				return False, None
			ok = 0 <= local_need <= local_free and local_need <= local_total
			return ok, {node_id: (local_need, local_free)}
		if not dist.is_initialized():
			raise RuntimeError(
				"distributed process group is required for host-KV prefill admission"
			)
		gathered = [torch.zeros_like(report) for _ in range(self.world_size)]
		dist.all_gather(gathered, report)

		verdict = verdict_node_capacity(
			[
				NodeCapacityReport(
					node_id=int(item[0].item()),
					need=int(item[1].item()),
					free=int(item[2].item()),
					total=int(item[3].item()),
					snapshot_error=int(item[4].item()),
				)
				for item in gathered
			],
			want_rows=True,
		)

		decision = torch.tensor(
			[1 if verdict.ok else 0, 1 if verdict.rows_usable else 0],
			dtype=torch.int32, device=self.torch_device,
		)
		dist.all_reduce(decision, op=dist.ReduceOp.MIN)
		if not int(decision[1].item()):
			return bool(int(decision[0].item())), None
		# The reduced rows_usable is the MIN over ranks, so it is 1 only when
		# this rank's local verdict was usable and carries its rows.
		return bool(int(decision[0].item())), verdict.rows

	def _reclaim_host_prefix_pages_collective(
		self,
		page_target: Callable[[], int],
		*,
		reason: str,
	) -> int:
		"""Reclaim retained prefix pages on each node leader; all ranks align.

		The node leader evaluates ``page_target()`` (the Host pages this node is
		short) and releases up to that many cold retained prefix pages -- only
		pages with no live attachment and no pending load, so nothing a running
		row or an attached wave reads is touched. Every other rank of the node
		then retires the trajectory frontiers standing on the released pages.

		Collective discipline: every rank reaches every collective, a
		leader-side failure (including in ``page_target``) becomes a MAX-reduced
		flag, and no rank raises before that flag is reduced. ``reason`` names
		the caller in logs, the ``prefix_reclaim`` metric and errors. Returns
		the largest page count any leader reclaimed.
		"""
		from batchgen.prefix_reuse.eviction import (
			EvictedPageRecorder,
			reclaim_prefix_pages_for_host_admission,
			retire_node_frontiers,
		)
		node_id = self.rank_ctx.node_id
		failure = 0
		local_frontier_hook = self._trajectory_frontier_eviction_hook()
		reclaimed_pages = EvictedPageRecorder(local_frontier_hook)
		if self.local_rank == 0:
			try:
				target = int(page_target())
				if target:
					released = reclaim_prefix_pages_for_host_admission(
						core_engine_module=core_engine,
						coordinator=self.prefix_cache_coordinator,
						worker_views_by_group=self._host_views_by_group(),
						group_id=0,
						page_target=target,
						max_scan_nodes=self.prefix_cache_runtime_config.max_nodes,
						frontier_hook=reclaimed_pages,
					)
					logging.info(
						"[PREFIX_CACHE] %s reclaimed %s Host pages on node %s "
						"(free=%s, target=%s)",
						reason, released, node_id, self._get_host_kv_free_pages(),
						target,
					)
					if self.prefix_cache_debug_stats:
						self._emit_prefix_cache_metric(
							"prefix_reclaim",
							reason=reason,
							node=int(node_id),
							page_target=int(target),
							released_pages=int(released),
							**self._prefix_host_resource_fields(),
						)
			except Exception:
				logging.exception("[PREFIX_CACHE] %s eviction failed", reason)
				failure = 1
		reclaimed_page_count = reclaimed_pages.page_count
		if self.world_size > 1:
			failed = torch.tensor(
				[failure, reclaimed_page_count],
				dtype=torch.int64, device=self.torch_device,
			)
			dist.all_reduce(failed, op=dist.ReduceOp.MAX)
			failure = int(failed[0].item())
			reclaimed_page_count = int(failed[1].item())
		if failure != 0:
			raise RuntimeError(f"prefix cache {reason} eviction failed on a worker")
		# The leader already retired its own frontiers inside the release. Any
		# other rank of the node may own frontier records standing on the
		# reclaimed pages (G=1, or several serve groups per node), so every rank
		# runs its own hook on its node's reclaimed map.
		if self.world_size > 1 and reclaimed_page_count != 0:
			frontier_failure = 0
			gathered_reclaims = [None] * self.world_size
			dist.all_gather_object(
				gathered_reclaims,
				(
					node_id,
					reclaimed_pages.pages_by_group
					if self.local_rank == 0 else None,
				),
			)
			if self.local_rank != 0:
				try:
					retire_node_frontiers(
						gathered=gathered_reclaims,
						node_id=node_id,
						frontier_hook=local_frontier_hook,
					)
				except Exception:
					logging.exception(
						"[PREFIX_CACHE] frontier retirement after %s reclaim "
						"failed", reason,
					)
					frontier_failure = 1
			retired = torch.tensor(
				[frontier_failure], dtype=torch.int64, device=self.torch_device
			)
			dist.all_reduce(retired, op=dist.ReduceOp.MAX)
			if int(retired.item()) != 0:
				raise RuntimeError(
					f"trajectory frontier retirement after {reason} reclaim "
					"failed on a worker"
				)
		return reclaimed_page_count

	def _retry_prefill_host_capacity_after_reclaim(
		self,
		owner_requests: List[Tuple[str, int]],
		capacity_rows: Dict[int, Tuple[int, int]],
	) -> bool:
		"""Reclaim this node's host-page deficit once, then re-check capacity.

		Exactly one attempt: a wave sized against free + reclaimable pages can
		be short only by what is retained, and this releases it. A second
		shortfall is a real capacity change (a concurrent decode grew, a peer
		node is the short one and has nothing cold left), which the caller
		rejects exactly as before.

		Safe by construction: the wave's prefix attachments are already held,
		so every prefix this wave HIT is pinned and ``NodeIsProtectedLocked``
		keeps it -- only cold retained pages go, which is precisely the supply
		the admission budget counted.

		Collective discipline is ``_reclaim_host_prefix_pages_collective``'s:
		every rank reaches every collective, and the deficit is computed on the
		leader inside its MAX-reduced failure flag.
		"""
		node_id = node_of_rank(self.rank, self.local_world_size)
		self._reclaim_host_prefix_pages_collective(
			lambda: prefill_host_reserve_page_deficit(capacity_rows, node_id),
			reason="host_reserve",
		)
		capacity_ok, _ = self._collective_prefill_host_capacity_ok(owner_requests)
		return capacity_ok

	def _reclaim_host_prefix_pages_for_growth(
		self,
		growth_entries: List[Tuple[str, int, object]],
		worker_view: Optional[object],
	) -> int:
		"""Release the retained prefix pages this boundary's growth plan counted.

		The rank-0 growth plan covers each node's growth debt with free pages,
		completed/evicted releases and the node's reclaimable retained prefix
		pages. After every rank has released (the caller's barrier), the node
		leader reads the physical free count and reclaims exactly the part of
		this node's growth the free pages do not cover:
		``max(0, node_growth_pages - free_after_releases)``. A shortfall the
		reclaim cannot close is left to the fresh collective preflight, which
		fails loud. ``growth_entries`` is the broadcast ``(uuid, pages, seq)``
		list, identical on every rank.
		"""
		node_id = node_of_rank(self.rank, self.local_world_size)
		group_size = self._decode_attn_tp_size()

		def node_growth_deficit() -> int:
			from batchgen.decode_dp_group import host_kv_owner_rank

			node_growth_pages = 0
			for uuid, growth_pages, seq in growth_entries:
				if seq is None:
					raise RuntimeError(f"growth sequence {uuid} is not in global_batch")
				if group_size > 1:
					if seq.decode_dp_group is None:
						raise RuntimeError(
							f"growth sequence {uuid} has no decode_dp_group for "
							f"attn_tp_size={group_size}"
						)
					owner = host_kv_owner_rank(seq.decode_dp_group, group_size)
				else:
					owner = seq.assigned_rank
				if node_of_rank(int(owner), self.local_world_size) == node_id:
					node_growth_pages += int(growth_pages)
			if node_growth_pages == 0:
				return 0
			if worker_view is None:
				raise RuntimeError("host-KV worker view is unavailable")
			free_after_releases = int(worker_view.get_stats().num_free_pages)
			return max(0, node_growth_pages - free_after_releases)

		return self._reclaim_host_prefix_pages_collective(
			node_growth_deficit, reason="host_growth",
		)

	def _attach_prefix_cache_for_prefill(self, prefill_uuids: List[str]) -> None:
		"""Attach page-aligned cached prefixes for sequences this rank owns.

		Runs before the reservation plan so attached pages reduce the owner's
		private demand. The attachments participate in the wave's rollback:
		both the fresh-capacity rejection and the allocation rollback release
		them through ``_release_prefix_cache_attachments``. Prompts are read
		from the unified trajectory book (``seq.input_ids`` is retired).
		"""
		from batchgen.prefix_reuse.prefill import (
			PrefixCacheSequenceState,
			lookup_prefix_cache_for_prefill,
		)

		# Every rank that SERVES a sequence plans the wave (the G>1 plan is a
		# group collective); only the host-KV LEADER looks up and attaches.
		served = [
			uuid for uuid in prefill_uuids
			if self._owns_local_sequence(self.global_batch.get_sequence(uuid))
		]
		owned = [
			uuid for uuid in served
			if self._owns_host_kv(self.global_batch.get_sequence(uuid))
		]
		if not served:
			return
		prefix_states = {}
		if owned:
			prompt_token_ids = []
			for uuid in owned:
				seq = self.global_batch.get_sequence(uuid)
				prompt_token_ids.append(
					self._trajectory_tokens(seq, length=seq.prompt_length).tolist()
				)
			lookup = lookup_prefix_cache_for_prefill(
				coordinator=self.prefix_cache_coordinator,
				namespace_digest=self.prefix_cache_runtime_config.namespace_digest,
				prompt_token_ids=prompt_token_ids,
				page_size_tokens=(
					self.prefix_cache_runtime_config.group_specs[0].raw_page_tokens
				),
				num_groups=len(self.prefix_cache_runtime_config.group_specs),
			)
			for uuid, result, attached, compute_cached in zip(
				owned,
				lookup.lookup_results,
				lookup.attached_tokens,
				lookup.compute_cached_tokens,
			):
				seq = self.global_batch.get_sequence(uuid)
				state = PrefixCacheSequenceState(
					lookup_result=result,
					attached_tokens=int(attached),
					compute_cached_tokens=int(compute_cached),
				)
				self._prefix.prefix_sequence_states[seq.global_idx] = state
				prefix_states[seq.global_idx] = state
		# M3 Slice 2, SAME batch-level flag as the commit side: one flag
		# governs the whole trajectory path, so lookups cannot be on while
		# commits are off. With it off nothing here runs and no attachment is
		# taken, so the restore seam stays empty and this wave is
		# byte-identical to a pre-Slice-2 one.
		#
		# Slice 3 moved this IN FRONT of the pool plan: a hit whose prefix the
		# wave materializes is a plan input. The reference this lookup takes
		# keeps the frontier's snapshot and registry entry from the plan to the
		# restore; the MLA pages the load reads are kept resident by the row's
		# prefix-cache attachment above, which the plan retains for the wave.
		if owned and self._trajectory_commit_requested(
			[self.global_batch.get_sequence(uuid) for uuid in served]
		):
			self._lookup_trajectory_frontiers(owned)
		# In-wave pool: plan this rank's shared segments from the same wave.
		# The plan only reads tokens and states; Host pages for the segments
		# are allocated inside the reservation transaction below.
		self._prefill_pool_wave = None
		if self._prefill_prefix_pool_enabled():
			self._prefill_pool_wave = self._plan_prefill_pool_wave(
				served, prefix_states
			)
			if self._prefill_pool_wave is None:
				from batchgen.prefix_reuse.config import (
					prefix_reuse_requires_prefill_pool,
					require_prefix_cache_model_support,
				)
				if prefix_reuse_requires_prefill_pool(
					require_prefix_cache_model_support(self.huggingface_ckpt_name)
				):
					# A pool-only model's hit (DSA dual-group, MLA-latent K3)
					# has no cross-wave materialization path; a wave the pool
					# declines recomputes, so its hits must not stay attached.
					# Full-K/V waves keep theirs for attach+suffix.
					seqs = [self.global_batch.get_sequence(uuid) for uuid in served]
					self._replace_prefix_lookup_hits(
						seqs, prefix_states, [0] * len(seqs)
					)
		else:
			# No pool, no plan: nothing materializes a frontier, so no row of
			# this wave restores. Declined here, at plan time, like every
			# other attachment the plan does not take.
			self._settle_trajectory_restore_plan(
				[self.global_batch.get_sequence(uuid) for uuid in served],
				{},
				"the prefill prefix pool is off, so no plan materializes a "
				"frontier",
			)

	def _reserve_prefill_host_kv(self, prefill_uuids: List[str]) -> bool:
		"""Atomically reserve host pages before IN_PREFILL is published.

		Selection is only a snapshot.  This second, immediate snapshot catches a
		capacity change before status, trajectory metadata, or local QueryBook
		bindings are mutated.  Allocator exceptions are converted into a
		collective failure and successful peers release their tentative pages.
		"""
		self._prefill_pool_wave = None
		wave_global_ids = [
			seq.global_idx
			for seq in (
				self.global_batch.get_sequence(uuid) for uuid in prefill_uuids
			)
			if seq is not None
		]
		reservations: Dict[str, Tuple[int, int]] = {}
		owner_requests: List[Tuple[str, int]] = []
		plan_error = False
		try:
			if self.enable_prefix_cache and self.prefix_cache_coordinator is not None:
				self._attach_prefix_cache_for_prefill(prefill_uuids)
			reservations, owner_requests = self._prefill_host_reservation_plan(
				prefill_uuids
			)
		except Exception:
			# Every rank must still reach the capacity collective below, or the
			# peers wait in all_gather forever while this rank raises.
			plan_error = True
			logging.exception(
				"Rank %s: prefix lookup or reservation plan failed before the "
				"host-KV capacity collective",
				self.rank,
			)
		capacity_ok, capacity_rows = self._collective_prefill_host_capacity_ok(
			owner_requests, local_error=plan_error
		)
		if plan_error:
			# A planned wave never allocated: dropping it frees its GPU pool.
			self._prefill_pool_wave = None
			self._release_prefix_cache_attachments(wave_global_ids)
			raise RuntimeError(
				f"Rank {self.rank}: host-KV prefill plan failed; the wave was "
				"rejected collectively before any allocator mutation"
			)
		if (
			not capacity_ok and capacity_rows is not None
			and self.enable_prefix_cache
		):
			# Selection sized this wave against free PLUS reclaimable pages, so
			# a shortfall here is the part of the budget that is still retained
			# rather than free. Release it once and re-decide, instead of
			# rejecting a wave the budget was right about (and re-selecting the
			# same wave on the next iteration). Entered on the rank-uniform
			# (verdict, rows) pair, so every rank reaches the retry's
			# collectives and the second capacity collective.
			capacity_ok = self._retry_prefill_host_capacity_after_reclaim(
				owner_requests, capacity_rows
			)
		if not capacity_ok:
			# Record the KIND of outcome (never a counter) for the no-progress
			# guard: a rejected transaction is a different stall cause from a
			# wave that selected nothing, so it must not share their streak --
			# but a rejection that repeats with everything else unchanged is
			# deterministic, not contention, and must still reach the limit.
			self._prefill_reservation_outcome = "rejected_by_capacity_snapshot"
			self._suspend_shared_prefix_credit("rejected_by_capacity_snapshot")
			if self.rank == 0:
				logging.warning(
					"[PREFILL] Host KV admission rejected by fresh capacity snapshot; "
					"leaving candidates queueing/evicted"
				)
			# The rejected wave stays QUEUEING/EVICTED; its lookup attachments
			# must not pin prefix pages past this decision, and the planned
			# wave (GPU pool included) must not outlive the rejection.
			self._prefill_pool_wave = None
			self._release_prefix_cache_attachments(wave_global_ids)
			return False

		host_view = getattr(self, "host_paged_kv_worker_view", None)
		if host_view is None:
			host_view = getattr(
				getattr(self, "core_engine", None),
				"host_paged_kv_worker_view",
				None,
			)
		global_ids = [
			self.global_batch.get_sequence(uuid).global_idx
			for uuid, _ in owner_requests
		]
		allocation_requests = [
			(self.global_batch.get_sequence(uuid).global_idx, tokens)
			for uuid, tokens in owner_requests
			if int(tokens) > 0
		]
		local_success = True
		registered = bool(owner_requests)
		try:
			if owner_requests:
				host_view.register_sequences(global_ids)
				if self._prefill_pool_wave is not None:
					# One Host copy per pooled segment and per KV group;
					# members reference the segment chains. Segment
					# pseudo-sids join this rollback.
					self._allocate_prefill_pool_host_pages(
						self._host_views_by_group(),
						{
							self.global_batch.get_sequence(uuid).global_idx:
							self._prefix.prefix_sequence_states[
								self.global_batch.get_sequence(uuid).global_idx
							]
							for uuid, _ in owner_requests
							if self.global_batch.get_sequence(uuid).global_idx
							in self._prefix.prefix_sequence_states
						},
					)
				if self.enable_prefix_cache:
					# Shared prefix pages join the sequence's page table before
					# private allocation so release_sequence_pages covers both.
					# A group's shared pages live in that group's own region.
					host_views = self._host_views_by_group()
					for uuid, _ in owner_requests:
						seq = self.global_batch.get_sequence(uuid)
						state = self._prefix.prefix_sequence_states.get(seq.global_idx)
						if state is None:
							continue
						for group_id, view in sorted(host_views.items()):
							shared = state.shared_page_ids_for_group(group_id)
							if shared:
								view.attach_shared_prefix_pages(
									seq.global_idx, list(shared)
								)
				if allocation_requests:
					host_view.allocate_pages_for_sequences(allocation_requests)
		except Exception:
			local_success = False
			logging.exception(
				"Rank %s: host-KV prefill allocation failed after collective preflight",
				self.rank,
			)

		if self.world_size > 1:
			allocation = torch.tensor(
				[1 if local_success else 0],
				dtype=torch.int32,
				device=self.torch_device,
			)
			dist.all_reduce(allocation, op=dist.ReduceOp.MIN)
			all_success = bool(int(allocation.item()))
		else:
			all_success = local_success

		if not all_success:
			if registered:
				try:
					host_view.release_sequence_pages(global_ids)
				except Exception:
					logging.exception(
						"Rank %s: failed to rollback host-KV prefill pages",
						self.rank,
					)
				try:
					host_view.unregister_sequences(global_ids)
				except Exception:
					logging.exception(
						"Rank %s: failed to rollback host-KV prefill registrations",
						self.rank,
					)
			if self._prefill_pool_wave is not None:
				wave = self._prefill_pool_wave
				# Members release first (their rows merely reference segment
				# pages), then each pseudo-sid frees and unregisters. Pop only
				# THIS wave's entries: earlier waves' members still hold refs.
				for gid in wave.global_ids:
					self._prefix.pool_member_sids.pop(gid, None)
				for sid in set(wave.segment_sids.values()):
					self._prefix.pool_sid_members.pop(sid, None)
					try:
						host_view.release_sequence_pages([sid])
					except Exception:
						pass
					try:
						host_view.unregister_sequences([sid])
					except Exception:
						pass
				self._prefill_pool_wave = None
			self._release_prefix_cache_attachments(wave_global_ids)
			self._prefill_reservation_outcome = "rolled_back_after_allocator_failure"
			self._suspend_shared_prefix_credit(
				"rolled_back_after_allocator_failure"
			)
			return False

		# Commit scalar reservation metadata only after every allocator agrees.
		for uuid, (pages, tokens) in reservations.items():
			seq = self.global_batch.get_sequence(uuid)
			seq.host_pages_allocated = pages
			seq.host_token_capacity = tokens
			# Attached prefix pages never enter the sequence's own chain, so
			# releasing the row returns only the private remainder. Only the
			# host-KV owner holds the prefix state here; every other rank (G>1
			# followers included) takes the owner's value at the post-prefill
			# _sync_sequence_metadata.
			state = (
				self._prefix.prefix_sequence_states.get(seq.global_idx)
				if self.enable_prefix_cache
				else None
			)
			attached_pages = (
				state.attached_tokens // seq.PAGE_SIZE if state is not None else 0
			)
			seq.host_owned_pages = host_owned_pages_after_sharing(
				chain_pages=pages,
				shared_pages=attached_pages,
				sequence_id=seq.global_idx,
				where="prefill reservation commit",
			)
		self._prefill_reservation_outcome = "committed"
		# A committed wave proves the credit (if any) matched what the
		# allocator consumed, so the next round may credit again.
		self._prefix.shared_prefix_credit_suspended = False
		return True

	def _collective_host_kv_growth_capacity_ok(
		self,
		host_grow_requests: List[Tuple[int, int]],
		worker_view: Optional[object] = None,
	) -> bool:
		"""Preflight one boundary's shared host-KV growth transaction.

		The boundary planner runs from a gathered snapshot taken before completed
		and evicted rows release their pages.  That snapshot is intentionally
		pure, but it cannot prove that the shared allocator still has the pages
		when Phase 4.C executes.  Every rank therefore reports its owner demand;
		only one rank per node reports the node-shared allocator's fresh free/total
		counts.  The all-rank MIN makes the decision identical before any view is
		mutated.

		The second all-rank consensus is performed by the caller after the local
		allocator call.  A post-preflight allocator failure is fatal: the native
		worker-view API has no page-level rollback for a successful peer, and
		continuing would make scalar metadata and shared page tables disagree.
		"""
		local_need = sum(int(pages) for _, pages in host_grow_requests)
		node_id = self.rank_ctx.node_id
		local_free = -1
		local_total = -1
		snapshot_error = 0
		if worker_view is None:
			worker_view = getattr(self, "host_paged_kv_worker_view", None)
		if worker_view is None:
			worker_view = getattr(
				getattr(self, "core_engine", None),
				"host_paged_kv_worker_view",
				None,
			)
		# The host region is shared by all ranks on a node.  Use the local
		# representative for the snapshot, matching _gather_host_kv_stats_by_node
		# and _collective_prefill_host_capacity_ok.
		if self.local_rank == 0:
			try:
				if worker_view is None:
					raise RuntimeError("host-KV worker view is unavailable")
				stats = worker_view.get_stats()
				local_free = int(stats.num_free_pages)
				local_total = int(stats.num_total_pages)
			except Exception:
				snapshot_error = 1
				logging.exception(
					"Rank %s: host-KV boundary growth snapshot failed", self.rank
				)

		report = torch.tensor(
			[node_id, local_need, local_free, local_total, snapshot_error],
			dtype=torch.int64,
			device=self.torch_device,
		)
		if self.world_size == 1:
			if snapshot_error:
				return False
			return 0 <= local_need <= local_free and local_need <= local_total
		if not dist.is_initialized():
			raise RuntimeError(
				"distributed process group is required for host-KV growth preflight"
			)
		gathered = [torch.zeros_like(report) for _ in range(self.world_size)]
		dist.all_gather(gathered, report)

		verdict = verdict_node_capacity(
			[
				NodeCapacityReport(
					node_id=int(item[0].item()),
					need=int(item[1].item()),
					free=int(item[2].item()),
					total=int(item[3].item()),
					snapshot_error=int(item[4].item()),
				)
				for item in gathered
			],
			want_rows=False,
		)

		decision = torch.tensor(
			[1 if verdict.ok else 0], dtype=torch.int32, device=self.torch_device
		)
		dist.all_reduce(decision, op=dist.ReduceOp.MIN)
		return bool(int(decision.item()))

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
		self._prefill_admission_state = None
		self._prefill_admission_selection = None
		self._prefill_admission_totals = None
		self._prefill_admission_live = False
		# Reset here, not at the reservation: _prepare_prefill_batch always runs
		# before _reserve_prefill_host_kv in the scheduler loop, so an iteration
		# that selects nothing reports "not attempted" rather than carrying the
		# previous iteration's verdict into the guard.
		self._prefill_reservation_outcome = None
		if not all_candidates:
			return []

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

		# This iteration's live set, reported into the gather below so the
		# reclaim branch and the scheduler loop's no-progress guard both decide
		# on the OR across ranks. Read here because nothing between this point
		# and the guard changes a sequence status -- selection, admission
		# reclaim and the Host reservation do not -- while a rank-LOCAL read
		# lets ranks raise on different iterations, i.e. diverge on the very
		# failure path the guard exists to make clean.
		local_live = (
			self.global_batch.has_prefilled()
			or self.global_batch.has_in_decode()
			or self.global_batch.has_on_hold()
		)
		# Admission reclaim and the guard both branch on the candidate snapshot
		# each rank builds from its own global_batch. Carry a digest of the
		# fields that snapshot is derived from, so a divergence is named here
		# instead of sending one rank into a reclaim collective the others
		# never enter (same pattern as _check_pool_plan_identical).
		candidate_digest = self._prefill_candidate_digest(all_candidates)
		# Prefix-cache pressure moves while a prefix load or an attachment
		# release is in flight -- an attachment release in particular makes
		# pages reclaimable WITHOUT changing the free count -- which is what
		# separates "waiting on something" from "stuck" for the guard. Reading
		# it scans the coordinator's whole shared node table under its lock
		# (max_nodes tracks pool pages), so it is taken only when this node has
		# nothing live, the sole case where the guard can fire: a busy worker
		# pays nothing. Node-leader only, like the pool stats, and every rank of
		# the node uses the leader's gathered value, so gating on the leader's
		# own live set keeps the component rank-uniform.
		prefix_pressure = (
			self._prefill_prefix_pressure()
			if self.local_rank == 0 and not local_live
			else (-1, -1, -1)
		)
		# Retained prefix pages this node can reclaim right now (resident, no
		# attachment, no pending load). They join the admission budget below: a
		# retained prefix either gets attached at config time, dropping the
		# wave's demand and the evictable supply by the same page count, or
		# stays evictable for the reserve-time reclaim to release. Either way
		# demand <= free + reclaimable holds, so full-charge stays. Node-leader
		# only (one shared region per node); one O(max_nodes) scan under the
		# coordinator lock, same order as the pressure scan above, and unlike it
		# taken on every iteration -- which is why it mutates nothing.
		report_releasable = (
			self.prefix_cache_coordinator.count_releasable_pages(0)
			if (
				self.local_rank == 0
				and self.enable_prefix_cache
				and self.prefix_cache_coordinator is not None
			)
			else -1
		)

		# Step 2: Gather host KV plus persistent sequence capacity. Host KV is
		# one shared region per node, so only the node leader reports it --
		# free AND total pages, because the total is what makes "this request
		# can never fit the pool" decidable identically on every rank. KDA
		# state is per GPU; every rank reports its free count and the scheduler
		# uses the node minimum. This keeps every rank's selection identical
		# even if a prior lifecycle bug left one TP group asymmetric.
		report_node = self.rank_ctx.node_of(self.rank)
		report_host_free = local_host_free if self.local_rank == 0 else -1
		report_host_total = (
			self._get_host_kv_total_pages() if self.local_rank == 0 else -1
		)
		# One row per rank; -1 marks "not reported by this rank":
		#   [0] node id of this rank
		#   [1] Host KV free pages (node leader only)
		#   [2] sequence-slot limit scope: 0 none, 1 per rank, 2 per node
		#   [3] free sequence slots in that scope (-1 when unlimited)
		#   [4] Host KV total pages (node leader only): decides "can never
		#       fit" for the per-request rejection and the stall verdict
		#   [5] live bit: this rank has a prefilled/decoding/on-hold sequence;
		#       OR-reduced for the reclaim stall check and the progress guard
		#   [6] candidate digest (_prefill_candidate_digest); must be equal on
		#       every rank or PrefillAdmissionStateDivergence is raised
		#   [7..9] prefix-cache pressure of the node (resident nodes, active
		#       attachments, pending loads; node leader with nothing live
		#       only), so the guard tells an in-flight wait from a stall
		#   [10] reclaimable retained prefix pages of the node (node leader
		#       with the prefix cache only): joins the host admission budget
		free_tensor = torch.tensor(
			[
				report_node, report_host_free, limit_scope, local_sequence_free,
				report_host_total, int(local_live), candidate_digest,
				prefix_pressure[0], prefix_pressure[1], prefix_pressure[2],
				report_releasable,
			],
			dtype=torch.int64,
			device=self.torch_device,
		)
		gathered = [torch.zeros_like(free_tensor) for _ in range(self.world_size)]
		dist.all_gather(gathered, free_tensor)

		gathered_digests = {int(item[6].item()) for item in gathered}
		if len(gathered_digests) != 1:
			raise PrefillAdmissionStateDivergence(
				f"prefill admission candidates diverged across ranks: rank "
				f"{self.rank} digest {candidate_digest} over "
				f"{len(all_candidates)} candidates; gathered "
				f"{sorted(gathered_digests)}"
			)
		# Rank-uniform liveness. OR, not AND: a false "live" only postpones the
		# guard, a false "not live" fails a worker that still had work to do.
		self._prefill_admission_live = any(
			int(item[5].item()) != 0 for item in gathered
		)

		# Extract per-node host KV free/total pages and prefix pressure
		reports_by_node = {}
		totals_by_node = {}
		pressure_by_node = {}
		reclaimable_by_node = {}
		for item in gathered:
			node_id = int(item[0].item())
			host_free = int(item[1].item())
			if node_id >= 0 and host_free >= 0:
				reports_by_node[node_id] = host_free
				totals_by_node[node_id] = int(item[4].item())
				pressure_by_node[node_id] = (
					int(item[7].item()), int(item[8].item()), int(item[9].item())
				)
				# -1 is "this leader runs no prefix cache", i.e. no reclaimable
				# supply -- never a negative budget contribution.
				reclaimable_by_node[node_id] = max(0, int(item[10].item()))
		per_node_host_free = []
		per_node_host_total = []
		per_node_prefix_pressure = []
		per_node_host_reclaimable = []
		for node in range(num_nodes):
			per_node_host_free.append(reports_by_node.get(node, 0))
			per_node_host_reclaimable.append(reclaimable_by_node.get(node, 0))
			# -1 means "no leader report": an unknown total must not be read as
			# a zero-page pool that can never fit anything.
			per_node_host_total.append(totals_by_node.get(node, -1))
			per_node_prefix_pressure.append(
				pressure_by_node.get(node, (-1, -1, -1))
			)
		# Pool sizes for the stall report. Gathered, so a multi-node stall can
		# name "can never fit" too, where the old rank-local read could not.
		self._prefill_admission_totals = tuple(per_node_host_total)

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

		# Under G>1 the selection request maps each candidate to a node via its
		# decode_dp_group, so every candidate must already carry one. Newly
		# queued sequences are grouped later (at the decode transition), so a
		# wave mixing queued with evicted candidates would raise "sequence ...
		# has no decode_dp_group before TP prefill selection". Stamp them here:
		# _assign_decode_dp_groups is documented to run at admission/prefill
		# config, is idempotent (re-entrant sequences keep the group their
		# head-sharded KDA state lives on), and is deterministic across ranks.
		self._assign_decode_dp_groups(all_candidates)
		# The map decides this wave's node mapping (below), its bindings and
		# its KDA state, so verify agreement BEFORE any of them is derived:
		# _prepare_prefill_batch is already collective, and this is the first
		# all-ranks point after the stamp.
		self._check_decode_dp_group_plan_identical(all_candidates)

		# Step 3: Select sequences considering per-node host KV capacity.
		# The NCCL gather above and the logging below stay here; the greedy
		# per-node admission is delegated to PrefillScheduler. Kimi-Linear
		# additionally supplies a persistent KDA-state limit: unlike the token
		# cap used later by prepack, a KDA slot remains occupied until the
		# sequence completes or is evicted.
		selection = self._make_prefill_selection_request(
			all_candidates,
			per_node_host_free,
			num_nodes,
			chunk_size,
			per_rank_sequence_free=per_rank_sequence_free,
			per_node_sequence_free=per_node_sequence_free,
			per_node_host_reclaimable=per_node_host_reclaimable,
		)
		# One reservation per candidate, reused by the rejection below, the
		# reclaim target and the guard state. Independent of free pages, so it
		# survives the post-reclaim `replace(selection, ...)`.
		required_pages = {
			candidate.uuid: PrefillScheduler.required_host_pages(
				candidate, selection
			)
			for candidate in selection.candidates
		}
		# A request whose initial Host reservation exceeds its node's WHOLE pool
		# can never be admitted -- not by this wave, not by any later one, no
		# matter how much is reclaimed. Fail that REQUEST, not the worker: the
		# stall exceptions below leave _prepare_prefill_batch for
		# generate_persistent(), which has no handler for them, so they reach
		# the worker fatal boundary and take every rank down for one oversized
		# prompt. Rank-uniform: per_node_host_total is gathered and the
		# candidate snapshot was digest-checked above, so every rank rejects the
		# same uuids with no further collective.
		if self._reject_unsatisfiable_prefill_requests(
			selection, per_node_host_total, required_pages
		):
			# The candidate set changed under the snapshot. Rebuild it on the
			# next iteration instead of selecting against removed sequences; the
			# cleared state also resets the no-progress streak, which is correct
			# -- this iteration did make progress.
			self._prefill_admission_state = None
			self._prefill_admission_selection = None
			return []
		prefill_batch = PrefillScheduler.select_prefill_batch(selection)
		# The in-wave prefix pool releases every cross-wave lookup hit and
		# recomputes it (_plan_prefill_pool_wave), so a hit saves no Host pages
		# there: discounting it over-admits, and the allocation fails once the
		# retained prefixes are all pinned by live sequences. Those retained
		# pages are reclaimable instead.
		pool_mode = self._prefill_prefix_pool_enabled()
		if (
			self.enable_prefix_cache and selection.candidates and not pool_mode
			and per_rank_sequence_free is None
			and per_node_sequence_free is None
		):
			# A cached prefix already occupies Host pages, so charging a hit its
			# full reservation lets one full-cost prompt fill the wave while
			# cheap hits wait (one sequence per wave under tight Host KV).
			# Estimate on each node leader and select with the discount whenever
			# any candidate has a hit; the actual lookup and attach still happen
			# in _config_prefill_for_batch. EVICTED prompts are reconstructed
			# there, so retain their existing full-page path.
			from batchgen.prefix_reuse.prefill import (
				estimate_prefix_cached_pages_for_prefill,
			)
			estimated_values = [0] * (len(selection.candidates) + 1)
			if self.local_rank == 0:
				try:
					for index, candidate in enumerate(selection.candidates):
						if candidate.node_id != report_node or candidate.is_evicted:
							continue
						seq = self.global_batch.get_sequence(candidate.uuid)
						estimated_values[index] = estimate_prefix_cached_pages_for_prefill(
							coordinator=self.prefix_cache_coordinator,
							namespace_digest=(
								self.prefix_cache_runtime_config.namespace_digest
							),
							prompt_token_ids=self._trajectory_tokens(
								seq, length=seq.prompt_length
							).tolist(),
							page_size_tokens=(
								self.prefix_cache_runtime_config.group_specs[0]
								.raw_page_tokens
							),
						)
				except Exception:
					logging.exception("[PREFIX_CACHE] admission estimate failed")
					estimated_values[-1] = 1
			estimated = torch.tensor(
				estimated_values, dtype=torch.int64,
				device=self.torch_device,
			)
			dist.all_reduce(estimated, op=dist.ReduceOp.MAX)
			if int(estimated[-1].item()):
				raise RuntimeError("prefix cache admission estimate failed on a worker")
			cached_pages = estimated[:-1].tolist()
			if any(cached_pages):
				estimated_selection = replace(
					selection,
					candidates=tuple(
						replace(candidate, cached_prefix_pages=int(cached_pages[index]))
						for index, candidate in enumerate(selection.candidates)
					),
				)
				prefill_batch = PrefillScheduler.select_prefill_batch(
					estimated_selection
				)
		# Admission reclaim targets what the persistent sequence-slot limits
		# (Kimi-K3/Kimi-Linear KDA state) would admit with a fully reclaimed
		# Host pool. Without slot limits that is every candidate, i.e. the
		# previous trigger. A slot-limited model used to be excluded outright,
		# so retained prefixes filled its Host pool and every later wave
		# selected zero sequences forever with nothing ever evicted.
		slot_admissible = PrefillScheduler.slot_admissible_candidates(selection)
		# Split the slot-admissible set into what this node's pool can hold at all
		# and what it can never hold, from the GATHERED totals rather than a
		# rank-local stats read, so every rank enters the branch below -- and
		# raises its stall -- on one shared verdict. A node whose leader reported
		# no total (-1) counts as "unknown", never as a zero-page pool.
		def _fits(candidate) -> bool:
			total = per_node_host_total[candidate.node_id]
			return total < 0 or required_pages[candidate.uuid] <= total

		# The reclaim has to buy room for the next candidate the pool CANNOT
		# take yet, so exclude the ones this wave already admitted: in pool mode
		# the trigger fires while part of the set is in, and targeting the head
		# of the whole set would evict for a sequence already admitted.
		admitted_uuids = set(prefill_batch)
		fitting_requirements = [
			required_pages[candidate.uuid] for candidate in slot_admissible
			if candidate.node_id == report_node
			and candidate.uuid not in admitted_uuids
			and _fits(candidate)
		]
		any_node_can_fit = any(_fits(c) for c in slot_admissible)
		reclaim_outcome = (0, 0)
		if (
			self.enable_prefix_cache
			and slot_admissible
			and (
				len(prefill_batch) < len(slot_admissible) if pool_mode
				else not prefill_batch
			)
		):
			if not any_node_can_fit:
				# Every slot-admissible candidate needs more than its node's whole
				# pool. Oversized QUEUEING requests were already failed one by one
				# above, so what is left is an EVICTED sequence that no longer fits
				# a pool it did fit once: an engine invariant violation, not a bad
				# request. Raised before this branch's first collective, on a
				# verdict every rank computed from the same gathered totals.
				self._prefill_admission_selection = selection
				raise PrefillAdmissionStalled(
					"no prefix-cache request fits the total Host KV pool: "
					+ self._describe_prefill_admission_blockers()
				)
			# No candidate fits even after a non-mutating prefix estimate.
			# Reclaim against the original full-page requirement so a later lookup
			# miss cannot make the allocation unsafe, and only as far as the head
			# of the priority order needs (prefill_reclaim_target_pages):
			# reclaiming for the whole admissible set evicts retained prefixes
			# this wave will not use, and every one of them is a hot prefix a
			# later prompt has to recompute.
			from batchgen.prefix_reuse.eviction import (
				EvictedPageRecorder,
				reclaim_prefix_pages_for_host_admission,
				retire_node_frontiers,
			)
			failure = 0
			updated_local_free = -1
			# A reclaim moves pages from reclaimable to free. Re-report BOTH, or
			# the refreshed budget below counts the released pages twice.
			updated_local_releasable = -1
			local_frontier_hook = self._trajectory_frontier_eviction_hook()
			reclaimed_pages = EvictedPageRecorder(local_frontier_hook)
			if self.local_rank == 0:
				try:
					worker_view = self.core_engine.host_paged_kv_worker_view
					stats = worker_view.get_stats()
					updated_local_free = int(stats.num_free_pages)
					required, page_target = prefill_reclaim_target_pages(
						fitting_requirements=fitting_requirements,
						free_pages=updated_local_free,
					)
					if required:
						released = reclaim_prefix_pages_for_host_admission(
							core_engine_module=core_engine,
							coordinator=self.prefix_cache_coordinator,
							worker_views_by_group=self._host_views_by_group(),
							group_id=0,
							page_target=page_target,
							max_scan_nodes=self.prefix_cache_runtime_config.max_nodes,
							frontier_hook=reclaimed_pages,
						)
						updated_local_free = self._get_host_kv_free_pages()
						updated_local_releasable = (
							self.prefix_cache_coordinator.count_releasable_pages(0)
						)
						logging.info(
							"[PREFIX_CACHE] admission reclaimed %s Host pages on node %s "
							"(free=%s, target=%s)",
							released, report_node, updated_local_free, required,
						)
						if self.prefix_cache_debug_stats:
							self._emit_prefix_cache_metric(
								"prefix_reclaim",
								reason="host_admission",
								node=int(report_node),
								page_target=int(page_target),
								released_pages=int(released),
								**self._prefix_host_resource_fields(),
							)
				except Exception:
					logging.exception("[PREFIX_CACHE] admission eviction failed")
					failure = 1
			failed = torch.tensor(
				[failure, reclaimed_pages.page_count],
				dtype=torch.int64, device=self.torch_device,
			)
			dist.all_reduce(failed, op=dist.ReduceOp.MAX)
			if int(failed[0].item()) != 0:
				raise RuntimeError("prefix cache admission eviction failed on a worker")
			reclaimed_page_count = int(failed[1].item())
			reclaim_outcome = (1, reclaimed_page_count)
			# The leader already retired its own frontiers inside the release.
			# Any other rank of the node may own frontier records standing on
			# the reclaimed pages (G=1, or several serve groups per node), so
			# every rank runs its own hook on its node's reclaimed map.
			frontier_failure = 0
			if reclaimed_page_count != 0:
				gathered_reclaims = [None] * self.world_size
				dist.all_gather_object(
					gathered_reclaims,
					(
						report_node,
						reclaimed_pages.pages_by_group
						if self.local_rank == 0 else None,
					),
				)
				if self.local_rank != 0:
					try:
						retire_node_frontiers(
							gathered=gathered_reclaims,
							node_id=report_node,
							frontier_hook=local_frontier_hook,
						)
					except Exception:
						logging.exception(
							"[PREFIX_CACHE] frontier retirement after admission "
							"reclaim failed"
						)
						frontier_failure = 1
			updated = torch.tensor(
				[
					report_node, updated_local_free, frontier_failure,
					updated_local_releasable,
				],
				dtype=torch.int64, device=self.torch_device,
			)
			gathered = [torch.zeros_like(updated) for _ in range(self.world_size)]
			dist.all_gather(gathered, updated)
			if any(int(item[2].item()) != 0 for item in gathered):
				raise RuntimeError(
					"trajectory frontier retirement after admission reclaim "
					"failed on a worker"
				)
			for item in gathered:
				node, free = int(item[0].item()), int(item[1].item())
				if free >= 0:
					per_node_host_free[node] = free
				releasable = int(item[3].item())
				if releasable >= 0:
					per_node_host_reclaimable[node] = releasable
			selection = replace(
				selection,
				per_node_host_free=tuple(per_node_host_free),
				per_node_host_reclaimable=tuple(per_node_host_reclaimable),
			)
			prefill_batch = PrefillScheduler.select_prefill_batch(selection)
			if not prefill_batch and not self._prefill_admission_live:
				# Nothing is live on ANY rank, so nothing will decode, complete, or
				# free a page or a slot: an empty wave here would repeat forever.
				self._prefill_admission_selection = selection
				raise PrefillAdmissionStalled(
					"prefix cache prefill admission made no progress after "
					"eviction with no live sequence: "
					+ self._describe_prefill_admission_blockers()
				)

		# Rank-uniform admission state for the scheduler loop's no-progress guard
		# (PrefillProgressGuard). Every component is gathered, reduced, or derived
		# from the digest-checked candidate snapshot, and the set is wide enough to
		# separate a recoverable wait from a true stall: free AND total pages, the
		# slot vectors, the candidate IDENTITIES with their required pages (a count
		# alone cannot see one candidate swapped for another), the per-node
		# prefix-cache pressure counters that move while a load or an attachment
		# release is in flight, this wave's reclaim outcome, and the reduced live
		# bit. The scheduler loop folds the Host reservation outcome in on top
		# (_prefill_admission_guard_state).
		self._prefill_admission_state = (
			tuple(per_node_host_free),
			tuple(per_node_host_total),
			None if per_rank_sequence_free is None else tuple(per_rank_sequence_free),
			None if per_node_sequence_free is None else tuple(per_node_sequence_free),
			tuple(
				(
					candidate.uuid,
					candidate.node_id,
					candidate.is_evicted,
					required_pages[candidate.uuid],
				)
				for candidate in selection.candidates
			),
			tuple(per_node_prefix_pressure),
			reclaim_outcome,
			self._prefill_admission_live,
		)
		self._prefill_admission_selection = selection

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

	def _describe_prefill_admission_blockers(self) -> str:
		"""Name the candidates the last admission left out, and why.

		Pool sizes come from the admission gather, not a rank-local stats read:
		the report is then identical on every rank and covers every node, so a
		multi-node stall can still say "can never fit".
		"""
		selection = self._prefill_admission_selection
		if selection is None:
			return "no admission snapshot"
		totals = self._prefill_admission_totals
		if (
			totals is None
			or len(totals) != selection.num_nodes
			or any(int(value) < 0 for value in totals)
		):
			# A node without a leader report has an unknown pool size; omit the
			# totals rather than printing a bogus "can never fit".
			totals = None
		return describe_prefill_blockers(selection, total_pages_by_node=totals)

	def _get_host_kv_total_pages(self) -> int:
		"""This node's shared Host-KV pool size in pages."""
		return int(self.host_paged_kv_worker_view.get_stats().num_total_pages)

	def _prefill_candidate_digest(self, candidate_uuids: List[str]) -> int:
		"""Digest the inputs the prefill candidate snapshot is derived from.

		Covers identity and ORDER plus every field
		``_make_prefill_selection_request`` reads, including the two that decide
		a candidate's node (``assigned_rank``, ``decode_dp_group``). Gathered by
		the admission collective so a divergence is named instead of letting one
		rank reclaim, raise, or enter a collective the others do not.
		"""
		hasher = hashlib.blake2b(digest_size=7)
		for uuid in candidate_uuids:
			seq = self.global_batch.get_sequence(uuid)
			hasher.update(
				f"{uuid}:{int(seq.status)}:{seq.global_idx}:{seq.prompt_length}:"
				f"{seq.kv_token_budget}:{seq.PAGE_SIZE}:{seq.assigned_rank}:"
				f"{seq.decode_dp_group}:"
				f"{getattr(seq, 'prefill_order', None)};".encode()
			)
		return int.from_bytes(hasher.digest(), "little")

	def _prefill_prefix_pressure(self) -> Tuple[int, int, int]:
		"""``(resident_nodes, active_attachments, pending_loads)`` of this node.

		Folded into the admission state so the no-progress guard can tell a wave
		waiting on an in-flight prefix load or an attachment release from one
		that is truly stuck. Read on the node leader only and distributed by the
		admission gather, so every rank of the node sees one value. Diagnostic
		read: an unreadable coordinator reports the ``-1`` sentinel instead of
		masking the stall, and the sentinel is itself a state component.
		"""
		coordinator = getattr(self, "prefix_cache_coordinator", None)
		if not self.enable_prefix_cache or coordinator is None:
			return (-1, -1, -1)
		try:
			stats = coordinator.get_stats()
			return (
				int(stats.resident_nodes),
				int(stats.active_attachments),
				int(stats.pending_load_entries) + int(stats.pending_load_refs),
			)
		except Exception:
			logging.exception(
				"[PREFILL] prefix cache pressure unavailable for the admission state"
			)
			return (-1, -1, -1)

	def _prefill_admission_guard_state(self):
		"""The guard's state: the admission snapshot plus the wave's outcome.

		``_prefill_admission_state`` is built before the Host reservation, so a
		wave the fresh capacity transaction rejected would otherwise be
		indistinguishable from a wave that selected nothing — two different
		causes sharing one streak. ``None`` (no candidates) stays ``None``, which
		resets the guard.
		"""
		if self._prefill_admission_state is None:
			return None
		return self._prefill_admission_state + (self._prefill_reservation_outcome,)

	def _reject_unsatisfiable_prefill_requests(
		self,
		selection,
		per_node_host_total: List[int],
		required_pages: Dict[str, int],
	) -> List[str]:
		"""Fail every QUEUEING request larger than its node's whole Host pool.

		Such a request can never be admitted, by this wave or any later one, so
		leaving it queued blocks the queue until the no-progress guard kills the
		worker over one oversized prompt. It is reported on the same per-request
		error channel admission uses for ``context_length_exceeded``
		(``SequenceStatus`` has no failure state) and dropped from
		``global_batch``. Admission already bound the request a full
		model-context reservation in the unified trajectory pool (its token
		store), so that reservation is released here exactly as completion
		releases it, together with any local QueryBook slot; otherwise every
		rejected request would shrink the pool for the process lifetime.

		EVICTED candidates are deliberately NOT rejected here: a sequence that
		fitted the pool once and no longer does is an engine invariant
		violation, not a bad request, and must keep failing by name.

		Pure decision, no collective: ``per_node_host_total`` is gathered and
		the candidate snapshot was digest-checked, so every rank rejects exactly
		the same uuids. Returns the rejected uuids in candidate order.
		"""
		impossible: List[Tuple[str, int, int]] = []
		for candidate in selection.candidates:
			if candidate.is_evicted:
				continue
			total = int(per_node_host_total[candidate.node_id])
			if total < 0:
				continue
			needed = int(required_pages[candidate.uuid])
			if needed > total:
				impossible.append((candidate.uuid, needed, total))
		if not impossible:
			return []
		rejected: List[str] = []
		for uuid, needed, total in impossible:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None:
				continue
			rejected.append(uuid)
			if self.rank == 0:
				logging.error(
					"[PREFILL] Rejecting %s: initial Host KV reservation %s pages "
					"exceeds the whole node pool (total=%s)",
					uuid[:8], needed, total,
				)
			if (
				self._traj.trajectory_book is not None
				and self._traj.trajectory_book.has_sequence(uuid)
			):
				self._traj.trajectory_book.release(self._traj.trajectory_book.slot_for(uuid))
				seq._buffer_slot = -1
			release_query_book_slot(
				uuid,
				query_book=self.query_book,
				table=self.seq_table,
			)
			if self.rank == 0 and self._response_queue is not None:
				self._response_queue.put({
					"type": "completion",
					"request_id": uuid,
					"batch_id": getattr(seq, "batch_id", None),
					"error": {
						"code": "host_kv_capacity_exceeded",
						"message": (
							f"Prompt needs {needed} Host KV pages, more than the "
							f"whole node pool ({total} pages)"
						),
					},
					"text": "",
				})
			self.global_batch.remove_sequence(uuid)
		self._publish_trajectory_pool_capacity()
		return rejected

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

	def _plan_prefill_micro_batches(self, seq_lengths: List[int], full_seq_lengths: List[int], compute_cached_tokens: List[int]) -> Tuple[List[Tuple[int, int]], int, bool]:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._plan_prefill_micro_batches(
			self,
			seq_lengths,
			full_seq_lengths,
			compute_cached_tokens,
		)

	def _prefix_materialization_kv_config(self, full_seq_lengths: List[int]):
		"""The temporary GPU KV config of one prefix-hit micro-batch."""
		from batchgen.kv_cache.host_kv_mananger_config import build_gpu_kv_config
		from batchgen.prefix_reuse.materialization import temporary_gpu_kv_config

		return temporary_gpu_kv_config(
			build_gpu_kv_config(self.huggingface_ckpt_name, full_seq_lengths),
			full_seq_lengths,
		)

	def _prefill_model_pass_count(self, batch: List[int]) -> int:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefill_model_pass_count(
			self,
			batch,
		)

	def _streamed_sp8_prefill_pass_alignment(
		self, local_prefill_indices: List[int]
	) -> Tuple[int, int]:
		"""Return local/global pass counts for hierarchical streamed-SP8 and
		for resident-EP prefill (every pass joins EP-world collectives, so a
		rank with fewer micro-batches pads with collective-only passes)."""
		method = getattr(
			self.parallel_manager,
			"streamed_sp8_requires_global_pass_alignment",
			None,
		)
		resident = (
			hasattr(self.parallel_manager, "prefill_uses_resident_ep")
			and self.parallel_manager.prefill_uses_resident_ep()
		)
		if not resident and (method is None or not method()):
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

	def _suspend_shared_prefix_credit(self, reason: str) -> None:
		"""Delegate the credit latch while retaining worker logging context."""
		self._prefix.suspend_shared_prefix_credit(reason, rank=self.rank)

	def _shared_prefix_credit_enabled(self) -> bool:
		"""Whether this round may charge an in-wave shared prefix once."""
		# Keep the old short-circuit order: with the feature flag off the pool
		# configuration is intentionally allowed to be absent.
		if not self.shared_prefix_admission or not self.enable_prefix_cache:
			return False
		if self._prefix.shared_prefix_credit_suspended:
			return False
		if not self._prefill_prefix_pool_enabled():
			return False
		return self._prefix.shared_prefix_credit_enabled(
			shared_prefix_admission=self.shared_prefix_admission,
			enable_prefix_cache=self.enable_prefix_cache,
			pool_enabled=True,
			prefill_pool_page_budget=self._prefill_pool_page_budget,
		)

	def _prefix_page_digests(self, seq) -> Tuple[int, ...]:
		"""Worker identity wrapper around the shared digest walk."""
		return self._prefix.prefix_page_digests(
			global_id=int(seq.global_idx),
			prompt_length=int(seq.prompt_length),
			page_size=int(seq.PAGE_SIZE),
			tokens_for=lambda: self._trajectory_tokens(
				seq, length=seq.prompt_length
			).tolist(),
		)

	def _prune_prefix_page_digest_cache(self) -> None:
		"""Pass the live global ids; identity resolution stays on the worker."""
		if not self._prefix.prefix_page_digest_cache:
			return
		self._prefix.prune_prefix_page_digest_cache(
			[int(seq.global_idx) for seq in self.global_batch.sequences.values()]
		)

	def _admission_shared_prefix_segments(
		self, all_candidates: List[str]
	) -> Dict[str, Tuple[Tuple[int, int], ...]]:
		"""Resolve candidates on the worker, then run the pure store walk."""
		credit_enabled = self._shared_prefix_credit_enabled()
		if credit_enabled:
			self._prune_prefix_page_digest_cache()
		return self._prefix.admission_shared_prefix_segments(
			all_candidates,
			sequence_for=self.global_batch.get_sequence,
			tokens_for=lambda seq: self._trajectory_tokens(
				seq, length=seq.prompt_length
			).tolist(),
			credit_enabled=credit_enabled,
		)

	def _admission_order_key(self, seq) -> int:
		"""The order ``select_prefill_batch`` walks QUEUEING candidates in."""
		order = getattr(seq, "prefill_order", None)
		return int(seq.global_idx if order is None else order)

	def _make_prefill_selection_request(
		self, all_candidates: List[str], per_node_host_free: List[int],
		num_nodes: int, chunk_size: int, *,
		per_rank_sequence_free=None, per_node_sequence_free=None,
		per_node_host_reclaimable=None,
	) -> PrefillSelectionRequest:
		"""Snapshot the candidate metadata `select_prefill_batch` consumes."""
		shared_segments = self._admission_shared_prefix_segments(all_candidates)
		self._prefix.shared_prefix_credit_applied = bool(shared_segments)
		candidates = []
		for uuid in all_candidates:
			seq = self.global_batch.get_sequence(uuid)
			seq_node = self.rank_ctx.node_of(seq.assigned_rank)
			G = self._decode_attn_tp_size()
			if G > 1:
				if seq.decode_dp_group is None:
					raise RuntimeError(
						f"sequence {uuid[:8]} has no decode_dp_group before "
						"TP prefill selection"
					)
				from batchgen.decode_dp_group import host_kv_owner_rank
				seq_node = self.rank_ctx.node_of(
					host_kv_owner_rank(seq.decode_dp_group, G)
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
				# G>1 no longer replicates the reservation: only the serve-group
				# leader (_owns_host_kv) registers / allocates in the per-node
				# shared host region and the other G-1 ranks mirror its page
				# table (_mirror_group_host_kv), so one admitted sequence costs
				# the node allocator exactly one copy of req_pages.
				host_kv_replication_factor=1,
				prefill_order=getattr(seq, "prefill_order", None),
				shared_prefix_segments=shared_segments.get(uuid, ()),
			))
		return PrefillSelectionRequest(
			candidates=tuple(candidates),
			per_node_host_free=tuple(per_node_host_free),
			chunk_size=chunk_size,
			num_nodes=num_nodes,
			gpus_per_node=self.local_world_size,
			initial_gpu_page_buffer=_sequence_module.INITIAL_GPU_PAGE_BUFFER,
			per_rank_sequence_free=(
				tuple(per_rank_sequence_free)
				if per_rank_sequence_free is not None else None
			),
			per_node_sequence_free=(
				tuple(per_node_sequence_free)
				if per_node_sequence_free is not None else None
			),
			per_node_host_reclaimable=(
				tuple(per_node_host_reclaimable)
				if per_node_host_reclaimable is not None else None
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
		# ON_HOLD sequences are continuing decode when resumed, so they should
		# get EXTENSION_GPU_PAGE_BUFFER (smaller), not INITIAL_GPU_PAGE_BUFFER.
		for uuid in uuids:
			seq = self.global_batch.get_sequence(uuid)
			seq.gpu_pages_allocated = 0
			seq.log_event(SeqEvent.ON_HOLD, self.rank, "trigger=watermark")
			self.global_batch.update_status(uuid, SequenceStatus.ON_HOLD)

		# Synchronize state across all ranks
		dist.barrier()

	def _snapshot_decode_capacity(self) -> DecodeCapacitySnapshot:
		"""Collect the live GPU-KV capacity before decode admission.

		Total pages are immutable after initialization and must match on every
		rank.  Free pages are dynamic; the pure reducer takes the tightest rank
		in each replicated attention group so the admission plan is safe for all
		ranks that will allocate the selected sequence.

		The whole read — local stats and the cross-rank all_gather — runs inside
		the manager's ``capacity_snapshot()`` section (#424), so the phase
		switch cannot release the pool underneath a decision that is already
		being taken against it: a release attempted here fails loud instead of
		producing a plan for pages that no longer exist.
		"""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None or not manager.is_initialized:
			raise RuntimeError(
				"GPU KV manager must be initialized before taking a decode capacity snapshot"
			)
		with manager.capacity_snapshot():
			stats = manager.get_stats()
			local = torch.tensor(
				[int(stats.num_total_pages), int(stats.num_free_pages)],
				dtype=torch.int64,
				device=self.torch_device,
			)
			if self.world_size == 1:
				rank_stats = [local]
			else:
				if not dist.is_initialized():
					raise RuntimeError(
						"distributed process group is required for a multi-rank decode capacity snapshot"
					)
				rank_stats = [torch.zeros_like(local) for _ in range(self.world_size)]
				dist.all_gather(rank_stats, local)
		return reduce_decode_capacity_snapshot(
			tuple(int(item[0].item()) for item in rank_stats),
			tuple(int(item[1].item()) for item in rank_stats),
			world_size=self.world_size,
			attn_tp_size=self._decode_attn_tp_size(),
		)

	def _prepare_decode_batch(self) -> List[str]:
		"""
		Select sequences for decode phase from PREFILLED sequences.
		Greedily fill the current physical GPU-KV free capacity.
		"""
		prefilled_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		onhold_uuids = self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
		
		# Combine and sort for deterministic ordering
		all_candidates = prefilled_uuids + onhold_uuids
		all_candidates.sort(key=lambda uuid: self.global_batch.get_sequence(uuid).global_idx)
		
		# Get a collective live capacity snapshot.  The old implementation used
		# only one rank's total_pages and a metadata-derived resident count, which
		# could admit against stale state or strand a candidate at the 90% guard.
		capacity = self._snapshot_decode_capacity()
		if not all_candidates:
			return []

		# Candidate enumeration and logging stay here; pure physical-capacity
		# selection is delegated.
		# Owner-reported shared GPU prefixes, gathered before the pure
		# selection so every rank charges the same marginal cost: rows that
		# reference one Host page get ONE physical GPU page, so charging each
		# row its whole context would bound decode by the logical page sum.
		shared_segments = self._gather_decode_shared_prefix_segments(
			all_candidates
		)
		decode_batch = DecodeScheduler.select_decode_batch(
			self._make_decode_batch_request(
				all_candidates, capacity, shared_segments=shared_segments
			)
		)

		# An all_short whole-model capture never runs indexer scoring, so a
		# sequence whose KV budget can outgrow index_topk must never join the
		# decode batch of such a lifetime. Admission is deterministic across
		# ranks, so this raises on every rank together (a clean failure)
		# instead of one rank leaving the graph mid-collective.
		if decode_batch and self._graph.glm5_whole_model_all_short:
			index_topk = int(self._graph.glm5_whole_model_index_topk or 0)
			if index_topk:
				over_budget = [
					uuid for uuid in decode_batch
					if int(self.global_batch.get_sequence(uuid).kv_token_budget) > index_topk
				]
				if over_budget:
					raise RuntimeError(
						"GLM-5 all_short whole-model graph lifetime cannot admit "
						f"{len(over_budget)} sequence(s) with kv_token_budget > "
						f"index_topk={index_topk} (first: {over_budget[0]}); "
						"restart the server for workloads with longer budgets"
					)

		if self.rank == 0:
			logging.info(
				f"[DECODE] Prepared batch: {len(decode_batch)} sequences"
			)

		return decode_batch

	def _decode_shared_prefix_credit_enabled(self) -> bool:
		"""Whether decode admission may charge a shared GPU prefix once.

		The credit is only honest where the ALLOCATOR shares: GPU pages are
		deduplicated by Host page id inside
		``allocate_pages_for_sequences_with_page_keys``, which
		``_allocate_gpu_pages_with_host_identity`` reaches exactly when
		``_shared_page_identity_sources`` names the pools to key. A DSA dual
		coordinator is now one of those: both of its pools deduplicate, each
		on its own Host region's page ids. Everywhere else each row really
		does load its own copy, so each row must pay for it -- the same
		condition ``_host_identity_page_keys`` tests before returning page
		keys at all.
		"""
		if not self.shared_prefix_admission:
			return False
		if not self.enable_prefix_cache:
			return False
		manager = self.gpu_paged_kv_cache_manager
		if manager is None or not manager.is_initialized:
			return False
		return self._shared_page_identity_sources(manager) is not None

	def _local_decode_shared_prefix_runs(
		self, all_candidates: List[str]
	) -> Dict[str, Tuple[Tuple[int, int], ...]]:
		"""Shared leading page runs among the candidates THIS rank owns.

		Grouped per owner because sharing only pays inside one GPU pool: the
		pool belongs to one DP rank (or one replicated TP group whose Host-KV
		leader owns every row), so two rows loading onto different pools each
		need their own physical copy.

		The identity is the HOST page id, which is what the allocator dedups
		on, so a run here means those rows provably reference one physical
		copy -- not merely equal tokens, which two distinct Host copies can
		also have.

		The view comes from ``_shared_page_identity_sources``, so the runs are
		read from exactly the key space the allocator will dedup on. On a DSA
		dual coordinator that is the PRIMARY pool (the MLA latent region): the
		auxiliary pool shares the same rows page for page, and the charge is
		counted in primary-pool pages, so one pool's identity describes the
		credit for both.
		"""
		sources = self._shared_page_identity_sources(
			self.gpu_paged_kv_cache_manager
		)
		if not sources:
			return {}
		worker_view = sources[0][2]
		mine = []
		for uuid in all_candidates:
			if uuid not in self._uuid_to_local_map:
				continue
			seq = self.global_batch.get_sequence(uuid)
			req_pages = int(seq.get_gpu_pages_for_two_page_buffer())
			if req_pages <= 1:
				continue
			mine.append((uuid, seq, req_pages))
		if len(mine) < 2:
			return {}
		host_tables = worker_view.build_page_table(
			[int(seq.global_idx) for _, seq, _ in mine]
		)
		tables: Dict[int, List[int]] = {}
		for (_uuid, seq, req_pages), host_pages in zip(mine, host_tables):
			# Credit only pages that exist, and never the row's whole
			# reservation: one page stays private so no row is charged zero.
			# A Host table shorter than the reservation is named loudly by
			# _host_identity_page_keys at allocation time; admission must not
			# credit pages it cannot see.
			usable = min(req_pages - 1, len(host_pages))
			if usable <= 0:
				continue
			tables[int(seq.global_idx)] = [
				int(page) for page in host_pages[:usable]
			]
		runs = plan_shared_runs(tables)
		local: Dict[str, Tuple[Tuple[int, int], ...]] = {}
		for uuid, seq, _req_pages in mine:
			run = runs.get(int(seq.global_idx))
			if run is None:
				continue
			local[uuid] = ((int(run[0]), int(run[1])),)
		return local

	def _gather_decode_shared_prefix_segments(
		self, all_candidates: List[str]
	) -> Dict[str, Tuple[Tuple[int, int], ...]]:
		"""Collect every rank's shared runs so selection stays rank-identical.

		``select_decode_batch`` is pure and runs on every rank; a rank that
		credited a run its peers did not would admit a different batch and the
		next collective would deadlock. Only an owner can read its rows' Host
		page tables, so the ground truth is gathered rather than re-derived.
		``all_candidates`` is already rank-uniform (it comes from replicated
		sequence status), so every rank reaches this gather together.
		"""
		if not self._decode_shared_prefix_credit_enabled():
			return {}
		local = self._local_decode_shared_prefix_runs(all_candidates)
		if self.world_size == 1:
			return local
		if not dist.is_initialized():
			raise RuntimeError(
				"distributed process group is required to gather decode "
				"shared-prefix runs"
			)
		gathered: List[Optional[Dict[str, Tuple[Tuple[int, int], ...]]]] = (
			[None] * self.world_size
		)
		dist.all_gather_object(gathered, local)
		merged: Dict[str, Tuple[Tuple[int, int], ...]] = {}
		for part in gathered:
			if part:
				merged.update(part)
		return merged

	def _make_decode_batch_request(
		self, all_candidates: List[str], capacity: DecodeCapacitySnapshot,
		shared_segments: Optional[Dict[str, Tuple[Tuple[int, int], ...]]] = None,
	) -> DecodeBatchRequest:
		"""Snapshot the candidate metadata `select_decode_batch` consumes."""
		shared_segments = shared_segments or {}
		attn_tp_size = self._decode_attn_tp_size()
		num_capacity_groups = self.world_size // attn_tp_size
		existing_pages = [0] * num_capacity_groups
		existing_sequence_counts = [0] * num_capacity_groups
		for uuid in self.global_batch.get_sequences_by_status(SequenceStatus.IN_DECODE):
			seq = self.global_batch.get_sequence(uuid)
			if attn_tp_size > 1:
				r = seq.decode_dp_group
				if r is None:
					raise ValueError(
						f"sequence {uuid[:8]} has no decode_dp_group for "
						f"attn_tp_size={attn_tp_size}"
					)
			else:
				r = seq.assigned_rank
			if not 0 <= r < num_capacity_groups:
				raise ValueError(
					f"sequence {uuid[:8]} has invalid decode capacity group {r}"
				)
			existing_pages[r] += max(0, seq.gpu_pages_allocated)
			existing_sequence_counts[r] += 1

		candidates = []
		for uuid in all_candidates:
			seq = self.global_batch.get_sequence(uuid)
			candidates.append(DecodeCandidate(
				uuid=uuid,
				assigned_rank=seq.assigned_rank,
				global_idx=seq.global_idx,
				req_pages=seq.get_gpu_pages_for_two_page_buffer(),
				decode_dp_group=seq.decode_dp_group,
				shared_prefix_segments=shared_segments.get(uuid, ()),
			))
		return DecodeBatchRequest(
			candidates=tuple(candidates),
			total_pages=capacity.total_pages,
			world_size=self.world_size,
			max_rank_bsz=getattr(self, "_decode_padding_bsz", 0) or 0,
			attn_tp_size=attn_tp_size,
			existing_pages=tuple(existing_pages),
			free_pages=capacity.free_pages,
			existing_sequence_counts=tuple(existing_sequence_counts),
		)

	def _check_and_handle_completions(
		self, 
		decode_uuids: List[str], 
		local_decode_indices: List[int],
		new_token_idx: int
	) -> Tuple[List[str], List[int], List[str]]:
		"""
		Check for completed sequences at page boundaries.
		A per-request ignore_eos is already honoured at the source: it keeps
		seq.eos_reached unset, so the EOS column below never fires for it.
		"""
		n = len(decode_uuids)
		if n == 0:
			return [], [], []

		# Vectorized completion check: build tensors once, compare in batch
		decoded_lens = torch.empty(n, dtype=torch.int64)
		max_lens = torch.empty(n, dtype=torch.int64)
		ctx_lens = torch.empty(n, dtype=torch.int64)
		eos_flags = torch.empty(n, dtype=torch.bool)

		seqs = []
		for i, uuid in enumerate(decode_uuids):
			seq = self.global_batch.get_sequence(uuid)
			seqs.append(seq)
			decoded_lens[i] = seq.decoded_length
			max_lens[i] = seq.max_decode_length
			ctx_lens[i] = seq.current_context_length
			eos_flags[i] = seq.eos_reached

		# Variable-length N-gram repetition detection at decision boundary
		# Catches repeating patterns of length 2-100 tokens (32 repetitions required)
		if REP_DETECTION:
			for i in range(n):
				seq = seqs[i]
				if (seq.decoded_length >= 64 and not seq._rep_detected
						and _repetition_check_enabled(seq)):
					uuid = decode_uuids[i]
					local_idx = self._uuid_to_local_map.get(uuid)
					if local_idx is not None and local_idx in self.query_book:
						dl = seq.decoded_length
						tokens = self._trajectory_decoded_tokens(seq)
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
					f"(decoded_length={seq.decoded_length}, eos_reached={seq.eos_reached})"
				)
			else:
				active_uuids.append(uuid)
				if uuid in self._uuid_to_local_map:
					active_local_indices.append(self._uuid_to_local_map[uuid])

		return active_uuids, active_local_indices, completed_uuids

	def _finish_prefill_completed_sequences(self, prefill_uuids: List[str]) -> List[str]:
		"""Complete the sequences whose budget is satisfied by the prefill token.

		Prefill already samples the first token and appends it through the
		unified trajectory book at the current tail (then increments
		``decoded_length``; see the writeback loop at the end of
		``prefill``/``prefill_prepacked``). For a
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
		# gather text -> release KV -> scalar cleanup -> status -> report.
		# _gather_completed_tokens is a collective; every rank calls it with
		# the identical uuid list.
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
				seq.host_owned_pages = 0
				seq.host_token_capacity = 0
				seq.evicted_from_dp_group = None
				self._sequences_with_gpu_kv.discard(uuid)

		self._update_batch_status(completed_uuids, SequenceStatus.COMPLETED)

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
		# Legacy attention modes still run on a live worker, so route their
		# boundary admission through the same collective-capacity contract as
		# mode 3.  The old local-free/pages-required loop could desynchronize
		# ranks and bypass TP replication or the decode row cap.
		# DecodeScheduler needs a TP group on every candidate, so stamp all
		# PREFILLED/ON_HOLD candidates before taking its live capacity snapshot.
		admission_candidates = (
			self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
			+ self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
		)
		self._assign_decode_dp_groups(admission_candidates)
		new_uuids = [
			uuid for uuid in self._prepare_decode_batch()
			if uuid not in set(current_decode_uuids)
		]
		
		if not new_uuids:
			return current_decode_uuids, current_local_indices

		# Get local indices for sequences belonging to THIS rank
		new_local_indices = self._get_local_indices_for_uuids(new_uuids)
		
		# M2b: assign the decode DP-group before the transition (no-op for G==1).
		self._assign_decode_dp_groups(new_uuids)

		# Every rank participates, including ranks with no local rows, before any
		# allocator mutation.  A failed preflight is global and leaves statuses
		# untouched, so the next caller can retry safely.
		alloc_ok = self._allocate_gpu_kv_two_page_buffer(
			new_local_indices,
			load_from_host=True,
			collective_preflight=True,
		)
		if not alloc_ok:
			raise RuntimeError(
				f"Rank {self.rank}: collective GPU KV load preflight failed for "
				f"{len(new_uuids)} legacy-mode sequences"
			)

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
			self.Init(None, self.max_decoding_length, 0)

		# Initialize empty global batch
		self.global_batch = SequenceBatch()

		# The unified trajectory pool was allocated once during core init. It is
		# fixed for the worker lifetime and derives scheduling capacity from its
		# total reservation capacity; free reservations are telemetry only.
		self._buffer_pool = None
		logging.info(
			f"Rank {self.rank}: unified trajectory pool active "
			f"(capacity={self._traj.trajectory_book.reservation_total_capacity(self.model_context_length)})"
		)

		# Initialize index maps
		self.seq_table.reset()
		self._bump_decode_row_epoch()
		self.num_global_queries = 0
		self.num_local_queries = 0
		self._rejected_sequences = []

		# Reset max_input_length from Init's 8192 default to 0.
		# Init(None,...) defaults max_input_length to 8192 for the initializer,
		# but once core components are ready we must reset it so the first
		# admission batch sets it from actual prompt lengths — which is what
		# _update_config_after_tokenization then propagates to max_prompt_length
		# in engine config BEFORE prefill/decode.
		# Without this, max_prompt_length stays at 8192 which causes wrong
		# KV_Storage_Config.reserved_length and GPU buffer sizing.
		self.max_input_length = 0

		# Enter the persistent generate loop
		self.generate()

	def _nsys_prefill_profile_begin(
		self,
		*,
		local_bsz: int,
		global_bsz: int,
	) -> bool:
		"""Start a batch-debug-gated nsys prefill capture window."""
		debug = self._batchgen_debug or {}
		if not self._debug_flag_enabled(debug.get("glm5_prefill_nsys_profile")):
			return False

		torch.cuda.synchronize(self.torch_device)
		if dist.is_available() and dist.is_initialized():
			dist.barrier()
		if self.rank == 0:
			logging.info(
				"[NSYS_PREFILL_PROFILE] starting cuda profiler capture "
				"global_bsz=%s",
				global_bsz,
			)
			torch.cuda.cudart().cudaProfilerStart()
		if dist.is_available() and dist.is_initialized():
			dist.barrier()
		torch.cuda.nvtx.range_push(
			f"BatchGen_prefill_rank_{self.rank}_local_bsz_{local_bsz}"
			f"_global_bsz_{global_bsz}"
		)
		return True

	def _nsys_prefill_profile_end(self, active: bool) -> None:
		"""Finish one batch-debug-gated nsys prefill capture window."""
		if not active:
			return
		torch.cuda.nvtx.range_pop()
		torch.cuda.synchronize(self.torch_device)
		if dist.is_available() and dist.is_initialized():
			dist.barrier()
		if self.rank == 0:
			logging.info("[NSYS_PREFILL_PROFILE] stopping cuda profiler capture")
			torch.cuda.cudart().cudaProfilerStop()
		if dist.is_available() and dist.is_initialized():
			dist.barrier()

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

	def _k3_decode_profile_step(self, batchgen_debug, batch_rows: int) -> None:
		"""Batch-level ``batchgen_debug.k3_decode_profile_steps = N``: after 3
		warm-up steps, wrap the next N decode steps of rank 0 in
		``torch.profiler`` (CUDA + CPU) and log the kernel table sorted by CUDA
		time — a per-kernel breakdown of the GPU-bound step without a server
		restart. One capture per server lifetime."""
		# ``k3_decode_profile_ranks``: which ranks profile (default rank 0; the
		# first rank of every node shows cross-node skew inside the exchange).
		_prof_ranks = (batchgen_debug or {}).get("k3_decode_profile_ranks") or [0]
		if self.rank not in _prof_ranks or getattr(self, "_k3_decode_profile_done", False):
			return
		try:
			steps = int((batchgen_debug or {}).get("k3_decode_profile_steps", 0) or 0)
		except (TypeError, ValueError):
			return
		if steps <= 0:
			return
		count = getattr(self, "_k3_decode_profile_count", 0) + 1
		self._k3_decode_profile_count = count
		warmup = 3
		if count == warmup + 1:
			from torch.profiler import ProfilerActivity, profile
			self._k3_decode_profiler = profile(
				activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
			)
			self._k3_decode_profiler.__enter__()
			self._k3_decode_profile_rows = batch_rows
			logging.info("[K3_DECODE_PROFILE] capturing %d decode steps at %d rows", steps, batch_rows)
		elif count == warmup + 1 + steps:
			prof = getattr(self, "_k3_decode_profiler", None)
			if prof is None:
				return
			torch.cuda.synchronize(self.torch_device)
			prof.__exit__(None, None, None)
			self._k3_decode_profiler = None
			self._k3_decode_profile_done = True
			table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
			logging.info(
				"[K3_DECODE_PROFILE] %d steps at %d rows (per-step = table / %d)\n%s",
				steps, self._k3_decode_profile_rows, steps, table,
			)
			# ``k3_decode_profile_trace = "<dir>"``: also dump the Chrome trace
			# (kernel sequence per step, for per-layer node counts); the table
			# alone loses the order.
			trace_dir = (batchgen_debug or {}).get("k3_decode_profile_trace")
			if trace_dir:
				try:
					os.makedirs(trace_dir, exist_ok=True)
					path = os.path.join(
						trace_dir, f"k3_decode_profile_rows{self._k3_decode_profile_rows}_rank{self.rank}.json"
					)
					prof.export_chrome_trace(path)
					logging.info("[K3_DECODE_PROFILE] trace written to %s", path)
				except Exception as exc:  # noqa: BLE001 - diagnostics only
					logging.warning("[K3_DECODE_PROFILE] trace export failed: %s", exc)

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
			# Local: only this diagnostic exit needs it.
			from batchgen.eventlog.process import exit_process

			logging.info(
				"[NSYS_DECODE_PROFILE] rank=%s exiting after %s profiled decode forwards",
				self.rank,
				forward_idx,
			)
			exit_process(0, "nsys profile exit")

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
			# Rank 0 finds an available port, then broadcasts to all ranks
			selected_port = self._nccl_port
			if self.rank == 0:
				try:
					selected_port = self._find_available_pynccl_port(
						comm_master_addr,
						self._nccl_port,
					)
					logging.debug(f"Rank 0: Found available port {selected_port} for PyNccl")
				except RuntimeError as e:
					logging.error(f"Rank 0: Failed to find available port: {e}")
					selected_port = -1

			# Broadcast the chosen port from rank 0 to all ranks
			port_tensor = torch.tensor([selected_port], dtype=torch.int32, device=self.torch_device)
			dist.broadcast(port_tensor, src=0)
			self._nccl_port = port_tensor.item()
			if self._nccl_port < 0:
				raise RuntimeError(
					"No available port remains in this runtime's PyNccl range"
				)

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

	def _find_available_pynccl_port(self, host: str, start_port: int) -> int:
		end_port = self.pynccl_port_base + self.pynccl_port_span
		start_port = max(start_port, self.pynccl_port_base)
		remaining = end_port - start_port
		if remaining <= 0:
			raise RuntimeError(
				"PyNccl port range exhausted: "
				f"[{self.pynccl_port_base}, {end_port})"
			)
		return _find_available_port(host, start_port, remaining)

	def _verify_deployment_max_model_len(self, pool_gb: float) -> None:
		"""Collect every region before allocating GPU pages or publishing L."""
		from batchgen.kv_cache.host_kv_mananger_config import (
			_resolve_profile, _resolve_indexer_profile, is_dsa_model,
		)

		profiles = [_resolve_profile(self.huggingface_ckpt_name)]
		if is_dsa_model(self.huggingface_ckpt_name):
			profiles.append(_resolve_indexer_profile(self.huggingface_ckpt_name))
		page_bytes = [p.bytes_per_page() * p.num_layers for p in profiles]
		# This is the same split used by _initialize_gpu_kv_manager_fixed_size:
		# all regions have equal page counts, each page spans its physical layers.
		gpu_pages = max(0, int(pool_gb * (1024 ** 3))) // sum(page_bytes)
		node = node_of_rank(self.rank, self.local_world_size)
		capacities = []
		host_views = self._host_views_by_group()
		for region, (profile, size) in enumerate(zip(profiles, page_bytes)):
			capacities.append(SequenceLengthCapacity(
				"GPU KV", str(region), node, self.rank,
				profile.page_size, size, gpu_pages,
			))
			if self.local_rank == 0:
				capacities.append(SequenceLengthCapacity(
					"host KV", str(region), node, self.rank,
					profile.page_size, size,
					int(host_views[region].get_stats().num_total_pages),
					copies=self._decode_attn_tp_size(),
				))
		if self.local_rank == 0:
			book = self._traj.trajectory_book
			capacities.append(SequenceLengthCapacity(
				"input IDs", "trajectory", node, self.rank,
				book.page_tokens, book.page_tokens * QueryBook.TOKEN_BYTES, book.page_count,
			))
		all_capacities = [None] * self.world_size
		dist.all_gather_object(all_capacities, capacities)
		length = verify_max_model_len(
			self.model_context_length,
			[capacity for rank_capacities in all_capacities for capacity in rank_capacities],
			auto_fit=(
				getattr(self.args, "max_model_len", None) == -1
				and not getattr(self, "_deployment_length_verified", False)
			),
		)
		self.model_context_length = length
		self.engine_config.Basic_Config.max_model_len = length
		self._deployment_length_verified = True

	def prepare_deployment(self) -> None:
		"""Resolve and verify L against decode residency before worker readiness.

		The measured provisional budget is retained for every model and mode;
		it reserves the configured fraction's slack for later allocations.
		Both lifecycles probe the startup decode transient at the per-rank cap
		right after the pool is sized (``_probe_startup_transient``); it is
		published, not yet reserved.

		On the default lifecycle a decode instance the strategy manager keeps
		across the switch to prefill has its decode graphs captured here,
		inside that probe and before readiness
		(``_startup_decode_graph_capture``); every later activation replays
		them. An instance the manager rebuilds per phase captures at its
		decode configure.
		"""
		with self._startup_phase("model-build"):
			if not self._core_initialized:
				self.Init(None, 1, 0)
		with self._startup_phase("communicator-init"):
			self._ensure_pynccl_communicator()
		with self._startup_phase("decode-shard-build"):
			if self._persistent_phase_enabled():
				self._build_persistent_phase_instances()
			else:
				prefill_key = self._desired_phase_key("prefill")
				self._prefill_moe_mode_in_effect = prefill_key.prefill_moe_mode
				self._load_decode_model(comm=self.comm)
				self._init_gpu_kv_with_actual_size()
				self._probe_startup_transient(
					seam=_startup_transient.SEAM_DEPLOYMENT_PREPARE,
					capture=self._startup_decode_graph_capture(prefill_key),
				)
				self._transition_phase(prefill_key)
				self.set_phase("prefill")
				self._install_prefill_weight_copy_pipeline(False)
		logging.info("Deployment max_model_len verified: %s", self.model_context_length)

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
		# Each step below reports itself as a named startup phase: every one is
		# long, silent and workload-independent, so a step that STOPS -- a
		# checkpoint read that never returns, a rendezvous a peer never joins --
		# is indistinguishable from a step that is merely slow until the
		# server's watchdog names the phase this rank has been sitting in.
		with self._startup_phase("model-build"):
			self.Init(None, _K3_STARTUP_MAX_DECODING_LENGTH, 0)
		with self._startup_phase("extension-preload"):
			self._preload_kimi_k3_runtime_extensions()
		# Model-specific native dependencies beyond the family's (e.g. a
		# sparse-attention indexer's kernels), owned by the model package.
		preload_model_extensions = getattr(
			self.parallel_manager, "preload_runtime_extensions", None
		)
		if preload_model_extensions is not None:
			with self._startup_phase("model-extension-preload"):
				preload_model_extensions()
		with self._startup_phase("communicator-init"):
			self._ensure_pynccl_communicator()
			# The resident-EP decode MoE runs its own all_gather/all_reduce, so
			# the manager needs the communicator before configure_decoding, not
			# at the first decode phase.
			self.parallel_manager.set_comm(self.comm)

		# Decode first: configure_decoding materializes the stacked MXFP4 EP
		# shard that both phases keep resident. _load_decode_model sizes it at
		# the per-rank decode width (_max_decode_rank_bsz), the same width the
		# runtime decode configure and the decode admission cap use, so the
		# padded MoE buffers are already sized for the largest admitted batch.
		# This decode entry precedes any prefill entry, so the worker's record
		# of the prefill mode these instances are built for must be primed
		# here: the decode phase key carries it, and a None here against the
		# startup mode below would turn the resumable decode exit into RETIRE,
		# which leaves the resident shard mapped and OOMs the prefill build.
		startup_prefill_mode = self.parallel_manager.default_prefill_moe_mode()
		self._prefill_moe_mode_in_effect = startup_prefill_mode
		with self._startup_phase("decode-shard-build"):
			self._load_decode_model(comm=self.comm)

		# Then hand the built model to the prefill mode this weight topology
		# will actually use, so the first admission inherits this phase directly
		# instead of rebuilding the streamed-SP8 buffers itself.
		with self._startup_phase("prefill-configure"):
			self.parallel_manager.set_prefill_moe_mode(startup_prefill_mode)
			prefill_key = PhaseKey("prefill", startup_prefill_mode, ())
			self._phase_profile_overrides = ()
			self._transition_phase(prefill_key)
			self.set_phase("prefill")
			self._install_prefill_weight_copy_pipeline(k3_prefill_profile=False)

		# Slice 3: the DeepEP buffer lives for the process; build it before
		# readiness so the first decode configure does not pay the NVSHMEM
		# handshake inside the measured wall.
		prewarm_exchange = getattr(self.parallel_manager, "prewarm_deepep_exchange", None)
		if prewarm_exchange is not None:
			with self._startup_phase("prewarm"):
				prewarm_exchange()

		self._k3_startup_completed = True
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
		# Warm the same capability-selected kernel the decode forward uses
		# (flash_mla on Hopper, flashinfer trtllm-gen on Blackwell).
		from batchgen.attention.mla.paged_decode import mla_paged_decode

		mla_paged_decode(query, blocked_k, block_table, cache_seqlens, 512, 576 ** -0.5)
		torch.cuda.synchronize(device)

	def _group_token_totals(self):
		"""(sequences, prompt tokens, decoded tokens) of the current batch group."""
		group = [self.global_batch.get_sequence(u) for u in self._group_uuids]
		group = [s for s in group if s is not None]  # rejected at admission
		return (
			len(group),
			sum(s.prompt_length for s in group),
			sum(s.decoded_length for s in group),
		)

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
		# Sequences of the current batch group. global_batch keeps completed
		# sequences across groups, so the summaries count only these to match
		# the per-group timers.
		self._group_uuids = {seq.uuid for seq in self.global_batch}
		self._decode_group_idx = 0  # Track decode groups for diagnostic logging
		self._cumulative_decode_iterations = 0
		self._cumulative_decode_boundaries = 0
		self._cumulative_boundary_ms = 0.0
		self._cumulative_forward_ms = 0.0
		
		# NOTE: torch.distributed was initialized at worker startup, long before
		# this loop. This is just a sanity check - should never fail here.
		logging.info(f"Rank {self.rank}: Verifying distributed connections...")
		if not dist.is_initialized():
			raise RuntimeError(f"Rank {self.rank}: torch.distributed not initialized")
		if not self._check_and_reinit_pynccl():
			raise RuntimeError(f"Rank {self.rank}: Failed to ensure healthy PyNccl communicator")
		logging.info(f"Rank {self.rank}: Distributed connections verified")
		
		# Ensure communicator is ready
		self._ensure_pynccl_communicator()

		iteration = 0

		if self._persistent_phase_enabled():
			self._build_persistent_phase_instances()

		# Persistent loop: continues until all completed AND no more admissions expected
		while True:
			# --- Cohort teardown boundary ---
			# Carries the decode-step exit the rest of the way out.  Without it
			# the decode phase below is re-entered while its sequences are still
			# IN_DECODE -- its setup issues collectives before decoding_continuous
			# returns at its own check again -- so the rank would spin on
			# aborted communicators instead of leaving.
			#
			# This tests the LATCH, deliberately not the probe.  Reading the flag
			# here would also retire an IDLE pool, which today leaves through the
			# queue sentinel a few lines below and is counted as
			# ordered-via-sentinel; the cohort flag is set just before that
			# sentinel, so a probe test would silently take over every clean
			# teardown.  Only a rank that actually left the decode loop at its
			# boundary exits here.
			if self._cohort_shutdown_observed:
				break

			# --- ADMISSION CHECK: Poll for new sequences from IntakePool ---
			if self._admission_queue is not None:
				admitted = self._poll_admissions()
				if admitted and self.rank == 0:
					logging.info(f"[POOL] Admitted new sequences, total in batch: {len(self.global_batch)}")
					self._timing_logged = False  # Reset for new batch group

			# --- TERMINATION CHECK ---
			if self.global_batch.all_completed():
				self._pause_phase_switch_clock()
				# Print timing summary when all current work is done
				if not self._timing_logged and self.rank == 0:
					gen_time = time.perf_counter() - generation_start_time
					num_seq, total_prompt, total_decoded = self._group_token_totals()
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
				if self._shutdown_requested:
					break  # Pool mode: shutdown requested and all done
				# Pool mode: wait briefly for more work before exiting
				# status tensor encoding: [has_new_work, shutdown]
				import queue as queue_mod
				if self.rank == 0:
					try:
						msg = self._admission_queue.get(timeout=1.0)
						if msg is None:
							self._shutdown_requested = True
							status = torch.tensor([0, 1], dtype=torch.int32, device=self.torch_device)
							dist.broadcast(status, src=0)
						elif isinstance(msg, dict) and msg.get("type") == "admit":
							# Broadcast that we got new work
							status = torch.tensor([1, 0], dtype=torch.int32, device=self.torch_device)
							dist.broadcast(status, src=0)
							container = [msg]
							dist.broadcast_object_list(container, src=0)
							self._group_uuids = set()
							self._admit_sequences_from_message(msg)
							# Reset per-batch-group timing so each admission cycle
							# emits its own "Pool batch group completed" summary.
							prefill_time = 0.0
							decoding_time = 0.0
							generation_start_time = time.perf_counter()
							self._timing_logged = False
							# Continue loop — new sequences will be picked up
						elif isinstance(msg, dict) and "prompts" in msg:
							# Legacy /v1/inference payload -- see the matching
							# guard in _poll_admissions. The `else` below would
							# swallow it and the caller would then steal the next
							# completion off the shared response queue.
							raise LegacyInferenceDeprecated()
						else:
							status = torch.tensor([0, 0], dtype=torch.int32, device=self.torch_device)
							dist.broadcast(status, src=0)
					except queue_mod.Empty:
						# No work arrived, broadcast no-work to other ranks
						status = torch.tensor([0, 0], dtype=torch.int32, device=self.torch_device)
						dist.broadcast(status, src=0)
						continue  # Try again
				else:
					# Non-rank-0: wait for rank 0's broadcast
					status = torch.tensor([0, 0], dtype=torch.int32, device=self.torch_device)
					dist.broadcast(status, src=0)
					has_new = status[0].item() == 1
					is_shutdown = status[1].item() == 1
					if has_new:
						container = [None]
						dist.broadcast_object_list(container, src=0)
						self._group_uuids = set()
						self._admit_sequences_from_message(container[0])
						# Reset per-batch-group timing (matches rank-0 branch).
						prefill_time = 0.0
						decoding_time = 0.0
						generation_start_time = time.perf_counter()
						self._timing_logged = False
					elif is_shutdown:
						self._shutdown_requested = True
					# Continue loop regardless
				if self.global_batch.all_completed() and self._shutdown_requested:
					break
				if self.global_batch.all_completed():
					continue  # Keep waiting

			iteration += 1
			self._resume_phase_switch_clock()
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
				pool_wave_start = (
					time.perf_counter()
					if prefill_uuids and self._prefill_prefix_pool_enabled()
					else None
				)
				# One-node pure-DP Host admission is rank-independent. Place
				# this selected wave once, BEFORE the reservation registers
				# Host pages on an owner rank — reassigning afterwards would
				# strand registered pages on the old owner. Evicted sequences
				# keep the rank owning their Host KV.
				if (
					pool_wave_start is not None
					and self._get_num_nodes() == 1
					and not self._prefill_sequence_limits()
				):
					wave_seqs = [
						self.global_batch.get_sequence(uuid) for uuid in prefill_uuids
					]
					queueing = [
						seq.status == SequenceStatus.QUEUEING for seq in wave_seqs
					]
					# The packer pins an evicted member on the rank holding
					# its Host KV, so it only needs one placeable member.
					placeable = (
						any(queueing) if self._prefix_rank_packing_mode(wave_seqs)
						else all(queueing)
					)
					if placeable:
						self._assign_ranks_for_prefix_sharing(prefill_uuids, current_wave=True)
				if prefill_uuids and not self._reserve_prefill_host_kv(prefill_uuids):
					# The fresh capacity transaction rejected the snapshot.  Keep
					# QUEUEING/EVICTED state untouched and retry on a later scheduler
					# iteration after the live pool has changed.
					prefill_uuids = []
				# Hard guard: an iteration that admits nothing while nothing is
				# live runs no decode either, so it repeats unchanged. Fail by
				# name after a bounded streak instead of spinning forever.
				# Both inputs are rank-uniform -- the live bit is the OR the
				# admission collective gathered, the state is gathered or
				# derived from a digest-checked snapshot -- so every rank
				# raises on the SAME iteration instead of diverging on the
				# failure path.
				self._prefill_progress_guard.observe(
					admitted=len(prefill_uuids),
					live=self._prefill_admission_live,
					state=self._prefill_admission_guard_state(),
					describe=self._describe_prefill_admission_blockers,
				)
				
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
					self._prefill_global_passes = global_prefill_passes

					# B. Execute Prefill
					self._log_phase_switch("prefill")
					_prefill_profile_active = self._nsys_prefill_profile_begin(
						local_bsz=len(local_prefill_indices),
						global_bsz=len(prefill_uuids),
					)
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
								# M3 Slice 2 restore. THE single
								# pre-suffix-prefill seam: above the
								# prepack/plain dispatch so both paths get the
								# restored KDA state, and above every forward
								# launch of this wave, so the slot write is
								# ordered before any kernel that reads it.
								self._restore_trajectory_frontiers(
									local_prefill_indices
								)
								if self.enable_prepack:
									self.prefill_prepacked(local_prefill_indices)
								else:
									self.prefill(local_prefill_indices)
						finally:
							self._close_active_prefix_materializations()
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
						_offload_wait_start = time.perf_counter()
						num_retired = _AWB.retire_pending_prefill_offloads(
							device=self.torch_device,
							reason="end of prefill",
						)
						# Every D2H of the wave has retired, and the segment
						# pseudo sequences still hold their page tables: the
						# only window where a row's destination Host bytes can
						# be read back and compared to its source.
						self._verify_prefill_pool_offload_bytes()
						self._finish_prefill_pool_wave()
						self._commit_prefix_cache_for_prefill(
							local_prefill_indices
						)
						if num_retired:
							# Every rank logs: r28 showed node 1 entering the first
							# decode configure ~30 s after node 0 with nothing in
							# its log; the offload wait is the first suspect.
							logging.info(
								f"[PREFILL_SYNC] Rank {self.rank}: waited on {num_retired} "
								f"async KV offload tasks in "
								f"{time.perf_counter() - _offload_wait_start:.2f}s"
							)

					transport_only_passes = (
						global_prefill_passes - local_prefill_passes
					)
					if transport_only_passes:
						resident_padding = (
							hasattr(self.parallel_manager, "prefill_uses_resident_ep")
							and self.parallel_manager.prefill_uses_resident_ep()
						)
						logging.info(
							f"Rank {self.rank}: joining {transport_only_passes} "
							+ ("resident-EP prefill passes with no local rows"
							   if resident_padding else
							   "streamed-SP8 prefill weight schedules with no local rows")
						)
						transport_start = time.perf_counter()
						# A hierarchical source rank still acquires every layer's
						# shard, so each transport-only pass is a weight pass.
						with self._weight_passes.scope("prefill"):
							for transport_pass in range(transport_only_passes):
								self.feed_watchdog()
								self._weight_passes.begin(
									carry=transport_pass + 1 < transport_only_passes
								)
								with torch.inference_mode():
									if resident_padding:
										self._sync_prefill_moe_rank_counts(
											0, reason="resident_padding_pass",
										)
										self.parallel_manager.run_resident_ep_collective_only_prefill()
									else:
										self.parallel_manager.run_streamed_sp8_transport_only_prefill(
											1
										)
								self._weight_passes.end()
						prefill_time += time.perf_counter() - transport_start
					self._nsys_prefill_profile_end(_prefill_profile_active)
					self._mark_phase_end("prefill")

					# Cleanup & Status Update
					self._unregister_fp8_weights()
					for uuid in prefill_uuids:
						seq = self.global_batch.get_sequence(uuid)
						seq.log_event(SeqEvent.PREFILL_DONE, self.rank,
							f"decoded_len={seq.decoded_length}")
					self._update_batch_status(prefill_uuids, SequenceStatus.PREFILLED)
					pool_rank_ready_s = (
						time.perf_counter() - pool_wave_start
						if pool_wave_start is not None else None
					)
					dist.barrier()
					# M3 Slice 1c: THE post-barrier finalize. Every rank of
					# every serve group has returned from its own shard write
					# before any rank gets past the barrier above, so a frontier
					# record installed here binds a complete G-shard snapshot.
					# This is also where an arena no commit consumed gives its
					# Host pages back.
					self._finalize_trajectory_frontiers()
					if pool_wave_start is not None:
						logging.info("[METRICS] %s", json.dumps({
							"component": "prefix_pool",
							"phase": "wave",
							"rank": self.rank,
							"sequences": len(prefill_uuids),
							"rank_ready_s": pool_rank_ready_s,
							"wave_s": time.perf_counter() - pool_wave_start,
						}, separators=(",", ":")))

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
				# The decode configure is sized at the per-rank decode width
				# (_max_decode_rank_bsz), never at an estimate of this wave's
				# candidates: the padded MoE buffers, the GPU KV pool measured
				# after them and the decode admission cap (_decode_padding_bsz)
				# all derive from that one width, so a cold configure and a
				# reused instance build the same ledger.
				logging.info(
					f"[DECODE] Rank {self.rank}: entering decode configure "
					f"(decode_width={self._max_decode_rank_bsz()})"
				)
				self._load_decode_model(self.comm)

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

				if global_completed:
					# Refresh rank-0's sequence replicas first: _report_completion
					# reads prompt_length/decoded_length from the local entry,
					# which is stale here for sequences owned by other ranks
					# (only the completion BIT was all-reduced above).
					self._sync_sequence_metadata(list(global_completed))
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
							seq.host_owned_pages = 0
							seq.host_token_capacity = 0
							seq.evicted_from_dp_group = None
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
				# The local rows this resolves are the rows the decode forward
				# asks KDA state for, so refuse a row that the group map does
				# not own here instead of inside prepare_decode_step.
				self._check_local_binding_matches_group(
					decode_uuids, "decode entry"
				)
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

				# The round's criticality probe request. A request that differs
				# from the probe in the resident decode graph (none is a value)
				# arms a recapture, which the warmup below then runs.
				self._configure_criticality_probe(
					self._active_batchgen_debug_for_sequences(global_decode_sequences)
				)

				# CUDA Graph Warmup (configure-time). A decode instance kept
				# across phases had its graphs captured before readiness
				# (_startup_decode_graph_capture), so this finds them and
				# captures nothing.
				has_queueing = self.global_batch.has_queueing()
				if self._decode_graph_capture_needed_at_configure(has_queueing):
					if has_queueing:
						logging.info(
							f"Rank {self.rank}: warming GLM-5 CUDA graph with queued "
							"prefill work still pending when the requested graph path supports it"
						)
					self._warmup_cuda_graphs()

				# C. Execute Continuous Decode
				decode_start = time.perf_counter()
				# One scope around the whole decode loop: every exit of it,
				# including the early cohort-shutdown returns, closes the pass
				# the last step carried.
				with torch.inference_mode(), self._weight_passes.scope("decode"):
					if local_decode_indices:
						new_tokens = self._rebuild_input_tokens(local_decode_indices)
					else:
						new_tokens = torch.empty((0, 1), dtype=torch.int64, device=self.torch_device)

					self._log_phase_switch("decode")
					self.decoding_continuous(new_tokens, decode_uuids, local_decode_indices)
				self._mark_phase_end("decode")
				decoding_time += time.perf_counter() - decode_start

				# The decode loop returned because this cohort is being torn
				# down.  Stop here: the cleanup immediately below ends in
				# dist.barrier(), a fresh collective on a communicator that was
				# very likely just aborted, and the rank that reaches it either
				# raises or re-wedges.  Leaving both loops instead carries the
				# rank out of generate() having issued nothing further, which is
				# what lets the supervisor run the ordered teardown.
				if self._cohort_shutdown_observed:
					break

				# D. Cleanup
				self._finish_decode_interval()
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
				# DECISION_INTERVAL, puts all seqs ON_HOLD during a phase transition, prefills
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

		# Compute throughput metrics from the last batch group (the timers
		# above restart with each pool admission group)
		num_sequences, total_prompt_tokens, total_decoded_tokens = self._group_token_totals()

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
			self._log_decode_timing()

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
		prefill config, so the install rules live in one place no matter which
		caller runs first.

		Every install restarts the schedule at its head. That is where the last
		weight pass of the previous phase parked the producer, and the node
		daemon's schedule is cyclic, so no transport carries a cursor or a slot
		across the phase boundary.
		"""
		streamed_sp8 = (
			hasattr(self.parallel_manager, "prefill_uses_streamed_sp8")
			and self.parallel_manager.prefill_uses_streamed_sp8()
		)
		# Resident decode leaves a streamed-SP8 ring in its prefill layout, so
		# only the first streamed-SP8 install allocates it.
		sp8_reentry = streamed_sp8 and self._streamed_sp8_h2d_installed
		self._weight_passes.close()
		self.core_engine.stop_h2d_worker()
		fingerprint = self._weight_copy_task_fingerprint(self.weight_copy_task)
		if sp8_reentry:
			# configure_prefill rebuilt the task from scratch, but the
			# distributed transport still serves the first install's schedule.
			if fingerprint != self._streamed_sp8_weight_copy_fingerprint:
				raise RuntimeError(
					f"Rank {self.rank}: streamed-SP8 weight-copy schedule "
					"changed after the H2D pipeline was installed; the "
					"distributed weight schedule no longer matches the rebuilt task"
				)
		if not sp8_reentry:
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
		self.core_engine.set_weight_copy_queue(self.weight_copy_task)
		if any(self.weight_copy_task.values()):
			self.core_engine.start_h2d_worker()
		self._streamed_sp8_h2d_installed = streamed_sp8
		self._streamed_sp8_weight_copy_fingerprint = (
			fingerprint if streamed_sp8 else None
		)

	def _assert_phase_switch_clean(
		self, direction: str, *, reused: bool
	) -> None:
		"""Fail loud when a phase switch left the other phase's state behind.

		Checks worker-owned switch state only, which is what this module can
		observe cheaply and deterministically:

		* ``prefill_to_decode`` — the prefill routed-expert weight ring must
		  have been released BEFORE the decode model was built. It used to stay
		  mapped through the whole decode configure, so the switch ran at the
		  tightest free-HBM point of the batch; that is a cleanup bug, not a
		  design choice, so it fails the switch instead of degrading it.
		* ``decode_to_prefill`` — the decode-only graph state must be gone.
		  A stale manager or captured segment keeps Python references to the
		  decode model and its static KV buffers alive, so they survive
		  empty_cache() and the next prefill runs with that memory still held.

		Raises RuntimeError; there is no degraded mode. A violated invariant
		means the next phase runs with the previous phase's HBM still mapped,
		which shows up later as an unrelated OOM or a silent capacity loss.
		"""
		if direction == "prefill_to_decode":
			if reused:
				pm = self.parallel_manager
				if not getattr(pm, "decode_experts_resident", False):
					raise RuntimeError(
						f"Rank {self.rank}: prefill->decode switch activated a "
						"reused decode instance whose routed experts are not "
						"mapped; a graph replay would read unmapped memory"
					)
				self._assert_kv_region_mapped("prefill->decode")
				self._assert_decode_graph_state_kept("prefill->decode")
				self._assert_decode_attention_weights_registered()
			return
		if direction == "decode_to_prefill":
			if reused:
				# Partly inverted on the reuse path, deliberately: the decode
				# graphs are supposed to outlive this switch (tearing them down
				# is what forces the recapture this lifecycle exists to remove),
				# while the two large memory classes — the KV pool and the
				# routed experts — give their physical pages back to the
				# prefill phase at addresses those same graphs baked.
				self._assert_decode_graph_state_kept("decode->prefill")
				self._assert_kv_region_released("decode->prefill")
				pm = self.parallel_manager
				if getattr(pm, "decode_experts_resident", False):
					raise RuntimeError(
						f"Rank {self.rank}: decode->prefill switch left the decode "
						"routed experts mapped; prefill needs that memory back"
					)
				return
			leftovers = [
				name
				for name in (
					"cuda_graph_manager",
					"whole_model_segment",
				)
				if getattr(self._graph, name, None) is not None
			]
			if leftovers:
				raise RuntimeError(
					f"Rank {self.rank}: decode->prefill switch left decode-only GPU "
					f"state mapped: {', '.join(leftovers)} still set; their Python "
					"references keep the decode model and captured static buffers "
					"alive across the prefill phase"
				)
			return
		raise ValueError(f"unknown phase-switch direction {direction!r}")

	def _desired_phase_key(self, phase: str) -> PhaseKey:
		"""Resolve the compatibility key for the next phase entry."""
		if phase == "decode":
			# The decode key carries the prefill MoE mode of the instance it
			# follows. The worker records that mode on every prefill entry
			# (_transition_phase), so the decode key and the live prefill key
			# compare the same value; a manager attribute committed only on a
			# rebuild would read None after a resumed or builder-made entry.
			return PhaseKey(
				"decode",
				self._prefill_moe_mode_in_effect,
				getattr(self, "_phase_profile_overrides", ()),
			)
		if phase != "prefill":
			raise ValueError(f"unsupported phase {phase!r}")

		prefill_sequences = []
		for uuid in getattr(self, "_phase_request_uuids", ()):
			seq = self.global_batch.get_sequence(uuid)
			if seq is not None:
				prefill_sequences.append(seq)
		prefill_debug = (
			self._active_batchgen_debug_for_sequences(prefill_sequences) or {}
		)
		k3_prefill_profile = self._debug_flag_enabled(
			prefill_debug.get("k3_prefill_profile")
		)
		self._weight_passes.end_delay_s = self._weight_pass_end_delay(
			prefill_debug.get("weight_pass_end_delay")
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
		resident_token_cap = int(getattr(
			self.engine_config.Module_Batching_Config,
			"k3_resident_prefill_token_cap", 0,
		) or 0)
		resident_handoff = (
			"k3_prefill_moe_mode" not in prefill_debug
			and resident_token_cap > 0
			and hasattr(self.parallel_manager, "resident_ep_prefill_available")
			and self.parallel_manager.resident_ep_prefill_available()
		)
		if resident_handoff:
			k3_prefill_moe_mode = "resident_ep"
		if hasattr(self.parallel_manager, "set_prefill_moe_mode"):
			self.parallel_manager.set_prefill_moe_mode(k3_prefill_moe_mode)
		self._phase_prefill_debug = prefill_debug
		self._phase_k3_prefill_profile = k3_prefill_profile
		self._phase_resident_handoff = resident_handoff
		self._phase_resident_token_cap = resident_token_cap
		self._phase_profile_overrides = (
			(("k3_prefill_profile", k3_prefill_profile),)
			if "k3_prefill_profile" in prefill_debug
			else ()
		)
		return PhaseKey(
			"prefill", k3_prefill_moe_mode, self._phase_profile_overrides
		)

	def _verify_manager_phase_contract(self) -> None:
		"""Refuse service when the active manager misses a required hook."""
		errors = verify_phase_contract(self.parallel_manager)
		if errors:
			raise RuntimeError(
				f"phase lifecycle contract failed for "
				f"{type(self.parallel_manager).__name__}: "
				+ "; ".join(errors)
			)

	def _phase_exit_mode(self, live: PhaseKey, desired: PhaseKey) -> ExitMode:
		"""Select suspend only when the outgoing instance can resume safely."""
		key_fields_unchanged = (
			live.prefill_moe_mode == desired.prefill_moe_mode
			and live.profile_overrides == desired.profile_overrides
		)
		if (
			key_fields_unchanged
			and self.parallel_manager.holds_instance(live.phase)
		):
			return ExitMode.SUSPEND
		return ExitMode.RETIRE

	def _prepare_phase_exit(self, live: PhaseKey, desired: PhaseKey) -> None:
		"""Stop worker-owned consumers before the manager exit hook."""
		self._weight_passes.close()
		self.core_engine.stop_h2d_worker()
		self.core_engine.clear_kv_copy_queue()
		if live.phase == "prefill" and desired.phase == "decode":
			if not self._streamed_sp8_h2d_installed:
				self.core_engine.reset_decoding_buffer()

	def _release_suspended_decode_kv_pool(self) -> None:
		"""Return the suspended decode KV pool's physical pages, keep its addresses.

		The decode graphs baked the pool bases, page-pointer tables and
		page-table storage, so a resumable decode instance must find them at
		the same virtual addresses; only the physical memory goes back to the
		driver for the prefill phase, and ``_remap_reused_kv_pool`` maps it
		again on resume.
		"""
		self._assert_kv_pool_addresses_are_fixed("decode->prefill")
		manager = self.gpu_paged_kv_cache_manager
		pages = manager.get_stats().num_total_pages if manager is not None else 0
		start = time.perf_counter()
		self._destroy_gpu_paged_kv_cache()
		if manager is not None:
			logging.info(
				"[GPU-KV] rank=%d released the reserved KV pool's physical "
				"pages in %.3fs (%d pages, %.2f GB)",
				self.rank,
				time.perf_counter() - start,
				pages,
				self.gpu_kv_cache_size_gb,
			)

	def _transition_phase(
		self, desired: PhaseKey, *, comm=None, max_num_seq: int | None = None
	) -> tuple[bool, bool]:
		"""Run one worker phase transition and return (changed, resumed)."""
		live = self._live_phase_key
		if live == desired:
			return False, False
		if live is not None:
			exit_mode = self._phase_exit_mode(live, desired)
			self._prepare_phase_exit(live, desired)
			if live.phase == "decode":
				if exit_mode is ExitMode.RETIRE:
					self._destroy_gpu_paged_kv_cache()
				elif self._persistent_phase_enabled():
					self._destroy_gpu_paged_kv_cache(keep_buffers=True)
				else:
					self._release_suspended_decode_kv_pool()
			self.parallel_manager.phase_exit(live, exit_mode)
			if exit_mode is ExitMode.RETIRE:
				if live.phase != "decode":
					self._destroy_gpu_paged_kv_cache()
				self.deep_free_model_memory()
		# A decode entry resumes exactly when the manager still holds a decode
		# instance after the outgoing phase has exited: that is the instance
		# phase_enter activates. It does not depend on how the outgoing phase
		# left — a manager that keeps no prefill instance retires prefill and
		# still resumes its kept decode instance.
		resumed = (
			desired.phase == "decode"
			and self.parallel_manager.holds_instance("decode")
		)
		if desired.phase == "decode":
			self.init_nvshmem()
		elif hasattr(self.parallel_manager, "set_comm"):
			self.parallel_manager.set_comm(self.comm)
		self.model, self.weight_copy_task = self.parallel_manager.phase_enter(
			desired,
			PhaseEnterContext(comm=comm, max_num_seq=max_num_seq),
		)
		self._live_phase_key = desired
		if desired.phase == "prefill":
			self._prefill_moe_mode_in_effect = desired.prefill_moe_mode
		return True, resumed

	def _log_prefill_entry_hbm(self, when: str) -> None:
		if not torch.cuda.is_available():
			return
		free_mem, _ = torch.cuda.mem_get_info(self.local_rank)
		allocated = torch.cuda.memory_allocated(self.local_rank) / 1e9
		reserved = torch.cuda.memory_reserved(self.local_rank) / 1e9
		logging.info(
			f"[HBM] Rank {self.rank} {when} configure_prefill: "
			f"free={free_mem/1e9:.2f}GB alloc={allocated:.2f}GB rsv={reserved:.2f}GB"
		)

	def _config_prefill_for_batch(self, prefill_uuids: List[str]) -> None:
		"""Configure prefill phase for a batch of sequences."""
		start_time = time.perf_counter()
		self._phase_request_uuids = tuple(prefill_uuids)
		desired_key = self._desired_phase_key("prefill")
		prefill_sequences = [
			self.global_batch.get_sequence(uuid)
			for uuid in prefill_uuids
			if self.global_batch.get_sequence(uuid) is not None
		]
		prefill_debug = self._phase_prefill_debug
		k3_prefill_profile = self._phase_k3_prefill_profile
		resident_handoff = self._phase_resident_handoff
		resident_token_cap = self._phase_resident_token_cap
		persistent = self._persistent_phase_enabled()
		if not persistent:
			self._request_phase_instance_reuse()
		if self.rank == 0:
			logging.info(
				f"[PREFILL] Configuring prefill phase for {len(prefill_uuids)} sequences"
			)

		in_decode = self.global_batch.get_sequences_by_status(SequenceStatus.IN_DECODE)
		on_hold = self.global_batch.get_sequences_by_status(SequenceStatus.ON_HOLD)
		prefilling = self.global_batch.get_sequences_by_status(SequenceStatus.PREFILLED)
		if (in_decode or on_hold) and BATCHGEN_CB_DEBUG:
			logging.debug(
				f"Rank {self.rank}: _config_prefill_for_batch called while "
				f"{len(in_decode)} IN_DECODE, {len(on_hold)} ON_HOLD, {len(prefilling)} PREFILLED sequences exist. "
				"This is a decode-to-prefill transition."
			)
			for uuid in (in_decode + on_hold)[:5]:
				seq = self.global_batch.get_sequence(uuid)
				logging.debug(
					f"Rank {self.rank}: Affected seq {seq.uuid[:8]}: "
					f"status={seq.status.name}, decoded_len={seq.decoded_length}, "
					f"ctx_len={seq.current_context_length}, gpu_pages={seq.gpu_pages_allocated}, "
					f"had_initial={seq.had_initial_gpu_reservation}"
				)

		if hasattr(self, "_pending_kv_append_tasks") and self._pending_kv_append_tasks:
			logging.info(
				f"Rank {self.rank}: Flushing {len(self._pending_kv_append_tasks)} pending KV append tasks before prefill config"
			)
			self._wait_pending_kv_append_tasks()
			torch.cuda.synchronize(self.torch_device)

		self._assign_decode_dp_groups(prefill_uuids)
		if self._live_phase_key != desired_key:
			self._log_prefill_entry_hbm("BEFORE")
		phase_changed, resumed = self._transition_phase(desired_key)
		if phase_changed:
			self._log_prefill_entry_hbm("AFTER")
		if phase_changed and not resumed:
			self._log_phase_memory_ledger("after_configure_prefill")
		if phase_changed:
			if resident_handoff:
				self.engine_config.Module_Batching_Config.prefill_micro_batch_token_cap = (
					resident_token_cap
				)
				if torch.cuda.is_available():
					torch.cuda.synchronize(self.torch_device)
			self.set_phase("prefill")
			if self.enable_prefix_cache:
				self._measure_prefill_prefix_kv_budget()
			self._install_prefill_weight_copy_pipeline(k3_prefill_profile)
		else:
			# phase_enter readies the instance for the first wave of a phase;
			# admit prepares every later wave of the same live phase.
			self.parallel_manager.admit(
				desired_key, AdmitContext(uuids=list(prefill_uuids))
			)

		if phase_changed and not persistent and k3_prefill_profile and torch.cuda.is_available():
			torch.cuda.reset_peak_memory_stats(self.local_rank)

		if (
			not persistent
			and hasattr(self.parallel_manager, "prefill_uses_resident_ep")
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

		# STEP 3: Prepare evicted sequences for re-entry (before host KV allocation)
		#
		# The phase transition and manager admission are complete above. The
		# remaining body prepares metadata and binds this wave's local rows.

		#
		# Split into two loops:
		#   (a) All-ranks scalar metadata update (runs on every rank using
		#       fields already synchronized via Phase 4.C of the eviction
		#       boundary and via _sync_sequence_metadata).
		#   (b) Validate the retained trajectory and clear the legacy marker.
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
			n_old = _reentry_history_length(
				seq.prompt_length, seq.original_prompt_length,
				seq.original_max_decode_length,
			)
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

		# (b) The trajectory is already the source of truth. Host-KV eviction
		# releases only KV pages; it must not reconstruct or copy a second token
		# buffer. Validate the retained trajectory and clear the legacy handoff
		# marker so old callers cannot accidentally reintroduce dense storage.
		for uuid in prefill_uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq.total_decoded_before_eviction == 0:
				continue
			prev_decoded = seq.total_decoded_before_eviction
			seq.log_event(SeqEvent.REENTRY_START, self.rank,
				f"new_prompt_len={seq.prompt_length}, prev_decoded={prev_decoded}")
			if self._traj.trajectory_book is None or not self._traj.trajectory_book.has_sequence(uuid):
				raise QueryBookPoolCapacityError(
					f"Rank {self.rank}: re-entry of {uuid[:8]} has no trajectory slot"
				)
			trajectory_slot = self._trajectory_slot(seq)
			trajectory_length = self._traj.trajectory_book.metadata(trajectory_slot).token_length
			if trajectory_length < seq.prompt_length:
				raise RuntimeError(
					f"Rank {self.rank}: trajectory for {uuid[:8]} is shorter than "
					f"re-entry prompt ({trajectory_length} < {seq.prompt_length})"
				)
			# Record the replayed prompt boundary and make the local QueryBook
			# metadata agree with the scalar effective prompt length. No token bytes
			# are copied: the trajectory already contains this prefix.
			self._traj.trajectory_book.begin_reentry_turn(
				trajectory_slot, prompt_length=seq.prompt_length
			)
			# Clear the legacy owner-only tensor marker. No token copy occurs.
			seq.evicted_token_ids = None

			# Recreate query_book entry for this rank's evicted sequences (Q4).
			# G>1 binds the row on all G ranks of its serve group, and each of
			# them decodes from its own query_book, so every holder must rebuild
			# the entry — not just assigned_rank. The group id an evicted re-entry
			# lost is already restored by the _assign_decode_dp_groups call at the
			# top of this method. G==1: the same single rank as before.
			if self._owns_local_sequence(seq) and uuid in self._uuid_to_local_map:
				local_idx = self._uuid_to_local_map[uuid]
				self.query_book[local_idx] = make_query_book_entry(seq)

			logging.info(
				f"Rank {self.rank}: Prepared EVICTED seq {uuid[:8]} for re-entry: "
				f"new_prompt={seq.prompt_length}, prev_decoded={prev_decoded}, "
				f"remaining_decode={seq.max_decode_length}, kv_budget={seq.kv_token_budget}"
			)

		# STEP 3.5 (Option 1, CORE): (re)assign the serve-group before binding.
		# Idempotent for fresh admits (already grouped at admission); RE-groups
		# evicted re-entries (whose decode_dp_group was cleared on eviction) so
		# the group predicate below binds them on all G ranks. No-op for G==1.
		self._assign_decode_dp_groups(prefill_uuids)

		# STEP 4: Bind local rows for sequences this rank serves.  Host-KV pages
		# were reserved and committed before IN_PREFILL was published, so this
		# phase must not perform another allocator mutation.
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
		if self.rank == 0:
			logging.debug(
				"[PREFILL] Host KV reservation already committed before phase config"
			)
		# The rows bound above are the rows this wave's prefill computes, and
		# the prefill forward is what creates each row's KDA state item. Catch
		# a binding that disagrees with the group map here, before the forward,
		# so the error names the seam instead of the decode step that misses
		# the state item.
		self._check_local_binding_matches_group(prefill_uuids, "prefill config")

		# Reached by EVERY rank — including ranks that serve no sequence this
		# wave — so the gather inside stays aligned. No-op for G==1; for G>1 it
		# hands the leader's page tables, prefix metadata and pool segment
		# pages to the rest of its serve group.
		#
		# M3 Slice 1c rides this same gather when the trajectory flag is on and
		# the group is head-sharded: the leader's G-wide snapshot arenas and
		# its frontier hits travel with the page tables, so no new collective
		# is introduced. The predicate is broadcast-identical (the flag is on
		# the batch's own sequences and the batch is replicated), so every rank
		# reaches the gather with the same expectation.
		self._traj.trajectory_arena_payload = {}
		self._traj.trajectory_mirrored_restores = {}
		mirror_trajectory = (
			self._decode_attn_tp_size() > 1
			and self.enable_prefix_cache
			and self._trajectory_commit_requested(
				[
					self.global_batch.get_sequence(uuid)
					for uuid in prefill_uuids
				]
			)
		)
		self._mirror_group_host_kv(
			my_prefill_uuids,
			self._prefix.prefix_sequence_states,
			trajectory=mirror_trajectory,
		)

		if self.rank == 0:
			logging.info(f"[PREFILL] Config completed: {(time.perf_counter() - start_time)*1000:.1f}ms")

	def _load_decode_model(self, comm=None) -> None:
		"""
		Load model for decoding phase. Must be called ONCE at the start of decode phase,
		BEFORE batch selection, so we know actual GPU KV capacity.

		Uses unified configure_decoding() which handles all scenarios:
		- Multi-node (world_size > 8): all experts persistent
		- Single-node with EP offloading: partial persistence based on offloading_ratio
		- Single-node without offloading: all experts persistent

		The per-rank buffer width is always the decode width
		(``_max_decode_rank_bsz``), the single source every decode configure
		and the decode admission cap (``_decode_padding_bsz``) derive from.

		Args:
			comm: NCCL communicator for distributed MoE forward.
		"""
		desired_key = self._desired_phase_key("decode")
		self._request_phase_instance_reuse()
		padding_bsz = self._max_decode_rank_bsz()
		self._check_decode_width_within_graph_buckets(padding_bsz)
		phase_changed, resumed = self._transition_phase(
			desired_key, comm=comm, max_num_seq=padding_bsz
		)
		if not phase_changed:
			return
		if resumed:
			self._remap_reused_kv_pool()
		else:
			self._log_phase_memory_ledger("after_configure_decoding")
			self._decode_padding_bsz = padding_bsz
		self._assert_phase_switch_clean(
			"prefill_to_decode", reused=resumed
		)
		self.set_phase("decode")
		self._install_decode_weight_copy_queue()
		if self.rank == 0:
			logging.info(
				"[DECODE] %s",
				"Reused decode instance activated" if resumed else "Model loaded for decoding phase",
			)
			if not resumed:
				self._log_decode_parameter_residency()

	def _install_decode_weight_copy_queue(self) -> None:
		"""Point the stopped H2D copy engine at the decode weight-copy schedule.

		Only a rank that offloads part of its routed experts has one. Every
		other rank installs an empty schedule, which also uninstalls the prefill
		lists, so its decode forwards arm no weight pass. Shared by the first
		decode build, by every later activation of a reused instance, whose task
		is the one its build produced, and by the persistent-phase activation.
		"""
		task = {}
		if self.weight_copy_task.get("routed_expert"):
			if self._streamed_sp8_h2d_installed:
				raise RuntimeError(
					f"Rank {self.rank}: streamed decode cannot replace an "
					"installed streamed-SP8 prefill pipeline; use resident-EP "
					"decode so the distributed weight schedule stays aligned"
				)
			task = self.weight_copy_task
		self.core_engine.set_weight_copy_queue(task)
		# Only start the H2D worker if there are experts to offload.
		if any(task.values()):
			self.core_engine.start_h2d_worker()

	def _max_decode_rank_bsz(self) -> int:
		"""Per-rank in-decode cap: the generic guard, lifted to the planner's
		per-node cap when a TP-group model (every rank holds the same rows)
		plans more slots than the guard — e.g. K3 H200 with sharded latent
		projections plans 160 rows per node.

		The legacy BATCHGEN_MAX_RANK_BSZ env name sizes the deepseek PSM
		padding separately; silently diverging values corrupt the padded
		buffer contract, so both names must agree when both are set."""
		explicit = os.environ.get("BATCHGEN_MAX_DECODE_RANK_BSZ")
		legacy = os.environ.get("BATCHGEN_MAX_RANK_BSZ")
		if explicit is not None and legacy is not None and explicit != legacy:
			raise RuntimeError(
				"BATCHGEN_MAX_DECODE_RANK_BSZ and BATCHGEN_MAX_RANK_BSZ "
				"must match when both are set"
			)
		planned = int(getattr(
			self.engine_config.Module_Batching_Config, "MoE_decoding_micro_batch_size", 0,
		) or 0)
		if self._decode_attn_tp_size() > 1 and planned > _MAX_DECODE_RANK_BSZ:
			return planned
		if explicit is None and legacy is None:
			# On a whole-model-graph deployment the bucket flag is the width
			# knob: raising --cuda-graph-max-bucket-size widens decode with it,
			# so capture above the admission width cannot exist (the bucket
			# set and this width share one source). An explicit env cap or a
			# TP-planner value still wins above.
			flag_bucket = int(getattr(
				getattr(self, "args", None), "cuda_graph_max_bucket_size", 0,
			) or 0)
			if (
				flag_bucket > _MAX_DECODE_RANK_BSZ
				and self._whole_model_decode_graphs_requested()
			):
				return flag_bucket
		return _MAX_DECODE_RANK_BSZ

	def _whole_model_decode_graphs_requested(self) -> bool:
		"""Whether this deployment requests whole-model decode graphs: GLM-5
		whole-model, GPT-OSS-120B or Kimi K2.5 under ``--enable-cuda-graph``.
		Eager deployments answer False, and nothing that keys on this
		predicate changes for them."""
		args = getattr(self, "args", None)
		if args is None or getattr(args, "disable_cuda_graphs", False):
			return False
		model_name = getattr(self, "model_name", None)
		if (
			"glm" in (model_name or "").lower()
			and self._glm5_whole_model_graph_requested_for_current_batch()
		):
			return True
		return cli_enables_whole_model_graph(
			model_name,
			enable_cuda_graph=getattr(args, "enable_cuda_graph", False),
		)

	def _decode_graph_bucket_sizes(self) -> list:
		"""The decode graph bucket ladder: the configured ladder with its top
		raised to the per-rank decode width (``_max_decode_rank_bsz``) when
		the width is wider, and the width itself always a bucket."""
		return decode_graph_bucket_sizes(
			int(self.args.cuda_graph_max_bucket_size),
			int(self.args.cuda_graph_num_buckets),
			self._max_decode_rank_bsz(),
		)

	def _decode_graph_top_bucket(self) -> Optional[int]:
		"""Largest decode graph bucket when whole-model graphs are requested,
		else None."""
		if not self._whole_model_decode_graphs_requested():
			return None
		return max(self._decode_graph_bucket_sizes())

	def _assert_decode_step_within_graph_buckets(self, max_rank_bsz: int) -> None:
		"""Decode-step invariant on a whole-model graph deployment: the step's
		globally-synced max per-rank batch fits the top bucket. The top is the
		captured ladder's when a graph is live, else the derived ladder's.
		Admission is bounded by the decode width the ladder covers, so a wider
		step is a regression; every rank sees the same width and raises
		together. Eager deployments return without a check."""
		bucketing = self._graph.whole_model_bucketing
		if (
			self._graph.whole_model_graph
			and self._graph.cuda_graph_manager is not None
			and bucketing is not None
		):
			top = int(max(bucketing.bucket_sizes))
		elif self._whole_model_decode_graphs_requested():
			top = int(max(self._decode_graph_bucket_sizes()))
		else:
			return
		if int(max_rank_bsz) <= top:
			return
		raise DecodeWidthExceedsGraphBucketError(
			f"Rank {self.rank}: decode step max per-rank batch {int(max_rank_bsz)} "
			"exceeds the largest whole-model decode graph bucket "
			f"{top} while decode admission is "
			f"bounded by the per-rank decode width {self._max_decode_rank_bsz()} "
			f"(admission cap _decode_padding_bsz="
			f"{getattr(self, '_decode_padding_bsz', None)}); refusing to run the "
			"step eager. Raise --cuda-graph-max-bucket-size to at least "
			f"{int(max_rank_bsz)} and report the admission path that exceeded "
			"the decode width."
		)

	def _check_decode_width_within_graph_buckets(self, decode_width: int) -> None:
		"""Configure-time invariant: on a deployment that requests whole-model
		decode graphs, the planned per-rank decode width fits the ladder that
		will be captured. Raises ``DecodeWidthExceedsGraphBucketError``;
		a no-op on eager deployments."""
		if not self._whole_model_decode_graphs_requested():
			return
		check_decode_width_within_buckets(
			decode_width, self._decode_graph_bucket_sizes()
		)

	def _log_decode_parameter_residency(self) -> None:
		"""One-line HBM breakdown of the decode model's resident parameters and
		buffers by top-level module kind (rank 0, once per decode config), so
		the non-expert residency — what competes with KV pool and KDA slots
		for concurrency — is visible next to the resident expert shard line."""
		model = getattr(self, "model", None)
		if model is None or not hasattr(model, "named_parameters"):
			return
		try:
			by_kind = {}
			seen = set()
			for name, t in list(model.named_parameters()) + list(model.named_buffers()):
				if not t.is_cuda or t.data_ptr() in seen:
					continue
				seen.add(t.data_ptr())
				parts = name.split(".")
				kind = parts[0]
				for i, part in enumerate(parts):
					if part == "layers" and i + 2 < len(parts):
						kind = parts[i + 2]
						break
					if part in ("embed_tokens", "lm_head", "norm"):
						kind = part
						break
				by_kind[kind] = by_kind.get(kind, 0) + t.numel() * t.element_size()
			total = sum(by_kind.values())
			items = sorted(by_kind.items(), key=lambda kv: -kv[1])
			logging.info(
				"[HBM] decode parameters/buffers resident on rank %s: %.2f GiB total | %s",
				self.rank, total / (1024 ** 3),
				", ".join(f"{k}={v / (1024 ** 3):.2f}" for k, v in items[:12]),
			)
		except Exception as e:  # informational only
			logging.debug("parameter residency breakdown failed: %r", e)

	def _remap_reused_kv_pool(self) -> None:
		"""P->D on the default lifecycle: map the KV pool back where it was.

		``initialize()`` on a reserved-range pool maps fresh physical memory
		under the SAME virtual addresses, zeroes it the way a freshly built
		pool hands out zeroed pages, and refreshes the page-pointer tables in
		place. Nothing is allocated, nothing is sized: the page count is the
		one the first decode configure measured.

		Called from inside the activation so the pool is live before the
		capacity snapshot, before decode admission allocates a page and before
		a captured graph dereferences a pool pointer.

		A decode instance can be built before readiness (Kimi-K3's startup
		pass), so its first activation precedes the first pool sizing. With no
		pool ever created in this process there is nothing to remap, and
		``_init_gpu_kv_with_actual_size()``, which the caller runs next from
		the decode phase, sizes it. A pool that existed and is gone is a
		lifecycle bug and still fails here.
		"""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			if not self._gpu_kv_pool_created:
				return
			raise RuntimeError(
				f"Rank {self.rank}: prefill->decode activated a reused decode "
				"instance with no GPU KV manager although this process already "
				"created a GPU KV pool; the reserved pool must survive every "
				"phase switch"
			)
		self._assert_kv_pool_addresses_are_fixed("prefill->decode")
		if manager.pool_is_mapped:
			# decode -> decode returns before the activation, so this is only
			# reached if a future caller activates an already-active instance.
			return
		start = time.perf_counter()
		manager.initialize()
		stats = manager.get_stats()
		logging.info(
			"[GPU-KV] rank=%d remapped the reserved KV pool in %.3fs (%d pages)",
			self.rank,
			time.perf_counter() - start,
			stats.num_total_pages,
		)

	def _assert_kv_pool_addresses_are_fixed(self, context: str) -> None:
		"""A reused decode instance requires an address-stable KV pool.

		Its captured graphs dereference the pool bases and page-pointer tables
		recorded at capture time. Releasing an allocator-backed pool under them
		would hand those addresses to the next allocation and the following
		replay would read someone else's memory, so an unsupported pool fails
		the switch instead of silently taking the old destroy/recreate path.
		"""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			return
		if not getattr(manager, "uses_fixed_address_pool", False):
			raise RuntimeError(
				f"Rank {self.rank}: {context} switch would release a GPU KV "
				"pool that is not held at fixed addresses while the decode "
				"instance is reused; its captured graphs bake the pool "
				"pointers, so the pool must be a reserved-range pool"
			)

	def _assert_kv_region_mapped(self, context: str) -> None:
		"""Before decode configure and replay the KV pool is mapped.

		Decode admission reads live all-rank capacity through it (#424) and the
		decode graphs bake its pointers, so the activation maps the pool back
		before either can run.

		The one exception is a process that has never created a pool: a decode
		instance built before readiness is activated before the first sizing,
		which ``_init_gpu_kv_with_actual_size()`` performs right after this
		check, before admission or capture can read it.
		"""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None and not self._gpu_kv_pool_created:
			return
		if manager is None or not manager.is_initialized:
			raise RuntimeError(
				f"Rank {self.rank}: {context} switch left the GPU KV region "
				"unmapped while phase instances are reused; decode admission "
				"reads live capacity through it and the decode graphs bake its "
				"pointers"
			)
		if not manager.pool_is_mapped:
			raise RuntimeError(
				f"Rank {self.rank}: {context} switch left the GPU KV pool's "
				"physical pages unmapped at their reserved addresses; a decode "
				"replay or capacity read would touch unmapped device memory"
			)

	def _assert_kv_region_released(self, context: str) -> None:
		"""After decode->prefill the KV pool's physical pages are back.

		The addresses stay reserved — that is what lets the activation map the
		same range again — but the pages themselves belong to the prefill phase
		until then. A pool still mapped here is the cleanup bug this lifecycle
		exists to avoid: the prefill phase would run with the whole decode KV
		pool charged against it.
		"""
		manager = self.gpu_paged_kv_cache_manager
		if manager is None:
			# Never created: a lifetime whose decode phase has not sized a pool
			# yet has nothing to release.
			return
		if manager.pool_is_mapped:
			raise RuntimeError(
				f"Rank {self.rank}: {context} switch left the GPU KV pool "
				"mapped; prefill needs its physical pages back, and the "
				"reserved addresses are what make the remap free"
			)

	def _request_phase_instance_reuse(self) -> bool:
		"""Ask the strategy manager to keep this process's phase instances.

		Returns whether the request was made. The manager still decides whether
		it can honour it: a rank that streams part of its routed experts has no
		fixed-address stacks to unmap and remap, so it keeps its legacy loader and
		leaves ``decode_instance`` None, and ``holds_instance("decode")`` stays
		false for the rest of the run — that rank rebuilds each phase
		exactly as it did before.

		This is not a mode and not a flag. Every deployment that alternates
		prefill and decode runs through here; legacy non-pool serving, which never
		alternated, was removed from the server.
		``--persistent-phase-instances`` has its own gate and never reaches here.
		"""
		pm = self.parallel_manager
		if not hasattr(pm, "reuse_phase_instances") or not hasattr(
			pm, "activate_decoding"
		):
			return False
		pm.reuse_phase_instances = True
		return True


	def _decode_graph_state_present(self) -> bool:
		"""Whether any captured decode graph state is currently held."""
		return any(
			getattr(self._graph, name, None) is not None
			for name in (
				"cuda_graph_manager",
				"whole_model_segment",
			)
		)


	def _finish_decode_interval(self) -> None:
		"""End-of-interval cleanup for the decode model that just ran.

		The rebuild path unregisters the model's FP8 weight pointers and
		deep-frees it, because its next decode configure builds a new model.
		Persistent and reused instances are never rebuilt; they are switched
		back in and their captured graphs replayed. Running this cleanup on a
		reused instance nulled the graph manager, so the next activation
		re-entered capture, and nulled every attention wrapper's fused
		Q-A/KV-A pointers, which that capture requires.
		"""
		if self._persistent_phase_enabled() or self.parallel_manager.holds_instance("decode"):
			return
		self._unregister_fp8_weights()
		self.deep_free_model_memory()

	def _assert_decode_graph_state_kept(self, context: str) -> None:
		"""A reused instance's captured graphs outlive every phase switch."""
		if self._graph.decode_graph_captures > 0 and not (
			self._decode_graph_state_present()
		):
			raise RuntimeError(
				f"Rank {self.rank}: {context} switch tore down the captured "
				"decode graph state of a reused instance; it is captured once "
				"per process and must survive every phase switch"
			)

	def _assert_decode_graph_capture_allowed(self) -> None:
		"""Capture-once on the reuse lifecycle, for every decode graph path.

		The building switch captures a reused instance's graphs and every
		later activation replays them. Getting here again means their state
		was torn down; recapturing would hide that, and it needs model state
		(fused attention pointers, MoE pointer tables) a torn-down instance no
		longer has. The criticality probe's armed recapture is the one
		exception: it releases only the graph state, never the model state.
		"""
		if (
			self._graph.decode_graph_captures > 0
			and self.parallel_manager.holds_instance("decode")
			and not getattr(self, "_criticality_probe_recapture", False)
		):
			raise RuntimeError(
				f"Rank {self.rank}: decode graph capture requested again for a "
				"reused decode instance; its graphs are captured once per "
				"process and their state must survive every phase switch"
			)

	def _record_decode_graph_capture(self) -> None:
		"""Count one completed decode graph capture for the live decode owner."""
		self._graph.decode_graph_captures = self._graph.decode_graph_captures + 1

	def _assert_decode_attention_weights_registered(self) -> None:
		"""A reused decode instance keeps its resident attention weights.

		The build registers them once; its graphs were captured against them,
		and an eager step or a capture reads the fused Q-A/KV-A pointers. Nothing
		on the reuse lifecycle may unregister them.
		"""
		missing = []
		for layer_idx, layer in enumerate(self.model.model.layers):
			wrapper = layer.self_attn
			if not getattr(wrapper, "persistent", False):
				continue
			if not hasattr(wrapper, "_fp8_qkv_a_proj"):
				continue
			if (
				wrapper._fp8_qkv_a_proj is None
				or getattr(wrapper, "_fp8_qkv_a_scale", None) is None
			):
				missing.append(layer_idx)
		if missing:
			raise RuntimeError(
				f"Rank {self.rank}: prefill->decode switch activated a reused "
				"decode instance whose fused Q-A/KV-A attention weights are "
				f"unregistered on {len(missing)} layer(s) (first {missing[:5]}); "
				"they must stay registered across the prefill phase"
			)

	def _persistent_phase_enabled(self) -> bool:
		"""True when --persistent-phase-instances keeps both phase instances."""
		if not getattr(self.args, "persistent_phase_instances", False):
			return False
		if self._traj.trajectory_book is None:
			raise RuntimeError(
				"--persistent-phase-instances requires the unified trajectory pool"
			)
		pm = self.parallel_manager
		if not hasattr(pm, "activate_decoding"):
			raise RuntimeError(
				f"--persistent-phase-instances is not supported by {type(pm).__name__}"
			)
		pm.persistent_phase_instances = True
		return True


	def _build_persistent_phase_instances(self) -> None:
		"""Build persistent phase instances and the decode pool before serving."""
		pm = self.parallel_manager
		if pm.decode_instance is not None:
			return
		start = time.perf_counter()
		if hasattr(pm, "set_comm"):
			pm.set_comm(self.comm)
		if hasattr(pm, "share_skeleton_on_device"):
			pm.share_skeleton_on_device()
		prefill_mode = (
			pm.default_prefill_moe_mode()
			if hasattr(pm, "default_prefill_moe_mode")
			else "streamed"
		)
		prefill_key = PhaseKey("prefill", prefill_mode, ())
		self._phase_profile_overrides = ()
		if hasattr(pm, "set_prefill_moe_mode"):
			pm.set_prefill_moe_mode(prefill_mode)
		self._transition_phase(prefill_key)
		self.set_phase("prefill")
		self._install_prefill_weight_copy_pipeline(False)
		decode_key = self._desired_phase_key("decode")
		decode_width = self._max_decode_rank_bsz()
		self._check_decode_width_within_graph_buckets(decode_width)
		_, resumed = self._transition_phase(
			decode_key,
			comm=self.comm,
			max_num_seq=decode_width,
		)
		if resumed:
			self._remap_reused_kv_pool()
		self._assert_phase_switch_clean("prefill_to_decode", reused=resumed)
		self.set_phase("decode")
		self._install_decode_weight_copy_queue()
		self._decode_padding_bsz = decode_width
		self._init_gpu_kv_with_actual_size()
		capture = self._glm5_whole_model_graph_requested_for_current_batch()
		# The probe runs the capture exactly once whether or not it can
		# measure it, so capture behavior is unchanged by the probe.
		self._probe_startup_transient(
			seam=_startup_transient.SEAM_PERSISTENT_BUILD,
			capture=self._capture_startup_decode_graphs if capture else None,
		)
		self._transition_phase(prefill_key)
		self.set_phase("prefill")
		self._install_prefill_weight_copy_pipeline(False)
		logging.info(
			"[PERSISTENT_PHASE] rank=%d instances built at startup in %.1fs graphs_captured=%s",
			self.rank,
			time.perf_counter() - start,
			capture,
		)

	def _startup_decode_graph_capture(
		self, next_key: PhaseKey
	) -> Optional[Callable[[], None]]:
		"""The decode graph capture the default lifecycle runs before readiness.

		Asked at the startup decode configure, right before the switch to
		``next_key``. Decode graphs belong to the decode instance that captured
		them. When the strategy manager keeps that instance across the switch
		(the exit-mode decision ``_transition_phase`` makes next is SUSPEND,
		i.e. ``holds_instance("decode")``), every later activation replays its
		graphs, so they are captured here, once per process, and the first
		decode configure finds ``_cuda_graph_manager`` set and captures nothing.
		A manager that retires its decode instance on the switch (build-always
		managers; a rank streaming part of its routed experts) rebuilds it, and
		its graphs, at every decode configure, so this returns None for it, as
		for every deployment that requests no whole-model decode graph.
		"""
		if not self._whole_model_decode_graphs_requested():
			return None
		live = self._live_phase_key
		if live is None or live.phase != "decode":
			raise RuntimeError(
				f"Rank {self.rank}: startup decode graph capture is decided at "
				f"the startup decode configure, which requires a live decode "
				f"phase; live phase key={live!r}"
			)
		if self._phase_exit_mode(live, next_key) is not ExitMode.SUSPEND:
			return None
		return self._capture_startup_decode_graphs

	def _capture_startup_decode_graphs(self) -> None:
		"""Capture the whole decode graph bucket ladder before readiness.

		No decode row exists yet: every rank binds an empty local batch and the
		global maximum is pinned to one row, which makes ``_setup_cuda_graphs``
		capture every bucket of ``_decode_graph_bucket_sizes``. Shared by both
		lifecycles that keep their decode instance.

		A capture that returns without graphs fails the startup: otherwise the
		capture silently moves back to the first decode configure, after
		readiness.

		Prewarm seam: a decode JIT prewarm (Triton/DeepGEMM) belongs right
		before ``_warmup_cuda_graphs`` below, where the KV pool is sized and
		the decode instance is live.
		"""
		captures = self._graph.decode_graph_captures
		self._sync_decode_moe_rank_counts([], reason="startup_graph_capture")
		self._current_decode_max_rank_batch_size = 1
		self._bind_decode_attention_metadata_for_graph_config([])
		self._warmup_cuda_graphs()
		if (
			self._graph.decode_graph_captures != captures + 1
			or not self._decode_graph_state_present()
		):
			raise RuntimeError(
				f"Rank {self.rank}: startup decode graph capture produced no "
				f"decode graphs (captures {captures} -> "
				f"{self._graph.decode_graph_captures}, graph state "
				f"present={self._decode_graph_state_present()}); the graphs "
				"are requested and the decode instance is kept, so they must "
				"be captured before readiness"
			)

	def _probe_startup_transient(
		self, *, seam: str, capture: Optional[Callable[[], None]]
	) -> None:
		"""Measure the startup decode transient at the per-rank cap; publish it.

		Runs at the startup decode configure on every rank: after
		``configure_decoding`` at the cap and after the GPU KV pool is sized,
		while decode is the live phase, before readiness. Collective when CUDA
		and torch.distributed are up (one ``all_gather_object``).

		Flavors:
		- eager: the cap-sized synthetic decode step the parallel strategy
		  manager provides as ``startup_decode_probe_step(padding_bsz)``. The
		  eager decode step is otherwise inline in the decode loop and needs
		  admitted rows, so a manager without the hook is reported, by name,
		  as unavailable rather than approximated.
		- capture: ``capture`` is the decode graph capture this lifecycle
		  performs at startup (None when it captures none). It is invoked
		  exactly once, measured or not.

		Report-only: the result is logged and stored in
		``self.startup_transient`` for the deployment-capacity publication;
		nothing reads it for sizing.
		"""
		padding_bsz = getattr(self, "_decode_padding_bsz", None)
		if getattr(self, "_phase", None) != "decode" or not padding_bsz:
			raise RuntimeError(
				f"Rank {self.rank}: startup transient probe requires the decode "
				f"phase configured at its padding; phase={getattr(self, '_phase', None)!r} "
				f"padding_bsz={padding_bsz!r}"
			)
		transient = _startup_transient
		cuda = torch.cuda if torch.cuda.is_available() else None
		pm = self.parallel_manager
		step = getattr(pm, "startup_decode_probe_step", None)
		if cuda is None:
			eager = transient.StartupTransient.missing(
				transient.FLAVOR_EAGER, seam, padding_bsz, "CUDA is unavailable"
			)
		elif step is None:
			eager = transient.StartupTransient.missing(
				transient.FLAVOR_EAGER, seam, padding_bsz,
				f"{type(pm).__name__} provides no cap-sized synthetic decode step",
			)
		else:
			try:
				eager = transient.measure_eager(
					lambda: step(padding_bsz),
					cuda=cuda, device=self.local_rank, seam=seam, padding_bsz=padding_bsz,
				)
			except Exception as exc:
				# The synthetic step is a measurement, not startup work: while
				# the value is report-only, a probe defect on one model must
				# not kill every deployment of it. Published by name instead;
				# once the reservation feeds the pool budget, this becomes a
				# startup failure.
				transient.logger.exception(
					"[STARTUP_TRANSIENT] rank=%s eager probe step failed", self.rank
				)
				eager = transient.StartupTransient.missing(
					transient.FLAVOR_EAGER, seam, padding_bsz,
					f"probe step failed: {type(exc).__name__}: {exc}",
				)
		if capture is None:
			captured = transient.StartupTransient.missing(
				transient.FLAVOR_CAPTURE, seam, padding_bsz,
				"this lifecycle captures no decode graph at startup",
			)
		elif cuda is None:
			capture()
			captured = transient.StartupTransient.missing(
				transient.FLAVOR_CAPTURE, seam, padding_bsz, "CUDA is unavailable"
			)
		else:
			captured = transient.measure_capture(
				capture, cuda=cuda, device=self.local_rank, seam=seam,
				padding_bsz=padding_bsz,
			)
		local = [eager, captured]
		transient.log_ledger_line(f"rank={self.rank}", local)
		if cuda is not None and dist.is_initialized():
			gathered = [None] * self.world_size
			dist.all_gather_object(gathered, local)
			deployment = transient.reduce_across_ranks(gathered)
		else:
			deployment = local
		if self.rank == 0:
			transient.log_ledger_line("deployment", deployment)
		self.startup_transient = [item.as_dict() for item in deployment]

	def _mark_phase_end(self, phase: str) -> None:
		# [phase, running since (None while idle), busy seconds so far]
		self._phase_end = [phase, time.perf_counter(), 0.0]

	def _pause_phase_switch_clock(self) -> None:
		"""Stop counting while the pool is idle and waits for new admissions."""
		end = getattr(self, "_phase_end", None)
		if end is not None and end[1] is not None:
			end[2] += time.perf_counter() - end[1]
			end[1] = None

	def _resume_phase_switch_clock(self) -> None:
		end = getattr(self, "_phase_end", None)
		if end is not None and end[1] is None:
			end[1] = time.perf_counter()

	def _log_phase_memory_ledger(self, label: str) -> None:
		"""One line per phase-switch stage: who holds device memory right now.

		The rebuild path tears a phase instance down and stages the next one, so
		"every cycle gives back what it took" is an invariant with no owner: a
		slice retained per switch only shows up as an OOM several cycles later,
		at whichever allocation happens to be the largest (the KV pool remap's
		``cuMemCreate``). Torch's own counters cannot see that slice — the KV
		pool is CUDA-VMM, the captured graphs own private pools, and the weight
		buffers are C++ — so the driver's free/total is reported beside them,
		and the holders that survive a model are named explicitly.

		Cheap: four driver queries and attribute reads, once per switch stage,
		on rank 0 only. No synchronize (the callers that need one do it) and no
		``.item()``.
		"""
		if self.rank != 0 or not torch.cuda.is_available():
			return
		free_bytes, total_bytes = torch.cuda.mem_get_info(self.local_rank)
		gib = 1024 ** 3
		pm = getattr(self, "parallel_manager", None)
		kda_slots, kda_live = self._traj.kda_stats()
		gpu_kv = getattr(self, "gpu_paged_kv_cache_manager", None)
		logging.info(
			"[PHASE_MEM] rank=%d stage=%s switch=%d driver_used=%.2f "
			"driver_total=%.2f torch_alloc=%.2f torch_reserved=%.2f "
			"model=%d pm_model=%d resident_ep=%d sp8_buffer=%d decode_graph=%d "
			"kv_pool_mapped=%s kda_slots=%d kda_live=%d",
			self.rank,
			label,
			getattr(self, "_phase_mem_switch_index", 0),
			(total_bytes - free_bytes) / gib,
			total_bytes / gib,
			torch.cuda.memory_allocated(self.local_rank) / gib,
			torch.cuda.memory_reserved(self.local_rank) / gib,
			int(getattr(self, "model", None) is not None),
			int(getattr(pm, "model", None) is not None),
			int(bool(getattr(pm, "_resident_ep_built", False))),
			int(getattr(pm, "_streamed_sp8_buffer", None) is not None),
			int(getattr(pm, "_decode_graph", None) is not None),
			getattr(gpu_kv, "pool_is_mapped", "na"),
			kda_slots,
			kda_live,
		)

	def _log_phase_switch(self, phase: str) -> None:
		"""Per-rank busy time from the previous phase's compute end to this phase's start."""
		previous = getattr(self, "_phase_end", None)
		self._phase_end = None
		if previous is None or previous[0] == phase:
			return
		seconds = previous[2]
		if previous[1] is not None:
			seconds += time.perf_counter() - previous[1]
		logging.info(
			"[PHASE_SWITCH] rank=%d direction=%s_to_%s seconds=%.3f persistent=%s",
			self.rank,
			previous[0],
			phase,
			seconds,
			self._persistent_phase_enabled(),
		)

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

		# The pool is sized from what is free once the decode model is in
		# place, so sizing it while the prefill phase is still resident
		# permanently charges that phase's bytes against decode capacity. On
		# the reuse lifecycle the size is then kept for the whole process,
		# which is exactly how an earlier startup-time sizing attempt ended up
		# with roughly half the pages the per-batch path got. Sizing therefore
		# only happens from the decode phase, after the prefill ring and the
		# prefill model have been released.
		if getattr(self, "_phase", None) != "decode":
			raise RuntimeError(
				f"Rank {self.rank}: GPU KV pool sizing attempted in phase "
				f"{self._phase!r}; it must run from the decode phase so the "
				"measurement cannot charge prefill-only residency against the "
				"pool it then keeps"
			)

		# A reserved-range pool is never re-created: its virtual range is still
		# held and its page count was fixed by the sizing below on the first
		# decode configure, so a later decode phase maps the same pages back
		# instead of sizing a second, different pool. The reuse lifecycle has
		# already remapped inside its activation; this covers a rank that
		# rebuilds its decode instance, and keeps every rank's num_total_pages
		# identical, which the cross-rank capacity snapshot requires.
		manager = self.gpu_paged_kv_cache_manager
		if manager is not None and getattr(manager, "uses_fixed_address_pool", False):
			self._remap_reused_kv_pool()
			return

		# Measure-once-cache: once startup decode residency has been
		# measured, every later pool (re)creation reuses the cached size —
		# never a fresh measurement (contract invariant 1: the pool is
		# identical across decode rounds of the same server).
		if self._gpu_kv_cached_pool_gb is not None:
			self.gpu_kv_cache_size_gb = self._gpu_kv_cached_pool_gb
			# The previous pool's segments already returned to CUDA at destroy
			# (dedicated MemPool, whole-segment free). This flushes the
			# GENERAL cache's all-free segments (prefill transients) too, so
			# the full-size cudaMalloc below sees the maximum free memory. The
			# collection first drops tensors still held only by cycles, so
			# empty_cache can return their segments.
			torch.cuda.synchronize(self.torch_device)
			gc.collect()
			torch.cuda.empty_cache()
			# Later residency may consume the fraction's reserved slack, but
			# the verified pool itself must still fit. Name any shortfall
			# instead of silently shrinking the deployment's capacity.
			free_mem_gb = torch.cuda.mem_get_info(self.local_rank)[0] / (1024 ** 3)
			KVCacheManager.check_cached_pool_fits(
				self.gpu_kv_cache_size_gb, free_mem_gb, rank=self.rank
			)
			if self.rank == 0:
				logging.info(
					f"[GPU-KV] Cached pool size: {self.gpu_kv_cache_size_gb:.2f} GB "
					f"(startup budget; {free_mem_gb:.2f} GB free)"
				)
			self._initialize_gpu_kv_manager_fixed_size()
			if self.rank == 0:
				stats = self.gpu_paged_kv_cache_manager.get_stats()
				logging.info(
					f"[GPU-KV] Initialized: {self.gpu_kv_cache_size_gb:.2f} GB, "
					f"{stats.num_total_pages} pages"
				)
			return

		# First startup decode entry: measure after weights are resident. The
		# remaining fraction is reserved for graph and eager allocations; it
		# is not counted as KV capacity. Every later entry uses the cache.
		torch.cuda.synchronize(self.torch_device)
		torch.cuda.empty_cache()

		free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(self.local_rank)
		free_mem_gb = free_mem_bytes / (1024 ** 3)
		total_mem_gb = total_mem_bytes / (1024 ** 3)
		used_mem_gb = total_mem_gb - free_mem_gb

		local_pool_gb = provisional_pool_gb(
			total_mem_gb, self.gpu_memory_frac, used_mem_gb, floor_gb=None
		)
		self._verify_deployment_max_model_len(local_pool_gb)
		# Identical page counts are required by decode admission. The tightest
		# rank supplies the common pool, rather than rank zero's residency.
		size_tensor = torch.tensor([local_pool_gb], dtype=torch.float64, device=self.torch_device)
		dist.all_reduce(size_tensor, op=dist.ReduceOp.MIN)
		self.gpu_kv_cache_size_gb = float(size_tensor.item())
		# Freeze the measured startup pool for every mode before capture. A
		# lazy capture must not change it after L has been published. The
		# fraction's slack covers graph/eager allocations until startup can
		# measure those allocations at the per-rank cap.
		self._gpu_kv_sizing_provisional = False

		# Initialize GPU KV manager with actual size
		self._initialize_gpu_kv_manager_fixed_size()
		self._gpu_kv_cached_pool_gb = self.gpu_kv_cache_size_gb

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
		if self._live_phase_key is None or self._live_phase_key.phase != "decode":
			raise RuntimeError("decode admission requires a live decode phase")
		self.parallel_manager.admit(
			self._live_phase_key,
			AdmitContext(
				uuids=list(decode_uuids),
				local_indices=list(local_decode_indices),
			),
		)

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
		# Every rank participates, including ranks with no locally owned rows, so
		# one rank cannot mutate its allocator after another rank reports failure.
		alloc_ok = self._allocate_gpu_kv_two_page_buffer(
			local_decode_indices,
			load_from_host=True,
			collective_preflight=True,
		)
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
				f"Rank {self.rank}: collective GPU KV allocation preflight failed for "
				f"{len(local_decode_indices)} locally owned sequences; "
				"decode admission exceeded the available replica capacity"
			)
		
		if self.rank == 0:
			logging.info(f"[DECODE] Config completed: {(time.perf_counter() - start_time)*1000:.1f}ms, {len(decode_uuids)} sequences")


	def _release_host_kv_pages_for_batch(self, uuids: List[str]) -> None:
		"""Release host KV pages for completed sequences owned by this rank.
		
		NOTE: This function only releases HOST KV pages. GPU KV pages should be
		released separately by calling _release_gpu_kv_pages() BEFORE this function.
		"""
		if not uuids:
			return
		
		# Keep lifecycle mutations on the worker-owned view.  For DSA this is the
		# DualHostKVCoordinator; the engine keeps only its primary view for the
		# attention wrapper and must not be used to release a sequence by itself.
		worker_view = getattr(self, "host_paged_kv_worker_view", None)
		if worker_view is None:
			worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		if worker_view is None:
			logging.warning("Host paged KV worker view is unavailable")
			return
		
		my_uuids = [uuid for uuid in uuids if uuid in self._uuid_to_local_map]

		# Host KV is ONE run-namespaced per-node SHARED shm region keyed
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

			# Release host KV pages through the mirrored coordinator when DSA is
			# active.  It owns both primary and auxiliary page tables.
			worker_view.release_sequence_pages(global_sequence_ids)
			# A model with an aux region OUTSIDE the coordinator (K3 keeps its
			# indexer keys there) releases it explicitly; the dual coordinator
			# already mirrored its own aux above.
			aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)
			if aux_view is not None and not isinstance(
				worker_view, DualHostKVCoordinator
			):
				aux_view.release_sequence_pages(global_sequence_ids)
			self._release_prefix_cache_attachments(global_sequence_ids)

		# The G-1 followers of a serve group never touched the shared pages, but
		# they do hold the leader's page table MIRRORED into their own view plus
		# a mirrored prefix state. Drop those rank-local records too, or the next
		# register of the same global_idx finds a stale row. unregister_sequences
		# only edits the view's own HostKVPageTable; the pages themselves were
		# freed exactly once, by the leader above. A mirrored prefix state has
		# lookup_result=None (no attachment handle, so no coordinator call) and a
		# follower owns no pool segment refs, so the shared release path inside
		# _release_prefix_cache_attachments is a no-op here. G==1: the local-map
		# owner IS the host-KV owner, so this list is always empty.
		mirror_release_uuids = [
			uuid for uuid in my_uuids
			if not self._owns_host_kv(self.global_batch.get_sequence(uuid))
		]
		if mirror_release_uuids:
			mirror_global_ids = [
				self.global_batch.get_sequence(uuid).global_idx
				for uuid in mirror_release_uuids
			]
			for view in self._host_views_by_group().values():
				view.unregister_sequences(mirror_global_ids)
			# An aux region OUTSIDE the prefix groups (K3 indexer keys) holds
			# follower rows too; dropping only the group views would leave
			# stale aux rows behind for the next use of the same global id.
			aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)
			if aux_view is not None and not isinstance(
				getattr(self, "host_paged_kv_worker_view", None),
				DualHostKVCoordinator,
			):
				aux_view.unregister_sequences(mirror_global_ids)
			self._release_prefix_cache_attachments(mirror_global_ids)

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

	def _emit_prefix_cache_metric(self, phase: str, **fields) -> None:
		if not self.prefix_cache_debug_stats:
			return
		from batchgen.prefix_reuse.metrics import emit_prefix_cache_metric

		emit_prefix_cache_metric(
			phase=phase,
			rank=self.rank,
			**fields,
		)

	def _host_views_by_group(self) -> dict:
		"""One Host-KV worker view per prefix-cache KV group.

		A DSA model keeps group 0 (MLA latent) in the primary region and group 1
		(indexer keys) in the auxiliary one, so every prefix-cache Host page
		operation has to run on both. Every other model has one region.
		"""
		if isinstance(self.host_paged_kv_worker_view, DualHostKVCoordinator):
			views = self.host_paged_kv_worker_view.views_by_group()
		else:
			view = getattr(self, "host_paged_kv_worker_view", None)
			if view is None:
				view = self.core_engine.host_paged_kv_worker_view
			views = {0: view}
		runtime = self.prefix_cache_runtime_config
		if runtime is not None and len(runtime.group_specs) != len(views):
			raise RuntimeError(
				"prefix cache groups do not match the Host KV regions: "
				f"groups={len(runtime.group_specs)}, views={len(views)}"
			)
		return views

	def _prefix_host_resource_fields(self) -> dict[str, int]:
		from batchgen.prefix_reuse.metrics import host_resource_fields

		return host_resource_fields(
			coordinator=self.prefix_cache_coordinator,
			worker_view=self.core_engine.host_paged_kv_worker_view,
		)

	def _maybe_emit_prefix_drain_metric(self) -> None:
		"""Emit one prefix_drain row when the live sequence set drains to empty.

		Called from _report_completion, which every completion path runs after
		the sequence reached COMPLETED and its Host pages and prefix
		attachments were released, so the row records active_attachments at the
		drain moment. Live means any status other than COMPLETED: a QUEUEING,
		IN_PREFILL, PREFILLED, IN_DECODE, ON_HOLD or EVICTED sequence can still
		hold or regain an attachment. Edge-triggered: one row per drain,
		re-armed by _admit_sequences_from_message.
		"""
		if not (self.enable_prefix_cache and self.prefix_cache_debug_stats):
			return
		if self._prefix.prefix_drain_metric_emitted:
			return
		if not self.global_batch.all_completed():
			return
		self._prefix.prefix_drain_metric_emitted = True
		self._emit_prefix_cache_metric(
			"prefix_drain",
			**self._prefix_host_resource_fields(),
			# M3 Slice 2: live attachments must be zero at the drain, beside
			# the coordinator's own active_attachments. Absent on a run that
			# never turned the trajectory flag on.
			**self._trajectory_restore_fields(),
		)

	def _close_prefix_materialization(
		self, materialization, *, empty_cuda_cache: bool
	) -> None:
		try:
			materialization.close(empty_cuda_cache=empty_cuda_cache)
		finally:
			if self.prefix_cache_debug_stats:
				fields = materialization.take_metrics_fields()
				if fields is not None:
					self._emit_prefix_cache_metric(
						"prefix_h2d",
						**fields,
					)

	def _close_active_prefix_materializations(self) -> None:
		"""Release temporary prefix-prefill GPU buffers after success or error."""

		from batchgen.models.wrappers.attention import AttnWrapperBase as _AWB

		_AWB.prefill_prefix_materialization = None
		if not self._prefix.active_prefix_materializations:
			return
		try:
			_AWB.retire_pending_prefill_offloads(
				device=self.torch_device,
				reason="prefix materialization cleanup",
			)
		except Exception:
			logging.exception(
				"Rank %s failed to retire prefix prefill offloads",
				self.rank,
			)
		while self._prefix.active_prefix_materializations:
			materialization = self._prefix.active_prefix_materializations.pop()
			try:
				self._close_prefix_materialization(
					materialization,
					empty_cuda_cache=True,
				)
			except Exception:
				logging.exception(
					"Rank %s failed to close prefix materialization",
					self.rank,
				)

	# ============ IN-WAVE PREFIX POOL (batchgen/prefix_reuse) ============

	def _prefill_prefix_pool_enabled(self) -> bool:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefill_prefix_pool_enabled(
			self,
		)

	def _pool_prefill_group_safe(self) -> bool:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._pool_prefill_group_safe(
			self,
		)

	def _prefill_pool_group(self):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefill_pool_group(
			self,
		)

	def _prefill_pool_free_state_slots(self) -> int:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefill_pool_free_state_slots(
			self,
		)

	def _next_pool_segment_sid(self) -> int:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._next_pool_segment_sid(
			self,
		)

	def _prefix_rank_packing_mode(self, seqs) -> bool:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefix_rank_packing_mode(
			self,
			seqs,
		)

	def _prefix_rank_cost_profile_disabled(self, seqs) -> bool:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefix_rank_cost_profile_disabled(
			self,
			seqs,
		)

	def _cached_prefix_rank_cost_profile(self, disabled: bool=False):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._cached_prefix_rank_cost_profile(
			self,
			disabled,
		)

	def _stamp_prefix_prefill_order(self, uuids: List[str]) -> None:
		"""Order a ``packing_v2`` admission by prompt prefix, not by arrival.

		Prefill waves are cut by free Host-KV pages in candidate order, and a
		wave releases its prefix pool before the next one starts. A prefix
		family split across that cut therefore recomputes its shared prefix.
		Sorting the admission lexicographically by prompt token ids is a
		prefix-tree DFS order, so a family stays contiguous and the cut falls
		between families instead of inside one.

		Identical on every rank: admission is broadcast from rank 0 and every
		rank tokenizes, so the prompt ids read here are already replicated.
		``global_idx`` grows across admissions and the base is this
		admission's smallest one, so earlier admissions still order first.
		"""
		seqs = [self.global_batch.get_sequence(uuid) for uuid in uuids]
		seqs = [seq for seq in seqs if seq is not None]
		if not seqs or not self._prefix_rank_packing_mode(seqs):
			return
		base = min(seq.global_idx for seq in seqs)
		ordered = sorted(
			seqs,
			key=lambda seq: (
				self._trajectory_tokens(seq, length=seq.prompt_length).tolist(),
				seq.global_idx,
			),
		)
		for offset, seq in enumerate(ordered):
			seq.prefill_order = base + offset
		if self.rank == 0:
			logging.info(
				f"[PREFIX_ORDER] Stamped prefill_order on {len(ordered)} sequences"
			)

	def _assign_ranks_for_prefix_sharing(
		self, uuids: List[str], *, current_wave: bool = False
	) -> None:
		"""Keep prompts sharing a prefix on one DP rank: each rank has its own pool.

		``packing_v2`` batches balance predicted executor seconds instead of
		computed tokens, and leave a re-entering sequence on the rank that still
		holds its Host KV: that sequence only pins load onto its rank.
		"""
		from batchgen.prefix_reuse.wave_plan import assign_ranks_for_sharing

		pending = set(uuids)
		seqs = [self.global_batch.get_sequence(uuid) for uuid in uuids]
		seqs = [seq for seq in seqs if seq is not None]

		def prompt_tokens(seq):
			# Prompts live in the node-shared trajectory book after admission
			# (a re-entry's reconstructed prompt included), so every rank reads
			# the same tokens.
			return self._trajectory_tokens(seq, length=seq.prompt_length)

		packing = None
		if self._prefix_rank_packing_mode(seqs):
			from batchgen.prefix_reuse.rank_packing import (
				pack_wave_ranks,
				prompt_score_from_length,
			)

			profile = self._cached_prefix_rank_cost_profile(
				self._prefix_rank_cost_profile_disabled(seqs)
			)
			page_tokens = int(self.prefix_cache_runtime_config.group_specs[0].raw_page_tokens)
			chunk_tokens = int(
				self.engine_config.Module_Batching_Config.prefill_micro_batch_token_cap
			)

			def pinned_load(seq):
				# Length-only on purpose: a pinned sequence's token storage may
				# live on another node (its Host KV migrated away), and the
				# score of a lone prompt depends only on its length anyway.
				return prompt_score_from_length(
					int(seq.prompt_length),
					block_tokens=page_tokens,
					chunk_tokens=chunk_tokens,
					profile=profile,
				)

			# Only sequences that still owe a prefill load a rank. Completed or
			# already-prefilled ones from earlier batches do not, and counting
			# them skews the packing (one rank received a whole admission).
			owes_prefill = (
				SequenceStatus.QUEUEING, SequenceStatus.IN_PREFILL, SequenceStatus.EVICTED
			)
			fixed = []
			if not current_wave:
				for seq in self.global_batch:
					if (
						seq.uuid in pending or seq.assigned_rank is None
						or seq.status not in owes_prefill
					):
						continue
					fixed.append((seq.assigned_rank, pinned_load(seq)))
			queued = []
			for seq in seqs:
				reentry = (
					seq.status == SequenceStatus.EVICTED
					or getattr(seq, "evicted_token_ids", None) is not None
				)
				if reentry:
					# A re-entry stays on the rank that holds its Host KV; an
					# ownerless re-entry cannot be packed identically anywhere.
					if seq.assigned_rank is None:
						raise RuntimeError(
							f"re-entering sequence {seq.uuid} has no Host-KV owner rank"
						)
					fixed.append((seq.assigned_rank, pinned_load(seq)))
				else:
					queued.append(seq)
			seqs = queued
			prompts = [prompt_tokens(seq).tolist() for seq in seqs]
			packing = pack_wave_ranks(
				prompts,
				world_size=self.world_size,
				block_tokens=page_tokens,
				chunk_tokens=chunk_tokens,
				profile=profile,
				fixed=fixed,
			)
			ranks = packing.ranks
			# Every rank packs the same wave from synced metadata. A divergence
			# would hand two ranks different owners for one sequence, so compare
			# a digest rather than discover it as a missing local slot later.
			# The input fingerprint tells an input divergence (unsynced prompts or
			# pinned loads) apart from a non-deterministic packer.
			mod = (1 << 61) - 1
			digest = 0
			for index, rank in enumerate(ranks):
				digest = (digest * 1_000_003 + (index + 1) * 131 + rank + 1) % mod
			hasher = hashlib.blake2b(digest_size=7)
			for seq in seqs:
				hasher.update(prompt_tokens(seq).numpy().tobytes())
				hasher.update(int(seq.prompt_length).to_bytes(8, "little"))
			for rank, load in fixed:
				hasher.update(f"{rank}:{load:.9e};".encode())
			inputs = int.from_bytes(hasher.digest(), "little")
			check = torch.tensor(
				[inputs, digest], dtype=torch.int64, device=self.torch_device
			)
			gathered = [torch.zeros_like(check) for _ in range(self.world_size)]
			dist.all_gather(gathered, check)
			if len({int(item[1].item()) for item in gathered}) != 1:
				same_inputs = len({int(item[0].item()) for item in gathered}) == 1
				if not same_inputs:
					# Every rank saw the digests differ, so this extra collective is
					# reached everywhere. Name the first prompt whose tokens differ,
					# with its trajectory slot on each rank.
					per_prompt = torch.tensor(
						[
							int.from_bytes(hashlib.blake2b(
								prompt_tokens(seq).numpy().tobytes(),
								digest_size=7,
							).digest(), "little")
							for seq in seqs
						] + [getattr(seq, "_buffer_slot", -1) for seq in seqs],
						dtype=torch.int64, device=self.torch_device,
					)
					views = [torch.zeros_like(per_prompt) for _ in range(self.world_size)]
					dist.all_gather(views, per_prompt)
					n = len(seqs)
					for index in range(n):
						hashes = {int(view[index].item()) for view in views}
						if len(hashes) > 1:
							logging.error(
								"[PREFIX_POOL] rank %d: first diverging prompt #%d uuid=%s "
								"prompt_length=%d slots_by_rank=%s differing_ranks_by_hash=%d",
								self.rank, index, seqs[index].uuid, seqs[index].prompt_length,
								[int(view[n + index].item()) for view in views], len(hashes),
							)
							break
				raise RuntimeError(
					"packing_v2 prefix rank assignment diverged across ranks "
					f"({'identical inputs: packer non-determinism' if same_inputs else 'inputs differ across ranks'}): "
					f"rank {self.rank} digest {digest} inputs {inputs} "
					f"prompts {len(prompts)} tokens {sum(map(len, prompts))} fixed {len(fixed)}"
				)
		else:
			loads = [0] * self.world_size
			if not current_wave:
				for seq in self.global_batch:
					if seq.uuid in pending or seq.assigned_rank is None:
						continue
					loads[seq.assigned_rank] += int(getattr(seq, "prompt_length", 0) or 0)
			ranks = assign_ranks_for_sharing(
				[prompt_tokens(seq).tolist() for seq in seqs],
				world_size=self.world_size,
				block_tokens=int(self.prefix_cache_runtime_config.group_specs[0].raw_page_tokens),
				initial_loads=loads,
			)
		for seq, rank in zip(seqs, ranks):
			if current_wave and seq.assigned_rank == self.rank and rank != self.rank:
				local_idx = release_query_book_slot(
					seq.uuid,
					query_book=self.query_book,
					table=self.seq_table,
				)
				if local_idx is None:
					raise RuntimeError(
						f"Rank {self.rank}: missing local slot for reassigned {seq.uuid}"
					)
			self.global_batch.assign_rank(seq.uuid, rank)
		if self.rank == 0:
			counts = [0] * self.world_size
			for rank in ranks:
				counts[rank] += 1
			if packing is not None:
				logging.info(
					"[PREFIX_POOL] packing_v2 rank assignment: %d sequences, per-rank %s, "
					"predicted seconds %s, round %d",
					len(seqs), counts,
					[round(value, 4) for value in packing.predicted],
					packing.round_used,
				)
			else:
				logging.info(
					"[PREFIX_POOL] %s rank assignment: %d sequences, per-rank %s",
					"wave" if current_wave else "sharing-aware", len(seqs), counts,
				)

	def _materialized_prefix_hits(self, seqs, *, page_tokens: int, group):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._materialized_prefix_hits(
			self,
			seqs,
			page_tokens=page_tokens,
			group=group,
		)

	def _unmaterializable_frontier(self, seq, *, boundary: int, owning_group_id: int, page_tokens: int) -> Optional[str]:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._unmaterializable_frontier(
			self,
			seq,
			boundary=boundary,
			owning_group_id=owning_group_id,
			page_tokens=page_tokens,
		)

	def _plan_prefill_pool_wave(self, uuids, prefix_states):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._plan_prefill_pool_wave(
			self,
			uuids,
			prefix_states,
		)

	def _prefill_pool_psm_layout(self):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefill_pool_psm_layout(
			self,
		)

	def _prefill_pool_page_dims(self, geometry) -> tuple:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefill_pool_page_dims(
			self,
			geometry,
		)

	def _prefill_pool_page_bytes(self) -> int:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefill_pool_page_bytes(
			self,
		)

	def _measure_prefill_prefix_kv_budget(self) -> None:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._measure_prefill_prefix_kv_budget(
			self,
		)

	def _construct_prefill_pool(self, wave) -> None:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._construct_prefill_pool(
			self,
			wave,
		)

	def _check_pool_plan_identical(self, plan, group) -> None:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._check_pool_plan_identical(
			self,
			plan,
			group,
		)

	def _prefill_pool_layout(self, geometry) -> tuple:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefill_pool_layout(
			self,
			geometry,
		)

	def _replace_prefix_lookup_hits(self, seqs, prefix_states, chain_tokens, retain=frozenset(), restore_boundaries=None) -> None:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._replace_prefix_lookup_hits(
			self,
			seqs,
			prefix_states,
			chain_tokens,
			retain,
			restore_boundaries,
		)

	def _settle_trajectory_restore_plan(self, seqs, drops, default_reason) -> None:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._settle_trajectory_restore_plan(
			self,
			seqs,
			drops,
			default_reason,
		)

	def _prefill_host_view_is_dual(self) -> bool:
		# S5a rewrite carve-out #5 (spec §S5b F3 block): the engine module must
		# stay importable without the CUDA/native kv_cache chain, so the
		# DualHostKVCoordinator isinstance stays on the worker (this file
		# already imports the class) and the moved body reads this predicate
		# through the rebound self.
		return isinstance(self.host_paged_kv_worker_view, DualHostKVCoordinator)

	def _prefill_pool_segment_host_pages(self) -> int:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefill_pool_segment_host_pages(
			self,
		)

	def _allocate_prefill_pool_host_pages(self, host_views, prefix_states) -> None:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._allocate_prefill_pool_host_pages(
			self,
			host_views,
			prefix_states,
		)

	def _materialized_pool_layers(self):
		"""The attention layers that materialize, in Host physical order.

		The n-th materializing layer must be pool row n: both the pool and the
		Host cache hold one dense row per cached attention layer, in engine
		order, and the load addresses them by that one index. Asserted rather
		than assumed -- a model whose two orders disagree would load every
		layer's prefix into the wrong row, and attention would read plausible
		numbers from the wrong depth.
		"""
		from batchgen.models.wrappers.attention import AttnWrapperBase

		layer_map = AttnWrapperBase.prefill_pool_layer_map
		if not layer_map:
			raise RuntimeError(
				"a materialized segment needs the pool's engine-layer map"
			)
		layers = []
		for engine_layer, decoder_layer in enumerate(self.model.model.layers):
			attn = decoder_layer.self_attn
			if not hasattr(attn, "materialize_pool_segment"):
				continue
			row = int(layer_map[engine_layer])
			if row < 0:
				continue
			if row != len(layers):
				raise RuntimeError(
					f"engine layer {engine_layer} is pool row {row} but the "
					f"{len(layers)}th materializing layer; the pool and Host "
					"orders must agree for a prefix load"
				)
			layers.append(attn)
		return layers

	def _materialized_segment_loader(self, wave):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._materialized_segment_loader(
			self,
			wave,
		)

	def _prefix_pool_index_key_carrier(self, aux_geometry) -> int:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefix_pool_index_key_carrier(
			self,
			aux_geometry,
		)

	def _materialized_pool_index_key_layers(self, pool):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._materialized_pool_index_key_layers(
			self,
			pool,
		)

	def _load_materialized_segments(self, wave, rows, loaders) -> int:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._load_materialized_segments(
			self,
			wave,
			rows,
			loaders,
		)

	def _prefill_prepacked_pool(self, batch: list[int]) -> torch.Tensor:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._prefill_prepacked_pool(
			self,
			batch,
		)

	def _pool_probe_debug(self, wave):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._pool_probe_debug(
			self,
			wave,
		)

	def _new_pool_epilogue_tracer(self, wave):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._new_pool_epilogue_tracer(
			self,
			wave,
		)

	def _record_unsharded_epilogue_gather(self, tracer, selected, *, num_global_rows) -> None:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._record_unsharded_epilogue_gather(
			self,
			tracer,
			selected,
			num_global_rows=num_global_rows,
		)

	def _new_pool_kernel_dump_recorder(self, wave):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._new_pool_kernel_dump_recorder(
			self,
			wave,
		)

	def _pool_pad_poison_enabled(self, wave) -> bool:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._pool_pad_poison_enabled(
			self,
			wave,
		)

	def _new_pool_offload_check_recorder(self, wave):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._new_pool_offload_check_recorder(
			self,
			wave,
		)

	def _verify_prefill_pool_offload_bytes(self) -> None:
		"""Compare every pooled row's Host destination bytes to its source.

		Runs at the one point where that is both meaningful and safe: after the
		wave's D2H has retired, so the destination pages are written, and before
		``_finish_prefill_pool_wave`` publishes the segments and drops the
		pseudo-sequence page tables the check reads back through.
		"""
		from batchgen.models.wrappers.attention import AttnWrapperBase
		from batchgen.prefix_reuse.offload_check import verify_pool_offload_rows

		from batchgen.prefix_reuse.probe_guard import run_probe

		recorder = AttnWrapperBase.prefill_pool_offload_recorder
		if recorder is None:
			return
		# Cleared before the comparison, not after: the recorder is this wave's
		# and must not survive a failure in the verification of its own rows.
		AttnWrapperBase.prefill_pool_offload_recorder = None
		wave = self._prefill_pool_wave
		wave_tag = (
			f"m{len(wave.uuids)}s{len(wave.plan.segments)}"
			if wave is not None
			else "-"
		)
		run_probe(
			"offload_check.verify",
			lambda: verify_pool_offload_rows(
				recorder.rows, logger=logging, wave_tag=wave_tag
			),
			rank=self.rank,
		)

	def _finish_prefill_pool_wave(self) -> None:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._finish_prefill_pool_wave(
			self,
		)

	def _release_pool_segment_refs(self, global_id: int) -> None:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._release_pool_segment_refs(
			self,
			global_id,
		)

	def _release_prefix_cache_attachments(
		self, global_sequence_ids: Sequence[int]
	) -> None:
		if not self.enable_prefix_cache:
			return
		# M3 Slice 2: a trajectory attachment that never reached its restore
		# leaves through the SAME seam, so a rejected wave, a rolled back
		# allocation and a completed row all return the live count to zero.
		self._release_trajectory_attachments(
			[int(sequence_id) for sequence_id in global_sequence_ids]
		)
		# A pass that died between the config-time arena reservation and the
		# post-barrier finalize never installed a record, so its arena is
		# holding Host pages nothing names. This is the teardown seam every
		# such path already goes through.
		self._rollback_trajectory_reservations(
			[int(sequence_id) for sequence_id in global_sequence_ids]
		)
		for sequence_id in global_sequence_ids:
			self._release_pool_segment_refs(int(sequence_id))
			state = self._prefix.pop_state(int(sequence_id))
			if state is None:
				continue
			for handle in (*state.commit_attachment_handles, state.attachment_handle):
				if handle:
					self.prefix_cache_coordinator.release_attachment(handle)

	def _trajectory_commit_requested(self, batch_sequences) -> bool:
		"""Batch-level ``batchgen_debug.prefix_trajectory_commit`` (M3 1b).

		Same seam as ``prefix_pool_offload_check``: the flag rides on the
		batch's own sequences, so it can be turned on against a server that is
		already up. A value other than ``"on"``/``"off"`` is refused by name
		rather than read as off -- a mistyped flag would look like a clean run
		with no frontier committed.
		"""
		return self._traj.commit_requested(
			self._active_batchgen_debug_for_sequences(batch_sequences)
		)

	def _trajectory_group_geometry(self) -> tuple:
		"""(G, this rank's shard index, whether it leads its serve group).

		The KDA state is head-sharded across the G ranks of a serve group, so a
		complete snapshot is G shards and rank ``r`` of the group owns shard
		``r``. The group leader is ``decode_dp_group * G`` (``host_kv_owner
		_rank``), so within the group the leader is index 0 and this rank's
		shard index is simply ``rank % G`` -- the same arithmetic, read from the
		other end, so there is no second notion of group membership.
		"""
		return self._traj.group_geometry(
			decode_attn_tp_size=self._decode_attn_tp_size(), rank=self.rank
		)

	def _get_trajectory_frontier_committer(self):
		"""One trajectory-frontier committer per rank, built on first use.

		The snapshot arena shares the pool-segment negative pseudo-sid
		namespace (``_next_pool_segment_sid``), so the two Host-page
		mechanisms cannot collide, and its page geometry comes from the
		prefix-cache runtime config -- the same ``host_page_bytes_all_layers``
		the frontier charge is expressed in.
		"""
		runtime = self.prefix_cache_runtime_config
		return self._traj.build_committer(
			rank=self.rank,
			runtime=runtime,
			group_geometry=self._trajectory_group_geometry(),
			host_views_provider=self._host_views_by_group,
			next_pseudo_sid=self._next_pool_segment_sid,
		)

	def _trajectory_frontier_eviction_hook(self):
		"""The frontier eviction seam, or None when no frontier is committed.

		Returns the committer's hook only once a committer EXISTS, so the
		prefix eviction path keeps its exact previous behavior on every run
		that never turned the flag on.
		"""
		committer = self._traj.trajectory_frontier_committer
		if committer is None:
			return None
		return committer.on_prefix_pages_released

	def _get_trajectory_restorer(self):
		"""M3 Slice 2 restorer, or None when no frontier has been committed.

		Deliberately does NOT construct a committer: with nothing committed on
		this rank every lookup is a guaranteed miss, so a lookup must not be
		the thing that builds the commit-side machinery (and must not raise
		``TrajectoryStateUnavailable`` on a deployment that has no KDA pool).
		The arena reader shares the committer's negative pseudo-sid namespace
		(``_next_pool_segment_sid``), so its read row cannot collide with a
		pool segment or with an arena being written.
		"""
		return self._traj.build_restorer(
			rank=self.rank,
			host_views_provider=self._host_views_by_group,
			next_pseudo_sid=self._next_pool_segment_sid,
		)

	def _trajectory_commit_tokens(self, seq) -> int:
		"""Page-aligned boundary this sequence would commit at, or 0.

		ONE owner of the figure, read by the config-time reservation and by the
		commit boundary, so the arena is sized for exactly the boundary that is
		later stored. Returns 0 when the prompt does not END on a publish
		boundary: Slice 1c refuses to store a snapshot whose live slot is ahead
		of the boundary it would claim, so such a sequence gets no arena at all
		and no rank writes a shard for it.
		"""
		return self._traj.commit_tokens(self.prefix_cache_runtime_config, seq)

	def _reserve_trajectory_arenas(self, uuids) -> dict:
		"""Pre-allocate each committing sequence's G-wide snapshot arena.

		Runs in prefill config, on the serve-group LEADER only, because the
		arena's page ids have to be on the host-KV mirror that runs at the end
		of that same config -- that mirror is the existing channel every rank
		uses to find its own shard window, and allocating later would leave the
		followers with no way to learn it without a new collective.

		The pages leave the Host free pool here, which is the R1 invariant
		unchanged: the boundary planner sees them gone and cannot double-promise
		them. Every arena reserved here is either consumed by a commit or given
		back by ``_finalize_trajectory_frontiers``.

		This is also where the committer takes its INSTALL DECISION, which is
		why the reservation is handed the frontier identity and not just a page
		count: the session key and the boundary are already known here, so a
		sequence whose frontier is already committed reserves nothing and pays
		neither the D2H nor the shard writes at the commit boundary.

		No-op at G==1: one rank owns the whole arena, so it allocates, writes
		and retains it inside the commit step exactly as before.
		"""
		shard_count, _shard_index, is_leader = self._trajectory_group_geometry()
		if shard_count <= 1 or not is_leader:
			return {}
		committer = self._get_trajectory_frontier_committer()
		runtime = self.prefix_cache_runtime_config
		page_tokens = int(runtime.group_specs[0].raw_page_tokens)
		payload = {}
		for uuid in uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None or not self._owns_host_kv(seq):
				continue
			commit_tokens = self._trajectory_commit_tokens(seq)
			if commit_tokens == 0:
				continue
			global_id = int(seq.global_idx)
			if global_id in committer.reserved_sequence_ids():
				continue
			arena = committer.reserve_arena(
				sequence_id=global_id,
				prefix_page_count=commit_tokens // page_tokens,
				namespace_digest=runtime.namespace_digest,
				# The same page-aligned prefix the commit boundary will pass,
				# so the decision taken here and the one taken at the barrier
				# are about the same frontier.
				frontier_token_ids=self._trajectory_tokens(
					seq, length=int(commit_tokens)
				).tolist(),
			)
			if arena is None:
				# Pool pressure, or a frontier that is already committed and
				# needs no second record: either way the committer named and
				# counted it, and this sequence gets no shard write this pass.
				continue
			payload[global_id] = {
				"pseudo_sid": int(arena.pseudo_sid),
				"shard_count": int(arena.shard_count),
				"pages_per_group": int(arena.pages_per_group),
				"capacity_bytes": int(arena.capacity_bytes),
				"pages": {
					int(group_id): [int(page) for page in pages]
					for group_id, pages in arena.page_ids_by_group.items()
				},
			}
		return payload

	def _trajectory_restore_payload(self) -> dict:
		"""The leader's frontier hits for this pass, for the mirror.

		A follower keeps no registry, so it restores on the LEADER's decision:
		one decision for the whole group means hit/miss and boundary cannot
		differ across ranks that must run prefill in lockstep.

		Only rows the wave PLANNED to restore are mirrored (their state
		carries the attachment's boundary as ``trajectory_restore_boundary``);
		every other attachment was declined when the wave was planned, and a
		row with no planned restore has no restore for a follower to apply.
		The leader's lookup agreement rides along for the follower's
		token-identity check.
		"""
		shard_count, _shard_index, is_leader = self._trajectory_group_geometry()
		def _planned_boundary(global_id: int) -> int:
			state = self._prefix.prefix_sequence_states.get(int(global_id))
			return 0 if state is None else int(state.trajectory_restore_boundary)

		return self._traj.restore_payload(
			shard_count=shard_count,
			is_leader=is_leader,
			planned_boundary_for=_planned_boundary,
		)

	def _trajectory_session_summary(self) -> dict:
		"""This serve group's committed sessions, for the world map.

		Leader-only: the leader holds its group's ONE frontier registry (the
		1c single-registry ruling), so its view is the whole truth about the
		group and a follower has nothing of its own to publish. The summary is
		compact -- one (owning group, boundary, frontier index key) triple per
		session, no tokens -- and rides the host-KV mirror's existing
		world-wide object gather, so no collective is added.

		Published at mirror time, which is the END of a pass's prefill config,
		while this pass's own records are installed later at the post-barrier
		finalize. A session therefore becomes pinnable one pass after it is
		committed; a same-pass continuation could not hit anyway, since its
		record does not exist until finalize.
		"""
		shard_count, _shard_index, is_leader = self._trajectory_group_geometry()
		return self._traj.session_summary(
			shard_count=shard_count, is_leader=is_leader, rank=self.rank
		)

	def _fold_trajectory_session_groups(self, gathered) -> None:
		"""Delegate the rank-identical fold to the trajectory owner."""
		self._traj.fold_session_groups(gathered, rank=self.rank)

	def _trajectory_session_residency(self, uuids, *, num_groups: int) -> dict:
		"""``{uuid: (owning group, boundary tokens)}`` for continuations.

		The cost-credit input of ``_assign_decode_dp_groups``; no group is
		stamped here. Already grouped sequences keep their group. Evicted
		re-entries use their saved group and never read node-local tokens here.

		Resolution is the map's, which probes the SAME
		``frontier_index_key`` derivation the registry index is built from, so
		nothing here re-implements the hashing or the boundary search. A miss
		returns nothing for that sequence and default assignment is unchanged.
		"""
		index = self._traj.trajectory_session_groups
		if not index:
			return {}
		if not self.enable_prefix_cache:
			return {}
		runtime = self.prefix_cache_runtime_config
		if runtime is None or self._traj.trajectory_book is None:
			return {}
		sequences = [self.global_batch.get_sequence(uuid) for uuid in sorted(uuids)]
		if not self._trajectory_commit_requested(
			[seq for seq in sequences if seq is not None]
		):
			return {}
		namespace_digest = runtime.namespace_digest
		residency = {}
		for seq in sequences:
			if seq is None or seq.decode_dp_group is not None:
				continue
			if seq.evicted_from_dp_group is not None:
				continue
			if not self._traj.trajectory_book.has_sequence(seq.uuid):
				# No trajectory reservation: the prompt is not readable yet, so
				# this sequence cannot supply a residency credit yet.
				continue
			entry = index.resolve(
				namespace_digest=namespace_digest,
				prompt_token_ids=self._trajectory_tokens(
					seq, length=seq.prompt_length
				).tolist(),
			)
			if entry is None:
				continue
			if not 0 <= int(entry.owning_group_id) < int(num_groups):
				raise RuntimeError(
					f"Rank {self.rank}: session {entry.session_key} is owned "
					f"by serve group {entry.owning_group_id}, which does not "
					f"exist in a {int(num_groups)}-group world"
				)
			residency[seq.uuid] = (int(entry.owning_group_id), int(entry.boundary_tokens))
		return residency

	def _trajectory_arena_from_payload(self, entry: dict):
		"""Rebuild the leader's arena object from its mirrored page ids."""
		return self._traj.arena_from_payload(entry)

	def _adopt_mirrored_trajectory_arenas(self, arenas: dict) -> None:
		"""Follower side: take the leader's arenas for this pass."""
		if not arenas:
			return
		committer = self._get_trajectory_frontier_committer()
		self._traj.adopt_mirrored_arenas(arenas, committer=committer)

	def _lookup_trajectory_frontiers(self, uuids) -> None:
		"""M3 Slice 2 admission lookup, beside the prefix-cache lookup.

		Runs on the sequences this rank owns the Host KV for -- the same filter
		the prefix lookup and the frontier COMMIT use, so a record is only ever
		probed on the rank whose registry holds it and whose node holds its
		arena pages. A hit takes a reference immediately; the reference is what
		keeps the record un-evictable until the restore finishes.

		Group affinity is M2's, not a new mechanism: ``decode_dp_group`` is
		already stamped and ``_assign_decode_dp_groups`` is idempotent (a
		sequence that carries a group keeps it), which is exactly how an
		evicted re-entry stays on the ranks holding its head-sharded KDA state.
		The restore asserts the record's owning group against that stamp rather
		than re-deriving an affinity of its own.
		"""
		restorer = self._get_trajectory_restorer()
		if restorer is None:
			return
		namespace_digest = self.prefix_cache_runtime_config.namespace_digest
		for uuid in uuids:
			seq = self.global_batch.get_sequence(uuid)
			if seq is None or not self._owns_host_kv(seq):
				continue
			global_id = int(seq.global_idx)
			if global_id in self._traj.trajectory_attachments:
				# Back-to-back prefill of the same row: the earlier attachment
				# is still live, so do not take a second reference for it.
				continue
			attachment = restorer.lookup(
				namespace_digest=namespace_digest,
				prompt_token_ids=self._trajectory_tokens(
					seq, length=seq.prompt_length
				).tolist(),
				sequence_id=global_id,
			)
			if attachment is not None:
				self._traj.trajectory_attachments[global_id] = attachment

	def _restore_trajectory_frontiers(self, local_indices) -> None:
		"""Restore every pinned snapshot into its KDA slot before prefill.

		THE ordering guarantee: this runs in the worker's host-side sequence
		before any suffix-prefill kernel of the wave is launched, and the slot
		write is a ``copy_`` from pageable host memory on the current stream,
		so the restored bytes are in the slot before the forward reads it.

		Empty whenever the flag is off -- nothing is attached and no row is
		planned to restore, so this is one state lookup per row and a return.

		Executes the wave plan's decision, it does not take one: every
		attachment still live here belongs to a row the plan materialized
		(``trajectory_restore_boundary``), because the plan declined all the
		others. One that is not is a broken invariant and is raised, after
		its reference is consumed.

		The converse is raised too. A planned row's compute starts at the
		boundary and the wave is already built around that start, so a
		planned row must reach the forward with its restored state: one with
		no source to restore from (no live attachment on the rank holding the
		registry, no mirrored leader decision on a follower) fails here as
		``TrajectoryPlannedRestoreFailed``, and every refusal inside the
		restore of a planned row fails loud the same way.
		"""
		from batchgen.prefix_reuse.trajectory_restore import (
			DECLINE_NOT_PLANNED,
			FAIL_NO_RESTORE_SOURCE,
			TrajectoryPlannedRestoreFailed,
		)

		for local_idx in local_indices:
			uuid = self._local_to_uuid_map.get(local_idx)
			seq = (
				self.global_batch.get_sequence(uuid)
				if uuid is not None
				else None
			)
			if seq is None:
				continue
			global_id = int(seq.global_idx)
			state = self._prefix.prefix_sequence_states.get(global_id)
			planned = 0 if state is None else int(state.trajectory_restore_boundary)
			if (
				planned
				and global_id not in self._traj.trajectory_attachments
				and global_id not in self._traj.trajectory_mirrored_restores
			):
				raise TrajectoryPlannedRestoreFailed(
					reason=FAIL_NO_RESTORE_SOURCE,
					detail=(
						"the wave plan restores this row but the restore seam "
						"holds neither its live attachment nor a mirrored "
						"leader decision for it"
					),
					rank=self.rank,
					sequence_id=global_id,
					session_key="unknown",
					boundary_tokens=planned,
				)
		if self._traj.trajectory_mirrored_restores:
			self._restore_mirrored_trajectory_shards(local_indices)
		if not self._traj.trajectory_attachments:
			return
		restorer = self._get_trajectory_restorer()
		if restorer is None:
			raise RuntimeError(
				f"Rank {self.rank}: {len(self._traj.trajectory_attachments)} live "
				"trajectory attachment(s) reached the restore seam but this "
				"rank has no restorer; their planned rows would run without "
				"their restored state"
			)

		for local_idx in local_indices:
			uuid = self._local_to_uuid_map.get(local_idx)
			seq = (
				self.global_batch.get_sequence(uuid)
				if uuid is not None
				else None
			)
			if seq is None:
				continue
			attachment = self._traj.trajectory_attachments.pop(
				int(seq.global_idx), None
			)
			if attachment is None:
				continue
			state = self._prefix.prefix_sequence_states.get(int(seq.global_idx))
			if state is None or not int(state.trajectory_restore_boundary):
				restorer.decline(
					attachment,
					reason=DECLINE_NOT_PLANNED,
					detail="reached the restore seam with no planned restore",
				)
				raise RuntimeError(
					f"Rank {self.rank}: sequence {int(seq.global_idx)} reached "
					"the trajectory restore with a live attachment its wave "
					"plan never planned; the plan must decline it"
				)
			restorer.restore(
				attachment,
				serving_group_id=int(seq.decode_dp_group or 0),
				# The position the suffix prefill will start at, which the plan
				# set to the snapshot boundary.
				compute_cached_tokens=int(state.compute_cached_tokens),
			)

	def _restore_mirrored_trajectory_shards(self, local_indices) -> None:
		"""Follower side of the restore: this rank's shard, leader's decision.

		A follower holds no frontier record -- the leader's registry is the
		group's only one, so a leader-side eviction can never leave a stale
		record here pointing at freed arena pages. What it holds is the hit the
		leader mirrored for this pass, which is why the whole group restores or
		recomputes together.
		"""
		mirrored = self._traj.trajectory_mirrored_restores
		if not mirrored:
			return
		# Build the committer FIRST. It is this rank's only source of the
		# snapshot layout, and ``_get_trajectory_restorer`` refuses to build
		# one for itself -- so without this a follower whose leader mirrored a
		# hit but no arena this pass would find no restorer and skip the
		# restore silently, while its peers restored. That is the divergence
		# the mirrored decision exists to prevent, so it must not be
		# reintroduced by construction order.
		committer = self._get_trajectory_frontier_committer()
		restorer = self._get_trajectory_restorer()
		if restorer is None:
			raise RuntimeError(
				f"Rank {self.rank}: its serve-group leader mirrored "
				f"{len(mirrored)} trajectory restore(s) but this rank has no "
				"restorer; the group's prefill would diverge"
			)
		for local_idx in local_indices:
			uuid = self._local_to_uuid_map.get(local_idx)
			seq = (
				self.global_batch.get_sequence(uuid)
				if uuid is not None
				else None
			)
			if seq is None:
				continue
			entry = mirrored.pop(int(seq.global_idx), None)
			if entry is None:
				continue
			state = self._prefix.prefix_sequence_states.get(int(seq.global_idx))
			restorer.restore_mirrored_shard(
				sequence_id=int(seq.global_idx),
				arena=self._trajectory_arena_from_payload(entry),
				layout=committer.layout,
				session_key=str(entry["session_key"]),
				boundary_tokens=int(entry["boundary_tokens"]),
				lcp_tokens=int(entry["lcp_tokens"]),
				owning_group_id=int(entry["owning_group_id"]),
				serving_group_id=int(seq.decode_dp_group or 0),
				compute_cached_tokens=(
					int(state.compute_cached_tokens) if state is not None else 0
				),
			)

	def _release_trajectory_attachments(self, global_sequence_ids) -> None:
		"""Drop every un-consumed attachment for these rows.

		Joins the prefix-attachment release seam, so a rejected wave, a rolled
		back allocation and a completed sequence all return the live count to
		zero through the same call.
		"""
		self._traj.release_attachments(global_sequence_ids, rank=self.rank)

	def _rollback_trajectory_reservations(self, global_sequence_ids) -> None:
		"""Give back any snapshot arena these rows reserved and never used."""
		self._traj.rollback_reservations(global_sequence_ids, rank=self.rank)

	def _trajectory_restore_fields(self) -> dict:
		"""Drain-counter fields, or empty when the path never ran."""
		return self._traj.restore_fields()

	def _resident_frontier_page_ids(self, *, token_ids, commit_tokens):
		"""Resident Host pages backing a committed prefix, per KV group.

		``retain_inserted_prefix_pages`` reports only the pages THIS commit
		inserted; a frontier whose leading nodes already existed is backed by
		the coordinator's older pages. A frontier record has to name the pages
		eviction will actually report, or it would outlive them, so the set is
		read back from the coordinator. ``estimate_lookup`` is the read-only
		form (it takes no attachment), and the commit's own active attachment
		already protects the nodes while this runs. ``None`` means the
		coordinator does not hold the whole prefix as resident pages.
		"""
		runtime = self.prefix_cache_runtime_config
		estimate = self.prefix_cache_coordinator.estimate_lookup(
			list(runtime.namespace_digest),
			[int(token_id) for token_id in token_ids[: int(commit_tokens)]],
		)
		if int(estimate.common_cached_tokens) != int(commit_tokens):
			return None
		pages = {}
		for span in estimate.materialization_spans:
			pages[int(span.group_id)] = [
				int(getattr(page, "page_id", page)) for page in span.pages
			]
		if len(pages) != len(runtime.group_specs):
			return None
		return pages

	def _write_trajectory_shards(self, local_sequence_ids) -> None:
		"""Follower side of the commit boundary: write THIS rank's shard.

		Runs on every rank of a head-sharded serve group that is NOT the
		host-KV leader. It mutates no allocator state: the arena's pages were
		allocated by the leader and mirrored here, and the write narrows this
		rank's own page-table row to its own window before copying into it.

		Which sequences get a shard is the LEADER's decision, carried here as
		the set of arenas it mirrored, so the ranks of a group cannot disagree
		about which frontiers exist.
		"""
		shard_count, shard_index, is_leader = self._trajectory_group_geometry()
		if shard_count <= 1 or is_leader or not self._traj.trajectory_arena_payload:
			return
		committer = self._get_trajectory_frontier_committer()
		for local_idx in local_sequence_ids:
			uuid = self._local_to_uuid_map.get(local_idx)
			seq = (
				self.global_batch.get_sequence(uuid)
				if uuid is not None
				else None
			)
			if seq is None:
				continue
			global_id = int(seq.global_idx)
			if global_id not in self._traj.trajectory_arena_payload:
				continue
			commit_tokens = self._trajectory_commit_tokens(seq)
			if commit_tokens == 0:
				continue
			token_ids = self._trajectory_tokens(
				seq, length=seq.prompt_length
			).tolist()
			try:
				committer.write_shard(
					sequence_id=global_id,
					frontier_token_ids=token_ids[:commit_tokens],
					prompt_token_count=len(token_ids),
				)
			except Exception:
				# A shard write is a memcpy into pages this rank already holds
				# the ids of: the only way it fails is a bug, and a bug here
				# would leave the leader installing a record whose shard is
				# missing. Named and re-raised rather than degraded.
				logging.exception(
					"[TRAJECTORY_FRONTIER] Rank %s shard=%s/%s FAILED to write "
					"its snapshot shard for sequence %s; the group's frontier "
					"must not be installed",
					self.rank,
					shard_index,
					shard_count,
					global_id,
				)
				raise

	def _finalize_trajectory_frontiers(self) -> None:
		"""Install every pending frontier; give back every unused arena.

		THE post-barrier step. It is called immediately after the pass's
		existing unconditional ``dist.barrier()``, which is the ordering point
		that makes a G-shard record sound: every rank of every serve group has
		returned from its own shard write before any rank reaches here, so a
		record installed now binds a COMPLETE snapshot. Before the barrier no
		record exists at all, so a crash, a rejected wave or any named skip
		leaves the previously committed frontier exactly as it was.

		No-op at G==1, where the commit step installs its own record: one rank
		owns the whole arena and there is nothing to wait for.
		"""
		committer = self._traj.trajectory_frontier_committer
		if committer is None:
			return
		shard_count, _shard_index, _is_leader = self._trajectory_group_geometry()
		if shard_count <= 1:
			return
		try:
			outcome = committer.finalize()
		except Exception:
			logging.exception(
				"[TRAJECTORY_FRONTIER] Rank %s: finalizing the pass's frontier "
				"records failed; previously committed frontiers are unchanged",
				self.rank,
			)
			return
		self._traj.trajectory_arena_payload = {}
		if outcome.installed or outcome.rolled_back_sequence_ids or outcome.failed_sequence_ids:
			logging.info(
				"[TRAJECTORY_FRONTIER] Rank %s finalize: installed=%s "
				"rolled_back=%s deduped=%s failed=%s dedup_total=%s",
				self.rank,
				[item.session_key for item in outcome.installed],
				list(outcome.rolled_back_sequence_ids),
				list(outcome.deduped_sequence_ids),
				list(outcome.failed_sequence_ids),
				committer.dedup_skips(),
			)

	def _commit_trajectory_frontier(self, *, seq, token_ids, commit_tokens):
		"""M3 Slice 1b: snapshot the KDA slot into Host KV pages (flag-gated).

		Runs INSIDE the retention transaction, after ``retain_inserted_prefix
		_pages`` and its ``host_owned_pages`` deduction. Per R1 the snapshot
		bytes go into Host KV pages taken from the SAME allocator, so
		``host_owned_pages`` is not touched here and nothing is credited back
		to it when the snapshot is freed.
		"""
		pages = self._resident_frontier_page_ids(
			token_ids=token_ids, commit_tokens=commit_tokens
		)
		if pages is None:
			logging.info(
				"[%s] rank=%s seq=%s SKIPPED frontier commit: the coordinator "
				"does not hold all %s committed tokens as resident pages in "
				"every KV group",
				self._traj.FRONTIER_PREFIX,
				self.rank,
				seq.global_idx,
				int(commit_tokens),
			)
			return None
		committer = self._get_trajectory_frontier_committer()
		return committer.commit(
			namespace_digest=self.prefix_cache_runtime_config.namespace_digest,
			frontier_token_ids=token_ids[: int(commit_tokens)],
			page_ids_by_group=pages,
			# Group affinity: the KDA state is head-sharded for the layout its
			# decode group holds, so a snapshot is only restorable into that
			# group (same invariant as today's re-entry pinning).
			owning_group_id=int(seq.decode_dp_group or 0),
			sequence_id=seq.global_idx,
			# Slice 1c's boundary identity: the live slot holds the state after
			# every token the prefill just ran, so it only describes this
			# boundary when the two are the same number.
			prompt_token_count=len(token_ids),
		)

	def _commit_prefix_cache_for_prefill(
		self, local_sequence_ids: Sequence[int]
	) -> None:
		"""Publish completed prompt pages after every D2H offload retires."""

		if not self.enable_prefix_cache or not local_sequence_ids:
			return
		from batchgen.prefix_reuse.commit import (
			build_prefix_commit_request,
			collect_group_pages_for_commit,
			retain_inserted_prefix_pages,
		)
		from batchgen.prefix_reuse.eviction import (
			commit_prefix_pages_with_capacity_retry,
		)

		host_views = self._host_views_by_group()
		runtime = self.prefix_cache_runtime_config
		page_size = int(runtime.group_specs[0].raw_page_tokens)
		page_tokens_by_group = {
			int(spec.group_id): int(spec.raw_page_tokens)
			for spec in runtime.group_specs
		}
		# ONE guard for the whole pass: with the flag off nothing below changes.
		commit_sequences = [
			self.global_batch.get_sequence(self._local_to_uuid_map[local_idx])
			for local_idx in local_sequence_ids
		]
		trajectory_frontier = self._trajectory_commit_requested(
			commit_sequences
		)
		if trajectory_frontier:
			# M3 Slice 1c: at G>1 every rank of the serve group snapshots its
			# own head shard, so the followers do their write here too. The
			# loop below stays leader-only (it publishes prefix pages and moves
			# host_owned_pages, which exactly one rank may do).
			self._write_trajectory_shards(local_sequence_ids)
		for local_idx in local_sequence_ids:
			uuid = self._local_to_uuid_map[local_idx]
			seq = self.global_batch.get_sequence(uuid)
			if not self._owns_host_kv(seq):
				continue
			token_ids = self._trajectory_tokens(
				seq, length=seq.prompt_length
			).tolist()
			commit_tokens = (
				len(token_ids) // runtime.publish_boundary_tokens
			) * runtime.publish_boundary_tokens
			if commit_tokens == 0:
				continue
			pages = collect_group_pages_for_commit(
				worker_views_by_group=host_views,
				sequence_id=seq.global_idx,
				commit_tokens=commit_tokens,
				raw_page_tokens_by_group=page_tokens_by_group,
			)
			request = build_prefix_commit_request(
				namespace_digest=runtime.namespace_digest,
				token_ids=token_ids,
				publish_boundary_tokens=runtime.publish_boundary_tokens,
				pages_by_group=pages,
			)
			if request is None:
				continue

			commit_outcome = commit_prefix_pages_with_capacity_retry(
				request=request,
				coordinator=self.prefix_cache_coordinator,
				worker_views_by_group=host_views,
				max_scan_nodes=runtime.max_nodes,
				frontier_hook=self._trajectory_frontier_eviction_hook(),
			)
			result = commit_outcome.commit_result
			commit_handle = int(result.active_attachment_handle)
			if result.inserted_nodes and not commit_handle:
				raise RuntimeError(
					"prefix commit inserted pages without protecting the active sequence"
				)
			if commit_handle:
				# Keep the new resident pages pinned until this sequence releases
				# its Host page table, including during back-to-back prefill.
				state = self._prefix.prefix_sequence_states.get(seq.global_idx)
				if state is None:
					self.prefix_cache_coordinator.release_attachment(commit_handle)
					raise RuntimeError(
						"prefix commit has no active sequence state"
					)
				self._prefix.prefix_sequence_states[seq.global_idx] = replace(
					state,
					commit_attachment_handles=(
						*state.commit_attachment_handles, commit_handle
					),
				)
			retained = retain_inserted_prefix_pages(
				commit_result=result,
				request=request,
				worker_views_by_group=host_views,
				sequence_id=seq.global_idx,
			)
			# Retained pages left the sequence chain for prefix residency. Every
			# group retains the same page count; host_owned_pages counts one.
			# Leader-only at G>1: the followers adopt the result at the
			# post-prefill _sync_sequence_metadata (host_kv_owner_reports).
			seq.host_owned_pages = host_owned_pages_after_sharing(
				chain_pages=seq.host_owned_pages,
				shared_pages=len(retained.get(0, ())),
				sequence_id=seq.global_idx,
				where="prefix commit retain",
			)
			if trajectory_frontier:
				# M3 Slice 1b, inside this same transaction: the snapshot's
				# Host pages come from the SAME allocator, so the deduction
				# above is the only host_owned_pages move at this boundary.
				self._commit_trajectory_frontier(
					seq=seq,
					token_ids=token_ids,
					commit_tokens=commit_tokens,
				)
			logging.info(
				"[PREFIX_CACHE] Rank %s committed sequence=%s tokens=%s "
				"inserted_nodes=%s retained_pages=%s",
				self.rank,
				seq.global_idx,
				result.committed_tokens,
				result.inserted_nodes,
				sum(len(value) for value in retained.values()),
			)
			if self.prefix_cache_debug_stats:
				eviction = commit_outcome.eviction_result
				self._emit_prefix_cache_metric(
					"prefix_host_commit",
					sequence_id=int(seq.global_idx),
					committed_tokens=int(result.committed_tokens),
					inserted_nodes=int(result.inserted_nodes),
					existing_nodes=int(result.existing_nodes),
					retained_pages=sum(
						len(value) for value in retained.values()
					),
					retry_evicted_nodes=(
						int(eviction.evicted_nodes)
						if eviction is not None
						else 0
					),
					retry_protected_nodes=(
						int(eviction.protected_nodes)
						if eviction is not None
						else 0
					),
					retry_freed_group_entries=(
						int(eviction.freed_group_entries)
						if eviction is not None
						else 0
					),
					retry_freed_page_handles=(
						int(eviction.freed_page_handles)
						if eviction is not None
						else 0
					),
					**self._prefix_host_resource_fields(),
				)

	# ============ Prefill and Decode ============

	def prefill(self, batch: list[int]):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		# PrefillEngine is stateless; this lazy path (shared by every delegator) only serves test fakes built without __init__ -- deliberate, not a special-case branch.
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine.prefill(
			self,
			batch,
		)

	def prefill_prepacked(self, batch: list[int]):
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine.prefill_prepacked(
			self,
			batch,
		)

	def _write_prefill_tokens(self, batch: list[int], new_tokens: torch.Tensor) -> torch.Tensor:
		# The engine receives this worker as its sole state collaborator.
		engine = getattr(self, "_prefill_engine", None)
		if engine is None:
			from batchgen.worker.prefill_engine import PrefillEngine
			engine = PrefillEngine()
			self._prefill_engine = engine
		return engine._write_prefill_tokens(
			self,
			batch,
			new_tokens,
		)

	# ============ PREFILL MODEL-STATE BINDER (S5b) ============
	# The worker implements PrefillModelStateBinder (batchgen/worker/prefill_engine.py):
	# every model-wrapper class-state statement the moved prefill bodies made
	# stays here, one method per contiguous run of the original body, moved
	# verbatim; PrefillEngine calls each at its run's former position, so the
	# engine module never names a model wrapper.

	def _bind_prefill_host_views(self) -> None:
		"""Bind the host KV worker views the prefill forward offloads through."""
		AttnWrapperBase.host_paged_kv_worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		AttnWrapperBase.host_paged_kv_worker_view_aux = getattr(self, "host_paged_kv_worker_view_aux", None)

	def _bind_prefill_pool_segments(self, wave) -> None:
		"""Bind one pool wave's views, pool, segment sids and loaded segments."""
		AttnWrapperBase.host_paged_kv_worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		AttnWrapperBase.host_paged_kv_worker_view_aux = getattr(self, "host_paged_kv_worker_view_aux", None)
		AttnWrapperBase.prefill_pool = wave.pool
		# A recurrent layer forks a row's state from its parent segment's state
		# slot, which is keyed by that segment's pseudo sequence id.
		AttnWrapperBase.prefill_pool_segment_sids = wave.segment_sids
		# A LOADED segment ran no row, so it left no recurrent state to fork:
		# its reader resumes from the slot the trajectory restore bound before
		# this wave, which the level cut verifies.
		AttnWrapperBase.prefill_pool_materialized_segments = tuple(
			sorted(wave.materialized)
		)

	def _bind_prefill_pool_layer_state(self, wave, pool_layout) -> None:
		"""Bind the wave's pool layer map and its fresh offload recorder."""
		AttnWrapperBase.prefill_pool_layer_map = (
			tuple(pool_layout()["layer_map"]) if pool_layout is not None else None
		)
		# Always reassigned, never left over: a wave that died before its
		# verification point cannot leak its records into the next one.
		AttnWrapperBase.prefill_pool_offload_recorder = (
			self._new_pool_offload_check_recorder(wave)
		)

	def _install_prefill_pool_rows(self, rows, host_ids, host_starts) -> None:
		"""Install one pool chunk's packed-row metadata on the attention wrappers."""
		AttnWrapperBase.prepack_mode = True
		AttnWrapperBase.prepack_cu_seqlens = rows.cu_seqlens_q
		AttnWrapperBase.prepack_max_seqlen = rows.max_seqlen_q
		AttnWrapperBase.prepack_num_sequences = len(rows.items)
		AttnWrapperBase.prepack_seq_lengths = [item.tokens for item in rows.items]
		AttnWrapperBase.position_ids = rows.position_ids
		AttnWrapperBase.cur_batch = host_ids
		AttnWrapperBase.prefill_pool_batch = rows
		AttnWrapperBase.prefill_pool_row_starts = host_starts

	def _drop_prefill_pool_offload_recorder(self) -> None:
		"""Drop the wave's offload recorder (not handed to verification)."""
		AttnWrapperBase.prefill_pool_offload_recorder = None

	def _unbind_prefill_pool_segments(self) -> None:
		"""Clear every pool-wave binding the wrappers carried for the wave."""
		AttnWrapperBase.prefill_pool = None
		AttnWrapperBase.prefill_pool_batch = None
		AttnWrapperBase.prefill_pool_row_starts = None
		AttnWrapperBase.prefill_pool_segment_sids = None
		AttnWrapperBase.prefill_pool_materialized_segments = ()
		AttnWrapperBase.prefill_pool_layer_map = None
		AttnWrapperBase.prepack_mode = False
		AttnWrapperBase.prepack_cu_seqlens = None
		AttnWrapperBase.prepack_max_seqlen = None
		AttnWrapperBase.prepack_num_sequences = None
		AttnWrapperBase.prepack_seq_lengths = None

	def _bind_prefill_step_metadata(self, attention_mask) -> None:
		"""Bind one padded prefill micro-batch's mask and derived positions."""
		Attn_Wrapper.attention_mask = attention_mask
		Attn_Wrapper.position_ids = create_position_ids_from_attention_mask(
			attention_mask
		)

	def _bind_prefill_step_batch(self, cur_batch_local) -> None:
		"""Bind the global sequence ids of one padded prefill micro-batch."""
		Attn_Wrapper.cur_batch = self._local_indices_to_global_seq_ids(cur_batch_local)

	def _install_prefill_prepack_metadata(
		self,
		batch_cu_seqlens,
		batch_max_seqlen,
		batch_num_seqs,
		batch_seq_lengths,
		batch_position_ids_flat,
		batch_spans,
		prefix_materialization,
		batch_full_seq_lengths,
		batch_compute_cached,
		batch_attached_tokens,
	) -> None:
		"""Install one prepacked micro-batch's metadata on the wrappers."""
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
		AttnWrapperBase.prepack_full_seq_lengths = (
			batch_full_seq_lengths
			if prefix_materialization is not None
			else None
		)
		AttnWrapperBase.prepack_compute_cached_tokens = (
			batch_compute_cached
			if prefix_materialization is not None
			else None
		)
		AttnWrapperBase.prepack_attached_tokens = (
			batch_attached_tokens
			if prefix_materialization is not None
			else None
		)
		AttnWrapperBase.prefill_prefix_materialization = (
			prefix_materialization
		)
		AttnWrapperBase.position_ids = batch_position_ids_flat
		AttnWrapperBase.cur_batch = Attn_Wrapper.cur_batch

	def _retire_prefill_pending_offloads(self, device, reason):
		"""Retire pending prefill KV offloads; returns the retired task count."""
		return AttnWrapperBase.retire_pending_prefill_offloads(
			device=device,
			reason=reason,
		)

	def _clear_prefill_prefix_materialization(self) -> None:
		"""Detach the released prefix materialization from the wrappers."""
		AttnWrapperBase.prefill_prefix_materialization = None

	def _reset_prefill_prepack_metadata(self) -> None:
		"""Reset the prepack metadata after the prepacked forward loop."""
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
		AttnWrapperBase.prepack_full_seq_lengths = None
		AttnWrapperBase.prepack_compute_cached_tokens = None
		AttnWrapperBase.prepack_attached_tokens = None
		AttnWrapperBase.prefill_prefix_materialization = None

	# S5b model-name routing (spec F3): the moved prefill bodies read these
	# model-family constants through the worker so no model-name literal enters
	# batchgen/worker/. Each returns exactly the original string constant.

	def _prefill_deepseek_family_name(self) -> str:
		return "deepseek"

	def _k3_prefill_profile_flag_name(self) -> str:
		return "k3_prefill_profile"

	def _glm5_lmhead_fp32_flag_name(self) -> str:
		return "glm5_lmhead_fp32"

	def _new_prefill_temporary_gpu_kv_manager(self, gpu_config) -> GPUPagedKVCacheManager:
		# S5b carve-out #6 (spec §S5b): the engine module must stay importable
		# without the native kv_cache chain, so the temporary prefix-hit GPU KV
		# manager is constructed here, where the class is already imported.
		return GPUPagedKVCacheManager(
			config=gpu_config,
			device=self.local_rank,
		)

	def _emit_k3_prefill_profile_log(self, k3_profile_topk) -> None:
		"""Log the per-rank K3 prefill profile record (one JSON line)."""
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
			# The exact module counts of the profiled forward's passes.
			"weight_passes": self._weight_passes.totals,
			"expert_consumer": (
				KimiK3MXFP4ExpertWrapper.prefill_profile_snapshot()
			),
			"streamed_sp8": (
				StreamedSP8MXFP4MoELayer.prefill_profile_snapshot()
			),
			"logit_topk": k3_profile_topk,
		}, separators=(",", ":")))

	# ============ RANK-0 BOUNDARY DECISION COMPUTATION ============

	def _make_boundary_decision_request(
		self,
		decode_uuids: List[str],
		global_seq_state: Dict[str, Dict],
		global_candidate_info: Dict[str, Dict],
		per_rank_free: List[int],
		per_rank_total: List[int],
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
			per_rank_total=tuple(per_rank_total),
			chunk_size=chunk_size,
			per_node_host_stats=tuple(per_node_host_stats) if per_node_host_stats else None,
			seq_meta=seq_meta,
			world_size=self.world_size,
			num_gpus_per_node=self.local_world_size,
			max_rank_bsz=getattr(self, "_decode_padding_bsz", 0) or 0,
			enable_host_kv_eviction=self.enable_host_kv_eviction,
			attn_tp_size=self._decode_attn_tp_size(),
		)

	def _compute_boundary_decisions(
		self,
		decode_uuids: List[str],
		global_seq_state: Dict[str, Dict],
		global_candidate_info: Dict[str, Dict],
		per_rank_free: List[int],
		per_rank_total: List[int],
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
				per_rank_free, per_rank_total, chunk_size, per_node_host_stats,
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
		# Completion, eviction and load decisions read every row's stop state;
		# a deferred step's tokens must already be consumed on this rank.
		self._require_decode_tokens_read("_page_boundary_fast")
		timing = FastBoundaryTimingStats()
		boundary_start = time.perf_counter()
		
		# ========== PHASE 0: Wait for pending async operations ==========
		t0 = time.perf_counter()
		timing.num_kv_append_tasks = self._wait_pending_kv_append_tasks(sync_distributed_errors=True)
		timing.wait_kv_append_ms = (time.perf_counter() - t0) * 1000
		
		# Validate the entry UUID set before any conditional pending-load/empty
		# return.  Otherwise one rank can return while another enters the Phase 1
		# all-gather, deadlocking the process group.  This is deliberately a
		# cheap object gather at the boundary; the normal non-empty path performs
		# the full payload gather below as its next collective.
		t_sync = time.perf_counter()
		if self.world_size > 1:
			local_decode_set = tuple(sorted(set(decode_uuids)))
			all_decode_sets = [None] * self.world_size
			dist.all_gather_object(all_decode_sets, local_decode_set)
			all_sets_equal = all(s == local_decode_set for s in all_decode_sets if s is not None)
			if not all_sets_equal:
				logging.error(
					f"Rank {self.rank}: decode_uuids DESYNC detected at boundary start: "
					f"rank_sets={all_decode_sets}"
				)
				raise RuntimeError(
					"[SCHED_INVARIANT] decode UUID set diverged before boundary "
					"pending-load/empty handling"
				)
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
		
		local_payload_error = None
		try:
			if gpu_manager is not None and gpu_manager.is_initialized:
				gpu_stats = gpu_manager.get_stats()
				local_free_pages = int(gpu_stats.num_free_pages)
				local_total_pages = int(gpu_stats.num_total_pages)
			else:
				local_free_pages = 0
				local_total_pages = 0
		except Exception as exc:
			# Carry allocator stats failures through the payload gather so
			# peers do not block while this rank raises before all-gather.
			local_free_pages = 0
			local_total_pages = 0
			local_payload_error = (
				f"GPU-KV stats: {type(exc).__name__}: {exc}"
			)
		
		# DEBUG: Log decode_uuids and which ones this rank owns
		my_owned = [u for u in decode_uuids if u in self._uuid_to_local_map]
		if self.rank == 0:
			logging.debug(
				f"Rank {self.rank}: State gathering - decode_uuids_len={len(decode_uuids)}, "
				f"my_owned_count={len(my_owned)}"
			)
		
		# Build local state for sequences owned by this rank
		try:
			chunk_size = self._get_effective_chunk_size()
		except Exception as exc:
			chunk_size = 0
			local_payload_error = (
				f"chunk size: {type(exc).__name__}: {exc}"
			)
		local_seq_state = {}
		for uuid in decode_uuids:
			if uuid in self._uuid_to_local_map:
				seq = self.global_batch.get_sequence(uuid)
				try:
					is_completed = self._is_sequence_completed(seq)
					seq.validate_metadata(
						f"rank {self.rank} _page_boundary_fast/decode_state",
						allow_terminal_capacity_gap=is_completed,
					)
				except Exception as exc:
					local_payload_error = (
						f"decode {uuid}: {type(exc).__name__}: {exc}"
					)
					continue
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
					'host_owned_pages': seq.host_owned_pages,
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
					# QueryBook metadata is process-local even though its token
					# pages are node-shared. Publish the owner's lengths so
					# non-owning ranks can repair their mirror before re-entry.
					'trajectory_token_length': (
						self._traj.trajectory_book.metadata(self._trajectory_slot(seq)).token_length
						if self._traj.trajectory_book is not None
						and self._traj.trajectory_book.has_sequence(uuid)
						else None
					),
					'trajectory_decoded_length': (
						self._traj.trajectory_book.metadata(self._trajectory_slot(seq)).decoded_length
						if self._traj.trajectory_book is not None
						and self._traj.trajectory_book.has_sequence(uuid)
						else None
					),
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
			try:
				seq.validate_metadata(f"rank {self.rank} _page_boundary_fast/load_candidate")
			except Exception as exc:
				local_payload_error = (
					f"load candidate {uuid}: {type(exc).__name__}: {exc}"
				)
				continue
			try:
				pages_needed = seq.get_gpu_pages_for_two_page_buffer()
			except Exception as exc:
				local_payload_error = (
					f"load candidate {uuid} page count: {type(exc).__name__}: {exc}"
				)
				continue
			# Report this as a potential load candidate
			local_candidate_state[uuid] = {
				'pages_needed': pages_needed,
				'assigned_rank': seq.assigned_rank,
				'decode_dp_group': seq.decode_dp_group,  # Option 1 group-ownership key
				'status': seq.status.name,  # Include status for debugging
				'decoded_length': seq.decoded_length,  # For prioritized loading
			}
		
		# Pack everything into one dict for single all_gather
		local_payload = {
			'free_pages': local_free_pages,
			'total_pages': local_total_pages,
			'error': local_payload_error,
			'seq_state': local_seq_state,
			'candidate_state': local_candidate_state,
		}
		
		all_payloads = [None] * self.world_size
		dist.all_gather_object(all_payloads, local_payload)
		payload_errors = [
			f"rank {rank_idx}: {payload.get('error')}"
			for rank_idx, payload in enumerate(all_payloads)
			if isinstance(payload, dict) and payload.get('error')
		]
		if payload_errors:
			raise RuntimeError(
				"[SCHED_INVARIANT] boundary payload validation failed: "
				+ " | ".join(payload_errors)
			)
		validate_boundary_payload_alignment(
			decode_uuids, all_payloads, group_size=self._decode_attn_tp_size()
		)
		
		timing.gather_ms = (time.perf_counter() - t0) * 1000
		
		# ========== PHASE 2: MERGE GATHERED DATA + RANK-0 DECISIONS ==========
		t0 = time.perf_counter()

		# Extract per-rank live capacity.  Total pages are immutable after GPU-KV
		# initialization; rank 0 validates that invariant before planning any
		# extension or load.
		per_rank_free = [p['free_pages'] for p in all_payloads]
		per_rank_total = [p['total_pages'] for p in all_payloads]
		# Validate the live allocator snapshot on EVERY rank before rank 0
		# computes or broadcasts decisions.  Keeping this check inside the
		# rank-0 planner would strand peers in broadcast when one total differs.
		reduce_decode_capacity_snapshot(
			tuple(per_rank_total),
			tuple(per_rank_free),
			world_size=self.world_size,
			attn_tp_size=self._decode_attn_tp_size(),
		)

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
		post_gather_errors = []
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
					seq.host_owned_pages = state['host_owned_pages']
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
					try:
						seq.validate_metadata(
							f"rank {self.rank} _page_boundary_fast/gathered_state",
							require_owner_tensors=False,
							allow_terminal_capacity_gap=bool(state.get('completed', False)),
						)
					except Exception as exc:
						post_gather_errors.append(
							f"gathered {uuid}: {type(exc).__name__}: {exc}"
						)
			# The token pages are shared, but the QueryBook metadata is not.
			# Repair every local mirror from the canonical owner state before
			# Phase 4 can call begin_reentry_turn on all ranks.
			seq = self.global_batch.get_sequence(uuid)
			if seq is not None:
				try:
					self._sync_trajectory_metadata(seq, state)
				except Exception as exc:
					post_gather_errors.append(
						f"trajectory {uuid}: {type(exc).__name__}: {exc}"
					)

		post_gather_reports = [None] * self.world_size
		dist.all_gather_object(post_gather_reports, post_gather_errors)
		post_gather_failures = [
			f"rank {rank_idx}: {error}"
			for rank_idx, errors in enumerate(post_gather_reports)
			for error in (errors or [])
		]
		if post_gather_failures:
			raise RuntimeError(
				"[SCHED_INVARIANT] gathered boundary state validation failed: "
				+ " | ".join(post_gather_failures)
			)

		# ========== RANK 0 COMPUTES ALL DECISIONS ==========
		# Only rank 0 makes batching decisions. All other ranks receive via broadcast.
		# This eliminates desync from independent decision-making.
		# The worker owns the mirrored coordinator for DSA.  Using the engine's
		# primary view here would let a partial primary/aux growth split escape the
		# coordinator's transaction guard.
		worker_view = getattr(self, "host_paged_kv_worker_view", None)
		if worker_view is None:
			worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)
		per_node_host_stats = self._gather_host_kv_stats_by_node(worker_view)

		if self.rank == 0:
			try:
				decisions = self._compute_boundary_decisions(
					decode_uuids, global_seq_state, global_candidate_info,
					per_rank_free, per_rank_total, chunk_size, per_node_host_stats,
				)
			except Exception as exc:
				# Rank 0 owns the pure planner, but a malformed gathered state
				# must become a broadcast scheduler error rather than strand peers
				# waiting in the decision broadcast.
				logging.exception("Rank 0 boundary planner failed")
				decisions = BoundaryDecisions(
					completed_uuids=[], active_uuids=[], host_growth_uuids=[],
					host_growth_pages=[], growth_feasible=False,
					host_evicted_uuids=[], onhold_uuids=[],
					seqs_needing_extension=[], new_load_uuids=[],
					decode_uuids_final=[],
					scheduler_error=(
						"[SCHED_INVARIANT] rank-0 boundary planner failed: "
						f"{type(exc).__name__}: {exc}"
					),
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
					seq.host_owned_pages = 0
					seq.host_token_capacity = 0
					seq.evicted_from_dp_group = None
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
		# requiring a follow-up _sync_sequence_metadata call. The trajectory book
		# is the token source of truth and remains resident across KV eviction;
		# only scalar metadata — prompt_length, current_context_length,
		# total_decoded_before_eviction, host/gpu page counters, status — is
		# updated on ALL ranks deterministically, using values already
		# synchronized in Phase 1/2 of this same boundary call.
		#
		# For re-entry length: new_reentry_len = seq.prompt_length +
		# seq.decoded_length. At Phase 4.C time, both operands are consistent
		# across ranks because Phase 2 synced them from the owner.
		host_evicted_uuids = decisions.host_evicted_uuids
		if host_evicted_uuids:
			# Release on-device/host KV resources. The unified trajectory stays
			# bound to the sequence, so no prompt or decoded-token copy is made.
			my_evicted = [u for u in host_evicted_uuids if u in self._uuid_to_local_map]
			if my_evicted:
				# Host-eviction usually targets seqs already in DECODE (so they
				# have GPU pages), but defensively intersect with the source-of-
				# truth set in case an EVICTED seq never reached decode.
				gpu_allocated = [u for u in my_evicted if u in self._sequences_with_gpu_kv]
				if gpu_allocated:
					self._release_gpu_kv_pages(self._get_local_indices_for_uuids(gpu_allocated))
				if BATCHGEN_CB_DEBUG:
					for uuid in my_evicted:
						seq = self.global_batch.get_sequence(uuid)
						logging.debug(
							f"[HOST_KV_EVICT_DETAIL] seq={uuid[:8]} "
							f"decoded={seq.decoded_length} "
							f"host_pages={seq.host_pages_allocated} "
							f"trajectory_tokens={self._traj.trajectory_book.metadata(self._trajectory_slot(seq)).token_length}"
						)
				# Host KV is ONE per-node SHARED shm region keyed by global_idx, so
				# release/unregister must fire EXACTLY once per sequence. Under G>1
				# all G ranks of a group hold the evicted uuid, so releasing on every
				# rank double-frees: the first releaser tombstones the entry and the
				# rest raise "Sequence ID ... not found during release". Filter to the
				# group leader (_owns_host_kv), matching the validated release path in
				# _release_host_kv_pages_for_batch. The all-ranks scalar-metadata loop
				# below iterates host_evicted_uuids and is deliberately NOT filtered.
				evicted_global_ids = [
					self.global_batch.get_sequence(u).global_idx for u in my_evicted
					if self._owns_host_kv(self.global_batch.get_sequence(u))
				]
				# A follower of the group holds the leader's page table MIRRORED in
				# its own view plus a mirrored prefix state, and the re-entry of
				# this row registers the same global_idx again, so those rank-local
				# records must go as well. unregister_sequences edits only the
				# view's own HostKVPageTable — the shared pages are freed exactly
				# once, by the leader above. Always empty at G==1.
				mirrored_global_ids = [
					self.global_batch.get_sequence(u).global_idx for u in my_evicted
					if not self._owns_host_kv(self.global_batch.get_sequence(u))
				]
				if worker_view is not None:
					# The worker-owned view is a DualHostKVCoordinator for DSA and
					# mirrors both page tables transactionally.  Calling the auxiliary
					# view again here would double-release its pages and leave the two
					# allocators out of sync.
					worker_view.release_sequence_pages(evicted_global_ids)
					worker_view.unregister_sequences(evicted_global_ids)
					self._release_prefix_cache_attachments(
						evicted_global_ids
					)
					if mirrored_global_ids:
						for view in self._host_views_by_group().values():
							view.unregister_sequences(mirrored_global_ids)
						self._release_prefix_cache_attachments(mirrored_global_ids)

			# All-ranks: update scalar metadata deterministically. The retained
			# trajectory length is original_prompt_length + decoded_length; the
			# effective prompt for the next prefill is the previous effective prompt
			# plus only the newly generated suffix from this KV residency cycle.
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
				if self._traj.trajectory_book is not None and self._traj.trajectory_book.has_sequence(uuid):
					slot = self._traj.trajectory_book.slot_for(uuid)
					# Record the replayed prompt boundary at eviction time. Re-entry
					# calls this idempotently, so migration while ON_HOLD carries a
					# ledger consistent with the scalar prompt length.
					self._traj.trajectory_book.begin_reentry_turn(
						slot, prompt_length=new_reentry_len
					)
				saved = new_reentry_len
				seq.log_event(SeqEvent.EVICTED, self.rank,
					f"saved_tokens={saved}, decoded={seq.decoded_length}, "
					f"new_this_cycle={new_decoded_count}")
				seq.gpu_pages_allocated = 0
				seq.host_pages_allocated = 0
				seq.host_owned_pages = 0
				seq.host_token_capacity = 0
				self._sequences_with_gpu_kv.discard(uuid)
				# Destroy the head-sharded KDA state the comment below relies
				# on. Host eviction frees the sequence's pages but reached the
				# group drop without releasing its state item, so the slot
				# stayed charged to a sequence that no longer has a group: the
				# pool leaked a slot per eviction, and a re-entry re-grouped
				# onto another group found its stale state on the old one.
				# Released only by the ranks holding the row, exactly like the
				# completion path, because the item is per rank.
				if uuid in self._uuid_to_local_map:
					self._release_kda_state_slots([int(seq.global_idx)])
				# M2b (d): eviction destroys the head-sharded KDA state on the
				# group's ranks, so drop the decode DP-group. A re-prefilled
				# sequence must re-group fresh — otherwise _assign_decode_dp_groups
				# treats the stale non-None group id as a preserved re-entry and
				# skips it, stranding the sequence with no live state. The group
				# is remembered: the decoded suffix lives only in that group's
				# node's trajectory book, so re-grouping must land on it. No-op
				# for G==1 (the field is always None on the validated pure-DP path).
				from batchgen.decode_dp_group import evict_from_decode_dp_group
				evict_from_decode_dp_group(seq)
				self.global_batch.update_status(uuid, SequenceStatus.EVICTED)

			evicted_set = set(host_evicted_uuids)
			decode_uuids = [u for u in decode_uuids if u not in evicted_set]
			batch = self._get_local_indices_for_uuids(decode_uuids)

			if self.rank == 0:
				logging.info(
					f"[HOST_KV_EVICT] Evicted {len(host_evicted_uuids)} sequences"
				)

		# The host pool is shared by every rank on a node, but each rank
		# released only its own completed/evicted rows above, while rank 0's
		# plan already counts all of those pages as free. No rank may acquire
		# host pages (growth below, or prefill after an early return) until
		# every rank has finished releasing; otherwise a fast rank sees the
		# pre-release free count and AcquirePages fails. All conditions come
		# from the broadcast decisions, so every rank takes the same branch.
		if self.world_size > 1 and (
			decisions.completed_uuids
			or decisions.host_evicted_uuids
			or (decisions.growth_feasible and decisions.host_growth_uuids)
		):
			dist.barrier()

		# C. Host KV growth. This intentionally runs after completed/evicted
		# host pages have been released so worker_view free pages match the
		# growth-debt-aware plan computed on rank 0.
		if decisions.growth_feasible and decisions.host_growth_uuids:
			host_grow_requests = []
			growth_entries = []
			for uuid, growth_pages in zip(decisions.host_growth_uuids, decisions.host_growth_pages):
				seq = self.global_batch.get_sequence(uuid)
				growth_entries.append((uuid, growth_pages, seq))
				# Host KV is one shared region per node.  G>1 ranks replicate
				# GPU/KDA state, but only the host-KV owner may mutate its page table.
				if self._owns_host_kv(seq):
					host_grow_requests.append((seq.global_idx, growth_pages))

			# The growth plan counted each node's reclaimable retained prefix
			# pages as supply; release the part free pages do not cover now,
			# after every rank's releases and before the fresh preflight.
			# Gated on the rank-uniform prefix-cache config so every rank
			# reaches the reclaim's collectives.
			if self.enable_prefix_cache:
				self._reclaim_host_prefix_pages_for_growth(
					growth_entries, worker_view
				)

			# The rank-0 planner's stats are an admission snapshot.  Recheck the
			# node-shared allocator after all releases have completed and before any
			# owner mutates it.  This mirrors GPU-KV admission and prevents one rank
			# from consuming pages on a stale free-page estimate.
			if not self._collective_host_kv_growth_capacity_ok(
				host_grow_requests, worker_view
			):
				raise RuntimeError(
					"[SCHED_INVARIANT] host-KV boundary growth rejected by fresh "
					"collective capacity preflight"
				)

			local_growth_success = True
			if host_grow_requests:
				try:
					if worker_view is None:
						raise RuntimeError("host-KV worker view is unavailable")
					worker_view.grow_pages_for_sequences(host_grow_requests)
				except Exception:
					local_growth_success = False
					logging.exception(
						"Rank %s: host-KV boundary growth failed after collective preflight",
						self.rank,
					)

			if self.world_size > 1:
				growth_success = torch.tensor(
					[1 if local_growth_success else 0],
					dtype=torch.int32,
					device=self.torch_device,
				)
				dist.all_reduce(growth_success, op=dist.ReduceOp.MIN)
				all_growth_success = bool(int(growth_success.item()))
			else:
				all_growth_success = local_growth_success
			if not all_growth_success:
				raise RuntimeError(
					"[SCHED_INVARIANT] host-KV boundary growth failed on at least "
					"one rank after collective preflight; worker state is not safe to continue"
				)

			# Commit scalar metadata only after every owner allocator call has
			# succeeded.  A failed growth therefore cannot leave a sequence claiming
			# pages that were never appended to its host chain.
			for uuid, growth_pages, seq in growth_entries:
				seq.host_token_capacity += growth_pages * seq.PAGE_SIZE
				seq.host_pages_allocated += growth_pages
				seq.host_owned_pages += growth_pages

			if host_grow_requests:
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

			# Every rank reads and writes host KV through its OWN worker-view
			# page table, so a follower whose leader just extended the shared
			# chain still holds the pre-growth table and would append the next
			# token past its last known page. Re-mirror the grown rows (page
			# tables only — the prefix attachment stays the leader's). Reached
			# by every rank under the broadcast growth condition, so the
			# gather inside stays aligned; no-op for G==1.
			self._mirror_group_host_kv(
				[u for u in decisions.host_growth_uuids if u in self._uuid_to_local_map],
				{},
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
		extension_preflight_ok = True
		if remaining_needing_ext:
			try:
				local_extension_pages = sum(
					self.global_batch.get_sequence(uuid).get_additional_gpu_pages_needed()
					for uuid in my_remaining_ext
				)
				local_extension_free = (
					gpu_manager.get_stats().num_free_pages
					if gpu_manager is not None and gpu_manager.is_initialized else 0
				)
			except Exception as exc:
				logging.exception("Rank %s: boundary extension snapshot failed", self.rank)
				self._collective_gpu_kv_allocation_preflight(1, 0)
				raise RuntimeError(
					f"[SCHED_INVARIANT] boundary extension snapshot failed: "
					f"{type(exc).__name__}: {exc}"
				)
			extension_preflight_ok = self._collective_gpu_kv_allocation_preflight(
				local_extension_pages, local_extension_free
			)

		local_extension_success = extension_preflight_ok
		if extension_preflight_ok and my_remaining_ext:
			try:
				local_extension_success = self._extend_gpu_kv_allocation(my_remaining_ext)
			except Exception:
				# The stats preflight is only advisory if the allocator itself
				# rejects a request. Convert that local failure into the second
				# collective so peers can roll back without deadlocking here.
				logging.exception(
					"Rank %s: GPU-KV extension raised after collective preflight",
					self.rank,
				)
				local_extension_success = False
		if remaining_needing_ext and extension_preflight_ok:
			# An allocator can still reject a request after the stats check. Make
			# that failure global before any rank updates status or proceeds to
			# loading.  Peers that extended successfully release the sequence's
			# full allocation below; this rolls back both the old and new pages
			# before the sequence is moved to ON_HOLD.
			local_extension_success = self._collective_gpu_kv_allocation_preflight(
				0 if local_extension_success else 1,
				1 if local_extension_success else 0,
			)

		if remaining_needing_ext and not local_extension_success:
			if my_remaining_ext:
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
				if ext_failed_global and gpu_manager is not None:
					gpu_manager.free_pages_for_sequences(ext_failed_global)
				for uuid in my_remaining_ext:
					self._sequences_with_gpu_kv.discard(uuid)

			# All ranks: zero scalars and update status for ALL failed seqs
			# (remaining_needing_ext is the globally-consistent list).
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
			try:
				new_load_local = self._get_local_indices_for_uuids(my_new_uuids)
				local_load_pages = sum(
					self.global_batch.get_sequence(self._local_to_uuid_map[local_idx])
					.get_gpu_pages_for_two_page_buffer()
					for local_idx in new_load_local
				)
				local_load_free = (
					gpu_manager.get_stats().num_free_pages
					if gpu_manager is not None and gpu_manager.is_initialized else 0
				)
			except Exception as exc:
				logging.exception("Rank %s: boundary load snapshot failed", self.rank)
				# The failed rank still joins the preflight so peers cannot
				# advance into allocation while this rank raises.
				self._collective_gpu_kv_allocation_preflight(1, 0)
				raise RuntimeError(
					f"[SCHED_INVARIANT] boundary load snapshot failed: "
					f"{type(exc).__name__}: {exc}"
				)
			load_preflight_ok = self._collective_gpu_kv_allocation_preflight(
				local_load_pages, local_load_free
			)
			if not load_preflight_ok:
				raise RuntimeError(
					f"Rank {self.rank}: collective GPU KV load preflight failed for "
					f"{len(new_load_uuids)} globally selected sequences; "
					"boundary capacity changed before allocation"
				)

			local_load_success = True
			if new_load_local:
				new_load_global = self._local_indices_to_global_seq_ids(new_load_local)
				tokens = [
					self.global_batch.get_sequence(self._local_to_uuid_map[local_idx])
					.get_gpu_pages_for_two_page_buffer() * self.PAGE_SIZE
					for local_idx in new_load_local
				]

				try:
					self._allocate_gpu_pages_with_host_identity(
						gpu_manager, new_load_global, tokens
					)
				except Exception:
					# Convert allocator failures into the same collective result
					# before any rank launches an async host-KV copy.
					logging.exception(
						"Rank %s: GPU-KV load allocation raised after collective preflight",
						self.rank,
					)
					local_load_success = False

			load_success = self._collective_gpu_kv_allocation_preflight(
				0 if local_load_success else 1,
				1 if local_load_success else 0,
			)
			if not load_success:
				if local_load_success and new_load_global and gpu_manager is not None:
					gpu_manager.free_pages_for_sequences(new_load_global)
				raise RuntimeError(
					"[SCHED_INVARIANT] collective GPU-KV load allocation failed "
					f"for {len(new_load_uuids)} globally selected sequences"
				)

			if new_load_local:
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

		# This returns a NEW ordered row list, so stamp it here rather than at
		# each caller: both the page-boundary planner and the decode loop call
		# this, and a caller that forgets would leave row-keyed state stale.
		self._bump_decode_row_epoch()

		return updated_uuids, updated_batch

	def _integrate_pending_async_load(
		self,
		pending_async_task: Optional[object],
		pending_load_uuids: List[str],
		pending_load_local_indices: List[int],
		pending_load_global_ids: List[int],
		decode_uuids: List[str],
		batch: List[int],
		gpu_manager,
	) -> Tuple[List[str], List[int]]:
		"""Complete an in-flight boundary GPU-KV load and fold its rows in.

		A launched boundary load has already allocated GPU pages -- and
		registered their shared-page keys -- for rows that are NOT yet in
		``decode_uuids``; only the finalize step moves them in. Every decode
		exit that drops its pending-load state must therefore run this first:
		an exit that forgets the pending rows strands their GPU pages (no
		later path frees pages for a row outside ``decode_uuids``), and the
		next load of such a row finds its own stale keyed pages still mapped
		and fails the writable-page sharing guard, which the collective
		converts into a cohort-fatal SCHED_INVARIANT.

		All ranks hold identical ``pending_load_uuids`` and reach this from
		broadcast boundary decisions, so the wait/barrier/finalize protocol
		stays rank-aligned.
		"""
		if pending_async_task is not None:
			pending_async_task.wait()
			torch.cuda.synchronize(self.torch_device)
		dist.barrier()
		return self._finalize_async_load_minimal(
			pending_async_task,
			pending_load_uuids,
			pending_load_local_indices,
			pending_load_global_ids,
			decode_uuids,
			batch,
			gpu_manager,
		)

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
			# Module-attribute read: the flag's single definition lives on
			# decode_engine and is documented as flippable on that module at
			# runtime; a from-import here would freeze a copy.
			_deferred = _decode_engine.DECODE_DEFERRED_TOKEN_READBACK
			if _deferred:
				# With the readback deferred the host runs one step ahead of
				# the stream, so the previous step's copy out of this pinned
				# buffer may not have executed yet: rewriting it now would
				# hand that step this step's lengths. The wait is on a copy
				# queued before the previous forward, so it is already done
				# or about to be; it never waits for a forward.
				_staging_event = getattr(self, "_decode_metadata_staging_event", None)
				if _staging_event is not None:
					_staging_event.synchronize()
			for i, value in enumerate(cache_seqlens_key):
				staging[i] = value
			cache_view.copy_(staging[:batch_size], non_blocking=True)
			if _deferred:
				if getattr(self, "_decode_metadata_staging_event", None) is None:
					self._decode_metadata_staging_event = torch.cuda.Event()
				self._decode_metadata_staging_event.record(
					torch.cuda.current_stream(self.torch_device)
				)
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
		self._finalize_gpu_kv_pool_after_capture(gpu_manager)

	@staticmethod
	def _glm5_dsa_graph_score_capacity_tokens(
		primary_page_table,
		primary_page_size: int,
		aux_page_table,
		aux_page_size: int,
		*,
		deployment_context_tokens: int,
		span_cap_tokens: int | None = None,
	) -> int:
		"""Token span the whole-model decode graph is captured for.

		`deployment_context_tokens` is the bound from
		`_deployment_context_bound_tokens` and is REQUIRED: it is the only term
		here that does not move with the batch, and dropping it (which an
		optional argument defaulting to None used to allow whenever the model
		config carried no `max_position_embeddings`) leaves the span at
		whatever the page tables happened to be sized for when the FIRST
		decode batch created them.

		The page-table terms stay in the `min` as hardware limits — a span
		wider than a table's columns is unaddressable — and `span_cap_tokens`
		applies the context-proportional statics budget.
		"""
		if int(deployment_context_tokens) <= 0:
			raise RuntimeError(
				"the DSA CUDA graph span requires a positive deployment context "
				f"bound, got {deployment_context_tokens}; model_context_length is "
				"set during core initialization and must be available before capture"
			)
		primary_capacity = int(primary_page_table.shape[1]) * int(primary_page_size)
		aux_capacity = int(aux_page_table.shape[1]) * int(aux_page_size)
		capacities = [
			primary_capacity,
			aux_capacity,
			int(deployment_context_tokens),
		]
		if span_cap_tokens is not None and int(span_cap_tokens) > 0:
			capacities.append(int(span_cap_tokens))
		capacity = min(capacities)
		if capacity <= 0:
			raise RuntimeError(
				"GLM-5 DSA CUDA graph requires positive primary/aux page-table capacity"
			)
		return capacity

	def _glm5_dsa_graph_required_tokens(
		self,
		active_sequence_ids: Sequence[int],
		*,
		page_size: int,
	) -> int:
		"""Smallest page-aligned decode length the current graph must cover.

		Budget-based: ``kv_token_budget`` bounds every row's final length for
		the whole graph lifetime, so a graph specialized on this value can never
		be outgrown mid-lifetime. With no live row to read a budget from, the
		graph must cover any row admission can still produce, which is the
		deployment bound. The worker-wide ``max_input_length +
		max_decoding_length`` describes only the first pool batch.
		"""
		budgets = [
			int(seq.kv_token_budget)
			for seq in self.global_batch
			if seq.status != SequenceStatus.COMPLETED
		] if self.global_batch is not None else []
		if not budgets and active_sequence_ids:
			by_gid = {
				int(seq.global_idx): seq
				for seq in self.global_batch
			} if self.global_batch is not None else {}
			budgets = [
				int(by_gid[seq_id].kv_token_budget)
				for seq_id in active_sequence_ids
				if int(seq_id) in by_gid
			]
		if budgets:
			required = max(budgets)
		else:
			required = self._deployment_context_bound_tokens()
			if required <= 0:
				raise RuntimeError(
					"GLM-5 DSA graph span with no live row requires the deployment "
					"max_model_len, which is unset before core init"
				)
		return math.ceil(required / int(page_size)) * int(page_size)

	def _debug_flag_enabled(self, value) -> bool:
		if isinstance(value, bool):
			return value
		if isinstance(value, (int, float)):
			return value != 0
		if isinstance(value, str):
			return value.strip().lower() in {"1", "true", "yes", "on"}
		return False

	def _weight_pass_end_delay(self, value) -> float:
		"""Seconds this rank sleeps before closing each prefill weight pass.

		``value`` is the batch-level ``weight_pass_end_delay`` debug entry,
		``{"rank": <global rank>, "seconds": <float>}``; any other rank, or no
		entry, gets 0.
		"""
		if value is None:
			return 0.0
		if (
			not isinstance(value, dict)
			or not isinstance(value.get("rank"), int)
			or not isinstance(value.get("seconds"), (int, float))
			or not 0 < value["seconds"] <= 60
		):
			raise ValueError(
				"batchgen_debug.weight_pass_end_delay must be "
				'{"rank": <int>, "seconds": <0 < s <= 60>}, '
				f"got {value!r}"
			)
		return float(value["seconds"]) if value["rank"] == self.rank else 0.0

	def _glm5_graph_path_log_requested_for_current_batch(self) -> bool:
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
		return glm5_whole_model_cuda_graph_requested(
			getattr(self, "model_name", None),
			enable_cuda_graph=getattr(
				getattr(self, "args", None),
				"enable_cuda_graph",
				False,
			),
		)

	def _glm5_whole_model_graph_compare_requested_for_current_batch(self) -> bool:
		debug = self._batchgen_debug or getattr(AttnWrapperBase, "batchgen_debug", None) or {}
		if not isinstance(debug, dict):
			return False
		return self._debug_flag_enabled(debug.get("glm5_whole_model_graph_compare"))

	def _glm5_whole_model_graph_timing_requested_for_current_batch(self) -> bool:
		debug = self._batchgen_debug or getattr(AttnWrapperBase, "batchgen_debug", None) or {}
		if isinstance(debug, dict) and self._debug_flag_enabled(debug.get("glm5_whole_model_graph_timing")):
			return True
		return self._glm5_whole_model_graph_compare_requested_for_current_batch()

	def _glm5_whole_model_graph_compare_fail_on_mismatch(self) -> bool:
		debug = self._batchgen_debug or getattr(AttnWrapperBase, "batchgen_debug", None) or {}
		if not isinstance(debug, dict):
			return False
		return self._debug_flag_enabled(debug.get("glm5_whole_model_graph_compare_fail"))

	def _glm5_whole_model_graph_debug_value(self, key: str):
		"""The current batch's ``batchgen_debug[key]`` (None when absent)."""
		debug = self._batchgen_debug or getattr(AttnWrapperBase, "batchgen_debug", None) or {}
		if not isinstance(debug, dict):
			return None
		return debug.get(key)

	def _glm5_whole_model_graph_compare_tolerance(self, key: str) -> float:
		"""``glm5_whole_model_graph_compare_atol`` / ``_rtol``: default 1e-2.

		A number, or a string ``float()`` parses (the grammar of the env vars
		these keys replace). Anything else is refused by name.
		"""
		value = self._glm5_whole_model_graph_debug_value(key)
		if value is None:
			return 1e-2
		if isinstance(value, bool) or not isinstance(value, (int, float, str)):
			raise ValueError(f"batchgen_debug.{key} must be a number, got {value!r}")
		try:
			return float(value)
		except ValueError as exc:
			raise ValueError(f"batchgen_debug.{key} must be a number, got {value!r}") from exc

	def _glm5_whole_model_graph_probe_layers(self, num_layers: int) -> tuple:
		"""``glm5_whole_model_graph_probe_layers``: decoder layers to probe.

		``"all"``, a comma-separated index string (the grammar of the env var
		this key replaces), or a list of ints; absent/empty means no probes.
		Anything else is refused by name.
		"""
		value = self._glm5_whole_model_graph_debug_value("glm5_whole_model_graph_probe_layers")
		if value is None:
			return ()
		if isinstance(value, str):
			if value.strip().lower() == "all":
				return tuple(range(num_layers))
			return tuple(int(part.strip()) for part in value.split(",") if part.strip())
		if isinstance(value, (list, tuple)) and all(
			isinstance(item, int) and not isinstance(item, bool) for item in value
		):
			return tuple(value)
		raise ValueError(
			"batchgen_debug.glm5_whole_model_graph_probe_layers must be \"all\", a "
			f"comma-separated index string or a list of ints, got {value!r}"
		)

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
		if self._glm5_whole_model_graph_requested_for_current_batch():
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
			# The round's criticality probe request (None without one): its
			# spin nodes change the graph, so a graph captured for another
			# request must not replay.
			getattr(self, "_criticality_probe_request", None),
		)

	def _configure_criticality_probe(self, debug: Optional[dict]) -> None:
		"""Decode configure: record the round's criticality probe request.

		The request, its flag and its validation belong to the model's graph
		adapter; a model without one has no probe and the request is None.
		The resident decode graph outlives decode rounds, so a request that
		differs from the installed probe's (None is a value on both sides)
		arms ONE recapture: the warmup that follows releases the resident
		graph and captures it again, instrumented for a new request and plain
		for the first round without one. An equal request keeps the graph.
		Every rank merges the same global decode sequences, so every rank
		takes the same decision; the recapture runs behind the capture path's
		rank barriers.
		"""
		adapter = self._graph.cuda_graph_adapter
		requested = None
		if adapter is not None:
			args = getattr(self, "args", None)
			graph_enabled = bool(getattr(args, "enable_cuda_graph", False)) and not bool(
				getattr(args, "disable_cuda_graphs", False)
			)
			requested = adapter.criticality_probe_request(debug, graph_enabled=graph_enabled)
			adapter.set_criticality_probe_request(requested)
		installed_probe = getattr(self, "_criticality_probe", None)
		installed = None if installed_probe is None else installed_probe.signature()
		self._criticality_probe_request = requested
		if installed == requested:
			return
		if self.rank == 0:
			logging.info(
				"[CRITICALITY_PROBE] decode configure: request changed "
				"(installed=%s, requested=%s); recapturing the decode graph",
				"none" if installed is None else "probe",
				"none" if requested is None else "probe",
			)
		self._criticality_probe_recapture = True

	def _decode_graph_capture_needed_at_configure(self, has_queueing: bool) -> bool:
		"""Whether this decode configure runs the CUDA graph warmup/capture.

		Phase C: layer / DSA / MoE / segmented graph modes are retired; only
		the whole-model graph remains for GLM-5. Warmup triggers when (a) the
		generic should_warmup helper says so, (b) the whole-model graph is
		requested but no graph manager exists, (c) the current bucket is
		missing, or (d) a changed criticality probe request armed a recapture.
		A decode instance whose graphs were captured before readiness answers
		False on every configure.
		"""
		from batchgen.models.glm.glm5.cuda_graph_policy import (
			should_warmup_cuda_graphs_before_decode,
		)

		glm5_whole_graph_requested = (
			self._glm5_whole_model_graph_requested_for_current_batch()
		)
		generic_cuda_graph_warmup_needed = should_warmup_cuda_graphs_before_decode(
			graph_manager_is_initialized=self._graph.cuda_graph_manager is not None,
			global_batch_has_queueing=has_queueing,
			model_name=getattr(self, "model_name", None),
			enable_cuda_graph=getattr(self.args, "enable_cuda_graph", False),
		)
		return bool(
			generic_cuda_graph_warmup_needed
			or (glm5_whole_graph_requested and self._graph.cuda_graph_manager is None)
			or self._glm5_whole_model_graph_current_bucket_missing()
			or self._criticality_probe_recapture
		)

	def _glm5_whole_model_graph_current_bucket_missing(self) -> bool:
		model_name_l = (getattr(self, 'model_name', '') or '').lower()
		if (
			not self._glm5_whole_model_graph_requested_for_current_batch()
			or "glm" not in model_name_l
		):
			return False
		if self._graph.glm5_whole_model_graph_unavailable_reason:
			return False
		max_bsz = int(getattr(self, "_current_decode_max_rank_batch_size", 0) or 0)
		if max_bsz <= 0:
			return False
		capture_attempted = bool(
			self._graph.glm5_whole_model_graph_capture_attempted_for_batch
		)
		if self._graph.cuda_graph_manager is None or not self._graph.glm5_whole_model_graph:
			return not capture_attempted
		try:
			bucket = self._graph.cuda_graph_manager.bucketing.get_padded_size(max_bsz)
		except ValueError:
			return False
		if bucket in self._graph.glm5_whole_model_graph_failed_buckets:
			return False
		if not self._graph.cuda_graph_manager.has_bucket_for_all_segments(max_bsz):
			if capture_attempted:
				return False
			return True
		current_max_seqlen = int(getattr(AttnWrapperBase, "max_seqlen", 0) or 0)
		captured_max_seqlen = int(getattr(self._graph.whole_model_segment, "max_seqlen", 0) or 0)
		if (
			current_max_seqlen > 0
			and captured_max_seqlen > 0
			and current_max_seqlen > captured_max_seqlen
		):
			if (
				capture_attempted
				and not self._graph.glm5_whole_model_graph_state_change_after_capture_logged
			):
				logging.info(
					f"Rank {self.rank}: GLM-5 whole-model CUDA graph max_seqlen "
					f"{captured_max_seqlen} is below current decode max_seqlen "
					f"{current_max_seqlen}; using eager decode instead of recapturing"
				)
				self._graph.glm5_whole_model_graph_state_change_after_capture_logged = True
			return not capture_attempted
		signature = self._glm5_whole_model_graph_capture_signature(bucket)
		if signature != self._graph.glm5_whole_model_graph_signature:
			if (
				capture_attempted
				and not self._graph.glm5_whole_model_graph_state_change_after_capture_logged
			):
				logging.info(
					f"Rank {self.rank}: GLM-5 whole-model CUDA graph page-table storage "
					"changed after configured capture; using eager decode instead of recapturing"
				)
				self._graph.glm5_whole_model_graph_state_change_after_capture_logged = True
			return not capture_attempted
		return False

	def _release_glm5_whole_model_graph_state(self, *, empty_cuda_cache: bool = False) -> None:
		manager = self._graph.cuda_graph_manager
		segment = self._graph.whole_model_segment
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
				bucket_sizes = list(getattr(self._graph.whole_model_bucketing, "bucket_sizes", []) or [])
				for bucket in bucket_sizes:
					release(int(bucket))
		self._graph.cuda_graph_manager = None
		self._graph.whole_model_segment = None
		self._graph.glm5_whole_model_all_short = False
		self._graph.glm5_whole_model_index_topk = 0
		self._graph.whole_model_bucketing = None
		self._graph.whole_model_graph = False
		self._graph.glm5_whole_model_graph = False
		self._graph.glm5_whole_model_graph_signature = None
		# The probe belongs to the released graph (its spin nodes bake the
		# probe's delay table), so it goes with it.
		self._criticality_probe = None
		# Phase B: also drop the adapter's reference to the captured segment so
		# the next worker generation starts from a clean adapter context.
		if self._graph.cuda_graph_adapter is not None:
			try:
				self._graph.cuda_graph_adapter.release_all(manager=None)
			except Exception as _exc:
				logging.warning(
					"Phase B: failed to release adapter state: %s", _exc,
				)
		manager = None
		segment = None
		gc.collect()
		if empty_cuda_cache and torch.cuda.is_available():
			torch.cuda.empty_cache()

	# Path states that mean "this step cannot replay the graph" but are still
	# served correctly by eager decode, so a required graph must NOT make them
	# fatal. A step longer than the captured span is the only such state: the
	# eager decode path is length-agnostic. Every other reason means the
	# capture itself is missing or stale, which stays fail-loud.
	_WHOLE_GRAPH_EAGER_SERVABLE_REASONS = frozenset({"max_seqlen_exceeds_capture"})

	def _warn_decode_graph_span_exceeded(self, step_max_seqlen: int, span: int) -> None:
		"""Rate-limited WARN naming this step's max_seqlen and the captured span."""
		now = time.monotonic()
		last = self._graph.decode_graph_span_warn_monotonic
		if last is not None and now - last < _GRAPH_SPAN_WARN_INTERVAL_S:
			return
		self._graph.decode_graph_span_warn_monotonic = now
		logging.warning(
			"Rank %s: the whole-model decode graph covers %d context tokens but "
			"this step's max_seqlen is %d; running the step in eager decode. The "
			"span is the deployment context bound trimmed by the graph's "
			"context-proportional statics budget (%d bytes).",
			self.rank,
			int(span),
			int(step_max_seqlen),
			GRAPH_CONTEXT_STATICS_BUDGET_BYTES,
		)

	def _glm5_whole_graph_path_state(self, max_rank_bsz: int):
		model_name_l = (getattr(self, "model_name", "") or "").lower()
		if (
			not self._glm5_whole_model_graph_requested_for_current_batch()
			or "glm" not in model_name_l
		):
			return "disabled", None, "not_requested"
		if max_rank_bsz <= 0:
			return "eager", None, "empty_global_batch"
		if self._graph.glm5_whole_model_graph_unavailable_reason:
			return "eager", None, "unavailable"
		manager = self._graph.cuda_graph_manager
		if manager is None or not self._graph.glm5_whole_model_graph:
			return "eager", None, "no_manager"
		try:
			bucket = manager.bucketing.get_padded_size(max_rank_bsz)
		except ValueError:
			return "eager", None, "over_bucket"
		if bucket in self._graph.glm5_whole_model_graph_failed_buckets:
			return "eager", bucket, "failed_bucket"
		if not manager.has_bucket_for_all_segments(max_rank_bsz):
			return "eager", bucket, "bucket_not_captured"
		current_max_seqlen = int(getattr(AttnWrapperBase, "max_seqlen", 0) or 0)
		captured_max_seqlen = int(getattr(self._graph.whole_model_segment, "max_seqlen", 0) or 0)
		if (
			current_max_seqlen > 0
			and captured_max_seqlen > 0
			and current_max_seqlen > captured_max_seqlen
		):
			return "eager", bucket, "max_seqlen_exceeds_capture"
		# An all_short capture never runs indexer scoring, so rows past
		# index_topk must leave the graph; the eager path still selects
		# top-k correctly for them.
		index_topk_cfg = int(self._graph.glm5_whole_model_index_topk or 0)
		if (
			self._graph.glm5_whole_model_all_short
			and index_topk_cfg > 0
			and current_max_seqlen > index_topk_cfg
		):
			return "eager", bucket, "all_short_exceeded"
		signature = self._glm5_whole_model_graph_capture_signature(bucket)
		if signature != self._graph.glm5_whole_model_graph_signature:
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

		num_valid_tokens = torch.empty((1,), dtype=torch.int32, device=self.torch_device)
		num_valid_tokens.fill_(local_bsz)
		return {
			"cache_seqlens": cache_seqlens_i32,
			"position_ids": position_ids_i64,
			"primary_slot_indices": _graph_slots(primary_manager),
			"aux_slot_indices": _graph_slots(aux_manager),
			"num_valid_tokens": num_valid_tokens,
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

	def _make_glm5_whole_model_capture_inputs(
		self,
		*,
		bucket: int,
		num_heads: int,
	):
		return {
			"input_ids": torch.empty((0, 1), dtype=torch.int64, device=self.torch_device),
			"cache_seqlens": torch.empty((0,), dtype=torch.int32, device=self.torch_device),
			"position_ids": torch.empty((0, 1), dtype=torch.int64, device=self.torch_device),
			"primary_slot_indices": torch.empty((0,), dtype=torch.int32, device=self.torch_device),
			"aux_slot_indices": torch.empty((0,), dtype=torch.int32, device=self.torch_device),
			"rank_token_counts": torch.zeros((self.world_size,), dtype=torch.int64, device=self.torch_device),
			"num_valid_tokens": torch.zeros((1,), dtype=torch.int32, device=self.torch_device),
		}

	def _finalize_gpu_kv_pool_after_capture(self, gpu_manager) -> None:
		"""Measure-once-cache finalization (semantic contract, POIS 2026-10-05).

		Called once, right after the full graph warmup/capture, while the
		PROVISIONAL pool is still allocated. The capture pass has materialized
		every persistent allocation (graph pools, static buffers, NCCL
		per-collective resources, JIT caches), so the steady residency is
		``used - provisional_pool`` and the final pool is everything the memory
		fraction allows above it. The result is CACHED; every later pool
		(re)creation reuses it, making the pool identical across decode rounds
		(invariant 1) and retiring the fresh-server round-1 protocol.

		The size is the device total minus that steady residency minus the
		explicit ``GPU_KV_POOL_HEADROOM_GB``. ``gpu_memory_frac`` is NOT
		applied again here: its slack is what covers the graph footprint the
		PRE-capture measurement cannot see, and once that footprint is inside
		the measurement, re-applying the fraction charges it twice and parks
		``(1 - frac) * total`` permanently free with no named consumer. The
		headroom constant is that consumer, measured and auditable.

		The current (first) round keeps the provisional pool: its sequences
		already hold pages, the captured graphs bake its addresses, and this
		codebase deliberately refuses to recapture after a configured capture
		(``_setup_cuda_graphs`` falls back to eager instead, and the reuse
		lifecycle raises), so the live pool cannot be resized under them. With
		the headroom sizing the provisional is the SMALLER pool (it still
		carries the unmeasured-graph slack), so round 1 is a conservative
		startup artifact, never an over-commitment. The cached size applies
		from the NEXT pool creation (the per-round destroy/recreate that
		already exists), from which point the page count is identical for
		every round.
		"""
		if not self._gpu_kv_sizing_provisional:
			return
		self._gpu_kv_sizing_provisional = False

		torch.cuda.synchronize(self.torch_device)
		torch.cuda.empty_cache()
		free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(self.local_rank)
		total_mem_gb = total_mem_bytes / (1024 ** 3)
		used_mem_gb = total_mem_gb - free_mem_bytes / (1024 ** 3)
		provisional_gb = self.gpu_kv_cache_size_gb
		steady_used_gb = used_mem_gb - provisional_gb

		final_gb = KVCacheManager.compute_final_pool_gb(
			total_mem_gb,
			used_mem_gb,
			provisional_gb,
		)

		# Broadcast rank 0's decision so every rank resizes identically.
		size_tensor = torch.tensor([final_gb], dtype=torch.float32, device=self.torch_device)
		dist.broadcast(size_tensor, src=0)
		final_gb = float(size_tensor.item())

		# Fail-loud floor (contract invariant 3).
		manager = self.gpu_paged_kv_cache_manager
		stats = manager.get_stats()
		bytes_per_page = (
			provisional_gb * (1024 ** 3) / max(stats.num_total_pages, 1)
		)
		final_pages = KVCacheManager.pool_pages_for_gb(final_gb, bytes_per_page)
		bucket_sizes = list(getattr(self._graph.whole_model_bucketing, "bucket_sizes", []) or [1])
		floor_pages = KVCacheManager.pool_floor_pages(max(bucket_sizes))
		if final_gb <= 0 or final_pages < floor_pages:
			raise RuntimeError(
				f"[GPU-KV] Final pool below floor: {final_gb:.2f} GB "
				f"(~{final_pages} pages) < {floor_pages} pages required for the "
				f"largest decode bucket. Steady residency {steady_used_gb:.2f} GB "
				f"of {total_mem_gb:.2f} GB leaves nothing above the "
				f"{GPU_KV_POOL_HEADROOM_GB:.2f} GB pool headroom; fix the "
				"configuration."
			)

		# Cache only: the live round keeps the provisional pool (its sequences
		# hold pages and the captured graphs bake its addresses); every later
		# pool creation uses the cached final size.
		self._gpu_kv_cached_pool_gb = final_gb
		if self.rank == 0:
			logging.info(
				f"[GPU-KV] Finalized pool size: {final_gb:.2f} GB "
				f"(~{final_pages} pages; steady used {steady_used_gb:.2f} GB of "
				f"{total_mem_gb:.2f} GB; headroom {GPU_KV_POOL_HEADROOM_GB:.2f} GB; "
				f"provisional {provisional_gb:.2f} GB stays for this round, delta "
				f"{final_gb - provisional_gb:+.2f} GB; cached for every later pool "
				"creation)"
			)
			logging.info(
				"[GPU-KV] Pool margin is now the explicit %.2f GB headroom; "
				"gpu_memory_frac=%.3f bounds only the pre-capture provisional "
				"measurement (the frac-based final size would have been %.2f GB)",
				GPU_KV_POOL_HEADROOM_GB,
				self.gpu_memory_frac,
				provisional_pool_gb(
					total_mem_gb, self.gpu_memory_frac, steady_used_gb, floor_gb=None
				),
			)

	def _setup_cuda_graphs(self, gpu_manager):
		"""Capture CUDA graphs for decode: full attention block per layer.

		Each graph captures the entire attention block in one shot:
		  RMSNorm → QKV proj → split → reshape → RoPE → KV write → FA → O_proj
		  → residual add + post-attn RMSNorm

		Dynamic metadata (cache_seqlens) is passed as a static-address input buffer.
		KV cache, page table, and cos/sin tables are at fixed GPU addresses.
		"""
		from batchgen.models.openai.gpt_oss_120b.cuda_graph_segments import (
			FullAttnSegment, MoESegment, MoEComputeSegment, SharedMoEBufferPool,
			WholeModelSegment,
		)
		from batchgen.models.wrappers.attention import AttnWrapperBase

		# Detect K2.5 model for specialized graph segment
		_is_k25 = is_kimi_k25_backend_model(self.model_name)

		# The ladder covers the per-rank decode width: decode admission is
		# bounded by that width, so every admitted step has a bucket to
		# replay and a step above the top bucket is an invariant violation,
		# not an eager fallback.
		bucket_sizes = self._decode_graph_bucket_sizes()
		decode_width = self._max_decode_rank_bsz()
		check_decode_width_within_buckets(decode_width, bucket_sizes)
		if not getattr(self, "_decode_graph_ladder_logged", False):
			logging.info(
				"CUDA graph bucket sizes: %s (decode_width=%d, "
				"--cuda-graph-max-bucket-size=%d, --cuda-graph-num-buckets=%d)",
				bucket_sizes,
				decode_width,
				int(self.args.cuda_graph_max_bucket_size),
				int(self.args.cuda_graph_num_buckets),
			)
			self._decode_graph_ladder_logged = True
		bucketing = self._graph.new_bucketing(bucket_sizes)
		manager = self._graph.new_manager(bucketing, device=self.torch_device)

		# Use model's max_position_embeddings (not the deployment max_model_len) so the
		# RoPE cos/sin cache captured in the graph covers ALL possible positions.
		max_rope_len = getattr(self.model_config, 'max_position_embeddings', 131072)
		# Phase C: only the whole-model graph remains for GLM-5 (the dsa / moe /
		# dsa_full / layer modes are retired and their branches deleted).
		glm5_whole_graph_enabled = (
			self._glm5_whole_model_graph_requested_for_current_batch()
		)
		if glm5_whole_graph_enabled:
			# The request is the requirement: requested here means required.
			whole_graph_required = glm5_whole_graph_enabled
			local_bsz = int(getattr(self, "_current_decode_local_batch_size", 0) or 0)
			max_bsz = int(getattr(self, "_current_decode_max_rank_batch_size", 0) or 0)
			capture_attempted = bool(
				self._graph.glm5_whole_model_graph_capture_attempted_for_batch
			)
			if max_bsz <= 0:
				logging.info(
					f"Rank {self.rank}: no global GLM-5 decode rows; skipping whole-model graph capture"
				)
				return
			failed_buckets = self._graph.glm5_whole_model_graph_failed_buckets
			capture_buckets = [
				int(bucket)
				for bucket in bucketing.bucket_sizes
				if int(bucket) not in failed_buckets
			]
			if not capture_buckets:
				self._graph.glm5_whole_model_graph_capture_attempted_for_batch = True
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
			existing_manager = self._graph.cuda_graph_manager
			existing_signature = self._glm5_whole_model_graph_capture_signature()
			if existing_manager is not None and self._graph.glm5_whole_model_graph:
				missing_configured = any(
					not existing_manager.has_bucket_for_all_segments(bucket)
					for bucket in capture_buckets
				)
				if (
					not missing_configured
					and existing_signature == self._graph.glm5_whole_model_graph_signature
				):
					self._graph.glm5_whole_model_graph_capture_attempted_for_batch = True
					return
				# A recapture armed by a changed criticality probe request is
				# the one deliberate exception: it falls through to the stale
				# graph release below and captures in place.
				if capture_attempted and not getattr(self, "_criticality_probe_recapture", False):
					logging.info(
						f"Rank {self.rank}: GLM-5 whole-model CUDA graph state changed after "
						"configure-time capture; using eager decode instead of recapturing"
					)
					return
			self._assert_decode_graph_capture_allowed()

			from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
				Glm5WholeModelSegment,
				make_glm5_whole_model_graph_segment_name,
			)
			from batchgen.models.glm.glm5.cuda_graph_segments import Glm5FullDsaAttnSegment
			from batchgen.models.glm.glm5.layer_cuda_graph_segments import (
				Glm5DecoderLayerGraphSegment,
			)
			from batchgen.models.glm.glm5.model import Glm5MoE, resolve_glm5_3d_mtp
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
			# The span comes from the DEPLOYMENT bound, never from the batch in
			# front of us: the first decode batch used to fix it for the life
			# of the process, so a later, longer turn had no replayable graph.
			# A 1-byte aux element type is the page-split FP8 layout, which
			# decides which scorer owns the context-proportional statics.
			aux_is_page_split = int(aux_manager.config.kv_dtype.itemsize) == 1
			capacity_seqlen = self._glm5_dsa_graph_score_capacity_tokens(
				primary_page_table,
				primary_page_size,
				aux_page_table,
				aux_page_size,
				deployment_context_tokens=self._deployment_context_bound_tokens(),
				span_cap_tokens=graph_context_span_cap_tokens(
					bucket_sizes=bucketing.bucket_sizes,
					aux_is_page_split=aux_is_page_split,
				),
			)
			graph_max_seqlen = int(capacity_seqlen)
			# A batch already past the span is not a capture error: capturing
			# at the span still serves every shorter step, and this step takes
			# the eager path (see the decode-step fallback).
			current_capture_max_seqlen = int(getattr(AttnWrapperBase, "max_seqlen", 0) or 0)
			if current_capture_max_seqlen > graph_max_seqlen:
				self._warn_decode_graph_span_exceeded(
					current_capture_max_seqlen, graph_max_seqlen
				)
			# all_short lifetime: every admissible row's final length fits in
			# index_topk, so the DSA segments attend densely to the resident
			# page cache and skip indexer scoring/top-k for the graph lifetime.
			# index_topk comes from the live indexer module — the value that
			# actually drives scoring; a local-checkpoint model_config may not
			# carry the field at all.
			required_seqlen = self._glm5_dsa_graph_required_tokens(
				active_sequence_ids,
				page_size=primary_page_size,
			)
			_first_indexer = next(
				(
					getattr(layer.self_attn.module, "indexer", None)
					for layer in self.model.model.layers
					if getattr(layer.self_attn.module, "indexer", None) is not None
				),
				None,
			)
			index_topk_cfg = int(getattr(_first_indexer, "index_topk", 0) or 0)
			# ONE general graph lifetime (POIS 2026-10-13). The all_short lifetime
			# is a second, narrower lifetime: it only serves rows whose final
			# length fits in index_topk, so a batch that grows past that bound has
			# to fall back to eager or force a recapture. That is why the default
			# path recaptured every bucket on every phase switch, and why the
			# persistent path rejected long rows outright. Pinning the selection to
			# the general lifetime makes the decode graphs capture-once and
			# length-agnostic (their span still comes from KV capacity via
			# graph_max_seqlen above), at the cost of always scoring the indexer.
			# The all_short capture code itself is left in place and never selected.
			all_short = False
			self._graph.glm5_whole_model_all_short = all_short
			self._graph.glm5_whole_model_index_topk = index_topk_cfg
			if self.rank == 0:
				logging.info(
					"Rank %s: GLM-5 whole-model graph attention span "
					"required_seqlen=%s index_topk=%s all_short=%s "
					"(general lifetime pinned)",
					self.rank,
					required_seqlen,
					index_topk_cfg,
					all_short,
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
				if indexer is None and layer_idx == 0:
					# Reuse-topk segments need a producing full layer first.
					raise RuntimeError(
						"Layer 0: GLM-5 whole-model graph requires a DSA "
						"indexer on the first layer")
				if getattr(wrapper, "_fp8_absorb_weights", None) is None:
					wrapper.initialize_decode_absorb()
				if indexer is not None and (
					getattr(wrapper, "_fused_wqb_weights", None) is None
					or getattr(wrapper, "_indexer_cuda_module", None) is None
				):
					# Indexer-only kernels; GLM-5.2 skip layers have none and
					# get Glm5ReuseTopkAttnSegment instead.
					wrapper.initialize_fused_kernels()
				if getattr(wrapper, "_fp8_absorb_weights", None) is None:
					raise RuntimeError(f"Layer {layer_idx}: GLM-5 whole-model graph requires FP8 absorb weights")
				if (
					getattr(wrapper, "_fp8_qkv_a_proj", None) is None
					or getattr(wrapper, "_fp8_qkv_a_scale", None) is None
				):
					raise RuntimeError(
						f"Layer {layer_idx}: GLM-5 whole-model graph requires "
						"fused Q-A/KV-A weights"
					)
				if indexer is not None:
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
				self._graph.glm5_whole_model_graph_unavailable_reason = reason
				if whole_graph_required:
					raise RuntimeError(reason)
				logging.warning("%s Using eager decode without whole-model graph compare.", reason)
				return

			if self._graph.glm5_whole_model_graph or self._graph.whole_model_segment is not None:
				logging.info(
					f"Rank {self.rank}: releasing stale GLM-5 whole-model CUDA graph "
					"before configure-time capture"
				)
				self._release_glm5_whole_model_graph_state(empty_cuda_cache=True)

			manager = self._graph.new_manager(bucketing, device=self.torch_device)
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
					# Graph decode fan-in is bounded by the largest captured
					# bucket across all ranks; the pool still takes
					# max(base_mtp, round_up(world_size x max_bucket)).
					base_mtp=resolve_glm5_3d_mtp(
						self.world_size * max(bucket_sizes)
					),
				)
			shared_dsa_buffers = {}
			shared_reuse_buffers = {}
			last_full_segment = None
			last_cos_table = None
			last_sin_table = None
			last_index_topk = None
			layer_segments = []
			for layer_idx, decoder_layer in enumerate(self.model.model.layers):
				wrapper = decoder_layer.self_attn
				indexer = wrapper.module.indexer
				primary_blocked_k = primary_k_cache[layer_idx]
				if indexer is None:
					# GLM-5.2 skip_topk layer: no indexer module. Reuse the
					# most recent full layer's top-k (eager semantics,
					# glm5_decode_selector.py:443-461) via the reuse segment.
					# Rope tables are layer-invariant; borrow the producer's.
					if last_full_segment is None:
						raise RuntimeError(
							"GLM-5 whole-model graph: first layer has no DSA "
							"indexer; layer 0 must be a full indexer layer"
						)
					from batchgen.models.glm.glm5.reuse_topk_segment import (
						Glm5ReuseTopkAttnSegment,
					)
					dsa_segment = Glm5ReuseTopkAttnSegment(
						wrapper=wrapper,
						primary_blocked_k=primary_blocked_k,
						primary_page_table=primary_page_table,
						absorb_weights=wrapper._fp8_absorb_weights,
						cos_table=last_cos_table,
						sin_table=last_sin_table,
						max_seqlen=graph_max_seqlen,
						index_topk=last_index_topk,
						page_size=primary_page_size,
						topk_source=last_full_segment,
						all_short=all_short,
						shared_buffers=shared_reuse_buffers,
					)
				else:
					# The aux pool may hold only the indexer layers (GLM-5.2):
					# resolve the physical slot instead of indexing by engine layer.
					aux_blocked_k = aux_k_cache[aux_manager.resolve_physical_layer(layer_idx)]
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
						all_short=all_short,
						shared_buffers=shared_dsa_buffers,
					)
					last_full_segment = dsa_segment
					last_cos_table = cos_table
					last_sin_table = sin_table
					last_index_topk = indexer.index_topk
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
			compare_probe_layers = self._glm5_whole_model_graph_probe_layers(
				len(self.model.model.layers)
			)
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
				# e4m3 for a page-split FP8 indexer pool, bf16 for AoS: the aux
				# staging clone must not requantize the stream.
				aux_kv_staging_dtype=aux_manager.config.kv_dtype,
				stage_kv_for_offload=not self.glm5_disable_decode_host_kv_staging,
			)
			segment_name = make_glm5_whole_model_graph_segment_name()
			manager.register_segment(segment_name, whole_seg)
			first_wrapper = self.model.model.layers[0].self_attn.module
			num_heads = int(getattr(first_wrapper, "num_heads", 64))
			logging.info(
				f"Rank {self.rank}: capturing GLM-5 whole-model CUDA graph "
				f"segment={segment_name} buckets={capture_buckets}, "
				f"context_capacity={graph_max_seqlen}"
			)
			# Criticality probe: installed before warmup and capture, because
			# its spin nodes are issued by the segments' forward. Without a
			# request the adapter is not asked and nothing is constructed.
			self._criticality_probe = None
			probe_request = getattr(self, "_criticality_probe_request", None)
			if probe_request is not None:
				self._criticality_probe = self._graph.cuda_graph_adapter.install_criticality_probe(
					probe_request,
					whole_model_segment=whole_seg,
					device=self.torch_device,
				)
			torch.cuda.synchronize(self.torch_device)
			dist.barrier()
			self._graph.cuda_graph_manager = manager
			self._graph.whole_model_graph = True
			self._graph.glm5_whole_model_graph = True
			self._graph.whole_model_bucketing = bucketing
			self._graph.whole_model_segment = whole_seg
			self._graph.glm5_whole_model_graph_capture_attempted_for_batch = True
			try:
				# Largest-first: the big bucket primes the shared graph pool and
				# smaller buckets reuse its blocks instead of growing the pool.
				for capture_bucket in sorted(capture_buckets, reverse=True):
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
				self._graph.glm5_whole_model_graph_failed_buckets.add(capture_bucket)
				self._graph.cuda_graph_manager = None
				self._graph.whole_model_segment = None
				self._graph.glm5_whole_model_all_short = False
				self._graph.glm5_whole_model_index_topk = 0
				self._graph.whole_model_bucketing = None
				self._graph.glm5_whole_model_capture_input_ids = None
				self._graph.whole_model_graph = False
				self._graph.glm5_whole_model_graph = False
				self._criticality_probe = None
				self._criticality_probe_recapture = False
				torch.cuda.empty_cache()
				if whole_graph_required:
					raise
				logging.error(
					f"Rank {self.rank}: GLM-5 whole-model CUDA graph configure-time "
					f"capture for bucket BS={capture_bucket} ran out of memory; using eager decode: {exc}"
				)
				return
			self._graph.glm5_whole_model_graph_signature = self._glm5_whole_model_graph_capture_signature()
			self._record_decode_graph_capture()
			self._criticality_probe_recapture = False
			# Phase B: hand the just-captured segment to the adapter so its
			# eligibility() / prepare_replay_inputs() / stage_post_graph_kv()
			# can run against the same captured graph the legacy path uses.
			# Without this attach, adapter._ctx stays None and eligibility()
			# returns EAGER/adapter_not_built every step (the worker silently
			# falls back to the legacy path).
			if self._graph.cuda_graph_adapter is not None:
				try:
					_adapter_max_seqlen = int(getattr(whole_seg, "max_seqlen", 0) or 0)
					_adapter_gpu_mgr = gpu_manager
					_attach = getattr(self._graph.cuda_graph_adapter, "attach_existing_segment", None)
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
						_sig = self._graph.cuda_graph_adapter.capture_signature(
							bucket=int(_bucket),
							gpu_kv_manager=_adapter_gpu_mgr,
							max_seqlen=_adapter_max_seqlen,
						)
						self._graph.cuda_graph_adapter.record_capture(
							segment_name=segment_name,
							bucket=int(_bucket),
							signature=_sig,
						)
				except Exception as _exc:
					logging.warning(
						"Phase B: failed to attach/record_capture on adapter: %s", _exc,
					)
			# The decode graphs are captured and active, so the eager MoE
			# buffers these layers share (one set for the whole model) are dead
			# residency competing with the KV pool that
			# _finalize_gpu_kv_pool_after_capture sizes from the post-capture
			# steady usage. The eager fallback paths re-create them on demand
			# at their own entry, so this is a residency switch only.
			for _mlp in moe_layers[:1]:
				_eager_moe_freed = _mlp.release_eager_decode_buffers()
				if _eager_moe_freed and self.rank == 0:
					logging.info(
						"Rank %s: released the eager MoE decode buffers "
						"(%.3f GiB) after graph capture",
						self.rank,
						_eager_moe_freed / (1024 ** 3),
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

		# Every other decode graph path is under the same capture-once rule.
		self._assert_decode_graph_capture_allowed()

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
		if _is_k25 and self._graph.cuda_graph_adapter is not None:
			from batchgen.models.moonshotai.kimi_k25.cuda_graph_adapter import (
				SEGMENT_NAME_WHOLE_MODEL as _K25_WM_NAME,
			)
			bundle = self._graph.cuda_graph_adapter.build_segments(
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
				_sig = self._graph.cuda_graph_adapter.capture_signature(
					bucket=_bucket, gpu_kv_manager=gpu_manager, max_seqlen=max_rope_len,
				)
				self._graph.cuda_graph_adapter.record_capture(
					segment_name=_K25_WM_NAME, bucket=_bucket, signature=_sig,
				)
			self._graph.cuda_graph_manager = manager
			self._graph.whole_model_graph = True
			self._graph.k25_whole_model_graph = True
			self._graph.whole_model_segment_name = _K25_WM_NAME
			self._graph.whole_model_bucketing = bucketing
			self._graph.whole_model_segment = whole_seg
			self._record_decode_graph_capture()
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
			manager = self._graph.new_manager(bucketing, device=self.torch_device)

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

			self._graph.cuda_graph_manager = manager
			self._graph.whole_model_graph = True
			self._graph.whole_model_bucketing = bucketing
			self._graph.whole_model_segment = whole_seg
			if self.rank == 0:
				stats = manager.get_capture_stats()
				logging.info(
					f"CUDA graphs ready (whole-model): {stats['total_capture_time_ms']:.0f}ms"
				)
		else:
			# Per-layer mode: capture attention graph per layer, MoE stays eager.
			self._graph.whole_model_graph = False

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

			self._graph.cuda_graph_manager = manager
			if self.rank == 0:
				stats = manager.get_capture_stats()
				logging.info(
					f"CUDA graphs ready: {stats['total_capture_time_ms']:.0f}ms"
				)
		self._record_decode_graph_capture()

	def _bind_decode_entry_model_state(
		self,
		gpu_manager,
		worker_view,
		scale_dict,
		past_key_states,
		past_value_states,
		batch,
	):
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

	def _read_decode_step_model_metadata(
		self,
		global_decode_sequences,
		batch,
		probe_adapter,
		probe_round_request,
		probe_mismatch_logged,
	):
		AttnWrapperBase.batchgen_debug = self._active_batchgen_debug_for_sequences(
			global_decode_sequences
		)
		if (
			AttnWrapperBase.batchgen_debug is not None
			and probe_adapter is not None
			and not probe_mismatch_logged
		):
			try:
				_step_probe_request = probe_adapter.criticality_probe_request(
					AttnWrapperBase.batchgen_debug, graph_enabled=True
				)
			except ValueError as _probe_exc:
				_step_probe_request = _probe_exc
			if _step_probe_request != probe_round_request:
				probe_mismatch_logged = True
				logging.warning(
					"Rank %s: [CRITICALITY_PROBE] a job admitted mid-round requests a "
					"different criticality probe (%s); this round keeps the request "
					"it was configured with until the next decode configure",
					self.rank,
					_step_probe_request,
				)
		self._k3_decode_profile_step(AttnWrapperBase.batchgen_debug, len(batch))
		self._configure_glm5_dispatch_trace(global_decode_sequences)
		return probe_mismatch_logged

	def _rebind_decode_replay_model_metadata(
		self,
		batch_sequences,
		cache_seqlens,
		max_ctx,
		batch,
		new_tokens,
		page_table_verified,
		gpu_manager,
	):
		if batch:
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
			_reset_glm5_dsa_step_state(
				GLM5AttnWrapper,
				Attn_Wrapper.cache_seqlens,
				getattr(self.model_config, "index_topk", None),
			)

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
			self._decode_metadata_batch_key = None
			self._decode_metadata_cpu_seqlens = None

		if batch:
			Attn_Wrapper.cur_batch = self._local_indices_to_global_seq_ids(batch)
			AttnWrapperBase.cur_batch = Attn_Wrapper.cur_batch

			# OPTIMIZATION: Only check page table if not already verified this batch
			# Between boundaries, batch doesn't change so page table stays valid
			if not page_table_verified:
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
				page_table_verified = True

		return new_tokens, page_table_verified

	def _install_decode_kv_append_callbacks(self, batch, batch_sequences):
		# KV append callback — deferred: accumulate during forward, single sync after
		current_batch = list(batch)
		_kv_worker_view = getattr(self.core_engine, "host_paged_kv_worker_view", None)

		if _kv_worker_view is not None:
			# ``batch_sequences`` was resolved from this same ``batch``
			# above and neither the batch nor the local->uuid map moves
			# inside a step, so re-resolving every row through
			# _local_to_uuid_map + get_sequence here was pure
			# duplication.
			_kv_seq_ids = []
			_kv_seq_lengths = []
			for seq in batch_sequences:
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
		# to _deferred_kv_entries_aux; _flush_deferred_kv_to_host launches
		# primary and aux appends together and the per-step
		# [KV_APPEND_DRAIN] after the token readback completes both.
		aux_view = getattr(self, "host_paged_kv_worker_view_aux", None)
		if aux_view is not None:
			if BATCHGEN_SYNC_KV:
				def kv_append_callback_aux(
					layer_idx: int,
					k_tensor: torch.Tensor,
					v_tensor: torch.Tensor = None,
					scale_tensor: torch.Tensor = None,
				):
					self._append_decode_kv_to_host_aux_async(
						layer_idx, current_batch, k_tensor, v_tensor,
						scale_tensor,
					)
			else:
				def kv_append_callback_aux(
					layer_idx: int,
					k_tensor: torch.Tensor,
					v_tensor: torch.Tensor = None,
					scale_tensor: torch.Tensor = None,
				):
					valid_layers = self._aux_layers_with_slots()
					if valid_layers is not None and layer_idx not in valid_layers:
						return
					self._deferred_kv_entries_aux.append(
						(layer_idx, k_tensor, v_tensor, scale_tensor)
					)
			AttnWrapperBase.kv_append_callback_aux = kv_append_callback_aux
		else:
			AttnWrapperBase.kv_append_callback_aux = None

	# One reader per bound decode-step term (S6 fold: the S4b binder named
	# one reader per call site; the bodies were identical, and ``Attn_Wrapper``
	# is ``AttnWrapperBase``).
	def _read_decode_max_seqlen(self):
		return int(getattr(AttnWrapperBase, "max_seqlen", 0) or 0)

	def _read_decode_cur_batch(self):
		return tuple(getattr(AttnWrapperBase, "cur_batch", None) or ())

	def _read_decode_cache_seqlens(self):
		return AttnWrapperBase.cache_seqlens

	def _read_decode_position_ids(self):
		return AttnWrapperBase.position_ids

	def _read_decode_attention_mask(self):
		return AttnWrapperBase.attention_mask

	def _read_decode_kv_append_callbacks(self):
		return (
			getattr(AttnWrapperBase, 'kv_append_callback', None),
			getattr(AttnWrapperBase, 'kv_append_callback_aux', None),
		)

	def _teardown_decode_model_state(self):
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

	def _whole_model_replay_page_inputs(self, **kwargs):
		from batchgen.models.openai.gpt_oss_120b.whole_model_graph_inputs import (
			whole_model_replay_page_inputs,
		)
		return whole_model_replay_page_inputs(**kwargs)

	def _compare_glm5_whole_model_graph_logits(self, **kwargs):
		from batchgen.models.glm.glm5.whole_model_cuda_graph_segments import (
			compare_glm5_whole_model_graph_logits,
		)
		return compare_glm5_whole_model_graph_logits(
			**kwargs,
			atol=self._glm5_whole_model_graph_compare_tolerance("glm5_whole_model_graph_compare_atol"),
			rtol=self._glm5_whole_model_graph_compare_tolerance("glm5_whole_model_graph_compare_rtol"),
		)

	def _decode_whole_model_segment_name(self):
		return self._graph.whole_model_segment_name or "glm5_whole_model"

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

		strategy = strategy_for(self.engine_config.Basic_Config.attn_mode)
		if strategy is not DecodeStrategy.DECOUPLED_GPU:
			self._decoding_legacy_modes(new_tokens, decode_uuids, batch, 1, strategy)
			return decode_uuids, batch

		return DecoupledGpuDecode().run(
			self,
			self,
			decode_uuids,
			batch,
			new_tokens,
			past_key_states,
			past_value_states,
			scale_dict,
		)

	def _wait_pending_kv_append_tasks(
		self,
		*,
		sync_distributed_errors: bool = False,
		defer_errors: bool = False,
		device_sync: bool = True,
	) -> int:
		"""
		Wait for all pending KV append tasks at page boundary.
		Returns the number of tasks that were waited for.
		
		CRITICAL: Also syncs CUDA to ensure all D2H DMA operations complete.
		Without this, KV data may not be fully written to host memory when
		sequences are later resumed, causing KV corruption.

		``device_sync=False`` is only for the deferred-readback decode step,
		where the next forward is already enqueued and a device-wide sync
		would wait for it; see _finish_deferred_decode_step for why each
		task's own copy-stream sync suffices there.
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
		if device_sync and num_tasks > 0 and not wait_errors:
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


	def _rebuild_input_tokens(self, batch: List[int]) -> torch.Tensor:
		"""Build input tokens from each sequence's last decoded position."""
		self._require_decode_tokens_read("_rebuild_input_tokens")
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
			if self._traj.trajectory_book is None:
				continue
			# Use the turn ledger: later user prompts make the latest generated
			# token non-contiguous with the original prompt boundary.
			token = self._traj.trajectory_book.last_generated_token(
				self._trajectory_slot(seq)
			).to(dtype=torch.int64).view(1, 1)
			tokens.append(token)

		result = torch.cat(tokens, dim=0).to(self.torch_device) if tokens else torch.empty((0, 1), dtype=torch.int64, device=self.torch_device)
		
		if result.shape[0] != len(batch):
			logging.error(
				f"Rank {self.rank}: _rebuild_input_tokens MISMATCH: "
				f"batch_size={len(batch)}, result_size={result.shape[0]}, "
				f"tokens_collected={len(tokens)}"
			)
		
		return result



	def _legacy_decode_step_bound(self, decode_uuids: List[str]) -> int:
		"""Exclusive step bound of a legacy decode call, from the decode set's budgets.

		Every row enters the call with its prefill token decoded, so no row
		needs more than ``max_decode_length - 1`` further steps and the largest
		budget in the set bounds the call. ``decode_uuids`` is the rank-synced
		decode set and admission sets ``max_decode_length`` identically on every
		rank, so every rank runs the same number of steps. The worker-wide
		``max_decoding_length`` must not be used: in pool mode it is the first
		batch's budget, so it would end every call at that budget and, at 1,
		run no step at all for any later batch.
		"""
		return max(
			(self.global_batch.get_sequence(uuid).max_decode_length for uuid in decode_uuids),
			default=0,
		)

	def _decoding_legacy_modes(
		self,
		new_tokens: torch.Tensor,
		decode_uuids: List[str],
		batch: List[int],
		start_token_idx: int,
		strategy: DecodeStrategy,
	) -> None:
		"""Legacy decoding for modes 0, 1, 2 with continuous batching support."""
		new_token_idx = start_token_idx

		while new_token_idx < self._legacy_decode_step_bound(decode_uuids) and (decode_uuids or batch):
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
			

			if strategy is DecodeStrategy.CPU_ATTN:
				"""CPU ATTN MODE - NO ATTN MICRO BATCH"""
				with torch.inference_mode():
					Attn_Wrapper.cur_batch = [batch]
					# Build attention mask on-the-fly from sequence metadata
					cache_seqlens = []
					for query_idx in batch:
						uuid = self._local_to_uuid_map[query_idx]
						seq = self.global_batch.get_sequence(uuid)
						cache_seqlens.append(seq.current_context_length)
					max_len = max(cache_seqlens, default=0)
					seqlens_tensor = torch.tensor(cache_seqlens, dtype=torch.int64)
					positions = torch.arange(max_len)
					attention_mask = (positions.unsqueeze(0) < seqlens_tensor.unsqueeze(1)).to(torch.int64)
					if "deepseek" not in self.model_config.model_type:
						position_ids = (seqlens_tensor - 1).unsqueeze(-1)
					else:
						position_ids = create_position_ids_from_attention_mask(attention_mask)

					Attn_Wrapper.attention_mask = attention_mask
					Attn_Wrapper.position_ids = position_ids
					self._weight_passes.begin(carry=True)
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
					self._weight_passes.end()
					self.update_new_token(new_tokens, batch, new_token_idx)

					# Update sequence state
					for i, local_idx in enumerate(batch):
						uuid = self._local_to_uuid_map[local_idx]
						seq = self.global_batch.get_sequence(uuid)
						seq.decoded_length += 1
						seq.current_context_length = (
							seq.original_prompt_length + seq.decoded_length
						)

						# Only mark eos_reached if we should stop at EOS
						token_id = new_tokens[i].item()
						if self._should_stop_at_eos(token_id, seq):
							seq.eos_reached = True

						# Always check max length
						if seq.decoded_length >= seq.max_decode_length:
							seq.eos_reached = True

						# Repetition detection (BATCHGEN_REP_DETECTION=1)
						if REP_DETECTION and not seq._rep_detected and _repetition_check_enabled(seq):
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

			elif strategy is DecodeStrategy.LEGACY_MICROBATCH_GPU:
				"""GPU ATTN MODE - ATTN MICRO BATCH"""
				micro_batch_size = self.engine_config.Module_Batching_Config.attn_decoding_micro_batch_size
				num_micro_batches = math.ceil(len(batch) / micro_batch_size)
				micro_batches = [
					batch[micro_batch_idx * micro_batch_size : (micro_batch_idx + 1) * micro_batch_size]
					for micro_batch_idx in range(num_micro_batches)
				]
				Attn_Wrapper.cur_batch = micro_batches
				cache_seqlens = [
					self.global_batch.get_sequence(self._local_to_uuid_map[query_idx]).current_context_length
					for query_idx in batch
				]
				max_cache_len = max(cache_seqlens, default=0)
				
				if (new_token_idx - 1) % 32 == 0:
					for idx in range(new_token_idx - 1, new_token_idx + 31):
						past_kv_byte_size = legacy_past_kv_byte_size(
							self.model_config,
							max_input_length=self.max_input_length,
							token_idx=idx,
							max_cache_len=max_cache_len,
						)

						for layer_idx in range(self.model_config.num_hidden_layers):
							for micro_batch_idx in range(num_micro_batches):
								cur_batch = micro_batches[micro_batch_idx]
								self.core_engine.submit_to_KV_queue(
									cur_batch, micro_batch_idx, layer_idx, past_kv_byte_size,
								)

				with torch.inference_mode():
					# Build attention mask on-the-fly from sequence metadata
					max_len = max_cache_len
					seqlens_tensor = torch.tensor(cache_seqlens, dtype=torch.int64, device=self.torch_device)
					positions = torch.arange(max_len, device=self.torch_device)
					attention_mask = (positions.unsqueeze(0) < seqlens_tensor.unsqueeze(1)).to(torch.int64)
					if "deepseek" in self.model_config.model_type:
						position_ids = create_position_ids_from_attention_mask(attention_mask)
					else:
						position_ids = (seqlens_tensor - 1).unsqueeze(-1)

					Attn_Wrapper.attention_mask = attention_mask
					Attn_Wrapper.position_ids = position_ids
					self._weight_passes.begin(carry=True)
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
					self._weight_passes.end()
					self.update_new_token(new_tokens, batch, new_token_idx)

					# Update sequence state
					for i, local_idx in enumerate(batch):
						uuid = self._local_to_uuid_map[local_idx]
						seq = self.global_batch.get_sequence(uuid)
						seq.decoded_length += 1
						seq.current_context_length = (
							seq.original_prompt_length + seq.decoded_length
						)

						# Only mark eos_reached if we should stop at EOS
						token_id = new_tokens[i].item()
						if self._should_stop_at_eos(token_id, seq):
							seq.eos_reached = True

						# Always check max length
						if seq.decoded_length >= seq.max_decode_length:
							seq.eos_reached = True

						# Repetition detection (BATCHGEN_REP_DETECTION=1)
						if REP_DETECTION and not seq._rep_detected and _repetition_check_enabled(seq):
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

			elif strategy is DecodeStrategy.HYBRID_CPU_GPU:
				"""CPU-GPU Parallel ATTN - Deprecated"""
				logging.warning("RUNTIME_ATTN_MODE 2 is deprecated")
				new_token_idx += 1

	# ============ Utility Methods ============

	def set_phase(self, phase: str):
		"""Control different behavior of the engine in different phases."""
		# Kept on the worker too: the KV sizing guard needs to know which
		# phase it is running in, and the wrapper class attributes below are
		# model-side state.
		self._phase = phase
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
			uuid = self._local_to_uuid_map[q_idx]
			seq = self.global_batch.get_sequence(uuid)
			self._append_trajectory_token(seq, int(new_tokens[idx].item()))

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
			comm_master_addr = os.getenv("COMM_MASTER_ADDR", "127.0.0.1")
			selected_port = self._nccl_port
			if self.rank == 0:
				try:
					selected_port = self._find_available_pynccl_port(
						comm_master_addr,
						self._nccl_port + 1,
					)
					logging.debug(f"Rank 0: Found available port {selected_port} for PyNccl reinit")
				except RuntimeError as e:
					logging.error(f"Rank 0: Failed to find available port: {e}")
					selected_port = -1
			# Broadcast port to all ranks
			port_tensor = torch.tensor([selected_port], dtype=torch.int32, device=self.torch_device)
			dist.broadcast(port_tensor, src=0)
			self._nccl_port = port_tensor.item()
			if self._nccl_port < 0:
				return False
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

	def _release_decode_graph_state(self) -> None:
		"""Drop the decode instance's captured graphs and their bookkeeping."""
		self._graph.cuda_graph_manager = None
		self._graph.whole_model_segment = None
		self._graph.glm5_whole_model_all_short = False
		self._graph.glm5_whole_model_index_topk = 0
		self._graph.whole_model_bucketing = None
		self._graph.glm5_whole_model_capture_input_ids = None
		self._graph.whole_model_graph = False
		# The K2.5 capture's two descriptors go with the graphs they describe,
		# so a later capture of any mode starts from the field defaults.
		self._graph.k25_whole_model_graph = False
		self._graph.whole_model_segment_name = None
		self._graph.glm5_whole_model_graph = False
		self._graph.glm5_whole_model_graph_failed_buckets = set()
		self._graph.glm5_whole_model_graph_capture_attempted_for_batch = False
		self._graph.glm5_whole_model_graph_state_change_after_capture_logged = False
		self._graph.glm5_whole_model_graph_signature = None
		self._graph.glm5_whole_model_graph_unavailable_reason = None
		# The probe's spin nodes live in the graphs dropped above.
		self._criticality_probe = None
		# Adapter holds Python refs to model/segment/KV manager via _ctx;
		# without release_context() the captured segment's static KV buffers
		# survive empty_cache() and prefill OOMs on the next batch.
		if self._graph.cuda_graph_adapter is not None:
			self._graph.cuda_graph_adapter.release_context()
		# A later decode instance is a new owner and captures its own graphs.
		self._graph.decode_graph_captures = 0

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

		self._phase_mem_switch_index = (
			getattr(self, "_phase_mem_switch_index", 0) + 1
		)
		self._log_phase_memory_ledger("deep_free_enter")

		# Stage timings on every rank: the first decode configure after the
		# streamed wave-1 prefill spent 34 s here on rank 0 (r30) / ~30 s on
		# node 1 (r29), intermittently, with nothing logged.
		_df_t0 = time.perf_counter()
		# Ensure all GPU operations complete before deletion
		if torch.cuda.is_available():
			torch.cuda.synchronize(self.torch_device)
		_df_t1 = time.perf_counter()

		# Free WGMMA shared buffers if they exist (class-level, survives model deletion).
		# Only when the GLM-5 model module is already loaded: importing it here
		# on a non-GLM-5 server walks the GLM-5 JIT extensions -- MEASURED r33
		# (K3, 2x8 H200, shared TORCH_EXTENSIONS_DIR): del=30.2 s on every rank
		# of one node (0.2 s on the other) at the first decode configure.
		_glm5_model_module = sys.modules.get("batchgen.models.glm.glm5.model")
		if _glm5_model_module is not None:
			Glm5MoE = getattr(_glm5_model_module, "Glm5MoE", None)
			if Glm5MoE is not None and getattr(Glm5MoE, '_wgmma_shared_bufs', None) is not None:
				Glm5MoE._wgmma_shared_bufs.free_buffers()
				Glm5MoE._wgmma_shared_bufs = None
				Glm5MoE._wgmma_next_layer_id = 0

		# Delete model directly without CPU transfer
		del self.model
		self.model = None
		# Decode graph state belongs to the decode instance that captured it.
		# A held decode instance replays those graphs on its next activation
		# (capture-once), so freeing the live prefill model must not drop
		# them; they go when the decode instance itself is retired, whose
		# phase_exit clears the manager's hold before this runs.
		if not self.parallel_manager.holds_instance("decode"):
			self._release_decode_graph_state()

		# Defense-in-depth: free PSM-owned GPU buffers that survive model deletion
		# (INT4 contiguous weight buffers, MoE class-level buffers)
		_df_t2 = time.perf_counter()
		if hasattr(self, 'parallel_manager') and self.parallel_manager is not None:
			pm = self.parallel_manager
			if hasattr(pm, "_release_streamed_sp8_prefill"):
				pm._release_streamed_sp8_prefill()
			# The worker is not the only owner of the model: the parallel strategy
			# manager keeps its own reference from configure_prefill/
			# configure_decoding. Leaving that reference set is what made this
			# function a measured no-op on the decode->prefill switch (allocated
			# bytes unchanged across the call): the decode model was really freed
			# later, by the refcount drop when configure_prefill rebound pm.model,
			# so it stayed resident through the KV destroy and the first part of the
			# prefill configure. Dropping it here is what makes the function free
			# what its name says it frees.
			#
			# Persistent instances own their model tuples; the guard keeps those
			# references intact while a suspend transition is in progress.
			if not getattr(pm, "persistent_phase_instances", False):
				if getattr(pm, "model", None) is not None:
					pm.model = None
			for attr in ('_int4_packed_gpu_buf', '_int4_scale_gpu_buf'):
				if hasattr(pm, attr):
					delattr(pm, attr)
		_df_t3 = time.perf_counter()

		# Release memory
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		_df_t4 = time.perf_counter()
		gc.collect()
		logging.info(
			f"[DECODE] Rank {self.rank}: deep_free_model_memory sync={_df_t1 - _df_t0:.1f}s "
			f"del={_df_t2 - _df_t1:.1f}s release_streamed={_df_t3 - _df_t2:.1f}s "
			f"empty_cache={_df_t4 - _df_t3:.1f}s gc={time.perf_counter() - _df_t4:.1f}s"
		)
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
		# After empty_cache: what is still held here is what this switch
		# did NOT give back, which is the number the rebuild path's
		# net-zero invariant is read from.
		self._log_phase_memory_ledger("deep_free_done")
